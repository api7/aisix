//! Surviving an etcd auth token the server no longer accepts.
//!
//! `etcd-client` authenticates once, inside `Client::connect`, and never
//! again. The token it gets there is the only one that connection will
//! ever carry, and etcd stops accepting it in two ordinary situations:
//! the server restarts (its token store is in memory), or the token's
//! `--auth-token-ttl` elapses while nothing is using it. Every later call
//! is then refused, and before this the only cure was restarting the
//! gateway.
//!
//! Both need a server that really issues and forgets tokens — the status
//! code, the wording and the *timing* are etcd's, not something a stub
//! can stand in for. Bracketed by `ETCD_AUTH_TTL_TEST_URL` (the pattern
//! `auth_connect_integration.rs` uses): the tests no-op when it is unset,
//! so a local `cargo test` without docker still passes. CI starts the
//! cluster and sets the variables in `.github/workflows/ci.yml`.
//!
//! The cluster this file uses is a **third** etcd, separate from the
//! `ETCD_AUTH_TEST_URL` one, for two reasons: its `--auth-token-ttl` is
//! seconds rather than the five-minute default, which would make every
//! other authenticated test depend on this feature; and one test here
//! momentarily turns authentication off on it, which no other test can
//! be reading through.
//!
//! The container and short-TTL cluster this needs were built by community
//! PR api7/aisix#763 (`okaybase`), which proposed the scheduled-refresh
//! fix these tests replace.

#![allow(clippy::expect_used)]

use std::sync::LazyLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use aisix_etcd::{ConfigProvider, EtcdConfigProvider, ProviderError};
use etcd_client::ConnectOptions;

/// One test here turns the cluster's authentication off and on again,
/// which every other test in this file is reading through, so they take
/// turns. Test binaries are run one at a time, so nothing outside this
/// file is talking to it.
static ETCD: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

struct AuthEtcd {
    url: String,
    user: String,
    password: String,
    /// The cluster's `--auth-token-ttl`. Read from the environment rather
    /// than assumed, so the sleep below tracks whatever CI started.
    token_ttl: Duration,
}

fn auth_ttl_etcd() -> Option<AuthEtcd> {
    Some(AuthEtcd {
        url: std::env::var("ETCD_AUTH_TTL_TEST_URL").ok()?,
        user: std::env::var("ETCD_AUTH_TTL_TEST_USER").ok()?,
        password: std::env::var("ETCD_AUTH_TTL_TEST_PASSWORD").ok()?,
        token_ttl: Duration::from_secs(
            std::env::var("ETCD_AUTH_TTL_TEST_SECS")
                .ok()
                .and_then(|s| s.parse().ok())?,
        ),
    })
}

fn unique_prefix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("/aisix-token-it-{nanos}")
}

impl AuthEtcd {
    fn credentials(&self) -> ConnectOptions {
        ConnectOptions::new().with_user(self.user.clone(), self.password.clone())
    }

    /// A client of this cluster that is not the one under test — used to
    /// seed the configuration the gateway then reads.
    async fn writer(&self) -> etcd_client::Client {
        etcd_client::Client::connect([self.url.clone()], Some(self.credentials()))
            .await
            .expect("seed client")
    }

    async fn provider(&self, prefix: &str) -> EtcdConfigProvider {
        EtcdConfigProvider::connect(
            std::slice::from_ref(&self.url),
            prefix,
            Some(self.credentials()),
            Some(Duration::from_secs(10)),
            Some(Duration::from_secs(10)),
        )
        .await
        .expect("the cluster is up and the credentials are good")
    }
}

const SEEDED_MODEL: &str = r#"{"display_name":"token-it-model","provider":"openai","model_name":"gpt-4o-mini","provider_key_id":"11111111-1111-1111-1111-111111111111"}"#;

/// Long enough past the TTL that etcd's own expiry sweep has run.
fn idle_past_expiry(ttl: Duration) -> Duration {
    ttl + Duration::from_secs(4)
}

#[tokio::test]
async fn a_token_that_expired_while_idle_is_replaced_without_a_restart() {
    let Some(etcd) = auth_ttl_etcd() else {
        eprintln!("skipping: ETCD_AUTH_TTL_TEST_URL not set");
        return;
    };
    let _serialised = ETCD.lock().await;
    let prefix = unique_prefix();
    let mut writer = etcd.writer().await;
    writer
        .put(format!("{prefix}/models/m-token-1"), SEEDED_MODEL, None)
        .await
        .expect("seed put");

    let provider = etcd.provider(&prefix).await;
    let (entries, _) = provider.load_all().await.expect("the first read works");
    assert_eq!(entries.len(), 1, "the seeded model must be readable");

    // A gateway whose configuration is not changing makes no etcd calls,
    // which is exactly the state in which etcd forgets its token — the
    // TTL is refreshed by use, so only an idle connection ever reaches
    // it. This is the shape of the reported failure: a gateway that was
    // fine all day stops being able to read the moment anything changes.
    tokio::time::sleep(idle_past_expiry(etcd.token_ttl)).await;

    let (entries, _) = provider
        .load_all()
        .await
        .expect("an expired token must be replaced, not fail the read");
    assert_eq!(
        entries.len(),
        1,
        "the read after re-authenticating must return the same configuration",
    );

    // A fresh writer: this one's own token expired during the sleep too,
    // and nothing re-authenticates a bare `etcd_client::Client` — which
    // is the whole bug, seen from the other side.
    let mut writer = etcd.writer().await;
    writer
        .delete(format!("{prefix}/models/m-token-1"), None)
        .await
        .expect("cleanup");
}

#[tokio::test]
async fn a_token_the_server_has_discarded_is_replaced_without_a_restart() {
    let Some(etcd) = auth_ttl_etcd() else {
        eprintln!("skipping: ETCD_AUTH_TTL_TEST_URL not set");
        return;
    };
    let _serialised = ETCD.lock().await;
    let prefix = unique_prefix();
    let mut writer = etcd.writer().await;
    writer
        .put(format!("{prefix}/models/m-token-2"), SEEDED_MODEL, None)
        .await
        .expect("seed put");

    let provider = etcd.provider(&prefix).await;
    let (entries, _) = provider.load_all().await.expect("the first read works");
    assert_eq!(entries.len(), 1);

    // The other half of the failure, and the one no schedule could have
    // covered: the token stops being valid at a moment nothing on the
    // client can predict. Toggling authentication clears etcd's token
    // store, which is what an operator re-enabling auth, a JWT signing
    // key regenerated at startup, and a member brought up on an empty
    // data directory all do to the tokens already handed out. Unlike an
    // expiry this never comes back on its own — the read keeps being
    // refused until something authenticates again.
    discard_every_token(&etcd).await;

    let (entries, _) = provider
        .load_all()
        .await
        .expect("a discarded token must be replaced, not fail the read");
    assert_eq!(
        entries.len(),
        1,
        "the read after re-authenticating must return the same configuration",
    );

    let mut writer = etcd.writer().await;
    writer
        .delete(format!("{prefix}/models/m-token-2"), None)
        .await
        .expect("cleanup");
}

#[tokio::test]
async fn a_user_etcd_refuses_is_answered_once_and_not_re_authenticated() {
    let Some(etcd) = auth_ttl_etcd() else {
        eprintln!("skipping: ETCD_AUTH_TTL_TEST_URL not set");
        return;
    };
    let _serialised = ETCD.lock().await;

    // A user that authenticates and is then refused by the read itself —
    // the only way a real etcd refuses a *call* rather than a dial, and
    // so the case the new retry path could have turned into a spin. It
    // must fail on the first answer, the way a wrong password does at the
    // dial.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let user = format!("norole-{nanos}");
    let password = "norole-pw";
    let mut root = etcd.writer().await;
    root.user_add(user.clone(), password, None)
        .await
        .expect("add a user with no roles");

    let prefix = unique_prefix();
    let provider = EtcdConfigProvider::connect(
        std::slice::from_ref(&etcd.url),
        &prefix,
        Some(ConnectOptions::new().with_user(user.clone(), password)),
        Some(Duration::from_secs(10)),
        Some(Duration::from_secs(10)),
    )
    .await
    .expect("the credentials themselves are good — the dial succeeds");

    let started = Instant::now();
    let err = provider
        .load_all()
        .await
        .expect_err("a user with no roles cannot read");
    assert!(
        matches!(err, ProviderError::Rejected(_)),
        "a refusal must stay a refusal, got {err:?}",
    );
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "a refusal must be answered, not retried until something times out: {:?}",
        started.elapsed(),
    );

    root.user_delete(user).await.expect("cleanup");
}

/// Make etcd forget every token it has issued, this one included.
///
/// `auth disable` drops the token store and `auth enable` starts a fresh
/// one, so the token the provider holds is refused from here on with
/// `Unauthenticated` — the same answer an elapsed TTL produces, from a
/// cause that never heals by waiting.
async fn discard_every_token(etcd: &AuthEtcd) {
    let mut admin = etcd.writer().await;
    admin.auth_disable().await.expect("auth disable");
    admin.auth_enable().await.expect("auth enable");
}
