//! Surviving an etcd auth token the server no longer accepts.
//!
//! `etcd-client` authenticates once, inside `Client::connect`, and never
//! again. The token it gets there is the only one that connection will
//! ever carry, and etcd stops accepting it in two ordinary situations:
//!
//! 1. **The token's TTL elapses.** Answered `Unauthenticated`
//!    (`etcdserver: invalid auth token`).
//! 2. **The auth store's revision changes** — someone runs `etcdctl user
//!    add`, grants a role, edits a permission. Every token issued before
//!    that is refused from the moment it lands, and answered
//!    `InvalidArgument` (`etcdserver: revision of auth store is old`),
//!    sharing a status code with a wrong password.
//!
//! Every later call is then refused, and before this the only cure was
//! restarting the gateway. Note what is *not* on the list: an etcd
//! restart on its own does not invalidate anything, because
//! `Authenticate` is a raft entry and replaying the WAL re-registers the
//! tokens it minted.
//!
//! Both need a server that really issues and forgets tokens — the status
//! code, the wording and the *timing* are etcd's, not something a stub
//! can stand in for. Bracketed by `ETCD_AUTH_TTL_TEST_URL` (the pattern
//! `auth_connect_integration.rs` uses): the tests no-op when it is unset,
//! so a local `cargo test` without docker still passes. CI starts the
//! cluster and sets the variables in `.github/workflows/ci.yml`.
//!
//! The cluster this file uses is a **third** etcd, separate from the
//! `ETCD_AUTH_TEST_URL` one, for two reasons. Its token lifetime is
//! seconds rather than the five-minute default, which would make every
//! other authenticated test depend on this feature. And it runs
//! `--auth-token jwt` rather than the default `simple`, which is what
//! reaches case 2 above: under `simple` the auth revision is read at
//! request time, so only JWT — what a multi-member cluster runs anyway —
//! refuses a token for it.
//!
//! The container and short-TTL cluster this needs were built by community
//! PR api7/aisix#763 (`okaybase`), which proposed the scheduled-refresh
//! fix these tests replace.

#![allow(clippy::expect_used)]

use std::sync::LazyLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use aisix_etcd::{ConfigProvider, EtcdConfigProvider, ProviderError};
use etcd_client::ConnectOptions;

/// Two tests here change the cluster's auth store, which stales every
/// token every other test in this file is holding, so they take turns.
/// Test binaries are run one at a time, so nothing outside this file is
/// talking to it.
static ETCD: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

struct AuthEtcd {
    url: String,
    user: String,
    password: String,
    /// The lifetime of a token this cluster issues (the `ttl` inside its
    /// `--auth-token`). Read from the environment rather than assumed, so
    /// the sleep below tracks whatever CI started.
    token_ttl: Duration,
}

fn auth_ttl_etcd() -> Option<AuthEtcd> {
    Some(AuthEtcd {
        url: std::env::var("ETCD_AUTH_TTL_TEST_URL").ok()?,
        user: std::env::var("ETCD_AUTH_TTL_TEST_USER").ok()?,
        password: std::env::var("ETCD_AUTH_TTL_TEST_PASSWORD").ok()?,
        // Parsed strictly on purpose: a typo in the workflow would
        // otherwise turn every test in this file into a silent skip,
        // which on a check is the same colour as a pass.
        token_ttl: Duration::from_secs(
            std::env::var("ETCD_AUTH_TTL_TEST_SECS")
                .ok()?
                .parse()
                .expect("ETCD_AUTH_TTL_TEST_SECS must be a whole number of seconds"),
        ),
    })
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default()
}

fn unique_prefix() -> String {
    format!("/aisix-token-it-{}", unique_suffix())
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
async fn a_token_an_auth_store_change_staled_is_replaced_without_a_restart() {
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

    // The half no schedule could have covered: adding a user bumps the
    // auth store's revision, and from that instant every token issued
    // before it is refused — nothing about it is predictable from the
    // client, and it does not come back on its own. It is also refused
    // with `InvalidArgument`, the same status code as a wrong password,
    // so this is the test that keeps those two apart end to end: if the
    // gateway treated it as a wrong password it would stop reading its
    // configuration for good, over an operator adding a user.
    let stale_user = format!("stale-{}", unique_suffix());
    let mut admin = etcd.writer().await;
    admin
        .user_add(stale_user.clone(), "stale-pw", None)
        .await
        .expect("bump the auth store revision");

    let (entries, _) = provider
        .load_all()
        .await
        .expect("a staled token must be replaced, not fail the read");
    assert_eq!(
        entries.len(),
        1,
        "the read after re-authenticating must return the same configuration",
    );

    // Key first, user second: deleting the user bumps the revision
    // again and stales this writer's own token, so nothing may follow it.
    let mut writer = etcd.writer().await;
    writer
        .delete(format!("{prefix}/models/m-token-2"), None)
        .await
        .expect("cleanup");
    writer.user_delete(stale_user).await.expect("cleanup user");
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
    let user = format!("norole-{}", unique_suffix());
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

    // A writer of its own. This cluster's token TTL is seconds, and the
    // connect and read above can outlast it on a loaded runner — a
    // cleanup on the original `root` would then fail for a reason that
    // has nothing to do with what this test asserts.
    let mut root = etcd.writer().await;
    root.user_delete(user).await.expect("cleanup");
}
