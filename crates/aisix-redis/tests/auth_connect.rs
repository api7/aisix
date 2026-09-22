//! What a boot does with a credential the Redis server has an opinion
//! about, against a live `redis:7 --requirepass`.
//!
//! Runs only when `REDIS_AUTH_TEST_URL` + `REDIS_AUTH_TEST_PASSWORD` are
//! set (CI starts a password-protected Redis for it; absence is a no-op
//! so local unit runs stay hermetic).
//!
//! Two properties, and they are the same property seen from both sides:
//! a server that ANSWERS and refuses must be told apart from one that
//! never answered, and a credential supplied through the `password`
//! field must reach the handshake — because the failure that made both
//! of these tests necessary was the gateway reporting `connected` in one
//! case and `redis connect timed out` in the other, never once naming
//! the password.
//!
//! Neither outcome ends a boot. What the classification decides is what
//! the operator is TOLD while the gateway serves degraded, and whether
//! the background re-attach is waiting on a network or on them.

use aisix_core::{RedisConnConfig, RedisMode};
use aisix_redis::{classify_connect_failure, connect_bounded, ConnectFailure, FailurePolicy};

/// `redis://host:port`, with NO credential in it.
fn plain_url() -> Option<String> {
    std::env::var("REDIS_AUTH_TEST_URL").ok()
}

fn password() -> Option<String> {
    std::env::var("REDIS_AUTH_TEST_PASSWORD").ok()
}

fn single(url: &str) -> RedisConnConfig {
    RedisConnConfig {
        mode: RedisMode::Single,
        url: Some(url.to_string()),
        // Short: every refusal case here must return long before this,
        // and a case that does time out should say so quickly.
        timeout_secs: 3,
        ..Default::default()
    }
}

/// `redis://:<password>@host:port` from a plain `redis://host:port`.
fn with_url_credential(url: &str, password: &str) -> String {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    format!("redis://:{password}@{rest}")
}

/// Connect the way boot does, then prove the connection can actually
/// run a command.
///
/// The round trip is the point, not decoration: a `requirepass` server
/// lets an unauthenticated client open a connection and only refuses the
/// commands, so a connect that returns `Ok` proves nothing at all about
/// the credential. That is exactly how an ignored `password` field
/// produced a boot that logged `connected` and then failed every
/// counter operation for the life of the process.
async fn connect_and_use(cfg: &RedisConnConfig) -> Result<(), redis::RedisError> {
    let policy = FailurePolicy::new(cfg);
    let conn = connect_bounded(cfg, &policy).await?;
    let mut handle = conn.acquire().await?;
    redis::cmd("SET")
        .arg("aisix:auth-connect-test")
        .arg("1")
        .query_async::<()>(&mut handle)
        .await
}

/// The whole finding in one assertion: the server answered, so this is a
/// refusal rather than an outage, and the error must carry what it
/// answered rather than a timeout that never happened.
#[tokio::test]
async fn a_wrong_password_in_the_url_reads_as_a_refusal() {
    let (Some(url), Some(_)) = (plain_url(), password()) else {
        return;
    };
    let cfg = single(&with_url_credential(&url, "definitely-not-the-password"));
    let err = connect_and_use(&cfg)
        .await
        .expect_err("a refused credential must fail the connect");
    assert_eq!(
        classify_connect_failure(&err),
        ConnectFailure::Refused,
        "the server answered; classifying it as an outage is what produced the \
         timeout diagnostic: {err}"
    );
    let text = err.to_string().to_lowercase();
    assert!(
        text.contains("auth"),
        "the error must name the refusal: {err}"
    );
    assert!(
        !text.contains("timed out"),
        "the server answered in milliseconds; calling it a timeout sends the operator \
         looking at the network: {err}"
    );
}

/// The `password` field is the documented way to keep the secret out of
/// the config file. Before this it was parsed and then never applied in
/// `single` mode: the boot logged `connected` and every command failed.
#[tokio::test]
async fn the_password_field_authenticates_when_the_url_carries_no_credential() {
    let (Some(url), Some(pw)) = (plain_url(), password()) else {
        return;
    };
    let cfg = RedisConnConfig {
        password: Some(pw),
        ..single(&url)
    };
    connect_and_use(&cfg)
        .await
        .expect("the configured password must reach the handshake");
}

#[tokio::test]
async fn a_wrong_password_field_reads_as_a_refusal() {
    let (Some(url), Some(_)) = (plain_url(), password()) else {
        return;
    };
    let cfg = RedisConnConfig {
        password: Some("definitely-not-the-password".into()),
        ..single(&url)
    };
    let err = connect_and_use(&cfg)
        .await
        .expect_err("a refused credential must fail the connect");
    assert_eq!(
        classify_connect_failure(&err),
        ConnectFailure::Refused,
        "{err}"
    );
}

/// Precedence, stated in the field's rustdoc and in `config.example.yaml`:
/// the explicit field wins. A stale credential left in the URL must not
/// outrank the one the operator injected through the environment.
#[tokio::test]
async fn the_password_field_overrides_a_credential_in_the_url() {
    let (Some(url), Some(pw)) = (plain_url(), password()) else {
        return;
    };
    let cfg = RedisConnConfig {
        password: Some(pw),
        ..single(&with_url_credential(&url, "the-stale-one"))
    };
    connect_and_use(&cfg)
        .await
        .expect("the explicit password must override the URL's");
}

/// The ACL user, on a server that has only `default`. Pins that
/// `username` travels with the password rather than being dropped.
#[tokio::test]
async fn the_username_field_reaches_the_handshake() {
    let (Some(url), Some(pw)) = (plain_url(), password()) else {
        return;
    };
    let ok = RedisConnConfig {
        username: Some("default".into()),
        password: Some(pw.clone()),
        ..single(&url)
    };
    connect_and_use(&ok)
        .await
        .expect("the default ACL user must authenticate");

    let bad = RedisConnConfig {
        username: Some("no-such-acl-user".into()),
        password: Some(pw),
        ..single(&url)
    };
    let err = connect_and_use(&bad)
        .await
        .expect_err("an unknown ACL user must fail the connect");
    assert_eq!(
        classify_connect_failure(&err),
        ConnectFailure::Refused,
        "{err}"
    );
}

/// The other half of the classification: nothing answered, so this must
/// not be dressed up as a refusal — the operator would go looking at a
/// credential when the server is simply not there.
#[tokio::test]
async fn an_endpoint_that_does_not_answer_reads_as_unreachable() {
    // Port 1 on loopback: refused immediately, so this stays hermetic
    // and fast. The blackhole *timing* is pinned by the e2e cases.
    let cfg = single("redis://127.0.0.1:1");
    let err = connect_and_use(&cfg)
        .await
        .expect_err("nothing is listening there");
    assert_eq!(
        classify_connect_failure(&err),
        ConnectFailure::Unreachable,
        "nothing answered, so this must not be reported as a refusal: {err}"
    );
}

/// `cluster` and `sentinel` mode, where the explicit fields are the ONLY
/// way to authenticate the data node — a Sentinel-discovered master has
/// no URL of its own, and a cluster's slot map names nodes the seed list
/// never mentioned.
///
/// Pointed at topologies that need NO password, so a *bogus* one is the
/// discriminator: if the field reaches the handshake the server refuses
/// it, and if it is dropped on the floor the connect quietly succeeds.
/// That is the same assertion as the `single` cases above, read in the
/// mirror.
mod data_node_credentials {
    use super::*;

    fn cluster_nodes() -> Option<Vec<String>> {
        let nodes = std::env::var("REDIS_AUTH_TEST_CLUSTER_NODES").ok()?;
        Some(nodes.split(',').map(|s| s.trim().to_string()).collect())
    }

    fn sentinel_topology() -> Option<(Vec<String>, String)> {
        let sentinels = std::env::var("REDIS_AUTH_TEST_SENTINELS").ok()?;
        let master = std::env::var("REDIS_AUTH_TEST_MASTER").ok()?;
        Some((
            sentinels.split(',').map(|s| s.trim().to_string()).collect(),
            master,
        ))
    }

    #[tokio::test]
    async fn cluster_applies_the_password_field_to_its_nodes() {
        let Some(nodes) = cluster_nodes() else { return };
        let cfg = RedisConnConfig {
            mode: RedisMode::Cluster,
            nodes,
            password: Some("a-password-this-cluster-does-not-want".into()),
            timeout_secs: 3,
            ..Default::default()
        };
        let err = connect_and_use(&cfg)
            .await
            .expect_err("a password the cluster refuses must fail the connect");
        assert_eq!(
            classify_connect_failure(&err),
            ConnectFailure::Refused,
            "a refused cluster credential must read as a refusal: {err}"
        );
    }

    #[tokio::test]
    async fn sentinel_applies_the_password_field_to_the_master() {
        let Some((sentinels, master_name)) = sentinel_topology() else {
            return;
        };
        let cfg = RedisConnConfig {
            mode: RedisMode::Sentinel,
            sentinels,
            master_name: Some(master_name),
            password: Some("a-password-this-master-does-not-want".into()),
            timeout_secs: 3,
            ..Default::default()
        };
        let err = connect_and_use(&cfg)
            .await
            .expect_err("a password the master refuses must fail the connect");
        assert_eq!(
            classify_connect_failure(&err),
            ConnectFailure::Refused,
            "a refused master credential must read as a refusal: {err}"
        );
    }
}

/// The boot check must not be stricter than what the gateway actually
/// needs, and it must still catch a setting the server rejects outright.
mod what_the_server_rejects {
    use super::*;

    /// An ACL user scoped to the commands a subsystem really runs has no
    /// `+ping` — and it works, so it must start.
    ///
    /// `NOPERM` is an answer from a connection the server has already
    /// authenticated: it knows who we are and is declining one command.
    /// Reading it as a bad credential would refuse to start a deployment
    /// that serves correctly.
    #[tokio::test]
    async fn an_acl_user_without_ping_still_boots() {
        let (Some(url), Some(pw)) = (plain_url(), password()) else {
            return;
        };
        let admin = RedisConnConfig {
            password: Some(pw),
            ..single(&url)
        };
        let policy = FailurePolicy::new(&admin);
        let conn = connect_bounded(&admin, &policy)
            .await
            .expect("the admin connection must come up");
        let mut handle = conn.acquire().await.expect("a handle");
        redis::cmd("ACL")
            .arg("SETUSER")
            .arg("aisix-scoped")
            .arg("on")
            .arg(">scoped-pw")
            .arg("~*")
            .arg("+get")
            .arg("+set")
            .query_async::<()>(&mut handle)
            .await
            .expect("the scoped ACL user must be creatable");

        let scoped = RedisConnConfig {
            username: Some("aisix-scoped".into()),
            password: Some("scoped-pw".into()),
            ..single(&url)
        };
        let policy = FailurePolicy::new(&scoped);
        connect_bounded(&scoped, &policy)
            .await
            .expect("a user that cannot PING but can serve must still boot");
    }

    /// A `database` the server does not have is the same class as a
    /// refused password and was hidden the same way — the connection
    /// manager's retry ladder turned the server's `SELECT` refusal into
    /// a timeout, so the operator was told the network was down.
    ///
    /// `single` mode is where this became reachable: `database` was
    /// ignored there until the field started being applied.
    #[tokio::test]
    async fn a_database_the_server_does_not_have_reads_as_a_refusal() {
        let (Some(url), Some(pw)) = (plain_url(), password()) else {
            return;
        };
        let cfg = RedisConnConfig {
            password: Some(pw.clone()),
            // Far past the 16 a default Redis provides.
            database: Some(9_999),
            ..single(&url)
        };
        let err = connect_and_use(&cfg)
            .await
            .expect_err("an out-of-range database must fail the connect");
        assert_eq!(
            classify_connect_failure(&err),
            ConnectFailure::Refused,
            "{err}"
        );
        assert!(
            !err.to_string().to_lowercase().contains("timed out"),
            "the server answered; calling it a timeout is what hid this: {err}"
        );

        // …and one the server does have still connects, so the check
        // above is about the value rather than about the field existing.
        let ok = RedisConnConfig {
            password: Some(pw),
            database: Some(3),
            ..single(&url)
        };
        connect_and_use(&ok)
            .await
            .expect("a database the server has must connect");
    }

    /// The credential is one value, not two independent ones. Overriding
    /// only the username used to compose a login from both sources —
    /// the URL's password under the field's user — which is a credential
    /// nobody configured, and whose refusal points at config that looks
    /// right in both places.
    #[tokio::test]
    async fn a_half_specified_override_does_not_borrow_the_other_half() {
        let (Some(url), Some(pw)) = (plain_url(), password()) else {
            return;
        };
        let cfg = RedisConnConfig {
            username: Some("aisix-no-such-user".into()),
            ..single(&with_url_credential(&url, &pw))
        };
        let err = connect_and_use(&cfg)
            .await
            .expect_err("the URL's password must not be lent to another user");
        assert_eq!(
            classify_connect_failure(&err),
            ConnectFailure::Refused,
            "{err}"
        );
        // NOAUTH is the discriminator, and it is the whole point: the
        // connection authenticated as NOTHING, because a username with
        // no password is not a credential. Borrowing the URL's password
        // would have sent one — and been refused as a user/password pair
        // that appears nowhere in the configuration.
        assert!(
            err.to_string().contains("NOAUTH"),
            "a username with no password must send no credential at all: {err}"
        );
    }
}
