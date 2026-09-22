//! What a boot does with a credential the Redis server has an opinion
//! about, against a live `redis:7 --requirepass`.
//!
//! Runs only when `REDIS_AUTH_TEST_URL` + `REDIS_AUTH_TEST_PASSWORD` are
//! set (CI starts a password-protected Redis for it; absence is a no-op
//! so local unit runs stay hermetic).
//!
//! Two properties, and they are the same property seen from both sides:
//! a server that ANSWERS and refuses the credential must stop the boot
//! with the refusal in the message, and a credential supplied through
//! the `password` field must reach the handshake — because the failure
//! that made both of these tests necessary was the gateway reporting
//! `connected` in one case and `redis connect timed out` in the other,
//! never once naming the password.

use aisix_core::{RedisConnConfig, RedisMode};
use aisix_redis::{connect_bounded, is_permanent_config_error, FailurePolicy};

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

/// The whole finding in one assertion: the server answered, so the boot
/// must stop, and the message must name what it answered rather than a
/// timeout that never happened.
#[tokio::test]
async fn a_wrong_password_in_the_url_is_a_permanent_error() {
    let (Some(url), Some(_)) = (plain_url(), password()) else {
        return;
    };
    let cfg = single(&with_url_credential(&url, "definitely-not-the-password"));
    let err = connect_and_use(&cfg)
        .await
        .expect_err("a refused credential must fail the connect");
    assert!(
        is_permanent_config_error(&err),
        "a refused credential must be permanent, not retried forever: {err}"
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
async fn a_wrong_password_field_is_a_permanent_error() {
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
    assert!(is_permanent_config_error(&err), "{err}");
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
    assert!(is_permanent_config_error(&err), "{err}");
}

/// The other half of the classification, and the one the fail-open path
/// depends on: nothing answered, so this is NOT permanent and the
/// background re-attach keeps trying.
#[tokio::test]
async fn an_endpoint_that_does_not_answer_is_not_permanent() {
    // Port 1 on loopback: refused immediately, so this stays hermetic
    // and fast. The blackhole *timing* is pinned by the e2e cases.
    let cfg = single("redis://127.0.0.1:1");
    let err = connect_and_use(&cfg)
        .await
        .expect_err("nothing is listening there");
    assert!(
        !is_permanent_config_error(&err),
        "an unreachable endpoint must keep being retried in the background: {err}"
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
        assert!(
            is_permanent_config_error(&err),
            "a refused cluster credential must be permanent: {err}"
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
        assert!(
            is_permanent_config_error(&err),
            "a refused master credential must be permanent: {err}"
        );
    }
}
