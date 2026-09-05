//! Connecting to an etcd that really has authentication enabled.
//!
//! Bracketed by `ETCD_AUTH_TEST_URL` (the pattern
//! `crates/aisix-admin/tests/etcd_integration.rs` uses for
//! `ADMIN_TEST_ETCD_URL`): the tests no-op when it is unset so a local
//! `cargo test` without docker still passes. CI starts the cluster and
//! sets the variables in `.github/workflows/ci.yml`.
//!
//! Two things need a server that answers rather than a stub. The wording
//! and status code etcd really returns for a wrong password is what the
//! whole "refused, not unreachable" split rests on; and the recovery
//! path — an etcd that was unreachable at connect time coming back, being
//! authenticated on the retry, and the gateway applying a configuration
//! from it — is only meaningful end to end.

#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use aisix_etcd::{EtcdConfigProvider, ProviderError, Supervisor};
use etcd_client::ConnectOptions;

fn auth_etcd() -> Option<(String, String, String)> {
    let url = std::env::var("ETCD_AUTH_TEST_URL").ok()?;
    let user = std::env::var("ETCD_AUTH_TEST_USER").ok()?;
    let password = std::env::var("ETCD_AUTH_TEST_PASSWORD").ok()?;
    Some((url, user, password))
}

fn unique_prefix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("/aisix-auth-it-{nanos}")
}

fn host_port(url: &str) -> String {
    url.trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/')
        .to_string()
}

/// Accept on `listen` and pipe both directions to `target`, from the
/// moment this is called — so a test can leave the endpoint dead first
/// and bring it up afterwards.
async fn start_forwarder(listen: std::net::SocketAddr, target: String) {
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .expect("relay bind");
    tokio::spawn(async move {
        while let Ok((mut client, _)) = listener.accept().await {
            let target = target.clone();
            tokio::spawn(async move {
                let Ok(mut upstream) = tokio::net::TcpStream::connect(&target).await else {
                    return;
                };
                let _ = tokio::io::copy_bidirectional(&mut client, &mut upstream).await;
            });
        }
    });
}

/// A dead endpoint: bound to learn a free port, then released.
async fn dead_endpoint() -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("probe bind");
    let addr = listener.local_addr().expect("probe addr");
    drop(listener);
    addr
}

#[tokio::test]
async fn a_wrong_password_is_refused_by_the_real_server() {
    let Some((url, user, _password)) = auth_etcd() else {
        eprintln!("skipping: ETCD_AUTH_TEST_URL not set");
        return;
    };
    let options = ConnectOptions::new().with_user(user, "definitely-not-the-password");
    let err = EtcdConfigProvider::connect(
        &[url],
        unique_prefix(),
        Some(options),
        None,
        Some(Duration::from_secs(5)),
    )
    .await
    .expect_err("a wrong password cannot produce a usable provider");

    let ProviderError::Rejected(msg) = err else {
        panic!("a real etcd's refusal must be classified as refused, got {err:?}");
    };
    // etcd 3.5 answers `Authenticate` with InvalidArgument and this
    // message. The classification is on the status code, not on the
    // text, but the text is what an operator reads out of the boot
    // failure — so it has to survive to the error.
    assert!(
        msg.contains("authentication failed"),
        "the error must carry what etcd said: {msg}",
    );
}

#[tokio::test]
async fn an_etcd_that_comes_back_is_authenticated_and_applied() {
    let Some((url, user, password)) = auth_etcd() else {
        eprintln!("skipping: ETCD_AUTH_TEST_URL not set");
        return;
    };
    let prefix = unique_prefix();

    // Seed one model the way an operator or the control plane would,
    // before the gateway has anything to read it through.
    let mut writer = etcd_client::Client::connect(
        [url.clone()],
        Some(ConnectOptions::new().with_user(user.clone(), password.clone())),
    )
    .await
    .expect("seed client");
    writer
        .put(
            format!("{prefix}/models/m-auth-1"),
            r#"{"display_name":"auth-it-model","provider":"openai","model_name":"gpt-4o-mini","provider_key_id":"11111111-1111-1111-1111-111111111111"}"#,
            None,
        )
        .await
        .expect("seed put");

    // The gateway starts against an endpoint nothing is listening on.
    // With credentials configured this used to fail the connect and end
    // the process; it now leaves the connection pending.
    let addr = dead_endpoint().await;
    let provider = Arc::new(
        EtcdConfigProvider::connect(
            &[format!("http://{addr}")],
            prefix.clone(),
            Some(ConnectOptions::new().with_user(user, password)),
            Some(Duration::from_secs(5)),
            Some(Duration::from_secs(5)),
        )
        .await
        .expect("an unreachable etcd must not fail the connect"),
    );

    let supervisor = Arc::new(Supervisor::new(provider, prefix.clone()));
    let status = supervisor.config_status();
    let handle = supervisor.handle();
    let (_cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(supervisor.run(cancel_rx));

    // Nothing is applied while the source is away — which is what holds
    // the proxy listener closed for the whole of this window.
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert!(
        !status.is_ready(),
        "no configuration can be applied from an etcd that is not there",
    );

    // The source comes back. Nothing is restarted or rebuilt: the
    // supervisor's own retry loop dials again, authenticates, and applies.
    start_forwarder(addr, host_port(&url)).await;

    let deadline = Instant::now() + Duration::from_secs(60);
    while !status.is_ready() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(
        status.is_ready(),
        "the gateway must apply a configuration once etcd answers",
    );
    assert_eq!(
        handle.load().models.len(),
        1,
        "the applied snapshot must carry the seeded model",
    );

    writer
        .delete(format!("{prefix}/models/m-auth-1"), None)
        .await
        .expect("cleanup");
    task.abort();
}
