//! The AWS default credential chain reaches its link-local endpoints
//! directly even when the proxy environment would send it elsewhere.
//!
//! Driven here rather than through the gateway e2e suite because the
//! endpoints themselves (169.254.169.254, 169.254.170.2, …) cannot be bound
//! by a test: the client is built with a loopback host in their place,
//! through the same constructor production passes the real list to, and
//! the ECS credential provider is pointed at it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aws_config::BehaviorVersion;
use aws_credential_types::provider::ProvideCredentials as _;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpListener;

/// Answers every connection with `body` and counts the connections.
async fn serve(status: &'static str, body: &'static str) -> (u16, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = hits.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            counter.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let _ = sock.read(&mut buf).await;
                let reply = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(reply.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (port, hits)
}

async fn resolve(
    client: aws_smithy_runtime_api::client::http::SharedHttpClient,
) -> Result<aws_credential_types::Credentials, String> {
    let sdk = aws_config::defaults(BehaviorVersion::latest())
        .region(aws_config::Region::new("us-east-1"))
        .http_client(client)
        .load()
        .await;
    sdk.credentials_provider()
        .expect("the default chain")
        .provide_credentials()
        .await
        .map_err(|e| format!("{e:?}"))
}

#[tokio::test]
async fn credential_endpoints_bypass_the_proxy_environment() {
    let (creds_port, creds_hits) = serve(
        "200 OK",
        r#"{"AccessKeyId":"AKIDDIRECTCREDS00001","SecretAccessKey":"secret","Token":"token","Expiration":"2099-01-01T00:00:00Z"}"#,
    )
    .await;
    let (proxy_port, proxy_hits) = serve("502 Bad Gateway", "{}").await;

    // One test function, so no other test in this binary races these.
    for var in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "no_proxy",
        "http_proxy",
    ] {
        std::env::remove_var(var);
    }
    std::env::set_var("HTTP_PROXY", format!("http://127.0.0.1:{proxy_port}"));
    // Set, and not covering the credential host.
    std::env::set_var("NO_PROXY", "example.invalid");
    std::env::set_var(
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        format!("http://127.0.0.1:{creds_port}/credentials"),
    );
    std::env::set_var("AWS_EC2_METADATA_DISABLED", "true");
    std::env::set_var("AWS_CONFIG_FILE", "/nonexistent/aisix-test-aws-config");
    std::env::set_var(
        "AWS_SHARED_CREDENTIALS_FILE",
        "/nonexistent/aisix-test-aws-credentials",
    );

    // Control: without the host on the direct list, the proxy environment
    // takes the credential call — so the proxy is really in force here.
    let proxied = resolve(aisix_gateway::upstream_tls::build_aws_http_client_direct_to(&[])).await;
    assert!(
        proxied.is_err(),
        "the proxy answers 502, not credentials: {proxied:?}"
    );
    assert!(
        proxy_hits.load(Ordering::SeqCst) > 0,
        "the control call never reached the proxy"
    );
    assert_eq!(creds_hits.load(Ordering::SeqCst), 0);

    let proxy_before = proxy_hits.load(Ordering::SeqCst);
    let creds =
        resolve(aisix_gateway::upstream_tls::build_aws_http_client_direct_to(&["127.0.0.1"]))
            .await
            .expect("credentials from the endpoint, reached directly");
    assert_eq!(creds.access_key_id(), "AKIDDIRECTCREDS00001");
    assert!(creds_hits.load(Ordering::SeqCst) > 0);
    assert_eq!(
        proxy_hits.load(Ordering::SeqCst),
        proxy_before,
        "a credential-endpoint call reached the proxy"
    );
}
