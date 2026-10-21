//! The AWS default credential chain reaches its credential endpoints
//! directly while the proxy environment is in force, and its STS calls
//! still go through the proxy.
//!
//! Driven here rather than through the gateway e2e suite because the fixed
//! endpoints (169.254.169.254, 169.254.170.2, …) cannot be bound by a test.
//! What a test can bind is the two configurable cases, driven through the
//! production credential path (`aws_auth::resolve`): a local credential
//! agent on loopback named by `AWS_CONTAINER_CREDENTIALS_FULL_URI`, and a
//! custom instance-metadata endpoint on this machine's own non-loopback
//! address named by `AWS_EC2_METADATA_SERVICE_ENDPOINT`. Matching of the
//! fixed addresses is pinned by a unit test in `upstream_tls`.

use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aisix_gateway::aws_auth::{self, AssumeRole, AwsAuth, AwsAuthMode, AwsAuthSpec};
use aws_credential_types::provider::ProvideCredentials as _;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpListener;

type Responder = fn(&str) -> (&'static str, Vec<(&'static str, &'static str)>, String);

/// Answers each request with `respond(request line)` and counts requests.
async fn serve(bind: IpAddr, respond: Responder) -> (SocketAddr, Arc<AtomicUsize>) {
    let listener = TcpListener::bind((bind, 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = hits.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            counter.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                let mut buf = vec![0u8; 16384];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]);
                let line = head.lines().next().unwrap_or_default().to_string();
                let (status, headers, body) = respond(&line);
                let mut reply = format!(
                    "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n",
                    body.len()
                );
                for (k, v) in headers {
                    reply.push_str(&format!("{k}: {v}\r\n"));
                }
                reply.push_str("\r\n");
                reply.push_str(&body);
                let _ = sock.write_all(reply.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (addr, hits)
}

const CREDS: &str = r#"{"Code":"Success","AccessKeyId":"AKIDDIRECTCREDS00001","SecretAccessKey":"secret","Token":"token","Expiration":"2099-01-01T00:00:00Z"}"#;

fn credential_agent(_: &str) -> (&'static str, Vec<(&'static str, &'static str)>, String) {
    (
        "200 OK",
        vec![("content-type", "application/json")],
        CREDS.into(),
    )
}

fn imds(line: &str) -> (&'static str, Vec<(&'static str, &'static str)>, String) {
    if line.starts_with("PUT /latest/api/token") {
        return (
            "200 OK",
            vec![("x-aws-ec2-metadata-token-ttl-seconds", "21600")],
            "imds-token".into(),
        );
    }
    if line.contains("/latest/meta-data/iam/security-credentials/ ")
        || line.ends_with("/latest/meta-data/iam/security-credentials/")
        || line.contains("/latest/meta-data/iam/security-credentials/ HTTP")
    {
        return ("200 OK", vec![], "instance-role".into());
    }
    if line.contains("/latest/meta-data/iam/security-credentials/instance-role") {
        return (
            "200 OK",
            vec![("content-type", "application/json")],
            CREDS.into(),
        );
    }
    ("404 Not Found", vec![], String::new())
}

fn refusing_proxy(_: &str) -> (&'static str, Vec<(&'static str, &'static str)>, String) {
    ("502 Bad Gateway", vec![], "{}".into())
}

/// This machine's own address on its default route — not loopback, and so
/// not exempted for being loopback. No packet is sent.
fn own_address() -> Option<IpAddr> {
    let sock = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("192.0.2.1:9").ok()?;
    let ip = sock.local_addr().ok()?.ip();
    (!ip.is_loopback() && !ip.is_unspecified()).then_some(ip)
}

async fn credentials(spec: AwsAuthSpec) -> Result<aws_credential_types::Credentials, String> {
    let AwsAuth::SigV4(provider) = aws_auth::resolve(&spec) else {
        panic!("a default-chain spec signs");
    };
    provider
        .provide_credentials()
        .await
        .map_err(|e| format!("{e:?}"))
}

fn default_chain(region: &str, assume_role: Option<AssumeRole>) -> AwsAuthSpec {
    AwsAuthSpec {
        mode: AwsAuthMode::DefaultChain,
        assume_role,
        region: region.into(),
    }
}

#[tokio::test]
async fn credential_endpoints_bypass_the_proxy_environment_and_sts_does_not() {
    let own = own_address().expect("a non-loopback address on the default route");
    let (proxy, proxy_hits) = serve(IpAddr::from([127, 0, 0, 1]), refusing_proxy).await;
    let (imds_addr, imds_hits) = serve(own, imds).await;
    let (agent_addr, agent_hits) = serve(IpAddr::from([127, 0, 0, 1]), credential_agent).await;
    let (sts_addr, sts_hits) = serve(own, refusing_proxy).await;

    // One test function, so nothing else in this binary races these.
    for var in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_EC2_METADATA_DISABLED",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "http_proxy",
        "no_proxy",
    ] {
        std::env::remove_var(var);
    }
    std::env::set_var("HTTP_PROXY", format!("http://{proxy}"));
    // Set, and covering none of the endpoints below.
    std::env::set_var("NO_PROXY", "example.invalid");
    std::env::set_var("AWS_CONFIG_FILE", "/nonexistent/aisix-test-aws-config");
    std::env::set_var(
        "AWS_SHARED_CREDENTIALS_FILE",
        "/nonexistent/aisix-test-aws-credentials",
    );
    std::env::set_var(
        "AWS_EC2_METADATA_SERVICE_ENDPOINT",
        format!("http://{imds_addr}"),
    );
    std::env::set_var("AWS_ENDPOINT_URL_STS", format!("http://{sts_addr}"));

    // A custom instance-metadata endpoint is reached directly.
    let creds = credentials(default_chain("us-east-1", None))
        .await
        .expect("instance credentials, reached directly");
    assert_eq!(creds.access_key_id(), "AKIDDIRECTCREDS00001");
    assert!(imds_hits.load(Ordering::SeqCst) > 0);
    assert_eq!(
        proxy_hits.load(Ordering::SeqCst),
        0,
        "an instance-metadata call reached the proxy"
    );

    // The chain's STS call is the proxy's: the proxy is really in force,
    // and only the credential endpoints are exempt from it.
    let role = AssumeRole {
        role_arn: "arn:aws:iam::111122223333:role/ViaProxy".into(),
        external_id: None,
        role_session_name: None,
    };
    let assumed = credentials(default_chain("us-west-2", Some(role))).await;
    assert!(assumed.is_err(), "the proxy refuses STS: {assumed:?}");
    assert!(
        proxy_hits.load(Ordering::SeqCst) > 0,
        "the STS call did not go through the proxy"
    );
    assert_eq!(
        sts_hits.load(Ordering::SeqCst),
        0,
        "STS was reached directly"
    );

    // A local credential agent on loopback is reached directly.
    std::env::set_var(
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        format!("http://{agent_addr}/v1/credentials"),
    );
    let proxy_before = proxy_hits.load(Ordering::SeqCst);
    let creds = credentials(default_chain("eu-west-1", None))
        .await
        .expect("container credentials, reached directly");
    assert_eq!(creds.access_key_id(), "AKIDDIRECTCREDS00001");
    assert!(agent_hits.load(Ordering::SeqCst) > 0);
    assert_eq!(
        proxy_hits.load(Ordering::SeqCst),
        proxy_before,
        "a container-credentials call reached the proxy"
    );
}
