//! Periodic `POST /dp/heartbeat` so cp-api knows the DP is alive.
//!
//! Protocol (prd-09a §9A.7.2 + §9A.10A.3): the DP authenticates via
//! its mTLS client certificate. cp-api reads the peer cert SAN URI
//! (`x-aisix://env/<env_id>/dp/<dp_id>`) to derive identity — there
//! is no `Authorization` header on the v3 wire. The body still
//! carries `{ dp_id, uptime_seconds, version }` for diagnostics and
//! to keep parity with the legacy log shape.
//!
//! Shape:
//!   - spawned once from `main` after registration / bundle-on-disk
//!     load is complete
//!   - ticks at the interval returned by the register response
//!     (default 15s)
//!   - individual heartbeats fail fast on network errors; the ticker
//!     keeps running so a transient outage doesn't stop the DP from
//!     being seen when the CP comes back
//!   - cancelled via the shared `watch::Receiver<bool>` so graceful
//!     shutdown doesn't leave an in-flight request dangling

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context};
use serde::Serialize;
use tokio::sync::watch;

/// File paths to the on-disk mTLS bundle the heartbeat client presents
/// to cp-api. Same three files written by `register::register_and_persist`
/// (and re-used on every subsequent boot when the bundle is already on
/// disk).
///
/// `extra_ca_pem` is an optional second CA bundle the operator points
/// at via `managed.cp_ca_cert_file` — needed in e2e / on-prem
/// deployments where dp-manager's *server* cert is signed by a CA
/// distinct from the cert-manager-issued one (which only signs DP
/// *client* certs). When set, every outbound mTLS client built from
/// this bundle (heartbeat, telemetry, BudgetClient) appends it to
/// the verify chain. Production with public-CA certs leaves this
/// `None`.
#[derive(Debug, Clone)]
pub struct MtlsBundle {
    pub ca_cert_path: PathBuf,
    pub client_cert_path: PathBuf,
    pub client_key_path: PathBuf,
    pub extra_ca_pem: Option<Vec<u8>>,
}

/// Configuration captured at register time. `url`, `dp_id`, `interval`
/// come from the register response (or are synthesised on bundle-on-disk
/// boots); `mtls` points at the persisted bundle.
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    pub url: String,
    pub dp_id: String,
    pub interval: Duration,
    pub mtls: MtlsBundle,
}

impl HeartbeatConfig {
    /// Clamp the server-suggested interval into a safe band. Defence
    /// against a buggy CP config that returns 0 or a week.
    pub fn sanitised(url: String, dp_id: String, interval: Duration, mtls: MtlsBundle) -> Self {
        const MIN: Duration = Duration::from_secs(5);
        const MAX: Duration = Duration::from_secs(300);
        let interval = interval.clamp(MIN, MAX);
        Self {
            url,
            dp_id,
            interval,
            mtls,
        }
    }
}

/// Spawn the heartbeat worker. Returns the JoinHandle so `main` can
/// await it at shutdown. Errors during individual heartbeats are
/// logged, not propagated — a heartbeat that can't reach the CP is
/// noisy, not fatal.
pub fn spawn(
    cfg: HeartbeatConfig,
    mut cancel: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        run(cfg, &mut cancel).await;
    })
}

async fn run(cfg: HeartbeatConfig, cancel: &mut watch::Receiver<bool>) {
    let client = match build_client(&cfg.mtls) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            tracing::error!(error = %e, "heartbeat: build mTLS reqwest client failed; disabled");
            return;
        }
    };
    let started = Instant::now();
    let mut ticker = tokio::time::interval(cfg.interval);
    // Skip the catch-up fire — we want the first beat to happen
    // immediately at spawn but subsequent ones to follow the tick
    // schedule without bursting if we fall behind.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    tracing::info!(
        url = %cfg.url,
        dp_id = %cfg.dp_id,
        interval_secs = cfg.interval.as_secs(),
        "heartbeat started (mTLS)",
    );

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let uptime = started.elapsed().as_secs() as i64;
                match send(&client, &cfg, uptime).await {
                    Ok(()) => tracing::debug!("heartbeat ok"),
                    Err(e) => tracing::warn!(error = %e, "heartbeat failed"),
                }
            }
            _ = cancel.changed() => {
                if *cancel.borrow() {
                    tracing::info!("heartbeat shutting down");
                    return;
                }
            }
        }
    }
}

#[derive(Debug, Serialize)]
struct HeartbeatBody<'a> {
    dp_id: &'a str,
    uptime_seconds: i64,
    version: &'a str,
}

async fn send(client: &reqwest::Client, cfg: &HeartbeatConfig, uptime: i64) -> anyhow::Result<()> {
    let resp = client
        .post(&cfg.url)
        // No bearer header — cp-api authenticates via the peer
        // certificate SAN URI (§9A.7.2).
        .json(&HeartbeatBody {
            dp_id: &cfg.dp_id,
            uptime_seconds: uptime,
            version: env!("CARGO_PKG_VERSION"),
        })
        .send()
        .await
        .with_context(|| format!("POST {}", cfg.url))?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!(
            "heartbeat {} returned {} — {}",
            cfg.url,
            status,
            body.trim().chars().take(200).collect::<String>()
        ));
    }
    Ok(())
}

/// Build a reqwest client wired up with the on-disk mTLS bundle: the
/// CA from cp-api as a trust root, plus the DP's client cert + key as
/// the presenting identity. Files are read here (not at config-load
/// time) so an unreadable/rotated bundle surfaces an actionable error
/// at the same place every other heartbeat error does.
///
/// Public so the BudgetClient (aisix-proxy) and any future per-request
/// CP caller can reuse the same identity without duplicating the PEM-
/// loading dance. `aisix-server::telemetry` keeps its own copy because
/// it predates the extraction; consolidating later is fine.
pub fn build_mtls_client(mtls: &MtlsBundle) -> anyhow::Result<reqwest::Client> {
    build_client(mtls)
}

fn build_client(mtls: &MtlsBundle) -> anyhow::Result<reqwest::Client> {
    let ca_pem = std::fs::read(&mtls.ca_cert_path)
        .with_context(|| format!("read {}", mtls.ca_cert_path.display()))?;
    let cert_pem = std::fs::read(&mtls.client_cert_path)
        .with_context(|| format!("read {}", mtls.client_cert_path.display()))?;
    let key_pem = std::fs::read(&mtls.client_key_path)
        .with_context(|| format!("read {}", mtls.client_key_path.display()))?;

    // reqwest::Identity::from_pem expects a single PEM blob containing
    // BOTH the private key and the cert chain. Concatenate in that
    // order — rustls is order-tolerant but it's the convention.
    let mut identity_pem = Vec::with_capacity(cert_pem.len() + key_pem.len());
    identity_pem.extend_from_slice(&key_pem);
    identity_pem.extend_from_slice(&cert_pem);
    let identity = reqwest::Identity::from_pem(&identity_pem)
        .context("build mTLS Identity from client cert + key")?;

    let ca = reqwest::Certificate::from_pem(&ca_pem).context("parse CA certificate")?;

    let mut builder = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent(format!("aisix-dp/{}", env!("CARGO_PKG_VERSION")))
        .identity(identity)
        .add_root_certificate(ca)
        .use_rustls_tls();
    // Operator-supplied extra root (e2e / on-prem). Covers the dp-
    // manager server cert when it's signed by a CA distinct from the
    // cert-manager-issued client-cert CA.
    if let Some(extra) = mtls.extra_ca_pem.as_ref() {
        let extra_ca = reqwest::Certificate::from_pem(extra)
            .context("parse managed.cp_ca_cert_file as PEM certificate")?;
        builder = builder.add_root_certificate(extra_ca);
    }
    builder
        .build()
        .context("build reqwest client with mTLS")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rcgen::{CertificateParams, DistinguishedName, KeyPair, PKCS_ECDSA_P256_SHA256};
    use std::path::Path;
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Bundle on disk used by the build_client test: generates a real
    /// self-signed CA + leaf so reqwest's PEM parser actually accepts
    /// it. Lives in a TempDir that's dropped at end of test.
    fn write_test_bundle(dir: &Path) -> MtlsBundle {
        // Self-signed CA: subject = issuer = "aisix-test-ca".
        let ca_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.distinguished_name = {
            let mut dn = DistinguishedName::new();
            dn.push(rcgen::DnType::CommonName, "aisix-test-ca");
            dn
        };
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_kp).unwrap();

        // Leaf signed by the CA.
        let leaf_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut leaf_params = CertificateParams::new(vec!["dp-test".to_string()]).unwrap();
        leaf_params.distinguished_name = {
            let mut dn = DistinguishedName::new();
            dn.push(rcgen::DnType::CommonName, "dp-test");
            dn
        };
        let leaf_cert = leaf_params.signed_by(&leaf_kp, &ca_cert, &ca_kp).unwrap();

        let ca_path = dir.join("ca.crt");
        let cert_path = dir.join("client.crt");
        let key_path = dir.join("client.key");
        std::fs::write(&ca_path, ca_cert.pem()).unwrap();
        std::fs::write(&cert_path, leaf_cert.pem()).unwrap();
        std::fs::write(&key_path, leaf_kp.serialize_pem()).unwrap();

        MtlsBundle {
            ca_cert_path: ca_path,
            client_cert_path: cert_path,
            client_key_path: key_path,
            extra_ca_pem: None,
        }
    }

    fn cfg_with_bundle(url: String, mtls: MtlsBundle) -> HeartbeatConfig {
        HeartbeatConfig::sanitised(
            url,
            "dp_test_node_42".into(),
            Duration::from_millis(50),
            mtls,
        )
    }

    fn plain_client() -> reqwest::Client {
        // Plain HTTP client used by the wiremock-based send tests.
        // We don't want to drag the wiremock test through TLS termination
        // — the protocol-level assertions (body shape, HTTP error
        // propagation) are independent of the mTLS handshake.
        reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn send_omits_authorization_header_and_posts_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/heartbeat"))
            .and(body_string_contains("\"dp_id\":\"dp_test_node_42\""))
            .and(body_string_contains("\"uptime_seconds\":"))
            // Negative-match the Authorization header below via
            // `received_requests()` since wiremock has no built-in
            // header-absence matcher.
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let c = plain_client();
        send(
            &c,
            &cfg_with_bundle(format!("{}/dp/heartbeat", server.uri()), mtls),
            7,
        )
        .await
        .unwrap();

        // Inspect the recorded request to confirm no Authorization header.
        let received = server.received_requests().await.unwrap();
        let req = received.first().expect("expected one request");
        assert!(
            req.headers.get("authorization").is_none(),
            "v3 heartbeat MUST NOT carry Authorization header (mTLS-only auth)",
        );
    }

    #[tokio::test]
    async fn send_propagates_non_success_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/heartbeat"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": {"code": "DP_NOT_FOUND", "message": "no registered DP matches this id"}
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let c = plain_client();
        let err = send(
            &c,
            &cfg_with_bundle(format!("{}/dp/heartbeat", server.uri()), mtls),
            7,
        )
        .await
        .unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("404"), "expected status in error: {s}");
        assert!(s.contains("DP_NOT_FOUND"), "expected body in error: {s}");
    }

    #[tokio::test]
    async fn run_stops_on_cancel() {
        // Start a server that 200s fast enough that the first tick
        // completes, then cancel and make sure the task exits.
        // Bundle is real but the tick uses plain HTTP wiremock — the
        // mTLS client builder still has to succeed for `run` to enter
        // its loop, which is what this test exercises.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/heartbeat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let (tx, rx) = watch::channel(false);
        let handle = spawn(
            cfg_with_bundle(format!("{}/dp/heartbeat", server.uri()), mtls),
            rx,
        );

        tokio::time::sleep(Duration::from_millis(150)).await;
        tx.send(true).unwrap();

        // Runs to completion within a small grace window.
        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("heartbeat did not stop after cancel")
            .unwrap();
    }

    #[test]
    fn sanitised_interval_clamps_extremes() {
        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let a = HeartbeatConfig::sanitised(
            "http://x".into(),
            "id".into(),
            Duration::from_millis(10),
            mtls.clone(),
        );
        assert_eq!(a.interval, Duration::from_secs(5));

        let b = HeartbeatConfig::sanitised(
            "http://x".into(),
            "id".into(),
            Duration::from_secs(86_400),
            mtls.clone(),
        );
        assert_eq!(b.interval, Duration::from_secs(300));

        let c = HeartbeatConfig::sanitised(
            "http://x".into(),
            "id".into(),
            Duration::from_secs(30),
            mtls,
        );
        assert_eq!(c.interval, Duration::from_secs(30));
    }

    #[test]
    fn build_client_loads_real_mtls_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let _client = build_client(&mtls).expect("build_client should succeed with valid bundle");
    }

    #[test]
    fn build_client_surfaces_missing_files() {
        let mtls = MtlsBundle {
            ca_cert_path: "/definitely/missing/ca.crt".into(),
            client_cert_path: "/definitely/missing/client.crt".into(),
            client_key_path: "/definitely/missing/client.key".into(),
            extra_ca_pem: None,
        };
        let err = build_client(&mtls).unwrap_err();
        // The error must mention which file was missing — operators
        // should not have to diff config against filesystem state.
        assert!(
            err.to_string().contains("ca.crt"),
            "unexpected error: {err}"
        );
    }
}
