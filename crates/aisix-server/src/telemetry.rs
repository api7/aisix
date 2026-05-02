//! Sender worker for the CP-side `/dp/telemetry` surface.
//!
//! Per prd-09a §9A.7B Phase 1:
//!
//! - Proxy handlers call [`UsageSink::try_emit`] (defined in aisix-obs)
//!   to push one event per chat completion onto an mpsc channel.
//! - This worker drains the channel, batches up to [`MAX_BATCH`]
//!   events or every [`FLUSH_INTERVAL`] (whichever fires first),
//!   POSTs the batch as `{ events: [...] }` to the CP's
//!   `/dp/telemetry` URL, and logs the outcome.
//! - On HTTP error the batch is dropped (NOT retried). Phase 1
//!   accepts a small loss window in exchange for not building a
//!   persistent disk queue. The `received_at` column on the cp-api
//!   side records when CP saw the row, so dashboards distinguish
//!   "DP never sent" from "DP sent but CP rejected" via log
//!   correlation.
//!
//! mTLS: the sender presents the same on-disk bundle the heartbeat
//! worker uses (the bundle persisted by `register::register_and_persist`
//! at first boot). cp-api derives `env_id` and `dp_id` from the peer
//! cert SAN URI, so the request body doesn't carry them — same wire
//! shape as `/dp/heartbeat`.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context};
use serde::Serialize;
use tokio::sync::watch;

use aisix_obs::{UsageEvent, UsageSink};

use crate::heartbeat::MtlsBundle;

/// Maximum number of events accumulated per outbound POST. Pinned in
/// code rather than config — at >100 events/batch the request body
/// approaches gin's default `MaxMultipartMemory` plumbing on the
/// receiving side, and we'd rather flush more often than tune that.
const MAX_BATCH: usize = 100;

/// Cadence at which the worker flushes whatever it has buffered, even
/// if the buffer hasn't filled. Keeps fresh requests visible on
/// /usage and /logs within ~5s end-to-end.
const FLUSH_INTERVAL: Duration = Duration::from_secs(5);

/// In-memory bound on the proxy → worker channel. At 1024 events,
/// 5s flush, 100 batch ceiling, the proxy can sustain a sustained
/// 200 req/s burst without dropping. Beyond that `try_emit` warns
/// and the event is dropped — telemetry must not back-pressure the
/// request hot path.
const QUEUE_CAPACITY: usize = 1024;

/// Configuration for the sender. Mirrors `HeartbeatConfig` — the URL
/// is the absolute `/dp/telemetry` endpoint on cp-api, the bundle is
/// the on-disk mTLS material persisted by register, and `interval`
/// is the flush cadence (kept overridable so tests can speed up).
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub url: String,
    pub interval: Duration,
    pub mtls: MtlsBundle,
}

impl TelemetryConfig {
    /// Build with default flush interval (5s).
    pub fn new(url: String, mtls: MtlsBundle) -> Self {
        Self {
            url,
            interval: FLUSH_INTERVAL,
            mtls,
        }
    }
}

/// Spawn the worker. Returns:
///   - a [`UsageSink`] the proxy uses to enqueue events;
///   - a [`tokio::task::JoinHandle`] the caller awaits at shutdown
///     so the final in-flight batch drains cleanly.
///
/// The worker stops when `cancel` flips to `true` AND the channel is
/// drained (one final flush so we don't lose the tail). Errors during
/// individual flushes are logged, not propagated — same contract as
/// heartbeat::spawn.
pub fn spawn(
    cfg: TelemetryConfig,
    mut cancel: watch::Receiver<bool>,
) -> (UsageSink, tokio::task::JoinHandle<()>) {
    let (tx, rx) = tokio::sync::mpsc::channel(QUEUE_CAPACITY);
    let sink = UsageSink::new(tx);
    let handle = tokio::spawn(async move {
        run(cfg, rx, &mut cancel).await;
    });
    (sink, handle)
}

async fn run(
    cfg: TelemetryConfig,
    mut rx: tokio::sync::mpsc::Receiver<UsageEvent>,
    cancel: &mut watch::Receiver<bool>,
) {
    let client = match build_client(&cfg.mtls) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            tracing::error!(error = %e, "telemetry: build mTLS client failed; worker disabled");
            return;
        }
    };
    let mut ticker = tokio::time::interval(cfg.interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut buffer: Vec<UsageEvent> = Vec::with_capacity(MAX_BATCH);

    tracing::info!(
        url = %cfg.url,
        flush_interval_secs = cfg.interval.as_secs(),
        max_batch = MAX_BATCH,
        "telemetry sender started (mTLS)",
    );

    loop {
        tokio::select! {
            // New event from the proxy. Buffer it; flush if we hit
            // the batch ceiling so a steady high-throughput stream
            // doesn't starve cp-api on a 5s cadence.
            maybe_event = rx.recv() => {
                match maybe_event {
                    Some(event) => {
                        buffer.push(event);
                        if buffer.len() >= MAX_BATCH {
                            flush(&client, &cfg, &mut buffer).await;
                        }
                    }
                    None => {
                        // All senders dropped — proxy is shutting down.
                        // Drain anything left and exit.
                        if !buffer.is_empty() {
                            flush(&client, &cfg, &mut buffer).await;
                        }
                        tracing::info!("telemetry sender: channel closed, exiting");
                        return;
                    }
                }
            }
            _ = ticker.tick() => {
                if !buffer.is_empty() {
                    flush(&client, &cfg, &mut buffer).await;
                }
            }
            _ = cancel.changed() => {
                if *cancel.borrow() {
                    // Final drain — try to grab whatever is still in
                    // the channel without blocking, then post it.
                    while let Ok(ev) = rx.try_recv() {
                        buffer.push(ev);
                    }
                    if !buffer.is_empty() {
                        flush(&client, &cfg, &mut buffer).await;
                    }
                    tracing::info!("telemetry sender shutting down");
                    return;
                }
            }
        }
    }
}

/// POST one batch and clear the buffer. Errors are logged, not
/// propagated — telemetry losses must not stall the worker.
async fn flush(client: &reqwest::Client, cfg: &TelemetryConfig, buffer: &mut Vec<UsageEvent>) {
    if buffer.is_empty() {
        return;
    }
    let count = buffer.len();
    // Move events out into the request body; clear the buffer
    // unconditionally so a hung CP doesn't grow the buffer
    // indefinitely (worst case we drop the batch on error).
    let events: Vec<UsageEvent> = std::mem::take(buffer);
    buffer.reserve(MAX_BATCH);

    match send(client, cfg, &events).await {
        Ok(()) => tracing::debug!(count, "telemetry batch flushed"),
        Err(e) => tracing::warn!(count, error = %e, "telemetry batch failed (events dropped)"),
    }
}

#[derive(Debug, Serialize)]
struct TelemetryBody<'a> {
    events: &'a [UsageEvent],
}

async fn send(
    client: &reqwest::Client,
    cfg: &TelemetryConfig,
    events: &[UsageEvent],
) -> anyhow::Result<()> {
    let resp = client
        .post(&cfg.url)
        // Same as /dp/heartbeat — no Authorization header; cp-api
        // derives identity from the peer cert SAN URI.
        .json(&TelemetryBody { events })
        .send()
        .await
        .with_context(|| format!("POST {}", cfg.url))?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!(
            "telemetry {} returned {} — {}",
            cfg.url,
            status,
            body.trim().chars().take(200).collect::<String>()
        ));
    }
    Ok(())
}

/// Build the mTLS reqwest client. Identical shape to heartbeat's
/// build_client — extracted out of heartbeat.rs would be nicer but
/// pulling that into a shared module is post-MVP polish and would
/// expand this PR's scope. The two clients are independently
/// constructed so a botched bundle on one path doesn't cascade.
fn build_client(mtls: &MtlsBundle) -> anyhow::Result<reqwest::Client> {
    let ca_pem = std::fs::read(&mtls.ca_cert_path)
        .with_context(|| format!("read {}", mtls.ca_cert_path.display()))?;
    let cert_pem = std::fs::read(&mtls.client_cert_path)
        .with_context(|| format!("read {}", mtls.client_cert_path.display()))?;
    let key_pem = std::fs::read(&mtls.client_key_path)
        .with_context(|| format!("read {}", mtls.client_key_path.display()))?;

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
    // Mirror heartbeat::build_client — pick up the operator-supplied
    // extra trust root (managed.cp_ca_cert_file) when set.
    if let Some(extra) = mtls.extra_ca_pem.as_ref() {
        let extra_ca = reqwest::Certificate::from_pem(extra)
            .context("parse managed.cp_ca_cert_file as PEM certificate")?;
        builder = builder.add_root_certificate(extra_ca);
    }
    builder.build().context("build reqwest client with mTLS")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rcgen::{CertificateParams, DistinguishedName, KeyPair, PKCS_ECDSA_P256_SHA256};
    use std::path::Path;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn write_test_bundle(dir: &Path) -> MtlsBundle {
        let ca_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.distinguished_name = {
            let mut dn = DistinguishedName::new();
            dn.push(rcgen::DnType::CommonName, "aisix-test-ca");
            dn
        };
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_kp).unwrap();

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

    fn sample_event(id: &str) -> UsageEvent {
        UsageEvent {
            request_id: id.into(),
            occurred_at: "2026-04-29T12:00:00Z".into(),
            model_id: "mod-uuid".into(),
            api_key_id: "ak-uuid".into(),
            prompt_tokens: 10,
            completion_tokens: 20,
            latency_ms: 30,
            status_code: 200,
            cost_usd: 0.001,
            guardrail_blocked: false,
            ..Default::default()
        }
    }

    fn plain_client() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn send_posts_events_array_with_no_authorization() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/telemetry"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "accepted": 2
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let cfg = TelemetryConfig::new(format!("{}/dp/telemetry", server.uri()), mtls);
        let events = vec![sample_event("req-1"), sample_event("req-2")];

        send(&plain_client(), &cfg, &events).await.unwrap();

        let received = server.received_requests().await.unwrap();
        let req = received.first().unwrap();
        // v3 telemetry MUST NOT carry Authorization — mTLS only.
        assert!(req.headers.get("authorization").is_none());
        // Body wraps the array in `{ events: [...] }`.
        let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(body["events"].as_array().unwrap().len(), 2);
        assert_eq!(body["events"][0]["request_id"], "req-1");
    }

    #[tokio::test]
    async fn send_propagates_non_success_with_body_excerpt() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/telemetry"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": {"code": "INVALID_REQUEST", "message": "event 0: bad uuid"}
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let cfg = TelemetryConfig::new(format!("{}/dp/telemetry", server.uri()), mtls);
        let err = send(&plain_client(), &cfg, &[sample_event("req-1")])
            .await
            .unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("400"), "expected status: {s}");
        assert!(s.contains("INVALID_REQUEST"), "expected body excerpt: {s}");
    }

    #[test]
    fn build_client_loads_real_mtls_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let _ = build_client(&mtls).expect("real bundle should build");
    }
}
