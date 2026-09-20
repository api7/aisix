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
//! worker uses. cp-api derives `env_id` and `dp_id` from the peer
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

/// In-memory bound on the proxy → worker channel. Beyond it `try_emit`
/// warns and the event is dropped (`sink_full`) — telemetry must not
/// back-pressure the request hot path.
///
/// Sized for the control plane being SLOW rather than for the steady
/// rate: one POST is in flight at a time, so while cp-api takes seconds
/// per batch the queue is the only thing holding the traffic that
/// arrives meanwhile, and a batch this worker drops is gone for good
/// (there is no retry). A stability round measured 1.1–1.8s per batch
/// for 8s at 173 req/s — ~1.4k events behind a queue that held 1024, so
/// 793 were dropped. At 16384 the same stall is absorbed whole, and the
/// queue only overflows once a sustained arrival rate outruns delivery
/// for minutes rather than seconds.
///
/// Cost is bounded by occupancy, not by the bound: the channel allocates
/// in small blocks as events are pushed, so a queue that never fills
/// never holds the memory for one that did.
const QUEUE_CAPACITY: usize = 16_384;

/// Path the telemetry worker POSTs to, under `managed.cp_base_url`.
/// Derived from the heartbeat URL by swapping the suffix, so the two
/// stay in lock-step on a `cp_base_url` change.
pub const TELEMETRY_PATH: &str = "/dp/telemetry";

/// Configuration for the sender. Mirrors `HeartbeatConfig` — the URL
/// is the absolute `/dp/telemetry` endpoint on cp-api, the bundle is
/// the externally provisioned on-disk mTLS material, and `interval`
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
                    fill_ready_batch(&mut buffer, &mut rx);
                    flush(&client, &cfg, &mut buffer).await;
                }
            }
            _ = cancel.changed() => {
                if *cancel.borrow() {
                    // Final drain — post whatever is still queued, in
                    // batches of at most MAX_BATCH. This is the one path
                    // that can meet a FULL queue (every other flush
                    // happens at or below the ceiling), and one POST of
                    // everything queued is exactly what the ceiling
                    // exists to prevent: cp-api rejects an oversized body
                    // and the whole backlog is gone, unretried.
                    loop {
                        fill_ready_batch(&mut buffer, &mut rx);
                        if buffer.is_empty() {
                            break;
                        }
                        flush(&client, &cfg, &mut buffer).await;
                    }
                    tracing::info!("telemetry sender shutting down");
                    return;
                }
            }
        }
    }
}

fn fill_ready_batch(
    buffer: &mut Vec<UsageEvent>,
    rx: &mut tokio::sync::mpsc::Receiver<UsageEvent>,
) {
    while buffer.len() < MAX_BATCH {
        match rx.try_recv() {
            Ok(event) => buffer.push(event),
            Err(_) => break,
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

    // Ensure a newline separates the two PEM blocks (see heartbeat.rs).
    let mut identity_pem = Vec::with_capacity(cert_pem.len() + key_pem.len() + 1);
    identity_pem.extend_from_slice(&key_pem);
    if !key_pem.ends_with(b"\n") {
        identity_pem.push(b'\n');
    }
    identity_pem.extend_from_slice(&cert_pem);
    let identity = reqwest::Identity::from_pem(&identity_pem)
        .context("build mTLS Identity from client cert + key")?;

    let ca = reqwest::Certificate::from_pem(&ca_pem).context("parse CA certificate")?;

    let mut builder = aisix_gateway::client_builder()
        .timeout(Duration::from_secs(10))
        .user_agent(format!("aisix-dp/{}", &*crate::heartbeat::BUILD_VERSION))
        .identity(identity)
        .add_root_certificate(ca)
        // Pin HTTP/1.1 — see heartbeat::build_client. dp-manager cmux
        // routes one TLS port to gRPC (h2) vs REST (http1) by ALPN; once
        // the cloud-sink crates pulled reqwest's `http2` feature into the
        // workspace, this telemetry client advertised `h2` and cmux
        // misrouted the /dp/telemetry POSTs to the gRPC handler.
        .http1_only()
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
            upstream_latency_ms: 30,
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

    /// Mirror heartbeat regression: PEM without trailing newline.
    #[test]
    fn build_client_works_without_trailing_newlines() {
        let dir = tempfile::tempdir().unwrap();

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

        let ca_path = dir.path().join("ca.crt");
        let cert_path = dir.path().join("client.crt");
        let key_path = dir.path().join("client.key");
        std::fs::write(&ca_path, ca_cert.pem().trim_end()).unwrap();
        std::fs::write(&cert_path, leaf_cert.pem().trim_end()).unwrap();
        std::fs::write(&key_path, leaf_kp.serialize_pem().trim_end()).unwrap();

        let mtls = MtlsBundle {
            ca_cert_path: ca_path,
            client_cert_path: cert_path,
            client_key_path: key_path,
            extra_ca_pem: None,
        };
        build_client(&mtls).expect("build_client must tolerate PEM without trailing newline");
    }
    type RecordedBatches = Arc<std::sync::Mutex<Vec<serde_json::Value>>>;

    async fn recording_server(first_status: u16) -> (MockServer, RecordedBatches) {
        let server = MockServer::start().await;
        let batches = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = Arc::clone(&batches);
        Mock::given(method("POST"))
            .and(path("/dp/telemetry"))
            .respond_with(move |request: &wiremock::Request| {
                assert!(request.headers.get("authorization").is_none());
                let mut batches = recorded.lock().unwrap();
                batches.push(serde_json::from_slice(&request.body).unwrap());
                ResponseTemplate::new(if batches.len() == 1 {
                    first_status
                } else {
                    200
                })
            })
            .mount(&server)
            .await;
        (server, batches)
    }

    /// [`recording_server`] that holds its FIRST response for `held`,
    /// leaving the worker inside one flush while a test stages what
    /// arrives behind it. The body is recorded before the wait, so the
    /// test can tell "in flight" from "not sent yet".
    async fn recording_server_holding_first(held: Duration) -> (MockServer, RecordedBatches) {
        let server = MockServer::start().await;
        let batches = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = Arc::clone(&batches);
        Mock::given(method("POST"))
            .and(path("/dp/telemetry"))
            .respond_with(move |request: &wiremock::Request| {
                let mut batches = recorded.lock().unwrap();
                batches.push(serde_json::from_slice(&request.body).unwrap());
                let response = ResponseTemplate::new(200);
                if batches.len() == 1 {
                    response.set_delay(held)
                } else {
                    response
                }
            })
            .mount(&server)
            .await;
        (server, batches)
    }

    fn ordered_events(count: usize) -> Vec<UsageEvent> {
        (0..count)
            .map(|i| {
                // Distinct attempts may share a request ID; none may be deduplicated.
                let mut event = sample_event(&format!("request-{}", i / 3));
                event.model_id = format!("model-{}", i % 7);
                event.provider_kind = format!("provider-{}", i % 3);
                event.user_id = format!("user-{i}");
                event.prompt_tokens = i as u32;
                event.completion_tokens = (i * 2) as u32;
                event
            })
            .collect()
    }

    async fn poll_sender(
        mut sender: std::pin::Pin<&mut impl std::future::Future<Output = ()>>,
    ) -> std::task::Poll<()> {
        std::future::poll_fn(|cx| std::task::Poll::Ready(sender.as_mut().poll(cx))).await
    }

    async fn finish_sender(mut sender: std::pin::Pin<&mut impl std::future::Future<Output = ()>>) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while poll_sender(sender.as_mut()).await.is_pending() {
            assert!(std::time::Instant::now() < deadline, "sender did not exit");
            // Keep the paused runtime runnable while real HTTP I/O completes.
            tokio::task::yield_now().await;
        }
    }

    fn assert_recorded_events(batches: &RecordedBatches, expected: &[UsageEvent]) -> Vec<usize> {
        let batches = batches.lock().unwrap();
        let actual: Vec<_> = batches
            .iter()
            .flat_map(|batch| batch["events"].as_array().unwrap().iter().cloned())
            .collect();
        assert_eq!(
            serde_json::json!(actual),
            serde_json::to_value(expected).unwrap()
        );
        batches
            .iter()
            .map(|batch| batch["events"].as_array().unwrap().len())
            .collect()
    }

    /// Channel bound for the backlog cases below. They are about what the
    /// worker does with a queue it has filled, not about how deep the real
    /// one is, so they size their own channel rather than staging
    /// [`QUEUE_CAPACITY`] events to fill it.
    const BACKLOG_QUEUE: usize = 1024;

    async fn run_interval_with_backlog(first_status: u16) {
        let (server, batches) = recording_server(first_status).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = TelemetryConfig::new(
            format!("{}/dp/telemetry", server.uri()),
            write_test_bundle(dir.path()),
        );
        let (tx, rx) = tokio::sync::mpsc::channel(BACKLOG_QUEUE);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        let expected = ordered_events(BACKLOG_QUEUE + 2);
        let mut events = ordered_events(BACKLOG_QUEUE + 2).into_iter();
        tokio::time::pause();
        let mut sender = Box::pin(run(cfg, rx, &mut cancel_rx));
        // Consume the initial empty tick before staging a partial batch.
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        for event in events.by_ref().take(2) {
            tx.try_send(event).unwrap();
        }
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        assert_eq!(tx.capacity(), BACKLOG_QUEUE);
        assert!(batches.lock().unwrap().is_empty());
        for event in events {
            tx.try_send(event).unwrap();
        }
        assert_eq!(tx.capacity(), 0);
        tokio::time::advance(FLUSH_INTERVAL).await;

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while batches.lock().unwrap().is_empty() {
            assert!(poll_sender(sender.as_mut()).await.is_pending());
            assert!(std::time::Instant::now() < deadline, "no telemetry POST");
            tokio::task::yield_now().await;
        }
        // Both recv and the tick are ready: this must hold whichever wins.
        assert_eq!(
            batches.lock().unwrap()[0]["events"]
                .as_array()
                .unwrap()
                .len(),
            MAX_BATCH
        );
        drop(tx);
        finish_sender(sender.as_mut()).await;
        let sizes = assert_recorded_events(&batches, &expected);
        assert_eq!(sizes, [vec![MAX_BATCH; 10], vec![26]].concat());
    }

    #[tokio::test]
    async fn interval_fills_ready_backlog_without_reordering_events() {
        run_interval_with_backlog(200).await;
    }

    #[tokio::test]
    async fn failed_interval_batch_is_not_retried_or_carried_into_next_batch() {
        run_interval_with_backlog(500).await;
    }

    #[tokio::test]
    async fn interval_flushes_partial_batch_without_waiting_for_more_events() {
        let (server, batches) = recording_server(200).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = TelemetryConfig::new(
            format!("{}/dp/telemetry", server.uri()),
            write_test_bundle(dir.path()),
        );
        let (tx, rx) = tokio::sync::mpsc::channel(QUEUE_CAPACITY);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let mut sender = Box::pin(run(cfg, rx, &mut cancel_rx));
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        for event in ordered_events(2) {
            tx.try_send(event).unwrap();
        }
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        assert_eq!(tx.capacity(), QUEUE_CAPACITY);
        tokio::time::advance(FLUSH_INTERVAL - Duration::from_millis(1)).await;
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        assert!(batches.lock().unwrap().is_empty());
        tokio::time::advance(Duration::from_millis(1)).await;
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while batches.lock().unwrap().is_empty() {
            assert!(poll_sender(sender.as_mut()).await.is_pending());
            assert!(
                std::time::Instant::now() < deadline,
                "partial batch was not flushed"
            );
            tokio::task::yield_now().await;
        }
        drop(tx);
        finish_sender(sender.as_mut()).await;
        assert_eq!(
            assert_recorded_events(&batches, &ordered_events(2)),
            vec![2]
        );
    }

    #[tokio::test]
    async fn cancellation_drains_buffer_and_ready_queue_without_losing_attempts() {
        let (server, batches) = recording_server(200).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = TelemetryConfig::new(
            format!("{}/dp/telemetry", server.uri()),
            write_test_bundle(dir.path()),
        );
        let (tx, rx) = tokio::sync::mpsc::channel(QUEUE_CAPACITY);
        let (cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let mut sender = Box::pin(run(cfg, rx, &mut cancel_rx));
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        let mut events = ordered_events(250).into_iter();
        for event in events.by_ref().take(2) {
            tx.try_send(event).unwrap();
        }
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        assert_eq!(tx.capacity(), QUEUE_CAPACITY);
        for event in events {
            tx.try_send(event).unwrap();
        }
        cancel_tx.send(true).unwrap();
        // Keep tx alive: completion must come from cancellation, not channel closure.
        finish_sender(sender.as_mut()).await;
        assert_recorded_events(&batches, &ordered_events(250));
        assert!(tx.is_closed());
    }

    /// The shutdown drain is the one flush that can meet a FULL queue —
    /// every other one happens at or below MAX_BATCH — and one POST of
    /// everything queued is what the ceiling exists to prevent: cp-api
    /// rejects an oversized body, and a rejected batch is not retried, so
    /// the whole backlog would go at once.
    #[tokio::test]
    async fn cancellation_posts_the_backlog_in_batches_not_in_one_body() {
        const QUEUED: usize = MAX_BATCH * 3 + 7;

        let (server, batches) = recording_server(200).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = TelemetryConfig::new(
            format!("{}/dp/telemetry", server.uri()),
            write_test_bundle(dir.path()),
        );
        let (tx, rx) = tokio::sync::mpsc::channel(QUEUED);
        let (cancel_tx, mut cancel_rx) = watch::channel(false);
        let mut sender = Box::pin(run(cfg, rx, &mut cancel_rx));
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        for event in ordered_events(QUEUED) {
            tx.try_send(event).unwrap();
        }
        cancel_tx.send(true).unwrap();
        finish_sender(sender.as_mut()).await;

        let sizes = assert_recorded_events(&batches, &ordered_events(QUEUED));
        assert!(
            sizes.iter().all(|n| *n <= MAX_BATCH),
            "no POST may carry more than the batch ceiling: {sizes:?}"
        );
    }

    /// A batch this worker gives up on is gone — there is no retry — so the
    /// queue is the whole defence against a control plane that has gone
    /// slow. One POST is in flight at a time, and everything the proxy
    /// emits meanwhile has to fit.
    ///
    /// 1550 events is the shape a stability round measured behind an 8s
    /// control-plane stall at 173 req/s: the queue that held 1024 dropped
    /// 793 of them.
    #[tokio::test]
    async fn a_burst_arriving_while_one_post_is_in_flight_is_not_dropped() {
        // Staged first, to get the worker into a POST. The mock records a
        // request before it answers, so seeing the batch means the flush
        // is in flight rather than finished.
        const OPENING: usize = 150;
        const BURST: usize = 1_400;
        // Outlasts the staging below, which is a few microseconds of
        // non-blocking sends.
        const HELD: Duration = Duration::from_secs(5);

        let (server, batches) = recording_server_holding_first(HELD).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = TelemetryConfig::new(
            format!("{}/dp/telemetry", server.uri()),
            write_test_bundle(dir.path()),
        );
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let (sink, worker) = spawn(cfg, cancel_rx);

        let expected = ordered_events(OPENING + BURST);
        let mut events = expected.clone().into_iter();
        for event in events.by_ref().take(OPENING) {
            sink.try_emit("test", event, aisix_obs::UsageEventLabels::default());
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while batches.lock().unwrap().is_empty() {
            assert!(std::time::Instant::now() < deadline, "no telemetry POST");
            tokio::task::yield_now().await;
        }

        // The control plane is now holding that POST, so nothing is being
        // drained while the rest of the burst arrives.
        for event in events {
            sink.try_emit("test", event, aisix_obs::UsageEventLabels::default());
        }
        drop(sink);
        worker.await.unwrap();

        assert_recorded_events(&batches, &expected);
    }
}
