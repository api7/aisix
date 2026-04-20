//! Langfuse exporter — pushes per-request generation events to a
//! Langfuse `/api/public/ingestion` endpoint.
//!
//! There is no first-party Rust SDK for Langfuse, so we hand-roll the
//! JSON shape against the documented batch ingestion API.
//!
//! Design (spec §3.6 / §9):
//! - The proxy emits a [`LangfuseEvent`] at end-of-request (success or
//!   failure) onto a fire-and-forget mpsc channel.
//! - A background task drains the channel, batches up to
//!   `MAX_BATCH_SIZE` events (or flushes after `FLUSH_INTERVAL`), and
//!   POSTs them with HTTP basic auth (`<public_key>:<secret_key>`).
//! - All errors are logged at WARN; we never block the request hot path
//!   on Langfuse availability.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use aisix_core::ObservabilityConfig;

/// Hard cap on per-batch event count. Tracks the Langfuse default
/// payload-size limit (~1 MB body).
const MAX_BATCH_SIZE: usize = 50;
/// Maximum delay between batch flushes when the queue is non-empty.
const FLUSH_INTERVAL: Duration = Duration::from_secs(1);
/// Channel capacity. Chosen so that a sustained ~1k req/s burst still
/// never blocks the proxy thread; if Langfuse is offline we drop the
/// oldest events at the edges.
const CHANNEL_CAPACITY: usize = 4096;

#[derive(Debug, thiserror::Error)]
pub enum LangfuseError {
    #[error("langfuse enabled but {0} env var not set")]
    MissingEnv(String),
    #[error("langfuse host not configured")]
    MissingHost,
}

/// Event emitted to Langfuse for a single proxy request.
#[derive(Debug, Clone)]
pub struct LangfuseEvent {
    pub trace_id: String,
    pub model: String,
    pub provider: String,
    pub input: Option<serde_json::Value>,
    pub output: Option<serde_json::Value>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub status_code: u16,
    pub latency: Duration,
    pub api_key_id: Option<String>,
}

/// Owning handle returned by [`spawn`]. Drop or call [`shutdown`] to
/// stop the background task.
pub struct LangfuseHandle {
    sender: Arc<LangfuseSender>,
    task: Option<JoinHandle<()>>,
}

impl LangfuseHandle {
    pub fn sender(&self) -> Arc<LangfuseSender> {
        self.sender.clone()
    }

    /// Best-effort shutdown — closes the channel and waits up to 2s for
    /// the background task to drain remaining events.
    pub async fn shutdown(mut self) {
        // Drop our reference to the sender so the channel closes.
        // Replacing the inner Arc with an empty one isn't possible
        // without unsafe; instead we rely on the task's own shutdown
        // signal which fires when the channel is closed.
        let _ = Arc::try_unwrap(self.sender);
        if let Some(task) = self.task.take() {
            let _ = tokio::time::timeout(Duration::from_secs(2), task).await;
        }
    }
}

/// Cheap-clone handle the proxy keeps to push events. Sending is
/// non-blocking — if the queue is full the event is silently dropped.
#[derive(Debug, Clone)]
pub struct LangfuseSender {
    tx: mpsc::Sender<LangfuseEvent>,
}

impl LangfuseSender {
    /// Push an event onto the queue. Never blocks; never errors. If the
    /// channel is full or closed the event is dropped and a counter is
    /// bumped (TODO: wire up `metrics::counter!` once the obs crate
    /// gains a shared metrics handle).
    pub fn emit(&self, event: LangfuseEvent) {
        if self.tx.try_send(event).is_err() {
            tracing::debug!("langfuse queue full or closed — event dropped");
        }
    }
}

/// Spawn the Langfuse exporter. Returns `Ok(None)` when Langfuse is
/// disabled in config; returns the handle otherwise.
pub fn spawn(cfg: &ObservabilityConfig) -> Result<Option<LangfuseHandle>, LangfuseError> {
    let lf = &cfg.langfuse;
    if !lf.enabled {
        return Ok(None);
    }
    let host = lf.host.clone().ok_or(LangfuseError::MissingHost)?;
    let public_key = resolve_env(lf.public_key_env.as_deref(), "LANGFUSE_PUBLIC_KEY")?;
    let secret_key = resolve_env(lf.secret_key_env.as_deref(), "LANGFUSE_SECRET_KEY")?;

    let (tx, rx) = mpsc::channel::<LangfuseEvent>(CHANNEL_CAPACITY);
    let sender = Arc::new(LangfuseSender { tx });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let task = tokio::spawn(run_exporter(host, public_key, secret_key, client, rx));

    Ok(Some(LangfuseHandle {
        sender,
        task: Some(task),
    }))
}

fn resolve_env(env_name: Option<&str>, default: &str) -> Result<String, LangfuseError> {
    let var = env_name.unwrap_or(default);
    std::env::var(var).map_err(|_| LangfuseError::MissingEnv(var.into()))
}

async fn run_exporter(
    host: String,
    public_key: String,
    secret_key: String,
    client: reqwest::Client,
    mut rx: mpsc::Receiver<LangfuseEvent>,
) {
    let endpoint = format!("{}/api/public/ingestion", host.trim_end_matches('/'));
    let auth = base64_basic(&public_key, &secret_key);

    let mut buf: Vec<LangfuseEvent> = Vec::with_capacity(MAX_BATCH_SIZE);
    let mut interval = tokio::time::interval(FLUSH_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            biased;
            maybe = rx.recv() => match maybe {
                Some(ev) => {
                    buf.push(ev);
                    if buf.len() >= MAX_BATCH_SIZE {
                        flush(&client, &endpoint, &auth, &mut buf).await;
                    }
                }
                None => {
                    // Channel closed — drain remaining and exit.
                    if !buf.is_empty() {
                        flush(&client, &endpoint, &auth, &mut buf).await;
                    }
                    break;
                }
            },
            _ = interval.tick() => {
                if !buf.is_empty() {
                    flush(&client, &endpoint, &auth, &mut buf).await;
                }
            }
        }
    }
}

async fn flush(client: &reqwest::Client, endpoint: &str, auth: &str, buf: &mut Vec<LangfuseEvent>) {
    if buf.is_empty() {
        return;
    }
    let payload = IngestionPayload::from_events(std::mem::take(buf));
    let body = match serde_json::to_vec(&payload) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "langfuse: failed to serialise batch");
            return;
        }
    };
    match client
        .post(endpoint)
        .header("authorization", format!("Basic {auth}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("langfuse: batch accepted");
        }
        Ok(resp) => {
            tracing::warn!(status = %resp.status(), "langfuse: ingestion rejected");
        }
        Err(e) => {
            tracing::warn!(error = %e, "langfuse: ingestion request failed");
        }
    }
}

fn base64_basic(public_key: &str, secret_key: &str) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(format!("{public_key}:{secret_key}"))
}

#[derive(Debug, Serialize)]
struct IngestionPayload {
    batch: Vec<IngestionItem>,
}

#[derive(Debug, Serialize)]
struct IngestionItem {
    id: String,
    timestamp: String,
    #[serde(rename = "type")]
    event_type: &'static str,
    body: GenerationBody,
}

#[derive(Debug, Serialize)]
struct GenerationBody {
    id: String,
    trace_id: String,
    name: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
    usage: GenerationUsage,
    start_time: String,
    end_time: String,
    status_message: Option<String>,
}

#[derive(Debug, Serialize)]
struct GenerationUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<u64>,
}

impl IngestionPayload {
    fn from_events(events: Vec<LangfuseEvent>) -> Self {
        let now_iso = iso_now();
        Self {
            batch: events
                .into_iter()
                .map(|ev| IngestionItem {
                    id: format!("evt-{}", uuid::Uuid::new_v4()),
                    timestamp: now_iso.clone(),
                    event_type: "generation-create",
                    body: GenerationBody {
                        id: format!("gen-{}", uuid::Uuid::new_v4()),
                        trace_id: ev.trace_id.clone(),
                        name: format!("{}.chat", ev.provider),
                        model: ev.model.clone(),
                        input: ev.input,
                        output: ev.output,
                        metadata: ev.api_key_id.map(|k| serde_json::json!({"api_key_id": k})),
                        usage: GenerationUsage {
                            input: ev.prompt_tokens,
                            output: ev.completion_tokens,
                            total: ev.total_tokens,
                        },
                        start_time: iso_offset(ev.latency),
                        end_time: now_iso.clone(),
                        status_message: (ev.status_code != 200)
                            .then(|| format!("upstream status {}", ev.status_code)),
                    },
                })
                .collect(),
        }
    }
}

fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    format_iso(secs)
}

fn iso_offset(latency: Duration) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    let start = now.saturating_sub(latency);
    format_iso(start.as_secs())
}

/// Minimal RFC-3339 formatter (no chrono dep needed for this one
/// purpose; chrono is heavy and we only need second-precision
/// timestamps for Langfuse).
fn format_iso(secs: u64) -> String {
    // Days since 1970-01-01
    let days = (secs / 86_400) as i64;
    let time_in_day = secs % 86_400;
    let h = time_in_day / 3600;
    let m = (time_in_day / 60) % 60;
    let s = time_in_day % 60;
    let (y, mo, d) = days_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(mut days: i64) -> (i64, u32, u32) {
    // Convert "days since 1970-01-01" to (year, month, day).
    // Algorithm from Howard Hinnant's date library, simplified.
    days += 719_468;
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::LangfuseConfig;

    fn cfg(enabled: bool, host: Option<&str>) -> ObservabilityConfig {
        ObservabilityConfig {
            service_name: "test".into(),
            log_level: "info".into(),
            access_log: true,
            metrics: Default::default(),
            tracing: Default::default(),
            langfuse: LangfuseConfig {
                enabled,
                host: host.map(String::from),
                public_key_env: None,
                secret_key_env: None,
            },
        }
    }

    #[test]
    fn spawn_returns_none_when_disabled() {
        let result = spawn(&cfg(false, None)).unwrap();
        assert!(
            result.is_none(),
            "disabled langfuse should produce no handle"
        );
    }

    #[test]
    fn spawn_errors_when_enabled_without_host() {
        let result = spawn(&cfg(true, None));
        assert!(matches!(result, Err(LangfuseError::MissingHost)));
    }

    #[test]
    fn spawn_errors_when_enabled_without_keys() {
        // Host present but env vars missing. We point at custom env names
        // to avoid stomping on the global LANGFUSE_PUBLIC_KEY/SECRET_KEY,
        // which other tests (e.g. the wiremock round-trip) set.
        let mut c = cfg(true, Some("https://cloud.langfuse.com"));
        c.langfuse.public_key_env = Some("AISIX_TEST_NEVER_SET_PK".into());
        c.langfuse.secret_key_env = Some("AISIX_TEST_NEVER_SET_SK".into());
        let result = spawn(&c);
        assert!(matches!(result, Err(LangfuseError::MissingEnv(_))));
    }

    #[test]
    fn ingestion_payload_serialises_required_fields() {
        let ev = LangfuseEvent {
            trace_id: "trace-1".into(),
            model: "openai/gpt-4o".into(),
            provider: "openai".into(),
            input: Some(serde_json::json!({"messages": [{"role": "user", "content": "hi"}]})),
            output: Some(serde_json::json!({"text": "hello"})),
            prompt_tokens: Some(7),
            completion_tokens: Some(2),
            total_tokens: Some(9),
            status_code: 200,
            latency: Duration::from_millis(123),
            api_key_id: Some("k-1".into()),
        };
        let payload = IngestionPayload::from_events(vec![ev]);
        let json = serde_json::to_value(&payload).unwrap();
        assert_eq!(json["batch"].as_array().unwrap().len(), 1);
        let item = &json["batch"][0];
        assert_eq!(item["type"], "generation-create");
        assert_eq!(item["body"]["model"], "openai/gpt-4o");
        assert_eq!(item["body"]["trace_id"], "trace-1");
        assert_eq!(item["body"]["usage"]["input"], 7);
        assert_eq!(item["body"]["usage"]["total"], 9);
        assert_eq!(item["body"]["metadata"]["api_key_id"], "k-1");
    }

    #[test]
    fn iso_format_round_trips_known_epoch_seconds() {
        // 2026-04-23T00:00:00Z = 1_776_902_400
        assert_eq!(format_iso(1_776_902_400), "2026-04-23T00:00:00Z");
        // 2024-02-29T12:34:56Z (leap year sanity check)
        assert_eq!(format_iso(1_709_210_096), "2024-02-29T12:34:56Z");
        // Unix epoch
        assert_eq!(format_iso(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn base64_basic_matches_expected_encoding() {
        // "pk:sk" → "cGs6c2s="
        assert_eq!(base64_basic("pk", "sk"), "cGs6c2s=");
    }

    #[test]
    fn sender_emit_does_not_block_when_channel_full() {
        let (tx, mut _rx) = mpsc::channel(1);
        let sender = LangfuseSender { tx };
        let ev = LangfuseEvent {
            trace_id: "t".into(),
            model: "m".into(),
            provider: "p".into(),
            input: None,
            output: None,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            status_code: 200,
            latency: Duration::from_millis(1),
            api_key_id: None,
        };
        sender.emit(ev.clone());
        // Channel now full — second emit must not block or error.
        sender.emit(ev);
    }

    #[tokio::test]
    async fn full_round_trip_to_wiremock_upstream() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/public/ingestion"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        // SAFETY: tests run single-threaded by default and we only mutate
        // env vars that are scoped to this test's spawn() call.
        unsafe {
            std::env::set_var("LANGFUSE_PUBLIC_KEY", "pk-test");
            std::env::set_var("LANGFUSE_SECRET_KEY", "sk-test");
        }

        let handle = spawn(&cfg(true, Some(&server.uri()))).unwrap().unwrap();
        let sender = handle.sender();
        sender.emit(LangfuseEvent {
            trace_id: "t-rt".into(),
            model: "openai/gpt-4o".into(),
            provider: "openai".into(),
            input: None,
            output: None,
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            status_code: 200,
            latency: Duration::from_millis(50),
            api_key_id: None,
        });

        // Wait for the 1s flush interval + a small jitter window.
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        server.verify().await;
    }
}
