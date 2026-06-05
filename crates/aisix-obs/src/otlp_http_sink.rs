//! Per-env OTLP/HTTP exporter — emits one OTLP-shaped POST per chat
//! request to each configured `ObservabilityExporter` (kind=otlp_http).
//!
//! ## Design
//!
//! cp-api projects every configured exporter onto kine at
//! `/aisix/<env>/observability_exporters/<uuid>`. The DP loads them
//! via the existing etcd watch into
//! `AisixSnapshot::observability_exporters`. After every chat
//! completion the proxy hot path hands the resulting `UsageEvent` plus
//! the live snapshot's exporter list to [`fan_out`], which:
//!
//! 1. Filters to enabled exporters with `kind = OtlpHttp`.
//! 2. Builds one OTLP/HTTP-JSON span per exporter, encoded per
//!    OpenTelemetry's GenAI semantic conventions
//!    (<https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md>).
//! 3. Spawns a fire-and-forget tokio task per (event, exporter) pair
//!    that POSTs the span. Failures get a `tracing::warn!` and are
//!    dropped — observability MUST NOT block the request hot path.
//!
//! ## What's intentionally NOT in MVP
//!
//! - **No batching** — one HTTP POST per request per exporter. Phase 2
//!   will move to a worker-task model with a bounded mpsc + 1s flush
//!   interval once the patterns are exercised by real load.
//! - **No retry / backoff** — best-effort fire-and-forget. If the
//!   user's OTLP receiver is unreachable the span is lost. Phase 2
//!   adds a tiny exponential-backoff wrapper.
//! - **No gRPC** — `otlp_grpc` is a separate kind we'll add when a
//!   user actually asks for it; the JSON-over-HTTP form works against
//!   every receiver in the wild and avoids pulling in tonic on the
//!   hot path.
//! - **No content_mode redaction** — defaults to `metadata_only`
//!   (no prompt/response bodies in the span). The MVP cannot leak
//!   user content because it never accepts content fields in the
//!   first place.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{ExporterKind, ObservabilityExporter};
use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::sink::{
    BatchUnit, EventBatch, IdempotencyMarker, IdempotencyScheme, ObservabilitySink, OrderingScope,
    SinkAck, SinkCapabilities, SinkError, SinkHealth, SinkResult,
};
use crate::usage::UsageEvent;

/// Wall-clock duration of an OTLP/HTTP POST before we abandon it.
/// Tight on purpose — we never want a slow exporter to backlog tokio
/// tasks for a wedged user receiver.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum concurrent in-flight POSTs per exporter. Past this point we
/// drop further events on the request hot path rather than queueing
/// them — the queue would just grow unbounded behind a slow receiver,
/// hold the per-event JSON body in memory, and eventually OOM the DP.
/// 64 is generous enough that a healthy receiver never trips the cap
/// (even at a sustained 100 RPS, a 200 ms p50 keeps in-flight under
/// 20) but still bounds the worst case to ~64 × payload-size bytes.
/// See issue #113.
const MAX_INFLIGHT_PER_EXPORTER: usize = 64;

/// `User-Agent` header so vendor receivers can attribute traces back
/// to AISIX in their own analytics. Not a contract; informational.
const USER_AGENT: &str = concat!("aisix-dp/", env!("CARGO_PKG_VERSION"));

/// Cheap clonable handle the proxy hands to request handlers. Holds a
/// reusable `reqwest::Client` so connection pools survive across
/// requests — even with per-event POSTs the kept-alive socket
/// amortises TLS for the common case where one DP exports to one
/// vendor.
///
/// Per-exporter concurrency is bounded: a [`Semaphore`] with
/// [`MAX_INFLIGHT_PER_EXPORTER`] permits is created lazily on first
/// sighting of each exporter name. When the cap is hit, further events
/// for that exporter are *dropped* on the hot path (logged at debug)
/// rather than queued. This is intentional — the alternative is letting
/// task count + memory grow unbounded behind a slow receiver, which
/// caused real OOMs in production. See issue #113.
#[derive(Debug, Clone)]
pub struct OtlpHttpFanOut {
    inner: Arc<FanOutInner>,
}

#[derive(Debug)]
struct FanOutInner {
    client: reqwest::Client,
    /// Per-exporter semaphores keyed by name. Created lazily; never
    /// pruned (a Semaphore is small and the operator's exporter set
    /// is bounded by configuration). The Mutex is parking_lot so
    /// uncontended lookups are basically a single atomic load.
    permits: Mutex<HashMap<String, Arc<Semaphore>>>,
}

impl OtlpHttpFanOut {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .user_agent(USER_AGENT)
            .build()
            // The client builder only fails on illegal TLS roots; the
            // default config is always valid.
            .expect("reqwest::Client default config is valid");
        Self {
            inner: Arc::new(FanOutInner {
                client,
                permits: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Look up (or lazily insert) the per-exporter permit semaphore.
    /// Returning `Arc<Semaphore>` lets the spawned task hold the permit
    /// past `fan_out`'s lifetime — the permit drops with the task.
    fn permits_for(&self, exporter_name: &str) -> Arc<Semaphore> {
        let mut guard = self.inner.permits.lock();
        if let Some(sem) = guard.get(exporter_name) {
            return Arc::clone(sem);
        }
        let sem = Arc::new(Semaphore::new(MAX_INFLIGHT_PER_EXPORTER));
        guard.insert(exporter_name.to_string(), Arc::clone(&sem));
        sem
    }

    /// Test-only: how many in-flight slots are currently held for
    /// `exporter_name`. Used by tests to assert the bounded-fan-out
    /// invariant. Returns 0 if the exporter has never been seen.
    #[doc(hidden)]
    pub fn in_flight_for(&self, exporter_name: &str) -> usize {
        let guard = self.inner.permits.lock();
        match guard.get(exporter_name) {
            Some(sem) => MAX_INFLIGHT_PER_EXPORTER.saturating_sub(sem.available_permits()),
            None => 0,
        }
    }

    /// Fan out one event to every enabled `otlp_http` exporter in the
    /// supplied list. Returns immediately — the actual POSTs run on
    /// detached tokio tasks and never block the caller.
    ///
    /// Per-exporter concurrency is capped at
    /// [`MAX_INFLIGHT_PER_EXPORTER`]. Past the cap, further events for
    /// that exporter are dropped (logged at `debug`) — the alternative
    /// is unbounded queueing behind a slow / down receiver, which OOMs
    /// the DP. See issue #113.
    ///
    /// The `exporters` slice is what the proxy's snapshot lookup
    /// returns. Empty slice = no-op (the common case for envs that
    /// haven't configured any exporters yet, so this is the cheap
    /// path).
    pub fn fan_out<'a, I>(&self, event: &UsageEvent, exporters: I)
    where
        I: IntoIterator<Item = &'a ObservabilityExporter>,
    {
        for exp in exporters {
            if !exp.enabled {
                continue;
            }
            // Single-variant enum today; the `let ExporterKind::OtlpHttp`
            // pattern is exhaustive but deliberately written with the
            // type tag spelled out so adding a new variant in Phase 2
            // forces a compile error here.
            let ExporterKind::OtlpHttp(cfg) = &exp.kind;

            // Try to claim a permit BEFORE building the payload — if
            // we're at the cap, drop early so we don't even pay the
            // JSON-serialisation cost.
            let sem = self.permits_for(&exp.name);
            let permit = match sem.try_acquire_owned() {
                Ok(p) => p,
                Err(_) => {
                    tracing::debug!(
                        exporter = %exp.name,
                        cap = MAX_INFLIGHT_PER_EXPORTER,
                        "otlp_http fan-out: exporter at concurrency cap; dropping span",
                    );
                    continue;
                }
            };

            // Build the wire body once per exporter (cheap — small
            // JSON) so the spawned task only owns the bytes.
            let body = build_otlp_traces_payload(event, &exp.name);
            let endpoint = cfg.endpoint.clone();
            let headers = cfg.headers.clone();
            let client = self.inner.client.clone();
            let exporter_name = exp.name.clone();

            tokio::spawn(async move {
                // Permit released when the task ends. `_permit` keeps
                // it alive across the await point.
                let _permit = permit;
                if let Err(err) = post_one(client, endpoint, headers, body).await {
                    tracing::warn!(
                        exporter = %exporter_name,
                        error = %err,
                        "otlp_http exporter POST failed; span dropped",
                    );
                }
            });
        }
    }
}

impl Default for OtlpHttpFanOut {
    fn default() -> Self {
        Self::new()
    }
}

/// An [`ObservabilitySink`] over the OTLP/HTTP-JSON traces protocol — the
/// same wire shape as [`OtlpHttpFanOut`], but driven by the shared
/// [`crate::sink::SinkPipeline`] (batched, retried, backpressured) rather
/// than a per-event fire-and-forget spawn. One instance per configured
/// `otlp_http` exporter.
pub struct OtlpSink {
    name: String,
    endpoint: String,
    headers: BTreeMap<String, String>,
    client: reqwest::Client,
}

impl OtlpSink {
    /// Build a sink for one exporter. The `client` is shared across sinks so
    /// connection pools and TLS sessions are reused.
    pub fn new(
        name: impl Into<String>,
        endpoint: impl Into<String>,
        headers: BTreeMap<String, String>,
        client: reqwest::Client,
    ) -> Self {
        Self {
            name: name.into(),
            endpoint: endpoint.into(),
            headers,
            client,
        }
    }
}

#[async_trait]
impl ObservabilitySink for OtlpSink {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities {
            idempotency: IdempotencyScheme::None,
            ordering: OrderingScope::None,
            batch_unit: BatchUnit::Records,
            // OTLP spans are small and receivers accept large payloads; the
            // sink does not split by bytes, so no pipeline-enforced ceiling.
            max_batch_bytes: None,
            supports_partial_batch: false,
            supports_streaming_ingest: false,
        }
    }

    async fn append_batch(&self, batch: &EventBatch, _marker: &IdempotencyMarker) -> SinkResult {
        if batch.is_empty() {
            return Ok(SinkAck::default());
        }
        // One export request carrying every record's span — one POST, one
        // atomic retry unit (vs. the per-event fan-out's N spawns).
        let spans: Vec<Value> = batch
            .records
            .iter()
            .map(|record| build_otlp_span(&record.usage, &self.name))
            .collect();
        let body = otlp_export_request(spans);
        let bytes = serde_json::to_vec(&body)
            .map_err(|e| SinkError::Permanent(format!("otlp encode: {e}")))?;

        let mut req = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .body(bytes);
        for (key, value) in &self.headers {
            req = req.header(key, value);
        }

        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(SinkAck {
                        accepted: batch.len(),
                        ..SinkAck::default()
                    });
                }
                let text = resp.text().await.unwrap_or_default();
                let detail = format!(
                    "HTTP {}: {}",
                    status,
                    text.chars().take(200).collect::<String>()
                );
                // 5xx / 408 / 429 are worth retrying; other 4xx are
                // config/auth/payload errors that will fail identically.
                if status.is_server_error()
                    || status == reqwest::StatusCode::REQUEST_TIMEOUT
                    || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                {
                    Err(SinkError::Transient(detail))
                } else {
                    Err(SinkError::Permanent(detail))
                }
            }
            // Connect / DNS / timeout — transient by nature.
            Err(e) => Err(SinkError::Transient(format!("POST {}: {e}", self.endpoint))),
        }
    }

    async fn healthcheck(&self) -> SinkHealth {
        // A real connectivity probe (and the control-plane "test connection"
        // affordance) lands with the health/metrics surface; until then a
        // sink reports healthy and its delivery errors surface via
        // `SinkStats::last_error`.
        SinkHealth::healthy()
    }
}

async fn post_one(
    client: reqwest::Client,
    endpoint: String,
    headers: BTreeMap<String, String>,
    body: Value,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut req = client
        .post(&endpoint)
        .header("Content-Type", "application/json")
        .body(serde_json::to_vec(&body)?);
    for (k, v) in headers {
        req = req.header(k, v);
    }
    let resp = req.send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!(
            "HTTP {}: {}",
            status,
            body.chars().take(200).collect::<String>()
        )
        .into());
    }
    Ok(())
}

/// Build the single OTLP span object for one usage event. Attribute names
/// match the OpenTelemetry GenAI semantic conventions:
/// <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md>.
///
/// Per-attribute encoding:
/// - String / int values use the canonical `{"stringValue": ...}` /
///   `{"intValue": "..."}` (string-encoded int per OTLP/JSON spec).
/// - Trace ID + span ID are random 16-byte / 8-byte hex values.
/// - Timestamps are nanos-since-epoch, OTLP's required unit.
fn build_otlp_span(event: &UsageEvent, exporter_name: &str) -> Value {
    let trace_id = random_trace_id();
    let span_id = random_span_id();

    // The DP records `occurred_at` as RFC 3339; convert to nanos.
    // On parse failure (shouldn't happen in practice) fall back to
    // "now" so the span isn't silently dropped.
    let end_unix_nano =
        parse_rfc3339_to_unix_nano(&event.occurred_at).unwrap_or_else(now_unix_nano);
    // Latency landed in milliseconds; widen + multiply.
    let latency_nanos = (event.latency_ms as u128).saturating_mul(1_000_000);
    let start_unix_nano = end_unix_nano.saturating_sub(latency_nanos);

    // Status: OK (1) for 2xx, ERROR (2) otherwise.
    let status_code = if (200..300).contains(&event.status_code) {
        1
    } else {
        2
    };

    let mut attributes = vec![
        attr_string("gen_ai.system", "aisix"),
        attr_string("gen_ai.operation.name", "chat"),
    ];
    if !event.provider_model_version.is_empty() {
        attributes.push(attr_string(
            "gen_ai.response.model",
            &event.provider_model_version,
        ));
    }
    if !event.provider_request_id.is_empty() {
        attributes.push(attr_string(
            "gen_ai.response.id",
            &event.provider_request_id,
        ));
    }
    if !event.finish_reason.is_empty() {
        attributes.push(attr_string_array(
            "gen_ai.response.finish_reasons",
            std::slice::from_ref(&event.finish_reason),
        ));
    }
    attributes.push(attr_int(
        "gen_ai.usage.input_tokens",
        event.prompt_tokens as i64,
    ));
    attributes.push(attr_int(
        "gen_ai.usage.output_tokens",
        event.completion_tokens as i64,
    ));
    attributes.push(attr_int(
        "http.response.status_code",
        event.status_code as i64,
    ));
    if !event.api_key_id.is_empty() {
        // Custom attribute (no semconv yet) so reviewers can join
        // spans back to the AISIX api_key dashboard.
        attributes.push(attr_string("aisix.api_key_id", &event.api_key_id));
    }
    if !event.model_id.is_empty() {
        attributes.push(attr_string("aisix.model_id", &event.model_id));
    }
    attributes.push(attr_string("aisix.exporter_name", exporter_name));
    attributes.push(attr_string("aisix.request_id", &event.request_id));
    if event.ttft_ms > 0 {
        attributes.push(attr_int("aisix.ttft_ms", event.ttft_ms as i64));
    }
    // Downstream client attribution (#492). Custom attrs so exporters
    // can slice by source IP / client type; the OTLP encoder is an
    // explicit allowlist, so new UsageEvent fields must be added here.
    if !event.client_source_ip.is_empty() {
        attributes.push(attr_string(
            "aisix.client_source_ip",
            &event.client_source_ip,
        ));
    }
    if !event.client_user_agent.is_empty() {
        attributes.push(attr_string(
            "aisix.client_user_agent",
            &event.client_user_agent,
        ));
    }

    json!({
        "traceId": trace_id,
        "spanId":  span_id,
        "name":    "chat.completions",
        "kind":    3, // SPAN_KIND_CLIENT (DP → upstream LLM)
        "startTimeUnixNano": start_unix_nano.to_string(),
        "endTimeUnixNano":   end_unix_nano.to_string(),
        "attributes": attributes,
        "status": { "code": status_code },
    })
}

/// Wrap one or more spans into an OTLP/HTTP-JSON `ExportTraceServiceRequest`.
fn otlp_export_request(spans: Vec<Value>) -> Value {
    json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    attr_string("service.name", "aisix-dp"),
                ],
            },
            "scopeSpans": [{
                "scope": { "name": "aisix-obs.otlp_http_sink" },
                "spans": spans,
            }],
        }],
    })
}

/// One event -> one-span export request (used by the per-event fan-out).
fn build_otlp_traces_payload(event: &UsageEvent, exporter_name: &str) -> Value {
    otlp_export_request(vec![build_otlp_span(event, exporter_name)])
}

fn attr_string(key: &str, value: &str) -> Value {
    json!({
        "key": key,
        "value": { "stringValue": value },
    })
}

fn attr_int(key: &str, value: i64) -> Value {
    json!({
        "key": key,
        // OTLP/JSON encodes int as a string to avoid JS Number precision loss.
        "value": { "intValue": value.to_string() },
    })
}

fn attr_string_array(key: &str, values: &[String]) -> Value {
    let arr: Vec<Value> = values.iter().map(|v| json!({"stringValue": v})).collect();
    json!({
        "key": key,
        "value": { "arrayValue": { "values": arr } },
    })
}

/// 16 random bytes as 32 lowercase-hex chars per OTLP/JSON spec.
fn random_trace_id() -> String {
    let bytes: [u8; 16] = rand_16();
    hex32(&bytes)
}

/// 8 random bytes as 16 lowercase-hex chars per OTLP/JSON spec.
fn random_span_id() -> String {
    let bytes: [u8; 8] = rand_8();
    hex16(&bytes)
}

fn rand_16() -> [u8; 16] {
    let u = uuid::Uuid::new_v4();
    *u.as_bytes()
}

fn rand_8() -> [u8; 8] {
    let u = uuid::Uuid::new_v4();
    let b = u.as_bytes();
    [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]
}

fn hex32(bytes: &[u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex16(bytes: &[u8; 8]) -> String {
    let mut s = String::with_capacity(16);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn parse_rfc3339_to_unix_nano(s: &str) -> Option<u128> {
    // Use chrono if available, fall back to naive epoch parsing.
    // We avoid pulling chrono into this crate by hand-parsing the
    // common DP-emitted RFC3339 form: `2006-01-02T15:04:05Z` or with
    // fractional seconds `.<digits>`.
    let dt = chrono_like_parse(s)?;
    let secs = dt.0 as u128;
    let nanos = dt.1 as u128;
    secs.checked_mul(1_000_000_000)
        .and_then(|n| n.checked_add(nanos))
}

/// Returns (unix_secs, sub_seconds_in_nanos) on success.
fn chrono_like_parse(s: &str) -> Option<(i64, u32)> {
    // Cheap-and-cheerful: split on the 'T', the seconds field, and 'Z'.
    // Wrong handling of timezone offsets — but the DP serialises UTC
    // with a 'Z' suffix everywhere, so this is sufficient for our
    // own emit shape.
    let s = s.strip_suffix('Z')?;
    let (date, time) = s.split_once('T')?;
    let mut date_parts = date.split('-');
    let y: i32 = date_parts.next()?.parse().ok()?;
    let mo: u32 = date_parts.next()?.parse().ok()?;
    let d: u32 = date_parts.next()?.parse().ok()?;

    let (h_m_s, frac_str) = match time.split_once('.') {
        Some((a, b)) => (a, b),
        None => (time, "0"),
    };
    let mut t_parts = h_m_s.split(':');
    let h: u32 = t_parts.next()?.parse().ok()?;
    let mi: u32 = t_parts.next()?.parse().ok()?;
    let se: u32 = t_parts.next()?.parse().ok()?;

    let secs = days_from_civil(y, mo, d).checked_mul(86_400)?
        + (h as i64) * 3600
        + (mi as i64) * 60
        + se as i64;

    // Truncate to 9 fractional digits.
    let frac_padded: String = frac_str
        .chars()
        .chain(std::iter::repeat('0'))
        .take(9)
        .collect();
    let nanos: u32 = frac_padded.parse().ok()?;

    Some((secs, nanos))
}

/// Howard Hinnant's `days_from_civil` (https://howardhinnant.github.io/date_algorithms.html).
/// Avoids depending on chrono just for the e2e build.
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let doy = (153 * if m > 2 { m - 3 } else { m + 9 } + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era as i64) * 146_097 + doe as i64 - 719_468
}

fn now_unix_nano() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

#[allow(dead_code)]
fn _ensure_arc_clone(_: Arc<()>) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event() -> UsageEvent {
        UsageEvent {
            request_id: "req-test-123".into(),
            occurred_at: "2026-05-01T12:00:00Z".into(),
            model_id: "mod-uuid".into(),
            api_key_id: "ak-uuid".into(),
            prompt_tokens: 10,
            completion_tokens: 5,
            latency_ms: 250,
            status_code: 200,
            provider_request_id: "chatcmpl-abc".into(),
            provider_model_version: "gpt-4o-2024-08-06".into(),
            finish_reason: "stop".into(),
            cost_usd: 0.001,
            ..Default::default()
        }
    }

    fn sample_exporter() -> ObservabilityExporter {
        // Round-trip through serde so the runtime_id (private) gets
        // populated by the loader path, just like in production. Kept
        // off the public API on purpose — callers must go through
        // the loader, not poke the field directly.
        serde_json::from_value(serde_json::json!({
            "name": "test-exp",
            "enabled": true,
            "kind": "otlp_http",
            "endpoint": "http://mock-otlp:4318/v1/traces",
            "headers": {"authorization": "Bearer xyz"}
        }))
        .unwrap()
    }

    #[test]
    fn payload_carries_genai_semconv_attributes() {
        let body = build_otlp_traces_payload(&sample_event(), "test-exp");
        let span = &body["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(span["name"], "chat.completions");
        assert_eq!(span["status"]["code"], 1);
        // Attribute set must include the GenAI required + recommended fields
        // we promised the user.
        let attrs = span["attributes"].as_array().unwrap();
        let keys: Vec<&str> = attrs.iter().map(|a| a["key"].as_str().unwrap()).collect();
        assert!(keys.contains(&"gen_ai.system"));
        assert!(keys.contains(&"gen_ai.operation.name"));
        assert!(keys.contains(&"gen_ai.response.model"));
        assert!(keys.contains(&"gen_ai.response.id"));
        assert!(keys.contains(&"gen_ai.usage.input_tokens"));
        assert!(keys.contains(&"gen_ai.usage.output_tokens"));
        assert!(keys.contains(&"gen_ai.response.finish_reasons"));
        assert!(keys.contains(&"http.response.status_code"));
        assert!(keys.contains(&"aisix.api_key_id"));
        assert!(keys.contains(&"aisix.model_id"));
        assert!(keys.contains(&"aisix.exporter_name"));
        assert!(keys.contains(&"aisix.request_id"));
    }

    #[test]
    fn payload_carries_client_attribution_when_present() {
        let mut ev = sample_event();
        ev.client_source_ip = "203.0.113.7".into();
        ev.client_user_agent = "codex-cli/1.2".into();
        let body = build_otlp_traces_payload(&ev, "test-exp");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let ip = attrs.iter().find(|a| a["key"] == "aisix.client_source_ip");
        let ua = attrs.iter().find(|a| a["key"] == "aisix.client_user_agent");
        assert_eq!(
            ip.expect("client_source_ip attr")["value"]["stringValue"],
            "203.0.113.7"
        );
        assert_eq!(
            ua.expect("client_user_agent attr")["value"]["stringValue"],
            "codex-cli/1.2"
        );
    }

    #[test]
    fn payload_omits_client_attribution_when_empty() {
        let body = build_otlp_traces_payload(&sample_event(), "x");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let keys: Vec<&str> = attrs.iter().map(|a| a["key"].as_str().unwrap()).collect();
        assert!(!keys.contains(&"aisix.client_source_ip"));
        assert!(!keys.contains(&"aisix.client_user_agent"));
    }

    #[test]
    fn payload_marks_5xx_as_error_status() {
        let mut ev = sample_event();
        ev.status_code = 503;
        let body = build_otlp_traces_payload(&ev, "x");
        assert_eq!(
            body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["status"]["code"],
            2
        );
    }

    fn otlp_test_client() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    }

    fn batch_of(n: usize) -> EventBatch {
        let records = (0..n)
            .map(|_| Arc::new(crate::sink::SinkRecord::metadata_only(sample_event())))
            .collect();
        EventBatch::new(records)
    }

    #[tokio::test]
    async fn otlp_sink_posts_one_request_with_all_spans() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/traces"))
            .respond_with(wiremock::ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let sink = OtlpSink::new(
            "test-exp",
            format!("{}/v1/traces", server.uri()),
            BTreeMap::new(),
            otlp_test_client(),
        );

        let ack = sink
            .append_batch(&batch_of(3), &IdempotencyMarker::None)
            .await
            .expect("2xx delivers the batch");
        assert_eq!(ack.accepted, 3);

        let reqs = server.received_requests().await.unwrap();
        assert_eq!(reqs.len(), 1, "one batched request, not three spawns");
        let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
        let spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
            .as_array()
            .unwrap();
        assert_eq!(spans.len(), 3, "all three spans in one export request");
    }

    #[tokio::test]
    async fn otlp_sink_classifies_5xx_as_transient() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let sink = OtlpSink::new("e", server.uri(), BTreeMap::new(), otlp_test_client());
        let err = sink
            .append_batch(&batch_of(1), &IdempotencyMarker::None)
            .await
            .unwrap_err();
        assert!(err.is_transient(), "5xx must be retryable: {err}");
    }

    #[tokio::test]
    async fn otlp_sink_classifies_4xx_as_permanent() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(400))
            .mount(&server)
            .await;
        let sink = OtlpSink::new("e", server.uri(), BTreeMap::new(), otlp_test_client());
        let err = sink
            .append_batch(&batch_of(1), &IdempotencyMarker::None)
            .await
            .unwrap_err();
        assert!(!err.is_transient(), "4xx must be permanent: {err}");
    }

    #[test]
    fn payload_omits_empty_optional_fields() {
        let mut ev = sample_event();
        ev.provider_request_id = String::new();
        ev.provider_model_version = String::new();
        ev.finish_reason = String::new();
        let body = build_otlp_traces_payload(&ev, "x");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let keys: Vec<&str> = attrs.iter().map(|a| a["key"].as_str().unwrap()).collect();
        assert!(!keys.contains(&"gen_ai.response.id"));
        assert!(!keys.contains(&"gen_ai.response.model"));
        assert!(!keys.contains(&"gen_ai.response.finish_reasons"));
        // ttft_ms = 0 (default) → omitted
        assert!(!keys.contains(&"aisix.ttft_ms"));
    }

    #[test]
    fn payload_includes_ttft_when_set() {
        let mut ev = sample_event();
        ev.ttft_ms = 42;
        let body = build_otlp_traces_payload(&ev, "test-exp");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let ttft_attr = attrs.iter().find(|a| a["key"] == "aisix.ttft_ms");
        assert!(ttft_attr.is_some(), "aisix.ttft_ms should be present");
        assert_eq!(ttft_attr.unwrap()["value"]["intValue"], "42");
    }

    #[test]
    fn rfc3339_round_trip() {
        // 2026-05-01T12:00:00Z = 1_777_636_800 unix seconds.
        // (epoch + 56 years + 14 leap days + 120 days into 2026 + 12h)
        let nanos = parse_rfc3339_to_unix_nano("2026-05-01T12:00:00Z").unwrap();
        assert_eq!(nanos, 1_777_636_800 * 1_000_000_000);
    }

    #[test]
    fn rfc3339_with_fractional_seconds() {
        let nanos = parse_rfc3339_to_unix_nano("2026-05-01T12:00:00.5Z").unwrap();
        assert_eq!(nanos, 1_777_636_800 * 1_000_000_000 + 500_000_000);
    }

    #[test]
    fn fan_out_is_a_no_op_on_empty_exporter_list() {
        // Smoke: building the fan-out struct + calling on an empty
        // iterator shouldn't panic and shouldn't spawn tasks. We
        // can't easily count spawned tasks, but if the call returned
        // and the test process didn't hang, we're good.
        let f = OtlpHttpFanOut::new();
        f.fan_out(&sample_event(), std::iter::empty());
    }

    #[test]
    fn disabled_exporter_is_skipped() {
        // Build a disabled exporter with a deliberately bogus
        // endpoint; if the fan-out tried to POST to it the spawned
        // task would log a warning, but never panic. We can't easily
        // assert "no task was spawned" without instrumentation;
        // contenting ourselves with "doesn't crash" + the
        // production code path's `if !exp.enabled { continue }`.
        let mut exp = sample_exporter();
        exp.enabled = false;
        let f = OtlpHttpFanOut::new();
        f.fan_out(&sample_event(), std::iter::once(&exp));
    }

    // ---- regression coverage for issue #113 -------------------------
    // Pre-fix, fan_out spawned one detached tokio::spawn per (event,
    // exporter) with no concurrency cap. A slow / down OTLP receiver
    // would let task count + per-task payload memory grow unbounded
    // until OOM. The fix bounds in-flight POSTs per exporter to
    // MAX_INFLIGHT_PER_EXPORTER via a Semaphore; past the cap, events
    // are dropped on the request hot path rather than queued.

    /// A receiver that hangs forever — simulates a wedged OTLP backend.
    /// We point exporters at it to wedge the spawned tasks past the
    /// cap so we can observe the bound.
    fn wedged_endpoint(server: &wiremock::MockServer) -> ObservabilityExporter {
        // wiremock without registering any Mock returns 404 — that's
        // fast (not the wedged behaviour we want). Instead point at a
        // path that has a long delay registered.
        let exp_json = serde_json::json!({
            "name": "wedged-exporter",
            "enabled": true,
            "kind": "otlp_http",
            "endpoint": format!("{}/v1/traces", server.uri()),
            "headers": {}
        });
        serde_json::from_value(exp_json).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fan_out_caps_in_flight_when_exporter_is_slow() {
        use std::time::Duration;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Each POST hangs for 5 minutes; production REQUEST_TIMEOUT
        // is 5s so in steady state these all time out, but for the
        // test window we drive the cap deterministically.
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(300)))
            .mount(&server)
            .await;

        let exp = wedged_endpoint(&server);
        let f = OtlpHttpFanOut::new();
        // Push more events than the cap; tasks block on the wedged
        // receiver, so in_flight must saturate at MAX_INFLIGHT_PER_EXPORTER
        // and further calls must be dropped at the hot path.
        let pushes = MAX_INFLIGHT_PER_EXPORTER + 50;
        for _ in 0..pushes {
            f.fan_out(&sample_event(), std::iter::once(&exp));
        }
        // Yield so spawned tasks get a chance to acquire their
        // permits before we sample. Multi-thread runtime + a couple
        // of yields is plenty for try_acquire to settle.
        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
        let inflight = f.in_flight_for(&exp.name);
        assert_eq!(
            inflight, MAX_INFLIGHT_PER_EXPORTER,
            "in-flight should saturate exactly at the cap; got {inflight}",
        );
        // Pre-fix (no cap), inflight would equal `pushes` here —
        // i.e. ~114, growing unboundedly. The assertion above pins
        // the bound.
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fan_out_recovers_after_permits_release() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Fast 200 — every permit released quickly.
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let exp = wedged_endpoint(&server);
        let f = OtlpHttpFanOut::new();
        // Drive a burst that exceeds the cap; under fast-receiver
        // conditions the permits cycle through quickly.
        for _ in 0..(MAX_INFLIGHT_PER_EXPORTER * 2) {
            f.fan_out(&sample_event(), std::iter::once(&exp));
        }
        // Generous wait — wiremock + reqwest + tokio handshake is
        // hundreds of ms in CI. The point: in-flight should drop back
        // to (close to) zero after the burst clears.
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let inflight = f.in_flight_for(&exp.name);
        assert!(
            inflight < MAX_INFLIGHT_PER_EXPORTER,
            "permits should release as POSTs complete; in_flight stuck at {inflight}",
        );
    }
}
