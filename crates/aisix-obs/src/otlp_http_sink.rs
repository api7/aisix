//! Per-env OTLP/HTTP exporter — emits one OTLP-shaped span per chat request
//! to each configured `ObservabilityExporter` (kind=otlp_http).
//!
//! ## Design
//!
//! cp-api projects every configured exporter onto kine at
//! `/aisix/<env>/observability_exporters/<uuid>`. The DP loads them via the
//! existing etcd watch into `AisixSnapshot::observability_exporters`. After
//! every chat completion the proxy hot path hands the resulting `UsageEvent`
//! plus the live snapshot's exporter list to [`OtlpHttpFanOut::fan_out`],
//! which:
//!
//! 1. Filters to enabled exporters with `kind = OtlpHttp`.
//! 2. Resolves each exporter's [`crate::sink::SinkPipeline`] (lazily started
//!    on first sighting, immediately consistent with the snapshot) and
//!    enqueues one OTLP span record into it. The pipeline batches, retries
//!    transient failures with backoff, and drops-with-metric under
//!    backpressure — all off the request hot path. Spans are encoded per
//!    OpenTelemetry's GenAI semantic conventions
//!    (<https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md>).
//!
//! ## What's intentionally NOT here yet
//!
//! - **No gRPC** — `otlp_grpc` is a separate kind we'll add when a user
//!   actually asks for it; the JSON-over-HTTP form works against every
//!   receiver in the wild and avoids pulling in tonic on the hot path.
//! - **No content_mode redaction** — defaults to `metadata_only` (no
//!   prompt/response bodies in the span). The record never carries content
//!   fields, so it cannot leak.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{
    AliyunSlsConfig, ExporterKind, ObjectStoreConfig, ObservabilityExporter, OtlpHttpConfig,
    SlsContentMode,
};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::sink::{
    build_object_store_sink, resolve_sls_credential, AliyunSlsSink, BatchUnit, CapturedContent,
    EventBatch, ExporterPipelines, IdempotencyMarker, IdempotencyScheme, ObservabilitySink,
    OrderingScope, PipelineConfig, SinkAck, SinkCapabilities, SinkContent, SinkError, SinkHealth,
    SinkRecord, SinkResult,
};
use crate::usage::UsageEvent;

/// Wall-clock duration of an OTLP/HTTP POST before we abandon it.
/// Tight on purpose — we never want a slow exporter to backlog tokio
/// tasks for a wedged user receiver.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// `User-Agent` header so vendor receivers can attribute traces back
/// to AISIX in their own analytics. Not a contract; informational.
const USER_AGENT: &str = concat!("aisix-dp/", env!("CARGO_PKG_VERSION"));

/// Fans usage events out to every configured observability exporter — any
/// [`ExporterKind`], dispatched per kind to the matching
/// [`crate::sink::ObservabilitySink`] — each via its own
/// [`crate::sink::SinkPipeline`] (batched, retried, backpressured). Cheap
/// clonable handle; the per-exporter pipelines and the shared `reqwest::Client`
/// live behind an `Arc`. Pipelines start lazily on first sighting of an
/// exporter (immediately consistent with the snapshot) and are GC'd by
/// [`OtlpHttpFanOut::gc`] when an exporter leaves it.
///
/// NOTE: the type name is historical — it drove only `otlp_http` originally
/// and now fans out all kinds. A rename to `ExporterFanOut` (plus the
/// `ProxyState::otlp_fan_out` field) is a mechanical, behaviour-preserving
/// follow-up kept out of this change to avoid churning every call site.
#[derive(Clone)]
pub struct OtlpHttpFanOut {
    inner: Arc<FanOutInner>,
}

struct FanOutInner {
    /// Per-exporter delivery pipelines (one batched worker each).
    exporters: ExporterPipelines,
    /// Shared HTTP client handed to every sink (connection-pool reuse).
    client: reqwest::Client,
}

/// Delivery tuning for otlp exporter pipelines. A short flush keeps a
/// single-request span visible quickly (the old fan-out posted each event
/// immediately); batching + retry + drop accounting come from the pipeline.
fn exporter_pipeline_config() -> PipelineConfig {
    PipelineConfig {
        flush_interval: Duration::from_secs(1),
        ..PipelineConfig::default()
    }
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
                exporters: ExporterPipelines::new(exporter_pipeline_config()),
                client,
            }),
        }
    }

    /// Fan one event out to every enabled exporter, dispatched per kind and
    /// enqueued into that exporter's pipeline (lazily started on first
    /// sighting). The pipeline owns batching / retry / backpressure; enqueue is
    /// non-blocking, so this never blocks the request hot path. Empty list =
    /// cheap no-op.
    ///
    /// `content` is the request's captured prompt/response, or `None` when the
    /// handler captured none (the default). It is attached ONLY to an
    /// `aliyun_sls` exporter whose `content_mode = full`; every other exporter
    /// — and the CP telemetry path, which is not in this loop — receives the
    /// shared metadata-only record, so prompt/response can never leak there.
    pub fn fan_out<'a, I>(
        &self,
        event: &UsageEvent,
        content: Option<&CapturedContent>,
        exporters: I,
    ) where
        I: IntoIterator<Item = &'a ObservabilityExporter>,
    {
        // The shared metadata-only record, built once on first sighting and
        // reused by every exporter that does not capture content.
        let mut metadata_record: Option<Arc<SinkRecord>> = None;
        for exp in exporters {
            if !exp.enabled {
                continue;
            }

            // Dispatch on kind: each arm fingerprints its delivery-relevant
            // config (so a dashboard edit rebuilds the pipeline) and lazily
            // builds the matching sink. All kinds share one `ExporterPipelines`
            // manager and the pooled client. A new `ExporterKind` variant
            // makes this `match` non-exhaustive — a compile-time prompt to wire
            // its sink here.
            let client = self.inner.client.clone();
            let handle = match &exp.kind {
                ExporterKind::OtlpHttp(cfg) => {
                    let fingerprint = fingerprint_otlp(cfg);
                    let name = exp.name.clone();
                    let endpoint = cfg.endpoint.clone();
                    let headers = cfg.headers.clone();
                    self.inner
                        .exporters
                        .get_or_create(&exp.name, fingerprint, move || {
                            Arc::new(OtlpSink::new(name, endpoint, headers, client))
                                as Arc<dyn ObservabilitySink>
                        })
                }
                ExporterKind::AliyunSls(cfg) => {
                    let fingerprint = fingerprint_sls(cfg);
                    let name = exp.name.clone();
                    let cfg = cfg.clone();
                    self.inner
                        .exporters
                        .get_or_create(&exp.name, fingerprint, move || {
                            // Resolve the AccessKey from the DP's local env at
                            // build time (the key never rode the kine path).
                            // Missing creds → empty key → SLS 401 surfaces as a
                            // delivery-health auth error, not a silent drop.
                            let (ak_id, ak_secret) =
                                resolve_sls_credential(&cfg.credential_ref).unwrap_or_default();
                            Arc::new(AliyunSlsSink::new(
                                name,
                                &cfg.endpoint,
                                &cfg.project,
                                &cfg.logstore,
                                ak_id,
                                ak_secret,
                                client,
                            )) as Arc<dyn ObservabilitySink>
                        })
                }
                ExporterKind::ObjectStore(cfg) => {
                    let fingerprint = fingerprint_object_store(cfg);
                    let name = exp.name.clone();
                    let cfg = cfg.clone();
                    self.inner
                        .exporters
                        .get_or_create(&exp.name, fingerprint, move || {
                            // Resolve cloud creds from the DP's local env and
                            // build the backend at build time. Missing creds or
                            // an un-buildable backend yield a sink that reports
                            // the reason via delivery health (never a silent
                            // drop) — mirroring the SLS path.
                            build_object_store_sink(name, &cfg)
                        })
                }
            };

            // A content-bearing record for an SLS exporter that opted into
            // full capture (and only when the handler captured content); the
            // shared metadata-only record for everyone else.
            let record = content_record(&exp.kind, event, content).unwrap_or_else(|| {
                Arc::clone(
                    metadata_record
                        .get_or_insert_with(|| Arc::new(SinkRecord::metadata_only(event.clone()))),
                )
            });
            handle.try_enqueue(record);
        }
    }

    /// Stop pipelines for exporters no longer present in `live` (the current
    /// snapshot's enabled exporter names, across all kinds). Called
    /// periodically by the server to GC pipelines for deleted / disabled
    /// exporters.
    pub fn gc(&self, live: &std::collections::HashSet<String>) {
        self.inner.exporters.retain(live);
    }

    /// Drain every exporter pipeline at graceful shutdown.
    pub async fn shutdown(&self) {
        self.inner.exporters.shutdown().await;
    }
}

impl Default for OtlpHttpFanOut {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash an `otlp_http` exporter's delivery-relevant config. A change to
/// endpoint or headers yields a new fingerprint, so the manager rebuilds the
/// exporter's pipeline against the edited target.
fn fingerprint_otlp(cfg: &OtlpHttpConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cfg.endpoint.hash(&mut hasher);
    cfg.headers.hash(&mut hasher);
    hasher.finish()
}

/// Hash an `aliyun_sls` exporter's delivery-relevant config. The fingerprint
/// covers only kine-visible fields (endpoint / project / logstore /
/// credential_ref), never the resolved AccessKey — rotating the secret under
/// the *same* reference therefore takes effect on the next DP restart, not
/// live. A ref change does rebuild the pipeline.
fn fingerprint_sls(cfg: &AliyunSlsConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cfg.endpoint.hash(&mut hasher);
    cfg.project.hash(&mut hasher);
    cfg.logstore.hash(&mut hasher);
    cfg.credential_ref.hash(&mut hasher);
    hasher.finish()
}

/// The content-bearing [`SinkRecord`] for one exporter, or `None` to fall back
/// to the shared metadata-only record.
///
/// Content is attached ONLY to an `aliyun_sls` exporter whose
/// `content_mode = full`, and ONLY when the handler captured content — every
/// other exporter (and the CP telemetry path, which never enters the fan-out)
/// gets metadata only. The captured prompt/response are truncated to the
/// exporter's `content_max_bytes`, ORing in any truncation the handler already
/// applied at capture time.
fn content_record(
    kind: &ExporterKind,
    event: &UsageEvent,
    content: Option<&CapturedContent>,
) -> Option<Arc<SinkRecord>> {
    let ExporterKind::AliyunSls(cfg) = kind else {
        return None;
    };
    if cfg.content_mode != SlsContentMode::Full {
        return None;
    }
    let captured = content?;
    let mut sc = SinkContent::capture(
        &captured.prompt,
        &captured.response,
        cfg.content_max_bytes as usize,
    );
    sc.truncated = sc.truncated || captured.truncated;
    Some(Arc::new(
        SinkRecord::metadata_only(event.clone()).with_content(sc),
    ))
}

/// The largest `content_max_bytes` among the env's enabled exporters that
/// capture full content, or `None` if none do.
///
/// A request handler calls this BEFORE doing any capture work: `None` means no
/// exporter wants prompt/response, so the handler skips capture entirely (no
/// body clone, no stream accumulation — zero hot-path cost). `Some(cap)` is the
/// bound the handler caps its capture at; each exporter then re-truncates to
/// its own (≤ cap) limit at delivery.
pub fn content_capture_cap<'a>(
    exporters: impl IntoIterator<Item = &'a ObservabilityExporter>,
) -> Option<u32> {
    exporters
        .into_iter()
        .filter(|e| e.enabled)
        .filter_map(|e| match &e.kind {
            ExporterKind::AliyunSls(cfg) if cfg.content_mode == SlsContentMode::Full => {
                Some(cfg.content_max_bytes)
            }
            _ => None,
        })
        .max()
}

/// Hash an `object_store` exporter's delivery-relevant config. Covers only
/// kine-visible fields (provider / bucket / prefix / region / endpoint /
/// compression / credential_ref), never the resolved cloud key — rotating the
/// secret under the *same* reference therefore takes effect on the next DP
/// restart, not live. Any field change rebuilds the pipeline.
fn fingerprint_object_store(cfg: &ObjectStoreConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cfg.provider.hash(&mut hasher);
    cfg.bucket.hash(&mut hasher);
    cfg.prefix.hash(&mut hasher);
    cfg.region.hash(&mut hasher);
    cfg.endpoint.hash(&mut hasher);
    cfg.compression.hash(&mut hasher);
    cfg.credential_ref.hash(&mut hasher);
    hasher.finish()
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
    // Per-attempt telemetry (#655). `request_id` is the trace/group key; a
    // failover request emits one span per attempt sharing it, ordered by
    // `aisix.attempt_index`. The OTLP encoder is an explicit allowlist, so
    // these are added here alongside the wire fields.
    attributes.push(attr_int("aisix.attempt_index", event.attempt_index as i64));
    if !event.attempt_kind.is_empty() {
        attributes.push(attr_string("aisix.attempt_kind", &event.attempt_kind));
    }
    if !event.attempt_model.is_empty() {
        attributes.push(attr_string("aisix.attempt_model", &event.attempt_model));
    }
    if !event.error_class.is_empty() {
        attributes.push(attr_string("aisix.error_class", &event.error_class));
    }
    if !event.error_message.is_empty() {
        attributes.push(attr_string("aisix.error_message", &event.error_message));
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

/// One event -> one-span export request. Test-only helper for the payload
/// assertions; production paths build spans via [`build_otlp_span`] and batch
/// them through [`otlp_export_request`] inside [`OtlpSink`].
#[cfg(test)]
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

    fn sls_kind(content_mode: SlsContentMode, max_bytes: u32) -> ExporterKind {
        ExporterKind::AliyunSls(AliyunSlsConfig {
            endpoint: "ap-southeast-3.log.aliyuncs.com".into(),
            project: "p".into(),
            logstore: "l".into(),
            credential_ref: "r".into(),
            content_mode,
            content_max_bytes: max_bytes,
        })
    }

    #[test]
    fn content_record_targets_only_full_capture_sls() {
        let event = sample_event();
        let captured = CapturedContent {
            prompt: "the prompt".into(),
            response: "the response".into(),
            truncated: false,
        };

        // otlp never carries content, even when content was captured.
        let otlp = ExporterKind::OtlpHttp(OtlpHttpConfig {
            endpoint: "https://x/v1/traces".into(),
            headers: Default::default(),
        });
        assert!(content_record(&otlp, &event, Some(&captured)).is_none());

        // object_store never carries content either — content_record gates on
        // the AliyunSls variant, so any other kind gets metadata-only even when
        // content was captured (no prompt/response leak into S3 / GCS / Azure).
        let objstore = serde_json::from_value::<ObservabilityExporter>(serde_json::json!({
            "name": "o",
            "enabled": true,
            "kind": "object_store",
            "provider": "s3",
            "bucket": "b",
            "prefix": "p",
            "credential_ref": "r"
        }))
        .unwrap()
        .kind;
        assert!(content_record(&objstore, &event, Some(&captured)).is_none());

        // sls metadata_only → no content.
        let meta = sls_kind(SlsContentMode::MetadataOnly, 1024);
        assert!(content_record(&meta, &event, Some(&captured)).is_none());

        // sls full but nothing captured → falls back to metadata.
        let full = sls_kind(SlsContentMode::Full, 1024);
        assert!(content_record(&full, &event, None).is_none());

        // sls full + captured content → a content-bearing record that still
        // carries the metadata.
        let rec = content_record(&full, &event, Some(&captured))
            .expect("full-capture sls with content yields a content record");
        let c = rec.content.as_ref().expect("content attached");
        assert_eq!(c.prompt, "the prompt");
        assert_eq!(c.response, "the response");
        assert!(!c.truncated);
        assert_eq!(rec.usage.request_id, "req-test-123");

        // Per-exporter cap truncates + flags.
        let big = CapturedContent {
            prompt: "a".repeat(500),
            response: "ok".into(),
            truncated: false,
        };
        let rec = content_record(&sls_kind(SlsContentMode::Full, 16), &event, Some(&big)).unwrap();
        let c = rec.content.as_ref().unwrap();
        assert_eq!(c.prompt.len(), 16);
        assert!(c.truncated, "oversize content must flag truncated");

        // Handler-side truncation propagates even when the per-exporter cap
        // did not cut.
        let pre = CapturedContent {
            prompt: "short".into(),
            response: "short".into(),
            truncated: true,
        };
        let rec = content_record(&full, &event, Some(&pre)).unwrap();
        assert!(
            rec.content.as_ref().unwrap().truncated,
            "source truncation must propagate"
        );
    }

    #[test]
    fn content_capture_cap_picks_max_enabled_full_sls() {
        fn sls(name: &str, enabled: bool, mode: &str, max: u32) -> ObservabilityExporter {
            serde_json::from_value(serde_json::json!({
                "name": name,
                "enabled": enabled,
                "kind": "aliyun_sls",
                "endpoint": "ap-southeast-3.log.aliyuncs.com",
                "project": "p",
                "logstore": "l",
                "credential_ref": "r",
                "content_mode": mode,
                "content_max_bytes": max,
            }))
            .unwrap()
        }

        // No exporter wants content → None (handler skips capture).
        let otlp = sample_exporter();
        let meta = sls("a", true, "metadata_only", 1024);
        assert_eq!(content_capture_cap([&otlp, &meta]), None);

        // One full-capture sls → its cap.
        let full = sls("a", true, "full", 4096);
        assert_eq!(content_capture_cap([&full]), Some(4096));

        // Max across several full-capture exporters.
        let full_b = sls("b", true, "full", 8192);
        assert_eq!(content_capture_cap([&full, &full_b]), Some(8192));

        // A disabled full-capture exporter is ignored.
        let disabled = sls("a", false, "full", 4096);
        assert_eq!(content_capture_cap([&disabled]), None);
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
    fn payload_carries_per_attempt_attributes() {
        // A failed fallback attempt (#655): zero tokens, error info, target.
        let mut ev = sample_event();
        ev.attempt_index = 1;
        ev.attempt_kind = "fallback".into();
        ev.attempt_model = "secondary".into();
        ev.error_class = "upstream_status".into();
        ev.error_message = "upstream returned 502".into();
        let body = build_otlp_traces_payload(&ev, "test-exp");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let find = |k: &str| attrs.iter().find(|a| a["key"] == k);
        assert_eq!(
            find("aisix.attempt_index").unwrap()["value"]["intValue"],
            "1"
        );
        assert_eq!(
            find("aisix.attempt_kind").unwrap()["value"]["stringValue"],
            "fallback"
        );
        assert_eq!(
            find("aisix.attempt_model").unwrap()["value"]["stringValue"],
            "secondary"
        );
        assert_eq!(
            find("aisix.error_class").unwrap()["value"]["stringValue"],
            "upstream_status"
        );
        assert_eq!(
            find("aisix.error_message").unwrap()["value"]["stringValue"],
            "upstream returned 502"
        );

        // A direct (non-routing) success omits the routing-only attrs but
        // still carries attempt_index=0.
        let plain = build_otlp_traces_payload(&sample_event(), "test-exp");
        let plain_attrs = plain["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let keys: Vec<&str> = plain_attrs
            .iter()
            .map(|a| a["key"].as_str().unwrap())
            .collect();
        assert!(keys.contains(&"aisix.attempt_index"));
        assert!(!keys.contains(&"aisix.attempt_kind"));
        assert!(!keys.contains(&"aisix.attempt_model"));
        assert!(!keys.contains(&"aisix.error_class"));
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
        f.fan_out(&sample_event(), None, std::iter::empty());
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
        f.fan_out(&sample_event(), None, std::iter::once(&exp));
    }

    #[tokio::test]
    async fn fan_out_delivers_a_span_to_a_real_receiver() {
        // The new fan-out enqueues into a per-exporter pipeline (1s flush);
        // a single request's span lands at the receiver after the flush.
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/traces"))
            .respond_with(wiremock::ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let exp: ObservabilityExporter = serde_json::from_value(serde_json::json!({
            "name": "real-otlp",
            "enabled": true,
            "kind": "otlp_http",
            "endpoint": format!("{}/v1/traces", server.uri()),
            "headers": {}
        }))
        .unwrap();

        let f = OtlpHttpFanOut::new();
        f.fan_out(&sample_event(), None, std::iter::once(&exp));

        // Poll for the batched POST (flush_interval is 1s).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if !server.received_requests().await.unwrap().is_empty() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no OTLP POST within 5s"
            );
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        f.shutdown().await;
    }
}
