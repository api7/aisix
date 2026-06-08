//! Datadog native **Logs HTTP intake** sink — the `http_batch` family's
//! Datadog vendor (ai-gateway#688).
//!
//! A batch becomes one Datadog **logs intake** request: every [`SinkRecord`]
//! maps to one JSON log object (the canonical usage metadata flattened into
//! sibling fields under OTel GenAI semconv names, plus the Datadog reserved
//! attributes `ddsource` / `ddtags` / `service` / `message`, plus opt-in
//! captured prompt/response), the JSON array is gzip-compressed, and POSTed
//! to `https://http-intake.logs.<site>/api/v2/logs` over the crate's shared
//! rustls client.
//!
//! Wire details are taken from Datadog's official "Send logs" HTTP API
//! reference (<https://docs.datadoghq.com/api/latest/logs/#send-logs>):
//! the endpoint, the `DD-API-KEY` header, the `Content-Encoding: gzip`
//! support, the JSON-array body, and the intake limits (1000 logs and 5 MB
//! uncompressed per request; 1 MB per log). GenAI attribute names follow the
//! OpenTelemetry GenAI semantic conventions, matching the OTLP span builder
//! the other sinks emit, so a Datadog log and an OTLP span carry the same
//! `gen_ai.*` keys.

use std::io::Write as _;

use async_trait::async_trait;
use http::header::{CONTENT_ENCODING, CONTENT_TYPE};
use http::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::{
    BatchUnit, EventBatch, IdempotencyMarker, IdempotencyScheme, ObservabilitySink, OrderingScope,
    SinkAck, SinkCapabilities, SinkError, SinkHealth, SinkRecord, SinkResult,
};

/// `DD-API-KEY`: Datadog's API-key header. The resolved key rides here and
/// nowhere else (never in the body, the URL, logs, or error text).
const DD_API_KEY: HeaderName = HeaderName::from_static("dd-api-key");

/// Cap on a masked error-detail string surfaced to logs / health.
const DETAIL_MAX_CHARS: usize = 200;

/// A delivery target for one Datadog Logs intake.
pub struct DatadogSink {
    name: String,
    /// Full POST target — `https://http-intake.logs.<site>/api/v2/logs`
    /// (or `http://<loopback>/api/v2/logs` for a vetted mock intake).
    endpoint_url: String,
    /// Resolved Datadog API key. Sent only as the `DD-API-KEY` header.
    api_key: String,
    /// Datadog `ddsource` reserved attribute.
    ddsource: String,
    /// Comma-joined `ddtags` reserved attribute (empty string = omitted).
    ddtags: String,
    /// Datadog `service` reserved attribute.
    service: String,
    client: reqwest::Client,
}

impl DatadogSink {
    /// Build a sink for one Datadog site.
    ///
    /// `site` is a bare Datadog site host, e.g. `datadoghq.com`; the request
    /// host is `http-intake.logs.<site>` over https. A `site` that is a vetted
    /// loopback host (the e2e's `mock-datadog` / `127.0.0.1` / `localhost`,
    /// optionally with a `:port`) is posted to over http directly, so a local
    /// mock intake needs no TLS. `tags` are rendered into a comma-joined
    /// `ddtags` value once at build time. The `client` is shared across sinks
    /// so connection pools and TLS sessions are reused.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        site: &str,
        api_key: impl Into<String>,
        ddsource: impl Into<String>,
        tags: &[String],
        service: impl Into<String>,
        client: reqwest::Client,
    ) -> Self {
        Self {
            name: name.into(),
            endpoint_url: intake_url_for(site),
            api_key: api_key.into(),
            ddsource: ddsource.into(),
            ddtags: tags.join(","),
            service: service.into(),
            client,
        }
    }
}

#[async_trait]
impl ObservabilitySink for DatadogSink {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities {
            // Datadog logs intake is at-least-once: no server-side dedup token.
            idempotency: IdempotencyScheme::None,
            // Independent posts; Datadog does not require cross-record ordering.
            ordering: OrderingScope::None,
            // Count-bounded today (parity with the SLS / OTLP sinks). Datadog
            // caps the intake body (1000 logs and 5 MB uncompressed per
            // request, 1 MB per log), but byte-aware chunking is only needed
            // once full prompt/response content rides these records; on the
            // metadata-only path the pipeline's per-batch record cap keeps
            // bodies far under the limit. Declaring a byte ceiling the sink
            // does not yet self-enforce would be a false promise (and a
            // silent-drop bug under content) — so it stays `None` until the
            // chunking lands (api7/ai-gateway#556), matching the SLS sink's
            // resolved shape.
            batch_unit: BatchUnit::Records,
            max_batch_bytes: None,
            // The intake accepts or rejects the whole request.
            supports_partial_batch: false,
            supports_streaming_ingest: false,
        }
    }

    async fn append_batch(&self, batch: &EventBatch, _marker: &IdempotencyMarker) -> SinkResult {
        if batch.is_empty() {
            return Ok(SinkAck::default());
        }

        // 1. One JSON log object per record → a single JSON array body.
        let logs: Vec<Value> = batch.records.iter().map(|r| self.to_log(r)).collect();
        let raw = serde_json::to_vec(&Value::Array(logs))
            .map_err(|e| SinkError::Permanent(format!("datadog: json encode: {e}")))?;

        // 2. gzip the JSON (Datadog accepts `Content-Encoding: gzip`).
        let compressed = gzip(&raw)?;

        // 3. Headers: JSON body, gzip encoding, and the API key.
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        // The API key is the only place the secret appears. An un-resolvable
        // key (empty) is sent as-is so Datadog answers 403 and the failure
        // surfaces as a delivery-health auth error rather than a silent drop.
        let key = HeaderValue::from_str(&self.api_key).map_err(|_| {
            SinkError::Permanent("datadog: api key has invalid header bytes".into())
        })?;
        headers.insert(DD_API_KEY, key);

        // 4. Deliver.
        let resp = match self
            .client
            .post(&self.endpoint_url)
            .headers(headers)
            .body(compressed)
            .send()
            .await
        {
            Ok(resp) => resp,
            // Connect / DNS / timeout — transient by nature. The endpoint URL
            // carries no secret, so it is safe in the error detail.
            Err(e) => {
                return Err(SinkError::Transient(format!(
                    "datadog: POST {}: {e}",
                    self.endpoint_url
                )))
            }
        };

        let status = resp.status();
        if status.is_success() {
            return Ok(SinkAck {
                accepted: batch.len(),
                ..SinkAck::default()
            });
        }

        let body = resp.text().await.unwrap_or_default();
        let detail = parse_datadog_error(status, &body);
        // 429 (rate limit) and 5xx (502/503/504, transient server faults) are
        // worth retrying; other 4xx (400 malformed / 401/403 auth / 413 too
        // large) are config/auth/payload errors that fail identically on retry.
        if is_transient_status(status) {
            Err(SinkError::Transient(detail))
        } else {
            Err(SinkError::Permanent(detail))
        }
    }

    async fn healthcheck(&self) -> SinkHealth {
        // A real connectivity probe (and the control-plane "test connection"
        // affordance) lands with the health/metrics surface; until then a sink
        // reports healthy and its delivery errors surface via
        // `SinkStats::last_error`. (Mirrors the SLS / OTLP sinks.)
        SinkHealth::healthy()
    }
}

impl DatadogSink {
    /// Map one canonical record to a Datadog log JSON object.
    ///
    /// The metadata is produced by *serializing* [`crate::usage::UsageEvent`]
    /// rather than hand-listing its 30-plus fields: that single-sources the
    /// schema and inherits the exact `skip_serializing_if` emptiness rules the
    /// cp-api wire uses (empty strings / zero counters are omitted). Field
    /// names are remapped to the OTel GenAI semconv where one exists
    /// (`gen_ai.system`, `gen_ai.request.model` / `gen_ai.response.model`,
    /// `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`, …) so a
    /// Datadog log and an OTLP span carry the same keys; the remainder land
    /// under an `aisix.` prefix. The Datadog reserved attributes (`ddsource`,
    /// `ddtags`, `service`, `message`) are set as siblings.
    fn to_log(&self, record: &SinkRecord) -> Value {
        let mut obj = Map::new();

        // Datadog reserved attributes.
        obj.insert("ddsource".into(), json!(self.ddsource));
        if !self.ddtags.is_empty() {
            obj.insert("ddtags".into(), json!(self.ddtags));
        }
        obj.insert("service".into(), json!(self.service));
        obj.insert("message".into(), json!(summary_message(&record.usage)));
        obj.insert("schema_version".into(), json!(record.schema_version));

        // Flatten the usage metadata, remapping each field to its GenAI semconv
        // (or `aisix.`) key. Empty strings are skipped; numeric 0 / false stay.
        if let Ok(Value::Object(fields)) = serde_json::to_value(&record.usage) {
            for (key, value) in fields {
                let Some(rendered) = render_scalar(&value) else {
                    continue;
                };
                obj.insert(map_field_name(&key), rendered);
            }
        }

        // Opt-in captured content as flat, queryable fields. Absent on the
        // default metadata-only path, so prompts never leak there.
        if let Some(content) = &record.content {
            obj.insert("gen_ai.prompt".into(), json!(content.prompt));
            obj.insert("gen_ai.completion".into(), json!(content.response));
            if content.truncated {
                obj.insert("content_truncated".into(), json!(true));
            }
        }

        Value::Object(obj)
    }
}

/// Render a JSON scalar as a Datadog log field value; `None` skips the field.
///
/// Empty strings are skipped so the log omits blank fields uniformly: most
/// optional `UsageEvent` fields already drop out via `skip_serializing_if`,
/// but a few (`model_id`, `api_key_id`) carry only `#[serde(default)]` and
/// would otherwise serialize as `""`. Numeric `0` and `false` are kept — a
/// zero token count or status code is real data. Nested fields (e.g.
/// `applied_guardrails`) ride through structurally so Datadog can facet them.
fn render_scalar(value: &Value) -> Option<Value> {
    match value {
        Value::Null => None,
        Value::String(s) if s.is_empty() => None,
        other => Some(other.clone()),
    }
}

/// Map a `UsageEvent` field name to the key it lands under in the Datadog log.
///
/// The OTel GenAI semantic conventions own the names for the LLM dimensions
/// that have a convention; everything else keeps an `aisix.`-prefixed custom
/// key so the field set is self-describing and collision-free in Datadog's
/// attribute namespace. This is the same naming the OTLP span builder uses.
fn map_field_name(field: &str) -> String {
    match field {
        // ── OTel GenAI semconv ──
        "provider_model_version" => "gen_ai.response.model",
        "provider_request_id" => "gen_ai.response.id",
        "finish_reason" => "gen_ai.response.finish_reason",
        "prompt_tokens" => "gen_ai.usage.input_tokens",
        "completion_tokens" => "gen_ai.usage.output_tokens",
        // ── HTTP semconv ──
        "status_code" => "http.response.status_code",
        // ── AISIX custom dimensions (no semconv) ──
        other => return format!("aisix.{other}"),
    }
    .to_string()
}

/// A short human-readable summary used as the Datadog log `message`. Datadog's
/// Log Explorer shows `message` as the row text, so a compact one-liner keyed
/// on the request makes the log readable without expanding attributes.
fn summary_message(usage: &crate::usage::UsageEvent) -> String {
    let model = if !usage.provider_model_version.is_empty() {
        usage.provider_model_version.as_str()
    } else if !usage.model_id.is_empty() {
        usage.model_id.as_str()
    } else {
        "-"
    };
    format!(
        "ai-gateway request {} model={} status={}",
        usage.request_id, model, usage.status_code
    )
}

/// The Datadog logs-intake error envelope — `{"errors": ["...", ...]}`.
#[derive(Deserialize)]
struct DatadogErrorBody {
    #[serde(default)]
    errors: Vec<String>,
}

/// Parse the Datadog error body into a masked detail, falling back to the raw
/// (truncated) body when it isn't the JSON envelope. The detail never contains
/// the API key — Datadog error messages echo neither the key nor the header.
fn parse_datadog_error(status: reqwest::StatusCode, body: &str) -> String {
    if let Ok(parsed) = serde_json::from_str::<DatadogErrorBody>(body) {
        if !parsed.errors.is_empty() {
            return truncate(&format!("HTTP {status}: {}", parsed.errors.join("; ")));
        }
    }
    truncate(&format!("HTTP {status}: {body}"))
}

/// Whether a Datadog intake HTTP status means "retry with backoff".
///
/// Datadog returns `429 Too Many Requests` on rate-limit and `5xx`
/// (`502`/`503`/`504`) on transient server faults; both are retried. Other
/// `4xx` (`400` malformed payload, `401`/`403` bad API key, `413` payload too
/// large) are permanent — a retry of the same batch fails identically.
fn is_transient_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS
}

/// gzip a byte slice (RFC 1952). Permanent on the rare encode failure.
fn gzip(data: &[u8]) -> Result<Vec<u8>, SinkError> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(data)
        .and_then(|_| enc.finish())
        .map_err(|e| SinkError::Permanent(format!("datadog: gzip: {e}")))
}

/// Truncate a masked detail string to a bounded length for logs / health.
fn truncate(s: &str) -> String {
    s.chars().take(DETAIL_MAX_CHARS).collect()
}

/// Compute the full intake POST URL for a Datadog `site`.
///
/// A real Datadog site host becomes
/// `https://http-intake.logs.<site>/api/v2/logs`. A vetted loopback host (the
/// e2e's `mock-datadog` / `127.0.0.1` / `localhost`, optionally with a
/// `host:port`) is posted to over http directly, so a local mock receiver
/// needs no TLS. The exporter schema only admits a site from the allow-list
/// (the seven real sites plus those three loopback hosts), so this branch
/// can't be used to redirect real traffic to an arbitrary host.
fn intake_url_for(site: &str) -> String {
    let site = site.trim().trim_end_matches('/');
    if is_loopback_site(site) {
        format!("http://{site}/api/v2/logs")
    } else {
        format!("https://http-intake.logs.{site}/api/v2/logs")
    }
}

/// Whether a site token is a vetted loopback mock host (optionally `host:port`)
/// rather than a real Datadog site. Matches the schema's loopback allow-list.
fn is_loopback_site(site: &str) -> bool {
    let host = site.split(':').next().unwrap_or(site);
    matches!(host, "mock-datadog" | "127.0.0.1" | "localhost")
}

/// Resolve an exporter's `credential_ref` to the Datadog API key from the DP's
/// local environment.
///
/// The API key never travels on the kine path (the control plane stores only
/// the reference), so the DP looks it up where it actually runs. The reference
/// is upper-cased with non-alphanumerics folded to `_`, then read from
/// `DD_CRED_<REF>_API_KEY`. The prefix is deliberately NOT `AISIX_`: that
/// namespace is owned by the config loader (`Environment::with_prefix("AISIX")`),
/// so an `AISIX_`-named secret would be reinterpreted as a config override.
/// Returns `None` when the key is unset or blank — the caller then lets the
/// misconfiguration surface as a delivery-health auth error rather than POST
/// with an empty key. (Mirrors `resolve_sls_credential` /
/// `resolve_object_store_credential`.)
pub fn resolve_datadog_credential(credential_ref: &str) -> Option<String> {
    resolve_datadog_credential_with(credential_ref, |key| std::env::var(key).ok())
}

/// Reference-resolution core, parameterized over the variable source so it is
/// testable without mutating the process environment.
fn resolve_datadog_credential_with(
    credential_ref: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    let slug: String = credential_ref
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    let key = lookup(&format!("DD_CRED_{slug}_API_KEY"))?;
    if key.is_empty() {
        return None;
    }
    Some(key)
}

#[cfg(test)]
mod tests {
    use super::{
        intake_url_for, is_transient_status, parse_datadog_error, resolve_datadog_credential_with,
        DatadogSink,
    };
    use crate::sink::{EventBatch, IdempotencyMarker, ObservabilitySink, SinkContent, SinkRecord};
    use crate::usage::UsageEvent;
    use serde_json::Value;
    use std::sync::Arc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const POST_PATH: &str = "/api/v2/logs";

    /// Decode the request body a mock server captured back into the JSON array
    /// of log objects (gunzip, then JSON-parse).
    fn decode_logs(req: &wiremock::Request) -> Vec<Value> {
        assert_eq!(
            req.headers
                .get("content-encoding")
                .expect("content-encoding header")
                .to_str()
                .unwrap(),
            "gzip"
        );
        let mut gz = flate2::read::GzDecoder::new(&req.body[..]);
        let mut text = String::new();
        std::io::Read::read_to_string(&mut gz, &mut text).expect("gunzip");
        match serde_json::from_str(&text).expect("valid json") {
            Value::Array(a) => a,
            other => panic!("expected a JSON array body, got {other}"),
        }
    }

    fn sink_for(server: &MockServer, tags: &[String]) -> DatadogSink {
        // The mock server URI is `http://127.0.0.1:<port>` — a vetted loopback
        // site, so the sink posts to it directly over http.
        let host = server.uri().strip_prefix("http://").unwrap().to_string();
        DatadogSink::new(
            "datadog-test",
            &host,
            "test-dd-api-key",
            "aisix-ai-gateway",
            tags,
            "ai-gateway",
            reqwest::Client::new(),
        )
    }

    fn batch_of(records: Vec<SinkRecord>) -> EventBatch {
        EventBatch::new(records.into_iter().map(Arc::new).collect())
    }

    #[tokio::test]
    async fn appends_batch_posts_gzipped_json_with_api_key_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(POST_PATH))
            .respond_with(ResponseTemplate::new(202))
            .mount(&server)
            .await;

        let sink = sink_for(&server, &["team:platform".into(), "tier:prod".into()]);
        let event = UsageEvent {
            request_id: "req-42".into(),
            occurred_at: "2026-05-01T12:00:00Z".into(),
            model_id: "gpt-4o".into(),
            status_code: 200,
            prompt_tokens: 5,
            completion_tokens: 7,
            latency_ms: 123,
            provider_model_version: "gpt-4o-2024-08-06".into(),
            finish_reason: "stop".into(),
            ..UsageEvent::default()
        };
        let ack = sink
            .append_batch(
                &batch_of(vec![SinkRecord::metadata_only(event)]),
                &IdempotencyMarker::None,
            )
            .await
            .expect("delivery succeeds");
        assert_eq!(ack.accepted, 1);

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let req = &requests[0];

        // Wire shape: gzipped JSON to /api/v2/logs with the DD-API-KEY header.
        assert_eq!(req.method.as_str(), "POST");
        assert_eq!(req.url.path(), POST_PATH);
        assert_eq!(
            req.headers.get("content-type").unwrap().to_str().unwrap(),
            "application/json"
        );
        assert_eq!(
            req.headers.get("dd-api-key").unwrap().to_str().unwrap(),
            "test-dd-api-key"
        );

        // Body round-trips: one log object with the reserved attrs + GenAI
        // semconv token fields.
        let logs = decode_logs(req);
        assert_eq!(logs.len(), 1, "exactly one log object");
        let log = &logs[0];
        assert_eq!(log["ddsource"], "aisix-ai-gateway");
        assert_eq!(log["ddtags"], "team:platform,tier:prod");
        assert_eq!(log["service"], "ai-gateway");
        assert!(log["message"].as_str().unwrap().contains("req-42"));
        assert_eq!(log["schema_version"], "1.0");
        // GenAI semconv names for the LLM dimensions.
        assert_eq!(log["gen_ai.usage.input_tokens"], 5);
        assert_eq!(log["gen_ai.usage.output_tokens"], 7);
        assert_eq!(log["gen_ai.response.model"], "gpt-4o-2024-08-06");
        assert_eq!(log["gen_ai.response.finish_reason"], "stop");
        assert_eq!(log["http.response.status_code"], 200);
        // AISIX custom dimensions under the `aisix.` prefix.
        assert_eq!(log["aisix.request_id"], "req-42");
        assert_eq!(log["aisix.model_id"], "gpt-4o");
        assert_eq!(log["aisix.latency_ms"], 123);

        // The API key must NEVER appear in the body anywhere.
        let body_text = serde_json::to_string(&logs).unwrap();
        assert!(
            !body_text.contains("test-dd-api-key"),
            "api key must never be on the wire body: {body_text}"
        );
        // Empty metadata is omitted uniformly (no blank `aisix.api_key_id`).
        assert!(log.get("aisix.api_key_id").is_none());
        // Metadata-only path never carries a prompt.
        assert!(log.get("gen_ai.prompt").is_none());
    }

    #[tokio::test]
    async fn full_content_record_emits_prompt_and_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(POST_PATH))
            .respond_with(ResponseTemplate::new(202))
            .mount(&server)
            .await;

        let sink = sink_for(&server, &[]);
        let record = SinkRecord::metadata_only(UsageEvent {
            request_id: "req-content".into(),
            status_code: 200,
            ..UsageEvent::default()
        })
        .with_content(SinkContent {
            prompt: "what is 2+2?".into(),
            response: "4".into(),
            truncated: true,
        });
        sink.append_batch(&batch_of(vec![record]), &IdempotencyMarker::None)
            .await
            .expect("delivery succeeds");

        let requests = server.received_requests().await.unwrap();
        let logs = decode_logs(&requests[0]);
        let log = &logs[0];
        assert_eq!(log["gen_ai.prompt"], "what is 2+2?");
        assert_eq!(log["gen_ai.completion"], "4");
        assert_eq!(log["content_truncated"], true);
        // ddtags is omitted entirely when no tags are configured.
        assert!(log.get("ddtags").is_none(), "empty ddtags must be omitted");
    }

    #[tokio::test]
    async fn server_error_is_transient() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(POST_PATH))
            .respond_with(
                ResponseTemplate::new(503).set_body_string(r#"{"errors":["service unavailable"]}"#),
            )
            .mount(&server)
            .await;

        let err = sink_for(&server, &[])
            .append_batch(
                &batch_of(vec![SinkRecord::metadata_only(UsageEvent::default())]),
                &IdempotencyMarker::None,
            )
            .await
            .expect_err("5xx fails");
        assert!(err.is_transient(), "5xx must be retried: {err}");
    }

    #[tokio::test]
    async fn rate_limited_429_is_transient() {
        // Datadog signals back-pressure with 429; a log sink must back off and
        // retry, not drop the batch.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(POST_PATH))
            .respond_with(
                ResponseTemplate::new(429).set_body_string(r#"{"errors":["rate limit"]}"#),
            )
            .mount(&server)
            .await;

        let err = sink_for(&server, &[])
            .append_batch(
                &batch_of(vec![SinkRecord::metadata_only(UsageEvent::default())]),
                &IdempotencyMarker::None,
            )
            .await
            .expect_err("429 fails this attempt");
        assert!(err.is_transient(), "429 must be retried: {err}");
    }

    #[tokio::test]
    async fn auth_error_403_is_permanent() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(POST_PATH))
            .respond_with(ResponseTemplate::new(403).set_body_string(r#"{"errors":["Forbidden"]}"#))
            .mount(&server)
            .await;

        let err = sink_for(&server, &[])
            .append_batch(
                &batch_of(vec![SinkRecord::metadata_only(UsageEvent::default())]),
                &IdempotencyMarker::None,
            )
            .await
            .expect_err("bad api key fails");
        assert!(!err.is_transient(), "auth error must not be retried: {err}");
    }

    #[test]
    fn transient_status_classification() {
        use reqwest::StatusCode;
        for s in [
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
        ] {
            assert!(is_transient_status(s), "{s} should be transient");
        }
        for s in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::PAYLOAD_TOO_LARGE,
        ] {
            assert!(!is_transient_status(s), "{s} should be permanent");
        }
    }

    #[test]
    fn parse_error_extracts_messages_and_masks_body() {
        let detail = parse_datadog_error(
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"errors":["Invalid log format","bad ddsource"]}"#,
        );
        assert!(detail.contains("Invalid log format"));
        assert!(detail.contains("bad ddsource"));
        assert!(detail.contains("400"));

        // Non-envelope body falls back to the raw (truncated) text.
        let detail = parse_datadog_error(
            reqwest::StatusCode::BAD_GATEWAY,
            "<html>502 Bad Gateway</html>",
        );
        assert!(detail.contains("502"));
    }

    #[test]
    fn intake_url_builds_https_for_a_real_site() {
        assert_eq!(
            intake_url_for("datadoghq.com"),
            "https://http-intake.logs.datadoghq.com/api/v2/logs"
        );
        assert_eq!(
            intake_url_for("ap1.datadoghq.com"),
            "https://http-intake.logs.ap1.datadoghq.com/api/v2/logs"
        );
        // Trailing slash trimmed.
        assert_eq!(
            intake_url_for("datadoghq.eu/"),
            "https://http-intake.logs.datadoghq.eu/api/v2/logs"
        );
    }

    #[test]
    fn intake_url_uses_http_for_a_loopback_site() {
        // A loopback mock is posted to directly over http — no `http-intake`
        // prefix, no TLS.
        assert_eq!(
            intake_url_for("mock-datadog:8080"),
            "http://mock-datadog:8080/api/v2/logs"
        );
        assert_eq!(
            intake_url_for("127.0.0.1:9001"),
            "http://127.0.0.1:9001/api/v2/logs"
        );
        assert_eq!(intake_url_for("localhost"), "http://localhost/api/v2/logs");
    }

    #[test]
    fn resolve_credential_maps_reference_to_env_key() {
        // The ref folds to an upper `_`-joined slug; the key comes from
        // `DD_CRED_<SLUG>_API_KEY`.
        let store = |key: &str| match key {
            "DD_CRED_DATADOG_PROD_API_KEY" => Some("dd-key-123".to_string()),
            _ => None,
        };
        assert_eq!(
            resolve_datadog_credential_with("datadog-prod", store),
            Some("dd-key-123".to_string())
        );
        // Case-insensitive: `Datadog.Prod` folds to the same slug.
        assert_eq!(
            resolve_datadog_credential_with("Datadog.Prod", store),
            Some("dd-key-123".to_string())
        );
    }

    #[test]
    fn resolve_credential_is_none_when_unset_or_blank() {
        // Key absent → None.
        assert_eq!(resolve_datadog_credential_with("missing", |_| None), None);
        // Blank value is treated as unset (never POST with an empty key).
        assert_eq!(
            resolve_datadog_credential_with("x", |_| Some(String::new())),
            None
        );
    }
}
