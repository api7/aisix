//! Aliyun SLS (Simple Log Service) sink — the `http_batch` family's first
//! concrete vendor (AISIX-Cloud#687, the pre-sales deliverable on
//! ai-gateway#432).
//!
//! A batch becomes one SLS **PutLogs** request: every [`SinkRecord`] maps to
//! one protobuf `Log` (the canonical usage metadata flattened into key/value
//! contents, plus opt-in captured prompt/response), the `LogGroup` is
//! lz4-block compressed, signed with SLS signature v1 (HMAC-SHA1), and POSTed
//! to `/logstores/<logstore>/shards/lb` over the crate's shared rustls client.
//!
//! Wire details are taken from the official Aliyun SLS sub-crates
//! (`aliyun-log-sdk-protobuf`, `aliyun-log-sdk-sign`) rather than re-derived.
//! In particular the signer sets `Date` / `Content-MD5` / `Content-Length` /
//! `x-log-apiversion` / `x-log-signaturemethod` / `Authorization` itself and
//! signs over *every* `x-log-*` header, so `Content-Type` plus
//! `x-log-bodyrawsize` and `x-log-compresstype` are populated before the call.

use aliyun_log_sdk_protobuf::{Log, LogGroup};
use aliyun_log_sdk_sign::{sign_v1, QueryParams};
use async_trait::async_trait;
use http::header::CONTENT_TYPE;
use http::{HeaderMap, HeaderName, HeaderValue, Method};
use serde::Deserialize;

use super::{
    BatchUnit, EventBatch, IdempotencyMarker, IdempotencyScheme, ObservabilitySink, OrderingScope,
    SinkAck, SinkCapabilities, SinkError, SinkHealth, SinkRecord, SinkResult,
};

/// `x-log-bodyrawsize`: the *uncompressed* protobuf size. SLS uses it to size
/// the lz4 decompression buffer (the block format carries no length itself).
const X_LOG_BODYRAWSIZE: HeaderName = HeaderName::from_static("x-log-bodyrawsize");
/// `x-log-compresstype`: the body compression. This sink always lz4-block-
/// compresses, so the value is constant.
const X_LOG_COMPRESSTYPE: HeaderName = HeaderName::from_static("x-log-compresstype");

/// Conservative ceiling on a single PutLogs body, comfortably under SLS's
/// documented per-request limit. The shared pipeline never hands the sink a
/// batch larger than this; an oversize body would otherwise come back as a
/// permanent 400.
const SLS_MAX_REQUEST_BYTES: u64 = 3 * 1024 * 1024;

/// Cap on a masked error-detail string surfaced to logs / health.
const DETAIL_MAX_CHARS: usize = 200;

/// A delivery target for one Aliyun SLS logstore.
pub struct AliyunSlsSink {
    name: String,
    /// Full POST target — `https://<project>.<endpoint>/logstores/<ls>/shards/lb`.
    endpoint_url: String,
    /// The request path used for the SLS signature (no scheme/host/query).
    path: String,
    access_key_id: String,
    access_key_secret: String,
    client: reqwest::Client,
}

impl AliyunSlsSink {
    /// Build a sink for one `<project>/<logstore>` in an SLS region.
    ///
    /// `endpoint` is the bare region host, e.g.
    /// `ap-southeast-3.log.aliyuncs.com`; the request host is
    /// `<project>.<endpoint>`. An endpoint that already carries a scheme
    /// (the e2e's `http://mock-sls:9000`) is used verbatim so a local mock
    /// needs no `<project>.` DNS or TLS. The `client` is shared across sinks
    /// so connection pools and TLS sessions are reused.
    pub fn new(
        name: impl Into<String>,
        endpoint: &str,
        project: &str,
        logstore: &str,
        access_key_id: impl Into<String>,
        access_key_secret: impl Into<String>,
        client: reqwest::Client,
    ) -> Self {
        Self::from_base_url(
            name,
            base_url_for(endpoint, project),
            logstore,
            access_key_id,
            access_key_secret,
            client,
        )
    }

    /// Construct from an explicit `scheme://host[:port]` base. Production goes
    /// through [`AliyunSlsSink::new`]; tests point this at a mock server.
    fn from_base_url(
        name: impl Into<String>,
        base_url: impl Into<String>,
        logstore: &str,
        access_key_id: impl Into<String>,
        access_key_secret: impl Into<String>,
        client: reqwest::Client,
    ) -> Self {
        let path = format!("/logstores/{logstore}/shards/lb");
        let endpoint_url = format!("{}{}", base_url.into(), path);
        Self {
            name: name.into(),
            endpoint_url,
            path,
            access_key_id: access_key_id.into(),
            access_key_secret: access_key_secret.into(),
            client,
        }
    }
}

#[async_trait]
impl ObservabilitySink for AliyunSlsSink {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities {
            // SLS PutLogs is at-least-once: no server-side dedup token.
            idempotency: IdempotencyScheme::None,
            // Independent posts; SLS does not require cross-record ordering.
            ordering: OrderingScope::None,
            batch_unit: BatchUnit::Both,
            max_batch_bytes: Some(SLS_MAX_REQUEST_BYTES),
            // PutLogs accepts or rejects the whole LogGroup.
            supports_partial_batch: false,
            supports_streaming_ingest: false,
        }
    }

    async fn append_batch(&self, batch: &EventBatch, _marker: &IdempotencyMarker) -> SinkResult {
        if batch.is_empty() {
            return Ok(SinkAck::default());
        }

        // 1. One protobuf `Log` per record → a single `LogGroup`.
        let mut group = LogGroup::new();
        for record in &batch.records {
            group.add_log(to_log(record)?);
        }
        let raw = group
            .encode()
            .map_err(|e| SinkError::Permanent(format!("sls: protobuf encode: {e}")))?;
        let raw_size = raw.len();

        // 2. lz4 *block* compression (bare block; the size lives in the header).
        let compressed = lz4_flex::block::compress(&raw);

        // 3. Headers SLS signs over. The signer adds Date / Content-MD5 /
        //    Content-Length / apiversion / signaturemethod / Authorization and
        //    signs every `x-log-*`, so these three are set beforehand.
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/x-protobuf"),
        );
        headers.insert(X_LOG_BODYRAWSIZE, HeaderValue::from(raw_size as u64));
        headers.insert(X_LOG_COMPRESSTYPE, HeaderValue::from_static("lz4"));
        sign_v1(
            &self.access_key_id,
            &self.access_key_secret,
            None,
            Method::POST,
            &self.path,
            &mut headers,
            QueryParams::empty(),
            Some(&compressed),
        )
        .map_err(|e| SinkError::Permanent(format!("sls: sign request: {e}")))?;

        // 4. Deliver. The exact `compressed` bytes were signed (Content-MD5).
        let resp = match self
            .client
            .post(&self.endpoint_url)
            .headers(headers)
            .body(compressed)
            .send()
            .await
        {
            Ok(resp) => resp,
            // Connect / DNS / timeout — transient by nature.
            Err(e) => {
                return Err(SinkError::Transient(format!(
                    "sls: POST {}: {e}",
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
        let (error_code, detail) = parse_sls_error(status, &body);
        // 5xx / 408 / 429 are always worth retrying; SLS also signals back-
        // pressure (quota / busy / clock skew) on a 4xx with a code whose plain
        // meaning is transient, so promote those rather than drop a log batch.
        if status.is_server_error()
            || status == reqwest::StatusCode::REQUEST_TIMEOUT
            || status == reqwest::StatusCode::TOO_MANY_REQUESTS
            || is_transient_error_code(&error_code)
        {
            Err(SinkError::Transient(detail))
        } else {
            Err(SinkError::Permanent(detail))
        }
    }

    async fn healthcheck(&self) -> SinkHealth {
        // A real connectivity probe (and the control-plane "test connection"
        // affordance) lands with the health/metrics surface; until then a sink
        // reports healthy and its delivery errors surface via
        // `SinkStats::last_error`. (Mirrors `OtlpSink`.)
        SinkHealth::healthy()
    }
}

/// Map one canonical record to an SLS protobuf `Log`.
///
/// Metadata is produced by *serializing* [`crate::usage::UsageEvent`] rather
/// than hand-listing its 30-plus fields: that single-sources the schema and
/// inherits the exact `skip_serializing_if` emptiness rules the cp-api wire
/// already uses (empty strings / zero counters are omitted). The SLS log time
/// is ingest time (now); the event's own `occurred_at` is preserved as a
/// content field.
fn to_log(record: &SinkRecord) -> Result<Log, SinkError> {
    let mut log = Log::from_unixtime(now_unix_secs());
    log.add_content_kv("schema_version", record.schema_version);

    let usage = serde_json::to_value(&record.usage)
        .map_err(|e| SinkError::Permanent(format!("sls: serialize usage: {e}")))?;
    if let serde_json::Value::Object(fields) = usage {
        for (key, value) in fields {
            if let Some(rendered) = render_scalar(&value) {
                log.add_content_kv(key, rendered);
            }
        }
    }

    // Opt-in captured content as flat, queryable fields (the SLS customer case
    // wants prompt / response columns, not a nested blob). Absent on the
    // default metadata-only path, so prompts never leak there.
    if let Some(content) = &record.content {
        log.add_content_kv("prompt", content.prompt.as_str());
        log.add_content_kv("response", content.response.as_str());
        if content.truncated {
            log.add_content_kv("content_truncated", "true");
        }
    }

    Ok(log)
}

/// Render a JSON scalar as an SLS content value; `None` skips the field.
///
/// Empty strings are skipped so the SLS log omits blank columns uniformly:
/// most optional `UsageEvent` fields already drop out via
/// `skip_serializing_if`, but a few (`model_id`, `api_key_id`) carry only
/// `#[serde(default)]` and would otherwise serialize as `""`. Numeric `0` and
/// `false` are kept — a zero token count or status code is real data.
fn render_scalar(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) if s.is_empty() => None,
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        // Nested fields (e.g. routing_attempts) round-trip as compact JSON in
        // a single content value.
        other => Some(other.to_string()),
    }
}

/// The SLS JSON error envelope — `{"errorCode": ..., "errorMessage": ...}`.
#[derive(Deserialize)]
struct SlsErrorBody {
    #[serde(rename = "errorCode", default)]
    error_code: String,
    #[serde(rename = "errorMessage", default)]
    error_message: String,
}

/// Parse the SLS error body into `(errorCode, masked detail)`, falling back to
/// the raw (truncated) body when it isn't the JSON envelope. The detail never
/// contains the access key — SLS error messages echo neither id nor secret.
fn parse_sls_error(status: reqwest::StatusCode, body: &str) -> (String, String) {
    if let Ok(parsed) = serde_json::from_str::<SlsErrorBody>(body) {
        if !parsed.error_code.is_empty() {
            let detail = truncate(&format!(
                "HTTP {status}: {} {}",
                parsed.error_code, parsed.error_message
            ));
            return (parsed.error_code, detail);
        }
    }
    (String::new(), truncate(&format!("HTTP {status}: {body}")))
}

/// Whether an SLS `errorCode` means "retry with backoff", keyed on the plain
/// semantics of the code string. SLS returns throttling / overload / clock
/// skew on a 4xx in some configurations, which a status-only check would
/// mis-classify as permanent and silently drop a log batch. See the SLS error
/// code reference:
/// <https://www.alibabacloud.com/help/en/sls/developer-reference/api-error-codes>.
fn is_transient_error_code(code: &str) -> bool {
    const TRANSIENT_TOKENS: [&str; 6] = [
        "QuotaExceed",          // Write/Read/ShardWriteQuotaExceed — throttle
        "Busy",                 // ServerBusy — transient overload
        "InternalServerError",  // server-side, retryable
        "Unavailable",          // ServiceUnavailable
        "RequestTimeExpired",   // stale Date — a fresh re-sign on retry fixes it
        "RequestTimeTooSkewed", // clock skew — likewise
    ];
    TRANSIENT_TOKENS.iter().any(|token| code.contains(token))
}

/// Truncate a masked detail string to a bounded length for logs / health.
fn truncate(s: &str) -> String {
    s.chars().take(DETAIL_MAX_CHARS).collect()
}

/// Compute the base URL (`scheme://host[:port]`) the sink posts to.
///
/// A bare SLS region host becomes `https://<project>.<host>` — the real
/// PutLogs target. An endpoint that already carries a scheme (the e2e's
/// `http://mock-sls:9000`) is used verbatim, so a local mock receiver needs
/// no `<project>.` DNS or TLS. The exporter schema only admits a
/// scheme-qualified endpoint for vetted loopback hosts, so this branch can't
/// be used to redirect real traffic.
fn base_url_for(endpoint: &str, project: &str) -> String {
    let endpoint = endpoint.trim().trim_end_matches('/');
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.to_string()
    } else {
        format!("https://{project}.{endpoint}")
    }
}

/// Resolve an exporter's `credential_ref` to `(access_key_id,
/// access_key_secret)` from the DP's local environment.
///
/// The AccessKey never travels on the kine path (the control plane stores
/// only the reference), so the DP looks it up where it actually runs. The
/// reference is upper-cased with non-alphanumerics folded to `_`, then read
/// from `SLS_CRED_<REF>_AK_ID` / `SLS_CRED_<REF>_AK_SECRET`. The prefix is
/// deliberately NOT `AISIX_`: that namespace is owned by the config loader
/// (`Environment::with_prefix("AISIX")`), so an `AISIX_`-named secret would be
/// reinterpreted as a config override. Returns `None` when either half is
/// unset or blank — the caller then lets the misconfiguration surface as a
/// delivery-health auth error rather than signing with an empty key. BYOK
/// variants (customer KMS / uploaded key) plug in here as alternative
/// resolution paths keyed off the same reference.
pub fn resolve_sls_credential(credential_ref: &str) -> Option<(String, String)> {
    resolve_sls_credential_with(credential_ref, |key| std::env::var(key).ok())
}

/// Reference-resolution core, parameterized over the variable source so it is
/// testable without mutating the process environment.
fn resolve_sls_credential_with(
    credential_ref: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Option<(String, String)> {
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
    let id = lookup(&format!("SLS_CRED_{slug}_AK_ID"))?;
    let secret = lookup(&format!("SLS_CRED_{slug}_AK_SECRET"))?;
    if id.is_empty() || secret.is_empty() {
        return None;
    }
    Some((id, secret))
}

/// Unix seconds (UTC) for the SLS log time. SLS expects a `u32` (valid until
/// 2106); falls back to 0 only if the clock predates the epoch.
fn now_unix_secs() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{
        base_url_for, is_transient_error_code, parse_sls_error, resolve_sls_credential_with,
        AliyunSlsSink,
    };
    use crate::sink::{EventBatch, IdempotencyMarker, ObservabilitySink, SinkContent, SinkRecord};
    use crate::usage::UsageEvent;
    use aliyun_log_sdk_protobuf::LogGroupList;
    use std::collections::HashMap;
    use std::sync::Arc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const LOGSTORE: &str = "request-events";
    const POST_PATH: &str = "/logstores/request-events/shards/lb";

    /// Wrap a bare `LogGroup` message as the single element of a `LogGroupList`
    /// (protobuf field #1, length-delimited) so the SDK decoder reads it back.
    fn wrap_as_log_group_list(group: &[u8]) -> Vec<u8> {
        let mut out = vec![0x0A]; // tag: field 1, wire type 2 (length-delimited)
        let mut len = group.len();
        loop {
            let mut byte = (len & 0x7F) as u8;
            len >>= 7;
            if len != 0 {
                byte |= 0x80; // continuation bit
            }
            out.push(byte);
            if len == 0 {
                break;
            }
        }
        out.extend_from_slice(group);
        out
    }

    /// Decode the request body a mock server captured back into the flat
    /// key→value content map of its single log (lz4-decompress using the
    /// raw-size header, then protobuf-decode).
    fn decode_single_log(req: &wiremock::Request) -> HashMap<String, String> {
        let raw_size: usize = req
            .headers
            .get("x-log-bodyrawsize")
            .expect("x-log-bodyrawsize header")
            .to_str()
            .unwrap()
            .parse()
            .expect("raw size is a number");
        let raw = lz4_flex::block::decompress(&req.body, raw_size).expect("lz4 decompress");
        let list = LogGroupList::decode(&wrap_as_log_group_list(&raw)).expect("protobuf decode");
        let groups = list.log_groups();
        assert_eq!(groups.len(), 1, "exactly one LogGroup");
        let logs = groups[0].logs();
        assert_eq!(logs.len(), 1, "exactly one Log");
        logs[0]
            .contents()
            .iter()
            .map(|c| (c.key().clone(), c.value().clone()))
            .collect()
    }

    fn sink_for(server: &MockServer) -> AliyunSlsSink {
        AliyunSlsSink::from_base_url(
            "sls-test",
            server.uri(),
            LOGSTORE,
            "test-akid",
            "test-secret",
            reqwest::Client::new(),
        )
    }

    fn batch_of(records: Vec<SinkRecord>) -> EventBatch {
        EventBatch::new(records.into_iter().map(Arc::new).collect())
    }

    #[tokio::test]
    async fn appends_batch_posts_signed_lz4_protobuf() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(POST_PATH))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let sink = sink_for(&server);
        let event = UsageEvent {
            request_id: "req-42".into(),
            occurred_at: "2026-05-01T12:00:00Z".into(),
            model_id: "gpt-4o".into(),
            status_code: 200,
            prompt_tokens: 5,
            completion_tokens: 7,
            latency_ms: 123,
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

        // Wire shape: signed PutLogs over lz4 protobuf.
        assert_eq!(req.method.as_str(), "POST");
        assert_eq!(req.url.path(), POST_PATH);
        assert_eq!(
            req.headers.get("content-type").unwrap().to_str().unwrap(),
            "application/x-protobuf"
        );
        assert_eq!(
            req.headers
                .get("x-log-compresstype")
                .unwrap()
                .to_str()
                .unwrap(),
            "lz4"
        );
        assert_eq!(
            req.headers
                .get("x-log-apiversion")
                .unwrap()
                .to_str()
                .unwrap(),
            "0.6.0"
        );
        let auth = req.headers.get("authorization").unwrap().to_str().unwrap();
        assert!(
            auth.starts_with("LOG test-akid:"),
            "signature carries the access-key id, not the secret: {auth}"
        );
        assert!(
            !auth.contains("test-secret"),
            "secret must never be on the wire"
        );
        assert!(req.headers.contains_key("date"));
        assert!(req.headers.contains_key("content-md5"));

        // Body round-trips: the mapped fields land as flat contents.
        let contents = decode_single_log(req);
        assert_eq!(
            contents.get("schema_version").map(String::as_str),
            Some("1.0")
        );
        assert_eq!(
            contents.get("request_id").map(String::as_str),
            Some("req-42")
        );
        assert_eq!(
            contents.get("occurred_at").map(String::as_str),
            Some("2026-05-01T12:00:00Z")
        );
        assert_eq!(contents.get("model_id").map(String::as_str), Some("gpt-4o"));
        assert_eq!(contents.get("status_code").map(String::as_str), Some("200"));
        assert_eq!(contents.get("prompt_tokens").map(String::as_str), Some("5"));
        assert_eq!(
            contents.get("completion_tokens").map(String::as_str),
            Some("7")
        );
        assert_eq!(contents.get("latency_ms").map(String::as_str), Some("123"));
        // Empty metadata is omitted uniformly: `api_key_id` (serde `default`
        // only, would serialize as "") and `finish_reason` (`skip_serializing_if`)
        // both drop out, so the SLS log carries no blank columns.
        assert!(!contents.contains_key("api_key_id"));
        assert!(!contents.contains_key("finish_reason"));
        // Metadata-only path never carries a prompt.
        assert!(!contents.contains_key("prompt"));
    }

    #[tokio::test]
    async fn full_content_record_emits_prompt_and_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(POST_PATH))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let sink = sink_for(&server);
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
        let contents = decode_single_log(&requests[0]);
        assert_eq!(
            contents.get("prompt").map(String::as_str),
            Some("what is 2+2?")
        );
        assert_eq!(contents.get("response").map(String::as_str), Some("4"));
        assert_eq!(
            contents.get("content_truncated").map(String::as_str),
            Some("true")
        );
    }

    #[tokio::test]
    async fn server_error_is_transient() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(POST_PATH))
            .respond_with(ResponseTemplate::new(500).set_body_string(
                r#"{"errorCode":"InternalServerError","errorMessage":"try again"}"#,
            ))
            .mount(&server)
            .await;

        let err = sink_for(&server)
            .append_batch(
                &batch_of(vec![SinkRecord::metadata_only(UsageEvent::default())]),
                &IdempotencyMarker::None,
            )
            .await
            .expect_err("5xx fails");
        assert!(err.is_transient(), "5xx must be retried: {err}");
    }

    #[tokio::test]
    async fn auth_error_is_permanent() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(POST_PATH))
            .respond_with(ResponseTemplate::new(403).set_body_string(
                r#"{"errorCode":"SignatureNotMatch","errorMessage":"signature mismatch"}"#,
            ))
            .mount(&server)
            .await;

        let err = sink_for(&server)
            .append_batch(
                &batch_of(vec![SinkRecord::metadata_only(UsageEvent::default())]),
                &IdempotencyMarker::None,
            )
            .await
            .expect_err("bad signature fails");
        assert!(!err.is_transient(), "auth error must not be retried: {err}");
    }

    #[tokio::test]
    async fn quota_exceeded_on_4xx_is_transient() {
        // SLS signals shard write throttling with a 4xx + a quota code; a log
        // sink must back off and retry, not drop the batch.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(POST_PATH))
            .respond_with(ResponseTemplate::new(403).set_body_string(
                r#"{"errorCode":"WriteQuotaExceed","errorMessage":"shard write quota exceeded"}"#,
            ))
            .mount(&server)
            .await;

        let err = sink_for(&server)
            .append_batch(
                &batch_of(vec![SinkRecord::metadata_only(UsageEvent::default())]),
                &IdempotencyMarker::None,
            )
            .await
            .expect_err("quota exceeded fails this attempt");
        assert!(
            err.is_transient(),
            "quota-exceed on a 4xx must be retried: {err}"
        );
    }

    #[test]
    fn transient_error_codes_are_recognised() {
        for code in [
            "WriteQuotaExceed",
            "ShardWriteQuotaExceed",
            "ReadQuotaExceed",
            "ServerBusy",
            "InternalServerError",
            "ServiceUnavailable",
            "RequestTimeExpired",
            "RequestTimeTooSkewed",
        ] {
            assert!(is_transient_error_code(code), "{code} should be transient");
        }
        for code in [
            "SignatureNotMatch",
            "Unauthorized",
            "ParameterInvalid",
            "PostBodyInvalid",
            "LogStoreNotExist",
            "",
        ] {
            assert!(!is_transient_error_code(code), "{code} should be permanent");
        }
    }

    #[test]
    fn parse_sls_error_extracts_code_and_masks_body() {
        let (code, detail) = parse_sls_error(
            reqwest::StatusCode::FORBIDDEN,
            r#"{"errorCode":"SignatureNotMatch","errorMessage":"bad sig"}"#,
        );
        assert_eq!(code, "SignatureNotMatch");
        assert!(detail.contains("SignatureNotMatch"));
        assert!(detail.contains("403"));

        // Non-envelope body falls back to the raw (truncated) text.
        let (code, detail) = parse_sls_error(reqwest::StatusCode::BAD_GATEWAY, "<html>502</html>");
        assert_eq!(code, "");
        assert!(detail.contains("502"));
    }

    #[test]
    fn base_url_prepends_project_for_a_region_host() {
        assert_eq!(
            base_url_for("ap-southeast-3.log.aliyuncs.com", "aisix-obs"),
            "https://aisix-obs.ap-southeast-3.log.aliyuncs.com"
        );
        // Trailing slash trimmed.
        assert_eq!(
            base_url_for("cn-hangzhou.log.aliyuncs.com/", "p"),
            "https://p.cn-hangzhou.log.aliyuncs.com"
        );
    }

    #[test]
    fn base_url_uses_a_scheme_qualified_endpoint_verbatim() {
        // A loopback mock is posted to directly — no `<project>.` prefix.
        assert_eq!(
            base_url_for("http://mock-sls:9000", "aisix-obs"),
            "http://mock-sls:9000"
        );
    }

    #[test]
    fn resolve_credential_maps_reference_to_env_keys() {
        // The ref folds to an upper `_`-joined slug; both halves come from
        // `SLS_CRED_<SLUG>_AK_{ID,SECRET}`.
        let store = |key: &str| match key {
            "SLS_CRED_SLS_PROD_AK_ID" => Some("akid-123".to_string()),
            "SLS_CRED_SLS_PROD_AK_SECRET" => Some("secret-456".to_string()),
            _ => None,
        };
        assert_eq!(
            resolve_sls_credential_with("sls-prod", store),
            Some(("akid-123".to_string(), "secret-456".to_string()))
        );
        // Case-insensitive: `SLS.Prod` folds to the same slug.
        assert_eq!(
            resolve_sls_credential_with("SLS.Prod", store),
            Some(("akid-123".to_string(), "secret-456".to_string()))
        );
    }

    #[test]
    fn resolve_credential_is_none_when_unset_or_blank() {
        // Neither half present.
        assert_eq!(resolve_sls_credential_with("missing", |_| None), None);
        // Id present, secret missing → still None (never sign half-credentialed).
        let id_only = |key: &str| (key == "SLS_CRED_X_AK_ID").then(|| "id".to_string());
        assert_eq!(resolve_sls_credential_with("x", id_only), None);
        // Blank value is treated as unset.
        assert_eq!(
            resolve_sls_credential_with("x", |_| Some(String::new())),
            None
        );
    }
}

/// Real-SLS smoke test — the one-off validation that Aliyun actually accepts
/// our signed PutLogs (the signature / lz4 / protobuf a mock receiver cannot
/// check). `#[ignore]` + env-gated, so the normal suite never touches the
/// network; run against real SLS with the creds present via
/// `cargo test -p aisix-obs -- --ignored sls_smoke`.
#[cfg(test)]
mod smoke {
    use super::{base_url_for, now_unix_secs, AliyunSlsSink};
    use crate::sink::{EventBatch, IdempotencyMarker, ObservabilitySink, SinkError, SinkRecord};
    use crate::usage::UsageEvent;
    use aliyun_log_sdk_sign::{sign_v1, QueryParams};
    use http::header::CONTENT_TYPE;
    use http::{HeaderMap, HeaderValue, Method};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn env(key: &str) -> Option<String> {
        std::env::var(key).ok().filter(|v| !v.is_empty())
    }

    #[tokio::test]
    #[ignore = "hits real Aliyun SLS; needs AISIX_E2E_SLS_* creds. \
                Run: cargo test -p aisix-obs -- --ignored sls_smoke"]
    async fn sls_smoke_putlogs_and_readback() {
        // Env-gate so `--ignored` is safe without creds (CI injects them).
        let (Some(ak_id), Some(ak_secret), Some(endpoint), Some(project), Some(logstore)) = (
            env("AISIX_E2E_SLS_AK_ID"),
            env("AISIX_E2E_SLS_AK_SECRET"),
            env("AISIX_E2E_SLS_ENDPOINT"),
            env("AISIX_E2E_SLS_PROJECT"),
            env("AISIX_E2E_SLS_LOGSTORE"),
        ) else {
            eprintln!("sls_smoke: AISIX_E2E_SLS_* not set — skipping real-SLS smoke");
            return;
        };

        let client = reqwest::Client::new();
        let marker = format!("smoke-{}", uuid::Uuid::new_v4());

        // ── 1. PutLogs through the real sink path (the hard assertion). ──
        // A 2xx here means Aliyun accepted our v1 signature, lz4 block, and
        // protobuf LogGroup — exactly what a mock receiver cannot validate.
        let sink = AliyunSlsSink::new(
            "smoke",
            &endpoint,
            &project,
            &logstore,
            ak_id.clone(),
            ak_secret.clone(),
            client.clone(),
        );
        let event = UsageEvent {
            request_id: marker.clone(),
            occurred_at: "2026-01-01T00:00:00Z".into(),
            model_id: "smoke-model".into(),
            status_code: 200,
            prompt_tokens: 1,
            completion_tokens: 2,
            latency_ms: 7,
            ..UsageEvent::default()
        };
        let batch = EventBatch::new(vec![Arc::new(SinkRecord::metadata_only(event))]);
        // A Permanent error (bad signature / auth / payload) is a real
        // regression — fail fast. A Transient one (network / 5xx) is retried a
        // few times so an SLS blip doesn't flake the CI gate.
        let mut transient: Option<SinkError> = None;
        for attempt in 0..3 {
            match sink.append_batch(&batch, &IdempotencyMarker::None).await {
                Ok(_) => {
                    transient = None;
                    break;
                }
                Err(e @ SinkError::Permanent(_)) => {
                    panic!("real SLS rejected the signed PutLogs (permanent): {e}")
                }
                Err(e) => {
                    eprintln!("sls_smoke: PutLogs attempt {attempt} transient: {e}");
                    transient = Some(e);
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
            }
        }
        if let Some(e) = transient {
            panic!("real SLS PutLogs failed after retries (transient): {e}");
        }
        eprintln!("sls_smoke: PutLogs accepted by real SLS (request_id={marker})");

        // ── 2. GetLogs readback (best-effort confirmation). ──
        // Whether the event is queryable depends on the logstore having an
        // index and SLS's indexing lag, so a miss is reported, not asserted —
        // the PutLogs 2xx above is the real signal. The diagnostics make the
        // first real run debuggable so this can later be hardened to assert.
        let base_url = base_url_for(&endpoint, &project);
        match readback(&client, &base_url, &logstore, &ak_id, &ak_secret, &marker).await {
            Ok(true) => eprintln!("sls_smoke: readback found request_id={marker}"),
            Ok(false) => eprintln!(
                "sls_smoke: readback did NOT find request_id={marker} within timeout \
                 (logstore index / indexing lag?) — PutLogs still succeeded"
            ),
            Err(e) => eprintln!("sls_smoke: readback error: {e} — PutLogs still succeeded"),
        }
    }

    /// Poll GetLogs (`POST /logstores/<ls>/logs`, signed like PutLogs) for the
    /// marker request_id. Returns `Ok(true)` once found, `Ok(false)` after the
    /// poll budget, `Err` on a transport/encode failure.
    async fn readback(
        client: &reqwest::Client,
        base_url: &str,
        logstore: &str,
        ak_id: &str,
        ak_secret: &str,
        marker: &str,
    ) -> Result<bool, String> {
        let path = format!("/logstores/{logstore}/logs");
        let url = format!("{base_url}{path}");
        let now = now_unix_secs() as i64;
        for attempt in 0..20 {
            let body = serde_json::json!({
                "from": now - 600,
                "to": now + 120,
                "query": format!("request_id: \"{marker}\""),
                "line": 100,
                "offset": 0,
                "reverse": true,
            });
            let body_bytes = serde_json::to_vec(&body).map_err(|e| e.to_string())?;
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            sign_v1(
                ak_id,
                ak_secret,
                None,
                Method::POST,
                &path,
                &mut headers,
                QueryParams::empty(),
                Some(&body_bytes),
            )
            .map_err(|e| format!("sign: {e}"))?;
            let resp = client
                .post(&url)
                .headers(headers)
                .body(body_bytes)
                .send()
                .await
                .map_err(|e| e.to_string())?;
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                eprintln!(
                    "sls_smoke: GetLogs attempt {attempt}: HTTP {status}: {}",
                    text.chars().take(200).collect::<String>()
                );
            } else if response_contains(&text, marker) {
                return Ok(true);
            }
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }
        Ok(false)
    }

    /// True if any returned log's `request_id` equals `marker`. Tolerates both
    /// the `{ "data": [...] }` envelope and a bare `[...]` array.
    fn response_contains(text: &str, marker: &str) -> bool {
        #[derive(serde::Deserialize)]
        struct Envelope {
            data: Vec<HashMap<String, String>>,
        }
        let logs: Vec<HashMap<String, String>> = serde_json::from_str::<Envelope>(text)
            .map(|e| e.data)
            .or_else(|_| serde_json::from_str::<Vec<HashMap<String, String>>>(text))
            .unwrap_or_default();
        logs.iter()
            .any(|log| log.get("request_id").map(String::as_str) == Some(marker))
    }
}
