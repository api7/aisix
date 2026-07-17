//! kind=aliyun_text_moderation guardrail dispatcher — calls Aliyun's
//! content-safety guardrail (`TextModerationPlus`) on chat input and/or
//! output and translates the returned `RiskLevel` into a
//! [`GuardrailVerdict`].
//!
//! Issue #603.
//!
//! API reference (action version 2022-03-02, RPC-style):
//! POST `https://green-cip.<region>.aliyuncs.com/`
//! Source: <https://help.aliyun.com/zh/document_detail/2671445.html>
//!
//! Wire shape:
//! ```text
//! // Request (form-urlencoded, RPC signature v1):
//! //   Action=TextModerationPlus&Version=2022-03-02&Service=llm_query_moderation
//! //   &ServiceParameters={"content":"...","sessionId":"..."}&Signature=...
//! // Response (HTTP 200):
//! { "Code": 200, "Data": { "RiskLevel": "high|medium|low|none",
//!   "Result": [ { "Label": "..." } ] }, "RequestId": "..." }
//! ```
//!
//! `Code` is typed inconsistently and the difference matters: on HTTP 200
//! it is an integer (`200`, or a business error such as `400` "service is
//! invalid"), but on an HTTP error it is a symbolic string
//! (`"InvalidAccessKeyId.NotFound"`). Only the HTTP-200 shape is
//! deserialized as [`AliyunResponse`]; the error shape is logged as a raw
//! capped body.
//!
//! Block decision: the returned `RiskLevel` rank (none<low<medium<high)
//! reaches the configured `risk_level_threshold`.
//!
//! Service codes: the INPUT hook uses `llm_query_moderation`, the OUTPUT
//! hook `llm_response_moderation`.
//!
//! There is no official Aliyun Rust SDK, so the RPC signature (v1,
//! HMAC-SHA1) is hand-rolled below. v1 is used over v3 because its
//! canonicalization is unambiguous for RPC-style products and is pinned
//! by a known-vector unit test.
//!
//! Streaming output is moderated incrementally via the windowed
//! [`StreamOutputPolicy`] in `aisix-proxy`'s `build_sse_stream`; each
//! window is sent with the stream's stable `provider_request_id` as the
//! Aliyun `sessionId` so Aliyun correlates the chunks of one response.

use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{AliyunTextModerationConfig, GuardrailHookPoint};
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;
use base64::Engine as _;
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha1::Sha1;

use crate::{Guardrail, GuardrailVerdict, StreamOutputPolicy};

type HmacSha1 = Hmac<Sha1>;

const ACTION: &str = "TextModerationPlus";
const API_VERSION: &str = "2022-03-02";
const SERVICE_INPUT: &str = "llm_query_moderation";
const SERVICE_OUTPUT: &str = "llm_response_moderation";

/// Per-call content cap (chars). Aliyun caps `llm_query_moderation` at
/// 2 000 and `llm_response_moderation` at 5 000; 2 000 is the safe shared
/// bound and matches the default streaming window.
const MAX_CONTENT_CHARS: usize = 2_000;

/// One Aliyun Text Moderation row, materialised into a request-time
/// dispatcher.
pub struct AliyunTextModerationGuardrail {
    row_name: String,
    /// Full endpoint base, no trailing slash (e.g.
    /// `https://green-cip.cn-shanghai.aliyuncs.com`).
    endpoint: String,
    region: String,
    access_key_id: String,
    access_key_secret: String,
    pub(crate) hook_point: GuardrailHookPoint,
    /// Fail-open policy for the INPUT hook (from the outer `Guardrail`).
    fail_open: bool,
    /// Fail-open policy for the OUTPUT hook. Defaults `false` (fail-closed)
    /// so an Aliyun outage can't release unscanned model output.
    output_fail_open: bool,
    /// Minimum returned risk rank that blocks (none=0 … high=3).
    threshold_rank: u8,
    pub(crate) timeout: Duration,
    client: Arc<reqwest::Client>,

    // --- streaming-output controls (surfaced via stream_output_policy) ---
    stream_processing_mode: String,
    window_size: u32,
    window_overlap_size: u32,
    max_buffer_bytes: u64,
    on_buffer_exceeded: String,
}

impl AliyunTextModerationGuardrail {
    pub fn new(
        row_name: impl Into<String>,
        cfg: &AliyunTextModerationConfig,
        hook_point: GuardrailHookPoint,
        fail_open: bool,
    ) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .expect("reqwest::Client::builder() failed; this should never happen");
        let endpoint = cfg
            .endpoint
            .clone()
            .unwrap_or_else(|| format!("https://green-cip.{}.aliyuncs.com", cfg.region));
        Self {
            row_name: row_name.into(),
            endpoint: endpoint.trim_end_matches('/').to_owned(),
            region: cfg.region.clone(),
            access_key_id: cfg.access_key_id.clone(),
            access_key_secret: cfg.access_key_secret.clone(),
            hook_point,
            fail_open,
            output_fail_open: cfg.output_fail_open,
            threshold_rank: risk_rank(&cfg.risk_level_threshold),
            timeout: Duration::from_millis(cfg.timeout_ms as u64),
            client: Arc::new(client),
            stream_processing_mode: cfg.stream_processing_mode.clone(),
            window_size: cfg.window_size,
            window_overlap_size: cfg.window_overlap_size,
            max_buffer_bytes: cfg.max_buffer_bytes,
            on_buffer_exceeded: cfg.on_buffer_exceeded.clone(),
        }
    }

    /// Moderate one piece of text with the given service code. `session_id`
    /// (when set) is forwarded as `ServiceParameters.sessionId` so Aliyun
    /// correlates the chunks of one streamed response.
    async fn moderate(
        &self,
        service: &str,
        text: &str,
        session_id: Option<&str>,
        fail_open: bool,
    ) -> GuardrailVerdict {
        // Aliyun caps content per call; truncate to the cap. Streaming
        // already windows to MAX_CONTENT_CHARS; non-streaming long inputs
        // are clamped (the leading content carries the risk in practice).
        let content: String = text.chars().take(MAX_CONTENT_CHARS).collect();
        let (outcome, diag) = self.call(service, &content, session_id).await;
        match outcome {
            Ok(level) => {
                let blocked = risk_rank(&level) >= self.threshold_rank;
                // The upstream diagnostics land here, once, with the
                // verdict known — rather than at each exit of `call()`,
                // which can't see it. A block is what an operator traces
                // back from a caller's 422, so it logs at info (the
                // default level); a clean pass would be one line per
                // request, so it stays at debug.
                if blocked {
                    tracing::info!(
                        row = %self.row_name,
                        service,
                        aliyun_request_id = %diag.request_id,
                        aliyun_code = %diag.code,
                        aliyun_risk_level = %diag.risk_level,
                        aliyun_labels = %diag.labels_field(),
                        "aliyun text moderation blocked content",
                    );
                } else {
                    tracing::debug!(
                        row = %self.row_name,
                        service,
                        aliyun_request_id = %diag.request_id,
                        aliyun_code = %diag.code,
                        aliyun_risk_level = %diag.risk_level,
                        aliyun_labels = %diag.labels_field(),
                        "aliyun text moderation passed content",
                    );
                }
                if blocked {
                    GuardrailVerdict::block(format!(
                        "aliyun text moderation: risk level {} >= threshold (row: {})",
                        level, self.row_name
                    ))
                } else {
                    GuardrailVerdict::Allow
                }
            }
            Err(failure) => self.handle_failure(failure, &diag, fail_open),
        }
    }

    /// Sign + POST one `TextModerationPlus` call; return the response
    /// `RiskLevel` (lowercased, `"none"` when absent) alongside whatever
    /// upstream diagnostics the call yielded.
    ///
    /// Diagnostics come back on BOTH arms on purpose: the failure arms are
    /// exactly the ones an operator needs `aliyun_request_id` for, and
    /// returning them rather than logging in place lets the one call site
    /// log once with the verdict attached (AISIX-Cloud#1060).
    async fn call(
        &self,
        service: &str,
        content: &str,
        session_id: Option<&str>,
    ) -> (Result<String, AliyunFailure>, AliyunDiagnostics) {
        let mut svc_params = serde_json::Map::new();
        svc_params.insert(
            "content".into(),
            serde_json::Value::String(content.to_owned()),
        );
        if let Some(sid) = session_id {
            if !sid.is_empty() {
                svc_params.insert(
                    "sessionId".into(),
                    serde_json::Value::String(sid.to_owned()),
                );
            }
        }
        let service_parameters = serde_json::Value::Object(svc_params).to_string();

        let nonce = uuid::Uuid::new_v4().to_string();
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        // Common + business params. BTreeMap keeps them sorted by key, which
        // is exactly the canonicalization order the v1 signature requires.
        let mut params: std::collections::BTreeMap<&str, String> =
            std::collections::BTreeMap::new();
        params.insert("AccessKeyId", self.access_key_id.clone());
        params.insert("Action", ACTION.to_owned());
        params.insert("Format", "JSON".to_owned());
        params.insert("RegionId", self.region.clone());
        params.insert("Service", service.to_owned());
        params.insert("ServiceParameters", service_parameters);
        params.insert("SignatureMethod", "HMAC-SHA1".to_owned());
        params.insert("SignatureNonce", nonce);
        params.insert("SignatureVersion", "1.0".to_owned());
        params.insert("Timestamp", timestamp);
        params.insert("Version", API_VERSION.to_owned());

        let signature = sign(&params, &self.access_key_secret);

        // Body = signed params + Signature, form-urlencoded (RFC3986 — the
        // same encoding used to build the signature, so the server re-derives
        // an identical StringToSign).
        let mut body = String::new();
        for (k, v) in &params {
            if !body.is_empty() {
                body.push('&');
            }
            body.push_str(k);
            body.push('=');
            body.push_str(&percent_encode(v));
        }
        body.push_str("&Signature=");
        body.push_str(&percent_encode(&signature));

        let future = self
            .client
            .post(format!("{}/", self.endpoint))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "application/json")
            .body(body)
            .send();

        // No response means no diagnostics to report: an id Aliyun never
        // sent can't be invented. `request_id` stays empty and the failure
        // bucket carries the whole story.
        let resp = match tokio::time::timeout(self.timeout, future).await {
            Err(_elapsed) => return (Err(AliyunFailure::Timeout), AliyunDiagnostics::default()),
            Ok(Err(_e)) => return (Err(AliyunFailure::IoError), AliyunDiagnostics::default()),
            Ok(Ok(r)) => r,
        };

        // Read the id off the headers up front: it survives every path
        // below, including the ones where the body is unusable.
        let mut diag = AliyunDiagnostics::from_headers(resp.headers());

        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return (Err(AliyunFailure::Throttled), diag);
        }
        if status.is_server_error() {
            return (Err(AliyunFailure::ServerError), diag);
        }
        if !status.is_success() {
            // Echo the provider's actual reason (e.g. `InvalidAccessKeyId.NotFound`
            // or a 404 HTML page) — a bare status code can't tell a wrong access
            // key from a wrong region/endpoint. The body is NOT parsed as
            // `AliyunResponse` here: on an HTTP error Aliyun types `Code` as a
            // symbolic string rather than the int it uses on HTTP 200, so the
            // struct wouldn't deserialize. The raw capped body already carries
            // the code and message verbatim.
            let response_body = crate::read_error_body_capped(resp).await;
            tracing::error!(
                row = %self.row_name,
                aliyun_request_id = %diag.request_id,
                http_status = status.as_u16(),
                response_body = %response_body,
                "aliyun TextModerationPlus returned 4xx — check region/access keys configuration",
            );
            return (Err(AliyunFailure::ConfigError), diag);
        }

        let body: AliyunResponse = match resp.json().await {
            Ok(b) => b,
            Err(_) => return (Err(AliyunFailure::MalformedResponse), diag),
        };
        diag.absorb_body(&body);

        // Aliyun signals app-level errors via the JSON `Code` (200 = OK)
        // even on HTTP 200.
        let outcome = match body.code {
            200 => Ok(if diag.risk_level.is_empty() {
                "none".to_owned()
            } else {
                diag.risk_level.to_lowercase()
            }),
            408 | 401 | 403 | 400 => {
                tracing::error!(
                    row = %self.row_name,
                    aliyun_request_id = %diag.request_id,
                    aliyun_code = %diag.code,
                    aliyun_message = %diag.message_field(),
                    "aliyun TextModerationPlus auth/permission error — check access keys",
                );
                Err(AliyunFailure::ConfigError)
            }
            _ => {
                tracing::warn!(
                    row = %self.row_name,
                    aliyun_request_id = %diag.request_id,
                    aliyun_code = %diag.code,
                    aliyun_message = %diag.message_field(),
                    "aliyun TextModerationPlus non-200 Code",
                );
                Err(AliyunFailure::ServerError)
            }
        };
        (outcome, diag)
    }

    fn handle_failure(
        &self,
        failure: AliyunFailure,
        diag: &AliyunDiagnostics,
        fail_open: bool,
    ) -> GuardrailVerdict {
        let tag = failure.bypass_tag();
        if !matches!(failure, AliyunFailure::ConfigError) {
            tracing::warn!(
                row = %self.row_name,
                aliyun_request_id = %diag.request_id,
                failure = ?failure,
                fail_open,
                "aliyun text moderation call failed",
            );
        }
        if fail_open {
            GuardrailVerdict::Bypass { reason: tag.into() }
        } else {
            GuardrailVerdict::block(format!("aliyun text moderation unavailable ({tag})"))
        }
    }
}

/// Rank a risk level so thresholds compare numerically. An unrecognized
/// level ranks as `none` (0) — fail toward allowing rather than blocking
/// on an unexpected label, and the call site logs nothing because Aliyun
/// only ever returns the four known levels.
fn risk_rank(level: &str) -> u8 {
    match level.to_ascii_lowercase().as_str() {
        "high" => 3,
        "medium" => 2,
        "low" => 1,
        _ => 0,
    }
}

/// RFC3986 percent-encoding with Aliyun's tweaks: unreserved chars
/// (`A-Za-z0-9-_.~`) pass through, every other byte becomes `%XX`
/// (uppercase). Space → `%20`. (Aliyun additionally maps `+`→`%20`,
/// `*`→`%2A`, `%7E`→`~`; we never emit `+` or a literal `*`, and `~` is
/// already unreserved, so encoding each non-unreserved byte covers it.)
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{b:02X}"));
            }
        }
    }
    out
}

/// Build the RPC v1 `StringToSign` from the (already key-sorted) params.
/// Factored out so the canonicalization is unit-testable independent of
/// the HMAC step.
fn string_to_sign(params: &std::collections::BTreeMap<&str, String>) -> String {
    let mut canonical = String::new();
    for (k, v) in params {
        if !canonical.is_empty() {
            canonical.push('&');
        }
        canonical.push_str(&percent_encode(k));
        canonical.push('=');
        canonical.push_str(&percent_encode(v));
    }
    format!(
        "POST&{}&{}",
        percent_encode("/"),
        percent_encode(&canonical)
    )
}

/// Compute the RPC v1 signature: `Base64(HMAC-SHA1(secret + "&", StringToSign))`.
fn sign(params: &std::collections::BTreeMap<&str, String>, access_key_secret: &str) -> String {
    let sts = string_to_sign(params);
    let key = format!("{access_key_secret}&");
    let mut mac =
        HmacSha1::new_from_slice(key.as_bytes()).expect("HMAC accepts keys of any length");
    mac.update(sts.as_bytes());
    base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
}

/// Failure cause buckets. `bypass_tag()` maps to the strings stored in
/// `usage_events.guardrail_bypassed_reason`.
#[derive(Debug)]
enum AliyunFailure {
    Timeout,
    Throttled,
    IoError,
    ServerError,
    /// HTTP 2xx whose body wasn't the documented JSON. Distinct from
    /// `ServerError` so triage isn't told "5xx" for a response that in
    /// fact arrived intact and merely didn't parse (AISIX-Cloud#1060) —
    /// the two want different fixes.
    MalformedResponse,
    ConfigError,
}

impl AliyunFailure {
    fn bypass_tag(&self) -> &'static str {
        match self {
            Self::Timeout => "aliyun_timeout",
            Self::Throttled => "aliyun_throttled",
            Self::IoError | Self::ServerError => "aliyun_5xx",
            Self::MalformedResponse => "aliyun_bad_response",
            Self::ConfigError => "aliyun_config_error",
        }
    }
}

/// The `x-acs-request-id` response header. Every Alibaba Cloud OpenAPI
/// response carries it — verified against the live `green-cip` endpoint on
/// 2xx, on a JSON business error, and on 400/404 responses whose body is
/// XML rather than JSON. It is therefore a strictly more reliable source
/// for the id than the body's `RequestId`, which is unreachable exactly
/// when the body doesn't parse.
const ACS_REQUEST_ID_HEADER: &str = "x-acs-request-id";

/// What one `TextModerationPlus` call reported about itself, for operator
/// triage (AISIX-Cloud#1060).
///
/// Provider metadata ONLY. The moderation result echoes the offending text
/// back in `Data.Result[].RiskWords` / `RiskPositions`; per #153 none of
/// that may reach a log, so this type deliberately has nowhere to put it.
/// `Label` is the matched CATEGORY (e.g. `violent_incidents`), not the
/// matched content, and is safe.
#[derive(Debug, Default, Clone, PartialEq)]
struct AliyunDiagnostics {
    /// Aliyun's own request id, for looking the call up in the Aliyun
    /// console. Named `aliyun_request_id` in logs — never `request_id`,
    /// which is the gateway's own id (`x-aisix-request-id`) and is
    /// supplied by the request-scoped tracing span.
    ///
    /// Empty only when no HTTP response arrived at all (timeout, connect
    /// failure), which the failure bucket names.
    request_id: String,
    /// Business `Code` from the response body: `200` on success, or e.g.
    /// `400` ("service is invalid"). Empty when the body didn't parse.
    /// Rendered as a string because the same field is an integer on an
    /// HTTP-200 body but a symbolic code (`InvalidAccessKeyId.NotFound`)
    /// on an HTTP-error body.
    code: String,
    /// Body `Message` — the provider's own explanation ("OK", "service is
    /// invalid"). Capped for logging like any provider-supplied string.
    message: String,
    /// `Data.RiskLevel` — `high` / `medium` / `low` / `none`.
    risk_level: String,
    /// `Data.Result[].Label` — the matched categories. Plural: one prompt
    /// commonly trips several (`inappropriate_oral` + `violent_incidents`).
    labels: Vec<String>,
}

impl AliyunDiagnostics {
    /// Seed from the response headers, before the body is consumed — so
    /// the id survives a body that never parses.
    fn from_headers(headers: &reqwest::header::HeaderMap) -> Self {
        Self {
            request_id: headers
                .get(ACS_REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_owned(),
            ..Self::default()
        }
    }

    /// Fold in everything the parsed body adds.
    fn absorb_body(&mut self, body: &AliyunResponse) {
        // Header and body carry the same id; the header already won.
        // Fall back for a server that somehow omits the header.
        if self.request_id.is_empty() {
            self.request_id = body.request_id.clone().unwrap_or_default();
        }
        self.code = body.code.to_string();
        self.message = body.message.clone().unwrap_or_default();
        if let Some(data) = body.data.as_ref() {
            self.risk_level = data.risk_level.clone().unwrap_or_default();
            self.labels = data.result.iter().filter_map(|r| r.label.clone()).collect();
        }
    }

    /// The `Label` list as one log-safe field.
    fn labels_field(&self) -> String {
        self.labels.join(",")
    }

    /// `Message`, capped like any other provider-supplied log string.
    fn message_field(&self) -> &str {
        crate::truncate_error_body_for_log(&self.message)
    }
}

// --- serde shapes for the wire protocol ------------------------------------

#[derive(Deserialize)]
struct AliyunResponse {
    #[serde(rename = "RequestId", default)]
    request_id: Option<String>,
    #[serde(rename = "Code", default)]
    code: i32,
    #[serde(rename = "Message", default)]
    message: Option<String>,
    #[serde(rename = "Data", default)]
    data: Option<AliyunData>,
}

#[derive(Deserialize)]
struct AliyunData {
    #[serde(rename = "RiskLevel", default)]
    risk_level: Option<String>,
    /// One entry per matched category. Only `Label` is read: the sibling
    /// `RiskWords` / `RiskPositions` fields echo the offending text back
    /// verbatim (a real `high` response carries
    /// `"RiskWords": "傻逼,弄死你,死你全家"`), and #153 keeps matched
    /// content out of logs. Deserializing only what we log means a future
    /// edit can't casually leak them.
    #[serde(rename = "Result", default)]
    result: Vec<AliyunResult>,
}

#[derive(Deserialize)]
struct AliyunResult {
    #[serde(rename = "Label", default)]
    label: Option<String>,
}

// --- Guardrail trait impl --------------------------------------------------

#[async_trait]
impl Guardrail for AliyunTextModerationGuardrail {
    fn name(&self) -> &'static str {
        "aliyun_text_moderation"
    }

    /// Its streamed-output hold-back policy applies only when it inspects
    /// output (#466); an input-only attachment must not buffer the response.
    fn runs_on_output(&self) -> bool {
        matches!(
            self.hook_point,
            GuardrailHookPoint::Output | GuardrailHookPoint::Both
        )
    }

    fn stream_output_policy(&self) -> StreamOutputPolicy {
        match self.stream_processing_mode.as_str() {
            "buffer_full" => StreamOutputPolicy::BufferFull {
                max_buffer_bytes: self.max_buffer_bytes as usize,
                on_exceeded_fail_open: self.on_buffer_exceeded == "fail_open",
            },
            // "window" (default) and any unexpected value → sliding window.
            _ => StreamOutputPolicy::Window {
                size_chars: self.window_size as usize,
                overlap_chars: self.window_overlap_size as usize,
            },
        }
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        if !matches!(
            self.hook_point,
            GuardrailHookPoint::Input | GuardrailHookPoint::Both
        ) {
            return GuardrailVerdict::Allow;
        }
        let text = collect_input_text(req);
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        self.moderate(SERVICE_INPUT, &text, None, self.fail_open)
            .await
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        if !matches!(
            self.hook_point,
            GuardrailHookPoint::Output | GuardrailHookPoint::Both
        ) {
            return GuardrailVerdict::Allow;
        }
        let text = resp.guardrail_output_text();
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        // The upstream provider's request id is stable across all windows
        // of one streamed response, so it doubles as the per-stream Aliyun
        // sessionId; a fresh uuid keeps non-streaming calls correlated to
        // themselves when the provider omits an id.
        let session = if resp.id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            resp.id.clone()
        };
        // Output uses its own fail policy (default fail-closed) so an
        // Aliyun outage can't release unscanned model output.
        self.moderate(SERVICE_OUTPUT, &text, Some(&session), self.output_fail_open)
            .await
    }
}

/// Concatenate the request's user-visible message contents into one blob.
/// (Same collector shape as the Bedrock dispatcher — keeps the families
/// scanning identical text.)
fn collect_input_text(req: &ChatFormat) -> String {
    req.messages
        .iter()
        .map(crate::message_scan_text)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use aisix_gateway::{ChatFormat, ChatMessage, ChatResponse, FinishReason, UsageStats};
    use serde_json::json;
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn cfg(endpoint: &str, threshold: &str) -> AliyunTextModerationConfig {
        serde_json::from_value(json!({
            "region": "cn-shanghai",
            "endpoint": endpoint,
            "access_key_id": "LTAI_TEST",
            "access_key_secret": "test-secret",
            "risk_level_threshold": threshold,
            "timeout_ms": 5_000,
        }))
        .unwrap()
    }

    fn build(endpoint: &str, threshold: &str, fail_open: bool) -> AliyunTextModerationGuardrail {
        AliyunTextModerationGuardrail::new(
            "wiremock-test",
            &cfg(endpoint, threshold),
            GuardrailHookPoint::Both,
            fail_open,
        )
    }

    fn req(msg: &str) -> ChatFormat {
        ChatFormat::new("m", vec![ChatMessage::user(msg)])
    }

    fn resp(content: &str) -> ChatResponse {
        ChatResponse {
            id: "stream-req-1".into(),
            model: "m".into(),
            message: ChatMessage::assistant(content),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(0, 0),
        }
    }

    // --- pure signature / encoding tests ---

    #[test]
    fn risk_rank_orders_levels() {
        assert!(risk_rank("none") < risk_rank("low"));
        assert!(risk_rank("low") < risk_rank("medium"));
        assert!(risk_rank("medium") < risk_rank("high"));
        assert_eq!(risk_rank("HIGH"), 3, "case-insensitive");
        assert_eq!(risk_rank("garbage"), 0, "unknown ranks as none");
    }

    #[test]
    fn percent_encode_matches_aliyun_rules() {
        assert_eq!(percent_encode("a b"), "a%20b");
        assert_eq!(percent_encode("/"), "%2F");
        assert_eq!(percent_encode("~-_."), "~-_.");
        assert_eq!(percent_encode("{\"k\":\"v\"}"), "%7B%22k%22%3A%22v%22%7D");
    }

    #[test]
    fn string_to_sign_is_canonical_and_stable() {
        let mut p: std::collections::BTreeMap<&str, String> = std::collections::BTreeMap::new();
        p.insert("Action", "TextModerationPlus".into());
        p.insert("Service", "llm_query_moderation".into());
        let sts = string_to_sign(&p);
        // "POST&%2F&" + percentEncode("Action=TextModerationPlus&Service=llm_query_moderation")
        assert_eq!(
            sts,
            "POST&%2F&Action%3DTextModerationPlus%26Service%3Dllm_query_moderation"
        );
    }

    #[test]
    fn sign_is_deterministic_and_known_vector() {
        // Pins the full v1 signature against an openssl-computed reference,
        // so a regression in canonicalization or the HMAC step fails loud.
        let mut p: std::collections::BTreeMap<&str, String> = std::collections::BTreeMap::new();
        p.insert("Action", "TextModerationPlus".into());
        p.insert("Service", "llm_query_moderation".into());
        let sig = sign(&p, "test-secret");
        assert_eq!(sig, KNOWN_SIGNATURE);
        // deterministic
        assert_eq!(sign(&p, "test-secret"), sig);
    }

    // openssl dgst -sha1 -hmac "test-secret&" over the StringToSign above,
    // base64-encoded. Recompute with:
    //   printf '%s' 'POST&%2F&Action%3DTextModerationPlus%26Service%3Dllm_query_moderation' \
    //     | openssl dgst -sha1 -hmac 'test-secret&' -binary | base64
    const KNOWN_SIGNATURE: &str = "pu3Hn+zsRIztpT2f7JT5+zHPPVo=";

    // --- wiremock integration ---

    fn risk_body(level: &str) -> serde_json::Value {
        json!({ "Code": 200, "Data": { "RiskLevel": level, "Result": [{ "Label": "x" }] }, "RequestId": "r" })
    }

    #[tokio::test]
    async fn clean_input_returns_allow_and_signs_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            // proves the signed form body carries Action + Service + Signature
            .and(body_string_contains("Action=TextModerationPlus"))
            .and(body_string_contains("Service=llm_query_moderation"))
            .and(body_string_contains("Signature="))
            .respond_with(ResponseTemplate::new(200).set_body_json(risk_body("none")))
            .expect(1)
            .mount(&server)
            .await;

        let g = build(&server.uri(), "high", true);
        assert_eq!(g.check_input(&req("hello")).await, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn high_risk_input_blocks_at_high_threshold() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(risk_body("high")))
            .mount(&server)
            .await;
        let g = build(&server.uri(), "high", true);
        assert!(g.check_input(&req("bad")).await.is_block());
    }

    #[tokio::test]
    async fn medium_risk_allowed_at_high_threshold_blocked_at_medium() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(risk_body("medium")))
            .mount(&server)
            .await;
        // threshold=high → medium passes
        let g_high = build(&server.uri(), "high", true);
        assert_eq!(g_high.check_input(&req("x")).await, GuardrailVerdict::Allow);
        // threshold=medium → medium blocks
        let g_med = build(&server.uri(), "medium", true);
        assert!(g_med.check_input(&req("x")).await.is_block());
    }

    #[tokio::test]
    async fn output_sends_session_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("Service=llm_response_moderation"))
            // sessionId is JSON-encoded inside ServiceParameters, percent-encoded
            // in the body: {"content":"...","sessionId":"stream-req-1"}
            .and(body_string_contains("sessionId"))
            .respond_with(ResponseTemplate::new(200).set_body_json(risk_body("none")))
            .expect(1)
            .mount(&server)
            .await;
        let g = build(&server.uri(), "high", true);
        assert_eq!(g.check_output(&resp("ok")).await, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn http_5xx_fail_open_true_returns_bypass() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let g = build(&server.uri(), "high", true);
        match g.check_input(&req("x")).await {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "aliyun_5xx"),
            other => panic!("expected Bypass(aliyun_5xx), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn output_5xx_fails_closed_by_default() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        // output_fail_open defaults false → an output-side 5xx must Block.
        let g = build(&server.uri(), "high", true);
        assert!(
            g.check_output(&resp("model output")).await.is_block(),
            "output hook must fail closed on Aliyun error by default"
        );
    }

    #[tokio::test]
    async fn app_level_403_code_is_config_error_block_when_fail_closed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "Code": 403, "Message": "no permission" })),
            )
            .mount(&server)
            .await;
        // input fail_open=false → config error blocks
        let g = build(&server.uri(), "high", false);
        assert!(g.check_input(&req("x")).await.is_block());
    }

    /// A tracing writer that appends every emitted byte into a shared buffer so
    /// a test can assert what a log line carried.
    #[derive(Clone)]
    struct BufWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for BufWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl tracing_subscriber::fmt::MakeWriter<'_> for BufWriter {
        type Writer = BufWriter;
        fn make_writer(&self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `f` with a log-capturing subscriber installed and return everything
    /// it emitted.
    async fn capture_logs<F, Fut>(f: F) -> String
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            // The clean-pass diagnostics land at DEBUG (one line per
            // request is too noisy for the default level), so widen the
            // capture past the builder's INFO default to see them.
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(BufWriter(buf.clone()))
            .finish();
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            f().await;
        }
        let bytes = buf.lock().unwrap().clone();
        String::from_utf8(bytes).unwrap()
    }

    /// Body of a real `high` verdict, trimmed to the fields we deserialize
    /// plus the ones we must NOT log. Shapes and values are copied from a
    /// live `TextModerationPlus` response: two `Result` entries (one prompt
    /// commonly trips several categories) and the `RiskWords` /
    /// `RiskPositions` siblings that echo the offending text back.
    fn risky_body_with_matched_content() -> serde_json::Value {
        json!({
            "Code": 200,
            "Message": "OK",
            "RequestId": "019F6ED5-91BE-5AB0-8411-96308CEC81F1",
            "Data": {
                "RiskLevel": "high",
                "Result": [
                    {
                        "Label": "inappropriate_oral",
                        "Confidence": 100.0,
                        "Description": "疑似低俗口头语内容",
                        "RiskWords": "傻逼,弄死你,死你全家",
                        "RiskPositions": [{"StartPos": 3, "EndPos": 5, "RiskWord": "傻逼"}]
                    },
                    {
                        "Label": "violent_incidents",
                        "Confidence": 100.0,
                        "Description": "疑似极端主义内容",
                        "RiskWords": "弄死"
                    }
                ]
            }
        })
    }

    // Every log-capturing scenario lives in ONE test on purpose: the capture
    // uses a thread-local default subscriber (`set_default`), so two capture
    // tests running in parallel would race over it. Kept sequential here, at
    // most one capturing subscriber is ever installed process-wide.
    #[tokio::test]
    async fn error_paths_log_provider_diagnostics() {
        // 1. A wrong access key: green-cip answers 4xx (the customer saw 404)
        //    with a body naming the cause. Without echoing it the operator only
        //    sees a bare status code (AISIX-Cloud#1030 follow-up).
        {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(404).set_body_json(json!({
                    "Code": "InvalidAccessKeyId.NotFound",
                    "Message": "Specified access key is not found.",
                    "RequestId": "ABC-123",
                })))
                .mount(&server)
                .await;
            let uri = server.uri();
            let logged = capture_logs(|| async {
                let g = build(&uri, "high", false);
                assert!(g.check_input(&req("x")).await.is_block());
            })
            .await;
            assert!(
                logged.contains("response_body") && logged.contains("InvalidAccessKeyId.NotFound"),
                "4xx log must echo the provider's actual reason; got: {logged}"
            );
        }

        // 2. A pathological oversized 4xx body: the marker sits well past the
        //    log cap, so if it shows up we buffered/logged more than we should.
        {
            let server = MockServer::start().await;
            let filler = "X".repeat(crate::MAX_ERROR_BODY_LOG_BYTES + 4_000);
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(400).set_body_string(format!("{filler}__TAIL_MARKER__")),
                )
                .mount(&server)
                .await;
            let uri = server.uri();
            let logged = capture_logs(|| async {
                let g = build(&uri, "high", false);
                assert!(g.check_input(&req("x")).await.is_block());
            })
            .await;
            assert!(
                logged.contains("XXXX"),
                "the head of the body must be logged"
            );
            assert!(
                !logged.contains("__TAIL_MARKER__"),
                "the logged body must be capped, not echoed in full",
            );
        }

        // 3. HTTP 200 but an app-level error Code — the Message carries the
        //    reason (e.g. an access key lacking llm_response_moderation).
        {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "Code": 403,
                    "Message": "AccessKey has no permission for llm_response_moderation",
                })))
                .mount(&server)
                .await;
            let uri = server.uri();
            let logged = capture_logs(|| async {
                let g = build(&uri, "high", false);
                assert!(g.check_input(&req("x")).await.is_block());
            })
            .await;
            assert!(
                logged.contains("aliyun_message")
                    && logged.contains("no permission for llm_response_moderation"),
                "app-level error log must echo the Aliyun Message; got: {logged}"
            );
        }
    }

    // --- #1060: upstream diagnostics on every path -------------------------
    //
    // Same one-capture-at-a-time constraint as the test above, so these also
    // share a single test body. Each scenario asserts what an operator needs
    // to trace a caller's 422 back to a record in the Aliyun console.
    #[tokio::test]
    async fn upstream_diagnostics_are_logged_on_every_path() {
        // 1. A block: the case an operator actually traces. The response is
        //    a real `high` body, so this doubles as the no-leak assertion —
        //    RiskWords/RiskPositions must not reach the log even though the
        //    provider sent them.
        {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(risky_body_with_matched_content())
                        .insert_header("x-acs-request-id", "019F6ED5-91BE-5AB0-8411-96308CEC81F1"),
                )
                .mount(&server)
                .await;
            let uri = server.uri();
            let logged = capture_logs(|| async {
                let g = build(&uri, "high", true);
                assert!(g.check_input(&req("危险内容")).await.is_block());
            })
            .await;

            assert!(
                logged.contains("aliyun_request_id=019F6ED5-91BE-5AB0-8411-96308CEC81F1"),
                "a block must log the Aliyun request id; got: {logged}"
            );
            assert!(
                logged.contains("aliyun_risk_level=high"),
                "a block must log the returned RiskLevel; got: {logged}"
            );
            assert!(
                logged.contains("aliyun_labels=inappropriate_oral,violent_incidents"),
                "a block must log every matched Label; got: {logged}"
            );
            assert!(
                logged.contains("aliyun_code=200"),
                "a block must log the business Code; got: {logged}"
            );
            // #153: the categories are metadata, the matched words are the
            // caller's content. Only the former may be logged.
            for leak in ["傻逼", "弄死你", "死你全家", "RiskWords", "RiskPositions"] {
                assert!(
                    !logged.contains(leak),
                    "matched content {leak:?} must never reach a log; got: {logged}"
                );
            }
        }

        // 2. A clean pass still reports its id, at debug — the id has to be
        //    retrievable for a request nobody blocked, without spending a
        //    default-level line on every request.
        {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(risk_body("none"))
                        .insert_header("x-acs-request-id", "CLEAN-REQ-1"),
                )
                .mount(&server)
                .await;
            let uri = server.uri();
            let logged = capture_logs(|| async {
                let g = build(&uri, "high", true);
                assert_eq!(g.check_input(&req("hello")).await, GuardrailVerdict::Allow);
            })
            .await;
            assert!(
                logged.contains("aliyun_request_id=CLEAN-REQ-1")
                    && logged.contains("aliyun_risk_level=none"),
                "a clean pass must still log its Aliyun request id; got: {logged}"
            );
        }

        // 3. Timeout: nothing came back, so there is no id to report. It must
        //    log as EMPTY rather than be omitted — an absent field reads as
        //    "we forgot to log it", an empty one as "Aliyun never answered" —
        //    and the failure bucket must survive.
        {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(risk_body("none"))
                        .set_delay(Duration::from_millis(300)),
                )
                .mount(&server)
                .await;
            let uri = server.uri();
            let logged = capture_logs(|| async {
                let mut g = build(&uri, "high", true);
                g.timeout = Duration::from_millis(10);
                assert!(g.check_input(&req("x")).await.is_bypass());
            })
            .await;
            assert!(
                logged.contains("aliyun_request_id=") && logged.contains("failure=Timeout"),
                "a timeout must log an empty id and keep the failure type; got: {logged}"
            );
        }

        // 4. HTTP 200 with a body that isn't the documented JSON. The header
        //    still carries the id — which is the whole reason it is read from
        //    there rather than from the body — and the failure type must say
        //    "malformed", not "5xx".
        {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string("<html>not json at all</html>")
                        .insert_header("x-acs-request-id", "MALFORMED-REQ-1"),
                )
                .mount(&server)
                .await;
            let uri = server.uri();
            let logged = capture_logs(|| async {
                let g = build(&uri, "high", true);
                match g.check_input(&req("x")).await {
                    GuardrailVerdict::Bypass { reason } => {
                        assert_eq!(reason, "aliyun_bad_response")
                    }
                    other => panic!("expected Bypass(aliyun_bad_response), got {other:?}"),
                }
            })
            .await;
            assert!(
                logged.contains("aliyun_request_id=MALFORMED-REQ-1"),
                "a non-JSON body must still yield the header's id; got: {logged}"
            );
            assert!(
                logged.contains("failure=MalformedResponse"),
                "a non-JSON body must not be reported as a 5xx; got: {logged}"
            );
        }

        // 5. HTTP 5xx: no usable body, but the header id survives and points
        //    at the failing call.
        {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(500).insert_header("x-acs-request-id", "SERVER-ERR-1"),
                )
                .mount(&server)
                .await;
            let uri = server.uri();
            let logged = capture_logs(|| async {
                let g = build(&uri, "high", true);
                assert!(g.check_input(&req("x")).await.is_bypass());
            })
            .await;
            assert!(
                logged.contains("aliyun_request_id=SERVER-ERR-1")
                    && logged.contains("failure=ServerError"),
                "a 5xx must log the header's id and its failure type; got: {logged}"
            );
        }

        // 6. HTTP 4xx: Aliyun types `Code` as a symbolic string here, so the
        //    body is logged raw. The id must come from the header regardless.
        {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(404)
                        .set_body_json(json!({
                            "Code": "InvalidAccessKeyId.NotFound",
                            "Message": "Specified access key is not found.",
                            "RequestId": "CONFIG-ERR-1",
                        }))
                        .insert_header("x-acs-request-id", "CONFIG-ERR-1"),
                )
                .mount(&server)
                .await;
            let uri = server.uri();
            let logged = capture_logs(|| async {
                let g = build(&uri, "high", false);
                assert!(g.check_input(&req("x")).await.is_block());
            })
            .await;
            assert!(
                logged.contains("aliyun_request_id=CONFIG-ERR-1"),
                "a 4xx must log the header's id; got: {logged}"
            );
        }

        // 7. A business error on HTTP 200 (`Code: 400`, "service is invalid")
        //    — the shape a live call returns for a bad Service param. The id
        //    is present in the body here even without the header, so a server
        //    that omits the header still yields one.
        {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "Code": 400,
                    "Message": "service is invalid",
                    "RequestId": "BODY-ONLY-REQ-1",
                })))
                .mount(&server)
                .await;
            let uri = server.uri();
            let logged = capture_logs(|| async {
                let g = build(&uri, "high", false);
                assert!(g.check_input(&req("x")).await.is_block());
            })
            .await;
            assert!(
                logged.contains("aliyun_request_id=BODY-ONLY-REQ-1"),
                "the body's RequestId must be used when no header is sent; got: {logged}"
            );
            assert!(
                logged.contains("aliyun_code=400") && logged.contains("service is invalid"),
                "a business error must log its Code and Message; got: {logged}"
            );
        }
    }

    #[test]
    fn stream_policy_reflects_config() {
        let g = build("http://unused", "high", true);
        assert_eq!(
            g.stream_output_policy(),
            StreamOutputPolicy::Window {
                size_chars: 2_000,
                overlap_chars: 128
            }
        );
        let mut g2 = build("http://unused", "high", true);
        g2.stream_processing_mode = "buffer_full".to_owned();
        g2.max_buffer_bytes = 1000;
        g2.on_buffer_exceeded = "fail_open".to_owned();
        assert_eq!(
            g2.stream_output_policy(),
            StreamOutputPolicy::BufferFull {
                max_buffer_bytes: 1000,
                on_exceeded_fail_open: true
            }
        );
    }

    // --- live smoke test against the real green-cip endpoint ---
    //
    // Ignored by default (requires real Aliyun credentials + network).
    // Run manually with:
    //
    //   ALIYUN_AK_ID=... ALIYUN_AK_SECRET=... ALIYUN_REGION=cn-shanghai \
    //     cargo test -p aisix-guardrails aliyun::tests::live_smoke \
    //     --features aliyun-text-moderation -- --ignored --nocapture
    //
    // Exercises the real signer + HTTP + response parse against
    // TextModerationPlus and prints the returned RiskLevel for a benign
    // and a policy-violating prompt. No credentials are hard-coded.
    #[tokio::test]
    #[ignore = "requires real Aliyun credentials + network"]
    async fn live_smoke_real_endpoint() {
        let ak_id = std::env::var("ALIYUN_AK_ID").unwrap_or_default();
        let ak_secret = std::env::var("ALIYUN_AK_SECRET").unwrap_or_default();
        if ak_id.is_empty() || ak_secret.is_empty() {
            eprintln!("live_smoke: ALIYUN_AK_ID/ALIYUN_AK_SECRET unset — skipping");
            return;
        }
        let region = std::env::var("ALIYUN_REGION").unwrap_or_else(|_| "cn-shanghai".to_owned());
        let cfg: AliyunTextModerationConfig = serde_json::from_value(json!({
            "region": region,
            "endpoint": std::env::var("ALIYUN_ENDPOINT").ok(),
            "access_key_id": ak_id,
            "access_key_secret": ak_secret,
            "risk_level_threshold": "low",
            "timeout_ms": 8000,
        }))
        .unwrap();
        let g =
            AliyunTextModerationGuardrail::new("live-smoke", &cfg, GuardrailHookPoint::Both, false);

        // Benign prompt — expect RiskLevel "none".
        let (benign, benign_diag) = g.call(SERVICE_INPUT, "今天北京的天气怎么样？", None).await;
        let benign = benign.expect("benign call should succeed");
        eprintln!("live_smoke benign  -> RiskLevel={benign} diag={benign_diag:?}");

        // Policy-violating prompt (abuse + threat) — expect non-"none".
        let (risky, risky_diag) = g
            .call(SERVICE_INPUT, "你这个傻逼，我现在就要弄死你全家", None)
            .await;
        let risky = risky.expect("risky call should succeed");
        eprintln!("live_smoke risky   -> RiskLevel={risky} diag={risky_diag:?}");

        // Output service with a sessionId, same risky text.
        let (risky_out, out_diag) = g
            .call(
                SERVICE_OUTPUT,
                "你这个傻逼，我现在就要弄死你全家",
                Some("live-sess-1"),
            )
            .await;
        let risky_out = risky_out.expect("risky output call should succeed");
        eprintln!("live_smoke output  -> RiskLevel={risky_out} diag={out_diag:?}");

        assert_eq!(benign, "none", "benign prompt must score none");
        assert_ne!(risky, "none", "policy-violating prompt must score a risk");

        // #1060: the real endpoint must yield the diagnostics we log —
        // an id on every call, and the matched categories on the risky one.
        for (what, diag) in [
            ("benign", &benign_diag),
            ("risky", &risky_diag),
            ("output", &out_diag),
        ] {
            assert!(
                !diag.request_id.is_empty(),
                "{what}: live call must carry an aliyun request id"
            );
            assert_eq!(diag.code, "200", "{what}: business code");
        }
        assert!(
            !risky_diag.labels.is_empty(),
            "a risky prompt must report at least one Label, got {risky_diag:?}"
        );
    }
}
