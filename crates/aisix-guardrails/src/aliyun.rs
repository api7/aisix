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
        match self.call(service, &content, session_id).await {
            Ok(level) => {
                if risk_rank(&level) >= self.threshold_rank {
                    GuardrailVerdict::block(format!(
                        "aliyun text moderation: risk level {} >= threshold (row: {})",
                        level, self.row_name
                    ))
                } else {
                    GuardrailVerdict::Allow
                }
            }
            Err(failure) => self.handle_failure(failure, fail_open),
        }
    }

    /// Sign + POST one `TextModerationPlus` call; return the response
    /// `RiskLevel` (lowercased, `"none"` when absent).
    async fn call(
        &self,
        service: &str,
        content: &str,
        session_id: Option<&str>,
    ) -> Result<String, AliyunFailure> {
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

        let resp = match tokio::time::timeout(self.timeout, future).await {
            Err(_elapsed) => return Err(AliyunFailure::Timeout),
            Ok(Err(_e)) => return Err(AliyunFailure::IoError),
            Ok(Ok(r)) => r,
        };

        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(AliyunFailure::Throttled);
        }
        if status.is_server_error() {
            return Err(AliyunFailure::ServerError);
        }
        if !status.is_success() {
            tracing::error!(
                row = %self.row_name,
                http_status = status.as_u16(),
                "aliyun TextModerationPlus returned 4xx — check region/access keys configuration",
            );
            return Err(AliyunFailure::ConfigError);
        }

        let body: AliyunResponse = resp.json().await.map_err(|_| AliyunFailure::ServerError)?;

        // Aliyun signals app-level errors via the JSON `Code` (200 = OK)
        // even on HTTP 200.
        match body.code {
            200 => Ok(body
                .data
                .and_then(|d| d.risk_level)
                .unwrap_or_else(|| "none".to_owned())
                .to_lowercase()),
            408 | 401 | 403 | 400 => {
                tracing::error!(
                    row = %self.row_name,
                    aliyun_code = body.code,
                    "aliyun TextModerationPlus auth/permission error — check access keys",
                );
                Err(AliyunFailure::ConfigError)
            }
            other => {
                tracing::warn!(
                    row = %self.row_name,
                    aliyun_code = other,
                    "aliyun TextModerationPlus non-200 Code",
                );
                Err(AliyunFailure::ServerError)
            }
        }
    }

    fn handle_failure(&self, failure: AliyunFailure, fail_open: bool) -> GuardrailVerdict {
        let tag = failure.bypass_tag();
        if !matches!(failure, AliyunFailure::ConfigError) {
            tracing::warn!(
                row = %self.row_name,
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
    ConfigError,
}

impl AliyunFailure {
    fn bypass_tag(&self) -> &'static str {
        match self {
            Self::Timeout => "aliyun_timeout",
            Self::Throttled => "aliyun_throttled",
            Self::IoError | Self::ServerError => "aliyun_5xx",
            Self::ConfigError => "aliyun_config_error",
        }
    }
}

// --- serde shapes for the wire protocol ------------------------------------

#[derive(Deserialize)]
struct AliyunResponse {
    #[serde(rename = "Code", default)]
    code: i32,
    #[serde(rename = "Data", default)]
    data: Option<AliyunData>,
}

#[derive(Deserialize)]
struct AliyunData {
    #[serde(rename = "RiskLevel", default)]
    risk_level: Option<String>,
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
        let benign = g
            .call(SERVICE_INPUT, "今天北京的天气怎么样？", None)
            .await
            .expect("benign call should succeed");
        eprintln!("live_smoke benign  -> RiskLevel={benign}");

        // Policy-violating prompt (abuse + threat) — expect non-"none".
        let risky = g
            .call(SERVICE_INPUT, "你这个傻逼，我现在就要弄死你全家", None)
            .await
            .expect("risky call should succeed");
        eprintln!("live_smoke risky   -> RiskLevel={risky}");

        // Output service with a sessionId, same risky text.
        let risky_out = g
            .call(
                SERVICE_OUTPUT,
                "你这个傻逼，我现在就要弄死你全家",
                Some("live-sess-1"),
            )
            .await
            .expect("risky output call should succeed");
        eprintln!("live_smoke output  -> RiskLevel={risky_out}");

        assert_eq!(benign, "none", "benign prompt must score none");
        assert_ne!(risky, "none", "policy-violating prompt must score a risk");
    }
}
