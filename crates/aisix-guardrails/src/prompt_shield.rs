//! kind=azure_content_safety guardrail dispatcher — calls Azure AI Content
//! Safety Prompt Shield on every chat request and translates the response
//! into a [`GuardrailVerdict`].
//!
//! PRD-09c §6 P1.
//!
//! API reference (2024-09-01):
//! POST `{endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01`
//! Source: <https://learn.microsoft.com/en-us/azure/ai-services/content-safety/reference>
//!
//! Wire shape:
//! ```json
//! // Request
//! { "userPrompt": "...", "documents": [] }
//! // Response
//! { "userPromptAnalysis": { "attackDetected": bool }, "documentsAnalysis": [] }
//! ```
//!
//! The API caps `userPrompt` at 10 000 characters. Longer inputs are
//! auto-split on whitespace boundaries; the first chunk that returns
//! `attackDetected=true` short-circuits and produces a `Block` verdict.
//!
//! The cp-api decrypts the envelope-encrypted `api_key` at kine-projection
//! time so this module only handles plaintext keys. The key is never logged.
//!
//! Behavior matrix (failure modes):
//!
//! | API response                    | `fail_open` | Verdict                               |
//! |---------------------------------|-------------|---------------------------------------|
//! | `attackDetected=false` (all)    | n/a         | Allow                                 |
//! | `attackDetected=true` (any)     | n/a         | Block { reason }                      |
//! | timeout                         | true        | Bypass { "azure_cs_timeout" }         |
//! | timeout                         | false       | Block { "azure content safety …" }    |
//! | 429 Throttling                  | true        | Bypass { "azure_cs_throttled" }       |
//! | 429 Throttling                  | false       | Block { "azure content safety …" }    |
//! | 5xx / IO error                  | true        | Bypass { "azure_cs_5xx" }             |
//! | 5xx / IO error                  | false       | Block { "azure content safety …" }    |
//! | 4xx (non-429, e.g. 401/400/404) | true        | Bypass { "azure_cs_config_error" }    |
//! | 4xx (non-429, e.g. 401/400/404) | false       | Block { "azure content safety …" }    |

use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{AzureContentSafetyConfig, GuardrailHookPoint};
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Guardrail, GuardrailVerdict};

/// Maximum characters per Prompt Shield API call. Azure CS enforces a
/// 10 000-char limit on `userPrompt`.
/// Source: <https://learn.microsoft.com/en-us/azure/ai-services/content-safety/reference>
const MAX_PROMPT_CHARS: usize = 10_000;

/// Path + query appended to the configured `endpoint`.
const SHIELD_PATH: &str = "/contentsafety/text:shieldPrompt?api-version=2024-09-01";

/// One Azure Content Safety Prompt Shield row, materialised into a
/// request-time dispatcher. Built once per snapshot from
/// [`AzureContentSafetyConfig`] + the outer `Guardrail` fields.
pub struct PromptShieldGuardrail {
    /// Operator-facing row name. Kept for log labels; the trait's static
    /// `name()` returns "azure_content_safety" so metric cardinality stays bounded.
    row_name: String,
    /// Endpoint with trailing slash stripped, e.g.
    /// `https://my-resource.cognitiveservices.azure.com`.
    endpoint: String,
    /// Plaintext subscription key (decrypted by cp-api before kine write).
    api_key: String,
    pub(crate) hook_point: GuardrailHookPoint,
    fail_open: bool,
    /// Call timeout. 0 ms in config → `Duration::ZERO` here, which
    /// `tokio::time::timeout` treats as "already elapsed"; callers that
    /// want no timeout should pass a very large value instead.
    pub(crate) timeout: Duration,
    /// Shared HTTP client. Pre-configured at construction time; kept in
    /// `Arc` so snapshot swaps don't drop a client mid-request.
    client: Arc<reqwest::Client>,
}

impl PromptShieldGuardrail {
    /// Build the dispatcher from a parsed [`AzureContentSafetyConfig`].
    /// Caller owns `row_name`, `hook_point`, and `fail_open` (they live
    /// on the outer `Guardrail` struct, not on the kind config).
    pub fn new(
        row_name: impl Into<String>,
        cfg: &AzureContentSafetyConfig,
        hook_point: GuardrailHookPoint,
        fail_open: bool,
    ) -> Self {
        let client = reqwest::Client::builder()
            // Per-call timeout is enforced via tokio::time::timeout in
            // call_api(); the connection pool uses reqwest's default idle
            // timeout (90 s). No pool customisation is needed here.
            .build()
            .expect("reqwest::Client::builder() failed; this should never happen");
        Self {
            row_name: row_name.into(),
            // Strip trailing slash so we can always append SHIELD_PATH with
            // a leading slash without getting a double slash.
            endpoint: cfg.endpoint.trim_end_matches('/').to_owned(),
            api_key: cfg.api_key.clone(),
            hook_point,
            fail_open,
            timeout: Duration::from_millis(cfg.timeout_ms as u64),
            client: Arc::new(client),
        }
    }

    /// Check `text` against Prompt Shield, splitting it into ≤10 000-char
    /// chunks. Returns `Block` on the first chunk where
    /// `attackDetected=true`; returns `Allow` when all chunks pass.
    async fn shield(&self, text: &str) -> GuardrailVerdict {
        for chunk in chunk_text(text, MAX_PROMPT_CHARS) {
            match self.call_api(&chunk).await {
                Ok(true) => {
                    return GuardrailVerdict::Block {
                        reason: format!(
                            "azure content safety prompt shield detected attack (row: {})",
                            self.row_name
                        ),
                    };
                }
                Ok(false) => {} // clean — continue to next chunk
                Err(failure) => return self.handle_failure(failure),
            }
        }
        GuardrailVerdict::Allow
    }

    /// POST to the Prompt Shield endpoint. Returns `Ok(true)` if any
    /// attack was detected, `Ok(false)` if the request is clean, `Err`
    /// if the call failed.
    async fn call_api(&self, user_prompt: &str) -> Result<bool, AcsFailure> {
        let url = format!("{}{}", self.endpoint, SHIELD_PATH);
        let body = ShieldRequest {
            user_prompt,
            documents: &[],
        };

        let future = self
            .client
            .post(&url)
            // Auth header: <https://learn.microsoft.com/en-us/azure/ai-services/content-safety/reference>
            .header("Ocp-Apim-Subscription-Key", &self.api_key)
            .json(&body)
            .send();

        let resp = match tokio::time::timeout(self.timeout, future).await {
            Err(_elapsed) => return Err(AcsFailure::Timeout),
            Ok(Err(_e)) => return Err(AcsFailure::IoError),
            Ok(Ok(r)) => r,
        };

        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(AcsFailure::Throttled);
        }
        if status.is_server_error() {
            return Err(AcsFailure::ServerError);
        }
        if !status.is_success() {
            // 4xx other than 429 — almost always a misconfiguration.
            // Log at error level (not warn): with fail_open=true this
            // silently bypasses the guardrail on every request until
            // the operator notices. A persistent error-level log is
            // the only signal they get.
            tracing::error!(
                row = %self.row_name,
                http_status = status.as_u16(),
                "azure content safety returned 4xx — check endpoint and api_key configuration",
            );
            return Err(AcsFailure::ConfigError);
        }

        let parsed: ShieldResponse = resp.json().await.map_err(|_| AcsFailure::ServerError)?;
        let attacked = parsed.user_prompt_analysis.attack_detected
            || parsed.documents_analysis.iter().any(|d| d.attack_detected);
        Ok(attacked)
    }

    fn handle_failure(&self, failure: AcsFailure) -> GuardrailVerdict {
        let tag = failure.bypass_tag();
        // ConfigError is already logged at error level in call_api(); skip
        // the generic warn here so operators see exactly one log line per
        // event and alert rules don't fire twice for the same failure.
        if !matches!(failure, AcsFailure::ConfigError) {
            tracing::warn!(
                row = %self.row_name,
                failure = ?failure,
                fail_open = self.fail_open,
                "azure content safety call failed",
            );
        }
        if self.fail_open {
            GuardrailVerdict::Bypass { reason: tag.into() }
        } else {
            GuardrailVerdict::Block {
                reason: format!("azure content safety unavailable ({tag})"),
            }
        }
    }
}

/// Failure cause buckets. `bypass_tag()` maps to the strings stored in
/// `usage_events.guardrail_bypassed_reason` — changing them is a breaking
/// change for operators who filter on these values.
#[derive(Debug)]
enum AcsFailure {
    Timeout,
    Throttled,
    IoError,
    /// HTTP 5xx or unparseable response body — Azure CS infrastructure error.
    ServerError,
    /// HTTP 4xx (other than 429) — almost always a misconfiguration
    /// (wrong `api_key`, wrong `endpoint`, etc.). Tagged separately from
    /// `ServerError` so operators can distinguish transient infrastructure
    /// problems from persistent config bugs without reading raw access logs.
    ConfigError,
}

impl AcsFailure {
    fn bypass_tag(&self) -> &'static str {
        match self {
            Self::Timeout => "azure_cs_timeout",
            Self::Throttled => "azure_cs_throttled",
            Self::IoError | Self::ServerError => "azure_cs_5xx",
            Self::ConfigError => "azure_cs_config_error",
        }
    }
}

// --- serde shapes for the wire protocol ------------------------------------

#[derive(Serialize)]
struct ShieldRequest<'a> {
    #[serde(rename = "userPrompt")]
    user_prompt: &'a str,
    // Always empty for the gateway use case (we check user messages, not
    // retrieved documents). The field is required by the API.
    documents: &'a [&'a str],
}

#[derive(Deserialize)]
struct ShieldResponse {
    #[serde(rename = "userPromptAnalysis")]
    user_prompt_analysis: AttackAnalysis,
    #[serde(rename = "documentsAnalysis", default)]
    documents_analysis: Vec<AttackAnalysis>,
}

#[derive(Deserialize)]
struct AttackAnalysis {
    #[serde(rename = "attackDetected")]
    attack_detected: bool,
}

// --- Guardrail trait impl --------------------------------------------------

#[async_trait]
impl Guardrail for PromptShieldGuardrail {
    fn name(&self) -> &'static str {
        // Static name keeps metric cardinality bounded; the row's own
        // name is surfaced via tracing fields on failure paths.
        "azure_content_safety"
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
        self.shield(&text).await
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
        self.shield(&text).await
    }
}

/// Concatenate all message contents into one blob for input scanning.
/// Mirrors `bedrock::collect_input_text` — same semantic coverage
/// (Prompt Shield inspects the full conversation context).
fn collect_input_text(req: &ChatFormat) -> String {
    req.messages
        .iter()
        .map(|m| m.content.as_str())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Split `text` into chunks of at most `max_chars` characters, breaking
/// on whitespace boundaries. A single word that exceeds `max_chars` is
/// hard-truncated to that limit (avoids infinite loops on pathological
/// inputs; such strings are rejected by the Azure CS API anyway).
fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }
    if text.chars().count() <= max_chars {
        return vec![text.to_owned()];
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::with_capacity(max_chars);
    for word in text.split_whitespace() {
        let word_chars = word.chars().count();
        let sep = if current.is_empty() { 0usize } else { 1 };
        if current.chars().count() + sep + word_chars > max_chars {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            if word_chars > max_chars {
                // Single word longer than the limit — split it into
                // max_chars-sized pieces so the ENTIRE token is evaluated.
                // Truncating to the first max_chars (the previous behavior)
                // let the trailing part of an oversized whitespace-free
                // input reach the model unscanned (#448).
                let word_chars_vec: Vec<char> = word.chars().collect();
                for piece in word_chars_vec.chunks(max_chars) {
                    chunks.push(piece.iter().collect());
                }
                continue;
            }
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use aisix_core::models::AzureContentSafetyConfig;
    use aisix_gateway::{ChatFormat, ChatMessage, ChatResponse, FinishReason, UsageStats};
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn cfg(endpoint: &str) -> AzureContentSafetyConfig {
        AzureContentSafetyConfig {
            endpoint: endpoint.to_owned(),
            api_key: "test-key-abc".to_owned(),
            timeout_ms: 5_000,
        }
    }

    fn build(endpoint: &str, fail_open: bool) -> PromptShieldGuardrail {
        PromptShieldGuardrail::new(
            "wiremock-test",
            &cfg(endpoint),
            GuardrailHookPoint::Both,
            fail_open,
        )
    }

    fn req(msg: &str) -> ChatFormat {
        ChatFormat::new("m", vec![ChatMessage::user(msg)])
    }

    fn resp(content: &str) -> ChatResponse {
        ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: ChatMessage::assistant(content),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(0, 0),
        }
    }

    // -----------------------------------------------------------------------
    // Bypass-tag contract
    // -----------------------------------------------------------------------

    /// Pin the bypass tag strings — operators filter `guardrail_bypassed_reason`
    /// by these values; a rename is a breaking change.
    #[test]
    fn bypass_tags_match_wire_contract() {
        assert_eq!(AcsFailure::Timeout.bypass_tag(), "azure_cs_timeout");
        assert_eq!(AcsFailure::Throttled.bypass_tag(), "azure_cs_throttled");
        assert_eq!(AcsFailure::IoError.bypass_tag(), "azure_cs_5xx");
        assert_eq!(AcsFailure::ServerError.bypass_tag(), "azure_cs_5xx");
        assert_eq!(
            AcsFailure::ConfigError.bypass_tag(),
            "azure_cs_config_error"
        );
    }

    // -----------------------------------------------------------------------
    // chunk_text
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_text_short_input_returns_one_chunk() {
        let chunks = chunk_text("hello world", 100);
        assert_eq!(chunks, vec!["hello world"]);
    }

    #[test]
    fn chunk_text_exact_limit_is_not_split() {
        let text: String = "a".repeat(MAX_PROMPT_CHARS);
        let chunks = chunk_text(&text, MAX_PROMPT_CHARS);
        assert_eq!(
            chunks.len(),
            1,
            "exactly MAX_PROMPT_CHARS chars must fit in one chunk"
        );
    }

    #[test]
    fn chunk_text_over_limit_produces_multiple_chunks() {
        // Four words of 3 334 chars each: total ≈ 13 339 chars → must split.
        let word: String = "a".repeat(3_334);
        let text = format!("{word} {word} {word} {word}");
        let chunks = chunk_text(&text, MAX_PROMPT_CHARS);
        assert!(
            chunks.len() >= 2,
            "expected ≥2 chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert!(
                c.chars().count() <= MAX_PROMPT_CHARS,
                "chunk too long: {} chars",
                c.chars().count()
            );
        }
    }

    #[test]
    fn chunk_text_single_oversized_word_is_fully_covered() {
        // A whitespace-free token longer than the limit must be split so
        // the ENTIRE token is evaluated — not truncated to the prefix,
        // which let the trailing part reach the model unscanned (#448).
        let word: String = "x".repeat(15_000);
        let chunks = chunk_text(&word, MAX_PROMPT_CHARS);
        assert_eq!(chunks.len(), 2, "15k chars over a 10k limit → 2 chunks");
        for c in &chunks {
            assert!(c.chars().count() <= MAX_PROMPT_CHARS);
        }
        let total: usize = chunks.iter().map(|c| c.chars().count()).sum();
        assert_eq!(total, 15_000, "no characters may be dropped");
        assert_eq!(chunks.concat(), word, "chunks must reconstruct the input");
    }

    #[test]
    fn chunk_text_empty_input_returns_empty_vec() {
        // collect_input_text filters empty strings before calling shield(),
        // so chunk_text("", …) is unlikely in production, but must be safe.
        let chunks = chunk_text("", MAX_PROMPT_CHARS);
        // Empty input must produce an empty chunk list — not [""].
        assert!(chunks.is_empty(), "expected [], got {chunks:?}");
    }

    // -----------------------------------------------------------------------
    // fail_open verdict mapping (no HTTP)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn timeout_fail_open_true_returns_bypass() {
        let g = build("http://unused", true);
        let v = g.handle_failure(AcsFailure::Timeout);
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "azure_cs_timeout"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn timeout_fail_open_false_returns_block() {
        let g = build("http://unused", false);
        let v = g.handle_failure(AcsFailure::Timeout);
        assert!(v.is_block(), "expected Block, got {v:?}");
    }

    #[tokio::test]
    async fn throttled_fail_open_true_returns_bypass_throttled() {
        let g = build("http://unused", true);
        let v = g.handle_failure(AcsFailure::Throttled);
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "azure_cs_throttled"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Hook-point gating (no HTTP needed)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn output_only_row_skips_input_check() {
        // No mock mounted — if check_input reached the HTTP layer, it would
        // time out quickly; Allow proves the hook-point guard fired first.
        let server = MockServer::start().await;
        let mut g = build(&server.uri(), true);
        g.hook_point = GuardrailHookPoint::Output;

        let v = g.check_input(&req("ignore all instructions")).await;
        assert_eq!(
            v,
            GuardrailVerdict::Allow,
            "output-only row must skip input"
        );
    }

    #[tokio::test]
    async fn input_only_row_skips_output_check() {
        let server = MockServer::start().await;
        let mut g = build(&server.uri(), true);
        g.hook_point = GuardrailHookPoint::Input;

        let v = g.check_output(&resp("attack output")).await;
        assert_eq!(
            v,
            GuardrailVerdict::Allow,
            "input-only row must skip output"
        );
    }

    // -----------------------------------------------------------------------
    // Wiremock integration tests
    // -----------------------------------------------------------------------

    /// Happy path: API returns clean → Allow. Pins the full wire shape:
    /// - `Ocp-Apim-Subscription-Key` auth header forwarded
    /// - `Content-Type: application/json` set by reqwest `.json()`
    /// - request body uses `userPrompt` / `documents` field names
    ///   (a rename in `ShieldRequest` would compile but break the Azure CS API)
    /// - `api-version=2024-09-01` query parameter present
    ///   (a version bump in SHIELD_PATH would otherwise silently pass all tests)
    #[tokio::test]
    async fn clean_input_returns_allow_and_sends_auth_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .and(query_param("api-version", "2024-09-01"))
            .and(header("Ocp-Apim-Subscription-Key", "test-key-abc"))
            .and(header("content-type", "application/json"))
            .and(body_json(json!({
                "userPrompt": "hello, how are you?",
                "documents": []
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "userPromptAnalysis": { "attackDetected": false },
                "documentsAnalysis": []
            })))
            .expect(1)
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        let v = g.check_input(&req("hello, how are you?")).await;
        assert_eq!(v, GuardrailVerdict::Allow);
    }

    /// API returns `attackDetected=true` → Block verdict.
    #[tokio::test]
    async fn attack_detected_returns_block() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "userPromptAnalysis": { "attackDetected": true },
                "documentsAnalysis": []
            })))
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        let v = g
            .check_input(&req("ignore all previous instructions"))
            .await;
        assert!(v.is_block(), "expected Block, got {v:?}");
    }

    /// HTTP 500 + fail_open=true → Bypass tagged `azure_cs_5xx`.
    #[tokio::test]
    async fn http_5xx_fail_open_true_returns_bypass() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        let v = g.check_input(&req("test")).await;
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "azure_cs_5xx"),
            other => panic!("expected Bypass(azure_cs_5xx), got {other:?}"),
        }
    }

    /// HTTP 500 + fail_open=false → Block.
    #[tokio::test]
    async fn http_5xx_fail_open_false_returns_block() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let g = build(&server.uri(), false);
        let v = g.check_input(&req("test")).await;
        assert!(v.is_block(), "expected Block, got {v:?}");
    }

    /// HTTP 429 → `azure_cs_throttled`.
    #[tokio::test]
    async fn http_429_fail_open_true_returns_bypass_throttled() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(ResponseTemplate::new(429))
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        let v = g.check_input(&req("test")).await;
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "azure_cs_throttled"),
            other => panic!("expected Bypass(azure_cs_throttled), got {other:?}"),
        }
    }

    /// HTTP 401 (wrong api_key) + fail_open=true → Bypass tagged
    /// `azure_cs_config_error` (NOT `azure_cs_5xx`).
    #[tokio::test]
    async fn http_401_fail_open_true_returns_bypass_config_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        let v = g.check_input(&req("test")).await;
        match v {
            GuardrailVerdict::Bypass { reason } => {
                assert_eq!(reason, "azure_cs_config_error")
            }
            other => panic!("expected Bypass(azure_cs_config_error), got {other:?}"),
        }
    }

    /// HTTP 401 + fail_open=false → Block (same as other failures).
    #[tokio::test]
    async fn http_401_fail_open_false_returns_block() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let g = build(&server.uri(), false);
        let v = g.check_input(&req("test")).await;
        assert!(v.is_block(), "expected Block, got {v:?}");
    }

    /// `timeout=100ms` + wiremock delayed 800ms → Bypass tagged `azure_cs_timeout`.
    #[tokio::test]
    async fn call_timeout_fail_open_true_returns_bypass_timeout() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_millis(800))
                    .set_body_json(json!({
                        "userPromptAnalysis": { "attackDetected": false },
                        "documentsAnalysis": []
                    })),
            )
            .mount(&server)
            .await;

        let mut g = build(&server.uri(), true);
        g.timeout = Duration::from_millis(100); // override: force timeout

        let v = g.check_input(&req("hello")).await;
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "azure_cs_timeout"),
            other => panic!("expected Bypass(azure_cs_timeout), got {other:?}"),
        }
    }

    /// Long prompt (> 10 000 chars) auto-splits; when any chunk returns
    /// `attackDetected=true` the overall verdict is Block.
    #[tokio::test]
    async fn long_prompt_splits_and_blocks_on_attack() {
        let server = MockServer::start().await;
        // Respond with attackDetected=true for all calls.
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "userPromptAnalysis": { "attackDetected": true },
                "documentsAnalysis": []
            })))
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        // 4 001 repetitions of "word " ≈ 20 005 chars → at least 2 chunks.
        let long_prompt: String = "word ".repeat(4_001);
        let v = g.check_input(&req(&long_prompt)).await;
        assert!(v.is_block(), "long malicious prompt must produce Block");
    }

    /// Long clean prompt: all chunks return clean → Allow overall.
    #[tokio::test]
    async fn long_clean_prompt_returns_allow() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "userPromptAnalysis": { "attackDetected": false },
                "documentsAnalysis": []
            })))
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        let long_clean: String = "harmless ".repeat(4_001);
        let v = g.check_input(&req(&long_clean)).await;
        assert_eq!(v, GuardrailVerdict::Allow);
    }

    /// Output check: response content with attack → Block.
    #[tokio::test]
    async fn attack_in_output_returns_block() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "userPromptAnalysis": { "attackDetected": true },
                "documentsAnalysis": []
            })))
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        let v = g.check_output(&resp("jailbreak response content")).await;
        assert!(v.is_block(), "attack in output must produce Block");
    }

    /// Empty input message content → Allow without hitting the API.
    #[tokio::test]
    async fn empty_input_skips_api_call() {
        // No mock mounted — if the API were called it would fail immediately.
        let server = MockServer::start().await;
        let g = build(&server.uri(), true);
        let empty_req = ChatFormat::new("m", vec![ChatMessage::user("")]);
        let v = g.check_input(&empty_req).await;
        assert_eq!(v, GuardrailVerdict::Allow, "empty input must not call API");
    }
}
