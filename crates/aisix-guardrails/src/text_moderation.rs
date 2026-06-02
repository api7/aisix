//! kind=azure_content_safety_text_moderation guardrail dispatcher — calls
//! Azure AI Content Safety `text:analyze` on chat input and/or output and
//! translates the category-severity + blocklist result into a
//! [`GuardrailVerdict`].
//!
//! PRD-09c §6 P2 (#379).
//!
//! API reference (2024-09-01):
//! POST `{endpoint}/contentsafety/text:analyze?api-version=2024-09-01`
//! Source: <https://learn.microsoft.com/en-us/azure/ai-services/content-safety/reference>
//!
//! Wire shape:
//! ```json
//! // Request
//! { "text": "...", "categories": ["Hate","Sexual","SelfHarm","Violence"],
//!   "blocklistNames": [], "haltOnBlocklistHit": false,
//!   "outputType": "FourSeverityLevels" }
//! // Response
//! { "categoriesAnalysis": [ { "category": "Hate", "severity": 2 }, ... ],
//!   "blocklistsMatch": [ { "blocklistName": "...", "blocklistItemText": "..." } ] }
//! ```
//!
//! Block decision: a category whose `severity` reaches its threshold
//! (per-category override → general threshold → default 2), OR a
//! non-empty `blocklistsMatch`.
//!
//! Streaming output is moderated separately in `aisix-proxy`'s
//! `build_sse_stream` using the `stream_processing_mode` / `window_*`
//! config; this dispatcher only implements the non-streaming
//! `check_input` / `check_output` hooks.
//!
//! NOTE: the HTTP transport (`chunk_text`, the `AcsFailure` buckets, the
//! fail-open verdict mapping, the `Ocp-Apim-Subscription-Key` + tokio
//! timeout call shape) is duplicated from `prompt_shield.rs`. Extracting a
//! shared `azure_common` module is tracked as a follow-up so this slice
//! stays surgical (it does not touch the shipped P1 dispatcher).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{AzureContentSafetyTextModerationConfig, GuardrailHookPoint};
use aisix_gateway::{ChatFormat, ChatResponse, Role};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Guardrail, GuardrailVerdict, StreamOutputPolicy};

/// Maximum characters per `text:analyze` call. Azure CS enforces a
/// 10 000-char limit on `text`.
const MAX_TEXT_CHARS: usize = 10_000;

/// Path + query appended to the configured `endpoint`.
const ANALYZE_PATH: &str = "/contentsafety/text:analyze?api-version=2024-09-01";

/// One Azure Content Safety Text Moderation row, materialised into a
/// request-time dispatcher.
pub struct TextModerationGuardrail {
    row_name: String,
    endpoint: String,
    api_key: String,
    pub(crate) hook_point: GuardrailHookPoint,
    /// Fail-open policy for the INPUT hook (from the outer `Guardrail`).
    fail_open: bool,
    /// Fail-open policy for the OUTPUT hook. Defaults to `false`
    /// (fail-closed) so an Azure outage can't release unscanned model
    /// output — otherwise output moderation is defeated by a timeout.
    output_fail_open: bool,
    pub(crate) timeout: Duration,
    client: Arc<reqwest::Client>,

    // --- moderation parameters ---
    categories: Vec<String>,
    output_type: String,
    severity_threshold: u8,
    severity_threshold_by_category: BTreeMap<String, u8>,
    blocklist_names: Vec<String>,
    halt_on_blocklist_hit: bool,
    /// `concatenate_user_content` (default) scans only user messages on
    /// the input hook; `concatenate_all_content` scans every message.
    /// Ignored on the output hook (always the assistant message).
    text_source: String,

    // --- streaming-output controls (surfaced via stream_output_policy;
    // consumed by aisix-proxy's build_sse_stream) ---
    stream_processing_mode: String,
    window_size: u32,
    window_overlap_size: u32,
    max_buffer_bytes: u64,
    on_buffer_exceeded: String,
}

impl TextModerationGuardrail {
    pub fn new(
        row_name: impl Into<String>,
        cfg: &AzureContentSafetyTextModerationConfig,
        hook_point: GuardrailHookPoint,
        fail_open: bool,
    ) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .expect("reqwest::Client::builder() failed; this should never happen");
        Self {
            row_name: row_name.into(),
            endpoint: cfg.endpoint.trim_end_matches('/').to_owned(),
            api_key: cfg.api_key.clone(),
            hook_point,
            fail_open,
            output_fail_open: cfg.output_fail_open,
            timeout: Duration::from_millis(cfg.timeout_ms as u64),
            client: Arc::new(client),
            categories: cfg.categories.clone(),
            output_type: cfg.output_type.clone(),
            severity_threshold: cfg.severity_threshold,
            severity_threshold_by_category: cfg.severity_threshold_by_category.clone(),
            blocklist_names: cfg.blocklist_names.clone(),
            halt_on_blocklist_hit: cfg.halt_on_blocklist_hit,
            text_source: cfg.text_source.clone(),
            stream_processing_mode: cfg.stream_processing_mode.clone(),
            window_size: cfg.window_size,
            window_overlap_size: cfg.window_overlap_size,
            max_buffer_bytes: cfg.max_buffer_bytes,
            on_buffer_exceeded: cfg.on_buffer_exceeded.clone(),
        }
    }

    /// Scan `text` in ≤10 000-char chunks. Returns `Block` on the first
    /// chunk that crosses a category threshold or hits a blocklist;
    /// `Allow` when every chunk is clean; the fail-open mapping on error.
    async fn scan(&self, text: &str, fail_open: bool) -> GuardrailVerdict {
        for chunk in chunk_text(text, MAX_TEXT_CHARS) {
            match self.analyze(&chunk).await {
                Ok(resp) => {
                    if let Some(reason) = self.violation_reason(&resp) {
                        return GuardrailVerdict::Block { reason };
                    }
                }
                Err(failure) => return self.handle_failure(failure, fail_open),
            }
        }
        GuardrailVerdict::Allow
    }

    /// POST one chunk to `text:analyze` and return the parsed result.
    async fn analyze(&self, text: &str) -> Result<AnalyzeResponse, AcsFailure> {
        let url = format!("{}{}", self.endpoint, ANALYZE_PATH);
        let body = AnalyzeRequest {
            text,
            categories: &self.categories,
            blocklist_names: &self.blocklist_names,
            halt_on_blocklist_hit: self.halt_on_blocklist_hit,
            output_type: &self.output_type,
        };

        let future = self
            .client
            .post(&url)
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
            tracing::error!(
                row = %self.row_name,
                http_status = status.as_u16(),
                "azure content safety text:analyze returned 4xx — check endpoint and api_key configuration",
            );
            return Err(AcsFailure::ConfigError);
        }

        resp.json::<AnalyzeResponse>()
            .await
            .map_err(|_| AcsFailure::ServerError)
    }

    /// Apply the threshold + blocklist policy to one analyze response.
    /// Returns the block reason, or `None` when the chunk is clean.
    fn violation_reason(&self, resp: &AnalyzeResponse) -> Option<String> {
        for cat in &resp.categories_analysis {
            let threshold = self
                .severity_threshold_by_category
                .get(&cat.category)
                .copied()
                .unwrap_or(self.severity_threshold);
            if cat.severity >= threshold {
                return Some(format!(
                    "azure content safety: {} severity {} >= threshold {} (row: {})",
                    cat.category, cat.severity, threshold, self.row_name
                ));
            }
        }
        if let Some(first) = resp.blocklists_match.first() {
            return Some(format!(
                "azure content safety: blocklist {:?} matched (row: {})",
                first.blocklist_name, self.row_name
            ));
        }
        None
    }

    fn handle_failure(&self, failure: AcsFailure, fail_open: bool) -> GuardrailVerdict {
        let tag = failure.bypass_tag();
        if !matches!(failure, AcsFailure::ConfigError) {
            tracing::warn!(
                row = %self.row_name,
                failure = ?failure,
                fail_open,
                "azure content safety text moderation call failed",
            );
        }
        if fail_open {
            GuardrailVerdict::Bypass { reason: tag.into() }
        } else {
            GuardrailVerdict::Block {
                reason: format!("azure content safety unavailable ({tag})"),
            }
        }
    }

    /// Collect the text the INPUT hook scans, honoring `text_source`.
    fn collect_input_text(&self, req: &ChatFormat) -> String {
        let all = self.text_source == "concatenate_all_content";
        req.messages
            .iter()
            .filter(|m| all || m.role == Role::User)
            .map(crate::message_scan_text)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Failure cause buckets. `bypass_tag()` maps to the strings stored in
/// `usage_events.guardrail_bypassed_reason`; these match the P1 Prompt
/// Shield tags so operators filter both Azure kinds the same way.
#[derive(Debug)]
enum AcsFailure {
    Timeout,
    Throttled,
    IoError,
    ServerError,
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
struct AnalyzeRequest<'a> {
    text: &'a str,
    categories: &'a [String],
    #[serde(rename = "blocklistNames")]
    blocklist_names: &'a [String],
    #[serde(rename = "haltOnBlocklistHit")]
    halt_on_blocklist_hit: bool,
    #[serde(rename = "outputType")]
    output_type: &'a str,
}

#[derive(Deserialize)]
struct AnalyzeResponse {
    #[serde(
        rename = "categoriesAnalysis",
        default,
        deserialize_with = "null_as_empty"
    )]
    categories_analysis: Vec<CategoryAnalysis>,
    #[serde(
        rename = "blocklistsMatch",
        default,
        deserialize_with = "null_as_empty"
    )]
    blocklists_match: Vec<BlocklistMatch>,
}

/// Deserialize an array field that may arrive as JSON `null` (or be
/// absent) into an empty `Vec`. `#[serde(default)]` alone only covers
/// an *absent* field; a field present as `null` fails with "invalid
/// type: null, expected a sequence", which previously fail-closed the
/// whole request as an Azure `ServerError`. Conformant Azure always
/// sends `[]`, but a non-conformant upstream/proxy can send `null`
/// (#470).
fn null_as_empty<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    Ok(Option::<Vec<T>>::deserialize(d)?.unwrap_or_default())
}

#[derive(Deserialize)]
struct CategoryAnalysis {
    category: String,
    severity: u8,
}

#[derive(Deserialize)]
struct BlocklistMatch {
    #[serde(rename = "blocklistName", default)]
    blocklist_name: String,
}

// --- Guardrail trait impl --------------------------------------------------

#[async_trait]
impl Guardrail for TextModerationGuardrail {
    fn name(&self) -> &'static str {
        "azure_content_safety_text_moderation"
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
        let text = self.collect_input_text(req);
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        self.scan(&text, self.fail_open).await
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
        // Output uses its own fail policy (default fail-closed) so an
        // Azure outage can't release unscanned model output.
        self.scan(&text, self.output_fail_open).await
    }
}

/// Split `text` into chunks of at most `max_chars` characters on
/// whitespace boundaries. A single word over the limit is split into
/// max_chars-sized pieces so the entire token is evaluated (#448).
/// (Forked from `prompt_shield::chunk_text`; see the module note.)
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
                // Split the oversized token fully instead of truncating to
                // the prefix, which let the trailing part bypass scanning.
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
    use aisix_core::models::AzureContentSafetyTextModerationConfig;
    use aisix_gateway::{ChatFormat, ChatMessage, ChatResponse, FinishReason, UsageStats};
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[test]
    fn analyze_response_tolerates_null_and_absent_arrays() {
        // #470: a non-conformant upstream may send the arrays as JSON
        // `null`; that must deserialize to empty, not fail-closed.
        let from_null: AnalyzeResponse =
            serde_json::from_value(json!({ "categoriesAnalysis": null, "blocklistsMatch": null }))
                .expect("null arrays must deserialize to empty");
        assert!(from_null.categories_analysis.is_empty());
        assert!(from_null.blocklists_match.is_empty());

        // Absent fields (today's path) still work.
        let from_absent: AnalyzeResponse =
            serde_json::from_value(json!({})).expect("absent arrays default to empty");
        assert!(from_absent.categories_analysis.is_empty());
        assert!(from_absent.blocklists_match.is_empty());

        // Conformant Azure (`[]` + populated) is unaffected.
        let from_conformant: AnalyzeResponse = serde_json::from_value(json!({
            "categoriesAnalysis": [{ "category": "Hate", "severity": 4 }],
            "blocklistsMatch": []
        }))
        .expect("conformant body must deserialize");
        assert_eq!(from_conformant.categories_analysis.len(), 1);
        assert!(from_conformant.blocklists_match.is_empty());
    }

    fn cfg(endpoint: &str) -> AzureContentSafetyTextModerationConfig {
        // Mirrors what cp-api projects after applying its defaults.
        serde_json::from_value(json!({
            "endpoint": endpoint,
            "api_key": "test-key-abc",
            "timeout_ms": 5_000,
        }))
        .unwrap()
    }

    fn build(endpoint: &str, fail_open: bool) -> TextModerationGuardrail {
        TextModerationGuardrail::new(
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

    /// A clean analyze response (all categories below threshold).
    fn clean_body() -> serde_json::Value {
        json!({
            "categoriesAnalysis": [
                { "category": "Hate", "severity": 0 },
                { "category": "Violence", "severity": 0 }
            ],
            "blocklistsMatch": []
        })
    }

    // --- bypass-tag contract (shared with P1) ---

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

    #[test]
    fn chunk_text_exact_limit_is_not_split() {
        let text: String = "a".repeat(MAX_TEXT_CHARS);
        assert_eq!(chunk_text(&text, MAX_TEXT_CHARS).len(), 1);
    }

    #[test]
    fn stream_policy_reflects_config() {
        // Defaults → sliding window 10000/256.
        let g = build("http://unused", true);
        assert_eq!(
            g.stream_output_policy(),
            StreamOutputPolicy::Window {
                size_chars: 10_000,
                overlap_chars: 256
            }
        );
        // buffer_full mode surfaces the cap + on_exceeded policy.
        let mut g2 = build("http://unused", true);
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

    // --- severity threshold logic (no HTTP) ---

    fn mk_resp(cats: &[(&str, u8)], blocklist: bool) -> AnalyzeResponse {
        AnalyzeResponse {
            categories_analysis: cats
                .iter()
                .map(|(c, s)| CategoryAnalysis {
                    category: (*c).to_owned(),
                    severity: *s,
                })
                .collect(),
            blocklists_match: if blocklist {
                vec![BlocklistMatch {
                    blocklist_name: "corp-terms".to_owned(),
                }]
            } else {
                vec![]
            },
        }
    }

    #[test]
    fn default_threshold_blocks_at_two() {
        let g = build("http://unused", true);
        // severity 2 >= default 2 → block
        assert!(g
            .violation_reason(&mk_resp(&[("Hate", 2)], false))
            .is_some());
        // severity 0 < 2 → clean
        assert!(g
            .violation_reason(&mk_resp(&[("Hate", 0)], false))
            .is_none());
    }

    #[test]
    fn per_category_override_is_independent() {
        let mut g = build("http://unused", true);
        g.severity_threshold = 2;
        g.severity_threshold_by_category = BTreeMap::from([("Violence".to_owned(), 6u8)]);
        // Violence severity 4 < its override 6 → clean, even though it
        // would trip the general threshold of 2.
        assert!(g
            .violation_reason(&mk_resp(&[("Violence", 4)], false))
            .is_none());
        // Violence severity 6 >= override 6 → block.
        assert!(g
            .violation_reason(&mk_resp(&[("Violence", 6)], false))
            .is_some());
        // Hate has no override → general threshold 2 applies.
        assert!(g
            .violation_reason(&mk_resp(&[("Hate", 2)], false))
            .is_some());
    }

    #[test]
    fn blocklist_match_blocks_regardless_of_severity() {
        let g = build("http://unused", true);
        assert!(g.violation_reason(&mk_resp(&[("Hate", 0)], true)).is_some());
    }

    // --- fail-open mapping (no HTTP) ---

    #[test]
    fn timeout_fail_open_true_returns_bypass() {
        let g = build("http://unused", true);
        match g.handle_failure(AcsFailure::Timeout, true) {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "azure_cs_timeout"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    #[test]
    fn output_defaults_fail_closed() {
        // cp-api omits output_fail_open when unset → serde default false.
        let g = build("http://unused", true);
        assert!(!g.output_fail_open, "output must default to fail-closed");
        // An output-side failure with fail_open=false must Block.
        assert!(g
            .handle_failure(AcsFailure::Timeout, g.output_fail_open)
            .is_block());
    }

    // --- text_source ---

    #[test]
    fn user_content_source_skips_assistant_messages() {
        let g = build("http://unused", true);
        let mut c = ChatFormat::new(
            "m",
            vec![
                ChatMessage::user("user says hi"),
                ChatMessage::assistant("assistant reply"),
            ],
        );
        c.messages.push(ChatMessage::user("more user text"));
        let text = g.collect_input_text(&c);
        assert!(text.contains("user says hi"));
        assert!(text.contains("more user text"));
        assert!(
            !text.contains("assistant reply"),
            "default concatenate_user_content must skip assistant messages"
        );
    }

    #[test]
    fn all_content_source_includes_assistant_messages() {
        let mut g = build("http://unused", true);
        g.text_source = "concatenate_all_content".to_owned();
        let c = ChatFormat::new(
            "m",
            vec![
                ChatMessage::user("user says hi"),
                ChatMessage::assistant("assistant reply"),
            ],
        );
        let text = g.collect_input_text(&c);
        assert!(text.contains("user says hi"));
        assert!(
            text.contains("assistant reply"),
            "concatenate_all_content must include assistant messages"
        );
    }

    // --- wiremock integration ---

    #[tokio::test]
    async fn clean_input_returns_allow_and_sends_wire_shape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .and(query_param("api-version", "2024-09-01"))
            .and(header("Ocp-Apim-Subscription-Key", "test-key-abc"))
            .and(body_partial_json(json!({
                "text": "hello there",
                "outputType": "FourSeverityLevels"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(clean_body()))
            .expect(1)
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        assert_eq!(
            g.check_input(&req("hello there")).await,
            GuardrailVerdict::Allow
        );
    }

    #[tokio::test]
    async fn high_severity_input_returns_block() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "categoriesAnalysis": [ { "category": "Hate", "severity": 6 } ],
                "blocklistsMatch": []
            })))
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        assert!(g.check_input(&req("hateful content")).await.is_block());
    }

    #[tokio::test]
    async fn http_5xx_fail_open_true_returns_bypass() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        match g.check_input(&req("test")).await {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "azure_cs_5xx"),
            other => panic!("expected Bypass(azure_cs_5xx), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn output_5xx_fails_closed_by_default() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        // output_fail_open defaults false → an output-side 5xx must Block.
        let g = build(&server.uri(), true);
        assert!(
            g.check_output(&resp("some model output")).await.is_block(),
            "output hook must fail closed on Azure error by default"
        );
    }

    #[tokio::test]
    async fn high_severity_output_returns_block() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "categoriesAnalysis": [ { "category": "Violence", "severity": 4 } ],
                "blocklistsMatch": []
            })))
            .mount(&server)
            .await;

        let g = build(&server.uri(), true);
        assert!(g.check_output(&resp("violent output")).await.is_block());
    }
}
