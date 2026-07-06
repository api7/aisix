//! kind=presidio guardrail dispatcher (#52) — self-hosted Microsoft
//! Presidio PII detection + anonymization.
//!
//! Two-step API (customer-run containers, no vendor secret):
//! - `POST {analyzer_url}/analyze` `{ text, language, entities?,
//!   score_threshold? }` → `[{ entity_type, start, end, score }]`
//! - `POST {anonymizer_url}/anonymize` `{ text, analyzer_results,
//!   anonymizers }` → `{ text, items: [{ entity_type, ... }] }`
//!
//! Source: <https://microsoft.github.io/presidio/api-docs/api-docs.html>
//!
//! Decision rule (per-entity actions, same shape as `kind: "pii"`):
//! - any detected entity whose effective action is `block` → Block;
//! - otherwise detected entities (action `mask`) → anonymize the text
//!   with the configured operator and continue — honored on the segment
//!   path (`moderate_*_segments`); the blob path (`check_*`) has no mask
//!   write-back channel, so a maskable outcome maps to Block there (same
//!   contract as kind=bedrock ANONYMIZE);
//! - nothing detected → Allow.
//!
//! vs. the built-in `kind: "pii"`: Presidio adds NER/ML entities a regex
//! cannot express (`PERSON`, `LOCATION`, `NRP`, …) and selectable
//! anonymize operators (`replace`, `mask`, `hash`, `redact`). LiteLLM's
//! `presidio` guardrail is the behavior baseline (per-entity MASK/BLOCK,
//! `language`, skip-empty-text); operator selection is our superset —
//! LiteLLM always uses Presidio's default replace.
//!
//! Block reasons and telemetry counts carry entity type NAMES only, never
//! matched values (#153 / #932 no-leak criterion).
//!
//! Behavior matrix (failure modes). The effective `fail_open` is the outer
//! `Guardrail::fail_open` on the INPUT hook and the independent
//! `PresidioConfig::output_fail_open` (default fail-closed) on the OUTPUT
//! hook:
//!
//! | API response                    | `fail_open` | Verdict                            |
//! |---------------------------------|-------------|------------------------------------|
//! | no entities                     | n/a         | Allow                              |
//! | entity with action=block        | n/a         | Block { reason }                   |
//! | entities, all action=mask       | n/a         | mask write-back (segment path)     |
//! | timeout                         | true        | Bypass { "presidio_timeout" }      |
//! | 429 Throttling                  | true        | Bypass { "presidio_throttled" }    |
//! | 5xx / IO error                  | true        | Bypass { "presidio_5xx" }          |
//! | 4xx (non-429, e.g. 400/404)     | true        | Bypass { "presidio_config_error" } |
//! | any failure                     | false       | Block { "presidio unavailable …" } |

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{GuardrailHookPoint, PresidioConfig};
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::pii::PiiAction;
use crate::{Guardrail, GuardrailVerdict, SegmentsOutcome, StreamOutputPolicy};

/// One Presidio row, materialised into a request-time dispatcher. Built
/// once per snapshot from [`PresidioConfig`] + the outer `Guardrail`
/// fields.
pub struct PresidioGuardrail {
    /// Operator-facing row name. Kept for log labels; the trait's static
    /// `name()` returns "presidio" so metric cardinality stays bounded.
    row_name: String,
    /// Analyzer base URL with trailing slash stripped.
    analyzer_url: String,
    /// Anonymizer base URL with trailing slash stripped.
    anonymizer_url: String,
    /// Entities to analyze for; empty → Presidio's full recognizer set.
    entities: Vec<String>,
    /// Per-entity action overrides (uppercased entity type → action).
    entity_actions: BTreeMap<String, PiiAction>,
    default_action: PiiAction,
    /// Anonymizer operator config for masked entities, pre-serialised.
    anonymizers: serde_json::Value,
    language: String,
    score_threshold: Option<f64>,
    hook_point: GuardrailHookPoint,
    /// Fail-open policy for the INPUT hook (the outer `Guardrail::fail_open`).
    fail_open: bool,
    /// Fail-open policy for the OUTPUT hook (default fail-closed).
    output_fail_open: bool,
    timeout: Duration,
    max_buffer_bytes: usize,
    on_buffer_exceeded_fail_open: bool,
    client: Arc<reqwest::Client>,
}

/// The anonymizer operator payload for one operator name. Presidio's
/// `/anonymize` takes `{"anonymizers": {"DEFAULT": { "type": ... }}}`;
/// unknown names are rejected at build time (`BuildError::InvalidValue`).
/// Source: <https://microsoft.github.io/presidio/anonymizer/>
pub fn operator_config(operator: &str) -> Option<serde_json::Value> {
    match operator {
        // Presidio's default: replace the span with `<ENTITY_TYPE>`.
        "replace" => Some(serde_json::json!({ "type": "replace" })),
        "mask" => Some(serde_json::json!({
            "type": "mask",
            "masking_char": "*",
            "chars_to_mask": 512,
            "from_end": false,
        })),
        "hash" => Some(serde_json::json!({ "type": "hash", "hash_type": "sha256" })),
        "redact" => Some(serde_json::json!({ "type": "redact" })),
        _ => None,
    }
}

impl PresidioGuardrail {
    /// Build the dispatcher from a parsed [`PresidioConfig`]. Caller owns
    /// `row_name`, `hook_point`, and `fail_open`, and has already
    /// validated `default_action`, per-entity actions, and `operator`
    /// (build.rs maps bad values to `BuildError::InvalidValue`).
    /// `operator` is the [`operator_config`] payload; it applies to every
    /// masked entity via the anonymizer's `DEFAULT` slot.
    pub fn new(
        row_name: impl Into<String>,
        cfg: &PresidioConfig,
        hook_point: GuardrailHookPoint,
        fail_open: bool,
        default_action: PiiAction,
        entity_actions: BTreeMap<String, PiiAction>,
        operator: serde_json::Value,
    ) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .expect("reqwest::Client::builder() failed; this should never happen");
        Self {
            row_name: row_name.into(),
            analyzer_url: cfg.analyzer_url.trim_end_matches('/').to_owned(),
            anonymizer_url: cfg.anonymizer_url.trim_end_matches('/').to_owned(),
            entities: cfg
                .entities
                .iter()
                .map(|e| e.entity_type.to_uppercase())
                .collect(),
            entity_actions,
            default_action,
            anonymizers: serde_json::json!({ "DEFAULT": operator }),
            language: cfg.language.clone(),
            score_threshold: cfg.score_threshold,
            hook_point,
            fail_open,
            output_fail_open: cfg.output_fail_open,
            timeout: Duration::from_millis(cfg.timeout_ms as u64),
            max_buffer_bytes: usize::try_from(cfg.max_buffer_bytes).unwrap_or(usize::MAX),
            on_buffer_exceeded_fail_open: cfg.on_buffer_exceeded == "fail_open",
            client: Arc::new(client),
        }
    }

    fn hook_enabled(&self, hook: GuardrailHookPoint) -> bool {
        self.hook_point == GuardrailHookPoint::Both || self.hook_point == hook
    }

    fn action_for(&self, entity_type: &str) -> PiiAction {
        self.entity_actions
            .get(&entity_type.to_uppercase())
            .copied()
            .unwrap_or(self.default_action)
    }

    /// `POST {analyzer_url}/analyze` for one text. Empty/whitespace-only
    /// text short-circuits to no results (Presidio 500s on it; LiteLLM
    /// skips it the same way).
    async fn analyze(&self, text: &str) -> Result<Vec<AnalyzerResult>, PresidioFailure> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/analyze", self.analyzer_url);
        let body = AnalyzeRequest {
            text,
            language: &self.language,
            entities: (!self.entities.is_empty()).then_some(&self.entities),
            score_threshold: self.score_threshold,
        };
        let parsed: Vec<AnalyzerResult> = self.post_json(&url, &body).await?;
        Ok(parsed)
    }

    /// `POST {anonymizer_url}/anonymize` — rewrite `text` per
    /// `analyzer_results` with the configured operator.
    async fn anonymize(
        &self,
        text: &str,
        results: &[AnalyzerResult],
    ) -> Result<AnonymizeResponse, PresidioFailure> {
        let url = format!("{}/anonymize", self.anonymizer_url);
        let body = AnonymizeRequest {
            text,
            analyzer_results: results,
            anonymizers: &self.anonymizers,
        };
        self.post_json(&url, &body).await
    }

    async fn post_json<B: Serialize, T: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        body: &B,
    ) -> Result<T, PresidioFailure> {
        let future = self.client.post(url).json(body).send();
        let resp = match tokio::time::timeout(self.timeout, future).await {
            Err(_elapsed) => return Err(PresidioFailure::Timeout),
            Ok(Err(_e)) => return Err(PresidioFailure::IoError),
            Ok(Ok(r)) => r,
        };
        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(PresidioFailure::Throttled);
        }
        if status.is_server_error() {
            return Err(PresidioFailure::ServerError);
        }
        if !status.is_success() {
            // 4xx other than 429 — almost always a misconfiguration
            // (bad URL path, unsupported language, malformed entity list).
            tracing::error!(
                row = %self.row_name,
                http_status = status.as_u16(),
                url = %url,
                "presidio returned 4xx — check analyzer_url/anonymizer_url, language, and entities configuration",
            );
            return Err(PresidioFailure::ConfigError);
        }
        resp.json().await.map_err(|_| PresidioFailure::ServerError)
    }

    /// Analyze one text and fold the entity hits into a decision.
    async fn decide(&self, text: &str) -> Result<TextDecision, PresidioFailure> {
        let results = self.analyze(text).await?;
        if results.is_empty() {
            return Ok(TextDecision::Clean);
        }
        let blocking: Vec<&str> = results
            .iter()
            .filter(|r| self.action_for(&r.entity_type) == PiiAction::Block)
            .map(|r| r.entity_type.as_str())
            .collect();
        if !blocking.is_empty() {
            let mut names: Vec<&str> = blocking;
            names.sort_unstable();
            names.dedup();
            return Ok(TextDecision::Block(
                names.iter().map(|s| s.to_string()).collect(),
            ));
        }
        Ok(TextDecision::Mask(results))
    }

    /// Blob-mode check: verdict only. Serves `check_input`/`check_output`
    /// — the families with no mask write-back channel — so a maskable
    /// outcome maps to Block there.
    async fn check_blob(&self, text: &str, fail_open: bool) -> GuardrailVerdict {
        match self.decide(text).await {
            Ok(TextDecision::Clean) => GuardrailVerdict::Allow,
            Ok(TextDecision::Block(entities)) => self.block_verdict(&entities),
            Ok(TextDecision::Mask(results)) => {
                let mut names: Vec<&str> = results.iter().map(|r| r.entity_type.as_str()).collect();
                names.sort_unstable();
                names.dedup();
                GuardrailVerdict::block(format!(
                    "presidio detected PII ({}) (row: {})",
                    names.join(", "),
                    self.row_name
                ))
            }
            Err(failure) => self.handle_failure(failure, fail_open),
        }
    }

    /// Segment-mode moderation: analyze every slot (sequentially — the
    /// self-hosted analyzer is typically a single container; a burst of
    /// concurrent calls per request would thundering-herd it), block if
    /// any slot has a blocking entity, else anonymize the slots that had
    /// hits and return the positionally-aligned masked vec.
    async fn moderate_segments(&self, texts: &[String], fail_open: bool) -> SegmentsOutcome {
        let mut decisions: Vec<Option<Vec<AnalyzerResult>>> = Vec::with_capacity(texts.len());
        for text in texts {
            match self.decide(text).await {
                Ok(TextDecision::Clean) => decisions.push(None),
                Ok(TextDecision::Block(entities)) => {
                    return SegmentsOutcome::from_verdict(self.block_verdict(&entities));
                }
                Ok(TextDecision::Mask(results)) => decisions.push(Some(results)),
                Err(failure) => {
                    return SegmentsOutcome::from_verdict(self.handle_failure(failure, fail_open));
                }
            }
        }
        if decisions.iter().all(Option::is_none) {
            return SegmentsOutcome::allow();
        }

        let mut masked: Vec<String> = texts.to_vec();
        let mut counts: BTreeMap<String, u32> = BTreeMap::new();
        for (i, results) in decisions.into_iter().enumerate() {
            let Some(results) = results else { continue };
            match self.anonymize(&texts[i], &results).await {
                Ok(resp) => {
                    for item in &resp.items {
                        *counts.entry(item.entity_type.clone()).or_insert(0) += 1;
                    }
                    masked[i] = resp.text;
                }
                Err(failure) => {
                    // The analyzer FOUND PII but the anonymizer can't
                    // rewrite it — releasing the original would defeat the
                    // policy, so the failure verdict (fail_open → Bypass)
                    // replaces the whole segment outcome.
                    return SegmentsOutcome::from_verdict(self.handle_failure(failure, fail_open));
                }
            }
        }
        SegmentsOutcome {
            verdict: GuardrailVerdict::Allow,
            masked: Some(masked),
            counts,
        }
    }

    fn block_verdict(&self, entities: &[String]) -> GuardrailVerdict {
        GuardrailVerdict::block(format!(
            "presidio blocked on entity ({}) (row: {})",
            entities.join(", "),
            self.row_name
        ))
    }

    fn handle_failure(&self, failure: PresidioFailure, fail_open: bool) -> GuardrailVerdict {
        let tag = failure.bypass_tag();
        // ConfigError is already logged at error level in post_json().
        if !matches!(failure, PresidioFailure::ConfigError) {
            tracing::warn!(
                row = %self.row_name,
                failure = ?failure,
                fail_open = fail_open,
                "presidio call failed",
            );
        }
        if fail_open {
            GuardrailVerdict::Bypass { reason: tag.into() }
        } else {
            GuardrailVerdict::block(format!("presidio unavailable ({tag})"))
        }
    }
}

/// What one analyzed text resolves to before write-back.
enum TextDecision {
    Clean,
    /// Entity types (deduped) whose action is `block`.
    Block(Vec<String>),
    /// Entities detected, all maskable — the analyzer results feed
    /// `/anonymize`.
    Mask(Vec<AnalyzerResult>),
}

/// Failure cause buckets. `bypass_tag()` maps to the strings stored in
/// `usage_events.guardrail_bypassed_reason` — changing them is a breaking
/// change for operators who filter on these values.
#[derive(Debug)]
enum PresidioFailure {
    Timeout,
    Throttled,
    IoError,
    ServerError,
    ConfigError,
}

impl PresidioFailure {
    fn bypass_tag(&self) -> &'static str {
        match self {
            Self::Timeout => "presidio_timeout",
            Self::Throttled => "presidio_throttled",
            Self::IoError | Self::ServerError => "presidio_5xx",
            Self::ConfigError => "presidio_config_error",
        }
    }
}

// --- serde shapes for the wire protocol ------------------------------------

#[derive(Serialize)]
struct AnalyzeRequest<'a> {
    text: &'a str,
    language: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    entities: Option<&'a Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score_threshold: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnalyzerResult {
    entity_type: String,
    start: usize,
    end: usize,
    score: f64,
}

#[derive(Serialize)]
struct AnonymizeRequest<'a> {
    text: &'a str,
    analyzer_results: &'a [AnalyzerResult],
    anonymizers: &'a serde_json::Value,
}

#[derive(Deserialize)]
struct AnonymizeResponse {
    text: String,
    #[serde(default)]
    items: Vec<AnonymizedItem>,
}

#[derive(Deserialize)]
struct AnonymizedItem {
    entity_type: String,
}

// --- Guardrail trait impl --------------------------------------------------

#[async_trait]
impl Guardrail for PresidioGuardrail {
    fn name(&self) -> &'static str {
        "presidio"
    }

    fn runs_on_output(&self) -> bool {
        matches!(
            self.hook_point,
            GuardrailHookPoint::Output | GuardrailHookPoint::Both
        )
    }

    /// Masking a streamed response requires the whole response held back —
    /// a masked span can cross any chunk boundary. Cap + overflow policy
    /// come from the row config, like kind=pii.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        StreamOutputPolicy::BufferFull {
            max_buffer_bytes: self.max_buffer_bytes,
            on_exceeded_fail_open: self.on_buffer_exceeded_fail_open,
        }
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        if !self.hook_enabled(GuardrailHookPoint::Input) {
            return GuardrailVerdict::Allow;
        }
        let text = collect_input_text(req);
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        self.check_blob(&text, self.fail_open).await
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        if !self.hook_enabled(GuardrailHookPoint::Output) {
            return GuardrailVerdict::Allow;
        }
        let text = resp.guardrail_output_text();
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        self.check_blob(&text, self.output_fail_open).await
    }

    /// Presidio moderates via the segment pass on call sites that support
    /// mask write-back; those sites pair `moderate_*_segments` with
    /// `check_*_non_segment`, so the guardrail is called exactly once.
    fn moderates_segments(&self) -> bool {
        true
    }

    async fn moderate_input_segments(&self, texts: &[String]) -> SegmentsOutcome {
        if !self.hook_enabled(GuardrailHookPoint::Input) {
            return SegmentsOutcome::allow();
        }
        self.moderate_segments(texts, self.fail_open).await
    }

    async fn moderate_output_segments(&self, texts: &[String]) -> SegmentsOutcome {
        if !self.hook_enabled(GuardrailHookPoint::Output) {
            return SegmentsOutcome::allow();
        }
        self.moderate_segments(texts, self.output_fail_open).await
    }
}

/// Concatenate all message contents into one blob for the blob-path input
/// scan. Mirrors `bedrock::collect_input_text` — same semantic coverage.
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
    use aisix_core::models::{PresidioConfig, PresidioEntityConfig};
    use aisix_gateway::{ChatFormat, ChatMessage};
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn cfg(analyzer: &str, anonymizer: &str) -> PresidioConfig {
        PresidioConfig {
            analyzer_url: analyzer.to_owned(),
            anonymizer_url: anonymizer.to_owned(),
            entities: vec![
                PresidioEntityConfig {
                    entity_type: "EMAIL_ADDRESS".to_owned(),
                    action: None,
                },
                PresidioEntityConfig {
                    entity_type: "US_SSN".to_owned(),
                    action: Some("block".to_owned()),
                },
            ],
            default_action: "mask".to_owned(),
            operator: "replace".to_owned(),
            language: "en".to_owned(),
            score_threshold: Some(0.5),
            timeout_ms: 5_000,
            output_fail_open: false,
            max_buffer_bytes: 262_144,
            on_buffer_exceeded: "fail_closed".to_owned(),
        }
    }

    fn build(analyzer: &str, anonymizer: &str, fail_open: bool) -> PresidioGuardrail {
        let c = cfg(analyzer, anonymizer);
        let mut entity_actions = BTreeMap::new();
        entity_actions.insert("US_SSN".to_owned(), PiiAction::Block);
        PresidioGuardrail::new(
            "wiremock-test",
            &c,
            GuardrailHookPoint::Both,
            fail_open,
            PiiAction::Mask,
            entity_actions,
            operator_config("replace").unwrap(),
        )
    }

    fn req(msg: &str) -> ChatFormat {
        ChatFormat::new("m", vec![ChatMessage::user(msg)])
    }

    #[tokio::test]
    async fn clean_input_allows_without_anonymizer_call() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .and(body_partial_json(json!({
                "language": "en",
                "entities": ["EMAIL_ADDRESS", "US_SSN"],
                "score_threshold": 0.5
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
            .expect(1)
            .mount(&server)
            .await;
        // No /anonymize mock mounted — a call to it would 404 → ConfigError
        // → Block, so Allow also proves the anonymizer was never consulted.
        let g = build(&server.uri(), &server.uri(), false);
        assert_eq!(g.check_input(&req("hello")).await, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn blocking_entity_blocks() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                { "entity_type": "US_SSN", "start": 4, "end": 15, "score": 0.9 }
            ])))
            .mount(&server)
            .await;
        let g = build(&server.uri(), &server.uri(), false);
        match g.check_input(&req("ssn 123-45-6789")).await {
            GuardrailVerdict::Block { reason, .. } => {
                assert!(reason.contains("US_SSN"), "reason: {reason}");
                // no-leak criterion: the matched value never appears.
                assert!(!reason.contains("123-45-6789"), "value leaked: {reason}");
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn maskable_blob_path_blocks() {
        // The blob path has no write-back channel, so a maskable outcome
        // maps to Block there (kind=bedrock ANONYMIZE contract).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                { "entity_type": "EMAIL_ADDRESS", "start": 0, "end": 5, "score": 0.9 }
            ])))
            .mount(&server)
            .await;
        let g = build(&server.uri(), &server.uri(), false);
        match g.check_input(&req("a@b.c hello")).await {
            GuardrailVerdict::Block { reason, .. } => {
                assert!(reason.contains("EMAIL_ADDRESS"), "reason: {reason}");
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn segment_path_masks_via_anonymizer() {
        let analyzer = MockServer::start().await;
        let anonymizer = MockServer::start().await;
        // Slot 0 is clean, slot 1 carries the email.
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .and(body_partial_json(json!({ "text": "no pii here" })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
            .mount(&analyzer)
            .await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .and(body_partial_json(json!({ "text": "mail a@b.c" })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                { "entity_type": "EMAIL_ADDRESS", "start": 5, "end": 10, "score": 0.85 }
            ])))
            .mount(&analyzer)
            .await;
        Mock::given(method("POST"))
            .and(path("/anonymize"))
            .and(body_partial_json(json!({
                "text": "mail a@b.c",
                "anonymizers": { "DEFAULT": { "type": "replace" } }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "text": "mail <EMAIL_ADDRESS>",
                "items": [
                    { "operator": "replace", "entity_type": "EMAIL_ADDRESS",
                      "start": 5, "end": 20, "text": "<EMAIL_ADDRESS>" }
                ]
            })))
            .expect(1)
            .mount(&anonymizer)
            .await;

        let g = build(&analyzer.uri(), &anonymizer.uri(), false);
        let texts = vec!["no pii here".to_owned(), "mail a@b.c".to_owned()];
        let out = g.moderate_input_segments(&texts).await;
        assert_eq!(out.verdict, GuardrailVerdict::Allow);
        let masked = out.masked.expect("mask write-back expected");
        assert_eq!(masked[0], "no pii here");
        assert_eq!(masked[1], "mail <EMAIL_ADDRESS>");
        assert_eq!(out.counts.get("EMAIL_ADDRESS"), Some(&1));
    }

    #[tokio::test]
    async fn segment_path_block_entity_short_circuits() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                { "entity_type": "US_SSN", "start": 0, "end": 11, "score": 0.95 }
            ])))
            .mount(&server)
            .await;
        let g = build(&server.uri(), &server.uri(), false);
        let out = g.moderate_input_segments(&["123-45-6789".to_owned()]).await;
        assert!(out.verdict.is_block(), "got {:?}", out.verdict);
        assert!(out.masked.is_none());
    }

    #[tokio::test]
    async fn anonymizer_failure_does_not_release_unmasked_content() {
        let analyzer = MockServer::start().await;
        let anonymizer = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                { "entity_type": "EMAIL_ADDRESS", "start": 0, "end": 5, "score": 0.9 }
            ])))
            .mount(&analyzer)
            .await;
        Mock::given(method("POST"))
            .and(path("/anonymize"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&anonymizer)
            .await;
        // fail_open=false: the failure blocks.
        let g = build(&analyzer.uri(), &anonymizer.uri(), false);
        let out = g.moderate_input_segments(&["a@b.c".to_owned()]).await;
        assert!(out.verdict.is_block(), "got {:?}", out.verdict);
        assert!(out.masked.is_none());
    }

    #[tokio::test]
    async fn empty_text_skips_analyzer() {
        // No /analyze mock mounted — a call would 404 → ConfigError → Block,
        // so Allow proves empty slots never reach the analyzer.
        let server = MockServer::start().await;
        let g = build(&server.uri(), &server.uri(), false);
        let out = g
            .moderate_input_segments(&[String::new(), "   ".to_owned()])
            .await;
        assert_eq!(out, SegmentsOutcome::allow());
    }

    #[tokio::test]
    async fn five_xx_fail_open_bypasses_fail_closed_blocks() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let open = build(&server.uri(), &server.uri(), true);
        assert_eq!(
            open.check_input(&req("x")).await,
            GuardrailVerdict::Bypass {
                reason: "presidio_5xx".into()
            }
        );
        let closed = build(&server.uri(), &server.uri(), false);
        assert!(closed.check_input(&req("x")).await.is_block());
    }

    #[tokio::test]
    async fn config_error_4xx_tagged_separately() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(400))
            .mount(&server)
            .await;
        let g = build(&server.uri(), &server.uri(), true);
        assert_eq!(
            g.check_input(&req("x")).await,
            GuardrailVerdict::Bypass {
                reason: "presidio_config_error".into()
            }
        );
    }

    #[test]
    fn operator_configs() {
        assert_eq!(
            operator_config("replace").unwrap(),
            json!({ "type": "replace" })
        );
        assert_eq!(
            operator_config("hash").unwrap(),
            json!({ "type": "hash", "hash_type": "sha256" })
        );
        assert!(operator_config("rot13").is_none());
    }

    #[tokio::test]
    async fn input_only_hook_skips_output() {
        let server = MockServer::start().await;
        let c = cfg(&server.uri(), &server.uri());
        let g = PresidioGuardrail::new(
            "t",
            &c,
            GuardrailHookPoint::Input,
            false,
            PiiAction::Mask,
            BTreeMap::new(),
            operator_config("replace").unwrap(),
        );
        assert!(!g.runs_on_output());
        let out = g.moderate_output_segments(&["x".to_owned()]).await;
        assert_eq!(out, SegmentsOutcome::allow());
    }
}
