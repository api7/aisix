//! kind=openai_moderation guardrail dispatcher (#52) — calls the OpenAI
//! Moderation API and translates the result into a [`GuardrailVerdict`].
//! Detection-only: it blocks, never rewrites. Monitor-before-enforce comes
//! from the row's `enforcement_mode`.
//!
//! API reference:
//! POST `{endpoint}/moderations`, `Authorization: Bearer <key>`
//! Source: <https://platform.openai.com/docs/guides/moderation>
//!
//! Wire shape:
//! ```json
//! // Request
//! { "model": "omni-moderation-latest", "input": "..." }
//! // Response
//! { "results": [{ "flagged": bool,
//!                 "categories": { "violence": true, ... },
//!                 "category_scores": { "violence": 0.97, ... } }] }
//! ```
//!
//! Decision rule: with `category_thresholds` empty (the default), the
//! API's `flagged` boolean decides — the LiteLLM `openai_moderation`
//! baseline behavior. With thresholds configured, ONLY the listed
//! categories are enforced: a category blocks when its score reaches its
//! threshold, and the API's own `flagged` is ignored (LiteLLM has no
//! equivalent knob).
//!
//! The cp-api decrypts the envelope-encrypted `api_key` at kine-projection
//! time so this module only handles plaintext keys. The key is never
//! logged; block reasons carry category NAMES only, never matched content
//! (#153).
//!
//! Behavior matrix (failure modes). The effective `fail_open` is the outer
//! `Guardrail::fail_open` on the INPUT hook and the independent
//! `OpenaiModerationConfig::output_fail_open` (default fail-closed) on the
//! OUTPUT hook:
//!
//! | API response                    | `fail_open` | Verdict                                     |
//! |---------------------------------|-------------|---------------------------------------------|
//! | not flagged / under thresholds  | n/a         | Allow                                       |
//! | flagged / over a threshold      | n/a         | Block { reason }                            |
//! | timeout                         | true        | Bypass { "openai_moderation_timeout" }      |
//! | 429 Throttling                  | true        | Bypass { "openai_moderation_throttled" }    |
//! | 5xx / IO error                  | true        | Bypass { "openai_moderation_5xx" }          |
//! | 4xx (non-429, e.g. 401/400)     | true        | Bypass { "openai_moderation_config_error" } |
//! | any failure                     | false       | Block { "openai moderation unavailable …" } |

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{GuardrailHookPoint, OpenaiModerationConfig};
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Guardrail, GuardrailVerdict};

/// Default OpenAI API base (the config's `endpoint` overrides it).
const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1";

/// Path appended to the configured `endpoint`.
const MODERATIONS_PATH: &str = "/moderations";

/// One OpenAI Moderation row, materialised into a request-time dispatcher.
/// Built once per snapshot from [`OpenaiModerationConfig`] + the outer
/// `Guardrail` fields.
pub struct OpenaiModerationGuardrail {
    /// Operator-facing row name. Kept for log labels; the trait's static
    /// `name()` returns "openai_moderation" so metric cardinality stays
    /// bounded.
    row_name: String,
    /// Endpoint with trailing slash stripped.
    endpoint: String,
    /// Plaintext Bearer key (decrypted by cp-api before kine write).
    api_key: String,
    model: String,
    category_thresholds: BTreeMap<String, f64>,
    hook_point: GuardrailHookPoint,
    /// Fail-open policy for the INPUT hook (the outer `Guardrail::fail_open`).
    fail_open: bool,
    /// Fail-open policy for the OUTPUT hook (default fail-closed).
    output_fail_open: bool,
    timeout: Duration,
    client: Arc<reqwest::Client>,
}

impl OpenaiModerationGuardrail {
    /// Build the dispatcher from a parsed [`OpenaiModerationConfig`].
    /// Caller owns `row_name`, `hook_point`, and `fail_open` (they live on
    /// the outer `Guardrail` struct, not on the kind config).
    pub fn new(
        row_name: impl Into<String>,
        cfg: &OpenaiModerationConfig,
        hook_point: GuardrailHookPoint,
        fail_open: bool,
    ) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .expect("reqwest::Client::builder() failed; this should never happen");
        Self {
            row_name: row_name.into(),
            endpoint: cfg
                .endpoint
                .as_deref()
                .unwrap_or(DEFAULT_ENDPOINT)
                .trim_end_matches('/')
                .to_owned(),
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            category_thresholds: cfg.category_thresholds.clone(),
            hook_point,
            fail_open,
            output_fail_open: cfg.output_fail_open,
            timeout: Duration::from_millis(cfg.timeout_ms as u64),
            client: Arc::new(client),
        }
    }

    fn hook_enabled(&self, hook: GuardrailHookPoint) -> bool {
        self.hook_point == GuardrailHookPoint::Both || self.hook_point == hook
    }

    /// Check `text` against the Moderation API and map the result to a
    /// verdict per the decision rule in the module docs.
    async fn moderate(&self, text: &str, fail_open: bool) -> GuardrailVerdict {
        match self.call_api(text).await {
            Ok(resp) => self.evaluate(&resp),
            Err(failure) => self.handle_failure(failure, fail_open),
        }
    }

    async fn call_api(&self, input: &str) -> Result<ModerationResponse, ModerationFailure> {
        let url = format!("{}{}", self.endpoint, MODERATIONS_PATH);
        let body = ModerationRequest {
            model: &self.model,
            input,
        };

        let future = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send();

        let resp = match tokio::time::timeout(self.timeout, future).await {
            Err(_elapsed) => return Err(ModerationFailure::Timeout),
            Ok(Err(_e)) => return Err(ModerationFailure::IoError),
            Ok(Ok(r)) => r,
        };

        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ModerationFailure::Throttled);
        }
        if status.is_server_error() {
            return Err(ModerationFailure::ServerError);
        }
        if !status.is_success() {
            // 4xx other than 429 — almost always a misconfiguration
            // (bad api_key / endpoint / model). Error level: with
            // fail_open=true this silently bypasses the guardrail on every
            // request until the operator notices.
            tracing::error!(
                row = %self.row_name,
                http_status = status.as_u16(),
                "openai moderation returned 4xx — check endpoint, api_key, and model configuration",
            );
            return Err(ModerationFailure::ConfigError);
        }

        resp.json()
            .await
            .map_err(|_| ModerationFailure::ServerError)
    }

    fn evaluate(&self, resp: &ModerationResponse) -> GuardrailVerdict {
        let Some(result) = resp.results.first() else {
            return GuardrailVerdict::Allow;
        };

        if self.category_thresholds.is_empty() {
            // LiteLLM-baseline behavior: the API's own flagged boolean.
            if !result.flagged {
                return GuardrailVerdict::Allow;
            }
            let violated: Vec<&str> = result
                .categories
                .iter()
                .filter(|(_, &v)| v)
                .map(|(k, _)| k.as_str())
                .collect();
            return GuardrailVerdict::block(format!(
                "openai moderation flagged content ({}) (row: {})",
                violated.join(", "),
                self.row_name
            ));
        }

        // Threshold mode: only the configured categories are enforced.
        let over: Vec<String> = self
            .category_thresholds
            .iter()
            .filter_map(|(category, &threshold)| {
                let score = result.category_scores.get(category).copied()?;
                (score >= threshold).then(|| format!("{category}={score:.3}"))
            })
            .collect();
        if over.is_empty() {
            GuardrailVerdict::Allow
        } else {
            GuardrailVerdict::block(format!(
                "openai moderation category threshold exceeded ({}) (row: {})",
                over.join(", "),
                self.row_name
            ))
        }
    }

    fn handle_failure(&self, failure: ModerationFailure, fail_open: bool) -> GuardrailVerdict {
        let tag = failure.bypass_tag();
        // ConfigError is already logged at error level in call_api().
        if !matches!(failure, ModerationFailure::ConfigError) {
            tracing::warn!(
                row = %self.row_name,
                failure = ?failure,
                fail_open = fail_open,
                "openai moderation call failed",
            );
        }
        if fail_open {
            GuardrailVerdict::Bypass { reason: tag.into() }
        } else {
            GuardrailVerdict::block(format!("openai moderation unavailable ({tag})"))
        }
    }
}

/// Failure cause buckets. `bypass_tag()` maps to the strings stored in
/// `usage_events.guardrail_bypassed_reason` — changing them is a breaking
/// change for operators who filter on these values.
#[derive(Debug)]
enum ModerationFailure {
    Timeout,
    Throttled,
    IoError,
    ServerError,
    ConfigError,
}

impl ModerationFailure {
    fn bypass_tag(&self) -> &'static str {
        match self {
            Self::Timeout => "openai_moderation_timeout",
            Self::Throttled => "openai_moderation_throttled",
            Self::IoError | Self::ServerError => "openai_moderation_5xx",
            Self::ConfigError => "openai_moderation_config_error",
        }
    }
}

// --- serde shapes for the wire protocol ------------------------------------

#[derive(Serialize)]
struct ModerationRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct ModerationResponse {
    #[serde(default)]
    results: Vec<ModerationResult>,
}

#[derive(Deserialize)]
struct ModerationResult {
    #[serde(default)]
    flagged: bool,
    #[serde(default)]
    categories: BTreeMap<String, bool>,
    #[serde(default)]
    category_scores: BTreeMap<String, f64>,
}

// --- Guardrail trait impl --------------------------------------------------

#[async_trait]
impl Guardrail for OpenaiModerationGuardrail {
    fn name(&self) -> &'static str {
        "openai_moderation"
    }

    /// Its streamed-output hold-back policy applies only when it inspects
    /// output (#466); moderation is normally input-only, so it must not
    /// buffer the response unless attached on the output hook.
    fn runs_on_output(&self) -> bool {
        matches!(
            self.hook_point,
            GuardrailHookPoint::Output | GuardrailHookPoint::Both
        )
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        if !self.hook_enabled(GuardrailHookPoint::Input) {
            return GuardrailVerdict::Allow;
        }
        let text = collect_input_text(req);
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        self.moderate(&text, self.fail_open).await
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        if !self.hook_enabled(GuardrailHookPoint::Output) {
            return GuardrailVerdict::Allow;
        }
        let text = resp.guardrail_output_text();
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        self.moderate(&text, self.output_fail_open).await
    }
}

/// Concatenate all message contents into one blob for input scanning
/// (LiteLLM joins texts with `\n` for its single-string moderation call).
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
    use aisix_gateway::{ChatFormat, ChatMessage};
    use serde_json::json;
    use wiremock::matchers::{bearer_token, body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn cfg(endpoint: &str) -> OpenaiModerationConfig {
        OpenaiModerationConfig {
            api_key: "sk-test-key".to_owned(),
            endpoint: Some(endpoint.to_owned()),
            model: "omni-moderation-latest".to_owned(),
            category_thresholds: BTreeMap::new(),
            timeout_ms: 5_000,
            output_fail_open: false,
        }
    }

    fn build(endpoint: &str, fail_open: bool) -> OpenaiModerationGuardrail {
        OpenaiModerationGuardrail::new(
            "wiremock-test",
            &cfg(endpoint),
            GuardrailHookPoint::Both,
            fail_open,
        )
    }

    fn req(msg: &str) -> ChatFormat {
        ChatFormat::new("m", vec![ChatMessage::user(msg)])
    }

    fn flagged_response() -> serde_json::Value {
        json!({
            "id": "modr-1",
            "model": "omni-moderation-latest",
            "results": [{
                "flagged": true,
                "categories": { "violence": true, "hate": false },
                "category_scores": { "violence": 0.97, "hate": 0.01 }
            }]
        })
    }

    fn clean_response() -> serde_json::Value {
        json!({
            "id": "modr-2",
            "model": "omni-moderation-latest",
            "results": [{
                "flagged": false,
                "categories": { "violence": false },
                "category_scores": { "violence": 0.12 }
            }]
        })
    }

    #[tokio::test]
    async fn clean_input_allows() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .and(bearer_token("sk-test-key"))
            .and(body_partial_json(
                json!({ "model": "omni-moderation-latest" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(clean_response()))
            .expect(1)
            .mount(&server)
            .await;
        let g = build(&server.uri(), false);
        assert_eq!(g.check_input(&req("hello")).await, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn flagged_blocks_with_category_names() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(flagged_response()))
            .mount(&server)
            .await;
        let g = build(&server.uri(), false);
        match g.check_input(&req("violent text")).await {
            GuardrailVerdict::Block { reason, .. } => {
                assert!(reason.contains("violence"), "reason: {reason}");
                assert!(
                    !reason.contains("hate"),
                    "unflagged category leaked: {reason}"
                );
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn threshold_mode_overrides_flagged_boolean() {
        // score 0.4: flagged=false from the API, but the operator set a
        // 0.3 threshold — threshold mode blocks anyway.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{
                    "flagged": false,
                    "categories": { "violence": false },
                    "category_scores": { "violence": 0.4, "hate": 0.9 }
                }]
            })))
            .mount(&server)
            .await;
        let mut c = cfg(&server.uri());
        c.category_thresholds.insert("violence".into(), 0.3);
        let g = OpenaiModerationGuardrail::new("t", &c, GuardrailHookPoint::Both, false);
        match g.check_input(&req("x")).await {
            GuardrailVerdict::Block { reason, .. } => {
                assert!(reason.contains("violence=0.400"), "reason: {reason}");
                // hate scored 0.9 but is NOT configured — not enforced.
                assert!(
                    !reason.contains("hate"),
                    "unconfigured category enforced: {reason}"
                );
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn threshold_mode_under_threshold_allows_even_when_flagged() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{
                    "flagged": true,
                    "categories": { "violence": true },
                    "category_scores": { "violence": 0.4 }
                }]
            })))
            .mount(&server)
            .await;
        let mut c = cfg(&server.uri());
        c.category_thresholds.insert("violence".into(), 0.8);
        let g = OpenaiModerationGuardrail::new("t", &c, GuardrailHookPoint::Both, false);
        assert_eq!(g.check_input(&req("x")).await, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn five_xx_fail_open_bypasses_fail_closed_blocks() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let open = build(&server.uri(), true);
        assert_eq!(
            open.check_input(&req("x")).await,
            GuardrailVerdict::Bypass {
                reason: "openai_moderation_5xx".into()
            }
        );
        let closed = build(&server.uri(), false);
        assert!(closed.check_input(&req("x")).await.is_block());
    }

    #[tokio::test]
    async fn config_error_4xx_tagged_separately() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;
        let g = build(&server.uri(), true);
        assert_eq!(
            g.check_input(&req("x")).await,
            GuardrailVerdict::Bypass {
                reason: "openai_moderation_config_error".into()
            }
        );
    }

    #[tokio::test]
    async fn output_hook_uses_its_own_fail_policy() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        // fail_open=true on input, output_fail_open=false (default) — an
        // outage must still block the OUTPUT hook.
        let g = build(&server.uri(), true);
        let resp = aisix_gateway::ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: ChatMessage::assistant("model output"),
            finish_reason: aisix_gateway::FinishReason::Stop,
            usage: aisix_gateway::UsageStats::new(0, 0),
        };
        assert!(g.check_output(&resp).await.is_block());
    }

    #[tokio::test]
    async fn input_only_hook_skips_output() {
        let server = MockServer::start().await;
        let g = OpenaiModerationGuardrail::new(
            "t",
            &cfg(&server.uri()),
            GuardrailHookPoint::Input,
            false,
        );
        assert!(!g.runs_on_output());
    }
}
