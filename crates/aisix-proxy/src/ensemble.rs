//! Ensemble executor — fan a chat request out to a panel of models in
//! parallel, then synthesize one answer via a judge model.
//!
//! This module is the pure orchestration core. It is parameterized over a
//! [`ModelCaller`] so the fan-out / min-responses / synthesis logic is
//! unit-testable without real bridges or a network. The proxy dispatch
//! layer (`chat.rs`) supplies the production caller that resolves a model
//! `display_name` to its Bridge + ProviderKey and emits per-sub-call usage.
//!
//! Shape of one ensemble request:
//! 1. Build a per-member [`ChatFormat`] (the member's `temperature`/`seed`
//!    override the request's, which is what makes self-ensemble produce
//!    diverse answers) and dispatch every member concurrently.
//! 2. Keep the successful responses; require at least
//!    [`EnsembleConfig::min_responses_or_default`] of them, else error.
//! 3. Ask the judge model to synthesize a single answer from the labeled
//!    candidate answers, at a fixed low temperature. Retry the judge once
//!    on a transient failure.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::future::join_all;

use aisix_core::models::{EnsembleConfig, Judge, PanelMember};
use aisix_core::AisixSnapshot;
use aisix_gateway::bridge::{BridgeContext, BridgeError};
use aisix_gateway::chat::{ChatFormat, ChatMessage, ChatResponse, Role, UsageStats};

use crate::state::ProxyState;

/// Fixed judge sampling temperature. The judge runs cool for stable
/// synthesis; this is intentionally not operator-configurable in v1.
const JUDGE_TEMPERATURE: f32 = 0.2;

/// Per-candidate text cap (bytes) fed to the judge. Bounds the synthesis
/// prompt so a panel of N long answers cannot overflow the judge's
/// context window; oversized answers are truncated, not dropped.
const MAX_CANDIDATE_BYTES: usize = 8 * 1024;

/// Default synthesis instructions. Filled with the original request and
/// the neutrally-labeled candidate answers, then sent to the judge as a
/// single user message. Operators can override the whole template via
/// `Judge.synthesis_prompt` (it must carry the two `{...}` placeholders).
const DEFAULT_SYNTHESIS_TEMPLATE: &str = "You are the synthesis step of a multi-model answer pipeline. You are given the user's original request and several independent candidate answers to it.

Your job: produce the single best possible answer to the user's ORIGINAL request.

Use the candidates as evidence, not as text to copy:
- Identify where they agree — treat consensus as likely reliable.
- Identify where they contradict — resolve it by reasoning about which is correct and why; do not just take the majority.
- Keep unique, correct, well-supported insights that only one candidate raised.
- Discard claims that are unsupported, internally inconsistent, or look hallucinated, even if several candidates share them.

Output rules (strict):
- Respond ONLY with the final answer to the user's request — no meta-commentary.
- Do NOT mention that multiple models, candidates, or a panel were involved.
- Do NOT name or attribute any candidate (\"Answer 1 says...\", model names, etc.).
- Write in the SAME language as the user's original request.
- Match the format the user asked for (code, list, prose, length).

User's original request:
{original_request}

Candidate answers:
{labeled_candidates}";

/// Dispatches one resolved chat request to a model by its `display_name`.
/// The production impl resolves the name to a Bridge + ProviderKey and
/// calls `Bridge::chat`; tests supply a scripted mock.
#[async_trait]
pub trait ModelCaller: Send + Sync {
    async fn call(&self, target: &str, req: &ChatFormat) -> Result<ChatResponse, BridgeError>;
}

/// Production [`ModelCaller`] used by the chat dispatch layer. Resolves
/// a panel/judge member's `display_name` to its Bridge + ProviderKey
/// from the live snapshot and dispatches a single non-streaming
/// `Bridge::chat`. Borrows everything it needs for the duration of one
/// ensemble run, so it holds no owned state of its own.
pub(crate) struct ProxyModelCaller<'a> {
    pub state: &'a ProxyState,
    pub snapshot: &'a AisixSnapshot,
    pub request_id: &'a str,
}

#[async_trait]
impl ModelCaller for ProxyModelCaller<'_> {
    async fn call(&self, target: &str, req: &ChatFormat) -> Result<ChatResponse, BridgeError> {
        // Resolve the member's display_name against the live snapshot.
        // A missing entry is a misconfigured ensemble (a panel/judge name
        // that no longer points at a real Model) → 400 InvalidUpstreamConfig.
        // The member's display_name is operator-internal config; keep it in
        // server logs but out of the client-visible `error.message` (a
        // misconfigured judge surfaces its bridge error to the caller).
        let entry = self.snapshot.models.get_by_name(target).ok_or_else(|| {
            tracing::warn!(member = %target, "ensemble member references an unknown model");
            BridgeError::InvalidUpstreamConfig(
                "ensemble member references an unknown model".to_string(),
            )
        })?;
        let model = &entry.value;

        // ProviderKey + Bridge resolution mirrors the direct-dispatch path.
        // `resolve_provider_key` yields a `ProxyError` whose message embeds
        // the member's `display_name` + `provider_key_id`. For a misconfigured
        // judge that error reaches the client envelope verbatim (judge err →
        // EnsembleError::Judge → ProxyError::Bridge → transparent), so redact
        // it the same way as the get_by_name path: detail to server logs only.
        let pk_entry = crate::dispatch::resolve_provider_key(self.snapshot, model).map_err(|e| {
            tracing::warn!(member = %target, error = %e, "ensemble member provider key unresolved");
            BridgeError::InvalidUpstreamConfig(
                "ensemble member has an unresolved provider key".to_string(),
            )
        })?;
        let bridge =
            crate::dispatch::resolve_bridge(&self.state.hub, &pk_entry.value).ok_or_else(|| {
                tracing::warn!(member = %target, "ensemble member has no registered bridge");
                BridgeError::Config("ensemble member has no registered bridge".to_string())
            })?;

        // Per-member request deadline from the member Model's own `timeout`.
        let mut ctx = BridgeContext::new(
            self.request_id,
            Arc::new(model.clone()),
            Arc::new(pk_entry.value.clone()),
        );
        if let Some(deadline) = model.request_timeout() {
            ctx = ctx.with_deadline(deadline);
        }

        // Enforce THIS member's own model rate limit before the sub-call and
        // bill its own tokens after (#620). The request-level layers (api_key
        // / team / member) are already reserved on the entry alias; here we add
        // only the member's `model:` layer + model-scope policies. A member
        // that exceeds its own limit becomes a failed sub-call: the panel drops
        // it toward `min_responses`, and the judge surfaces it as a 429 judge
        // failure. An unlimited member reserves nothing (zero overhead).
        let reservation = crate::quota::reserve_model_only(self.state, target, &entry.id, model)
            .await
            .map_err(|_| {
                BridgeError::upstream_status(429, "rate limit exceeded for an ensemble sub-call")
            })?;

        // On a bridge error the reservation drops here → concurrency slots
        // release and no tokens are counted. On success we commit the member's
        // own token cost to its model bucket.
        let response = bridge.chat(req, &ctx).await?;
        reservation
            .commit_tokens(u64::from(response.usage.total_tokens))
            .await;
        Ok(response)
    }
}

/// Per successful panel member: which model answered and what it cost.
/// Consumed by the dispatch layer to emit one usage event per sub-call.
#[derive(Debug)]
pub struct PanelOutcome {
    pub model: String,
    pub usage: UsageStats,
    /// The member's answer text (content + reasoning + tool-call text),
    /// captured so the dispatch layer can estimate this sub-call's
    /// completion tokens when the member backend reports no usage
    /// (AISIX-Cloud#1074). Never billed directly — only a fallback.
    pub est_output_text: String,
}

/// Everything the dispatch layer needs after an ensemble run: the judge's
/// response (the client-facing answer, whose `usage` is the judge call's
/// own), plus the per-sub-call accounting for telemetry and the aggregate
/// client-facing usage.
#[derive(Debug)]
pub struct EnsembleOutcome {
    pub response: ChatResponse,
    pub panel: Vec<PanelOutcome>,
    pub judge_model: String,
    /// The judge's synthesis request, kept so the dispatch layer can
    /// estimate the judge sub-call's prompt tokens when the judge backend
    /// reports no usage (AISIX-Cloud#1074). The streaming path builds its
    /// own judge estimator from `run_ensemble_panel`'s `judge_req`; this
    /// field carries the same request out of the buffered path.
    pub judge_req: ChatFormat,
}

#[derive(Debug, thiserror::Error)]
pub enum EnsembleError {
    #[error("ensemble panel produced {got} response(s), need at least {needed}")]
    InsufficientPanel {
        got: usize,
        needed: usize,
        /// The panel members that DID succeed before the run was abandoned.
        /// Each one already hit an upstream and was billed, so the dispatch
        /// layer must still commit + emit their usage even though the request
        /// fails — dropping them would under-report panel usage to cp-api.
        panel: Vec<PanelOutcome>,
    },
    #[error("ensemble judge call failed")]
    Judge {
        #[source]
        source: BridgeError,
        /// The panel members that DID succeed before the judge failed. The
        /// full panel met `min_responses` and was billed, so the dispatch
        /// layer must still commit + emit their usage on this exit too —
        /// same invariant as `InsufficientPanel`.
        panel: Vec<PanelOutcome>,
    },
}

impl EnsembleError {
    /// Client-facing HTTP status. An exhausted panel is an upstream fault
    /// (502); a judge failure preserves the bridge's own mapping, so a
    /// customer-fixable 4xx (bad judge config/credentials) stays 4xx while
    /// upstream 5xx collapses to 502.
    pub fn http_status(&self) -> u16 {
        match self {
            EnsembleError::InsufficientPanel { .. } => 502,
            EnsembleError::Judge { source, .. } => source.http_status(),
        }
    }
}

/// Run only the panel phase of an ensemble request: fan out to every
/// panel member concurrently (phase 1), enforce `min_responses` (phase 2),
/// and build the judge's synthesis request from the labeled candidates.
/// The judge itself is NOT called — that is left to the caller so the
/// streaming dispatch path can stream the judge's tokens while the
/// non-streaming path ([`run_ensemble`]) buffers them.
///
/// Returns the per-member accounting (`panel`), the raw candidate
/// responses (`candidates`, kept so the caller can re-derive context if
/// needed), and the ready-to-dispatch `judge_req`. The returned
/// `judge_req` is non-streaming (`stream = Some(false)`); a streaming
/// caller flips it to `Some(true)` before dispatching.
///
/// An exhausted panel returns [`EnsembleError::InsufficientPanel`] with
/// the already-billed survivors, exactly as before — the dispatch layer
/// must still commit + emit their usage.
pub(crate) async fn run_ensemble_panel(
    req: &ChatFormat,
    config: &EnsembleConfig,
    caller: &dyn ModelCaller,
) -> Result<(Vec<PanelOutcome>, Vec<ChatResponse>, ChatFormat), EnsembleError> {
    let per_call_timeout = config.timeout();

    // Phase 1: fan out to every panel member concurrently.
    let calls = config.panel.iter().map(|member| {
        let model = member.model.clone();
        let member_req = panel_request(req, member);
        async move {
            let result =
                call_with_optional_timeout(caller, &model, &member_req, per_call_timeout).await;
            (model, result)
        }
    });
    let results = join_all(calls).await;

    // Phase 2: keep the successes, enforce min_responses.
    let mut panel: Vec<PanelOutcome> = Vec::new();
    let mut candidates: Vec<ChatResponse> = Vec::new();
    for (model, result) in results {
        match result {
            Ok(resp) => {
                panel.push(PanelOutcome {
                    model,
                    usage: resp.usage.clone(),
                    est_output_text: crate::chat::estimation_output_text(&resp),
                });
                candidates.push(resp);
            }
            Err(err) => {
                tracing::warn!(panel_model = %model, error = %err, "ensemble panel member failed");
            }
        }
    }
    let needed = config.min_responses_or_default();
    if candidates.len() < needed {
        // Carry the survivors out on the error: they were already billed,
        // so the dispatch layer commits + emits their usage before failing.
        return Err(EnsembleError::InsufficientPanel {
            got: candidates.len(),
            needed,
            panel,
        });
    }

    // Build the judge's synthesis request from the labeled candidates.
    let judge_req = judge_request(req, &config.judge, &candidates);
    Ok((panel, candidates, judge_req))
}

/// Run one ensemble request to completion. See the module docs for the
/// three phases.
pub async fn run_ensemble(
    req: &ChatFormat,
    config: &EnsembleConfig,
    caller: &dyn ModelCaller,
) -> Result<EnsembleOutcome, EnsembleError> {
    // Phases 1-2 + judge-request construction.
    let (panel, _candidates, judge_req) = run_ensemble_panel(req, config, caller).await?;

    // Phase 3: synthesize via the judge, retrying once on a transient error.
    // The judge call gets the same per-call timeout as each panel member
    // (`config.timeout()`), so the ensemble-level deadline applies uniformly
    // across the whole fan-out.
    let response = match call_judge_with_retry(
        caller,
        &config.judge.model,
        &judge_req,
        config.timeout(),
    )
    .await
    {
        Ok(r) => r,
        // Carry the (already-billed) panel survivors out on the judge
        // failure so the dispatch layer commits + emits them too.
        Err(source) => return Err(EnsembleError::Judge { source, panel }),
    };

    Ok(EnsembleOutcome {
        response,
        panel,
        judge_model: config.judge.model.clone(),
        judge_req,
    })
}

/// Build a panel member's request: the inbound request with this member's
/// sampling overrides applied. Always non-streaming — the executor needs
/// the full response to synthesize.
fn panel_request(req: &ChatFormat, member: &PanelMember) -> ChatFormat {
    let mut out = req.clone();
    out.model = member.model.clone();
    if let Some(temperature) = member.temperature {
        out.temperature = Some(temperature);
    }
    if let Some(seed) = member.seed {
        out.extra
            .insert("seed".to_string(), serde_json::Value::from(seed));
    }
    out.stream = Some(false);
    out
}

/// Build the judge's request from the labeled candidate answers.
fn judge_request(req: &ChatFormat, judge: &Judge, candidates: &[ChatResponse]) -> ChatFormat {
    let template = judge
        .synthesis_prompt
        .as_deref()
        .unwrap_or(DEFAULT_SYNTHESIS_TEMPLATE);
    let prompt = template
        .replace(
            "{original_request}",
            &render_original_request(&req.messages),
        )
        .replace("{labeled_candidates}", &label_candidates(candidates));

    let mut out = ChatFormat::new(judge.model.clone(), vec![ChatMessage::user(prompt)]);
    out.temperature = Some(JUDGE_TEMPERATURE);
    out.stream = Some(false);
    out
}

/// Render the original conversation as plain text for the judge prompt.
fn render_original_request(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("{}: {}", role_label(m.role), m.content_str()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Present the candidates to the judge with neutral labels. Model names
/// are deliberately omitted so the operator's provider/model choices never
/// leak to the client through the synthesized answer.
fn label_candidates(candidates: &[ChatResponse]) -> String {
    candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "Answer {}:\n{}",
                i + 1,
                truncate_bytes(c.message.content_str(), MAX_CANDIDATE_BYTES)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::System => "System",
        Role::User => "User",
        Role::Assistant => "Assistant",
        Role::Tool => "Tool",
    }
}

/// Truncate to at most `max` bytes on a UTF-8 boundary, marking the cut.
fn truncate_bytes(s: &str, max: usize) -> Cow<'_, str> {
    if s.len() <= max {
        return Cow::Borrowed(s);
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    Cow::Owned(format!("{}…", &s[..end]))
}

async fn call_with_optional_timeout(
    caller: &dyn ModelCaller,
    target: &str,
    req: &ChatFormat,
    timeout: Option<Duration>,
) -> Result<ChatResponse, BridgeError> {
    match timeout {
        Some(d) => match tokio::time::timeout(d, caller.call(target, req)).await {
            Ok(result) => result,
            Err(_) => Err(BridgeError::Timeout {
                elapsed_ms: d.as_millis() as u64,
                cause: String::new(),
            }),
        },
        None => caller.call(target, req).await,
    }
}

/// Call the judge, retrying once if the first failure is transient
/// (resolved decision: judge failure → 5xx after one in-process retry; no
/// silent fallback to a raw panel answer). Each attempt is bound by the
/// same per-call `timeout` as the panel members; a timed-out judge call
/// surfaces as a transient [`BridgeError::Timeout`] and so is retried once.
async fn call_judge_with_retry(
    caller: &dyn ModelCaller,
    target: &str,
    req: &ChatFormat,
    timeout: Option<Duration>,
) -> Result<ChatResponse, BridgeError> {
    match call_with_optional_timeout(caller, target, req, timeout).await {
        Ok(resp) => Ok(resp),
        Err(first) if is_transient(&first) => {
            call_with_optional_timeout(caller, target, req, timeout).await
        }
        Err(err) => Err(err),
    }
}

/// Conservative transient-failure classification for the judge retry:
/// timeouts, transport faults, mid-stream aborts, and upstream 5xx are
/// retryable; everything else (4xx, config, credentials, decode) is not.
fn is_transient(err: &BridgeError) -> bool {
    match err {
        BridgeError::Timeout { .. } | BridgeError::Transport(_) | BridgeError::StreamAborted => {
            true
        }
        BridgeError::UpstreamStatus { status, .. } => *status >= 500,
        BridgeError::UpstreamDecode(_)
        | BridgeError::Config(_)
        | BridgeError::InvalidUpstreamConfig(_)
        | BridgeError::InvalidUpstreamCredentials(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Mutex;

    use aisix_gateway::chat::FinishReason;

    /// A scripted [`ModelCaller`]: each target has a queue of results
    /// returned on successive calls. Records every call for assertions.
    struct MockCaller {
        scripted: Mutex<HashMap<String, VecDeque<Result<ChatResponse, BridgeError>>>>,
        calls: Mutex<Vec<(String, ChatFormat)>>,
    }

    impl MockCaller {
        fn new() -> Self {
            Self {
                scripted: Mutex::new(HashMap::new()),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn on(self, target: &str, results: Vec<Result<ChatResponse, BridgeError>>) -> Self {
            self.scripted
                .lock()
                .unwrap()
                .insert(target.to_string(), results.into_iter().collect());
            self
        }

        fn calls_to(&self, target: &str) -> Vec<ChatFormat> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .filter(|(t, _)| t == target)
                .map(|(_, req)| req.clone())
                .collect()
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl ModelCaller for MockCaller {
        async fn call(&self, target: &str, req: &ChatFormat) -> Result<ChatResponse, BridgeError> {
            self.calls
                .lock()
                .unwrap()
                .push((target.to_string(), req.clone()));
            let mut scripted = self.scripted.lock().unwrap();
            let queue = scripted
                .get_mut(target)
                .unwrap_or_else(|| panic!("no scripted response for target {target:?}"));
            queue
                .pop_front()
                .unwrap_or_else(|| panic!("scripted responses exhausted for target {target:?}"))
        }
    }

    fn ok(model: &str, content: &str) -> Result<ChatResponse, BridgeError> {
        Ok(ChatResponse {
            id: format!("id-{model}"),
            model: model.to_string(),
            message: ChatMessage::assistant(content),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(10, 20),
        })
    }

    fn config(json: &str) -> EnsembleConfig {
        serde_json::from_str(json).unwrap()
    }

    fn user_request(text: &str) -> ChatFormat {
        ChatFormat::new("council", vec![ChatMessage::user(text)])
    }

    #[tokio::test]
    async fn happy_path_fans_out_then_synthesizes() {
        let caller = MockCaller::new()
            .on("gpt", vec![ok("gpt", "answer from gpt")])
            .on("claude", vec![ok("claude", "answer from claude")])
            .on("judge", vec![ok("judge", "synthesized answer")]);
        let cfg =
            config(r#"{"panel":[{"model":"gpt"},{"model":"claude"}],"judge":{"model":"judge"}}"#);

        let out = run_ensemble(&user_request("what is 2+2?"), &cfg, &caller)
            .await
            .unwrap();

        assert_eq!(out.response.message.content_str(), "synthesized answer");
        assert_eq!(out.panel.len(), 2);
        assert_eq!(out.judge_model, "judge");
        assert_eq!(caller.call_count(), 3); // 2 panel + 1 judge

        // The judge saw both candidates, neutrally labeled, plus the request.
        let judge_prompt = caller.calls_to("judge")[0].messages[0]
            .content_str()
            .to_string();
        assert!(judge_prompt.contains("Answer 1:"));
        assert!(judge_prompt.contains("Answer 2:"));
        assert!(judge_prompt.contains("answer from gpt"));
        assert!(judge_prompt.contains("answer from claude"));
        assert!(judge_prompt.contains("what is 2+2?"));
    }

    #[tokio::test]
    async fn errors_when_min_responses_not_met() {
        let caller = MockCaller::new()
            .on("gpt", vec![ok("gpt", "only survivor")])
            .on(
                "claude",
                vec![Err(BridgeError::upstream_status(503, "busy"))],
            )
            .on("judge", vec![ok("judge", "should not be called")]);
        // Two-member panel defaults to min_responses = 2; only one succeeds.
        let cfg =
            config(r#"{"panel":[{"model":"gpt"},{"model":"claude"}],"judge":{"model":"judge"}}"#);

        let err = run_ensemble(&user_request("hi"), &cfg, &caller)
            .await
            .unwrap_err();

        match err {
            EnsembleError::InsufficientPanel { got, needed, panel } => {
                assert_eq!(got, 1);
                assert_eq!(needed, 2);
                // The one survivor (gpt) is carried out so the dispatch
                // layer can still commit + emit its (already-billed) usage.
                assert_eq!(panel.len(), 1);
                assert_eq!(panel[0].model, "gpt");
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(caller.calls_to("judge").is_empty());
    }

    #[tokio::test]
    async fn proceeds_on_partial_panel_when_min_met() {
        // 3-member panel, one fails. Default min_responses = 2, so the two
        // survivors are enough to synthesize.
        let caller = MockCaller::new()
            .on("a", vec![ok("a", "alpha")])
            .on("b", vec![Err(BridgeError::Transport("reset".into()))])
            .on("c", vec![ok("c", "gamma")])
            .on("judge", vec![ok("judge", "merged")]);
        let cfg = config(
            r#"{"panel":[{"model":"a"},{"model":"b"},{"model":"c"}],"judge":{"model":"judge"}}"#,
        );

        let out = run_ensemble(&user_request("hi"), &cfg, &caller)
            .await
            .unwrap();

        assert_eq!(out.panel.len(), 2);
        let judge_prompt = caller.calls_to("judge")[0].messages[0]
            .content_str()
            .to_string();
        assert!(judge_prompt.contains("alpha"));
        assert!(judge_prompt.contains("gamma"));
        assert!(!judge_prompt.contains("reset"));
    }

    #[tokio::test]
    async fn retries_judge_once_on_transient_failure() {
        let caller = MockCaller::new()
            .on("a", vec![ok("a", "x")])
            .on("b", vec![ok("b", "y")])
            .on(
                "judge",
                vec![
                    Err(BridgeError::Timeout {
                        cause: String::new(),
                        elapsed_ms: 1,
                    }),
                    ok("judge", "second attempt wins"),
                ],
            );
        let cfg = config(r#"{"panel":[{"model":"a"},{"model":"b"}],"judge":{"model":"judge"}}"#);

        let out = run_ensemble(&user_request("hi"), &cfg, &caller)
            .await
            .unwrap();

        assert_eq!(out.response.message.content_str(), "second attempt wins");
        assert_eq!(caller.calls_to("judge").len(), 2); // retried once
    }

    #[tokio::test]
    async fn does_not_retry_judge_on_non_transient_failure() {
        let caller = MockCaller::new()
            .on("a", vec![ok("a", "x")])
            .on("b", vec![ok("b", "y")])
            .on(
                "judge",
                vec![Err(BridgeError::InvalidUpstreamConfig("bad model".into()))],
            );
        let cfg = config(r#"{"panel":[{"model":"a"},{"model":"b"}],"judge":{"model":"judge"}}"#);

        let err = run_ensemble(&user_request("hi"), &cfg, &caller)
            .await
            .unwrap_err();

        assert_eq!(err.http_status(), 400); // judge 4xx preserved
        assert_eq!(caller.calls_to("judge").len(), 1); // not retried
                                                       // The full panel succeeded and was billed before the judge failed,
                                                       // so the survivors are carried out on the error for the dispatch
                                                       // layer to commit + emit (FIX G).
        match err {
            EnsembleError::Judge { source, panel } => {
                assert_eq!(source.http_status(), 400);
                assert_eq!(panel.len(), 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn applies_per_member_temperature_and_judge_runs_cool() {
        let caller = MockCaller::new()
            .on("hot", vec![ok("hot", "h")])
            .on("inherit", vec![ok("inherit", "i")])
            .on("judge", vec![ok("judge", "done")]);
        // `hot` overrides temperature; `inherit` has none so it keeps the
        // request's temperature.
        let cfg = config(
            r#"{"panel":[{"model":"hot","temperature":1.0},{"model":"inherit"}],"judge":{"model":"judge"}}"#,
        );
        let mut req = user_request("hi");
        req.temperature = Some(0.2);

        run_ensemble(&req, &cfg, &caller).await.unwrap();

        assert_eq!(caller.calls_to("hot")[0].temperature, Some(1.0));
        assert_eq!(caller.calls_to("inherit")[0].temperature, Some(0.2));
        assert_eq!(
            caller.calls_to("judge")[0].temperature,
            Some(JUDGE_TEMPERATURE)
        );
    }

    #[tokio::test]
    async fn judge_prompt_does_not_leak_panel_model_names() {
        let caller = MockCaller::new()
            .on("secret-vendor-x", vec![ok("secret-vendor-x", "ans one")])
            .on("secret-vendor-y", vec![ok("secret-vendor-y", "ans two")])
            .on("judge", vec![ok("judge", "final")]);
        let cfg = config(
            r#"{"panel":[{"model":"secret-vendor-x"},{"model":"secret-vendor-y"}],"judge":{"model":"judge"}}"#,
        );

        run_ensemble(&user_request("hi"), &cfg, &caller)
            .await
            .unwrap();

        let judge_prompt = caller.calls_to("judge")[0].messages[0]
            .content_str()
            .to_string();
        assert!(!judge_prompt.contains("secret-vendor-x"));
        assert!(!judge_prompt.contains("secret-vendor-y"));
    }

    #[tokio::test]
    async fn custom_synthesis_prompt_overrides_default() {
        let caller = MockCaller::new()
            .on("a", vec![ok("a", "alpha")])
            .on("b", vec![ok("b", "beta")])
            .on("judge", vec![ok("judge", "done")]);
        let cfg = config(
            r#"{"panel":[{"model":"a"},{"model":"b"}],"judge":{"model":"judge","synthesis_prompt":"PICK BEST. Q={original_request} A={labeled_candidates}"}}"#,
        );

        run_ensemble(&user_request("which?"), &cfg, &caller)
            .await
            .unwrap();

        let judge_prompt = caller.calls_to("judge")[0].messages[0]
            .content_str()
            .to_string();
        assert!(judge_prompt.starts_with("PICK BEST."));
        assert!(judge_prompt.contains("which?"));
        assert!(judge_prompt.contains("alpha"));
    }
}
