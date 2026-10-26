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
//!    candidate answers, at a fixed low temperature.
//!
//! Retrying a transient sub-call failure — panel member or judge alike — is
//! the [`ModelCaller`]'s job, using that member model's own retry budget.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::future::join_all;

use aisix_core::models::{EnsembleConfig, Judge, PanelMember};
use aisix_core::AisixSnapshot;
use aisix_gateway::bridge::BridgeError;
use aisix_gateway::chat::{ChatFormat, ChatMessage, ChatResponse, Role, UsageStats};

use crate::state::ProxyState;

/// Fixed judge sampling temperature. The judge runs cool for stable
/// synthesis; this is intentionally not operator-configurable in v1.
const JUDGE_TEMPERATURE: f64 = 0.2;

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
#[derive(Debug)]
pub struct ModelCallOutcome {
    pub response: ChatResponse,
    pub effective_usage: UsageStats,
    pub usage_estimated: bool,
}

/// A failed sub-call: the bridge error the caller is answered with, and
/// whether the gateway refused the call on the model's own rate limit before
/// dispatching it.
#[derive(Debug)]
pub struct CallError {
    pub source: BridgeError,
    pub rate_limited: bool,
}

impl From<BridgeError> for CallError {
    fn from(source: BridgeError) -> Self {
        Self {
            source,
            rate_limited: false,
        }
    }
}

#[async_trait]
pub trait ModelCaller: Send + Sync {
    /// `req` builds the sub-call's request. It is called once per upstream
    /// attempt and the result handed to the bridge by value, so a panel
    /// member's copy of the caller's request is not held across the
    /// upstream wait — the member's wire body is its only copy then.
    async fn call(
        &self,
        target: &str,
        req: &RequestBuilder<'_>,
    ) -> Result<ModelCallOutcome, CallError>;
}

/// Builds one sub-call's request on demand; see [`ModelCaller::call`].
pub type RequestBuilder<'a> = dyn Fn() -> ChatFormat + Send + Sync + 'a;

/// Production [`ModelCaller`] used by the chat dispatch layer. Resolves
/// a panel/judge member's `display_name` to its Bridge + ProviderKey
/// from the live snapshot and dispatches a single non-streaming
/// `Bridge::chat`. Borrows everything it needs for the duration of one
/// ensemble run, so it holds no owned state of its own.
pub(crate) struct ProxyModelCaller<'a> {
    pub state: &'a ProxyState,
    /// Caller identity — the per-member quota gate needs the identity
    /// dimensions for conditional policy rows (AISIX-Cloud#892).
    pub auth: &'a crate::auth::AuthenticatedKey,
    pub snapshot: &'a AisixSnapshot,
    pub request_id: &'a str,
    /// The originating request's context. Member calls are dispatched on
    /// the caller's behalf, so they carry the same caller identity and
    /// forwardable client headers as a single-upstream dispatch would.
    pub client: &'a crate::client_ip::ClientContext,
    /// The ensemble entry the caller addressed, so per-member policy
    /// matching sees the {member, parent} pair (AISIX-Cloud#1267).
    pub routing_parent: crate::quota::RoutingParent<'a>,
}

#[async_trait]
impl ModelCaller for ProxyModelCaller<'_> {
    async fn call(
        &self,
        target: &str,
        req: &RequestBuilder<'_>,
    ) -> Result<ModelCallOutcome, CallError> {
        // Members run concurrently, outside the attempt windows the
        // upstream clock and the response-header tap key on; say so rather
        // than report their time as the gateway's.
        crate::attribution::note_upstream_wait_unmeasured();
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
        let mut ctx = crate::dispatch::bridge_ctx(
            self.request_id,
            &entry.id,
            Arc::new(model.clone()),
            &pk_entry.id,
            Arc::new(pk_entry.value.clone()),
            Some(self.client),
        );
        if let Some(deadline) =
            crate::routing::effective_timeouts(model, None, self.state.default_timeouts).request
        {
            ctx = ctx.with_deadline(deadline);
        }

        // Enforce THIS member's own model rate limit before the sub-call and
        // bill its own tokens after (#620). The request-level layers (api_key
        // / team / member) are already reserved on the entry alias; here we add
        // only the member's `model:` layer + model-scope policies. A member
        // that exceeds its own limit becomes a failed sub-call: the panel drops
        // it toward `min_responses`, and the judge surfaces it as a 429 judge
        // failure. An unlimited member reserves nothing (zero overhead).
        let reservation = crate::quota::reserve_model_only(
            self.state,
            self.snapshot,
            self.auth,
            target,
            &entry.id,
            model,
            Some(self.routing_parent),
        )
        .await
        .map_err(|e| {
            // Client-visible message stays generic; the cause
            // (which layer / which policy fired) goes to the
            // logs so an operator can attribute the throttled
            // member.
            tracing::warn!(member = %target, error = %e, "ensemble sub-call rate limited");
            CallError {
                source: BridgeError::upstream_status(
                    429,
                    "rate limit exceeded for an ensemble sub-call",
                ),
                rate_limited: true,
            }
        })?;

        // On a bridge error the reservation drops here → concurrency slots
        // release and no tokens are counted. On success we commit the member's
        // own token cost to its model bucket.
        //
        // Retries use the MEMBER model's own budget, so a flaky panel member
        // is absorbed instead of silently shrinking the panel toward
        // `min_responses`. The reservation is deliberately outside the retry
        // loop (one sub-call bills once, however many attempts it took), and
        // `ctx`'s deadline is per attempt while the ensemble's own
        // `config.timeout()` — applied by the caller — remains the ceiling on
        // the whole attempt sequence.
        let response = crate::routing::retrying_dispatch(self.state, model, "ensemble", || {
            let attempt_req = req();
            let upstream_req = match crate::effort_mapping::chat_request(&attempt_req, model) {
                std::borrow::Cow::Owned(mapped) => mapped,
                std::borrow::Cow::Borrowed(_) => attempt_req,
            };
            bridge.chat_cow(std::borrow::Cow::Owned(upstream_req), &ctx)
        })
        .await?;
        let effective = crate::chat::effective_subcall_usage(
            req,
            model.upstream_model().unwrap_or(target),
            &response.usage,
            &crate::chat::estimation_output_text(&response),
        );
        reservation
            .commit_tokens(u64::from(effective.usage.total_tokens))
            .await;
        Ok(ModelCallOutcome {
            response,
            effective_usage: effective.usage,
            usage_estimated: effective.estimated,
        })
    }
}

/// Per successful panel member: which model answered and what it cost.
/// Consumed by the dispatch layer to emit one usage event per sub-call.
#[derive(Debug)]
pub struct PanelOutcome {
    /// The member's slot in the configured panel — its row's `attempt_index`.
    pub slot: u32,
    pub model: String,
    /// Provider-reported usage retained for the client-facing aggregate.
    pub usage: UsageStats,
    /// Final accounting shared by member quota, parent quota, telemetry, and
    /// request-level metrics. Local estimates never replace `usage` above.
    pub effective_usage: UsageStats,
    pub usage_estimated: bool,
    /// The tier the member's upstream reported it served on.
    pub service_tier: Option<String>,
}

impl PanelOutcome {
    fn of(slot: u32, model: &str, outcome: &ModelCallOutcome) -> Self {
        Self {
            slot,
            model: model.to_string(),
            usage: outcome.response.usage.clone(),
            effective_usage: outcome.effective_usage.clone(),
            usage_estimated: outcome.usage_estimated,
            service_tier: tier_value(&outcome.response),
        }
    }
}

/// A panel member or judge call that failed. It earns a zero-token usage row
/// of its own, the same shape a failed routing attempt gets.
#[derive(Debug, Clone)]
pub struct SubcallFailure {
    /// The member's panel slot, or the panel size for the judge.
    pub index: u32,
    /// `"panel"` or `"judge"`.
    pub kind: &'static str,
    pub model: String,
    pub status: u16,
    pub error_class: &'static str,
    pub error_message: String,
    pub latency_ms: u32,
    /// Whether the call reached the upstream.
    pub dispatched: bool,
}

impl SubcallFailure {
    /// `rate_limited` marks a call the gateway refused on the model's own
    /// rate limit: classed and flagged as a routing attempt refused the same
    /// way is, not as an answer from the upstream.
    pub(crate) fn new(
        index: u32,
        kind: &'static str,
        model: &str,
        err: &BridgeError,
        rate_limited: bool,
        started: Instant,
    ) -> Self {
        Self {
            index,
            kind,
            model: model.to_string(),
            status: err.http_status(),
            error_class: if rate_limited {
                "rate_limit_exceeded"
            } else {
                crate::attempt::routing_error_class(err)
            },
            error_message: crate::attempt::attempt_error_message(err),
            latency_ms: crate::attempt::ms_since(started),
            dispatched: !rate_limited && err.reached_upstream(),
        }
    }
}

pub(crate) fn tier_value(resp: &ChatResponse) -> Option<String> {
    resp.service_tier.as_ref().map(|t| t.as_str().to_owned())
}

/// Everything the dispatch layer needs after an ensemble run: the judge's
/// response (the client-facing answer, whose `usage` is the judge call's
/// own), plus the per-sub-call accounting for telemetry and the aggregate
/// client-facing usage.
#[derive(Debug)]
pub struct EnsembleOutcome {
    pub response: ChatResponse,
    pub panel: Vec<PanelOutcome>,
    /// The panel members that failed, in slot order.
    pub failed: Vec<SubcallFailure>,
    pub judge_model: String,
    pub judge_effective_usage: UsageStats,
    pub judge_usage_estimated: bool,
}

/// The sub-calls of one ensemble run that have already returned, recorded as
/// each one completes rather than when the run does. A run whose future is
/// dropped part-way (the caller hung up) never returns its outcome, yet every
/// sub-call recorded here already round-tripped an upstream and is owed its
/// usage row; the dispatch layer reads this to emit them.
#[derive(Debug, Default)]
pub struct EnsembleProgress {
    inner: std::sync::Mutex<ProgressInner>,
}

#[derive(Debug, Default)]
struct ProgressInner {
    /// Every panel member that returned, answered or failed, in completion
    /// order.
    panel: Vec<Result<PanelOutcome, SubcallFailure>>,
    judge: Option<JudgeReturn>,
}

/// A judge call that returned: its accounting plus the response identity its
/// usage row reports.
#[derive(Debug)]
pub(crate) struct JudgeReturn {
    pub outcome: PanelOutcome,
    pub response_id: String,
    pub response_model: String,
    pub finish_reason: aisix_gateway::FinishReason,
}

impl EnsembleProgress {
    fn record_panel(&self, outcome: Result<PanelOutcome, SubcallFailure>) {
        self.inner.lock().unwrap().panel.push(outcome);
    }

    fn record_judge(&self, index: u32, model: &str, judge: &ModelCallOutcome) {
        self.inner.lock().unwrap().judge = Some(JudgeReturn {
            outcome: PanelOutcome::of(index, model, judge),
            response_id: judge.response.id.clone(),
            response_model: judge.response.model.clone(),
            finish_reason: judge.response.finish_reason.clone(),
        });
    }

    /// The panel members that returned, answered or failed, in configured
    /// order, and the judge if it returned.
    #[allow(clippy::type_complexity)]
    pub(crate) fn take(
        &self,
    ) -> (
        Vec<Result<PanelOutcome, SubcallFailure>>,
        Option<JudgeReturn>,
    ) {
        let mut inner = self.inner.lock().unwrap();
        let mut panel = std::mem::take(&mut inner.panel);
        panel.sort_by_key(|member| match member {
            Ok(answered) => answered.slot,
            Err(failed) => failed.index,
        });
        (panel, inner.judge.take())
    }
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
        /// The panel members that failed, each owed its own usage row.
        failed: Vec<SubcallFailure>,
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
        failed: Vec<SubcallFailure>,
        /// The judge call's own failure row.
        judge: SubcallFailure,
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
/// Returns the per-member accounting (`panel`), the members that failed
/// (`failed`), the raw candidate responses (`candidates`, kept so the caller
/// can re-derive context if needed), and the ready-to-dispatch `judge_req`. The returned
/// `judge_req` is non-streaming (`stream = Some(false)`); a streaming
/// caller flips it to `Some(true)` before dispatching.
///
/// An exhausted panel returns [`EnsembleError::InsufficientPanel`] with
/// the already-billed survivors and the failed members — the dispatch layer
/// must still emit a row for each and commit the survivors' usage.
#[allow(clippy::type_complexity)]
pub(crate) async fn run_ensemble_panel(
    req: &ChatFormat,
    config: &EnsembleConfig,
    caller: &dyn ModelCaller,
    progress: &EnsembleProgress,
) -> Result<
    (
        Vec<PanelOutcome>,
        Vec<SubcallFailure>,
        Vec<ChatResponse>,
        ChatFormat,
    ),
    EnsembleError,
> {
    let per_call_timeout = config.timeout();

    // Phase 1: fan out to every panel member concurrently.
    let calls = config.panel.iter().enumerate().map(|(slot, member)| {
        let model = member.model.clone();
        let slot = slot as u32;
        async move {
            let member_req = || panel_request(req, member);
            let started = Instant::now();
            let result =
                call_with_optional_timeout(caller, &model, &member_req, per_call_timeout).await;
            let result = result.map_err(|err| {
                tracing::warn!(panel_model = %model, error = %err.source, "ensemble panel member failed");
                SubcallFailure::new(
                    slot,
                    "panel",
                    &model,
                    &err.source,
                    err.rate_limited,
                    started,
                )
            });
            progress.record_panel(match &result {
                Ok(outcome) => Ok(PanelOutcome::of(slot, &model, outcome)),
                Err(failed) => Err(failed.clone()),
            });
            (slot, model, result)
        }
    });
    let results = join_all(calls).await;

    // Phase 2: keep the successes, enforce min_responses.
    let mut panel: Vec<PanelOutcome> = Vec::new();
    let mut failed: Vec<SubcallFailure> = Vec::new();
    let mut candidates: Vec<ChatResponse> = Vec::new();
    for (slot, model, result) in results {
        match result {
            Ok(outcome) => {
                let resp = outcome.response;
                panel.push(PanelOutcome {
                    slot,
                    model,
                    usage: resp.usage.clone(),
                    effective_usage: outcome.effective_usage,
                    usage_estimated: outcome.usage_estimated,
                    service_tier: tier_value(&resp),
                });
                candidates.push(resp);
            }
            Err(failure) => failed.push(failure),
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
            failed,
        });
    }

    // Build the judge's synthesis request from the labeled candidates.
    let judge_req = judge_request(req, &config.judge, &candidates);
    Ok((panel, failed, candidates, judge_req))
}

/// Run one ensemble request to completion. See the module docs for the
/// three phases.
pub async fn run_ensemble(
    req: &ChatFormat,
    config: &EnsembleConfig,
    caller: &dyn ModelCaller,
    progress: &EnsembleProgress,
) -> Result<EnsembleOutcome, EnsembleError> {
    // Phases 1-2 + judge-request construction.
    let (panel, failed, candidates, judge_req) =
        run_ensemble_panel(req, config, caller, progress).await?;
    // The judge's prompt renders the caller's messages, so it is rebuilt per
    // attempt like a panel member's request rather than held for the wait.
    drop(judge_req);

    // Phase 3: synthesize via the judge. The judge call gets the same
    // per-call timeout as each panel member (`config.timeout()`), so the
    // ensemble-level deadline applies uniformly across the whole fan-out.
    //
    // Retrying a transient judge failure is the ModelCaller's job now, using
    // the judge model's own retry budget — it used to be a hardcoded single
    // retry here, which both ignored the operator's configuration and would
    // have stacked on top of the caller-level budget.
    let judge_index = config.panel.len() as u32;
    let judge_started = Instant::now();
    let judge = match call_with_optional_timeout(
        caller,
        &config.judge.model,
        &|| judge_request(req, &config.judge, &candidates),
        config.timeout(),
    )
    .await
    {
        Ok(r) => r,
        // Carry the (already-billed) panel survivors out on the judge
        // failure so the dispatch layer commits + emits them too.
        Err(err) => {
            let judge = SubcallFailure::new(
                judge_index,
                "judge",
                &config.judge.model,
                &err.source,
                err.rate_limited,
                judge_started,
            );
            return Err(EnsembleError::Judge {
                source: err.source,
                panel,
                failed,
                judge,
            });
        }
    };
    progress.record_judge(judge_index, &config.judge.model, &judge);

    Ok(EnsembleOutcome {
        response: judge.response,
        panel,
        failed,
        judge_model: config.judge.model.clone(),
        judge_effective_usage: judge.effective_usage,
        judge_usage_estimated: judge.usage_estimated,
    })
}

/// Build a panel member's request: the inbound request with this member's
/// sampling overrides applied. Always non-streaming — the executor needs
/// the full response to synthesize.
fn panel_request(req: &ChatFormat, member: &PanelMember) -> ChatFormat {
    let mut out = req.clone();
    out.model = member.model.clone();
    if let Some(temperature) = member.temperature {
        // Through its shortest decimal, so a configured `0.7` is sent as
        // `0.7` rather than the widened `0.699999988`.
        out.temperature = temperature.to_string().parse().ok();
    }
    if let Some(seed) = member.seed {
        out.extra
            .insert("seed".to_string(), serde_json::Value::from(seed));
    }
    out.stream = Some(false);
    // A streaming caller's `stream_options` would reach a non-streaming
    // upstream call, which OpenAI-protocol upstreams refuse.
    out.extra.remove("stream_options");
    out
}

/// Build the judge's request from the labeled candidate answers.
pub(crate) fn judge_request(
    req: &ChatFormat,
    judge: &Judge,
    candidates: &[ChatResponse],
) -> ChatFormat {
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
    if let Some(effort) = req.extra.get("reasoning_effort") {
        out.extra
            .insert("reasoning_effort".to_string(), effort.clone());
    }
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
                truncate_bytes(&c.message.content_str(), MAX_CANDIDATE_BYTES)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::System => "System",
        Role::Developer => "Developer",
        Role::User => "User",
        Role::Assistant => "Assistant",
        Role::Tool => "Tool",
        Role::Function => "Function",
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
    req: &RequestBuilder<'_>,
    timeout: Option<Duration>,
) -> Result<ModelCallOutcome, CallError> {
    match timeout {
        Some(d) => match tokio::time::timeout(d, caller.call(target, req)).await {
            Ok(result) => result,
            Err(_) => Err(BridgeError::Timeout {
                elapsed_ms: d.as_millis() as u64,
                cause: String::new(),
            }
            .into()),
        },
        None => caller.call(target, req).await,
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
        async fn call(
            &self,
            target: &str,
            req: &RequestBuilder<'_>,
        ) -> Result<ModelCallOutcome, CallError> {
            self.calls.lock().unwrap().push((target.to_string(), req()));
            let mut scripted = self.scripted.lock().unwrap();
            let queue = scripted
                .get_mut(target)
                .unwrap_or_else(|| panic!("no scripted response for target {target:?}"));
            let response = queue
                .pop_front()
                .unwrap_or_else(|| panic!("scripted responses exhausted for target {target:?}"))?;
            Ok(ModelCallOutcome {
                effective_usage: response.usage.clone(),
                response,
                usage_estimated: false,
            })
        }
    }

    fn ok(model: &str, content: &str) -> Result<ChatResponse, BridgeError> {
        Ok(ChatResponse {
            logprobs: None,
            more_choices: Vec::new(),
            id: format!("id-{model}"),
            model: model.to_string(),
            message: ChatMessage::assistant(content),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(10, 20),
            service_tier: None,
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

        let mut req = user_request("what is 2+2?");
        req.extra
            .insert("reasoning_effort".to_string(), "medium".into());
        let out = run_ensemble(&req, &cfg, &caller, &EnsembleProgress::default())
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
        assert_eq!(
            caller.calls_to("judge")[0].extra["reasoning_effort"],
            "medium"
        );
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

        let err = run_ensemble(
            &user_request("hi"),
            &cfg,
            &caller,
            &EnsembleProgress::default(),
        )
        .await
        .unwrap_err();

        match err {
            EnsembleError::InsufficientPanel {
                got,
                needed,
                panel,
                failed,
            } => {
                assert_eq!(got, 1);
                assert_eq!(needed, 2);
                // The one survivor (gpt) is carried out so the dispatch
                // layer can still commit + emit its (already-billed) usage.
                assert_eq!(panel.len(), 1);
                assert_eq!(panel[0].model, "gpt");
                assert_eq!(panel[0].slot, 0);
                // The failed member is carried out too, at its own slot.
                assert_eq!(failed.len(), 1);
                assert_eq!(failed[0].model, "claude");
                assert_eq!(failed[0].index, 1);
                assert_eq!(failed[0].kind, "panel");
                // The status the same failure would give the caller.
                assert_eq!(failed[0].status, 502);
                assert_eq!(failed[0].error_class, "upstream_status");
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

        let out = run_ensemble(
            &user_request("hi"),
            &cfg,
            &caller,
            &EnsembleProgress::default(),
        )
        .await
        .unwrap();

        assert_eq!(out.panel.len(), 2);
        // Members keep their configured slots; the failed one is reported at
        // its own.
        let slots: Vec<u32> = out.panel.iter().map(|p| p.slot).collect();
        assert_eq!(slots, [0, 2]);
        assert_eq!(out.failed.len(), 1);
        assert_eq!(
            (out.failed[0].index, out.failed[0].model.as_str()),
            (1, "b")
        );
        assert_eq!(out.failed[0].error_class, "transport");
        let judge_prompt = caller.calls_to("judge")[0].messages[0]
            .content_str()
            .to_string();
        assert!(judge_prompt.contains("alpha"));
        assert!(judge_prompt.contains("gamma"));
        assert!(!judge_prompt.contains("reset"));
    }

    /// Retrying a transient judge failure belongs to the `ModelCaller`, which
    /// applies the judge model's own retry budget. `run_ensemble` used to
    /// wrap the judge in a hardcoded single retry on top of that; this pins
    /// that it no longer does, so the two layers can't multiply (a judge with
    /// `retries: 2` would otherwise have cost up to six upstream calls).
    #[tokio::test]
    async fn does_not_itself_retry_a_transient_judge_failure() {
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

        let err = run_ensemble(
            &user_request("hi"),
            &cfg,
            &caller,
            &EnsembleProgress::default(),
        )
        .await
        .unwrap_err();

        assert_eq!(caller.calls_to("judge").len(), 1);
        assert!(matches!(err, EnsembleError::Judge { .. }));
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

        let err = run_ensemble(
            &user_request("hi"),
            &cfg,
            &caller,
            &EnsembleProgress::default(),
        )
        .await
        .unwrap_err();

        assert_eq!(err.http_status(), 400); // judge 4xx preserved
        assert_eq!(caller.calls_to("judge").len(), 1); // not retried
                                                       // The full panel succeeded and was billed before the judge failed,
                                                       // so the survivors are carried out on the error for the dispatch
                                                       // layer to commit + emit (FIX G).
        match err {
            EnsembleError::Judge {
                source,
                panel,
                failed,
                judge,
            } => {
                assert_eq!(source.http_status(), 400);
                assert_eq!(panel.len(), 2);
                assert!(failed.is_empty());
                // The judge's row follows the panel's two slots.
                assert_eq!(judge.index, 2);
                assert_eq!(judge.kind, "judge");
                assert_eq!(judge.status, 400);
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

        run_ensemble(&req, &cfg, &caller, &EnsembleProgress::default())
            .await
            .unwrap();

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

        run_ensemble(
            &user_request("hi"),
            &cfg,
            &caller,
            &EnsembleProgress::default(),
        )
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

        run_ensemble(
            &user_request("which?"),
            &cfg,
            &caller,
            &EnsembleProgress::default(),
        )
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
