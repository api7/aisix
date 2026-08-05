//! Mid-stream failover for streaming responses (AISIX-Cloud#1222,
//! `routing.stream_failure: continue`) — `/v1/chat/completions`,
//! `/v1/messages`, and `/v1/responses`.
//!
//! Once the first chunk of a streaming response has been committed the
//! HTTP 200 can no longer be revised, so the pre-stream retry/failover
//! loop is out of reach. This module wraps the winning upstream
//! [`ChatChunkStream`] in a combinator that, when a qualifying error
//! arrives mid-stream, dispatches the remaining fallback targets and
//! splices their chunks into the SAME client stream — asking the
//! fallback model to continue the already-delivered partial text
//! (LiteLLM's mid-stream fallback semantics: original messages + a
//! continuation system instruction + an assistant message carrying the
//! partial; Anthropic-wire targets consume the trailing assistant
//! message as native prefill).
//!
//! The endpoints' native-protocol passthrough legs (`/v1/messages` on
//! an Anthropic target, `/v1/responses` on an OpenAI target) forward
//! opaque bytes, not [`ChatChunk`]s — their byte-level combinators live
//! in their endpoint modules and reuse [`acquire_fallback_stream`] +
//! [`classify_trigger`] from here, re-encoding the fallback's chunks
//! onto the client's already-committed wire envelope.
//!
//! [`ChatChunk`]: aisix_gateway::ChatChunk
//!
//! Client-cancel safety is structural: the combinator only makes
//! progress when the pump polls it, and the pump only advances when
//! the client connection pulls — a disconnected client stops the
//! generator at its suspension point, so no fallback dispatch can
//! fire for an abandoned stream.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use aisix_core::{Model, StreamFailure, StreamFailureTrigger};
use aisix_gateway::{BridgeError, ChatFormat, ChatMessage};
use futures::StreamExt;

use crate::attempt::{attempt_error_message, routing_error_class, AttemptRecord};
use crate::client_ip::ClientContext;
use crate::routing::AttemptModel;
use crate::ProxyState;

/// Verbatim LiteLLM continuation instruction (`litellm/router.py`,
/// `_build_responses_continuation_input` — the one continuation surface
/// LiteLLM still ships, after upstream removed the chat-completions
/// one) — kept byte-identical so the two gateways' fallback models
/// receive the same steering. LiteLLM sends it as a `developer` turn on the
/// Responses input; our internal shape carries it as a system message,
/// which each provider bridge maps to its own instruction tier. The
/// partial text is NOT interpolated here; it rides the assistant
/// message that follows.
const CONTINUATION_SYSTEM_PROMPT: &str = "The previous assistant response was interrupted mid-stream. Continue exactly where it stopped — do not repeat any of its content. Your response must read as a seamless continuation.";

/// Which endpoint's stream the combinator is serving. Selects the
/// UsageEvent stamps (`inbound_protocol` + the usage-sink handler
/// label) for the per-attempt events emitted on a mid-stream switch, so
/// they land in telemetry alongside the endpoint's own events.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MidStreamEndpoint {
    Chat,
    Messages,
}

impl MidStreamEndpoint {
    pub(crate) fn sink_label(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Messages => "messages",
        }
    }

    pub(crate) fn inbound_protocol(self) -> &'static str {
        match self {
            Self::Chat => "openai",
            Self::Messages => "anthropic",
        }
    }
}

/// The serving attempt behind the live client stream. Starts as the
/// pre-stream winner; rewritten by the combinator on every mid-stream
/// switch. The pump's `on_complete` closure reads it at stream end so
/// the terminal UsageEvent attributes tokens/latency to the target
/// that actually finished the response.
pub(crate) struct ServingAttempt {
    pub target_id: String,
    /// Routing-target display name for the event's `attempt_model`
    /// (empty for direct models, same convention as the dispatch loop).
    pub target_model: String,
    pub provider: String,
    pub provider_key_id: String,
    pub upstream_model: String,
    /// The serving target's cooldown config, carried here so a
    /// mid-stream failure can run the cooldown decision without a
    /// snapshot lookup.
    pub cooldown: Option<aisix_core::CooldownConfig>,
    pub attempt_index: u32,
    pub attempt_kind: &'static str,
    pub attempt_started: Instant,
}

/// Everything the combinator needs to dispatch fallback targets and
/// keep the request's telemetry coherent while doing so.
pub(crate) struct MidStreamPlan {
    pub cfg: StreamFailure,
    /// Endpoint stamps for the per-attempt events (see
    /// [`MidStreamEndpoint`]).
    pub endpoint: MidStreamEndpoint,
    /// Whether the request pins the output to a structured shape the
    /// continuation cannot safely extend. Computed by the caller from
    /// its own inbound request model (`response_format` on the chat
    /// shape, `text.format` on the Responses shape; the Anthropic shape
    /// has no equivalent).
    pub structured_output: bool,
    /// Targets after the pre-stream winner, in strategy order.
    pub remaining: Vec<AttemptModel>,
    pub state: ProxyState,
    /// Caller identity — the per-target quota gate needs the identity
    /// dimensions for conditional policy rows (AISIX-Cloud#892).
    pub auth: crate::auth::AuthenticatedKey,
    /// The routing (group) model — resolves group-level timeout
    /// defaults for each fallback target.
    pub group: Model,
    /// The original client request (pre-continuation).
    pub req: ChatFormat,
    pub request_id: String,
    pub client: ClientContext,
    pub retry_on_429: bool,
    pub fallback_on_statuses: Vec<u16>,
    /// Client-facing model name (`req.model`) for the failed-attempt
    /// events.
    pub requested_model: String,
    pub api_key_id: String,
    pub applied_guardrails: Vec<aisix_core::AppliedGuardrail>,
    /// Shared with the pump's completion closure.
    pub serving: Arc<Mutex<ServingAttempt>>,
    /// Cross-task state shared with the SSE pump and its completion
    /// closure.
    pub shared: MidStreamShared,
}

/// State the failover combinator shares with `build_sse_stream` and the
/// completion closure. Cheap to clone (all `Arc`s).
#[derive(Clone)]
pub(crate) struct MidStreamShared {
    /// Estimated usage of failed partial attempts, folded into the
    /// final stream's client-facing usage frames by the pump (LiteLLM
    /// merges partial + fallback usage the same way).
    pub extra_usage: Arc<Mutex<aisix_gateway::UsageStats>>,
    /// Bumped on every fallback dispatch. The pump watches it to reset
    /// its usage accumulators when the serving attempt changes —
    /// max-wins folding across attempts would otherwise mix a
    /// per-chunk-usage provider's (e.g. Gemini) partial counters into
    /// the serving attempt's totals. The completion closure reads it
    /// as the fallbacks-attempted count.
    pub attempt_seq: Arc<AtomicU32>,
    /// Rate-limit keys of fallback targets that served this stream —
    /// the completion closure bills their TPM post-stream the same way
    /// it bills the pre-stream reservation's keys (#450 / #1087
    /// family).
    pub extra_post_stream_keys: Arc<Mutex<Vec<String>>>,
    /// Terminal stream failure `(error_class, error_message)` recorded
    /// by a byte-level (passthrough) combinator, which ends the wire
    /// with a synthesized protocol error frame the downstream builder
    /// cannot distinguish from clean bytes — unlike the typed pumps,
    /// which observe the `Err` item directly. `None` = no terminal
    /// failure.
    pub terminal_failure: Arc<Mutex<Option<(String, String)>>>,
}

impl MidStreamShared {
    pub fn new() -> Self {
        Self {
            extra_usage: Arc::new(Mutex::new(aisix_gateway::UsageStats::default())),
            attempt_seq: Arc::new(AtomicU32::new(0)),
            extra_post_stream_keys: Arc::new(Mutex::new(Vec::new())),
            terminal_failure: Arc::new(Mutex::new(None)),
        }
    }
}

/// Classify a mid-stream [`BridgeError`] into the configurable trigger
/// taxonomy. `UpstreamStatus` cannot occur after the 200 is committed;
/// config/credential errors are pre-dispatch by construction. Both map
/// to `None` (never fall back) defensively.
pub(crate) fn classify_trigger(err: &BridgeError) -> Option<StreamFailureTrigger> {
    match err {
        BridgeError::Transport(_) | BridgeError::StreamAborted => {
            Some(StreamFailureTrigger::TransportError)
        }
        BridgeError::Timeout { .. } => Some(StreamFailureTrigger::ReadTimeout),
        BridgeError::UpstreamDecode(_) => Some(StreamFailureTrigger::UpstreamDecodeError),
        BridgeError::UpstreamInBand { .. } => Some(StreamFailureTrigger::UpstreamInBandError),
        BridgeError::UpstreamStatus { .. }
        | BridgeError::Config(_)
        | BridgeError::InvalidUpstreamConfig(_)
        | BridgeError::InvalidUpstreamCredentials(_) => None,
    }
}

/// Whether the request pins the output to a structured shape
/// (`response_format: json_object` / `json_schema`). A fallback model
/// cannot safely continue a half-emitted JSON document, so these
/// requests keep the terminate behavior regardless of config.
pub(crate) fn expects_structured_output(req: &ChatFormat) -> bool {
    req.extra
        .get("response_format")
        .and_then(|rf| rf.get("type"))
        .and_then(|t| t.as_str())
        .is_some_and(|t| t == "json_object" || t == "json_schema")
}

/// Build the continuation request for a fallback target: the original
/// messages, then the continuation instruction, then an assistant
/// message carrying the partial text. An empty partial (the failure
/// beat the first content delta) retries with the untouched messages —
/// LiteLLM's `is_pre_first_chunk` branch: a continuation prompt there
/// would only waste tokens and confuse the model.
pub(crate) fn continuation_request(orig: &ChatFormat, partial: &str) -> ChatFormat {
    let mut req = orig.clone();
    if !partial.is_empty() {
        req.messages
            .push(ChatMessage::system(CONTINUATION_SYSTEM_PROMPT));
        req.messages.push(ChatMessage::assistant(partial));
    }
    req
}

/// Wrap the winning upstream stream with the mid-stream failover
/// combinator. The caller has already checked `mode: continue` and
/// that `plan.remaining` is non-empty.
pub(crate) fn wrap(
    upstream: aisix_gateway::ChatChunkStream,
    plan: MidStreamPlan,
) -> aisix_gateway::ChatChunkStream {
    Box::pin(async_stream::stream! {
        let mut current = upstream;
        // Generated content accumulated across every attempt — the
        // continuation baseline. Capped at the same bound as the
        // pump's estimation buffer; past it a faithful continuation
        // prompt can no longer be built, so fallback disarms.
        let mut partial = String::new();
        let mut partial_overflow = false;
        // Output shapes a fallback model cannot safely continue:
        // half-emitted tool calls, provider-signed reasoning streams,
        // structured output. Sticky once observed.
        let mut unsafe_output = plan.structured_output;
        let mut used: u32 = 0;
        let mut cursor = 0usize;
        let max = plan.cfg.max_fallbacks_or_default();
        // The serving fallback target's rate-limit hold (concurrency
        // slot). Replaced on every switch — releasing the previous
        // fallback's slot — and released when the generator drops at
        // stream end or client cancellation (#450 semantics).
        let mut _fallback_hold: Option<aisix_ratelimit::StreamConcurrencyGuard> = None;
        loop {
            match current.next().await {
                Some(Ok(chunk)) => {
                    if chunk.delta.tool_calls.is_some()
                        || chunk.delta.reasoning_content.is_some()
                    {
                        unsafe_output = true;
                    }
                    if let Some(text) = chunk.delta.content.as_deref() {
                        if partial.len() + text.len()
                            > crate::token_estimate::OUTPUT_ACCUMULATION_CAP
                        {
                            partial_overflow = true;
                        } else {
                            partial.push_str(text);
                        }
                    }
                    yield Ok(chunk);
                }
                Some(Err(err)) => {
                    let eligible = classify_trigger(&err)
                        .is_some_and(|t| plan.cfg.on_or_default().contains(&t))
                        && crate::routing::is_retryable(
                            &err,
                            plan.retry_on_429,
                            &plan.fallback_on_statuses,
                        )
                        && !unsafe_output
                        && !partial_overflow
                        && used < max;
                    if !eligible {
                        yield Err(err);
                        return;
                    }
                    match acquire_fallback_stream(&plan, &mut cursor, &mut used, err, &partial)
                        .await
                    {
                        Ok((next, hold)) => {
                            current = next;
                            _fallback_hold = hold;
                        }
                        Err(last) => {
                            yield Err(last);
                            return;
                        }
                    }
                }
                None => return,
            }
        }
    })
}

/// Record the outgoing (failed) serving attempt: per-attempt UsageEvent
/// with the estimated partial spend, cooldown + health bookkeeping.
/// Mirrors what the pre-stream loop does for a failed attempt, minus
/// the pieces that only exist before the 200 (routing telemetry is
/// already finalized; the access log already went out).
pub(crate) fn finalize_failed_attempt(plan: &MidStreamPlan, err: &BridgeError, partial: &str) {
    let (rec, failed_cooldown, failed_target_id, failed_upstream_model, failed_display);
    {
        let serving = plan.serving.lock().expect("serving lock");
        rec = AttemptRecord {
            index: serving.attempt_index,
            kind: serving.attempt_kind,
            target_model: serving.target_model.clone(),
            target_model_id: serving.target_id.clone(),
            provider_key_id: serving.provider_key_id.clone(),
            status: err.http_status(),
            success: false,
            error_class: routing_error_class(err).to_string(),
            error_message: attempt_error_message(err),
            latency_ms: serving
                .attempt_started
                .elapsed()
                .as_millis()
                .min(u32::MAX as u128) as u32,
        };
        failed_cooldown = serving.cooldown.clone();
        failed_target_id = serving.target_id.clone();
        failed_upstream_model = serving.upstream_model.clone();
        failed_display = if serving.target_model.is_empty() {
            plan.requested_model.clone()
        } else {
            serving.target_model.clone()
        };
    }
    if let Some((ttl, reason)) = crate::cooldown::decide_cooldown(err, failed_cooldown.as_ref()) {
        plan.state
            .runtime_status
            .mark_cooldown(&failed_target_id, ttl, reason);
    }
    plan.state.health.record_failure(&failed_display);

    // Bill the failed attempt's real partial spend: prompt from the
    // original request, completion from the delivered partial text
    // (the same estimator the pump uses when an upstream reports no
    // usage — AISIX-Cloud#1074).
    let est = crate::token_estimate::Estimator::new(
        &failed_upstream_model,
        crate::token_estimate::PromptInput::Chat(Box::new(plan.req.clone())),
    );
    let prompt_tokens = est.count_prompt();
    let completion_tokens = if partial.is_empty() {
        0
    } else {
        est.count_output(partial)
    };
    {
        let mut extra = plan.shared.extra_usage.lock().expect("extra_usage lock");
        *extra = extra.saturating_add(&aisix_gateway::UsageStats::new(
            prompt_tokens,
            completion_tokens,
        ));
    }
    crate::chat::emit_mid_stream_failed_attempt(
        &plan.state,
        plan.endpoint,
        &plan.request_id,
        &plan.requested_model,
        &plan.api_key_id,
        &plan.client,
        &plan.applied_guardrails,
        &rec,
        prompt_tokens,
        completion_tokens,
    );
    tracing::warn!(
        request_id = %plan.request_id,
        failed_target = %failed_display,
        error = %err,
        partial_bytes = partial.len(),
        "mid-stream failure; attempting fallback targets",
    );
}

/// Try the remaining targets (from `cursor`, bounded by the episode's
/// `max_fallbacks`) until one produces a live stream. Targets in
/// cooldown / unhealthy state are skipped without burning fallback
/// budget; a dispatched target that fails to connect burns one. On
/// success the serving handle is rewritten and the caller splices the
/// returned stream into the client response. On exhaustion the most
/// recent error is returned — the pump then terminates the stream with
/// it (in-band error frame, no `[DONE]`), same as LiteLLM surfacing
/// the fallback's own failure.
pub(crate) async fn acquire_fallback_stream(
    plan: &MidStreamPlan,
    cursor: &mut usize,
    used: &mut u32,
    original_err: BridgeError,
    partial: &str,
) -> Result<
    (
        aisix_gateway::ChatChunkStream,
        Option<aisix_ratelimit::StreamConcurrencyGuard>,
    ),
    BridgeError,
> {
    finalize_failed_attempt(plan, &original_err, partial);
    let max = plan.cfg.max_fallbacks_or_default();
    let mut last_err = original_err;
    let cont_req = continuation_request(&plan.req, partial);

    while *cursor < plan.remaining.len() && *used < max {
        let attempt = &plan.remaining[*cursor];
        *cursor += 1;
        // Re-check runtime state at switch time — the pre-stream filter
        // ran before this stream started and the world has moved (the
        // failed target itself may just have been cooled down).
        let stale_after = attempt
            .model
            .background_model_check
            .as_ref()
            .map(|cfg| std::time::Duration::from_secs(cfg.stale_after_seconds));
        let status = plan
            .state
            .runtime_status
            .status_with_stale(&attempt.id, stale_after)
            .status;
        if matches!(
            status,
            crate::RuntimeStatus::Unhealthy | crate::RuntimeStatus::Cooldown
        ) {
            tracing::debug!(
                target = %attempt.model.display_name,
                ?status,
                "skipping mid-stream fallback candidate (runtime state)",
            );
            continue;
        }
        let model = &attempt.model;
        let Ok(provider) = crate::dispatch::require_provider(model) else {
            continue;
        };
        let provider = provider.to_ascii_lowercase();
        let snapshot = plan.state.snapshot.load();
        let Ok(pk_entry) = crate::dispatch::resolve_provider_key(&snapshot, model) else {
            continue;
        };
        let Some(bridge) = crate::dispatch::resolve_bridge(&plan.state.hub, &pk_entry.value) else {
            continue;
        };
        // Reserve the fallback target's own rate-limit layers before
        // dispatching to it, exactly like the pre-stream loop
        // (AISIX-Cloud#1087) — without this a mid-stream continuation
        // would be invisible to the target model's rpm/concurrency
        // caps. A refused reservation skips the candidate (recorded as
        // a 429 attempt) without burning fallback budget: nothing was
        // dispatched upstream.
        let member_reservation = match crate::quota::reserve_routing_target(
            &plan.state,
            &plan.auth,
            true,
            &model.display_name,
            &attempt.id,
            model,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                let rec = AttemptRecord {
                    index: {
                        let mut serving = plan.serving.lock().expect("serving lock");
                        serving.attempt_index += 1;
                        serving.attempt_index
                    },
                    kind: "mid_stream_fallback",
                    target_model: model.display_name.clone(),
                    target_model_id: attempt.id.clone(),
                    provider_key_id: pk_entry.id.clone(),
                    status: 429,
                    success: false,
                    error_class: "rate_limit_exceeded".to_string(),
                    error_message: e.to_string(),
                    latency_ms: 0,
                };
                crate::chat::emit_mid_stream_failed_attempt(
                    &plan.state,
                    plan.endpoint,
                    &plan.request_id,
                    &plan.requested_model,
                    &plan.api_key_id,
                    &plan.client,
                    &plan.applied_guardrails,
                    &rec,
                    0,
                    0,
                );
                continue;
            }
        };
        *used += 1;
        plan.shared.attempt_seq.fetch_add(1, Ordering::Relaxed);
        let attempt_started = Instant::now();
        let mut ctx = crate::dispatch::bridge_ctx(
            &plan.request_id,
            &attempt.id,
            Arc::new(model.clone()),
            &pk_entry.id,
            Arc::new(pk_entry.value.clone()),
            Some(&plan.client),
        );
        let timeouts = crate::routing::effective_timeouts(
            model,
            Some(&plan.group),
            plan.state.default_timeouts,
        );
        if let Some(d) = timeouts.stream {
            ctx = ctx.with_deadline(d);
        }
        match bridge.chat_stream(&cont_req, &ctx).await {
            Ok(up) => {
                let up = crate::stream_timeout::with_read_timeout(up, timeouts.stream);
                plan.state.health.record_success(&model.display_name);
                plan.state.runtime_status.mark_healthy(&attempt.id);
                // Convert the reservation into a stream-lifetime hold
                // and register its keys so the completion closure bills
                // this target's TPM post-stream too.
                let hold = member_reservation.map(|r| {
                    plan.shared
                        .extra_post_stream_keys
                        .lock()
                        .expect("extra keys lock")
                        .extend(r.keys());
                    r.into_stream_hold()
                });
                {
                    let mut serving = plan.serving.lock().expect("serving lock");
                    serving.attempt_index += 1;
                    serving.target_id = attempt.id.clone();
                    serving.target_model = model.display_name.clone();
                    serving.provider = provider;
                    serving.provider_key_id = pk_entry.id.clone();
                    serving.upstream_model =
                        model.upstream_model().unwrap_or("unknown").to_string();
                    serving.cooldown = model.cooldown.clone();
                    serving.attempt_kind = "mid_stream_fallback";
                    serving.attempt_started = attempt_started;
                }
                tracing::info!(
                    request_id = %plan.request_id,
                    fallback_target = %model.display_name,
                    continuation_bytes = partial.len(),
                    "mid-stream fallback target streaming; continuing client response",
                );
                return Ok((up, hold));
            }
            Err(err) => {
                // The candidate never produced a stream — the refused
                // reservation drops here, rolling its counters back.
                // Record it as a failed attempt (zero tokens) and move
                // on.
                let rec = AttemptRecord {
                    index: {
                        let mut serving = plan.serving.lock().expect("serving lock");
                        serving.attempt_index += 1;
                        serving.attempt_index
                    },
                    kind: "mid_stream_fallback",
                    target_model: model.display_name.clone(),
                    target_model_id: attempt.id.clone(),
                    provider_key_id: pk_entry.id.clone(),
                    status: err.http_status(),
                    success: false,
                    error_class: routing_error_class(&err).to_string(),
                    error_message: attempt_error_message(&err),
                    latency_ms: attempt_started.elapsed().as_millis().min(u32::MAX as u128) as u32,
                };
                if let Some((ttl, reason)) =
                    crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
                {
                    plan.state
                        .runtime_status
                        .mark_cooldown(&attempt.id, ttl, reason);
                }
                if crate::routing::is_retryable(&err, plan.retry_on_429, &plan.fallback_on_statuses)
                {
                    plan.state.health.record_failure(&model.display_name);
                }
                crate::chat::emit_mid_stream_failed_attempt(
                    &plan.state,
                    plan.endpoint,
                    &plan.request_id,
                    &plan.requested_model,
                    &plan.api_key_id,
                    &plan.client,
                    &plan.applied_guardrails,
                    &rec,
                    0,
                    0,
                );
                last_err = err;
            }
        }
    }
    Err(last_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_with_extra(extra: serde_json::Value) -> ChatFormat {
        let mut req = ChatFormat::new("m", vec![ChatMessage::user("hi")]);
        if let serde_json::Value::Object(map) = extra {
            req.extra = map;
        }
        req
    }

    #[test]
    fn continuation_appends_instruction_and_partial() {
        let orig = ChatFormat::new("m", vec![ChatMessage::user("write a story")]);
        let cont = continuation_request(&orig, "Once upon a time");
        assert_eq!(cont.messages.len(), 3);
        assert_eq!(cont.messages[1].content_str(), CONTINUATION_SYSTEM_PROMPT);
        assert_eq!(cont.messages[2].content_str(), "Once upon a time");
        // Empty partial → untouched messages (LiteLLM pre-first-chunk
        // branch).
        let plain = continuation_request(&orig, "");
        assert_eq!(plain.messages.len(), 1);
    }

    #[test]
    fn structured_output_detection() {
        assert!(!expects_structured_output(&req_with_extra(
            serde_json::json!({})
        )));
        assert!(expects_structured_output(&req_with_extra(
            serde_json::json!({"response_format": {"type": "json_object"}})
        )));
        assert!(expects_structured_output(&req_with_extra(
            serde_json::json!({"response_format": {"type": "json_schema", "json_schema": {}}})
        )));
        assert!(!expects_structured_output(&req_with_extra(
            serde_json::json!({"response_format": {"type": "text"}})
        )));
    }

    #[test]
    fn trigger_classification_covers_the_mid_stream_taxonomy() {
        use StreamFailureTrigger as T;
        assert_eq!(
            classify_trigger(&BridgeError::Transport("reset".into())),
            Some(T::TransportError)
        );
        assert_eq!(
            classify_trigger(&BridgeError::StreamAborted),
            Some(T::TransportError)
        );
        assert_eq!(
            classify_trigger(&BridgeError::Timeout {
                elapsed_ms: 1,
                cause: String::new()
            }),
            Some(T::ReadTimeout)
        );
        assert_eq!(
            classify_trigger(&BridgeError::UpstreamDecode("x".into())),
            Some(T::UpstreamDecodeError)
        );
        assert_eq!(
            classify_trigger(&BridgeError::UpstreamInBand {
                status: Some(529),
                message: "overloaded".into(),
                parsed: None,
                wire: aisix_gateway::UpstreamWire::Anthropic,
            }),
            Some(T::UpstreamInBandError)
        );
        assert_eq!(
            classify_trigger(&BridgeError::upstream_status(500, "http")),
            None
        );
        assert_eq!(classify_trigger(&BridgeError::Config("c".into())), None);
    }
}
