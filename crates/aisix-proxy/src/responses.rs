//! `POST /v1/responses` — OpenAI Responses API pass-through.
//!
//! The Responses API is an OpenAI-specific endpoint that lets callers
//! interact with the stateful responses surface (`gpt-4o` + tools). The
//! gateway proxies it transparently:
//!
//! 1. Authenticate and authorise the API key + model.
//! 2. Validate the model is an OpenAI provider.
//! 3. Rewrite the `model` field to the upstream model name.
//! 4. Forward verbatim — streaming SSE and non-streaming JSON both work.
//!
//! Only OpenAI models support this endpoint. Non-OpenAI models receive a
//! 400 with an explanatory message.

use aisix_gateway::{ChatFormat, ChatMessage, ChatResponse, FinishReason, UsageStats};
use aisix_obs::{
    content_capture_cap, AccessLog, CapturedContent, LatencyLabels, RequestOutcome, UsageEvent,
};
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::attempt::{
    attempt_error_from_proxy, ms_since, AttemptInfo, AttemptRecord, RoutingTelemetry,
};
use crate::auth::AuthenticatedKey;
use crate::chat::sanitize_tag;
use crate::client_ip::ClientContext;
use crate::error::ProxyError;
use crate::state::ProxyState;
use crate::usage_attr::{provider_telemetry_tags, total_tokens_with_cache};

/// Per-request payload from a successful dispatch — carries the
/// response + provider label + the bits of usage data needed for
/// UsageEvent emission (#404). On the verbatim streaming path the
/// emission is owned by the response stream's Drop guard (#808), so
/// `usage = None` here and `usage_handled_by_stream = true` tells the
/// handler not to double-emit.
struct ResponseDispatchSuccess {
    response: Response,
    provider: String,
    /// Set on the non-streaming 2xx path and the buffered output-guardrail
    /// path (both parse the full body here). `None` on the verbatim
    /// streaming path, where the stream's Drop guard emits the UsageEvent
    /// from the terminal SSE event instead (#808).
    usage: Option<ResponseUsage>,
    /// UUID of the resolved Model row — needed for UsageEvent
    /// `model_id` field. Always present on success.
    model_id: String,
    /// UUID of the resolved ProviderKey for the winning target — feeds the
    /// per-PK telemetry attribution tags (provider_kind / branded_provider /
    /// pk_label / …) on the emitted UsageEvent (AISIX-Cloud#867). Empty when
    /// the target carried no provider_key_id.
    provider_key_id: String,
    /// Per-attempt routing telemetry (#655): the failed attempts that
    /// preceded the winner plus the winning attempt itself.
    routing: RoutingTelemetry,
    /// #543: set when an OUTPUT guardrail blocked this response. The
    /// upstream already billed, so this is returned as a "success" carrying
    /// the billed `usage` + a 422 body, and the emitted UsageEvent is marked
    /// `guardrail_blocked` so the dashboard's Blocked tab + budget ledger see
    /// it (silently zeroing the tokens would underreport spend the operator
    /// paid the provider for).
    guardrail_blocked: bool,
    /// `true` on the verbatim streaming path: the response stream's Drop
    /// guard owns the UsageEvent emit (parsed from the terminal SSE event),
    /// so the top-level handler must NOT emit the winner event again (#808).
    usage_handled_by_stream: bool,
    /// Captured request/response content for content-capturing exporters
    /// (AISIX-Cloud#947). `Some` only when an enabled exporter opted into
    /// `content_mode = full`; threaded to `fan_out` via the handler's emit,
    /// never to the CP sink. `None` on the streaming paths, whose
    /// end-of-stream emit owns the capture.
    captured_content: Option<CapturedContent>,
    /// Per-detector PII mask counts applied to the response body (#932),
    /// non-streaming + buffered paths. Merged with the input-side counts by
    /// the handler before the terminal emit. Empty on the live streaming
    /// paths — their end-of-stream closures own the output-side counts.
    output_redactions: crate::redact::RedactionCounts,
    /// Monitor-mode guardrail observations on the response side
    /// (AISIX-Cloud#562), same lifecycle as `output_redactions`.
    output_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
}

/// Dispatch error carrying the per-attempt telemetry accumulated before
/// the request ultimately failed (#655). Mirrors `chat::DispatchFailure`.
struct ResponsesDispatchError {
    err: ProxyError,
    routing: RoutingTelemetry,
}

impl From<ProxyError> for ResponsesDispatchError {
    /// Pre-attempt `?` failures (model-not-found, auth, budget) carry no
    /// recorded attempts.
    fn from(err: ProxyError) -> Self {
        Self {
            err,
            routing: RoutingTelemetry::default(),
        }
    }
}

/// Subset of the OpenAI Responses-API `usage` block the gateway
/// surfaces for telemetry (plus the two Anthropic cache counters carried
/// only on the #825 cross-provider bridge path). Other fields (`total_tokens`,
/// `output_tokens_details.audio_tokens`, etc.) are intentionally
/// dropped here — cp-api's `dpmgr_usage_events` table records only
/// the ones below.
#[derive(Default, Clone)]
struct ResponseUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    /// True when any token counter was filled by the local estimator
    /// because the upstream reported no usage (AISIX-Cloud#1074).
    usage_estimated: bool,
    /// o1/o3/GPT-5 class models surface reasoning tokens as a
    /// subset of `completion_tokens` via
    /// `usage.output_tokens_details.reasoning_tokens`. Zero for
    /// models that don't expose this.
    reasoning_tokens: u32,
    /// OpenAI prompt-cache hit count, subset of `prompt_tokens`,
    /// surfaced via `usage.input_tokens_details.cached_tokens`.
    cached_prompt_tokens: u32,
    /// Anthropic `cache_creation_input_tokens` (cache write). Always 0 on
    /// the verbatim OpenAI path; carried for the cross-provider bridge
    /// path (#825) so an Anthropic-backed /v1/responses call bills cache
    /// writes the same way /v1/messages does.
    cache_creation_tokens: u32,
    /// Anthropic `cache_read_input_tokens` (cache read). Always 0 on the
    /// verbatim OpenAI path (OpenAI surfaces cache hits via
    /// `cached_prompt_tokens` instead).
    cache_read_tokens: u32,
}

pub async fn responses(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    Json(mut body): Json<Value>,
) -> Response {
    let started = Instant::now();
    let request_id = client.request_id.clone();
    let api_key_id = auth.entry.id.clone();

    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Filled by `dispatch` with per-detector PII mask counts (#932); attached
    // to the terminal usage event on both the success and failure paths.
    let mut redaction_counts = crate::redact::RedactionCounts::new();
    // Filled by `dispatch` with monitor-mode guardrail observations
    // (AISIX-Cloud#562), same lifecycle as `redaction_counts`.
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    match dispatch(
        &state,
        &auth,
        &mut body,
        &request_id,
        started,
        &client,
        &mut redaction_counts,
        &mut monitor_hits,
    )
    .await
    {
        Ok(success) => {
            // #932: fold the non-streaming response-side mask counts into
            // the per-request total before the terminal emit below.
            crate::redact::merge_counts(&mut redaction_counts, success.output_redactions.clone());
            monitor_hits.extend(success.output_monitor_hits.clone());
            let elapsed = started.elapsed();
            let status = success.response.status().as_u16();
            emit_access_log(
                &model_name,
                &success.provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
                &success.routing,
                None,
            );
            state.metrics.record_request(
                &success.provider,
                &model_name,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            // Per #655: one zero-token UsageEvent per failed attempt that
            // preceded the winner (non-streaming failover).
            emit_failed_attempts(
                &state,
                &request_id,
                &model_name,
                &api_key_id,
                &client,
                &success.routing,
                // The winner's success event carries the content.
                /* content_for_last */
                None,
            );
            // Issue #404: emit UsageEvent so cp-api's budget ledger
            // and customer-facing /logs analytics see /v1/responses
            // spend. Pre-#404 the responses handler dropped the event
            // entirely — every o1/o3/GPT-5 traffic via Responses API
            // was invisible to budget enforcement and billing
            // reconciliation.
            //
            // #808: the verbatim streaming path can't extract usage
            // synchronously here (the SSE bytes are consumed by the client
            // after this handler returns), so its UsageEvent is emitted from
            // the response stream's Drop guard, which parses the terminal
            // `response.completed` event. `usage_handled_by_stream` guards
            // against a double-emit; `usage` is `None` on that path.
            if !success.usage_handled_by_stream {
                // SLO e2e histogram (AISIX-Cloud#1011): recorded even when
                // the upstream response carried no parseable usage block —
                // latency observation must not depend on token accounting.
                state.metrics.record_request_e2e_latency(
                    LatencyLabels {
                        endpoint: "/v1/responses",
                        model: &model_name,
                        provider: &success.provider,
                        status,
                        streaming: body
                            .get("stream")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                    },
                    elapsed,
                );
                if let Some(usage) = success.usage {
                    // Winning-attempt classification (#655). Direct models
                    // have no recorded attempt → AttemptInfo defaults.
                    let attempt = success
                        .routing
                        .winner()
                        .map(AttemptInfo::from_record)
                        .unwrap_or_default();
                    emit_usage_event(
                        &state,
                        &request_id,
                        &success.model_id,
                        &model_name,
                        &api_key_id,
                        &success.provider_key_id,
                        status,
                        elapsed,
                        &usage,
                        &client,
                        attempt,
                        success.guardrail_blocked,
                        redaction_counts.clone(),
                        monitor_hits.clone(),
                        success.captured_content.as_ref(),
                    );
                }
            }
            success.response
        }
        Err(ResponsesDispatchError { err, routing }) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                &model_name,
                "unknown",
                &api_key_id,
                status,
                elapsed,
                &request_id,
                &routing,
                Some(&err),
            );
            let snap = state.snapshot.load();
            let metric_model = crate::usage_attr::metric_model_label(&snap, &model_name);
            state.metrics.record_request(
                "unknown",
                metric_model,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            state.metrics.record_request_e2e_latency(
                LatencyLabels {
                    endpoint: "/v1/responses",
                    model: metric_model,
                    provider: "unknown",
                    status,
                    streaming: body
                        .get("stream")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                },
                elapsed,
            );
            // AISIX-Cloud#1013: failed requests carry the (post-mask)
            // request body so a 4xx/5xx can be triaged from the log alone.
            // Same opt-in gate and cap as the success path; 401/403 stay
            // body-less (a 401 here is upstream-auth passthrough — caller
            // 401s are rejected by the auth extractor before any event
            // exists) (the body adds nothing to an authorization failure).
            let mut failure_content = if status == 401 || status == 403 {
                None
            } else {
                content_capture_cap(
                    snap.observability_exporters
                        .entries()
                        .iter()
                        .map(|e| &e.value),
                )
                .map(|cap| {
                    CapturedContent::new(
                        &serde_json::to_string(&body).unwrap_or_default(),
                        "",
                        cap as usize,
                    )
                })
            };
            // When every target failed there is no terminal event below —
            // the content rides the last failed attempt instead.
            let content_for_last = if !routing.attempts.is_empty() {
                failure_content.take()
            } else {
                None
            };
            // Per #655: emit one zero-token UsageEvent per FAILED attempt so
            // the dashboard's Logs tab surfaces each failed upstream try.
            emit_failed_attempts(
                &state,
                &request_id,
                &model_name,
                &api_key_id,
                &client,
                &routing,
                content_for_last,
            );
            // Pre-dispatch failure (model-not-found, auth, budget) records no
            // attempts — emit a single terminal event carrying the failure
            // class (`model_id` empty: the model never resolved). When
            // attempts were recorded, each was already emitted.
            if routing.attempts.is_empty() {
                emit_zero_token_event(
                    &state,
                    &request_id,
                    "",
                    &model_name,
                    &api_key_id,
                    // Pre-dispatch failure resolved no provider key → wire NULL.
                    "",
                    status,
                    elapsed,
                    &client,
                    AttemptInfo {
                        kind: "initial".to_string(),
                        error_class: err.kind().to_string(),
                        ..Default::default()
                    },
                    // Input masking may have fired before the failure.
                    redaction_counts.clone(),
                    monitor_hits.clone(),
                    failure_content.take(),
                );
            }
            err.into_response()
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    // `&mut` so mask-action PII guardrails (#932) can rewrite the request
    // text in place before it reaches the upstream.
    body: &mut Value,
    request_id: &str,
    // Request-scoped clock + downstream client attribution, threaded so the
    // streaming path's Drop guard can stamp latency + client IP/UA on the
    // end-of-stream UsageEvent it emits (#808).
    started: Instant,
    client: &ClientContext,
    // Out-param: per-detector PII mask counts (#932). Input-side counts land
    // here as soon as the request is rewritten; the non-streaming output side
    // arrives via `ResponseDispatchSuccess::output_redactions`; streaming
    // output counts travel via the stream completion instead.
    redactions_out: &mut crate::redact::RedactionCounts,
    // Out-param: monitor-mode guardrail observations (AISIX-Cloud#562),
    // same lifecycle as `redactions_out`.
    monitor_hits_out: &mut Vec<aisix_core::GuardrailMonitorHit>,
) -> Result<ResponseDispatchSuccess, ResponsesDispatchError> {
    let snapshot = state.snapshot.load();

    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::InvalidRequest("`model` field missing".into()))?
        .to_string();

    let model_entry = crate::model_resolve::resolve_model(&snapshot, &model_name)
        .ok_or_else(|| ProxyError::ModelNotFound(model_name.clone()))?;

    if !auth.key().can_access(&model_name) {
        return Err(ProxyError::ModelForbidden(model_name.clone()).into());
    }

    // Client-IP allowlist gate (#557): reject before guardrails / upstream.
    crate::dispatch::check_ip_access(&model_entry.value, &client.source_ip)?;

    // #719: /v1/responses must run input guardrails like /v1/chat/completions
    // and /v1/messages. Before this, user input reached the upstream without
    // any configured content/DLP check, so a content block enforced on the
    // chat surface was bypassable simply by calling /v1/responses (the same
    // violent input that 422s on chat returned 200 with the content echoed
    // here). Translate the Responses-API body into the internal ChatFormat
    // and run the resolved input guardrail chain; a Block short-circuits
    // before dispatch. (Input Bypass is not applied to the outgoing
    // Responses body — only Block is enforced, matching /v1/messages.)
    //
    // #542: run this BEFORE the rate-limit reservation so a content-policy
    // block doesn't burn an RPM slot (matching /v1/chat/completions).
    let guardrail_ctx = aisix_guardrails::RequestContext {
        model_id: &model_entry.id,
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    // Arc so the chain can be cloned into the cross-provider streaming
    // response body (which outlives this handler) for end-of-stream output
    // guardrails (#825), mirroring /v1/messages.
    let resolved_chain = Arc::new(state.guardrail_index.resolve(&guardrail_ctx));
    if !resolved_chain.is_empty() {
        let chat = responses_input_to_chat(&model_name, body);
        let (verdict, hits) = aisix_guardrails::Guardrail::check_input_non_segment_observed(
            resolved_chain.as_ref(),
            &chat,
        )
        .await;
        monitor_hits_out.extend(hits);
        // Segment pass: one Bedrock call over the body's text slots; an
        // ANONYMIZE disposition writes the masked text back into the
        // Responses body (#932 bedrock follow-up).
        let verdict = crate::redact::moderate_body(
            resolved_chain.as_ref(),
            crate::redact::Direction::Input,
            verdict,
            redactions_out,
            monitor_hits_out,
            |g| crate::redact::redact_responses_request(g, body),
        )
        .await;
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } = verdict
        {
            // Per #153 the matched-pattern detail stays in ops logs only; the
            // wire envelope names only the guardrail that fired (#519 B.4b)
            // so callers can't enumerate the blocklist by probing error
            // responses.
            // AISIX-Cloud#1013: mask before returning so the failure
            // content capture exports post-mask text (see chat.rs).
            crate::redact::merge_counts(
                redactions_out,
                crate::redact::redact_responses_request(resolved_chain.as_ref(), body),
            );
            tracing::warn!(
                guardrail_hook = "input",
                model = %model_name,
                reason = %reason,
                "guardrail blocked /v1/responses request",
            );
            return Err(
                ProxyError::ContentFiltered(crate::error::guardrail_block_message(
                    "request",
                    guardrail_name.as_deref(),
                ))
                .into(),
            );
        }
        // #932: mask-action PII rules rewrite the Responses body in place
        // AFTER the block check passes — both the verbatim passthrough and
        // the cross-provider bridge forward from this body.
        crate::redact::merge_counts(
            redactions_out,
            crate::redact::redact_responses_request(resolved_chain.as_ref(), body),
        );
    }

    let model_rl =
        crate::quota::ModelRateLimit::from_model(&model_name, &model_entry.id, &model_entry.value);
    // `Option` so the winning streaming attempt can `take()` the reservation
    // and carry it into the end-of-stream guard (#688); non-streaming / failed
    // attempts leave it in place for the post-dispatch commit or a retry.
    let mut reservation = Some(crate::quota::enforce(state, auth, Some(&model_rl)).await?);

    // Resolve the attempt list (routing-aware). A Model Group walks its
    // targets in order; a direct model resolves to itself (#471). OpenAI
    // targets take the verbatim Responses passthrough; every other provider
    // is bridged through ChatFormat (#825), so a group can mix and fail over
    // across both kinds.
    let attempt_models = crate::routing::resolve_attempt_models(
        &state.routing,
        &state.runtime_status,
        &snapshot,
        &model_name,
        &model_entry.id,
        &model_entry.value,
        crate::routing::RoutingRequest {
            tags: &client.routing_tags,
            stability_key: Some(
                client
                    .routing_key
                    .as_deref()
                    .unwrap_or(auth.entry.id.as_str()),
            ),
            source_ip: &client.source_ip,
        },
    )?;
    let retry_on_429 = model_entry
        .value
        .routing
        .as_ref()
        .map(|r| r.retry_on_429_or_default())
        .unwrap_or(false);
    let fallback_statuses: &[u16] = model_entry
        .value
        .routing
        .as_ref()
        .map(|r| r.fallback_on_statuses_or_default())
        .unwrap_or(&[]);
    let is_routing_request = model_entry.value.routing.is_some();
    let mut routing = RoutingTelemetry::default();
    // `routing.retries` — same-target retries (with backoff) before failing
    // over, honoured exactly like chat.rs / messages.rs (#641). 0 (default)
    // keeps fail-over-only; /v1/responses previously ignored it entirely.
    let retries = model_entry
        .value
        .routing
        .as_ref()
        .map(|r| r.retries_or_default())
        .unwrap_or(0);

    // Walk the targets, failing over on a retryable failure. Streaming and
    // non-streaming share this loop: the per-target dispatch branches
    // internally and, for streaming, only returns Ok once the first chunk
    // has arrived under `stream_timeout` (#554) — so the 200 is committed to
    // exactly one target and a slow first chunk fails over.
    let mut last_err: Option<ProxyError> = None;
    'targets: for target in &attempt_models {
        // Resolved ProviderKey UUID for this target — feeds the per-PK
        // telemetry attribution tags on the emitted UsageEvent
        // (AISIX-Cloud#867). Recorded on the AttemptRecord (success + failure)
        // so both the winner and each failed-attempt event can attribute it.
        let pk_id = target.model.provider_key_id.clone().unwrap_or_default();
        for attempt_idx in 0..=retries {
            // Exponential backoff + jitter before re-hitting the SAME target
            // (#641); cross-target fall-over (the outer loop) stays immediate.
            if attempt_idx > 0 {
                tokio::time::sleep(crate::routing::retry_backoff(attempt_idx as u32)).await;
            }
            let (idx, kind) = routing.begin_attempt(&target.model.display_name);
            let target_model = if is_routing_request {
                target.model.display_name.clone()
            } else {
                String::new()
            };
            let attempt_started = Instant::now();
            // Winning-attempt classification (#655) for the streaming path's
            // end-of-stream UsageEvent. The non-streaming / buffered paths emit
            // from the handler and ignore it.
            let attempt = AttemptInfo {
                index: idx,
                kind: kind.to_string(),
                model: target_model.clone(),
                ..Default::default()
            };
            // Reserve THIS target's own model rate-limit layers before
            // dispatching to it (AISIX-Cloud#1087). Over-limit → record a
            // 429 attempt and move on to the remaining targets in strategy
            // order (same-target retries can't help — the window won't
            // reset mid-loop).
            let mut member_reservation = match crate::quota::reserve_routing_target(
                state,
                is_routing_request,
                &target.model.display_name,
                &target.id,
                &target.model,
            )
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    routing.attempts.push(AttemptRecord {
                        index: idx,
                        kind,
                        target_model,
                        target_model_id: target.id.clone(),
                        provider_key_id: pk_id.clone(),
                        status: 429,
                        success: false,
                        error_class: "rate_limit_exceeded".to_string(),
                        error_message: e.to_string(),
                        latency_ms: 0,
                    });
                    last_err = Some(e);
                    continue 'targets;
                }
            };
            let result = if target.model.provider.as_deref() == Some("openai") {
                responses_to_target(
                    state,
                    &snapshot,
                    body,
                    &target.model,
                    &target.id,
                    request_id,
                    resolved_chain.clone(),
                    started,
                    &model_name,
                    &auth.entry.id,
                    client,
                    attempt,
                    &mut reservation,
                    &mut member_reservation,
                    redactions_out.clone(),
                    monitor_hits_out.clone(),
                )
                .await
            } else {
                responses_cross_provider_to_target(
                    state,
                    &snapshot,
                    body,
                    &target.model,
                    &target.id,
                    request_id,
                    resolved_chain.clone(),
                    started,
                    &model_name,
                    &auth.entry.id,
                    client,
                    attempt,
                    &mut reservation,
                    &mut member_reservation,
                    redactions_out.clone(),
                    monitor_hits_out.clone(),
                )
                .await
            };
            match result {
                Ok(mut success) => {
                    let latency_ms = ms_since(attempt_started);
                    // Feed the least_latency EWMA for this target.
                    state.runtime_status.record_latency(&target.id, latency_ms);
                    routing.attempts.push(AttemptRecord {
                        index: idx,
                        kind,
                        target_model,
                        target_model_id: target.id.clone(),
                        provider_key_id: pk_id.clone(),
                        status: success.response.status().as_u16(),
                        success: true,
                        error_class: String::new(),
                        error_message: String::new(),
                        latency_ms,
                    });
                    success.routing = routing;
                    // #911 [21]: commit the reserved layers with the actual
                    // token cost so TPM/TPD is enforced for /v1/responses like
                    // chat + embeddings. The buffered / non-streaming paths
                    // carry `usage` here and commit now; the streaming path
                    // already `take()`-d the reservation into its end-of-stream
                    // guard (#688), so `reservation` is `None` and this is
                    // skipped.
                    if !success.usage_handled_by_stream {
                        if let Some(mut r) = reservation.take() {
                            // Fold this target's model-layer reservation in
                            // (AISIX-Cloud#1087) so one commit bills the
                            // member's TPM/TPD too. Already `None` when the
                            // streaming path folded it into the guard.
                            if let Some(member) = member_reservation.take() {
                                r.merge(member);
                            }
                            let total = success
                                .usage
                                .as_ref()
                                .map(|u| {
                                    total_tokens_with_cache(
                                        u.prompt_tokens,
                                        u.completion_tokens,
                                        u.cache_creation_tokens,
                                        u.cache_read_tokens,
                                    )
                                })
                                .unwrap_or(0);
                            r.commit_tokens(total).await;
                        }
                    }
                    return Ok(success);
                }
                Err(e) => {
                    let retryable = matches!(
                        &e,
                        ProxyError::Bridge(be) if crate::routing::is_retryable(be, retry_on_429, fallback_statuses)
                    );
                    let (error_class, error_message) = attempt_error_from_proxy(&e);
                    routing.attempts.push(AttemptRecord {
                        index: idx,
                        kind,
                        target_model,
                        target_model_id: target.id.clone(),
                        provider_key_id: pk_id.clone(),
                        status: e.status().as_u16(),
                        success: false,
                        error_class,
                        error_message,
                        latency_ms: ms_since(attempt_started),
                    });
                    last_err = Some(e);
                    // Non-retryable → stop entirely. Retryable → re-hit the
                    // same target until `retries` is exhausted, then fall over
                    // to the next target (the outer loop advances).
                    if !retryable {
                        break 'targets;
                    }
                    if attempt_idx == retries {
                        break;
                    }
                }
            }
        }
    }

    Err(ResponsesDispatchError {
        err: last_err.unwrap_or(ProxyError::ProviderUnavailable),
        routing,
    })
}

/// Translate a `/v1/responses` request body into the internal
/// [`ChatFormat`] so the input guardrail chain can scan the
/// user-supplied content (#719). Only scannable text matters here — this
/// is **not** a faithful Responses→Chat transform and is never sent
/// upstream; the original `body` is forwarded verbatim.
///
/// The Responses-API `input` field is either a bare string or an array of
/// input items; a message item is `{role, content}` whose `content` is a
/// string or an array of typed parts (`input_text` / `output_text` /
/// `text`). The optional top-level `instructions` maps to a system
/// message. Roles are preserved so the guardrail's user-vs-all message
/// selection behaves the same as on /v1/chat/completions.
/// <https://platform.openai.com/docs/api-reference/responses/create>
fn responses_input_to_chat(model: &str, body: &Value) -> ChatFormat {
    let mut messages = Vec::new();

    if let Some(instructions) = body.get("instructions").and_then(|v| v.as_str()) {
        if !instructions.is_empty() {
            messages.push(ChatMessage::system(instructions.to_string()));
        }
    }

    match body.get("input") {
        Some(Value::String(text)) => {
            if !text.is_empty() {
                messages.push(ChatMessage::user(text.clone()));
            }
        }
        Some(Value::Array(items)) => {
            for item in items {
                // A bare-string array element is treated as user text; an
                // object element is a message whose role we preserve.
                if let Some(text) = item.as_str() {
                    if !text.is_empty() {
                        messages.push(ChatMessage::user(text.to_string()));
                    }
                    continue;
                }
                let text = responses_item_text(item);
                if text.is_empty() {
                    continue;
                }
                let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                messages.push(match role {
                    "assistant" => ChatMessage::assistant(text),
                    "system" | "developer" => ChatMessage::system(text),
                    _ => ChatMessage::user(text),
                });
            }
        }
        _ => {}
    }

    ChatFormat::new(model, messages)
}

/// Collect the plain, caller-supplied text of one Responses-API input
/// item, across every key on the `input`-item union that carries text the
/// model will see:
/// - `content` — message items;
/// - `output` — tool-result items (`function_call_output`,
///   `custom_tool_call_output`, `*_call_output`) the caller feeds back;
/// - `reason` — an `mcp_approval_response` justification.
///
/// All are user-controlled content entering the model — the
/// `/v1/chat/completions` equivalent (a `role:"tool"` message) is
/// scanned, so leaving any of these unscanned would let the #719
/// surface-switch bypass survive on that channel. Each slot is a string
/// or an array of typed parts; we gather the `text` of each part and
/// ignore non-text parts (images, files). These items carry no `role`,
/// so the caller maps them to a user message (scanned by every guardrail
/// kind). Reading a key absent on other item types is a harmless no-op.
/// <https://platform.openai.com/docs/api-reference/responses/create>
fn responses_item_text(item: &Value) -> String {
    [item.get("content"), item.get("output"), item.get("reason")]
        .into_iter()
        .flatten()
        .map(responses_value_text)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Plain text of one Responses-API content slot: a bare string, or the
/// concatenated `text` of an array of typed parts.
fn responses_value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Dispatch one concrete OpenAI target's Responses-API passthrough to
/// `{api_base}/v1/responses`. The caller has already confirmed
/// `model.provider == openai`.
#[allow(clippy::too_many_arguments)]
async fn responses_to_target(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    body: &Value,
    model: &aisix_core::Model,
    model_id: &str,
    request_id: &str,
    // Arc so the live-forward streaming path can carry the chain into its
    // end-of-stream observation (AISIX-Cloud#1010).
    chain: Arc<aisix_guardrails::GuardrailChain>,
    // #808: end-of-stream UsageEvent context for the verbatim streaming
    // path's Drop guard. Unused by the non-streaming / buffered paths,
    // which emit from the handler.
    started: Instant,
    requested_model: &str,
    api_key_id: &str,
    client_ctx: &ClientContext,
    attempt: AttemptInfo,
    reservation: &mut Option<aisix_ratelimit::MultiReservation>,
    // This target's own model-layer reservation (routing dispatch only,
    // AISIX-Cloud#1087). The streaming path folds it into `reservation`
    // before the take below; the non-streaming path leaves it for the
    // handler to commit alongside `reservation`.
    member_reservation: &mut Option<aisix_ratelimit::MultiReservation>,
    // Input-side PII mask counts (#932) for the verbatim streaming path's
    // end-of-stream emit; the non-streaming/buffered emits happen in the
    // handler, which already holds them.
    input_redactions: crate::redact::RedactionCounts,
    // Input-side monitor hits (AISIX-Cloud#562), same lifecycle as
    // `input_redactions`.
    input_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
) -> Result<ResponseDispatchSuccess, ProxyError> {
    let chain_arc = chain;
    let chain = chain_arc.as_ref();
    // Largest content cap any enabled content-capturing exporter wants, or
    // `None` when none do (AISIX-Cloud#947). The captured prompt is the
    // client-facing request body (post-#932-redaction), taken BEFORE the
    // upstream model rewrite below so the log shows what the caller sent.
    let content_cap = content_capture_cap(
        snapshot
            .observability_exporters
            .entries()
            .iter()
            .map(|e| &e.value),
    );
    let captured_prompt = content_cap.map(|_| serde_json::to_string(body).unwrap_or_default());
    let mut body = body.clone();
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
    // Resolved PK id for per-PK telemetry attribution on the emitted
    // UsageEvent (AISIX-Cloud#867).
    let provider_key_id = pk_entry.id.clone();
    let api_key = crate::dispatch::require_api_key(&pk_entry.value, model)?.to_string();
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();

    // Rewrite model field to upstream name.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model.clone());
    }

    // Apply the PK's `request.*` overrides to the outbound body, matching the
    // OpenAI bridge's chat() path and the /v1/messages passthrough. The
    // verbatim /v1/responses path builds the request directly (bypassing the
    // Hub), so without this the override pipeline silently no-ops for Codex
    // traffic (AISIX-Cloud#867 follow-up). Apply order: renames → constraints
    // → defaults; each is a no-op when its configured map is empty.
    if let Some(r) = pk_entry.value.request.as_ref() {
        aisix_provider_openai::overrides::apply_param_renames(&mut body, &r.param_renames);
        if let Some(constraints) = &r.param_constraints {
            aisix_provider_openai::overrides::apply_param_constraints(&mut body, constraints);
        }
        aisix_provider_openai::overrides::apply_default_body_fields(
            &mut body,
            &r.default_body_fields,
        );
    }

    let base = crate::dispatch::resolve_base_url(&pk_entry.value)?;
    // build_v1_url tolerates both `https://api.openai.com` (provider
    // default) and `https://api.openai.com/v1` (the OpenAI-SDK form
    // the dashboard's provider-keys placeholder pre-fills). Without
    // it, the SDK convention double-prefixes to `/v1/v1/responses`
    // and 404s.
    let url = crate::dispatch::build_v1_url(&base, "/responses");

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Build headers explicitly so the PK's `request.default_headers` can inject
    // operator headers. Bridge-owned headers go in FIRST; `apply_default_headers`
    // skips already-present keys + the reserved auth blacklist, so an operator
    // header can never clobber auth.
    let mut headers = axum::http::HeaderMap::new();
    let auth_hv = HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "api key contains invalid header chars: {e}"
        )))
    })?;
    headers.insert(axum::http::header::AUTHORIZATION, auth_hv);
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    let rid_hv = HeaderValue::from_str(request_id).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "request_id contains invalid header chars: {e}"
        )))
    })?;
    headers.insert(HeaderName::from_static("x-aisix-request-id"), rid_hv);
    if let Some(r) = pk_entry.value.request.as_ref() {
        aisix_provider_openai::overrides::apply_default_headers(&mut headers, &r.default_headers);
    }

    let client = crate::http_client::client();
    let mut req = client.post(&url).headers(headers).json(&body);
    // #554: non-streaming gets the E2E request timeout via reqwest's
    // request-level timeout. Streaming must NOT use it (it would cap the
    // whole stream); the streaming branch below enforces the per-chunk
    // read timeout instead.
    if !is_stream {
        if let Some(d) = model.request_timeout() {
            req = req.timeout(d);
        }
    }
    let send_started = Instant::now();
    // least_busy: count this target as in-flight for the upstream call
    // (mirrors chat.rs). Non-streaming / error paths drop the guard at
    // function return; the streaming branch moves it into the
    // end-of-stream closure next to `stream_hold`, so the count stays
    // raised for the stream's full lifetime.
    let in_flight = state.runtime_status.begin_in_flight(model_id);
    // Streaming bounds the connect by the stream deadline (reqwest's
    // request-level timeout can't be used — it would cap the whole stream);
    // non-streaming relies on the request-level timeout set above.
    let connect_deadline = if is_stream {
        model.stream_timeout_effective()
    } else {
        None
    };
    let upstream_resp =
        crate::stream_timeout::send_with_deadline(req, connect_deadline, send_started)
            .await
            .map_err(|be| {
                crate::cooldown::note_failure(
                    &state.runtime_status,
                    model_id,
                    model.cooldown.as_ref(),
                    be,
                )
            })
            .map_err(ProxyError::Bridge)?;

    let status = upstream_resp.status();

    if !status.is_success() {
        let status_u16 = status.as_u16();
        let retry_after = aisix_gateway::parse_retry_after(upstream_resp.headers());
        let message = upstream_resp.text().await.unwrap_or_default();
        let err = aisix_gateway::BridgeError::upstream_status_with_retry_after(
            status_u16,
            message.chars().take(1024).collect::<String>(),
            retry_after,
        );
        if let Some((ttl, reason)) = crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
        {
            state.runtime_status.mark_cooldown(model_id, ttl, reason);
        }
        return Err(ProxyError::Bridge(err));
    }

    state.health.record_success(&model.display_name);
    state.runtime_status.mark_healthy(model_id);

    let provider_label = "openai".to_string();

    if is_stream {
        let headers = upstream_resp.headers().clone();

        // #719: when an output-hook guardrail with a hold-back streaming
        // policy (Window/BufferFull — any block-capable output chain) is
        // attached, the streaming response can't be forwarded token-by-token
        // — a blocked phrase would already be on the wire before it scans
        // clean, the same surface-switch bypass via `stream:true`. Mirror the
        // chat surface's secure default (BufferFull): hold the whole SSE
        // response, scan the assistant output text, then release the bytes
        // verbatim or block with 422. A monitor-only chain resolves to
        // EndOfStreamCheck — it can never block, so it must never hold the
        // stream back nor fail closed on the buffer cap (AISIX-Cloud#1010);
        // it takes the live-forward path below, which scans at end-of-stream
        // for observation. Requests with no output-hook guardrail keep the
        // zero-copy verbatim passthrough.
        let output_policy = aisix_guardrails::Guardrail::stream_output_policy(chain);
        if aisix_guardrails::Guardrail::runs_on_output(chain) && output_policy.holds_back() {
            // Hold the whole SSE response back to scan it, but cap the
            // buffer so a huge (or malicious) upstream response can't OOM the
            // gateway. Mirror the chat surface's secure BufferFull default
            // (#466): read with a running byte count and fail closed if the
            // response exceeds the cap — an output-hook guardrail must never
            // release content it couldn't fully buffer to scan. The cap is
            // taken from the chain's resolved streaming policy.
            let max_buffer_bytes = match output_policy {
                aisix_guardrails::StreamOutputPolicy::BufferFull {
                    max_buffer_bytes, ..
                } => max_buffer_bytes,
                _ => aisix_guardrails::DEFAULT_STREAM_OUTPUT_BUFFER_BYTES,
            };
            let stream = upstream_resp.bytes_stream();
            futures::pin_mut!(stream);
            // Effective streaming budget: `stream_timeout`, falling back to
            // `timeout` — applied to every buffered read, consistent with the
            // verbatim branch and the connect deadline.
            let read_to = model.stream_timeout_effective();
            let mut buf: Vec<u8> = Vec::new();
            let mut saw_chunk = false;
            loop {
                // #554: bound each read so a stalled upstream fails over —
                // the buffer path hasn't sent anything to the client yet, so
                // a read timeout is a retryable failure, not a truncation.
                let next = match read_to {
                    Some(d) => tokio::time::timeout(d, stream.next())
                        .await
                        .map_err(|_| {
                            crate::cooldown::note_failure(
                                &state.runtime_status,
                                model_id,
                                model.cooldown.as_ref(),
                                aisix_gateway::BridgeError::Timeout {
                                    elapsed_ms: d.as_millis() as u64,
                                    cause: String::new(),
                                },
                            )
                        })
                        .map_err(ProxyError::Bridge)?,
                    None => stream.next().await,
                };
                let Some(chunk) = next else {
                    // #554: an upstream that returns 200 then closes with zero
                    // bytes is a first-chunk failure — fail over rather than
                    // serving an empty 200, matching the verbatim branch. Only
                    // when a stream timeout is configured, so a model without
                    // one keeps the pre-#554 behavior.
                    if !saw_chunk && read_to.is_some() {
                        let err = crate::cooldown::note_failure(
                            &state.runtime_status,
                            model_id,
                            model.cooldown.as_ref(),
                            aisix_gateway::BridgeError::StreamAborted,
                        );
                        return Err(ProxyError::Bridge(err));
                    }
                    break;
                };
                saw_chunk = true;
                let chunk = chunk
                    .map_err(|e| {
                        crate::cooldown::note_failure(
                            &state.runtime_status,
                            model_id,
                            model.cooldown.as_ref(),
                            aisix_gateway::BridgeError::UpstreamDecode(e.to_string()),
                        )
                    })
                    .map_err(ProxyError::Bridge)?;
                if buf.len() + chunk.len() > max_buffer_bytes {
                    // Unlike chat's BufferFull, we always fail closed on
                    // overflow regardless of `on_exceeded_fail_open`: an
                    // output-hook guardrail must not release a response it
                    // couldn't fully buffer to scan. No shipped guardrail
                    // configures `BufferFull { on_exceeded_fail_open: true }`
                    // on this surface today.
                    tracing::warn!(
                        guardrail_hook = "output",
                        model = %model.display_name,
                        max_buffer_bytes,
                        "streaming /v1/responses output exceeded buffer cap; failing closed",
                    );
                    return Err(ProxyError::ContentFiltered(
                        "response blocked by content policy".into(),
                    ));
                }
                buf.extend_from_slice(&chunk);
            }
            let out_text = responses_sse_output_text(&buf);
            let synth = synth_chat_response(&upstream_model, out_text);
            let (verdict, hits) =
                aisix_guardrails::Guardrail::check_output_non_segment_observed(chain, &synth).await;
            let mut output_monitor_hits = hits;
            // Segment pass over the held SSE frames: one Bedrock call; an
            // ANONYMIZE disposition rewrites `buf` in place (#932 bedrock
            // follow-up). The capture below reads the post-mask buffer.
            let mut output_redactions = crate::redact::RedactionCounts::new();
            let verdict = crate::redact::moderate_body(
                chain,
                crate::redact::Direction::Output,
                verdict,
                &mut output_redactions,
                &mut output_monitor_hits,
                |g| match crate::redact::redact_responses_sse(g, &buf) {
                    Some((rewritten, counts)) => {
                        buf = rewritten;
                        counts
                    }
                    None => crate::redact::RedactionCounts::new(),
                },
            )
            .await;
            if let aisix_guardrails::GuardrailVerdict::Block {
                reason,
                guardrail_name,
            } = verdict
            {
                // Per #153 the matched-pattern detail stays in ops logs only.
                tracing::warn!(
                    guardrail_hook = "output",
                    model = %model.display_name,
                    reason = %reason,
                    "guardrail blocked streaming /v1/responses response",
                );
                return Err(ProxyError::ContentFiltered(
                    crate::error::guardrail_block_message("response", guardrail_name.as_deref()),
                ));
            }
            // #932: the whole SSE response is held here — mask the frames
            // (channel reassembly) before anything reaches the wire.
            let buf = match crate::redact::redact_responses_sse(chain, &buf) {
                Some((rewritten, counts)) => {
                    crate::redact::merge_counts(&mut output_redactions, counts);
                    rewritten
                }
                None => buf,
            };
            // #808: the whole SSE response is buffered here, so parse its
            // terminal event for usage and let the handler emit (the body is
            // a single complete chunk now, not a live stream).
            //
            // Token-estimation fallback (AISIX-Cloud#1074): a buffered
            // stream with zero/missing usage fills the counters locally —
            // telemetry only, the buffered bytes forward untouched.
            let usage = {
                let mut u = responses_sse_usage(&buf).unwrap_or_default();
                if u.prompt_tokens == 0 || u.completion_tokens == 0 {
                    let est = crate::token_estimate::Estimator::new(
                        &upstream_model,
                        crate::token_estimate::PromptInput::Responses(body.clone()),
                    );
                    let filled = crate::token_estimate::fill_missing(
                        &est,
                        u.prompt_tokens,
                        u.completion_tokens,
                        Some(&responses_sse_output_text(&buf)),
                    );
                    if filled.estimated {
                        u.prompt_tokens = filled.prompt_tokens;
                        u.completion_tokens = filled.completion_tokens;
                        u.usage_estimated = true;
                    }
                }
                Some(u)
            };
            // Content capture (AISIX-Cloud#947): the assembled output text,
            // read from the POST-redaction buffer so masked PII stays masked
            // in the exported content.
            let captured_content = match (&captured_prompt, content_cap) {
                (Some(prompt), Some(cap)) => Some(CapturedContent::new(
                    prompt,
                    &responses_sse_output_text(&buf),
                    cap as usize,
                )),
                _ => None,
            };
            let mut response = axum::response::Response::new(axum::body::Body::from(buf));
            apply_passthrough_headers(&mut response, &headers, request_id);
            return Ok(ResponseDispatchSuccess {
                response,
                provider: provider_label,
                usage,
                model_id: model_id.to_string(),
                provider_key_id: provider_key_id.clone(),
                routing: RoutingTelemetry::default(),
                guardrail_blocked: false,
                usage_handled_by_stream: false,
                output_redactions,
                output_monitor_hits,
                captured_content,
            });
        }

        // #554: enforce the per-chunk read timeout on the forwarded bytes.
        // When a `stream_timeout` is configured, peek the first byte so a
        // slow/erroring first token fails over before the 200 is committed;
        // without one, forward directly (pre-#554 behavior). A mid-stream
        // stall truncates the forwarded stream (no in-band error frame for
        // an opaque byte passthrough).
        let stream_budget = model.stream_timeout_effective();
        let wrapped: std::pin::Pin<
            Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>,
        > = Box::pin(crate::stream_timeout::with_read_timeout_bytes(
            upstream_resp.bytes_stream(),
            stream_budget,
        ));
        let body_stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>,
        > = if stream_budget.is_some() {
            let mut wrapped = wrapped;
            let first_bytes = match wrapped.next().await {
                Some(Ok(b)) => b,
                Some(Err(e)) => {
                    let err = crate::dispatch::reqwest_error_to_bridge(&e, send_started);
                    if let Some((ttl, reason)) =
                        crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
                    {
                        state.runtime_status.mark_cooldown(model_id, ttl, reason);
                    }
                    return Err(ProxyError::Bridge(err));
                }
                None => {
                    let err = aisix_gateway::BridgeError::StreamAborted;
                    if let Some((ttl, reason)) =
                        crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
                    {
                        state.runtime_status.mark_cooldown(model_id, ttl, reason);
                    }
                    return Err(ProxyError::Bridge(err));
                }
            };
            Box::pin(
                futures::stream::once(std::future::ready(Ok::<bytes::Bytes, reqwest::Error>(
                    first_bytes,
                )))
                .chain(wrapped),
            )
        } else {
            wrapped
        };
        // #808: wrap the verbatim byte stream so the terminal
        // `response.completed` SSE event's `usage` block is parsed in-flight
        // and a UsageEvent is emitted from the stream's Drop guard at
        // end-of-stream (or client-disconnect). Bytes forward unchanged — the
        // client still sees the exact upstream SSE wire shape. Pre-#808 this
        // path dropped the event entirely, so every streaming /v1/responses
        // call (e.g. all Codex traffic, which always streams) was invisible
        // to the dashboard Logs and the budget ledger.
        let state_c = state.clone();
        let request_id_c = request_id.to_string();
        let model_id_c = model_id.to_string();
        let requested_model_c = requested_model.to_string();
        let api_key_id_c = api_key_id.to_string();
        let provider_key_id_c = provider_key_id.clone();
        let provider_c = provider_label.clone();
        let client_c = client_ctx.clone();
        // #688: carry the reservation into the end-of-stream guard — keys drive
        // post-stream TPM/TPD accounting, the hold keeps the concurrency slot(s)
        // until the stream ends. `take()` leaves the handler's `reservation` as
        // `None` so it won't also `commit_tokens`.
        // Fold this target's model-layer reservation in first (AISIX-Cloud#1087)
        // so the guard covers the member's limits too; `take()` leaves it `None`
        // so the handler won't also commit it.
        if let Some(member) = member_reservation.take() {
            match reservation.as_mut() {
                Some(main) => main.merge(member),
                None => *reservation = Some(member),
            }
        }
        let post_stream_keys = reservation.as_ref().map(|r| r.keys()).unwrap_or_default();
        let stream_hold = reservation.take().map(|r| r.into_stream_hold());
        let limiter_c = std::sync::Arc::clone(&state.limiter);
        let captured_prompt_c = captured_prompt.clone();
        // AISIX-Cloud#1010: a monitor-only output chain (EndOfStreamCheck —
        // the only way an output-hook chain reaches this live-forward branch)
        // still gets its end-of-stream scan, so would-block / would-mask
        // observations reach telemetry. `None` without an output hook.
        let eos_scan = aisix_guardrails::Guardrail::runs_on_output(chain).then(|| EosOutputScan {
            chain: Arc::clone(&chain_arc),
            upstream_model: upstream_model.clone(),
        });
        // Token-estimation fallback context (AISIX-Cloud#1074): the request
        // body is cloned because the closure runs at end-of-stream Drop.
        // Tokenized only if the upstream never reports usage.
        let estimator_c = crate::token_estimate::Estimator::new(
            &upstream_model,
            crate::token_estimate::PromptInput::Responses(body.clone()),
        );
        let parsed_stream = build_responses_passthrough_stream(
            body_stream,
            content_cap,
            eos_scan,
            move |mut usage, out_text, output_hits| {
                // Streams that reach here are committed 200s — the
                // `!status.is_success()` guard above returned early on errors.
                //
                // Token-estimation fallback (AISIX-Cloud#1074): a stream that
                // ends without a terminal usage event (client abort, relay
                // that omits usage) fills the missing counters from the
                // request + the assembled output text, BEFORE the TPM
                // accounting and the emit below.
                if usage.prompt_tokens == 0 || usage.completion_tokens == 0 {
                    let filled = crate::token_estimate::fill_missing(
                        &estimator_c,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        Some(&out_text),
                    );
                    if filled.estimated {
                        usage.prompt_tokens = filled.prompt_tokens;
                        usage.completion_tokens = filled.completion_tokens;
                        usage.usage_estimated = true;
                    }
                }
                // #688: apply the terminal token cost to TPM/TPD and release the
                // concurrency hold now the stream has ended (sync analog of the
                // reservation's async `commit_tokens`, which this closure can't await).
                let streamed_tokens = total_tokens_with_cache(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.cache_creation_tokens,
                    usage.cache_read_tokens,
                );
                for key in &post_stream_keys {
                    limiter_c.add_tokens_post_stream(key, streamed_tokens);
                }
                drop(stream_hold);
                // least_busy: stream over — this target is no longer
                // in-flight.
                drop(in_flight);
                // Content capture (AISIX-Cloud#947): prompt captured up front,
                // output text assembled by the stream wrapper (empty when no
                // exporter wants content).
                let captured_content = match (&captured_prompt_c, content_cap) {
                    (Some(prompt), Some(cap)) => {
                        Some(CapturedContent::new(prompt, &out_text, cap as usize))
                    }
                    _ => None,
                };
                // SLO e2e histogram: full stream duration (verbatim path).
                state_c.metrics.record_request_e2e_latency(
                    LatencyLabels {
                        endpoint: "/v1/responses",
                        model: &requested_model_c,
                        provider: &provider_c,
                        status: 200,
                        streaming: true,
                    },
                    started.elapsed(),
                );
                // Live-forward path: no output masking possible (a masking
                // guardrail holds back → buffered branch; a monitor-mode one
                // suppresses its masks), so only the input-side counts apply.
                // The end-of-stream scan's monitor observations ride along
                // with the input-side hits (AISIX-Cloud#1010).
                let mut monitor_hits = input_monitor_hits.clone();
                monitor_hits.extend(output_hits);
                emit_usage_event(
                    &state_c,
                    &request_id_c,
                    &model_id_c,
                    &requested_model_c,
                    &api_key_id_c,
                    &provider_key_id_c,
                    200,
                    started.elapsed(),
                    &usage,
                    &client_c,
                    attempt,
                    /* guardrail_blocked */ false,
                    input_redactions.clone(),
                    monitor_hits,
                    captured_content.as_ref(),
                );
            },
        );
        let mut response =
            axum::response::Response::new(axum::body::Body::from_stream(Box::pin(parsed_stream)));
        apply_passthrough_headers(&mut response, &headers, request_id);

        Ok(ResponseDispatchSuccess {
            response,
            provider: provider_label,
            // The Drop guard owns the emit; the handler must not double-emit.
            usage: None,
            model_id: model_id.to_string(),
            provider_key_id,
            routing: RoutingTelemetry::default(),
            guardrail_blocked: false,
            usage_handled_by_stream: true,
            // The Drop guard's emit carries the counts.
            output_redactions: crate::redact::RedactionCounts::new(),
            output_monitor_hits: Vec::new(),
            // The Drop guard's emit carries the captured content too.
            captured_content: None,
        })
    } else {
        let json_body: Value = upstream_resp
            .json()
            .await
            .map_err(|e| {
                crate::cooldown::note_failure(
                    &state.runtime_status,
                    model_id,
                    model.cooldown.as_ref(),
                    aisix_gateway::BridgeError::UpstreamDecode(e.to_string()),
                )
            })
            .map_err(ProxyError::Bridge)?;

        // Extract the upstream-reported usage block for telemetry
        // emission. Pulled here (before the response is moved into
        // `Json::into_response`) so the success struct can carry
        // typed counters rather than re-parsing JSON downstream.
        // Token-estimation fallback (AISIX-Cloud#1074): a body with zero or
        // missing usage fills the counters locally — telemetry only, the
        // response body is forwarded untouched. A body with no `usage`
        // object at all becomes a wholly-estimated record instead of None.
        let usage = {
            let mut u = extract_response_usage(&json_body).unwrap_or_default();
            if u.prompt_tokens == 0 || u.completion_tokens == 0 {
                let est = crate::token_estimate::Estimator::new(
                    &upstream_model,
                    crate::token_estimate::PromptInput::Responses(body.clone()),
                );
                let filled = crate::token_estimate::fill_missing(
                    &est,
                    u.prompt_tokens,
                    u.completion_tokens,
                    Some(&responses_output_text(&json_body)),
                );
                if filled.estimated {
                    u.prompt_tokens = filled.prompt_tokens;
                    u.completion_tokens = filled.completion_tokens;
                    u.usage_estimated = true;
                }
            }
            Some(u)
        };

        // #719: run the output guardrail chain on the assistant's text so a
        // configured output block isn't bypassable by calling /v1/responses
        // (the input half is enforced in `dispatch`). Only when an
        // output-hook guardrail is attached; otherwise this is a no-op.
        let mut json_body = json_body;
        let mut output_seg_counts = crate::redact::RedactionCounts::new();
        let mut output_monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
        if aisix_guardrails::Guardrail::runs_on_output(chain) {
            let synth = synth_chat_response(&upstream_model, responses_output_text(&json_body));
            let (verdict, hits) =
                aisix_guardrails::Guardrail::check_output_non_segment_observed(chain, &synth).await;
            output_monitor_hits.extend(hits);
            let verdict = crate::redact::moderate_body(
                chain,
                crate::redact::Direction::Output,
                verdict,
                &mut output_seg_counts,
                &mut output_monitor_hits,
                |g| crate::redact::redact_responses_response(g, &mut json_body),
            )
            .await;
            if let aisix_guardrails::GuardrailVerdict::Block {
                reason,
                guardrail_name,
            } = verdict
            {
                // Per #153 the matched-pattern detail stays in ops logs only.
                tracing::warn!(
                    guardrail_hook = "output",
                    model = %model.display_name,
                    reason = %reason,
                    "guardrail blocked /v1/responses response",
                );
                // #543: the provider already billed for this response, so
                // return a 422 body BUT carry the billed `usage` (marked
                // guardrail_blocked) — recording zero tokens would let the
                // customer's ledger underreport spend they were charged for.
                // This is the output analog of chat.rs's UpstreamCharge.
                return Ok(ResponseDispatchSuccess {
                    response: ProxyError::ContentFiltered(crate::error::guardrail_block_message(
                        "response",
                        guardrail_name.as_deref(),
                    ))
                    .into_response(),
                    provider: provider_label,
                    usage,
                    model_id: model_id.to_string(),
                    provider_key_id: provider_key_id.clone(),
                    routing: RoutingTelemetry::default(),
                    guardrail_blocked: true,
                    usage_handled_by_stream: false,
                    output_redactions: crate::redact::RedactionCounts::new(),
                    // Hits observed by the blocking check still count.
                    output_monitor_hits,
                    // AISIX-Cloud#1013: the billed-then-blocked event carries
                    // the (post-mask) prompt; the blocked output itself stays
                    // out of the log — blocking it and then archiving it would
                    // defeat the block.
                    captured_content: match (&captured_prompt, content_cap) {
                        (Some(p), Some(cap)) => Some(CapturedContent::new(p, "", cap as usize)),
                        _ => None,
                    },
                });
            }
        }

        // #932: mask-action PII rules rewrite the response body AFTER the
        // block check passes.
        let mut output_redactions = crate::redact::redact_responses_response(chain, &mut json_body);
        crate::redact::merge_counts(&mut output_redactions, output_seg_counts);

        // Content capture (AISIX-Cloud#947): the assistant's assembled output
        // text, read from the POST-redaction body so masked PII stays masked
        // in the exported content.
        let captured_content = match (&captured_prompt, content_cap) {
            (Some(prompt), Some(cap)) => Some(CapturedContent::new(
                prompt,
                &responses_output_text(&json_body),
                cap as usize,
            )),
            _ => None,
        };

        Ok(ResponseDispatchSuccess {
            response: Json(json_body).into_response(),
            provider: provider_label,
            usage,
            model_id: model_id.to_string(),
            provider_key_id,
            routing: RoutingTelemetry::default(),
            guardrail_blocked: false,
            usage_handled_by_stream: false,
            output_redactions,
            output_monitor_hits,
            captured_content,
        })
    }
}

/// Dispatch one non-OpenAI target by bridging the Responses-API request
/// through the gateway's canonical [`ChatFormat`] and the provider
/// [`Bridge`](aisix_gateway::Bridge), then re-encoding the response back
/// into the Responses-API shape (#825). This is what lets clients like
/// `codex` — which speak only the OpenAI Responses API — reach an
/// Anthropic (or any other) backend. Mirrors `messages::cross_provider_dispatch`.
#[allow(clippy::too_many_arguments)]
async fn responses_cross_provider_to_target(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    body: &Value,
    model: &aisix_core::Model,
    model_id: &str,
    request_id: &str,
    chain: Arc<aisix_guardrails::GuardrailChain>,
    started: Instant,
    requested_model: &str,
    api_key_id: &str,
    client_ctx: &ClientContext,
    attempt: AttemptInfo,
    reservation: &mut Option<aisix_ratelimit::MultiReservation>,
    // This target's own model-layer reservation (routing dispatch only,
    // AISIX-Cloud#1087). The streaming path folds it into `reservation`
    // before the take below; the non-streaming path leaves it for the
    // handler to commit alongside `reservation`.
    member_reservation: &mut Option<aisix_ratelimit::MultiReservation>,
    // Input-side PII mask counts (#932), merged into the streamed judge
    // path's end-of-stream emit; non-streaming emits happen in the handler.
    input_redactions: crate::redact::RedactionCounts,
    // Input-side monitor hits (AISIX-Cloud#562), same lifecycle as
    // `input_redactions`.
    input_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
) -> Result<ResponseDispatchSuccess, ProxyError> {
    use aisix_gateway::{Bridge, BridgeContext};

    // Content capture (AISIX-Cloud#947), same contract as the verbatim
    // target: prompt = the client-facing Responses request body
    // (post-#932-redaction), gated on an exporter actually wanting content.
    let content_cap = content_capture_cap(
        snapshot
            .observability_exporters
            .entries()
            .iter()
            .map(|e| &e.value),
    );
    let captured_prompt = content_cap.map(|_| serde_json::to_string(body).unwrap_or_default());

    let provider = model
        .provider
        .as_deref()
        .ok_or_else(|| {
            ProxyError::InvalidRequest(format!("model `{requested_model}` has no provider prefix"))
        })?
        .to_string();
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
    // Resolved PK id for per-PK telemetry attribution on the emitted
    // UsageEvent (AISIX-Cloud#867).
    let provider_key_id = pk_entry.id.clone();
    let bridge: Arc<dyn Bridge> = crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value)
        .ok_or(ProxyError::ProviderUnavailable)?;

    // Faithful Responses → ChatFormat transform; `chat.model` stays the
    // operator-facing name so the bridge re-resolves the upstream id via
    // `ctx.model.upstream_model()` exactly like chat.rs.
    let chat = crate::responses_bridge::responses_request_to_chat(requested_model, body);

    let is_stream = chat.is_streaming();
    let model_arc = Arc::new(model.clone());
    let pk_arc = Arc::new(pk_entry.value.clone());
    let mut ctx = BridgeContext::new(request_id, model_arc, pk_arc);
    let connect_deadline = if is_stream {
        model.stream_timeout_effective()
    } else {
        model.request_timeout()
    };
    if let Some(d) = connect_deadline {
        ctx = ctx.with_deadline(d);
    }
    let provider_label = provider.to_ascii_lowercase();

    // least_busy: count this target as in-flight for the upstream call
    // (mirrors chat.rs). Non-streaming / error paths drop the guard at
    // function return; the streaming branch moves it into the
    // end-of-stream closure next to `stream_hold`, so the count stays
    // raised for the stream's full lifetime.
    let in_flight = state.runtime_status.begin_in_flight(model_id);

    if is_stream {
        let upstream = bridge.chat_stream(&chat, &ctx).await.map_err(|err| {
            if let Some((ttl, reason)) =
                crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
            {
                state.runtime_status.mark_cooldown(model_id, ttl, reason);
            }
            ProxyError::Bridge(err)
        })?;
        // #554: peek the first chunk so a slow/erroring first token fails
        // over before the 200 is committed (when a stream budget is set);
        // the wrapper keeps enforcing the per-chunk read timeout either way.
        let stream_budget = model.stream_timeout_effective();
        let upstream = crate::stream_timeout::with_read_timeout(upstream, stream_budget);
        let upstream: aisix_gateway::ChatChunkStream = if stream_budget.is_some() {
            let mut upstream = upstream;
            let first_chunk = match upstream.next().await {
                Some(Ok(chunk)) => chunk,
                Some(Err(err)) => {
                    if let Some((ttl, reason)) =
                        crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
                    {
                        state.runtime_status.mark_cooldown(model_id, ttl, reason);
                    }
                    return Err(ProxyError::Bridge(err));
                }
                None => {
                    let err = aisix_gateway::BridgeError::StreamAborted;
                    if let Some((ttl, reason)) =
                        crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
                    {
                        state.runtime_status.mark_cooldown(model_id, ttl, reason);
                    }
                    return Err(ProxyError::Bridge(err));
                }
            };
            Box::pin(
                futures::stream::once(std::future::ready(Ok::<_, aisix_gateway::BridgeError>(
                    first_chunk,
                )))
                .chain(upstream),
            )
        } else {
            upstream
        };
        // Health tracks the concrete resolved target, not the (possibly
        // group-alias) requested model — matching the non-streaming branch.
        state.health.record_success(&model.display_name);
        state.runtime_status.mark_healthy(model_id);

        let response_id = format!("resp_{}", Uuid::new_v4().simple());
        let created_at = chrono::Utc::now().timestamp();
        let encoder = crate::responses_bridge::ResponsesSseEncoder::new(
            response_id,
            requested_model,
            created_at,
        );
        // Only an output-hook guardrail needs the streamed response text.
        // When attached with a hold-back policy (Window/BufferFull — any
        // block-capable chain), the bridge buffers the SSE and scans before
        // releasing it (#719 secure default); cap the buffer the same way the
        // verbatim path does so a huge response can't OOM the gateway. A
        // monitor-only chain resolves to EndOfStreamCheck — it can never
        // block, so the bridge forwards live and scans at end-of-stream for
        // observation only (AISIX-Cloud#1010).
        let output_guardrail = (!chain.is_empty()
            && aisix_guardrails::Guardrail::runs_on_output(chain.as_ref()))
        .then(|| chain.clone());
        let output_policy = aisix_guardrails::Guardrail::stream_output_policy(chain.as_ref());
        let hold_back = output_policy.holds_back();
        let max_buffer_bytes = match output_policy {
            aisix_guardrails::StreamOutputPolicy::BufferFull {
                max_buffer_bytes, ..
            } => max_buffer_bytes,
            _ => aisix_guardrails::DEFAULT_STREAM_OUTPUT_BUFFER_BYTES,
        };

        let state_c = state.clone();
        let request_id_c = request_id.to_string();
        let model_id_c = model_id.to_string();
        let requested_model_c = requested_model.to_string();
        let api_key_id_c = api_key_id.to_string();
        let provider_key_id_c = provider_key_id.clone();
        let provider_c = provider_label.clone();
        let client_c = client_ctx.clone();
        let attempt_c = attempt.clone();
        // #688: carry the reservation into the end-of-stream guard — keys drive
        // post-stream TPM/TPD accounting, the hold keeps the concurrency slot(s)
        // until the stream ends. `take()` leaves the handler's `reservation` as
        // `None` so it won't also `commit_tokens`.
        // Fold this target's model-layer reservation in first (AISIX-Cloud#1087)
        // so the guard covers the member's limits too; `take()` leaves it `None`
        // so the handler won't also commit it.
        if let Some(member) = member_reservation.take() {
            match reservation.as_mut() {
                Some(main) => main.merge(member),
                None => *reservation = Some(member),
            }
        }
        let post_stream_keys = reservation.as_ref().map(|r| r.keys()).unwrap_or_default();
        let stream_hold = reservation.take().map(|r| r.into_stream_hold());
        let limiter_c = std::sync::Arc::clone(&state.limiter);
        let captured_prompt_c = captured_prompt.clone();
        // Token-estimation fallback context (AISIX-Cloud#1074): the request
        // body is cloned because the stream owns it until an end-of-stream
        // Drop. Tokenized only if the bridged upstream never reports usage.
        let estimator = crate::token_estimate::Estimator::new(
            model.upstream_model().unwrap_or("unknown"),
            crate::token_estimate::PromptInput::Responses(body.clone()),
        );
        let sse_body = crate::responses_bridge::build_responses_bridge_stream(
            upstream,
            encoder,
            started,
            output_guardrail,
            hold_back,
            max_buffer_bytes,
            requested_model.to_string(),
            content_cap,
            Some(estimator),
            move |comp| {
                // #688: apply the terminal token cost to TPM/TPD and release the
                // concurrency hold now the stream has ended (sync analog of the
                // reservation's async `commit_tokens`). Tokens count even on an
                // output-guardrail block — the upstream still billed them.
                let streamed_tokens = total_tokens_with_cache(
                    comp.prompt_tokens,
                    comp.completion_tokens,
                    comp.cache_creation_tokens,
                    comp.cache_read_tokens,
                );
                for key in &post_stream_keys {
                    limiter_c.add_tokens_post_stream(key, streamed_tokens);
                }
                drop(stream_hold);
                // least_busy: stream over — this target is no longer
                // in-flight.
                drop(in_flight);
                let usage = ResponseUsage {
                    prompt_tokens: comp.prompt_tokens,
                    completion_tokens: comp.completion_tokens,
                    reasoning_tokens: comp.reasoning_tokens,
                    cached_prompt_tokens: comp.cached_prompt_tokens,
                    cache_creation_tokens: comp.cache_creation_tokens,
                    cache_read_tokens: comp.cache_read_tokens,
                    usage_estimated: comp.usage_estimated,
                };
                // A clean stream is a committed 200; an output-guardrail block
                // (or fail-closed overflow) bills the upstream tokens but is
                // recorded as a 422 marked guardrail_blocked, matching the
                // non-streaming path so the Blocked tab + ledger see it.
                let status = if comp.guardrail_blocked { 422 } else { 200 };
                // Content capture (AISIX-Cloud#947): prompt captured up front,
                // response assembled across the bridged stream into
                // `comp.response_text` (empty when no exporter wants content
                // or when the response was blocked before release).
                let captured_content = match (&captured_prompt_c, content_cap) {
                    (Some(prompt), Some(cap)) if !comp.guardrail_blocked => Some(
                        CapturedContent::new(prompt, &comp.response_text, cap as usize),
                    ),
                    _ => None,
                };
                // SLO e2e histogram: full stream duration (bridge path).
                // Blocked streams keep this guard's 422 status.
                state_c.metrics.record_request_e2e_latency(
                    LatencyLabels {
                        endpoint: "/v1/responses",
                        model: &requested_model_c,
                        provider: &provider_c,
                        status,
                        streaming: true,
                    },
                    started.elapsed(),
                );
                emit_usage_event(
                    &state_c,
                    &request_id_c,
                    &model_id_c,
                    &requested_model_c,
                    &api_key_id_c,
                    &provider_key_id_c,
                    status,
                    started.elapsed(),
                    &usage,
                    &client_c,
                    attempt_c,
                    comp.guardrail_blocked,
                    // #932: input-side counts merged with the hold-back
                    // release's output-side counts.
                    {
                        let mut merged = input_redactions.clone();
                        crate::redact::merge_counts(&mut merged, comp.redacted_entity_counts);
                        merged
                    },
                    {
                        let mut merged = input_monitor_hits.clone();
                        merged.extend(comp.monitor_hits);
                        merged
                    },
                    captured_content.as_ref(),
                );
            },
        );
        let mut response = axum::response::Response::new(sse_body);
        response.headers_mut().insert(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        response.headers_mut().insert(
            axum::http::header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        );
        if let Ok(hv) = HeaderValue::from_str(request_id) {
            response
                .headers_mut()
                .insert(HeaderName::from_static("x-aisix-request-id"), hv);
        }
        return Ok(ResponseDispatchSuccess {
            response,
            provider: provider_label,
            usage: None,
            model_id: model_id.to_string(),
            provider_key_id,
            routing: RoutingTelemetry::default(),
            guardrail_blocked: false,
            usage_handled_by_stream: true,
            // The stream's end-of-stream emit carries the counts.
            output_redactions: crate::redact::RedactionCounts::new(),
            output_monitor_hits: Vec::new(),
            // The stream's end-of-stream emit carries the captured content.
            captured_content: None,
        });
    }

    // Non-streaming.
    let mut resp = bridge.chat(&chat, &ctx).await.map_err(|err| {
        if let Some((ttl, reason)) = crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
        {
            state.runtime_status.mark_cooldown(model_id, ttl, reason);
        }
        ProxyError::Bridge(err)
    })?;
    state.health.record_success(&model.display_name);
    state.runtime_status.mark_healthy(model_id);

    let usage = {
        let mut u = ResponseUsage {
            prompt_tokens: resp.usage.prompt_tokens,
            completion_tokens: resp.usage.completion_tokens,
            reasoning_tokens: resp.usage.reasoning_tokens,
            cached_prompt_tokens: resp.usage.cached_prompt_tokens,
            cache_creation_tokens: resp.usage.cache_creation_tokens,
            cache_read_tokens: resp.usage.cache_read_tokens,
            usage_estimated: false,
        };
        // Token-estimation fallback (AISIX-Cloud#1074): fill counters the
        // bridged upstream never reported. Telemetry only — the re-encoded
        // Responses JSON below carries the upstream's own usage.
        if u.prompt_tokens == 0 || u.completion_tokens == 0 {
            let est = crate::token_estimate::Estimator::new(
                model.upstream_model().unwrap_or("unknown"),
                crate::token_estimate::PromptInput::Responses(body.clone()),
            );
            let filled = crate::token_estimate::fill_missing(
                &est,
                u.prompt_tokens,
                u.completion_tokens,
                Some(&crate::chat::estimation_output_text(&resp)),
            );
            if filled.estimated {
                u.prompt_tokens = filled.prompt_tokens;
                u.completion_tokens = filled.completion_tokens;
                u.usage_estimated = true;
            }
        }
        u
    };

    // #719: run output guardrails on the bridged response before re-encoding
    // it as Responses JSON — the assistant text + tool calls are
    // client-visible output, scanned the same way /v1/chat/completions does.
    let mut output_seg_counts = crate::redact::RedactionCounts::new();
    let mut output_monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    if aisix_guardrails::Guardrail::runs_on_output(chain.as_ref()) {
        let (verdict, hits) =
            aisix_guardrails::Guardrail::check_output_non_segment_observed(chain.as_ref(), &resp)
                .await;
        output_monitor_hits.extend(hits);
        let verdict = crate::redact::moderate_body(
            chain.as_ref(),
            crate::redact::Direction::Output,
            verdict,
            &mut output_seg_counts,
            &mut output_monitor_hits,
            |g| crate::redact::redact_chat_response(g, &mut resp),
        )
        .await;
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } = verdict
        {
            tracing::warn!(
                guardrail_hook = "output",
                model = %requested_model,
                reason = %reason,
                "guardrail blocked /v1/responses (cross-provider) response",
            );
            // #543: the upstream already billed — return the 422 body but
            // carry the billed usage (marked guardrail_blocked) so the
            // ledger doesn't underreport spend.
            return Ok(ResponseDispatchSuccess {
                response: ProxyError::ContentFiltered(crate::error::guardrail_block_message(
                    "response",
                    guardrail_name.as_deref(),
                ))
                .into_response(),
                provider: provider_label,
                usage: Some(usage),
                model_id: model_id.to_string(),
                provider_key_id: provider_key_id.clone(),
                routing: RoutingTelemetry::default(),
                guardrail_blocked: true,
                usage_handled_by_stream: false,
                output_redactions: crate::redact::RedactionCounts::new(),
                // Hits observed by the blocking check still count.
                output_monitor_hits,
                // AISIX-Cloud#1013: the billed-then-blocked event carries
                // the (post-mask) prompt; the blocked output itself stays
                // out of the log — blocking it and then archiving it would
                // defeat the block.
                captured_content: match (&captured_prompt, content_cap) {
                    (Some(p), Some(cap)) => Some(CapturedContent::new(p, "", cap as usize)),
                    _ => None,
                },
            });
        }
    }

    // #932: mask-action PII rules rewrite the bridged response AFTER the
    // block check passes, BEFORE it is re-encoded as Responses JSON.
    let mut output_redactions = crate::redact::redact_chat_response(chain.as_ref(), &mut resp);
    crate::redact::merge_counts(&mut output_redactions, output_seg_counts);

    let created_at = chrono::Utc::now().timestamp();
    let json_body = crate::responses_bridge::chat_response_to_responses_json(
        &resp,
        requested_model,
        created_at,
    );
    // Content capture (AISIX-Cloud#947): the client-visible Responses JSON
    // (post-redaction) is the source, so the exported text matches what the
    // caller received.
    let captured_content = match (&captured_prompt, content_cap) {
        (Some(prompt), Some(cap)) => Some(CapturedContent::new(
            prompt,
            &responses_output_text(&json_body),
            cap as usize,
        )),
        _ => None,
    };
    let mut response = Json(json_body).into_response();
    if let Ok(hv) = HeaderValue::from_str(request_id) {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-aisix-request-id"), hv);
    }
    Ok(ResponseDispatchSuccess {
        response,
        provider: provider_label,
        usage: Some(usage),
        model_id: model_id.to_string(),
        provider_key_id,
        routing: RoutingTelemetry::default(),
        guardrail_blocked: false,
        usage_handled_by_stream: false,
        output_redactions,
        output_monitor_hits,
        captured_content,
    })
}

/// Pull the usage counters out of a Responses-API non-streaming
/// response body. Returns `None` only when:
///   - The `usage` block is missing entirely, OR
///   - `usage.input_tokens` is missing / non-numeric
///
/// Those cases skip UsageEvent emission rather than attributing a
/// zero-everything noise row to the api_key. The `input_tokens` gate
/// distinguishes "no upstream usage at all" from a legitimate reply.
///
/// `output_tokens`, by contrast, defaults to 0 when absent: a 200 that
/// reports an input side but omits the output side is still a real
/// billable call and must be recorded. A missing completion/output side
/// coerces to 0 and the event is still logged/billed
/// (#429 follow-up; mirrors the tolerant wire-layer decode of
/// #474). Spec:
/// <https://platform.openai.com/docs/api-reference/responses/object>
fn extract_response_usage(body: &Value) -> Option<ResponseUsage> {
    let usage = body.get("usage")?;
    let prompt_tokens = usage.get("input_tokens").and_then(|v| v.as_u64())? as u32;
    let completion_tokens = usage
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let reasoning_tokens = usage
        .get("output_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let cached_prompt_tokens = usage
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    Some(ResponseUsage {
        prompt_tokens,
        completion_tokens,
        usage_estimated: false,
        reasoning_tokens,
        cached_prompt_tokens,
        // OpenAI verbatim path: no Anthropic-style cache counters.
        cache_creation_tokens: 0,
        cache_read_tokens: 0,
    })
}

/// Pull usage out of one parsed Responses-API SSE event if it is a terminal
/// event that carries the authoritative `usage` block (#808). The full usage
/// rides `response.completed`, and **also `response.incomplete` /
/// `response.failed`** which fire on `max_output_tokens` truncation or
/// cancellation — billing those keeps streaming parity with non-streaming.
/// The `usage` lives under the nested `response` object (unlike the
/// non-streaming body where it is top-level), so the same `extract_response_usage`
/// gate is applied to `json.response`.
/// <https://platform.openai.com/docs/api-reference/responses-streaming>
fn parse_responses_terminal_usage(json: &Value) -> Option<ResponseUsage> {
    matches!(
        json.get("type").and_then(|t| t.as_str()),
        Some("response.completed" | "response.incomplete" | "response.failed")
    )
    .then(|| json.get("response").and_then(extract_response_usage))
    .flatten()
}

/// Scan a fully-buffered Responses-API SSE body for the terminal event's
/// usage block (#808). Used by the buffered output-guardrail path, which
/// already holds the whole response. Returns `None` (skip emission, matching
/// the non-streaming gate) when no terminal event carried a usage block.
fn responses_sse_usage(bytes: &[u8]) -> Option<ResponseUsage> {
    let text = String::from_utf8_lossy(bytes);
    let mut usage = None;
    for line in text.lines() {
        let data = match line.strip_prefix("data:") {
            Some(d) => d.trim(),
            None => continue,
        };
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<Value>(data) {
            if let Some(u) = parse_responses_terminal_usage(&json) {
                usage = Some(u);
            }
        }
    }
    usage
}

/// Drain every complete SSE frame from `buf`, updating `acc` with the latest
/// terminal-event usage (#808) and feeding each parsed event to the optional
/// content capture (AISIX-Cloud#947). A frame ends at the first blank line;
/// an incomplete trailing frame is left in `buf` for the next chunk. Reuses
/// the shared SSE framing helpers from the `/v1/messages` passthrough so the
/// two surfaces parse identically.
fn drain_responses_sse_frames(
    buf: &mut Vec<u8>,
    acc: &mut Option<ResponseUsage>,
    mut capture: Option<&mut SseTextCapture>,
) {
    while let Some(end) = crate::messages::find_frame_end(buf) {
        let frame: Vec<u8> = buf.drain(..end).collect();
        if let Some(data) = crate::messages::extract_sse_data_line(&frame) {
            if data == b"[DONE]" {
                continue;
            }
            if let Ok(json) = serde_json::from_slice::<Value>(data) {
                if let Some(u) = parse_responses_terminal_usage(&json) {
                    *acc = Some(u);
                }
                if let Some(c) = capture.as_deref_mut() {
                    c.observe(&json);
                }
            }
        }
    }
}

/// Streamed output-text accumulator for content-capturing exporters
/// (AISIX-Cloud#947). Mirrors `responses_sse_output_text`'s precedence: a
/// terminal `response.*` event's full output (incl. tool-call items) wins;
/// concatenated `*.delta` text is the fallback for streams that abort before
/// a terminal object. Delta accumulation is bounded to the capture cap so a
/// long stream can't grow the buffer without limit.
struct SseTextCapture {
    cap: usize,
    deltas: String,
    terminal: Option<String>,
}

impl SseTextCapture {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            deltas: String::new(),
            terminal: None,
        }
    }

    /// Feed one parsed SSE event's JSON.
    fn observe(&mut self, json: &Value) {
        match json.get("type").and_then(|t| t.as_str()) {
            Some("response.completed" | "response.incomplete" | "response.failed") => {
                if let Some(resp) = json.get("response") {
                    let full = responses_output_text(resp);
                    if !full.is_empty() {
                        self.terminal = Some(full);
                    }
                }
            }
            Some(
                "response.output_text.delta"
                | "response.function_call_arguments.delta"
                | "response.mcp_call_arguments.delta"
                | "response.custom_tool_call_input.delta",
            ) => {
                if let Some(d) = json.get("delta").and_then(|d| d.as_str()) {
                    if self.deltas.len() < self.cap {
                        self.deltas.push_str(d);
                    }
                }
            }
            _ => {}
        }
    }

    /// The captured output text: terminal full output when seen, else the
    /// accumulated deltas. `CapturedContent::new` re-truncates to the cap.
    fn into_text(self) -> String {
        self.terminal.unwrap_or(self.deltas)
    }

    /// [`Self::into_text`] without consuming — the end-of-stream scan reads
    /// the text while the completion guard stays armed (a client disconnect
    /// mid-scan must still fire the guard's Drop emit with the captured
    /// text), so it clones instead of taking.
    fn text(&self) -> String {
        self.terminal.clone().unwrap_or_else(|| self.deltas.clone())
    }
}

/// Drop guard that fires `on_complete` exactly once with the usage parsed from
/// the stream's terminal SSE event — on normal end-of-stream AND on
/// client-disconnect (the async-stream generator drops at its suspension
/// point), mirroring the `/v1/messages` and chat.rs CompleteOnDrop pattern.
/// `None` means no terminal usage was seen (e.g. an abort before completion);
/// the emit then records a zero-token 200 so the request still appears in the
/// dashboard Logs. The second callback argument is the captured output text
/// (AISIX-Cloud#947) — empty when no exporter wants content. The third is the
/// end-of-stream scan's monitor observations (AISIX-Cloud#1010) — the Drop
/// (disconnect) path passes none: the response never completed, so there is
/// nothing final to observe, matching the chat surface's disconnect behavior.
struct ResponsesUsageGuard<F: FnOnce(ResponseUsage, String, Vec<aisix_core::GuardrailMonitorHit>)> {
    slot: Option<(F, Option<ResponseUsage>, Option<SseTextCapture>)>,
}

impl<F: FnOnce(ResponseUsage, String, Vec<aisix_core::GuardrailMonitorHit>)>
    ResponsesUsageGuard<F>
{
    fn parts(&mut self) -> (&mut Option<ResponseUsage>, Option<&mut SseTextCapture>) {
        let slot = self
            .slot
            .as_mut()
            .expect("ResponsesUsageGuard accessed after take");
        (&mut slot.1, slot.2.as_mut())
    }
}

impl<F: FnOnce(ResponseUsage, String, Vec<aisix_core::GuardrailMonitorHit>)> Drop
    for ResponsesUsageGuard<F>
{
    fn drop(&mut self) {
        if let Some((f, usage, capture)) = self.slot.take() {
            f(
                usage.unwrap_or_default(),
                capture.map(SseTextCapture::into_text).unwrap_or_default(),
                Vec::new(),
            );
        }
    }
}

/// End-of-stream output observation for the live-forward verbatim path
/// (AISIX-Cloud#1010). Reachable only when the output-hook chain's resolved
/// streaming policy is `EndOfStreamCheck` — today that is exactly the
/// monitor-only chains, which can never block. Runs the same two-phase scan
/// as the buffered branch (blob check + segment pass) so would-block /
/// would-mask hits reach telemetry; the bytes are already on the wire, so a
/// `Block` verdict (unreachable for monitor members) is logged, not enforced.
struct EosOutputScan {
    chain: Arc<aisix_guardrails::GuardrailChain>,
    upstream_model: String,
}

impl EosOutputScan {
    async fn observe(self, text: &str) -> Vec<aisix_core::GuardrailMonitorHit> {
        // Bound the provider calls the same way the buffered branch's byte
        // cap does — scan at most the cap's worth of text.
        let mut end = text
            .len()
            .min(aisix_guardrails::DEFAULT_STREAM_OUTPUT_BUFFER_BYTES);
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        let scan_text = &text[..end];
        if scan_text.is_empty() {
            return Vec::new();
        }
        let synth = synth_chat_response(&self.upstream_model, scan_text.to_string());
        let (verdict, mut hits) = aisix_guardrails::Guardrail::check_output_non_segment_observed(
            self.chain.as_ref(),
            &synth,
        )
        .await;
        // Segment pass (bedrock/lakera/presidio members): offer the flattened
        // text as one segment so monitor-mode segment moderators record their
        // observations too. Masks are suppressed in monitor mode, and nothing
        // could be rewritten anyway — the counts are discarded.
        let mut seg_counts = crate::redact::RedactionCounts::new();
        let verdict = crate::redact::moderate_body(
            self.chain.as_ref(),
            crate::redact::Direction::Output,
            verdict,
            &mut seg_counts,
            &mut hits,
            |g| {
                let _ = g.redact_output_text(scan_text);
                crate::redact::RedactionCounts::new()
            },
        )
        .await;
        if let aisix_guardrails::GuardrailVerdict::Block { reason, .. } = verdict {
            tracing::warn!(
                guardrail_hook = "output",
                model = %self.upstream_model,
                reason = %reason,
                "output guardrail returned a block after live forward; \
                 response already sent (EndOfStreamCheck policy)",
            );
        }
        hits
    }
}

/// Wrap a Responses-API upstream byte stream so the terminal event's usage is
/// parsed in-flight and `on_complete` fires once at end-of-stream (or
/// client-disconnect) with the accumulated counts (#808) plus the captured
/// output text (AISIX-Cloud#947, empty when `content_cap` is `None`) and the
/// end-of-stream scan's monitor hits (AISIX-Cloud#1010, empty without
/// `eos_scan`). Bytes forward verbatim — the client sees the exact upstream
/// SSE wire shape.
fn build_responses_passthrough_stream<S, F>(
    upstream: S,
    content_cap: Option<u32>,
    eos_scan: Option<EosOutputScan>,
    on_complete: F,
) -> impl futures::Stream<Item = reqwest::Result<bytes::Bytes>>
where
    S: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
    F: FnOnce(ResponseUsage, String, Vec<aisix_core::GuardrailMonitorHit>) + Send + 'static,
{
    // The scan and the token-estimation fallback read the same assembled
    // output text the capture produces, so the accumulator is now ALWAYS
    // on (AISIX-Cloud#1074) — whether estimation is needed is only known
    // at end-of-stream — with the estimation cap as the floor. The cap
    // bounds delta accumulation; a terminal event's full output text is
    // instead bounded by MAX_SSE_FRAME_BUF_BYTES (an oversized frame
    // never parses) and re-truncated by each consumer —
    // `CapturedContent::new` at the exporter cap and
    // `EosOutputScan::observe` at the scan bound — so none sees beyond
    // its own limit.
    let capture_cap = Some(
        content_cap
            .map(|cap| cap as usize)
            .unwrap_or(0)
            .max(if eos_scan.is_some() {
                aisix_guardrails::DEFAULT_STREAM_OUTPUT_BUFFER_BYTES
            } else {
                0
            })
            .max(crate::token_estimate::OUTPUT_ACCUMULATION_CAP),
    );
    // Re-attach the request span: the body is polled after the request-id
    // middleware returns, so the end-of-stream output-guardrail scan
    // (`EosOutputScan::observe`) would otherwise log without a
    // `request_id` (AISIX-Cloud#1060).
    crate::request_id::in_request_span(async_stream::stream! {
        let mut guard = ResponsesUsageGuard {
            slot: Some((
                on_complete,
                None,
                capture_cap.map(SseTextCapture::new),
            )),
        };
        futures::pin_mut!(upstream);
        let mut buf: Vec<u8> = Vec::new();
        while let Some(item) = upstream.next().await {
            if let Ok(bytes) = &item {
                // Side-channel parse: copy into the frame buffer (the original
                // `bytes` is yielded unchanged below) and drain complete frames.
                buf.extend_from_slice(bytes);
                let (usage_acc, capture) = guard.parts();
                drain_responses_sse_frames(&mut buf, usage_acc, capture);
                // Bound the frame buffer: the happy path drains complete frames
                // above so `buf` only holds a partial trailing frame. A
                // non-conformant upstream streaming bytes without a blank-line
                // terminator would otherwise grow `buf` unboundedly; drop it
                // (losing usage parsing for that pathological case) rather than
                // OOM. Bytes still forward verbatim — only telemetry is affected.
                if buf.len() > crate::messages::MAX_SSE_FRAME_BUF_BYTES {
                    tracing::warn!(
                        buffered = buf.len(),
                        "responses stream: SSE frame buffer exceeded cap without a \
                         terminator; dropping buffer (usage parsing skipped)"
                    );
                    buf.clear();
                }
            }
            // Forward the original item verbatim (Ok bytes OR a mid-stream Err).
            yield item;
        }
        // Clean end-of-stream: run the monitor observation (needs async, so
        // it can't live in the Drop guard), then complete explicitly. The
        // scan awaits a remote guardrail provider, and SDK clients routinely
        // close the connection right after the terminal frame — dropping
        // this generator at that await. The guard therefore MUST stay armed
        // across the scan: its Drop then still emits the usage event (with
        // the captured text, without hits — same as any disconnect). Taking
        // the slot before the await would silently lose the event for a
        // fully-delivered stream.
        let hits = match eos_scan {
            Some(scan) => {
                let text = guard.parts().1.map(|c| c.text()).unwrap_or_default();
                scan.observe(&text).await
            }
            None => Vec::new(),
        };
        if let Some((f, usage, capture)) = guard.slot.take() {
            f(
                usage.unwrap_or_default(),
                capture.map(SseTextCapture::into_text).unwrap_or_default(),
                hits,
            );
        }
    })
}

/// Collect the assistant's output text from a Responses-API response object
/// for output-guardrail scanning (#719/#546):
/// - the `text` of every `output_text` content part of message items, and
/// - the `name` + `arguments` (function calls) / `input` (custom tool calls)
///   of tool-call items — these are **top-level** item fields, not under
///   `content[]`, so without scanning them a blocked literal placed in a
///   tool-call's arguments would bypass the output guardrail. The chat
///   surface scans tool-call output too (`ChatResponse::guardrail_output_text`,
///   the #448 fix); this keeps the surfaces symmetric.
///
/// Reasoning items are intentionally excluded (out of output-guardrail
/// scope, matching the chat surface) — they carry `summary`, not `content`
/// / `arguments`, so they're naturally skipped.
/// <https://platform.openai.com/docs/api-reference/responses/object>
fn responses_output_text(resp: &Value) -> String {
    let Some(items) = resp.get("output").and_then(|v| v.as_array()) else {
        return String::new();
    };
    let mut parts: Vec<&str> = Vec::new();
    for it in items {
        if let Some(content) = it.get("content").and_then(|c| c.as_array()) {
            parts.extend(
                content
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str())),
            );
        }
        // Tool-call items carry caller-visible model output under top-level
        // `name`/`arguments` (function_call) or `name`/`input` (custom tool).
        for key in ["name", "arguments", "input"] {
            if let Some(s) = it.get(key).and_then(|v| v.as_str()) {
                parts.push(s);
            }
        }
    }
    parts
        .into_iter()
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Collect the assistant's streamed output text from a buffered
/// Responses-API SSE response (#719/#546). Prefers the authoritative full
/// output carried on a terminal `response` event — `response.completed`,
/// **and also `response.incomplete` / `response.failed`**, which carry the
/// same full `output[]` (incl. tool-call items) and fire routinely (e.g.
/// `max_output_tokens` truncation). Falls back to concatenating the streamed
/// deltas when no terminal `response` object is present (truncated/aborted):
/// both `response.output_text.delta` (assistant text) and
/// `response.function_call_arguments.delta` (tool-call args stream via their
/// own event, NOT output_text) — otherwise blocked tool-call args would leak
/// on a stream that never reaches a terminal object. The `type` field on each
/// `data:` JSON line drives the dispatch.
/// <https://platform.openai.com/docs/api-reference/responses-streaming>
fn responses_sse_output_text(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut deltas = String::new();
    for line in text.lines() {
        let data = match line.strip_prefix("data:") {
            Some(d) => d.trim(),
            None => continue,
        };
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let json: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match json.get("type").and_then(|t| t.as_str()) {
            Some("response.completed" | "response.incomplete" | "response.failed") => {
                if let Some(resp) = json.get("response") {
                    let full = responses_output_text(resp);
                    if !full.is_empty() {
                        return full;
                    }
                }
            }
            Some("response.output_text.delta") => {
                if let Some(d) = json.get("delta").and_then(|d| d.as_str()) {
                    deltas.push_str(d);
                }
            }
            // Tool-call argument deltas across all tool kinds — function calls,
            // MCP tool calls, and custom tools each stream their args/input via
            // their own event, not output_text.delta. On a terminal `response`
            // object these are already covered by responses_output_text; this
            // matters only when the stream aborts before any terminal object.
            // Concatenate WITHOUT a separator — these are pieces of one call's
            // string; a separator would split a literal that streamed across
            // two deltas (e.g. "BLOCK"+"ME") and miss the match.
            Some(
                "response.function_call_arguments.delta"
                | "response.mcp_call_arguments.delta"
                | "response.custom_tool_call_input.delta",
            ) => {
                if let Some(d) = json.get("delta").and_then(|d| d.as_str()) {
                    deltas.push_str(d);
                }
            }
            _ => {}
        }
    }
    deltas
}

/// Build the minimal internal `ChatResponse` an output guardrail needs to
/// scan: the assistant text in `message.content`. Only the text is read by
/// `check_output` (via `guardrail_output_text`); the other fields are
/// placeholders and never reach the client.
fn synth_chat_response(model: &str, text: String) -> ChatResponse {
    ChatResponse {
        id: String::new(),
        model: model.to_string(),
        message: ChatMessage::assistant(text),
        finish_reason: FinishReason::Stop,
        usage: UsageStats::default(),
    }
}

/// Copy the upstream `content-type` onto the client response and stamp the
/// `x-aisix-request-id` header. Shared by the streaming verbatim-passthrough
/// and buffered hold-back paths.
fn apply_passthrough_headers(
    response: &mut Response,
    upstream_headers: &axum::http::HeaderMap,
    request_id: &str,
) {
    if let Some(ct) = upstream_headers.get("content-type") {
        if let Ok(hv) = HeaderValue::from_bytes(ct.as_bytes()) {
            response
                .headers_mut()
                .insert(axum::http::header::CONTENT_TYPE, hv);
        }
    }
    if let Ok(hv) = HeaderValue::from_str(request_id) {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-aisix-request-id"), hv);
    }
}

/// Issue #404: push one `UsageEvent` onto cp-api's telemetry sink
/// and fan it out to per-env OTLP exporters. Mirrors the shape of
/// `embeddings::emit_usage_event` (#402) for the fields that matter
/// to /v1/responses, with one extension: `reasoning_tokens` is
/// surfaced for o1/o3/GPT-5 class models. `inbound_protocol` is
/// `"openai"` — the Responses API is OpenAI-shaped on the wire even when
/// the resolved model is bridged to a non-OpenAI provider (#825).
///
/// Other fields left at `UsageEvent::default()`:
///   - cache_creation_tokens / cache_read_tokens — populated only on the
///     #825 cross-provider bridge path (Anthropic backends); 0 otherwise
///   - provider_request_id / provider_model_version / finish_reason
///     — not yet plumbed for non-chat handlers (follow-up)
///   - cost_usd — cp-api computes server-side from pricing catalog
///   - cache_status / cache_hit_* / ttft_ms — no caching/streaming
///     surface on Responses API non-streaming
///   - served_by_model / routing_* — Responses doesn't run routing
///
/// `provider_kind` / `provider_featured` / `branded_provider` / `pk_label` /
/// `byo_label` are populated from the resolved target's ProviderKey
/// `telemetry_tags` (AISIX-Cloud#867) — same lookup as `/v1/messages` and
/// `/v1/chat/completions`, so Codex (`/v1/responses`) logs carry the upstream
/// vendor + PK label the dashboard's Logs detail shows. Empty `provider_key_id`
/// (pre-dispatch error) bypasses the lookup → wire NULL.
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    requested_model: &str,
    api_key_id: &str,
    provider_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    usage: &ResponseUsage,
    client: &ClientContext,
    attempt: AttemptInfo,
    guardrail_blocked: bool,
    // Per-detector PII mask counts (#932), input + output merged. Detector
    // names only, never matched values. Empty = no redaction.
    redacted_entity_counts: crate::redact::RedactionCounts,
    // Monitor-mode guardrail observations (AISIX-Cloud#562), input +
    // output merged.
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    // Captured request/response content for content-capturing exporters
    // (AISIX-Cloud#947). Forwarded only to `fan_out`, never to the CP sink.
    content: Option<&CapturedContent>,
) {
    let snap = state.snapshot.load();
    let tags = provider_telemetry_tags(&snap, provider_key_id);
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        cached_prompt_tokens: usage.cached_prompt_tokens,
        reasoning_tokens: usage.reasoning_tokens,
        // Anthropic cache counters (#825 cross-provider path); 0 on the
        // verbatim OpenAI path.
        cache_creation_tokens: usage.cache_creation_tokens,
        cache_read_tokens: usage.cache_read_tokens,
        usage_estimated: usage.usage_estimated,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "openai".to_string(),
        attempt_index: attempt.index,
        attempt_kind: attempt.kind,
        attempt_model: attempt.model,
        error_class: attempt.error_class,
        error_message: attempt.error_message,
        provider_kind: sanitize_tag(tags.kind.map(|k| k.as_str().to_owned()).unwrap_or_default()),
        provider_featured: tags.featured,
        branded_provider: sanitize_tag(tags.branded_provider.unwrap_or_default()),
        pk_label: sanitize_tag(tags.pk_label.unwrap_or_default()),
        byo_label: sanitize_tag(tags.byo_label.unwrap_or_default()),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        guardrail_blocked,
        redacted_entity_counts,
        guardrail_monitor_hits,
        ..Default::default()
    };
    state.usage_sink.try_emit("responses", event.clone());
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, content, exporters.iter().map(|e| &e.value));
    // AISIX-Cloud#1044: token volume by inbound client type × model. Codex
    // traffic arrives on /v1/responses, so leaving this endpoint out of the
    // by-client series made an allowlisted client invisible in it. All three
    // usage-bearing paths (non-streaming, verbatim streaming, bridge
    // streaming) funnel through here. `requested_model` resolved at dispatch
    // on every path that reaches this emit, so the label is bounded by the
    // configured model set. The per-key `aisix_llm_*_tokens_total` family
    // intentionally stays chat/messages-scoped (cross-API audit #646-652).
    // #1002: cache-inclusive total via the shared helper — cache counters are
    // non-zero only on the #825 Anthropic bridge path.
    state.metrics.record_llm_tokens_by_client(
        state.client_classifier.classify(&client.user_agent),
        requested_model,
        u64::from(usage.prompt_tokens),
        u64::from(usage.completion_tokens),
        total_tokens_with_cache(
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.cache_creation_tokens,
            usage.cache_read_tokens,
        ),
    );
}

/// Emit a zero-token `UsageEvent` for a failed / pre-dispatch attempt
/// (#655). Tokens stay 0; `status_code` + `error_*` carry the failure.
#[allow(clippy::too_many_arguments)]
fn emit_zero_token_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    requested_model: &str,
    api_key_id: &str,
    provider_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    client: &ClientContext,
    attempt: AttemptInfo,
    // Per-detector PII mask counts (#932): input masking may have fired
    // before the failure. Empty for most failure classes.
    redacted_entity_counts: crate::redact::RedactionCounts,
    // Monitor-mode guardrail observations (AISIX-Cloud#562) that fired
    // before the failure.
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    // AISIX-Cloud#1013: captured (post-mask) request body for failed
    // requests. Forwarded only to `fan_out`, never to the CP sink.
    content: Option<CapturedContent>,
) {
    let snap = state.snapshot.load();
    let tags = provider_telemetry_tags(&snap, provider_key_id);
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        redacted_entity_counts,
        guardrail_monitor_hits,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "openai".to_string(),
        attempt_index: attempt.index,
        attempt_kind: attempt.kind,
        attempt_model: attempt.model,
        error_class: attempt.error_class,
        error_message: attempt.error_message,
        provider_kind: sanitize_tag(tags.kind.map(|k| k.as_str().to_owned()).unwrap_or_default()),
        provider_featured: tags.featured,
        branded_provider: sanitize_tag(tags.branded_provider.unwrap_or_default()),
        pk_label: sanitize_tag(tags.pk_label.unwrap_or_default()),
        byo_label: sanitize_tag(tags.byo_label.unwrap_or_default()),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        ..Default::default()
    };
    state.usage_sink.try_emit("responses", event.clone());
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, content.as_ref(), exporters.iter().map(|e| &e.value));
}

/// Emit one zero-token `UsageEvent` per FAILED attempt of a `/v1/responses`
/// request (#655). The winner / terminal event is emitted separately.
#[allow(clippy::too_many_arguments)]
fn emit_failed_attempts(
    state: &ProxyState,
    request_id: &str,
    requested_model: &str,
    api_key_id: &str,
    client: &ClientContext,
    routing: &RoutingTelemetry,
    // AISIX-Cloud#1013: when every target failed there is no terminal
    // event, so the captured request body rides the LAST failed attempt —
    // the one whose status the caller saw. Other attempts (and the
    // success-path caller) stay content-less.
    mut content_for_last: Option<CapturedContent>,
) {
    let last_failed = routing.attempts.iter().rposition(|a| !a.success);
    for (i, rec) in routing
        .attempts
        .iter()
        .enumerate()
        .filter(|(_, a)| !a.success)
    {
        let content = if Some(i) == last_failed {
            content_for_last.take()
        } else {
            None
        };
        emit_zero_token_event(
            state,
            request_id,
            // Each failed attempt records the TARGET it actually hit
            // (AISIX-Cloud#790), not the group it was resolved from.
            &rec.target_model_id,
            requested_model,
            api_key_id,
            &rec.provider_key_id,
            rec.status,
            Duration::from_millis(u64::from(rec.latency_ms)),
            client,
            AttemptInfo::from_record(rec),
            // Failed attempts carry no per-request redaction detail; the
            // terminal event does.
            crate::redact::RedactionCounts::new(),
            Vec::new(),
            content,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_access_log(
    model: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    elapsed: Duration,
    request_id: &str,
    routing: &RoutingTelemetry,
    error: Option<&ProxyError>,
) {
    let (error_kind, error) = match error {
        Some(e) => {
            let (kind, msg) = crate::attempt::access_log_error(e);
            (Some(kind), Some(msg))
        }
        None => (None, None),
    };
    // Per #655 the access log stays ONE line per request, carrying the
    // user-perceived `latency` + final status plus a routing summary.
    let served_by = routing
        .winner()
        .map(|w| w.target_model.as_str())
        .filter(|s| !s.is_empty());
    AccessLog {
        method: "POST",
        path: "/v1/responses",
        status,
        latency: elapsed,
        provider: Some(provider),
        model: Some(model),
        api_key_id: Some(api_key_id),
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id,
        served_by_model: served_by,
        routing_attempt_count: match routing.attempt_count() {
            0 => None,
            n => Some(n),
        },
        routing_fallback_count: match routing.fallback_count() {
            0 => None,
            n => Some(n),
        },
        error_kind,
        error: error.as_deref(),
    }
    .emit();
}

#[cfg(test)]
mod tests {

    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use aisix_provider_anthropic::AnthropicBridge;
    use aisix_provider_openai::OpenAiBridge;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    const OPENAI_PK_ID: &str = "11111111-1111-1111-1111-111111111111";
    const ANTHROPIC_PK_ID: &str = "22222222-2222-2222-2222-222222222222";

    fn openai_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"openai","model_name":"gpt-4o","provider_key_id":"{OPENAI_PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn anthropic_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"anthropic","model_name":"claude-3-haiku-20240307","provider_key_id":"{ANTHROPIC_PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-2", m, 1)
    }

    fn openai_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-test","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(OPENAI_PK_ID, pk, 1)
    }

    /// An OpenAI PK carrying per-PK `request.*` overrides (AISIX-Cloud#867):
    /// a `default_body_fields` injection and a `default_headers` injection,
    /// so the verbatim Responses path can be asserted to apply both to the
    /// outbound upstream call.
    fn openai_pk_with_overrides(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-test","api_base":"{api_base}","provider":"openai","adapter":"openai","request":{{"default_body_fields":{{"safe_flag":true}},"default_headers":{{"x-custom":"trace-on"}}}}}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(OPENAI_PK_ID, pk, 1)
    }

    /// An OpenAI PK carrying per-PK telemetry attribution tags
    /// (AISIX-Cloud#867) so emitted UsageEvents can be asserted to surface the
    /// upstream vendor + PK label the dashboard's Logs detail shows.
    fn openai_pk_tagged(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-test","api_base":"{api_base}","provider":"openai","adapter":"openai","telemetry_tags":{{"kind":"catalog","featured":true,"branded_provider":"openai","pk_label":"prod-codex-key"}}}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(OPENAI_PK_ID, pk, 1)
    }

    fn new_snap_openai_tagged(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(openai_pk_tagged(api_base));
        snap
    }

    fn anthropic_pk_at(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"anthropic-up","secret":"sk-ant-test","api_base":"{api_base}","provider":"anthropic","adapter":"anthropic"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(ANTHROPIC_PK_ID, pk, 1)
    }

    fn new_snap_openai(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(openai_pk(api_base));
        snap
    }

    fn new_snap_anthropic_at(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk_at(api_base));
        snap
    }

    fn apikey_entry(allowed: &[&str]) -> ResourceEntry<ApiKey> {
        let json = format!(
            r#"{{"key_hash":"8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891","allowed_models":{}}}"#,
            serde_json::to_string(&allowed).unwrap()
        );
        let k: ApiKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("k-1", k, 1)
    }

    fn build_app(snap: AisixSnapshot) -> axum::Router {
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        // #825: the cross-provider /v1/responses path bridges non-OpenAI
        // targets through the provider Bridge; register Anthropic so those
        // tests resolve a bridge.
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let handle = SnapshotHandle::new(snap);
        crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    fn make_req(body: serde_json::Value) -> Request<axum::body::Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    /// An env-scoped keyword input guardrail (no attachment row → applies to
    /// every request via the backward-compat fallback) that blocks on a
    /// literal substring. Keyword is local (no remote call), so it's the
    /// deterministic stand-in for any input-hook guardrail kind.
    fn keyword_input_guardrail(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"test-block","enabled":true,"hook_point":"input","fail_open":false,"kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-1", g, 1)
    }

    /// #719 (the core fix): a configured INPUT guardrail that blocks on
    /// /v1/chat/completions must also fire on /v1/responses. The same blocked
    /// input must return 422 content_filter here — not 200 with the input
    /// echoed back — and the upstream must never be contacted (`expect(0)`).
    #[tokio::test]
    async fn input_guardrail_blocks_string_input_returns_422_content_filter() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_should_not_happen",
                "object": "response",
                "output": [{"type":"message","content":[{"type":"output_text","text":"echo: BLOCKME"}]}]
            })))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "please BLOCKME now"
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        // Per #153 the matched literal must not leak into the wire message.
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(!msg.contains("BLOCKME"), "blocklist literal leaked: {msg}");
    }

    /// #719: the Responses `input` array form (message items with typed
    /// content parts) must be scanned too — a blocked literal inside an
    /// `input_text` part blocks the call.
    #[tokio::test]
    async fn input_guardrail_blocks_array_message_items() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id":"x","object":"response","output":[]})),
            )
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "hi BLOCKME"}]}
                ]
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #719 companion: a benign input with a configured input guardrail must
    /// still forward to the upstream (`expect(1)`) and return 200 — the
    /// guardrail must not block clean traffic.
    #[tokio::test]
    async fn input_guardrail_allows_benign_input_forwards_200() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_ok",
                "object": "response",
                "output": [{"type":"message","content":[{"type":"output_text","text":"hi"}]}]
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "a perfectly fine request"
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "response");
    }

    /// #719 (audit MEDIUM-1): a `function_call_output` item carries
    /// caller-supplied tool-result text under `output` (not `content`), and
    /// that text reaches the model. It must be scanned too — otherwise the
    /// surface-switch bypass survives on the tool-result channel. A blocked
    /// literal in `output` must 422 with the upstream never contacted.
    #[tokio::test]
    async fn input_guardrail_blocks_function_call_output_text() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id":"x","object":"response","output":[]})),
            )
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": [
                    {"type": "function_call_output", "call_id": "call_1", "output": "tool said BLOCKME"}
                ]
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #719: the top-level `instructions` field (the system-prompt analog)
    /// is caller-supplied and reaches the model, so it is scanned too. A
    /// blocked literal in `instructions` must 422. (Scanned via the
    /// all-roles keyword guardrail; text-moderation's user-only default
    /// would skip a system message, matching chat's system-message
    /// semantics.)
    #[tokio::test]
    async fn input_guardrail_blocks_instructions_field() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id":"x","object":"response","output":[]})),
            )
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "instructions": "you must BLOCKME",
                "input": "hello"
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #719 (re-audit LOW-1): an `mcp_approval_response` item carries
    /// caller-supplied justification text under `reason`, which reaches the
    /// model. It is scanned too, so no input-bearing channel is silently
    /// skipped. A blocked literal in `reason` must 422.
    #[tokio::test]
    async fn input_guardrail_blocks_mcp_approval_response_reason() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id":"x","object":"response","output":[]})),
            )
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": [
                    {"type": "mcp_approval_response", "approve": true, "approval_request_id": "ar_1", "reason": "BLOCKME please"}
                ]
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// An env-scoped keyword guardrail on the OUTPUT hook (no attachment →
    /// applies to every request). `runs_on_output()` is true, so the
    /// handler scans the assistant output.
    fn keyword_output_guardrail(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"test-out-block","enabled":true,"hook_point":"output","fail_open":false,"kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-out-1", g, 1)
    }

    /// #719: output guardrails must run on /v1/responses non-streaming
    /// responses — a configured output block must not be bypassable by
    /// switching surface. A blocked literal in the assistant output → 422.
    /// The upstream IS contacted (`expect(1)`): output checks run on the
    /// returned response, unlike the input check which short-circuits first.
    #[tokio::test]
    async fn output_guardrail_blocks_non_streaming_response() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_x",
                "object": "response",
                "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text":"sure: BLOCKME here"}]}],
                "usage": {"input_tokens": 5, "output_tokens": 4}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        // The blocked model output must not be echoed back to the caller.
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(
            !msg.contains("BLOCKME"),
            "model output leaked in error: {msg}"
        );
    }

    /// #719 companion: a clean non-streaming response with an output
    /// guardrail configured passes through unchanged → 200 with body.
    #[tokio::test]
    async fn output_guardrail_allows_clean_non_streaming_response() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_ok",
                "object": "response",
                "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text":"a clean answer"}]}],
                "usage": {"input_tokens": 5, "output_tokens": 3}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["output"][0]["content"][0]["text"], "a clean answer");
    }

    /// #719: streaming /v1/responses must also enforce output guardrails —
    /// else `stream:true` bypasses the output block. The blocked content is
    /// held back (BufferFull): the client gets 422 and never the tokens.
    #[tokio::test]
    async fn output_guardrail_blocks_streaming_response_holds_back() {
        let upstream = MockServer::start().await;
        let sse = "event: response.output_text.delta\n\
                   data: {\"type\":\"response.output_text.delta\",\"delta\":\"sure: BLOCKME\"}\n\n\
                   event: response.completed\n\
                   data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"sure: BLOCKME\"}]}]}}\n\n\
                   data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        // The held-back content must never reach the client.
        assert!(
            !String::from_utf8_lossy(&bytes).contains("BLOCKME"),
            "streamed content leaked despite output block",
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #719 companion: a clean streaming response with an output guardrail
    /// is scanned then released in full → 200 + the SSE body.
    #[tokio::test]
    async fn output_guardrail_allows_clean_streaming_response() {
        let upstream = MockServer::start().await;
        let sse = "event: response.output_text.delta\n\
                   data: {\"type\":\"response.output_text.delta\",\"delta\":\"a clean answer\"}\n\n\
                   data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            body.contains("a clean answer"),
            "clean SSE body must be released in full",
        );
        assert!(
            body.starts_with("event: ") || body.starts_with("data: "),
            "SSE shape preserved on release",
        );
    }

    /// Anthropic Messages streaming SSE carrying a single text delta.
    fn anthropic_text_sse(text: &str) -> String {
        format!(
            "event: message_start\n\
             data: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_g\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-haiku-20240307\",\"content\":[],\"usage\":{{\"input_tokens\":5,\"output_tokens\":0}}}}}}\n\n\
             event: content_block_start\n\
             data: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
             event: content_block_delta\n\
             data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":{text}}}}}\n\n\
             event: content_block_stop\n\
             data: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
             event: message_delta\n\
             data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":3}}}}\n\n\
             event: message_stop\n\
             data: {{\"type\":\"message_stop\"}}\n\n",
            text = serde_json::to_string(text).unwrap(),
        )
    }

    /// #825 + #719: the cross-provider (bridged) streaming path must enforce
    /// output guardrails too — else `stream:true` against a non-OpenAI model
    /// bypasses the block. The bridge buffers the encoded SSE and, on a
    /// block, emits only a terminal `error` event; no output_text delta with
    /// the blocked literal reaches the client.
    #[tokio::test]
    async fn output_guardrail_blocks_streaming_cross_provider_response() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(anthropic_text_sse("sure: BLOCKME here")),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_at(&upstream.uri());
        snap.models.insert(anthropic_model("claude-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"claude-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        // The SSE 200 is committed by the first-chunk failover peek; the block
        // surfaces as an in-band terminal error event.
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            body.contains("content_filter"),
            "missing block error: {body}"
        );
        assert!(
            !body.contains("BLOCKME"),
            "blocked content leaked in stream: {body}"
        );
        assert!(
            !body.contains("response.output_text.delta"),
            "held-back deltas leaked: {body}"
        );
    }

    /// #825 companion: a clean bridged streaming response with an output
    /// guardrail is scanned then released in full.
    #[tokio::test]
    async fn output_guardrail_allows_clean_streaming_cross_provider_response() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(anthropic_text_sse("a clean answer")),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_at(&upstream.uri());
        snap.models.insert(anthropic_model("claude-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"claude-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            body.contains("a clean answer"),
            "clean body withheld: {body}"
        );
        assert!(body.contains("response.completed"));
    }

    /// #825: a blocked cross-provider STREAM still bills the upstream tokens
    /// but the emitted UsageEvent is marked guardrail_blocked (status 422) —
    /// matching the non-streaming path — so the dashboard's Blocked tab and
    /// the budget ledger see it rather than recording it as clean usage.
    #[tokio::test]
    async fn streaming_cross_provider_block_emits_guardrail_blocked_usage_event() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(anthropic_text_sse("sure: BLOCKME")),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_at(&upstream.uri());
        snap.models.insert(anthropic_model("claude-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"claude-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Drain the body so the stream's Drop guard fires the usage event.
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("BLOCKME"));

        let event = tokio::time::timeout(std::time::Duration::from_millis(1000), rx.recv())
            .await
            .expect("usage event must be emitted")
            .expect("usage_sink sender dropped");
        assert!(
            event.guardrail_blocked,
            "a blocked stream must mark guardrail_blocked"
        );
        assert_eq!(event.status_code, 422);
        // The upstream-billed tokens are still recorded.
        assert_eq!(event.prompt_tokens, 5);
        assert_eq!(event.completion_tokens, 3);
    }

    /// #719 (audit HIGH-1): the streaming hold-back buffer is capped so a
    /// huge (or malicious) upstream response can't OOM the gateway. A
    /// response exceeding the BufferFull cap fails closed (422) rather than
    /// being released unscanned — even when its content is otherwise clean.
    #[tokio::test]
    async fn output_guardrail_streaming_oversized_response_fails_closed() {
        let upstream = MockServer::start().await;
        // One delta larger than the 256 KiB default BufferFull cap.
        let big = "x".repeat(300_000);
        let sse =
            format!("data: {{\"type\":\"response.output_text.delta\",\"delta\":\"{big}\"}}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "oversized streamed response must fail closed, not be released unscanned",
        );
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// `keyword_output_guardrail` with `enforcement_mode: monitor` — the
    /// chain resolves to `EndOfStreamCheck` (never holds back, never blocks).
    fn keyword_output_guardrail_monitor(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"test-out-mon","enabled":true,"hook_point":"output","fail_open":false,"enforcement_mode":"monitor","kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-out-mon-1", g, 1)
    }

    /// AISIX-Cloud#1010: a MONITOR-mode output guardrail must never make a
    /// streaming /v1/responses request fail closed. Same oversized stream as
    /// the fail-closed test above, but the chain resolves to EndOfStreamCheck
    /// — the bytes forward live and the client gets the full 200 SSE, not a
    /// 422. Pre-fix, any output-hook guardrail (monitor included) forced the
    /// hold-back branch and a >256 KiB response was rejected with
    /// `content_filter` — a monitor rule "blocking", which it must never do.
    #[tokio::test]
    async fn monitor_output_guardrail_oversized_stream_released() {
        let upstream = MockServer::start().await;
        let big = "x".repeat(300_000);
        let sse =
            format!("data: {{\"type\":\"response.output_text.delta\",\"delta\":\"{big}\"}}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails
            .insert(keyword_output_guardrail_monitor("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a monitor-only chain must never fail a stream closed on the buffer cap",
        );
        let bytes = to_bytes(resp.into_body(), 4 * 1024 * 1024).await.unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            body.len() >= 300_000 && body.contains("[DONE]"),
            "the full SSE must be released, got {} bytes",
            body.len(),
        );
    }

    /// AISIX-Cloud#1010 companion: on the live-forward monitor path the
    /// end-of-stream scan still runs — a violating stream is delivered
    /// verbatim (200, content included) and the emitted usage event carries
    /// the `would_block` observation instead of `guardrail_blocked`.
    #[tokio::test]
    async fn monitor_output_guardrail_streams_live_and_records_would_block() {
        use aisix_obs::UsageSink;
        let upstream = MockServer::start().await;
        let sse = "event: response.output_text.delta\n\
                   data: {\"type\":\"response.output_text.delta\",\"delta\":\"sure: BLOCKME\"}\n\n\
                   event: response.completed\n\
                   data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"sure: BLOCKME\"}]}],\"usage\":{\"input_tokens\":5,\"output_tokens\":3}}}\n\n\
                   data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails
            .insert(keyword_output_guardrail_monitor("BLOCKME"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            String::from_utf8_lossy(&bytes).contains("BLOCKME"),
            "monitor mode must deliver the stream verbatim",
        );

        let event = tokio::time::timeout(std::time::Duration::from_millis(1000), rx.recv())
            .await
            .expect("usage event must be emitted")
            .expect("usage_sink sender dropped");
        assert!(!event.guardrail_blocked);
        assert_eq!(event.status_code, 200);
        assert!(
            event
                .guardrail_monitor_hits
                .iter()
                .any(|h| h.hook == "output" && h.action == "would_block"),
            "the end-of-stream scan must record the would-block observation, got {:?}",
            event.guardrail_monitor_hits,
        );
    }

    /// AISIX-Cloud#1010 audit H1: the end-of-stream observation awaits a
    /// remote guardrail provider, and SDK clients close the connection right
    /// after the terminal frame — the generator is dropped at that await.
    /// The completion guard must stay armed across the scan so the Drop
    /// still emits the usage event: a fully-delivered 200 stream must never
    /// lose its billing/logs record to a disconnect during the observation.
    #[tokio::test]
    async fn monitor_scan_disconnect_still_emits_usage_event() {
        use aisix_obs::UsageSink;
        use futures::StreamExt;
        let upstream = MockServer::start().await;
        let sse = "event: response.output_text.delta\n\
                   data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello there\"}\n\n\
                   event: response.completed\n\
                   data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello there\"}]}],\"usage\":{\"input_tokens\":5,\"output_tokens\":3}}}\n\n\
                   data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        // Remote moderation backend that never answers within the test —
        // parks the end-of-stream scan so the disconnect lands mid-await.
        let acs = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(30))
                    .set_body_json(serde_json::json!({
                        "categoriesAnalysis": [],
                        "blocklistsMatch": []
                    })),
            )
            .mount(&acs)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let textmod_json = format!(
            r#"{{"name":"textmod-mon","enabled":true,"kind":"azure_content_safety_text_moderation","hook_point":"output","enforcement_mode":"monitor","endpoint":"{}","api_key":"k"}}"#,
            acs.uri()
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&textmod_json).unwrap();
        snap.guardrails
            .insert(ResourceEntry::new("g-textmod-mon", g, 1));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Read frames until the terminal [DONE] is on the wire, then poll
        // once more so the generator advances past the loop and parks on the
        // remote-scan await — and drop the body there, like an SDK client
        // closing after the terminal frame.
        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while !String::from_utf8_lossy(&wire).contains("[DONE]") {
            let chunk =
                tokio::time::timeout(std::time::Duration::from_millis(2000), body_stream.next())
                    .await
                    .expect("stream must deliver the terminal frame promptly")
                    .expect("stream ended before [DONE]")
                    .expect("stream errored");
            wire.extend_from_slice(chunk.as_ref());
        }
        let parked =
            tokio::time::timeout(std::time::Duration::from_millis(300), body_stream.next()).await;
        assert!(
            parked.is_err(),
            "generator should be parked on the remote scan await",
        );
        drop(body_stream);

        let event = tokio::time::timeout(std::time::Duration::from_millis(1000), rx.recv())
            .await
            .expect("disconnect during the EOS scan must still emit the usage event")
            .expect("usage_sink sender dropped");
        assert_eq!(event.status_code, 200);
        assert!(!event.guardrail_blocked);
        assert_eq!(event.prompt_tokens, 5);
        assert_eq!(event.completion_tokens, 3);
    }

    /// AISIX-Cloud#1010, cross-provider bridge path: a monitor-mode output
    /// guardrail must not hold back or fail the bridged stream on the buffer
    /// cap either. An oversized bridged response is released in full with no
    /// `content_filter` error frame, and the usage event stays an unblocked
    /// 200.
    #[tokio::test]
    async fn monitor_output_guardrail_oversized_cross_provider_stream_released() {
        use aisix_obs::UsageSink;
        let upstream = MockServer::start().await;
        let big = serde_json::to_string(&"y".repeat(300_000)).unwrap();
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(anthropic_text_sse(&big)),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_at(&upstream.uri());
        snap.models.insert(anthropic_model("claude-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails
            .insert(keyword_output_guardrail_monitor("BLOCKME"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"claude-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 8 * 1024 * 1024).await.unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            body.contains(&"y".repeat(1000)),
            "the bridged stream must be released, not withheld",
        );
        assert!(
            !body.contains("content_filter"),
            "no fail-closed error frame on a monitor-only chain",
        );

        let event = tokio::time::timeout(std::time::Duration::from_millis(1000), rx.recv())
            .await
            .expect("usage event must be emitted")
            .expect("usage_sink sender dropped");
        assert!(!event.guardrail_blocked);
        assert_eq!(event.status_code, 200);
    }

    /// #546: output tool-call arguments must be scanned. A blocked literal in
    /// a `function_call` item's `arguments` (a top-level item field, not under
    /// `content[]`) must block the non-streaming response — else tool-call
    /// output is an output-guardrail bypass.
    #[tokio::test]
    async fn output_guardrail_blocks_tool_call_arguments_non_streaming() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_tc",
                "object": "response",
                "output": [{
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": "{\"q\":\"BLOCKME\"}"
                }],
                "usage": {"input_tokens": 5, "output_tokens": 4}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #546: tool-call arguments are scanned on the streaming path too (via
    /// the `response.completed` event), and held back — the args never reach
    /// the client.
    #[tokio::test]
    async fn output_guardrail_blocks_tool_call_arguments_streaming() {
        let upstream = MockServer::start().await;
        let completed = serde_json::json!({
            "type": "response.completed",
            "response": {"output": [{
                "type": "function_call",
                "name": "lookup",
                "arguments": "{\"q\":\"BLOCKME\"}"
            }]}
        });
        let sse = format!("event: response.completed\ndata: {completed}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            !String::from_utf8_lossy(&bytes).contains("BLOCKME"),
            "tool-call arguments leaked despite output block",
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #546 (audit HIGH): tool-call args on a `response.incomplete` terminal
    /// (fires routinely on `max_output_tokens` truncation — it carries the
    /// full `output[]`) must also be scanned and held back, not only
    /// `response.completed`. Same for `response.failed`.
    #[tokio::test]
    async fn output_guardrail_blocks_tool_call_on_incomplete_terminal() {
        let upstream = MockServer::start().await;
        let incomplete = serde_json::json!({
            "type": "response.incomplete",
            "response": {"output": [{
                "type": "function_call",
                "name": "lookup",
                "arguments": "{\"q\":\"BLOCKME\"}"
            }]}
        });
        let sse = format!("event: response.incomplete\ndata: {incomplete}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            !String::from_utf8_lossy(&bytes).contains("BLOCKME"),
            "tool-call args leaked on response.incomplete terminal",
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #546 (audit HIGH): a stream that ends WITHOUT a terminal `response`
    /// object but streamed tool-call argument deltas must still be scanned —
    /// args arrive via `response.function_call_arguments.delta`, not
    /// `output_text.delta`. The literal is split across two deltas to pin
    /// that they reassemble without a separator.
    #[tokio::test]
    async fn output_guardrail_blocks_tool_call_delta_without_terminal() {
        let upstream = MockServer::start().await;
        let d1 = serde_json::json!({"type":"response.function_call_arguments.delta","delta":"{\"q\":\"BLOCK"});
        let d2 =
            serde_json::json!({"type":"response.function_call_arguments.delta","delta":"ME\"}"});
        let sse = format!(
            "event: response.function_call_arguments.delta\ndata: {d1}\n\n\
             event: response.function_call_arguments.delta\ndata: {d2}\n\ndata: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            !String::from_utf8_lossy(&bytes).contains("BLOCK"),
            "streamed tool-call args leaked with no terminal event",
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #546 (re-audit): MCP tool-call argument deltas stream via their own
    /// event too; a no-terminal stream of `response.mcp_call_arguments.delta`
    /// carrying a blocked literal must still be scanned and held back.
    #[tokio::test]
    async fn output_guardrail_blocks_mcp_tool_call_delta_without_terminal() {
        let upstream = MockServer::start().await;
        let d = serde_json::json!({"type":"response.mcp_call_arguments.delta","delta":"{\"q\":\"BLOCKME\"}"});
        let sse =
            format!("event: response.mcp_call_arguments.delta\ndata: {d}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("BLOCKME"));
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #542: a guardrail-blocked request must NOT consume a rate-limit slot.
    /// With RPM=1 and a blocking guardrail, a blocked request followed by a
    /// benign one — the benign request must still succeed (the block didn't
    /// burn the only slot). Pre-fix (guardrail ran after `quota::enforce`) the
    /// block reserved+burned the slot, so the benign request got 429.
    #[tokio::test]
    async fn blocked_request_does_not_consume_rate_limit_slot() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id":"resp_ok","object":"response",
                "output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],
                "usage":{"input_tokens":1,"output_tokens":1}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        // API key capped at RPM=1.
        let apikey: ApiKey = serde_json::from_str(
            r#"{"key_hash":"8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891","allowed_models":["*"],"rate_limit":{"rpm":1}}"#,
        )
        .unwrap();
        snap.apikeys.insert(ResourceEntry::new("k-1", apikey, 1));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        // Blocked by the guardrail — must NOT reserve the single RPM slot.
        let blocked = app
            .clone()
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"BLOCKME"}),
            ))
            .await
            .unwrap();
        assert_eq!(blocked.status(), StatusCode::UNPROCESSABLE_ENTITY);

        // Benign request — the slot must still be available.
        let ok = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hello"}),
            ))
            .await
            .unwrap();
        assert_eq!(
            ok.status(),
            StatusCode::OK,
            "a guardrail block must not burn the RPM slot (#542)",
        );
    }

    #[tokio::test]
    async fn unauthenticated_returns_401() {
        let snap = new_snap_openai("http://unused");
        let app = build_app(snap);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"model":"m","input":"hi"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn unknown_model_returns_404() {
        let snap = new_snap_openai("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "no-such-model",
                "input": "hello"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    /// #825: an Anthropic-backed model is no longer rejected on
    /// /v1/responses — the request is bridged through ChatFormat to the
    /// Anthropic Messages upstream and the reply is re-encoded into the
    /// Responses-API shape. This is the codex-against-Anthropic path.
    #[tokio::test]
    async fn non_openai_model_bridges_to_responses_shape() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-ant-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_xprov",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-haiku-20240307",
                "content": [{"type": "text", "text": "Hi from Claude"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 9, "output_tokens": 4}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_at(&upstream.uri());
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "input": "hello"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        // Operator-facing model name echoed, not the upstream id.
        assert_eq!(body["model"], "claude-haiku");
        assert_eq!(body["output"][0]["type"], "message");
        assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(body["output"][0]["content"][0]["text"], "Hi from Claude");
        assert_eq!(body["usage"]["input_tokens"], 9);
        assert_eq!(body["usage"]["output_tokens"], 4);
    }

    /// #825 streaming: a streamed Anthropic-backed /v1/responses call emits
    /// the canonical Responses SSE event sequence ending in
    /// `response.completed` (the exact codex-tui path).
    #[tokio::test]
    async fn non_openai_streaming_bridges_to_responses_sse() {
        let sse = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-haiku-20240307\",\"content\":[],\"usage\":{\"input_tokens\":6,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_at(&upstream.uri());
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "input": "hi",
                "stream": true
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            text.contains("event: response.created"),
            "missing created: {text}"
        );
        assert!(text.contains("event: response.output_text.delta"));
        assert!(text.contains("\"delta\":\"Hi\""));
        assert!(text.contains("event: response.completed"));
    }

    #[tokio::test]
    async fn happy_path_forwards_to_upstream() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_abc",
                "object": "response",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}]
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "Hello"
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "response");
    }

    #[tokio::test]
    async fn upstream_error_returns_502() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "Hello"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    /// Issue #404: a successful non-streaming /v1/responses call must
    /// emit a `UsageEvent` onto the `usage_sink`. Pre-#404 the
    /// responses handler dropped the event entirely, so every
    /// o1/o3/GPT-5 traffic through Responses API was invisible to
    /// cp-api's budget ledger and customer-facing /logs analytics.
    /// This test pins the contract: after a 200 with a real
    /// upstream usage block, exactly one event arrives with the
    /// input_tokens / output_tokens / reasoning_tokens / cached
    /// counters and `inbound_protocol = "openai"`.
    #[tokio::test]
    async fn emits_usage_event_on_200_non_streaming_issue_404() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // Responses-API wire shape. Pin specific token counts so a
        // regression that swapped semantics (input vs output) or
        // dropped reasoning_tokens would fail here. Mirrors the
        // canonical OpenAI Responses API response object.
        let upstream_body = serde_json::json!({
            "id": "resp-abc",
            "object": "response",
            "model": "gpt-4o-2024-08-06",
            "output": [{
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}]
            }],
            "usage": {
                "input_tokens": 17,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens": 23,
                "output_tokens_details": {"reasoning_tokens": 8},
                "total_tokens": 40
            }
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hello world"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/responses 200")
            .expect("usage_sink sender dropped");

        assert_eq!(
            event.prompt_tokens, 17,
            "prompt_tokens must mirror upstream usage.input_tokens",
        );
        assert_eq!(
            event.completion_tokens, 23,
            "completion_tokens must mirror upstream usage.output_tokens",
        );
        assert_eq!(
            event.reasoning_tokens, 8,
            "reasoning_tokens must mirror usage.output_tokens_details.reasoning_tokens \
             (o1/o3/GPT-5 class models)",
        );
        assert_eq!(
            event.cached_prompt_tokens, 5,
            "cached_prompt_tokens must mirror usage.input_tokens_details.cached_tokens",
        );
        assert_eq!(event.status_code, 200);
        assert_eq!(event.api_key_id, "k-1");
        assert_eq!(event.model_id, "m-1");
        assert_eq!(event.inbound_protocol, "openai");
        assert!(!event.request_id.is_empty());
        assert!(!event.occurred_at.is_empty());
    }

    /// AISIX-Cloud#867: the verbatim-OpenAI /v1/responses path must apply the
    /// resolved ProviderKey's `request.*` overrides to the outbound call —
    /// both the `default_body_fields` injection (body) and the
    /// `default_headers` injection (header) must reach the upstream. The mock
    /// only matches (200) when BOTH the injected body field AND header are
    /// present, so a 200 proves the overrides were applied. Before the fix the
    /// outbound body/headers carried neither → mock wouldn't match → wiremock
    /// 404 → non-200.
    #[tokio::test]
    async fn responses_verbatim_applies_pk_request_overrides_issue_867() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(body_partial_json(serde_json::json!({"safe_flag": true})))
            .and(header("x-custom", "trace-on"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp-1",
                "object": "response",
                "output": [],
                "usage": {"input_tokens": 3, "output_tokens": 1}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(openai_pk_with_overrides(&upstream.uri()));
        snap.models.insert(openai_model("gpt-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-resp",
                "input": "hi"
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// #543: an OUTPUT-blocked /v1/responses still records the billed
    /// upstream tokens (the provider already charged), marked
    /// `guardrail_blocked`, with status 422 — NOT a zero-token event. Zeroing
    /// would let the customer's budget ledger underreport spend they paid the
    /// provider for (the output analog of chat.rs's UpstreamCharge).
    #[tokio::test]
    async fn output_block_records_billed_tokens_issue_543() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "id": "resp-blk",
            "object": "response",
            "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text":"sure: BLOCKME"}]}],
            "usage": {"input_tokens": 11, "output_tokens": 7}
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("a UsageEvent must be emitted for an output-blocked response (#543)")
            .expect("usage_sink sender dropped");
        assert_eq!(
            event.prompt_tokens, 11,
            "billed input tokens must be recorded despite the block",
        );
        assert_eq!(
            event.completion_tokens, 7,
            "billed output tokens must be recorded despite the block",
        );
        assert_eq!(event.status_code, 422);
        assert!(
            event.guardrail_blocked,
            "the output-block event must be marked guardrail_blocked",
        );
    }

    /// Companion: an upstream 200 missing the `usage` block entirely
    /// (some relay / compat backends) now emits an ESTIMATED usage
    /// event (AISIX-Cloud#1074) — the call must not stay invisible to
    /// billing. (Pre-#1074 this edge kept the pre-#404 no-emit
    /// behaviour.)
    #[tokio::test]
    async fn estimates_usage_event_when_upstream_omits_usage_block() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // 200 OK but no `usage` field — should NOT emit.
        let upstream_body = serde_json::json!({
            "id": "resp-abc",
            "object": "response",
            "model": "gpt-4o-2024-08-06",
            "output": []
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hello"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("estimated UsageEvent must be emitted when `usage` is absent")
            .expect("usage_sink sender dropped");
        // input "hello" per the cookbook message scheme: 3 per-message
        // + "user" (1) + "hello" (1) + 3 reply priming = 8. Empty
        // `output` → completion stays 0.
        assert_eq!(event.prompt_tokens, 8);
        assert_eq!(event.completion_tokens, 0);
        assert!(
            event.usage_estimated,
            "locally-counted tokens must be flagged"
        );
    }

    /// #808: a streaming `/v1/responses` 200 (e.g. all Codex traffic, which
    /// always streams) MUST emit a UsageEvent with the tokens carried on the
    /// terminal `response.completed` event — pre-#808 the streaming path
    /// dropped the event entirely, so successful streamed calls were invisible
    /// to the dashboard Logs and the budget ledger while 4xx/5xx still logged.
    /// Bytes must still pass through verbatim (SSE shape preserved). Fails
    /// before the fix (no event), passes after.
    #[tokio::test]
    async fn streaming_path_emits_usage_event_from_terminal_event_issue_808() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // Real Responses-API streaming: deltas then a terminal
        // `response.completed` carrying the authoritative `usage` block
        // (nested under `response`, with reasoning + cached sub-counts).
        let sse_body = "\
data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":11,\"output_tokens\":7,\"output_tokens_details\":{\"reasoning_tokens\":3},\"input_tokens_details\":{\"cached_tokens\":2}}}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hi",
                "stream": true
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Draining the body runs the stream to completion → the Drop guard
        // fires the end-of-stream emit. Bytes must survive verbatim.
        let body_bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            body_bytes.starts_with(b"data: "),
            "SSE shape must pass through verbatim",
        );
        assert!(
            body_bytes.windows(b"[DONE]".len()).any(|w| w == b"[DONE]"),
            "terminal frames must reach the client unchanged",
        );

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for a streaming 200 (#808)")
            .expect("usage event sender dropped");
        assert_eq!(ev.status_code, 200);
        assert_eq!(ev.inbound_protocol, "openai");
        assert_eq!(ev.prompt_tokens, 11);
        assert_eq!(ev.completion_tokens, 7);
        assert_eq!(ev.reasoning_tokens, 3);
        assert_eq!(ev.cached_prompt_tokens, 2);
        assert!(
            rx.try_recv().is_err(),
            "exactly one UsageEvent for a single streamed request",
        );
    }

    /// AISIX-Cloud#867: a streaming `/v1/responses` 200 (every Codex request,
    /// which always streams) MUST carry the resolved ProviderKey's telemetry
    /// attribution tags — provider_kind / provider_featured / branded_provider
    /// / pk_label — exactly like `/v1/messages` and `/v1/chat/completions`.
    /// Pre-fix the responses handler left these at default, so Codex logs were
    /// missing the upstream vendor + PK label that Claude-Code (Anthropic SDK)
    /// logs show. Fails before the fix (empty tags), passes after.
    #[tokio::test]
    async fn streaming_path_emits_provider_telemetry_tags_issue_867() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let sse_body = "\
data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":11,\"output_tokens\":7}}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_openai_tagged(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hi",
                "stream": true
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = to_bytes(resp.into_body(), 65536).await.unwrap();

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for a streaming 200")
            .expect("usage event sender dropped");
        assert_eq!(ev.status_code, 200);
        assert_eq!(
            ev.provider_kind, "catalog",
            "provider_kind must mirror the resolved PK's telemetry_tags.kind",
        );
        assert!(
            ev.provider_featured,
            "provider_featured must mirror telemetry_tags.featured",
        );
        assert_eq!(
            ev.branded_provider, "openai",
            "branded_provider must mirror telemetry_tags.branded_provider",
        );
        assert_eq!(
            ev.pk_label, "prod-codex-key",
            "pk_label must mirror telemetry_tags.pk_label",
        );
    }

    /// AISIX-Cloud#867 (non-streaming sibling): the same per-PK telemetry
    /// attribution must land on a non-streaming `/v1/responses` 200, which
    /// emits from the handler via `ResponseDispatchSuccess.provider_key_id`
    /// (a different threading path than the streaming Drop guard above).
    #[tokio::test]
    async fn non_streaming_emits_provider_telemetry_tags_issue_867() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "id": "resp-abc",
            "object": "response",
            "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}],
            "usage": {"input_tokens": 17, "output_tokens": 23}
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai_tagged(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hello"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/responses 200")
            .expect("usage_sink sender dropped");
        assert_eq!(ev.provider_kind, "catalog");
        assert!(ev.provider_featured);
        assert_eq!(ev.branded_provider, "openai");
        assert_eq!(ev.pk_label, "prod-codex-key");
    }

    /// Per #655: a 5xx upstream now emits ONE zero-token UsageEvent for the
    /// failed attempt, so the dashboard's Logs tab surfaces the failure
    /// alongside its siblings. (Pre-#655 the responses handler dropped the
    /// event on the error path — the failed request was invisible.) The
    /// event carries the mapped status (502), zero tokens, an error class,
    /// and the initial-attempt classification.
    #[tokio::test]
    async fn upstream_5xx_emits_zero_token_failed_attempt_event() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal"))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hello"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let recv = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await;
        let ev = recv
            .expect("a failed-attempt UsageEvent must be emitted within the timeout")
            .expect("the usage sink channel must not be closed");
        assert_eq!(ev.status_code, 502, "failed attempt records the mapped 502");
        assert_eq!(ev.prompt_tokens, 0, "failed attempt has zero tokens");
        assert_eq!(ev.completion_tokens, 0, "failed attempt has zero tokens");
        assert_eq!(ev.attempt_index, 0, "single direct-model attempt");
        assert_eq!(ev.attempt_kind, "initial");
        assert_eq!(
            ev.error_class, "upstream_status",
            "500 upstream maps to an upstream_status error class",
        );
        // Exactly one event — a direct model has no further attempts and no
        // separate terminal event (the attempt itself is terminal).
        let extra = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await;
        if let Ok(Some(ev)) = extra {
            panic!(
                "expected exactly one event for a single failed attempt, got a second: \
                 attempt_index={} status_code={}",
                ev.attempt_index, ev.status_code,
            );
        }
    }

    /// A 200 with `usage: {}` (malformed — `input_tokens` is required
    /// by the Responses-API spec) now emits an ESTIMATED usage event
    /// (AISIX-Cloud#1074) instead of dropping the record. Same edge as
    /// the omitted-usage test but the gate is one layer deeper —
    /// `usage` exists but is empty. (Pre-#1074, per issue #404 audit
    /// MEDIUM-1, this dropped the event entirely.)
    #[tokio::test]
    async fn estimates_usage_event_when_usage_block_is_empty() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "id": "resp-abc",
            "object": "response",
            "output": [],
            "usage": {}  // malformed — input_tokens required by spec
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hi"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("estimated UsageEvent must be emitted for malformed `usage: {}`")
            .expect("usage_sink sender dropped");
        // input "hi" counts per the cookbook message scheme: 3 per-message
        // + "user" (1) + "hi" (1) + 3 reply priming = 8. Empty output → 0.
        assert_eq!(event.prompt_tokens, 8);
        assert_eq!(event.completion_tokens, 0);
        assert!(
            event.usage_estimated,
            "locally-counted tokens must be flagged"
        );
    }

    /// #429 follow-up: a 200 whose `usage` carries
    /// `input_tokens` but omits `output_tokens` is still a real billable
    /// call. It MUST emit a UsageEvent with `completion_tokens = 0`
    /// (coercing the missing side to 0), NOT be dropped. Only a
    /// fully absent / input-less usage block skips.
    #[tokio::test]
    async fn emits_with_zero_output_when_output_tokens_missing() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "id": "resp-abc",
            "object": "response",
            "output": [],
            "usage": { "input_tokens": 17 }  // missing output_tokens
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hi"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must still be emitted when only output_tokens is missing")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 17, "input side must be recorded");
        assert_eq!(
            event.completion_tokens, 0,
            "missing output_tokens must default to 0, not drop the event"
        );
    }

    /// AISIX-Cloud#947: the streamed content capture prefers the terminal
    /// `response.completed` event's full output over accumulated deltas —
    /// the terminal object is authoritative (includes tool-call items the
    /// deltas may have missed).
    #[test]
    fn sse_text_capture_prefers_terminal_full_output() {
        let mut cap = super::SseTextCapture::new(1024);
        cap.observe(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "partial "
        }));
        cap.observe(&serde_json::json!({
            "type": "response.completed",
            "response": {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "the full text"}]}
                ]
            }
        }));
        assert_eq!(cap.into_text(), "the full text");
    }

    /// AISIX-Cloud#947: a stream that aborts before any terminal event falls
    /// back to the concatenated deltas — including tool-call argument deltas,
    /// which stream via their own event type.
    #[test]
    fn sse_text_capture_falls_back_to_deltas_on_abort() {
        let mut cap = super::SseTextCapture::new(1024);
        cap.observe(&serde_json::json!({
            "type": "response.output_text.delta",
            "delta": "hello "
        }));
        cap.observe(&serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "delta": "{\"city\":\"SF\"}"
        }));
        assert_eq!(cap.into_text(), "hello {\"city\":\"SF\"}");
    }

    /// AISIX-Cloud#947: delta accumulation is bounded to the capture cap so
    /// a long stream can't grow the buffer without limit.
    #[test]
    fn sse_text_capture_bounds_delta_accumulation() {
        let mut cap = super::SseTextCapture::new(10);
        for _ in 0..100 {
            cap.observe(&serde_json::json!({
                "type": "response.output_text.delta",
                "delta": "0123456789"
            }));
        }
        let text = cap.into_text();
        // One push may land after the buffer crosses the cap; the point is
        // the accumulation stops near the cap instead of growing 100x.
        assert!(text.len() <= 20, "delta buffer must stay near the cap");
    }
}
