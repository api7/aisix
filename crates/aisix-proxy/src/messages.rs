//! `POST /v1/messages` — Anthropic Messages API, any upstream.
//!
//! Two dispatch paths share this entry point:
//!
//! - **Anthropic upstream** (`Model.provider == anthropic`) — byte-for-byte
//!   passthrough to `{api_base}/v1/messages`. Preserves features the
//!   gateway-internal `ChatFormat` can't lossily round-trip (cache_control,
//!   thinking blocks, tool_use, image blocks). Adds `x-api-key` +
//!   `anthropic-version` headers, rewrites the `model` field to the
//!   upstream id, and streams the SSE response verbatim.
//!
//! - **Non-Anthropic upstream** (`Model.provider == openai|gemini|deepseek`)
//!   — translates the Anthropic-shape body to the gateway's internal
//!   [`ChatFormat`], dispatches through the [`Hub`] to the matching
//!   [`Bridge`], and re-encodes the bridge's [`ChatResponse`] / chunk
//!   stream as Anthropic JSON or Anthropic SSE events
//!   (`message_start` / `content_block_*` / `message_delta` /
//!   `message_stop`). The translation helpers live in
//!   `aisix-provider-anthropic::wire`. Content blocks translate per the
//!   LiteLLM map (#722): text / image / document / tool_use /
//!   tool_result; thinking history blocks drop (non-replayable on the
//!   OpenAI wire).
//!
//! Both paths share the same auth, model lookup, allowed_models check,
//! access-log emission, metrics labels, and health tracker hooks.
//!
//! Errors use the Anthropic-shape envelope
//! `{type:"error", error:{type, message}}` (per
//! <https://docs.anthropic.com/en/api/errors>) so Claude SDKs and the
//! official `anthropic-sdk-python` envelope parser see a wire shape they
//! recognise. The inner `error.type` follows the Anthropic SDK's strict
//! `ErrorType` literal — `authentication_error` / `rate_limit_error` /
//! `api_error` / etc. — NOT the OpenAI envelope's DP-stable taxonomy.
//! See [`crate::error::ProxyError::into_anthropic_response`] for the
//! status-to-type mapping. (`/v1/chat/completions` continues to emit
//! the OpenAI-shape envelope with its DP-stable taxonomy.)

use aisix_core::AppliedGuardrail;
use aisix_obs::{
    content_capture_cap, AccessLog, CapturedContent, LlmUsage, RequestLabels, RequestOutcome,
    UsageEvent, UsageLabels,
};
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
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
use crate::usage_attr::total_tokens_with_cache;

/// Anthropic API version header value injected on every forwarded request.
/// Shared with the `/v1/messages/count_tokens` handler so both Anthropic
/// passthrough paths pin the same version.
pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";

pub async fn messages(
    State(state): State<ProxyState>,
    auth: Result<AuthenticatedKey, ProxyError>,
    client: ClientContext,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    // Catch extractor rejections (auth fail / malformed JSON) HERE
    // and re-wrap as Anthropic envelope. Without this, axum's default
    // `IntoResponse for ProxyError` emits the OpenAI shape, which the
    // Claude SDK can't parse on a /v1/messages 401/400 response
    // (#336). Same envelope policy as dispatch-side errors below.
    let auth = match auth {
        Ok(a) => a,
        Err(e) => return e.into_anthropic_response(),
    };
    let Json(mut body) = match body {
        Ok(j) => j,
        Err(rej) => {
            // Classify the body-extractor failure (malformed JSON vs
            // 413 cap vs transport read error) via the shared helper so
            // /v1/messages and /v1/messages/count_tokens stay in lockstep
            // on the discrimination rules, then render the Anthropic-
            // shape envelope the Claude SDK can parse (#336).
            return crate::error::proxy_error_from_json_rejection(
                rej,
                state.request_body_limit_bytes,
            )
            .into_anthropic_response();
        }
    };
    let started = Instant::now();
    let request_id = client.request_id.clone();
    let api_key_id = auth.entry.id.clone();

    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let snapshot = state.snapshot.load();
    let model_id = crate::model_resolve::resolve_model(&snapshot, &model_name)
        .map(|e| e.id.clone())
        .unwrap_or_default();
    drop(snapshot);

    // Filled by `dispatch` once the per-request guardrail chain resolves;
    // read below to attach `applied_guardrails` to the telemetry event on both
    // the success and failure (input-block) paths (#379).
    let mut applied_guardrails: Vec<AppliedGuardrail> = Vec::new();
    // Filled by `dispatch` with per-detector PII mask counts (#932), same
    // dual-path lifecycle as `applied_guardrails`. Streaming output counts
    // travel via the stream builders' end-of-stream emit instead.
    let mut redaction_counts = crate::redact::RedactionCounts::new();
    // Filled by `dispatch` with monitor-mode guardrail observations
    // (AISIX-Cloud#562), same lifecycle as `redaction_counts`.
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    // #890 req-1: capture the client's streaming intent before dispatch
    // (which mutates the body).
    let stream_requested = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    match dispatch(
        &state,
        &auth,
        &mut body,
        &request_id,
        started,
        &client,
        &mut applied_guardrails,
        &mut redaction_counts,
        &mut monitor_hits,
    )
    .await
    {
        Ok(DispatchOutcome {
            response,
            provider_label,
            provider_key_id,
            upstream_model,
            metrics,
            usage_handled_by_stream,
            routing,
            captured_content,
            output_redactions,
            output_monitor_hits,
        }) => {
            // #932: fold the non-streaming response-side mask counts into
            // the per-request total before the terminal emit below.
            crate::redact::merge_counts(&mut redaction_counts, output_redactions);
            monitor_hits.extend(output_monitor_hits);
            let elapsed = started.elapsed();
            let status = response.status().as_u16();
            emit_access_log(
                &model_name,
                &provider_label,
                &api_key_id,
                status,
                elapsed,
                &request_id,
                &routing,
            );
            state.metrics.record_request(
                &provider_label,
                &model_name,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            let outcome = RequestOutcome::from_status(status);
            // #890 req-3: readable provider-key name resolved from the snapshot.
            let provider_key_name = {
                let snap = state.snapshot.load();
                crate::usage_attr::provider_key_metric_name(&snap, &provider_key_id)
            };
            let labels = RequestLabels {
                endpoint: "/v1/messages",
                inbound_protocol: "anthropic",
                provider: &provider_label,
                model: &model_name,
                upstream_model: &upstream_model,
                provider_key_id: &provider_key_id,
                provider_key_name: &provider_key_name,
                api_key_id: &api_key_id,
                team_id: auth.key().team_id.as_deref().unwrap_or("unknown"),
                user_id: auth.key().user_id.as_deref().unwrap_or("unknown"),
                user_name: auth.key().user_name.as_deref().unwrap_or("unknown"),
                stream: stream_requested,
                is_fallback: routing.fallback_count() > 0,
                status,
                outcome,
            };
            state.metrics.record_proxy_request(labels, elapsed);
            state.metrics.record_llm_request(labels, elapsed);
            // Per #655: one zero-token UsageEvent per failed attempt that
            // preceded the winner (non-streaming failover). No-op for a
            // first-try success and for the single-attempt streaming path.
            emit_failed_attempts_anthropic(
                &state,
                &request_id,
                &api_key_id,
                &provider_label,
                &model_name,
                &upstream_model,
                auth.key().team_id.as_deref(),
                auth.key().user_id.as_deref(),
                auth.key().user_name.as_deref(),
                &client,
                &applied_guardrails,
                &routing,
            );
            if !usage_handled_by_stream {
                // Winning-attempt classification (#655). Direct models have
                // no recorded attempt → AttemptInfo defaults (index 0,
                // "initial", empty target). The streaming path emits the
                // winner from its Drop guard, so it is skipped here.
                let winner = routing.winner();
                // AISIX-Cloud#790: the event's model_id is the winning
                // TARGET's id so pricing resolves against it.
                let event_model_id = winner
                    .map(|w| w.target_model_id.as_str())
                    .unwrap_or(&model_id);
                let attempt = winner.map(AttemptInfo::from_record).unwrap_or_default();
                emit_anthropic_usage_event(
                    &state,
                    &request_id,
                    event_model_id,
                    &api_key_id,
                    &provider_label,
                    &model_name,
                    &provider_key_id,
                    &upstream_model,
                    auth.key().team_id.as_deref(),
                    auth.key().user_id.as_deref(),
                    auth.key().user_name.as_deref(),
                    status,
                    elapsed,
                    metrics,
                    &client,
                    attempt,
                    applied_guardrails.clone(),
                    redaction_counts.clone(),
                    monitor_hits.clone(),
                    captured_content,
                );
            }
            response
        }
        Err(MessagesDispatchError { err, routing }) => {
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
            // #890 req-2: count the FAILED request on the rich request metrics
            // so a success rate is computable (denominator incl. failures).
            // Provider/upstream/provider_key are unknown on the failure path.
            let fail_labels = RequestLabels {
                endpoint: "/v1/messages",
                inbound_protocol: "anthropic",
                provider: "unknown",
                model: metric_model,
                upstream_model: "unknown",
                provider_key_id: "unknown",
                provider_key_name: "unknown",
                api_key_id: &api_key_id,
                team_id: auth.key().team_id.as_deref().unwrap_or("unknown"),
                user_id: auth.key().user_id.as_deref().unwrap_or("unknown"),
                user_name: auth.key().user_name.as_deref().unwrap_or("unknown"),
                stream: stream_requested,
                is_fallback: routing.fallback_count() > 0,
                status,
                outcome: RequestOutcome::from_status(status),
            };
            state.metrics.record_proxy_request(fail_labels, elapsed);
            state.metrics.record_llm_request(fail_labels, elapsed);
            // Per #655: emit one zero-token UsageEvent per FAILED attempt so
            // the dashboard's Logs tab surfaces each failed upstream try.
            emit_failed_attempts_anthropic(
                &state,
                &request_id,
                &api_key_id,
                "unknown",
                &model_name,
                "unknown",
                auth.key().team_id.as_deref(),
                auth.key().user_id.as_deref(),
                auth.key().user_name.as_deref(),
                &client,
                &applied_guardrails,
                &routing,
            );
            // Pre-dispatch failure (model-not-found, auth, budget, guardrail
            // block before any upstream attempt) records no attempts — emit a
            // single terminal event carrying the failure class. When attempts
            // were recorded, each was already emitted above.
            if routing.attempts.is_empty() {
                emit_anthropic_usage_event(
                    &state,
                    &request_id,
                    &model_id,
                    &api_key_id,
                    "unknown",
                    &model_name,
                    "unknown",
                    "unknown",
                    auth.key().team_id.as_deref(),
                    auth.key().user_id.as_deref(),
                    auth.key().user_name.as_deref(),
                    status,
                    elapsed,
                    AnthropicUsageMetrics::default(),
                    &client,
                    AttemptInfo {
                        kind: "initial".to_string(),
                        error_class: err.kind().to_string(),
                        ..Default::default()
                    },
                    applied_guardrails.clone(),
                    // Input masking may have fired before the failure.
                    redaction_counts.clone(),
                    monitor_hits.clone(),
                    /* content */ None,
                );
            }
            // /v1/messages must return Anthropic-shape error envelope
            // `{type:"error", error:{type, message}}` so Claude SDKs
            // can parse it — closes #336. The DP-stable taxonomy
            // (`upstream_error`, `invalid_api_key`, …) is preserved
            // on the nested `error.type` per ai-gateway#327.
            err.into_anthropic_response()
        }
    }
}

/// Emit one zero-token `UsageEvent` per FAILED attempt of a `/v1/messages`
/// request (#655). The winner / pre-dispatch event is emitted separately.
/// No-op when there are no failed attempts. Each event shares `request_id`.
#[allow(clippy::too_many_arguments)]
fn emit_failed_attempts_anthropic(
    state: &ProxyState,
    request_id: &str,
    api_key_id: &str,
    provider: &str,
    model: &str,
    upstream_model: &str,
    team_id: Option<&str>,
    user_id: Option<&str>,
    user_name: Option<&str>,
    client: &ClientContext,
    applied_guardrails: &[AppliedGuardrail],
    routing: &RoutingTelemetry,
) {
    for rec in routing.attempts.iter().filter(|a| !a.success) {
        emit_anthropic_usage_event(
            state,
            request_id,
            // Each failed attempt records the TARGET it actually hit
            // (AISIX-Cloud#790), not the group it was resolved from.
            &rec.target_model_id,
            api_key_id,
            provider,
            model,
            &rec.provider_key_id,
            upstream_model,
            team_id,
            user_id,
            user_name,
            rec.status,
            Duration::from_millis(u64::from(rec.latency_ms)),
            AnthropicUsageMetrics::default(),
            client,
            AttemptInfo::from_record(rec),
            applied_guardrails.to_vec(),
            // Failed attempts carry no per-request redaction detail; the
            // terminal (winner / pre-dispatch) event does.
            crate::redact::RedactionCounts::new(),
            Vec::new(),
            /* content */ None,
        );
    }
}

#[allow(clippy::too_many_arguments)]
async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    body: &mut Value,
    request_id: &str,
    started: Instant,
    client: &ClientContext,
    // Out-param: filled with the resolved chain's `{kind, hook}` set as soon as
    // the guardrail chain resolves, so `messages()` can attach it to telemetry
    // on both the success and error (input-block) paths. Empty for requests
    // rejected before resolution. The streaming paths capture the same set
    // directly from `resolved_chain` for their end-of-stream emit.
    applied_out: &mut Vec<AppliedGuardrail>,
    // Out-param: per-detector PII mask counts (#932), same lifecycle as
    // `applied_out`. Streaming output counts travel via the stream
    // builders' own end-of-stream emit instead.
    redactions_out: &mut crate::redact::RedactionCounts,
    // Out-param: monitor-mode guardrail observations (AISIX-Cloud#562),
    // same lifecycle as `redactions_out`.
    monitor_hits_out: &mut Vec<aisix_core::GuardrailMonitorHit>,
) -> Result<DispatchOutcome, MessagesDispatchError> {
    let snapshot = state.snapshot.load();

    // Extract and resolve model.
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

    // #448 (#22): /v1/messages must run input guardrails like
    // /v1/chat/completions — previously prompts reached the upstream without
    // any content/DLP check. Translate the Anthropic-shaped body into the
    // internal ChatFormat and run the resolved input guardrail chain; a Block
    // short-circuits before dispatch. (Input Rewrite/Bypass on this endpoint
    // is not yet applied to the outgoing Anthropic body — only Block is
    // enforced here.)
    //
    // #542: run this BEFORE the rate-limit reservation so a content-policy
    // block doesn't burn an RPM slot (matching /v1/chat/completions).
    let guardrail_ctx = aisix_guardrails::RequestContext {
        model_id: &model_entry.id,
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    // Arc so the chain can be cloned into the streaming-response body
    // (which outlives this handler) for end-of-stream output guardrails.
    let resolved_chain = std::sync::Arc::new(state.guardrail_index.resolve(&guardrail_ctx));
    // Surface the applied `{kind, hook}` set to the caller so the telemetry
    // event records which guardrails governed the request even when the input
    // check below blocks it (#379 / closes the anthropic gap in #519).
    *applied_out = resolved_chain.applied().to_vec();
    if !resolved_chain.is_empty() {
        if let Ok(chat) = aisix_provider_anthropic::parse_inbound_request(body) {
            let (verdict, hits) = aisix_guardrails::Guardrail::check_input_non_segment_observed(
                resolved_chain.as_ref(),
                &chat,
            )
            .await;
            monitor_hits_out.extend(hits);
            // Segment pass: one Bedrock call over the body's text slots;
            // an ANONYMIZE disposition writes the masked text back into
            // the Anthropic-native body (#932 bedrock follow-up).
            let verdict = crate::redact::moderate_body(
                resolved_chain.as_ref(),
                crate::redact::Direction::Input,
                verdict,
                redactions_out,
                monitor_hits_out,
                |g| crate::redact::redact_anthropic_request(g, body),
            )
            .await;
            if let aisix_guardrails::GuardrailVerdict::Block {
                reason,
                guardrail_name,
            } = verdict
            {
                tracing::warn!(
                    guardrail_hook = "input",
                    model = %model_name,
                    reason = %reason,
                    "guardrail blocked /v1/messages request",
                );
                return Err(
                    ProxyError::ContentFiltered(crate::error::guardrail_block_message(
                        "request",
                        guardrail_name.as_deref(),
                    ))
                    .into(),
                );
            }
        }
        // #932: mask-action PII rules rewrite the Anthropic-native body in
        // place AFTER the block check passes — both the passthrough and the
        // cross-provider bridge forward from this body, so the masked text
        // is what reaches the upstream.
        crate::redact::merge_counts(
            redactions_out,
            crate::redact::redact_anthropic_request(resolved_chain.as_ref(), body),
        );
    }

    let model_rl =
        crate::quota::ModelRateLimit::from_model(&model_name, &model_entry.id, &model_entry.value);
    // `Option` so the winning streaming attempt can `take()` the reservation
    // and carry it into the end-of-stream guard (#688); non-streaming / failed
    // attempts leave it in place for the post-dispatch commit or a retry.
    let mut reservation = Some(crate::quota::enforce(state, auth, Some(&model_rl)).await?);

    // Budget pre-check via cp-api (mirrors /v1/chat/completions).
    let budget_decision = state.budgets.check(&auth.entry.id).await;
    if !budget_decision.allowed {
        return Err(
            ProxyError::BudgetExceeded(Box::new(budget_decision.reason.unwrap_or_else(|| {
                crate::budget::BudgetReason::message_only(auth.entry.id.clone())
            })))
            .into(),
        );
    }

    // Resolve the attempt list. For a Model Group (routing model) this
    // walks `routing.targets` and health-filters them; for a direct
    // model it's just the model itself. Shared with /v1/chat/completions
    // so both endpoints dispatch Model Groups identically (#471).
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
        },
    )?;

    let retry_on_429 = model_entry
        .value
        .routing
        .as_ref()
        .map(|r| r.retry_on_429_or_default())
        .unwrap_or(false);
    // Routing target names only matter on the telemetry for a real Model
    // Group; a direct model leaves `attempt_model` empty (its `model_id`
    // already identifies it), matching chat.rs.
    let is_routing_request = model_entry.value.routing.is_some();
    let mut routing = RoutingTelemetry::default();

    // `routing.retries` — how many times to re-hit the SAME target (with
    // backoff) on a retryable failure before failing over to the next target.
    // Honoured here exactly like chat.rs (#641); 0 (the default) keeps the
    // fail-over-only behaviour. /v1/messages previously ignored it entirely.
    let retries = model_entry
        .value
        .routing
        .as_ref()
        .map(|r| r.retries_or_default())
        .unwrap_or(0);

    // Walk targets, failing over to the next only on a retryable upstream
    // failure. A 4xx / config error is returned as-is — retrying other
    // targets won't help. Streaming and non-streaming share this loop:
    // `dispatch_to_target` branches internally and, for streaming, only
    // returns Ok once the first chunk has arrived under `stream_timeout`
    // (#554) — so the 200 is committed to exactly one target and a slow
    // first chunk fails over like any other retryable error. Each attempt
    // (initial / same-target retry / fallover) becomes its own per-attempt
    // record (#655).
    let n = attempt_models.len();
    let mut last_err: Option<ProxyError> = None;
    'targets: for (i, target) in attempt_models.iter().enumerate() {
        let pk_id = crate::dispatch::resolve_provider_key(&snapshot, &target.model)
            .map(|e| e.id.clone())
            .unwrap_or_default();
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
            match dispatch_to_target(
                state,
                &snapshot,
                body,
                target,
                &model_name,
                request_id,
                started,
                &auth.entry.id,
                auth.key().team_id.clone(),
                auth.key().user_id.clone(),
                auth.key().user_name.clone(),
                resolved_chain.clone(),
                client,
                AttemptInfo {
                    index: idx,
                    kind: kind.to_string(),
                    model: target_model.clone(),
                    ..Default::default()
                },
                &mut reservation,
                redactions_out.clone(),
                monitor_hits_out.clone(),
            )
            .await
            {
                Ok(mut outcome) => {
                    let latency_ms = ms_since(attempt_started);
                    // Feed the least_latency EWMA for this target.
                    state.runtime_status.record_latency(&target.id, latency_ms);
                    routing.attempts.push(AttemptRecord {
                        index: idx,
                        kind,
                        target_model,
                        target_model_id: target.id.clone(),
                        provider_key_id: outcome.provider_key_id.clone(),
                        status: 200,
                        success: true,
                        error_class: String::new(),
                        error_message: String::new(),
                        latency_ms,
                    });
                    outcome.routing = routing;
                    // #911 [21]: commit the reserved layers with the actual
                    // token cost so TPM/TPD is enforced for /v1/messages like
                    // chat + embeddings. The non-streaming path carries the
                    // counts in `outcome.metrics` and commits here; the
                    // streaming path already `take()`-d the reservation into its
                    // end-of-stream guard (#688), so `reservation` is `None` and
                    // this is skipped.
                    if !outcome.usage_handled_by_stream {
                        if let Some(r) = reservation.take() {
                            let total = total_tokens_with_cache(
                                outcome.metrics.prompt_tokens,
                                outcome.metrics.completion_tokens,
                                outcome.metrics.cache_creation_tokens,
                                outcome.metrics.cache_read_tokens,
                            );
                            r.commit_tokens(total).await;
                        }
                    }
                    return Ok(outcome);
                }
                Err(e) => {
                    let retryable = matches!(
                        &e,
                        ProxyError::Bridge(be) if crate::routing::is_retryable(be, retry_on_429)
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
                    // Non-retryable → stop entirely (retrying or failing over
                    // won't help). Retryable → re-hit the same target until
                    // `retries` is exhausted, then fall over to the next target
                    // if there is one.
                    if !retryable {
                        break 'targets;
                    }
                    if attempt_idx == retries {
                        if i + 1 >= n {
                            break 'targets;
                        }
                        break;
                    }
                }
            }
        }
    }
    Err(MessagesDispatchError {
        err: last_err.unwrap_or(ProxyError::ProviderUnavailable),
        routing,
    })
}

/// Dispatch one concrete (non-routing) target Model. Branches on the
/// target's provider: Anthropic upstreams go through the byte-for-byte
/// passthrough, everything else through the cross-provider translation.
#[allow(clippy::too_many_arguments)]
async fn dispatch_to_target(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    body: &Value,
    target: &crate::routing::AttemptModel,
    model_name: &str,
    request_id: &str,
    started: Instant,
    api_key_id: &str,
    team_id: Option<String>,
    user_id: Option<String>,
    user_name: Option<String>,
    resolved_chain: std::sync::Arc<aisix_guardrails::GuardrailChain>,
    client: &ClientContext,
    // Winning-attempt classification (#655) — used by the streaming paths
    // whose Drop guard owns the UsageEvent emit. Non-streaming paths emit
    // from the handler and ignore it.
    attempt: AttemptInfo,
    // #688: the streaming paths `take()` this to carry the concurrency hold +
    // post-stream token accounting into the end-of-stream guard. Left in place
    // on the non-streaming / error paths for the handler to commit or retry.
    reservation: &mut Option<aisix_ratelimit::MultiReservation>,
    // Input-side PII mask counts (#932) — the streaming paths merge these
    // into their end-of-stream telemetry emit (the non-streaming emit
    // happens in `messages()`, which already holds them).
    input_redactions: crate::redact::RedactionCounts,
    // Input-side monitor hits (AISIX-Cloud#562), same lifecycle as
    // `input_redactions`.
    input_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
) -> Result<DispatchOutcome, ProxyError> {
    let model = &target.model;
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;

    if model.provider.as_deref() != Some("anthropic") {
        return cross_provider_dispatch(
            state,
            body,
            model,
            &target.id,
            &pk_entry.value,
            model_name,
            request_id,
            started,
            api_key_id,
            team_id,
            user_id,
            user_name,
            resolved_chain,
            client,
            attempt,
            reservation,
            input_redactions,
            input_monitor_hits,
        )
        .await;
    }

    anthropic_passthrough_dispatch(
        state,
        body,
        model,
        &target.id,
        &pk_entry.value,
        &pk_entry.id,
        model_name,
        request_id,
        started,
        api_key_id,
        team_id,
        user_id,
        user_name,
        resolved_chain,
        client,
        attempt,
        reservation,
        input_redactions,
        input_monitor_hits,
    )
    .await
}

/// Anthropic-protocol input -> Anthropic upstream: byte-for-byte
/// passthrough to `{api_base}/v1/messages`. Adds the `x-api-key` +
/// `anthropic-version` headers, rewrites the `model` field to the
/// upstream id, and streams the SSE response verbatim.
#[allow(clippy::too_many_arguments)]
async fn anthropic_passthrough_dispatch(
    state: &ProxyState,
    body: &Value,
    model: &aisix_core::Model,
    model_id: &str,
    pk_value: &aisix_core::ProviderKey,
    pk_id: &str,
    model_name: &str,
    request_id: &str,
    started: Instant,
    api_key_id: &str,
    team_id: Option<String>,
    user_id: Option<String>,
    user_name: Option<String>,
    resolved_chain: std::sync::Arc<aisix_guardrails::GuardrailChain>,
    client_ctx: &ClientContext,
    attempt: AttemptInfo,
    reservation: &mut Option<aisix_ratelimit::MultiReservation>,
    input_redactions: crate::redact::RedactionCounts,
    input_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
) -> Result<DispatchOutcome, ProxyError> {
    let mut body = body.clone();
    let api_key = crate::dispatch::require_secret(pk_value, model)?;

    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();

    // Rewrite the `model` field to the upstream value.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model.clone());
    }

    // Apply the PK's `request.*` override block to the outbound
    // body. Mirrors the OpenAI dispatch path's `prepare_outbound_body`
    // in `crates/aisix-provider-openai/src/bridge.rs:317-323`. The
    // OpenAI bridge applies the same primitives via the Hub dispatch,
    // but the Anthropic-passthrough path bypasses the Hub and builds
    // the request directly here — without this block the override
    // pipeline silently no-ops on `/v1/messages` (issue #302 §5
    // contract; tracked as ai-gateway#335 for the gap-as-shipped).
    //
    // Apply order matches §5: renames → constraints → defaults. Each
    // primitive is a no-op when its configured map is empty.
    if let Some(r) = pk_value.request.as_ref() {
        aisix_provider_openai::overrides::apply_param_renames(&mut body, &r.param_renames);
        if let Some(constraints) = &r.param_constraints {
            aisix_provider_openai::overrides::apply_param_constraints(&mut body, constraints);
        }
        aisix_provider_openai::overrides::apply_default_body_fields(
            &mut body,
            &r.default_body_fields,
        );
    }

    // Build the target URL. build_v1_url tolerates the rare case
    // where the customer mistakenly puts `/v1` in the Anthropic
    // api_base (the dashboard placeholder uses the OpenAI form, so
    // this is a copy-paste hazard).
    let base = crate::dispatch::resolve_base_url(pk_value)?;
    let url = crate::dispatch::build_v1_url(&base, "/messages");

    // Check if the request wants streaming.
    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Build the outbound HeaderMap explicitly so the PK's
    // `request.default_headers` block can inject operator-supplied
    // headers via the shared apply pipeline. The bridge-owned
    // headers (x-api-key, anthropic-version, content-type,
    // x-aisix-request-id) are inserted FIRST — `apply_default_headers`
    // skips keys already present + the reserved auth-header blacklist
    // (`x-api-key` is in `RESERVED_DEFAULT_HEADERS`), so operator
    // headers can never clobber auth here (ai-gateway#337).
    let mut headers = axum::http::HeaderMap::new();
    let api_key_hv = HeaderValue::from_str(api_key).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "api key contains invalid header chars: {e}"
        )))
    })?;
    headers.insert(HeaderName::from_static("x-api-key"), api_key_hv);
    headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static(ANTHROPIC_VERSION),
    );
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
    if let Some(r) = pk_value.request.as_ref() {
        aisix_provider_openai::overrides::apply_default_headers(&mut headers, &r.default_headers);
    }

    let client = crate::http_client::client();
    let mut req_builder = client.post(&url).headers(headers).json(&body);
    // #554: non-streaming gets the E2E request timeout via reqwest's
    // request-level timeout. Streaming must NOT use it (it would cap the
    // whole stream); the streaming branch below enforces the per-chunk
    // read timeout instead.
    if !is_stream {
        if let Some(d) = model.request_timeout() {
            req_builder = req_builder.timeout(d);
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
        crate::stream_timeout::send_with_deadline(req_builder, connect_deadline, send_started)
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
        let truncated = crate::util::truncate_on_char_boundary(&message, 1024);
        let err = aisix_gateway::BridgeError::upstream_status_with_retry_after(
            status_u16,
            truncated,
            retry_after,
        );
        // Apply the cross-request cooldown contract to the
        // Anthropic-passthrough path too — without this, a 401 / 429 /
        // 5xx via /v1/messages would never mark the direct model and
        // subsequent requests would keep hitting the same broken
        // upstream. See `crate::cooldown` for the shared decision.
        if let Some((ttl, reason)) = crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
        {
            state.runtime_status.mark_cooldown(model_id, ttl, reason);
        }
        return Err(ProxyError::Bridge(err));
    }

    // Update health trackers on success — both the display-name-keyed
    // observational signal AND the id-keyed runtime status that
    // routing filters consult. Without `mark_healthy` here, a target
    // that recovered via the Anthropic passthrough would stay in
    // `cooldown` on /admin/v1/models/status until its TTL naturally
    // expired (round-2 audit MEDIUM on PR #268).
    state.health.record_success(&model.display_name);
    state.runtime_status.mark_healthy(model_id);

    let provider_label = "anthropic".to_string();

    if is_stream {
        // For SSE streaming: pass through the response body as a streaming
        // `text/event-stream` response.
        let headers = upstream_resp.headers().clone();
        // #554: enforce the per-chunk read timeout on the forwarded bytes.
        // When a `stream_timeout` is configured, peek the first byte so a
        // slow/erroring first token fails over (the caller loops to the next
        // target) before the 200 is committed; without one, forward directly
        // (pre-#554 behavior). A mid-stream stall truncates the forwarded
        // stream — there is no in-band error frame for an opaque passthrough.
        let stream_budget = model.stream_timeout_effective();
        let wrapped: std::pin::Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>> =
            Box::pin(crate::stream_timeout::with_read_timeout_bytes(
                upstream_resp.bytes_stream(),
                stream_budget,
            ));
        let body_stream: std::pin::Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>> =
            if stream_budget.is_some() {
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
                    // Read timeout before the first byte (or an upstream that
                    // closed immediately): retryable stream-abort so the
                    // caller fails over.
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
                    futures::stream::once(std::future::ready(Ok::<Bytes, reqwest::Error>(
                        first_bytes,
                    )))
                    .chain(wrapped),
                )
            } else {
                wrapped
            };

        // Issue #245: parity with the OpenAI streaming fix (#225 /
        // #196). Pre-fix this path forwarded raw bytes and emitted a
        // UsageEvent with `prompt_tokens=0 completion_tokens=0` —
        // every streaming /v1/messages request billed as zero. Wrap
        // the byte stream in an Anthropic-shape SSE parser that
        // side-channels the upstream `usage` block (input_tokens from
        // `message_start`, running output_tokens from `message_delta`)
        // while forwarding bytes verbatim, then fires
        // `emit_anthropic_usage_event` from a Drop guard so the event
        // ships even on client-disconnect mid-stream (same
        // CompleteOnDrop pattern as chat.rs::build_sse_stream).
        let state_c = state.clone();
        let request_id_c = request_id.to_string();
        let model_id_c = model_id.to_string();
        let api_key_id_c = api_key_id.to_string();
        let provider_c = provider_label.clone();
        let model_name_c = model_name.to_string();
        let provider_key_id_c = pk_id.to_string();
        let upstream_model_c = upstream_model.clone();
        let team_id_c = team_id.clone();
        let user_id_c = user_id.clone();
        let user_name_c = user_name.clone();
        // #492: log the same client IP/UA on streamed responses.
        let client_ctx_c = client_ctx.clone();
        // Winning-attempt classification (#655) for the stream-end emit.
        let attempt_c = attempt.clone();

        // Applied guardrail set (#379), owned for the move into the
        // end-of-stream telemetry closure.
        let applied_guardrails_c = resolved_chain.applied().to_vec();
        let stream_guardrail = if resolved_chain.is_empty() {
            None
        } else {
            Some(resolved_chain.clone())
        };
        // Content capture: prompt up front; the response is already assembled
        // into `usage.response_text` by the frame parser and preserved (not
        // taken) when `content_cap` is set. Both gated.
        let content_cap = content_capture_cap(
            state
                .snapshot
                .load()
                .observability_exporters
                .entries()
                .iter()
                .map(|e| &e.value),
        );
        let captured_prompt_c =
            content_cap.map(|_| serde_json::to_string(&body).unwrap_or_default());
        // #688: carry the rate-limit reservation into the end-of-stream guard.
        // The winning streaming attempt owns it now; the keys drive post-stream
        // TPM/TPD accounting and `into_stream_hold` keeps the concurrency slot(s)
        // until the stream ends (mirrors chat.rs). `take()` leaves the handler's
        // `reservation` as `None`, so it won't also `commit_tokens`.
        let post_stream_keys = reservation.as_ref().map(|r| r.keys()).unwrap_or_default();
        let stream_hold = reservation.take().map(|r| r.into_stream_hold());
        let limiter_c = std::sync::Arc::clone(&state.limiter);
        let parsed_stream = build_anthropic_passthrough_stream(
            body_stream,
            started,
            stream_guardrail,
            model_name.to_string(),
            content_cap,
            move |usage| {
                // Streaming responses that got this far are 200 — the
                // !status.is_success() guard above returned early on
                // upstream errors.
                //
                // #688: apply the terminal token cost to TPM/TPD and release the
                // concurrency hold now the stream has ended. `add_tokens_post_stream`
                // is the sync analog of the reservation's async `commit_tokens`
                // (this end-of-stream closure can't await); dropping the hold frees
                // the concurrency slot(s) held for the stream's full lifetime.
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

                let metrics = AnthropicUsageMetrics {
                    prompt_tokens: usage.prompt_tokens,
                    completion_tokens: usage.completion_tokens,
                    cache_creation_tokens: usage.cache_creation_tokens,
                    cache_read_tokens: usage.cache_read_tokens,
                    provider_request_id: usage.provider_request_id,
                    provider_model_version: usage.provider_model_version,
                    finish_reason: usage.finish_reason,
                    ttft_ms: usage.ttft_ms,
                };
                emit_anthropic_usage_event(
                    &state_c,
                    &request_id_c,
                    &model_id_c,
                    &api_key_id_c,
                    &provider_c,
                    &model_name_c,
                    &provider_key_id_c,
                    &upstream_model_c,
                    team_id_c.as_deref(),
                    user_id_c.as_deref(),
                    user_name_c.as_deref(),
                    200,
                    started.elapsed(),
                    metrics,
                    &client_ctx_c,
                    attempt_c.clone(),
                    applied_guardrails_c.clone(),
                    // #932: input-side mask counts captured before dispatch,
                    // merged with the hold-back release's output-side counts.
                    {
                        let mut merged = input_redactions.clone();
                        crate::redact::merge_counts(&mut merged, usage.redacted_entity_counts);
                        merged
                    },
                    {
                        let mut merged = input_monitor_hits.clone();
                        merged.extend(usage.monitor_hits);
                        merged
                    },
                    // Prompt captured up front; response assembled by the frame
                    // parser into `usage.response_text`. Both gated on the cap.
                    match (&captured_prompt_c, content_cap) {
                        (Some(prompt), Some(cap)) => Some(CapturedContent::new(
                            prompt,
                            &usage.response_text,
                            cap as usize,
                        )),
                        _ => None,
                    },
                );
            },
        );

        let mut response =
            axum::response::Response::new(axum::body::Body::from_stream(parsed_stream));

        // Copy content-type from upstream (should be text/event-stream).
        if let Some(ct) = headers.get("content-type") {
            if let Ok(hv) = HeaderValue::from_bytes(ct.as_bytes()) {
                response
                    .headers_mut()
                    .insert(axum::http::header::CONTENT_TYPE, hv);
            }
        }
        // Set cache-control to no-cache for SSE.
        response.headers_mut().insert(
            axum::http::header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        );
        // Expose the request-id header.
        if let Ok(hv) = HeaderValue::from_str(request_id) {
            response
                .headers_mut()
                .insert(HeaderName::from_static("x-aisix-request-id"), hv);
        }

        // `usage_handled_by_stream: true` — the Drop guard inside
        // `build_anthropic_passthrough_stream` owns the UsageEvent
        // emission, so the top-level handler must NOT double-emit.
        // `metrics` here is unused on this path (the stream computes
        // the real counts at end-of-stream).
        Ok(DispatchOutcome {
            response,
            provider_label,
            provider_key_id: pk_id.to_string(),
            upstream_model: upstream_model.clone(),
            metrics: AnthropicUsageMetrics::default(),
            usage_handled_by_stream: true,
            routing: RoutingTelemetry::default(),
            // Streaming content capture lands in C3b.
            captured_content: None,
            // Streaming: the end-of-stream closure owns the counts.
            output_redactions: crate::redact::RedactionCounts::new(),
            output_monitor_hits: Vec::new(),
        })
    } else {
        // Non-streaming: deserialise and re-serialise as JSON. Decode
        // failures cool down the target — a body the bridge can't
        // parse is a real upstream problem worth taking out of
        // rotation, not a caller bug.
        let mut json_body: Value = upstream_resp
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

        let metrics = anthropic_metrics_from_response_json(&json_body);

        // #448 (#22): run output guardrails on the passthrough response.
        // The body is forwarded verbatim, so extract its text (content
        // blocks + the raw content array, which covers tool_use args) into
        // a synthetic ChatResponse for inspection before returning it.
        let mut output_seg_counts = crate::redact::RedactionCounts::new();
        let mut output_monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
        if !resolved_chain.is_empty() {
            if let Some(content) = json_body.get("content").and_then(|v| v.as_array()) {
                let mut out_text = String::new();
                for block in content {
                    if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                        if !out_text.is_empty() {
                            out_text.push('\n');
                        }
                        out_text.push_str(t);
                    }
                }
                if !out_text.is_empty() {
                    out_text.push('\n');
                }
                out_text.push_str(&Value::Array(content.clone()).to_string());

                let synth = aisix_gateway::ChatResponse {
                    id: String::new(),
                    model: model_name.to_string(),
                    message: aisix_gateway::ChatMessage::assistant(out_text),
                    finish_reason: aisix_gateway::FinishReason::Stop,
                    usage: aisix_gateway::UsageStats::new(0, 0),
                };
                let (verdict, hits) =
                    aisix_guardrails::Guardrail::check_output_non_segment_observed(
                        resolved_chain.as_ref(),
                        &synth,
                    )
                    .await;
                output_monitor_hits.extend(hits);
                let verdict = crate::redact::moderate_body(
                    resolved_chain.as_ref(),
                    crate::redact::Direction::Output,
                    verdict,
                    &mut output_seg_counts,
                    &mut output_monitor_hits,
                    |g| crate::redact::redact_anthropic_response(g, &mut json_body),
                )
                .await;
                if let aisix_guardrails::GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } = verdict
                {
                    tracing::warn!(
                        guardrail_hook = "output",
                        model = %model_name,
                        reason = %reason,
                        "guardrail blocked /v1/messages passthrough response",
                    );
                    return Err(ProxyError::ContentFiltered(
                        crate::error::guardrail_block_message(
                            "response",
                            guardrail_name.as_deref(),
                        ),
                    ));
                }
            }
        }

        // Restore the gateway-facing model name so callers see what they asked for.
        if let Some(m) = json_body.get_mut("model") {
            // If the upstream echoes the model name, rewrite to the gateway name.
            if m.as_str().map(|s| s == upstream_model).unwrap_or(false) {
                *m = Value::String(model_name.to_string());
            }
        }

        // #932: mask-action PII rules rewrite the passthrough response body
        // (text blocks + tool_use input) AFTER the block check passes.
        let mut output_redactions =
            crate::redact::redact_anthropic_response(resolved_chain.as_ref(), &mut json_body);
        crate::redact::merge_counts(&mut output_redactions, output_seg_counts);

        // Capture the prompt (the outbound request body) + assembled assistant
        // text for content-capturing exporters (gated). Built here, before
        // `json_body` is rendered into the response; threaded to `fan_out` via
        // `DispatchOutcome`, never to the CP sink.
        let captured_content = content_capture_cap(
            state
                .snapshot
                .load()
                .observability_exporters
                .entries()
                .iter()
                .map(|e| &e.value),
        )
        .map(|cap| {
            CapturedContent::new(
                &serde_json::to_string(&body).unwrap_or_default(),
                &anthropic_response_text(&json_body),
                cap as usize,
            )
        });

        Ok(DispatchOutcome {
            response: Json(json_body).into_response(),
            provider_label,
            provider_key_id: pk_id.to_string(),
            upstream_model,
            metrics,
            usage_handled_by_stream: false,
            routing: RoutingTelemetry::default(),
            captured_content,
            output_redactions,
            output_monitor_hits,
        })
    }
}

/// Concatenate the text from an Anthropic response's `content` blocks — the
/// assistant's assembled output text, for content-capturing exporters.
fn anthropic_response_text(body: &Value) -> String {
    body.get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

/// Pull `usage.input_tokens` / `output_tokens` / `cache_creation_input_tokens`
/// / `cache_read_input_tokens`, plus `id`, `model`, `stop_reason` from
/// an Anthropic non-streaming response body. Best-effort: missing
/// fields land as zero / empty string.
fn anthropic_metrics_from_response_json(body: &Value) -> AnthropicUsageMetrics {
    let usage = body.get("usage");
    AnthropicUsageMetrics {
        prompt_tokens: usage
            .and_then(|u| u.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        completion_tokens: usage
            .and_then(|u| u.get("output_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        cache_creation_tokens: usage
            .and_then(|u| u.get("cache_creation_input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        cache_read_tokens: usage
            .and_then(|u| u.get("cache_read_input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        provider_request_id: body
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        provider_model_version: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        finish_reason: body
            .get("stop_reason")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        ttft_ms: 0,
    }
}

/// Anthropic-protocol input → non-Anthropic upstream output.
///
/// Symmetric to `chat.rs::dispatch` but with Anthropic wire shapes on
/// both ends of the gateway:
///
/// 1. parse_inbound_request(body) → ChatFormat (gateway-internal)
/// 2. hub.get(model.provider) → Bridge for the configured upstream
/// 3. For non-streaming: bridge.chat → ChatResponse →
///    chat_response_into_anthropic_json
/// 4. For streaming: bridge.chat_stream → AnthropicSseEncoder pumps
///    each ChatChunk through the message_start / content_block_* /
///    message_* state machine and writes SSE bytes
#[allow(clippy::too_many_arguments)]
async fn cross_provider_dispatch(
    state: &ProxyState,
    body: &Value,
    model: &aisix_core::Model,
    model_id: &str,
    provider_key: &aisix_core::ProviderKey,
    model_name: &str,
    request_id: &str,
    started: Instant,
    api_key_id: &str,
    team_id: Option<String>,
    user_id: Option<String>,
    user_name: Option<String>,
    resolved_chain: std::sync::Arc<aisix_guardrails::GuardrailChain>,
    client: &ClientContext,
    attempt: AttemptInfo,
    reservation: &mut Option<aisix_ratelimit::MultiReservation>,
    input_redactions: crate::redact::RedactionCounts,
    input_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
) -> Result<DispatchOutcome, ProxyError> {
    use aisix_gateway::{Bridge, BridgeContext};
    use aisix_provider_anthropic::{
        chat_response_into_anthropic_json, parse_inbound_request, translate_extras_to_openai_shape,
        AnthropicSseEncoder,
    };
    use std::sync::Arc;

    let provider = model
        .provider
        .as_deref()
        .ok_or_else(|| {
            ProxyError::InvalidRequest(format!("model `{model_name}` has no provider prefix"))
        })?
        .to_string();
    let bridge: Arc<dyn Bridge> = crate::dispatch::resolve_bridge(&state.hub, provider_key)
        .ok_or(ProxyError::ProviderUnavailable)?;

    // Parse the Anthropic-shape body into the gateway's normalised
    // ChatFormat. Errors here are 400 — the request is malformed
    // before it even hits the bridge.
    let mut chat = parse_inbound_request(body)
        .map_err(|e| ProxyError::InvalidRequest(format!("invalid Anthropic body: {e}")))?;
    // Force the bridge dispatch to use the operator's display name
    // (`model_name`) so the bridge can re-resolve the upstream id
    // through `ctx.model.upstream_model()` exactly like chat.rs does.
    chat.model = model_name.to_string();

    // Rewrite Anthropic-shaped extras into the OpenAI chat shape the
    // non-Anthropic bridges expect: tools/tool_choice (#236),
    // stop_sequences/metadata/thinking are translated; Anthropic-only
    // fields (context_management, top_k, mcp_servers, …) are dropped —
    // flattened onto an OpenAI-compatible upstream they 400 as unknown
    // parameters (AISIX-Cloud#953).
    translate_extras_to_openai_shape(&mut chat.extra);

    let is_stream = chat.is_streaming();
    let model_arc = Arc::new(model.clone());
    let pk_arc = Arc::new(provider_key.clone());
    // #554: bound the upstream connect with the appropriate deadline —
    // the streaming read budget for stream calls, the E2E request timeout
    // otherwise. The streaming path additionally enforces the per-chunk
    // read timeout below.
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
    let provider_key_id = model.provider_key_id.as_deref().unwrap_or("unknown");
    let upstream_model = model.upstream_model().unwrap_or("unknown").to_string();

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
        // #554: when a streaming budget is configured (`stream_timeout`,
        // falling back to `timeout`), peek the first chunk so a slow/erroring
        // first token fails over (the caller loops to the next target) before
        // the 200 is committed. Without one, commit the stream directly
        // (pre-#554 behavior; a first-chunk error then surfaces in-band). The
        // wrapper keeps enforcing the read timeout on the remaining chunks
        // either way (no-op when unset).
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
            // Re-prepend the peeked chunk so the SSE encoder sees the whole
            // stream (and records TTFT on the first content chunk).
            Box::pin(
                futures::stream::once(std::future::ready(Ok::<_, aisix_gateway::BridgeError>(
                    first_chunk,
                )))
                .chain(upstream),
            )
        } else {
            upstream
        };
        state.health.record_success(model_name);
        state.runtime_status.mark_healthy(model_id);

        let message_id = format!("msg_{}", Uuid::new_v4().simple());
        let encoder = AnthropicSseEncoder::new(message_id, model_name, 0);
        let state_for_telem = state.clone();
        let request_id_for_telem = request_id.to_string();
        let model_id_for_telem = model_id.to_string();
        let api_key_id_for_telem = api_key_id.to_string();
        let provider_for_telem = provider_label.clone();
        let model_for_telem = model_name.to_string();
        let provider_key_id_for_telem = provider_key_id.to_string();
        let upstream_model_for_telem = upstream_model.clone();
        let team_id_for_telem = team_id;
        let user_id_for_telem = user_id;
        let user_name_for_telem = user_name;
        let started_for_telem = started;
        // #492: log the same client IP/UA on streamed responses.
        let client_for_telem = client.clone();
        // Winning-attempt classification (#655) for the stream-end emit.
        let attempt_for_telem = attempt.clone();
        // Applied guardrail set (#379), owned for the move into the
        // end-of-stream telemetry closure.
        let applied_guardrails_for_telem = resolved_chain.applied().to_vec();
        let stream_guardrail = if resolved_chain.is_empty() {
            None
        } else {
            Some(resolved_chain.clone())
        };
        // Content capture: prompt up front, response assembled in the stream
        // into `comp.response_text`. Both gated on `content_cap`.
        let content_cap = content_capture_cap(
            state
                .snapshot
                .load()
                .observability_exporters
                .entries()
                .iter()
                .map(|e| &e.value),
        );
        let captured_prompt_for_telem =
            content_cap.map(|_| serde_json::to_string(body).unwrap_or_default());
        // #688: carry the rate-limit reservation into the end-of-stream guard —
        // keys drive post-stream TPM/TPD accounting, the hold keeps the
        // concurrency slot(s) until the stream ends. `take()` leaves the
        // handler's `reservation` as `None` so it won't also `commit_tokens`.
        let post_stream_keys = reservation.as_ref().map(|r| r.keys()).unwrap_or_default();
        let stream_hold = reservation.take().map(|r| r.into_stream_hold());
        let limiter_for_stream = std::sync::Arc::clone(&state.limiter);
        let sse_body = build_anthropic_sse_stream(
            upstream,
            encoder,
            started,
            stream_guardrail,
            model_name.to_string(),
            content_cap,
            move |comp| {
                // #688: apply the terminal token cost to TPM/TPD and release the
                // concurrency hold now the stream has ended (sync analog of the
                // reservation's async `commit_tokens`, which this closure can't
                // await).
                let streamed_tokens = total_tokens_with_cache(
                    comp.prompt_tokens,
                    comp.completion_tokens,
                    comp.cache_creation_tokens,
                    comp.cache_read_tokens,
                );
                for key in &post_stream_keys {
                    limiter_for_stream.add_tokens_post_stream(key, streamed_tokens);
                }
                drop(stream_hold);
                // least_busy: stream over — this target is no longer
                // in-flight.
                drop(in_flight);

                let metrics = AnthropicUsageMetrics {
                    prompt_tokens: comp.prompt_tokens,
                    completion_tokens: comp.completion_tokens,
                    cache_creation_tokens: comp.cache_creation_tokens,
                    cache_read_tokens: comp.cache_read_tokens,
                    provider_request_id: comp.provider_request_id,
                    provider_model_version: comp.provider_model_version,
                    finish_reason: comp.finish_reason,
                    ttft_ms: comp.ttft_ms,
                };
                emit_anthropic_usage_event(
                    &state_for_telem,
                    &request_id_for_telem,
                    &model_id_for_telem,
                    &api_key_id_for_telem,
                    &provider_for_telem,
                    &model_for_telem,
                    &provider_key_id_for_telem,
                    &upstream_model_for_telem,
                    team_id_for_telem.as_deref(),
                    user_id_for_telem.as_deref(),
                    user_name_for_telem.as_deref(),
                    200,
                    started_for_telem.elapsed(),
                    metrics,
                    &client_for_telem,
                    attempt_for_telem.clone(),
                    applied_guardrails_for_telem.clone(),
                    // #932: input-side mask counts captured before dispatch,
                    // merged with the hold-back release's output-side counts.
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
                    // Prompt captured up front, response assembled across the
                    // stream into `comp.response_text`; both gated on the cap.
                    match (&captured_prompt_for_telem, content_cap) {
                        (Some(prompt), Some(cap)) => Some(CapturedContent::new(
                            prompt,
                            &comp.response_text,
                            cap as usize,
                        )),
                        _ => None,
                    },
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
        return Ok(DispatchOutcome {
            response,
            provider_label,
            provider_key_id: provider_key_id.to_string(),
            upstream_model,
            metrics: AnthropicUsageMetrics::default(),
            usage_handled_by_stream: true,
            routing: RoutingTelemetry::default(),
            // Streaming content capture lands in C3b.
            captured_content: None,
            // Streaming: the end-of-stream closure owns the counts.
            output_redactions: crate::redact::RedactionCounts::new(),
            output_monitor_hits: Vec::new(),
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

    // #448 (#22): run output guardrails on the cross-provider response
    // before rendering it back as Anthropic JSON — the response is
    // client-visible output just like /v1/chat/completions.
    let mut output_seg_counts = crate::redact::RedactionCounts::new();
    let mut output_monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    if !resolved_chain.is_empty() {
        let (verdict, hits) = aisix_guardrails::Guardrail::check_output_non_segment_observed(
            resolved_chain.as_ref(),
            &resp,
        )
        .await;
        output_monitor_hits.extend(hits);
        let verdict = crate::redact::moderate_body(
            resolved_chain.as_ref(),
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
                model = %model_name,
                reason = %reason,
                "guardrail blocked /v1/messages response",
            );
            return Err(ProxyError::ContentFiltered(
                crate::error::guardrail_block_message("response", guardrail_name.as_deref()),
            ));
        }
    }

    // #932: mask-action PII rules rewrite the bridged response AFTER the
    // block check passes, BEFORE it is rendered back as Anthropic JSON.
    let mut output_redactions =
        crate::redact::redact_chat_response(resolved_chain.as_ref(), &mut resp);
    crate::redact::merge_counts(&mut output_redactions, output_seg_counts);

    let metrics = AnthropicUsageMetrics {
        prompt_tokens: resp.usage.prompt_tokens,
        completion_tokens: resp.usage.completion_tokens,
        cache_creation_tokens: resp.usage.cache_creation_tokens,
        cache_read_tokens: resp.usage.cache_read_tokens,
        provider_request_id: resp.id.clone(),
        provider_model_version: resp.model.clone(),
        finish_reason: finish_reason_label(&resp.finish_reason),
        ttft_ms: 0,
    };
    // Capture the prompt (the Anthropic request body) + assembled assistant
    // text for content-capturing exporters (gated); threaded to `fan_out` via
    // `DispatchOutcome`, never to the CP sink.
    let captured_content = content_capture_cap(
        state
            .snapshot
            .load()
            .observability_exporters
            .entries()
            .iter()
            .map(|e| &e.value),
    )
    .map(|cap| {
        CapturedContent::new(
            &serde_json::to_string(body).unwrap_or_default(),
            resp.message.content.as_deref().unwrap_or(""),
            cap as usize,
        )
    });
    let json = chat_response_into_anthropic_json(&resp, model_name);
    Ok(DispatchOutcome {
        response: Json(json).into_response(),
        provider_label,
        provider_key_id: provider_key_id.to_string(),
        upstream_model,
        metrics,
        usage_handled_by_stream: false,
        routing: RoutingTelemetry::default(),
        captured_content,
        output_redactions,
        output_monitor_hits,
    })
}

/// Pump `ChatChunk`s through an `AnthropicSseEncoder` and emit each
/// resulting `AnthropicSseEvent` as `event: …\ndata: …\n\n` bytes.
/// Errors in the stream surface as a final `event: error` frame so
/// SSE clients see something actionable rather than a half-complete
/// stream.
fn build_anthropic_sse_stream(
    upstream: aisix_gateway::ChatChunkStream,
    encoder: aisix_provider_anthropic::AnthropicSseEncoder,
    started: Instant,
    output_guardrail: Option<std::sync::Arc<aisix_guardrails::GuardrailChain>>,
    model_label: String,
    // Largest content cap any content-capturing exporter wants, or `None` to
    // skip response accumulation (the common, content-free path).
    content_cap: Option<u32>,
    on_complete: impl FnOnce(AnthropicStreamCompletion) + Send + 'static,
) -> axum::body::Body {
    use futures::StreamExt;

    let mut encoder = encoder;
    // #932 / #466-class: when the chain's streamed-output policy is the
    // whole-response hold-back (BufferFull — keyword/pii/bedrock output
    // guardrails), chunks are withheld from the encoder until the
    // end-of-stream scan clears (and masks) them: a block keeps matched
    // content off the wire entirely, and a mask can't rewrite bytes that
    // already left. Window-policy guardrails (Azure/Aliyun) keep the
    // pre-existing live-forward + end-of-stream check on this surface.
    let hold_policy = output_guardrail.as_ref().and_then(|c| {
        match aisix_guardrails::Guardrail::stream_output_policy(c.as_ref()) {
            aisix_guardrails::StreamOutputPolicy::BufferFull {
                max_buffer_bytes, ..
            } => Some(max_buffer_bytes),
            _ => None,
        }
    });
    let stream = async_stream::stream! {
        let mut guard = CompleteAnthropicStreamOnDrop {
            slot: Some((on_complete, AnthropicStreamCompletion::default())),
        };
        let mut upstream = upstream;
        let mut first_chunk_seen = false;
        // Accumulate assistant text for the end-of-stream output guardrail
        // (#448). Without a hold-back policy, bytes are forwarded live and
        // a blocked response is signalled with a terminal `error` event.
        let mut content_text = String::new();
        // Also collect streamed tool-call fragments so tool-call output is
        // scanned too (parity with the non-streaming path). Fragments are
        // kept raw — the guardrail scans their serialized text, no need to
        // reassemble by index.
        let mut tool_call_fragments: Vec<serde_json::Value> = Vec::new();
        // Chunks withheld from the encoder until the end-of-stream scan
        // clears them (hold-back policies only). Held PRE-encode so the
        // mask rewrite can run on the normalised chunks.
        let mut held_chunks: Vec<aisix_gateway::ChatChunk> = Vec::new();
        let mut held_bytes: usize = 0;
        while let Some(item) = upstream.next().await {
            match item {
                Ok(chunk) => {
                    if !first_chunk_seen
                        && (chunk.delta.content.is_some() || chunk.delta.tool_calls.is_some())
                    {
                        first_chunk_seen = true;
                        guard.comp().ttft_ms =
                            started.elapsed().as_millis().min(u32::MAX as u128) as u32;
                    }
                    let comp = guard.comp();
                    if !chunk.id.is_empty() {
                        comp.provider_request_id = chunk.id.clone();
                    }
                    if !chunk.model.is_empty() {
                        comp.provider_model_version = chunk.model.clone();
                    }
                    if let Some(fr) = chunk.finish_reason.as_ref() {
                        comp.finish_reason = finish_reason_label(fr);
                    }
                    if let Some(u) = chunk.usage.as_ref() {
                        comp.prompt_tokens = comp.prompt_tokens.max(u.prompt_tokens);
                        comp.completion_tokens = comp.completion_tokens.max(u.completion_tokens);
                        comp.cache_creation_tokens =
                            comp.cache_creation_tokens.max(u.cache_creation_tokens);
                        comp.cache_read_tokens = comp.cache_read_tokens.max(u.cache_read_tokens);
                    }
                    if output_guardrail.is_some() {
                        if let Some(t) = chunk.delta.content.as_deref() {
                            content_text.push_str(t);
                        }
                        if let Some(tcs) = chunk.delta.tool_calls.as_ref() {
                            tool_call_fragments.extend(tcs.iter().cloned());
                        }
                    }
                    // Content capture: assemble the response (bounded to the
                    // cap), only when an exporter wants full content.
                    if let Some(cap) = content_cap {
                        if let Some(t) = chunk.delta.content.as_deref() {
                            if comp.response_text.len() < cap as usize {
                                comp.response_text.push_str(t);
                            }
                        }
                    }
                    if let Some(max_hold) = hold_policy {
                        // Hold-back: withhold the chunk until the end-of-
                        // stream scan clears it. Overflow fails closed —
                        // unscannable content must not be released.
                        held_bytes += chunk.delta.content.as_deref().map_or(0, str::len);
                        if held_bytes > max_hold {
                            tracing::warn!(
                                guardrail_hook = "output",
                                max_buffer_bytes = max_hold,
                                "streaming /v1/messages response exceeded hold-back cap; failing closed",
                            );
                            yield Ok(bytes::Bytes::from(guardrail_block_frame(None)));
                            return;
                        }
                        held_chunks.push(chunk);
                        continue;
                    }
                    for ev in encoder.next_events(&chunk) {
                        yield Ok::<_, std::io::Error>(bytes::Bytes::from(ev.to_sse_string()));
                    }
                    if encoder.is_finished() {
                        break;
                    }
                }
                Err(e) => {
                    // Hold-back: the held (unscanned) chunks are dropped —
                    // fail closed; only the error frame reaches the client.
                    let frame = format!(
                        "event: error\ndata: {{\"type\":\"error\",\"error\":{{\"type\":\"{}\",\"message\":{}}}}}\n\n",
                        e.error_type(),
                        serde_json::to_string(&e.to_string()).unwrap_or_else(|_| "\"error\"".into()),
                    );
                    yield Ok(bytes::Bytes::from(frame));
                    return;
                }
            }
        }
        // End-of-stream output guardrail (#448): scan the accumulated
        // assistant text and, on a block, emit a terminal Anthropic
        // `error` event instead of completing the stream cleanly.
        if let Some(chain) = output_guardrail.as_ref() {
            if !content_text.is_empty() || !tool_call_fragments.is_empty() {
                let mut message =
                    aisix_gateway::ChatMessage::assistant(std::mem::take(&mut content_text));
                if !tool_call_fragments.is_empty() {
                    // guardrail_output_text() serializes extra["tool_calls"],
                    // so streamed tool-call arguments are scanned too.
                    message.extra.insert(
                        "tool_calls".to_string(),
                        serde_json::Value::Array(std::mem::take(&mut tool_call_fragments)),
                    );
                }
                let synth = aisix_gateway::ChatResponse {
                    id: String::new(),
                    model: model_label.clone(),
                    message,
                    finish_reason: aisix_gateway::FinishReason::Stop,
                    usage: aisix_gateway::UsageStats::new(0, 0),
                };
                let (verdict, hits) =
                    aisix_guardrails::Guardrail::check_output_non_segment_observed(
                        chain.as_ref(),
                        &synth,
                    )
                    .await;
                guard.comp().monitor_hits.extend(hits);
                let mut seg_counts = crate::redact::RedactionCounts::new();
                let mut seg_hits = Vec::new();
                let verdict = crate::redact::moderate_body(
                    chain.as_ref(),
                    crate::redact::Direction::Output,
                    verdict,
                    &mut seg_counts,
                    &mut seg_hits,
                    |g| crate::redact::redact_chat_chunks(g, &mut held_chunks),
                )
                .await;
                guard.comp().monitor_hits.extend(seg_hits);
                if !seg_counts.is_empty() {
                    // Bedrock masked the held chunks — rebuild the content-
                    // capture accumulator from the masked content channel
                    // (the sync redactor below can't reproduce a provider-
                    // side mask), keeping the original soft cap
                    // (#932 × AISIX-Cloud#947).
                    if let Some(cap) = content_cap {
                        let mut rebuilt = String::new();
                        for c in held_chunks.iter() {
                            if rebuilt.len() >= cap as usize {
                                break;
                            }
                            if let Some(t) = c.delta.content.as_deref() {
                                rebuilt.push_str(t);
                            }
                        }
                        guard.comp().response_text = rebuilt;
                    }
                    crate::redact::merge_counts(
                        &mut guard.comp().redacted_entity_counts,
                        seg_counts,
                    );
                }
                if let aisix_guardrails::GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } = verdict
                {
                    tracing::warn!(
                        guardrail_hook = "output",
                        model = %model_label,
                        reason = %reason,
                        "guardrail blocked streaming /v1/messages response",
                    );
                    // Hold-back: the held chunks are dropped — the matched
                    // content never reached the wire.
                    let frame = guardrail_block_frame(guardrail_name.as_deref());
                    yield Ok(bytes::Bytes::from(frame));
                    return;
                }
            }
        }
        // Hold-back release (#932): the scan cleared — mask the held
        // chunks (channel reassembly across chunk boundaries), then feed
        // them through the encoder as if they had streamed live.
        if !held_chunks.is_empty() {
            if let Some(chain) = output_guardrail.as_ref() {
                let counts =
                    crate::redact::redact_chat_chunks(chain.as_ref(), &mut held_chunks);
                if !counts.is_empty() {
                    // The wire chunks were masked — mask the content-capture
                    // accumulator too, or the exported content would carry
                    // PII the client never saw (#932 × AISIX-Cloud#947).
                    crate::redact::redact_captured_output(
                        chain.as_ref(),
                        &mut guard.comp().response_text,
                    );
                    crate::redact::merge_counts(
                        &mut guard.comp().redacted_entity_counts,
                        counts,
                    );
                }
            }
            for chunk in held_chunks.drain(..) {
                for ev in encoder.next_events(&chunk) {
                    yield Ok::<_, std::io::Error>(bytes::Bytes::from(ev.to_sse_string()));
                }
                if encoder.is_finished() {
                    break;
                }
            }
        }
        if !encoder.is_finished() {
            for ev in encoder.force_finish() {
                yield Ok(bytes::Bytes::from(ev.to_sse_string()));
            }
        }
    };
    axum::body::Body::from_stream(stream)
}

/// Anthropic-shape SSE error frame for a streaming guardrail block. Built
/// with serde_json so an operator-supplied guardrail name is JSON-escaped
/// correctly; the message carries the firing guardrail's name (#519 B.4b)
/// but never the matched-pattern detail (#153).
fn guardrail_block_frame(guardrail_name: Option<&str>) -> String {
    format!(
        "event: error\ndata: {}\n\n",
        serde_json::json!({
            "type": "error",
            "error": {
                "type": "content_filter",
                "message": crate::error::guardrail_block_message("response", guardrail_name),
            }
        })
    )
}

fn finish_reason_label(reason: &aisix_gateway::FinishReason) -> String {
    use aisix_gateway::FinishReason;
    match reason {
        FinishReason::Stop => "stop".into(),
        FinishReason::Length => "length".into(),
        FinishReason::ContentFilter => "content_filter".into(),
        FinishReason::ToolCalls => "tool_calls".into(),
        FinishReason::Other(s) => s.clone(),
    }
}

#[derive(Default)]
struct AnthropicStreamCompletion {
    prompt_tokens: u32,
    completion_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    provider_request_id: String,
    provider_model_version: String,
    finish_reason: String,
    ttft_ms: u32,
    /// Assembled assistant text for content-capturing exporters, accumulated
    /// across chunks ONLY when an exporter wants full content (bounded to the
    /// capture cap). Empty otherwise. Read by the on_complete closure; never
    /// reaches the CP sink.
    response_text: String,
    /// Per-detector PII mask counts applied to the held stream at release
    /// (#932). Merged with the input-side counts by the on_complete emit.
    redacted_entity_counts: crate::redact::RedactionCounts,
    /// Monitor-mode guardrail observations made by the end-of-stream output
    /// check (AISIX-Cloud#562). Merged with the input-side hits by the
    /// on_complete emit.
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
}

struct CompleteAnthropicStreamOnDrop<F: FnOnce(AnthropicStreamCompletion)> {
    slot: Option<(F, AnthropicStreamCompletion)>,
}

impl<F: FnOnce(AnthropicStreamCompletion)> CompleteAnthropicStreamOnDrop<F> {
    fn comp(&mut self) -> &mut AnthropicStreamCompletion {
        &mut self
            .slot
            .as_mut()
            .expect("stream completion guard accessed after drop")
            .1
    }
}

impl<F: FnOnce(AnthropicStreamCompletion)> Drop for CompleteAnthropicStreamOnDrop<F> {
    fn drop(&mut self) {
        if let Some((f, c)) = self.slot.take() {
            f(c);
        }
    }
}

/// What `dispatch` produces alongside the wire response: enough
/// metadata for the outer wrapper to emit a UsageEvent with the
/// proper token counts and provider-detail fields.
struct DispatchOutcome {
    response: Response,
    provider_label: String,
    provider_key_id: String,
    upstream_model: String,
    metrics: AnthropicUsageMetrics,
    usage_handled_by_stream: bool,
    /// Per-attempt routing telemetry (#655). Carries every attempt that
    /// preceded the winner plus the winning attempt itself, so the
    /// handler can emit one `UsageEvent` per attempt sharing `request_id`.
    routing: RoutingTelemetry,
    /// Captured request/response content for the observability fan-out, gated
    /// on the snapshot's content-capturing exporters. `None` when none want it
    /// or on the streaming path (filled at stream end). Forwarded only to
    /// `fan_out`, never to the CP telemetry sink.
    captured_content: Option<CapturedContent>,
    /// Per-detector PII mask counts applied to the NON-streaming response
    /// body (#932). Merged with the input-side counts by `messages()`
    /// before the terminal emit. Empty on the streaming paths — their
    /// end-of-stream closures own the output-side counts.
    output_redactions: crate::redact::RedactionCounts,
    /// Monitor-mode guardrail observations on the response side
    /// (AISIX-Cloud#562), same lifecycle as `output_redactions`.
    output_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
}

/// Dispatch error carrying the per-attempt telemetry accumulated before
/// the request ultimately failed (#655). Mirrors `chat::DispatchFailure`.
struct MessagesDispatchError {
    err: ProxyError,
    routing: RoutingTelemetry,
}

impl MessagesDispatchError {
    /// Pre-dispatch failure (model-not-found, auth, budget, guardrail
    /// block before any upstream attempt): no recorded attempts.
    fn pre_dispatch(err: ProxyError) -> Self {
        Self {
            err,
            routing: RoutingTelemetry::default(),
        }
    }
}

impl From<ProxyError> for MessagesDispatchError {
    /// Every `?` in `dispatch`'s pre-attempt prelude converts here — those
    /// errors fire before any upstream attempt, so they carry no routing.
    fn from(err: ProxyError) -> Self {
        Self::pre_dispatch(err)
    }
}

/// Bundle of optional fields a UsageEvent emit-call wants when the
/// upstream actually returned tokens. All-defaults when called from
/// the error path or before token info is available.
#[derive(Default)]
struct AnthropicUsageMetrics {
    prompt_tokens: u32,
    completion_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    provider_request_id: String,
    provider_model_version: String,
    finish_reason: String,
    ttft_ms: u32,
}

/// Emit a UsageEvent for a `/v1/messages` request. Mirrors
/// `chat::emit_usage_event` but tagged `inbound_protocol = "anthropic"`
/// so the dashboard's Logs view can disambiguate the inbound SDK
/// from the upstream provider label.
///
/// Called from `messages()` once dispatch has produced a Response and
/// (for non-streaming) we know the token counts. Cross-provider
/// streaming calls invoke it from the stream completion callback after
/// observing the upstream chunks.
#[allow(clippy::too_many_arguments)]
fn emit_anthropic_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    api_key_id: &str,
    provider: &str,
    model: &str,
    provider_key_id: &str,
    upstream_model: &str,
    team_id: Option<&str>,
    user_id: Option<&str>,
    // #890 req-3: readable owner name (1:1 with user_id) for the metric label.
    user_name: Option<&str>,
    status_code: u16,
    elapsed: Duration,
    metrics: AnthropicUsageMetrics,
    client: &ClientContext,
    attempt: AttemptInfo,
    // The `{kind, hook}` set of guardrails that governed this request (#379).
    // Empty for the guardrail-free path and pre-resolution failures.
    applied_guardrails: Vec<AppliedGuardrail>,
    // Per-detector PII mask counts (#932), input + output merged. Detector
    // names only, never matched values. Empty = no redaction.
    redacted_entity_counts: crate::redact::RedactionCounts,
    // Monitor-mode guardrail observations (AISIX-Cloud#562), input +
    // output merged.
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    content: Option<CapturedContent>,
) {
    // Per-PK telemetry attribution (#302 M17 / AISIX-Cloud#436).
    // Same shape as chat.rs's emit_usage_event — look up the
    // resolved ProviderKey from the live snapshot and copy its
    // `telemetry_tags` into wire fields. Empty `provider_key_id`
    // (pre-dispatch error path) bypasses the lookup → wire NULL.
    let snap = state.snapshot.load();
    let tags = if !provider_key_id.is_empty() {
        snap.provider_keys
            .get_by_id(provider_key_id)
            .map(|e| e.value.telemetry_tags.clone())
            .unwrap_or_default()
    } else {
        Default::default()
    };
    // #890 req-3: readable provider-key name for the metric label (shared
    // resolver so chat + messages can't drift).
    let provider_key_name = crate::usage_attr::provider_key_metric_name(&snap, provider_key_id);
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        // `model` is the client-sent alias on every call path
        // (AISIX-Cloud#790) — the group name for routed requests.
        requested_model: model.to_string(),
        prompt_tokens: metrics.prompt_tokens,
        completion_tokens: metrics.completion_tokens,
        cache_creation_tokens: metrics.cache_creation_tokens,
        cache_read_tokens: metrics.cache_read_tokens,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        provider_request_id: metrics.provider_request_id,
        provider_model_version: metrics.provider_model_version,
        finish_reason: metrics.finish_reason,
        ttft_ms: metrics.ttft_ms,
        inbound_protocol: "anthropic".to_string(),
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
        applied_guardrails,
        redacted_entity_counts,
        guardrail_monitor_hits,
        ..Default::default()
    };
    // Handler label "messages" — Anthropic /v1/messages inbound
    // path. Bucketed prometheus counter (#408).
    state.usage_sink.try_emit("messages", event.clone());
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, content.as_ref(), exporters.iter().map(|e| &e.value));
    state.metrics.record_llm_usage(
        UsageLabels {
            endpoint: "/v1/messages",
            inbound_protocol: "anthropic",
            provider,
            model,
            upstream_model,
            provider_key_id,
            provider_key_name: &provider_key_name,
            api_key_id,
            team_id: team_id.unwrap_or("unknown"),
            user_id: user_id.unwrap_or("unknown"),
            user_name: user_name.unwrap_or("unknown"),
        },
        LlmUsage {
            input_tokens: metrics.prompt_tokens,
            output_tokens: metrics.completion_tokens,
            total_tokens: total_tokens_with_cache(
                metrics.prompt_tokens,
                metrics.completion_tokens,
                metrics.cache_creation_tokens,
                metrics.cache_read_tokens,
            )
            .min(u64::from(u32::MAX)) as u32,
            spend_usd: 0.0,
        },
    );
    // #890 req-4: token volume by inbound client type (covers streaming and
    // non-streaming — every /v1/messages usage event flows through here).
    state.metrics.record_llm_tokens_by_client(
        aisix_obs::client_type_from_user_agent(&client.user_agent),
        u64::from(metrics.prompt_tokens),
        u64::from(metrics.completion_tokens),
    );
    if metrics.ttft_ms > 0 {
        state.metrics.record_time_to_first_token(
            UsageLabels {
                endpoint: "/v1/messages",
                inbound_protocol: "anthropic",
                provider,
                model,
                upstream_model,
                provider_key_id,
                provider_key_name: &provider_key_name,
                api_key_id,
                team_id: team_id.unwrap_or("unknown"),
                user_id: user_id.unwrap_or("unknown"),
                user_name: user_name.unwrap_or("unknown"),
            },
            Duration::from_millis(u64::from(metrics.ttft_ms)),
        );
    }
}

// ─── Anthropic streaming usage parser (#245) ───────────────────────
//
// The Anthropic `/v1/messages` passthrough forwards the upstream SSE
// byte stream verbatim. To recover token counts for telemetry without
// altering the bytes the client sees, `build_anthropic_passthrough_stream`
// wraps the byte stream: it appends each chunk to a frame buffer,
// extracts complete SSE events (delimited by a blank line), and parses
// their `data:` JSON to accumulate usage — then yields the *original*
// bytes unchanged. A Drop guard fires `on_complete` exactly once at
// end-of-stream OR on client-disconnect (mirroring chat.rs's
// `CompleteOnDrop`), so a streamed request always ships a UsageEvent.

/// Upper bound on the in-flight SSE frame buffer (PR #436 audit
/// MEDIUM-2). Real Anthropic SSE frames are a few KB at most; this
/// ceiling only trips on a non-conformant upstream that never emits a
/// frame terminator, guarding against per-request memory exhaustion.
/// Shared with the `/v1/responses` streaming usage parser (#808).
pub(crate) const MAX_SSE_FRAME_BUF_BYTES: usize = 1 << 20; // 1 MiB

/// Accumulated usage observed across an Anthropic SSE stream.
/// Sourced from `message_start` (input + cache tokens, id, model) and
/// `message_delta` (running output_tokens, stop_reason). All fields
/// default to zero / empty when the upstream never emits the
/// corresponding frame.
#[derive(Default)]
struct AnthropicStreamUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    provider_request_id: String,
    provider_model_version: String,
    finish_reason: String,
    ttft_ms: u32,
    /// Count of upstream byte-chunks actually delivered to the client
    /// (read by the Drop guard for the #419 cost-leak gate).
    chunks_delivered: u32,
    /// Assistant text accumulated from `content_block_delta` frames, for
    /// the end-of-stream output guardrail (#448).
    response_text: String,
    /// Per-detector PII mask counts applied to the held stream at release
    /// (#932). Merged with the input-side counts by the on_complete emit.
    redacted_entity_counts: crate::redact::RedactionCounts,
    /// Monitor-mode guardrail observations made by the end-of-stream output
    /// check (AISIX-Cloud#562). Merged with the input-side hits by the
    /// on_complete emit.
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
}

/// Update the accumulator from one parsed SSE `data:` JSON object.
/// Best-effort: unrecognised `type` values are ignored. `started` +
/// `first_token_seen` drive the TTFT measurement (first content frame).
fn update_anthropic_usage(
    acc: &mut AnthropicStreamUsage,
    json: &Value,
    started: Instant,
    first_token_seen: &mut bool,
) {
    match json.get("type").and_then(Value::as_str) {
        Some("message_start") => {
            let msg = json.get("message");
            if let Some(usage) = msg.and_then(|m| m.get("usage")) {
                if let Some(t) = usage.get("input_tokens").and_then(Value::as_u64) {
                    acc.prompt_tokens = t as u32;
                }
                if let Some(t) = usage
                    .get("cache_creation_input_tokens")
                    .and_then(Value::as_u64)
                {
                    acc.cache_creation_tokens = t as u32;
                }
                if let Some(t) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
                    acc.cache_read_tokens = t as u32;
                }
                // message_start carries an initial output_tokens (often
                // 1); take it as a floor — message_delta supersedes with
                // the real total. max-wins guards against a provider that
                // double-emits or re-orders.
                if let Some(t) = usage.get("output_tokens").and_then(Value::as_u64) {
                    acc.completion_tokens = acc.completion_tokens.max(t as u32);
                }
            }
            if let Some(id) = msg.and_then(|m| m.get("id")).and_then(Value::as_str) {
                acc.provider_request_id = id.to_string();
            }
            if let Some(m) = msg.and_then(|m| m.get("model")).and_then(Value::as_str) {
                acc.provider_model_version = m.to_string();
            }
        }
        Some("content_block_start") | Some("content_block_delta") => {
            // First content frame → record time-to-first-token.
            if !*first_token_seen {
                *first_token_seen = true;
                acc.ttft_ms = started.elapsed().as_millis().min(u32::MAX as u128) as u32;
            }
            // Accumulate assistant output for the end-of-stream output
            // guardrail (#448). text streams as `delta.text`; tool_use
            // streams its name in `content_block.{name,input}` on
            // content_block_start and its arguments as `delta.partial_json`
            // on input_json_delta — scan all of it.
            if let Some(delta) = json.get("delta") {
                if let Some(t) = delta.get("text").and_then(Value::as_str) {
                    acc.response_text.push('\n');
                    acc.response_text.push_str(t);
                }
                if let Some(pj) = delta.get("partial_json").and_then(Value::as_str) {
                    acc.response_text.push_str(pj);
                }
            }
            if let Some(cb) = json.get("content_block") {
                if let Some(name) = cb.get("name").and_then(Value::as_str) {
                    acc.response_text.push('\n');
                    acc.response_text.push_str(name);
                }
                if let Some(input) = cb.get("input") {
                    if !input.is_null() {
                        acc.response_text.push('\n');
                        acc.response_text.push_str(&input.to_string());
                    }
                }
            }
        }
        Some("message_delta") => {
            if let Some(usage) = json.get("usage") {
                if let Some(v) = usage.get("output_tokens") {
                    if let Some(t) = v.as_u64() {
                        acc.completion_tokens = acc.completion_tokens.max(t as u32);
                    } else {
                        // PR #436 audit LOW-1: a `usage` object present but
                        // with a non-numeric `output_tokens` leaves
                        // completion_tokens at the message_start floor
                        // (often 1) — a silent under-count. Surface it so a
                        // wire-shape drift is visible to operators.
                        tracing::debug!(
                            output_tokens = %v,
                            "anthropic stream: message_delta usage.output_tokens \
                             is non-numeric; completion_tokens left at floor"
                        );
                    }
                }
                // AISIX-Cloud#952: newer Anthropic wire (and some relays)
                // report cumulative input/cache counts on message_delta —
                // for some backends that is the ONLY place they appear
                // (message_start ships no usable usage), which recorded
                // prompt_tokens=0. Harvest them here too, max-wins with
                // the message_start values (LiteLLM reads both frames).
                if let Some(t) = usage.get("input_tokens").and_then(Value::as_u64) {
                    acc.prompt_tokens = acc.prompt_tokens.max(t as u32);
                }
                if let Some(t) = usage
                    .get("cache_creation_input_tokens")
                    .and_then(Value::as_u64)
                {
                    acc.cache_creation_tokens = acc.cache_creation_tokens.max(t as u32);
                }
                if let Some(t) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
                    acc.cache_read_tokens = acc.cache_read_tokens.max(t as u32);
                }
            }
            if let Some(sr) = json
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(Value::as_str)
            {
                acc.finish_reason = sr.to_string();
            }
        }
        _ => {}
    }
}

/// Drain every complete SSE frame from `buf`, updating `acc`. A frame
/// ends at the first blank line (`\n\n`). Incomplete trailing bytes are
/// left in `buf` for the next chunk. The `data:` payload is parsed as
/// JSON; non-JSON or non-`data` frames are skipped.
fn drain_anthropic_sse_frames(
    buf: &mut Vec<u8>,
    acc: &mut AnthropicStreamUsage,
    started: Instant,
    first_token_seen: &mut bool,
) {
    // SSE event delimiter is a blank line. Anthropic emits `\n\n`;
    // tolerate `\r\n\r\n` defensively by normalising the search.
    while let Some(end) = find_frame_end(buf) {
        let frame: Vec<u8> = buf.drain(..end).collect();
        if let Some(data) = extract_sse_data_line(&frame) {
            if let Ok(json) = serde_json::from_slice::<Value>(data) {
                update_anthropic_usage(acc, &json, started, first_token_seen);
            }
        }
    }
}

/// Find the byte index just past the first SSE frame terminator
/// (`\n\n` or `\r\n\r\n`). Returns the number of bytes to drain
/// (frame + terminator), or `None` if no complete frame is buffered.
/// Shared with the `/v1/responses` streaming usage parser (#808).
pub(crate) fn find_frame_end(buf: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some(i + 2);
        }
        if i + 3 < buf.len()
            && buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some(i + 4);
        }
        i += 1;
    }
    None
}

/// Extract the `data:` payload bytes from one SSE frame. Returns the
/// JSON slice (after `data:` and an optional leading space), or `None`
/// if the frame has no data line. Only the first data line is read —
/// Anthropic emits single-line data for the frames we care about.
/// Shared with the `/v1/responses` streaming usage parser (#808).
pub(crate) fn extract_sse_data_line(frame: &[u8]) -> Option<&[u8]> {
    for line in frame.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if let Some(rest) = line.strip_prefix(b"data:") {
            let rest = rest.strip_prefix(b" ").unwrap_or(rest);
            return Some(rest);
        }
    }
    None
}

/// Drop guard that fires `on_complete` exactly once with the
/// accumulated usage — on normal end-of-stream AND on client
/// disconnect (the async-stream generator drops at its suspension
/// point). Applies the #419 cost-leak gate: if no byte-chunk reached
/// the client, the completion-side counters are zeroed (the prompt was
/// processed upstream regardless, so `prompt_tokens` is kept).
struct AnthropicStreamGuard<F: FnOnce(AnthropicStreamUsage)> {
    slot: Option<(F, AnthropicStreamUsage)>,
    delivered: Arc<AtomicU32>,
}

impl<F: FnOnce(AnthropicStreamUsage)> AnthropicStreamGuard<F> {
    fn usage(&mut self) -> &mut AnthropicStreamUsage {
        &mut self
            .slot
            .as_mut()
            .expect("AnthropicStreamGuard accessed after take")
            .1
    }
}

impl<F: FnOnce(AnthropicStreamUsage)> Drop for AnthropicStreamGuard<F> {
    fn drop(&mut self) {
        if let Some((f, mut usage)) = self.slot.take() {
            let delivered = self.delivered.load(Ordering::Relaxed);
            usage.chunks_delivered = delivered;
            if delivered == 0 {
                // No bytes crossed the wire (client aborted before the
                // first chunk). Don't bill the completion side; keep
                // prompt_tokens per the "prompts always billed"
                // industry contract (#419 parity).
                usage.completion_tokens = 0;
                usage.cache_creation_tokens = 0;
                usage.cache_read_tokens = 0;
            }
            f(usage);
        }
    }
}

/// Stream wrapper that counts delivered items (`poll_next ->
/// Ready(Some)`) into a shared atomic, read by the Drop guard for the
/// #419 cost-leak gate. Mirrors chat.rs's `DeliveryCounter`.
struct AnthropicDeliveryCounter<T> {
    inner: Pin<Box<dyn Stream<Item = T> + Send>>,
    delivered: Arc<AtomicU32>,
}

impl<T> Stream for AnthropicDeliveryCounter<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => {
                self.delivered.fetch_add(1, Ordering::Relaxed);
                Poll::Ready(Some(item))
            }
            other => other,
        }
    }
}

/// Wrap an Anthropic upstream byte stream so token usage is parsed
/// in-flight and `on_complete` fires once at end-of-stream (or
/// client-disconnect) with the accumulated counts. Bytes are forwarded
/// verbatim — the client sees the exact upstream SSE wire shape.
fn build_anthropic_passthrough_stream<S, F>(
    upstream: S,
    started: Instant,
    output_guardrail: Option<std::sync::Arc<aisix_guardrails::GuardrailChain>>,
    model_label: String,
    // When `Some`, the assembled `response_text` is preserved (not taken by the
    // guardrail scan) so the on_complete content capture can read it.
    content_cap: Option<u32>,
    on_complete: F,
) -> AnthropicDeliveryCounter<reqwest::Result<Bytes>>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
    F: FnOnce(AnthropicStreamUsage) + Send + 'static,
{
    let delivered = Arc::new(AtomicU32::new(0));
    let delivered_for_drop = Arc::clone(&delivered);
    // #932 / #466-class: when the chain's streamed-output policy is the
    // whole-response hold-back (BufferFull — keyword/pii/bedrock output
    // guardrails), the passthrough must buffer the raw SSE bytes rather
    // than forward them live: a block must keep matched content off the
    // wire entirely, and a mask can't be applied to bytes already sent.
    // Window-policy guardrails (Azure/Aliyun incremental release) keep the
    // pre-existing live-forward + end-of-stream check on this surface.
    let hold_policy = output_guardrail.as_ref().and_then(|c| {
        match aisix_guardrails::Guardrail::stream_output_policy(c.as_ref()) {
            aisix_guardrails::StreamOutputPolicy::BufferFull {
                max_buffer_bytes, ..
            } => Some(max_buffer_bytes),
            _ => None,
        }
    });
    let inner = async_stream::stream! {
        let mut guard = AnthropicStreamGuard {
            slot: Some((on_complete, AnthropicStreamUsage::default())),
            delivered: delivered_for_drop,
        };
        futures::pin_mut!(upstream);
        let mut buf: Vec<u8> = Vec::new();
        let mut first_token_seen = false;
        // Whole-response hold-back buffer (BufferFull policies only).
        let mut held: Vec<u8> = Vec::new();
        while let Some(item) = upstream.next().await {
            if let Ok(bytes) = &item {
                // Side-channel parse: copy into the frame buffer (the
                // original `bytes` is yielded unchanged below) and drain
                // any complete SSE frames into the accumulator.
                buf.extend_from_slice(bytes);
                drain_anthropic_sse_frames(
                    &mut buf,
                    guard.usage(),
                    started,
                    &mut first_token_seen,
                );
                // Bound the frame buffer (PR #436 audit MEDIUM-2). The
                // happy path drains complete frames above, so `buf`
                // only retains a partial trailing frame — normally a
                // few hundred bytes. A malformed / hostile upstream
                // that streams bytes WITHOUT a blank-line terminator
                // would otherwise grow `buf` unboundedly (per-request
                // memory exhaustion). Real Anthropic SSE frames are
                // well under a few KB, so a 1 MiB ceiling can only be
                // hit by a non-conformant stream; drop the buffer
                // (losing usage parsing for that pathological case)
                // rather than OOM. The bytes themselves still forward
                // to the client verbatim — only telemetry parsing is
                // affected.
                if buf.len() > MAX_SSE_FRAME_BUF_BYTES {
                    tracing::warn!(
                        buffered = buf.len(),
                        "anthropic stream: SSE frame buffer exceeded cap without a \
                         terminator; dropping buffer (usage parsing skipped for the \
                         oversized frame)"
                    );
                    buf.clear();
                }
                if let Some(max_hold) = hold_policy {
                    // Hold-back: withhold the bytes until the end-of-stream
                    // scan clears (and masks) them. Overflow fails closed —
                    // content that can't be fully buffered to scan must not
                    // be released (mirrors /v1/responses).
                    if held.len() + bytes.len() > max_hold {
                        tracing::warn!(
                            guardrail_hook = "output",
                            max_buffer_bytes = max_hold,
                            "streaming /v1/messages passthrough exceeded hold-back cap; failing closed",
                        );
                        yield Ok(Bytes::from(guardrail_block_frame(None)));
                        return;
                    }
                    held.extend_from_slice(bytes);
                    continue;
                }
            }
            // Forward the original item verbatim (Ok bytes OR Err — an
            // upstream error mid-stream is passed through; the
            // accumulator keeps whatever was captured before it). In
            // hold-back mode an Err lands here too: it is forwarded and
            // the held (unscanned) content is dropped — fail closed.
            let errored = item.is_err();
            yield item;
            if errored && hold_policy.is_some() {
                return;
            }
        }
        // End-of-stream output guardrail (#448): scan the accumulated
        // assistant text. On a block, emit a terminal Anthropic `error`
        // event. On the hold-back path (BufferFull) nothing has been
        // forwarded yet, so a block keeps the matched content off the
        // wire entirely; on the live-forward path (Window /
        // EndOfStreamCheck) the bytes were already forwarded verbatim
        // and the error frame is the trailing signal.
        let mut blocked = false;
        if let Some(chain) = output_guardrail.as_ref() {
            // Clone (not take) when content capture is on, so the assembled
            // response survives for the on_complete content capture below;
            // otherwise take it (nothing downstream reads it).
            let text = if content_cap.is_some() {
                guard.usage().response_text.clone()
            } else {
                std::mem::take(&mut guard.usage().response_text)
            };
            if !text.is_empty() {
                let synth = aisix_gateway::ChatResponse {
                    id: String::new(),
                    model: model_label.clone(),
                    message: aisix_gateway::ChatMessage::assistant(text),
                    finish_reason: aisix_gateway::FinishReason::Stop,
                    usage: aisix_gateway::UsageStats::new(0, 0),
                };
                let (verdict, hits) =
                    aisix_guardrails::Guardrail::check_output_non_segment_observed(
                        chain.as_ref(),
                        &synth,
                    )
                    .await;
                guard.usage().monitor_hits.extend(hits);
                // Segment pass over the held SSE bytes. Only meaningful in
                // hold-back mode (`held` is empty otherwise — and a chain
                // with a segment member always folds to BufferFull, so a
                // live-forward stream never carries one).
                let mut seg_counts = crate::redact::RedactionCounts::new();
                let mut seg_hits = Vec::new();
                let verdict = crate::redact::moderate_body(
                    chain.as_ref(),
                    crate::redact::Direction::Output,
                    verdict,
                    &mut seg_counts,
                    &mut seg_hits,
                    |g| match crate::redact::redact_anthropic_sse(g, &held) {
                        Some((rewritten, counts)) => {
                            held = rewritten;
                            counts
                        }
                        None => crate::redact::RedactionCounts::new(),
                    },
                )
                .await;
                guard.usage().monitor_hits.extend(seg_hits);
                if !seg_counts.is_empty() {
                    // Bedrock masked the held bytes — rebuild the content-
                    // capture accumulator from the masked text channels
                    // (the sync redactor can't reproduce a provider-side
                    // mask) (#932 × AISIX-Cloud#947).
                    if let Some(cap) = content_cap {
                        let mut rebuilt = crate::redact::anthropic_sse_text(&held);
                        let mut cut = (cap as usize).min(rebuilt.len());
                        while cut < rebuilt.len() && !rebuilt.is_char_boundary(cut) {
                            cut += 1;
                        }
                        rebuilt.truncate(cut);
                        guard.usage().response_text = rebuilt;
                    }
                    crate::redact::merge_counts(
                        &mut guard.usage().redacted_entity_counts,
                        seg_counts,
                    );
                }
                if let aisix_guardrails::GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } = verdict
                {
                    tracing::warn!(
                        guardrail_hook = "output",
                        model = %model_label,
                        reason = %reason,
                        "guardrail blocked streaming /v1/messages passthrough response",
                    );
                    blocked = true;
                    let frame = guardrail_block_frame(guardrail_name.as_deref());
                    yield Ok(Bytes::from(frame));
                }
            }
        }
        // Hold-back release (#932): the scan cleared — mask the held SSE
        // bytes (channel reassembly across frames) and release them.
        if hold_policy.is_some() && !blocked && !held.is_empty() {
            match output_guardrail
                .as_ref()
                .and_then(|c| crate::redact::redact_anthropic_sse(c.as_ref(), &held))
            {
                Some((rewritten, counts)) => {
                    // The wire bytes were masked — mask the content-capture
                    // accumulator too, or the exported content would carry
                    // PII the client never saw (#932 × AISIX-Cloud#947).
                    if let Some(c) = output_guardrail.as_ref() {
                        crate::redact::redact_captured_output(
                            c.as_ref(),
                            &mut guard.usage().response_text,
                        );
                    }
                    crate::redact::merge_counts(
                        &mut guard.usage().redacted_entity_counts,
                        counts,
                    );
                    yield Ok(Bytes::from(rewritten));
                }
                None => {
                    yield Ok(Bytes::from(std::mem::take(&mut held)));
                }
            }
        }
        // guard drops here → on_complete fires (delivery-gated).
    };
    AnthropicDeliveryCounter {
        inner: Box::pin(inner),
        delivered,
    }
}

fn emit_access_log(
    model: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    latency: Duration,
    request_id: &str,
    routing: &RoutingTelemetry,
) {
    // Per #655 the access log stays ONE line per request, carrying the
    // user-perceived `latency` + final status plus a routing summary; the
    // per-attempt detail lives in telemetry.
    let served_by = routing
        .winner()
        .map(|w| w.target_model.as_str())
        .filter(|s| !s.is_empty());
    AccessLog {
        method: "POST",
        path: "/v1/messages",
        status,
        latency,
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
    }
    .emit();
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {

    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use aisix_provider_anthropic::AnthropicBridge;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use axum::response::Response;
    use std::sync::Arc;
    use tower::ServiceExt;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    const ANTHROPIC_PK_ID: &str = "11111111-1111-1111-1111-111111111111";
    const OPENAI_PK_ID: &str = "22222222-2222-2222-2222-222222222222";
    const GOOGLE_PK_ID: &str = "33333333-3333-3333-3333-333333333333";
    const DEEPSEEK_PK_ID: &str = "44444444-4444-4444-4444-444444444444";

    #[test]
    fn finish_reason_label_uses_wire_names() {
        use aisix_gateway::FinishReason;

        assert_eq!(super::finish_reason_label(&FinishReason::Stop), "stop");
        assert_eq!(super::finish_reason_label(&FinishReason::Length), "length");
        assert_eq!(
            super::finish_reason_label(&FinishReason::ContentFilter),
            "content_filter"
        );
        assert_eq!(
            super::finish_reason_label(&FinishReason::ToolCalls),
            "tool_calls"
        );
        assert_eq!(
            super::finish_reason_label(&FinishReason::Other("custom".into())),
            "custom"
        );
    }

    fn anthropic_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "anthropic",
                "model_name": "claude-3-5-haiku-20241022",
                "provider_key_id": "{ANTHROPIC_PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn openai_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "{OPENAI_PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-2", m, 1)
    }

    fn anthropic_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"anthropic-up","secret":"sk-ant-test","api_base":"{api_base}","provider":"anthropic","adapter":"anthropic"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(ANTHROPIC_PK_ID, pk, 1)
    }

    fn openai_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-openai-test","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(OPENAI_PK_ID, pk, 1)
    }

    fn new_snap_anthropic(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk(api_base));
        snap
    }

    fn new_snap_openai(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(openai_pk(api_base));
        snap
    }

    fn apikey_entry(allowed: &[&str]) -> ResourceEntry<ApiKey> {
        let json = format!(
            r#"{{"key_hash": "8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891", "allowed_models": {}}}"#,
            serde_json::to_string(&allowed).unwrap()
        );
        let k: ApiKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("k-1", k, 1)
    }

    fn build_app(snap: AisixSnapshot) -> axum::Router {
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let handle = SnapshotHandle::new(snap);
        crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    fn make_req(body: serde_json::Value) -> Request<axum::body::Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    fn anthropic_response() -> serde_json::Value {
        serde_json::json!({
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-5-haiku-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3}
        })
    }

    #[tokio::test]
    async fn happy_path_non_streaming_returns_anthropic_response() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-ant-test"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic(&upstream.uri());
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
    }

    #[tokio::test]
    async fn model_field_is_rewritten_to_upstream_name() {
        let upstream = MockServer::start().await;
        // Expect upstream receives "claude-3-5-haiku-20241022" (no prefix).
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic(&upstream.uri());
        snap.models.insert(anthropic_model("my-claude"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify mock received the request (meaning the model field was
        // rewritten and the call was forwarded).
        upstream.verify().await;
    }

    // ─── /v1/messages × Anthropic passthrough × RequestOverrides ──────
    //
    // The four override primitives the OpenAI bridge applies on every
    // outbound chat request (param_renames / param_constraints /
    // default_body_fields / default_headers) must apply identically on
    // the Anthropic passthrough path too. These tests boot a mock
    // upstream that strict-matches the EXPECTED outbound body shape /
    // header after each override is applied — if the override silently
    // no-ops the matcher rejects the request and wiremock 404s, which
    // surfaces as a non-200 status here.
    //
    // Issue refs: ai-gateway#335 (`apply_param_constraints` not wired
    // on /v1/messages), ai-gateway#337 (same gap for
    // `apply_default_headers`). Same site / same fix covers
    // `param_renames` and `default_body_fields`.

    /// Build an Anthropic ProviderKey JSON with the given request
    /// override block. Mirrors `anthropic_pk` plus a `request: {...}`
    /// field that round-trips through serde.
    fn anthropic_pk_with_request_overrides(
        api_base: &str,
        request_overrides: serde_json::Value,
    ) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = serde_json::json!({
            "display_name": "anthropic-up",
            "secret": "sk-ant-test",
            "api_base": api_base,
            "request": request_overrides,
        });
        let pk: aisix_core::ProviderKey = serde_json::from_value(json).unwrap();
        ResourceEntry::new(ANTHROPIC_PK_ID, pk, 1)
    }

    fn new_snap_anthropic_with_overrides(
        api_base: &str,
        request_overrides: serde_json::Value,
    ) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(anthropic_pk_with_request_overrides(
                api_base,
                request_overrides,
            ));
        snap
    }

    #[tokio::test]
    async fn anthropic_passthrough_applies_param_renames() {
        // ai-gateway#335 / #337 root cause: messages.rs bypassed the
        // override apply pipeline. This test verifies the rename
        // primitive now fires on outbound. mock-llm matcher is
        // strict on body — the rename MUST be applied or wiremock
        // returns 404.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({"max_tokens_to_sample": 100}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "param_renames": {"max_tokens": "max_tokens_to_sample"}
            }),
        );
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "rename must rewrite max_tokens → max_tokens_to_sample on outbound"
        );
    }

    #[tokio::test]
    async fn anthropic_passthrough_clamps_temperature_via_param_constraints() {
        // ai-gateway#335: caller temperature 0.9 with override max 0.5
        // must arrive upstream as 0.5. The mock body matcher strict-
        // checks temperature == 0.5 — wiremock 404s on mismatch.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({"temperature": 0.5}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "param_constraints": {"temperature_max": 0.5}
            }),
        );
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": 0.9
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "temperature must clamp from 0.9 to 0.5 on outbound"
        );
    }

    #[tokio::test]
    async fn anthropic_passthrough_fills_default_body_fields_when_caller_omits() {
        // ai-gateway#335 sibling: caller omits top_p, override
        // populates it on outbound.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({"top_p": 0.9}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "default_body_fields": {"top_p": 0.9}
            }),
        );
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "missing top_p must be filled with override default 0.9"
        );
    }

    #[tokio::test]
    async fn anthropic_passthrough_injects_default_headers() {
        // ai-gateway#337: operator-injected custom header reaches
        // upstream. Strict header matcher on wiremock surfaces a 404
        // on miss.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-tenant-id", "acme-prod-42"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "default_headers": {"x-tenant-id": "acme-prod-42"}
            }),
        );
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "operator-injected x-tenant-id header must reach upstream"
        );
    }

    #[tokio::test]
    async fn anthropic_passthrough_default_headers_cannot_overwrite_x_api_key() {
        // Defense-in-depth: `x-api-key` is in
        // `aisix_provider_openai::overrides::RESERVED_DEFAULT_HEADERS`
        // — even if cp-api validation slips and lets the operator
        // register a default_headers entry with `x-api-key`, the apply
        // function MUST drop it so the PK's secret remains the auth
        // value upstream sees.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            // Strict: must match the PK's secret, NOT the override value.
            .and(header("x-api-key", "sk-ant-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "default_headers": {"x-api-key": "sk-attacker-hijack"}
            }),
        );
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "reserved x-api-key header must NOT be overwritten by default_headers"
        );
    }

    #[tokio::test]
    async fn unauthenticated_request_returns_401() {
        let snap = new_snap_anthropic("http://unused");
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"claude-haiku","messages":[],"max_tokens":10}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // Anthropic envelope: 401 → authentication_error (#336).
        assert_anthropic_error_envelope(resp, StatusCode::UNAUTHORIZED, "authentication_error")
            .await;
    }

    #[tokio::test]
    async fn forbidden_model_returns_403() {
        let snap = new_snap_anthropic("http://unused");
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["other-model"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        // Anthropic envelope: 403 → permission_error (#336).
        assert_anthropic_error_envelope(resp, StatusCode::FORBIDDEN, "permission_error").await;
    }

    #[tokio::test]
    async fn unknown_model_returns_404() {
        let snap = new_snap_anthropic("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        // Anthropic envelope: 404 → not_found_error (#336).
        assert_anthropic_error_envelope(resp, StatusCode::NOT_FOUND, "not_found_error").await;
    }

    /// Cross-provider path: client speaks Anthropic protocol but the
    /// resolved Model points at an OpenAI upstream. The handler now
    /// translates Anthropic body → ChatFormat, dispatches through the
    /// OpenAi bridge, and re-encodes the OpenAI response as
    /// Anthropic-shape JSON (`{type:"message", role:"assistant",
    /// content:[{type:"text",...}], stop_reason, usage}`).
    #[tokio::test]
    async fn non_anthropic_model_dispatches_through_bridge_and_returns_anthropic_shape() {
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        // Mock an OpenAI /chat/completions response. The proxy will
        // translate it back to Anthropic shape on the way out.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-XYZ",
                "object": "chat.completion",
                "created": 1_715_000_000_u64,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from GPT!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("my-claude-alias"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let app = crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache());

        // Anthropic-shape inbound body.
        let body = serde_json::json!({
            "model": "my-claude-alias",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        // Anthropic-shape envelope.
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(
            v["model"], "my-claude-alias",
            "echoes operator alias, not upstream id"
        );
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "Hello from GPT!");
        assert_eq!(
            v["stop_reason"], "end_turn",
            "OpenAI 'stop' → Anthropic 'end_turn'"
        );
        assert_eq!(v["usage"]["input_tokens"], 7);
        assert_eq!(v["usage"]["output_tokens"], 3);
    }

    /// #597: Claude Code/cc-switch send `role: "system"` inside
    /// `messages[]`. The cross-provider path must keep it as an OpenAI
    /// system message instead of rejecting the request with a 400.
    /// The wiremock matcher is strict on the translated body — if the
    /// system turn is dropped or reordered the upstream 404s.
    #[tokio::test]
    async fn non_anthropic_model_preserves_system_role_in_messages() {
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "system", "content": "respond in French"},
                    {"role": "user", "content": "hello again"},
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-XYZ",
                "object": "chat.completion",
                "created": 1_715_000_000_u64,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Bonjour!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("my-claude-alias"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let app = crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache());

        let body = serde_json::json!({
            "model": "my-claude-alias",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "respond in French"},
                {"role": "user", "content": "hello again"},
            ],
            "max_tokens": 100
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["content"][0]["text"], "Bonjour!");
    }

    /// Streaming variant: the client asks for SSE; we translate
    /// OpenAI delta chunks to Anthropic message_start /
    /// content_block_delta / message_stop events.
    #[tokio::test]
    async fn non_anthropic_model_streams_anthropic_sse_events() {
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        // OpenAI-style SSE stream with two content deltas + a done marker.
        let sse = "\
data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1715000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1715000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1715000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("my-claude-alias"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let app = crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache());

        let body = serde_json::json!({
            "model": "my-claude-alias",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": true,
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream"),
        );
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        // Anthropic-shape SSE event sequence.
        assert!(
            body.contains("event: message_start"),
            "missing message_start in:\n{body}"
        );
        assert!(body.contains("event: content_block_start"));
        assert!(body.contains("event: content_block_delta"));
        assert!(body.contains("\"text\":\"hel\""));
        assert!(body.contains("\"text\":\"lo\""));
        assert!(body.contains("event: content_block_stop"));
        assert!(body.contains("event: message_delta"));
        assert!(body.contains("\"stop_reason\":\"end_turn\""));
        assert!(body.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn non_anthropic_streaming_records_anthropic_usage_event_with_ttft() {
        use aisix_obs::UsageSink;
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"cmpl-359\",\"object\":\"chat.completion.chunk\",\"created\":1715000000,\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-359\",\"object\":\"chat.completion.chunk\",\"created\":1715000000,\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":13,\"completion_tokens\":4,\"total_tokens\":17}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_delay(std::time::Duration::from_millis(20))
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("my-claude-alias"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({
            "model": "my-claude-alias",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": true,
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let streamed = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(streamed.contains("event: message_stop"));

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("usage event sender dropped");
        assert_eq!(event.inbound_protocol, "anthropic");
        assert_eq!(event.prompt_tokens, 13);
        assert_eq!(event.completion_tokens, 4);
        assert_eq!(event.provider_request_id, "cmpl-359");
        assert_eq!(event.provider_model_version, "gpt-4o-2024-08-06");
        assert_eq!(event.finish_reason, "stop");
        assert!(
            event.ttft_ms > 0,
            "streaming /v1/messages telemetry must record TTFT"
        );
        assert!(rx.try_recv().is_err(), "usage event should be emitted once");
    }

    #[tokio::test]
    async fn upstream_error_returns_502() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic(&upstream.uri());
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        // 5xx upstream → 502 BadGateway → api_error per Anthropic
        // SDK ErrorType literal (#336).
        assert_anthropic_error_envelope(resp, StatusCode::BAD_GATEWAY, "api_error").await;
    }

    /// /v1/messages must emit the Anthropic-shape error envelope
    /// `{type:"error", error:{type, message}}` on every error site —
    /// closes #336. The pre-#336 OpenAI-shape envelope on /v1/messages
    /// made the Claude SDK fall through to a generic exception that
    /// dumped the entire body to the message field, losing the
    /// structured error context that drives retry / fallback logic.
    ///
    /// Inner `error.type` follows the Anthropic SDK's `ErrorType`
    /// literal (NOT the OpenAI envelope's DP-stable taxonomy) so
    /// customers branching on `e.body['error']['type']` against
    /// Anthropic-canonical strings stay portable. See
    /// `crate::error::anthropic_kind_from_status` for the
    /// ecosystem-aligned status→type mapping.
    /// Strict envelope-shape helper used across every error-path
    /// test below — keeps regression coverage tight against a flip
    /// back to OpenAI shape (audit HIGH-2).
    async fn assert_anthropic_error_envelope(
        resp: Response,
        expected_status: StatusCode,
        expected_kind: &str,
    ) -> serde_json::Value {
        assert_eq!(resp.status(), expected_status);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let env: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            env["type"], "error",
            "top-level discriminator must be \"error\""
        );
        assert_eq!(
            env["error"]["type"], expected_kind,
            "inner error.type must follow Anthropic SDK ErrorType literal"
        );
        assert!(env["error"]["message"].is_string());
        assert!(
            env["error"].get("code").is_none(),
            "OpenAI-only field `code` must be absent"
        );
        assert!(
            env["error"].get("param").is_none(),
            "OpenAI-only field `param` must be absent"
        );
        env
    }

    #[tokio::test]
    async fn upstream_5xx_emits_anthropic_envelope_api_error() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(500).set_body_string("engine internal panic"))
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic(&upstream.uri());
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        // 5xx upstream → 502 BadGateway via Bridge collapse; status
        // maps to `api_error` per Anthropic SDK ErrorType literal.
        let env = assert_anthropic_error_envelope(resp, StatusCode::BAD_GATEWAY, "api_error").await;
        // 5xx body redaction is preserved.
        let msg = env["error"]["message"].as_str().unwrap_or("");
        assert!(
            !msg.contains("engine internal panic"),
            "upstream 5xx body must be redacted on the Anthropic envelope, got: {msg}",
        );
        assert!(
            msg.contains("500"),
            "redacted message must surface the upstream status, got: {msg}",
        );
    }

    #[tokio::test]
    async fn unknown_model_emits_anthropic_envelope_not_found_error() {
        let snap = new_snap_anthropic("http://unused");
        snap.apikeys.insert(apikey_entry(&["claude-haiku"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "claude-haiku",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_anthropic_error_envelope(resp, StatusCode::NOT_FOUND, "not_found_error").await;
    }

    #[tokio::test]
    async fn missing_model_field_returns_400() {
        let snap = new_snap_anthropic("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        // 400 Bad Request — `model` field missing. Anthropic
        // envelope: 400 → invalid_request_error (#336).
        assert_anthropic_error_envelope(resp, StatusCode::BAD_REQUEST, "invalid_request_error")
            .await;
    }

    // ─── Cross-protocol matrix (Anthropic inbound × non-Anthropic) ─

    fn gemini_model(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "google",
                "model_name": "gemini-2.0-flash",
                "provider_key_id": "{GOOGLE_PK_ID}"
            }}"#
        );
        ResourceEntry::new("m-3", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn deepseek_model(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "deepseek",
                "model_name": "deepseek-chat",
                "provider_key_id": "{DEEPSEEK_PK_ID}"
            }}"#
        );
        ResourceEntry::new("m-4", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn gemini_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"gemini-up","secret":"ya29-test","api_base":"{api_base}","provider":"google","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(GOOGLE_PK_ID, pk, 1)
    }

    fn deepseek_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"deepseek-up","secret":"sk-deepseek","api_base":"{api_base}","provider":"deepseek","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(DEEPSEEK_PK_ID, pk, 1)
    }

    fn new_snap_gemini(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(gemini_pk(api_base));
        snap
    }

    fn new_snap_deepseek(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(deepseek_pk(api_base));
        snap
    }

    /// (Anthropic inbound) × (Gemini upstream). Anthropic body comes
    /// in, the gateway translates → ChatFormat, dispatches via the
    /// Gemini bridge (OpenAi-compat wire), translates the response
    /// back to Anthropic JSON. Together with the OpenAI-upstream test
    /// above this proves the cross-provider path works for every
    /// non-Anthropic Bridge in the workspace.
    #[tokio::test]
    async fn matrix_anthropic_in_gemini_upstream_non_streaming() {
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-gemini",
                "object": "chat.completion",
                "created": 1_715_000_000_u64,
                "model": "gemini-2.0-flash",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Gemini!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap_gemini(&upstream.uri());
        snap.models.insert(gemini_model("my-claude-via-gemini"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_family(
            aisix_core::Adapter::Anthropic,
            Arc::new(AnthropicBridge::new()),
        );
        hub.register_family(aisix_core::Adapter::Openai, Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let app = crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache());

        let body = serde_json::json!({
            "model": "my-claude-via-gemini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["model"], "my-claude-via-gemini");
        assert_eq!(v["content"][0]["text"], "Hello from Gemini!");
        assert_eq!(v["stop_reason"], "end_turn");
        assert_eq!(v["usage"]["input_tokens"], 8);
        assert_eq!(v["usage"]["output_tokens"], 4);
    }

    /// (Anthropic inbound) × (DeepSeek upstream).
    #[tokio::test]
    async fn matrix_anthropic_in_deepseek_upstream_non_streaming() {
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-deepseek",
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from DeepSeek!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 6, "completion_tokens": 5, "total_tokens": 11}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap_deepseek(&upstream.uri());
        snap.models.insert(deepseek_model("my-claude-via-ds"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_family(
            aisix_core::Adapter::Anthropic,
            Arc::new(AnthropicBridge::new()),
        );
        hub.register_family(aisix_core::Adapter::Openai, Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let app = crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache());

        let body = serde_json::json!({
            "model": "my-claude-via-ds",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["content"][0]["text"], "Hello from DeepSeek!");
    }

    /// (Anthropic inbound) × (Anthropic upstream) × (streaming).
    /// The existing happy-path covers non-streaming passthrough; this
    /// one pins that the SSE byte stream from the Anthropic upstream
    /// is forwarded verbatim — the typed events stay typed, no
    /// translation layer in between.
    #[tokio::test]
    async fn matrix_anthropic_in_anthropic_upstream_streaming() {
        let upstream = MockServer::start().await;
        let sse = "\
event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-5-haiku-20241022\",\"stop_reason\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_anthropic(&upstream.uri());
        snap.models.insert(anthropic_model("my-claude"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": true,
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        // Verbatim Anthropic typed events on the way out (passthrough,
        // not re-encoded by AnthropicSseEncoder).
        assert!(body.contains("event: message_start"));
        assert!(body.contains("event: content_block_delta"));
        assert!(body.contains("\"text\":\"hi\""));
        assert!(body.contains("event: message_stop"));
    }

    /// Issue #245 (dp-blocker): the Anthropic passthrough STREAMING
    /// path must record the upstream-billed token counts on the
    /// UsageEvent — parity with the OpenAI streaming fix (#225/#196).
    /// Pre-fix this path forwarded raw bytes and emitted
    /// `prompt_tokens=0 completion_tokens=0`, so every streaming
    /// /v1/messages request billed as zero. This test drives a
    /// realistic Anthropic SSE response (input_tokens in
    /// `message_start`, running output_tokens in `message_delta`) and
    /// asserts the emitted UsageEvent carries the real counts, plus
    /// the response bytes still pass through verbatim.
    #[tokio::test]
    async fn anthropic_passthrough_streaming_records_usage_from_sse_frames() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // Canonical Anthropic streaming wire shape:
        // - message_start carries usage.input_tokens (+ cache fields)
        //   and the message id / model
        // - message_delta carries the running usage.output_tokens and
        //   the terminal stop_reason
        let sse = "\
event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stream_245\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-5-haiku-20241022\",\"stop_reason\":null,\"usage\":{\"input_tokens\":37,\"cache_creation_input_tokens\":4,\"cache_read_input_tokens\":9,\"output_tokens\":1}}}\n\n\
event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello there\"}}\n\n\
event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":52}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    // Small delay so TTFT measurement is non-zero.
                    .set_delay(std::time::Duration::from_millis(20))
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let snap = new_snap_anthropic(&upstream.uri());
        snap.models.insert(anthropic_model("my-claude"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": true,
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Bytes pass through verbatim — the client still sees the exact
        // Anthropic SSE wire shape.
        let streamed =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        assert!(streamed.contains("event: message_start"));
        assert!(streamed.contains("\"text\":\"hello there\""));
        assert!(streamed.contains("event: message_stop"));

        // The UsageEvent must carry the real upstream counts (#245).
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("streaming /v1/messages must emit a UsageEvent (#245)")
            .expect("usage event sender dropped");
        assert_eq!(event.inbound_protocol, "anthropic");
        assert_eq!(
            event.prompt_tokens, 37,
            "prompt_tokens must mirror message_start usage.input_tokens",
        );
        assert_eq!(
            event.completion_tokens, 52,
            "completion_tokens must mirror message_delta usage.output_tokens (running total)",
        );
        assert_eq!(
            event.cache_creation_tokens, 4,
            "cache_creation_tokens from message_start",
        );
        assert_eq!(
            event.cache_read_tokens, 9,
            "cache_read_tokens from message_start",
        );
        assert_eq!(event.provider_request_id, "msg_stream_245");
        assert_eq!(event.provider_model_version, "claude-3-5-haiku-20241022");
        assert_eq!(event.finish_reason, "end_turn");
        assert_eq!(event.status_code, 200);
        assert!(
            event.ttft_ms > 0,
            "streaming /v1/messages telemetry must record TTFT",
        );
        assert!(rx.try_recv().is_err(), "usage event should be emitted once");
    }

    /// AISIX-Cloud#952: relay backends that ship NO usage on
    /// `message_start` (id/model present) and report cumulative
    /// input/cache counts only on the terminal `message_delta`. Pre-fix
    /// the emitted UsageEvent carried prompt_tokens=0 (stored as NULL by
    /// cp-api, shown as 0 in the dashboard).
    #[tokio::test]
    async fn anthropic_passthrough_streaming_harvests_input_tokens_from_message_delta() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let sse = "\
event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"gen_01REPRO952\",\"role\":\"assistant\",\"content\":[],\"model\":\"mco-5\",\"stop_reason\":null}}\n\n\
event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":11504,\"cache_creation_input_tokens\":4,\"cache_read_input_tokens\":9,\"output_tokens\":136}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let snap = new_snap_anthropic(&upstream.uri());
        snap.models.insert(anthropic_model("my-claude"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 200,
            "stream": true,
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        to_bytes(resp.into_body(), 65536).await.unwrap();

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("streaming /v1/messages must emit a UsageEvent")
            .expect("usage event sender dropped");
        assert_eq!(
            event.prompt_tokens, 11504,
            "input_tokens reported only on message_delta must be harvested (#952)",
        );
        assert_eq!(event.completion_tokens, 136);
        assert_eq!(event.cache_creation_tokens, 4);
        assert_eq!(event.cache_read_tokens, 9);
        assert_eq!(event.provider_request_id, "gen_01REPRO952");
        assert_eq!(event.provider_model_version, "mco-5");
    }

    /// Issue #245: the SSE frame parser must reassemble events that
    /// arrive split across byte-chunk boundaries (reqwest's
    /// `bytes_stream()` makes no frame-alignment guarantees). Drives
    /// `drain_anthropic_sse_frames` directly with a buffer that holds
    /// one complete frame plus a partial second frame, then completes
    /// the second frame on the next call.
    #[test]
    fn sse_frame_parser_reassembles_split_chunks() {
        use super::{drain_anthropic_sse_frames, AnthropicStreamUsage};

        let mut acc = AnthropicStreamUsage::default();
        let mut first_token_seen = false;
        let started = std::time::Instant::now();

        // First "chunk": a complete message_start frame + the start of
        // a message_delta frame (no terminating blank line yet).
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(
            b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":11}}}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2",
        );
        drain_anthropic_sse_frames(&mut buf, &mut acc, started, &mut first_token_seen);
        // Only the complete first frame is consumed.
        assert_eq!(acc.prompt_tokens, 11, "input_tokens parsed from frame 1");
        assert_eq!(acc.provider_request_id, "m1");
        assert_eq!(
            acc.completion_tokens, 0,
            "partial frame 2 must NOT be parsed until its terminator arrives",
        );

        // Second "chunk": the remainder of the message_delta frame.
        buf.extend_from_slice(b"3}}\n\n");
        drain_anthropic_sse_frames(&mut buf, &mut acc, started, &mut first_token_seen);
        assert_eq!(
            acc.completion_tokens, 23,
            "output_tokens parsed once the split frame is reassembled",
        );
        assert!(buf.is_empty(), "buffer fully drained after both frames");
    }

    /// Issue #245 (audit angle 8c): a stream that carries NO usage
    /// blocks at all — e.g. an Anthropic error stream — must drain
    /// cleanly leaving the accumulator at zeros, without panicking.
    /// Guards the best-effort parser against a frame shape it doesn't
    /// recognise.
    #[test]
    fn sse_frame_parser_tolerates_streams_without_usage() {
        use super::{drain_anthropic_sse_frames, AnthropicStreamUsage};

        let mut acc = AnthropicStreamUsage::default();
        let mut first_token_seen = false;
        let started = std::time::Instant::now();

        let mut buf: Vec<u8> = Vec::new();
        // An error-style stream: a `ping` frame, an `error` frame, no
        // message_start / message_delta and so no usage anywhere.
        buf.extend_from_slice(
            b"event: ping\ndata: {\"type\":\"ping\"}\n\n\
event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"overloaded\"}}\n\n",
        );
        drain_anthropic_sse_frames(&mut buf, &mut acc, started, &mut first_token_seen);

        assert_eq!(acc.prompt_tokens, 0, "no usage → prompt_tokens stays zero");
        assert_eq!(acc.completion_tokens, 0, "no usage → completion stays zero");
        assert!(
            acc.provider_request_id.is_empty(),
            "no message_start → no provider_request_id",
        );
        assert!(buf.is_empty(), "both frames drained even without usage");
    }

    /// Issue #245 / #419 parity: the stream Drop guard must zero the
    /// completion-side counters when no byte-chunk reached the client
    /// (mid-stream disconnect), while preserving prompt_tokens. Drives
    /// `AnthropicStreamGuard::drop` directly with the delivered atomic
    /// pre-set, mirroring chat.rs's CompleteOnDrop test discipline.
    #[test]
    fn stream_guard_zeroes_completion_when_nothing_delivered() {
        use super::{AnthropicStreamGuard, AnthropicStreamUsage, AtomicU32};
        use std::sync::{Arc, Mutex};

        fn drop_and_capture(
            usage: AnthropicStreamUsage,
            delivered_count: u32,
        ) -> AnthropicStreamUsage {
            let captured: Arc<Mutex<Option<AnthropicStreamUsage>>> = Arc::new(Mutex::new(None));
            let cap = captured.clone();
            let delivered = Arc::new(AtomicU32::new(delivered_count));
            {
                let guard = AnthropicStreamGuard {
                    slot: Some((
                        move |u: AnthropicStreamUsage| {
                            *cap.lock().unwrap() = Some(u);
                        },
                        usage,
                    )),
                    delivered,
                };
                drop(guard);
            }
            let out = captured.lock().unwrap().take().expect("on_complete fired");
            out
        }

        // delivered==0: completion side zeroed, prompt kept.
        let usage = AnthropicStreamUsage {
            prompt_tokens: 30,
            completion_tokens: 17,
            cache_creation_tokens: 3,
            cache_read_tokens: 2,
            ..Default::default()
        };
        let out = drop_and_capture(usage, 0);
        assert_eq!(out.prompt_tokens, 30, "prompt_tokens preserved (#419)");
        assert_eq!(
            out.completion_tokens, 0,
            "completion zeroed when delivered==0"
        );
        assert_eq!(out.cache_creation_tokens, 0);
        assert_eq!(out.cache_read_tokens, 0);
        assert_eq!(out.chunks_delivered, 0);

        // delivered>0: counts preserved.
        let usage = AnthropicStreamUsage {
            prompt_tokens: 30,
            completion_tokens: 17,
            ..Default::default()
        };
        let out = drop_and_capture(usage, 5);
        assert_eq!(
            out.completion_tokens, 17,
            "completion kept when delivered>0"
        );
        assert_eq!(out.chunks_delivered, 5);
    }

    /// Helper for the streaming variants of (Anthropic inbound) ×
    /// (Gemini | DeepSeek upstream). Both upstreams expose the
    /// OpenAi-compat `/chat/completions` endpoint with OpenAi-shape
    /// SSE deltas, so the assertion shape is identical. The PK is
    /// stamped with `adapter: "openai"` so the family bridge handles
    /// dispatch.
    async fn assert_anthropic_streams_through_openai_compat_upstream(
        bridge_provider: &str,
        model_entry: ResourceEntry<Model>,
        model_name: &str,
    ) {
        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1715000000,\"model\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1715000000,\"model\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"yo\"},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        // Build a fresh ProviderKey pointing at the wiremock URI; the
        // model_entry passed in carries the right `provider_key_id` to
        // bind it to that PK.
        let pk_id = model_entry
            .value
            .provider_key_id
            .clone()
            .expect("matrix fixtures must reference a provider_key_id");
        // The PK's vendor identity must match `bridge_provider` so
        // `dispatch_two_tier` hits the specialized bridge this test
        // registered. `adapter: "openai"` is right for both gemini
        // and deepseek (OpenAI-compat wire shapes).
        let pk_json = format!(
            r#"{{"display_name":"matrix-up","secret":"k","api_base":"{}","provider":"{bridge_provider}","adapter":"openai"}}"#,
            upstream.uri()
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_json).unwrap();

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(ResourceEntry::new(pk_id, pk, 1));
        snap.models.insert(model_entry);
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_family(
            aisix_core::Adapter::Anthropic,
            Arc::new(AnthropicBridge::new()),
        );
        hub.register_family(
            aisix_core::Adapter::Openai,
            Arc::new(aisix_provider_openai::OpenAiBridge::new()),
        );
        let handle = SnapshotHandle::new(snap);
        let app = crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache());

        let body = serde_json::json!({
            "model": model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": true,
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream"),
        );
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        // Anthropic-typed SSE events on the way out, regardless of
        // upstream wire shape.
        assert!(
            body.contains("event: message_start"),
            "missing message_start"
        );
        assert!(body.contains("event: content_block_delta"));
        assert!(body.contains("\"text\":\"yo\""));
        assert!(body.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn matrix_anthropic_in_gemini_upstream_streaming() {
        assert_anthropic_streams_through_openai_compat_upstream(
            "google",
            // Placeholder; helper rebuilds with the wiremock uri.
            gemini_model("my-claude-via-gemini"),
            "my-claude-via-gemini",
        )
        .await;
    }

    #[tokio::test]
    async fn matrix_anthropic_in_deepseek_upstream_streaming() {
        assert_anthropic_streams_through_openai_compat_upstream(
            "deepseek",
            deepseek_model("my-claude-via-ds"),
            "my-claude-via-ds",
        )
        .await;
    }
}
