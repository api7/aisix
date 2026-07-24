//! `POST /v1/chat/completions` handler.
//!
//! Flow:
//! 1. [`AuthenticatedKey`] extractor runs first — rejects unauthenticated
//!    requests with a 401 envelope.
//! 2. Parse [`ChatFormat`] from the JSON body.
//! 3. Resolve `req.model` against the snapshot's Model table → 404 if
//!    absent.
//! 4. Check the ApiKey's `allowed_models` whitelist → 403 if disallowed.
//! 5. Look up the matching `Bridge` on the Hub by `Model::provider()` →
//!    503 if no bridge registered.
//! 6. Rate-limit pre-commit; build [`BridgeContext`] and dispatch:
//!    - `stream == true`  → `chat_stream` + Sse response
//!    - otherwise          → `chat` + JSON response rendered as OpenAI
//! 7. On completion: record metrics + emit one structured access log
//!    line. Errors surface via [`ProxyError`] which carries the right
//!    status, error type, and (for rate-limits) Retry-After.

use aisix_cache::{Cache, CacheKey};
use aisix_core::AppliedGuardrail;
use aisix_gateway::{BridgeContext, BridgeError, ChatFormat};
use aisix_guardrails::GuardrailVerdict;
use aisix_obs::{
    content_capture_cap, AccessLog, CapturedContent, LatencyLabels, LlmUsage, Metrics,
    RequestLabels, RequestOutcome, UsageEvent, UsageLabels,
};
use axum::extract::State;
use axum::http::HeaderValue;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::{Stream, StreamExt};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::attempt::{attempt_error_message, routing_error_class, AttemptRecord, RoutingTelemetry};
use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::ProxyError;
use crate::render::{render_chunk, render_response};
use crate::routing::{is_retryable, resolve_attempt_models, AttemptModel, RoutingRequest};
use crate::state::ProxyState;

/// Header set on every non-streaming response indicating whether the
/// response came from the cache (`hit`) or the upstream (`miss`).
pub const CACHE_HEADER: &str = "x-aisix-cache";

struct DispatchFailure {
    model_id: Option<String>,
    charge: Option<UpstreamCharge>,
    err: ProxyError,
    routing: RoutingTelemetry,
}

impl DispatchFailure {
    fn new(model_id: Option<String>, charge: Option<UpstreamCharge>, err: ProxyError) -> Self {
        Self {
            model_id,
            charge,
            err,
            routing: RoutingTelemetry::default(),
        }
    }

    fn with_routing(mut self, routing: RoutingTelemetry) -> Self {
        self.routing = routing;
        self
    }
}

// Per-attempt cooldown decision lives in `crate::cooldown` so every
// dispatch path (chat, messages, responses, audio, rerank) shares the
// same logic. See cooldown.rs for the audit context (#264 H-1).
use crate::cooldown::decide_cooldown;

pub async fn chat_completions(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    // Issue #324: catch the JSON-extractor rejection here so we can
    // map it to OpenAI's documented 400 invalid_request_error wire
    // shape. Axum's `Json<T>` extractor returns 422 on JsonDataError
    // (valid-JSON-but-missing-required-field, e.g. no `model`),
    // which diverges from OpenAI — every SDK that branches on 400
    // vs 422 sees different semantics here than it does talking to
    // api.openai.com. Same discriminate-then-map pattern as
    // messages.rs (#336) which already handles this for /v1/messages.
    body: Result<Json<ChatFormat>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let started = Instant::now();
    let method = "POST";
    let path = "/v1/chat/completions";
    let mut req = match body {
        Ok(Json(r)) => r,
        Err(rej) => {
            use axum::extract::rejection::JsonRejection;
            use axum::http::StatusCode;
            // BytesRejection → distinguish 413 (PAYLOAD_TOO_LARGE,
            // real per-extractor cap exceeded) from 400 (transport-
            // side read failure). `JsonRejection` is `#[non_exhaustive]`
            // so the fallback `_` arm catches today's JsonDataError
            // (the #324 case) / JsonSyntaxError / MissingJsonContentType
            // AND any future variant axum adds, defaulting to 400
            // until each new variant gets an explicit policy decision.
            return match rej {
                JsonRejection::BytesRejection(inner)
                    if inner.status() == StatusCode::PAYLOAD_TOO_LARGE =>
                {
                    ProxyError::RequestTooLarge {
                        limit_bytes: state.request_body_limit_bytes,
                    }
                }
                JsonRejection::BytesRejection(_) => {
                    ProxyError::InvalidRequest("failed to read request body".into())
                }
                _ => ProxyError::InvalidRequest("invalid JSON request body".into()),
            }
            .into_response();
        }
    };
    let request_id = client.request_id.clone();
    let api_key_id = auth.entry.id.clone();
    let model_name = req.model.clone();

    // Filled by `dispatch` once the per-request guardrail chain resolves;
    // read below to attach `applied_guardrails` to the telemetry event on
    // both the success and failure (guardrail-block) paths (#379).
    let mut applied_guardrails: Vec<AppliedGuardrail> = Vec::new();
    // Filled by `dispatch` with per-detector PII mask counts (#932), same
    // dual-path lifecycle as `applied_guardrails`.
    let mut redaction_counts = crate::redact::RedactionCounts::new();
    // Filled by `dispatch` with monitor-mode guardrail observations
    // (AISIX-Cloud#562), same dual-path lifecycle as `applied_guardrails`.
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    let outcome = dispatch(
        &state,
        &auth,
        &mut req,
        &request_id,
        started,
        &client,
        &mut applied_guardrails,
        &mut redaction_counts,
        &mut monitor_hits,
    )
    .await;

    match outcome {
        Ok(mut success) => {
            let status = 200;
            let elapsed = started.elapsed();
            // #890 req-3/req-4: resolve the readable provider-key name from the
            // snapshot and normalise the inbound client type once.
            let provider_key_name = {
                let snap = state.snapshot.load();
                crate::usage_attr::provider_key_metric_name(&snap, &success.provider_key_id)
            };
            let client_type = state.client_classifier.classify(&client.user_agent);
            record_success(
                &state.metrics,
                &success.provider,
                &model_name,
                &api_key_id,
                auth.key().team_id.as_deref(),
                auth.key().user_id.as_deref(),
                auth.key().user_name.as_deref(),
                &provider_key_name,
                client_type,
                req.is_streaming(),
                success.routing.fallback_count() > 0,
                status,
                &success,
                elapsed,
            );
            emit_access_log(
                method,
                path,
                status,
                elapsed,
                Some(success.provider.as_str()),
                Some(&model_name),
                Some(&api_key_id),
                success.prompt_tokens,
                success.completion_tokens,
                success.total_tokens,
                &request_id,
                &success.routing,
                None,
            );
            // Per #655: emit a zero-token event for each failed attempt
            // that preceded the winner (non-streaming fallover). No-op for
            // direct-model success, cache hits, and the single-attempt
            // streaming path.
            emit_failed_attempts(
                &state,
                &request_id,
                &model_name,
                &api_key_id,
                &client,
                &applied_guardrails,
                &success.routing,
                // The winner's success event carries the content.
                /* content_for_last */
                None,
            );
            // The streaming path wires the WINNER's telemetry into the SSE
            // stream's on_complete callback (it has to wait for the
            // terminal chunk to read the upstream's `usage` block). Calling
            // `emit_usage_event` here for streaming would double-emit — once
            // with all-zero tokens at handler return, then again with the
            // real values at stream end.
            if !success.telemetry_handled_by_stream {
                // Winning-attempt metadata (#655). Direct models / cache
                // hits have no recorded attempt → defaults (index 0,
                // "initial", empty target). `latency_ms` is scoped to the
                // winning attempt itself; the access log above carries the
                // user-perceived total.
                let winner = success.routing.winner();
                let winner_latency = winner
                    .map(|w| Duration::from_millis(u64::from(w.latency_ms)))
                    .unwrap_or(elapsed);
                // AISIX-Cloud#790: the event's model_id is the winning
                // TARGET's id so pricing resolves against it; cache hits
                // record no attempt and keep the requested entry's id.
                let event_model_id = winner
                    .map(|w| w.target_model_id.as_str())
                    .unwrap_or(&success.model_id);
                emit_usage_event(
                    &state,
                    &request_id,
                    event_model_id,
                    &model_name,
                    &api_key_id,
                    status,
                    winner_latency,
                    success.prompt_tokens.unwrap_or(0) as u32,
                    success.completion_tokens.unwrap_or(0) as u32,
                    UsageExtras {
                        cached_prompt_tokens: success.cached_prompt_tokens,
                        reasoning_tokens: success.reasoning_tokens,
                        cache_creation_tokens: success.cache_creation_tokens,
                        cache_read_tokens: success.cache_read_tokens,
                        usage_estimated: success.usage_estimated,
                        provider_request_id: success.provider_request_id.clone(),
                        provider_model_version: success.provider_model_version.clone(),
                        finish_reason: success.finish_reason.clone(),
                        bypass_reason: success.bypass_reason.clone().unwrap_or_default(),
                        cache_status: success.cache_status.as_str().to_string(),
                        cache_hit_saved_input_tokens: success.cache_hit_saved_input_tokens,
                        cache_hit_saved_output_tokens: success.cache_hit_saved_output_tokens,
                        ttft_ms: 0,
                        attempt_index: winner.map(|w| w.index).unwrap_or(0),
                        attempt_kind: winner.map(|w| w.kind).unwrap_or("initial").to_string(),
                        attempt_model: winner.map(|w| w.target_model.clone()).unwrap_or_default(),
                        error_class: String::new(),
                        error_message: String::new(),
                        applied_guardrails: applied_guardrails.clone(),
                        provider_key_id: success.provider_key_id.clone(),
                        redacted_entity_counts: redaction_counts.clone(),
                        guardrail_monitor_hits: monitor_hits.clone(),
                    },
                    success.cost_usd,
                    /* guardrail_blocked */ false,
                    &client,
                    success.captured_content.take(),
                );
            }
            // Inject x-ratelimit-* headers so OpenAI SDK clients see the
            // current window state. We peek *after* the commit so
            // remaining-requests reflects the post-dispatch tally.
            let rl_limits = auth.key().rate_limit.clone().unwrap_or_default();
            if let Some(rl_status) = state.limiter.peek(&api_key_id, &rl_limits).await {
                crate::render::inject_ratelimit_headers(&mut success.response, &rl_status);
                state.metrics.set_rate_limit_remaining(
                    &api_key_id,
                    &model_name,
                    rl_status.rpm_remaining(),
                    rl_status.tpm_remaining(),
                );
            }
            // Correlation / routing headers.
            if let Ok(v) = axum::http::HeaderValue::try_from(request_id.as_str()) {
                success.response.headers_mut().insert("x-aisix-call-id", v);
            }
            // `x-aisix-served-by` exposes which routing target served
            // the request — see AISIX-Cloud#410. Only emitted when a
            // routing group was the entry point (direct models would
            // just echo `req.model`, which the body already carries).
            //
            // `HeaderValue::try_from` rejects CR/LF and non-visible
            // ASCII (RFC 7230) — correct from a response-splitting
            // standpoint, but if a routing target's `display_name`
            // carries such bytes the header is silently absent and
            // a customer debugging failover would see the same wire
            // shape as a direct-model response. Surface the rejection
            // in DP logs so operators can rename the offending target.
            if let Some(target) = success.served_by_target.as_deref() {
                match axum::http::HeaderValue::try_from(target) {
                    Ok(v) => {
                        success
                            .response
                            .headers_mut()
                            .insert("x-aisix-served-by", v);
                    }
                    Err(err) => {
                        tracing::warn!(
                            target_display_name = %target,
                            error = %err,
                            "target display_name is not a valid HTTP header value; \
                             omitting x-aisix-served-by — rename the target to use \
                             only visible ASCII (no CR/LF, no non-ASCII characters)"
                        );
                    }
                }
            }
            // `x-aisix-route` exposes which semantic route matched (#641).
            // Only present for a semantic router that resolved to a route;
            // absent on a fall-through to `default` and on non-semantic
            // requests. Same RFC 7230 header-value guard as above.
            if let Some(route) = success.served_by_route.as_deref() {
                match axum::http::HeaderValue::try_from(route) {
                    Ok(v) => {
                        success.response.headers_mut().insert("x-aisix-route", v);
                    }
                    Err(err) => {
                        tracing::warn!(
                            route_name = %route,
                            error = %err,
                            "semantic route name is not a valid HTTP header value; \
                             omitting x-aisix-route — rename the route to use only \
                             visible ASCII (no CR/LF, no non-ASCII characters)"
                        );
                    }
                }
            }
            success.response
        }
        Err(failure) => {
            let DispatchFailure {
                model_id: resolved_model_id,
                charge,
                err,
                routing,
            } = failure;
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            // #911 [27]: bound the `model` metric label to the configured set.
            // A pre-resolution failure (model-not-found) carries an arbitrary
            // caller-supplied `model_name` that must never become a Prometheus
            // label (unbounded cardinality). The raw name still flows to the
            // per-request access log + usage events below (bounded by request
            // volume, not label cardinality).
            let snap = state.snapshot.load();
            let metric_model = crate::usage_attr::metric_model_label(&snap, &model_name);
            record_error(&state.metrics, &err, metric_model, status, elapsed);
            // Access log: surface the upstream-billed counts when the
            // error fired AFTER the upstream call (output-content-filter
            // block). Pre-upstream errors (input filter, budget,
            // model-not-found) carry no charge — log None there so the
            // line reflects "request never reached the model".
            let (al_prompt, al_completion, al_total) = match charge.as_ref() {
                Some(c) => (
                    Some(u64::from(c.prompt_tokens)),
                    Some(u64::from(c.completion_tokens)),
                    Some(u64::from(c.prompt_tokens) + u64::from(c.completion_tokens)),
                ),
                None => (None, None, None),
            };
            // Routing telemetry lives on the charge for an output-blocked
            // request (the upstream succeeded, then the output guardrail
            // blocked it); otherwise on the failure itself.
            let routing = charge
                .as_ref()
                .map(|c| c.routing.clone())
                .filter(|r| !r.attempts.is_empty())
                .unwrap_or(routing);
            // #890 req-2: record the FAILED request on the same rich request
            // metrics successes use, so a success rate is computable
            // (numerator outcome="success" over a denominator that includes
            // failures — previously these series were success-path-only).
            // Provider / upstream_model / provider_key are unknown on the
            // failure path; identity + status + outcome + stream + is_fallback
            // are what the success-rate query needs.
            let fail_labels = RequestLabels {
                endpoint: "/v1/chat/completions",
                inbound_protocol: "openai",
                provider: "unknown",
                model: metric_model,
                upstream_model: "unknown",
                provider_key_id: "unknown",
                provider_key_name: "unknown",
                api_key_id: &api_key_id,
                team_id: auth.key().team_id.as_deref().unwrap_or("unknown"),
                user_id: auth.key().user_id.as_deref().unwrap_or("unknown"),
                user_name: auth.key().user_name.as_deref().unwrap_or("unknown"),
                stream: req.is_streaming(),
                is_fallback: routing.fallback_count() > 0,
                status,
                outcome: RequestOutcome::from_status(status),
            };
            state.metrics.record_proxy_request(fail_labels, elapsed);
            state.metrics.record_llm_request(fail_labels, elapsed);
            state.metrics.record_request_e2e_latency(
                LatencyLabels {
                    endpoint: "/v1/chat/completions",
                    model: metric_model,
                    provider: "unknown",
                    status,
                    streaming: req.is_streaming(),
                },
                elapsed,
            );
            emit_access_log(
                method,
                path,
                status,
                elapsed,
                None,
                Some(&model_name),
                Some(&api_key_id),
                al_prompt,
                al_completion,
                al_total,
                &request_id,
                &routing,
                Some(&err),
            );
            // `resolved_model_id` is populated by `dispatch` once
            // `req.model` resolves against the snapshot, so a guardrail /
            // budget / rate-limit / bridge error after that point still
            // records which model the request targeted. ContentFiltered
            // (guardrail) sets `guardrail_blocked` for the Blocked tab.
            let guardrail_blocked = matches!(err, ProxyError::ContentFiltered(_));
            let model_id_str = resolved_model_id.as_deref().unwrap_or("");
            // AISIX-Cloud#1013: failed requests carry the (post-mask)
            // request body so a 4xx/5xx can be triaged from the log alone.
            // Same opt-in gate and cap as the success path; 401/403 stay
            // body-less (a 401 here is upstream-auth passthrough — caller
            // 401s are rejected by the auth extractor before any event
            // exists) — the body adds nothing to an authorization
            // failure and callers probing keys shouldn't get their
            // payloads archived.
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
                        &serde_json::to_string(&req).unwrap_or_default(),
                        "",
                        cap as usize,
                    )
                })
            };
            // When every target failed there is no terminal event below —
            // the content rides the last failed attempt instead.
            let content_for_last = if charge.is_none() && !routing.attempts.is_empty() {
                failure_content.take()
            } else {
                None
            };
            // Per #655: emit one zero-token event for each FAILED attempt.
            // `applied_guardrails` (resolved by `dispatch`) is recorded on
            // every attempt so an input-blocked / failed request still
            // reports which guardrails governed it.
            emit_failed_attempts(
                &state,
                &request_id,
                &model_name,
                &api_key_id,
                &client,
                &applied_guardrails,
                &routing,
                content_for_last,
            );
            // Terminal event:
            //  - output-blocked (charge present): the WINNING attempt was
            //    billed by the upstream then blocked — emit it with the
            //    charge tokens + guardrail_blocked + winner metadata.
            //  - all-targets-failed (charge None, attempts present): every
            //    attempt was already emitted above — nothing more.
            //  - pre-dispatch error (charge None, no attempts): emit a
            //    single terminal event (attempt 0, "initial").
            match charge {
                Some(c) => {
                    let winner = routing.winner();
                    // AISIX-Cloud#790: the billed-then-blocked event is the
                    // winning attempt's — carry its TARGET id.
                    let event_model_id = winner
                        .map(|w| w.target_model_id.as_str())
                        .unwrap_or(model_id_str);
                    emit_usage_event(
                        &state,
                        &request_id,
                        event_model_id,
                        &model_name,
                        &api_key_id,
                        status,
                        elapsed,
                        c.prompt_tokens,
                        c.completion_tokens,
                        UsageExtras {
                            cached_prompt_tokens: c.cached_prompt_tokens,
                            reasoning_tokens: c.reasoning_tokens,
                            cache_creation_tokens: c.cache_creation_tokens,
                            cache_read_tokens: c.cache_read_tokens,
                            usage_estimated: c.usage_estimated,
                            provider_request_id: c.provider_request_id,
                            provider_model_version: c.provider_model_version,
                            finish_reason: c.finish_reason,
                            bypass_reason: c.bypass_reason,
                            cache_status: c.cache_status.as_str().to_string(),
                            cache_hit_saved_input_tokens: 0,
                            cache_hit_saved_output_tokens: 0,
                            ttft_ms: 0,
                            attempt_index: winner.map(|w| w.index).unwrap_or(0),
                            attempt_kind: winner.map(|w| w.kind).unwrap_or("initial").to_string(),
                            attempt_model: winner
                                .map(|w| w.target_model.clone())
                                .unwrap_or_default(),
                            error_class: String::new(),
                            error_message: String::new(),
                            // The chain governed the request even though it
                            // ultimately blocked on the output filter.
                            applied_guardrails: applied_guardrails.clone(),
                            provider_key_id: c.provider_key_id,
                            // Input-side masking happened before the output
                            // block — the audit trail keeps it.
                            redacted_entity_counts: redaction_counts.clone(),
                            guardrail_monitor_hits: monitor_hits.clone(),
                        },
                        /* cost_usd */ 0.0,
                        guardrail_blocked,
                        &client,
                        failure_content.take(),
                    );
                }
                None if routing.attempts.is_empty() => {
                    emit_usage_event(
                        &state,
                        &request_id,
                        model_id_str,
                        &model_name,
                        &api_key_id,
                        status,
                        elapsed,
                        /* prompt_tokens */ 0,
                        /* completion_tokens */ 0,
                        UsageExtras {
                            attempt_kind: "initial".to_string(),
                            // Bounded ProxyError class so the dashboard can
                            // show why the request never reached an upstream.
                            error_class: err.kind().to_string(),
                            // Input-blocked / pre-upstream errors still record
                            // the resolved guardrail chain (empty if it failed
                            // before resolution).
                            applied_guardrails: applied_guardrails.clone(),
                            // Input masking may have fired before the failure.
                            redacted_entity_counts: redaction_counts.clone(),
                            guardrail_monitor_hits: monitor_hits.clone(),
                            ..UsageExtras::default()
                        },
                        /* cost_usd */ 0.0,
                        guardrail_blocked,
                        &client,
                        failure_content.take(),
                    );
                }
                None => {
                    // All attempts failed; each was emitted above (the last
                    // one carrying the captured request body).
                }
            }
            err.into_response()
        }
    }
}

/// Everything needed to populate post-call telemetry for a successful
/// request. For streaming we don't yet have token totals, so those are
/// `None` and `total_tokens` stays at 0.
struct Success {
    response: Response,
    provider: String,
    /// UUID of the v3 Model row this request resolved to (the virtual
    /// model the caller asked for, before any routing fan-out). Empty
    /// in the unlikely case the resolver was bypassed.
    model_id: String,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    /// True when the token counters were filled by the local estimator
    /// because the upstream response carried no usage block
    /// (AISIX-Cloud#1074). Lands on `UsageEvent::usage_estimated`.
    usage_estimated: bool,
    /// Provider-specific cache + reasoning token counters. Default 0
    /// for providers that don't expose them; cp-api falls back to the
    /// standard prompt / completion rate when these are 0.
    cached_prompt_tokens: u32,
    reasoning_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    /// Provider response `id` (OpenAI chat.completion.id or Anthropic
    /// message id) — empty when the cached path served the request
    /// (re-using a stored response's id would mislead reconciliation).
    provider_request_id: String,
    /// Resolved model the provider actually billed.
    provider_model_version: String,
    provider_key_id: String,
    upstream_model: String,
    /// finish_reason / stop_reason as the upstream returned it. Empty
    /// for streaming (no terminal event yet) and cache hits.
    finish_reason: String,
    /// Cost computed via the per-key Budget's pricing table, in USD.
    /// 0.0 when no budget is configured for the key.
    cost_usd: f64,
    /// Set when at least one guardrail returned `Bypass` (remote-API
    /// guardrail upstream unreachable + `fail_open=true`). The first
    /// bypass reason wins. Goes onto `usage_events.guardrail_bypassed_reason`
    /// so a compliance audit can see what slipped past during a Bedrock
    /// outage. None for the normal Allow / Block paths.
    bypass_reason: Option<String>,
    /// Cache outcome for this request. `disabled` when no enabled
    /// cache_policy is in snapshot for the env; `miss` when the cache
    /// was consulted but no entry matched; `hit` when a stored entry
    /// served the response. Lands on `usage_events.cache_status` for
    /// the dashboard's /logs column. See `aisix-core::CachePolicy`
    /// (Stage 2) and `aisix-cache::Cache` for the source of truth.
    cache_status: CacheStatus,
    /// True when telemetry emission is wired into the SSE stream's
    /// on_complete callback (streaming path). The top-level handler
    /// must NOT call `emit_usage_event` again — that would emit one
    /// event with all-zero tokens at handler return on top of the real
    /// one from stream completion. Always false for non-streaming
    /// paths (handler emits inline with `success.prompt_tokens` etc.).
    telemetry_handled_by_stream: bool,
    /// On a cache HIT, the prompt + completion tokens of the cached
    /// response — the work the upstream would have repeated had the
    /// cache not served the request. Both 0 on miss / disabled. cp-api
    /// multiplies these by its pricing catalog to derive
    /// `cost_saved_usd` on ingestion (matches the existing `cost_usd`
    /// pattern: DP records tokens, cp-api owns pricing). See #88.
    cache_hit_saved_input_tokens: u32,
    cache_hit_saved_output_tokens: u32,
    /// Display name of the routing target that actually served this
    /// request (AISIX-Cloud#410). Surfaces in the `x-aisix-served-by`
    /// response header so callers can tell which target inside a
    /// routing group won the failover loop.
    ///
    /// `None` in two cases:
    ///   - Direct (non-routing) model — `display_name` of the served
    ///     target equals `req.model`, so the header would be redundant.
    ///   - Cache hit — we don't know which target produced the stored
    ///     response. Re-stamping a stale name would lie.
    ///
    /// Streaming routing responses still set this to the selected target.
    /// The streaming path attempts only that target and does not fail over
    /// mid-stream, but the header remains useful because callers asked for
    /// the routing group's display name.
    served_by_target: Option<String>,
    /// Name of the semantic route that matched this request (#641).
    /// Surfaces in the `x-aisix-route` response header so callers can see
    /// which intent the router resolved to. `None` for non-semantic
    /// requests and for a semantic request that fell through to `default`.
    served_by_route: Option<String>,
    routing: RoutingTelemetry,
    /// Captured request/response content for the observability fan-out, built
    /// (gated on the snapshot's content-capturing exporters) where the request
    /// and upstream response are both in scope. `None` when no exporter
    /// captures content, and on the streaming path (filled at stream end). It
    /// is forwarded only to `fan_out`, never to the CP telemetry sink.
    captured_content: Option<CapturedContent>,
}

/// Cache decision attached to every successful request. Wire shape
/// (lowercase string) is what cp-api persists in
/// `dpmgr_usage_events.cache_status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheStatus {
    /// No enabled cache policy in snapshot — gate skipped the lookup.
    /// Stage 3 will refine this to "no policy matched applies_to".
    Disabled,
    /// Cache consulted, no entry matched. The successful upstream
    /// response is stored on the way out so future identical requests
    /// hit. Maps to `x-aisix-cache: miss` on the response header.
    Miss,
    /// Cache consulted, entry returned without an upstream call. Maps
    /// to `x-aisix-cache: hit` on the response header.
    Hit,
}

impl CacheStatus {
    /// Lowercase wire string the DP ships to cp-api in `cache_status`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            CacheStatus::Disabled => "disabled",
            CacheStatus::Miss => "miss",
            CacheStatus::Hit => "hit",
        }
    }
}

/// Text of the latest user turn, used as the semantic-routing query.
/// Walks `messages` in reverse and returns the first `role: "user"`
/// message's text. `ChatMessage::content` already carries the
/// concatenated text of multimodal content blocks, so this also covers
/// vision/array-shaped messages (non-text blocks are skipped upstream).
/// Generated output text for the token-estimation fallback
/// (AISIX-Cloud#1074) — the non-streaming analog of the stream loop's
/// `est_output_text` accumulation: message content + reasoning +
/// tool-call name/argument text. Only built when estimation runs.
pub(crate) fn estimation_output_text(resp: &aisix_gateway::ChatResponse) -> String {
    let mut out = resp.message.content.clone().unwrap_or_default();
    if let Some(s) = resp
        .message
        .extra
        .get("reasoning_content")
        .and_then(|v| v.as_str())
    {
        out.push_str(s);
    }
    if let Some(tcs) = resp
        .message
        .extra
        .get("tool_calls")
        .and_then(|v| v.as_array())
    {
        for tc in tcs {
            if let Some(f) = tc.get("function") {
                if let Some(n) = f.get("name").and_then(|v| v.as_str()) {
                    out.push_str(n);
                }
                if let Some(a) = f.get("arguments").and_then(|v| v.as_str()) {
                    out.push_str(a);
                }
            }
        }
    }
    out
}

/// #1074 ensemble sub-call token fallback. A sub-call backend (a panel
/// member, or the judge) that reports no usage would otherwise record
/// silent zeros; estimate the prompt from that sub-call's own request
/// (`req`: the client request for a panel member — every member shares the
/// inbound messages — or the synthesis request for the judge) and the
/// completion from the sub-call's answer text. `model` is the sub-call's
/// resolved **upstream** model (not the operator alias) so the tokenizer
/// encoding matches the real backend, consistent with the direct and
/// streaming-judge paths. Returns `(prompt, completion, usage_estimated)`
/// under the #794 or-semantics: a non-zero upstream value always wins,
/// estimation fills only zeros. One helper so every sub-call emit site
/// (non-streaming panel + judge, streaming panel) can't drift apart.
fn estimate_subcall_tokens(
    req: &ChatFormat,
    model: &str,
    usage: &aisix_gateway::chat::UsageStats,
    output_text: &str,
) -> (u32, u32, bool) {
    if usage.prompt_tokens != 0 && usage.completion_tokens != 0 {
        return (usage.prompt_tokens, usage.completion_tokens, false);
    }
    let est = crate::token_estimate::Estimator::new(
        model,
        crate::token_estimate::PromptInput::Chat(Box::new(req.clone())),
    );
    let filled = crate::token_estimate::fill_missing(
        &est,
        usage.prompt_tokens,
        usage.completion_tokens,
        Some(output_text),
    );
    (
        filled.prompt_tokens,
        filled.completion_tokens,
        filled.estimated,
    )
}

fn last_user_message_text(req: &ChatFormat) -> Option<String> {
    req.messages
        .iter()
        .rev()
        .find(|m| m.role == aisix_gateway::Role::User)
        .and_then(|m| m.content.clone())
}

/// Compute the value of [`Success::served_by_target`] for a request.
///
/// Centralises the "should we surface routing identity?" policy so
/// the same rule applies on every dispatch branch (non-streaming
/// success, streaming success, cache hit). The rule is:
///
/// - Routing-group request **and** an attempt won → `Some(target)`.
/// - Anything else (direct model, no winner, cache hit) → `None`.
///
/// `None` is the wire signal "this was not a routing request" — the
/// `x-aisix-served-by` response header is emitted only when the
/// returned `Option` is `Some`, so its presence alone tells callers
/// that failover routing fired. Direct-model responses must never
/// carry the header.
fn served_by_target_for_routing(
    is_routing_request: bool,
    served_target: Option<String>,
) -> Option<String> {
    if is_routing_request {
        served_target
    } else {
        None
    }
}

#[cfg(test)]
mod served_by_target_tests {
    use super::served_by_target_for_routing;

    #[test]
    fn direct_model_returns_none_even_when_a_target_was_captured() {
        // A direct model still walks the attempt loop (with one
        // candidate), so `served_target` may be `Some`. The routing
        // flag is what gates header emission — direct models must
        // never carry `x-aisix-served-by`.
        assert_eq!(
            served_by_target_for_routing(false, Some("gpt-4-direct".into())),
            None,
        );
    }

    #[test]
    fn direct_model_with_no_target_is_none() {
        assert_eq!(served_by_target_for_routing(false, None), None);
    }

    #[test]
    fn routing_request_passes_winning_target_through() {
        assert_eq!(
            served_by_target_for_routing(true, Some("healthy-target".into())),
            Some("healthy-target".into()),
        );
    }

    #[test]
    fn routing_request_with_no_winner_is_none() {
        // All targets failed: header omitted. The caller is already
        // surfacing an error response; routing identity is moot.
        assert_eq!(served_by_target_for_routing(true, None), None);
    }
}

/// Returns `Ok(Success)` on the happy path, or `Err((resolved_model_id, err))`
/// where `resolved_model_id` is `Some(id)` once the request's `req.model`
/// has been resolved against the snapshot, and `None` for errors that
/// fire before resolution (empty messages, ModelNotFound). The caller
/// surfaces this id on the failure-path telemetry event so the
/// dashboard's `/logs` page can show the model that was targeted by a
/// guardrail-blocked / budget-exceeded / bridge-failed request.
/// Upstream-billed token counts attached to dispatch errors that fired
/// AFTER the upstream call already responded. Today this is populated
/// only on the output-content-filter block path: the gateway received a
/// full upstream response (which the provider has already charged for),
/// then the output guardrail decided to block delivery to the client.
/// The customer is still on the hook for those tokens, so the charge
/// flows up to the handler so its `usage_events` row reflects the bill.
///
/// `None` on every other error path — input-filter blocks, budget
/// failures, rate-limit rejections, model-not-found, etc. all happen
/// BEFORE any upstream call and the customer pays nothing.
struct UpstreamCharge {
    /// UUID of the resolved ProviderKey. Threaded through so
    /// `emit_usage_event` can look up `telemetry_tags` even on the
    /// output-guardrail-block path (where dispatch reached the
    /// upstream and billed the request). Empty if for some reason the
    /// charge was assembled without a known PK.
    provider_key_id: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    /// True when the counters above were filled by the local estimator
    /// (AISIX-Cloud#1074) — the billed-then-blocked upstream response
    /// carried no usage block.
    usage_estimated: bool,
    cached_prompt_tokens: u32,
    reasoning_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    provider_request_id: String,
    provider_model_version: String,
    finish_reason: String,
    /// First bypass reason observed on this request (input or output
    /// guardrail returned `Bypass` because a remote-API guardrail was
    /// unreachable + `fail_open=true`). Empty string = no bypass.
    /// Carried so the output-block telemetry surfaces "this blocked
    /// request had a degraded input check" — without this, an operator
    /// auditing a guardrail bypass would only see the bypass on
    /// successfully-served requests, never on output-blocked ones.
    bypass_reason: String,
    /// Cache outcome decided BEFORE the output guardrail ran. The
    /// dashboard's cache-status filter counts misses vs disabled vs
    /// hits — without forwarding it on the output-block path the
    /// blocked-but-billed request is mis-bucketed as "no cache decision".
    cache_status: CacheStatus,
    routing: RoutingTelemetry,
}

#[allow(clippy::too_many_arguments)]
async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    // `&mut` so mask-action PII guardrails (#932) can rewrite the request
    // text in place before it reaches semantic routing, the cache key, or
    // the upstream.
    req: &mut ChatFormat,
    request_id: &str,
    started: Instant,
    client: &ClientContext,
    // Out-param: filled with the resolved chain's `{kind, hook}` set as soon
    // as the guardrail chain is resolved, so the caller can attach it to the
    // telemetry event on BOTH the success and error (guardrail-block) paths
    // without threading a field through every `Success`/`DispatchFailure`
    // construction site. Stays empty for requests rejected before resolution.
    applied_out: &mut Vec<AppliedGuardrail>,
    // Out-param: per-detector PII mask counts (#932), same lifecycle as
    // `applied_out`. Input-side counts land here as soon as the request is
    // rewritten; the non-streaming output side merges in later. Streaming
    // output counts travel via `StreamCompletion` instead (the event is
    // emitted at end-of-stream).
    redactions_out: &mut crate::redact::RedactionCounts,
    // Out-param: monitor-mode guardrail observations (AISIX-Cloud#562),
    // same lifecycle as `redactions_out`.
    monitor_hits_out: &mut Vec<aisix_core::GuardrailMonitorHit>,
) -> Result<Success, DispatchFailure> {
    if req.messages.is_empty() {
        return Err(DispatchFailure::new(
            None,
            None,
            ProxyError::InvalidRequest("messages array must not be empty".into()),
        ));
    }

    let snapshot = state.snapshot.load();
    // Largest content cap any enabled content-capturing exporter wants, or
    // `None` when none do — computed once so each response path (cache hit /
    // upstream) only captures when an exporter actually consumes it.
    let content_cap = content_capture_cap(
        snapshot
            .observability_exporters
            .entries()
            .iter()
            .map(|e| &e.value),
    );
    let virtual_entry =
        crate::model_resolve::resolve_model(&snapshot, &req.model).ok_or_else(|| {
            DispatchFailure::new(None, None, ProxyError::ModelNotFound(req.model.clone()))
        })?;
    let model_id = virtual_entry.id.clone();

    // Every error from here on attaches the resolved model_id so the
    // failure-path telemetry event in `chat_completions` can surface
    // which model the request targeted.
    // Pre-upstream errors carry `None` for the charge — by definition
    // they fire before any provider billing happens. The output-filter
    // path below is the only site that builds a `Some(UpstreamCharge)`
    // (manually, not via this helper).
    let with_model = |e: ProxyError| DispatchFailure::new(Some(model_id.clone()), None, e);

    if !auth.key().can_access(&req.model) {
        return Err(with_model(ProxyError::ModelForbidden(req.model.clone())));
    }

    // Client-IP allowlist gate (#557): reject before any guardrail / upstream
    // work when the resolved source IP is outside the model's allowed_cidrs.
    crate::dispatch::check_ip_access(&virtual_entry.value, &client.source_ip)
        .map_err(with_model)?;

    // Resolve the per-request guardrail chain from the index.
    // Done once here; `resolved_chain` is reused for both the input
    // check below, the output check later, and the streaming output
    // guardrail context.
    let guardrail_ctx = aisix_guardrails::RequestContext {
        model_id: &model_id,
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved = state.guardrail_index.resolve(&guardrail_ctx);
    // Capture the applied `{kind, hook}` set before the concrete chain is
    // erased to `Arc<dyn Guardrail>` (the trait has no `applied()`). Fill the
    // caller's out-param so the failure path can surface it too, and keep a
    // local copy for the streaming on_complete closure below.
    let applied_guardrails = resolved.applied().to_vec();
    *applied_out = applied_guardrails.clone();
    let resolved_chain: std::sync::Arc<dyn aisix_guardrails::Guardrail> =
        std::sync::Arc::new(resolved);

    // Input guardrails. Run before reservation so a blocked prompt
    // doesn't burn an RPM slot — content-policy refusals shouldn't
    // count against quota. Bypass (remote-API guardrail unavailable
    // + fail_open=true) doesn't short-circuit; the reason is stashed
    // and attached to the telemetry event when the request finishes.
    let mut bypass_reason: Option<String> = None;
    // Split moderation: local/blob members check first (on the original
    // text), then the segment pass runs the Bedrock call and writes
    // ANONYMIZE masks back into `req` (#932 bedrock follow-up).
    let (input_verdict, hits) = resolved_chain.check_input_non_segment_observed(req).await;
    monitor_hits_out.extend(hits);
    let input_verdict = crate::redact::moderate_body(
        resolved_chain.as_ref(),
        crate::redact::Direction::Input,
        input_verdict,
        redactions_out,
        monitor_hits_out,
        |g| crate::redact::redact_chat_format(g, req),
    )
    .await;
    match input_verdict {
        GuardrailVerdict::Allow => {}
        GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } => {
            // The verdict's `reason` carries matched-pattern detail
            // (e.g. `"input blocked by literal \"forbidden-token\""`).
            // Keep it for operator logs but DO NOT propagate it to the
            // wire envelope — see #153. The redacted public message
            // carries only the firing guardrail's name (#519 B.4b) so
            // callers can't enumerate the blocklist by inspecting
            // error responses.
            // AISIX-Cloud#1013: the blocked request's body is captured
            // into full-content exporters, so run the mask-action rewrite
            // BEFORE returning — otherwise the capture would export the
            // pre-mask text the success path would have masked. (Remote
            // segment masks — moderate_body's Bedrock pass — are still
            // skipped: the request is dead, don't burn a provider call;
            // those entities were never locally detected.)
            crate::redact::merge_counts(
                redactions_out,
                crate::redact::redact_chat_format(resolved_chain.as_ref(), req),
            );
            tracing::warn!(
                guardrail_hook = "input",
                model = %req.model,
                reason = %reason,
                "guardrail blocked request"
            );
            return Err(with_model(ProxyError::ContentFiltered(
                crate::error::guardrail_block_message("request", guardrail_name.as_deref()),
            )));
        }
        GuardrailVerdict::Bypass { reason } => {
            bypass_reason = Some(reason);
        }
    }

    // #932: mask-action PII rules rewrite the request in place AFTER the
    // block check passes and BEFORE any downstream use — the semantic-
    // routing embedding, the cache-key fingerprint, and the upstream
    // dispatch all see the masked text, so the matched values never leave
    // the gateway.
    crate::redact::merge_counts(
        redactions_out,
        crate::redact::redact_chat_format(resolved_chain.as_ref(), req),
    );

    // Budget pre-check via cp-api. The DP no longer owns budget state;
    // cp-api returns a cached/live decision per api_key.
    let decision = state.budgets.check(&auth.entry.id).await;
    if let Some(budget) = decision.budget.as_ref() {
        record_budget_gauges(&state.metrics, auth, Some(budget));
    } else {
        record_budget_gauges(&state.metrics, auth, None);
    }
    if !decision.allowed {
        return Err(with_model(ProxyError::BudgetExceeded(Box::new(
            decision.reason.unwrap_or_else(|| {
                crate::budget::BudgetReason::message_only(auth.entry.id.clone())
            }),
        ))));
    }

    // Resolve the attempt-list of underlying Model entries. A semantic
    // router embeds the request and scores it against route examples to
    // pick its single target; a routing group walks targets per strategy;
    // a direct model dispatches to itself. Either way the dispatch loop
    // below drives the resulting attempt list, so semantic routing reuses
    // the full streaming / failover / telemetry machinery and only adds
    // the "which target + which route" decision. `semantic_route` carries
    // the matched route name (None on a fall-through to `default` or a
    // non-semantic request) for the `x-aisix-route` response header.
    let (attempt_models, semantic_route): (Vec<AttemptModel>, Option<String>) =
        if virtual_entry.value.is_semantic() {
            let prompt = last_user_message_text(req).unwrap_or_default();
            crate::semantic::resolve(state, &snapshot, &virtual_entry, &prompt, request_id)
                .await
                .map_err(&with_model)?
        } else {
            let attempts = resolve_attempt_models(
                &state.routing,
                &state.runtime_status,
                &snapshot,
                &req.model,
                &virtual_entry.id,
                &virtual_entry.value,
                RoutingRequest {
                    tags: &client.routing_tags,
                    stability_key: Some(
                        client
                            .routing_key
                            .as_deref()
                            .unwrap_or(auth.entry.id.as_str()),
                    ),
                    source_ip: &client.source_ip,
                },
            )
            .map_err(&with_model)?;
            (attempts, None)
        };

    // For non-routing requests, surface a misconfigured bridge as a
    // proper 503 rather than burying it inside a generic Bridge error.
    // Routing requests rely on the loop's `is_retryable` path so a
    // single bad provider doesn't take down the whole request.
    //
    // Skip an ensemble entry: it intentionally carries no
    // provider/provider_key_id (those live on its panel/judge members),
    // so `require_provider` would 400 here. Its members are resolved
    // and dispatched in `dispatch_ensemble`, branched below.
    if attempt_models.len() == 1 && !virtual_entry.value.is_ensemble() {
        let only = &attempt_models[0].model;
        let _provider = crate::dispatch::require_provider(only).map_err(with_model)?;
        // Pre-flight the PK-based two-tier dispatch so a missing
        // family/specialized bridge surfaces as 503 here, before
        // we commit to a long upstream call.
        let pk_entry =
            crate::dispatch::resolve_provider_key(&snapshot, only).map_err(with_model)?;
        if crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value).is_none() {
            return Err(with_model(ProxyError::ProviderUnavailable));
        }
    }

    // Multi-layer rate-limit reservation (api_key inline + model inline + policies).
    // `mut` so a routing dispatch can fold the winning target's model-layer
    // reservation into it once the winner is known (AISIX-Cloud#1087).
    let model_rl = crate::quota::ModelRateLimit::from_model(
        &req.model,
        &virtual_entry.id,
        &virtual_entry.value,
    );
    let mut reservation = crate::quota::enforce_rate_limit(state, auth, Some(&model_rl))
        .await
        .map_err(&with_model)?;

    let now = created_ts();

    // Ensemble path: fan the request out to the panel + judge instead of
    // dispatching a single upstream. Branches here — after the single
    // entry-level rate-limit reservation, before the streaming/failover
    // machinery — because an ensemble entry has no provider/provider_key_id
    // and no `routing.targets`, so the attempt loop below does not apply to
    // it. `dispatch_ensemble` owns the fan-out, judge synthesis, output
    // guardrail, token commit, and per-sub-call telemetry.
    if virtual_entry.value.is_ensemble() {
        return dispatch_ensemble(
            state,
            &snapshot,
            &virtual_entry,
            req,
            request_id,
            now,
            reservation,
            &resolved_chain,
            &applied_guardrails,
            bypass_reason,
            &model_id,
            &auth.entry.id,
            client,
            redactions_out.clone(),
            monitor_hits_out.clone(),
        )
        .await;
    }

    // Streaming path (#554): walk the target list with first-chunk
    // fallback. For each target we connect, then peek the first stream
    // chunk under the per-chunk read timeout (`stream_timeout`, falling
    // back to `timeout`). A first-chunk failure — connect error, an error
    // chunk, an empty stream, or a read timeout — fails over to the next
    // target before any bytes reach the client, exactly like the
    // non-streaming loop. The peeked chunk is re-prepended so the SSE pump
    // sees the full stream; the wrapper keeps enforcing the read timeout on
    // the remaining chunks, so a mid-stream stall terminates the response
    // (no fallback once the 200 is committed).
    if req.is_streaming() {
        // `routing.retries` — how many times to re-hit the SAME target (with
        // backoff) on a retryable failure before failing over to the next one.
        // Read here for the same reason the non-streaming loop below reads it:
        // the two loops are written separately, and streaming used to skip
        // `retries` entirely (AISIX-Cloud#1119), so a retryable first-chunk
        // failure fell straight over to the next target — or, with a single
        // target, failed the request outright.
        let retries = virtual_entry
            .value
            .routing
            .as_ref()
            .map(|r| r.retries_or_default())
            .unwrap_or(0);
        let retry_on_429 = virtual_entry
            .value
            .routing
            .as_ref()
            .map(|r| r.retry_on_429_or_default())
            .unwrap_or(false);
        let fallback_statuses: &[u16] = virtual_entry
            .value
            .routing
            .as_ref()
            .map(|r| r.fallback_on_statuses_or_default())
            .unwrap_or(&[]);
        let is_routing_request =
            virtual_entry.value.routing.is_some() || virtual_entry.value.is_semantic();
        let mut stream_routing = RoutingTelemetry::default();
        let mut last_err: Option<BridgeError> = None;

        struct StreamWin {
            model: aisix_core::Model,
            /// Snapshot id of the winning target — the emitted event's
            /// `model_id` (AISIX-Cloud#790). Equals the requested
            /// entry's id for direct (non-routing) requests.
            target_id: String,
            provider_lc: String,
            pk_id: String,
            upstream: aisix_gateway::ChatChunkStream,
            idx: u32,
            kind: &'static str,
        }
        let mut won: Option<StreamWin> = None;
        // The winning target's own model-layer reservation (routing dispatch
        // only) — folded into `reservation` after the loop so the stream hold
        // and post-stream token accounting cover the member's limits too
        // (AISIX-Cloud#1087).
        let mut won_member_reservation: Option<aisix_ratelimit::MultiReservation> = None;

        'targets: for attempt in &attempt_models {
            let model = &attempt.model;
            let Ok(provider) = crate::dispatch::require_provider(model) else {
                last_err = Some(BridgeError::Config("model has no provider".into()));
                continue 'targets;
            };
            let Ok(pk_entry) = crate::dispatch::resolve_provider_key(&snapshot, model) else {
                last_err = Some(BridgeError::Config(
                    "model references unknown provider_key_id".into(),
                ));
                continue 'targets;
            };
            let Some(bridge) = crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value) else {
                last_err = Some(BridgeError::Config(
                    "no bridge registered for provider_key".into(),
                ));
                continue 'targets;
            };
            let model_arc = Arc::new(model.clone());
            let pk_arc = Arc::new(pk_entry.value.clone());
            // Streaming deadline (#554): bound the connect by the effective
            // stream timeout; the read-timeout wrapper below enforces the
            // same budget on the first and subsequent chunks.
            let mut ctx = BridgeContext::new(request_id, model_arc, pk_arc);
            if let Some(d) = model.stream_timeout_effective() {
                ctx = ctx.with_deadline(d);
            }
            // Effective streaming budget: `stream_timeout`, falling back to
            // `timeout`. Used for the connect deadline (above) AND the
            // per-chunk read timeout + first-chunk peek below, so the budget
            // is applied consistently.
            let stream_budget = model.stream_timeout_effective();

            // Re-hit the SAME target `retries` times before failing over,
            // mirroring the non-streaming loop below (AISIX-Cloud#1119).
            for attempt_idx in 0..=retries {
                let (idx, kind) = stream_routing.begin_attempt(&model.display_name);
                let target_model = if is_routing_request {
                    model.display_name.clone()
                } else {
                    String::new()
                };
                // Reserve THIS target's own model rate-limit layers before
                // dispatching to it (AISIX-Cloud#1087). Over-limit → record a
                // 429 attempt and move on to the remaining targets in strategy
                // order (same-target retries can't help — the window won't
                // reset mid-loop).
                let member_reservation = match crate::quota::reserve_routing_target(
                    state,
                    is_routing_request,
                    &model.display_name,
                    &attempt.id,
                    model,
                )
                .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        stream_routing.attempts.push(AttemptRecord {
                            index: idx,
                            kind,
                            target_model,
                            target_model_id: attempt.id.clone(),
                            provider_key_id: pk_entry.id.clone(),
                            status: 429,
                            success: false,
                            error_class: "rate_limit_exceeded".to_string(),
                            error_message: e.to_string(),
                            latency_ms: 0,
                        });
                        // Keep the limiter's own Retry-After hint on the wire:
                        // when every target is exhausted this error becomes the
                        // client's 429, and SDKs back off on that header.
                        last_err = Some(BridgeError::upstream_status_with_retry_after(
                            429,
                            format!(
                                "routing target {:?} is over its model rate limit: {e}",
                                model.display_name
                            ),
                            crate::quota::retry_after_of(&e).map(Duration::from_secs),
                        ));
                        continue 'targets;
                    }
                };
                let attempt_started = Instant::now();
                // Connect, then — only when a streaming budget is configured —
                // peek the first chunk so a slow or erroring first token fails
                // over before the 200 is committed. Without a budget there is
                // nothing to gate on, so the stream is committed directly (a
                // first-chunk error then surfaces in-band, exactly like the
                // pre-#554 behavior). The read-timeout wrapper is a no-op when
                // the budget is None.
                let attempt_stream: Result<aisix_gateway::ChatChunkStream, BridgeError> =
                    match bridge.chat_stream(req, &ctx).await {
                        Err(e) => Err(e),
                        Ok(up) => {
                            let up = crate::stream_timeout::with_read_timeout(up, stream_budget);
                            if stream_budget.is_some() {
                                let mut up = up;
                                match up.next().await {
                                    // Re-prepend the peeked chunk so the SSE pump
                                    // sees the whole stream (and records TTFT on
                                    // the first content chunk); the wrapper keeps
                                    // enforcing the read timeout on the rest.
                                    Some(Ok(chunk)) => Ok(Box::pin(
                                        futures::stream::once(std::future::ready(Ok::<
                                            _,
                                            BridgeError,
                                        >(
                                            chunk
                                        )))
                                        .chain(up),
                                    )
                                        as aisix_gateway::ChatChunkStream),
                                    Some(Err(e)) => Err(e),
                                    None => Err(BridgeError::StreamAborted),
                                }
                            } else {
                                Ok(up)
                            }
                        }
                    };
                let latency_ms = attempt_started.elapsed().as_millis().min(u32::MAX as u128) as u32;
                match attempt_stream {
                    Ok(upstream) => {
                        state.health.record_success(&model.display_name);
                        state.runtime_status.mark_healthy(&attempt.id);
                        // Feed the least_latency EWMA. For streaming this is
                        // time-to-first-response (upstream stream established) —
                        // the routing-relevant latency signal.
                        state.runtime_status.record_latency(&attempt.id, latency_ms);
                        stream_routing.attempts.push(AttemptRecord {
                            index: idx,
                            kind,
                            target_model,
                            target_model_id: attempt.id.clone(),
                            provider_key_id: pk_entry.id.clone(),
                            status: 200,
                            success: true,
                            error_class: String::new(),
                            error_message: String::new(),
                            latency_ms,
                        });
                        won = Some(StreamWin {
                            model: model.clone(),
                            target_id: attempt.id.clone(),
                            provider_lc: provider.to_ascii_lowercase(),
                            pk_id: pk_entry.id.clone(),
                            upstream,
                            idx,
                            kind,
                        });
                        won_member_reservation = member_reservation;
                        break 'targets;
                    }
                    Err(err) => {
                        stream_routing.attempts.push(AttemptRecord {
                            index: idx,
                            kind,
                            target_model,
                            target_model_id: attempt.id.clone(),
                            provider_key_id: pk_entry.id.clone(),
                            status: err.http_status(),
                            success: false,
                            error_class: routing_error_class(&err).to_string(),
                            error_message: attempt_error_message(&err),
                            latency_ms,
                        });
                        let retryable = is_retryable(&err, retry_on_429, fallback_statuses);
                        tracing::warn!(
                            target_model = %model.display_name,
                            target_attempt = attempt_idx + 1,
                            error = %err,
                            retryable,
                            "streaming routing target attempt failed",
                        );
                        if retryable {
                            state.health.record_failure(&model.display_name);
                        }
                        if let Some((ttl, reason)) =
                            decide_cooldown(&err, attempt.model.cooldown.as_ref())
                        {
                            state.runtime_status.mark_cooldown(&attempt.id, ttl, reason);
                        }
                        last_err = Some(err);
                        if !retryable {
                            break 'targets;
                        }
                        if attempt_idx == retries {
                            break;
                        }
                        // Exponential backoff + jitter before re-hitting the
                        // SAME target (#788 P2); cross-target fail-over (the
                        // outer loop) stays immediate.
                        let backoff = crate::routing::retry_backoff((attempt_idx + 1) as u32);
                        tracing::debug!(
                            target_model = %model.display_name,
                            next_attempt = attempt_idx + 2,
                            backoff_ms = backoff.as_millis() as u64,
                            "backing off before same-target retry",
                        );
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }

        let Some(won) = won else {
            let err = last_err.unwrap_or_else(|| {
                BridgeError::Config("streaming routing exhausted with no targets".into())
            });
            return Err(with_model(ProxyError::Bridge(err)).with_routing(stream_routing));
        };
        let StreamWin {
            model,
            target_id: winner_target_id,
            provider_lc: provider,
            pk_id,
            upstream,
            idx: winner_idx,
            kind: winner_kind,
        } = won;
        let model = &model;
        // Hold concurrency for the stream's full lifetime instead of
        // releasing it at handler return. RPM was already counted by
        // pre_commit; TPM is updated retroactively on stream-end by
        // `add_tokens_post_stream`. The borrow-based reservation can't be
        // carried into the stream, so convert it into an owned guard that
        // releases the permit(s) on drop — i.e. when the stream completes
        // or is cancelled (the guard is moved into the on_complete closure,
        // which the CompleteOnDrop guard fires on both paths). Pre-fix the
        // permit was released here, letting a key capped at N run far more
        // than N simultaneous streams (#450).
        //
        // Fold the winning target's model-layer reservation in first, so the
        // stream hold keeps its concurrency slot(s) and `post_stream_keys`
        // bills its TPM/TPD at stream end too (AISIX-Cloud#1087).
        if let Some(member) = won_member_reservation.take() {
            reservation.merge(member);
        }
        let post_stream_keys = reservation.keys();
        let stream_concurrency_hold = reservation.into_stream_hold();
        // least_busy: keep this target counted as in-flight for the stream's
        // full lifetime. Like `stream_concurrency_hold`, the guard is moved
        // into the on_complete closure (dropped there), so the count stays
        // raised until the stream completes or is cancelled — the window a
        // concurrent routing decision must see this target as loaded.
        let in_flight_hold = state.runtime_status.begin_in_flight(&winner_target_id);
        // Capture everything the stream-completion callback needs so
        // it can fire `emit_usage_event` once the terminal SSE chunk
        // has yielded its `usage` block. Telemetry emission has to
        // wait until end-of-stream because OpenAI / Anthropic only
        // populate `usage` on the last chunk; emitting at handler
        // return (the non-streaming path's spot) would record zeros.
        let limiter = Arc::clone(&state.limiter);
        let state_for_telem = state.clone();
        let metrics_for_stream = state.metrics.clone();
        let request_id_for_telem = request_id.to_string();
        // AISIX-Cloud#790: the per-attempt event carries the winning
        // TARGET's id, not the group's — pricing resolves against the
        // target. (Equal for direct models.)
        let model_id_for_telem = winner_target_id;
        let api_key_id_for_telem = auth.entry.id.clone();
        let team_id_for_metrics = auth.key().team_id.clone();
        let user_id_for_metrics = auth.key().user_id.clone();
        let provider_for_metrics = provider.to_ascii_lowercase();
        let model_for_metrics = req.model.clone();
        let provider_key_id_for_metrics = pk_id.clone();
        // #890 req-3/req-4: readable provider-key name + normalised inbound
        // client type, captured for the streaming on_complete metric emission
        // (mirrors the non-streaming `record_success` path).
        let provider_key_name_for_metrics = {
            let snap = state.snapshot.load();
            crate::usage_attr::provider_key_metric_name(&snap, &pk_id)
        };
        let user_name_for_metrics = auth.key().user_name.clone();
        let client_type_for_metrics = state
            .client_classifier
            .classify(&client.user_agent)
            .to_string();
        // Captured for the stream-end telemetry closure so
        // emit_usage_event can look up `telemetry_tags` for per-PK
        // attribution (#302 M17 / AISIX-Cloud#436). The metrics
        // variant above is `&str`-scoped to inner scopes that consume
        // it as a borrow; the telem variant is owned for the move
        // into the on_complete closure.
        let provider_key_id_for_telem = pk_id.clone();
        let upstream_model_for_metrics = model.upstream_model().unwrap_or("unknown").to_string();
        let bypass_reason_for_telem = bypass_reason.clone().unwrap_or_default();
        // Applied guardrail set (#379), owned for the move into on_complete so
        // the streamed-response telemetry event records which guardrails ran.
        let applied_guardrails_for_telem = applied_guardrails.clone();
        // Input-side PII mask counts (#932), captured before the stream so
        // the end-of-stream event merges them with the output-side counts
        // accumulated in `comp.redacted_entity_counts`.
        let input_redactions_for_telem = redactions_out.clone();
        // Input-side monitor hits (AISIX-Cloud#562), merged with the
        // output-side hits accumulated in `comp.monitor_hits`.
        let input_monitor_hits_for_telem = monitor_hits_out.clone();
        // Downstream client attribution (#492) moved into the on_complete
        // closure so streamed responses log the same IP/UA as non-streaming.
        let client_for_telem = client.clone();
        // Per-attempt target name for the on_complete winner event. The
        // real per-attempt RoutingTelemetry was accumulated by the
        // fallback loop above (`stream_routing`).
        let attempt_model_for_telem = if is_routing_request {
            model.display_name.clone()
        } else {
            String::new()
        };
        // Per #204: pass the resolved guardrail chain so the streaming
        // path can run output guardrails at end-of-stream
        // (buffer-then-check). Mirrors the non-streaming
        // `resolved_chain.check_output(...)` call site below.
        //
        // Fast-path: skip the context entirely when the resolved chain
        // for this request is empty (no attachment matched). When `None`,
        // `build_sse_stream` skips per-chunk accumulation — both noise on
        // the hot path for the dominant guardrail-free deployment.
        let stream_guardrail = if resolved_chain.is_empty() {
            None
        } else {
            Some(StreamGuardrailContext {
                chain: Arc::clone(&resolved_chain),
                model_name: req.model.clone(),
            })
        };
        // Capture the prompt for content-capturing exporters; the response is
        // assembled inside the stream into `comp.response_text`. Both gated on
        // `content_cap` (None on the common content-free path).
        let captured_prompt_for_telem =
            content_cap.map(|_| serde_json::to_string(req).unwrap_or_default());
        // #790: the bridge injects `stream_options.include_usage` on the
        // upstream leg whenever the client didn't set stream_options — the
        // terminal usage-only chunk that produces is for the gateway's own
        // telemetry. It reaches the client only when the client itself
        // asked for usage.
        let client_requested_usage = req
            .extra
            .get("stream_options")
            .and_then(|so| so.get("include_usage"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Token-estimation fallback context (AISIX-Cloud#1074): the
        // request is cloned here because the stream owns it until an
        // end-of-stream Drop — where the borrow is long gone. Tokenized
        // only if the upstream never reports usage.
        let estimator = crate::token_estimate::Estimator::new(
            &upstream_model_for_metrics,
            crate::token_estimate::PromptInput::Chat(Box::new(req.clone())),
        );
        let sse_stream = build_sse_stream(
            upstream,
            now,
            stream_guardrail,
            started,
            req.model.clone(),
            content_cap,
            client_requested_usage,
            // Single upstream: nothing pre-incurred, so no usage to fold in.
            aisix_gateway::chat::UsageStats::default(),
            Some(estimator),
            move |comp: StreamCompletion| {
                // Rate-limit accounting (TPM cap) for all layers.
                for key in &post_stream_keys {
                    limiter.add_tokens_post_stream(key, comp.total_tokens);
                }
                // Telemetry: emit with the actual upstream-reported counts.
                // cost_usd stays 0.0; cp-api recomputes server-side from
                // its model_pricing catalog (same pattern as the non-
                // streaming path's cost_usd handling).
                emit_usage_event(
                    &state_for_telem,
                    &request_id_for_telem,
                    &model_id_for_telem,
                    &model_for_metrics,
                    &api_key_id_for_telem,
                    /* status_code */ 200,
                    started.elapsed(),
                    comp.prompt_tokens,
                    comp.completion_tokens,
                    UsageExtras {
                        cached_prompt_tokens: comp.cached_prompt_tokens,
                        reasoning_tokens: comp.reasoning_tokens,
                        cache_creation_tokens: comp.cache_creation_tokens,
                        cache_read_tokens: comp.cache_read_tokens,
                        usage_estimated: comp.usage_estimated,
                        provider_request_id: comp.provider_request_id,
                        provider_model_version: comp.provider_model_version,
                        finish_reason: comp.finish_reason,
                        // Per #204 audit H1: merge input-side bypass
                        // (`bypass_reason_for_telem` captured before
                        // the upstream stream started) with the
                        // output-side bypass observed at end-of-stream
                        // (`comp.bypass_reason`). First-bypass-wins,
                        // matching the non-streaming convention.
                        bypass_reason: if !bypass_reason_for_telem.is_empty() {
                            bypass_reason_for_telem
                        } else {
                            comp.bypass_reason
                        },
                        // TODO(streaming-cache): when streaming responses
                        // become cacheable, this constant `Disabled` will
                        // silently mis-bucket cached streamed responses.
                        // Propagate the dispatch path's `cache_status`
                        // local at that point. Tracking issue to be filed
                        // alongside the streaming-cache implementation.
                        cache_status: CacheStatus::Disabled.as_str().to_string(),
                        cache_hit_saved_input_tokens: 0,
                        cache_hit_saved_output_tokens: 0,
                        ttft_ms: comp.ttft_ms,
                        // #554: the winning attempt may be a fallback target,
                        // not the initial one — record the real index/kind.
                        attempt_index: winner_idx,
                        attempt_kind: winner_kind.to_string(),
                        attempt_model: attempt_model_for_telem.clone(),
                        error_class: String::new(),
                        error_message: String::new(),
                        applied_guardrails: applied_guardrails_for_telem.clone(),
                        provider_key_id: provider_key_id_for_telem.clone(),
                        redacted_entity_counts: {
                            let mut merged = input_redactions_for_telem.clone();
                            crate::redact::merge_counts(&mut merged, comp.redacted_entity_counts);
                            merged
                        },
                        guardrail_monitor_hits: {
                            let mut merged = input_monitor_hits_for_telem.clone();
                            merged.extend(comp.monitor_hits);
                            merged
                        },
                    },
                    /* cost_usd */ 0.0,
                    comp.guardrail_blocked,
                    &client_for_telem,
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
                metrics_for_stream.record_llm_usage(
                    UsageLabels {
                        endpoint: "/v1/chat/completions",
                        inbound_protocol: "openai",
                        provider: &provider_for_metrics,
                        model: &model_for_metrics,
                        upstream_model: &upstream_model_for_metrics,
                        provider_key_id: &provider_key_id_for_metrics,
                        provider_key_name: &provider_key_name_for_metrics,
                        api_key_id: &api_key_id_for_telem,
                        team_id: team_id_for_metrics.as_deref().unwrap_or("unknown"),
                        user_id: user_id_for_metrics.as_deref().unwrap_or("unknown"),
                        user_name: user_name_for_metrics.as_deref().unwrap_or("unknown"),
                    },
                    LlmUsage {
                        input_tokens: comp.prompt_tokens,
                        output_tokens: comp.completion_tokens,
                        total_tokens: comp.total_tokens.min(u64::from(u32::MAX)) as u32,
                        spend_usd: 0.0,
                    },
                );
                // #890 req-4: streaming token volume by inbound client type.
                // #1002: comp.total_tokens is the cache-inclusive total (an
                // Anthropic upstream bridged to an OpenAI-shape client folds
                // cache tokens into total_tokens per #679).
                // AISIX-Cloud#1044: same requested logical model as the
                // UsageLabels above.
                metrics_for_stream.record_llm_tokens_by_client(
                    &client_type_for_metrics,
                    &model_for_metrics,
                    u64::from(comp.prompt_tokens),
                    u64::from(comp.completion_tokens),
                    comp.total_tokens,
                );
                metrics_for_stream.record_request_e2e_latency(
                    LatencyLabels {
                        endpoint: "/v1/chat/completions",
                        model: &model_for_metrics,
                        provider: &provider_for_metrics,
                        status: 200,
                        streaming: true,
                    },
                    started.elapsed(),
                );
                metrics_for_stream.record_request_ttft(
                    LatencyLabels {
                        endpoint: "/v1/chat/completions",
                        model: &model_for_metrics,
                        provider: &provider_for_metrics,
                        status: 200,
                        streaming: true,
                    },
                    Duration::from_millis(u64::from(comp.ttft_ms)),
                );
                metrics_for_stream.record_time_to_first_token(
                    UsageLabels {
                        endpoint: "/v1/chat/completions",
                        inbound_protocol: "openai",
                        provider: &provider_for_metrics,
                        model: &model_for_metrics,
                        upstream_model: &upstream_model_for_metrics,
                        provider_key_id: &provider_key_id_for_metrics,
                        provider_key_name: &provider_key_name_for_metrics,
                        api_key_id: &api_key_id_for_telem,
                        team_id: team_id_for_metrics.as_deref().unwrap_or("unknown"),
                        user_id: user_id_for_metrics.as_deref().unwrap_or("unknown"),
                        user_name: user_name_for_metrics.as_deref().unwrap_or("unknown"),
                    },
                    Duration::from_millis(u64::from(comp.ttft_ms)),
                );
                // Release the concurrency permit(s) now that the stream has
                // completed (or was cancelled). on_complete is fired by the
                // CompleteOnDrop guard on both paths, so the permit is held
                // for the stream's full lifetime and never leaked (#450).
                drop(stream_concurrency_hold);
                // Same lifetime for the least_busy in-flight count.
                drop(in_flight_hold);
            },
        );
        let response =
            Sse::new(sse_stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
        return Ok(Success {
            response: response.into_response(),
            provider: provider.to_ascii_lowercase(),
            model_id: model_id.clone(),
            // Token totals are populated on the SSE stream's terminal
            // chunk and forwarded into telemetry from on_complete; the
            // top-level handler skips its own `emit_usage_event` for
            // streaming via `telemetry_handled_by_stream` below.
            prompt_tokens: None,
            usage_estimated: false,
            completion_tokens: None,
            total_tokens: None,
            cost_usd: 0.0,
            cached_prompt_tokens: 0,
            reasoning_tokens: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
            provider_request_id: String::new(),
            provider_model_version: String::new(),
            provider_key_id: pk_id.clone(),
            upstream_model: model.upstream_model().unwrap_or("unknown").to_string(),
            finish_reason: String::new(),
            bypass_reason: bypass_reason.clone(),
            // Streaming responses aren't cached at this layer — see
            // crates/aisix-cache/src/lib.rs. Always surface as
            // `disabled` on the streaming path.
            cache_status: CacheStatus::Disabled,
            cache_hit_saved_input_tokens: 0,
            cache_hit_saved_output_tokens: 0,
            telemetry_handled_by_stream: true,
            // #554: the served target is whichever one won the first-chunk
            // race (may be a fallback). The helper enforces the
            // "direct model → None" rule consistently with the
            // non-streaming and cache paths.
            served_by_target: served_by_target_for_routing(
                is_routing_request,
                Some(model.display_name.clone()),
            ),
            served_by_route: semantic_route.clone(),
            routing: stream_routing,
            // Streaming content capture lands in C3b.
            captured_content: None,
        });
    }

    // Policy gate (Stage 3): the cache is only consulted when at
    // least one enabled `CachePolicy` in the snapshot has an
    // `applies_to` clause that matches THIS request. cp-api owns
    // the policy CRUD surface (`/api/environments/:env/cache_policies`,
    // see Stage 1); kine fans out the rows; the loader populates
    // `snapshot.cache_policies` (see aisix-etcd). Stage 4 will add
    // per-policy `ttl_seconds` propagation into the cache backend.
    //
    // Match order: first enabled policy whose `parsed_applies_to()`
    // accepts (req.model, auth.entry.id) wins. We grab the WHOLE
    // matching entry (not just `any`) so the backend selection and the
    // post-call write below can use that policy's `backend` and
    // `ttl_seconds`. When multiple policies match the same request,
    // the entry-table iteration order decides — that's an
    // unspecified-but-stable tiebreak we'll formalise (probably
    // "narrowest scope wins") in a follow-up if operators ever care.
    let matched_policy = snapshot
        .cache_policies
        .entries()
        .iter()
        .find(|entry| {
            entry.value.enabled
                && entry
                    .value
                    .parsed_applies_to()
                    .matches(&req.model, &auth.entry.id)
        })
        .cloned();

    // #519 B.8: the matched policy's `backend` selects the cache
    // instance. `memory` always resolves; `redis` resolves only when
    // the deployment configured `cache.redis` — otherwise caching is
    // INACTIVE for this request (`cache_status = disabled`, warn once
    // per policy inside `for_policy_backend`). Never fall back to
    // node-local memory: the operator asked for a shared cache, and a
    // silent memory stand-in would serve per-node answers while the
    // dashboard claims redis semantics.
    let policy_cache: Option<Arc<dyn Cache>> = match (matched_policy.as_ref(), state.cache.as_ref())
    {
        (Some(entry), Some(backends)) => backends
            .for_policy_backend(entry.value.backend, &entry.id, &entry.value.name)
            .cloned(),
        _ => None,
    };
    let matched_policy_ttl = policy_cache
        .as_ref()
        .and(matched_policy.as_ref())
        .map(|entry| Duration::from_secs(u64::from(entry.value.ttl_seconds)));

    // Cache lookup keyed on the *virtual* model name so a re-request
    // hits the cache regardless of which target served the original.
    let cache_key = policy_cache
        .as_ref()
        .map(|_| CacheKey::from_request(req).fingerprint());

    let cache_status = if policy_cache.is_some() {
        CacheStatus::Miss
    } else {
        CacheStatus::Disabled
    };

    if let (Some(cache), Some(key)) = (policy_cache.as_ref(), cache_key.as_ref()) {
        match cache.get(key).await {
            Ok(Some(mut cached)) => {
                reservation.commit_tokens(0).await;
                // #448: a cache hit is client-visible output just like a
                // fresh upstream response, so it must run output guardrails
                // before being returned — not bypass them.
                let (cached_verdict, hits) = resolved_chain
                    .check_output_non_segment_observed(&cached)
                    .await;
                monitor_hits_out.extend(hits);
                let cached_verdict = crate::redact::moderate_body(
                    resolved_chain.as_ref(),
                    crate::redact::Direction::Output,
                    cached_verdict,
                    redactions_out,
                    monitor_hits_out,
                    |g| crate::redact::redact_chat_response(g, &mut cached),
                )
                .await;
                match cached_verdict {
                    GuardrailVerdict::Block {
                        reason,
                        guardrail_name,
                    } => {
                        tracing::warn!(
                            guardrail_hook = "output",
                            model = %req.model,
                            reason = %reason,
                            "guardrail blocked cached response",
                        );
                        return Err(with_model(ProxyError::ContentFiltered(
                            crate::error::guardrail_block_message(
                                "response",
                                guardrail_name.as_deref(),
                            ),
                        )));
                    }
                    GuardrailVerdict::Bypass { reason } => {
                        if bypass_reason.is_none() {
                            bypass_reason = Some(reason);
                        }
                    }
                    _ => {}
                }
                // #932: cache hits are client-visible output like a fresh
                // upstream response — mask them too. Entries are stored
                // masked (see the write below), but entries written before
                // a rule was added/tightened must still be caught here.
                crate::redact::merge_counts(
                    redactions_out,
                    crate::redact::redact_chat_response(resolved_chain.as_ref(), &mut cached),
                );
                let prompt = cached.usage.prompt_tokens as u64;
                let completion = cached.usage.completion_tokens as u64;
                let total = cached.usage.total_tokens as u64;
                // Snapshot the cache / reasoning counters BEFORE the
                // value moves into render_response below. Cache hits
                // replay the original upstream's usage — we already
                // paid the cost first time around — so the dashboard's
                // "of which N were cache hits" stat reflects the
                // original event accurately.
                let cached_prompt_tokens = cached.usage.cached_prompt_tokens;
                let reasoning_tokens = cached.usage.reasoning_tokens;
                let cache_creation_tokens = cached.usage.cache_creation_tokens;
                let cache_read_tokens = cached.usage.cache_read_tokens;
                // The provider label points at the first attempt — for a
                // cache hit we don't know (or care) which target ran the
                // original call; the fingerprint identified the answer.
                let provider_label = attempt_models[0]
                    .model
                    .provider
                    .as_deref()
                    .map(|p| p.to_ascii_lowercase())
                    .unwrap_or_else(|| "unknown".into());
                let provider_key_id = attempt_models[0]
                    .model
                    .provider_key_id
                    .clone()
                    .unwrap_or_else(|| "unknown".into());
                let upstream_model = attempt_models[0]
                    .model
                    .upstream_model()
                    .unwrap_or("unknown")
                    .to_string();
                // Token-estimation fallback (AISIX-Cloud#1074): a stored
                // response whose original upstream never reported usage
                // replays zeros — fill them like the fresh-response path
                // so hit rows (and their saved-token stats) don't record
                // silent zeros.
                let (prompt, completion, total, usage_estimated) = if prompt == 0 || completion == 0
                {
                    let est = crate::token_estimate::Estimator::new(
                        &upstream_model,
                        crate::token_estimate::PromptInput::Chat(Box::new(req.clone())),
                    );
                    let filled = crate::token_estimate::fill_missing(
                        &est,
                        prompt as u32,
                        completion as u32,
                        Some(&estimation_output_text(&cached)),
                    );
                    if filled.estimated {
                        let total = crate::usage_attr::total_tokens_with_cache(
                            filled.prompt_tokens,
                            filled.completion_tokens,
                            cache_creation_tokens,
                            cache_read_tokens,
                        );
                        (
                            filled.prompt_tokens as u64,
                            filled.completion_tokens as u64,
                            total,
                            true,
                        )
                    } else {
                        (prompt, completion, total, false)
                    }
                } else {
                    (prompt, completion, total, false)
                };
                // Capture the prompt + cached response for content-capturing
                // exporters (gated). A cache hit still served content to the
                // caller, so it's logged like a fresh response.
                let captured_content = content_cap.map(|cap| {
                    CapturedContent::new(
                        &serde_json::to_string(req).unwrap_or_default(),
                        cached.message.content.as_deref().unwrap_or(""),
                        cap as usize,
                    )
                });
                let mut response = Json(render_response(now, cached, &req.model)).into_response();
                response
                    .headers_mut()
                    .insert(CACHE_HEADER, HeaderValue::from_static("hit"));
                return Ok(Success {
                    response,
                    provider: provider_label,
                    model_id: model_id.clone(),
                    prompt_tokens: Some(prompt),
                    completion_tokens: Some(completion),
                    total_tokens: Some(total),
                    usage_estimated,
                    cached_prompt_tokens,
                    reasoning_tokens,
                    cache_creation_tokens,
                    cache_read_tokens,
                    // The cache stored the original provider response;
                    // a stable id here would mislead reconciliation
                    // (the request didn't actually hit the upstream),
                    // so we leave these blank deliberately.
                    provider_request_id: String::new(),
                    provider_model_version: String::new(),
                    provider_key_id,
                    upstream_model,
                    finish_reason: String::new(),
                    // Cache hits don't burn cost on our side (we already
                    // paid the upstream price the first time around).
                    cost_usd: 0.0,
                    bypass_reason: bypass_reason.clone(),
                    cache_status: CacheStatus::Hit,
                    // The whole point of #88: cache hit replays the
                    // upstream's prompt + completion tokens. Surfacing
                    // them as a dedicated counter (rather than relying
                    // on the existing prompt_tokens column + cache_status
                    // filter) lets cp-api compute `cost_saved_usd`
                    // without joining on the status enum.
                    cache_hit_saved_input_tokens: prompt.try_into().unwrap_or(u32::MAX),
                    cache_hit_saved_output_tokens: completion.try_into().unwrap_or(u32::MAX),
                    telemetry_handled_by_stream: false,
                    // Cache hits don't carry routing-target identity:
                    // the stored response is decoupled from whichever
                    // target produced it on the original miss. See the
                    // `served_by_target` field docs on `Success`.
                    served_by_target: None,
                    served_by_route: None,
                    routing: RoutingTelemetry::default(),
                    captured_content,
                });
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(error = %err, key = %key, "cache lookup failed");
            }
        }
    }

    // Walk the target list. Retry the current target first, then fail over
    // to later targets only after retries are exhausted. Non-retryable
    // (non-429 4xx) errors stop immediately.
    let mut last_err: Option<BridgeError> = None;
    let mut chosen_provider: Option<String> = None;
    let mut chosen_provider_key_id: Option<String> = None;
    let mut chosen_upstream_model: Option<String> = None;
    // Display name of the target whose attempt finally succeeded. Used
    // to populate the `x-aisix-served-by` response header for routing
    // requests (AISIX-Cloud#410). Stays `None` until an attempt wins.
    let mut chosen_target_display_name: Option<String> = None;
    let mut upstream: Option<aisix_gateway::ChatResponse> = None;
    let retries = virtual_entry
        .value
        .routing
        .as_ref()
        .map(|routing| routing.retries_or_default())
        .unwrap_or(0);
    let retry_on_429 = virtual_entry
        .value
        .routing
        .as_ref()
        .map(|routing| routing.retry_on_429_or_default())
        .unwrap_or(false);
    let fallback_statuses: &[u16] = virtual_entry
        .value
        .routing
        .as_ref()
        .map(|routing| routing.fallback_on_statuses_or_default())
        .unwrap_or(&[]);
    let is_routing_request =
        virtual_entry.value.routing.is_some() || virtual_entry.value.is_semantic();
    let mut routing = RoutingTelemetry::default();
    // The winning target's own model-layer reservation (routing dispatch
    // only) — folded into `reservation` at the commit point below so the
    // member's TPM/TPD bills with the request-level layers
    // (AISIX-Cloud#1087).
    let mut won_member_reservation: Option<aisix_ratelimit::MultiReservation> = None;

    'targets: for attempt in &attempt_models {
        let model = &attempt.model;
        let Some(provider) = model.provider.as_deref() else {
            last_err = Some(BridgeError::Config("model has no provider".into()));
            continue;
        };
        let pk_entry = match crate::dispatch::resolve_provider_key(&snapshot, model) {
            Ok(pk) => pk,
            Err(_) => {
                last_err = Some(BridgeError::Config(
                    "model references unknown provider_key_id".into(),
                ));
                continue;
            }
        };
        // Two-tier dispatch via `Hub::dispatch_two_tier`: specialized
        // vendor (ProviderKey.provider) first, then adapter family
        // (ProviderKey.adapter). The legacy `Provider`-keyed registry
        // is gone after #302 Phase A; a PK that matches neither tier is
        // a misconfiguration and surfaces as 503.
        let Some(bridge) = crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value) else {
            last_err = Some(BridgeError::Config(format!(
                "no bridge registered for provider_key provider={:?} adapter={:?}",
                pk_entry.value.provider, pk_entry.value.adapter
            )));
            continue;
        };
        let model_arc = Arc::new(model.clone());
        let pk_arc = Arc::new(pk_entry.value.clone());
        // Per-attempt non-streaming deadline (#554): an elapsed `timeout`
        // surfaces as a retryable `BridgeError::Timeout`, so a slow target
        // fails over to the next one via the loop below.
        let mut ctx = BridgeContext::new(request_id, model_arc, pk_arc);
        if let Some(d) = model.request_timeout() {
            ctx = ctx.with_deadline(d);
        }

        for attempt_idx in 0..=retries {
            // Per-attempt telemetry kind (#655): the first attempt overall
            // is "initial"; a different target than the previous attempt is
            // a "fallback"; the same target again is a "retry".
            let (attempt_index, kind) = routing.begin_attempt(&model.display_name);
            // Routing target name only for routing groups; a direct model
            // leaves it empty since `model_id` already identifies it.
            let target_model = if is_routing_request {
                model.display_name.clone()
            } else {
                String::new()
            };

            // Reserve THIS target's own model rate-limit layers before
            // dispatching to it (AISIX-Cloud#1087). Over-limit → record a
            // 429 attempt and move on to the remaining targets in strategy
            // order (same-target retries can't help — the window won't
            // reset mid-loop).
            let member_reservation = match crate::quota::reserve_routing_target(
                state,
                is_routing_request,
                &model.display_name,
                &attempt.id,
                model,
            )
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    routing.attempts.push(AttemptRecord {
                        index: attempt_index,
                        kind,
                        target_model,
                        target_model_id: attempt.id.clone(),
                        provider_key_id: pk_entry.id.clone(),
                        status: 429,
                        success: false,
                        error_class: "rate_limit_exceeded".to_string(),
                        error_message: e.to_string(),
                        latency_ms: 0,
                    });
                    // Keep the limiter's own Retry-After hint on the wire:
                    // when every target is exhausted this error becomes the
                    // client's 429, and SDKs back off on that header.
                    last_err = Some(BridgeError::upstream_status_with_retry_after(
                        429,
                        format!(
                            "routing target {:?} is over its model rate limit: {e}",
                            model.display_name
                        ),
                        crate::quota::retry_after_of(&e).map(Duration::from_secs),
                    ));
                    continue 'targets;
                }
            };

            let attempt_started = Instant::now();
            // least_busy: count this target as in-flight for the upstream
            // call. The response is fully buffered, so the target is done
            // once `bridge.chat` returns; the guard drops at the end of this
            // attempt's scope on both the success-break and failure paths.
            let _in_flight = state.runtime_status.begin_in_flight(&attempt.id);
            let result = bridge.chat(req, &ctx).await;
            let attempt_latency_ms =
                attempt_started.elapsed().as_millis().min(u32::MAX as u128) as u32;
            match result {
                Ok(resp) => {
                    state.health.record_success(&model.display_name);
                    state.runtime_status.mark_healthy(&attempt.id);
                    // Feed the least_latency EWMA with this attempt's
                    // round-trip latency.
                    state
                        .runtime_status
                        .record_latency(&attempt.id, attempt_latency_ms);
                    chosen_provider = Some(provider.to_ascii_lowercase());
                    chosen_provider_key_id = Some(pk_entry.id.clone());
                    chosen_upstream_model =
                        Some(model.upstream_model().unwrap_or("unknown").to_string());
                    chosen_target_display_name = Some(model.display_name.clone());
                    routing.attempts.push(AttemptRecord {
                        index: attempt_index,
                        kind,
                        target_model,
                        target_model_id: attempt.id.clone(),
                        provider_key_id: pk_entry.id.clone(),
                        status: 200,
                        success: true,
                        error_class: String::new(),
                        error_message: String::new(),
                        latency_ms: attempt_latency_ms,
                    });
                    won_member_reservation = member_reservation;
                    upstream = Some(resp);
                    break;
                }
                Err(err) => {
                    routing.attempts.push(AttemptRecord {
                        index: attempt_index,
                        kind,
                        target_model,
                        target_model_id: attempt.id.clone(),
                        provider_key_id: pk_entry.id.clone(),
                        status: err.http_status(),
                        success: false,
                        error_class: routing_error_class(&err).to_string(),
                        error_message: attempt_error_message(&err),
                        latency_ms: attempt_latency_ms,
                    });
                    let retryable = is_retryable(&err, retry_on_429, fallback_statuses);
                    tracing::warn!(
                        target_model = %model.display_name,
                        target_attempt = attempt_idx + 1,
                        error = %err,
                        retryable,
                        "routing target attempt failed",
                    );
                    if retryable {
                        state.health.record_failure(&model.display_name);
                    }
                    // Cooldown decision is independent of retry — a
                    // non-retryable 401 still cools down because the
                    // same key will keep failing for upcoming
                    // requests; a retryable 502 also cools down so
                    // the next request prefers a different target.
                    if let Some((ttl, reason)) =
                        decide_cooldown(&err, attempt.model.cooldown.as_ref())
                    {
                        state.runtime_status.mark_cooldown(&attempt.id, ttl, reason);
                    }
                    last_err = Some(err);
                    if !retryable {
                        break;
                    }
                    if attempt_idx == retries {
                        break;
                    }
                    // #788 P2: exponential backoff + jitter before retrying
                    // the SAME target, so a transiently-failing upstream gets
                    // a pause instead of being hammered. Cross-target fallover
                    // (the outer loop) stays immediate.
                    let backoff = crate::routing::retry_backoff((attempt_idx + 1) as u32);
                    tracing::debug!(
                        target_model = %model.display_name,
                        next_attempt = attempt_idx + 2,
                        backoff_ms = backoff.as_millis() as u64,
                        "backing off before same-target retry",
                    );
                    tokio::time::sleep(backoff).await;
                }
            }
        }
        if upstream.is_some() {
            break;
        }
        if let Some(err) = last_err.as_ref() {
            if !is_retryable(err, retry_on_429, fallback_statuses) {
                break;
            }
        }
    }

    let Some(mut upstream) = upstream else {
        // Bubble the most recent BridgeError through ProxyError::Bridge.
        let err = last_err.unwrap_or_else(|| {
            BridgeError::Config("routing exhausted with no targets attempted".into())
        });
        return Err(with_model(ProxyError::Bridge(err)).with_routing(routing));
    };
    let provider_name = chosen_provider.unwrap_or_else(|| "unknown".into());
    let provider_key_id = chosen_provider_key_id.unwrap_or_else(|| "unknown".into());
    let upstream_model = chosen_upstream_model.unwrap_or_else(|| "unknown".into());

    // Output guardrail. Tokens still count against quota — the upstream
    // already burned them — so commit before the check, and refuse the
    // refusal-write to the cache so a re-request gets a fresh chance.
    //
    // Token-estimation fallback (AISIX-Cloud#1074): when the upstream
    // response carries no usage block, fill the missing counters locally
    // BEFORE the quota commit and telemetry below so neither records
    // silent zeros. Local variables only — `render_response` serialises
    // the upstream body untouched, so the client never sees synthesised
    // usage presented as the provider's.
    let (prompt_tokens_u32, completion_tokens_u32, usage_estimated) = {
        let (p, c) = (
            upstream.usage.prompt_tokens,
            upstream.usage.completion_tokens,
        );
        if p == 0 || c == 0 {
            let est = crate::token_estimate::Estimator::new(
                &upstream_model,
                crate::token_estimate::PromptInput::Chat(Box::new(req.clone())),
            );
            let filled = crate::token_estimate::fill_missing(
                &est,
                p,
                c,
                Some(&estimation_output_text(&upstream)),
            );
            (
                filled.prompt_tokens,
                filled.completion_tokens,
                filled.estimated,
            )
        } else {
            (p, c, false)
        }
    };
    let prompt = prompt_tokens_u32 as u64;
    let completion = completion_tokens_u32 as u64;
    let total = if usage_estimated {
        crate::usage_attr::total_tokens_with_cache(
            prompt_tokens_u32,
            completion_tokens_u32,
            upstream.usage.cache_creation_tokens,
            upstream.usage.cache_read_tokens,
        )
    } else {
        upstream.usage.total_tokens as u64
    };
    // Snapshot the cache / reasoning counters + provider identity before
    // the upstream gets moved into render_response below — we need them
    // on the Success struct for telemetry.
    let cached_prompt_tokens = upstream.usage.cached_prompt_tokens;
    let reasoning_tokens = upstream.usage.reasoning_tokens;
    let cache_creation_tokens = upstream.usage.cache_creation_tokens;
    let cache_read_tokens = upstream.usage.cache_read_tokens;
    let provider_request_id = upstream.id.clone();
    let provider_model_version = upstream.model.clone();
    let finish_reason = finish_reason_label(&upstream.finish_reason);
    // Fold the winning target's model-layer reservation in so one commit
    // bills the member's TPM/TPD alongside the request-level layers
    // (AISIX-Cloud#1087).
    if let Some(member) = won_member_reservation.take() {
        reservation.merge(member);
    }
    reservation.commit_tokens(total).await;

    // cp-api recomputes cost server-side from its pricing catalog when
    // ingesting telemetry; the DP just records 0.0 on the wire.
    let cost_usd = 0.0;

    let (output_verdict, hits) = resolved_chain
        .check_output_non_segment_observed(&upstream)
        .await;
    monitor_hits_out.extend(hits);
    let output_verdict = crate::redact::moderate_body(
        resolved_chain.as_ref(),
        crate::redact::Direction::Output,
        output_verdict,
        redactions_out,
        monitor_hits_out,
        |g| crate::redact::redact_chat_response(g, &mut upstream),
    )
    .await;
    match output_verdict {
        GuardrailVerdict::Allow => {}
        GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } => {
            // Output filter fires AFTER the upstream call, so the
            // provider has already billed for these tokens. Surface
            // the captured upstream `UsageStats` to the handler so
            // `usage_events` reflects the bill — silently zeroing
            // them would let the customer's dashboard underreport
            // tokens they paid the provider for.
            //
            // `bypass_reason` and `cache_status` are also forwarded
            // so an output-blocked request stays comparable to a
            // succeeded-but-billed one in the dashboard's bypass /
            // cache-status filters. Without these, an operator
            // auditing input-bypass would never see them on
            // output-blocked requests.
            let charge = UpstreamCharge {
                provider_key_id: provider_key_id.clone(),
                prompt_tokens: prompt_tokens_u32,
                completion_tokens: completion_tokens_u32,
                usage_estimated,
                cached_prompt_tokens,
                reasoning_tokens,
                cache_creation_tokens,
                cache_read_tokens,
                provider_request_id: provider_request_id.clone(),
                provider_model_version: provider_model_version.clone(),
                finish_reason: finish_reason.clone(),
                bypass_reason: bypass_reason.clone().unwrap_or_default(),
                cache_status,
                routing: routing.clone(),
            };
            // Per #153, the verdict's `reason` carries the matched-
            // pattern detail (the actual forbidden text from the
            // model's response). Echoing that back to the caller is
            // a real bypass of the output guardrail's purpose:
            // anyone who can trigger the rule can extract the model's
            // forbidden output via the error envelope. Redact on
            // the wire — naming only the guardrail that fired (#519
            // B.4b) — and keep the rich detail in tracing for ops.
            tracing::warn!(
                guardrail_hook = "output",
                model = %req.model,
                reason = %reason,
                "guardrail blocked response"
            );
            return Err(DispatchFailure::new(
                Some(model_id.clone()),
                Some(charge),
                ProxyError::ContentFiltered(crate::error::guardrail_block_message(
                    "response",
                    guardrail_name.as_deref(),
                )),
            )
            .with_routing(routing));
        }
        GuardrailVerdict::Bypass { reason } => {
            // First bypass wins — input bypass already populated
            // bypass_reason if it fired, in which case we keep the
            // earlier signal (it's the policy that failed first).
            if bypass_reason.is_none() {
                bypass_reason = Some(reason);
            }
        }
    }

    // #932: mask-action PII rules rewrite the response AFTER the block
    // check passes and BEFORE the cache write below, so cached entries
    // are stored masked — the matched values don't persist at rest.
    crate::redact::merge_counts(
        redactions_out,
        crate::redact::redact_chat_response(resolved_chain.as_ref(), &mut upstream),
    );

    // Cache write is gated on the same policy as the lookup at the
    // top of dispatch — without a matching enabled cache_policy in
    // snapshot, we skip both read AND write so the cache backend
    // doesn't fill up with entries that no policy ever asked for.
    //
    // `matched_policy_ttl` carries the policy's `ttl_seconds`; we feed
    // it to `put_with_ttl` so each entry expires per its own policy,
    // not the cache backend's global fallback. Backends without
    // per-entry support (defined via `Cache::put_with_ttl`'s default
    // impl) silently fall back to `put`.
    if let (Some(ttl), Some(cache), Some(key)) = (
        matched_policy_ttl,
        policy_cache.as_ref(),
        cache_key.as_ref(),
    ) {
        if let Err(err) = cache.put_with_ttl(key, upstream.clone(), ttl).await {
            tracing::warn!(error = %err, key = %key, "cache write failed");
        }
    }

    // Capture request/response content for the observability fan-out (gated:
    // `content_cap` is `None` when no exporter wants it). Built here, where both
    // the request and the upstream response text are in scope; threaded to
    // `fan_out` via `Success`, never to the CP sink.
    let captured_content = content_cap.map(|cap| {
        CapturedContent::new(
            &serde_json::to_string(req).unwrap_or_default(),
            upstream.message.content.as_deref().unwrap_or(""),
            cap as usize,
        )
    });

    let mut response = Json(render_response(now, upstream, &req.model)).into_response();
    if matches!(cache_status, CacheStatus::Miss) {
        // Miss header only when the cache was actually consulted —
        // policy-disabled requests have no cache header at all so a
        // user can tell at a glance whether the gate was open.
        response
            .headers_mut()
            .insert(CACHE_HEADER, HeaderValue::from_static("miss"));
    }

    // Header presence is the wire signal for "routing happened" —
    // see `served_by_target_for_routing`. The helper covers all
    // three branches (non-streaming, streaming, cache hit) with the
    // same policy so a refactor can't silently flip one of them.
    let served_by_target = served_by_target_for_routing(
        virtual_entry.value.routing.is_some() || virtual_entry.value.is_semantic(),
        chosen_target_display_name,
    );

    Ok(Success {
        response,
        provider: provider_name,
        model_id,
        prompt_tokens: Some(prompt),
        completion_tokens: Some(completion),
        total_tokens: Some(total),
        usage_estimated,
        cached_prompt_tokens,
        reasoning_tokens,
        cache_creation_tokens,
        cache_read_tokens,
        provider_request_id,
        provider_model_version,
        provider_key_id,
        upstream_model,
        finish_reason,
        cost_usd,
        bypass_reason,
        cache_status,
        // Cache-saved counters are zero on the upstream-served path —
        // the request *did* hit the upstream, no work was saved.
        cache_hit_saved_input_tokens: 0,
        cache_hit_saved_output_tokens: 0,
        telemetry_handled_by_stream: false,
        served_by_target,
        served_by_route: semantic_route.clone(),
        routing,
        captured_content,
    })
}

/// Orchestrate one ensemble request: fan out to the panel + judge via
/// [`crate::ensemble::run_ensemble`], run the output guardrail on the
/// synthesized answer, commit the aggregate token usage against the
/// single entry-level reservation, and emit one usage event per
/// sub-call (each panel member + the judge), all sharing `request_id`.
///
/// Lives here (not in `ensemble.rs`) so it can use the chat.rs-private
/// `Success` / `DispatchFailure` types, `emit_usage_event`, and
/// `UsageExtras`. The pure fan-out / min-responses / judge-synthesis
/// logic stays in `ensemble.rs`; this is only the dispatch glue.
///
/// `created_ts` is the shared `created` unix timestamp for the rendered
/// response. `reservation` is the SINGLE entry-level reservation taken
/// in `dispatch` — ensemble does not add per-sub-call reservations.
#[allow(clippy::too_many_arguments)]
async fn dispatch_ensemble(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    virtual_entry: &aisix_core::ResourceEntry<aisix_core::Model>,
    req: &ChatFormat,
    request_id: &str,
    created_ts: i64,
    reservation: aisix_ratelimit::MultiReservation,
    resolved_chain: &Arc<dyn aisix_guardrails::Guardrail>,
    applied_guardrails: &[AppliedGuardrail],
    mut bypass_reason: Option<String>,
    model_id: &str,
    api_key_id: &str,
    client: &ClientContext,
    // Input-side PII mask counts (#932) captured by `dispatch` before the
    // fan-out. Ensemble telemetry is emitted per sub-call inside this
    // function (the handler's main emit is skipped), so the counts ride
    // the judge event — the ensemble's terminal event.
    input_redactions: crate::redact::RedactionCounts,
    // Input-side monitor hits (AISIX-Cloud#562), same lifecycle as
    // `input_redactions`.
    input_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
) -> Result<Success, DispatchFailure> {
    let started = Instant::now();
    let with_model = |e: ProxyError| DispatchFailure::new(Some(model_id.to_string()), None, e);

    // The ensemble executor needs every panel member's full answer to
    // synthesize, so a tool-USING request can't be fanned out coherently
    // (panel members might emit tool calls the judge can't reconcile).
    // `tools` / `tool_choice` are flattened keys in `ChatFormat.extra`,
    // not struct fields. Reject only when tools is a NON-EMPTY array, or
    // `tool_choice` forces a call — an empty `tools: []` (which many SDKs
    // always send) means "no tools" and must still fan out. `tool_choice`
    // values `"none"` / `"auto"` don't force a call.
    let has_tools = req
        .extra
        .get("tools")
        .is_some_and(|v| v.as_array().is_none_or(|a| !a.is_empty()));
    let forces_tool = req
        .extra
        .get("tool_choice")
        .is_some_and(|v| v != "none" && v != "auto");
    if has_tools || forces_tool {
        return Err(with_model(ProxyError::InvalidRequest(
            "ensemble models do not support tools".into(),
        )));
    }
    // `is_ensemble()` is the branch guard, so `ensemble` is always Some
    // here; surface a 400 rather than panic if a future refactor breaks
    // that invariant.
    let ensemble_cfg = virtual_entry.value.ensemble.as_ref().ok_or_else(|| {
        with_model(ProxyError::InvalidRequest(
            "model is not an ensemble".into(),
        ))
    })?;

    // Resolve a sub-call's target by display_name → (model_id, provider_key_id,
    // upstream_model). The first two are telemetry attribution; the third is
    // the tokenizer key for the #1074 estimation fallback — the estimator keys
    // off the real upstream model, not the operator alias, exactly as the
    // direct and streaming-judge paths do (a `gpt-4o` alias must select
    // o200k_base, not the cl100k default an unrecognised alias falls back to).
    // Empty ids if the target was deleted between dispatch and emit (cp-api
    // stores NULL); the tokenizer then degrades to the display name.
    let resolve_sub = |display_name: &str| -> (String, String, String) {
        match snapshot.models.get_by_name(display_name) {
            Some(entry) => (
                entry.id.clone(),
                entry.value.provider_key_id.clone().unwrap_or_default(),
                entry
                    .value
                    .upstream_model()
                    .unwrap_or(display_name)
                    .to_string(),
            ),
            None => (String::new(), String::new(), display_name.to_string()),
        }
    };
    // Emit one usage event for a single (already-billed) panel member.
    // Defined before the `run_ensemble` match so the InsufficientPanel arm
    // can bill the survivors too — they hit upstream just like a full panel.
    // `bypass` is passed per call (not captured) so the closure holds no
    // borrow of the mutable `bypass_reason`. `attempt_index` is the member's
    // 0-based slot; `blocked` sets the event's `guardrail_blocked` flag.
    let emit_panel_member =
        |member: &crate::ensemble::PanelOutcome, index: usize, blocked: bool, bypass: &str| {
            let (sub_model_id, sub_provider_key_id, sub_upstream_model) =
                resolve_sub(&member.model);
            // #1074: a member backend that omitted usage gets its prompt
            // estimated from the shared client request and its completion
            // from the member's own answer text, tokenized with the member's
            // resolved upstream model.
            let (prompt_tokens, completion_tokens, usage_estimated) = estimate_subcall_tokens(
                req,
                &sub_upstream_model,
                &member.usage,
                &member.est_output_text,
            );
            emit_usage_event(
                state,
                request_id,
                &sub_model_id,
                &req.model,
                api_key_id,
                200,
                started.elapsed(),
                prompt_tokens,
                completion_tokens,
                UsageExtras {
                    cached_prompt_tokens: member.usage.cached_prompt_tokens,
                    reasoning_tokens: member.usage.reasoning_tokens,
                    cache_creation_tokens: member.usage.cache_creation_tokens,
                    cache_read_tokens: member.usage.cache_read_tokens,
                    usage_estimated,
                    bypass_reason: bypass.to_string(),
                    cache_status: CacheStatus::Disabled.as_str().to_string(),
                    attempt_index: index as u32,
                    attempt_kind: "panel".to_string(),
                    attempt_model: member.model.clone(),
                    applied_guardrails: applied_guardrails.to_vec(),
                    provider_key_id: sub_provider_key_id,
                    ..UsageExtras::default()
                },
                /* cost_usd */ 0.0,
                blocked,
                client,
                /* content */ None,
            );
        };

    let caller = crate::ensemble::ProxyModelCaller {
        state,
        snapshot,
        request_id,
    };

    // Streaming ensemble (OPTION A): the panel must be buffered to synthesize,
    // so phases 1-2 run NON-streaming; only the judge's tokens are streamed,
    // by reusing `build_sse_stream` exactly as the single-upstream path does.
    //
    // TODO(follow-up): keep the socket warm during the panel phase. The panel
    // fan-out runs BEFORE the `Sse` is constructed, so it is not covered by the
    // 15s keep-alive below. This is no worse than the non-streaming ensemble
    // path, which likewise holds the socket bytes-free across panel + judge;
    // warming it would need an early SSE handshake before the panel completes.
    if req.is_streaming() {
        // Emit the already-billed panel survivors, then build the failure.
        // Shared by both streaming error exits (panel exhausted, or judge
        // connect failed): in either case every surviving panel member
        // round-tripped an upstream and must be billed, exactly like the
        // non-streaming error paths. `charge: None` — the per-member events
        // already carry the bill.
        //
        // The token commit is NOT done here: `MultiReservation::commit_tokens`
        // is async (#607's cluster-Redis path), and this helper is sync so
        // `emit_panel_member` stays callable across every exit. Each call site
        // therefore `.await`s `reservation.commit_tokens(survivor_total)`
        // inline FIRST (mirroring the non-streaming ensemble path), then calls
        // this for the emit + `DispatchFailure`. `panel` is borrowed so the
        // call site still owns it to compute `survivor_total`.
        let survivor_total = |panel: &[crate::ensemble::PanelOutcome]| -> u64 {
            panel.iter().map(|p| u64::from(p.usage.total_tokens)).sum()
        };
        let emit_panel_then_fail =
            |panel: &[crate::ensemble::PanelOutcome], proxy_err: ProxyError| -> DispatchFailure {
                for (index, member) in panel.iter().enumerate() {
                    emit_panel_member(
                        member, index, /* blocked */ false, /* bypass */ "",
                    );
                }
                DispatchFailure::new(Some(model_id.to_string()), None, proxy_err)
            };

        // Phases 1-2 + judge-request construction. An exhausted panel bills
        // the survivors and returns 502 (same status mapping as the
        // non-streaming `InsufficientPanel` path).
        let (panel, _candidates, mut judge_req) =
            match crate::ensemble::run_ensemble_panel(req, ensemble_cfg, &caller).await {
                Ok(triple) => triple,
                Err(crate::ensemble::EnsembleError::InsufficientPanel { panel, .. }) => {
                    reservation.commit_tokens(survivor_total(&panel)).await;
                    return Err(emit_panel_then_fail(
                        &panel,
                        ProxyError::Bridge(BridgeError::upstream_status(
                            502,
                            "ensemble panel did not reach the required number of responses",
                        )),
                    ));
                }
                // `run_ensemble_panel` never returns `Judge` (it stops before
                // the judge call), but the enum is non-exhaustive to us here.
                Err(crate::ensemble::EnsembleError::Judge { panel, source }) => {
                    reservation.commit_tokens(survivor_total(&panel)).await;
                    return Err(emit_panel_then_fail(&panel, ProxyError::Bridge(source)));
                }
            };

        // Stream the judge's synthesized answer. Flip the judge request to
        // streaming (the executor built it non-streaming for the buffered
        // path) and resolve its bridge exactly as `ProxyModelCaller::call`
        // does for the non-streaming judge.
        judge_req.stream = Some(true);
        // Resolve the judge model from the snapshot. Effectively unreachable
        // (the panel calls already resolved member names against the same
        // snapshot, and the judge is required config), but stay total: bill the
        // panel + fail rather than panic if the entry vanished mid-request.
        let Some(judge_entry) = snapshot.models.get_by_name(&ensemble_cfg.judge.model) else {
            reservation.commit_tokens(survivor_total(&panel)).await;
            return Err(emit_panel_then_fail(
                &panel,
                ProxyError::Bridge(BridgeError::InvalidUpstreamConfig(
                    "ensemble judge references an unknown model".into(),
                )),
            ));
        };
        let judge_model = &judge_entry.value;
        let judge_pk = match crate::dispatch::resolve_provider_key(snapshot, judge_model) {
            Ok(pk) => pk,
            Err(e) => {
                tracing::warn!(error = %e, "ensemble judge provider key unresolved");
                reservation.commit_tokens(survivor_total(&panel)).await;
                return Err(emit_panel_then_fail(
                    &panel,
                    ProxyError::Bridge(BridgeError::InvalidUpstreamConfig(
                        "ensemble judge has an unresolved provider key".into(),
                    )),
                ));
            }
        };
        let Some(judge_bridge) = crate::dispatch::resolve_bridge(&state.hub, &judge_pk.value)
        else {
            reservation.commit_tokens(survivor_total(&panel)).await;
            return Err(emit_panel_then_fail(
                &panel,
                ProxyError::Bridge(BridgeError::Config(
                    "ensemble judge has no registered bridge".into(),
                )),
            ));
        };
        let mut judge_ctx = BridgeContext::new(
            request_id,
            Arc::new(judge_model.clone()),
            Arc::new(judge_pk.value.clone()),
        );
        if let Some(deadline) = judge_model.request_timeout() {
            judge_ctx = judge_ctx.with_deadline(deadline);
        }

        // #620: enforce the judge's OWN model rate limit for the streamed
        // synthesis too. The panel was reserved per-member in
        // `run_ensemble_panel` (via `ProxyModelCaller::call`), and the
        // non-streaming judge is reserved on that same path — but the streaming
        // judge is dispatched here via `chat_stream`, bypassing it. Reserve
        // before opening the stream so a rate-limited judge fails fast (429);
        // the slot is then held for the stream's lifetime and the judge's own
        // tokens are added post-stream, mirroring the entry reservation below.
        let judge_reservation = match crate::quota::reserve_model_only(
            state,
            &ensemble_cfg.judge.model,
            &judge_entry.id,
            judge_model,
        )
        .await
        {
            Ok(r) => r,
            Err(_) => {
                reservation.commit_tokens(survivor_total(&panel)).await;
                return Err(emit_panel_then_fail(
                    &panel,
                    ProxyError::Bridge(BridgeError::upstream_status(
                        429,
                        "rate limit exceeded for the ensemble judge",
                    )),
                ));
            }
        };

        let judge_stream = match judge_bridge.chat_stream(&judge_req, &judge_ctx).await {
            Ok(s) => s,
            // Judge connect failed AFTER the panel round-tripped: bill the
            // panel (same invariant as the non-streaming judge-failure path),
            // then surface the bridge error (5xx → 502, 4xx preserved).
            Err(be) => {
                reservation.commit_tokens(survivor_total(&panel)).await;
                return Err(emit_panel_then_fail(&panel, ProxyError::Bridge(be)));
            }
        };

        // Pre-resolve every owned value the `'static` on_complete needs. It
        // CANNOT borrow `state`/`snapshot`/`req`/`client` (the closure outlives
        // this frame — it fires on stream drop), so the `emit_panel_member` /
        // `resolve_sub` borrowing closures above are unusable inside it. Clone
        // the per-member + judge telemetry inputs up front.
        struct PanelTelem {
            model_id: String,
            provider_key_id: String,
            attempt_model: String,
            usage: aisix_gateway::chat::UsageStats,
            est_output_text: String,
            /// Resolved upstream model — the tokenizer key for the #1074
            /// estimate, pre-resolved here because the `'static` closure
            /// can't reach the snapshot (mirrors `model_id`).
            est_model: String,
        }
        let panel_telem: Vec<PanelTelem> = panel
            .iter()
            .map(|p| {
                let (model_id, provider_key_id, est_model) = resolve_sub(&p.model);
                PanelTelem {
                    model_id,
                    provider_key_id,
                    attempt_model: p.model.clone(),
                    usage: p.usage.clone(),
                    est_output_text: p.est_output_text.clone(),
                    est_model,
                }
            })
            .collect();
        // #1074: the `'static` on_complete closure cannot borrow `req`, so
        // capture one clone for the panel-member prompt estimate (used only
        // when a member backend omits usage — the guard in
        // `estimate_subcall_tokens` skips the tokenizer otherwise).
        let req_for_panel_est = req.clone();
        let panel_total: u64 = panel.iter().map(|p| u64::from(p.usage.total_tokens)).sum();
        // #614: field-wise panel usage sum (not just total_tokens) folded into
        // the client-facing terminal usage chunk via build_sse_stream's
        // `base_usage`, so a streamed ensemble reports the full panel+judge
        // aggregate — matching the non-streaming path.
        let panel_usage_sum = panel
            .iter()
            .fold(aisix_gateway::chat::UsageStats::default(), |acc, p| {
                acc.saturating_add(&p.usage)
            });
        // The streamed judge estimator keys off `judge_model.upstream_model()`
        // directly (below), so resolve_sub's upstream_model is unused here.
        let (judge_model_id, judge_provider_key_id, _) = resolve_sub(&ensemble_cfg.judge.model);
        let judge_attempt_model = ensemble_cfg.judge.model.clone();
        let judge_attempt_index = panel.len() as u32;

        // Output guardrail context — built the same way as the single-upstream
        // streaming path (skip entirely when the resolved chain is empty).
        let stream_guardrail = if resolved_chain.is_empty() {
            None
        } else {
            Some(StreamGuardrailContext {
                chain: Arc::clone(resolved_chain),
                model_name: req.model.clone(),
            })
        };
        // #790: the client only receives the terminal usage-only chunk when it
        // asked for `stream_options.include_usage`. Computed exactly as the
        // single-upstream path does.
        let client_requested_usage = req
            .extra
            .get("stream_options")
            .and_then(|so| so.get("include_usage"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Content-capturing exporters: the ensemble streams the JUDGE answer,
        // so content capture would mirror the single-upstream path. v1 keeps
        // the ensemble content-free (matching the non-streaming ensemble path,
        // which also passes `content: None`), so no per-chunk accumulation.
        let content_cap: Option<u32> = None;

        // Owned clones for the on_complete telemetry closure (mirrors the
        // single-upstream path's capture block).
        let state_for_telem = state.clone();
        let request_id_for_telem = request_id.to_string();
        let client_model_for_telem = req.model.clone();
        let api_key_id_for_telem = api_key_id.to_string();
        let applied_guardrails_for_telem = applied_guardrails.to_vec();
        let client_for_telem = client.clone();
        let bypass_for_telem = bypass_reason.clone().unwrap_or_default();
        // Input-side PII mask counts (#932); merged with the streamed judge's
        // output-side counts on the terminal (judge) event.
        let input_redactions_for_telem = input_redactions.clone();
        // Input-side monitor hits (AISIX-Cloud#562), merged the same way.
        let input_monitor_hits_for_telem = input_monitor_hits.clone();

        // Hold concurrency for the stream's full lifetime (#450). Snapshot the
        // keys BEFORE consuming the reservation into the owned guard.
        let post_stream_keys = reservation.keys();
        let stream_concurrency_hold = reservation.into_stream_hold();
        // #620: hold the judge's own concurrency slot for the stream lifetime
        // and add its tokens post-stream, the same way the entry reservation is
        // handled (snapshot keys before consuming into the owned guard).
        let judge_post_stream_keys = judge_reservation.keys();
        let judge_concurrency_hold = judge_reservation.into_stream_hold();
        let limiter = Arc::clone(&state.limiter);

        // Token-estimation fallback (AISIX-Cloud#1074) for the streamed
        // judge: `comp` carries the JUDGE's stream-only counts, so the
        // estimator gets the judge's own request (panel members buffered
        // their usage separately).
        let judge_estimator = crate::token_estimate::Estimator::new(
            judge_model.upstream_model().unwrap_or("unknown"),
            crate::token_estimate::PromptInput::Chat(Box::new(judge_req)),
        );
        let sse_stream = build_sse_stream(
            judge_stream,
            created_ts,
            stream_guardrail,
            started,
            // Re-stamp the client-facing ensemble model name (e.g. "council")
            // onto every chunk — never the judge's upstream model id.
            req.model.clone(),
            content_cap,
            client_requested_usage,
            panel_usage_sum,
            Some(judge_estimator),
            move |comp: StreamCompletion| {
                // Rate-limit accounting: the panel tokens (already round-tripped)
                // plus the streamed judge's final total, against every layer.
                for key in &post_stream_keys {
                    limiter.add_tokens_post_stream(key, panel_total + comp.total_tokens);
                }
                // #620: the judge's own model bucket gets only the judge's
                // streamed tokens (each panel member was billed per-member).
                for key in &judge_post_stream_keys {
                    limiter.add_tokens_post_stream(key, comp.total_tokens);
                }
                // Telemetry: one event per panel member (attempt_kind "panel",
                // index 0..N) carrying that member's own buffered usage, then
                // one judge event (attempt_kind "judge", index N) from the
                // streamed `StreamCompletion`. All share `request_id`. This
                // mirrors the non-streaming ensemble's per-sub-call emission,
                // moved into on_complete because the judge counts only land on
                // the terminal SSE chunk.
                for (index, member) in panel_telem.iter().enumerate() {
                    // #1074: same or-semantics fallback as the non-streaming
                    // panel emit — estimate a member's tokens when its backend
                    // omitted usage (members were buffered, so their answer
                    // text is available even though the judge is streamed).
                    let (prompt_tokens, completion_tokens, usage_estimated) =
                        estimate_subcall_tokens(
                            &req_for_panel_est,
                            &member.est_model,
                            &member.usage,
                            &member.est_output_text,
                        );
                    emit_usage_event(
                        &state_for_telem,
                        &request_id_for_telem,
                        &member.model_id,
                        &client_model_for_telem,
                        &api_key_id_for_telem,
                        200,
                        started.elapsed(),
                        prompt_tokens,
                        completion_tokens,
                        UsageExtras {
                            cached_prompt_tokens: member.usage.cached_prompt_tokens,
                            reasoning_tokens: member.usage.reasoning_tokens,
                            cache_creation_tokens: member.usage.cache_creation_tokens,
                            cache_read_tokens: member.usage.cache_read_tokens,
                            usage_estimated,
                            bypass_reason: bypass_for_telem.clone(),
                            cache_status: CacheStatus::Disabled.as_str().to_string(),
                            attempt_index: index as u32,
                            attempt_kind: "panel".to_string(),
                            attempt_model: member.attempt_model.clone(),
                            applied_guardrails: applied_guardrails_for_telem.clone(),
                            provider_key_id: member.provider_key_id.clone(),
                            ..UsageExtras::default()
                        },
                        /* cost_usd */ 0.0,
                        comp.guardrail_blocked,
                        &client_for_telem,
                        /* content */ None,
                    );
                }
                emit_usage_event(
                    &state_for_telem,
                    &request_id_for_telem,
                    &judge_model_id,
                    &client_model_for_telem,
                    &api_key_id_for_telem,
                    200,
                    started.elapsed(),
                    comp.prompt_tokens,
                    comp.completion_tokens,
                    UsageExtras {
                        cached_prompt_tokens: comp.cached_prompt_tokens,
                        reasoning_tokens: comp.reasoning_tokens,
                        cache_creation_tokens: comp.cache_creation_tokens,
                        cache_read_tokens: comp.cache_read_tokens,
                        usage_estimated: comp.usage_estimated,
                        provider_request_id: comp.provider_request_id,
                        provider_model_version: comp.provider_model_version,
                        finish_reason: comp.finish_reason,
                        bypass_reason: if !bypass_for_telem.is_empty() {
                            bypass_for_telem.clone()
                        } else {
                            comp.bypass_reason
                        },
                        cache_status: CacheStatus::Disabled.as_str().to_string(),
                        ttft_ms: comp.ttft_ms,
                        attempt_index: judge_attempt_index,
                        attempt_kind: "judge".to_string(),
                        attempt_model: judge_attempt_model.clone(),
                        applied_guardrails: applied_guardrails_for_telem.clone(),
                        provider_key_id: judge_provider_key_id.clone(),
                        redacted_entity_counts: {
                            let mut merged = input_redactions_for_telem.clone();
                            crate::redact::merge_counts(&mut merged, comp.redacted_entity_counts);
                            merged
                        },
                        guardrail_monitor_hits: {
                            let mut merged = input_monitor_hits_for_telem.clone();
                            merged.extend(comp.monitor_hits);
                            merged
                        },
                        ..UsageExtras::default()
                    },
                    /* cost_usd */ 0.0,
                    comp.guardrail_blocked,
                    &client_for_telem,
                    /* content */ None,
                );
                // SLO histograms (AISIX-Cloud#1011): the handler's
                // record_success is stream-gated, so the ensemble stream
                // records its e2e/TTFT here like the plain streaming path.
                state_for_telem.metrics.record_request_e2e_latency(
                    LatencyLabels {
                        endpoint: "/v1/chat/completions",
                        model: &client_model_for_telem,
                        // Matches the legacy series: no single provider
                        // governs an ensemble response.
                        provider: "ensemble",
                        status: 200,
                        streaming: true,
                    },
                    started.elapsed(),
                );
                state_for_telem.metrics.record_request_ttft(
                    LatencyLabels {
                        endpoint: "/v1/chat/completions",
                        model: &client_model_for_telem,
                        provider: "ensemble",
                        status: 200,
                        streaming: true,
                    },
                    Duration::from_millis(u64::from(comp.ttft_ms)),
                );
                // Release the concurrency permit(s) now the stream is done
                // (or was cancelled) — on_complete fires on both paths (#450).
                drop(stream_concurrency_hold);
                drop(judge_concurrency_hold);
            },
        );
        let response =
            Sse::new(sse_stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
        return Ok(Success {
            response: response.into_response(),
            // No single provider/model/key governs an ensemble response.
            provider: "ensemble".to_string(),
            model_id: model_id.to_string(),
            // Token totals land on the SSE terminal chunk and are forwarded
            // into telemetry from on_complete; the handler skips its own
            // emission via `telemetry_handled_by_stream`.
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            usage_estimated: false,
            cost_usd: 0.0,
            cached_prompt_tokens: 0,
            reasoning_tokens: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
            provider_request_id: String::new(),
            provider_model_version: String::new(),
            provider_key_id: String::new(),
            upstream_model: String::new(),
            finish_reason: String::new(),
            bypass_reason,
            cache_status: CacheStatus::Disabled,
            cache_hit_saved_input_tokens: 0,
            cache_hit_saved_output_tokens: 0,
            // Per-sub-call usage is emitted from on_complete; suppress the
            // handler's own entry-level emission.
            telemetry_handled_by_stream: true,
            // Not a routing request — no `x-aisix-served-by` header.
            served_by_target: None,
            served_by_route: None,
            routing: RoutingTelemetry::default(),
            captured_content: None,
        });
    }

    let mut outcome = match crate::ensemble::run_ensemble(req, ensemble_cfg, &caller).await {
        Ok(outcome) => outcome,
        Err(err) => {
            // Both error variants carry the panel members that already
            // succeeded — those hit upstream and were billed, so the
            // "successful panel members are always billed" invariant must
            // hold on the failure exits too (same class as the output-block
            // path). Extract the survivors + the client-facing ProxyError,
            // then commit + emit them BEFORE returning. No judge usage event:
            // the judge either never ran (InsufficientPanel) or produced no
            // response (Judge). `charge: None` — the per-member events
            // already carry the bill, so passing a charge would double-count.
            //
            // `EnsembleError::http_status` is the single source of truth for
            // the status (judge → its bridge status; exhausted panel → 502);
            // read it before destructuring.
            let status = err.http_status();
            let (panel, proxy_err) = match err {
                crate::ensemble::EnsembleError::Judge { source, panel } => {
                    // Carry the bridge error itself so its full envelope +
                    // status mapping survives (richer than a bare status).
                    (panel, ProxyError::Bridge(source))
                }
                crate::ensemble::EnsembleError::InsufficientPanel { panel, .. } => (
                    panel,
                    ProxyError::Bridge(BridgeError::upstream_status(
                        status,
                        "ensemble panel did not reach the required number of responses",
                    )),
                ),
            };
            let survivor_total: u64 = panel.iter().map(|p| u64::from(p.usage.total_tokens)).sum();
            reservation.commit_tokens(survivor_total).await;
            for (index, member) in panel.iter().enumerate() {
                emit_panel_member(
                    member, index, /* blocked */ false, /* bypass */ "",
                );
            }
            return Err(DispatchFailure::new(
                Some(model_id.to_string()),
                None,
                proxy_err,
            ));
        }
    };

    // Aggregate client-facing usage: every billed sub-call (panel members
    // + judge) counts against quota, since each one already hit an
    // upstream. Commit once against the single entry-level reservation.
    let panel_total: u64 = outcome
        .panel
        .iter()
        .map(|p| u64::from(p.usage.total_tokens))
        .sum();
    let judge_usage = outcome.response.usage.clone();
    let total_tokens = panel_total + u64::from(judge_usage.total_tokens);
    reservation.commit_tokens(total_tokens).await;

    // Emit one usage event per sub-call (each panel member + the judge),
    // all sharing `request_id`. `attempt_kind` is `"panel"` / `"judge"`;
    // `attempt_index` is 0..N for the panel and N for the judge.
    //
    // This MUST run before the output-guardrail check on BOTH paths
    // (allow and block): the panel + judge tokens are already committed
    // above, so skipping these events on a block would under-report panel
    // usage to cp-api. The `emit_subcalls` closure is therefore defined
    // here and invoked from each branch of the guardrail match below —
    // never from the post-match continuation alone.
    // Takes `outcome` as a parameter (not a capture) so the Allow branch
    // below can mask `outcome.response` in place (#932) between the
    // guardrail check and this emit. `redactions` lands on the judge
    // event — the ensemble's terminal telemetry event.
    let emit_subcalls = |outcome: &crate::ensemble::EnsembleOutcome,
                         blocked: bool,
                         bypass: &str,
                         redactions: &crate::redact::RedactionCounts,
                         hits: &[aisix_core::GuardrailMonitorHit]| {
        for (index, member) in outcome.panel.iter().enumerate() {
            emit_panel_member(member, index, blocked, bypass);
        }
        let (judge_model_id, judge_provider_key_id, judge_upstream_model) =
            resolve_sub(&outcome.judge_model);
        // #1074: estimate the judge sub-call when its backend omitted usage —
        // prompt from the judge's synthesis request, completion from the
        // synthesized answer text (read post-mask; token count is
        // mask-invariant to within placeholder length), tokenized with the
        // judge's resolved upstream model.
        let (judge_prompt, judge_completion, judge_estimated) = estimate_subcall_tokens(
            &outcome.judge_req,
            &judge_upstream_model,
            &judge_usage,
            &estimation_output_text(&outcome.response),
        );
        emit_usage_event(
            state,
            request_id,
            &judge_model_id,
            &req.model,
            api_key_id,
            200,
            started.elapsed(),
            judge_prompt,
            judge_completion,
            UsageExtras {
                cached_prompt_tokens: judge_usage.cached_prompt_tokens,
                reasoning_tokens: judge_usage.reasoning_tokens,
                cache_creation_tokens: judge_usage.cache_creation_tokens,
                cache_read_tokens: judge_usage.cache_read_tokens,
                usage_estimated: judge_estimated,
                provider_request_id: outcome.response.id.clone(),
                provider_model_version: outcome.response.model.clone(),
                finish_reason: finish_reason_label(&outcome.response.finish_reason),
                bypass_reason: bypass.to_string(),
                cache_status: CacheStatus::Disabled.as_str().to_string(),
                attempt_index: outcome.panel.len() as u32,
                attempt_kind: "judge".to_string(),
                attempt_model: outcome.judge_model.clone(),
                applied_guardrails: applied_guardrails.to_vec(),
                provider_key_id: judge_provider_key_id,
                redacted_entity_counts: redactions.clone(),
                guardrail_monitor_hits: hits.to_vec(),
                ..UsageExtras::default()
            },
            /* cost_usd */ 0.0,
            blocked,
            client,
            /* content */ None,
        );
    };

    // Output guardrail on the synthesized answer — same contract as the
    // non-streaming path. The tokens are already committed above and the
    // per-sub-call usage events fire inside each branch (so a block still
    // bills the full panel + judge): on a block we therefore pass
    // `charge: None` rather than a judge-only `UpstreamCharge`, which
    // would double-count the judge AND still miss the panel.
    let mut ensemble_redactions = input_redactions.clone();
    let mut ensemble_monitor_hits = input_monitor_hits.clone();
    let (ensemble_verdict, hits) = resolved_chain
        .check_output_non_segment_observed(&outcome.response)
        .await;
    ensemble_monitor_hits.extend(hits);
    let ensemble_verdict = crate::redact::moderate_body(
        resolved_chain.as_ref(),
        crate::redact::Direction::Output,
        ensemble_verdict,
        &mut ensemble_redactions,
        &mut ensemble_monitor_hits,
        |g| crate::redact::redact_chat_response(g, &mut outcome.response),
    )
    .await;
    match ensemble_verdict {
        GuardrailVerdict::Allow => {}
        GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } => {
            tracing::warn!(
                guardrail_hook = "output",
                model = %req.model,
                reason = %reason,
                "guardrail blocked ensemble response"
            );
            // Block is not a bypass, so the output guardrail did not mutate
            // `bypass_reason`; carry the input-only bypass (if any).
            emit_subcalls(
                &outcome,
                true,
                &bypass_reason.clone().unwrap_or_default(),
                &input_redactions,
                &ensemble_monitor_hits,
            );
            return Err(DispatchFailure::new(
                Some(model_id.to_string()),
                None,
                ProxyError::ContentFiltered(crate::error::guardrail_block_message(
                    "response",
                    guardrail_name.as_deref(),
                )),
            ));
        }
        GuardrailVerdict::Bypass { reason } => {
            // First bypass wins — keep an earlier input bypass if present.
            if bypass_reason.is_none() {
                bypass_reason = Some(reason);
            }
        }
    }
    // Allow / Bypass continuation: mask the synthesized answer (#932) —
    // the check above ran on the original text, the client gets the masked
    // one — then emit the (non-blocked) sub-call events with the final
    // bypass value (which an output bypass above may have set).
    crate::redact::merge_counts(
        &mut ensemble_redactions,
        crate::redact::redact_chat_response(resolved_chain.as_ref(), &mut outcome.response),
    );
    emit_subcalls(
        &outcome,
        false,
        &bypass_reason.clone().unwrap_or_default(),
        &ensemble_redactions,
        &ensemble_monitor_hits,
    );

    // The synthesized answer is the client-facing response, rendered with
    // the requested (ensemble) model name. `telemetry_handled_by_stream`
    // is reused as the "telemetry already emitted, do not double-emit"
    // flag — `chat_completions` skips its own `emit_usage_event` when it
    // is set, exactly as it does for the streaming path.
    // #614: the client-facing response reports the AGGREGATE usage — every
    // panel member plus the judge (api7/AISIX-Cloud#804) — so the caller sees
    // the full fan-out cost, not just the judge sub-call's. The per-sub-call
    // breakdown stays in the usage events emitted above (`judge_usage` carried
    // the judge-only count for its event; starting the fold from it adds the
    // judge once, then every panel member).
    let aggregate_usage = outcome
        .panel
        .iter()
        .fold(judge_usage.clone(), |acc, p| acc.saturating_add(&p.usage));
    outcome.response.usage = aggregate_usage;
    let response = Json(render_response(created_ts, outcome.response, &req.model)).into_response();
    Ok(Success {
        response,
        // No single provider/model/key governs an ensemble response.
        provider: "ensemble".to_string(),
        model_id: model_id.to_string(),
        // Per-sub-call usage was emitted above; the entry-level telemetry
        // event is suppressed, so these top-level token fields are unused.
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        usage_estimated: false,
        cached_prompt_tokens: 0,
        reasoning_tokens: 0,
        cache_creation_tokens: 0,
        cache_read_tokens: 0,
        provider_request_id: String::new(),
        provider_model_version: String::new(),
        provider_key_id: String::new(),
        upstream_model: String::new(),
        finish_reason: String::new(),
        cost_usd: 0.0,
        bypass_reason,
        cache_status: CacheStatus::Disabled,
        cache_hit_saved_input_tokens: 0,
        cache_hit_saved_output_tokens: 0,
        // Telemetry already emitted per-sub-call above; suppress the
        // handler's own entry-level emission.
        telemetry_handled_by_stream: true,
        // Not a routing request — no `x-aisix-served-by` header.
        served_by_target: None,
        served_by_route: None,
        routing: RoutingTelemetry::default(),
        captured_content: None,
    })
}

/// Wire-shape label for `FinishReason`. cp-api stores this verbatim
/// in `dpmgr_usage_events.finish_reason`; the dashboard reads it back
/// to distinguish normal stops from truncation / content_filter.
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

#[allow(clippy::too_many_arguments)]
fn record_success(
    metrics: &Metrics,
    provider: &str,
    model: &str,
    api_key_id: &str,
    team_id: Option<&str>,
    user_id: Option<&str>,
    // #890 req-3 readable name + req-4 client type + req-1/req-2 dimensions.
    user_name: Option<&str>,
    provider_key_name: &str,
    client_type: &str,
    stream: bool,
    is_fallback: bool,
    status: u16,
    s: &Success,
    elapsed: Duration,
) {
    let outcome = RequestOutcome::from_status(status);
    metrics.record_request(provider, model, status, outcome, elapsed);
    let request_labels = RequestLabels {
        endpoint: "/v1/chat/completions",
        inbound_protocol: "openai",
        provider,
        model,
        upstream_model: &s.upstream_model,
        provider_key_id: &s.provider_key_id,
        provider_key_name,
        api_key_id,
        team_id: team_id.unwrap_or("unknown"),
        user_id: user_id.unwrap_or("unknown"),
        user_name: user_name.unwrap_or("unknown"),
        stream,
        is_fallback,
        status,
        outcome,
    };
    metrics.record_proxy_request(request_labels, elapsed);
    metrics.record_llm_request(request_labels, elapsed);
    // SLO e2e histogram (AISIX-Cloud#1011): non-streaming only here —
    // `elapsed` for a stream is time-to-response-start; the stream's
    // on_complete records the full duration instead.
    if !stream {
        metrics.record_request_e2e_latency(
            LatencyLabels {
                endpoint: "/v1/chat/completions",
                model,
                provider,
                status,
                streaming: false,
            },
            elapsed,
        );
    }
    if let Some(total) = s.total_tokens {
        metrics.record_tokens(provider, model, total);
    }
    metrics.record_llm_usage(
        UsageLabels {
            endpoint: "/v1/chat/completions",
            inbound_protocol: "openai",
            provider,
            model,
            upstream_model: &s.upstream_model,
            provider_key_id: &s.provider_key_id,
            provider_key_name,
            api_key_id,
            team_id: team_id.unwrap_or("unknown"),
            user_id: user_id.unwrap_or("unknown"),
            user_name: user_name.unwrap_or("unknown"),
        },
        LlmUsage {
            input_tokens: s.prompt_tokens.unwrap_or(0).min(u64::from(u32::MAX)) as u32,
            output_tokens: s.completion_tokens.unwrap_or(0).min(u64::from(u32::MAX)) as u32,
            total_tokens: s.total_tokens.unwrap_or(0).min(u64::from(u32::MAX)) as u32,
            spend_usd: s.cost_usd,
        },
    );
    // #890 req-4: token volume by inbound client type (non-streaming path;
    // streaming tokens arrive in the SSE on_complete and are recorded there).
    // No-op when all counts are zero (e.g. the streaming branch here).
    // #1002: s.total_tokens is the cache-inclusive canonical total.
    // AISIX-Cloud#1044: `model` is the same requested logical model recorded
    // on the UsageLabels above.
    metrics.record_llm_tokens_by_client(
        client_type,
        model,
        s.prompt_tokens.unwrap_or(0),
        s.completion_tokens.unwrap_or(0),
        s.total_tokens.unwrap_or(0),
    );
}

fn record_budget_gauges(
    metrics: &Metrics,
    auth: &AuthenticatedKey,
    budget: Option<&crate::budget::BudgetDetails>,
) {
    let labels = aisix_obs::BudgetLabels {
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref().unwrap_or("unknown"),
        user_id: auth.key().user_id.as_deref().unwrap_or("unknown"),
    };
    if let Some(budget) = budget {
        metrics.set_budget_gauges(
            labels,
            aisix_obs::BudgetGauges {
                limit_usd: budget.limit_usd,
                spent_usd: budget.spent_usd,
                remaining_usd: budget.remaining_usd,
                reset_seconds: budget.reset_seconds,
            },
        );
    } else {
        metrics.clear_budget_gauges(labels);
    }
}

/// Push one telemetry event onto the CP-side sink **and** fan it out
/// to every per-env OTLP/HTTP exporter in the live snapshot.
/// Non-blocking on both legs: the CP sink drops on full queue, the
/// OTLP fan-out detaches a tokio task per exporter. Centralised here
/// so success / error / streaming / cache-hit paths share one event
/// construction and the two emit legs stay in lockstep.
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    requested_model: &str,
    api_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
    extras: UsageExtras,
    cost_usd: f64,
    guardrail_blocked: bool,
    client: &ClientContext,
    content: Option<CapturedContent>,
) {
    // Look up per-PK telemetry attribution tags from the live snapshot.
    // Empty `provider_key_id` (pre-dispatch error paths) → default
    // tags (all empty / false) → wire fields skip-serialize → cp-api
    // stores NULL. See AISIX-Cloud#436.
    let snap = state.snapshot.load();
    let tags = if !extras.provider_key_id.is_empty() {
        snap.provider_keys
            .get_by_id(&extras.provider_key_id)
            .map(|e| e.value.telemetry_tags.clone())
            .unwrap_or_default()
    } else {
        Default::default()
    };
    let event = UsageEvent {
        request_id: request_id.to_string(),
        // RFC 3339 UTC. cp-api parses with time.Parse(time.RFC3339, ...);
        // chrono's `to_rfc3339_opts(Secs, true)` emits the trailing Z.
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        prompt_tokens,
        completion_tokens,
        cached_prompt_tokens: extras.cached_prompt_tokens,
        reasoning_tokens: extras.reasoning_tokens,
        cache_creation_tokens: extras.cache_creation_tokens,
        cache_read_tokens: extras.cache_read_tokens,
        usage_estimated: extras.usage_estimated,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        provider_request_id: extras.provider_request_id,
        provider_model_version: extras.provider_model_version,
        finish_reason: extras.finish_reason,
        cost_usd,
        guardrail_blocked,
        guardrail_bypassed_reason: extras.bypass_reason,
        applied_guardrails: extras.applied_guardrails,
        redacted_entity_counts: extras.redacted_entity_counts,
        guardrail_monitor_hits: extras.guardrail_monitor_hits,
        cache_status: extras.cache_status,
        cache_hit_saved_input_tokens: extras.cache_hit_saved_input_tokens,
        cache_hit_saved_output_tokens: extras.cache_hit_saved_output_tokens,
        ttft_ms: extras.ttft_ms,
        // chat.rs is the OpenAI-shape /v1/chat/completions handler.
        // /v1/responses / /v1/embeddings / /v1/audio* / /v1/images* /
        // /v1/rerank don't emit UsageEvents today; when they do they
        // also pass `"openai"` here.
        inbound_protocol: "openai".to_string(),
        attempt_index: extras.attempt_index,
        attempt_kind: extras.attempt_kind,
        attempt_model: extras.attempt_model,
        error_class: extras.error_class,
        error_message: extras.error_message,
        // Per-PK telemetry attribution (#302 M17 / AISIX-Cloud#436).
        // Source struct is `aisix_core::TelemetryTags`; the wire
        // shape is flat strings + a bool, with skip_serializing_if
        // covering legacy PKs that pre-date attribution.
        // Each operator-defined string is run through `sanitize_tag`
        // as defence-in-depth against log/JSON injection downstream
        // (PR #382 audit MEDIUM-3; admission-side cap tracked
        // separately).
        provider_kind: sanitize_tag(tags.kind.map(|k| k.as_str().to_owned()).unwrap_or_default()),
        provider_featured: tags.featured,
        branded_provider: sanitize_tag(tags.branded_provider.unwrap_or_default()),
        pk_label: sanitize_tag(tags.pk_label.unwrap_or_default()),
        byo_label: sanitize_tag(tags.byo_label.unwrap_or_default()),
        // Downstream client attribution (#492). Already sanitised in the
        // extractor (control-char strip + cap); IP is a formatted addr.
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        // MCP attribution does not apply to the chat path.
        ..Default::default()
    };
    // Handler label "chat" matches the documented enumeration for
    // `aisix_usage_events_emitted_total` (#408). Keep `&'static str`
    // so prometheus cardinality stays bounded.
    state.usage_sink.try_emit("chat", event.clone());
    // Guardrail outcome counters (#379). Recorded here — the one place every
    // chat path (success / error / streaming / cache-hit) funnels through —
    // from the same guardrail fields the UsageEvent carries.
    state
        .metrics
        .record_guardrail_outcome(guardrail_blocked, &event.guardrail_bypassed_reason);
    // Per-env OTLP/HTTP fan-out. The snapshot's exporter table is
    // empty for envs that haven't configured any, so this is a cheap
    // no-op on the common path. Spawned tasks own the POST work and
    // never block the request return.
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, content.as_ref(), exporters.iter().map(|e| &e.value));
}

/// Defence-in-depth sanitiser for operator-defined `ProviderKey
/// .telemetry_tags` string fields before they hit the wire.
///
/// Tag values are operator-controlled (set via the dashboard's
/// provider-key form, persisted in etcd). A malicious operator with
/// PK-write privileges could craft a label like
/// `"production\u{0a}injected-internal-key: secret"` that, while
/// safely JSON-escaped on this gateway↔cp-api hop, may forge log
/// lines or muddle downstream consumers (cp-api logs, dashboards,
/// log-aggregation pipelines) if any of them ever uses
/// non-strict line-oriented parsing.
///
/// We mitigate two ways:
///   1. Strip ASCII control characters (`\n`, `\r`, `\0`, etc.)
///   2. Cap length at 256 chars
///
/// The right place to enforce this in depth is at PK admission
/// (cp-api / dashboard validation on `display_name` / tag fields).
/// This sanitiser is a belt-and-suspenders guard on the emit side
/// — it cannot prevent a malicious tag from being *stored*, but it
/// can prevent the stored value from corrupting downstream logs.
///
/// PR #382 audit MEDIUM-3.
pub(crate) fn sanitize_tag(s: String) -> String {
    if s.is_empty() {
        return s;
    }
    s.chars().filter(|c| !c.is_control()).take(256).collect()
}

/// Provider-detail bundle for `emit_usage_event`. Grouped here so the
/// helper signature stays under the clippy-too-many-arguments bar
/// without losing the structural relationship between the seven new
/// fields. All seven default to "" / 0, which is exactly the
/// "no provider info / no cache or reasoning detail" case (error path,
/// streaming pre-Phase-2).
#[derive(Default)]
struct UsageExtras {
    cached_prompt_tokens: u32,
    reasoning_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    /// True when any token counter was filled by the local estimator
    /// because the upstream reported no usage (AISIX-Cloud#1074). Lands
    /// on `UsageEvent::usage_estimated`.
    usage_estimated: bool,
    provider_request_id: String,
    provider_model_version: String,
    finish_reason: String,
    /// Set when at least one guardrail returned `Bypass` for this
    /// request (remote-API guardrail upstream unreachable +
    /// `fail_open=true`). Goes onto
    /// `dpmgr_usage_events.guardrail_bypassed_reason`. Default empty
    /// string = no bypass; cp-api stores NULL in that case.
    bypass_reason: String,
    /// Lowercased `CacheStatus` (`"hit"` / `"miss"` / `"disabled"`).
    /// Empty default for the error path where the cache lookup never
    /// fired. Goes onto `dpmgr_usage_events.cache_status`.
    cache_status: String,
    /// On a cache HIT, the cached response's prompt + completion
    /// tokens. Zero otherwise. cp-api derives `cost_saved_usd` on
    /// ingest from these + its pricing catalog (see #88).
    cache_hit_saved_input_tokens: u32,
    cache_hit_saved_output_tokens: u32,
    ttft_ms: u32,
    // ─── Per-attempt telemetry (#655) ───
    /// 0-based attempt index within the request.
    attempt_index: u32,
    /// `"initial"` / `"retry"` / `"fallback"`. Empty defaults to
    /// `"initial"` on the wire.
    attempt_kind: String,
    /// Routing target display name for this attempt; empty for direct.
    attempt_model: String,
    /// Bounded error class for a failed attempt; empty on success.
    error_class: String,
    /// Short error message for a failed attempt; empty on success.
    error_message: String,
    /// The `{kind, hook}` set of guardrails that governed this request,
    /// captured at chain-resolve time. Lands on
    /// `dpmgr_usage_events.applied_guardrails` so the dashboard can show
    /// which guardrails ran (#379). Empty for the guardrail-free path and
    /// for requests rejected before resolution.
    applied_guardrails: Vec<AppliedGuardrail>,
    /// UUID of the resolved ProviderKey. Used at emit time to look up
    /// `telemetry_tags` from the snapshot and populate UsageEvent's
    /// per-PK attribution fields (`provider_kind` / `provider_featured`
    /// / `branded_provider` / `pk_label` / `byo_label`).
    /// Empty for pre-dispatch error paths (auth fail, guardrail block
    /// before dispatch) where no ProviderKey was resolved — those
    /// emit events land in cp-api with the tag columns NULL.
    /// See AISIX-Cloud#436 / #302 M17.
    provider_key_id: String,
    /// Per-detector PII mask counts for this request, input + output
    /// merged (#932). Lands on `usage_events.redacted_entity_counts`.
    /// Detector names only, never matched values. Empty = no redaction.
    redacted_entity_counts: crate::redact::RedactionCounts,
    /// Monitor-mode guardrail observations for this request, input +
    /// output merged (AISIX-Cloud#562). Lands on
    /// `usage_events.guardrail_monitor_hits`. Empty = no monitor hit.
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
}

/// Emit one zero-token `UsageEvent` per FAILED attempt of a request
/// (#655). The winning attempt (and pre-dispatch errors) are emitted
/// separately by the caller. No-op when there are no failed attempts
/// (direct-model success / pre-dispatch error). Each event shares the
/// request's `request_id` (the trace key) and carries this attempt's
/// status / error / latency.
#[allow(clippy::too_many_arguments)]
fn emit_failed_attempts(
    state: &ProxyState,
    request_id: &str,
    requested_model: &str,
    api_key_id: &str,
    client: &ClientContext,
    applied_guardrails: &[AppliedGuardrail],
    routing: &RoutingTelemetry,
    // AISIX-Cloud#1013: when every target failed there is no terminal
    // event, so the captured request body rides the LAST failed attempt —
    // the one whose status the caller saw. Other attempts (and the
    // success-path caller) stay content-less to avoid duplicating a large
    // prompt across N events.
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
        emit_usage_event(
            state,
            request_id,
            // Each failed attempt records the TARGET it actually hit
            // (AISIX-Cloud#790), not the group it was resolved from.
            &rec.target_model_id,
            requested_model,
            api_key_id,
            rec.status,
            Duration::from_millis(u64::from(rec.latency_ms)),
            /* prompt_tokens */ 0,
            /* completion_tokens */ 0,
            UsageExtras {
                attempt_index: rec.index,
                attempt_kind: rec.kind.to_string(),
                attempt_model: rec.target_model.clone(),
                error_class: rec.error_class.clone(),
                error_message: rec.error_message.clone(),
                applied_guardrails: applied_guardrails.to_vec(),
                provider_key_id: rec.provider_key_id.clone(),
                ..UsageExtras::default()
            },
            /* cost_usd */ 0.0,
            /* guardrail_blocked */ false,
            client,
            content,
        );
    }
}

fn record_error(metrics: &Metrics, err: &ProxyError, model: &str, status: u16, elapsed: Duration) {
    let outcome = RequestOutcome::from_status(status);
    // Provider is unknown for pre-dispatch errors (auth, 404, etc.).
    metrics.record_request("unknown", model, status, outcome, elapsed);
    if let ProxyError::RateLimit(rl) = err {
        metrics.record_ratelimit_rejection(&rl.scope().to_string());
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_access_log(
    method: &str,
    path: &str,
    status: u16,
    latency: Duration,
    provider: Option<&str>,
    model: Option<&str>,
    api_key_id: Option<&str>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
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
    // Per #655 the access log stays ONE line per request (the transport
    // plane), carrying user-perceived `latency` + the final status plus a
    // routing summary. The per-attempt detail lives in telemetry only.
    let served_by = routing.winner().map(|w| w.target_model.as_str());
    AccessLog {
        method,
        path,
        status,
        latency,
        provider,
        model,
        api_key_id,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        request_id,
        served_by_model: served_by.filter(|s| !s.is_empty()),
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

fn created_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Build the SSE stream for a streaming chat response.
///
/// As chunks flow through, we observe each chunk's `usage` field. The
/// upstream typically emits the final usage block on the *last* chunk
/// (OpenAI when `stream_options.include_usage=true`; Anthropic on the
/// `message_delta` carrying `output_tokens`). We carry forward the
/// most recently seen `total_tokens` and, on stream end, hand it to
/// `on_complete` so the caller can post-account it against TPM/TPD.
///
/// Pre-fix the streaming path called `reservation.commit_tokens(0)`
/// up front and never revisited the counter, leaving TPM caps silently
/// bypassed for all streaming traffic and the cost/usage telemetry
/// reporting `$0`. See issue #108.
///
/// `on_complete` runs **once** at end-of-stream. It also runs on the
/// disconnect path (the Drop of the upstream Stream returning `None`
/// — the same loop exit). Errors mid-stream still terminate the loop
/// normally, so partial-response cases commit whatever tokens were
/// observed.
/// What `build_sse_stream` extracts from the upstream stream and hands
/// to `on_complete` once the terminal chunk arrives. Sourced from the
/// `usage` block on whichever chunk carried it (typically the last one
/// before `[DONE]`) plus the most-recently-seen `chunk.id` /
/// `chunk.model` / `chunk.finish_reason`. Each numeric field defaults
/// to 0 (and string field to "") when the upstream never emits a
/// `usage` block — for example, an OpenAI streaming request without
/// `stream_options.include_usage=true`. Callers are responsible for
/// treating 0 as "no signal" the same way the non-streaming path
/// treats `prompt_tokens.unwrap_or(0)`.
#[derive(Default)]
struct StreamCompletion {
    prompt_tokens: u32,
    completion_tokens: u32,
    /// `u64` because the rate-limit accounting consumer (TPM cap)
    /// uses u64; cp-api's wire-shape `prompt_tokens` is u32 but
    /// cumulative-tokens accounting can overflow u32 over a long key.
    total_tokens: u64,
    cached_prompt_tokens: u32,
    reasoning_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    provider_request_id: String,
    provider_model_version: String,
    finish_reason: String,
    /// `true` when an OUTPUT guardrail blocked the response at
    /// end-of-stream (per #204). Read by the on_complete telemetry
    /// closure so `usage_events.guardrail_blocked` reflects the
    /// streaming-blocked path the same way the non-streaming path
    /// already does.
    guardrail_blocked: bool,
    /// First Bypass reason observed at end-of-stream (per #204
    /// audit H1). The non-streaming path captures Bypass into the
    /// telemetry envelope so operators can audit which policy
    /// fail-opened on a streamed response. Empty string = no bypass.
    /// First-bypass-wins matches the non-streaming convention.
    bypass_reason: String,
    /// Time to first token in milliseconds. Set once when the first
    /// Ok(chunk) arrives in `build_sse_stream`.
    ttft_ms: u32,
    /// Count of SSE events the **consumer actually pulled** from the
    /// stream — incremented on the post-yield resume in
    /// `build_sse_stream`. `async_stream::stream!` semantics: code
    /// AFTER a `yield` runs only when the consumer asks for the
    /// next item, so this counter is a reliable "delivered-to-the-
    /// client" signal even when the consumer disconnects mid-stream.
    ///
    /// `CompleteOnDrop::drop` uses this to gate `completion_tokens`:
    /// if the consumer disconnected before any chunk reached them
    /// (`chunks_delivered == 0`), the upstream's `usage` block is
    /// irrelevant — the customer must not be billed for tokens that
    /// never crossed the wire. Issue #419.
    chunks_delivered: u32,
    /// Assembled assistant text for content-capturing exporters, accumulated
    /// across chunks ONLY when an exporter wants full content (bounded to the
    /// capture cap). Empty otherwise. Read by the on_complete telemetry
    /// closure; never reaches the CP sink.
    response_text: String,
    /// Generated output (content + reasoning + tool-call text) accumulated
    /// for the token-estimation fallback (AISIX-Cloud#1074). Always on —
    /// the terminal usage chunk that would make it unnecessary arrives
    /// only at end-of-stream — but bounded to
    /// `token_estimate::OUTPUT_ACCUMULATION_CAP`. Unlike `response_text`
    /// this never leaves the process: it is tokenized in
    /// `CompleteOnDrop::drop` when the upstream reported no usage, then
    /// discarded.
    est_output_text: String,
    /// True when `CompleteOnDrop::drop` filled any token counter from the
    /// local estimator because the upstream never reported it. Threaded
    /// into `UsageEvent::usage_estimated` by the on_complete closure.
    usage_estimated: bool,
    /// Per-detector PII mask counts applied to the held stream at release
    /// (#932). Merged with the input-side counts by the on_complete
    /// telemetry closure. Detector names only, never matched values.
    redacted_entity_counts: crate::redact::RedactionCounts,
    /// Monitor-mode guardrail observations made by the end-of-stream
    /// output checks (AISIX-Cloud#562). Merged with the input-side hits
    /// by the on_complete telemetry closure.
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
}

/// Parameters needed to run output-guardrail evaluation at
/// end-of-stream. Per #204 the streaming path used to skip output
/// guardrails entirely — a `kind: "keyword"` deny-list could be
/// trivially bypassed by setting `stream: true`. Buffer-then-check
/// is the right cadence for blocking guardrails (per-chunk evaluation
/// would still leak prefix bytes 1..N-1 by the time chunk N matches).
struct StreamGuardrailContext {
    chain: Arc<dyn aisix_guardrails::Guardrail>,
    /// Surface in tracing only; the wire envelope is intentionally
    /// generic per #153 ("response blocked by content policy").
    model_name: String,
}

/// Fires `on_complete` with whatever `StreamCompletion` has been
/// accumulated when the guard is dropped. The async_stream body holds
/// this guard for the whole stream lifetime so on_complete fires
/// reliably on BOTH normal completion AND mid-stream cancellation.
///
/// Why this is necessary: code AFTER a `yield` in `async_stream::stream!{}`
/// only runs when the consumer pulls. If axum drops the response
/// future (client disconnect, request timeout, etc.), the generator
/// is dropped at its last suspension point and post-yield code never
/// runs. Pre-Drop-guard, that meant a streaming chat with a client
/// disconnect emitted ZERO telemetry events — the customer was billed
/// upstream, the gateway recorded nothing. Drop runs on cancellation,
/// so the captured `StreamCompletion` (potentially zeros if the
/// disconnect beat the upstream's `usage` chunk) is always shipped.
struct CompleteOnDrop<F: FnOnce(StreamCompletion)> {
    /// `Option<(closure, accumulator)>` — Drop calls the closure
    /// exactly once via `.take()`, so manual disarm before the
    /// natural drop is also safe (we don't currently use that, but
    /// it leaves the door open).
    slot: Option<(F, StreamCompletion)>,
    /// Shared counter of SSE events that crossed the wire to the
    /// consumer. Written by `DeliveryCounter` on each `poll_next ->
    /// Ready(Some(_))`. Read at Drop time to gate completion-side
    /// counters (issue #419).
    ///
    /// Why a shared atomic rather than a field on `StreamCompletion`
    /// incremented post-yield: `async_stream::stream!` resumes
    /// post-yield code only on the **next** consumer poll, so a
    /// consumer that pulls N chunks then disconnects leaves the
    /// counter at N-1, not N. Counting at the wrapper's `poll_next`
    /// returning `Ready(Some(_))` is exact — that's the moment the
    /// item handed to the consumer.
    delivered: Arc<AtomicU32>,
    /// Token-estimation fallback (AISIX-Cloud#1074): when the stream
    /// ends with no upstream-reported usage, Drop fills the missing
    /// counters from this estimator (prompt from the captured request,
    /// completion from `est_output_text`) and sets `usage_estimated`.
    estimator: Option<crate::token_estimate::Estimator>,
}

impl<F: FnOnce(StreamCompletion)> CompleteOnDrop<F> {
    fn comp(&mut self) -> &mut StreamCompletion {
        &mut self
            .slot
            .as_mut()
            .expect("CompleteOnDrop guard accessed after take")
            .1
    }
}

impl<F: FnOnce(StreamCompletion)> Drop for CompleteOnDrop<F> {
    fn drop(&mut self) {
        if let Some((f, mut c)) = self.slot.take() {
            // Snapshot the wire-delivered count for telemetry visibility
            // AND for the cost-leak gate below.
            let delivered = self.delivered.load(Ordering::Relaxed);
            c.chunks_delivered = delivered;
            // Issue #419 — cost-leak gate. If the consumer disconnected
            // before any chunk was delivered (e.g. AbortController.abort()
            // mid-`upstream.next().await`, or axum dropped the response
            // future before the first chunk was pulled), the upstream's
            // `usage` block is meaningless for billing — no completion
            // tokens crossed the wire to the client. Zero out the
            // completion-side counters but keep `prompt_tokens` (the
            // prompt was processed by upstream regardless, and the
            // industry contract is "prompts always billed").
            if delivered == 0 {
                c.completion_tokens = 0;
                c.reasoning_tokens = 0;
                c.cache_creation_tokens = 0;
                c.cache_read_tokens = 0;
                c.total_tokens = c.prompt_tokens as u64;
            }
            // Token-estimation fallback (AISIX-Cloud#1074), after the #419
            // gate so a zero-delivered disconnect never bills estimated
            // completion tokens (the prompt still fills — upstream processed
            // it regardless, same "prompts always billed" contract). Runs
            // only for counters the upstream left at zero; a stream with a
            // real usage chunk is untouched.
            if let Some(est) = self.estimator.take() {
                let output = (delivered > 0).then_some(c.est_output_text.as_str());
                let filled = crate::token_estimate::fill_missing(
                    &est,
                    c.prompt_tokens,
                    c.completion_tokens,
                    output,
                );
                if filled.estimated {
                    c.prompt_tokens = filled.prompt_tokens;
                    c.completion_tokens = filled.completion_tokens;
                    c.usage_estimated = true;
                    // Estimation only fills counters the upstream never
                    // reported, so the cache adders are zero or upstream-
                    // authoritative — the recompute stays consistent with
                    // `total_tokens_with_cache`.
                    c.total_tokens = crate::usage_attr::total_tokens_with_cache(
                        c.prompt_tokens,
                        c.completion_tokens,
                        c.cache_creation_tokens,
                        c.cache_read_tokens,
                    );
                }
            }
            f(c);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_sse_stream<F>(
    upstream: aisix_gateway::ChatChunkStream,
    created: i64,
    output_guardrail: Option<StreamGuardrailContext>,
    started: Instant,
    // Customer-facing model name (alias / routing group), re-stamped
    // onto every SSE chunk's `model` field per AISIX-Cloud#410. Owned
    // so it can move into the `async_stream::stream!` closure.
    client_facing_model: String,
    // Largest content cap any content-capturing exporter wants, or `None` to
    // skip response accumulation entirely (the common, content-free path).
    content_cap: Option<u32>,
    // Whether the CLIENT asked for `stream_options.include_usage`. The
    // gateway asks the upstream regardless (#790, for telemetry); the
    // terminal usage-only chunk is forwarded only when this is true.
    client_requested_usage: bool,
    // Usage already incurred before this stream begins — the buffered panel
    // members of an ensemble (#614). Folded into the client-facing terminal
    // usage chunk so the caller sees the full panel+judge aggregate; the
    // `on_complete` (`comp`) counts stay stream-only. Zero for single-upstream
    // callers, where the fold is a no-op.
    base_usage: aisix_gateway::chat::UsageStats,
    // Token-estimation fallback context (AISIX-Cloud#1074); see
    // `CompleteOnDrop::estimator`.
    estimator: Option<crate::token_estimate::Estimator>,
    on_complete: F,
) -> impl Stream<Item = Result<Event, Infallible>>
where
    F: FnOnce(StreamCompletion) + Send + 'static,
{
    // Shared counter for delivered SSE events. Written by the
    // `DeliveryCounter` wrapper below on each `poll_next ->
    // Ready(Some(_))`; read by `CompleteOnDrop::drop` to gate
    // completion-side counters when the consumer disconnected before
    // any chunk reached the wire (#419).
    let delivered = Arc::new(AtomicU32::new(0));
    let delivered_for_drop = Arc::clone(&delivered);
    let inner = async_stream::stream! {
        // Hold on_complete + the running StreamCompletion accumulator
        // inside a Drop guard so on_complete fires even on client-
        // disconnect cancellation. See `CompleteOnDrop` above for the
        // why; tl;dr without this, async_stream! body code after a
        // yield only runs on consumer pulls.
        let mut guard = CompleteOnDrop {
            slot: Some((on_complete, StreamCompletion::default())),
            delivered: delivered_for_drop,
            estimator,
        };
        futures::pin_mut!(upstream);
        // Per #204: accumulate the assistant's content across chunks
        // so the output guardrail can evaluate the full response at
        // end-of-stream. Allocates only when an output guardrail is
        // configured AND the upstream actually emits content; for
        // requests without an output guardrail this is a no-op
        // borrow of the empty string.
        let mut content_buffer = if output_guardrail.is_some() {
            Some(String::new())
        } else {
            None
        };
        // #448 parity for streaming: accumulate tool-call text (function name +
        // arguments) so the end-of-stream output check scans tool calls too.
        // The chat non-streaming path (`guardrail_output_text`) and /v1/messages
        // streaming already scan tool calls, but chat streaming buffered only
        // `delta.content` — a blocked literal in tool-call `arguments` leaked.
        // Bounded to the same cap as the hold-back buffer so a huge tool-call
        // stream can't grow it without limit. Allocated only with a guardrail.
        let mut tool_calls_buf = if output_guardrail.is_some() {
            Some(String::new())
        } else {
            None
        };
        // P2 (#379) / #466: streamed-output policy folded over the output-hook
        // guardrails. EndOfStreamCheck (reached only when no output-hook
        // guardrail is present) leaves the live-forward path below byte-for-byte
        // unchanged. Window / BufferFull hold content back until it scans clean
        // (BufferFull is now the secure default for output-blocking guardrails).
        let stream_policy = output_guardrail
            .as_ref()
            .map(|ctx| ctx.chain.stream_output_policy())
            .unwrap_or_default();
        let hold_back = stream_policy.holds_back();
        // Content chunks withheld from the wire until their window (or the
        // whole response) scans clean. Hold-back path only. Held PRE-render
        // (#932): the BufferFull release rewrites masked spans across the
        // held chunks before they are rendered + serialised at drain time.
        let mut pending: Vec<aisix_gateway::ChatChunk> = Vec::new();
        // Content accumulated since the last window flush (Window mode);
        // bounded to ~window_size, unlike content_buffer (whole response).
        let mut window_buf = String::new();
        // Set when a BufferFull cap is exceeded with fail-open: stop
        // holding and forward the remainder live.
        let mut cap_released = false;
        // Accumulate the upstream's `usage` block + per-chunk metadata
        // across the stream. Providers typically populate `usage` on
        // the terminal chunk only; using "max" rather than "last" makes
        // the bookkeeping robust to a provider that double-emits.
        // `chunk.id` / `chunk.model` / `chunk.finish_reason` use
        // last-seen-wins because those are stable per stream.
        //
        // Per docs/api-proxy.md §5: "If the upstream stream
        // terminates abnormally, aisix sends a final error chunk
        // and closes the response without `[DONE]`." Track whether
        // an error has been yielded so we can skip the closing
        // `[DONE]` — without this skip, a downstream SDK that
        // treats `[DONE]` as clean-completion would mis-interpret
        // the truncated response as a successful one.
        let mut errored = false;
        let mut first_chunk_seen = false;
        // Render + serialise one held/live chunk into an SSE Event.
        // Serialisation of these plain structs can't realistically fail;
        // the Err arm mirrors the pre-hold-back defensive error frame.
        macro_rules! chunk_event {
            ($chunk:expr) => {{
                let rendered = render_chunk(created, $chunk, &client_facing_model);
                match serde_json::to_string(&rendered) {
                    Ok(json) => Event::default().data(json),
                    Err(err) => {
                        errored = true;
                        Event::default()
                            .event("error")
                            .data(error_frame_payload("internal_error", &err.to_string()))
                    }
                }
            }};
        }
        while let Some(item) = upstream.next().await {
            let maybe_chunk = match item {
                Ok(mut chunk) => {
                    // Record TTFT on the first chunk carrying generated
                    // output (content or tool calls). Skip role-only
                    // chunks that OpenAI emits before actual tokens.
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
                    // Accumulate the assistant's content for the output
                    // guardrail. Window mode keeps only the current window
                    // (bounded memory); other modes accumulate the whole
                    // response in content_buffer. No-op without a guardrail.
                    if let Some(text) = chunk.delta.content.as_deref() {
                        if matches!(
                            stream_policy,
                            aisix_guardrails::StreamOutputPolicy::Window { .. }
                        ) {
                            window_buf.push_str(text);
                        } else if let Some(buf) = content_buffer.as_mut() {
                            buf.push_str(text);
                        }
                        // Content capture: assemble the response for the
                        // observability fan-out, bounded to the cap so a long
                        // stream can't grow the buffer without limit. Only when
                        // an exporter wants full content.
                        if let Some(cap) = content_cap {
                            if comp.response_text.len() < cap as usize {
                                comp.response_text.push_str(text);
                            }
                        }
                    }
                    // Token-estimation accumulator (AISIX-Cloud#1074): all
                    // generated output — content, reasoning, tool-call text —
                    // bounded to its own cap. Always on: whether the fallback
                    // is needed is only known at end-of-stream.
                    {
                        use crate::token_estimate::push_capped;
                        if let Some(text) = chunk.delta.content.as_deref() {
                            push_capped(&mut comp.est_output_text, text);
                        }
                        if let Some(text) = chunk.delta.reasoning_content.as_deref() {
                            push_capped(&mut comp.est_output_text, text);
                        }
                        if let Some(tcs) = chunk.delta.tool_calls.as_ref() {
                            for tc in tcs {
                                if let Some(f) = tc.get("function") {
                                    if let Some(n) = f.get("name").and_then(|v| v.as_str()) {
                                        push_capped(&mut comp.est_output_text, n);
                                    }
                                    if let Some(a) =
                                        f.get("arguments").and_then(|v| v.as_str())
                                    {
                                        push_capped(&mut comp.est_output_text, a);
                                    }
                                }
                            }
                        }
                    }
                    // #448 streaming parity: collect tool-call name + arguments
                    // for the output guardrail. `delta.tool_calls` streams as
                    // partial JSON objects; concatenate their text WITHOUT a
                    // separator so a literal split across deltas reassembles.
                    if let (Some(tcs), Some(buf)) =
                        (chunk.delta.tool_calls.as_ref(), tool_calls_buf.as_mut())
                    {
                        for tc in tcs {
                            if buf.len() >= aisix_guardrails::DEFAULT_STREAM_OUTPUT_BUFFER_BYTES {
                                break;
                            }
                            if let Some(f) = tc.get("function") {
                                if let Some(n) = f.get("name").and_then(|v| v.as_str()) {
                                    buf.push_str(n);
                                }
                                if let Some(a) = f.get("arguments").and_then(|v| v.as_str()) {
                                    buf.push_str(a);
                                }
                            }
                        }
                    }
                    if let Some(u) = chunk.usage.as_ref() {
                        if u.prompt_tokens > comp.prompt_tokens {
                            comp.prompt_tokens = u.prompt_tokens;
                        }
                        if u.completion_tokens > comp.completion_tokens {
                            comp.completion_tokens = u.completion_tokens;
                        }
                        let t = u.total_tokens as u64;
                        if t > comp.total_tokens {
                            comp.total_tokens = t;
                        }
                        if u.cached_prompt_tokens > comp.cached_prompt_tokens {
                            comp.cached_prompt_tokens = u.cached_prompt_tokens;
                        }
                        if u.reasoning_tokens > comp.reasoning_tokens {
                            comp.reasoning_tokens = u.reasoning_tokens;
                        }
                        if u.cache_creation_tokens > comp.cache_creation_tokens {
                            comp.cache_creation_tokens = u.cache_creation_tokens;
                        }
                        if u.cache_read_tokens > comp.cache_read_tokens {
                            comp.cache_read_tokens = u.cache_read_tokens;
                        }
                    }
                    // #790: a usage-only terminal chunk (no delta payload, no
                    // finish_reason) exists because the gateway injected
                    // `include_usage` on the upstream leg. Its counts are
                    // already folded into `comp` above — forward it only when
                    // the client itself asked for usage.
                    if !client_requested_usage
                        && chunk.usage.is_some()
                        && chunk.finish_reason.is_none()
                        && chunk.delta.role.is_none()
                        && chunk.delta.content.is_none()
                        && chunk.delta.tool_calls.is_none()
                        && chunk.delta.reasoning_content.is_none()
                    {
                        continue;
                    }
                    // #614: fold the ensemble panel's usage (`base_usage`) into
                    // the client-facing usage frame, AFTER `comp` captured the
                    // stream-only counts above. No-op when `base_usage` is zero
                    // (every single-upstream caller).
                    //
                    // Assumes the judge emits `usage` on a SINGLE terminal frame
                    // (true for the OpenAI/Anthropic/DeepSeek bridges via the
                    // injected include_usage, so the panel sum lands exactly
                    // once). A judge that stamps usage on multiple chunks
                    // (Gemini/Vertex) would add the panel sum more than once —
                    // tracked in #617 (fix: synthesize one terminal usage frame
                    // from `comp + base_usage`).
                    if let Some(u) = chunk.usage.as_mut() {
                        *u = u.saturating_add(&base_usage);
                    }
                    Some(chunk)
                }
                Err(err) => {
                    errored = true;
                    let etype = err.error_type();
                    yield Ok::<_, Infallible>(
                        Event::default()
                            .event("error")
                            .data(error_frame_payload(etype, &err.to_string())),
                    );
                    None
                }
            };
            let Some(chunk) = maybe_chunk else {
                continue;
            };
            if !hold_back || cap_released {
                // The EndOfStreamCheck path and a released BufferFull cap
                // forward straight to the wire. (Error frames yielded in
                // the Err arm above also bypass the hold; the `errored`
                // skip at end-of-stream then drops any held unscanned
                // content — fail closed.)
                let ev = chunk_event!(chunk);
                yield Ok::<_, Infallible>(ev);
            } else {
                // Hold-back: withhold this chunk until its window (or the
                // whole response) scans clean.
                pending.push(chunk);
                match &stream_policy {
                    aisix_guardrails::StreamOutputPolicy::Window {
                        size_chars,
                        overlap_chars,
                    } => {
                        if window_buf.chars().count() >= *size_chars {
                            if let Some(ctx) = output_guardrail.as_ref() {
                                let synthesized = {
                                    let comp = guard.comp();
                                    aisix_gateway::ChatResponse {
                                        id: comp.provider_request_id.clone(),
                                        model: comp.provider_model_version.clone(),
                                        // Fold tool-call text into the window
                                        // scan (#558) so a blocked tool-call
                                        // arg is caught BEFORE this window's
                                        // held events (which include the
                                        // tool-call chunks) are released —
                                        // otherwise Window mode would leak
                                        // tool-call args ahead of the
                                        // end-of-stream check.
                                        message: aisix_gateway::ChatMessage::assistant({
                                            let w = window_buf.clone();
                                            match tool_calls_buf.as_deref() {
                                                Some(tc) if !tc.is_empty() && !w.is_empty() => {
                                                    format!("{w}\n{tc}")
                                                }
                                                Some(tc) if !tc.is_empty() => tc.to_string(),
                                                _ => w,
                                            }
                                        }),
                                        finish_reason: aisix_gateway::FinishReason::Stop,
                                        usage: aisix_gateway::UsageStats::new(
                                            comp.prompt_tokens,
                                            comp.completion_tokens,
                                        ),
                                    }
                                };
                                let (verdict, hits) =
                                    ctx.chain.check_output_observed(&synthesized).await;
                                guard.comp().monitor_hits.extend(hits);
                                match verdict {
                                    aisix_guardrails::GuardrailVerdict::Block { reason, guardrail_name } => {
                                        tracing::warn!(
                                            guardrail_hook = "output",
                                            model = %ctx.model_name,
                                            reason = %reason,
                                            "guardrail blocked streaming response (window)",
                                        );
                                        errored = true;
                                        guard.comp().guardrail_blocked = true;
                                        yield Ok::<_, Infallible>(
                                            Event::default().event("error").data(
                                                error_frame_payload(
                                                    "content_filter",
                                                    &crate::error::guardrail_block_message(
                                                        "response",
                                                        guardrail_name.as_deref(),
                                                    ),
                                                ),
                                            ),
                                        );
                                        break;
                                    }
                                    aisix_guardrails::GuardrailVerdict::Bypass { reason } => {
                                        let comp = guard.comp();
                                        if comp.bypass_reason.is_empty() {
                                            comp.bypass_reason = reason;
                                        }
                                    }
                                    _ => {}
                                }
                                // Clean (Allow / Bypass): release
                                // this window's chunks, then keep the
                                // trailing overlap as scan context for the
                                // next window (its chunks were already sent).
                                // No mask rewrite here: a redacting (PII)
                                // guardrail always folds the chain to
                                // BufferFull, so Window mode implies no
                                // redactor in the chain.
                                for chunk in pending.drain(..) {
                                    let ev = chunk_event!(chunk);
                                    yield Ok::<_, Infallible>(ev);
                                }
                                // Clamp the retained overlap to cc-1 so a
                                // misconfigured overlap >= window can't keep
                                // the whole buffer and re-scan every
                                // subsequent token (cost/latency guard).
                                let cc = window_buf.chars().count();
                                let keep = (*overlap_chars).min(cc.saturating_sub(1));
                                window_buf = if keep > 0 {
                                    window_buf.chars().skip(cc - keep).collect()
                                } else {
                                    String::new()
                                };
                            }
                        }
                    }
                    aisix_guardrails::StreamOutputPolicy::BufferFull {
                        max_buffer_bytes,
                        on_exceeded_fail_open,
                    } => {
                        let buffered = content_buffer.as_ref().map_or(0, |b| b.len());
                        if buffered > *max_buffer_bytes {
                            if *on_exceeded_fail_open {
                                // Fail-open overflow releases the held
                                // chunks unscanned AND unmasked — the
                                // operator opted into that trade-off via
                                // `on_buffer_exceeded: fail_open`.
                                cap_released = true;
                                for chunk in pending.drain(..) {
                                    let ev = chunk_event!(chunk);
                                    yield Ok::<_, Infallible>(ev);
                                }
                            } else {
                                tracing::warn!(
                                    guardrail_hook = "output",
                                    max_buffer_bytes = *max_buffer_bytes,
                                    "streaming response exceeded max_buffer_bytes; failing closed",
                                );
                                errored = true;
                                guard.comp().guardrail_blocked = true;
                                yield Ok::<_, Infallible>(
                                    Event::default().event("error").data(error_frame_payload(
                                        "content_filter",
                                        "response blocked by content policy",
                                    )),
                                );
                                break;
                            }
                        }
                    }
                    aisix_guardrails::StreamOutputPolicy::EndOfStreamCheck => {}
                }
            }
            // Delivery is counted by the outer DeliveryCounter wrapper
            // at poll_next time, not here — async_stream's yield
            // suspends BEFORE the consumer has actually pulled, so
            // a post-yield increment under-counts by 1 on every
            // abort path (#419, audit follow-up).
        }
        // Per #204: run the output guardrail on the accumulated
        // assistant content BEFORE emitting `[DONE]`. Buffer-then-
        // check is the right cadence for a blocking guardrail:
        // per-chunk evaluation would still leak prefix bytes
        // 1..N-1 to the caller by the time chunk N matches the
        // forbidden literal — the secret reaches the wire
        // regardless. We accept the latency cost (whole completion
        // buffered) in exchange for the security control.
        //
        // Skip the check entirely when:
        //   - no output guardrail is configured (`content_buffer` is `None`)
        //   - the upstream stream errored (already sending error frame; no `[DONE]`)
        // The `errored` skip means a partially-streamed forbidden
        // literal would have already reached the caller — but per
        // docs §5 abnormal termination, the gateway has already
        // signaled the failure via SSE error event. Running the
        // guardrail on the partial would only add a second error
        // frame, not retroactively redact the leak.
        if !errored && !cap_released {
            if hold_back {
                // Final scan of the held content (the last partial window,
                // or — for BufferFull — the whole response), then release
                // the held events if it scans clean.
                if let Some(ctx) = output_guardrail.as_ref() {
                    let content_part = match &stream_policy {
                        aisix_guardrails::StreamOutputPolicy::Window { .. } => window_buf.clone(),
                        _ => content_buffer.clone().unwrap_or_default(),
                    };
                    // Scan content + tool-call text together (#448 streaming
                    // parity) so blocked content in tool-call arguments can't
                    // leak. Joined with a newline; either part may be empty.
                    let tc_part = tool_calls_buf.as_deref().unwrap_or("");
                    let final_text = if tc_part.is_empty() {
                        content_part
                    } else if content_part.is_empty() {
                        tc_part.to_string()
                    } else {
                        format!("{content_part}\n{tc_part}")
                    };
                    let blocked = if final_text.is_empty() {
                        false
                    } else {
                        let synthesized = {
                            let comp = guard.comp();
                            aisix_gateway::ChatResponse {
                                id: comp.provider_request_id.clone(),
                                model: comp.provider_model_version.clone(),
                                message: aisix_gateway::ChatMessage::assistant(final_text),
                                finish_reason: aisix_gateway::FinishReason::Stop,
                                usage: aisix_gateway::UsageStats::new(
                                    comp.prompt_tokens,
                                    comp.completion_tokens,
                                ),
                            }
                        };
                        let (verdict, hits) = ctx
                            .chain
                            .check_output_non_segment_observed(&synthesized)
                            .await;
                        guard.comp().monitor_hits.extend(hits);
                        let mut seg_counts = crate::redact::RedactionCounts::new();
                        let mut seg_hits = Vec::new();
                        let verdict = crate::redact::moderate_body(
                            ctx.chain.as_ref(),
                            crate::redact::Direction::Output,
                            verdict,
                            &mut seg_counts,
                            &mut seg_hits,
                            |g| crate::redact::redact_chat_chunks(g, &mut pending),
                        )
                        .await;
                        guard.comp().monitor_hits.extend(seg_hits);
                        if !seg_counts.is_empty() {
                            // Bedrock masked the held chunks — rebuild the
                            // content-capture accumulator from the masked
                            // content channel (the sync redactor below can't
                            // reproduce a provider-side mask), keeping the
                            // original soft cap (#932 × AISIX-Cloud#947).
                            if let Some(cap) = content_cap {
                                let mut rebuilt = String::new();
                                for c in pending.iter() {
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
                        match verdict {
                            aisix_guardrails::GuardrailVerdict::Block { reason, guardrail_name } => {
                                tracing::warn!(
                                    guardrail_hook = "output",
                                    model = %ctx.model_name,
                                    reason = %reason,
                                    "guardrail blocked streaming response",
                                );
                                errored = true;
                                guard.comp().guardrail_blocked = true;
                                yield Ok::<_, Infallible>(
                                    Event::default().event("error").data(error_frame_payload(
                                        "content_filter",
                                        &crate::error::guardrail_block_message(
                                            "response",
                                            guardrail_name.as_deref(),
                                        ),
                                    )),
                                );
                                true
                            }
                            aisix_guardrails::GuardrailVerdict::Bypass { reason } => {
                                let comp = guard.comp();
                                if comp.bypass_reason.is_empty() {
                                    comp.bypass_reason = reason;
                                }
                                false
                            }
                            _ => false,
                        }
                    };
                    if !blocked {
                        // #932: the whole response is held here, so mask-
                        // action PII rules rewrite the held chunks before
                        // anything reaches the wire (channel reassembly —
                        // a masked span can cross chunk boundaries).
                        if let Some(ctx) = output_guardrail.as_ref() {
                            let counts =
                                crate::redact::redact_chat_chunks(ctx.chain.as_ref(), &mut pending);
                            if !counts.is_empty() {
                                // The wire chunks were masked — mask the
                                // content-capture accumulator too, or the
                                // exported content would carry PII the client
                                // never saw (#932 × AISIX-Cloud#947).
                                crate::redact::redact_captured_output(
                                    ctx.chain.as_ref(),
                                    &mut guard.comp().response_text,
                                );
                                crate::redact::merge_counts(
                                    &mut guard.comp().redacted_entity_counts,
                                    counts,
                                );
                            }
                        }
                        for chunk in pending.drain(..) {
                            let ev = chunk_event!(chunk);
                            yield Ok::<_, Infallible>(ev);
                        }
                    }
                }
            } else if let (Some(content), Some(ctx)) =
                (content_buffer.as_ref(), output_guardrail.as_ref())
            {
                // EndOfStreamCheck: scan content + tool-call text (#448 parity).
                let tc_part = tool_calls_buf.as_deref().unwrap_or("");
                let scan_text = if tc_part.is_empty() {
                    content.clone()
                } else if content.is_empty() {
                    tc_part.to_string()
                } else {
                    format!("{content}\n{tc_part}")
                };
                let synthesized = aisix_gateway::ChatResponse {
                    id: guard.comp().provider_request_id.clone(),
                    model: guard.comp().provider_model_version.clone(),
                    message: aisix_gateway::ChatMessage::assistant(scan_text),
                    finish_reason: aisix_gateway::FinishReason::Stop,
                    usage: aisix_gateway::UsageStats::new(
                        guard.comp().prompt_tokens,
                        guard.comp().completion_tokens,
                    ),
                };
                let (verdict, hits) = ctx.chain.check_output_observed(&synthesized).await;
                guard.comp().monitor_hits.extend(hits);
                match verdict {
                    aisix_guardrails::GuardrailVerdict::Block { reason, guardrail_name } => {
                        // Mirror the non-streaming path's #153
                        // redaction contract: the wire-level message
                        // names only the guardrail that fired (#519
                        // B.4b), and the rich verdict reason (which
                        // carries the matched-pattern detail) goes to
                        // operator logs only.
                        tracing::warn!(
                            guardrail_hook = "output",
                            model = %ctx.model_name,
                            reason = %reason,
                            "guardrail blocked streaming response",
                        );
                        errored = true;
                        guard.comp().guardrail_blocked = true;
                        yield Ok::<_, Infallible>(
                            Event::default()
                                .event("error")
                                .data(error_frame_payload(
                                    "content_filter",
                                    &crate::error::guardrail_block_message(
                                        "response",
                                        guardrail_name.as_deref(),
                                    ),
                                )),
                        );
                    }
                    aisix_guardrails::GuardrailVerdict::Bypass { reason } => {
                        // Per #204 audit H1: capture an output
                        // Bypass at end-of-stream so the post-stream
                        // telemetry payload carries it (parallel to
                        // the non-streaming path's first-bypass-wins
                        // capture). An input-side bypass that fired
                        // earlier already populated the closure's
                        // `bypass_reason_for_telem` snapshot; only
                        // record the output bypass when the slot is
                        // still empty.
                        let comp = guard.comp();
                        if comp.bypass_reason.is_empty() {
                            comp.bypass_reason = reason;
                        }
                    }
                    aisix_guardrails::GuardrailVerdict::Allow => {}
                }
            }
        }
        // Stream completed normally. Yield [DONE] BEFORE the guard
        // drops at end-of-scope so the SSE sentinel reaches the
        // client before the on_complete telemetry fan-out. (Any
        // ordering relationship between the two is non-causal — both
        // are independent side effects on this thread — but matching
        // the prior "on_complete then [DONE]" intent feels right.)
        // For providers that don't emit `usage` in the stream the
        // accumulator's numeric fields stay 0; on_complete callers
        // must treat 0 as "no signal" (cp-api does — its pricing
        // catalog falls back to the standard rate when absent).
        //
        // Per docs §5: skip `[DONE]` on abnormal termination so SDK
        // consumers can detect truncation. The `errored` flag is
        // set by the loop above whenever an error event is yielded
        // OR by the output-guardrail check above on a Block verdict.
        if !errored {
            yield Ok::<_, Infallible>(Event::default().data("[DONE]"));
        }
        // `guard` drops here. On client disconnect, the generator
        // drops at the suspension point inside the loop; Drop fires
        // there with whatever StreamCompletion has been captured up
        // to that point.
    };
    // Hyper polls this generator after the request-id middleware has
    // returned, so re-attach the request span here — while we're still
    // on the handler's stack and it is current. Without it the
    // end-of-stream output-guardrail checks below log without a
    // `request_id` and can't be traced back to the caller's
    // `x-aisix-request-id` (AISIX-Cloud#1060).
    crate::request_id::in_request_span(DeliveryCounter {
        inner: Box::pin(inner),
        delivered,
    })
}

/// Stream wrapper that increments `delivered` on every `poll_next ->
/// Ready(Some(_))`. This is the canonical "consumer actually received
/// this item" signal — it fires AT the moment axum's SSE driver pulls
/// the event off. Pair with `CompleteOnDrop` so the Drop guard can
/// read the exact count even on mid-stream cancellation. Issue #419.
///
/// The inner stream is `Pin<Box<dyn Stream<...> + Send>>` so this
/// wrapper itself is `Unpin` and avoids the `forbid(unsafe_code)`
/// pin-projection dance. The boxing is per-request and negligible
/// against the rest of the chat-completion hot path.
struct DeliveryCounter<T> {
    inner: std::pin::Pin<Box<dyn Stream<Item = T> + Send>>,
    delivered: Arc<AtomicU32>,
}

impl<T> Stream for DeliveryCounter<T> {
    type Item = T;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            std::task::Poll::Ready(Some(item)) => {
                self.delivered.fetch_add(1, Ordering::Relaxed);
                std::task::Poll::Ready(Some(item))
            }
            other => other,
        }
    }
}

/// Build the `data:` payload for an SSE `event: error` frame.
/// The OpenAI Node SDK calls `JSON.parse(sse.data)` BEFORE checking
/// `sse.event === "error"`, so a plain-string payload yields a
/// `SyntaxError` ("Could not parse message into JSON: ...") on the
/// SDK side rather than the typed `APIError` callers expect. Emit
/// the OpenAI error envelope shape per
/// <https://platform.openai.com/docs/guides/error-codes/api-errors>:
/// `{"error": {"message": "...", "type": "..."}}`.
fn error_frame_payload(error_type: &str, message: &str) -> String {
    serde_json::to_string(&serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
        }
    }))
    // unreachable in practice — `serde_json::to_string` of a
    // `Value` cannot fail. The fallback emits a minimal valid
    // envelope so SDK consumers' `JSON.parse` still succeeds.
    .unwrap_or_else(|_| {
        r#"{"error":{"message":"error","type":"internal_error"}}"#.into()
    })
}

#[cfg(test)]
mod cooldown_tests {
    use super::*;
    use aisix_core::CooldownConfig;
    use std::time::Duration as StdDuration;

    fn upstream(status: u16) -> BridgeError {
        BridgeError::upstream_status(status, "boom")
    }

    fn upstream_with_retry_after(status: u16, secs: u64) -> BridgeError {
        BridgeError::upstream_status_with_retry_after(
            status,
            "rate limited",
            Some(StdDuration::from_secs(secs)),
        )
    }

    #[test]
    fn default_config_cooldowns_429() {
        let (ttl, reason) = decide_cooldown(&upstream(429), None).unwrap();
        assert_eq!(ttl, StdDuration::from_secs(30));
        assert_eq!(reason, "upstream_rate_limited");
    }

    #[test]
    fn default_config_cooldowns_401_even_though_non_retryable() {
        // H1 contract: 401 cools down. Non-retryable upstream errors
        // (auth failure) should still take the target out of rotation,
        // because the same key will keep failing on subsequent
        // requests. The retry-vs-cooldown split is the whole point.
        let (ttl, reason) = decide_cooldown(&upstream(401), None).unwrap();
        assert_eq!(ttl, StdDuration::from_secs(30));
        assert_eq!(reason, "upstream_auth_failure");
    }

    #[test]
    fn default_config_cooldowns_408() {
        let (_, reason) = decide_cooldown(&upstream(408), None).unwrap();
        assert_eq!(reason, "upstream_request_timeout");
    }

    #[test]
    fn default_config_cooldowns_5xx() {
        for status in [500, 502, 503, 504] {
            let (_, reason) = decide_cooldown(&upstream(status), None).unwrap();
            assert_eq!(reason, "upstream_server_error", "status={status}");
        }
    }

    #[test]
    fn default_config_skips_400_and_other_4xx() {
        // Caller bugs (400, 403, 422) are not cooldown signals — the
        // model didn't fail, the request did.
        assert!(decide_cooldown(&upstream(400), None).is_none());
        assert!(decide_cooldown(&upstream(403), None).is_none());
        assert!(decide_cooldown(&upstream(422), None).is_none());
    }

    #[test]
    fn default_config_cooldowns_timeout_and_transport_errors() {
        assert!(decide_cooldown(
            &BridgeError::Timeout {
                cause: String::new(),
                elapsed_ms: 30_000
            },
            None
        )
        .is_some());
        assert!(decide_cooldown(&BridgeError::Transport("conn refused".into()), None).is_some());
        assert!(decide_cooldown(&BridgeError::StreamAborted, None).is_some());
        assert!(decide_cooldown(&BridgeError::UpstreamDecode("bad json".into()), None).is_some());
    }

    #[test]
    fn config_disabled_skips_cooldown() {
        let cfg = CooldownConfig {
            enabled: Some(false),
            ..Default::default()
        };
        assert!(decide_cooldown(&upstream(429), Some(&cfg)).is_none());
        assert!(decide_cooldown(&upstream(500), Some(&cfg)).is_none());
    }

    #[test]
    fn config_override_trigger_statuses_excludes_429() {
        // Operator wants 429 to NOT cool down (e.g. heavy retry policy
        // already handles burst). 500s still cool down.
        let cfg = CooldownConfig {
            trigger_statuses: Some(vec![500, 502, 503]),
            ..Default::default()
        };
        assert!(decide_cooldown(&upstream(429), Some(&cfg)).is_none());
        assert!(decide_cooldown(&upstream(503), Some(&cfg)).is_some());
    }

    #[test]
    fn honor_retry_after_uses_upstream_hint() {
        let (ttl, _) = decide_cooldown(&upstream_with_retry_after(429, 75), None).unwrap();
        assert_eq!(ttl, StdDuration::from_secs(75));
    }

    #[test]
    fn honor_retry_after_clamps_to_max_seconds() {
        // Upstream is misbehaving — Retry-After: 100000. Clamp to
        // configured max so we don't lose the target for hours.
        let cfg = CooldownConfig {
            max_seconds: Some(60),
            ..Default::default()
        };
        let (ttl, _) =
            decide_cooldown(&upstream_with_retry_after(429, 100_000), Some(&cfg)).unwrap();
        assert_eq!(ttl, StdDuration::from_secs(60));
    }

    #[test]
    fn honor_retry_after_disabled_falls_back_to_default() {
        let cfg = CooldownConfig {
            honor_retry_after: Some(false),
            default_seconds: Some(45),
            ..Default::default()
        };
        let (ttl, _) = decide_cooldown(&upstream_with_retry_after(429, 5), Some(&cfg)).unwrap();
        assert_eq!(ttl, StdDuration::from_secs(45));
    }

    #[test]
    fn trigger_on_timeout_false_disables_timeout_cooldown() {
        let cfg = CooldownConfig {
            trigger_on_timeout: Some(false),
            ..Default::default()
        };
        assert!(decide_cooldown(
            &BridgeError::Timeout {
                cause: String::new(),
                elapsed_ms: 1
            },
            Some(&cfg)
        )
        .is_none());
    }

    #[test]
    fn config_error_never_cools_down() {
        // Misconfig = WE are wrong; cooling down doesn't help.
        assert!(decide_cooldown(&BridgeError::Config("bad key".into()), None).is_none());
    }
}

#[cfg(test)]
mod sanitize_tag_tests {
    use super::*;

    #[test]
    fn sanitize_tag_empty_stays_empty() {
        assert_eq!(sanitize_tag(String::new()), "");
    }

    #[test]
    fn sanitize_tag_strips_newlines_carriage_returns_and_nul() {
        // Injection attempt: a label that, if echoed verbatim into a
        // line-oriented log on the cp-api side, would forge an
        // "injected-internal-key: secret" line. Sanitiser must strip
        // all control chars including \r and \0.
        let evil = "production\ninjected-internal-key: secret\r\0".to_string();
        let safe = sanitize_tag(evil);
        assert!(!safe.contains('\n'));
        assert!(!safe.contains('\r'));
        assert!(!safe.contains('\0'));
        // Visible chars survive untouched.
        assert!(safe.starts_with("production"));
        assert!(safe.contains("injected-internal-key: secret"));
    }

    #[test]
    fn sanitize_tag_caps_length_at_256() {
        let huge = "a".repeat(10_000);
        let safe = sanitize_tag(huge);
        assert_eq!(safe.len(), 256);
    }

    #[test]
    fn sanitize_tag_preserves_normal_ascii_and_unicode() {
        // Real-world labels include hyphens, slashes, spaces, and
        // non-ASCII (operator might label a team in their language).
        let normal = "team-α / prod-east-1".to_string();
        assert_eq!(sanitize_tag(normal.clone()), normal);
    }
}

#[cfg(test)]
mod complete_on_drop_tests {
    //! Issue #419: streaming-abort cost-leak gate.
    //!
    //! The DP's streaming SSE generator (`build_sse_stream`) wraps an
    //! `on_complete` callback in a `CompleteOnDrop` guard so the
    //! callback fires reliably on BOTH normal stream completion AND
    //! mid-stream cancellation. Before the gate, the upstream's
    //! `usage` block populated `completion_tokens` regardless of
    //! whether any chunk actually reached the client — a customer
    //! who aborted mid-await was still billed for tokens the gateway
    //! never delivered.
    //!
    //! Delivery counting is driven by the `DeliveryCounter` stream
    //! wrapper: every `poll_next -> Ready(Some(_))` increments the
    //! shared `Arc<AtomicU32>`. `CompleteOnDrop::drop` reads that
    //! atomic and gates the completion-side counters on zero.
    //!
    //! These tests exercise the Drop logic directly — no real
    //! upstream stream, no axum, no httptest. Construct a
    //! StreamCompletion, set the shared atomic to simulate "N
    //! chunks delivered to the consumer", drop, observe the
    //! callback args.
    use super::{estimate_subcall_tokens, AtomicU32, CompleteOnDrop, StreamCompletion};
    use std::sync::{Arc, Mutex};

    /// Build the guard with `delivered_count` pre-set on the
    /// shared atomic, drop it, return whatever on_complete received.
    fn drop_and_capture(comp: StreamCompletion, delivered_count: u32) -> StreamCompletion {
        let captured: Arc<Mutex<Option<StreamCompletion>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let delivered = Arc::new(AtomicU32::new(delivered_count));
        {
            let guard = CompleteOnDrop {
                slot: Some((
                    move |c: StreamCompletion| {
                        *cap.lock().unwrap() = Some(c);
                    },
                    comp,
                )),
                delivered,
                estimator: None,
            };
            drop(guard);
        }
        let out = captured.lock().unwrap().take().expect("on_complete fired");
        out
    }

    /// Same as [`drop_and_capture`] but with a token estimator armed
    /// (AISIX-Cloud#1074) — the request is one user message "Hello".
    fn drop_and_capture_with_estimator(
        comp: StreamCompletion,
        delivered_count: u32,
    ) -> StreamCompletion {
        let captured: Arc<Mutex<Option<StreamCompletion>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let delivered = Arc::new(AtomicU32::new(delivered_count));
        let req = aisix_gateway::chat::ChatFormat::new(
            "relay-model",
            vec![aisix_gateway::chat::ChatMessage {
                role: aisix_gateway::chat::Role::User,
                content: Some("Hello".into()),
                content_blocks: None,
                name: None,
                tool_call_id: None,
                extra: serde_json::Map::new(),
            }],
        );
        {
            let guard = CompleteOnDrop {
                slot: Some((
                    move |c: StreamCompletion| {
                        *cap.lock().unwrap() = Some(c);
                    },
                    comp,
                )),
                delivered,
                estimator: Some(crate::token_estimate::Estimator::new(
                    "relay-model",
                    crate::token_estimate::PromptInput::Chat(Box::new(req)),
                )),
            };
            drop(guard);
        }
        let out = captured.lock().unwrap().take().expect("on_complete fired");
        out
    }

    /// AISIX-Cloud#1074: a stream that ended with no upstream usage
    /// fills prompt + completion from the estimator and flags the
    /// completion. Expected prompt: 3 per-message + "user" (1) +
    /// "Hello" (1) + 3 reply priming = 8 (cl100k fallback encoding);
    /// completion: "Hello world" = 2.
    #[test]
    fn estimator_fills_missing_usage_at_drop() {
        let comp = StreamCompletion {
            est_output_text: "Hello world".into(),
            chunks_delivered: 0, // set by Drop, ignored on input
            ..Default::default()
        };
        let out = drop_and_capture_with_estimator(comp, 3);
        assert_eq!(out.prompt_tokens, 8);
        assert_eq!(out.completion_tokens, 2);
        assert_eq!(out.total_tokens, 10);
        assert!(out.usage_estimated);
    }

    /// AISIX-Cloud#1074 × #419: a zero-delivered disconnect still
    /// estimates the prompt (prompts are always billed) but must NOT
    /// bill estimated completion tokens for content that never
    /// crossed the wire.
    #[test]
    fn estimator_respects_zero_delivered_gate() {
        let comp = StreamCompletion {
            est_output_text: "Hello world".into(),
            ..Default::default()
        };
        let out = drop_and_capture_with_estimator(comp, 0);
        assert_eq!(out.prompt_tokens, 8);
        assert_eq!(
            out.completion_tokens, 0,
            "nothing delivered → nothing billed"
        );
        assert_eq!(out.total_tokens, 8);
        assert!(out.usage_estimated);
    }

    /// AISIX-Cloud#1074: upstream-reported usage wins — the armed
    /// estimator must not touch a stream that carried a real usage
    /// block, and the event stays unflagged.
    #[test]
    fn estimator_leaves_upstream_usage_untouched() {
        let comp = StreamCompletion {
            prompt_tokens: 17,
            completion_tokens: 23,
            total_tokens: 40,
            est_output_text: "Hello world".into(),
            ..Default::default()
        };
        let out = drop_and_capture_with_estimator(comp, 5);
        assert_eq!(out.prompt_tokens, 17);
        assert_eq!(out.completion_tokens, 23);
        assert_eq!(out.total_tokens, 40);
        assert!(!out.usage_estimated);
    }

    fn subcall_req(user: &str) -> aisix_gateway::chat::ChatFormat {
        aisix_gateway::chat::ChatFormat::new(
            "relay-model",
            vec![aisix_gateway::chat::ChatMessage {
                role: aisix_gateway::chat::Role::User,
                content: Some(user.into()),
                content_blocks: None,
                name: None,
                tool_call_id: None,
                extra: serde_json::Map::new(),
            }],
        )
    }

    /// #796: the shared ensemble sub-call estimator fills a fully-missing
    /// usage block — prompt from the sub-call's request (one user "Hello"
    /// = 8 in the cl100k fallback), completion from its answer text
    /// ("Hello world" = 2) — and flags it.
    #[test]
    fn estimate_subcall_fills_missing_usage() {
        let usage = aisix_gateway::chat::UsageStats::default();
        let (prompt, completion, estimated) =
            estimate_subcall_tokens(&subcall_req("Hello"), "relay-model", &usage, "Hello world");
        assert_eq!(prompt, 8);
        assert_eq!(completion, 2);
        assert!(estimated);
    }

    /// #796: a sub-call whose backend DID report usage is returned
    /// verbatim and unflagged — the answer text is ignored.
    #[test]
    fn estimate_subcall_preserves_reported_usage() {
        let usage = aisix_gateway::chat::UsageStats {
            prompt_tokens: 17,
            completion_tokens: 23,
            total_tokens: 40,
            ..Default::default()
        };
        let (prompt, completion, estimated) =
            estimate_subcall_tokens(&subcall_req("Hello"), "relay-model", &usage, "Hello world");
        assert_eq!(prompt, 17);
        assert_eq!(completion, 23);
        assert!(!estimated);
    }

    /// #796: per-field or-semantics — a reported prompt is kept while a
    /// missing completion is estimated (and the record is flagged).
    #[test]
    fn estimate_subcall_fills_only_missing_side() {
        let usage = aisix_gateway::chat::UsageStats {
            prompt_tokens: 17,
            completion_tokens: 0,
            total_tokens: 17,
            ..Default::default()
        };
        let (prompt, completion, estimated) =
            estimate_subcall_tokens(&subcall_req("Hello"), "relay-model", &usage, "Hello world");
        assert_eq!(prompt, 17, "reported prompt preserved");
        assert_eq!(completion, 2, "missing completion estimated");
        assert!(estimated);
    }

    #[test]
    fn no_chunks_delivered_zeroes_completion_tokens() {
        // Simulated state: upstream emitted a usage block populating
        // completion_tokens=5, but the consumer aborted before any
        // chunk crossed the wire (DeliveryCounter atomic stayed 0).
        // Drop must zero completion-side counters so the customer
        // isn't billed for tokens that never reached them.
        let comp = StreamCompletion {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            reasoning_tokens: 2,
            cache_creation_tokens: 1,
            cache_read_tokens: 1,
            ..Default::default()
        };

        let out = drop_and_capture(comp, 0);

        assert_eq!(
            out.completion_tokens, 0,
            "completion_tokens must zero when delivered==0 (#419)"
        );
        assert_eq!(out.reasoning_tokens, 0, "reasoning_tokens must zero");
        assert_eq!(out.cache_creation_tokens, 0, "cache_creation must zero");
        assert_eq!(out.cache_read_tokens, 0, "cache_read must zero");
        assert_eq!(
            out.prompt_tokens, 10,
            "prompt_tokens preserved — prompt was processed regardless of delivery"
        );
        assert_eq!(
            out.total_tokens, 10,
            "total_tokens recomputed = prompt_tokens only when completion-side zeroed"
        );
        assert_eq!(out.chunks_delivered, 0, "Drop must persist delivered count");
    }

    #[test]
    fn one_chunk_delivered_preserves_completion_tokens() {
        // At least one chunk reached the consumer (could be a
        // role-only chunk + abort, or full stream + clean exit).
        // Either way, the upstream usage block is the source of
        // truth — pass through unchanged.
        let comp = StreamCompletion {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            reasoning_tokens: 2,
            ..Default::default()
        };

        let out = drop_and_capture(comp, 1);

        assert_eq!(
            out.completion_tokens, 5,
            "completion_tokens preserved when delivered>=1"
        );
        assert_eq!(out.reasoning_tokens, 2);
        assert_eq!(out.total_tokens, 15);
        assert_eq!(out.prompt_tokens, 10);
        assert_eq!(out.chunks_delivered, 1);
    }

    #[test]
    fn no_chunks_no_prompt_zero_total() {
        // Edge: connection dropped so early the upstream didn't even
        // confirm prompt_tokens. Drop produces a clean zeroed
        // completion-side AND prompt_tokens=0.
        let comp = StreamCompletion::default(); // all zeros

        let out = drop_and_capture(comp, 0);

        assert_eq!(out.completion_tokens, 0);
        assert_eq!(out.prompt_tokens, 0);
        assert_eq!(out.total_tokens, 0);
        assert_eq!(out.chunks_delivered, 0);
    }

    #[test]
    fn many_chunks_delivered_preserves_full_state() {
        // Normal stream completion: dozens of chunks pulled, full
        // upstream usage block received, Drop fires on natural exit
        // of the async_stream body. All fields pass through.
        let comp = StreamCompletion {
            prompt_tokens: 100,
            completion_tokens: 250,
            total_tokens: 350,
            cached_prompt_tokens: 30,
            reasoning_tokens: 50,
            ..Default::default()
        };

        let out = drop_and_capture(comp, 42);

        assert_eq!(out.prompt_tokens, 100);
        assert_eq!(out.completion_tokens, 250);
        assert_eq!(out.total_tokens, 350);
        assert_eq!(out.cached_prompt_tokens, 30);
        assert_eq!(out.reasoning_tokens, 50);
        assert_eq!(out.chunks_delivered, 42);
    }
}

#[cfg(test)]
mod delivery_counter_tests {
    //! Integration-style test for the `DeliveryCounter` wrapper +
    //! `CompleteOnDrop` collaboration (audit follow-up to the
    //! original #419 fix). The audit's HIGH-1 finding showed that
    //! the original post-yield approach under-counted by 1 on
    //! every abort path. This test would have caught that.
    //!
    //! Builds a real `Stream` via async_stream::stream!, pulls N
    //! items via `.next().await`, drops the stream, asserts the
    //! `chunks_delivered` value the on_complete callback received
    //! matches N exactly.
    use super::{AtomicU32, CompleteOnDrop, DeliveryCounter, StreamCompletion};
    use futures::Stream;
    use futures::StreamExt;
    use std::sync::{Arc, Mutex};

    /// Build a minimal stream that mirrors `build_sse_stream`'s
    /// shape: an async_stream yielding N units, wrapped in
    /// CompleteOnDrop + DeliveryCounter. The on_complete callback
    /// stores the received StreamCompletion in `captured`.
    fn build_test_stream(
        items: u32,
        captured: Arc<Mutex<Option<StreamCompletion>>>,
    ) -> impl Stream<Item = u32> {
        let delivered = Arc::new(AtomicU32::new(0));
        let delivered_for_drop = Arc::clone(&delivered);
        let inner = async_stream::stream! {
            let mut guard = CompleteOnDrop {
                slot: Some((
                    move |c: StreamCompletion| {
                        *captured.lock().unwrap() = Some(c);
                    },
                    StreamCompletion {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                        ..Default::default()
                    },
                )),
                delivered: delivered_for_drop,
                estimator: None,
            };
            // Silence unused-field warning on the test side; guard
            // is held by the body for its full lifetime.
            let _ = &mut guard;
            for i in 0..items {
                yield i;
            }
        };
        DeliveryCounter {
            inner: Box::pin(inner),
            delivered,
        }
    }

    // NOTE: there is no "zero polls" test case here — `async_stream`
    // only runs its body on the first poll, so creating a stream
    // without ever polling means the inner guard is never even
    // constructed, and on_complete cannot fire. That's not the
    // production failure mode (#419) anyway: axum's Sse driver always
    // polls at least once. The zero-delivered case is covered
    // synthetically in `complete_on_drop_tests` where the Drop logic
    // is exercised directly.

    #[tokio::test]
    async fn pulling_one_chunk_then_drop_yields_one_delivered() {
        // This is the canary case the audit flagged: under the old
        // post-yield approach, this would have come back as 0
        // (off-by-one). With DeliveryCounter at poll_next, the count
        // is 1 — gate skipped, completion_tokens preserved.
        let captured = Arc::new(Mutex::new(None));
        {
            let stream = build_test_stream(5, captured.clone());
            futures::pin_mut!(stream);
            let v = stream.next().await;
            assert_eq!(v, Some(0));
            // Consumer aborts here. Drop fires.
        }
        let out = captured.lock().unwrap().take().expect("on_complete fired");
        assert_eq!(
            out.chunks_delivered, 1,
            "delivery counter must reflect the 1 pull"
        );
        assert_eq!(
            out.completion_tokens, 5,
            "completion_tokens preserved when delivered>=1"
        );
    }

    #[tokio::test]
    async fn pulling_three_then_drop_yields_three_delivered() {
        let captured = Arc::new(Mutex::new(None));
        {
            let stream = build_test_stream(5, captured.clone());
            futures::pin_mut!(stream);
            assert_eq!(stream.next().await, Some(0));
            assert_eq!(stream.next().await, Some(1));
            assert_eq!(stream.next().await, Some(2));
            // 3 pulls; abort.
        }
        let out = captured.lock().unwrap().take().expect("on_complete fired");
        assert_eq!(out.chunks_delivered, 3);
    }

    #[tokio::test]
    async fn pulling_full_stream_yields_count_equal_to_items() {
        let captured = Arc::new(Mutex::new(None));
        {
            let stream = build_test_stream(4, captured.clone());
            futures::pin_mut!(stream);
            let collected: Vec<u32> = stream.by_ref().collect().await;
            assert_eq!(collected, vec![0, 1, 2, 3]);
            // Stream is exhausted; consumer drops naturally.
        }
        let out = captured.lock().unwrap().take().expect("on_complete fired");
        assert_eq!(
            out.chunks_delivered, 4,
            "exact match — DeliveryCounter fires on every Ready(Some)"
        );
    }
}
