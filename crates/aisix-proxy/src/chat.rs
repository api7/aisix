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

use aisix_cache::CacheKey;
use aisix_gateway::{BridgeContext, BridgeError, ChatFormat};
use aisix_guardrails::GuardrailVerdict;
use aisix_obs::{AccessLog, LangfuseEvent, Metrics, RequestOutcome, UsageEvent, UsageSink};
use axum::extract::State;
use axum::http::HeaderValue;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::{Stream, StreamExt};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::render::{render_chunk, render_response};
use crate::routing::is_retryable;
use crate::state::ProxyState;

/// Header set on every non-streaming response indicating whether the
/// response came from the cache (`hit`) or the upstream (`miss`).
pub const CACHE_HEADER: &str = "x-aisix-cache";

pub async fn chat_completions(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    Json(req): Json<ChatFormat>,
) -> Response {
    let started = Instant::now();
    let method = "POST";
    let path = "/v1/chat/completions";
    // Pure UUID — cp-api's telemetry handler stores this in the
    // `request_id UUID` column of dpmgr_usage_events, so a "req-…"
    // prefix on the wire would be rejected (event 0: request_id must
    // be a uuid). The downstream `x-aisix-call-id` header carries the
    // same value for human correlation.
    let request_id = Uuid::new_v4().to_string();
    let api_key_id = auth.entry.id.clone();
    let model_name = req.model.clone();

    let outcome = dispatch(&state, &auth, &req, &request_id).await;

    match outcome {
        Ok(mut success) => {
            let status = 200;
            let elapsed = started.elapsed();
            record_success(
                &state.metrics,
                &success.provider,
                &model_name,
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
            );
            emit_usage_event(
                &state.usage_sink,
                &request_id,
                &success.model_id,
                &api_key_id,
                status,
                elapsed,
                success.prompt_tokens.unwrap_or(0) as u32,
                success.completion_tokens.unwrap_or(0) as u32,
                UsageExtras {
                    cached_prompt_tokens: success.cached_prompt_tokens,
                    reasoning_tokens: success.reasoning_tokens,
                    cache_creation_tokens: success.cache_creation_tokens,
                    cache_read_tokens: success.cache_read_tokens,
                    provider_request_id: success.provider_request_id.clone(),
                    provider_model_version: success.provider_model_version.clone(),
                    finish_reason: success.finish_reason.clone(),
                },
                success.cost_usd,
                /* guardrail_blocked */ false,
            );
            if let Some(lf) = state.langfuse.as_ref() {
                lf.emit(LangfuseEvent {
                    trace_id: request_id.clone(),
                    model: model_name.clone(),
                    provider: success.provider.clone(),
                    input: None,
                    output: None,
                    prompt_tokens: success.prompt_tokens,
                    completion_tokens: success.completion_tokens,
                    total_tokens: success.total_tokens,
                    status_code: status,
                    latency: elapsed,
                    api_key_id: Some(api_key_id.clone()),
                });
            }
            // Inject x-ratelimit-* headers so OpenAI SDK clients see the
            // current window state. We peek *after* the commit so
            // remaining-requests reflects the post-dispatch tally.
            let rl_limits = auth.key().rate_limit.clone().unwrap_or_default();
            if let Some(rl_status) = state.limiter.peek(&api_key_id, &rl_limits) {
                crate::render::inject_ratelimit_headers(&mut success.response, &rl_status);
            }
            // Correlation / routing headers.
            if let Ok(v) = axum::http::HeaderValue::try_from(request_id.as_str()) {
                success.response.headers_mut().insert("x-aisix-call-id", v);
            }
            success.response
        }
        Err((resolved_model_id, err)) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            record_error(&state.metrics, &err, &model_name, status, elapsed);
            emit_access_log(
                method,
                path,
                status,
                elapsed,
                None,
                Some(&model_name),
                Some(&api_key_id),
                None,
                None,
                None,
                &request_id,
            );
            // Telemetry for the failure path. `resolved_model_id` is
            // populated by `dispatch` once the request's `req.model`
            // has been resolved against the snapshot — so a guardrail
            // / budget / rate-limit / bridge error after that point
            // still records which model the request targeted. Errors
            // before resolution (empty messages, ModelNotFound) keep
            // an empty string. ContentFiltered (guardrail) is recorded
            // with the dedicated `guardrail_blocked` flag so the
            // dashboard can surface it on the Blocked tab.
            let guardrail_blocked = matches!(err, ProxyError::ContentFiltered(_));
            emit_usage_event(
                &state.usage_sink,
                &request_id,
                resolved_model_id.as_deref().unwrap_or(""),
                &api_key_id,
                status,
                elapsed,
                /* prompt_tokens */ 0,
                /* completion_tokens */ 0,
                // Error path never reached the upstream — no provider
                // id / model version / finish_reason to record, all
                // zero / empty.
                UsageExtras::default(),
                /* cost_usd */ 0.0,
                guardrail_blocked,
            );
            if let Some(lf) = state.langfuse.as_ref() {
                lf.emit(LangfuseEvent {
                    trace_id: request_id.clone(),
                    model: model_name.clone(),
                    provider: "unknown".to_string(),
                    input: None,
                    output: None,
                    prompt_tokens: None,
                    completion_tokens: None,
                    total_tokens: None,
                    status_code: status,
                    latency: elapsed,
                    api_key_id: Some(api_key_id.clone()),
                });
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
    /// finish_reason / stop_reason as the upstream returned it. Empty
    /// for streaming (no terminal event yet) and cache hits.
    finish_reason: String,
    /// Cost computed via the per-key Budget's pricing table, in USD.
    /// 0.0 when no budget is configured for the key.
    cost_usd: f64,
}

/// Returns `Ok(Success)` on the happy path, or `Err((resolved_model_id, err))`
/// where `resolved_model_id` is `Some(id)` once the request's `req.model`
/// has been resolved against the snapshot, and `None` for errors that
/// fire before resolution (empty messages, ModelNotFound). The caller
/// surfaces this id on the failure-path telemetry event so the
/// dashboard's `/logs` page can show the model that was targeted by a
/// guardrail-blocked / budget-exceeded / bridge-failed request.
async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    req: &ChatFormat,
    request_id: &str,
) -> Result<Success, (Option<String>, ProxyError)> {
    if req.messages.is_empty() {
        return Err((
            None,
            ProxyError::InvalidRequest("messages array must not be empty".into()),
        ));
    }

    let snapshot = state.snapshot.load();
    let virtual_entry = snapshot
        .models
        .get_by_name(&req.model)
        .ok_or_else(|| (None, ProxyError::ModelNotFound(req.model.clone())))?;
    let model_id = virtual_entry.id.clone();

    // Every error from here on attaches the resolved model_id so the
    // failure-path telemetry event in `chat_completions` can surface
    // which model the request targeted.
    let with_model = |e: ProxyError| (Some(model_id.clone()), e);

    if !auth.key().can_access(&req.model) {
        return Err(with_model(ProxyError::ModelForbidden(req.model.clone())));
    }

    // Input guardrails. Run before reservation so a blocked prompt
    // doesn't burn an RPM slot — content-policy refusals shouldn't
    // count against quota.
    if let GuardrailVerdict::Block { reason } = state.guardrails.check_input(req).await {
        return Err(with_model(ProxyError::ContentFiltered(reason)));
    }

    // Budget pre-check via cp-api. The DP no longer owns budget state;
    // cp-api returns a cached/live decision per api_key.
    let decision = state.budgets.check(&auth.entry.id).await;
    if !decision.allowed {
        return Err(with_model(ProxyError::BudgetExceeded(
            decision.reason.unwrap_or_else(|| auth.entry.id.clone()),
        )));
    }

    // Resolve the attempt-list of underlying Model entries. For a
    // routing model we walk targets per the configured strategy; for a
    // single-provider Model we just dispatch to it directly.
    let attempt_models: Vec<aisix_core::Model> =
        if let Some(routing) = virtual_entry.value.routing.as_ref() {
            let names = state.routing.pick_order(&req.model, routing);
            if names.is_empty() {
                return Err(with_model(ProxyError::InvalidRequest(
                    "routing model has no targets".into(),
                )));
            }
            let mut resolved = Vec::with_capacity(names.len());
            for name in &names {
                let target_entry = snapshot.models.get_by_name(name).ok_or_else(|| {
                    with_model(ProxyError::InvalidRequest(format!(
                        "routing target {name:?} does not resolve to a Model"
                    )))
                })?;
                resolved.push(target_entry.value.clone());
            }
            resolved
        } else {
            vec![virtual_entry.value.clone()]
        };

    // For non-routing requests, surface a misconfigured bridge as a
    // proper 503 rather than burying it inside a generic Bridge error.
    // Routing requests rely on the loop's `is_retryable` path so a
    // single bad provider doesn't take down the whole request.
    if attempt_models.len() == 1 {
        let only = &attempt_models[0];
        let provider = only.provider().ok_or_else(|| {
            with_model(ProxyError::InvalidRequest("model has no provider prefix".into()))
        })?;
        if state.hub.get(provider).is_none() {
            return Err(with_model(ProxyError::ProviderUnavailable));
        }
    }

    let rl_key = auth.entry.id.clone();
    let rl_limits = auth.key().rate_limit.clone().unwrap_or_default();
    let reservation = state
        .limiter
        .pre_commit(&rl_key, &rl_limits)
        .map_err(|e| with_model(ProxyError::from(e)))?;

    let now = created_ts();

    // Streaming path: only attempt the first target. Streaming fallback
    // is genuinely hard (we'd have to buffer the stream to detect
    // failure mid-flight) and not worth the complexity for V1.
    if req.is_streaming() {
        let model = &attempt_models[0];
        let provider = model.provider().ok_or_else(|| {
            with_model(ProxyError::InvalidRequest("model has no provider prefix".into()))
        })?;
        let bridge = state
            .hub
            .get(provider)
            .ok_or_else(|| with_model(ProxyError::ProviderUnavailable))?;
        let model_arc = Arc::new(model.clone());
        let ctx = BridgeContext::new(request_id, model_arc);
        let upstream = bridge
            .chat_stream(req, &ctx)
            .await
            .map_err(|e| with_model(ProxyError::Bridge(e)))?;
        reservation.commit_tokens(0);
        let sse_stream = build_sse_stream(upstream, now);
        let response =
            Sse::new(sse_stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
        return Ok(Success {
            response: response.into_response(),
            provider: format!("{provider:?}").to_lowercase(),
            model_id: model_id.clone(),
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            // Streaming path doesn't compute cost yet (no token totals
            // until the stream completes upstream). Phase 2 wires
            // mid-stream accumulation; for now telemetry records 0.
            cost_usd: 0.0,
            // Cache / reasoning counters require parsing the terminal
            // SSE chunk's usage block; not wired yet — Phase 2.
            cached_prompt_tokens: 0,
            reasoning_tokens: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
            provider_request_id: String::new(),
            provider_model_version: String::new(),
            finish_reason: String::new(),
        });
    }

    // Cache lookup keyed on the *virtual* model name so a re-request
    // hits the cache regardless of which target served the original.
    let cache_key = state
        .cache
        .as_ref()
        .map(|_| CacheKey::from_request(req).fingerprint());

    if let (Some(cache), Some(key)) = (state.cache.as_ref(), cache_key.as_ref()) {
        match cache.get(key).await {
            Ok(Some(cached)) => {
                reservation.commit_tokens(0);
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
                    .provider()
                    .map(|p| format!("{p:?}").to_lowercase())
                    .unwrap_or_else(|| "unknown".into());
                let mut response = Json(render_response(now, cached)).into_response();
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
                    finish_reason: String::new(),
                    // Cache hits don't burn cost on our side (we already
                    // paid the upstream price the first time around).
                    cost_usd: 0.0,
                });
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(error = %err, key = %key, "cache lookup failed");
            }
        }
    }

    // Walk the attempt-list. First retryable failure → next target.
    // Non-retryable (4xx) or exhausted budget → propagate the last error.
    let mut last_err: Option<BridgeError> = None;
    let mut chosen_provider: Option<String> = None;
    let mut upstream: Option<aisix_gateway::ChatResponse> = None;

    for model in &attempt_models {
        let Some(provider) = model.provider() else {
            last_err = Some(BridgeError::Config("model has no provider prefix".into()));
            continue;
        };
        let Some(bridge) = state.hub.get(provider) else {
            last_err = Some(BridgeError::Config(
                "no bridge registered for provider".into(),
            ));
            continue;
        };
        let model_arc = Arc::new(model.clone());
        let ctx = BridgeContext::new(request_id, model_arc);

        match bridge.chat(req, &ctx).await {
            Ok(resp) => {
                state.health.record_success(&model.name);
                chosen_provider = Some(format!("{provider:?}").to_lowercase());
                upstream = Some(resp);
                break;
            }
            Err(err) => {
                tracing::warn!(
                    target_model = %model.name,
                    error = %err,
                    retryable = is_retryable(&err),
                    "routing target attempt failed",
                );
                // Only retryable (server-side) errors indicate deployment
                // health deterioration; 4xx are caller mistakes.
                if is_retryable(&err) {
                    state.health.record_failure(&model.name);
                }
                if !is_retryable(&err) {
                    last_err = Some(err);
                    break;
                }
                last_err = Some(err);
                continue;
            }
        }
    }

    let Some(upstream) = upstream else {
        // Bubble the most recent BridgeError through ProxyError::Bridge.
        let err = last_err.unwrap_or_else(|| {
            BridgeError::Config("routing exhausted with no targets attempted".into())
        });
        return Err(with_model(ProxyError::Bridge(err)));
    };
    let provider_name = chosen_provider.unwrap_or_else(|| "unknown".into());

    // Output guardrail. Tokens still count against quota — the upstream
    // already burned them — so commit before the check, and refuse the
    // refusal-write to the cache so a re-request gets a fresh chance.
    let prompt = upstream.usage.prompt_tokens as u64;
    let completion = upstream.usage.completion_tokens as u64;
    let total = upstream.usage.total_tokens as u64;
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
    reservation.commit_tokens(total);

    // cp-api recomputes cost server-side from its pricing catalog when
    // ingesting telemetry; the DP just records 0.0 on the wire.
    let cost_usd = 0.0;

    if let GuardrailVerdict::Block { reason } = state.guardrails.check_output(&upstream).await {
        return Err(with_model(ProxyError::ContentFiltered(reason)));
    }

    if let (Some(cache), Some(key)) = (state.cache.as_ref(), cache_key.as_ref()) {
        if let Err(err) = cache.put(key, upstream.clone()).await {
            tracing::warn!(error = %err, key = %key, "cache write failed");
        }
    }

    let mut response = Json(render_response(now, upstream)).into_response();
    if cache_key.is_some() {
        response
            .headers_mut()
            .insert(CACHE_HEADER, HeaderValue::from_static("miss"));
    }

    Ok(Success {
        response,
        provider: provider_name,
        model_id,
        prompt_tokens: Some(prompt),
        completion_tokens: Some(completion),
        total_tokens: Some(total),
        cached_prompt_tokens,
        reasoning_tokens,
        cache_creation_tokens,
        cache_read_tokens,
        provider_request_id,
        provider_model_version,
        finish_reason,
        cost_usd,
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

fn record_success(
    metrics: &Metrics,
    provider: &str,
    model: &str,
    status: u16,
    s: &Success,
    elapsed: Duration,
) {
    let outcome = RequestOutcome::from_status(status);
    metrics.record_request(provider, model, status, outcome, elapsed);
    if let Some(total) = s.total_tokens {
        metrics.record_tokens(provider, model, total);
    }
}

/// Push one telemetry event onto the CP-side sink. Non-blocking;
/// drops with a tracing::warn! if the worker queue is full.
/// Centralised here so success / error / streaming / cache-hit paths
/// all share the same event-construction logic and the wire shape
/// stays consistent across them.
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    sink: &UsageSink,
    request_id: &str,
    model_id: &str,
    api_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
    extras: UsageExtras,
    cost_usd: f64,
    guardrail_blocked: bool,
) {
    sink.try_emit(UsageEvent {
        request_id: request_id.to_string(),
        // RFC 3339 UTC. cp-api parses with time.Parse(time.RFC3339, ...);
        // chrono's `to_rfc3339_opts(Secs, true)` emits the trailing Z.
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        prompt_tokens,
        completion_tokens,
        cached_prompt_tokens: extras.cached_prompt_tokens,
        reasoning_tokens: extras.reasoning_tokens,
        cache_creation_tokens: extras.cache_creation_tokens,
        cache_read_tokens: extras.cache_read_tokens,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        provider_request_id: extras.provider_request_id,
        provider_model_version: extras.provider_model_version,
        finish_reason: extras.finish_reason,
        cost_usd,
        guardrail_blocked,
    });
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
    provider_request_id: String,
    provider_model_version: String,
    finish_reason: String,
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
) {
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
    }
    .emit();
}

fn created_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn build_sse_stream(
    upstream: aisix_gateway::ChatChunkStream,
    created: i64,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        futures::pin_mut!(upstream);
        while let Some(item) = upstream.next().await {
            let ev = match item {
                Ok(chunk) => {
                    let rendered = render_chunk(created, chunk);
                    match serde_json::to_string(&rendered) {
                        Ok(json) => Event::default().data(json),
                        Err(err) => Event::default()
                            .event("error")
                            .data(err.to_string()),
                    }
                }
                Err(err) => Event::default()
                    .event("error")
                    .data(err.to_string()),
            };
            yield Ok::<_, Infallible>(ev);
        }
        yield Ok::<_, Infallible>(Event::default().data("[DONE]"));
    }
}
