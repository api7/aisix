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
use aisix_obs::{AccessLog, Metrics, RequestOutcome};
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
    let request_id = format!("req-{}", Uuid::new_v4());
    let api_key_id = auth.entry.id.clone();
    let model_name = req.model.clone();

    let outcome = dispatch(&state, &auth, &req, &request_id).await;

    match outcome {
        Ok(success) => {
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
            success.response
        }
        Err(err) => {
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
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    req: &ChatFormat,
    request_id: &str,
) -> Result<Success, ProxyError> {
    if req.messages.is_empty() {
        return Err(ProxyError::InvalidRequest(
            "messages array must not be empty".into(),
        ));
    }

    let snapshot = state.snapshot.load();
    let virtual_entry = snapshot
        .models
        .get_by_name(&req.model)
        .ok_or_else(|| ProxyError::ModelNotFound(req.model.clone()))?;

    if !auth.key().can_access(&req.model) {
        return Err(ProxyError::ModelForbidden(req.model.clone()));
    }

    // Input guardrails. Run before reservation so a blocked prompt
    // doesn't burn an RPM slot — content-policy refusals shouldn't
    // count against quota.
    if let GuardrailVerdict::Block { reason } = state.guardrails.check_input(req).await {
        return Err(ProxyError::ContentFiltered(reason));
    }

    // Resolve the attempt-list of underlying Model entries. For a
    // routing model we walk targets per the configured strategy; for a
    // single-provider Model we just dispatch to it directly.
    let attempt_models: Vec<aisix_core::Model> =
        if let Some(routing) = virtual_entry.value.routing.as_ref() {
            let names = state.routing.pick_order(&req.model, routing);
            if names.is_empty() {
                return Err(ProxyError::InvalidRequest(
                    "routing model has no targets".into(),
                ));
            }
            let mut resolved = Vec::with_capacity(names.len());
            for name in &names {
                let target_entry = snapshot.models.get_by_name(name).ok_or_else(|| {
                    ProxyError::InvalidRequest(format!(
                        "routing target {name:?} does not resolve to a Model"
                    ))
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
        let provider = only
            .provider()
            .ok_or_else(|| ProxyError::InvalidRequest("model has no provider prefix".into()))?;
        if state.hub.get(provider).is_none() {
            return Err(ProxyError::ProviderUnavailable);
        }
    }

    let rl_key = auth.entry.id.clone();
    let rl_limits = auth.key().rate_limit.clone().unwrap_or_default();
    let reservation = state.limiter.pre_commit(&rl_key, &rl_limits)?;

    let now = created_ts();

    // Streaming path: only attempt the first target. Streaming fallback
    // is genuinely hard (we'd have to buffer the stream to detect
    // failure mid-flight) and not worth the complexity for V1.
    if req.is_streaming() {
        let model = &attempt_models[0];
        let provider = model
            .provider()
            .ok_or_else(|| ProxyError::InvalidRequest("model has no provider prefix".into()))?;
        let bridge = state
            .hub
            .get(provider)
            .ok_or(ProxyError::ProviderUnavailable)?;
        let model_arc = Arc::new(model.clone());
        let ctx = BridgeContext::new(request_id, model_arc);
        let upstream = bridge.chat_stream(req, &ctx).await?;
        reservation.commit_tokens(0);
        let sse_stream = build_sse_stream(upstream, now);
        let response =
            Sse::new(sse_stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
        return Ok(Success {
            response: response.into_response(),
            provider: format!("{provider:?}").to_lowercase(),
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
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
                    prompt_tokens: Some(prompt),
                    completion_tokens: Some(completion),
                    total_tokens: Some(total),
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
        return Err(ProxyError::Bridge(err));
    };
    let provider_name = chosen_provider.unwrap_or_else(|| "unknown".into());

    // Output guardrail. Tokens still count against quota — the upstream
    // already burned them — so commit before the check, and refuse the
    // refusal-write to the cache so a re-request gets a fresh chance.
    let prompt = upstream.usage.prompt_tokens as u64;
    let completion = upstream.usage.completion_tokens as u64;
    let total = upstream.usage.total_tokens as u64;
    reservation.commit_tokens(total);

    if let GuardrailVerdict::Block { reason } = state.guardrails.check_output(&upstream).await {
        return Err(ProxyError::ContentFiltered(reason));
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
        prompt_tokens: Some(prompt),
        completion_tokens: Some(completion),
        total_tokens: Some(total),
    })
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
