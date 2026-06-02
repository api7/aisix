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
//!   `aisix-provider-anthropic::wire`. Scoped to text content blocks
//!   today (tool_use / thinking / image blocks land in a follow-up).
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

use aisix_obs::{AccessLog, LlmUsage, RequestLabels, RequestOutcome, UsageEvent, UsageLabels};
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

use crate::auth::AuthenticatedKey;
use crate::chat::sanitize_tag;
use crate::error::ProxyError;
use crate::request_id::new_request_id;
use crate::state::ProxyState;

/// Anthropic API version header value injected on every forwarded request.
/// Shared with the `/v1/messages/count_tokens` handler so both Anthropic
/// passthrough paths pin the same version.
pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";

pub async fn messages(
    State(state): State<ProxyState>,
    auth: Result<AuthenticatedKey, ProxyError>,
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
    let request_id = new_request_id();
    let api_key_id = auth.entry.id.clone();

    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let snapshot = state.snapshot.load();
    let model_id = snapshot
        .models
        .get_by_name(&model_name)
        .map(|e| e.id.clone())
        .unwrap_or_default();
    drop(snapshot);

    match dispatch(&state, &auth, &mut body, &request_id, started).await {
        Ok(DispatchOutcome {
            response,
            provider_label,
            provider_key_id,
            upstream_model,
            metrics,
            usage_handled_by_stream,
        }) => {
            let elapsed = started.elapsed();
            let status = response.status().as_u16();
            emit_access_log(
                &model_name,
                &provider_label,
                &api_key_id,
                status,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &provider_label,
                &model_name,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            let outcome = RequestOutcome::from_status(status);
            let labels = RequestLabels {
                endpoint: "/v1/messages",
                inbound_protocol: "anthropic",
                provider: &provider_label,
                model: &model_name,
                upstream_model: &upstream_model,
                provider_key_id: &provider_key_id,
                api_key_id: &api_key_id,
                team_id: auth.key().team_id.as_deref().unwrap_or("unknown"),
                user_id: auth.key().user_id.as_deref().unwrap_or("unknown"),
                status,
                outcome,
            };
            state.metrics.record_proxy_request(labels, elapsed);
            state.metrics.record_llm_request(labels, elapsed);
            if !usage_handled_by_stream {
                emit_anthropic_usage_event(
                    &state,
                    &request_id,
                    &model_id,
                    &api_key_id,
                    &provider_label,
                    &model_name,
                    &provider_key_id,
                    &upstream_model,
                    auth.key().team_id.as_deref(),
                    auth.key().user_id.as_deref(),
                    status,
                    elapsed,
                    metrics,
                );
            }
            response
        }
        Err(err) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                &model_name,
                "unknown",
                &api_key_id,
                status,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                "unknown",
                &model_name,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            // Emit a token-less UsageEvent so the dashboard's Logs tab
            // surfaces the failed Anthropic-SDK request alongside its
            // openai-shape siblings. status_code carries the failure
            // class; tokens stay zero.
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
                status,
                elapsed,
                AnthropicUsageMetrics::default(),
            );
            // /v1/messages must return Anthropic-shape error envelope
            // `{type:"error", error:{type, message}}` so Claude SDKs
            // can parse it — closes #336. The DP-stable taxonomy
            // (`upstream_error`, `invalid_api_key`, …) is preserved
            // on the nested `error.type` per ai-gateway#327.
            err.into_anthropic_response()
        }
    }
}

async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    body: &mut Value,
    request_id: &str,
    started: Instant,
) -> Result<DispatchOutcome, ProxyError> {
    let snapshot = state.snapshot.load();

    // Extract and resolve model.
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::InvalidRequest("`model` field missing".into()))?
        .to_string();

    let model_entry = snapshot
        .models
        .get_by_name(&model_name)
        .ok_or_else(|| ProxyError::ModelNotFound(model_name.clone()))?;

    if !auth.key().can_access(&model_name) {
        return Err(ProxyError::ModelForbidden(model_name.clone()));
    }

    let model_rl =
        crate::quota::ModelRateLimit::from_model(&model_name, &model_entry.id, &model_entry.value);
    let _reservation = crate::quota::enforce(state, auth, Some(&model_rl)).await?;

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
    )?;

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let retry_on_429 = model_entry
        .value
        .routing
        .as_ref()
        .map(|r| r.retry_on_429_or_default())
        .unwrap_or(false);

    // Streaming attempts only the first target (no mid-stream fallback),
    // matching /v1/chat/completions.
    if is_stream {
        return dispatch_to_target(
            state,
            &snapshot,
            body,
            &attempt_models[0],
            &model_name,
            request_id,
            started,
            &auth.entry.id,
            auth.key().team_id.clone(),
            auth.key().user_id.clone(),
        )
        .await;
    }

    // Non-streaming: walk targets, failing over to the next only on a
    // retryable upstream failure. A 4xx / config error is returned
    // as-is — retrying other targets won't help.
    let n = attempt_models.len();
    let mut last_err: Option<ProxyError> = None;
    for (i, target) in attempt_models.iter().enumerate() {
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
        )
        .await
        {
            Ok(outcome) => return Ok(outcome),
            Err(e) => {
                let retryable = matches!(
                    &e,
                    ProxyError::Bridge(be) if crate::routing::is_retryable(be, retry_on_429)
                );
                last_err = Some(e);
                if !(retryable && i + 1 < n) {
                    break;
                }
            }
        }
    }
    Err(last_err.unwrap_or(ProxyError::ProviderUnavailable))
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
    let req_builder = client.post(&url).headers(headers).json(&body);

    let upstream_resp = req_builder
        .send()
        .await
        .map_err(|e| {
            crate::cooldown::note_failure(
                &state.runtime_status,
                model_id,
                model.cooldown.as_ref(),
                aisix_gateway::BridgeError::Transport(e.to_string()),
            )
        })
        .map_err(ProxyError::Bridge)?;

    let status = upstream_resp.status();

    if !status.is_success() {
        let status_u16 = status.as_u16();
        let retry_after = aisix_gateway::parse_retry_after(upstream_resp.headers());
        let message = upstream_resp.text().await.unwrap_or_default();
        let truncated = if message.len() > 1024 {
            format!("{}…", &message[..1024])
        } else {
            message
        };
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
        let body_stream = upstream_resp.bytes_stream();

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

        let parsed_stream =
            build_anthropic_passthrough_stream(body_stream, started, move |usage| {
                // Streaming responses that got this far are 200 — the
                // !status.is_success() guard above returned early on
                // upstream errors.
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
                    200,
                    started.elapsed(),
                    metrics,
                );
            });

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
        })
    } else {
        // Non-streaming: deserialise and re-serialise as JSON. Decode
        // failures cool down the target — a body the bridge can't
        // parse is a real upstream problem worth taking out of
        // rotation, not a caller bug.
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

        let metrics = anthropic_metrics_from_response_json(&json_body);

        // Restore the gateway-facing model name so callers see what they asked for.
        let mut json_body = json_body;
        if let Some(m) = json_body.get_mut("model") {
            // If the upstream echoes the model name, rewrite to the gateway name.
            if m.as_str().map(|s| s == upstream_model).unwrap_or(false) {
                *m = Value::String(model_name.to_string());
            }
        }

        Ok(DispatchOutcome {
            response: Json(json_body).into_response(),
            provider_label,
            provider_key_id: pk_id.to_string(),
            upstream_model,
            metrics,
            usage_handled_by_stream: false,
        })
    }
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
) -> Result<DispatchOutcome, ProxyError> {
    use aisix_gateway::{Bridge, BridgeContext};
    use aisix_provider_anthropic::{
        chat_response_into_anthropic_json, parse_inbound_request,
        translate_anthropic_tool_choice_to_openai, translate_anthropic_tools_to_openai,
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
    let bridge: Arc<dyn Bridge> =
        crate::dispatch::resolve_bridge(&state.hub, provider_key, model.provider.as_deref())
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

    // Translate Anthropic-shape tools/tool_choice in `extra` to
    // OpenAI shape so the non-Anthropic bridge receives the format
    // it expects. Without this, tools are silently dropped (#236).
    if let Some(tools) = chat.extra.remove("tools") {
        if let Some(translated) = translate_anthropic_tools_to_openai(tools) {
            chat.extra.insert("tools".to_string(), translated);
        }
    }
    if let Some(tc) = chat.extra.remove("tool_choice") {
        if let Some(translated) = translate_anthropic_tool_choice_to_openai(tc) {
            chat.extra.insert("tool_choice".to_string(), translated);
        }
    }

    let is_stream = chat.is_streaming();
    let model_arc = Arc::new(model.clone());
    let pk_arc = Arc::new(provider_key.clone());
    let ctx = BridgeContext::new(request_id, model_arc, pk_arc);
    let provider_label = provider.to_ascii_lowercase();
    let provider_key_id = model.provider_key_id.as_deref().unwrap_or("unknown");
    let upstream_model = model.upstream_model().unwrap_or("unknown").to_string();

    if is_stream {
        let upstream = bridge.chat_stream(&chat, &ctx).await.map_err(|err| {
            if let Some((ttl, reason)) =
                crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
            {
                state.runtime_status.mark_cooldown(model_id, ttl, reason);
            }
            ProxyError::Bridge(err)
        })?;
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
        let started_for_telem = started;
        let sse_body = build_anthropic_sse_stream(upstream, encoder, started, move |comp| {
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
                200,
                started_for_telem.elapsed(),
                metrics,
            );
        });

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
        });
    }

    // Non-streaming.
    let resp = bridge.chat(&chat, &ctx).await.map_err(|err| {
        if let Some((ttl, reason)) = crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
        {
            state.runtime_status.mark_cooldown(model_id, ttl, reason);
        }
        ProxyError::Bridge(err)
    })?;
    state.health.record_success(&model.display_name);
    state.runtime_status.mark_healthy(model_id);

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
    let json = chat_response_into_anthropic_json(&resp, model_name);
    Ok(DispatchOutcome {
        response: Json(json).into_response(),
        provider_label,
        provider_key_id: provider_key_id.to_string(),
        upstream_model,
        metrics,
        usage_handled_by_stream: false,
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
    on_complete: impl FnOnce(AnthropicStreamCompletion) + Send + 'static,
) -> axum::body::Body {
    use futures::StreamExt;

    let mut encoder = encoder;
    let stream = async_stream::stream! {
        let mut guard = CompleteAnthropicStreamOnDrop {
            slot: Some((on_complete, AnthropicStreamCompletion::default())),
        };
        let mut upstream = upstream;
        let mut first_chunk_seen = false;
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
                    for ev in encoder.next_events(&chunk) {
                        yield Ok::<_, std::io::Error>(bytes::Bytes::from(ev.to_sse_string()));
                    }
                    if encoder.is_finished() {
                        break;
                    }
                }
                Err(e) => {
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
        if !encoder.is_finished() {
            for ev in encoder.force_finish() {
                yield Ok(bytes::Bytes::from(ev.to_sse_string()));
            }
        }
    };
    axum::body::Body::from_stream(stream)
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
    status_code: u16,
    elapsed: Duration,
    metrics: AnthropicUsageMetrics,
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
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
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
        provider_kind: sanitize_tag(tags.kind.unwrap_or_default()),
        provider_featured: tags.featured,
        branded_provider: sanitize_tag(tags.branded_provider.unwrap_or_default()),
        pk_label: sanitize_tag(tags.pk_label.unwrap_or_default()),
        byo_label: sanitize_tag(tags.byo_label.unwrap_or_default()),
        ..Default::default()
    };
    // Handler label "messages" — Anthropic /v1/messages inbound
    // path. Bucketed prometheus counter (#408).
    state.usage_sink.try_emit("messages", event.clone());
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, exporters.iter().map(|e| &e.value));
    state.metrics.record_llm_usage(
        UsageLabels {
            endpoint: "/v1/messages",
            inbound_protocol: "anthropic",
            provider,
            model,
            upstream_model,
            provider_key_id,
            api_key_id,
            team_id: team_id.unwrap_or("unknown"),
            user_id: user_id.unwrap_or("unknown"),
        },
        LlmUsage {
            input_tokens: metrics.prompt_tokens,
            output_tokens: metrics.completion_tokens,
            total_tokens: metrics
                .prompt_tokens
                .saturating_add(metrics.completion_tokens),
            spend_usd: 0.0,
        },
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
                api_key_id,
                team_id: team_id.unwrap_or("unknown"),
                user_id: user_id.unwrap_or("unknown"),
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
const MAX_SSE_FRAME_BUF_BYTES: usize = 1 << 20; // 1 MiB

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
        }
        Some("message_delta") => {
            if let Some(v) = json.get("usage").and_then(|u| u.get("output_tokens")) {
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
fn find_frame_end(buf: &[u8]) -> Option<usize> {
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
fn extract_sse_data_line(frame: &[u8]) -> Option<&[u8]> {
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
    on_complete: F,
) -> AnthropicDeliveryCounter<reqwest::Result<Bytes>>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
    F: FnOnce(AnthropicStreamUsage) + Send + 'static,
{
    let delivered = Arc::new(AtomicU32::new(0));
    let delivered_for_drop = Arc::clone(&delivered);
    let inner = async_stream::stream! {
        let mut guard = AnthropicStreamGuard {
            slot: Some((on_complete, AnthropicStreamUsage::default())),
            delivered: delivered_for_drop,
        };
        futures::pin_mut!(upstream);
        let mut buf: Vec<u8> = Vec::new();
        let mut first_token_seen = false;
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
            }
            // Forward the original item verbatim (Ok bytes OR Err — an
            // upstream error mid-stream is passed through; the
            // accumulator keeps whatever was captured before it).
            yield item;
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
) {
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
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
        routing_attempts: None,
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
    /// LiteLLM-aligned status→type mapping.
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
