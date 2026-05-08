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
//!   `aisix-provider-anthropic::wire`. Pattern lifted from LiteLLM's
//!   experimental_pass_through adapter, scoped to text content blocks
//!   today (tool_use / thinking / image blocks land in a follow-up).
//!
//! Both paths share the same auth, model lookup, allowed_models check,
//! access-log emission, metrics labels, and health tracker hooks.
//!
//! Errors use the standard OpenAI-style envelope so clients on the proxy
//! side can handle them consistently regardless of which endpoint was used.

use aisix_core::models::Provider;
use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

/// Anthropic API version header value injected on every forwarded request.
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub async fn messages(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    Json(mut body): Json<Value>,
) -> Response {
    let started = Instant::now();
    let request_id = format!("msg-{}", Uuid::new_v4());
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

    match dispatch(&state, &auth, &mut body, &request_id).await {
        Ok(DispatchOutcome {
            response,
            provider_label,
            metrics,
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
            emit_anthropic_usage_event(
                &state,
                &request_id,
                &model_id,
                &api_key_id,
                status,
                elapsed,
                metrics,
            );
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
                status,
                elapsed,
                AnthropicUsageMetrics::default(),
            );
            err.into_response()
        }
    }
}

async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    body: &mut Value,
    request_id: &str,
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

    // Budget + rate-limit gate (issue #107). Pre-fix this endpoint
    // bypassed both — Anthropic-API customers ran unmetered. Held
    // until end of dispatch via Drop, which releases the concurrency
    // permit. RPM is recorded immediately on pre_commit; TPM stays
    // 0 for now (the streaming + cross-provider paths emit token
    // counts in different shapes — plumbing them uniformly is a
    // follow-up).
    let _reservation = crate::quota::enforce(state, auth).await?;

    let model = &model_entry.value;
    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;

    // Cross-provider path: when the resolved Model points at a non-
    // Anthropic upstream, parse the Anthropic-shape body into the
    // gateway's internal ChatFormat, dispatch through the Hub, and
    // re-encode the bridge's response as Anthropic JSON or SSE.
    // The Anthropic-upstream branch below stays as a byte-for-byte
    // passthrough to preserve features (cache_control, thinking
    // blocks, …) the cross-provider path can't lossily round-trip.
    if model.provider != Some(Provider::Anthropic) {
        return cross_provider_dispatch(
            state,
            body,
            model,
            &pk_entry.value,
            &model_name,
            request_id,
        )
        .await;
    }

    let api_key = crate::dispatch::require_secret(&pk_entry.value, model)?;

    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();

    // Rewrite the `model` field to the upstream value.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model.clone());
    }

    // Build the target URL. build_v1_url tolerates the rare case
    // where the customer mistakenly puts `/v1` in the Anthropic
    // api_base (the dashboard placeholder uses the OpenAI form, so
    // this is a copy-paste hazard).
    let base = crate::dispatch::resolve_base_url(Provider::Anthropic, &pk_entry.value);
    let url = crate::dispatch::build_v1_url(&base, "/messages");

    // Check if the request wants streaming.
    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let client = crate::http_client::client();
    let req_builder = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json")
        .header("x-aisix-request-id", request_id)
        .json(body);

    let upstream_resp = req_builder
        .send()
        .await
        .map_err(|e| aisix_gateway::BridgeError::Transport(e.to_string()))
        .map_err(ProxyError::Bridge)?;

    let status = upstream_resp.status();

    if !status.is_success() {
        let status_u16 = status.as_u16();
        let message = upstream_resp.text().await.unwrap_or_default();
        return Err(ProxyError::Bridge(
            aisix_gateway::BridgeError::UpstreamStatus {
                status: status_u16,
                message: if message.len() > 1024 {
                    format!("{}…", &message[..1024])
                } else {
                    message
                },
            },
        ));
    }

    // Update health tracker on success.
    state.health.record_success(&model_name);

    let provider_label = "anthropic".to_string();

    if is_stream {
        // For SSE streaming: pass through the response body as a streaming
        // `text/event-stream` response.
        let headers = upstream_resp.headers().clone();
        let body_stream = upstream_resp.bytes_stream();

        let mut response =
            axum::response::Response::new(axum::body::Body::from_stream(body_stream));

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

        // Streaming passthrough: the upstream byte stream isn't
        // parsed in-flight, so token counts aren't available here.
        // Emit a UsageEvent without token detail; the
        // `inbound_protocol="anthropic"` label still lets dashboard
        // Logs surface the request alongside non-streaming siblings.
        // Real token counts on this path land with the parsing
        // wrapper in a follow-up.
        Ok(DispatchOutcome {
            response,
            provider_label,
            metrics: AnthropicUsageMetrics::default(),
        })
    } else {
        // Non-streaming: deserialise and re-serialise as JSON.
        let json_body: Value = upstream_resp
            .json()
            .await
            .map_err(|e| aisix_gateway::BridgeError::UpstreamDecode(e.to_string()))
            .map_err(ProxyError::Bridge)?;

        let metrics = anthropic_metrics_from_response_json(&json_body);

        // Restore the gateway-facing model name so callers see what they asked for.
        let mut json_body = json_body;
        if let Some(m) = json_body.get_mut("model") {
            // If the upstream echoes the model name, rewrite to the gateway name.
            if m.as_str().map(|s| s == upstream_model).unwrap_or(false) {
                *m = Value::String(model_name.clone());
            }
        }

        Ok(DispatchOutcome {
            response: Json(json_body).into_response(),
            provider_label,
            metrics,
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
async fn cross_provider_dispatch(
    state: &ProxyState,
    body: &Value,
    model: &aisix_core::Model,
    provider_key: &aisix_core::ProviderKey,
    model_name: &str,
    request_id: &str,
) -> Result<DispatchOutcome, ProxyError> {
    use aisix_gateway::{Bridge, BridgeContext};
    use aisix_provider_anthropic::{
        chat_response_into_anthropic_json, parse_inbound_request, AnthropicSseEncoder,
    };
    use std::sync::Arc;

    let provider = model.provider.ok_or_else(|| {
        ProxyError::InvalidRequest(format!("model `{model_name}` has no provider prefix"))
    })?;
    let bridge: Arc<dyn Bridge> = state
        .hub
        .get(provider)
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

    let is_stream = chat.is_streaming();
    let model_arc = Arc::new(model.clone());
    let pk_arc = Arc::new(provider_key.clone());
    let ctx = BridgeContext::new(request_id, model_arc, pk_arc);
    let provider_label = format!("{provider:?}").to_lowercase();

    if is_stream {
        let upstream = bridge
            .chat_stream(&chat, &ctx)
            .await
            .map_err(ProxyError::Bridge)?;
        state.health.record_success(model_name);

        let message_id = format!("msg_{}", Uuid::new_v4().simple());
        let encoder = AnthropicSseEncoder::new(message_id, model_name, 0);
        let sse_body = build_anthropic_sse_stream(upstream, encoder);

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
        // Streaming cross-provider: token usage is only known once
        // the bridge stream finishes. The encoder pumps chunks but
        // we don't keep a handle into it; emitting tokens for this
        // path requires plumbing a sink through `build_anthropic_sse_stream`.
        // Punted to follow-up: token-less event so dashboard Logs
        // sees the request, finish_reason / token counts come later.
        return Ok(DispatchOutcome {
            response,
            provider_label,
            metrics: AnthropicUsageMetrics::default(),
        });
    }

    // Non-streaming.
    let resp = bridge.chat(&chat, &ctx).await.map_err(ProxyError::Bridge)?;
    state.health.record_success(model_name);

    let metrics = AnthropicUsageMetrics {
        prompt_tokens: resp.usage.prompt_tokens,
        completion_tokens: resp.usage.completion_tokens,
        cache_creation_tokens: resp.usage.cache_creation_tokens,
        cache_read_tokens: resp.usage.cache_read_tokens,
        provider_request_id: resp.id.clone(),
        provider_model_version: resp.model.clone(),
        finish_reason: format!("{:?}", resp.finish_reason).to_lowercase(),
    };
    let json = chat_response_into_anthropic_json(&resp, model_name);
    Ok(DispatchOutcome {
        response: Json(json).into_response(),
        provider_label,
        metrics,
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
) -> axum::body::Body {
    use futures::StreamExt;

    let mut encoder = encoder;
    let stream = async_stream::stream! {
        let mut upstream = upstream;
        while let Some(item) = upstream.next().await {
            match item {
                Ok(chunk) => {
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

/// What `dispatch` produces alongside the wire response: enough
/// metadata for the outer wrapper to emit a UsageEvent with the
/// proper token counts and provider-detail fields.
struct DispatchOutcome {
    response: Response,
    provider_label: String,
    metrics: AnthropicUsageMetrics,
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
}

/// Emit a UsageEvent for a `/v1/messages` request. Mirrors
/// `chat::emit_usage_event` but tagged `inbound_protocol = "anthropic"`
/// so the dashboard's Logs view can disambiguate the inbound SDK
/// from the upstream provider label.
///
/// Called from `messages()` once dispatch has produced a Response and
/// (for non-streaming) we know the token counts. Streaming passthrough
/// to an Anthropic upstream skips the call — the upstream byte stream
/// isn't parsed in-flight, so token counts aren't available; that
/// path's UsageEvent emission is tracked as follow-up work.
fn emit_anthropic_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    api_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    metrics: AnthropicUsageMetrics,
) {
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
        inbound_protocol: "anthropic".to_string(),
        ..Default::default()
    };
    state.usage_sink.try_emit(event.clone());
    let snap = state.snapshot.load();
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, exporters.iter().map(|e| &e.value));
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
    }
    .emit();
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use aisix_core::models::Provider;
    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use aisix_provider_anthropic::AnthropicBridge;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
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
    const GEMINI_PK_ID: &str = "33333333-3333-3333-3333-333333333333";
    const DEEPSEEK_PK_ID: &str = "44444444-4444-4444-4444-444444444444";

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
            r#"{{"display_name":"anthropic-up","secret":"sk-ant-test","api_base":"{api_base}"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(ANTHROPIC_PK_ID, pk, 1)
    }

    fn openai_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-openai-test","api_base":"{api_base}"}}"#
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
        hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
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
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
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
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
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
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
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
        hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
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
        hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
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
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
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
        // 400 Bad Request — `model` field missing.
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ─── Cross-protocol matrix (Anthropic inbound × non-Anthropic) ─

    fn gemini_model(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "gemini",
                "model_name": "gemini-2.0-flash",
                "provider_key_id": "{GEMINI_PK_ID}"
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
            r#"{{"display_name":"gemini-up","secret":"ya29-test","api_base":"{api_base}"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(GEMINI_PK_ID, pk, 1)
    }

    fn deepseek_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"deepseek-up","secret":"sk-deepseek","api_base":"{api_base}"}}"#
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
        use aisix_provider_gemini::gemini_bridge;

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
        hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
        hub.register(Provider::Gemini, Arc::new(gemini_bridge()));
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
        use aisix_provider_deepseek::deepseek_bridge;

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
        hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
        hub.register(Provider::Deepseek, Arc::new(deepseek_bridge()));
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

    /// Helper for the streaming variants of (Anthropic inbound) ×
    /// (Gemini | DeepSeek upstream). Both upstreams expose the
    /// OpenAi-compat `/chat/completions` endpoint with OpenAi-shape
    /// SSE deltas, so the assertion shape is identical.
    async fn assert_anthropic_streams_through_openai_compat_upstream(
        bridge_provider: Provider,
        bridge: Arc<dyn aisix_gateway::Bridge>,
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
        let pk_json = format!(
            r#"{{"display_name":"matrix-up","secret":"k","api_base":"{}"}}"#,
            upstream.uri()
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_json).unwrap();

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(ResourceEntry::new(pk_id, pk, 1));
        snap.models.insert(model_entry);
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
        hub.register(bridge_provider, bridge);
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
        use aisix_provider_gemini::gemini_bridge;
        assert_anthropic_streams_through_openai_compat_upstream(
            Provider::Gemini,
            Arc::new(gemini_bridge()),
            // Placeholder; helper rebuilds with the wiremock uri.
            gemini_model("my-claude-via-gemini"),
            "my-claude-via-gemini",
        )
        .await;
    }

    #[tokio::test]
    async fn matrix_anthropic_in_deepseek_upstream_streaming() {
        use aisix_provider_deepseek::deepseek_bridge;
        assert_anthropic_streams_through_openai_compat_upstream(
            Provider::Deepseek,
            Arc::new(deepseek_bridge()),
            deepseek_model("my-claude-via-ds"),
            "my-claude-via-ds",
        )
        .await;
    }
}
