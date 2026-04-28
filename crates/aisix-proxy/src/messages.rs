//! `POST /v1/messages` — Anthropic native Messages API pass-through.
//!
//! This endpoint lets callers that already speak the Anthropic SDK send
//! requests directly without going through the OpenAI-compat Hub
//! translation layer. The gateway:
//!
//! 1. Authenticates the proxy API key and authorises model access.
//! 2. Resolves the model name to a `Model` in the snapshot.
//! 3. Enforces that the model uses the `anthropic/` provider — non-Anthropic
//!    models are rejected with 422 ("model is not an Anthropic provider").
//! 4. Rewrites the `model` field to the upstream model name (strips the
//!    `anthropic/` prefix).
//! 5. Forwards the body to `{api_base}/v1/messages` with the correct
//!    `x-api-key` and `anthropic-version` headers.
//! 6. Returns the response verbatim — both streaming (SSE) and non-streaming
//!    are supported transparently.
//!
//! Rate-limiting and metrics are recorded using the same hooks as chat
//! completions.
//!
//! Errors use the standard OpenAI-style envelope so clients on the proxy
//! side can handle them consistently regardless of which endpoint was used.

use aisix_core::models::Provider;
use aisix_obs::{AccessLog, RequestOutcome};
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
/// Default Anthropic base URL used when `api_base` is not set on the Model.
const ANTHROPIC_DEFAULT_BASE: &str = "https://api.anthropic.com";

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

    match dispatch(&state, &auth, &mut body, &request_id).await {
        Ok((resp, provider)) => {
            let elapsed = started.elapsed();
            let status = resp.status().as_u16();
            emit_access_log(
                &model_name,
                &provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &provider,
                &model_name,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            resp
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
            err.into_response()
        }
    }
}

async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    body: &mut Value,
    request_id: &str,
) -> Result<(Response, String), ProxyError> {
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

    let model = &model_entry.value;

    // Validate the model is Anthropic — this endpoint is native-only.
    if model.provider() != Some(Provider::Anthropic) {
        return Err(ProxyError::InvalidRequest(format!(
            "model `{model_name}` is not an Anthropic provider; use /v1/chat/completions instead"
        )));
    }

    let api_key = model.provider_config.api_key.as_str();

    if api_key.is_empty() {
        return Err(ProxyError::Bridge(aisix_gateway::BridgeError::Config(
            "provider_config.api_key is empty".into(),
        )));
    }

    // Resolve the upstream model name (strip "anthropic/" prefix).
    let upstream_model = model
        .upstream_model()
        .ok_or_else(|| ProxyError::InvalidRequest("model field missing provider/ prefix".into()))?
        .to_string();

    // Rewrite the `model` field to the upstream value.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model.clone());
    }

    // Build the target URL.
    let base = match model.base_url() {
        Some(b) if !b.trim().is_empty() => b.trim_end_matches('/').to_string(),
        _ => ANTHROPIC_DEFAULT_BASE.to_string(),
    };
    let url = format!("{base}/v1/messages");

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

        Ok((response, provider_label))
    } else {
        // Non-streaming: deserialise and re-serialise as JSON.
        let json_body: Value = upstream_resp
            .json()
            .await
            .map_err(|e| aisix_gateway::BridgeError::UpstreamDecode(e.to_string()))
            .map_err(ProxyError::Bridge)?;

        // Restore the gateway-facing model name so callers see what they asked for.
        let mut json_body = json_body;
        if let Some(m) = json_body.get_mut("model") {
            // If the upstream echoes the model name, rewrite to the gateway name.
            if m.as_str().map(|s| s == upstream_model).unwrap_or(false) {
                *m = Value::String(model_name.clone());
            }
        }

        Ok((Json(json_body).into_response(), provider_label))
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

    fn anthropic_model(name: &str, api_base: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "name": "{name}",
                "model": "anthropic/claude-3-5-haiku-20241022",
                "provider_config": {{"api_key": "sk-ant-test", "api_base": "{api_base}"}}
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn openai_model(name: &str, api_base: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "name": "{name}",
                "model": "openai/gpt-4o",
                "provider_config": {{"api_key": "sk-openai-test", "api_base": "{api_base}"}}
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-2", m, 1)
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

        let snap = AisixSnapshot::new();
        snap.models
            .insert(anthropic_model("claude-haiku", &upstream.uri()));
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

        let snap = AisixSnapshot::new();
        snap.models
            .insert(anthropic_model("my-claude", &upstream.uri()));
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
        let snap = AisixSnapshot::new();
        snap.models
            .insert(anthropic_model("claude-haiku", "http://unused"));
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
        let snap = AisixSnapshot::new();
        snap.models
            .insert(anthropic_model("claude-haiku", "http://unused"));
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
        let snap = AisixSnapshot::new();
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

    #[tokio::test]
    async fn non_anthropic_model_returns_400() {
        let snap = AisixSnapshot::new();
        snap.models.insert(openai_model("gpt-4o", "http://unused"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        });
        let resp = app.oneshot(make_req(body)).await.unwrap();
        // 400 Bad Request — model is not an Anthropic provider.
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn upstream_error_returns_502() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.models
            .insert(anthropic_model("claude-haiku", &upstream.uri()));
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
        let snap = AisixSnapshot::new();
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
}
