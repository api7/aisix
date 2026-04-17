//! aisix-proxy — client-facing proxy router (`:3000`).
//!
//! Mounts the OpenAI-compatible surface:
//! - `GET  /health`
//! - `POST /v1/chat/completions` (streaming + non-streaming)
//!
//! Handlers run behind the [`AuthenticatedKey`] extractor which reads
//! the Bearer token (or `x-api-key` fallback) and looks the key up in
//! the current [`AisixSnapshot`]. Model authorisation is enforced per
//! request against `ApiKey::allowed_models`. Upstream calls are
//! dispatched through the [`aisix_gateway::Hub`] to the registered
//! `Bridge` for the Model's provider.
//!
//! Errors surface as OpenAI-style envelopes:
//!
//! ```json
//! {"error":{"message":"…","type":"…"}}
//! ```
//!
//! Status codes follow [`crate::error::ProxyError::status`] — spec §3 auth
//! rules (401/403), `Bridge` mapping preserves upstream 4xx and collapses
//! upstream 5xx to 502.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod auth;
mod chat;
mod error;
mod render;
mod state;

pub use auth::AuthenticatedKey;
pub use error::{ErrorEnvelope, ProxyError};
pub use state::ProxyState;

use axum::routing::{get, post};
use axum::{http::StatusCode, Json, Router};
use serde_json::json;

/// Build the proxy router. Mounts `/health` plus the
/// OpenAI-compatible chat-completions surface.
pub fn build_router(state: ProxyState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .with_state(state)
}

async fn health(
    axum::extract::State(state): axum::extract::State<ProxyState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let snap = state.snapshot.load();
    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "models": snap.models.len(),
            "apikeys": snap.apikeys.len(),
            "providers": state.hub.len(),
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::models::Provider;
    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::{Hub, SseDecoder, SseEvent};
    use aisix_provider_openai::OpenAiBridge;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use futures::StreamExt;
    use std::sync::Arc;
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            tls: None,
        }
    }

    fn build_state(snapshot: AisixSnapshot, hub: Arc<Hub>) -> ProxyState {
        let handle = SnapshotHandle::new(snapshot);
        ProxyState::new(handle, hub, &cfg())
    }

    fn model_entry(name: &str, api_base: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "name": "{name}",
                "model": "openai/gpt-4o",
                "provider_config": {{"api_key": "sk-upstream", "api_base": "{api_base}"}}
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new("model-id-1", model, 1)
    }

    fn apikey_entry(key: &str, allowed: &[&str]) -> ResourceEntry<ApiKey> {
        let allowed_json = serde_json::to_string(&allowed).unwrap();
        let cfg = format!(r#"{{"key": "{key}", "allowed_models": {allowed_json}}}"#);
        let apikey: ApiKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new("key-id-1", apikey, 1)
    }

    fn seed_snapshot(model: &str, allowed: &[&str], api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.models.insert(model_entry(model, api_base));
        snap.apikeys.insert(apikey_entry("sk-caller", allowed));
        snap
    }

    async fn run(app: Router, req: Request<Body>) -> axum::http::Response<Body> {
        app.oneshot(req).await.unwrap()
    }

    #[tokio::test]
    async fn non_streaming_happy_path_returns_openai_shaped_json() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-upstream",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hi");
        assert_eq!(v["usage"]["total_tokens"], 3);
    }

    #[tokio::test]
    async fn missing_authorization_returns_401_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_api_key");
    }

    #[tokio::test]
    async fn unknown_api_key_returns_401() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-does-not-exist")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn model_not_in_allowed_list_returns_403() {
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        // ApiKey allows only "other-model", the caller asks for "my-gpt4".
        let snap = seed_snapshot("my-gpt4", &["other-model"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "permission_denied");
    }

    #[tokio::test]
    async fn unknown_model_returns_404_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["*"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "model_not_found");
    }

    #[tokio::test]
    async fn empty_messages_returns_400() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn upstream_429_passes_through_with_openai_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "upstream_error");
    }

    #[tokio::test]
    async fn provider_without_registered_bridge_returns_503() {
        // Snapshot has a Model targeting openai, but the Hub is empty.
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn streaming_response_emits_sse_then_done_sentinel() {
        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
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

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .contains("text/event-stream"));

        // Drain the body, decode SSE events, assert we got at least one
        // delta chunk plus the terminating [DONE].
        let mut body_stream = resp.into_body().into_data_stream();
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            let bytes = chunk.unwrap();
            events.extend(decoder.feed(bytes.as_ref()));
        }
        assert!(events.contains(&SseEvent::Done), "missing [DONE] sentinel");
        let data_count = events
            .iter()
            .filter(|e| matches!(e, SseEvent::Data(_)))
            .count();
        assert!(
            data_count >= 2,
            "expected at least two chat chunks, got {data_count}"
        );
    }
}
