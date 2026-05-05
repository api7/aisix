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

mod audio;
mod auth;
pub mod budget;
mod chat;
mod completions;
mod embeddings;
mod error;
pub mod health;
mod http_client;
mod images;
mod messages;
mod models;
mod passthrough;
mod render;
mod rerank;
mod responses;
mod routing;
mod state;

pub use auth::AuthenticatedKey;
pub use error::{ErrorEnvelope, ProxyError};
pub use health::HealthTracker;
pub use state::ProxyState;

use axum::routing::{any, get, post};
use axum::{http::StatusCode, Json, Router};
use serde_json::json;

/// Build the proxy router. Mounts `/health` plus the
/// OpenAI-compatible proxy surface.
pub fn build_router(state: ProxyState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/embeddings", post(embeddings::embeddings))
        .route("/v1/images/generations", post(images::image_generations))
        .route("/v1/messages", post(messages::messages))
        .route("/v1/rerank", post(rerank::rerank))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/audio/transcriptions", post(audio::transcriptions))
        .route("/v1/audio/translations", post(audio::translations))
        .route("/v1/audio/speech", post(audio::speech))
        .route(
            "/passthrough/:provider/*rest",
            any(passthrough::passthrough),
        )
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

    /// State used by the *existing* tests — cache disabled so the
    /// rate-limit / wiremock cases still see every request reach the
    /// upstream. Cache-specific tests build state with the default
    /// constructor (which keeps caching on) instead.
    fn build_state(snapshot: AisixSnapshot, hub: Arc<Hub>) -> ProxyState {
        let handle = SnapshotHandle::new(snapshot);
        ProxyState::new(handle, hub, &cfg()).without_cache()
    }

    fn build_state_with_cache(snapshot: AisixSnapshot, hub: Arc<Hub>) -> ProxyState {
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
        apikey_entry_with_limits(key, allowed, None)
    }

    fn apikey_entry_with_limits(
        key: &str,
        allowed: &[&str],
        rate_limit: Option<serde_json::Value>,
    ) -> ResourceEntry<ApiKey> {
        let allowed_json = serde_json::to_string(&allowed).unwrap();
        let rl_tail = match rate_limit {
            Some(v) => format!(", \"rate_limit\": {v}"),
            None => String::new(),
        };
        // Tests pass the plaintext bearer here (e.g. "sk-caller"); the
        // wire schema stores its SHA-256 (§9A.7B.4). Hash via the
        // canonical helper so request-side `Bearer <plaintext>` lookups
        // line up.
        let key_hash = ApiKey::hash_bearer(key);
        let cfg =
            format!(r#"{{"key_hash": "{key_hash}", "allowed_models": {allowed_json}{rl_tail}}}"#);
        let apikey: ApiKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new("key-id-1", apikey, 1)
    }

    fn seed_snapshot(model: &str, allowed: &[&str], api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.models.insert(model_entry(model, api_base));
        snap.apikeys.insert(apikey_entry("sk-caller", allowed));
        snap
    }

    /// Insert a default-enabled cache policy on the snapshot so the
    /// proxy's cache gate (chat::dispatch) opens the lookup path.
    /// Stage 2 honors existence + `enabled`; Stage 3 honors
    /// `applies_to`. The default `applies_to=all` (set by serde
    /// when omitted) matches every request, so existing tests that
    /// seed a bare policy keep passing.
    fn seed_cache_policy(snap: &AisixSnapshot, name: &str) {
        seed_cache_policy_with_applies_to(snap, name, "all");
    }

    /// Like `seed_cache_policy` but with a specific `applies_to`
    /// clause — used by the Stage 3 tests that pin the matcher's
    /// behaviour on `model:<name>` / `api_key:<id>`.
    fn seed_cache_policy_with_applies_to(snap: &AisixSnapshot, name: &str, applies_to: &str) {
        let cfg =
            format!(r#"{{"name": "{name}", "backend": "memory", "applies_to": "{applies_to}"}}"#,);
        let policy: aisix_core::models::CachePolicy = serde_json::from_str(&cfg).unwrap();
        snap.cache_policies
            .insert(ResourceEntry::new(format!("cp-id-{name}"), policy, 1));
    }

    fn seed_snapshot_with_limits(
        model: &str,
        allowed: &[&str],
        api_base: &str,
        rate_limit: serde_json::Value,
    ) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.models.insert(model_entry(model, api_base));
        snap.apikeys.insert(apikey_entry_with_limits(
            "sk-caller",
            allowed,
            Some(rate_limit),
        ));
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

    #[tokio::test]
    async fn rate_limit_rpm_returns_429_with_retry_after_header() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 1}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request succeeds.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Second request within the same minute trips rpm=1 → 429.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let retry = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .expect("missing or malformed retry-after header");
        assert!(retry >= 1);
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn rate_limit_tpm_blocks_after_token_commit_exhausts_window() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }],
                // Deliberately overshoot the TPM cap so the next
                // pre_commit observes an exhausted window.
                "usage": {"prompt_tokens": 10_000, "completion_tokens": 10_000, "total_tokens": 20_000}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"tpm": 1_000}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request goes through (pre-commit TPM is unchecked for an
        // empty bucket); usage counted at post-deduct overshoots the cap.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Second request sees TPM > 1000 and rejects.
        let resp = run(build_router(state), make_req()).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn request_lifecycle_increments_metrics_counters() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());

        let state = build_state(snap, hub);
        let metrics = state.metrics.clone();
        let app = build_router(state);

        // Pre-flight: counter family is absent until something writes.
        assert!(!metrics.render().contains("aisix_requests_total"));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
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

        let rendered = metrics.render();
        assert!(rendered.contains("aisix_requests_total"));
        assert!(rendered.contains("provider=\"openai\""));
        assert!(rendered.contains("outcome=\"success\""));
        assert!(rendered.contains("aisix_tokens_consumed_total"));
        // 7 tokens were committed.
        assert!(
            rendered.contains("7"),
            "expected tokens counter at 7:\n{rendered}"
        );
    }

    #[tokio::test]
    async fn ratelimit_rejection_increments_ratelimit_counter() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 1}),
        );
        let state = build_state(snap, hub);
        let metrics = state.metrics.clone();
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let _ = run(build_router(state.clone()), make_req()).await;
        let resp = run(build_router(state), make_req()).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        let rendered = metrics.render();
        assert!(rendered.contains("aisix_ratelimit_rejections_total"));
        assert!(rendered.contains("scope=\"requests\""));
    }

    #[tokio::test]
    async fn cache_hit_short_circuits_upstream_and_sets_header() {
        // Wiremock that *only* satisfies one upstream call. If the cache
        // ever lets a second request through, the test fails with a 500.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cached"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1) // hard expectation: exactly one upstream hit
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        // Cache gate opens only when an enabled policy exists in
        // snapshot. Without this seed step the test would 200 but
        // the cache header would be absent (policy-disabled path).
        seed_cache_policy(&snap, "test-cache");
        // Cache enabled — uses the default constructor.
        let state = build_state_with_cache(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request — miss.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("miss"),
        );

        // Second identical request — hit.
        let resp = run(build_router(state), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("hit"),
        );
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "cached");
    }

    /// Regression for #88. On a cache hit the DP must surface the
    /// cached response's prompt + completion tokens on a dedicated
    /// `cache_hit_saved_*` pair so cp-api can multiply by its pricing
    /// catalog server-side and report `cost_saved_usd` on `/usage`.
    /// Miss rows must keep the saved counters at zero.
    #[tokio::test]
    async fn cache_hit_emits_saved_token_counters_on_telemetry_event() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cached"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy(&snap, "test-cache");
        let state = build_state_with_cache(snap, hub).with_usage_sink(UsageSink::new(tx));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // Miss: saved counters must be zero (the request paid the upstream).
        let _ = run(build_router(state.clone()), make_req()).await;
        let miss_event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("miss event was never emitted")
            .expect("sender dropped");
        assert_eq!(miss_event.cache_status, "miss");
        assert_eq!(miss_event.cache_hit_saved_input_tokens, 0);
        assert_eq!(miss_event.cache_hit_saved_output_tokens, 0);

        // Hit: saved counters must mirror the cached response's usage.
        let _ = run(build_router(state), make_req()).await;
        let hit_event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("hit event was never emitted")
            .expect("sender dropped");
        assert_eq!(hit_event.cache_status, "hit");
        assert_eq!(hit_event.cache_hit_saved_input_tokens, 7);
        assert_eq!(hit_event.cache_hit_saved_output_tokens, 11);
        // `prompt_tokens` keeps mirroring the cached usage too — the
        // existing dashboard rollups stay correct. The new field is
        // additive, not a substitute.
        assert_eq!(hit_event.prompt_tokens, 7);
        assert_eq!(hit_event.completion_tokens, 11);
    }

    #[tokio::test]
    async fn cache_miss_when_request_payload_differs() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            // Two distinct payloads → expect exactly two upstream calls.
            .expect(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy(&snap, "test-cache");
        let state = build_state_with_cache(snap, hub);

        let body_a = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "first"}]
        });
        let body_b = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "second"}]
        });
        let mk = |body: &serde_json::Value| {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let r1 = run(build_router(state.clone()), mk(&body_a)).await;
        let r2 = run(build_router(state), mk(&body_b)).await;
        for r in [r1, r2] {
            assert_eq!(r.status(), StatusCode::OK);
            assert_eq!(
                r.headers()
                    .get("x-aisix-cache")
                    .and_then(|v| v.to_str().ok()),
                Some("miss"),
            );
        }
    }

    #[tokio::test]
    async fn applies_to_model_does_not_cache_unmatched_model() {
        // Stage 3 contract: a `cache_policy` with
        // `applies_to = "model:<other>"` must NOT enable the cache for
        // requests targeting a different model. Three identical
        // requests should hit the upstream three times — none of them
        // gets cached.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "always-fresh"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(3) // each call must reach the upstream
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        // The api key + model are named "my-gpt4"; the policy below
        // pins applies_to to a different model name so no request in
        // this test matches.
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy_with_applies_to(&snap, "scoped", "model:not-my-gpt4");
        let state = build_state_with_cache(snap, hub);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // All three calls: 200, no `x-aisix-cache` header (policy gate
        // closed for this model). The wiremock `.expect(3)` above is
        // the load-bearing assertion — it fails the test at server
        // teardown if any call short-circuited via the cache.
        for _ in 0..3 {
            let resp = run(build_router(state.clone()), make_req()).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert!(
                resp.headers().get("x-aisix-cache").is_none(),
                "policy-gate-closed responses must not carry x-aisix-cache",
            );
        }
    }

    #[tokio::test]
    async fn applies_to_model_caches_matched_model() {
        // Counterpart to the negative test above: when the policy
        // `applies_to` matches the request's model, the cache gate
        // opens and the second identical request hits.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "matched"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1) // only one upstream hit; second call must come from cache
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy_with_applies_to(&snap, "scoped", "model:my-gpt4");
        let state = build_state_with_cache(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let r1 = run(build_router(state.clone()), make_req()).await;
        assert_eq!(r1.status(), StatusCode::OK);
        assert_eq!(
            r1.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("miss"),
        );

        let r2 = run(build_router(state), make_req()).await;
        assert_eq!(r2.status(), StatusCode::OK);
        assert_eq!(
            r2.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("hit"),
        );
    }

    /// Build a `ResourceEntry<Model>` with a non-default id so the test
    /// can mount multiple Models in one snapshot.
    fn model_entry_with_id(id: &str, name: &str, api_base: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "name": "{name}",
                "model": "openai/gpt-4o",
                "provider_config": {{"api_key": "sk-upstream", "api_base": "{api_base}"}}
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, model, 1)
    }

    /// Build a virtual routing Model that points at `targets` (other
    /// Model.name values) using the given strategy.
    fn routing_entry(name: &str, strategy: &str, targets: &[&str]) -> ResourceEntry<Model> {
        let target_objs: Vec<serde_json::Value> = targets
            .iter()
            .map(|t| serde_json::json!({"model": t}))
            .collect();
        let cfg = serde_json::json!({
            "name": name,
            "model": format!("router/{name}"),
            // The provider_config is required by the schema but unused
            // by the proxy because routing.is_some() short-circuits.
            "provider_config": {"api_key": "ignored"},
            "routing": {
                "strategy": strategy,
                "targets": target_objs,
            }
        });
        let model: Model = serde_json::from_value(cfg).unwrap();
        ResourceEntry::new(format!("router-{name}"), model, 1)
    }

    #[tokio::test]
    async fn routing_failover_retries_to_second_target_when_first_5xxs() {
        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .mount(&bad_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "fallback worked"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));

        let snap = AisixSnapshot::new();
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", &bad_upstream.uri()));
        snap.models.insert(model_entry_with_id(
            "m-good",
            "secondary",
            &good_upstream.uri(),
        ));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
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
        assert_eq!(v["choices"][0]["message"]["content"], "fallback worked");
    }

    #[tokio::test]
    async fn routing_propagates_4xx_without_attempting_fallback() {
        // First target returns 400 — caller mistake, no point trying
        // the second target. We assert the request fails 400 *and* the
        // second wiremock never sees a request.
        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string("invalid_request"))
            .expect(1)
            .mount(&bad_upstream)
            .await;

        let standby_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            // Should never be hit; expect(0) enforces it on Drop.
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&standby_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));

        let snap = AisixSnapshot::new();
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", &bad_upstream.uri()));
        snap.models.insert(model_entry_with_id(
            "m-standby",
            "secondary",
            &standby_upstream.uri(),
        ));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn routing_to_missing_target_returns_400() {
        // Routing references a Model that isn't in the snapshot — this
        // is a misconfiguration and should surface as a clean 400.
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));

        let snap = AisixSnapshot::new();
        snap.models
            .insert(routing_entry("smart", "failover", &["nonexistent"]));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn ratelimit_response_headers_are_injected_on_success() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 100, "tpm": 50000}),
        );
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
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

        let headers = resp.headers();
        assert!(
            headers.contains_key("x-ratelimit-limit-requests"),
            "missing x-ratelimit-limit-requests"
        );
        assert_eq!(
            headers
                .get("x-ratelimit-limit-requests")
                .and_then(|v| v.to_str().ok()),
            Some("100"),
        );
        assert!(
            headers.contains_key("x-ratelimit-limit-tokens"),
            "missing x-ratelimit-limit-tokens"
        );
        assert_eq!(
            headers
                .get("x-ratelimit-limit-tokens")
                .and_then(|v| v.to_str().ok()),
            Some("50000"),
        );
        // Remaining should be limit - 1 (one request consumed).
        assert_eq!(
            headers
                .get("x-ratelimit-remaining-requests")
                .and_then(|v| v.to_str().ok()),
            Some("99"),
        );
    }

    #[tokio::test]
    async fn input_guardrail_block_returns_422_and_skips_upstream() {
        use aisix_guardrails::{GuardrailChain, KeywordBlocklist, KeywordRule};

        // wiremock that fails the test if it's hit at all.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0) // hard expectation — guardrail must short-circuit
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());

        let guardrails = Arc::new(GuardrailChain::new(vec![Arc::new(KeywordBlocklist::new(
            vec![KeywordRule::literal("forbidden-token")],
        ))]));
        let state = build_state(snap, hub).with_guardrails(guardrails);
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "say the forbidden-token please"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("forbidden-token"));
    }

    /// Regression: a guardrail-blocked request must record the resolved
    /// model_id on its telemetry event. Earlier the error path hard-coded
    /// model_id="" for every failure, which left the dashboard /logs
    /// "Guardrail blocks" tab showing an empty model column.
    #[tokio::test]
    async fn input_guardrail_block_records_resolved_model_id_in_telemetry() {
        use aisix_guardrails::{GuardrailChain, KeywordBlocklist, KeywordRule};
        use aisix_obs::UsageSink;

        // Capturing usage sink — we read the emitted event off the
        // receiver to assert telemetry shape, not just the HTTP response.
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let guardrails = Arc::new(GuardrailChain::new(vec![Arc::new(KeywordBlocklist::new(
            vec![KeywordRule::literal("forbidden-token")],
        ))]));
        let state = build_state(snap, hub)
            .with_guardrails(guardrails)
            .with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "say the forbidden-token please"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        // The seeded model_entry uses the literal id "model-id-1"
        // (see lib.rs::model_entry). Pinning the exact value catches
        // regressions where the id silently becomes empty.
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        assert_eq!(event.model_id, "model-id-1");
        assert_eq!(event.status_code, 422);
        assert!(event.guardrail_blocked);
    }

    #[tokio::test]
    async fn output_guardrail_block_returns_422_after_upstream_runs() {
        use aisix_guardrails::{GuardrailChain, KeywordBlocklist, KeywordRule};

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "here is your secret-string"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1) // upstream IS called; guardrail blocks the response
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());

        let guardrails = Arc::new(GuardrailChain::new(vec![Arc::new(
            KeywordBlocklist::output_only(vec![KeywordRule::literal("secret-string")]),
        )]));
        let state = build_state(snap, hub).with_guardrails(guardrails);
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "anything"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    #[tokio::test]
    async fn budget_exceeded_returns_429() {
        use crate::budget::BudgetClient;

        // cp-api stand-in: returns a deny decision for our key.
        // Wire shape mirrors cp-api's budgetCheckResponse — see
        // internal/cpapi/resources/budget_check.go (prd-09b rev 2 §5.5).
        let cp = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/dp/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "allow": false,
                "fail_mode": "closed",
                "reason": {
                    "type": "billing_error",
                    "code": "budget_exceeded",
                    "message": "monthly cap exceeded",
                    "scope": "api_key",
                    "scope_ref": "ak-uuid",
                    "limit_usd": "10.00",
                    "spent_usd": "10.50",
                    "period": "month",
                    "period_resets_at": "2026-05-01T00:00:00Z",
                    "retry_after_seconds": 86400
                }
            })))
            .mount(&cp)
            .await;

        // Upstream chat endpoint must NOT be hit — the budget check
        // fires before dispatch.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));

        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_budget_client(Arc::new(BudgetClient::new(
            cp.uri(),
            reqwest::Client::new(),
        )));

        let app = build_router(state);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "billing_error");
        assert_eq!(v["error"]["code"], "budget_exceeded");
    }
}
