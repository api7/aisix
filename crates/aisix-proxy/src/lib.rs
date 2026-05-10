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
mod dispatch;
mod embeddings;
mod error;
pub mod health;
mod http_client;
mod images;
mod messages;
mod models;
mod passthrough;
mod quota;
mod render;
mod rerank;
mod responses;
mod routing;
mod state;

pub use auth::AuthenticatedKey;
pub use error::{ErrorEnvelope, ProxyError};
pub use health::HealthTracker;
pub use state::ProxyState;

use axum::extract::State;
use axum::http::Request;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{http::StatusCode, Json, Router};
use serde_json::json;

/// Build the proxy router. Mounts `/health` plus the
/// OpenAI-compatible proxy surface.
pub fn build_router(state: ProxyState) -> Router {
    let body_limit = state.request_body_limit_bytes;
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
        // Wire the configured cap into axum's request-body extractor
        // chain (`Json<T>` defers to `Bytes` which honors this layer).
        // Without this, axum 0.7's `DefaultBodyLimit` falls back to
        // its built-in 2 MiB default, which silently rejects bodies
        // in the 2 MiB-to-cap band with a stock `BytesRejection`
        // response (NOT the OpenAI envelope). The middleware below
        // catches the Content-Length-known case ahead of the
        // extractor; this layer catches chunked / size-mismatched
        // bodies once their actual byte count exceeds the cap.
        .layer(axum::extract::DefaultBodyLimit::max(body_limit))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            enforce_request_body_limit,
        ))
        .with_state(state)
}

/// Per RFC 9110 §15.5.14, a request body that exceeds the gateway's
/// configured `request_body_limit_bytes` must surface as a clean
/// `413 Content Too Large` response — NOT an `ECONNRESET` from a
/// mid-write socket close. This middleware inspects the inbound
/// `Content-Length` header before any handler runs and short-circuits
/// with the OpenAI-shape error envelope when the declared size
/// exceeds the cap.
///
/// Bodies sent with chunked transfer encoding (no Content-Length)
/// fall through to handler-level body extraction, which still
/// enforces the limit but with the slower fail mode (the read errors
/// once the cap is hit). Catching the Content-Length-known case here
/// is the load-bearing user-visible win: the OpenAI Node SDK and
/// `fetch` both set Content-Length for non-streamed POSTs, and
/// without this middleware they see ECONNRESET (indistinguishable
/// from a network failure or a gateway crash) instead of 413.
async fn enforce_request_body_limit(
    State(state): State<ProxyState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // RFC 9110 §8.6 — a server SHOULD reject a request that carries
    // duplicate or conflicting `Content-Length` values rather than
    // act on the first one (which is a request-smuggling vector).
    let mut content_lengths = request
        .headers()
        .get_all(axum::http::header::CONTENT_LENGTH)
        .iter();
    let first = content_lengths.next();
    if content_lengths.next().is_some() {
        return ProxyError::InvalidRequest("conflicting Content-Length headers".into())
            .into_response();
    }
    if let Some(declared) = first
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())
    {
        if declared > state.request_body_limit_bytes {
            return ProxyError::RequestTooLarge {
                limit_bytes: state.request_body_limit_bytes,
            }
            .into_response();
        }
    }
    next.run(request).await
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
    use wiremock::matchers::{header, method, path};
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

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new("model-id-1", model, 1)
    }

    fn provider_key_entry(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"openai-up","secret":"sk-upstream","api_base":"{api_base}"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(provider_key_entry(api_base));
        snap
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
        let snap = new_snap(api_base);
        snap.models.insert(model_entry(model));
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

    /// Disabled-policy seeder for #154 regression coverage. Posts a
    /// `CachePolicy{enabled: false, applies_to: "all"}` so the
    /// cache-gate predicate at chat.rs (`entry.value.enabled && ...`)
    /// must skip it.
    fn seed_cache_policy_disabled(snap: &AisixSnapshot, name: &str) {
        let cfg = format!(
            r#"{{"name": "{name}", "backend": "memory", "applies_to": "all", "enabled": false}}"#,
        );
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
        let snap = new_snap(api_base);
        snap.models.insert(model_entry(model));
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

    /// Issue #159: a request body whose declared `Content-Length`
    /// exceeds the configured cap must surface as `413 Content Too
    /// Large` per RFC 9110 §15.5.14, NOT as `ECONNRESET` from a
    /// mid-write socket close. Regression: the handler-level body
    /// extractor's overflow path was racing the client write,
    /// surfacing as a network failure indistinguishable from a
    /// gateway crash. The new middleware short-circuits on the
    /// declared size before any handler runs.
    #[tokio::test]
    async fn oversize_body_returns_413_envelope_with_content_length_check() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        // Declare a Content-Length well over the test cfg's 1 MiB cap
        // but ship a tiny actual body — the middleware MUST reject
        // based on the declared size alone, before reading the body.
        // (A real caller's `JSON.stringify` would set Content-Length
        // matching the body size; the assertion is "we trust the
        // declared header for the early reject".)
        let oversized = 2 * 1024 * 1024; // 2 MiB > 1 MiB cap
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", oversized.to_string())
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // OpenAI-shape envelope per docs/api-proxy.md §3:
        // `{ "error": { "message": ..., "type": "..." } }`
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("limit"),
            "413 message should reference the limit; got {message:?}"
        );
    }

    /// Issue #159 audit MEDIUM-3: duplicate `Content-Length` headers
    /// are a classic request-smuggling vector — a server that acts on
    /// the first value while a downstream peer acts on the second can
    /// be tricked into framing the body wrongly. Per RFC 9110 §8.6 a
    /// server SHOULD reject the request rather than disambiguate.
    #[tokio::test]
    async fn duplicate_content_length_headers_return_400_invalid_request() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let body = r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#;
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        // Inject TWO Content-Length headers (simulating a smuggling
        // attempt). axum's HeaderMap supports `append` for duplicate
        // names; the middleware must reject rather than read the
        // first value.
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len()),
        );
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len() + 1),
        );

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("Content-Length"),
            "smuggling-rejection message should mention Content-Length; got {message:?}"
        );
    }

    /// Issue #159 companion: a body within the cap must NOT be
    /// rejected — the middleware short-circuits ONLY when the
    /// Content-Length exceeds the cap, leaving normal traffic
    /// untouched. Without this guard, a regression that always-
    /// rejected (e.g. comparing the wrong field) would be invisible
    /// since most existing tests don't set Content-Length.
    #[tokio::test]
    async fn within_limit_body_is_not_rejected_by_middleware() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-ok",
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
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", body.len().to_string())
            .body(Body::from(body))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
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

    /// Cross-provider contract: Anthropic upstream 5xx → client sees an
    /// OpenAI-shape envelope `{error:{type:"upstream_error",...}}` with
    /// status 502 (collapsed per `BridgeError::http_status`, see
    /// crates/aisix-gateway/src/bridge.rs).
    #[tokio::test]
    async fn upstream_anthropic_5xx_collapses_to_502_with_openai_envelope() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(503).set_body_string(
                r#"{"type":"error","error":{"type":"overloaded_error","message":"upstream busy"}}"#,
            ))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
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
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        // Anthropic-shape leaks must not bleed through to the client.
        assert_eq!(v["error"]["type"], "upstream_error");
        assert!(v["error"]["message"].is_string());
    }

    /// Cross-provider 4xx pass-through: Anthropic upstream 400 reaches
    /// the client as 400 + OpenAI-shape envelope (status flows from
    /// `BridgeError::UpstreamStatus.http_status()` 4xx branch).
    #[tokio::test]
    async fn upstream_anthropic_400_passes_through_with_openai_envelope() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_string(
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad input"}}"#,
            ))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
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
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        assert_eq!(v["error"]["type"], "upstream_error");
    }

    /// Garbage upstream body (200 + non-JSON) must surface as 502 with
    /// `error.type = "upstream_decode_error"` — distinct from the 4xx/5xx
    /// `upstream_error` token so dashboards can tell parsing failures
    /// apart from upstream errors.
    #[tokio::test]
    async fn upstream_unparseable_body_returns_502_decode_error_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{not valid json"))
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
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        assert_eq!(v["error"]["type"], "upstream_decode_error");
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

    /// Issue #177: per docs/api-proxy.md §5, abnormal upstream
    /// stream termination must close the response WITHOUT `[DONE]`
    /// — SDK consumers that key off `[DONE]` for clean-completion
    /// signal need to detect truncation. The previous behavior
    /// emitted `[DONE]` after the SSE error event, masking
    /// truncated responses as complete.
    #[tokio::test]
    async fn streaming_response_omits_done_when_upstream_returns_invalid_json_mid_stream() {
        let upstream = MockServer::start().await;
        // Upstream emits two valid SSE chunks then a malformed JSON
        // payload. The malformed payload triggers `serde_json::from_str`
        // to fail in the bridge's `build_chunk_stream`, surfacing as
        // `BridgeError::UpstreamDecode` to `build_sse_stream`, which
        // emits an SSE `event: error` frame. After that frame the
        // proxy MUST NOT emit `[DONE]`.
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial \"},\"finish_reason\":null}]}\n\n\
data: <not valid json>\n\n";
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

        // Drain the wire bytes — at this layer we want byte-level
        // assertions, not just decoded SSE events, because the
        // contract being verified is "did `[DONE]` appear at all
        // on the wire".
        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            wire.extend_from_slice(chunk.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("SSE bytes are utf8");

        // Per docs §5: NO `[DONE]` after abnormal termination.
        assert!(
            !wire_str.contains("data: [DONE]"),
            "abnormal termination MUST close without [DONE]; got wire:\n{wire_str}"
        );
        // The error event MUST be emitted so SDK consumers see a
        // failure signal.
        assert!(
            wire_str.contains("event: error"),
            "abnormal termination MUST emit `event: error`; got wire:\n{wire_str}"
        );
        // The error payload MUST be valid OpenAI-envelope JSON
        // (the SDK does `JSON.parse(sse.data)` BEFORE checking
        // event type, so plain-string payloads yield a SyntaxError
        // instead of the typed APIError callers expect).
        let err_event_idx = wire_str.find("event: error\n").unwrap();
        let after_err = &wire_str[err_event_idx + "event: error\n".len()..];
        let data_line = after_err
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("error event followed by a data line");
        let json_payload = &data_line["data: ".len()..];
        let parsed: serde_json::Value = serde_json::from_str(json_payload)
            .expect("error frame data must be valid OpenAI-envelope JSON");
        assert!(
            parsed.get("error").is_some(),
            "error frame data must be `{{\"error\": {{...}}}}` shape; got {json_payload}"
        );
    }

    // ---- regression coverage for issue #107 -------------------------
    // Pre-fix only /v1/chat/completions enforced rate-limit / budget;
    // every other LLM endpoint silently bypassed both. The test below
    // pins /v1/embeddings — representative of the class — to ensure
    // the gate fires after this PR. Adding the same coverage to every
    // endpoint would multiply the test surface without buying signal,
    // since the gate is centralised in `crate::quota::enforce`. If
    // any individual handler ever stops calling it, that handler's
    // own tests would still catch the breakage on the budget path
    // (BudgetExceeded surfaces as a 4xx the existing tests assert on).

    #[tokio::test]
    async fn rate_limit_rpm_applies_to_embeddings_endpoint_issue_107() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5}
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
        let body = serde_json::json!({"model": "my-gpt4", "input": "hello"});
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request consumes the only RPM slot.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Pre-fix: this would also return 200 (the gate didn't run on
        // /v1/embeddings). Post-fix: 429 because the rpm=1 cap is now
        // enforced uniformly via crate::quota::enforce.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "/v1/embeddings must enforce rate limits (issue #107); pre-fix it bypassed",
        );
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    /// Regression for issue #108: streaming chat used to commit
    /// `0` tokens up front and never look at the upstream's terminal
    /// usage frame. TPM caps were silently bypassed for all streaming
    /// traffic. The fix: build_sse_stream now passes the largest
    /// `total_tokens` seen on any chunk to a callback that calls
    /// `Limiter::add_tokens_post_stream`, after the SSE stream
    /// completes. This test exercises that path end-to-end:
    ///
    /// 1. Issue one streaming request whose terminal SSE chunk
    ///    carries `usage.total_tokens = 1500`. Pre-fix this would
    ///    leave TPM at 0; post-fix TPM should be 1500.
    /// 2. Issue a second streaming request with the same key.
    ///    With TPM cap at 1000, this must 429 (not 200) — the
    ///    pre-emptive `tpm.is_exceeded` check on pre_commit catches
    ///    the over-shoot left by the previous request.
    #[tokio::test]
    async fn streaming_chat_tpm_cap_enforced_after_post_stream_commit_issue_108() {
        let upstream = MockServer::start().await;
        // Final SSE chunk carries usage. OpenAI emits this when the
        // client sets `stream_options.include_usage=true`; the proxy
        // doesn't yet add that on the streamed leg, but the OpenAI
        // bridge does parse `usage` off any chunk that has one — so
        // our mock can include it on the terminal chunk and the
        // bridge will surface it.
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":500,\"completion_tokens\":1000,\"total_tokens\":1500}}\n\n\
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
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"tpm": 1000}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
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

        // First streaming request succeeds. Drive the body to
        // completion so build_sse_stream's on_complete fires.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body_stream = resp.into_body().into_data_stream();
        while let Some(chunk) = body_stream.next().await {
            let _ = chunk.unwrap();
        }

        // Second request must 429 — TPM is now over-shot at 1500/1000.
        // Pre-fix TPM stayed at 0 and this would have been a 200.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "streaming chat must commit upstream tokens to TPM (issue #108); \
             pre-fix this returned 200 and the cap was bypassed",
        );
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
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

    /// Issue #154 regression: a CachePolicy with `enabled: false`
    /// must NOT cache. The disabled policy must be filtered out by
    /// the find-first-enabled predicate at the chat.rs cache gate;
    /// every identical request must reach the upstream and the
    /// response must NOT carry an `x-aisix-cache` header (per the
    /// "policy-gate-closed = no header" contract pinned by the
    /// `applies_to_filters_out_unmatched_model` test above).
    #[tokio::test]
    async fn disabled_cache_policy_does_not_cache_and_emits_no_header() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-uncached",
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
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        // The disabled policy applies_to="all" (would match every
        // request), but `enabled: false` MUST cause the find-first-
        // enabled predicate to skip it.
        seed_cache_policy_disabled(&snap, "off-policy");
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

        // All three calls: 200, no `x-aisix-cache` header (policy
        // is disabled). wiremock's `.expect(3)` fails the test if
        // any call short-circuited via cache (= the disable flag
        // wasn't honored).
        for _ in 0..3 {
            let resp = run(build_router(state.clone()), make_req()).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert!(
                resp.headers().get("x-aisix-cache").is_none(),
                "disabled cache_policy must not emit x-aisix-cache header"
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
    /// can mount multiple Models in one snapshot. `pk_id` lets each
    /// model point at its own ProviderKey row — useful for routing
    /// tests that use multiple upstream MockServers.
    fn model_entry_with_id(id: &str, name: &str, pk_id: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "{pk_id}"
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, model, 1)
    }

    fn pk_entry_with_id(pk_id: &str, api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"openai-{pk_id}","secret":"sk-upstream","api_base":"{api_base}"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(pk_id, pk, 1)
    }

    /// Build a virtual routing Model that points at `targets` (other
    /// Model.display_name values) using the given strategy.
    fn routing_entry(name: &str, strategy: &str, targets: &[&str]) -> ResourceEntry<Model> {
        let target_objs: Vec<serde_json::Value> = targets
            .iter()
            .map(|t| serde_json::json!({"model": t}))
            .collect();
        let cfg = serde_json::json!({
            "display_name": name,
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
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
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
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-standby", &standby_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models
            .insert(model_entry_with_id("m-standby", "secondary", "pk-standby"));
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
        // No upstream provider_key needed — the routing target itself
        // is missing so dispatch fails before any provider lookup.

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
        // Per #153, the wire-level `error.message` MUST NOT carry
        // the matched-pattern detail. The previous assertion
        // `.contains("forbidden-token")` pinned the leaky behavior
        // (the literal value of the forbidden pattern showing up
        // in the caller-visible message). Redaction keeps the
        // matched literal in operator logs (`tracing`) only.
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            !message.contains("forbidden-token"),
            "wire-level error.message must not leak the matched literal; got {message:?}"
        );
        assert_eq!(message, "request blocked by content policy");
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
        // Per #153, the matched literal from the model's response
        // ("secret-string" — what the upstream returned and the
        // guardrail matched) MUST NOT appear in the caller-visible
        // error envelope. Echoing it would defeat the entire point
        // of an output guardrail: anyone who can trigger the rule
        // could extract the model's forbidden output via the error
        // message. This is the most security-critical assertion
        // for the whole guardrail surface.
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            !message.contains("secret-string"),
            "output guardrail leaked the matched literal back to the caller; got {message:?}"
        );
        // The full error envelope (any field) must also be clean —
        // future regressions might leak via a different field
        // (param/code) so check the whole serialized blob.
        let blob = serde_json::to_string(&v).unwrap();
        assert!(
            !blob.contains("secret-string"),
            "output guardrail leaked the matched literal in the envelope; got {blob}"
        );
        assert_eq!(message, "response blocked by content policy");
    }

    /// Regression for #226: when an output-content-filter blocks a
    /// response that the upstream already produced, the telemetry event
    /// MUST carry the upstream-billed `prompt_tokens` /
    /// `completion_tokens` instead of zeroing them. Pre-fix the error
    /// path uniformly emitted zeros for every error variant — the
    /// "request never reached the upstream" assumption baked into the
    /// failure-path comment was wrong for the output-block case where
    /// the upstream HAS run and the provider has already charged.
    #[tokio::test]
    async fn output_guardrail_block_records_upstream_usage_in_telemetry() {
        use aisix_guardrails::{GuardrailChain, KeywordBlocklist, KeywordRule};
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-blocked-1",
                "model": "gpt-4o-2024-08-06",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "leak the secret-string"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let guardrails = Arc::new(GuardrailChain::new(vec![Arc::new(
            KeywordBlocklist::output_only(vec![KeywordRule::literal("secret-string")]),
        )]));
        let state = build_state(snap, hub)
            .with_guardrails(guardrails)
            .with_usage_sink(UsageSink::new(tx));
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

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        // The customer paid the provider for these tokens — telemetry
        // must reflect that, not silently drop them to 0.
        assert_eq!(
            event.prompt_tokens, 11,
            "output-block must preserve the upstream's prompt_tokens"
        );
        assert_eq!(
            event.completion_tokens, 7,
            "output-block must preserve the upstream's completion_tokens"
        );
        assert!(event.guardrail_blocked);
        assert_eq!(event.status_code, 422);
        assert_eq!(event.provider_request_id, "cmpl-blocked-1");
        assert_eq!(event.provider_model_version, "gpt-4o-2024-08-06");
        assert_eq!(event.finish_reason, "stop");
        // cache_status reflects the per-policy gate the request went
        // through; "disabled" here because the test seeds no
        // cache_policy. A regression that drops cache_status on the
        // output-block path would surface as empty-string here.
        assert_eq!(event.cache_status, "disabled");
    }

    /// Regression for ai-gateway#196 audit HIGH-1: streaming chat
    /// telemetry must fire even when the client disconnects mid-
    /// stream. Pre-fix, on_complete lived in a `yield`-following
    /// branch of the async_stream! body that only ran when the
    /// consumer pulled — a dropped consumer (axum aborting the
    /// response future) skipped it entirely, so the customer's
    /// upstream call billed but the gateway recorded zero events.
    /// Post-fix, on_complete fires from a Drop guard so cancellation
    /// still produces a usage_event (with whatever counts were
    /// captured up to disconnect, typically 0 if disconnect beat
    /// the upstream's `usage` chunk).
    #[tokio::test]
    async fn streaming_chat_telemetry_fires_on_client_disconnect() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // Use a slow drip so we can disconnect before [DONE] arrives.
        let sse = "\
data: {\"id\":\"cmpl-cancel-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-cancel-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-cancel-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
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

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

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

        // Read ONE chunk then drop the response body — simulates a
        // client that hung up before the upstream's terminal chunk.
        // The Drop guard inside build_sse_stream must still fire
        // on_complete for this disconnected request.
        let mut body_stream = resp.into_body().into_data_stream();
        let _first = body_stream.next().await;
        drop(body_stream);

        let event = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("usage event was never emitted (Drop guard regression)")
            .expect("sender dropped without sending");
        // The exact token counts depend on how many chunks reached
        // the guard before disconnect — could be 0 (disconnect beat
        // the upstream emission entirely) or more. The contract is
        // simply that an event fires; counts are best-effort. Pin
        // the structural fields to confirm we didn't grab some
        // unrelated event.
        assert_eq!(event.status_code, 200);
        assert!(!event.guardrail_blocked);
    }

    /// Regression for #225: streaming chat must read the terminal SSE
    /// chunk's `usage` block and forward those counts into the
    /// telemetry event. Pre-fix the streaming path captured only
    /// `total_tokens` (for rate-limit accounting) and dropped
    /// `prompt_tokens` / `completion_tokens` — telemetry recorded zero
    /// for every streamed request even though the DP had the real
    /// counts in hand.
    #[tokio::test]
    async fn streaming_chat_telemetry_records_usage_from_terminal_chunk() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // OpenAI's stream_options.include_usage=true shape: the final
        // delta chunk before [DONE] carries a `usage` block.
        let sse = "\
data: {\"id\":\"cmpl-stream-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-stream-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-stream-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":13,\"completion_tokens\":4,\"total_tokens\":17}}\n\n\
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

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
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

        // Drain the SSE body so build_sse_stream's on_complete fires.
        // Telemetry emission is wired to that callback; the channel
        // stays empty until the full stream has been consumed.
        let mut body_stream = resp.into_body().into_data_stream();
        while let Some(chunk) = body_stream.next().await {
            let _ = chunk.unwrap();
        }

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        assert_eq!(
            event.prompt_tokens, 13,
            "streaming telemetry must capture prompt_tokens from the terminal chunk's usage block"
        );
        assert_eq!(
            event.completion_tokens, 4,
            "streaming telemetry must capture completion_tokens from the terminal chunk"
        );
        assert_eq!(event.status_code, 200);
        assert!(!event.guardrail_blocked);
        assert_eq!(event.provider_request_id, "cmpl-stream-1");
        assert_eq!(event.provider_model_version, "gpt-4o-2024-08-06");
        assert_eq!(event.finish_reason, "stop");
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

    // ─── Cross-protocol × upstream matrix ─────────────────────────
    //
    // Closes the gap noted in earlier review: the per-bridge wiremock
    // tests prove each Bridge translates ChatFormat ↔ its wire shape,
    // and the proxy lib tests above prove `/v1/chat/completions` end-
    // to-end against an OpenAi upstream — but the *integration* of an
    // OpenAI-protocol inbound request hitting an Anthropic / Gemini /
    // DeepSeek upstream had zero coverage. These tests pin the full
    // path: client body parser → Hub.get(provider) → Bridge.chat[_stream]
    // → upstream → Bridge response decoder → renderer → wire bytes.

    const MATRIX_ANTHROPIC_PK_ID: &str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    const MATRIX_GEMINI_PK_ID: &str = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
    const MATRIX_DEEPSEEK_PK_ID: &str = "cccccccc-cccc-cccc-cccc-cccccccccccc";

    fn anthropic_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "anthropic",
                "model_name": "claude-3-5-haiku-20241022",
                "provider_key_id": "{MATRIX_ANTHROPIC_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-anthropic-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn gemini_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "gemini",
                "model_name": "gemini-2.0-flash",
                "provider_key_id": "{MATRIX_GEMINI_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-gemini-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn deepseek_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "deepseek",
                "model_name": "deepseek-chat",
                "provider_key_id": "{MATRIX_DEEPSEEK_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-deepseek-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn matrix_pk_entry(
        id: &'static str,
        secret: &str,
        api_base: &str,
    ) -> ResourceEntry<aisix_core::ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"matrix-up","secret":"{secret}","api_base":"{api_base}"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, pk, 1)
    }

    fn matrix_anthropic_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(MATRIX_ANTHROPIC_PK_ID, "sk-ant-test", api_base)
    }

    fn matrix_gemini_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(MATRIX_GEMINI_PK_ID, "ya29-test", api_base)
    }

    fn matrix_deepseek_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(MATRIX_DEEPSEEK_PK_ID, "sk-deepseek", api_base)
    }

    /// (OpenAI inbound) × (Anthropic upstream) × (non-streaming).
    /// The most valuable cross-protocol cell — exercises real wire-shape
    /// translation in both directions inside `AnthropicBridge::chat`.
    #[tokio::test]
    async fn matrix_openai_in_anthropic_upstream_non_streaming() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-ant-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "model": "claude-3-5-haiku-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 4}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
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
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        // OpenAI-shape wire on the way out.
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "Hello from Claude!");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 5);
        assert_eq!(v["usage"]["completion_tokens"], 4);
    }

    /// (OpenAI inbound) × (Anthropic upstream) × (streaming).
    /// Pin the SSE event-stream translation: AnthropicBridge ingests
    /// typed Anthropic events (message_start / content_block_delta /
    /// message_delta / message_stop) and emits flat OpenAI deltas.
    #[tokio::test]
    async fn matrix_openai_in_anthropic_upstream_streaming() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        let sse = "\
event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-5-haiku-20241022\",\"stop_reason\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n\
event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n\
event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n\
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

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
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
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream"),
        );
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        // OpenAI-shape SSE deltas on the way out.
        assert!(
            body.contains("\"object\":\"chat.completion.chunk\""),
            "missing OpenAI chunk envelope in:\n{body}"
        );
        assert!(body.contains("\"content\":\"hel\""));
        assert!(body.contains("\"content\":\"lo\""));
        assert!(body.contains("\"finish_reason\":\"stop\""));
        assert!(body.contains("data: [DONE]"));
    }

    /// (OpenAI inbound) × (Gemini upstream). Gemini's bridge is a
    /// thin wrapper around the OpenAi-compat `/chat/completions`
    /// endpoint, so the upstream wire is OpenAI-shape — but the
    /// `Hub.get(Provider::Gemini)` lookup must still resolve to the
    /// Gemini-specific bridge instance (different metrics label,
    /// different default base URL behavior).
    #[tokio::test]
    async fn matrix_openai_in_gemini_upstream_non_streaming() {
        use aisix_provider_gemini::gemini_bridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-gemini",
                "model": "gemini-2.0-flash",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Gemini!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(matrix_gemini_pk(&upstream.uri()));
        snap.models.insert(gemini_model_entry("my-gemini"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-gemini"]));
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Gemini, Arc::new(gemini_bridge()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gemini",
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
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "Hello from Gemini!");
        assert_eq!(v["usage"]["total_tokens"], 9);
    }

    /// (OpenAI inbound) × (DeepSeek upstream). Mirrors the Gemini
    /// case: DeepSeek's bridge is OpenAi-compat with Bearer auth +
    /// a different `name()` for metrics. The integration test pins
    /// that `Provider::Deepseek` resolves correctly.
    #[tokio::test]
    async fn matrix_openai_in_deepseek_upstream_non_streaming() {
        use aisix_provider_deepseek::deepseek_bridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-deepseek"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-deepseek",
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from DeepSeek!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 6, "completion_tokens": 7, "total_tokens": 13}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_deepseek_pk(&upstream.uri()));
        snap.models.insert(deepseek_model_entry("my-deepseek"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-deepseek"]));
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Deepseek, Arc::new(deepseek_bridge()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-deepseek",
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
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(
            v["choices"][0]["message"]["content"],
            "Hello from DeepSeek!"
        );
        assert_eq!(v["usage"]["total_tokens"], 13);
    }
}
