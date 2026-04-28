//! `POST /playground/chat/completions` — in-process proxy to the chat
//! completions surface.
//!
//! The playground endpoint accepts any request carrying a *proxy* API key
//! (not an admin key) and forwards it to `/v1/chat/completions` through
//! the proxy router so the full middleware stack runs (auth, rate limit,
//! bridge, guardrails). Because both routers live in the same process,
//! the request does not touch the network — it is dispatched via
//! `tower::ServiceExt::oneshot` on the proxy `Router`.
//!
//! If the admin server was started without a wired proxy router (e.g. in
//! unit tests that only exercise the admin surface), the endpoint returns
//! `501 Not Implemented`.

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use tower::ServiceExt;

use crate::state::AdminState;

pub async fn playground_chat_completions(
    State(state): State<AdminState>,
    req: Request<Body>,
) -> Response {
    let Some(proxy) = state.proxy_router.clone() else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            axum::Json(serde_json::json!({
                "error_msg": "playground not wired: proxy router not configured"
            })),
        )
            .into_response();
    };

    // Rewrite the URI to the proxy's own path so the proxy router can
    // match it — the client POSTed to `/playground/chat/completions` but
    // the proxy listens on `/v1/chat/completions`.
    let (mut parts, body) = req.into_parts();
    parts.uri = "/v1/chat/completions".parse().unwrap_or(parts.uri.clone());
    let forwarded = Request::from_parts(parts, body);

    match proxy.oneshot(forwarded).await {
        Ok(resp) => resp,
        Err(infallible) => match infallible {},
    }
}

#[cfg(test)]
mod tests {
    use aisix_core::models::Provider;
    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AdminConfig, AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use aisix_provider_openai::OpenAiBridge;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::Router;
    use std::sync::Arc;
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::{build_router, AdminState, InMemoryStore};
    use aisix_proxy::ProxyState;

    fn admin_cfg() -> AdminConfig {
        AdminConfig {
            addr: "127.0.0.1:0".into(),
            admin_keys: vec!["admin-key".into()],
            tls: None,
        }
    }
    fn proxy_cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            tls: None,
        }
    }

    fn model_entry(name: &str, api_base: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "name": "{name}",
                "model": "openai/gpt-4o",
                "provider_config": {{"api_key": "sk-up", "api_base": "{api_base}"}}
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn apikey_entry() -> ResourceEntry<ApiKey> {
        let k: ApiKey = serde_json::from_str(
            r#"{"key_hash":"946bcbef196665b410dd95685673c8bd7d9d27209f1ae9b9e80aac336d57b26c","allowed_models":["*"]}"#,
        )
        .unwrap();
        ResourceEntry::new("k-1", k, 1)
    }

    fn build_test_app(upstream_uri: &str) -> Router {
        let snap = AisixSnapshot::new();
        snap.models.insert(model_entry("gpt4", upstream_uri));
        snap.apikeys.insert(apikey_entry());

        let snapshot = SnapshotHandle::new(snap);
        let hub = Arc::new(Hub::new());
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let proxy_state = ProxyState::new(snapshot.clone(), hub, &proxy_cfg()).without_cache();
        let proxy_router = aisix_proxy::build_router(proxy_state);

        let store = InMemoryStore::new() as Arc<dyn crate::ConfigStore>;
        let admin_state =
            AdminState::new(snapshot, store, &admin_cfg()).with_proxy_router(proxy_router);

        build_router(admin_state)
    }

    #[tokio::test]
    async fn playground_forwards_to_proxy_and_returns_completion() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-pg",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "playground ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let app = build_test_app(&upstream.uri());
        let body = serde_json::json!({
            "model": "gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/playground/chat/completions")
            .header("authorization", "Bearer sk-proxy") // proxy key
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "playground ok");
    }

    #[tokio::test]
    async fn playground_with_wrong_proxy_key_returns_401() {
        let app = build_test_app("http://unused");
        let req = Request::builder()
            .method("POST")
            .uri("/playground/chat/completions")
            .header("authorization", "Bearer wrong-key") // not a valid proxy key
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn playground_without_proxy_router_returns_501() {
        let snap = AisixSnapshot::new();
        let snapshot = SnapshotHandle::new(snap);
        let store = InMemoryStore::new() as Arc<dyn crate::ConfigStore>;
        // AdminState without with_proxy_router → proxy_router is None.
        let admin_state = AdminState::new(snapshot, store, &admin_cfg());
        let app = build_router(admin_state);

        let req = Request::builder()
            .method("POST")
            .uri("/playground/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    }
}
