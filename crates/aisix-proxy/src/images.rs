//! `POST /v1/images/generations` — image generation pass-through.
//!
//! Flow:
//! 1. [`AuthenticatedKey`] extractor — 401 if auth fails.
//! 2. Parse the body as a JSON object.
//! 3. Validate `model` field is present.
//! 4. Resolve model name → `Model` in snapshot → 404 if absent.
//! 5. Check `allowed_models` → 403 if denied.
//! 6. Look up Bridge on Hub → 503 if not registered.
//! 7. Call `bridge.generate_image(body, ctx)` → JSON response.
//! 8. Providers that don't support image generation return 501.

use aisix_gateway::{BridgeContext, BridgeError};
use aisix_obs::{AccessLog, RequestOutcome};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::auth::AuthenticatedKey;
use crate::error::{ErrorEnvelope, ProxyError};
use crate::state::ProxyState;

pub async fn image_generations(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    Json(body): Json<Value>,
) -> Response {
    let started = Instant::now();
    let request_id = format!("img-{}", Uuid::new_v4());
    let api_key_id = auth.entry.id.clone();
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    match dispatch(&state, &auth, body, &request_id).await {
        Ok((resp, provider)) => {
            let elapsed = started.elapsed();
            emit_access_log(
                &model_name,
                &provider,
                &api_key_id,
                200,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &provider,
                &model_name,
                200,
                RequestOutcome::Success,
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
    body: Value,
    request_id: &str,
) -> Result<(Response, String), ProxyError> {
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::InvalidRequest("missing `model` field".into()))?;

    let snapshot = state.snapshot.load();

    let model_entry = snapshot
        .models
        .get_by_name(model_name)
        .ok_or_else(|| ProxyError::ModelNotFound(model_name.to_string()))?;

    if !auth.key().can_access(model_name) {
        return Err(ProxyError::ModelForbidden(model_name.to_string()));
    }

    let model = &model_entry.value;
    let provider = model
        .provider()
        .ok_or_else(|| ProxyError::InvalidRequest("model has no provider prefix".into()))?;

    let bridge = state
        .hub
        .get(provider)
        .ok_or(ProxyError::ProviderUnavailable)?;

    let model_arc = Arc::new(model.clone());
    let ctx = BridgeContext::new(request_id, model_arc);

    let provider_label = format!("{provider:?}").to_lowercase();

    match bridge.generate_image(&body, &ctx).await {
        Ok(resp_json) => Ok((Json(resp_json).into_response(), provider_label)),
        Err(BridgeError::Config(msg)) if msg.contains("does not support image generation") => {
            let env = ErrorEnvelope::new(msg, "not_implemented");
            Ok((
                (StatusCode::NOT_IMPLEMENTED, Json(env)).into_response(),
                provider_label,
            ))
        }
        Err(e) => Err(ProxyError::Bridge(e)),
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
        path: "/v1/images/generations",
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

#[cfg(test)]
mod tests {
    use aisix_core::models::Provider;
    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use aisix_provider_openai::OpenAiBridge;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
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
                "model": "openai/dall-e-3",
                "provider_config": {{"api_key": "sk-up", "api_base": "{api_base}"}}
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
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
        hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    fn make_req(body: serde_json::Value) -> Request<axum::body::Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/images/generations")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    fn upstream_response() -> serde_json::Value {
        serde_json::json!({
            "created": 1_700_000_000i64,
            "data": [{"url": "https://example.com/image.png"}]
        })
    }

    #[tokio::test]
    async fn happy_path_returns_image_url() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.models.insert(model_entry("dall-e", &upstream.uri()));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "dall-e",
            "prompt": "A sunset over mountains",
            "n": 1,
            "size": "1024x1024"
        });
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["data"][0]["url"].as_str().is_some());
    }

    #[tokio::test]
    async fn unauthenticated_request_returns_401() {
        let snap = AisixSnapshot::new();
        snap.models.insert(model_entry("dall-e", "http://unused"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/images/generations")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"dall-e","prompt":"hi"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn forbidden_model_returns_403() {
        let snap = AisixSnapshot::new();
        snap.models.insert(model_entry("dall-e", "http://unused"));
        snap.apikeys.insert(apikey_entry(&["other-model"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "dall-e", "prompt": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn unknown_model_returns_404() {
        let snap = AisixSnapshot::new();
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "nonexistent", "prompt": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn upstream_error_propagates_as_502() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.models.insert(model_entry("dall-e", &upstream.uri()));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "dall-e", "prompt": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }
}
