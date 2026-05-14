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

    let model_rl =
        crate::quota::ModelRateLimit::from_model(model_name, &model_entry.id, &model_entry.value);
    let _reservation = crate::quota::enforce(state, auth, Some(&model_rl)).await?;

    let model = &model_entry.value;

    // Per #168: only OpenAI's API has the documented
    // `/v1/images/generations` route + body shape. Anthropic has no
    // image-generation API at all; Gemini's image generation lives
    // at a different URL (`/v1beta/models/...:generateContent`) with
    // a different body shape; DeepSeek doesn't expose image
    // generation. Routing a non-OpenAI Model here would silently
    // dispatch to an upstream that 404s — a confusing failure for
    // callers who follow `docs/api-proxy.md` §4.9 configuration
    // verbatim. Reject explicitly with 400 (parallel to
    // /v1/responses §4.6) so the configuration error is visible
    // at the gateway boundary.
    if model.provider != Some(aisix_core::models::Provider::Openai) {
        return Err(ProxyError::InvalidRequest(format!(
            "model `{model_name}` is not an OpenAI provider; \
             /v1/images/generations requires OpenAI"
        )));
    }

    let provider = crate::dispatch::require_provider(model)?;
    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;

    let bridge = state
        .hub
        .get(provider)
        .ok_or(ProxyError::ProviderUnavailable)?;

    let model_arc = Arc::new(model.clone());
    let pk_arc = Arc::new(pk_entry.value.clone());
    let ctx = BridgeContext::new(request_id, model_arc, pk_arc);

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

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn model_entry(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "dall-e-3",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn anthropic_model_entry(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "anthropic",
                "model_name": "claude-3-5-haiku-20241022",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn provider_key_entry(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json =
            format!(r#"{{"display_name":"openai-up","secret":"sk-up","api_base":"{api_base}"}}"#);
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(provider_key_entry(api_base));
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

    /// Issue #168 regression: only OpenAI's API has the documented
    /// `/v1/images/generations` route + body shape. A non-OpenAI
    /// Model configured here must be rejected at the gateway
    /// boundary with 400 (parallel to /v1/responses §4.6) rather
    /// than dispatched to an upstream that would 404 (or worse,
    /// hit a different Gemini route shape).
    #[tokio::test]
    async fn non_openai_provider_returns_400_invalid_request() {
        let snap = new_snap("https://api.anthropic.com");
        snap.models.insert(anthropic_model_entry("anthropic-image"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "anthropic-image",
            "prompt": "A sunset over mountains"
        });
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("requires OpenAI"),
            "rejection should reference OpenAI restriction; got {message:?}"
        );
    }

    #[tokio::test]
    async fn happy_path_returns_image_url() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("dall-e"));
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
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("dall-e"));
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
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("dall-e"));
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
        let snap = new_snap("http://unused");
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

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("dall-e"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "dall-e", "prompt": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }
}
