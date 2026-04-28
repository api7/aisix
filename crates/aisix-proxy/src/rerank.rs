//! `POST /v1/rerank` — Cohere-style rerank pass-through.
//!
//! This endpoint proxies rerank requests to the upstream provider.
//! The `model` field is resolved and authorised via the same path as
//! chat completions. The body is forwarded verbatim after rewriting the
//! `model` field to the upstream model name.
//!
//! Providers that support rerank natively (Cohere, Voyage, etc.) should
//! be configured with a `base_url` pointing to their rerank endpoint root.
//! The gateway appends `/v1/rerank`.

use aisix_obs::{AccessLog, RequestOutcome};
use axum::extract::State;
use axum::http::HeaderValue;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

pub async fn rerank(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    Json(mut body): Json<Value>,
) -> Response {
    let started = Instant::now();
    let request_id = format!("rerank-{}", Uuid::new_v4());
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

    let api_key = model.provider_config.api_key.as_str().to_string();
    if api_key.is_empty() {
        return Err(ProxyError::Bridge(aisix_gateway::BridgeError::Config(
            "provider_config.api_key is empty".into(),
        )));
    }

    let upstream_model = model
        .upstream_model()
        .ok_or_else(|| ProxyError::InvalidRequest("model field missing provider/ prefix".into()))?
        .to_string();

    let provider_label = model
        .provider()
        .map(|p| format!("{p:?}").to_lowercase())
        .unwrap_or_else(|| "unknown".to_string());

    // Rewrite model field.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model.clone());
    }

    // Build upstream URL: {base}/v1/rerank
    let base = match model.base_url() {
        Some(b) if !b.trim().is_empty() => b.trim_end_matches('/').to_string(),
        _ => {
            // Derive a sensible default base from the provider.
            model
                .provider()
                .and_then(default_base_for_provider)
                .unwrap_or_else(|| "https://api.cohere.ai".to_string())
        }
    };
    let url = format!("{base}/v1/rerank");

    let client = crate::http_client::client();
    let upstream_resp = client
        .post(&url)
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .header("x-aisix-request-id", request_id)
        .json(body)
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
                message: message.chars().take(1024).collect(),
            },
        ));
    }

    state.health.record_success(&model_name);

    let upstream_headers = upstream_resp.headers().clone();
    let body_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| aisix_gateway::BridgeError::UpstreamDecode(e.to_string()))
        .map_err(ProxyError::Bridge)?;

    let mut resp = axum::response::Response::new(axum::body::Body::from(body_bytes));

    // Forward content-type from upstream.
    if let Some(ct) = upstream_headers.get("content-type") {
        if let Ok(hv) = HeaderValue::from_bytes(ct.as_bytes()) {
            resp.headers_mut()
                .insert(axum::http::header::CONTENT_TYPE, hv);
        }
    }
    resp.headers_mut().insert(
        axum::http::header::HeaderName::from_static("x-aisix-request-id"),
        HeaderValue::from_str(request_id).unwrap_or_else(|_| HeaderValue::from_static("")),
    );

    Ok((resp, provider_label))
}

fn default_base_for_provider(provider: aisix_core::models::Provider) -> Option<String> {
    use aisix_core::models::Provider;
    match provider {
        Provider::Openai => Some("https://api.openai.com".to_string()),
        Provider::Anthropic => None, // Anthropic doesn't expose a rerank API
        Provider::Gemini => None,    // Gemini doesn't expose a rerank API
        Provider::Deepseek => None,
    }
}

fn emit_access_log(
    model: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    elapsed: Duration,
    request_id: &str,
) {
    AccessLog {
        method: "POST",
        path: "/v1/rerank",
        status,
        latency: elapsed,
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
    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use axum::http::{Request, StatusCode};
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

    fn openai_model(name: &str, api_base: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"name":"{name}","model":"openai/text-embedding-3-small","provider_config":{{"api_key":"sk-test","api_base":"{api_base}"}}}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn apikey_entry(allowed: &[&str]) -> ResourceEntry<ApiKey> {
        let json = format!(
            r#"{{"key_hash":"8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891","allowed_models":{}}}"#,
            serde_json::to_string(&allowed).unwrap()
        );
        let k: ApiKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("k-1", k, 1)
    }

    fn build_app(snap: AisixSnapshot) -> axum::Router {
        let hub = Arc::new(Hub::new());
        let handle = SnapshotHandle::new(snap);
        crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    fn make_req(body: serde_json::Value) -> Request<axum::body::Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/rerank")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn unauthenticated_returns_401() {
        let snap = AisixSnapshot::new();
        let app = build_app(snap);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/rerank")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"m","query":"hi","documents":["a"]}"#,
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn unknown_model_returns_404() {
        let snap = AisixSnapshot::new();
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "no-such/model",
                "query": "search",
                "documents": ["doc1"]
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn forbidden_model_returns_403() {
        let snap = AisixSnapshot::new();
        snap.models
            .insert(openai_model("rerank-model", "https://api.openai.com"));
        snap.apikeys.insert(apikey_entry(&["other-model"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "rerank-model",
                "query": "search",
                "documents": ["doc1"]
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn happy_path_forwards_to_upstream() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{"index": 0, "relevance_score": 0.9}]
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.models
            .insert(openai_model("my-reranker", &upstream.uri()));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "my-reranker",
                "query": "search query",
                "documents": ["doc1", "doc2"]
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        upstream.verify().await;
    }
}
