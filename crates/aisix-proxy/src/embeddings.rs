//! `POST /v1/embeddings` — OpenAI-compatible embeddings pass-through.
//!
//! Flow:
//! 1. [`AuthenticatedKey`] extractor — 401 if auth fails.
//! 2. Parse [`EmbeddingRequestBody`] from JSON.
//! 3. Resolve model name → `Model` in snapshot → 404 if absent.
//! 4. Check `allowed_models` → 403 if denied.
//! 5. Look up Bridge on Hub → 503 if not registered.
//! 6. Normalise `input` (single string → one-element vec).
//! 7. Call `bridge.embed(req, ctx)` → forward response as JSON.
//! 8. On completion: record metrics and emit access log.
//!
//! Errors follow the same OpenAI-style envelope as chat completions.
//! Providers that don't implement embeddings return a 501 with
//! `"type": "not_implemented"`.

use aisix_gateway::{BridgeContext, BridgeError, EmbeddingRequest};
use aisix_obs::{AccessLog, RequestOutcome};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::auth::AuthenticatedKey;
use crate::error::{ErrorEnvelope, ProxyError};
use crate::state::ProxyState;

/// The request body accepted by `POST /v1/embeddings`.
///
/// `input` may be a single string **or** an array of strings; both are
/// handled by the `InputField` helper so callers don't need to know.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequestBody {
    pub model: String,
    pub input: InputField,
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<u32>,
}

/// Deserialises both `"text"` and `["text", ...]` forms of the
/// OpenAI embeddings `input` field.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputField {
    Single(String),
    Multi(Vec<String>),
}

impl InputField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            InputField::Single(s) => vec![s],
            InputField::Multi(v) => v,
        }
    }
}

pub async fn embeddings(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    Json(body): Json<EmbeddingRequestBody>,
) -> Response {
    let started = Instant::now();
    let request_id = format!("emb-{}", Uuid::new_v4());
    let api_key_id = auth.entry.id.clone();
    let model_name = body.model.clone();

    match dispatch(&state, &auth, body, &request_id).await {
        Ok((resp, provider)) => {
            let elapsed = started.elapsed();
            let status = 200u16;
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
    body: EmbeddingRequestBody,
    request_id: &str,
) -> Result<(Response, String), ProxyError> {
    let snapshot = state.snapshot.load();

    let model_entry = snapshot
        .models
        .get_by_name(&body.model)
        .ok_or_else(|| ProxyError::ModelNotFound(body.model.clone()))?;

    if !auth.key().can_access(&body.model) {
        return Err(ProxyError::ModelForbidden(body.model.clone()));
    }

    let model = &model_entry.value;
    let provider = crate::dispatch::require_provider(model)?;
    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;

    let bridge = state
        .hub
        .get(provider)
        .ok_or(ProxyError::ProviderUnavailable)?;

    // Budget + rate-limit gate (issue #107). Pre-fix this endpoint
    // bypassed both. The reservation is held until commit_tokens at
    // the end of dispatch — embeddings don't surface a stable token
    // count across providers, so we commit 0 for now (RPM counts,
    // TPM doesn't). Plumbing per-provider token totals through is a
    // follow-up.
    let reservation = crate::quota::enforce(state, auth).await?;

    let upstream_model_id = crate::dispatch::require_upstream_model(model)?.to_string();

    let req = EmbeddingRequest {
        model: upstream_model_id,
        input: body.input.into_vec(),
        encoding_format: body.encoding_format,
        dimensions: body.dimensions,
    };

    let model_arc = Arc::new(model.clone());
    let pk_arc = Arc::new(pk_entry.value.clone());
    let ctx = BridgeContext::new(request_id, model_arc, pk_arc);

    match bridge.embed(&req, &ctx).await {
        Ok(embed_resp) => {
            // Commit the reservation — release the concurrency permit
            // and finalise RPM. Embeddings do report prompt_tokens via
            // EmbeddingResponse.usage; thread it through so TPM works
            // here even though other handlers commit 0.
            reservation.commit_tokens(embed_resp.usage.total_tokens as u64);
            let provider_label = format!("{provider:?}").to_lowercase();
            Ok((Json(embed_resp).into_response(), provider_label))
        }
        Err(BridgeError::Config(msg)) if msg.contains("does not support embeddings") => {
            // Provider doesn't implement embed → 501 Not Implemented.
            // Drop the reservation without committing — the request
            // didn't hit the upstream.
            reservation.commit_tokens(0);
            let env = ErrorEnvelope::new(msg, "not_implemented");
            Ok((
                (StatusCode::NOT_IMPLEMENTED, Json(env)).into_response(),
                format!("{provider:?}").to_lowercase(),
            ))
        }
        Err(e) => {
            reservation.commit_tokens(0);
            Err(ProxyError::Bridge(e))
        }
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
    let now_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = now_ts; // only used for context; access log uses elapsed
    AccessLog {
        method: "POST",
        path: "/v1/embeddings",
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
                "model_name": "text-embedding-3-small",
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
            .uri("/v1/embeddings")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    fn upstream_response() -> serde_json::Value {
        serde_json::json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": [0.1_f32, 0.2_f32, 0.3_f32]
            }],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        })
    }

    #[tokio::test]
    async fn happy_path_single_string_input() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hello world"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["object"], "embedding");
        let emb = v["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(emb.len(), 3);
    }

    #[tokio::test]
    async fn happy_path_array_input() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": ["a", "b"]});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unauthenticated_request_returns_401() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"my-embed","input":"hi"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn forbidden_model_returns_403() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["other-model"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hi"});
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
        let body = serde_json::json!({"model": "nonexistent", "input": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn upstream_error_propagates_as_502_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(503).set_body_string("overloaded"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "upstream_error");
    }
}
