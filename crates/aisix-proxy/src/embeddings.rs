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

    let model_rl =
        crate::quota::ModelRateLimit::from_model(&body.model, &model_entry.id, &model_entry.value);
    let reservation = crate::quota::enforce(state, auth, Some(&model_rl)).await?;

    let upstream_model_id = crate::dispatch::require_upstream_model(model)?.to_string();

    // Preserve the caller's original `input` shape per #162 /
    // `docs/api-proxy.md` §4.4 "both pass through". The bridge will
    // use this flag to serialise the upstream wire body as either a
    // single string or an array — without it, the gateway always
    // forwarded `["text"]` even when the caller sent `"text"`,
    // which contradicts the docs and confuses operator-side packet
    // captures during billing reconciliation / debugging.
    let input_was_single = matches!(body.input, InputField::Single(_));
    let req = EmbeddingRequest {
        model: upstream_model_id,
        input: body.input.into_vec(),
        input_was_single,
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

    /// Issue #162 regression: when the caller's `input` is a single
    /// string, the upstream wire body MUST be a single string (NOT a
    /// one-element array). Per `docs/api-proxy.md` §4.4, both shapes
    /// pass through; pre-fix the gateway always sent `["text"]` to
    /// the upstream regardless of the caller's shape.
    #[tokio::test]
    async fn single_string_input_preserves_string_shape_on_upstream_wire() {
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
        let body = serde_json::json!({"model": "my-embed", "input": "hello"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain the upstream's recorded request body and inspect the
        // `input` field on the upstream wire. A regression that
        // re-introduced the always-array normalisation would write
        // `["hello"]` here.
        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1, "exactly one upstream call expected");
        let upstream_body: serde_json::Value =
            serde_json::from_slice(&received[0].body).expect("upstream body is valid JSON");
        assert!(
            upstream_body["input"].is_string(),
            "single-string caller input must reach upstream as a string, not an array; got {:?}",
            upstream_body["input"]
        );
        assert_eq!(upstream_body["input"], "hello");
    }

    /// Counterpart to the above: when the caller's `input` is an
    /// array (even single-element), the upstream wire body is also
    /// an array. Without this companion test, a regression that
    /// over-corrected to "always single-string when len==1" would
    /// silently rewrite the caller's explicit array.
    #[tokio::test]
    async fn array_input_preserves_array_shape_on_upstream_wire() {
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
        // Caller uses array form even though there's only one
        // element. Gateway must NOT silently rewrite to a string.
        let body = serde_json::json!({"model": "my-embed", "input": ["only-one"]});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let upstream_body: serde_json::Value =
            serde_json::from_slice(&received[0].body).expect("upstream body is valid JSON");
        assert!(
            upstream_body["input"].is_array(),
            "array-form caller input must reach upstream as an array, not coerced to a string; got {:?}",
            upstream_body["input"]
        );
        let arr = upstream_body["input"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0], "only-one");
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

    #[tokio::test]
    async fn response_contains_usage_tokens_from_upstream() {
        // The existing `happy_path_*` tests assert response.data shape but
        // never pin the usage envelope; cp-api depends on it for billing.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1_f32, 0.2_f32]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 7, "total_tokens": 7}
            })))
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
        assert_eq!(v["usage"]["prompt_tokens"], 7);
        assert_eq!(v["usage"]["total_tokens"], 7);
    }

    #[tokio::test]
    async fn upstream_request_uses_provider_model_name_not_display_name() {
        // Model-alias resolution: the gateway's public display_name
        // (`my-embed`) must be rewritten to the upstream provider's
        // model id (`text-embedding-3-small`) before forwarding.
        // wiremock's body_partial_json matcher only fires on the
        // rewritten body; a 200 OK proves the alias was resolved.
        use wiremock::matchers::body_partial_json;
        let upstream = MockServer::start().await;
        // `.expect(1)` forces wiremock to assert on Drop that the mock
        // fired exactly once. The 200-status check below already catches
        // the wiremock-default-404 fallthrough path; the additional value
        // of `.expect(1)` is catching a regression class the status
        // check cannot — a future refactor that returns success WITHOUT
        // ever reaching the upstream (cached response, synthetic 200,
        // dry-run path). Status would still be 200, but the mock count
        // would be 0 and `.expect(1)` would fail on Drop.
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_partial_json(serde_json::json!({
                "model": "text-embedding-3-small"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hello"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "model alias was not rewritten to upstream provider model name"
        );
    }

    #[tokio::test]
    async fn upstream_429_propagates_status_to_client() {
        // ai-gateway's `BridgeError::UpstreamStatus` already maps 4xx
        // through (see crates/aisix-proxy/src/error.rs); this test pins
        // the contract for the embeddings path so a refactor can't
        // silently turn upstream 429 into a generic 502.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(429).set_body_string(
                r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#,
            ))
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
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }
}
