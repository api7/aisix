//! `/passthrough/:provider/*rest` — raw provider pass-through.
//!
//! This endpoint proxies any HTTP method to the upstream provider's API
//! without modification, giving callers access to provider-specific endpoints
//! that the gateway does not natively handle (e.g. fine-tuning, batch
//! management, assistants, etc.).
//!
//! ## Routing
//!
//! The `provider` path segment names a configured Model (or matches a Model
//! whose name starts with the provider prefix). The gateway resolves the
//! `api_key` and `api_base` from the first Model found for that provider.
//!
//! ## Request transformation
//!
//! The request body and headers are forwarded verbatim — only the
//! `Authorization` header is replaced with the provider's key. The incoming
//! API key (proxy key) is stripped and never forwarded.
//!
//! ## Auth
//!
//! Standard proxy authentication applies (`Authorization: Bearer <key>` or
//! `x-api-key`). No model-level authorisation is enforced beyond that.

use aisix_obs::{AccessLog, RequestOutcome};
use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

/// Provider defaults indexed by provider-prefix string.
fn default_base(provider_prefix: &str) -> Option<&'static str> {
    match provider_prefix {
        "openai" => Some("https://api.openai.com"),
        "anthropic" => Some("https://api.anthropic.com"),
        "gemini" => Some("https://generativelanguage.googleapis.com"),
        "deepseek" => Some("https://api.deepseek.com"),
        _ => None,
    }
}

/// Wildcard handler mounted at `/passthrough/:provider/*rest`.
///
/// `method` is not a path parameter — axum merges all HTTP methods for wildcard
/// routes; we read it from the request.
pub async fn passthrough(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    Path((provider, rest)): Path<(String, String)>,
    req: Request,
) -> Response {
    let started = Instant::now();
    let request_id = format!("pt-{}", Uuid::new_v4());
    let api_key_id = auth.entry.id.clone();
    let method = req.method().clone();
    let path = format!("/passthrough/{provider}/{rest}");

    match dispatch(state.clone(), &auth, &provider, &rest, req, &request_id).await {
        Ok((resp, provider_label)) => {
            let elapsed = started.elapsed();
            let status = resp.status().as_u16();
            emit_access_log(
                &method,
                &path,
                &provider_label,
                &api_key_id,
                status,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &provider_label,
                &rest,
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
                &method,
                &path,
                &provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &provider,
                &rest,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            err.into_response()
        }
    }
}

async fn dispatch(
    state: ProxyState,
    auth: &AuthenticatedKey,
    provider: &str,
    rest: &str,
    req: Request,
    request_id: &str,
) -> Result<(Response, String), ProxyError> {
    // Budget + rate-limit gate (issue #107). The previous _auth
    // binding ignored the AuthenticatedKey entirely — passthrough
    // ran completely unmetered, with no per-key budget cap and no
    // RPM/TPM limit. This was the most exploitable gap because the
    // /passthrough/* family covers everything OpenAI ships *plus*
    // every provider's own API. Held for the dispatch lifetime.
    let _reservation = crate::quota::enforce(&state, auth).await?;
    let snapshot = state.snapshot.load();

    // Find a model for this provider so we can borrow its provider_key.
    let provider_lower = provider.to_lowercase();
    let all_models = snapshot.models.entries();
    let model_entry = all_models
        .into_iter()
        .find(|e| {
            e.value
                .provider
                .map(|p| p.as_str().eq_ignore_ascii_case(&provider_lower))
                .unwrap_or(false)
        })
        .ok_or_else(|| {
            ProxyError::ModelNotFound(format!("no model found for provider `{provider}`"))
        })?;

    let model = &model_entry.value;
    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;
    let api_key = crate::dispatch::require_secret(&pk_entry.value, model)?.to_string();

    let base = match pk_entry.value.api_base.as_deref() {
        Some(b) if !b.trim().is_empty() => b.trim_end_matches('/').to_string(),
        _ => default_base(&provider_lower)
            .map(|s| s.to_string())
            .ok_or_else(|| {
                ProxyError::InvalidRequest(format!(
                    "no api_base configured for provider `{provider}` and no default known"
                ))
            })?,
    };

    // Build the target URL: {base}/{rest}
    let url = if rest.is_empty() {
        base.clone()
    } else {
        format!("{base}/{rest}")
    };

    // Preserve the query string.
    let url = if let Some(q) = req.uri().query() {
        format!("{url}?{q}")
    } else {
        url
    };

    let method = req.method().clone();
    let incoming_headers = req.headers().clone();
    let body_bytes: Bytes = axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024)
        .await
        .map_err(|e| ProxyError::InvalidRequest(format!("failed to read body: {e}")))?;

    let client = crate::http_client::client();
    let mut builder = client.request(method.clone(), &url);

    // Inject upstream Authorization; strip the incoming proxy auth.
    if api_key.is_empty() {
        // Some providers use special headers (anthropic uses x-api-key).
        if provider_lower == "anthropic" {
            builder = builder.header("x-api-key", &api_key);
        }
    } else {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {api_key}"));
        if provider_lower == "anthropic" {
            builder = builder.header("x-api-key", &api_key);
            builder = builder.header("anthropic-version", "2023-06-01");
        }
    }

    // Forward safe incoming headers (drop hop-by-hop and auth).
    for (name, value) in &incoming_headers {
        let n = name.as_str().to_lowercase();
        if matches!(
            n.as_str(),
            "authorization" | "x-api-key" | "host" | "content-length"
        ) {
            continue;
        }
        builder = builder.header(name, value);
    }

    builder = builder.header("x-aisix-request-id", request_id);

    if !body_bytes.is_empty() {
        builder = builder.body(body_bytes);
    }

    let upstream_resp = builder
        .send()
        .await
        .map_err(|e| aisix_gateway::BridgeError::Transport(e.to_string()))
        .map_err(ProxyError::Bridge)?;

    let status = upstream_resp.status();
    let resp_headers = upstream_resp.headers().clone();
    let resp_body = upstream_resp
        .bytes()
        .await
        .map_err(|e| aisix_gateway::BridgeError::UpstreamDecode(e.to_string()))
        .map_err(ProxyError::Bridge)?;

    let mut response = Response::builder()
        .status(status)
        .body(Body::from(resp_body))
        .unwrap();

    // Copy relevant response headers.
    copy_safe_headers(&resp_headers, response.headers_mut());

    if let Ok(hv) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert(
            axum::http::header::HeaderName::from_static("x-aisix-request-id"),
            hv,
        );
    }

    Ok((response, provider_lower))
}

/// Copy response headers that are safe to relay to the downstream caller.
fn copy_safe_headers(src: &HeaderMap, dst: &mut HeaderMap) {
    for (name, value) in src {
        let n = name.as_str().to_lowercase();
        // Skip hop-by-hop headers.
        if matches!(
            n.as_str(),
            "transfer-encoding"
                | "connection"
                | "keep-alive"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "te"
                | "trailers"
                | "upgrade"
        ) {
            continue;
        }
        dst.insert(name.clone(), value.clone());
    }
}

fn emit_access_log(
    method: &Method,
    path: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    elapsed: Duration,
    request_id: &str,
) {
    AccessLog {
        method: method.as_str(),
        path,
        status,
        latency: elapsed,
        provider: Some(provider),
        model: None,
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
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;
    use wiremock::matchers::{method as wm_method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            tls: None,
        }
    }

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn openai_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"openai","model_name":"gpt-4o","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn provider_key_entry(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json =
            format!(r#"{{"display_name":"openai-up","secret":"sk-test","api_base":"{api_base}"}}"#);
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

    #[tokio::test]
    async fn unauthenticated_returns_401() {
        let snap = new_snap("http://unused");
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn unknown_provider_returns_404() {
        let snap = new_snap("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/cohere/v1/embed")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn happy_path_forwards_to_upstream() {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": []
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
    }

    #[tokio::test]
    async fn upstream_non_200_is_relayed_verbatim() {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .and(path("/v1/fine_tuning/jobs"))
            .respond_with(ResponseTemplate::new(422).set_body_json(serde_json::json!({
                "error": {"code": "validation_error", "message": "invalid file_id"}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("POST")
            .uri("/passthrough/openai/v1/fine_tuning/jobs")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"training_file":"file-xyz","model":"gpt-4o"}"#,
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // 422 from upstream is relayed as-is (not remapped to 502).
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }
}
