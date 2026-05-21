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

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::request_id::new_request_id;
use crate::state::ProxyState;

pub async fn rerank(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    Json(mut body): Json<Value>,
) -> Response {
    let started = Instant::now();
    let request_id = new_request_id();
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

    let model_rl =
        crate::quota::ModelRateLimit::from_model(&model_name, &model_entry.id, &model_entry.value);
    let _reservation = crate::quota::enforce(state, auth, Some(&model_rl)).await?;

    let model = &model_entry.value;

    // Provider routing key, derived from `model.provider` as a
    // lowercase string. Per #302 Phase A this dispatch path
    // identifies Cohere/Jina by string rather than by `Provider`
    // enum variant so that, when the `Provider` enum is later
    // collapsed into the closed `Adapter` set, this file does not
    // depend on variants (`Provider::Cohere`, `Provider::Jina`)
    // that are slated for removal. The string values
    // ("openai", "cohere", "jina") are the same labels emitted in
    // metrics/access logs today, so dashboards keep working
    // unchanged.
    let provider_label = model
        .provider
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    // Per #168 + #213 Phases 1–2: `/v1/rerank` accepts OpenAI-,
    // Cohere-, and Jina-shape upstreams. All three speak the same
    // body shape (`{model, query, documents, top_n, ...}`) at
    // `…/v1/rerank` with `Authorization: Bearer …` auth, so the
    // gateway forwards verbatim with only the `model` field
    // rewritten. Anthropic, Gemini, and DeepSeek do not expose
    // this surface — routing a Model with one of those providers
    // here would silently 404 upstream, so reject explicitly at
    // the gateway boundary (parallel to `/v1/responses` §4.6).
    //
    // Voyage AI is intentionally NOT in this set despite also
    // having `/v1/rerank` — Voyage uses `top_k` (not `top_n`) on
    // request and `data` (not `results`) on response, so it
    // requires a request/response adapter that's out of scope
    // for this phase. Tracked in the #213 follow-up.
    let provider_allowed = matches!(provider_label.as_str(), "openai" | "cohere" | "jina");
    if !provider_allowed {
        return Err(ProxyError::InvalidRequest(format!(
            "model `{model_name}` is not an OpenAI, Cohere, or Jina provider; \
             /v1/rerank requires OpenAI, Cohere, or Jina"
        )));
    }

    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;
    let api_key = crate::dispatch::require_secret(&pk_entry.value, model)?.to_string();
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();

    // Rewrite model field.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model.clone());
    }

    // Build upstream URL. build_v1_url tolerates either base form —
    // `https://api.cohere.com` (bare host) and `https://api.openai.com/v1`
    // (OpenAI-SDK convention, with /v1) both end up at `…/v1/rerank`
    // instead of `…/v1/v1/rerank`.
    //
    // The provider arm of `default_base_for_provider` is guaranteed to
    // return `Some` here because the gate above already rejected any
    // provider label outside `{"openai", "cohere", "jina"}` — all three
    // have explicit arms in the helper. The `unwrap_or_else` is
    // defensive against a future provider string that gets through the
    // gate without an arm in the helper; the audit-trail-friendly
    // default is OpenAI's host (it's a 4xx-from-OpenAI rather than
    // dispatching to a stale legacy domain).
    let base = match pk_entry.value.api_base.as_deref() {
        Some(b) if !b.trim().is_empty() => b.trim_end_matches('/').to_string(),
        _ => default_base_for_provider(&provider_label)
            .unwrap_or_else(|| "https://api.openai.com".to_string()),
    };
    let url = crate::dispatch::build_v1_url(&base, "/rerank");

    let client = crate::http_client::client();
    let upstream_resp = client
        .post(&url)
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .header("x-aisix-request-id", request_id)
        .json(body)
        .send()
        .await
        .map_err(|e| {
            crate::cooldown::note_failure(
                &state.runtime_status,
                &model_entry.id,
                model.cooldown.as_ref(),
                aisix_gateway::BridgeError::Transport(e.to_string()),
            )
        })
        .map_err(ProxyError::Bridge)?;

    let status = upstream_resp.status();

    if !status.is_success() {
        let status_u16 = status.as_u16();
        let retry_after = aisix_gateway::parse_retry_after(upstream_resp.headers());
        let message = upstream_resp.text().await.unwrap_or_default();
        let err = aisix_gateway::BridgeError::upstream_status_with_retry_after(
            status_u16,
            message.chars().take(1024).collect::<String>(),
            retry_after,
        );
        if let Some((ttl, reason)) = crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
        {
            state
                .runtime_status
                .mark_cooldown(&model_entry.id, ttl, reason);
        }
        return Err(ProxyError::Bridge(err));
    }

    state.health.record_success(&model_name);
    state.runtime_status.mark_healthy(&model_entry.id);

    let upstream_headers = upstream_resp.headers().clone();
    let body_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| {
            crate::cooldown::note_failure(
                &state.runtime_status,
                &model_entry.id,
                model.cooldown.as_ref(),
                aisix_gateway::BridgeError::UpstreamDecode(e.to_string()),
            )
        })
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

/// Default upstream host for the rerank-supporting providers,
/// keyed by the lowercase provider string. Per #302 Phase A this
/// is intentionally a string-keyed match (not a `Provider` enum
/// match) so the file does not depend on `Provider::Cohere` /
/// `Provider::Jina` variants that are slated for removal. The
/// `{"openai", "cohere", "jina"}` set mirrors the rerank gate in
/// `dispatch`; any other string returns `None` and the caller
/// falls back to OpenAI's host.
fn default_base_for_provider(provider: &str) -> Option<String> {
    match provider {
        "openai" => Some("https://api.openai.com".to_string()),
        // Cohere v1 path (deprecated by Cohere but still functional)
        // is what the gateway's `build_v1_url` produces from this
        // base. Operators who want the Cohere v2 path can override
        // `api_base` to `https://api.cohere.com/v2` — see #213's v2
        // follow-up for the version-routing extension if needed.
        "cohere" => Some("https://api.cohere.com".to_string()),
        // Jina rerank is identity-mapped to the OpenAI-compat /
        // Cohere wire shape on both request AND response — same
        // body fields, same `results` array shape, Bearer auth.
        "jina" => Some("https://api.jina.ai".to_string()),
        _ => None,
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

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn openai_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"openai","model_name":"text-embedding-3-small","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn anthropic_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"anthropic","model_name":"claude-3-5-haiku-20241022","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn cohere_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"cohere","model_name":"rerank-english-v3.0","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn jina_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"jina","model_name":"jina-reranker-v2-base-multilingual","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn provider_key_entry(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-test","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
        );
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
        let snap = new_snap("http://unused");
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
        let snap = new_snap("http://unused");
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
        let snap = new_snap("https://api.openai.com");
        snap.models.insert(openai_model("rerank-model"));
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

    /// Issue #168 regression: only OpenAI's API exposes the
    /// documented `/v1/rerank` route + body shape. A non-OpenAI
    /// Model configured here must be rejected at the gateway
    /// boundary with 400 (parallel to /v1/responses §4.6) rather
    /// than dispatched to an upstream that would 404.
    #[tokio::test]
    async fn non_openai_provider_returns_400_invalid_request() {
        let snap = new_snap("https://api.anthropic.com");
        snap.models.insert(anthropic_model("anthropic-rerank"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "anthropic-rerank",
                "query": "search",
                "documents": ["doc1"]
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        // Per #213 Phases 1–2: the rejection message enumerates the
        // accepted set `{OpenAI, Cohere, Jina}`. Pin each provider
        // name individually so:
        //   - a regression that drops a provider from the gate's
        //     accepted set fails this assertion (the missing name
        //     wouldn't appear in the error message);
        //   - future Phase 2.5+ additions can reword the message
        //     freely without breaking this test (substring-per-
        //     provider is forward-compatible per audit LOW-2 on
        //     PR #227).
        assert!(message.contains("OpenAI"), "got {message:?}");
        assert!(message.contains("Cohere"), "got {message:?}");
        assert!(message.contains("Jina"), "got {message:?}");
    }

    /// Issue #213 Phase 2: a Model with `provider: "jina"` MUST
    /// dispatch successfully on `/v1/rerank`. Jina's rerank
    /// (https://api.jina.ai/v1/rerank) is identity-mapped to the
    /// Cohere/OpenAI-compat wire shape — same body fields
    /// (`{model, query, documents, top_n, ...}`), same Bearer
    /// auth, same `results: [{index, relevance_score}]` response
    /// shape — so the gateway forwards verbatim with only the
    /// `model` field rewritten.
    #[tokio::test]
    async fn jina_provider_dispatches_to_upstream_with_bearer_auth() {
        use wiremock::matchers::header;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .and(header("authorization", "Bearer jina_mock_secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "jina-reranker-v2-base-multilingual",
                "usage": {"total_tokens": 42},
                "results": [
                    {"index": 0, "relevance_score": 0.91},
                    {"index": 1, "relevance_score": 0.27}
                ]
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        // Operator-style configuration: bare host, no /v1 suffix.
        // The gateway's `build_v1_url` produces `/v1/rerank` correctly
        // for both `https://api.jina.ai` and `https://api.jina.ai/v1`.
        let pk_json = format!(
            r#"{{"display_name":"jina-up","secret":"jina_mock_secret","api_base":"{}","provider":"jina"}}"#,
            upstream.uri()
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        snap.models.insert(jina_model("jina-rerank"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "jina-rerank",
                "query": "search query",
                "documents": ["doc one", "doc two"],
                "top_n": 2
            })))
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Jina provider must dispatch successfully on /v1/rerank per #213 Phase 2"
        );

        // Pin the EXACT field set forwarded to Jina (parallel to the
        // Cohere case). Jina's documented body is
        // `{model, query, documents, top_n, return_documents}`; the
        // gateway forwards verbatim with only `model` rewritten. A
        // regression injecting an OpenAI-only field would 400 against
        // Jina without failing a happy-path 200 alone.
        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let upstream_body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(
            upstream_body["model"], "jina-reranker-v2-base-multilingual",
            "model field MUST be rewritten to upstream model_name"
        );
        let upstream_obj = upstream_body.as_object().unwrap();
        let mut keys: Vec<&str> = upstream_obj.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(keys, vec!["documents", "model", "query", "top_n"]);
    }

    /// Issue #213 Phase 1: a Model with `provider: "cohere"` MUST
    /// dispatch successfully on `/v1/rerank` (Cohere natively
    /// implements the same body shape OpenAI-compat servers use).
    /// Pre-#213 the gate only accepted OpenAI; this test pins the
    /// expansion at the unit level.
    #[tokio::test]
    async fn cohere_provider_dispatches_to_upstream_with_bearer_auth() {
        use wiremock::matchers::header;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .and(header("authorization", "Bearer sk-cohere-mock"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "rerank-resp-cohere-01",
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.42},
                ],
                "meta": {
                    "api_version": {"version": "1"},
                    "billed_units": {"search_units": 1}
                }
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        // Cohere's API base form: bare host, no /v1 suffix. The
        // gateway's `build_v1_url` appends /v1/rerank correctly for
        // both `https://api.cohere.com` and `https://api.cohere.com/v1`.
        let pk_json = format!(
            r#"{{"display_name":"cohere-up","secret":"sk-cohere-mock","api_base":"{}","provider":"cohere","adapter":"openai"}}"#,
            upstream.uri()
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        snap.models.insert(cohere_model("cohere-rerank"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "cohere-rerank",
                "query": "search query",
                "documents": ["doc one", "doc two"],
                "top_n": 2
            })))
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Cohere provider must dispatch successfully on /v1/rerank per #213 Phase 1"
        );

        // Verify the upstream-side body: model rewritten to the
        // `model_name` from the Cohere Model entry; everything else
        // verbatim. wiremock's `matchers::header` already pinned the
        // Bearer auth on the upstream request matcher.
        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1, "exactly one upstream call expected");
        let upstream_body: serde_json::Value =
            serde_json::from_slice(&received[0].body).expect("upstream body is valid JSON");
        assert_eq!(
            upstream_body["model"], "rerank-english-v3.0",
            "model field MUST be rewritten to upstream model_name; got {}",
            upstream_body["model"]
        );
        assert_eq!(upstream_body["query"], "search query");
        assert_eq!(
            upstream_body["documents"],
            serde_json::json!(["doc one", "doc two"])
        );
        assert_eq!(upstream_body["top_n"], 2);

        // Per #213 audit MEDIUM-2: pin the EXACT field set sent to
        // Cohere. Cohere's `/v1/rerank` documents `{model, query,
        // documents, top_n, return_documents, max_chunks_per_doc}`
        // (https://docs.cohere.com/reference/rerank). The gateway
        // forwards verbatim — but a future regression that injects
        // an OpenAI-only field (e.g. `dimensions` from embeddings,
        // or `stream` from chat) would break Cohere upstream
        // without failing a "happy path 200" test. Pinning the
        // exact key set catches that.
        let upstream_obj = upstream_body
            .as_object()
            .expect("upstream body is a JSON object");
        let mut keys: Vec<&str> = upstream_obj.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec!["documents", "model", "query", "top_n"],
            "upstream body must contain ONLY the fields the caller sent (no gateway-injected extras)"
        );
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

        let snap = new_snap(&upstream.uri());
        snap.models.insert(openai_model("my-reranker"));
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
