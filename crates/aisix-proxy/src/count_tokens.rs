//! `POST /v1/messages/count_tokens` — Anthropic token-counting passthrough.
//!
//! The Anthropic SDK exposes this as `anthropic.messages.countTokens(...)`,
//! the documented endpoint customers use to size a prompt (messages +
//! system + tools + images) before issuing a paid `/v1/messages` call.
//! Claude Code and most Anthropic-SDK apps call it, so a gateway that
//! omits the route forces callers to over-provision or bypass it (#418).
//!
//! This is the sibling sub-route of `/v1/messages`: same model-alias
//! resolution, same `x-api-key` + `anthropic-version` auth shape, same
//! Anthropic-shape error envelope (#336). The only differences from the
//! `/v1/messages` Anthropic passthrough are the upstream suffix
//! (`/messages/count_tokens`), the absence of streaming, and the tiny
//! `{"input_tokens": <int>}` response, which is forwarded verbatim.
//!
//! Scope: Anthropic-backed models only. `count_tokens` has no upstream
//! equivalent for OpenAI/Gemini/DeepSeek, so a non-Anthropic Model is
//! rejected with a 400 at the gateway boundary (parallel to `/v1/rerank`
//! §168 / `/v1/responses` §4.6) rather than dispatched to an upstream
//! that would 404 — and rather than the gateway emitting a misleading
//! 404 of its own, which was the bug this route closes.
//!
//! Reference:
//! - Anthropic Count Message Tokens API:
//!   <https://platform.claude.com/docs/en/api/messages-count-tokens>
//!   (`POST /v1/messages/count_tokens` → `{"input_tokens": <int>}`).
//! - Other OpenAI-compatible gateways expose the same route as a
//!   user-facing passthrough and hit the identical "route missing from
//!   the list" bug.

use aisix_obs::{AccessLog, RequestOutcome};
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue};
use axum::response::Response;
use axum::Json;
use serde_json::Value;
use std::time::{Duration, Instant};

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::messages::ANTHROPIC_VERSION;
use crate::request_id::new_request_id;
use crate::state::ProxyState;

pub async fn count_tokens(
    State(state): State<ProxyState>,
    auth: Result<AuthenticatedKey, ProxyError>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    // Auth / body-extractor rejections must render the Anthropic-shape
    // envelope so the Claude SDK's parser recognises them (#336) — same
    // policy as /v1/messages. The shared helper keeps the body-rejection
    // discrimination (malformed JSON vs 413 cap vs transport error) in
    // lockstep with the sibling route.
    let auth = match auth {
        Ok(a) => a,
        Err(e) => return e.into_anthropic_response(),
    };
    let Json(body) = match body {
        Ok(j) => j,
        Err(rej) => {
            return crate::error::proxy_error_from_json_rejection(
                rej,
                state.request_body_limit_bytes,
            )
            .into_anthropic_response();
        }
    };

    let started = Instant::now();
    let request_id = new_request_id();
    let api_key_id = auth.entry.id.clone();

    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    match dispatch(&state, &auth, &body, &request_id).await {
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
            // Anthropic-shape envelope (#336) — count_tokens callers are
            // the Anthropic SDK, not OpenAI-compatible clients.
            err.into_anthropic_response()
        }
    }
}

async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    body: &Value,
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

    // Resolve the attempt list (routing-aware). count_tokens is
    // Anthropic-only, so we attempt the group's Anthropic targets in
    // order; a direct model resolves to itself (#471).
    let attempt_models = crate::routing::resolve_attempt_models(
        &state.routing,
        &state.runtime_status,
        &snapshot,
        &model_name,
        &model_entry.id,
        &model_entry.value,
    )?;
    let retry_on_429 = model_entry
        .value
        .routing
        .as_ref()
        .map(|r| r.retry_on_429_or_default())
        .unwrap_or(false);

    let mut last_err: Option<ProxyError> = None;
    let mut any_anthropic = false;
    for target in &attempt_models {
        // count_tokens has no upstream equivalent for non-Anthropic
        // providers; skip foreign targets in a mixed group rather than
        // dispatching to an upstream that would 404.
        if target.model.provider.as_deref() != Some("anthropic") {
            continue;
        }
        any_anthropic = true;
        match count_tokens_to_target(
            state,
            &snapshot,
            body,
            &target.model,
            &target.id,
            request_id,
        )
        .await
        {
            Ok(resp) => return Ok((resp, "anthropic".to_string())),
            Err(e) => {
                let retryable = matches!(
                    &e,
                    ProxyError::Bridge(be) if crate::routing::is_retryable(be, retry_on_429)
                );
                last_err = Some(e);
                if !retryable {
                    break;
                }
            }
        }
    }

    // No Anthropic target to serve count_tokens. Reject at the boundary
    // with a 400 (parallel to /v1/rerank's provider gate) rather than
    // dispatching to an upstream that would 404.
    if !any_anthropic {
        return Err(ProxyError::InvalidRequest(format!(
            "model `{model_name}` is not an Anthropic provider; \
             /v1/messages/count_tokens requires an Anthropic-backed model"
        )));
    }
    Err(last_err.unwrap_or(ProxyError::ProviderUnavailable))
}

/// Dispatch one concrete Anthropic target's count_tokens passthrough to
/// `{api_base}/v1/messages/count_tokens`. The caller has already
/// confirmed `model.provider == anthropic`.
async fn count_tokens_to_target(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    body: &Value,
    model: &aisix_core::Model,
    model_id: &str,
    request_id: &str,
) -> Result<Response, ProxyError> {
    let mut body = body.clone();
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
    let api_key = crate::dispatch::require_secret(&pk_entry.value, model)?;
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();

    // Rewrite the `model` field to the upstream value, exactly as the
    // /v1/messages passthrough does — the caller speaks the gateway's
    // display name; the upstream expects its own id.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model.clone());
    }

    // Apply the PK's `request.*` override block to the outbound body,
    // identically to the /v1/messages passthrough — count_tokens shares
    // the same Anthropic ProviderKey, so operator-configured renames /
    // constraints / defaults must reach this sibling route too. Apply
    // order matches §5: renames → constraints → defaults; each is a
    // no-op when its configured map is empty.
    if let Some(r) = pk_entry.value.request.as_ref() {
        aisix_provider_openai::overrides::apply_param_renames(&mut body, &r.param_renames);
        if let Some(constraints) = &r.param_constraints {
            aisix_provider_openai::overrides::apply_param_constraints(&mut body, constraints);
        }
        aisix_provider_openai::overrides::apply_default_body_fields(
            &mut body,
            &r.default_body_fields,
        );
    }

    // `build_v1_url` tolerates an api_base with or without `/v1` (the
    // Anthropic dashboard placeholder and copy-pasted full URLs both
    // resolve to `…/v1/messages/count_tokens`).
    let base = crate::dispatch::resolve_base_url(&pk_entry.value)?;
    let url = crate::dispatch::build_v1_url(&base, "/messages/count_tokens");

    // Build the outbound HeaderMap explicitly so the PK's
    // `request.default_headers` block can inject operator-supplied
    // headers (e.g. `anthropic-beta`) via the shared apply pipeline.
    // The bridge-owned headers (x-api-key, anthropic-version,
    // content-type, x-aisix-request-id) are inserted FIRST;
    // `apply_default_headers` skips keys already present + the reserved
    // auth-header blacklist (`x-api-key`), so operator headers can never
    // clobber auth here (ai-gateway#337). Anthropic auth shape:
    // `x-api-key` + `anthropic-version`, NOT `Authorization: Bearer`.
    let mut headers = axum::http::HeaderMap::new();
    let api_key_hv = HeaderValue::from_str(api_key).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "api key contains invalid header chars: {e}"
        )))
    })?;
    headers.insert(HeaderName::from_static("x-api-key"), api_key_hv);
    headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static(ANTHROPIC_VERSION),
    );
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    let rid_hv = HeaderValue::from_str(request_id).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "request_id contains invalid header chars: {e}"
        )))
    })?;
    headers.insert(HeaderName::from_static("x-aisix-request-id"), rid_hv);
    if let Some(r) = pk_entry.value.request.as_ref() {
        aisix_provider_openai::overrides::apply_default_headers(&mut headers, &r.default_headers);
    }

    let client = crate::http_client::client();
    let upstream_resp = client
        .post(&url)
        .headers(headers)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            crate::cooldown::note_failure(
                &state.runtime_status,
                model_id,
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
        let truncated = crate::util::truncate_on_char_boundary(&message, 1024);
        let err = aisix_gateway::BridgeError::upstream_status_with_retry_after(
            status_u16,
            truncated,
            retry_after,
        );
        if let Some((ttl, reason)) = crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
        {
            state.runtime_status.mark_cooldown(model_id, ttl, reason);
        }
        return Err(ProxyError::Bridge(err));
    }

    state.health.record_success(&model.display_name);
    state.runtime_status.mark_healthy(model_id);

    // Forward the `{"input_tokens": <int>}` response body verbatim — the
    // gateway adds nothing to the token-counting contract.
    let upstream_headers = upstream_resp.headers().clone();
    let body_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| {
            crate::cooldown::note_failure(
                &state.runtime_status,
                model_id,
                model.cooldown.as_ref(),
                aisix_gateway::BridgeError::UpstreamDecode(e.to_string()),
            )
        })
        .map_err(ProxyError::Bridge)?;

    let mut resp = axum::response::Response::new(axum::body::Body::from(body_bytes));
    if let Some(ct) = upstream_headers.get("content-type") {
        if let Ok(hv) = HeaderValue::from_bytes(ct.as_bytes()) {
            resp.headers_mut()
                .insert(axum::http::header::CONTENT_TYPE, hv);
        }
    }
    // Only emit the request-id header when it parses — matching the
    // /v1/messages handler. An empty fallback value would hurt log
    // correlation more than an absent header.
    if let Ok(hv) = HeaderValue::from_str(request_id) {
        resp.headers_mut()
            .insert(HeaderName::from_static("x-aisix-request-id"), hv);
    }

    Ok(resp)
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
        path: "/v1/messages/count_tokens",
        status,
        latency: elapsed,
        provider: Some(provider),
        model: Some(model),
        api_key_id: Some(api_key_id),
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
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
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn anthropic_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"anthropic","model_name":"claude-haiku-4-5-20251001","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn openai_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"openai","model_name":"gpt-4o","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn anthropic_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"anthropic-up","secret":"sk-ant-test","api_base":"{api_base}","provider":"anthropic","adapter":"anthropic"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk(api_base));
        snap
    }

    fn apikey_entry(allowed: &[&str]) -> ResourceEntry<ApiKey> {
        // SHA-256 of "sk-caller".
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
        // Anthropic SDK auth shape: x-api-key + anthropic-version.
        Request::builder()
            .method("POST")
            .uri("/v1/messages/count_tokens")
            .header("x-api-key", "sk-caller")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn unauthenticated_returns_401_anthropic_envelope() {
        let snap = new_snap("http://unused");
        let app = build_app(snap);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        // Anthropic-shape envelope: `{type:"error", error:{type,message}}`.
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "authentication_error");
    }

    #[tokio::test]
    async fn unknown_model_returns_404() {
        let snap = new_snap("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "no-such-model",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn forbidden_model_returns_403() {
        let snap = new_snap("http://unused");
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["other-model"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    /// A non-Anthropic Model has no upstream count_tokens surface;
    /// reject at the boundary with 400 (Anthropic-shape) rather than
    /// 404-ing the caller or dispatching to an upstream that would 404.
    #[tokio::test]
    async fn non_anthropic_provider_returns_400() {
        let snap = AisixSnapshot::new();
        let pk_json = r#"{"display_name":"openai-up","secret":"sk-openai","api_base":"https://api.openai.com","provider":"openai","adapter":"openai"}"#;
        let pk: aisix_core::ProviderKey = serde_json::from_str(pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        snap.models.insert(openai_model("gpt-model"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-model",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(message.contains("Anthropic"), "got {message:?}");
    }

    /// Regression: a >1024-byte upstream error body whose 1024th byte
    /// falls mid-codepoint must not panic the handler — a raw
    /// `&message[..1024]` slice would. Reaching the assertions at all
    /// proves no panic; the upstream 5xx collapses to a gateway 5xx with
    /// the Anthropic-shape error envelope.
    #[tokio::test]
    async fn oversize_non_ascii_upstream_error_does_not_panic() {
        // 1023 ASCII bytes + a 3-byte '€' occupying bytes 1023..1026, so
        // byte index 1024 lands in the middle of a multibyte character.
        let big_body = format!("{}€", "a".repeat(1023));
        assert!(!big_body.is_char_boundary(1024), "test setup invariant");

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .respond_with(ResponseTemplate::new(500).set_body_string(big_body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();

        assert!(resp.status().is_server_error());
        let bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "error");
    }

    /// #418 happy path: the route is registered, dispatches to the
    /// Anthropic upstream at `…/v1/messages/count_tokens`, rewrites the
    /// model field, sends the Anthropic auth headers, and returns the
    /// `{"input_tokens": <n>}` body verbatim.
    #[tokio::test]
    async fn happy_path_forwards_to_anthropic_count_tokens() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .and(header("x-api-key", "sk-ant-test"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "input_tokens": 17
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hello"}]
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["input_tokens"], 17);

        // The model field must be rewritten to the upstream id, and the
        // request must reach the count_tokens sub-route (not /v1/messages).
        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].url.path(), "/v1/messages/count_tokens");
        let sent: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(sent["model"], "claude-haiku-4-5-20251001");
        assert_eq!(sent["messages"][0]["content"], "hello");
    }

    // ─── PK request.* overrides must apply identically to /v1/messages ──
    //
    // count_tokens shares the same Anthropic ProviderKey as /v1/messages,
    // so the operator's `request.*` overrides must reach this sibling too.
    // The mocks strict-match the EXPECTED post-override shape — if an
    // override silently no-ops, the matcher rejects the request and
    // wiremock 404s, surfacing here as a non-200.

    fn anthropic_pk_with_overrides(
        api_base: &str,
        request_overrides: serde_json::Value,
    ) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = serde_json::json!({
            "display_name": "anthropic-up",
            "secret": "sk-ant-test",
            "api_base": api_base,
            "provider": "anthropic",
            "adapter": "anthropic",
            "request": request_overrides,
        });
        let pk: aisix_core::ProviderKey = serde_json::from_value(json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    /// The concrete count_tokens case: an operator `default_headers`
    /// block injecting `anthropic-beta` must reach the upstream request.
    #[tokio::test]
    async fn applies_default_headers_anthropic_beta() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .and(header("anthropic-beta", "token-counting-2024-11-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "input_tokens": 5
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "default_headers": {"anthropic-beta": "token-counting-2024-11-01"}
            }),
        ));
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "default_headers must inject anthropic-beta on count_tokens"
        );
    }

    /// `param_renames` must rewrite the body field on the outbound
    /// count_tokens request, exactly as on /v1/messages.
    #[tokio::test]
    async fn applies_param_renames() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({"renamed_field": "v"}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "input_tokens": 5
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "param_renames": {"orig_field": "renamed_field"}
            }),
        ));
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}],
                "orig_field": "v"
            })))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "param_renames must rewrite orig_field → renamed_field on count_tokens"
        );
    }

    /// Operator `default_headers` must NOT be able to overwrite the
    /// gateway-owned `x-api-key` auth header (ai-gateway#337) — the
    /// reserved blacklist in `apply_default_headers` protects it.
    #[tokio::test]
    async fn default_headers_cannot_overwrite_x_api_key() {
        let upstream = MockServer::start().await;
        // Mock only 200s when x-api-key is the PK secret, NOT the value
        // the operator tried to smuggle via default_headers.
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .and(header("x-api-key", "sk-ant-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "input_tokens": 5
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "default_headers": {"x-api-key": "attacker-key"}
            }),
        ));
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "the PK secret x-api-key must survive an operator default_headers override attempt"
        );
    }
}
