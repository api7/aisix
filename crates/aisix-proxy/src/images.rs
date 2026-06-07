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
use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::{ErrorEnvelope, ProxyError};
use crate::request_id::new_request_id;
use crate::state::ProxyState;

/// Per-request payload from a successful dispatch — carries the
/// response + the bits the handler needs to emit a UsageEvent (#407).
struct ImageDispatchSuccess {
    response: Response,
    provider: String,
    /// UUID of the resolved Model row — required for UsageEvent
    /// `model_id`. Always present on success.
    model_id: String,
    /// `(prompt_tokens, completion_tokens)` from the upstream `usage`
    /// block when the model returns one (gpt-image-1). `None` for
    /// models that don't (dall-e-3) — those still emit a zero-token
    /// event so the request is visible + attributed.
    usage: Option<(u32, u32)>,
    /// `false` on the 501 NotImplemented branch (provider lacks image
    /// generation → no upstream call). Gates emission so the
    /// not-implemented path stays out of /logs (same convention as
    /// embeddings #402).
    upstream_called: bool,
}

pub async fn image_generations(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    Json(body): Json<Value>,
) -> Response {
    let started = Instant::now();
    let request_id = new_request_id();
    let api_key_id = auth.entry.id.clone();
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    match dispatch(&state, &auth, body, &request_id).await {
        Ok(success) => {
            let elapsed = started.elapsed();
            emit_access_log(
                &model_name,
                &success.provider,
                &api_key_id,
                200,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &success.provider,
                &model_name,
                200,
                RequestOutcome::Success,
                elapsed,
            );
            // Issue #407: emit UsageEvent so cp-api's budget ledger +
            // /logs see image-generation traffic. Pre-#407 the handler
            // dropped the event entirely. Emit on a real upstream call
            // (even zero tokens — request visible/attributed); skip the
            // 501 NotImplemented path. Tokens come from the upstream
            // `usage` block when present (gpt-image-1); dall-e-3 has no
            // usage block → zero tokens (precise per-image cost is a
            // documented cross-repo follow-up — needs image-count /
            // size / quality on the wire + cp-api pricing).
            if success.upstream_called {
                let (prompt_tokens, completion_tokens) = success.usage.unwrap_or((0, 0));
                emit_usage_event(
                    &state,
                    &request_id,
                    &success.model_id,
                    &api_key_id,
                    200,
                    elapsed,
                    prompt_tokens,
                    completion_tokens,
                    &client,
                );
            }
            success.response
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
) -> Result<ImageDispatchSuccess, ProxyError> {
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
    if model.provider.as_deref() != Some("openai") {
        return Err(ProxyError::InvalidRequest(format!(
            "model `{model_name}` is not an OpenAI provider; \
             /v1/images/generations requires OpenAI"
        )));
    }

    let provider = crate::dispatch::require_provider(model)?.to_string();
    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;

    let bridge = crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value)
        .ok_or(ProxyError::ProviderUnavailable)?;

    let model_arc = Arc::new(model.clone());
    let pk_arc = Arc::new(pk_entry.value.clone());
    let ctx = BridgeContext::new(request_id, model_arc, pk_arc);

    let provider_label = provider.to_ascii_lowercase();

    match bridge.generate_image(&body, &ctx).await {
        Ok(resp_json) => {
            // Extract usage tokens (gpt-image-1 returns a `usage` block;
            // dall-e-3 doesn't) BEFORE moving resp_json into the
            // Response, so the success struct carries typed counters.
            let usage = extract_token_usage(&resp_json);
            Ok(ImageDispatchSuccess {
                response: Json(resp_json).into_response(),
                provider: provider_label,
                model_id: model_entry.id.to_string(),
                usage,
                upstream_called: true,
            })
        }
        Err(BridgeError::Config(msg)) if msg.contains("does not support image generation") => {
            let env = ErrorEnvelope::new(msg, "not_implemented");
            Ok(ImageDispatchSuccess {
                response: (StatusCode::NOT_IMPLEMENTED, Json(env)).into_response(),
                provider: provider_label,
                model_id: model_entry.id.to_string(),
                usage: None,
                // No upstream call happened → handler skips emit.
                upstream_called: false,
            })
        }
        Err(e) => Err(ProxyError::Bridge(e)),
    }
}

/// Pull `(prompt_tokens, completion_tokens)` from an OpenAI image
/// response `usage` block. gpt-image-1 returns
/// `usage: {input_tokens, output_tokens, total_tokens, ...}`; dall-e-2/3
/// return no `usage` block → `None`. Wire shape:
/// <https://platform.openai.com/docs/api-reference/images/object>
fn extract_token_usage(body: &Value) -> Option<(u32, u32)> {
    let usage = body.get("usage")?;
    let input = usage.get("input_tokens").and_then(Value::as_u64)? as u32;
    let output = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    Some((input, output))
}

/// Issue #407: push one `UsageEvent` onto cp-api's telemetry sink and
/// fan it out to per-env OTLP exporters. Mirrors
/// `embeddings::emit_usage_event` (#402). `inbound_protocol = "openai"`
/// (images are an OpenAI-shape endpoint). Tokens are populated when the
/// upstream returned a `usage` block (gpt-image-1); zero otherwise —
/// the per-image cost basis (n × size × quality) is a cross-repo
/// follow-up needing a UsageEvent wire extension + cp-api pricing.
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    api_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
    client: &ClientContext,
) {
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        prompt_tokens,
        completion_tokens,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "openai".to_string(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        ..Default::default()
    };
    // Handler label "images" — bucketed prometheus counter (#408).
    state.usage_sink.try_emit("images", event.clone());
    let snap = state.snapshot.load();
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
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
    use aisix_obs::UsageEvent;
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
            real_ip: Default::default(),
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
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-up","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
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
            r#"{{"key_hash": "8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891", "allowed_models": {}}}"#,
            serde_json::to_string(&allowed).unwrap()
        );
        let k: ApiKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("k-1", k, 1)
    }

    fn build_app(snap: AisixSnapshot) -> axum::Router {
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
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

    fn build_app_with_sink(
        snap: AisixSnapshot,
        tx: tokio::sync::mpsc::Sender<UsageEvent>,
    ) -> axum::Router {
        use aisix_obs::UsageSink;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        crate::build_router(state)
    }

    /// Issue #407: gpt-image-1 returns a `usage` token block — a
    /// successful generation must emit a UsageEvent carrying those
    /// tokens, attributed to the api_key + model, inbound_protocol
    /// "openai".
    #[tokio::test]
    async fn emits_usage_event_with_tokens_when_upstream_returns_usage() {
        let upstream = MockServer::start().await;
        let body = serde_json::json!({
            "created": 1_700_000_000i64,
            "data": [{"b64_json": "aGVsbG8="}],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 1568,
                "total_tokens": 1618,
                "input_tokens_details": {"text_tokens": 10, "image_tokens": 40}
            }
        });
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("gpt-image"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let req = serde_json::json!({"model": "gpt-image", "prompt": "a cat", "n": 1});
        let resp = tower::ServiceExt::oneshot(app, make_req(req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/images/generations 200")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 50);
        assert_eq!(event.completion_tokens, 1568);
        assert_eq!(event.status_code, 200);
        assert_eq!(event.api_key_id, "k-1");
        assert_eq!(event.model_id, "m-1");
        assert_eq!(event.inbound_protocol, "openai");
    }

    /// Issue #407: dall-e-3 returns NO `usage` block — the request still
    /// emits a zero-token UsageEvent so it's visible in /logs and
    /// attributed (precise per-image cost is a cross-repo follow-up).
    #[tokio::test]
    async fn emits_zero_token_event_when_upstream_omits_usage() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("dall-e"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let req = serde_json::json!({"model": "dall-e", "prompt": "a dog", "n": 1});
        let resp = tower::ServiceExt::oneshot(app, make_req(req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("zero-token UsageEvent must still be emitted (visibility)")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 0);
        assert_eq!(event.completion_tokens, 0);
        assert_eq!(event.status_code, 200);
        assert_eq!(event.model_id, "m-1");
        assert_eq!(event.inbound_protocol, "openai");
    }

    /// Issue #456 (#226 family): the 501 NotImplemented path (resolved
    /// bridge doesn't support image generation) must NOT emit a
    /// UsageEvent — no upstream call happened. Mirrors
    /// `completions.rs::provider_lacking_complete_returns_501_without_emit`
    /// and the embeddings sibling. Unlike those, /v1/images/generations
    /// rejects non-OpenAI providers with 400 *before* dispatch (see
    /// `non_openai_provider_returns_400_invalid_request`) and the real
    /// `OpenAiBridge` overrides `generate_image`, so the only way to reach
    /// the 501 default is an openai-provider Model whose resolved bridge
    /// leaves `Bridge::generate_image` at the trait default. We register a
    /// minimal stub under the "openai" key to exercise exactly that.
    #[tokio::test]
    async fn resolved_bridge_lacking_generate_image_returns_501_without_emit_issue_456() {
        use aisix_gateway::{
            Bridge, BridgeContext, BridgeError, ChatChunkStream, ChatFormat, ChatMessage,
            ChatResponse, FinishReason, UsageStats,
        };
        use aisix_obs::UsageSink;

        // Minimal openai-family bridge that satisfies the required chat
        // methods but leaves `generate_image` at the 501 trait default.
        struct NoImageBridge;

        #[async_trait::async_trait]
        impl Bridge for NoImageBridge {
            fn name(&self) -> &'static str {
                "no-image"
            }

            async fn chat(
                &self,
                req: &ChatFormat,
                _ctx: &BridgeContext,
            ) -> Result<ChatResponse, BridgeError> {
                Ok(ChatResponse {
                    id: "stub".into(),
                    model: req.model.clone(),
                    message: ChatMessage::assistant("stub"),
                    finish_reason: FinishReason::Stop,
                    usage: UsageStats::new(0, 0),
                })
            }

            async fn chat_stream(
                &self,
                _req: &ChatFormat,
                _ctx: &BridgeContext,
            ) -> Result<ChatChunkStream, BridgeError> {
                Ok(Box::pin(futures::stream::iter(Vec::new())))
            }
        }

        let snap = new_snap("https://api.openai.com");
        snap.models.insert(model_entry("stub-image"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(NoImageBridge));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "stub-image", "prompt": "a cat", "n": 1});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_IMPLEMENTED,
            "a bridge without generate_image must surface /v1/images/generations as 501 \
             (default Bridge::generate_image returns BridgeError::Config)",
        );

        let recv = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await;
        if let Ok(Some(ev)) = recv {
            panic!(
                "501 NotImplemented must not emit UsageEvent, \
                 got prompt_tokens={}, status_code={}",
                ev.prompt_tokens, ev.status_code,
            );
        }
    }
}
