//! `POST /v1/completions` — OpenAI-compatible legacy text completions.
//!
//! This endpoint is a thin passthrough to the provider's `/completions`
//! surface. The upstream `model` field is rewritten to the provider's own
//! model id; everything else in the request body is forwarded verbatim.
//!
//! Flow:
//! 1. [`AuthenticatedKey`] extractor — 401 if auth fails.
//! 2. Parse the body as a JSON object.
//! 3. Validate `model` is present.
//! 4. Resolve model name → `Model` in snapshot → 404 if absent.
//! 5. Check `allowed_models` → 403 if denied.
//! 6. Look up Bridge on Hub → 503 if not registered.
//! 7. Call `bridge.complete(body, ctx)` → JSON response.
//! 8. Providers that don't support completions return 501.

use aisix_gateway::{BridgeContext, BridgeError};
use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::{ErrorEnvelope, ProxyError};
use crate::request_id::new_request_id;
use crate::state::ProxyState;

/// Per-request payload from a successful dispatch — carries the
/// response + provider + the bits the handler needs to emit a
/// UsageEvent on the success path (#403).
struct CompletionDispatchSuccess {
    response: Response,
    provider: String,
    /// UUID of the resolved Model row — required for UsageEvent
    /// `model_id`. Always populated on every success arm (including
    /// the 501 NotImplemented branch where no upstream call
    /// happened); the emit gate is `usage.is_some()`, not this
    /// field. Audit MEDIUM-1 on PR #426 clarified.
    model_id: String,
    /// Upstream-reported token counts. `None` on the 501
    /// NotImplemented path (provider doesn't support completions)
    /// or on a 200 with no `usage` block (rare edge). Handler
    /// gates UsageEvent emission on this being `Some`.
    usage: Option<CompletionUsage>,
}

/// Subset of the OpenAI legacy /v1/completions response `usage`
/// block surfaced for telemetry. Field naming mirrors the wire:
/// `prompt_tokens` + `completion_tokens` are both present (unlike
/// embeddings which has only prompt_tokens). Source:
/// <https://platform.openai.com/docs/api-reference/completions/object>
struct CompletionUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

pub async fn completions(
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
            // Audit MEDIUM-2 on PR #426: use the actual response
            // status, not a hardcoded 200. The 501 NotImplemented
            // branch returns `Ok(success)` with a 501 response —
            // logging status=200 there made it impossible for
            // operators to distinguish real successes from "provider
            // does not support completions". Matches the convention
            // PR #404 (responses) and PR #405 (rerank) adopted.
            let status = success.response.status().as_u16();
            emit_access_log(
                &model_name,
                &success.provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &success.provider,
                &model_name,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            // Issue #403: emit UsageEvent so cp-api's budget ledger
            // and customer-facing /logs see /v1/completions spend.
            // Pre-#403 the legacy completions handler dropped the
            // event entirely. Skip emit on the 501 NotImplemented
            // path (no upstream call) and on 200 without a usage
            // block (rare edge) — both surface as `usage: None`.
            if let Some(usage) = success.usage {
                emit_usage_event(
                    &state,
                    &request_id,
                    &success.model_id,
                    &api_key_id,
                    status,
                    elapsed,
                    &usage,
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
) -> Result<CompletionDispatchSuccess, ProxyError> {
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
    let provider = crate::dispatch::require_provider(model)?;
    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;

    let bridge = crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value)
        .ok_or(ProxyError::ProviderUnavailable)?;

    let model_arc = Arc::new(model.clone());
    let pk_arc = Arc::new(pk_entry.value.clone());
    let ctx = BridgeContext::new(request_id, model_arc, pk_arc);

    let provider_label = provider.to_ascii_lowercase();

    match bridge.complete(&body, &ctx).await {
        Ok(resp_json) => {
            // Extract usage BEFORE moving resp_json into the Response
            // so the success struct carries typed counters rather
            // than re-parsing JSON downstream.
            let usage = extract_completion_usage(&resp_json);
            Ok(CompletionDispatchSuccess {
                response: Json(resp_json).into_response(),
                provider: provider_label,
                model_id: model_entry.id.to_string(),
                usage,
            })
        }
        Err(BridgeError::Config(msg)) if msg.contains("does not support text completions") => {
            let env = ErrorEnvelope::new(msg, "not_implemented");
            Ok(CompletionDispatchSuccess {
                response: (StatusCode::NOT_IMPLEMENTED, Json(env)).into_response(),
                provider: provider_label,
                model_id: model_entry.id.to_string(),
                // No upstream call → no usage to attribute. Handler
                // gates emission on `usage.is_some()` so 501 stays
                // out of /logs noise (same convention as #402).
                usage: None,
            })
        }
        Err(e) => Err(ProxyError::Bridge(e)),
    }
}

/// Pull the usage counters out of a legacy /v1/completions response
/// body. Returns `None` only when:
///   - The `usage` block is missing entirely (non-conformant edge), or
///   - `usage.prompt_tokens` is missing / non-numeric (malformed)
///
/// Those cases skip UsageEvent emission rather than attributing a
/// zero-everything noise row to the api_key. The `prompt_tokens` gate
/// distinguishes "no upstream usage at all" from a legitimate reply.
///
/// `completion_tokens`, by contrast, defaults to 0 when absent: a 200
/// that reports a prompt side but omits the completion side is still a
/// real billable call (the prompt was processed) and must be recorded.
/// This matches LiteLLM, which coerces a missing completion side to 0
/// and still logs/bills the event (`completion_tokens=response["usage"].
/// get("completion_tokens", 0)` in convert_dict_to_response.py) — see
/// #429 follow-up. Dropping the whole event would under-record more than
/// the zeroed-completion it was meant to avoid. Wire shape:
/// <https://platform.openai.com/docs/api-reference/completions/object>
fn extract_completion_usage(body: &Value) -> Option<CompletionUsage> {
    let usage = body.get("usage")?;
    let prompt_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64())? as u32;
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    Some(CompletionUsage {
        prompt_tokens,
        completion_tokens,
    })
}

/// Issue #403: push one `UsageEvent` onto cp-api's telemetry sink
/// and fan it out to per-env OTLP exporters. Mirrors the shape of
/// `embeddings::emit_usage_event` (#402) and `responses::emit_usage_event`
/// (#404); the legacy /v1/completions endpoint has both prompt and
/// completion sides but no streaming / reasoning tokens.
///
/// `inbound_protocol = "openai"` per chat.rs convention. Per-PK
/// telemetry attribution (`provider_kind` / `branded_provider` /
/// `pk_label` / `byo_label`) intentionally deferred — wired for
/// chat only today; non-chat handlers gain it via the same
/// follow-up that covers #403-#407.
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    api_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    usage: &CompletionUsage,
    client: &ClientContext,
) {
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "openai".to_string(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        ..Default::default()
    };
    state.usage_sink.try_emit("completions", event.clone());
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
    let _now_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    AccessLog {
        method: "POST",
        path: "/v1/completions",
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
                "model_name": "gpt-3.5-turbo-instruct",
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
            .uri("/v1/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn happy_path_forwards_to_completions_endpoint() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-abc",
                "object": "text_completion",
                "created": 1_700_000_000i64,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [{
                    "text": " is a test",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "instruct", "prompt": "Say this"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["text"], " is a test");
    }

    #[tokio::test]
    async fn unauthenticated_request_returns_401() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"instruct","prompt":"hi"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn forbidden_model_returns_403() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["other-model"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "instruct", "prompt": "hi"});
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
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("error"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "instruct", "prompt": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    /// Issue #403: a successful /v1/completions call must emit a
    /// `UsageEvent` with the upstream-reported prompt + completion
    /// tokens, status_code, model_id, api_key_id, and
    /// `inbound_protocol = "openai"`. Pre-#403 the legacy
    /// completions handler dropped the event entirely.
    #[tokio::test]
    async fn emits_usage_event_on_200_with_tokens_issue_403() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // Legacy OpenAI completions wire shape. Pin specific token
        // counts so a regression that swapped prompt/completion
        // semantics would fail here.
        let upstream_body = serde_json::json!({
            "id": "cmpl-up-1",
            "object": "text_completion",
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{
                "text": "hi",
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        });
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "instruct", "prompt": "hello"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/completions 200")
            .expect("usage_sink sender dropped");

        assert_eq!(event.prompt_tokens, 11);
        assert_eq!(event.completion_tokens, 7);
        assert_eq!(event.status_code, 200);
        assert_eq!(event.api_key_id, "k-1");
        assert_eq!(event.model_id, "m-1");
        assert_eq!(event.inbound_protocol, "openai");
        assert!(!event.request_id.is_empty());
        assert!(!event.occurred_at.is_empty());
    }

    /// Companion: an upstream 200 with `usage: {}` (malformed —
    /// `prompt_tokens` is a required field on every legitimate
    /// completion response) must NOT emit a zero-everything noise
    /// row. Per audit MEDIUM-1 on PR #425 — applied preemptively
    /// here so /v1/completions and /v1/responses share the same
    /// edge-case gate.
    #[tokio::test]
    async fn skips_usage_event_when_upstream_usage_block_is_empty() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "id": "cmpl-up-1",
            "object": "text_completion",
            "choices": [],
            "usage": {}  // malformed — prompt_tokens required by spec
        });
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "instruct", "prompt": "x"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let recv = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await;
        if let Ok(Some(ev)) = recv {
            panic!(
                "no UsageEvent should be emitted when upstream usage block is malformed, \
                 but got prompt_tokens={}",
                ev.prompt_tokens,
            );
        }
    }

    /// #429 (LiteLLM-parity follow-up): a 200 whose `usage` carries
    /// `prompt_tokens` but omits `completion_tokens` is still a real
    /// billable call — the prompt was processed. It MUST emit a
    /// UsageEvent with `completion_tokens = 0` (matching LiteLLM's
    /// `get("completion_tokens", 0)`), NOT be dropped. Only a fully
    /// absent / prompt-less usage block skips (see the two tests below).
    #[tokio::test]
    async fn emits_with_zero_completion_when_completion_tokens_missing() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "id": "cmpl-up-1",
            "object": "text_completion",
            "choices": [],
            "usage": { "prompt_tokens": 50 }  // missing completion_tokens
        });
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "instruct", "prompt": "x"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must still be emitted when only completion_tokens is missing")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 50, "prompt side must be recorded");
        assert_eq!(
            event.completion_tokens, 0,
            "missing completion_tokens must default to 0, not drop the event"
        );
    }

    /// Issue #403 negative pinning: 4xx / 5xx responses must NOT
    /// emit a UsageEvent. Audit MEDIUM-2 on PR #425 — a future
    /// regression that moved `emit_usage_event` into the error
    /// branch would silently ship without this kind of negative
    /// assertion.
    #[tokio::test]
    async fn upstream_5xx_does_not_emit_usage_event() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "instruct", "prompt": "x"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let recv = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await;
        if let Ok(Some(ev)) = recv {
            panic!(
                "5xx must not emit UsageEvent, got status_code={}",
                ev.status_code,
            );
        }
    }

    /// Issue #403 audit MEDIUM-3: the 501 NotImplemented path
    /// (provider doesn't support text completions) must not emit
    /// a UsageEvent — no upstream call happened, so no usage to
    /// attribute. Without this test, a future regression that
    /// flipped `usage: None` → `Some(zero)` on the 501 branch
    /// would silently emit a bogus zero event. Triggers the path
    /// by routing /v1/completions at an Anthropic-backed model;
    /// `AnthropicBridge` doesn't override `Bridge::complete()`
    /// so the trait default returns `BridgeError::Config(...)`
    /// which maps to 501.
    #[tokio::test]
    async fn provider_lacking_complete_returns_501_without_emit() {
        use aisix_obs::UsageSink;
        use aisix_provider_anthropic::AnthropicBridge;

        const ANTHROPIC_PK_ID: &str = "22222222-2222-2222-2222-222222222222";

        let anthropic_pk_json = r#"{"display_name":"anthropic-up","secret":"sk-ant-test","provider":"anthropic","adapter":"anthropic"}"#;
        let anthropic_pk: aisix_core::ProviderKey =
            serde_json::from_str(anthropic_pk_json).unwrap();
        let anthropic_pk_entry = ResourceEntry::new(ANTHROPIC_PK_ID, anthropic_pk, 1);

        let anthropic_model_json = format!(
            r#"{{"display_name":"claude-instruct","provider":"anthropic","model_name":"claude-3-haiku-20240307","provider_key_id":"{ANTHROPIC_PK_ID}"}}"#
        );
        let anthropic_model: Model = serde_json::from_str(&anthropic_model_json).unwrap();
        let anthropic_model_entry = ResourceEntry::new("m-anthropic", anthropic_model, 1);

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk_entry);
        snap.models.insert(anthropic_model_entry);
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "claude-instruct", "prompt": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_IMPLEMENTED,
            "Anthropic-backed /v1/completions must surface as 501 \
             (default Bridge::complete returns BridgeError::Config)",
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

    /// Issue #403 audit LOW-1: a 200 response with NO `usage` block
    /// at all (vs `usage: {}` which is empty-but-present) must not
    /// emit. Pins the outer `body.get("usage")?` short-circuit in
    /// `extract_completion_usage` distinctly from the inner empty
    /// case.
    #[tokio::test]
    async fn skips_usage_event_when_upstream_omits_usage_block_entirely() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // No `usage` key at all — distinct from `usage: {}`.
        let upstream_body = serde_json::json!({
            "id": "cmpl-no-usage",
            "object": "text_completion",
            "choices": []
        });
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "instruct", "prompt": "x"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let recv = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await;
        if let Ok(Some(ev)) = recv {
            panic!(
                "no UsageEvent when `usage` key is entirely absent, \
                 got prompt_tokens={}",
                ev.prompt_tokens,
            );
        }
    }
}
