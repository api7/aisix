//! `POST /v1/responses` — OpenAI Responses API pass-through.
//!
//! The Responses API is an OpenAI-specific endpoint that lets callers
//! interact with the stateful responses surface (`gpt-4o` + tools). The
//! gateway proxies it transparently:
//!
//! 1. Authenticate and authorise the API key + model.
//! 2. Validate the model is an OpenAI provider.
//! 3. Rewrite the `model` field to the upstream model name.
//! 4. Forward verbatim — streaming SSE and non-streaming JSON both work.
//!
//! Only OpenAI models support this endpoint. Non-OpenAI models receive a
//! 400 with an explanatory message.

use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use std::time::{Duration, Instant};

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::request_id::new_request_id;
use crate::state::ProxyState;

/// Per-request payload from a successful dispatch — carries the
/// response + provider label + the bits of usage data needed for
/// UsageEvent emission (#404). The streaming path returns
/// `usage = None` because the byte stream is passed through verbatim
/// and the gateway doesn't parse SSE chunks here today; emission
/// for the streaming path is tracked as a follow-up.
struct ResponseDispatchSuccess {
    response: Response,
    provider: String,
    /// Set on non-streaming 2xx; `None` on streaming (where the
    /// gateway doesn't parse the SSE chunks to extract usage).
    usage: Option<ResponseUsage>,
    /// UUID of the resolved Model row — needed for UsageEvent
    /// `model_id` field. Always present on success.
    model_id: String,
}

/// Subset of the OpenAI Responses-API `usage` block the gateway
/// surfaces for telemetry. Other fields (`total_tokens`,
/// `output_tokens_details.audio_tokens`, etc.) are intentionally
/// dropped here — cp-api's `dpmgr_usage_events` table records only
/// the ones below.
struct ResponseUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    /// o1/o3/GPT-5 class models surface reasoning tokens as a
    /// subset of `completion_tokens` via
    /// `usage.output_tokens_details.reasoning_tokens`. Zero for
    /// models that don't expose this.
    reasoning_tokens: u32,
    /// OpenAI prompt-cache hit count, subset of `prompt_tokens`,
    /// surfaced via `usage.input_tokens_details.cached_tokens`.
    cached_prompt_tokens: u32,
}

pub async fn responses(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    Json(body): Json<Value>,
) -> Response {
    let started = Instant::now();
    let request_id = new_request_id();
    let api_key_id = auth.entry.id.clone();

    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    match dispatch(&state, &auth, &body, &request_id).await {
        Ok(success) => {
            let elapsed = started.elapsed();
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
            // Issue #404: emit UsageEvent so cp-api's budget ledger
            // and customer-facing /logs analytics see /v1/responses
            // spend. Pre-#404 the responses handler dropped the event
            // entirely — every o1/o3/GPT-5 traffic via Responses API
            // was invisible to budget enforcement and billing
            // reconciliation. Non-streaming-only MVP; streaming
            // emission requires SSE byte-stream interception and is
            // tracked as a follow-up.
            if let Some(usage) = success.usage {
                emit_usage_event(
                    &state,
                    &request_id,
                    &success.model_id,
                    &api_key_id,
                    status,
                    elapsed,
                    &usage,
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
    body: &Value,
    request_id: &str,
) -> Result<ResponseDispatchSuccess, ProxyError> {
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

    // Resolve the attempt list (routing-aware). /v1/responses is
    // OpenAI-only, so we attempt the group's OpenAI targets in order; a
    // direct model resolves to itself (#471).
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

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let not_openai = || {
        ProxyError::InvalidRequest(format!(
            "model `{model_name}` is not an OpenAI provider; /v1/responses requires OpenAI"
        ))
    };

    // Streaming attempts the first eligible target only (no mid-stream
    // fallback), matching /v1/chat/completions and /v1/messages.
    if is_stream {
        let target = attempt_models
            .iter()
            .find(|t| t.model.provider.as_deref() == Some("openai"))
            .ok_or_else(not_openai)?;
        return responses_to_target(
            state,
            &snapshot,
            body,
            &target.model,
            &target.id,
            request_id,
        )
        .await;
    }

    let mut last_err: Option<ProxyError> = None;
    let mut any_openai = false;
    for target in &attempt_models {
        if target.model.provider.as_deref() != Some("openai") {
            continue;
        }
        any_openai = true;
        match responses_to_target(
            state,
            &snapshot,
            body,
            &target.model,
            &target.id,
            request_id,
        )
        .await
        {
            Ok(success) => return Ok(success),
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

    if !any_openai {
        return Err(not_openai());
    }
    Err(last_err.unwrap_or(ProxyError::ProviderUnavailable))
}

/// Dispatch one concrete OpenAI target's Responses-API passthrough to
/// `{api_base}/v1/responses`. The caller has already confirmed
/// `model.provider == openai`.
async fn responses_to_target(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    body: &Value,
    model: &aisix_core::Model,
    model_id: &str,
    request_id: &str,
) -> Result<ResponseDispatchSuccess, ProxyError> {
    let mut body = body.clone();
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
    let api_key = crate::dispatch::require_secret(&pk_entry.value, model)?.to_string();
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();

    // Rewrite model field to upstream name.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model.clone());
    }

    let base = crate::dispatch::resolve_base_url(&pk_entry.value)?;
    // build_v1_url tolerates both `https://api.openai.com` (provider
    // default) and `https://api.openai.com/v1` (the OpenAI-SDK form
    // the dashboard's provider-keys placeholder pre-fills). Without
    // it, the SDK convention double-prefixes to `/v1/v1/responses`
    // and 404s.
    let url = crate::dispatch::build_v1_url(&base, "/responses");

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let client = crate::http_client::client();
    let upstream_resp = client
        .post(&url)
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .header("x-aisix-request-id", request_id)
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
        let err = aisix_gateway::BridgeError::upstream_status_with_retry_after(
            status_u16,
            message.chars().take(1024).collect::<String>(),
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

    let provider_label = "openai".to_string();

    if is_stream {
        let headers = upstream_resp.headers().clone();
        let body_stream = upstream_resp.bytes_stream();

        let mut response =
            axum::response::Response::new(axum::body::Body::from_stream(body_stream));

        if let Some(ct) = headers.get("content-type") {
            if let Ok(hv) = HeaderValue::from_bytes(ct.as_bytes()) {
                response
                    .headers_mut()
                    .insert(axum::http::header::CONTENT_TYPE, hv);
            }
        }

        if let Ok(hv) = HeaderValue::from_str(request_id) {
            response
                .headers_mut()
                .insert(HeaderName::from_static("x-aisix-request-id"), hv);
        }

        // Streaming path passes the upstream byte stream verbatim;
        // the gateway doesn't parse SSE chunks here today, so we
        // can't extract the usage block from `response.completed`
        // without a stream wrapper. Emission for the streaming
        // path is tracked as a #404 follow-up (see PR body).
        Ok(ResponseDispatchSuccess {
            response,
            provider: provider_label,
            usage: None,
            model_id: model_id.to_string(),
        })
    } else {
        let json_body: Value = upstream_resp
            .json()
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

        // Extract the upstream-reported usage block for telemetry
        // emission. Pulled here (before the response is moved into
        // `Json::into_response`) so the success struct can carry
        // typed counters rather than re-parsing JSON downstream.
        let usage = extract_response_usage(&json_body);

        Ok(ResponseDispatchSuccess {
            response: Json(json_body).into_response(),
            provider: provider_label,
            usage,
            model_id: model_id.to_string(),
        })
    }
}

/// Pull the usage counters out of a Responses-API non-streaming
/// response body. Returns `None` when:
///   - The `usage` block is missing entirely, OR
///   - `usage.input_tokens` is missing / non-numeric
///
/// Both cases skip UsageEvent emission rather than attributing a
/// zero-everything noise row to the api_key. Per the OpenAI
/// Responses API spec, `input_tokens` is required on every
/// successful non-streaming 200 — its absence (e.g. `usage: {}`)
/// is upstream-malformed, not a legitimate zero-spend reply.
/// Audit MEDIUM-1 on this PR. Spec:
/// <https://platform.openai.com/docs/api-reference/responses/object>
fn extract_response_usage(body: &Value) -> Option<ResponseUsage> {
    let usage = body.get("usage")?;
    // input_tokens and output_tokens are both required on a
    // spec-compliant 200. Gate emit on each so a malformed reply
    // missing the output side is skipped rather than under-billed with
    // a 0 (#429, audit MEDIUM-1).
    let prompt_tokens = usage.get("input_tokens").and_then(|v| v.as_u64())? as u32;
    let completion_tokens = usage.get("output_tokens").and_then(|v| v.as_u64())? as u32;
    let reasoning_tokens = usage
        .get("output_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let cached_prompt_tokens = usage
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    Some(ResponseUsage {
        prompt_tokens,
        completion_tokens,
        reasoning_tokens,
        cached_prompt_tokens,
    })
}

/// Issue #404: push one `UsageEvent` onto cp-api's telemetry sink
/// and fan it out to per-env OTLP exporters. Mirrors the shape of
/// `embeddings::emit_usage_event` (#402) for the fields that matter
/// to /v1/responses, with one extension: `reasoning_tokens` is
/// surfaced for o1/o3/GPT-5 class models. `inbound_protocol` is
/// `"openai"` — Responses API is OpenAI-only.
///
/// Other fields left at `UsageEvent::default()`:
///   - cache_creation_tokens / cache_read_tokens — Anthropic-only
///   - provider_request_id / provider_model_version / finish_reason
///     — not yet plumbed for non-chat handlers (follow-up)
///   - cost_usd — cp-api computes server-side from pricing catalog
///   - cache_status / cache_hit_* / ttft_ms — no caching/streaming
///     surface on Responses API non-streaming
///   - served_by_model / routing_* — Responses doesn't run routing
///   - provider_kind / provider_featured / branded_provider /
///     pk_label / byo_label — per-PK telemetry attribution wired
///     for chat only today (same deferred gap as #402)
fn emit_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    api_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    usage: &ResponseUsage,
) {
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        cached_prompt_tokens: usage.cached_prompt_tokens,
        reasoning_tokens: usage.reasoning_tokens,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "openai".to_string(),
        ..Default::default()
    };
    state.usage_sink.try_emit("responses", event.clone());
    let snap = state.snapshot.load();
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, exporters.iter().map(|e| &e.value));
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
        path: "/v1/responses",
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
        routing_attempts: None,
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
    use tower::ServiceExt;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            tls: None,
        }
    }

    const OPENAI_PK_ID: &str = "11111111-1111-1111-1111-111111111111";
    const ANTHROPIC_PK_ID: &str = "22222222-2222-2222-2222-222222222222";

    fn openai_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"openai","model_name":"gpt-4o","provider_key_id":"{OPENAI_PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn anthropic_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"anthropic","model_name":"claude-3-haiku-20240307","provider_key_id":"{ANTHROPIC_PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-2", m, 1)
    }

    fn openai_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-test","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(OPENAI_PK_ID, pk, 1)
    }

    fn anthropic_pk() -> ResourceEntry<aisix_core::ProviderKey> {
        let pk: aisix_core::ProviderKey =
            serde_json::from_str(r#"{"display_name":"anthropic-up","secret":"sk-ant-test","provider":"anthropic","adapter":"anthropic"}"#)
                .unwrap();
        ResourceEntry::new(ANTHROPIC_PK_ID, pk, 1)
    }

    fn new_snap_openai(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(openai_pk(api_base));
        snap
    }

    fn new_snap_anthropic() -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk());
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
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    fn make_req(body: serde_json::Value) -> Request<axum::body::Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn unauthenticated_returns_401() {
        let snap = new_snap_openai("http://unused");
        let app = build_app(snap);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"model":"m","input":"hi"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn unknown_model_returns_404() {
        let snap = new_snap_openai("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "no-such-model",
                "input": "hello"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn non_openai_model_returns_400() {
        let snap = new_snap_anthropic();
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "input": "hello"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn happy_path_forwards_to_upstream() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_abc",
                "object": "response",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}]
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "Hello"
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "response");
    }

    #[tokio::test]
    async fn upstream_error_returns_502() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "Hello"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    /// Issue #404: a successful non-streaming /v1/responses call must
    /// emit a `UsageEvent` onto the `usage_sink`. Pre-#404 the
    /// responses handler dropped the event entirely, so every
    /// o1/o3/GPT-5 traffic through Responses API was invisible to
    /// cp-api's budget ledger and customer-facing /logs analytics.
    /// This test pins the contract: after a 200 with a real
    /// upstream usage block, exactly one event arrives with the
    /// input_tokens / output_tokens / reasoning_tokens / cached
    /// counters and `inbound_protocol = "openai"`.
    #[tokio::test]
    async fn emits_usage_event_on_200_non_streaming_issue_404() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // Responses-API wire shape. Pin specific token counts so a
        // regression that swapped semantics (input vs output) or
        // dropped reasoning_tokens would fail here. Mirrors the
        // canonical OpenAI Responses API response object.
        let upstream_body = serde_json::json!({
            "id": "resp-abc",
            "object": "response",
            "model": "gpt-4o-2024-08-06",
            "output": [{
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}]
            }],
            "usage": {
                "input_tokens": 17,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens": 23,
                "output_tokens_details": {"reasoning_tokens": 8},
                "total_tokens": 40
            }
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hello world"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/responses 200")
            .expect("usage_sink sender dropped");

        assert_eq!(
            event.prompt_tokens, 17,
            "prompt_tokens must mirror upstream usage.input_tokens",
        );
        assert_eq!(
            event.completion_tokens, 23,
            "completion_tokens must mirror upstream usage.output_tokens",
        );
        assert_eq!(
            event.reasoning_tokens, 8,
            "reasoning_tokens must mirror usage.output_tokens_details.reasoning_tokens \
             (o1/o3/GPT-5 class models)",
        );
        assert_eq!(
            event.cached_prompt_tokens, 5,
            "cached_prompt_tokens must mirror usage.input_tokens_details.cached_tokens",
        );
        assert_eq!(event.status_code, 200);
        assert_eq!(event.api_key_id, "k-1");
        assert_eq!(event.model_id, "m-1");
        assert_eq!(event.inbound_protocol, "openai");
        assert!(!event.request_id.is_empty());
        assert!(!event.occurred_at.is_empty());
    }

    /// Companion: an upstream response missing the `usage` block
    /// entirely (some edge / error response shapes) must NOT emit
    /// — there's nothing meaningful to attribute. Pre-#404 the
    /// handler emitted nothing; we keep that behaviour for this
    /// edge so the api_key isn't credited with zero-everything
    /// noise rows.
    #[tokio::test]
    async fn skips_usage_event_when_upstream_omits_usage_block() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // 200 OK but no `usage` field — should NOT emit.
        let upstream_body = serde_json::json!({
            "id": "resp-abc",
            "object": "response",
            "model": "gpt-4o-2024-08-06",
            "output": []
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hello"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Wait briefly — no event should arrive. `Ok(None)` means
        // the channel closed (state dropped) without sending; `Err`
        // means timeout. Both are acceptable "no event" outcomes;
        // only `Ok(Some(_))` would be a real failure.
        let recv = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await;
        if let Ok(Some(ev)) = recv {
            panic!(
                "expected NO event when usage block absent, but got: \
                 prompt_tokens={}, completion_tokens={}, status_code={}",
                ev.prompt_tokens, ev.completion_tokens, ev.status_code,
            );
        }
    }

    /// Issue #404 streaming follow-up coverage: the streaming path
    /// MUST NOT regress on the non-streaming emission. A streaming
    /// request should pass the byte stream through verbatim and
    /// (today's MVP) skip emission — without crashing or hanging.
    /// Streaming emission requires SSE byte-stream interception
    /// and is tracked as a separate follow-up; this test pins the
    /// "no regression, no emission today" contract.
    #[tokio::test]
    async fn streaming_path_does_not_emit_today_but_passes_through() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // Minimal SSE-style body. Real Responses-API streaming uses
        // `response.completed` events with a usage block in the
        // final chunk; the gateway doesn't parse these today.
        let sse_body = "data: {\"type\":\"response.created\",\"response\":{}}\n\ndata: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hi",
                "stream": true
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Drain the body and pin its shape — audit LOW-2 on this
        // PR. A regression that dropped headers in the refactored
        // `ResponseDispatchSuccess` path could surface as a 200
        // with an empty body; the SSE prefix check guards against
        // the byte-stream being silently rewritten.
        let body_bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            !body_bytes.is_empty(),
            "streaming body must pass through verbatim",
        );
        assert!(
            body_bytes.starts_with(b"data: "),
            "SSE shape must survive the refactor",
        );

        // Streaming path MVP: no UsageEvent emitted today. Follow-up
        // tracked to wire SSE byte-stream interception. `Ok(None)`
        // = channel closed (state dropped); `Err` = timeout. Both
        // are "no event"; only `Ok(Some(_))` is a real failure.
        let recv = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await;
        if let Ok(Some(ev)) = recv {
            panic!(
                "streaming /v1/responses must not emit UsageEvent yet \
                 (follow-up tracked), but got prompt_tokens={}",
                ev.prompt_tokens,
            );
        }
    }

    /// Issue #404 negative pinning: 5xx responses must NOT emit
    /// a UsageEvent. Audit MEDIUM-2 on this PR — without negative
    /// pinning, a future regression that moved `emit_usage_event`
    /// into the error branch would silently ship.
    #[tokio::test]
    async fn upstream_5xx_does_not_emit_usage_event() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal"))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hello"
            })))
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

    /// Issue #404 audit MEDIUM-1: a 200 with `usage: {}` (malformed
    /// — `input_tokens` is required by the Responses-API spec) must
    /// NOT emit a zero-everything noise row. Same edge as the
    /// `skips_usage_event_when_upstream_omits_usage_block` test but
    /// the gate is one layer deeper — `usage` exists but is empty.
    #[tokio::test]
    async fn skips_usage_event_when_usage_block_is_empty_audit_m1() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "id": "resp-abc",
            "object": "response",
            "output": [],
            "usage": {}  // malformed — input_tokens required by spec
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hi"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let recv = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await;
        if let Ok(Some(ev)) = recv {
            panic!(
                "no UsageEvent should be emitted for malformed `usage: {{}}`, \
                 got prompt_tokens={}",
                ev.prompt_tokens,
            );
        }
    }

    /// #429: a 200 whose `usage` carries `input_tokens` but omits the
    /// required `output_tokens` is malformed — emitting it would
    /// under-bill the output side with a 0. It must be skipped.
    #[tokio::test]
    async fn skips_usage_event_when_output_tokens_missing() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "id": "resp-abc",
            "object": "response",
            "output": [],
            "usage": { "input_tokens": 17 }  // missing output_tokens
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "hi"
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let recv = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv()).await;
        if let Ok(Some(ev)) = recv {
            panic!(
                "no UsageEvent should be emitted when output_tokens is missing, \
                 got prompt_tokens={}",
                ev.prompt_tokens,
            );
        }
    }
}
