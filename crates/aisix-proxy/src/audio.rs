//! `POST /v1/audio/{transcriptions,translations,speech}` — audio API
//! pass-through.
//!
//! Three sub-endpoints with different request shapes:
//!
//! * **transcriptions** & **translations** — `multipart/form-data` with an
//!   audio `file`, a `model` field, and optional metadata fields.
//!   The gateway resolves the model name, swaps in the upstream model id,
//!   and re-assembles the multipart form before forwarding.
//!
//! * **speech** — JSON body `{model, input, voice, …}`.
//!   Standard JSON passthrough, identical to `/v1/completions`.
//!
//! In all cases the upstream response is returned verbatim: JSON for
//! transcription/translation results, binary audio bytes for speech.
//!
//! Auth and model authorisation follow the same rules as every other
//! proxy endpoint.

use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::body::Bytes;
use axum::extract::{Multipart, State};
use axum::http::{header, HeaderMap};
use axum::response::{IntoResponse, Response};
use axum::Json;
use reqwest::multipart;
use serde_json::Value;
use std::time::{Duration, Instant};

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::ProxyError;
use crate::request_id::new_request_id;
use crate::state::ProxyState;

/// Per-request payload from a successful multipart dispatch
/// (transcriptions/translations) — adds `model_id` + parsed `usage` to
/// the response/model/provider triplet so the handler can emit a
/// UsageEvent (#406).
struct AudioDispatchSuccess {
    response: Response,
    model_name: String,
    provider: String,
    model_id: String,
    /// `(prompt_tokens, completion_tokens)` from the upstream `usage`
    /// block when the model returns one (gpt-4o-transcribe). `None` for
    /// whisper-1 (no usage block) — those still emit a zero-token event
    /// so the request is visible + attributed.
    usage: Option<(u32, u32)>,
}

// ─────────────────────────────────────────────────────────────────────────────
// /v1/audio/transcriptions
// ─────────────────────────────────────────────────────────────────────────────

pub async fn transcriptions(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    multipart: Multipart,
) -> Response {
    let started = Instant::now();
    let request_id = new_request_id();
    let api_key_id = auth.entry.id.clone();

    match multipart_dispatch(
        &state,
        &auth,
        multipart,
        // Version-independent path — multipart_dispatch's URL builder
        // (build_v1_url) owns the `/v1` prefix.
        "/audio/transcriptions",
        &request_id,
    )
    .await
    {
        Ok(success) => {
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/transcriptions",
                &success.model_name,
                &success.provider,
                &api_key_id,
                200,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &success.provider,
                &success.model_name,
                200,
                RequestOutcome::Success,
                elapsed,
            );
            emit_audio_usage(&state, &request_id, &success, &api_key_id, elapsed, &client);
            success.response
        }
        Err(err) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/transcriptions",
                "unknown",
                "unknown",
                &api_key_id,
                status,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                "unknown",
                "unknown",
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            err.into_response()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// /v1/audio/translations
// ─────────────────────────────────────────────────────────────────────────────

pub async fn translations(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    multipart: Multipart,
) -> Response {
    let started = Instant::now();
    let request_id = new_request_id();
    let api_key_id = auth.entry.id.clone();

    match multipart_dispatch(
        &state,
        &auth,
        multipart,
        // Version-independent path — multipart_dispatch's URL builder
        // (build_v1_url) owns the `/v1` prefix.
        "/audio/translations",
        &request_id,
    )
    .await
    {
        Ok(success) => {
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/translations",
                &success.model_name,
                &success.provider,
                &api_key_id,
                200,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &success.provider,
                &success.model_name,
                200,
                RequestOutcome::Success,
                elapsed,
            );
            emit_audio_usage(&state, &request_id, &success, &api_key_id, elapsed, &client);
            success.response
        }
        Err(err) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/translations",
                "unknown",
                "unknown",
                &api_key_id,
                status,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                "unknown",
                "unknown",
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            err.into_response()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// /v1/audio/speech
// ─────────────────────────────────────────────────────────────────────────────

pub async fn speech(
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

    match speech_dispatch(&state, &auth, body, &request_id).await {
        Ok((resp, provider, model_id)) => {
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/speech",
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
            // Issue #406: /v1/audio/speech (TTS) returns binary audio
            // with no usage block — emit a zero-token UsageEvent so the
            // request is visible in /logs and attributed to the api_key.
            // (TTS is billed per input character; that cost basis is the
            // same cross-repo follow-up as audio duration.)
            emit_usage_event(
                &state,
                &request_id,
                &model_id,
                &api_key_id,
                200,
                elapsed,
                0,
                0,
                &client,
            );
            resp
        }
        Err(err) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/speech",
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

// ─────────────────────────────────────────────────────────────────────────────
// Shared dispatch functions
// ─────────────────────────────────────────────────────────────────────────────

/// Collect all multipart fields, resolve the model, swap in the upstream
/// model id, then rebuild and forward the multipart form.
async fn multipart_dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    mut multipart: Multipart,
    upstream_path: &str,
    request_id: &str,
) -> Result<AudioDispatchSuccess, ProxyError> {
    // Collect all fields first so we can find `model` before building the
    // outgoing reqwest multipart.
    let mut fields: Vec<(String, Option<String>, Option<String>, Bytes)> = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ProxyError::InvalidRequest(format!("multipart read error: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        let file_name = field.file_name().map(|s| s.to_string());
        let content_type = field.content_type().map(|s| s.to_string());
        let data = field
            .bytes()
            .await
            .map_err(|e| ProxyError::InvalidRequest(format!("multipart field read error: {e}")))?;
        fields.push((name, file_name, content_type, data));
    }

    // Extract the `model` field value.
    let model_name = fields
        .iter()
        .find(|(name, ..)| name == "model")
        .and_then(|(.., data)| std::str::from_utf8(data).ok())
        .map(|s| s.trim().to_string())
        .ok_or_else(|| ProxyError::InvalidRequest("`model` field missing from form".into()))?;

    let snapshot = state.snapshot.load();
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
    let provider = crate::dispatch::require_provider(model)?;
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();
    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;
    let api_key = crate::dispatch::require_secret(&pk_entry.value, model)?;

    let base = crate::dispatch::resolve_base_url(&pk_entry.value)?;
    // build_v1_url owns the /v1 prefix; callers pass the suffix
    // (e.g. `/audio/transcriptions`) so this code is agnostic to
    // whether the customer's api_base ends in /v1 or not.
    let url = crate::dispatch::build_v1_url(&base, upstream_path);
    let provider_label = provider.to_ascii_lowercase();

    // Rebuild the multipart form with `model` rewritten.
    let mut form = multipart::Form::new();
    for (name, file_name, content_type, data) in fields {
        let field_data = if name == "model" {
            Bytes::copy_from_slice(upstream_model.as_bytes())
        } else {
            data
        };

        let data_vec = field_data.to_vec();
        let mut part = if let Some(ct) = content_type {
            multipart::Part::bytes(data_vec.clone())
                .mime_str(&ct)
                .unwrap_or_else(|_| multipart::Part::bytes(data_vec))
        } else {
            multipart::Part::bytes(data_vec)
        };
        if let Some(fname) = file_name {
            part = part.file_name(fname);
        }
        form = form.part(name, part);
    }

    let client = crate::http_client::client();
    let resp = client
        .post(&url)
        .header(header::AUTHORIZATION, format!("Bearer {api_key}"))
        .header("x-aisix-request-id", request_id)
        .multipart(form)
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

    let status = resp.status();
    if !status.is_success() {
        let s = status.as_u16();
        let retry_after = aisix_gateway::parse_retry_after(resp.headers());
        let msg = resp.text().await.unwrap_or_default();
        let err = aisix_gateway::BridgeError::upstream_status_with_retry_after(
            s,
            msg.chars().take(1024).collect::<String>(),
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

    // Relay response headers that matter for the client.
    let upstream_headers = resp.headers().clone();
    let body_bytes = resp
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

    // Parse the response body best-effort for a `usage` token block
    // (gpt-4o-transcribe returns one; whisper-1 returns none, and the
    // `text`/`srt`/`vtt` response_formats aren't JSON at all). Parse
    // failure or absence → None → zero-token emit. Done before the
    // bytes move into the Body.
    let usage = serde_json::from_slice::<Value>(&body_bytes)
        .ok()
        .as_ref()
        .and_then(extract_token_usage);

    let mut out = axum::response::Response::new(axum::body::Body::from(body_bytes));
    copy_response_header(&upstream_headers, &mut out, header::CONTENT_TYPE);
    Ok(AudioDispatchSuccess {
        response: out,
        model_name,
        provider: provider_label,
        model_id: model_entry.id.to_string(),
        usage,
    })
}

/// JSON passthrough for `/v1/audio/speech` — returns binary audio bytes.
/// Returns `(response, provider_label, model_id)`; `model_id` lets the
/// handler attribute a (zero-token) UsageEvent (#406).
async fn speech_dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    mut body: Value,
    request_id: &str,
) -> Result<(Response, String, String), ProxyError> {
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::InvalidRequest("missing `model` field".into()))?
        .to_string();

    let snapshot = state.snapshot.load();
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
    let provider = crate::dispatch::require_provider(model)?;
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();
    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;
    let api_key = crate::dispatch::require_secret(&pk_entry.value, model)?;

    let base = crate::dispatch::resolve_base_url(&pk_entry.value)?;
    let provider_label = provider.to_ascii_lowercase();

    // Rewrite model field.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model);
    }

    let client = crate::http_client::client();
    let resp = client
        .post(crate::dispatch::build_v1_url(&base, "/audio/speech"))
        .header(header::AUTHORIZATION, format!("Bearer {api_key}"))
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-aisix-request-id", request_id)
        .json(&body)
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

    let status = resp.status();
    if !status.is_success() {
        let s = status.as_u16();
        let retry_after = aisix_gateway::parse_retry_after(resp.headers());
        let msg = resp.text().await.unwrap_or_default();
        let err = aisix_gateway::BridgeError::upstream_status_with_retry_after(
            s,
            msg.chars().take(1024).collect::<String>(),
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

    let upstream_headers = resp.headers().clone();
    let body_bytes = resp
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

    let mut out = axum::response::Response::new(axum::body::Body::from(body_bytes));
    copy_response_header(&upstream_headers, &mut out, header::CONTENT_TYPE);
    Ok((out, provider_label, model_entry.id.to_string()))
}

/// Pull `(prompt_tokens, completion_tokens)` from an audio response
/// `usage` block. gpt-4o-transcribe returns
/// `usage: {type:"tokens", input_tokens, output_tokens, ...}`;
/// whisper-1 (and the `text`/`srt`/`vtt` response formats) return no
/// token block → `None`. Spec:
/// <https://platform.openai.com/docs/api-reference/audio/json-object>
fn extract_token_usage(body: &Value) -> Option<(u32, u32)> {
    let usage = body.get("usage")?;
    let input = usage.get("input_tokens").and_then(Value::as_u64)? as u32;
    let output = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    Some((input, output))
}

/// Emit a UsageEvent for a successful transcription/translation. Tokens
/// come from the upstream `usage` block when present (gpt-4o-transcribe);
/// zero otherwise (whisper-1) — the request is still visible/attributed.
fn emit_audio_usage(
    state: &ProxyState,
    request_id: &str,
    success: &AudioDispatchSuccess,
    api_key_id: &str,
    elapsed: Duration,
    client: &ClientContext,
) {
    let (prompt_tokens, completion_tokens) = success.usage.unwrap_or((0, 0));
    emit_usage_event(
        state,
        request_id,
        &success.model_id,
        api_key_id,
        200,
        elapsed,
        prompt_tokens,
        completion_tokens,
        client,
    );
}

/// Issue #406: push one `UsageEvent` onto cp-api's telemetry sink and
/// fan it out to per-env OTLP exporters. Mirrors
/// `embeddings::emit_usage_event` (#402). `inbound_protocol = "openai"`.
/// Tokens are populated when the upstream returned a `usage` block
/// (gpt-4o-transcribe); zero otherwise — duration-based cost (whisper-1)
/// is a documented cross-repo follow-up (needs duration on the wire +
/// cp-api pricing).
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
    // Handler label "audio" — bucketed prometheus counter (#408).
    state.usage_sink.try_emit("audio", event.clone());
    let snap = state.snapshot.load();
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
}

fn copy_response_header(src: &HeaderMap, dst: &mut Response, name: header::HeaderName) {
    if let Some(val) = src.get(&name) {
        dst.headers_mut().insert(name, val.clone());
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_access_log(
    method: &'static str,
    path: &'static str,
    model: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    latency: Duration,
    request_id: &str,
) {
    AccessLog {
        method,
        path,
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
        routing_attempts: None,
    }
    .emit();
}

// The audio handler reuses the same client as messages.rs. It's exported
// from there to avoid creating multiple global Clients.
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
            request_body_limit_bytes: 10_485_760, // 10 MB for audio
            real_ip: Default::default(),
            tls: None,
        }
    }

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn whisper_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "whisper-1",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn tts_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "tts-1",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-2", m, 1)
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

    #[tokio::test]
    async fn speech_unauthenticated_returns_401() {
        let snap = new_snap("http://unused");
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"my-tts","input":"Hello","voice":"alloy"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn speech_unknown_model_returns_404() {
        let snap = new_snap("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"nonexistent","input":"Hello","voice":"alloy"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn speech_happy_path_returns_audio_bytes() {
        let upstream = MockServer::start().await;
        // TTS endpoint returns raw MP3 bytes.
        let fake_mp3 = b"ID3\x03\x00\x00\x00";
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(fake_mp3.to_vec()),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"my-tts","input":"Hello","voice":"alloy"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.contains("audio"),
            "expected audio content-type, got {ct}"
        );
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(&bytes[..3], b"ID3");
    }

    #[tokio::test]
    async fn transcriptions_unauthenticated_returns_401() {
        let snap = new_snap("http://unused");
        snap.models.insert(whisper_model("my-whisper"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        // A minimal multipart body.
        let body = "--boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nmy-whisper\r\n--boundary--\r\n";
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", "multipart/form-data; boundary=boundary")
            .body(axum::body::Body::from(body))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    fn build_app_with_sink(
        snap: AisixSnapshot,
        tx: tokio::sync::mpsc::Sender<aisix_obs::UsageEvent>,
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

    /// A minimal multipart body carrying `model` + a tiny fake audio
    /// `file` field — enough for the gateway to extract the model and
    /// forward the form.
    fn transcription_multipart(model: &str) -> (String, axum::body::Body) {
        let body = format!(
            "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n\
             --b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\n\
             Content-Type: audio/mpeg\r\n\r\nID3fakeaudio\r\n--b--\r\n"
        );
        (
            "multipart/form-data; boundary=b".to_string(),
            axum::body::Body::from(body),
        )
    }

    /// Issue #406: gpt-4o-transcribe returns a `usage` token block —
    /// a successful transcription must emit a UsageEvent with those
    /// tokens, attributed to the api_key + model, inbound_protocol
    /// "openai".
    #[tokio::test]
    async fn transcriptions_emit_usage_event_with_tokens() {
        let upstream = MockServer::start().await;
        let body = serde_json::json!({
            "text": "hello world",
            "usage": {
                "type": "tokens",
                "input_tokens": 14,
                "output_tokens": 4,
                "total_tokens": 18
            }
        });
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-transcribe"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = transcription_multipart("my-transcribe");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/audio/transcriptions 200")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 14);
        assert_eq!(event.completion_tokens, 4);
        assert_eq!(event.status_code, 200);
        assert_eq!(event.api_key_id, "k-1");
        assert_eq!(event.model_id, "m-1");
        assert_eq!(event.inbound_protocol, "openai");
    }

    /// Issue #406: whisper-1 `{"text":"..."}` has no `usage` block —
    /// the request still emits a zero-token UsageEvent so it's visible
    /// in /logs and attributed (duration-based cost is a cross-repo
    /// follow-up).
    #[tokio::test]
    async fn transcriptions_emit_zero_token_event_without_usage() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"text": "hi"})),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-whisper"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = transcription_multipart("my-whisper");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("zero-token UsageEvent must still be emitted (visibility)")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 0);
        assert_eq!(event.completion_tokens, 0);
        assert_eq!(event.model_id, "m-1");
        assert_eq!(event.inbound_protocol, "openai");
    }

    /// Issue #406: TTS speech returns binary audio (no usage). It still
    /// emits a zero-token UsageEvent so the request is visible +
    /// attributed.
    #[tokio::test]
    async fn speech_emits_zero_token_usage_event() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3\x03\x00\x00\x00".to_vec()),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"my-tts","input":"Hello","voice":"alloy"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("speech must emit a zero-token UsageEvent (visibility)")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 0);
        assert_eq!(event.completion_tokens, 0);
        assert_eq!(event.model_id, "m-2");
        assert_eq!(event.inbound_protocol, "openai");
    }

    /// gpt-4o-transcribe with a non-default `response_format` can return
    /// the *duration* usage variant — `{"type":"duration","seconds":N}` —
    /// which carries no `input_tokens`. `extract_token_usage` must degrade
    /// that to `None` (→ a zero-token emit, never a panic or mis-parse),
    /// consistent with the duration-cost being a cross-repo follow-up.
    /// Per OpenAI's `usage` oneOf (TranscriptTextUsageTokens |
    /// TranscriptTextUsageDuration):
    /// <https://platform.openai.com/docs/api-reference/audio/json-object>
    #[test]
    fn extract_token_usage_ignores_duration_variant() {
        let v = serde_json::json!({
            "text": "hello world",
            "usage": {"type": "duration", "seconds": 42.7}
        });
        assert_eq!(super::extract_token_usage(&v), None);
    }
}
