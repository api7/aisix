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

use aisix_obs::{AccessLog, RequestOutcome};
use axum::body::Bytes;
use axum::extract::{Multipart, State};
use axum::http::{header, HeaderMap};
use axum::response::{IntoResponse, Response};
use axum::Json;
use reqwest::multipart;
use serde_json::Value;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

// ─────────────────────────────────────────────────────────────────────────────
// /v1/audio/transcriptions
// ─────────────────────────────────────────────────────────────────────────────

pub async fn transcriptions(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    multipart: Multipart,
) -> Response {
    let started = Instant::now();
    let request_id = format!("atr-{}", Uuid::new_v4());
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
        Ok((resp, model_name, provider)) => {
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/transcriptions",
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
            resp
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
    multipart: Multipart,
) -> Response {
    let started = Instant::now();
    let request_id = format!("atr-{}", Uuid::new_v4());
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
        Ok((resp, model_name, provider)) => {
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/translations",
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
            resp
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
    Json(body): Json<Value>,
) -> Response {
    let started = Instant::now();
    let request_id = format!("asp-{}", Uuid::new_v4());
    let api_key_id = auth.entry.id.clone();
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    match speech_dispatch(&state, &auth, body, &request_id).await {
        Ok((resp, provider)) => {
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
) -> Result<(Response, String, String), ProxyError> {
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

    let base = crate::dispatch::resolve_base_url(provider, &pk_entry.value);
    // build_v1_url owns the /v1 prefix; callers pass the suffix
    // (e.g. `/audio/transcriptions`) so this code is agnostic to
    // whether the customer's api_base ends in /v1 or not.
    let url = crate::dispatch::build_v1_url(&base, upstream_path);
    let provider_label = format!("{provider:?}").to_lowercase();

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

    let mut out = axum::response::Response::new(axum::body::Body::from(body_bytes));
    copy_response_header(&upstream_headers, &mut out, header::CONTENT_TYPE);
    Ok((out, model_name, provider_label))
}

/// JSON passthrough for `/v1/audio/speech` — returns binary audio bytes.
async fn speech_dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    mut body: Value,
    request_id: &str,
) -> Result<(Response, String), ProxyError> {
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

    let base = crate::dispatch::resolve_base_url(provider, &pk_entry.value);
    let provider_label = format!("{provider:?}").to_lowercase();

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
    Ok((out, provider_label))
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
    }
    .emit();
}

// The audio handler reuses the same client as messages.rs. It's exported
// from there to avoid creating multiple global Clients.
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
            request_body_limit_bytes: 10_485_760, // 10 MB for audio
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
}
