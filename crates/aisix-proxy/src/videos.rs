//! `/v1/videos` — unified video-generation surface (AISIX-Cloud#1118 Phase 1).
//!
//! Three typed routes following the submit → poll → fetch contract
//! established by the upstream videos API
//! (<https://platform.openai.com/docs/api-reference/videos>; field shape
//! pinned against the official `openai-python` SDK `types/video.py`):
//!
//! 1. `POST /v1/videos` — submit a generation task. Returns the video
//!    job object (`object: "video"`, `status: "queued"`).
//! 2. `GET /v1/videos/:id` — poll the job. Statuses normalise to the
//!    four-value enum `queued` / `in_progress` / `completed` / `failed`.
//! 3. `GET /v1/videos/:id/content` — 302 redirect to the provider's
//!    video URL once the task has succeeded.
//!
//! **Stateless task addressing**: the gateway persists nothing. The
//! caller-visible id is `base64url_nopad("<model_entry_id>:<upstream_task_id>")`
//! — the GET routes decode it, re-resolve the Model row (and thus the
//! ProviderKey credential) from the live snapshot, and call the
//! provider's task endpoint. Same encode-routing-into-the-id approach as
//! the jobs surface (`jobs::encode_routed_id`), minus the `aisix-` prefix
//! since video ids never round-trip into other request bodies.
//!
//! **Phase 1 provider coverage**: Alibaba DashScope (Wan), per
//! <https://help.aliyun.com/zh/model-studio/text-to-video-api-reference>:
//! submit `POST {api_base}/api/v1/services/aigc/video-generation/video-synthesis`
//! with the mandatory `X-DashScope-Async: enable` header, poll
//! `GET {api_base}/api/v1/tasks/{task_id}`. Other providers on these
//! routes return 501 `not_implemented` (same convention as the
//! embeddings handler's unsupported-provider branch).
//!
//! **Rate limiting**: submit enforces the model-level layers exactly like
//! chat / embeddings. The two GET routes deliberately pass `None` for the
//! model layer — normal client polling must not burn the model's RPM
//! (AISIX-Cloud#1118 decision 3). Key-level layers still apply.
//!
//! **Usage**: one zero-token UsageEvent per submit (mirrors the
//! passthrough / jobs convention). Per-second cost accounting is a
//! control-plane follow-up; the GET routes emit no usage events by
//! design (poll traffic would flood /logs with no billing signal).

use aisix_core::AppliedGuardrail;
use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::{ErrorEnvelope, ProxyError};
use crate::state::ProxyState;

/// DashScope video-synthesis submit path (relative to the ProviderKey's
/// `api_base`). Source: Alibaba Model Studio text-to-video API reference.
const DASHSCOPE_SUBMIT_PATH: &str = "/api/v1/services/aigc/video-generation/video-synthesis";
/// DashScope async task poll path prefix.
const DASHSCOPE_TASK_PATH: &str = "/api/v1/tasks";

// ─────────────────────────── id codec ───────────────────────────

/// Encode `(model_entry_id, requested_alias, upstream_task_id)` into the
/// caller-visible video id:
/// `base64url_nopad("<model_entry_id>:<base64url_nopad(alias)>:<task_id>")`.
///
/// The alias segment carries the model name the caller REQUESTED at
/// submit time (which, for wildcard Models like `wan/*`, differs from
/// the stored display name). The GET routes re-run the key ACL against
/// this alias — the identical check the submit performed — so a key
/// allowlisted for `wan/turbo` can poll the task it submitted even
/// though it cannot access the literal `wan/*` display name. The alias
/// is itself base64url-encoded so it can never collide with the `:`
/// separators regardless of its content.
fn encode_video_id(model_entry_id: &str, alias: &str, task_id: &str) -> String {
    URL_SAFE_NO_PAD.encode(format!(
        "{model_entry_id}:{}:{task_id}",
        URL_SAFE_NO_PAD.encode(alias)
    ))
}

/// Decode a caller-supplied video id back to
/// `(model_entry_id, requested_alias, task_id)`. `None` for anything
/// that isn't a well-formed gateway id — the GET routes surface that as
/// 404 (the id can't name a task this gateway submitted). The first two
/// segments never contain `:` (UUID-shaped entry id, base64url alias),
/// so two `split_once` calls leave the remainder — which MAY contain
/// `:` — as the provider task id.
fn decode_video_id(id: &str) -> Option<(String, String, String)> {
    let bytes = URL_SAFE_NO_PAD.decode(id).ok()?;
    let s = String::from_utf8(bytes).ok()?;
    let (entry_id, rest) = s.split_once(':')?;
    let (alias_b64, task_id) = rest.split_once(':')?;
    let alias = String::from_utf8(URL_SAFE_NO_PAD.decode(alias_b64).ok()?).ok()?;
    if entry_id.is_empty() || alias.is_empty() || task_id.is_empty() {
        return None;
    }
    Some((entry_id.to_string(), alias, task_id.to_string()))
}

// ─────────────────────── status normalisation ───────────────────────

/// Map a DashScope `task_status` onto the four-value status enum of the
/// unified contract. `CANCELED` and `UNKNOWN` (task expired / not found
/// upstream) collapse to `failed` — from the caller's viewpoint the job
/// will never produce a video. Unrecognised strings also map to `failed`
/// rather than leaking provider taxonomy through the typed surface.
fn map_task_status(task_status: &str) -> &'static str {
    match task_status {
        "PENDING" => "queued",
        "RUNNING" => "in_progress",
        "SUCCEEDED" => "completed",
        _ => "failed",
    }
}

// ─────────────────────────── wire shapes ───────────────────────────

/// The request body accepted by `POST /v1/videos`. Field names and
/// semantics follow the upstream videos API (`VideoCreateParams` in the
/// official SDK): `prompt` required, `seconds` a duration in seconds,
/// `size` a `WIDTHxHEIGHT` resolution. `model` is required here (the
/// gateway has no default video model). Unknown fields (e.g.
/// `input_reference`) are ignored in Phase 1.
#[derive(Debug, Deserialize)]
pub struct VideoCreateBody {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub seconds: Option<SecondsField>,
    #[serde(default)]
    pub size: Option<String>,
}

/// The upstream contract types `seconds` as a string enum (`"4"`, `"8"`,
/// `"12"`); accept a bare integer too so curl-first callers aren't
/// punished for the obvious spelling.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SecondsField {
    Int(u64),
    Str(String),
}

impl SecondsField {
    /// Normalise to a positive integer number of seconds.
    fn as_secs(&self) -> Result<u64, ProxyError> {
        let n = match self {
            SecondsField::Int(n) => *n,
            SecondsField::Str(s) => s.trim().parse::<u64>().map_err(|_| {
                ProxyError::InvalidRequest(format!(
                    "`seconds` must be a positive integer, got {s:?}"
                ))
            })?,
        };
        if n == 0 {
            return Err(ProxyError::InvalidRequest(
                "`seconds` must be a positive integer".into(),
            ));
        }
        Ok(n)
    }
}

/// The video job object returned by all three routes — field names per
/// the upstream videos API (`Video` in the official SDK): `id`,
/// `object: "video"`, `model`, `status`, `progress`, `created_at`, plus
/// `seconds` / `size` / `error` when known. Optional fields are omitted
/// (not `null`) when the stateless gateway cannot recover them from the
/// provider's task response.
#[derive(Debug, Serialize)]
struct VideoObject {
    id: String,
    object: &'static str,
    model: String,
    status: &'static str,
    progress: u32,
    /// Unix seconds. Populated at submit time; the DashScope poll
    /// response reports no machine-readable creation timestamp and the
    /// gateway stores nothing, so poll responses carry 0.
    created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    seconds: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<VideoErrorObject>,
}

/// `error` payload on a failed job: `{code, message}` per the upstream
/// contract (`VideoCreateError` in the official SDK).
#[derive(Debug, Serialize)]
struct VideoErrorObject {
    code: String,
    message: String,
}

/// DashScope response envelope (submit and poll share it).
#[derive(Debug, Deserialize)]
struct DashScopeResponse {
    #[serde(default)]
    output: Option<DashScopeOutput>,
    #[serde(default)]
    usage: Option<DashScopeUsage>,
    /// Top-level error code/message on non-2xx responses.
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DashScopeOutput {
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    task_status: Option<String>,
    #[serde(default)]
    video_url: Option<String>,
    /// Task-level failure detail (set when `task_status == "FAILED"`).
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DashScopeUsage {
    #[serde(default)]
    output_video_duration: Option<u64>,
    #[serde(default)]
    duration: Option<u64>,
}

// ─────────────────────── DashScope param mapping ───────────────────────

/// Build the DashScope submit body from the unified request.
///
/// - `seconds` → `parameters.duration` (integer seconds).
/// - `size` `"WIDTHxHEIGHT"` → `parameters.size` `"WIDTH*HEIGHT"` — the
///   explicit-dimension parameter the Wan model families document. (The
///   wan2.7 line replaced `size` with `resolution`/`ratio` quality
///   tiers, which cannot represent an arbitrary WIDTHxHEIGHT without a
///   lossy invented mapping — callers targeting those models should omit
///   `size`; a tier mapping is a documented follow-up.)
/// - Unset params are omitted entirely; `parameters` itself is omitted
///   when empty.
fn dashscope_submit_body(
    upstream_model: &str,
    prompt: &str,
    seconds: Option<u64>,
    size: Option<&str>,
) -> Result<serde_json::Value, ProxyError> {
    let mut parameters = serde_json::Map::new();
    if let Some(secs) = seconds {
        parameters.insert("duration".into(), serde_json::json!(secs));
    }
    if let Some(size) = size {
        parameters.insert("size".into(), serde_json::json!(map_size(size)?));
    }
    let mut body = serde_json::json!({
        "model": upstream_model,
        "input": { "prompt": prompt },
    });
    if !parameters.is_empty() {
        body["parameters"] = serde_json::Value::Object(parameters);
    }
    Ok(body)
}

/// `"1280x720"` → `"1280*720"`. Strict `WIDTHxHEIGHT` digits-only shape;
/// anything else is a 400 before the provider is contacted.
fn map_size(size: &str) -> Result<String, ProxyError> {
    let valid = size
        .split_once('x')
        .is_some_and(|(w, h)| is_dim(w) && is_dim(h));
    if !valid {
        return Err(ProxyError::InvalidRequest(format!(
            "`size` must be formatted as WIDTHxHEIGHT (e.g. \"1280x720\"), got {size:?}"
        )));
    }
    Ok(size.replacen('x', "*", 1))
}

fn is_dim(s: &str) -> bool {
    !s.is_empty() && s.len() <= 5 && s.bytes().all(|b| b.is_ascii_digit())
}

// ─────────────────────── resolved dispatch target ───────────────────────

/// Everything the three handlers need after resolving a Model for the
/// videos surface: entry identity for telemetry, PK credential, base URL.
struct VideoTarget {
    model_entry: std::sync::Arc<aisix_core::ResourceEntry<aisix_core::Model>>,
    pk_id: String,
    provider_label: String,
    base_url: String,
    secret: String,
}

impl VideoTarget {
    fn display_name(&self) -> &str {
        &self.model_entry.value.display_name
    }
}

/// Shared resolve → ACL → IP-allowlist → provider gate for a Model on the
/// videos surface. `acl_name` is the name the caller's key is checked
/// against (the requested alias on submit, the stored display name on the
/// GET routes).
fn resolve_video_target(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    model_entry: std::sync::Arc<aisix_core::ResourceEntry<aisix_core::Model>>,
    acl_name: &str,
    source_ip: &str,
) -> Result<Result<VideoTarget, Response>, ProxyError> {
    let snapshot = state.snapshot.load();

    if !auth.key().can_access(acl_name) {
        return Err(ProxyError::ModelForbidden(acl_name.to_string()));
    }
    crate::dispatch::check_ip_access(&model_entry.value, source_ip)?;

    let provider = crate::dispatch::require_provider(&model_entry.value)?.to_string();
    // Phase 1 covers DashScope only; other providers get a typed 501 so
    // callers can tell "wrong provider" from "wrong request". Zhipu and
    // Volcano mappings are tracked follow-ups on the same surface.
    if !provider.eq_ignore_ascii_case("alibaba") {
        let env = ErrorEnvelope::new(
            format!("provider {provider:?} is not yet supported on /v1/videos"),
            "not_implemented",
        );
        return Ok(Err((StatusCode::NOT_IMPLEMENTED, Json(env)).into_response()));
    }

    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, &model_entry.value)?;
    // There is no built-in default api_base for the `alibaba` vendor —
    // the ProviderKey must carry it (regional DashScope endpoints).
    let base_url = crate::dispatch::resolve_base_url(&pk_entry.value)?;
    let secret = crate::dispatch::require_api_key(&pk_entry.value, &model_entry.value)?.to_string();

    Ok(Ok(VideoTarget {
        pk_id: pk_entry.id.to_string(),
        provider_label: provider.to_ascii_lowercase(),
        base_url,
        secret,
        model_entry,
    }))
}

// ─────────────────────── upstream plumbing ───────────────────────

/// The DashScope host root for the native task endpoints. Operators
/// routinely configure alibaba ProviderKeys with the OpenAI-compatible
/// base (`…/compatible-mode/v1`) or a versioned root (`…/api/v1`,
/// `…/v1`) because that is what every chat-endpoint example uses;
/// naively appending `/api/v1/services/…` to those produces an opaque
/// upstream 404. Strip ONE known suffix (most specific first) so both
/// conventions reach the same native root.
fn dashscope_root(base: &str) -> &str {
    let trimmed = base.trim_end_matches('/');
    for suffix in ["/compatible-mode/v1", "/api/v1", "/v1"] {
        if let Some(rest) = trimmed.strip_suffix(suffix) {
            return rest.trim_end_matches('/');
        }
    }
    trimmed
}

/// One DashScope HTTP round-trip: Bearer auth, optional async-submit
/// header, the model's E2E timeout, cooldown accounting on transport
/// failures (parity with jobs / passthrough, #701). Non-2xx responses
/// map to `BridgeError::UpstreamStatus` carrying the upstream `message`
/// (4xx pass through the standard envelope; 5xx bodies are redacted by
/// the shared renderer in error.rs).
async fn dashscope_call(
    state: &ProxyState,
    target: &VideoTarget,
    method: reqwest::Method,
    url: &str,
    body: Option<&serde_json::Value>,
    request_id: &str,
) -> Result<DashScopeResponse, ProxyError> {
    let client = crate::http_client::client();
    let mut builder = client
        .request(method, url)
        .header(header::AUTHORIZATION, format!("Bearer {}", target.secret))
        .header("x-aisix-request-id", request_id);
    if let Some(b) = body {
        // The submit endpoint requires the async-mode header — DashScope
        // rejects synchronous video-synthesis calls outright.
        builder = builder
            .header("X-DashScope-Async", "enable")
            .header(header::CONTENT_TYPE, "application/json")
            .json(b);
    }
    if let Some(d) = target.model_entry.value.request_timeout() {
        builder = builder.timeout(d);
    }

    let started = Instant::now();
    let note = |e: aisix_gateway::BridgeError| {
        crate::cooldown::note_failure(
            &state.runtime_status,
            &target.model_entry.id,
            target.model_entry.value.cooldown.as_ref(),
            e,
        )
    };
    let resp = builder
        .send()
        .await
        .map_err(|e| note(crate::dispatch::reqwest_error_to_bridge(&e, started)))
        .map_err(ProxyError::Bridge)?;
    let status = resp.status().as_u16();
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| note(aisix_gateway::BridgeError::UpstreamDecode(e.to_string())))
        .map_err(ProxyError::Bridge)?;

    if !(200..300).contains(&status) {
        // Best-effort parse of the DashScope `{code, message}` error
        // envelope; fall back to a generic marker for non-JSON bodies.
        let parsed: Option<DashScopeResponse> = serde_json::from_slice(&bytes).ok();
        let message = parsed
            .and_then(|p| match (p.code, p.message) {
                (Some(code), Some(msg)) => Some(format!("{code}: {msg}")),
                (None, Some(msg)) => Some(msg),
                (Some(code), None) => Some(code),
                (None, None) => None,
            })
            .unwrap_or_else(|| "upstream error".to_string());
        return Err(note(aisix_gateway::BridgeError::upstream_status(status, message)).into());
    }

    serde_json::from_slice(&bytes).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamDecode(format!(
            "invalid task response from upstream: {e}"
        )))
    })
}

/// Poll the DashScope task for a decoded video id. Shared by the two GET
/// routes. Charset-guards the decoded (attacker-suppliable) task id
/// before interpolating it into the upstream URL.
async fn poll_task(
    state: &ProxyState,
    target: &VideoTarget,
    task_id: &str,
    request_id: &str,
) -> Result<DashScopeResponse, ProxyError> {
    crate::jobs::require_safe_upstream_id(task_id)?;
    let url = format!(
        "{}{}/{}",
        dashscope_root(&target.base_url),
        DASHSCOPE_TASK_PATH,
        task_id
    );
    dashscope_call(state, target, reqwest::Method::GET, &url, None, request_id).await
}

/// Build the poll-shaped [`VideoObject`] from a DashScope task response.
fn video_object_from_poll(
    video_id: &str,
    model: &str,
    resp: &DashScopeResponse,
) -> Result<VideoObject, ProxyError> {
    let output = resp.output.as_ref().ok_or_else(|| {
        ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamDecode(
            "task response has no `output` object".into(),
        ))
    })?;
    let task_status = output.task_status.as_deref().unwrap_or("UNKNOWN");
    let status = map_task_status(task_status);
    let seconds = resp
        .usage
        .as_ref()
        .and_then(|u| u.output_video_duration.or(u.duration))
        .map(|d| d.to_string());
    let error = (status == "failed").then(|| VideoErrorObject {
        code: output
            .code
            .clone()
            .unwrap_or_else(|| "video_generation_failed".into()),
        message: output
            .message
            .clone()
            .unwrap_or_else(|| "video generation failed".into()),
    });
    Ok(VideoObject {
        id: video_id.to_string(),
        object: "video",
        model: model.to_string(),
        status,
        progress: if status == "completed" { 100 } else { 0 },
        created_at: 0,
        seconds,
        size: None,
        error,
    })
}

// ─────────────────────── shared handler tail ───────────────────────

/// Success/error tail shared by the three handlers: access log + bounded
/// request metrics. The `model` metric label is the RESOLVED display
/// name (or the fixed `unresolved` sentinel) — never raw caller input,
/// which would let a caller explode metric cardinality (#451).
struct Telemetry<'a> {
    state: &'a ProxyState,
    method: &'static str,
    path: String,
    api_key_id: String,
    request_id: String,
    started: Instant,
}

impl Telemetry<'_> {
    fn finish(&self, status: u16, provider: &str, model_label: &str) {
        let elapsed = self.started.elapsed();
        AccessLog {
            method: self.method,
            path: &self.path,
            status,
            latency: elapsed,
            provider: Some(provider).filter(|p| !p.is_empty()),
            model: Some(model_label),
            api_key_id: Some(&self.api_key_id),
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            request_id: &self.request_id,
            served_by_model: None,
            routing_attempt_count: None,
            routing_fallback_count: None,
        }
        .emit();
        self.state.metrics.record_request(
            provider,
            model_label,
            status,
            RequestOutcome::from_status(status),
            elapsed,
        );
    }
}

// ─────────────────────────── POST /v1/videos ───────────────────────────

pub async fn create_video(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    body: Result<Json<VideoCreateBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let started = Instant::now();
    let telemetry = Telemetry {
        state: &state,
        method: "POST",
        path: "/v1/videos".to_string(),
        api_key_id: auth.entry.id.clone(),
        request_id: client.request_id.clone(),
        started,
    };
    let body = match body {
        Ok(Json(b)) => b,
        Err(rej) => {
            // Same discriminate-then-map convention as embeddings (#401):
            // a missing `model`/`prompt` must be a 400 OpenAI-shaped
            // invalid_request_error, not axum's stock 422.
            let err =
                crate::error::proxy_error_from_json_rejection(rej, state.request_body_limit_bytes);
            telemetry.finish(
                err.status().as_u16(),
                "unknown",
                crate::usage_attr::UNRESOLVED_MODEL_LABEL,
            );
            return err.into_response();
        }
    };
    let model_name = body.model.clone();

    match dispatch_create(&state, &auth, body, &client).await {
        Ok(success) => {
            let status = success.response.status().as_u16();
            // Label by the RESOLVED entry's display_name, not the requested
            // alias: a wildcard Model (`wan/*`) accepts unbounded alias
            // strings, and raw caller input in the `model` label is the
            // #451 cardinality failure. Exact-match requests are unchanged
            // (alias == display_name); wildcard traffic aggregates under
            // the pattern itself.
            let snap = state.snapshot.load();
            let model_label = snap
                .models
                .get_by_id(&success.model_id)
                .map(|e| e.value.display_name.clone())
                .unwrap_or_else(|| crate::usage_attr::UNRESOLVED_MODEL_LABEL.to_string());
            telemetry.finish(status, &success.provider, &model_label);
            // One zero-token UsageEvent per accepted submit — visible in
            // /logs and the budget ledger like every other endpoint.
            // Per-second cost is computed control-plane-side once the
            // per-second cost schema lands (AISIX-Cloud#1118 decision 2);
            // token fields stay zero. Skipped when no upstream call
            // happened (the 501 unsupported-provider branch).
            if success.upstream_called {
                emit_submit_usage_event(
                    &state,
                    &client,
                    &auth.entry.id,
                    &success.model_id,
                    &model_name,
                    &success.provider_key_id,
                    &success.applied_guardrails,
                    success.monitor_hits.clone(),
                    status,
                    started.elapsed(),
                );
            }
            success.response
        }
        Err(err) => {
            let status = err.status().as_u16();
            let snap = state.snapshot.load();
            let metric_model = crate::usage_attr::metric_model_label(&snap, &model_name);
            telemetry.finish(status, "unknown", metric_model);
            // #655 parity: failed submits surface in Logs as zero-token
            // events instead of vanishing.
            crate::usage_attr::emit_error_usage_event(
                &state,
                "videos",
                "openai",
                &client.request_id,
                &model_name,
                &auth.entry.id,
                status,
                err.kind(),
                &client,
            );
            err.into_response()
        }
    }
}

struct CreateSuccess {
    response: Response,
    provider: String,
    model_id: String,
    provider_key_id: String,
    applied_guardrails: Vec<AppliedGuardrail>,
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    /// `false` on the 501 unsupported-provider branch — no upstream call
    /// happened, so no UsageEvent is attributed (embeddings convention).
    upstream_called: bool,
}

async fn dispatch_create(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    body: VideoCreateBody,
    client: &ClientContext,
) -> Result<CreateSuccess, ProxyError> {
    let snapshot = state.snapshot.load();
    let model_entry = crate::model_resolve::resolve_model(&snapshot, &body.model)
        .ok_or_else(|| ProxyError::ModelNotFound(body.model.clone()))?;
    let model_id = model_entry.id.to_string();

    // Validate the unified params before any upstream work.
    let seconds = body
        .seconds
        .as_ref()
        .map(SecondsField::as_secs)
        .transpose()?;
    if body.prompt.trim().is_empty() {
        return Err(ProxyError::InvalidRequest(
            "`prompt` must not be empty".into(),
        ));
    }

    let target =
        match resolve_video_target(state, auth, model_entry, &body.model, &client.source_ip)? {
            Ok(t) => t,
            Err(resp) => {
                return Ok(CreateSuccess {
                    response: resp,
                    provider: "unknown".into(),
                    model_id,
                    provider_key_id: String::new(),
                    applied_guardrails: Vec::new(),
                    monitor_hits: Vec::new(),
                    upstream_called: false,
                })
            }
        };

    // Input guardrail chain over the prompt — same resolution as chat /
    // embeddings, run BEFORE the rate-limit reservation so a policy
    // block doesn't burn an RPM slot (#542).
    let guardrail_ctx = aisix_guardrails::RequestContext {
        model_id: &target.model_entry.id,
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved_chain = state.guardrail_index.resolve(&guardrail_ctx);
    let applied_guardrails = resolved_chain.applied().to_vec();
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    if !resolved_chain.is_empty() {
        let chat = aisix_gateway::ChatFormat::new(
            &body.model,
            vec![aisix_gateway::ChatMessage::user(body.prompt.clone())],
        );
        let (verdict, hits) =
            aisix_guardrails::Guardrail::check_input_observed(&resolved_chain, &chat).await;
        monitor_hits.extend(hits);
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } = verdict
        {
            // Matched-pattern detail stays in ops logs only (#153).
            tracing::warn!(
                guardrail_hook = "input",
                model = %body.model,
                reason = %reason,
                "guardrail blocked /v1/videos request",
            );
            return Err(ProxyError::ContentFiltered(
                crate::error::guardrail_block_message("request", guardrail_name.as_deref()),
            ));
        }
    }

    // Model-level rate limiting — the submit is a full typed endpoint
    // (AISIX-Cloud#1118 decision 3; the #1116 shape).
    let model_rl = crate::quota::ModelRateLimit::from_model(
        &body.model,
        &target.model_entry.id,
        &target.model_entry.value,
    );
    let reservation = crate::quota::enforce(state, auth, Some(&model_rl)).await?;

    let upstream_model =
        crate::dispatch::require_upstream_model(&target.model_entry.value)?.to_string();
    let submit_body =
        dashscope_submit_body(&upstream_model, &body.prompt, seconds, body.size.as_deref())?;
    let url = format!(
        "{}{}",
        dashscope_root(&target.base_url),
        DASHSCOPE_SUBMIT_PATH
    );

    let result = dashscope_call(
        state,
        &target,
        reqwest::Method::POST,
        &url,
        Some(&submit_body),
        &client.request_id,
    )
    .await;
    // Zero tokens on the videos surface — commit releases the
    // concurrency permit and finalises RPM.
    reservation.commit_tokens(0).await;
    let resp = result?;

    state.health.record_success(&body.model);
    state.runtime_status.mark_healthy(&target.model_entry.id);

    let output = resp.output.as_ref().ok_or_else(|| {
        ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamDecode(
            "submit response has no `output` object".into(),
        ))
    })?;
    let task_id = output
        .task_id
        .as_deref()
        .filter(|t| !t.is_empty())
        .ok_or_else(|| {
            ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamDecode(
                "submit response has no task id".into(),
            ))
        })?;
    let status = map_task_status(output.task_status.as_deref().unwrap_or("PENDING"));

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let video = VideoObject {
        id: encode_video_id(&target.model_entry.id, &body.model, task_id),
        object: "video",
        // Echo the caller's requested model name, like every other
        // typed endpoint (wildcard aliases included).
        model: body.model.clone(),
        status,
        progress: if status == "completed" { 100 } else { 0 },
        created_at,
        seconds: seconds.map(|s| s.to_string()),
        size: body.size.clone(),
        error: None,
    };

    Ok(CreateSuccess {
        response: Json(video).into_response(),
        provider: target.provider_label.clone(),
        model_id,
        provider_key_id: target.pk_id.clone(),
        applied_guardrails,
        monitor_hits,
        upstream_called: true,
    })
}

// ─────────────────────────── GET routes ───────────────────────────

/// Decode + resolve the target for a GET route. 404 for ids this gateway
/// could not have minted (undecodable, or naming a Model entry that no
/// longer exists in the snapshot).
///
/// The key ACL runs against the REQUESTED ALIAS carried inside the id —
/// the identical check the submit performed — so wildcard-alias journeys
/// (Model `wan/*`, key allowlisted for `wan/turbo`) can poll their own
/// tasks. Two deliberate 404 folds close a disclosure oracle on guessed
/// entry UUIDs: an ACL denial and the unsupported-provider case both
/// surface as `video_not_found`, because a 403 (which would echo a model
/// identity) or a 501 (which reveals the entry exists and names its
/// provider family) would let a caller probe the Model table by minting
/// crafted ids. The submit path keeps its distinct 403/501 — there the
/// caller already knows the model name they asked for, so there is
/// nothing to disclose.
fn resolve_get_target(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    video_id: &str,
    source_ip: &str,
) -> Result<(VideoTarget, String), ProxyError> {
    let (entry_id, alias, task_id) =
        decode_video_id(video_id).ok_or_else(|| ProxyError::VideoNotFound(video_id.to_string()))?;
    let snapshot = state.snapshot.load();
    let model_entry = snapshot
        .models
        .get_by_id(&entry_id)
        .ok_or_else(|| ProxyError::VideoNotFound(video_id.to_string()))?;
    match resolve_video_target(state, auth, model_entry, &alias, source_ip) {
        Ok(Ok(target)) => Ok((target, task_id)),
        // Unsupported provider → uniform 404 (oracle fold, see above).
        Ok(Err(_)) => Err(ProxyError::VideoNotFound(video_id.to_string())),
        // ACL denial → uniform 404 (oracle fold, see above).
        Err(ProxyError::ModelForbidden(_)) => Err(ProxyError::VideoNotFound(video_id.to_string())),
        Err(e) => Err(e),
    }
}

pub async fn get_video(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    Path(video_id): Path<String>,
) -> Response {
    let telemetry = Telemetry {
        state: &state,
        method: "GET",
        path: "/v1/videos/:id".to_string(),
        api_key_id: auth.entry.id.clone(),
        request_id: client.request_id.clone(),
        started: Instant::now(),
    };

    let result: Result<(Response, String, String), ProxyError> = async {
        let (target, task_id) = resolve_get_target(&state, &auth, &video_id, &client.source_ip)?;
        // Poll traffic is exempt from model-level limits BY DESIGN
        // (AISIX-Cloud#1118 decision 3): a client polling a task it
        // already paid an RPM slot to submit must not starve itself.
        // Key-level layers still apply.
        let reservation = crate::quota::enforce(&state, &auth, None).await?;
        let result = poll_task(&state, &target, &task_id, &client.request_id).await;
        reservation.commit_tokens(0).await;
        let resp = result?;
        let video = video_object_from_poll(&video_id, target.display_name(), &resp)?;
        Ok((
            Json(video).into_response(),
            target.provider_label.clone(),
            target.display_name().to_string(),
        ))
    }
    .await;

    match result {
        Ok((resp, provider, model_label)) => {
            telemetry.finish(resp.status().as_u16(), &provider, &model_label);
            resp
        }
        Err(err) => {
            telemetry.finish(
                err.status().as_u16(),
                "unknown",
                crate::usage_attr::UNRESOLVED_MODEL_LABEL,
            );
            err.into_response()
        }
    }
}

pub async fn video_content(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    Path(video_id): Path<String>,
) -> Response {
    let telemetry = Telemetry {
        state: &state,
        method: "GET",
        path: "/v1/videos/:id/content".to_string(),
        api_key_id: auth.entry.id.clone(),
        request_id: client.request_id.clone(),
        started: Instant::now(),
    };

    let result: Result<(Response, String, String), ProxyError> = async {
        let (target, task_id) = resolve_get_target(&state, &auth, &video_id, &client.source_ip)?;
        // Same model-layer exemption as the poll route (see get_video).
        let reservation = crate::quota::enforce(&state, &auth, None).await?;
        let result = poll_task(&state, &target, &task_id, &client.request_id).await;
        reservation.commit_tokens(0).await;
        let resp = result?;

        let output = resp.output.as_ref().ok_or_else(|| {
            ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamDecode(
                "task response has no `output` object".into(),
            ))
        })?;
        let task_status = output.task_status.as_deref().unwrap_or("UNKNOWN");
        let response = match map_task_status(task_status) {
            // Phase 1 fetches by 302 redirect to the provider's own URL —
            // zero relay bandwidth (AISIX-Cloud#1118 decision 5). A
            // streaming proxy for providers whose URLs need gateway
            // credentials is a tracked follow-up.
            "completed" => {
                let video_url = output.video_url.as_deref().ok_or_else(|| {
                    ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamDecode(
                        "completed task has no video URL".into(),
                    ))
                })?;
                // The redirect target is provider-supplied: require a
                // well-formed absolute http(s) URL so a malformed or
                // exotic-scheme value (javascript:, file:, data:) can
                // never ride the Location header to the caller. The
                // host itself remains operator-trusted upstream
                // infrastructure — the gateway never fetches the URL.
                let parsed = url::Url::parse(video_url)
                    .ok()
                    .filter(|u| matches!(u.scheme(), "http" | "https"))
                    .ok_or_else(|| {
                        ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamDecode(
                            "completed task has a malformed or non-http video URL".into(),
                        ))
                    })?;
                let location =
                    axum::http::HeaderValue::from_str(parsed.as_str()).map_err(|_| {
                        ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamDecode(
                            "completed task has a malformed video URL".into(),
                        ))
                    })?;
                let mut resp = StatusCode::FOUND.into_response();
                resp.headers_mut().insert(header::LOCATION, location);
                resp
            }
            "failed" => {
                let detail = output
                    .message
                    .clone()
                    .or_else(|| output.code.clone())
                    .unwrap_or_else(|| "video generation failed".into());
                return Err(ProxyError::InvalidRequest(format!(
                    "video generation failed: {detail}"
                )));
            }
            // queued / in_progress — not downloadable yet. 400
            // invalid_request_error is the closest fit in the existing
            // taxonomy; the message tells the caller to keep polling.
            other => {
                return Err(ProxyError::InvalidRequest(format!(
                    "video is not ready for download (status: {other}); poll \
                     GET /v1/videos/{{video_id}} until status is \"completed\""
                )));
            }
        };
        Ok((
            response,
            target.provider_label.clone(),
            target.display_name().to_string(),
        ))
    }
    .await;

    match result {
        Ok((resp, provider, model_label)) => {
            telemetry.finish(resp.status().as_u16(), &provider, &model_label);
            resp
        }
        Err(err) => {
            telemetry.finish(
                err.status().as_u16(),
                "unknown",
                crate::usage_attr::UNRESOLVED_MODEL_LABEL,
            );
            err.into_response()
        }
    }
}

// ─────────────────────── telemetry plumbing ───────────────────────

/// One zero-token UsageEvent per accepted submit — the passthrough /
/// jobs shape (#699) with the resolved Model attributed and the
/// guardrail observability fields the chat family carries. Token fields
/// stay zero: video billing is duration-based and priced control-plane-
/// side once the per-second cost schema lands (AISIX-Cloud#1118
/// decision 2).
#[allow(clippy::too_many_arguments)]
fn emit_submit_usage_event(
    state: &ProxyState,
    client: &ClientContext,
    api_key_id: &str,
    model_id: &str,
    requested_model: &str,
    provider_key_id: &str,
    applied_guardrails: &[AppliedGuardrail],
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    status_code: u16,
    elapsed: Duration,
) {
    let snap = state.snapshot.load();
    let mut event = UsageEvent {
        request_id: client.request_id.clone(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        status_code,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        inbound_protocol: "openai".to_string(),
        applied_guardrails: applied_guardrails.to_vec(),
        guardrail_monitor_hits,
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        ..Default::default()
    };
    crate::usage_attr::apply_pk_telemetry(&mut event, &snap, provider_key_id);
    state.usage_sink.try_emit("videos", event.clone());
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
}

#[cfg(test)]
mod tests {
    use super::*;

    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use axum::body::to_bytes;
    use axum::http::Request;
    use std::sync::Arc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // ───────────────────────── pure-function tests ─────────────────────────

    #[test]
    fn task_status_mapping_table() {
        // The full mapping pinned by AISIX-Cloud#1118: PENDING→queued,
        // RUNNING→in_progress, SUCCEEDED→completed, FAILED/CANCELED/
        // UNKNOWN→failed. Unrecognised provider strings also collapse
        // to failed instead of leaking provider taxonomy.
        assert_eq!(map_task_status("PENDING"), "queued");
        assert_eq!(map_task_status("RUNNING"), "in_progress");
        assert_eq!(map_task_status("SUCCEEDED"), "completed");
        assert_eq!(map_task_status("FAILED"), "failed");
        assert_eq!(map_task_status("CANCELED"), "failed");
        assert_eq!(map_task_status("UNKNOWN"), "failed");
        assert_eq!(map_task_status("SOMETHING_NEW"), "failed");
    }

    #[test]
    fn video_id_roundtrip() {
        let id = encode_video_id(
            "11111111-2222-3333-4444-555555555555",
            "wan/turbo",
            "task-abc.123",
        );
        assert_eq!(
            decode_video_id(&id),
            Some((
                "11111111-2222-3333-4444-555555555555".to_string(),
                "wan/turbo".to_string(),
                "task-abc.123".to_string()
            ))
        );
        // A task id containing `:` survives — the first two segments
        // (entry id, base64url alias) never contain one.
        let id = encode_video_id("m-1", "my-video", "ns:task:9");
        assert_eq!(
            decode_video_id(&id),
            Some((
                "m-1".to_string(),
                "my-video".to_string(),
                "ns:task:9".to_string()
            ))
        );
        // An alias containing `:` cannot break the framing — it rides
        // base64url-encoded inside its segment.
        let id = encode_video_id("m-1", "weird:alias", "task-1");
        assert_eq!(
            decode_video_id(&id),
            Some((
                "m-1".to_string(),
                "weird:alias".to_string(),
                "task-1".to_string()
            ))
        );
    }

    #[test]
    fn video_id_tamper_rejection() {
        // Not base64url.
        assert_eq!(decode_video_id("!!!not-base64!!!"), None);
        // Valid base64 of a string with no separator.
        assert_eq!(
            decode_video_id(&URL_SAFE_NO_PAD.encode("no-separator")),
            None
        );
        // Two-segment (pre-alias) shape no longer decodes.
        assert_eq!(decode_video_id(&URL_SAFE_NO_PAD.encode("entry:task")), None);
        // Empty segments.
        let alias_b64 = URL_SAFE_NO_PAD.encode("alias");
        assert_eq!(
            decode_video_id(&URL_SAFE_NO_PAD.encode(format!(":{alias_b64}:task"))),
            None
        );
        assert_eq!(
            decode_video_id(&URL_SAFE_NO_PAD.encode("model::task")),
            None
        );
        assert_eq!(
            decode_video_id(&URL_SAFE_NO_PAD.encode(format!("model:{alias_b64}:"))),
            None
        );
        // Alias segment that isn't valid base64url.
        assert_eq!(
            decode_video_id(&URL_SAFE_NO_PAD.encode("model:!!!:task")),
            None
        );
        // Non-UTF8 payload.
        assert_eq!(
            decode_video_id(&URL_SAFE_NO_PAD.encode([0xff, 0xfe, b':', b'x'])),
            None
        );
        // Raw provider task id (never minted by this gateway) — decodes
        // as base64 only by coincidence, and then fails the separator
        // check; either way it must not resolve.
        assert_eq!(decode_video_id("task-123"), None);
    }

    #[test]
    fn dashscope_root_strips_known_api_base_suffixes() {
        // The OpenAI-compatible base operators paste from chat examples.
        assert_eq!(
            dashscope_root("https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "https://dashscope.aliyuncs.com"
        );
        assert_eq!(
            dashscope_root("https://dashscope.aliyuncs.com/compatible-mode/v1/"),
            "https://dashscope.aliyuncs.com"
        );
        // Native versioned roots.
        assert_eq!(
            dashscope_root("https://dashscope.aliyuncs.com/api/v1"),
            "https://dashscope.aliyuncs.com"
        );
        assert_eq!(
            dashscope_root("https://dashscope.aliyuncs.com/v1"),
            "https://dashscope.aliyuncs.com"
        );
        // Bare host is untouched; only ONE suffix is stripped.
        assert_eq!(
            dashscope_root("https://dashscope.aliyuncs.com"),
            "https://dashscope.aliyuncs.com"
        );
        assert_eq!(
            dashscope_root("https://host/api/v1/api/v1"),
            "https://host/api/v1"
        );
    }

    #[test]
    fn submit_param_mapping() {
        // seconds → parameters.duration, size WIDTHxHEIGHT → size WIDTH*HEIGHT.
        let body = dashscope_submit_body("wan-x", "a cat", Some(8), Some("1280x720")).unwrap();
        assert_eq!(body["model"], "wan-x");
        assert_eq!(body["input"]["prompt"], "a cat");
        assert_eq!(body["parameters"]["duration"], 8);
        assert_eq!(body["parameters"]["size"], "1280*720");

        // Unset params are omitted entirely — no empty `parameters` object.
        let body = dashscope_submit_body("wan-x", "a cat", None, None).unwrap();
        assert!(body.get("parameters").is_none());

        // Partial: only duration.
        let body = dashscope_submit_body("wan-x", "a cat", Some(5), None).unwrap();
        assert_eq!(body["parameters"]["duration"], 5);
        assert!(body["parameters"].get("size").is_none());
    }

    #[test]
    fn size_validation_rejects_malformed_values() {
        assert!(map_size("1280x720").is_ok());
        for bad in [
            "1280*720",
            "1280",
            "x720",
            "1280x",
            "12a0x720",
            "1280x720x1",
            "",
        ] {
            assert!(map_size(bad).is_err(), "size {bad:?} must be rejected");
        }
    }

    #[test]
    fn seconds_field_accepts_string_and_int() {
        assert_eq!(SecondsField::Int(8).as_secs().unwrap(), 8);
        assert_eq!(SecondsField::Str("12".into()).as_secs().unwrap(), 12);
        assert!(SecondsField::Str("abc".into()).as_secs().is_err());
        assert!(SecondsField::Int(0).as_secs().is_err());
        assert!(SecondsField::Str("-4".into()).as_secs().is_err());
    }

    // ───────────────────────── handler tests ─────────────────────────

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    const PK_ID: &str = "22222222-2222-2222-2222-222222222222";
    const MODEL_ID: &str = "m-video-1";

    fn model_entry_json(name: &str, provider: &str, extra: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "{provider}",
                "model_name": "wan-upstream",
                "provider_key_id": "{PK_ID}"
                {extra}
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(MODEL_ID, m, 1)
    }

    fn new_snap(api_base: &str, provider: &str, model_extra: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        let pk_json = format!(
            r#"{{"display_name":"ali-up","secret":"sk-up","api_base":"{api_base}","provider":"{provider}"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        snap.models
            .insert(model_entry_json("my-video", provider, model_extra));
        let key: ApiKey = serde_json::from_str(
            r#"{"key_hash": "8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891", "allowed_models": ["*"]}"#,
        )
        .unwrap();
        snap.apikeys.insert(ResourceEntry::new("k-1", key, 1));
        snap
    }

    fn build_app(snap: AisixSnapshot) -> axum::Router {
        let hub = Arc::new(Hub::new());
        let handle = SnapshotHandle::new(snap);
        crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    fn post_videos(body: serde_json::Value) -> Request<axum::body::Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/videos")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    fn get_uri(uri: &str) -> Request<axum::body::Body> {
        Request::builder()
            .method("GET")
            .uri(uri)
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap()
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn pending_submit_response() -> serde_json::Value {
        serde_json::json!({
            "output": {"task_id": "task-123", "task_status": "PENDING"},
            "request_id": "req-1"
        })
    }

    #[tokio::test]
    async fn submit_happy_path_returns_video_object_and_maps_upstream_wire() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(DASHSCOPE_SUBMIT_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(pending_submit_response()))
            .expect(1)
            .mount(&upstream)
            .await;

        let app = build_app(new_snap(&upstream.uri(), "alibaba", ""));
        let resp = tower::ServiceExt::oneshot(
            app,
            post_videos(serde_json::json!({
                "model": "my-video",
                "prompt": "a cardboard city at night",
                "seconds": "8",
                "size": "1280x720"
            })),
        )
        .await
        .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "video");
        assert_eq!(v["status"], "queued");
        // The caller's requested model name echoes back, not the
        // upstream `model_name`.
        assert_eq!(v["model"], "my-video");
        assert_eq!(v["progress"], 0);
        assert!(v["created_at"].as_i64().unwrap() > 0);
        assert_eq!(v["seconds"], "8");
        assert_eq!(v["size"], "1280x720");
        assert!(v["error"].is_null() || v.get("error").is_none());
        // The id decodes to (model entry id, upstream task id).
        let id = v["id"].as_str().unwrap();
        assert_eq!(
            decode_video_id(id),
            Some((
                MODEL_ID.to_string(),
                "my-video".to_string(),
                "task-123".to_string()
            ))
        );

        // Upstream wire shape: async header + DashScope envelope with
        // the resolved upstream model id and mapped parameters.
        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let req = &received[0];
        assert_eq!(
            req.headers
                .get("x-dashscope-async")
                .map(|v| v.to_str().unwrap()),
            Some("enable")
        );
        let wire: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(wire["model"], "wan-upstream");
        assert_eq!(wire["input"]["prompt"], "a cardboard city at night");
        assert_eq!(wire["parameters"]["duration"], 8);
        assert_eq!(wire["parameters"]["size"], "1280*720");
    }

    #[tokio::test]
    async fn poll_maps_running_to_in_progress() {
        let upstream = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("{DASHSCOPE_TASK_PATH}/task-123")))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {"task_id": "task-123", "task_status": "RUNNING"},
                "request_id": "req-2"
            })))
            .mount(&upstream)
            .await;

        let app = build_app(new_snap(&upstream.uri(), "alibaba", ""));
        let id = encode_video_id(MODEL_ID, "my-video", "task-123");
        let resp = tower::ServiceExt::oneshot(app, get_uri(&format!("/v1/videos/{id}")))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["object"], "video");
        assert_eq!(v["status"], "in_progress");
        assert_eq!(v["id"], id);
        assert_eq!(v["progress"], 0);
    }

    #[tokio::test]
    async fn poll_failed_task_carries_error_object() {
        let upstream = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("{DASHSCOPE_TASK_PATH}/task-bad")))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {
                    "task_id": "task-bad",
                    "task_status": "FAILED",
                    "code": "InvalidParameter",
                    "message": "prompt rejected"
                },
                "request_id": "req-3"
            })))
            .mount(&upstream)
            .await;

        let app = build_app(new_snap(&upstream.uri(), "alibaba", ""));
        let id = encode_video_id(MODEL_ID, "my-video", "task-bad");
        let resp = tower::ServiceExt::oneshot(app, get_uri(&format!("/v1/videos/{id}")))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["status"], "failed");
        assert_eq!(v["error"]["code"], "InvalidParameter");
        assert_eq!(v["error"]["message"], "prompt rejected");
    }

    #[tokio::test]
    async fn content_redirects_302_to_provider_url_when_completed() {
        let upstream = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("{DASHSCOPE_TASK_PATH}/task-123")))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {
                    "task_id": "task-123",
                    "task_status": "SUCCEEDED",
                    "video_url": "https://cdn.example.com/out.mp4?sig=abc"
                },
                "usage": {"duration": 8},
                "request_id": "req-4"
            })))
            .mount(&upstream)
            .await;

        let app = build_app(new_snap(&upstream.uri(), "alibaba", ""));
        let id = encode_video_id(MODEL_ID, "my-video", "task-123");
        let resp = tower::ServiceExt::oneshot(app, get_uri(&format!("/v1/videos/{id}/content")))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::FOUND);
        assert_eq!(
            resp.headers().get(header::LOCATION).unwrap(),
            "https://cdn.example.com/out.mp4?sig=abc"
        );
    }

    #[tokio::test]
    async fn content_not_ready_returns_400_invalid_request() {
        let upstream = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("{DASHSCOPE_TASK_PATH}/task-123")))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {"task_id": "task-123", "task_status": "RUNNING"},
                "request_id": "req-5"
            })))
            .mount(&upstream)
            .await;

        let app = build_app(new_snap(&upstream.uri(), "alibaba", ""));
        let id = encode_video_id(MODEL_ID, "my-video", "task-123");
        let resp = tower::ServiceExt::oneshot(app, get_uri(&format!("/v1/videos/{id}/content")))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not ready"));
    }

    #[tokio::test]
    async fn unknown_video_id_returns_404() {
        let app = build_app(new_snap("http://unused", "alibaba", ""));
        // Undecodable id.
        let resp = tower::ServiceExt::oneshot(app, get_uri("/v1/videos/not-a-real-id"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "video_not_found");

        // Decodable id naming a Model entry that doesn't exist.
        let app = build_app(new_snap("http://unused", "alibaba", ""));
        let ghost = encode_video_id("no-such-entry", "my-video", "task-1");
        let resp = tower::ServiceExt::oneshot(app, get_uri(&format!("/v1/videos/{ghost}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unsupported_provider_returns_501_not_implemented() {
        let app = build_app(new_snap("http://unused", "zhipu", ""));
        let resp = tower::ServiceExt::oneshot(
            app,
            post_videos(serde_json::json!({"model": "my-video", "prompt": "hi"})),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "not_implemented");
    }

    #[tokio::test]
    async fn submit_missing_prompt_returns_400_openai_envelope() {
        // Missing required field → 400 invalid_request_error (the #401
        // discriminate-then-map convention), not axum's stock 422.
        let app = build_app(new_snap("http://unused", "alibaba", ""));
        let resp =
            tower::ServiceExt::oneshot(app, post_videos(serde_json::json!({"model": "my-video"})))
                .await
                .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn submit_unknown_model_returns_404() {
        let app = build_app(new_snap("http://unused", "alibaba", ""));
        let resp = tower::ServiceExt::oneshot(
            app,
            post_videos(serde_json::json!({"model": "nonexistent", "prompt": "hi"})),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "model_not_found");
    }

    #[tokio::test]
    async fn input_guardrail_blocks_prompt_and_upstream_is_never_called() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(DASHSCOPE_SUBMIT_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(pending_submit_response()))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri(), "alibaba", "");
        let g: aisix_core::Guardrail = serde_json::from_str(
            r#"{"name":"test-block","enabled":true,"hook_point":"input","fail_open":false,"kind":"keyword","patterns":[{"kind":"literal","value":"BLOCKME"}]}"#,
        )
        .unwrap();
        snap.guardrails.insert(ResourceEntry::new("g-1", g, 1));

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(
            app,
            post_videos(serde_json::json!({"model": "my-video", "prompt": "please BLOCKME now"})),
        )
        .await
        .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "content_filter");
        // The matched literal must not leak into the wire message (#153).
        assert!(!v["error"]["message"].as_str().unwrap().contains("BLOCKME"));
    }

    #[tokio::test]
    async fn submit_hits_model_rpm_cap_but_polling_stays_exempt() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(DASHSCOPE_SUBMIT_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(pending_submit_response()))
            .expect(1)
            .mount(&upstream)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("{DASHSCOPE_TASK_PATH}/task-123")))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {"task_id": "task-123", "task_status": "RUNNING"},
                "request_id": "req-6"
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri(), "alibaba", r#", "rate_limit": {"rpm": 1}"#);
        let app = build_app(snap);
        let submit = serde_json::json!({"model": "my-video", "prompt": "hi"});

        // First submit consumes the model's single rpm slot.
        let first = tower::ServiceExt::oneshot(app.clone(), post_videos(submit.clone()))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let id = body_json(first).await["id"].as_str().unwrap().to_string();

        // Second submit inside the window → 429 with Retry-After; the
        // `.expect(1)` on the submit mock proves it never reached the
        // upstream.
        let second = tower::ServiceExt::oneshot(app.clone(), post_videos(submit))
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(second.headers().get("retry-after").is_some());
        let v = body_json(second).await;
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");

        // Polling the submitted task is NOT gated by the exhausted
        // model bucket (AISIX-Cloud#1118 decision 3).
        for _ in 0..3 {
            let poll =
                tower::ServiceExt::oneshot(app.clone(), get_uri(&format!("/v1/videos/{id}")))
                    .await
                    .unwrap();
            assert_eq!(poll.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn upstream_4xx_maps_to_error_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(DASHSCOPE_SUBMIT_PATH))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "code": "InvalidParameter",
                "message": "duration out of range",
                "request_id": "req-7"
            })))
            .mount(&upstream)
            .await;

        let app = build_app(new_snap(&upstream.uri(), "alibaba", ""));
        let resp = tower::ServiceExt::oneshot(
            app,
            post_videos(serde_json::json!({"model": "my-video", "prompt": "hi", "seconds": 999})),
        )
        .await
        .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        // Wire=Unknown 4xx keeps the upstream message under the generic
        // upstream_error type (error.rs render rules).
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("duration out of range"));
    }

    /// HIGH-1 (PR #811 audit): a GET on an id that resolves to a
    /// non-serving state must record the fixed `unresolved` metric
    /// label — never the caller-controlled video id, which would let an
    /// authenticated caller explode Prometheus cardinality by minting
    /// random ids against a known entry UUID.
    #[tokio::test]
    async fn get_error_paths_record_unresolved_metric_label_not_the_raw_id() {
        // A known entry UUID with a non-alibaba provider: pre-fix this
        // hit the unsupported-provider branch and recorded the raw id.
        let snap = new_snap("http://unused", "zhipu", "");
        let hub = Arc::new(Hub::new());
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg()).without_cache();
        let metrics = state.metrics.clone();
        let app = crate::build_router(state);

        let id = encode_video_id(MODEL_ID, "my-video", "task-zzz");
        let resp = tower::ServiceExt::oneshot(app, get_uri(&format!("/v1/videos/{id}")))
            .await
            .unwrap();
        // MED-3 fold: unsupported provider on a GET is a uniform 404.
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let rendered = metrics.render();
        assert!(
            rendered.contains(&format!(
                "model=\"{}\"",
                crate::usage_attr::UNRESOLVED_MODEL_LABEL
            )),
            "GET error path must record the bounded sentinel label: {rendered}"
        );
        assert!(
            !rendered.contains(&id),
            "the caller-controlled video id must never appear as a metric label"
        );
    }

    /// The submit success path labels metrics by the RESOLVED entry's
    /// display name, never the caller's alias: a wildcard Model accepts
    /// unbounded alias strings, so labeling by the request would be the
    /// #451 cardinality failure on this route too.
    #[tokio::test]
    async fn submit_success_metric_label_is_resolved_display_name_not_alias() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(DASHSCOPE_SUBMIT_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(pending_submit_response()))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        let pk_json = format!(
            r#"{{"display_name":"ali-up","secret":"sk-up","api_base":"{}","provider":"alibaba"}}"#,
            upstream.uri()
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        let m: Model = serde_json::from_str(&format!(
            r#"{{
                "display_name": "wan/*",
                "provider": "alibaba",
                "model_name": "*",
                "provider_key_id": "{PK_ID}"
            }}"#
        ))
        .unwrap();
        snap.models.insert(ResourceEntry::new(MODEL_ID, m, 1));
        let key: ApiKey = serde_json::from_str(
            r#"{"key_hash": "8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891", "allowed_models": ["wan/turbo-cardinality-probe"]}"#,
        )
        .unwrap();
        snap.apikeys.insert(ResourceEntry::new("k-1", key, 1));

        let hub = Arc::new(Hub::new());
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg()).without_cache();
        let metrics = state.metrics.clone();
        let app = crate::build_router(state);

        let resp = tower::ServiceExt::oneshot(
            app,
            post_videos(
                serde_json::json!({"model": "wan/turbo-cardinality-probe", "prompt": "hi"}),
            ),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let rendered = metrics.render();
        assert!(
            rendered.contains("model=\"wan/*\""),
            "submit success must label by the resolved wildcard pattern: {rendered}"
        );
        assert!(
            !rendered.contains("wan/turbo-cardinality-probe"),
            "the caller alias must never appear as a metric label: {rendered}"
        );
    }

    /// MED-2 (PR #811 audit): wildcard-alias journey. The key is
    /// allowlisted for the concrete alias (`wan/turbo`), NOT the
    /// wildcard Model's display name (`wan/*`). The id carries the
    /// alias, so poll + content re-run the SAME ACL check the submit
    /// passed — pre-fix the GETs checked the display name and 403'd
    /// the caller's own task.
    #[tokio::test]
    async fn wildcard_alias_submit_poll_content_all_succeed() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(DASHSCOPE_SUBMIT_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(pending_submit_response()))
            .expect(1)
            .mount(&upstream)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("{DASHSCOPE_TASK_PATH}/task-123")))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {
                    "task_id": "task-123",
                    "task_status": "SUCCEEDED",
                    "video_url": "https://cdn.example.com/wan-turbo.mp4"
                },
                "request_id": "req-w1"
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        let pk_json = format!(
            r#"{{"display_name":"ali-up","secret":"sk-up","api_base":"{}","provider":"alibaba"}}"#,
            upstream.uri()
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        let m: Model = serde_json::from_str(&format!(
            r#"{{
                "display_name": "wan/*",
                "provider": "alibaba",
                "model_name": "*",
                "provider_key_id": "{PK_ID}"
            }}"#
        ))
        .unwrap();
        snap.models.insert(ResourceEntry::new(MODEL_ID, m, 1));
        let key: ApiKey = serde_json::from_str(
            r#"{"key_hash": "8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891", "allowed_models": ["wan/turbo"]}"#,
        )
        .unwrap();
        snap.apikeys.insert(ResourceEntry::new("k-1", key, 1));

        let app = build_app(snap);
        let created = tower::ServiceExt::oneshot(
            app.clone(),
            post_videos(serde_json::json!({"model": "wan/turbo", "prompt": "hi"})),
        )
        .await
        .unwrap();
        assert_eq!(created.status(), StatusCode::OK);
        let v = body_json(created).await;
        assert_eq!(v["model"], "wan/turbo");
        let id = v["id"].as_str().unwrap().to_string();
        // The captured segment resolves the upstream model id.
        let received = upstream.received_requests().await.unwrap();
        let wire: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(wire["model"], "turbo");

        let poll = tower::ServiceExt::oneshot(app.clone(), get_uri(&format!("/v1/videos/{id}")))
            .await
            .unwrap();
        assert_eq!(poll.status(), StatusCode::OK);
        let polled = body_json(poll).await;
        assert_eq!(polled["status"], "completed");

        let content = tower::ServiceExt::oneshot(app, get_uri(&format!("/v1/videos/{id}/content")))
            .await
            .unwrap();
        assert_eq!(content.status(), StatusCode::FOUND);
    }

    /// MED-3 (PR #811 audit): a key NOT allowlisted for the model gets a
    /// uniform 404 on the GET routes — not a 403 echoing a model
    /// identity — and the upstream is never contacted. A crafted id
    /// must not turn the poll route into a Model-table existence probe.
    #[tokio::test]
    async fn restricted_key_gets_404_on_poll_and_upstream_untouched() {
        let upstream = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("{DASHSCOPE_TASK_PATH}/task-123")))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {"task_id": "task-123", "task_status": "RUNNING"}
            })))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri(), "alibaba", "");
        // Overwrite the caller key with one that cannot access the model.
        let key: ApiKey = serde_json::from_str(
            r#"{"key_hash": "8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891", "allowed_models": ["other-model"]}"#,
        )
        .unwrap();
        snap.apikeys.insert(ResourceEntry::new("k-1", key, 1));

        let app = build_app(snap);
        let id = encode_video_id(MODEL_ID, "my-video", "task-123");
        for suffix in ["", "/content"] {
            let resp = tower::ServiceExt::oneshot(
                app.clone(),
                get_uri(&format!("/v1/videos/{id}{suffix}")),
            )
            .await
            .unwrap();
            assert_eq!(resp.status(), StatusCode::NOT_FOUND);
            let v = body_json(resp).await;
            assert_eq!(v["error"]["type"], "video_not_found");
            let msg = v["error"]["message"].as_str().unwrap();
            assert!(
                !msg.contains("my-video"),
                "the model identity must not leak through the 404: {msg}"
            );
        }
    }

    /// MED-5 (PR #811 audit): an alibaba ProviderKey configured with the
    /// OpenAI-compatible base (`…/compatible-mode/v1` — what every chat
    /// example uses) must still reach the native task endpoints instead
    /// of producing `…/compatible-mode/v1/api/v1/services/…` 404s.
    #[tokio::test]
    async fn compatible_mode_api_base_reaches_native_dashscope_paths() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(DASHSCOPE_SUBMIT_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(pending_submit_response()))
            .expect(1)
            .mount(&upstream)
            .await;

        let base = format!("{}/compatible-mode/v1", upstream.uri());
        let app = build_app(new_snap(&base, "alibaba", ""));
        let resp = tower::ServiceExt::oneshot(
            app,
            post_videos(serde_json::json!({"model": "my-video", "prompt": "hi"})),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].url.path(), DASHSCOPE_SUBMIT_PATH);
    }

    /// MED-4 (PR #811 audit): the content redirect only forwards
    /// http(s) URLs — an exotic scheme from a misbehaving upstream must
    /// not ride the Location header to the caller.
    #[tokio::test]
    async fn content_rejects_non_http_video_url_scheme() {
        let upstream = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("{DASHSCOPE_TASK_PATH}/task-123")))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {
                    "task_id": "task-123",
                    "task_status": "SUCCEEDED",
                    "video_url": "javascript:alert(1)"
                }
            })))
            .mount(&upstream)
            .await;

        let app = build_app(new_snap(&upstream.uri(), "alibaba", ""));
        let id = encode_video_id(MODEL_ID, "my-video", "task-123");
        let resp = tower::ServiceExt::oneshot(app, get_uri(&format!("/v1/videos/{id}/content")))
            .await
            .unwrap();
        assert_ne!(resp.status(), StatusCode::FOUND);
        assert!(resp.headers().get(header::LOCATION).is_none());
    }

    /// MED-6 (PR #811 audit): the model client-IP allowlist gates the
    /// GET routes too. With `allowed_cidrs` set and no resolvable
    /// source IP the request is rejected before any upstream contact.
    #[tokio::test]
    async fn allowed_cidrs_enforced_on_poll_route() {
        let upstream = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("{DASHSCOPE_TASK_PATH}/task-123")))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {"task_id": "task-123", "task_status": "RUNNING"}
            })))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(
            &upstream.uri(),
            "alibaba",
            r#", "allowed_cidrs": ["203.0.113.0/24"]"#,
        );
        let app = build_app(snap);
        let id = encode_video_id(MODEL_ID, "my-video", "task-123");
        let resp = tower::ServiceExt::oneshot(app, get_uri(&format!("/v1/videos/{id}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["code"], "ip_restricted");
    }
}
