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

use aisix_core::AppliedGuardrail;
use aisix_gateway::{BridgeContext, BridgeError};
use aisix_obs::{content_capture_cap, AccessLog, CapturedContent, RequestOutcome, UsageEvent};
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
use crate::state::ProxyState;

/// Per-request payload from a successful dispatch — carries the
/// response + the bits the handler needs to emit a UsageEvent (#407).
struct ImageDispatchSuccess {
    response: Response,
    provider: String,
    /// UUID of the resolved Model row — required for UsageEvent
    /// `model_id`. Always present on success.
    model_id: String,
    /// Resolved ProviderKey UUID — feeds per-PK telemetry attribution
    /// (AISIX-Cloud#867 parity).
    provider_key_id: String,
    /// The `{kind, hook}` set of guardrails that governed this request (#379
    /// parity) — surfaced on the emitted UsageEvent.
    applied_guardrails: Vec<AppliedGuardrail>,
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
    /// Per-detector PII mask counts (#932/#696) applied to the prompt.
    /// Attached to the emitted UsageEvent. Empty = no redaction.
    redactions: crate::redact::RedactionCounts,
    /// Monitor-mode guardrail observations (AISIX-Cloud#562).
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    /// Captured request/response content for content-capturing exporters
    /// (#700, LiteLLM parity: the full image response JSON — url or
    /// b64_json — truncated at the capture cap). `Some` only when an
    /// exporter opted into `content_mode = full`.
    captured_content: Option<CapturedContent>,
}

pub async fn image_generations(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    Json(body): Json<Value>,
) -> Response {
    let started = Instant::now();
    let request_id = client.request_id.clone();
    let api_key_id = auth.entry.id.clone();
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    match dispatch(&state, &auth, body, &request_id, &client.source_ip).await {
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
                    &model_name,
                    &api_key_id,
                    &success.provider_key_id,
                    &success.applied_guardrails,
                    200,
                    elapsed,
                    prompt_tokens,
                    completion_tokens,
                    &client,
                    success.redactions.clone(),
                    success.monitor_hits.clone(),
                    success.captured_content.as_ref(),
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
            let snap = state.snapshot.load();
            let metric_model = crate::usage_attr::metric_model_label(&snap, &model_name);
            state.metrics.record_request(
                "unknown",
                metric_model,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            // Per #655 parity: surface the failed request in Logs with a
            // zero-token event (status + error class), instead of dropping it.
            crate::usage_attr::emit_error_usage_event(
                &state,
                "images",
                "openai",
                &request_id,
                &model_name,
                &api_key_id,
                status,
                err.kind(),
                &client,
            );
            err.into_response()
        }
    }
}

/// Build a [`ChatFormat`](aisix_gateway::ChatFormat) from the image
/// generation `prompt` so the input guardrail chain can scan it (#545).
/// Never sent upstream.
fn images_input_to_chat(model: &str, body: &Value) -> aisix_gateway::ChatFormat {
    let messages = match body.get("prompt").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => vec![aisix_gateway::ChatMessage::user(s.to_string())],
        _ => Vec::new(),
    };
    aisix_gateway::ChatFormat::new(model, messages)
}

async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    mut body: Value,
    request_id: &str,
    source_ip: &str,
) -> Result<ImageDispatchSuccess, ProxyError> {
    // Owned so the #696 in-place prompt masking below can borrow `body`
    // mutably.
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::InvalidRequest("missing `model` field".into()))?
        .to_string();
    let model_name = model_name.as_str();

    let snapshot = state.snapshot.load();

    let model_entry = crate::model_resolve::resolve_model(&snapshot, model_name)
        .ok_or_else(|| ProxyError::ModelNotFound(model_name.to_string()))?;

    if !auth.key().can_access(model_name) {
        return Err(ProxyError::ModelForbidden(model_name.to_string()));
    }

    // Client-IP allowlist gate (#557): reject before guardrails / upstream.
    crate::dispatch::check_ip_access(&model_entry.value, source_ip)?;

    // #545: /v1/images/generations must run input guardrails. Before this it
    // forwarded the user `prompt` with no configured content/DLP check, so a
    // block enforced on /v1/chat/completions was bypassable by switching
    // surface. Run before the rate-limit reservation so a content-policy
    // refusal doesn't burn an RPM slot. (Output is an image, not scannable
    // text, so there is no output hook.)
    let guardrail_ctx = aisix_guardrails::RequestContext {
        model_id: &model_entry.id,
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved_chain = state.guardrail_index.resolve(&guardrail_ctx);
    // Record which guardrails govern this request (#379 parity) so the emitted
    // UsageEvent surfaces them in Logs, like chat / messages. Empty when no
    // guardrail is attached.
    let applied_guardrails = resolved_chain.applied().to_vec();
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    if !resolved_chain.is_empty() {
        let chat = images_input_to_chat(model_name, &body);
        let (verdict, hits) =
            aisix_guardrails::Guardrail::check_input_observed(&resolved_chain, &chat).await;
        monitor_hits.extend(hits);
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } = verdict
        {
            // Per #153 the matched-pattern detail stays in ops logs only.
            tracing::warn!(
                guardrail_hook = "input",
                model = %model_name,
                reason = %reason,
                "guardrail blocked /v1/images/generations request",
            );
            return Err(ProxyError::ContentFiltered(
                crate::error::guardrail_block_message("request", guardrail_name.as_deref()),
            ));
        }
    }

    // #932/#696: mask-action PII rules rewrite the `prompt` in place AFTER
    // the block check passes, BEFORE the body is forwarded upstream.
    // Pre-#696 a mask-action detector was a silent no-op here.
    let redactions = crate::redact::redact_images_request(&resolved_chain, &mut body);

    // Content capture (#700): the client-facing request body
    // (post-#932-redaction) is the prompt; the response is the image JSON
    // (url or b64_json — LiteLLM logs it unstripped too), truncated at the
    // cap.
    let content_cap = content_capture_cap(
        snapshot
            .observability_exporters
            .entries()
            .iter()
            .map(|e| &e.value),
    );
    let captured_prompt = content_cap.map(|_| serde_json::to_string(&body).unwrap_or_default());

    let model_rl =
        crate::quota::ModelRateLimit::from_model(model_name, &model_entry.id, &model_entry.value);
    let reservation = crate::quota::enforce(state, auth, Some(&model_rl)).await?;

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
    // #554: apply the configured request `timeout` as the upstream deadline.
    let mut ctx = BridgeContext::new(request_id, model_arc, pk_arc);
    if let Some(d) = model.request_timeout() {
        ctx = ctx.with_deadline(d);
    }

    let provider_label = provider.to_ascii_lowercase();

    match bridge.generate_image(&body, &ctx).await {
        Ok(resp_json) => {
            // #701: clear any cooldown/unhealthy mark now the upstream
            // answered — same recovery signal as rerank/audio/chat.
            state.health.record_success(model_name);
            state.runtime_status.mark_healthy(&model_entry.id);
            // Extract usage tokens (gpt-image-1 returns a `usage` block;
            // dall-e-3 doesn't) BEFORE moving resp_json into the
            // Response, so the success struct carries typed counters.
            let usage = extract_token_usage(&resp_json);
            // #911 [21]: commit the actual token cost so TPM/TPD is enforced
            // for /v1/images/generations like chat + embeddings. Pre-fix the
            // reservation dropped uncommitted and the token counter never moved.
            let total_tokens = usage
                .map(|(prompt, completion)| u64::from(prompt) + u64::from(completion))
                .unwrap_or(0);
            reservation.commit_tokens(total_tokens).await;
            // Content capture (#700): the full response JSON (LiteLLM
            // parity); CapturedContent::new truncates to the cap.
            let captured_content = match (&captured_prompt, content_cap) {
                (Some(prompt), Some(cap)) => Some(CapturedContent::new(
                    prompt,
                    &serde_json::to_string(&resp_json).unwrap_or_default(),
                    cap as usize,
                )),
                _ => None,
            };
            Ok(ImageDispatchSuccess {
                response: Json(resp_json).into_response(),
                provider: provider_label,
                model_id: model_entry.id.to_string(),
                provider_key_id: pk_entry.id.to_string(),
                applied_guardrails: applied_guardrails.clone(),
                usage,
                upstream_called: true,
                redactions,
                monitor_hits: monitor_hits.clone(),
                captured_content,
            })
        }
        Err(BridgeError::Config(msg)) if msg.contains("does not support image generation") => {
            // No upstream call → no tokens to count; release the reservation.
            reservation.commit_tokens(0).await;
            let env = ErrorEnvelope::new(msg, "not_implemented");
            Ok(ImageDispatchSuccess {
                response: (StatusCode::NOT_IMPLEMENTED, Json(env)).into_response(),
                provider: provider_label,
                model_id: model_entry.id.to_string(),
                provider_key_id: pk_entry.id.to_string(),
                applied_guardrails: applied_guardrails.clone(),
                usage: None,
                // No upstream call happened → handler skips emit.
                upstream_called: false,
                redactions,
                monitor_hits: monitor_hits.clone(),
                captured_content: None,
            })
        }
        Err(e) => {
            reservation.commit_tokens(0).await;
            // #701: mark the failure on the runtime status so the cooldown /
            // circuit-breaker sees flapping upstreams reached only via this
            // endpoint — same policy as rerank/audio/chat. `note_failure` is
            // a no-op for non-triggering categories (e.g. Config errors).
            let e = crate::cooldown::note_failure(
                &state.runtime_status,
                &model_entry.id,
                model.cooldown.as_ref(),
                e,
            );
            Err(ProxyError::Bridge(e))
        }
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
///
/// The per-PK attribution tags (provider_kind / provider_featured /
/// branded_provider / pk_label / byo_label) are populated from the
/// resolved ProviderKey — same lookup as chat / messages / responses /
/// embeddings (AISIX-Cloud#867 parity) via `usage_attr::apply_pk_telemetry`.
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    requested_model: &str,
    api_key_id: &str,
    provider_key_id: &str,
    applied_guardrails: &[AppliedGuardrail],
    status_code: u16,
    elapsed: Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
    client: &ClientContext,
    // Per-detector PII mask counts (#932/#696). Empty = no redaction.
    redacted_entity_counts: crate::redact::RedactionCounts,
    // Monitor-mode guardrail observations (AISIX-Cloud#562).
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    // Captured request/response content (#700). Forwarded only to `fan_out`,
    // never to the CP sink.
    content: Option<&CapturedContent>,
) {
    let snap = state.snapshot.load();
    let mut event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        prompt_tokens,
        completion_tokens,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "openai".to_string(),
        applied_guardrails: applied_guardrails.to_vec(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        redacted_entity_counts,
        guardrail_monitor_hits,
        ..Default::default()
    };
    crate::usage_attr::apply_pk_telemetry(&mut event, &snap, provider_key_id);
    // Handler label "images" — bucketed prometheus counter (#408).
    state.usage_sink.try_emit("images", event.clone());
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, content, exporters.iter().map(|e| &e.value));
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

    /// AISIX-Cloud#867: same openai PK as `provider_key_entry` but carrying
    /// `telemetry_tags`, so the emitted UsageEvent gets the per-PK
    /// attribution fields stamped via `usage_attr::apply_pk_telemetry`.
    fn provider_key_entry_tagged(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-up","api_base":"{api_base}","provider":"openai","adapter":"openai","telemetry_tags":{{"kind":"catalog","featured":true,"branded_provider":"openai","pk_label":"prod-images-key"}}}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(provider_key_entry(api_base));
        snap
    }

    fn new_snap_tagged(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry_tagged(api_base));
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

    fn keyword_input_guardrail(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"t","enabled":true,"hook_point":"input","fail_open":false,"kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-1", g, 1)
    }

    /// #545: a configured input guardrail must fire on /v1/images/generations
    /// — a blocked `prompt` returns 422 content_filter, upstream never hit.
    #[tokio::test]
    async fn input_guardrail_blocks_prompt_returns_422() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("dalle"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "dalle", "prompt": "draw BLOCKME please"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        assert!(!v["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("BLOCKME"));
    }

    /// #545 companion: a benign prompt with a guardrail configured forwards
    /// (`expect(1)`) and returns 200.
    #[tokio::test]
    async fn input_guardrail_allows_benign_prompt_forwards_200() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("dalle"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "dalle", "prompt": "a serene landscape"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["data"][0]["url"].is_string());
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

    /// AISIX-Cloud#867 parity: a successful /v1/images/generations 200 must
    /// stamp the per-PK telemetry attribution fields (provider_kind /
    /// provider_featured / branded_provider / pk_label) on the emitted
    /// UsageEvent, sourced from the resolved ProviderKey's `telemetry_tags`
    /// — exactly like /v1/chat/completions, /v1/messages, /v1/responses, and
    /// /v1/embeddings. Pre-fix these five fields were left at Default, so
    /// image-generation spend showed up unattributed in cp-api analytics.
    #[tokio::test]
    async fn emits_provider_telemetry_tags_issue_867() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap_tagged(&upstream.uri());
        snap.models.insert(model_entry("dall-e"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let req = serde_json::json!({"model": "dall-e", "prompt": "a cat", "n": 1});
        let resp = tower::ServiceExt::oneshot(app, make_req(req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/images/generations 200")
            .expect("usage_sink sender dropped");
        assert_eq!(
            ev.provider_kind, "catalog",
            "provider_kind must mirror the resolved PK's telemetry_tags.kind"
        );
        assert!(
            ev.provider_featured,
            "provider_featured must mirror telemetry_tags.featured"
        );
        assert_eq!(
            ev.branded_provider, "openai",
            "branded_provider must mirror telemetry_tags.branded_provider"
        );
        assert_eq!(
            ev.pk_label, "prod-images-key",
            "pk_label must mirror telemetry_tags.pk_label"
        );
    }

    /// #379 parity: a successful /v1/images/generations 200 governed by a
    /// configured input guardrail must surface the applied `{kind, hook}` set
    /// on the emitted UsageEvent — exactly like chat / messages / embeddings.
    /// A benign prompt that does NOT match the keyword guardrail passes through
    /// (200), and the event records the guardrail that governed the request.
    #[tokio::test]
    async fn applied_guardrails_recorded_on_usage_event() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("dall-e"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        // Benign prompt — does not contain the blocked literal, so it passes
        // the guardrail and reaches the upstream.
        let req = serde_json::json!({"model": "dall-e", "prompt": "a serene landscape", "n": 1});
        let resp = tower::ServiceExt::oneshot(app, make_req(req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/images/generations 200")
            .expect("usage_sink sender dropped");
        assert!(
            !ev.applied_guardrails.is_empty(),
            "the configured guardrail must be recorded on the UsageEvent"
        );
        assert_eq!(
            ev.applied_guardrails[0].kind, "keyword",
            "applied guardrail kind must mirror the configured guardrail"
        );
        assert_eq!(
            ev.applied_guardrails[0].hook, "input",
            "applied guardrail hook must mirror the configured hook_point"
        );
    }

    /// #655 parity: an upstream 5xx on /v1/images/generations now emits ONE
    /// zero-token UsageEvent so the failed request is visible in Logs (status +
    /// error class) and attributed to the api_key — instead of being dropped.
    /// Mirrors `completions.rs::upstream_5xx_emits_zero_token_error_event`.
    #[tokio::test]
    async fn upstream_5xx_emits_zero_token_error_event() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("dall-e"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let req = serde_json::json!({"model": "dall-e", "prompt": "a cat", "n": 1});
        let resp = tower::ServiceExt::oneshot(app, make_req(req))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("a failed /v1/images/generations must emit a zero-token UsageEvent")
            .expect("usage_sink sender dropped");
        assert_eq!(ev.status_code, 502, "upstream 5xx maps to 502");
        assert_eq!(ev.prompt_tokens, 0);
        assert_eq!(ev.api_key_id, "k-1");
        assert_eq!(ev.requested_model, "dall-e");
        assert!(
            !ev.error_class.is_empty(),
            "error_class must classify the failure"
        );
        assert!(
            rx.try_recv().is_err(),
            "exactly one event per failed request"
        );
    }

    fn pii_mask_guardrail() -> ResourceEntry<aisix_core::Guardrail> {
        let json = r#"{"name":"pii","enabled":true,"hook_point":"input","kind":"pii","detectors":[{"type":"email","action":"mask"}]}"#;
        let g: aisix_core::Guardrail = serde_json::from_str(json).unwrap();
        ResourceEntry::new("g-pii", g, 1)
    }

    /// #696: a mask-action PII detector must rewrite the `prompt` before the
    /// body reaches the upstream. Pre-#696 the mask action was a silent
    /// no-op on /v1/images/generations — the raw text was forwarded. Also
    /// pins the counts on the emitted UsageEvent.
    #[tokio::test]
    async fn pii_mask_rewrites_prompt_before_upstream_issue_696() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "created": 1_700_000_000,
                "data": [{"url": "https://img.example/1.png"}]
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("img"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(pii_mask_guardrail());

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let body = serde_json::json!({
            "model": "img",
            "prompt": "a portrait of a@x.com at sunset"
        });
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let reqs = upstream.received_requests().await.unwrap();
        let sent = String::from_utf8_lossy(&reqs[0].body).into_owned();
        assert!(sent.contains("[EMAIL_REDACTED]"), "sent: {sent}");
        assert!(
            !sent.contains("a@x.com"),
            "raw PII forwarded upstream: {sent}"
        );

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted")
            .expect("usage_sink sender dropped");
        assert_eq!(
            event.redacted_entity_counts.get("email"),
            Some(&1),
            "mask counts must reach the UsageEvent"
        );
    }

    /// #701: an upstream 5xx must mark the model's runtime status (cooldown)
    /// — /v1/images/generations previously never touched it.
    #[tokio::test]
    async fn upstream_5xx_marks_cooldown_issue_701() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images/generations"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("img"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg()).without_cache();
        let app = crate::build_router(state.clone());

        let body = serde_json::json!({"model": "img", "prompt": "a cat"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let status = state.runtime_status.status("m-1");
        assert!(
            status.cooldown_until.is_some(),
            "a 500 must mark the model in cooldown, got {status:?}"
        );
    }
}
