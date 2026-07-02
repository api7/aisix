//! `POST /v1/embeddings` — OpenAI-compatible embeddings pass-through.
//!
//! Flow:
//! 1. [`AuthenticatedKey`] extractor — 401 if auth fails.
//! 2. Parse [`EmbeddingRequestBody`] from JSON.
//! 3. Resolve model name → `Model` in snapshot → 404 if absent.
//! 4. Check `allowed_models` → 403 if denied.
//! 5. Look up Bridge on Hub → 503 if not registered.
//! 6. Normalise `input` (single string → one-element vec).
//! 7. Call `bridge.embed(req, ctx)` → forward response as JSON.
//! 8. On completion: record metrics and emit access log.
//!
//! Errors follow the same OpenAI-style envelope as chat completions.
//! Providers that don't implement embeddings return a 501 with
//! `"type": "not_implemented"`.

use aisix_core::AppliedGuardrail;
use aisix_gateway::{BridgeContext, BridgeError, ChatFormat, ChatMessage, EmbeddingRequest};
use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::{ErrorEnvelope, ProxyError};
use crate::request_id::new_request_id;
use crate::state::ProxyState;

/// The request body accepted by `POST /v1/embeddings`.
///
/// `input` may be a single string **or** an array of strings; both are
/// handled by the `InputField` helper so callers don't need to know.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequestBody {
    pub model: String,
    pub input: InputField,
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<u32>,
}

/// Deserialises both `"text"` and `["text", ...]` forms of the
/// OpenAI embeddings `input` field.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputField {
    Single(String),
    Multi(Vec<String>),
}

impl InputField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            InputField::Single(s) => vec![s],
            InputField::Multi(v) => v,
        }
    }
}

/// Build a [`ChatFormat`] of user messages from the embeddings `input` so
/// the input guardrail chain can scan it (#719). Each non-empty string
/// becomes one user message; the synthesized request is never sent
/// upstream (the original `input` shape is forwarded verbatim).
fn embeddings_input_to_chat(model: &str, input: &InputField) -> ChatFormat {
    let messages = match input {
        InputField::Single(s) if !s.is_empty() => vec![ChatMessage::user(s.clone())],
        InputField::Single(_) => Vec::new(),
        InputField::Multi(v) => v
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| ChatMessage::user(s.clone()))
            .collect(),
    };
    ChatFormat::new(model, messages)
}

pub async fn embeddings(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    // Issue #401 (follow-up to #324): catch the JSON-extractor
    // rejection here so we can map it to OpenAI's documented 400
    // invalid_request_error wire shape. Axum's `Json<T>` extractor
    // returns 422 on JsonDataError (valid-JSON-but-missing-required-
    // field, e.g. no `model` / no `input`), which diverges from
    // OpenAI — every SDK that branches on 400 vs 422 sees different
    // semantics here than it does talking to api.openai.com. Same
    // discriminate-then-map pattern chat.rs uses for #324.
    body: Result<Json<EmbeddingRequestBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let started = Instant::now();
    let request_id = new_request_id();
    let api_key_id = auth.entry.id.clone();
    let body = match body {
        Ok(Json(b)) => b,
        Err(rej) => {
            use axum::extract::rejection::JsonRejection;
            // BytesRejection → distinguish 413 (PAYLOAD_TOO_LARGE,
            // real per-extractor cap exceeded) from 400 (transport-
            // side read failure). `JsonRejection` is `#[non_exhaustive]`
            // so the fallback `_` arm catches today's JsonDataError
            // (the #401 case) / JsonSyntaxError / MissingJsonContentType
            // AND any future variant axum adds, defaulting to 400
            // until each new variant gets an explicit policy decision.
            return match rej {
                JsonRejection::BytesRejection(inner)
                    if inner.status() == StatusCode::PAYLOAD_TOO_LARGE =>
                {
                    ProxyError::RequestTooLarge {
                        limit_bytes: state.request_body_limit_bytes,
                    }
                }
                JsonRejection::BytesRejection(_) => {
                    ProxyError::InvalidRequest("failed to read request body".into())
                }
                _ => ProxyError::InvalidRequest("invalid JSON request body".into()),
            }
            .into_response();
        }
    };
    let model_name = body.model.clone();

    match dispatch(&state, &auth, body, &request_id, &client.source_ip).await {
        Ok(success) => {
            let elapsed = started.elapsed();
            let status = 200u16;
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
                RequestOutcome::Success,
                elapsed,
            );
            // Issue #226: emit UsageEvent so cp-api's budget ledger
            // and customer-facing /logs analytics see embeddings
            // spend. Pre-#226 the embedding handler dropped the
            // event entirely, so any /v1/embeddings traffic was
            // invisible to budget enforcement and billing
            // reconciliation. Skip when `upstream_called == false`
            // — the dispatch's 501-NotImplemented path returns the
            // false flag because no upstream call happened, and
            // attributing a zero-everything event to the api_key
            // would bloat /logs with noise. Distinguished from
            // `prompt_tokens == 0` so a 200 with legitimately zero
            // tokens (empty input, provider-specific billing
            // convention) still emits. Same emit-on-success-only
            // convention as chat.rs.
            if success.upstream_called {
                emit_usage_event(
                    &state,
                    &request_id,
                    &success.model_id,
                    &model_name,
                    &api_key_id,
                    &success.provider_key_id,
                    &success.applied_guardrails,
                    status,
                    elapsed,
                    success.prompt_tokens,
                    &client,
                    success.redactions.clone(),
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
                "embeddings",
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

/// Per-request payload from a successful dispatch — carries
/// everything the handler needs to emit access-log + UsageEvent +
/// the actual HTTP response. Pre-#226 the dispatch only returned
/// the response + provider label; expanding to a struct surfaces
/// the model_id + prompt_tokens the UsageEvent emission needs
/// without threading two more positional args through the handler.
struct EmbedDispatchSuccess {
    response: Response,
    provider: String,
    model_id: String,
    /// Resolved ProviderKey UUID — feeds the per-PK telemetry attribution
    /// tags on the emitted UsageEvent (AISIX-Cloud#867 parity).
    provider_key_id: String,
    /// The `{kind, hook}` set of guardrails that governed this request (#379
    /// parity) — surfaced on the emitted UsageEvent.
    applied_guardrails: Vec<AppliedGuardrail>,
    /// Per-detector PII mask counts (#932) applied to the input.
    redactions: crate::redact::RedactionCounts,
    prompt_tokens: u32,
    /// `true` when the dispatch produced a real 200 from the upstream
    /// (we have authoritative usage data to attribute). `false` for the
    /// 501-NotImplemented branch where no upstream call was made.
    ///
    /// Explicitly distinguished from `prompt_tokens == 0` so a future
    /// provider that legitimately reports zero tokens on a 200 (empty
    /// input, special billing convention) still gets emitted — the
    /// "no upstream call" channel must not piggyback on a numeric
    /// sentinel that real responses could also produce.
    upstream_called: bool,
}

async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    mut body: EmbeddingRequestBody,
    request_id: &str,
    source_ip: &str,
) -> Result<EmbedDispatchSuccess, ProxyError> {
    let snapshot = state.snapshot.load();

    let model_entry = crate::model_resolve::resolve_model(&snapshot, &body.model)
        .ok_or_else(|| ProxyError::ModelNotFound(body.model.clone()))?;

    if !auth.key().can_access(&body.model) {
        return Err(ProxyError::ModelForbidden(body.model.clone()));
    }

    // Client-IP allowlist gate (#557): reject before guardrails / upstream.
    crate::dispatch::check_ip_access(&model_entry.value, source_ip)?;

    let model = &model_entry.value;
    let provider = crate::dispatch::require_provider(model)?;
    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;

    let bridge = crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value)
        .ok_or(ProxyError::ProviderUnavailable)?;

    // #719: /v1/embeddings must run input guardrails. Before this the
    // handler explicitly bypassed all guardrails, so a content block
    // enforced on /v1/chat/completions was bypassable by sending the same
    // text to /v1/embeddings. The embeddings `input` (a string or array of
    // strings) carries scannable user content; translate it into the
    // internal ChatFormat and run the resolved input guardrail chain. A
    // Block short-circuits before the upstream call. (Embeddings responses
    // are vectors, not text, so there is no output hook to run.)
    //
    // #542: run this BEFORE the rate-limit reservation so a content-policy
    // block doesn't burn an RPM slot (matching /v1/chat/completions).
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
    if !resolved_chain.is_empty() {
        let chat = embeddings_input_to_chat(&body.model, &body.input);
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } = aisix_guardrails::Guardrail::check_input(&resolved_chain, &chat).await
        {
            // Per #153 keep the matched-pattern detail in ops logs only; the
            // wire envelope names only the guardrail that fired (#519 B.4b).
            tracing::warn!(
                guardrail_hook = "input",
                model = %body.model,
                reason = %reason,
                "guardrail blocked /v1/embeddings request",
            );
            return Err(ProxyError::ContentFiltered(
                crate::error::guardrail_block_message("request", guardrail_name.as_deref()),
            ));
        }
    }

    // #932: mask-action PII rules rewrite the embeddings input in place
    // AFTER the block check passes — the masked text is what gets embedded,
    // so the matched values never reach the provider.
    let mut redactions = crate::redact::RedactionCounts::new();
    if aisix_guardrails::Guardrail::redacts_input(&resolved_chain) {
        match &mut body.input {
            InputField::Single(s) => {
                if let Some(r) = aisix_guardrails::Guardrail::redact_input_text(&resolved_chain, s)
                {
                    *s = r.text;
                    crate::redact::merge_counts(&mut redactions, r.counts);
                }
            }
            InputField::Multi(items) => {
                for s in items {
                    if let Some(r) =
                        aisix_guardrails::Guardrail::redact_input_text(&resolved_chain, s)
                    {
                        *s = r.text;
                        crate::redact::merge_counts(&mut redactions, r.counts);
                    }
                }
            }
        }
    }

    let model_rl =
        crate::quota::ModelRateLimit::from_model(&body.model, &model_entry.id, &model_entry.value);
    let reservation = crate::quota::enforce(state, auth, Some(&model_rl)).await?;

    let upstream_model_id = crate::dispatch::require_upstream_model(model)?.to_string();

    // Preserve the caller's original `input` shape per #162 /
    // `docs/api-proxy.md` §4.4 "both pass through". The bridge will
    // use this flag to serialise the upstream wire body as either a
    // single string or an array — without it, the gateway always
    // forwarded `["text"]` even when the caller sent `"text"`,
    // which contradicts the docs and confuses operator-side packet
    // captures during billing reconciliation / debugging.
    let input_was_single = matches!(body.input, InputField::Single(_));
    let req = EmbeddingRequest {
        model: upstream_model_id,
        input: body.input.into_vec(),
        input_was_single,
        encoding_format: body.encoding_format,
        dimensions: body.dimensions,
    };

    let model_arc = Arc::new(model.clone());
    let pk_arc = Arc::new(pk_entry.value.clone());
    // #554: apply the configured request `timeout` as the upstream deadline.
    let mut ctx = BridgeContext::new(request_id, model_arc, pk_arc);
    if let Some(d) = model.request_timeout() {
        ctx = ctx.with_deadline(d);
    }

    match bridge.embed(&req, &ctx).await {
        Ok(embed_resp) => {
            // #701: clear any cooldown/unhealthy mark now the upstream
            // answered — same recovery signal as rerank/audio/chat.
            state.health.record_success(&body.model);
            state.runtime_status.mark_healthy(&model_entry.id);
            // Commit the reservation — release the concurrency permit
            // and finalise RPM. Embeddings do report prompt_tokens via
            // EmbeddingResponse.usage; thread it through so TPM works
            // here even though other handlers commit 0.
            reservation
                .commit_tokens(embed_resp.usage.total_tokens as u64)
                .await;
            let provider_label = provider.to_ascii_lowercase();
            // Capture the prompt_tokens count BEFORE moving the
            // embed_resp into the JSON response — the handler needs
            // this for UsageEvent emission downstream (#226).
            let prompt_tokens = embed_resp.usage.prompt_tokens;
            Ok(EmbedDispatchSuccess {
                response: Json(embed_resp).into_response(),
                provider: provider_label,
                model_id: model_entry.id.to_string(),
                provider_key_id: pk_entry.id.to_string(),
                applied_guardrails: applied_guardrails.clone(),
                redactions: redactions.clone(),
                prompt_tokens,
                upstream_called: true,
            })
        }
        Err(BridgeError::Config(msg)) if msg.contains("does not support embeddings") => {
            // Provider doesn't implement embed → 501 Not Implemented.
            // Drop the reservation without committing — the request
            // didn't hit the upstream. No UsageEvent emission either
            // (`upstream_called: false` → handler skips emit per the
            // chat.rs convention that we only attribute usage on a
            // real upstream completion).
            reservation.commit_tokens(0).await;
            let env = ErrorEnvelope::new(msg, "not_implemented");
            Ok(EmbedDispatchSuccess {
                response: (StatusCode::NOT_IMPLEMENTED, Json(env)).into_response(),
                provider: provider.to_ascii_lowercase(),
                model_id: model_entry.id.to_string(),
                provider_key_id: pk_entry.id.to_string(),
                applied_guardrails: applied_guardrails.clone(),
                redactions: redactions.clone(),
                prompt_tokens: 0,
                // No upstream call happened — the handler reads this
                // and skips UsageEvent emission. Distinguished from
                // `prompt_tokens == 0` so a 200 that legitimately
                // reports zero tokens still emits.
                upstream_called: false,
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

fn emit_access_log(
    model: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    latency: Duration,
    request_id: &str,
) {
    let now_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let _ = now_ts; // only used for context; access log uses elapsed
    AccessLog {
        method: "POST",
        path: "/v1/embeddings",
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

/// Push one `UsageEvent` onto cp-api's telemetry sink **and** fan it
/// out to every per-env OTLP/HTTP exporter in the live snapshot.
/// Non-blocking on both legs: the CP sink drops on full queue, the
/// OTLP fan-out detaches a tokio task per exporter. Mirrors the
/// shape of chat.rs::emit_usage_event for the fields that matter
/// to /v1/embeddings:
///
///   - `completion_tokens = 0` — embeddings have no completion side.
///   - `inbound_protocol = "openai"` — match chat.rs convention.
///   - No cache / streaming / reasoning / finish_reason metadata —
///     none of these concepts apply to the embeddings endpoint;
///     cp-api reads these UsageEvent fields with `omitempty`-equivalent
///     defaults so leaving them zero is the same as omitting.
///   - `cost_usd = 0.0` — cp-api computes cost server-side from
///     pricing catalog + token counts on ingestion (same convention
///     as every chat.rs emit site).
///
/// Issue #226. /v1/embeddings is the first non-chat handler to gain
/// emission; follow-ups for completions / responses / rerank /
/// audio / images each get their own PR with the same shape.
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
    client: &ClientContext,
    // Per-detector PII mask counts (#932) applied to the input.
    redacted_entity_counts: crate::redact::RedactionCounts,
) {
    // Only populate fields meaningful to /v1/embeddings; rely on
    // UsageEvent's `#[derive(Default)]` for everything else. Wire-level
    // empty / zero / false maps to NULL on cp-api via skip_serializing_if,
    // identical to the legacy "field absent" semantics older DP images
    // emitted. Specifically left at Default:
    //   - completion_tokens / cached_prompt_tokens / reasoning_tokens /
    //     cache_creation_tokens / cache_read_tokens — embeddings have
    //     no completion side and no cache token concepts
    //   - provider_request_id / provider_model_version / finish_reason
    //     — not exposed by the OpenAI embeddings response shape
    //   - cost_usd — cp-api computes server-side from pricing catalog
    //   - guardrail_blocked — a blocked input short-circuits before this
    //     emit (success-only path), so it is never set here
    //   - guardrail_bypassed_reason — embeddings now run input guardrails
    //     (#719); fail-open bypass telemetry is not yet plumbed for the
    //     non-chat handlers (follow-up, same as the per-PK fields below)
    //   - cache_status / cache_hit_saved_* — no caching on embeddings
    //   - ttft_ms — embeddings are not streamed
    //   - served_by_model / routing_* — embeddings don't run routing
    //
    // The per-PK attribution tags (provider_kind / provider_featured /
    // branded_provider / pk_label / byo_label) ARE populated — same lookup as
    // chat / messages / responses (AISIX-Cloud#867 parity) via
    // `usage_attr::apply_pk_telemetry` below.
    let snap = state.snapshot.load();
    let mut event = UsageEvent {
        request_id: request_id.to_string(),
        // RFC 3339 UTC. cp-api parses with time.Parse(time.RFC3339, ...);
        // chrono's `to_rfc3339_opts(Secs, true)` emits the trailing Z.
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        prompt_tokens,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "openai".to_string(),
        applied_guardrails: applied_guardrails.to_vec(),
        redacted_entity_counts,
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        ..Default::default()
    };
    crate::usage_attr::apply_pk_telemetry(&mut event, &snap, provider_key_id);
    // Handler label "embeddings" — bucketed prometheus counter (#408).
    state.usage_sink.try_emit("embeddings", event.clone());
    // Per-env OTLP/HTTP fan-out — same shape as chat.rs:1334. The
    // snapshot's exporter table is empty for envs that haven't
    // configured any, so this is a cheap no-op on the common path.
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
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
                "model_name": "text-embedding-3-small",
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

    /// A PK carrying per-PK telemetry attribution tags (AISIX-Cloud#867
    /// parity) so emitted UsageEvents can be asserted to surface the upstream
    /// vendor + PK label the dashboard's Logs detail shows.
    fn provider_key_entry_tagged(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-up","api_base":"{api_base}","provider":"openai","adapter":"openai","telemetry_tags":{{"kind":"catalog","featured":true,"branded_provider":"openai","pk_label":"prod-embeddings-key"}}}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
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
            .uri("/v1/embeddings")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    fn upstream_response() -> serde_json::Value {
        serde_json::json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": [0.1_f32, 0.2_f32, 0.3_f32]
            }],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        })
    }

    /// An env-scoped keyword input guardrail (no attachment row → applies
    /// to every request via the backward-compat fallback) that blocks on a
    /// literal. `fail_open:false` is irrelevant for keyword (local, never
    /// errors) but keeps the row explicit.
    fn keyword_input_guardrail(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"test-block","enabled":true,"hook_point":"input","fail_open":false,"kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-1", g, 1)
    }

    /// #379 parity: a successful /v1/embeddings request whose input passes an
    /// attached input guardrail records that guardrail's `{kind, hook}` in the
    /// emitted UsageEvent's `applied_guardrails`. Before the fix the field was
    /// left empty on the non-chat handlers.
    #[tokio::test]
    async fn applied_guardrails_recorded_on_usage_event() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1_f32]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 3, "total_tokens": 3}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        // Benign input (does not contain "BLOCKME") → passes the guardrail.
        let body = serde_json::json!({"model": "my-embed", "input": "hello world"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted")
            .expect("usage_sink sender dropped");
        assert!(
            !ev.applied_guardrails.is_empty(),
            "the attached input guardrail must be recorded"
        );
        assert_eq!(ev.applied_guardrails[0].kind, "keyword");
        assert_eq!(ev.applied_guardrails[0].hook, "input");
    }

    /// #719: /v1/embeddings explicitly bypassed all guardrails, so a content
    /// block configured on /v1/chat/completions was evadable by sending the
    /// same text to /v1/embeddings. A configured input guardrail must now
    /// fire here: a blocked literal returns 422 content_filter and the
    /// upstream is never contacted (`expect(0)`).
    #[tokio::test]
    async fn input_guardrail_blocks_string_input_returns_422_content_filter() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "please BLOCKME now"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        // Per #153 the matched literal must not leak into the wire message.
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(!msg.contains("BLOCKME"), "blocklist literal leaked: {msg}");
    }

    /// #719: the array `input` form must be scanned too — a blocked literal
    /// in any element blocks the call.
    #[tokio::test]
    async fn input_guardrail_blocks_array_input() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let body =
            serde_json::json!({"model": "my-embed", "input": ["totally fine", "now BLOCKME"]});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #719 companion: a benign input with a configured input guardrail must
    /// still reach the upstream (`expect(1)`) and return 200 — the guardrail
    /// must not block clean traffic.
    #[tokio::test]
    async fn input_guardrail_allows_benign_input_forwards_200() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "a perfectly fine request"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
    }

    #[tokio::test]
    async fn happy_path_single_string_input() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hello world"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["object"], "embedding");
        let emb = v["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(emb.len(), 3);
    }

    #[tokio::test]
    async fn happy_path_array_input() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": ["a", "b"]});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Issue #162 regression: when the caller's `input` is a single
    /// string, the upstream wire body MUST be a single string (NOT a
    /// one-element array). Per `docs/api-proxy.md` §4.4, both shapes
    /// pass through; pre-fix the gateway always sent `["text"]` to
    /// the upstream regardless of the caller's shape.
    #[tokio::test]
    async fn single_string_input_preserves_string_shape_on_upstream_wire() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hello"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain the upstream's recorded request body and inspect the
        // `input` field on the upstream wire. A regression that
        // re-introduced the always-array normalisation would write
        // `["hello"]` here.
        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1, "exactly one upstream call expected");
        let upstream_body: serde_json::Value =
            serde_json::from_slice(&received[0].body).expect("upstream body is valid JSON");
        assert!(
            upstream_body["input"].is_string(),
            "single-string caller input must reach upstream as a string, not an array; got {:?}",
            upstream_body["input"]
        );
        assert_eq!(upstream_body["input"], "hello");
    }

    /// Counterpart to the above: when the caller's `input` is an
    /// array (even single-element), the upstream wire body is also
    /// an array. Without this companion test, a regression that
    /// over-corrected to "always single-string when len==1" would
    /// silently rewrite the caller's explicit array.
    #[tokio::test]
    async fn array_input_preserves_array_shape_on_upstream_wire() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        // Caller uses array form even though there's only one
        // element. Gateway must NOT silently rewrite to a string.
        let body = serde_json::json!({"model": "my-embed", "input": ["only-one"]});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let upstream_body: serde_json::Value =
            serde_json::from_slice(&received[0].body).expect("upstream body is valid JSON");
        assert!(
            upstream_body["input"].is_array(),
            "array-form caller input must reach upstream as an array, not coerced to a string; got {:?}",
            upstream_body["input"]
        );
        let arr = upstream_body["input"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0], "only-one");
    }

    #[tokio::test]
    async fn unauthenticated_request_returns_401() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"my-embed","input":"hi"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn forbidden_model_returns_403() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["other-model"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hi"});
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
        let body = serde_json::json!({"model": "nonexistent", "input": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn upstream_error_propagates_as_502_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(503).set_body_string("overloaded"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "upstream_error");
    }

    #[tokio::test]
    async fn response_contains_usage_tokens_from_upstream() {
        // The existing `happy_path_*` tests assert response.data shape but
        // never pin the usage envelope; cp-api depends on it for billing.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1_f32, 0.2_f32]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 7, "total_tokens": 7}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hello world"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["usage"]["prompt_tokens"], 7);
        assert_eq!(v["usage"]["total_tokens"], 7);
    }

    #[tokio::test]
    async fn upstream_request_uses_provider_model_name_not_display_name() {
        // Model-alias resolution: the gateway's public display_name
        // (`my-embed`) must be rewritten to the upstream provider's
        // model id (`text-embedding-3-small`) before forwarding.
        // wiremock's body_partial_json matcher only fires on the
        // rewritten body; a 200 OK proves the alias was resolved.
        use wiremock::matchers::body_partial_json;
        let upstream = MockServer::start().await;
        // `.expect(1)` forces wiremock to assert on Drop that the mock
        // fired exactly once. The 200-status check below already catches
        // the wiremock-default-404 fallthrough path; the additional value
        // of `.expect(1)` is catching a regression class the status
        // check cannot — a future refactor that returns success WITHOUT
        // ever reaching the upstream (cached response, synthetic 200,
        // dry-run path). Status would still be 200, but the mock count
        // would be 0 and `.expect(1)` would fail on Drop.
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_partial_json(serde_json::json!({
                "model": "text-embedding-3-small"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hello"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "model alias was not rewritten to upstream provider model name"
        );
    }

    #[tokio::test]
    async fn upstream_429_propagates_status_to_client() {
        // ai-gateway's `BridgeError::UpstreamStatus` already maps 4xx
        // through (see crates/aisix-proxy/src/error.rs); this test pins
        // the contract for the embeddings path so a refactor can't
        // silently turn upstream 429 into a generic 502.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(429).set_body_string(
                r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#,
            ))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "my-embed", "input": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    /// Issue #226: a successful /v1/embeddings call must emit a
    /// `UsageEvent` onto the `usage_sink`. Pre-#226 the embeddings
    /// handler dropped the event entirely, so any /v1/embeddings
    /// traffic was invisible to cp-api's budget ledger and the
    /// customer-facing /logs analytics. This test pins the contract:
    /// after a 200 response, exactly one event arrives with the
    /// caller's prompt_tokens, model_id, status, and `inbound_protocol
    /// = "openai"` (mirroring chat.rs's emission convention).
    #[tokio::test]
    async fn emits_usage_event_on_200_with_prompt_tokens_issue_226() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // Pin a specific upstream token count so a regression that
        // hardcoded 0 (or swapped prompt/completion semantics) would
        // surface here.
        let upstream_body = serde_json::json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": [0.1_f32, 0.2_f32]
            }],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 42, "total_tokens": 42}
        });
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "my-embed", "input": "hello world"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Event must arrive within a generous window — the emit is
        // try_send (non-blocking) so a 500ms ceiling is well clear of
        // schedule jitter.
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent was never emitted for /v1/embeddings 200")
            .expect("usage_sink sender dropped before emission");

        assert_eq!(
            event.prompt_tokens, 42,
            "UsageEvent.prompt_tokens must mirror upstream usage.prompt_tokens"
        );
        assert_eq!(
            event.completion_tokens, 0,
            "embeddings have no completion side; completion_tokens must be zero"
        );
        assert_eq!(
            event.status_code, 200,
            "status_code must reflect the response"
        );
        assert_eq!(
            event.api_key_id, "k-1",
            "api_key_id must reflect the authenticated caller"
        );
        assert_eq!(
            event.model_id, "m-1",
            "model_id must reflect the resolved model row"
        );
        assert_eq!(
            event.inbound_protocol, "openai",
            "inbound_protocol must be \"openai\" for /v1/embeddings (matches chat.rs convention)"
        );
        assert!(
            !event.request_id.is_empty(),
            "request_id must be set for join with x-aisix-call-id"
        );
        assert!(
            !event.occurred_at.is_empty(),
            "occurred_at must be set (RFC 3339 UTC)"
        );
    }

    /// AISIX-Cloud#867 parity: a successful /v1/embeddings 200 must carry the
    /// resolved ProviderKey's telemetry attribution tags (provider_kind /
    /// provider_featured / branded_provider / pk_label) — same lookup as
    /// chat / messages / responses. Fails before the fix (empty tags), passes
    /// after.
    #[tokio::test]
    async fn emits_provider_telemetry_tags_issue_867() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1_f32]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 7, "total_tokens": 7}
        });
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap_tagged(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "my-embed", "input": "hello"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/embeddings 200")
            .expect("usage_sink sender dropped");
        assert_eq!(event.provider_kind, "catalog");
        assert!(event.provider_featured);
        assert_eq!(event.branded_provider, "openai");
        assert_eq!(event.pk_label, "prod-embeddings-key");
    }

    /// Issue #456 (#226 family): the 501 NotImplemented path (provider
    /// doesn't support embeddings) must NOT emit a UsageEvent — no
    /// upstream call happened, so there's nothing to attribute. Mirrors
    /// the canonical `completions.rs::provider_lacking_complete_returns_501_without_emit`.
    /// Triggers the path by routing /v1/embeddings at an Anthropic-backed
    /// model; `AnthropicBridge` doesn't override `Bridge::embed()` so the
    /// trait default returns `BridgeError::Config(...)` → 501. Without this
    /// test, a regression flipping `upstream_called: false` → `true` (or
    /// `usage: None` → `Some(zero)`) on the 501 branch would silently emit
    /// a bogus zero event.
    #[tokio::test]
    async fn provider_lacking_embed_returns_501_without_emit_issue_456() {
        use aisix_obs::UsageSink;
        use aisix_provider_anthropic::AnthropicBridge;

        const ANTHROPIC_PK_ID: &str = "22222222-2222-2222-2222-222222222222";

        let anthropic_pk_json = r#"{"display_name":"anthropic-up","secret":"sk-ant-test","provider":"anthropic","adapter":"anthropic"}"#;
        let anthropic_pk: aisix_core::ProviderKey =
            serde_json::from_str(anthropic_pk_json).unwrap();
        let anthropic_pk_entry = ResourceEntry::new(ANTHROPIC_PK_ID, anthropic_pk, 1);

        let anthropic_model_json = format!(
            r#"{{"display_name":"claude-embed","provider":"anthropic","model_name":"claude-3-haiku-20240307","provider_key_id":"{ANTHROPIC_PK_ID}"}}"#
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

        let body = serde_json::json!({"model": "claude-embed", "input": "hello"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_IMPLEMENTED,
            "Anthropic-backed /v1/embeddings must surface as 501 \
             (default Bridge::embed returns BridgeError::Config)",
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

    /// Issue #226 audit M1: a 200 response with upstream-reported
    /// `prompt_tokens = 0` MUST still emit a UsageEvent. Pre-fix the
    /// handler gated emission on `prompt_tokens > 0`, conflating
    /// "no upstream call" (501 NotImplemented) with "upstream returned
    /// zero tokens" (legitimate 200, rare-but-possible for empty input
    /// or provider-specific billing conventions). The fix introduces an
    /// explicit `upstream_called: bool` flag on EmbedDispatchSuccess
    /// so the channel is unambiguous; this test pins the post-fix
    /// contract by driving a 200 with zero tokens and asserting the
    /// event still arrives.
    #[tokio::test]
    async fn emits_usage_event_on_200_with_zero_prompt_tokens_audit_m1() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // 200 with usage.prompt_tokens=0. The contract: emit anyway —
        // attribution belongs to the api_key for compliance / audit
        // even when the billable count is zero.
        let upstream_body = serde_json::json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": [0.1_f32, 0.2_f32]
            }],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 0, "total_tokens": 0}
        });
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "my-embed", "input": ""});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect(
                "UsageEvent must be emitted even when upstream reports prompt_tokens=0 \
                 (audit M1 — emission keyed on upstream_called, not on token count)",
            )
            .expect("usage_sink sender dropped before emission");

        assert_eq!(event.prompt_tokens, 0);
        assert_eq!(event.status_code, 200);
        assert_eq!(event.api_key_id, "k-1");
        assert_eq!(event.model_id, "m-1");
        assert_eq!(event.inbound_protocol, "openai");
    }

    /// Issue #401 (follow-up to #324): omitting the required `model`
    /// field on /v1/embeddings must surface as `400 Bad Request` with
    /// the OpenAI-shape `invalid_request_error` envelope — NOT 422.
    /// Pre-fix axum's `Json<EmbeddingRequestBody>` extractor returned
    /// `JsonRejection::JsonDataError` → 422 on missing fields, which
    /// every SDK that branches on 400 vs 422 (most of them) sees as a
    /// silent semantic divergence from api.openai.com. Same wire
    /// contract chat.rs pins for #324; this test pins it for
    /// embeddings.
    #[tokio::test]
    async fn missing_model_field_on_embeddings_returns_400_not_422() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        // Valid JSON, valid `input` field, but `model` omitted.
        let body = serde_json::json!({"input": "hello"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "missing model field must surface as 400 per OpenAI wire contract — #401",
        );
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            v["error"]["type"], "invalid_request_error",
            "envelope must be OpenAI-shape invalid_request_error — #401",
        );
    }

    /// Companion case: missing `input` field also surfaces as 400.
    /// Same JsonRejection path as the missing-model case — pinning
    /// it independently so a regression that only special-cased one
    /// required field would surface here.
    #[tokio::test]
    async fn missing_input_field_on_embeddings_returns_400_not_422() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let body = serde_json::json!({"model": "my-embed"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "missing input field must surface as 400 per OpenAI wire contract — #401",
        );
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            v["error"]["type"], "invalid_request_error",
            "envelope must be OpenAI-shape invalid_request_error — #401",
        );
    }

    /// #655 parity: an upstream 5xx on /v1/embeddings now emits ONE zero-token
    /// UsageEvent so the failed request is visible in Logs (status + error
    /// class) and attributed to the api_key — instead of being dropped, as the
    /// non-chat handlers used to do. Mirrors
    /// `completions.rs::upstream_5xx_emits_zero_token_error_event`.
    #[tokio::test]
    async fn upstream_5xx_emits_zero_token_error_event() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let body = serde_json::json!({"model": "my-embed", "input": "hi"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("a failed /v1/embeddings must emit a zero-token UsageEvent")
            .expect("usage_sink sender dropped");
        assert_eq!(ev.status_code, 502, "upstream 5xx maps to 502");
        assert_eq!(ev.prompt_tokens, 0);
        assert_eq!(ev.api_key_id, "k-1");
        assert_eq!(ev.requested_model, "my-embed");
        assert!(
            !ev.error_class.is_empty(),
            "error_class must classify the failure"
        );
        assert!(
            rx.try_recv().is_err(),
            "exactly one event per failed request"
        );
    }

    /// Malformed JSON (syntax error) on /v1/embeddings must also
    /// surface as 400, not 422. Same JsonRejection → InvalidRequest
    /// path as the missing-field cases.
    #[tokio::test]
    async fn malformed_json_on_embeddings_returns_400() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("my-embed"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{not even valid json"#))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "malformed JSON must surface as 400, not 422 — #401",
        );
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            v["error"]["type"], "invalid_request_error",
            "envelope must be OpenAI-shape invalid_request_error — #401",
        );
    }

    /// #701: an upstream 5xx must mark the model's runtime status (cooldown)
    /// — /v1/embeddings previously never touched it.
    #[tokio::test]
    async fn upstream_5xx_marks_cooldown_issue_701() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("embed-model"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg()).without_cache();
        let app = crate::build_router(state.clone());

        let body = serde_json::json!({"model": "embed-model", "input": "hi"});
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
