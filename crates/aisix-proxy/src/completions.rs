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

use aisix_gateway::{
    BridgeContext, BridgeError, ChatMessage, ChatResponse, FinishReason, UsageStats,
};
use aisix_obs::{content_capture_cap, AccessLog, CapturedContent, RequestOutcome, UsageEvent};
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
    /// Resolved ProviderKey UUID — feeds per-PK telemetry attribution
    /// (AISIX-Cloud#867 parity).
    provider_key_id: String,
    /// Upstream-reported token counts. `None` on the 501
    /// NotImplemented path (provider doesn't support completions)
    /// or on a 200 with no `usage` block (rare edge). Handler
    /// gates UsageEvent emission on this being `Some`.
    usage: Option<CompletionUsage>,
    /// Per-detector PII mask counts (#932), input + output merged.
    /// Attached to the emitted UsageEvent. Empty = no redaction.
    redactions: crate::redact::RedactionCounts,
    /// Monitor-mode guardrail observations (AISIX-Cloud#562), input +
    /// output merged. Attached to the emitted UsageEvent.
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    /// True when the response leg was blocked by an OUTPUT guardrail
    /// AFTER the upstream billed for it (#911 [23]). The response body is
    /// the redacted 422, but `usage` still carries the billed counts so
    /// the UsageEvent (marked `guardrail_blocked`) keeps cp-api's budget
    /// ledger + /logs from under-reporting spend the provider charged for
    /// — the output analog of chat.rs's UpstreamCharge / responses.rs #543.
    guardrail_blocked: bool,
    /// Captured request/response content for content-capturing exporters
    /// (AISIX-Cloud#947). `Some` only when an enabled exporter opted into
    /// `content_mode = full`; threaded to `fan_out` via the handler's emit,
    /// never to the CP sink.
    captured_content: Option<CapturedContent>,
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
                    &model_name,
                    &api_key_id,
                    &success.provider_key_id,
                    status,
                    elapsed,
                    &usage,
                    &client,
                    success.guardrail_blocked,
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
                "completions",
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

/// Build a [`ChatFormat`](aisix_gateway::ChatFormat) of user messages from
/// the legacy completions `prompt` so the input guardrail chain can scan it
/// (#545). `prompt` is a string, an array of strings, or an array of token
/// ids / token-id arrays; only the string forms carry scannable text (token
/// ids are integers — skipped). Never sent upstream.
fn completions_input_to_chat(model: &str, body: &Value) -> aisix_gateway::ChatFormat {
    let messages = match body.get("prompt") {
        Some(Value::String(s)) if !s.is_empty() => {
            vec![aisix_gateway::ChatMessage::user(s.clone())]
        }
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|it| it.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| aisix_gateway::ChatMessage::user(s.to_string()))
            .collect(),
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
) -> Result<CompletionDispatchSuccess, ProxyError> {
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

    // #545: /v1/completions must run input guardrails. Before this it
    // forwarded the user `prompt` to the upstream with no configured
    // content/DLP check, so a block enforced on /v1/chat/completions was
    // bypassable by switching surface. Run the check BEFORE the rate-limit
    // reservation so a content-policy refusal doesn't burn an RPM slot
    // (matching /v1/chat/completions).
    let guardrail_ctx = aisix_guardrails::RequestContext {
        model_id: &model_entry.id,
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved_chain = state.guardrail_index.resolve(&guardrail_ctx);
    let mut input_seg_counts = crate::redact::RedactionCounts::new();
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    if !resolved_chain.is_empty() {
        let chat = completions_input_to_chat(model_name, &body);
        let (verdict, hits) =
            aisix_guardrails::Guardrail::check_input_non_segment_observed(&resolved_chain, &chat)
                .await;
        monitor_hits.extend(hits);
        // Segment pass: one Bedrock call over the prompt slots; an
        // ANONYMIZE disposition writes the masked text back into the body
        // (#932 bedrock follow-up).
        let verdict = crate::redact::moderate_body(
            &resolved_chain,
            crate::redact::Direction::Input,
            verdict,
            &mut input_seg_counts,
            &mut monitor_hits,
            |g| crate::redact::redact_completions_request(g, &mut body),
        )
        .await;
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
                "guardrail blocked /v1/completions request",
            );
            return Err(ProxyError::ContentFiltered(
                crate::error::guardrail_block_message("request", guardrail_name.as_deref()),
            ));
        }
    }

    // #932: mask-action PII rules rewrite the prompt in place AFTER the
    // block check passes, BEFORE the body is forwarded upstream.
    let mut redactions = crate::redact::redact_completions_request(&resolved_chain, &mut body);
    crate::redact::merge_counts(&mut redactions, input_seg_counts);

    // Content capture (AISIX-Cloud#947): the client-facing request body
    // (post-redaction, so masked PII stays masked in the exported content),
    // gated on an exporter actually wanting content.
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
    let provider = crate::dispatch::require_provider(model)?;
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

    match bridge.complete(&body, &ctx).await {
        Ok(resp_json) => {
            // #701: clear any cooldown/unhealthy mark now the upstream
            // answered — same recovery signal as rerank/audio/chat.
            state.health.record_success(model_name);
            state.runtime_status.mark_healthy(&model_entry.id);
            // Extract usage BEFORE moving resp_json into the Response
            // so the success struct carries typed counters rather
            // than re-parsing JSON downstream.
            let usage = extract_completion_usage(&resp_json);
            // #911 [21]: commit the actual token cost so TPM/TPD is enforced
            // for /v1/completions the same way chat + embeddings enforce it.
            // Pre-fix the reservation dropped uncommitted, so the token
            // counter never moved and a caller could bypass token limits by
            // routing traffic through this endpoint.
            let total_tokens = usage
                .as_ref()
                .map(|u| u64::from(u.prompt_tokens) + u64::from(u.completion_tokens))
                .unwrap_or(0);
            reservation.commit_tokens(total_tokens).await;

            // #911 [23]: /v1/completions must run OUTPUT guardrails too. The
            // input hook above scans the prompt, but pre-fix the model's reply
            // was returned unscanned — a content/DLP block enforced on
            // /v1/chat/completions was bypassable by switching to this surface
            // for the response leg. Mirror chat's output check: buffer the reply
            // text into a synthetic ChatResponse and run the chain. The upstream
            // already billed (tokens committed above), so a block surfaces a
            // redacted 422 rather than the response.
            let mut resp_json = resp_json;
            if !resolved_chain.is_empty() {
                let synth = ChatResponse {
                    id: String::new(),
                    model: model_name.to_string(),
                    message: ChatMessage::assistant(completion_output_text(&resp_json)),
                    finish_reason: FinishReason::Stop,
                    usage: UsageStats::default(),
                };
                let (verdict, hits) =
                    aisix_guardrails::Guardrail::check_output_non_segment_observed(
                        &resolved_chain,
                        &synth,
                    )
                    .await;
                monitor_hits.extend(hits);
                let verdict = crate::redact::moderate_body(
                    &resolved_chain,
                    crate::redact::Direction::Output,
                    verdict,
                    &mut redactions,
                    &mut monitor_hits,
                    |g| crate::redact::redact_completions_response(g, &mut resp_json),
                )
                .await;
                if let aisix_guardrails::GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } = verdict
                {
                    // Per #153 the matched-pattern detail stays in ops logs only.
                    tracing::warn!(
                        guardrail_hook = "output",
                        model = %model_name,
                        reason = %reason,
                        "guardrail blocked /v1/completions response",
                    );
                    // The upstream already billed for this response (tokens
                    // committed above), so return the redacted 422 body BUT
                    // carry the billed `usage` marked `guardrail_blocked` —
                    // recording zero tokens here would let cp-api's ledger
                    // under-report spend the customer was charged for. Same
                    // output analog as responses.rs #543 / chat.rs UpstreamCharge.
                    return Ok(CompletionDispatchSuccess {
                        response: ProxyError::ContentFiltered(
                            crate::error::guardrail_block_message(
                                "response",
                                guardrail_name.as_deref(),
                            ),
                        )
                        .into_response(),
                        provider: provider_label,
                        model_id: model_entry.id.to_string(),
                        provider_key_id: pk_entry.id.to_string(),
                        usage,
                        redactions,
                        monitor_hits,
                        guardrail_blocked: true,
                        // Blocked responses never reached the client — no
                        // content capture, matching the chat surface.
                        captured_content: None,
                    });
                }
            }

            // #932: mask-action PII rules rewrite the reply text AFTER the
            // block check passes.
            crate::redact::merge_counts(
                &mut redactions,
                crate::redact::redact_completions_response(&resolved_chain, &mut resp_json),
            );

            // Content capture (AISIX-Cloud#947): the completion text from the
            // POST-redaction body, so the exported content matches what the
            // caller received.
            let captured_content = match (&captured_prompt, content_cap) {
                (Some(prompt), Some(cap)) => Some(CapturedContent::new(
                    prompt,
                    &completion_output_text(&resp_json),
                    cap as usize,
                )),
                _ => None,
            };

            Ok(CompletionDispatchSuccess {
                response: Json(resp_json).into_response(),
                provider: provider_label,
                model_id: model_entry.id.to_string(),
                provider_key_id: pk_entry.id.to_string(),
                usage,
                redactions,
                monitor_hits,
                guardrail_blocked: false,
                captured_content,
            })
        }
        Err(BridgeError::Config(msg)) if msg.contains("does not support text completions") => {
            // No upstream call → no tokens to count; release the reservation.
            reservation.commit_tokens(0).await;
            let env = ErrorEnvelope::new(msg, "not_implemented");
            Ok(CompletionDispatchSuccess {
                response: (StatusCode::NOT_IMPLEMENTED, Json(env)).into_response(),
                provider: provider_label,
                model_id: model_entry.id.to_string(),
                provider_key_id: pk_entry.id.to_string(),
                // No upstream call → no usage to attribute. Handler
                // gates emission on `usage.is_some()` so 501 stays
                // out of /logs noise (same convention as #402).
                usage: None,
                redactions,
                monitor_hits,
                guardrail_blocked: false,
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
/// A missing completion side coerces to 0 and the event is still
/// logged/billed (the usage block's `completion_tokens` defaults to 0
/// when absent) — see
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

/// Concatenate the `text` of every choice in a /v1/completions response for
/// output-guardrail scanning (#911 [23]). Missing/non-string `text` fields are
/// skipped; the result is the client-visible completion text the content/DLP
/// output hook must inspect.
fn completion_output_text(body: &Value) -> String {
    body.get("choices")
        .and_then(|c| c.as_array())
        .map(|choices| {
            choices
                .iter()
                .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// Issue #403: push one `UsageEvent` onto cp-api's telemetry sink
/// and fan it out to per-env OTLP exporters. Mirrors the shape of
/// `embeddings::emit_usage_event` (#402) and `responses::emit_usage_event`
/// (#404); the legacy /v1/completions endpoint has both prompt and
/// completion sides but no streaming / reasoning tokens.
///
/// `inbound_protocol = "openai"` per chat.rs convention. The per-PK
/// attribution tags (`provider_kind` / `provider_featured` /
/// `branded_provider` / `pk_label` / `byo_label`) ARE populated — same
/// lookup as chat / messages / responses / embeddings (AISIX-Cloud#867
/// parity) via `usage_attr::apply_pk_telemetry` below.
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    requested_model: &str,
    api_key_id: &str,
    provider_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    usage: &CompletionUsage,
    client: &ClientContext,
    guardrail_blocked: bool,
    // Per-detector PII mask counts (#932). Empty = no redaction.
    redacted_entity_counts: crate::redact::RedactionCounts,
    // Monitor-mode guardrail observations (AISIX-Cloud#562).
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    // Captured request/response content for content-capturing exporters
    // (AISIX-Cloud#947). Forwarded only to `fan_out`, never to the CP sink.
    content: Option<&CapturedContent>,
) {
    let snap = state.snapshot.load();
    let mut event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "openai".to_string(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        // #911 [23]: a billed-then-output-blocked completion surfaces on the
        // dashboard's Blocked tab while still carrying its billed token counts.
        guardrail_blocked,
        redacted_entity_counts,
        guardrail_monitor_hits,
        ..Default::default()
    };
    crate::usage_attr::apply_pk_telemetry(&mut event, &snap, provider_key_id);
    state.usage_sink.try_emit("completions", event.clone());
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

    /// Same PK as `provider_key_entry` (reuses `PK_ID` so existing model
    /// fixtures resolve to it) but carries `telemetry_tags` so the emitted
    /// UsageEvent picks up the per-PK attribution fields (AISIX-Cloud#867).
    fn provider_key_entry_tagged(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-up","api_base":"{api_base}","provider":"openai","adapter":"openai","telemetry_tags":{{"kind":"catalog","featured":true,"branded_provider":"openai","pk_label":"prod-completions-key"}}}}"#
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

    fn keyword_input_guardrail(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"t","enabled":true,"hook_point":"input","fail_open":false,"kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-1", g, 1)
    }

    /// #545: a configured input guardrail must fire on /v1/completions — a
    /// blocked `prompt` returns 422 content_filter and the upstream is never
    /// contacted (`expect(0)`).
    #[tokio::test]
    async fn input_guardrail_blocks_prompt_returns_422() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"object":"text_completion"})),
            )
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "instruct", "prompt": "please BLOCKME now"});
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

    /// #545 companion: a benign prompt with a guardrail configured still
    /// forwards (`expect(1)`) and returns 200.
    #[tokio::test]
    async fn input_guardrail_allows_benign_prompt_forwards_200() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "text_completion",
                "choices": [{"text": "ok", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let body = serde_json::json!({"model": "instruct", "prompt": "a fine prompt"});
        let resp = tower::ServiceExt::oneshot(app, make_req(body))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "text_completion");
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

    /// #429 follow-up: a 200 whose `usage` carries
    /// `prompt_tokens` but omits `completion_tokens` is still a real
    /// billable call — the prompt was processed. It MUST emit a
    /// UsageEvent with `completion_tokens = 0` (coercing the missing
    /// side to 0), NOT be dropped. Only a fully
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

    /// Per #655 parity (was #403 negative pinning): an upstream 5xx now emits
    /// ONE zero-token UsageEvent so the failed request is visible in Logs
    /// (status + error class) and attributed to the api_key — instead of being
    /// dropped, as the non-chat handlers used to do. The 501 NotImplemented
    /// path still emits nothing (no upstream call); see the test below.
    #[tokio::test]
    async fn upstream_5xx_emits_zero_token_error_event() {
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

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("a failed /v1/completions must emit a zero-token UsageEvent")
            .expect("usage_sink sender dropped");
        assert_eq!(ev.status_code, 502, "upstream 5xx maps to 502");
        assert_eq!(ev.prompt_tokens, 0);
        assert_eq!(ev.completion_tokens, 0);
        assert_eq!(ev.api_key_id, "k-1");
        assert_eq!(ev.requested_model, "instruct");
        assert!(
            !ev.error_class.is_empty(),
            "error_class must classify the failure"
        );
        assert!(
            rx.try_recv().is_err(),
            "exactly one event per failed request"
        );
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

    /// AISIX-Cloud#867 parity: a successful /v1/completions 200 must stamp
    /// the five per-PK telemetry attribution fields (provider_kind /
    /// provider_featured / branded_provider / pk_label / byo_label) onto the
    /// emitted UsageEvent, sourced from the resolved ProviderKey's
    /// `telemetry_tags` — exactly like `/v1/responses` and `/v1/embeddings`.
    /// Pre-fix the completions emitter left these at Default (wire NULL).
    #[tokio::test]
    async fn emits_provider_telemetry_tags_issue_867() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "id": "cmpl-up-1",
            "object": "text_completion",
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"text": "hi", "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        });
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry_tagged(&upstream.uri()));
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

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/completions 200")
            .expect("usage_sink sender dropped");
        assert_eq!(ev.provider_kind, "catalog");
        assert!(ev.provider_featured);
        assert_eq!(ev.branded_provider, "openai");
        assert_eq!(ev.pk_label, "prod-completions-key");
    }

    /// #701: an upstream 5xx must mark the model's runtime status (cooldown)
    /// so a flapping upstream reached only via /v1/completions trips the
    /// circuit breaker like rerank/audio/chat. Pre-#701 the status stayed
    /// Healthy.
    #[tokio::test]
    async fn upstream_5xx_marks_cooldown_issue_701() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("instruct"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg()).without_cache();
        let app = crate::build_router(state.clone());

        let body = serde_json::json!({"model": "instruct", "prompt": "x"});
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
