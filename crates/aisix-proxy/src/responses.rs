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

use aisix_gateway::{ChatFormat, ChatMessage, ChatResponse, FinishReason, UsageStats};
use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde_json::Value;
use std::time::{Duration, Instant};

use crate::attempt::{
    attempt_error_from_proxy, ms_since, AttemptInfo, AttemptRecord, RoutingTelemetry,
};
use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
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
    /// Per-attempt routing telemetry (#655): the failed attempts that
    /// preceded the winner plus the winning attempt itself.
    routing: RoutingTelemetry,
    /// #543: set when an OUTPUT guardrail blocked this response. The
    /// upstream already billed, so this is returned as a "success" carrying
    /// the billed `usage` + a 422 body, and the emitted UsageEvent is marked
    /// `guardrail_blocked` so the dashboard's Blocked tab + budget ledger see
    /// it (silently zeroing the tokens would underreport spend the operator
    /// paid the provider for).
    guardrail_blocked: bool,
}

/// Dispatch error carrying the per-attempt telemetry accumulated before
/// the request ultimately failed (#655). Mirrors `chat::DispatchFailure`.
struct ResponsesDispatchError {
    err: ProxyError,
    routing: RoutingTelemetry,
}

impl From<ProxyError> for ResponsesDispatchError {
    /// Pre-attempt `?` failures (model-not-found, auth, budget) carry no
    /// recorded attempts.
    fn from(err: ProxyError) -> Self {
        Self {
            err,
            routing: RoutingTelemetry::default(),
        }
    }
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
    client: ClientContext,
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
                &success.routing,
            );
            state.metrics.record_request(
                &success.provider,
                &model_name,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            // Per #655: one zero-token UsageEvent per failed attempt that
            // preceded the winner (non-streaming failover).
            emit_failed_attempts(
                &state,
                &request_id,
                &success.model_id,
                &api_key_id,
                &client,
                &success.routing,
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
                // Winning-attempt classification (#655). Direct models
                // have no recorded attempt → AttemptInfo defaults.
                let attempt = success
                    .routing
                    .winner()
                    .map(AttemptInfo::from_record)
                    .unwrap_or_default();
                emit_usage_event(
                    &state,
                    &request_id,
                    &success.model_id,
                    &api_key_id,
                    status,
                    elapsed,
                    &usage,
                    &client,
                    attempt,
                    success.guardrail_blocked,
                );
            }
            success.response
        }
        Err(ResponsesDispatchError { err, routing }) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                &model_name,
                "unknown",
                &api_key_id,
                status,
                elapsed,
                &request_id,
                &routing,
            );
            state.metrics.record_request(
                "unknown",
                &model_name,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            // Per #655: emit one zero-token UsageEvent per FAILED attempt so
            // the dashboard's Logs tab surfaces each failed upstream try.
            // `model_id` is empty on pre-dispatch failures (model never
            // resolved); the request_id still groups the rows.
            emit_failed_attempts(&state, &request_id, "", &api_key_id, &client, &routing);
            // Pre-dispatch failure (model-not-found, auth, budget) records no
            // attempts — emit a single terminal event carrying the failure
            // class. When attempts were recorded, each was already emitted.
            if routing.attempts.is_empty() {
                emit_zero_token_event(
                    &state,
                    &request_id,
                    "",
                    &api_key_id,
                    status,
                    elapsed,
                    &client,
                    AttemptInfo {
                        kind: "initial".to_string(),
                        error_class: err.kind().to_string(),
                        ..Default::default()
                    },
                );
            }
            err.into_response()
        }
    }
}

async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    body: &Value,
    request_id: &str,
) -> Result<ResponseDispatchSuccess, ResponsesDispatchError> {
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
        return Err(ProxyError::ModelForbidden(model_name.clone()).into());
    }

    // #719: /v1/responses must run input guardrails like /v1/chat/completions
    // and /v1/messages. Before this, user input reached the upstream without
    // any configured content/DLP check, so a content block enforced on the
    // chat surface was bypassable simply by calling /v1/responses (the same
    // violent input that 422s on chat returned 200 with the content echoed
    // here). Translate the Responses-API body into the internal ChatFormat
    // and run the resolved input guardrail chain; a Block short-circuits
    // before dispatch. (Input Rewrite/Bypass is not applied to the outgoing
    // Responses body — only Block is enforced, matching /v1/messages.)
    //
    // #542: run this BEFORE the rate-limit reservation so a content-policy
    // block doesn't burn an RPM slot (matching /v1/chat/completions).
    let guardrail_ctx = aisix_guardrails::RequestContext {
        model_id: &model_entry.id,
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved_chain = state.guardrail_index.resolve(&guardrail_ctx);
    if !resolved_chain.is_empty() {
        let chat = responses_input_to_chat(&model_name, body);
        if let aisix_guardrails::GuardrailVerdict::Block { reason } =
            aisix_guardrails::Guardrail::check_input(&resolved_chain, &chat).await
        {
            // Per #153 the matched-pattern detail stays in ops logs only; the
            // wire envelope stays generic so callers can't enumerate the
            // blocklist by probing error responses.
            tracing::warn!(
                guardrail_hook = "input",
                model = %model_name,
                reason = %reason,
                "guardrail blocked /v1/responses request",
            );
            return Err(
                ProxyError::ContentFiltered("request blocked by content policy".into()).into(),
            );
        }
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
    let is_routing_request = model_entry.value.routing.is_some();
    let mut routing = RoutingTelemetry::default();

    let not_openai = || {
        ProxyError::InvalidRequest(format!(
            "model `{model_name}` is not an OpenAI provider; /v1/responses requires OpenAI"
        ))
    };

    // Walk the OpenAI targets, failing over on a retryable failure.
    // Streaming and non-streaming share this loop: `responses_to_target`
    // branches internally and, for streaming, only returns Ok once the
    // first chunk has arrived under `stream_timeout` (#554) — so the 200 is
    // committed to exactly one target and a slow first chunk fails over.
    let mut last_err: Option<ProxyError> = None;
    let mut any_openai = false;
    for target in &attempt_models {
        if target.model.provider.as_deref() != Some("openai") {
            continue;
        }
        any_openai = true;
        let (idx, kind) = routing.begin_attempt(&target.model.display_name);
        let target_model = if is_routing_request {
            target.model.display_name.clone()
        } else {
            String::new()
        };
        let attempt_started = Instant::now();
        match responses_to_target(
            state,
            &snapshot,
            body,
            &target.model,
            &target.id,
            request_id,
            &resolved_chain,
        )
        .await
        {
            Ok(mut success) => {
                routing.attempts.push(AttemptRecord {
                    index: idx,
                    kind,
                    target_model,
                    provider_key_id: String::new(),
                    status: success.response.status().as_u16(),
                    success: true,
                    error_class: String::new(),
                    error_message: String::new(),
                    latency_ms: ms_since(attempt_started),
                });
                success.routing = routing;
                return Ok(success);
            }
            Err(e) => {
                let retryable = matches!(
                    &e,
                    ProxyError::Bridge(be) if crate::routing::is_retryable(be, retry_on_429)
                );
                let (error_class, error_message) = attempt_error_from_proxy(&e);
                routing.attempts.push(AttemptRecord {
                    index: idx,
                    kind,
                    target_model,
                    provider_key_id: String::new(),
                    status: e.status().as_u16(),
                    success: false,
                    error_class,
                    error_message,
                    latency_ms: ms_since(attempt_started),
                });
                last_err = Some(e);
                if !retryable {
                    break;
                }
            }
        }
    }

    if !any_openai {
        return Err(not_openai().into());
    }
    Err(ResponsesDispatchError {
        err: last_err.unwrap_or(ProxyError::ProviderUnavailable),
        routing,
    })
}

/// Translate a `/v1/responses` request body into the internal
/// [`ChatFormat`] so the input guardrail chain can scan the
/// user-supplied content (#719). Only scannable text matters here — this
/// is **not** a faithful Responses→Chat transform and is never sent
/// upstream; the original `body` is forwarded verbatim.
///
/// The Responses-API `input` field is either a bare string or an array of
/// input items; a message item is `{role, content}` whose `content` is a
/// string or an array of typed parts (`input_text` / `output_text` /
/// `text`). The optional top-level `instructions` maps to a system
/// message. Roles are preserved so the guardrail's user-vs-all message
/// selection behaves the same as on /v1/chat/completions.
/// <https://platform.openai.com/docs/api-reference/responses/create>
fn responses_input_to_chat(model: &str, body: &Value) -> ChatFormat {
    let mut messages = Vec::new();

    if let Some(instructions) = body.get("instructions").and_then(|v| v.as_str()) {
        if !instructions.is_empty() {
            messages.push(ChatMessage::system(instructions.to_string()));
        }
    }

    match body.get("input") {
        Some(Value::String(text)) => {
            if !text.is_empty() {
                messages.push(ChatMessage::user(text.clone()));
            }
        }
        Some(Value::Array(items)) => {
            for item in items {
                // A bare-string array element is treated as user text; an
                // object element is a message whose role we preserve.
                if let Some(text) = item.as_str() {
                    if !text.is_empty() {
                        messages.push(ChatMessage::user(text.to_string()));
                    }
                    continue;
                }
                let text = responses_item_text(item);
                if text.is_empty() {
                    continue;
                }
                let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                messages.push(match role {
                    "assistant" => ChatMessage::assistant(text),
                    "system" | "developer" => ChatMessage::system(text),
                    _ => ChatMessage::user(text),
                });
            }
        }
        _ => {}
    }

    ChatFormat::new(model, messages)
}

/// Collect the plain, caller-supplied text of one Responses-API input
/// item, across every key on the `input`-item union that carries text the
/// model will see:
/// - `content` — message items;
/// - `output` — tool-result items (`function_call_output`,
///   `custom_tool_call_output`, `*_call_output`) the caller feeds back;
/// - `reason` — an `mcp_approval_response` justification.
///
/// All are user-controlled content entering the model — the
/// `/v1/chat/completions` equivalent (a `role:"tool"` message) is
/// scanned, so leaving any of these unscanned would let the #719
/// surface-switch bypass survive on that channel. Each slot is a string
/// or an array of typed parts; we gather the `text` of each part and
/// ignore non-text parts (images, files). These items carry no `role`,
/// so the caller maps them to a user message (scanned by every guardrail
/// kind). Reading a key absent on other item types is a harmless no-op.
/// <https://platform.openai.com/docs/api-reference/responses/create>
fn responses_item_text(item: &Value) -> String {
    [item.get("content"), item.get("output"), item.get("reason")]
        .into_iter()
        .flatten()
        .map(responses_value_text)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Plain text of one Responses-API content slot: a bare string, or the
/// concatenated `text` of an array of typed parts.
fn responses_value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
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
    chain: &aisix_guardrails::GuardrailChain,
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
    let mut req = client
        .post(&url)
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .header("x-aisix-request-id", request_id)
        .json(&body);
    // #554: non-streaming gets the E2E request timeout via reqwest's
    // request-level timeout. Streaming must NOT use it (it would cap the
    // whole stream); the streaming branch below enforces the per-chunk
    // read timeout instead.
    if !is_stream {
        if let Some(d) = model.request_timeout() {
            req = req.timeout(d);
        }
    }
    let send_started = Instant::now();
    // Streaming bounds the connect by the stream deadline (reqwest's
    // request-level timeout can't be used — it would cap the whole stream);
    // non-streaming relies on the request-level timeout set above.
    let connect_deadline = if is_stream {
        model.stream_timeout_effective()
    } else {
        None
    };
    let upstream_resp =
        crate::stream_timeout::send_with_deadline(req, connect_deadline, send_started)
            .await
            .map_err(|be| {
                crate::cooldown::note_failure(
                    &state.runtime_status,
                    model_id,
                    model.cooldown.as_ref(),
                    be,
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

        // #719: when an output-hook guardrail is attached, the streaming
        // response can't be forwarded token-by-token — a blocked phrase
        // would already be on the wire before it scans clean, the same
        // surface-switch bypass via `stream:true`. Mirror the chat
        // surface's secure default (BufferFull): hold the whole SSE
        // response, scan the assistant output text, then release the bytes
        // verbatim or block with 422. Requests with no output-hook
        // guardrail keep the zero-copy verbatim passthrough below.
        if aisix_guardrails::Guardrail::runs_on_output(chain) {
            // Hold the whole SSE response back to scan it, but cap the
            // buffer so a huge (or malicious) upstream response can't OOM the
            // gateway. Mirror the chat surface's secure BufferFull default
            // (#466): read with a running byte count and fail closed if the
            // response exceeds the cap — an output-hook guardrail must never
            // release content it couldn't fully buffer to scan. The cap is
            // taken from the chain's resolved streaming policy.
            let max_buffer_bytes = match aisix_guardrails::Guardrail::stream_output_policy(chain) {
                aisix_guardrails::StreamOutputPolicy::BufferFull {
                    max_buffer_bytes, ..
                } => max_buffer_bytes,
                _ => aisix_guardrails::DEFAULT_STREAM_OUTPUT_BUFFER_BYTES,
            };
            let stream = upstream_resp.bytes_stream();
            futures::pin_mut!(stream);
            // Effective streaming budget: `stream_timeout`, falling back to
            // `timeout` — applied to every buffered read, consistent with the
            // verbatim branch and the connect deadline.
            let read_to = model.stream_timeout_effective();
            let mut buf: Vec<u8> = Vec::new();
            let mut saw_chunk = false;
            loop {
                // #554: bound each read so a stalled upstream fails over —
                // the buffer path hasn't sent anything to the client yet, so
                // a read timeout is a retryable failure, not a truncation.
                let next = match read_to {
                    Some(d) => tokio::time::timeout(d, stream.next())
                        .await
                        .map_err(|_| {
                            crate::cooldown::note_failure(
                                &state.runtime_status,
                                model_id,
                                model.cooldown.as_ref(),
                                aisix_gateway::BridgeError::Timeout {
                                    elapsed_ms: d.as_millis() as u64,
                                },
                            )
                        })
                        .map_err(ProxyError::Bridge)?,
                    None => stream.next().await,
                };
                let Some(chunk) = next else {
                    // #554: an upstream that returns 200 then closes with zero
                    // bytes is a first-chunk failure — fail over rather than
                    // serving an empty 200, matching the verbatim branch. Only
                    // when a stream timeout is configured, so a model without
                    // one keeps the pre-#554 behavior.
                    if !saw_chunk && read_to.is_some() {
                        let err = crate::cooldown::note_failure(
                            &state.runtime_status,
                            model_id,
                            model.cooldown.as_ref(),
                            aisix_gateway::BridgeError::StreamAborted,
                        );
                        return Err(ProxyError::Bridge(err));
                    }
                    break;
                };
                saw_chunk = true;
                let chunk = chunk
                    .map_err(|e| {
                        crate::cooldown::note_failure(
                            &state.runtime_status,
                            model_id,
                            model.cooldown.as_ref(),
                            aisix_gateway::BridgeError::UpstreamDecode(e.to_string()),
                        )
                    })
                    .map_err(ProxyError::Bridge)?;
                if buf.len() + chunk.len() > max_buffer_bytes {
                    // Unlike chat's BufferFull, we always fail closed on
                    // overflow regardless of `on_exceeded_fail_open`: an
                    // output-hook guardrail must not release a response it
                    // couldn't fully buffer to scan. No shipped guardrail
                    // configures `BufferFull { on_exceeded_fail_open: true }`
                    // on this surface today.
                    tracing::warn!(
                        guardrail_hook = "output",
                        model = %model.display_name,
                        max_buffer_bytes,
                        "streaming /v1/responses output exceeded buffer cap; failing closed",
                    );
                    return Err(ProxyError::ContentFiltered(
                        "response blocked by content policy".into(),
                    ));
                }
                buf.extend_from_slice(&chunk);
            }
            let out_text = responses_sse_output_text(&buf);
            let synth = synth_chat_response(&upstream_model, out_text);
            if let aisix_guardrails::GuardrailVerdict::Block { reason } =
                aisix_guardrails::Guardrail::check_output(chain, &synth).await
            {
                // Per #153 the matched-pattern detail stays in ops logs only.
                tracing::warn!(
                    guardrail_hook = "output",
                    model = %model.display_name,
                    reason = %reason,
                    "guardrail blocked streaming /v1/responses response",
                );
                return Err(ProxyError::ContentFiltered(
                    "response blocked by content policy".into(),
                ));
            }
            let mut response = axum::response::Response::new(axum::body::Body::from(buf));
            apply_passthrough_headers(&mut response, &headers, request_id);
            return Ok(ResponseDispatchSuccess {
                response,
                provider: provider_label,
                usage: None,
                model_id: model_id.to_string(),
                routing: RoutingTelemetry::default(),
                guardrail_blocked: false,
            });
        }

        // #554: enforce the per-chunk read timeout on the forwarded bytes.
        // When a `stream_timeout` is configured, peek the first byte so a
        // slow/erroring first token fails over before the 200 is committed;
        // without one, forward directly (pre-#554 behavior). A mid-stream
        // stall truncates the forwarded stream (no in-band error frame for
        // an opaque byte passthrough).
        let stream_budget = model.stream_timeout_effective();
        let wrapped: std::pin::Pin<
            Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>,
        > = Box::pin(crate::stream_timeout::with_read_timeout_bytes(
            upstream_resp.bytes_stream(),
            stream_budget,
        ));
        let body_stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>,
        > = if stream_budget.is_some() {
            let mut wrapped = wrapped;
            let first_bytes = match wrapped.next().await {
                Some(Ok(b)) => b,
                Some(Err(e)) => {
                    let err = crate::dispatch::reqwest_error_to_bridge(&e, send_started);
                    if let Some((ttl, reason)) =
                        crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
                    {
                        state.runtime_status.mark_cooldown(model_id, ttl, reason);
                    }
                    return Err(ProxyError::Bridge(err));
                }
                None => {
                    let err = aisix_gateway::BridgeError::StreamAborted;
                    if let Some((ttl, reason)) =
                        crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
                    {
                        state.runtime_status.mark_cooldown(model_id, ttl, reason);
                    }
                    return Err(ProxyError::Bridge(err));
                }
            };
            Box::pin(
                futures::stream::once(std::future::ready(Ok::<bytes::Bytes, reqwest::Error>(
                    first_bytes,
                )))
                .chain(wrapped),
            )
        } else {
            wrapped
        };
        let mut response =
            axum::response::Response::new(axum::body::Body::from_stream(body_stream));
        apply_passthrough_headers(&mut response, &headers, request_id);

        // Verbatim passthrough (no output guardrail). The gateway doesn't
        // parse SSE chunks here, so usage stays None — streaming usage
        // emission is tracked as a #404 follow-up (see PR body).
        Ok(ResponseDispatchSuccess {
            response,
            provider: provider_label,
            usage: None,
            model_id: model_id.to_string(),
            routing: RoutingTelemetry::default(),
            guardrail_blocked: false,
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

        // #719: run the output guardrail chain on the assistant's text so a
        // configured output block isn't bypassable by calling /v1/responses
        // (the input half is enforced in `dispatch`). Only when an
        // output-hook guardrail is attached; otherwise this is a no-op.
        if aisix_guardrails::Guardrail::runs_on_output(chain) {
            let synth = synth_chat_response(&upstream_model, responses_output_text(&json_body));
            if let aisix_guardrails::GuardrailVerdict::Block { reason } =
                aisix_guardrails::Guardrail::check_output(chain, &synth).await
            {
                // Per #153 the matched-pattern detail stays in ops logs only.
                tracing::warn!(
                    guardrail_hook = "output",
                    model = %model.display_name,
                    reason = %reason,
                    "guardrail blocked /v1/responses response",
                );
                // #543: the provider already billed for this response, so
                // return a 422 body BUT carry the billed `usage` (marked
                // guardrail_blocked) — recording zero tokens would let the
                // customer's ledger underreport spend they were charged for.
                // This is the output analog of chat.rs's UpstreamCharge.
                return Ok(ResponseDispatchSuccess {
                    response: ProxyError::ContentFiltered(
                        "response blocked by content policy".into(),
                    )
                    .into_response(),
                    provider: provider_label,
                    usage,
                    model_id: model_id.to_string(),
                    routing: RoutingTelemetry::default(),
                    guardrail_blocked: true,
                });
            }
        }

        Ok(ResponseDispatchSuccess {
            response: Json(json_body).into_response(),
            provider: provider_label,
            usage,
            model_id: model_id.to_string(),
            routing: RoutingTelemetry::default(),
            guardrail_blocked: false,
        })
    }
}

/// Pull the usage counters out of a Responses-API non-streaming
/// response body. Returns `None` only when:
///   - The `usage` block is missing entirely, OR
///   - `usage.input_tokens` is missing / non-numeric
///
/// Those cases skip UsageEvent emission rather than attributing a
/// zero-everything noise row to the api_key. The `input_tokens` gate
/// distinguishes "no upstream usage at all" from a legitimate reply.
///
/// `output_tokens`, by contrast, defaults to 0 when absent: a 200 that
/// reports an input side but omits the output side is still a real
/// billable call and must be recorded. A missing completion/output side
/// coerces to 0 and the event is still logged/billed
/// (#429 follow-up; mirrors the tolerant wire-layer decode of
/// #474). Spec:
/// <https://platform.openai.com/docs/api-reference/responses/object>
fn extract_response_usage(body: &Value) -> Option<ResponseUsage> {
    let usage = body.get("usage")?;
    let prompt_tokens = usage.get("input_tokens").and_then(|v| v.as_u64())? as u32;
    let completion_tokens = usage
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
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

/// Collect the assistant's output text from a Responses-API response object
/// for output-guardrail scanning (#719/#546):
/// - the `text` of every `output_text` content part of message items, and
/// - the `name` + `arguments` (function calls) / `input` (custom tool calls)
///   of tool-call items — these are **top-level** item fields, not under
///   `content[]`, so without scanning them a blocked literal placed in a
///   tool-call's arguments would bypass the output guardrail. The chat
///   surface scans tool-call output too (`ChatResponse::guardrail_output_text`,
///   the #448 fix); this keeps the surfaces symmetric.
///
/// Reasoning items are intentionally excluded (out of output-guardrail
/// scope, matching the chat surface) — they carry `summary`, not `content`
/// / `arguments`, so they're naturally skipped.
/// <https://platform.openai.com/docs/api-reference/responses/object>
fn responses_output_text(resp: &Value) -> String {
    let Some(items) = resp.get("output").and_then(|v| v.as_array()) else {
        return String::new();
    };
    let mut parts: Vec<&str> = Vec::new();
    for it in items {
        if let Some(content) = it.get("content").and_then(|c| c.as_array()) {
            parts.extend(
                content
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str())),
            );
        }
        // Tool-call items carry caller-visible model output under top-level
        // `name`/`arguments` (function_call) or `name`/`input` (custom tool).
        for key in ["name", "arguments", "input"] {
            if let Some(s) = it.get(key).and_then(|v| v.as_str()) {
                parts.push(s);
            }
        }
    }
    parts
        .into_iter()
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Collect the assistant's streamed output text from a buffered
/// Responses-API SSE response (#719/#546). Prefers the authoritative full
/// output carried on a terminal `response` event — `response.completed`,
/// **and also `response.incomplete` / `response.failed`**, which carry the
/// same full `output[]` (incl. tool-call items) and fire routinely (e.g.
/// `max_output_tokens` truncation). Falls back to concatenating the streamed
/// deltas when no terminal `response` object is present (truncated/aborted):
/// both `response.output_text.delta` (assistant text) and
/// `response.function_call_arguments.delta` (tool-call args stream via their
/// own event, NOT output_text) — otherwise blocked tool-call args would leak
/// on a stream that never reaches a terminal object. The `type` field on each
/// `data:` JSON line drives the dispatch.
/// <https://platform.openai.com/docs/api-reference/responses-streaming>
fn responses_sse_output_text(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut deltas = String::new();
    for line in text.lines() {
        let data = match line.strip_prefix("data:") {
            Some(d) => d.trim(),
            None => continue,
        };
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let json: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match json.get("type").and_then(|t| t.as_str()) {
            Some("response.completed" | "response.incomplete" | "response.failed") => {
                if let Some(resp) = json.get("response") {
                    let full = responses_output_text(resp);
                    if !full.is_empty() {
                        return full;
                    }
                }
            }
            Some("response.output_text.delta") => {
                if let Some(d) = json.get("delta").and_then(|d| d.as_str()) {
                    deltas.push_str(d);
                }
            }
            // Tool-call argument deltas across all tool kinds — function calls,
            // MCP tool calls, and custom tools each stream their args/input via
            // their own event, not output_text.delta. On a terminal `response`
            // object these are already covered by responses_output_text; this
            // matters only when the stream aborts before any terminal object.
            // Concatenate WITHOUT a separator — these are pieces of one call's
            // string; a separator would split a literal that streamed across
            // two deltas (e.g. "BLOCK"+"ME") and miss the match.
            Some(
                "response.function_call_arguments.delta"
                | "response.mcp_call_arguments.delta"
                | "response.custom_tool_call_input.delta",
            ) => {
                if let Some(d) = json.get("delta").and_then(|d| d.as_str()) {
                    deltas.push_str(d);
                }
            }
            _ => {}
        }
    }
    deltas
}

/// Build the minimal internal `ChatResponse` an output guardrail needs to
/// scan: the assistant text in `message.content`. Only the text is read by
/// `check_output` (via `guardrail_output_text`); the other fields are
/// placeholders and never reach the client.
fn synth_chat_response(model: &str, text: String) -> ChatResponse {
    ChatResponse {
        id: String::new(),
        model: model.to_string(),
        message: ChatMessage::assistant(text),
        finish_reason: FinishReason::Stop,
        usage: UsageStats::default(),
    }
}

/// Copy the upstream `content-type` onto the client response and stamp the
/// `x-aisix-request-id` header. Shared by the streaming verbatim-passthrough
/// and buffered hold-back paths.
fn apply_passthrough_headers(
    response: &mut Response,
    upstream_headers: &axum::http::HeaderMap,
    request_id: &str,
) {
    if let Some(ct) = upstream_headers.get("content-type") {
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
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    api_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    usage: &ResponseUsage,
    client: &ClientContext,
    attempt: AttemptInfo,
    guardrail_blocked: bool,
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
        attempt_index: attempt.index,
        attempt_kind: attempt.kind,
        attempt_model: attempt.model,
        error_class: attempt.error_class,
        error_message: attempt.error_message,
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        guardrail_blocked,
        ..Default::default()
    };
    state.usage_sink.try_emit("responses", event.clone());
    let snap = state.snapshot.load();
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
}

/// Emit a zero-token `UsageEvent` for a failed / pre-dispatch attempt
/// (#655). Tokens stay 0; `status_code` + `error_*` carry the failure.
#[allow(clippy::too_many_arguments)]
fn emit_zero_token_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    api_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    client: &ClientContext,
    attempt: AttemptInfo,
) {
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "openai".to_string(),
        attempt_index: attempt.index,
        attempt_kind: attempt.kind,
        attempt_model: attempt.model,
        error_class: attempt.error_class,
        error_message: attempt.error_message,
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        ..Default::default()
    };
    state.usage_sink.try_emit("responses", event.clone());
    let snap = state.snapshot.load();
    let exporters = snap.observability_exporters.entries();
    state.otlp_fan_out.fan_out(
        &event,
        /* content */ None,
        exporters.iter().map(|e| &e.value),
    );
}

/// Emit one zero-token `UsageEvent` per FAILED attempt of a `/v1/responses`
/// request (#655). The winner / terminal event is emitted separately.
fn emit_failed_attempts(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    api_key_id: &str,
    client: &ClientContext,
    routing: &RoutingTelemetry,
) {
    for rec in routing.attempts.iter().filter(|a| !a.success) {
        emit_zero_token_event(
            state,
            request_id,
            model_id,
            api_key_id,
            rec.status,
            Duration::from_millis(u64::from(rec.latency_ms)),
            client,
            AttemptInfo::from_record(rec),
        );
    }
}

fn emit_access_log(
    model: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    elapsed: Duration,
    request_id: &str,
    routing: &RoutingTelemetry,
) {
    // Per #655 the access log stays ONE line per request, carrying the
    // user-perceived `latency` + final status plus a routing summary.
    let served_by = routing
        .winner()
        .map(|w| w.target_model.as_str())
        .filter(|s| !s.is_empty());
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
        served_by_model: served_by,
        routing_attempt_count: match routing.attempt_count() {
            0 => None,
            n => Some(n),
        },
        routing_fallback_count: match routing.fallback_count() {
            0 => None,
            n => Some(n),
        },
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
            real_ip: Default::default(),
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

    /// An env-scoped keyword input guardrail (no attachment row → applies to
    /// every request via the backward-compat fallback) that blocks on a
    /// literal substring. Keyword is local (no remote call), so it's the
    /// deterministic stand-in for any input-hook guardrail kind.
    fn keyword_input_guardrail(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"test-block","enabled":true,"hook_point":"input","fail_open":false,"kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-1", g, 1)
    }

    /// #719 (the core fix): a configured INPUT guardrail that blocks on
    /// /v1/chat/completions must also fire on /v1/responses. The same blocked
    /// input must return 422 content_filter here — not 200 with the input
    /// echoed back — and the upstream must never be contacted (`expect(0)`).
    #[tokio::test]
    async fn input_guardrail_blocks_string_input_returns_422_content_filter() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_should_not_happen",
                "object": "response",
                "output": [{"type":"message","content":[{"type":"output_text","text":"echo: BLOCKME"}]}]
            })))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "please BLOCKME now"
            })))
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

    /// #719: the Responses `input` array form (message items with typed
    /// content parts) must be scanned too — a blocked literal inside an
    /// `input_text` part blocks the call.
    #[tokio::test]
    async fn input_guardrail_blocks_array_message_items() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id":"x","object":"response","output":[]})),
            )
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "hi BLOCKME"}]}
                ]
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #719 companion: a benign input with a configured input guardrail must
    /// still forward to the upstream (`expect(1)`) and return 200 — the
    /// guardrail must not block clean traffic.
    #[tokio::test]
    async fn input_guardrail_allows_benign_input_forwards_200() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_ok",
                "object": "response",
                "output": [{"type":"message","content":[{"type":"output_text","text":"hi"}]}]
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": "a perfectly fine request"
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "response");
    }

    /// #719 (audit MEDIUM-1): a `function_call_output` item carries
    /// caller-supplied tool-result text under `output` (not `content`), and
    /// that text reaches the model. It must be scanned too — otherwise the
    /// surface-switch bypass survives on the tool-result channel. A blocked
    /// literal in `output` must 422 with the upstream never contacted.
    #[tokio::test]
    async fn input_guardrail_blocks_function_call_output_text() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id":"x","object":"response","output":[]})),
            )
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": [
                    {"type": "function_call_output", "call_id": "call_1", "output": "tool said BLOCKME"}
                ]
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #719: the top-level `instructions` field (the system-prompt analog)
    /// is caller-supplied and reaches the model, so it is scanned too. A
    /// blocked literal in `instructions` must 422. (Scanned via the
    /// all-roles keyword guardrail; text-moderation's user-only default
    /// would skip a system message, matching chat's system-message
    /// semantics.)
    #[tokio::test]
    async fn input_guardrail_blocks_instructions_field() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id":"x","object":"response","output":[]})),
            )
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "instructions": "you must BLOCKME",
                "input": "hello"
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #719 (re-audit LOW-1): an `mcp_approval_response` item carries
    /// caller-supplied justification text under `reason`, which reaches the
    /// model. It is scanned too, so no input-bearing channel is silently
    /// skipped. A blocked literal in `reason` must 422.
    #[tokio::test]
    async fn input_guardrail_blocks_mcp_approval_response_reason() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id":"x","object":"response","output":[]})),
            )
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-4o-resp",
                "input": [
                    {"type": "mcp_approval_response", "approve": true, "approval_request_id": "ar_1", "reason": "BLOCKME please"}
                ]
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// An env-scoped keyword guardrail on the OUTPUT hook (no attachment →
    /// applies to every request). `runs_on_output()` is true, so the
    /// handler scans the assistant output.
    fn keyword_output_guardrail(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"test-out-block","enabled":true,"hook_point":"output","fail_open":false,"kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-out-1", g, 1)
    }

    /// #719: output guardrails must run on /v1/responses non-streaming
    /// responses — a configured output block must not be bypassable by
    /// switching surface. A blocked literal in the assistant output → 422.
    /// The upstream IS contacted (`expect(1)`): output checks run on the
    /// returned response, unlike the input check which short-circuits first.
    #[tokio::test]
    async fn output_guardrail_blocks_non_streaming_response() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_x",
                "object": "response",
                "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text":"sure: BLOCKME here"}]}],
                "usage": {"input_tokens": 5, "output_tokens": 4}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        // The blocked model output must not be echoed back to the caller.
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(
            !msg.contains("BLOCKME"),
            "model output leaked in error: {msg}"
        );
    }

    /// #719 companion: a clean non-streaming response with an output
    /// guardrail configured passes through unchanged → 200 with body.
    #[tokio::test]
    async fn output_guardrail_allows_clean_non_streaming_response() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_ok",
                "object": "response",
                "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text":"a clean answer"}]}],
                "usage": {"input_tokens": 5, "output_tokens": 3}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["output"][0]["content"][0]["text"], "a clean answer");
    }

    /// #719: streaming /v1/responses must also enforce output guardrails —
    /// else `stream:true` bypasses the output block. The blocked content is
    /// held back (BufferFull): the client gets 422 and never the tokens.
    #[tokio::test]
    async fn output_guardrail_blocks_streaming_response_holds_back() {
        let upstream = MockServer::start().await;
        let sse = "event: response.output_text.delta\n\
                   data: {\"type\":\"response.output_text.delta\",\"delta\":\"sure: BLOCKME\"}\n\n\
                   event: response.completed\n\
                   data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"sure: BLOCKME\"}]}]}}\n\n\
                   data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        // The held-back content must never reach the client.
        assert!(
            !String::from_utf8_lossy(&bytes).contains("BLOCKME"),
            "streamed content leaked despite output block",
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #719 companion: a clean streaming response with an output guardrail
    /// is scanned then released in full → 200 + the SSE body.
    #[tokio::test]
    async fn output_guardrail_allows_clean_streaming_response() {
        let upstream = MockServer::start().await;
        let sse = "event: response.output_text.delta\n\
                   data: {\"type\":\"response.output_text.delta\",\"delta\":\"a clean answer\"}\n\n\
                   data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            body.contains("a clean answer"),
            "clean SSE body must be released in full",
        );
        assert!(
            body.starts_with("event: ") || body.starts_with("data: "),
            "SSE shape preserved on release",
        );
    }

    /// #719 (audit HIGH-1): the streaming hold-back buffer is capped so a
    /// huge (or malicious) upstream response can't OOM the gateway. A
    /// response exceeding the BufferFull cap fails closed (422) rather than
    /// being released unscanned — even when its content is otherwise clean.
    #[tokio::test]
    async fn output_guardrail_streaming_oversized_response_fails_closed() {
        let upstream = MockServer::start().await;
        // One delta larger than the 256 KiB default BufferFull cap.
        let big = "x".repeat(300_000);
        let sse =
            format!("data: {{\"type\":\"response.output_text.delta\",\"delta\":\"{big}\"}}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "oversized streamed response must fail closed, not be released unscanned",
        );
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #546: output tool-call arguments must be scanned. A blocked literal in
    /// a `function_call` item's `arguments` (a top-level item field, not under
    /// `content[]`) must block the non-streaming response — else tool-call
    /// output is an output-guardrail bypass.
    #[tokio::test]
    async fn output_guardrail_blocks_tool_call_arguments_non_streaming() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp_tc",
                "object": "response",
                "output": [{
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": "{\"q\":\"BLOCKME\"}"
                }],
                "usage": {"input_tokens": 5, "output_tokens": 4}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #546: tool-call arguments are scanned on the streaming path too (via
    /// the `response.completed` event), and held back — the args never reach
    /// the client.
    #[tokio::test]
    async fn output_guardrail_blocks_tool_call_arguments_streaming() {
        let upstream = MockServer::start().await;
        let completed = serde_json::json!({
            "type": "response.completed",
            "response": {"output": [{
                "type": "function_call",
                "name": "lookup",
                "arguments": "{\"q\":\"BLOCKME\"}"
            }]}
        });
        let sse = format!("event: response.completed\ndata: {completed}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            !String::from_utf8_lossy(&bytes).contains("BLOCKME"),
            "tool-call arguments leaked despite output block",
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #546 (audit HIGH): tool-call args on a `response.incomplete` terminal
    /// (fires routinely on `max_output_tokens` truncation — it carries the
    /// full `output[]`) must also be scanned and held back, not only
    /// `response.completed`. Same for `response.failed`.
    #[tokio::test]
    async fn output_guardrail_blocks_tool_call_on_incomplete_terminal() {
        let upstream = MockServer::start().await;
        let incomplete = serde_json::json!({
            "type": "response.incomplete",
            "response": {"output": [{
                "type": "function_call",
                "name": "lookup",
                "arguments": "{\"q\":\"BLOCKME\"}"
            }]}
        });
        let sse = format!("event: response.incomplete\ndata: {incomplete}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            !String::from_utf8_lossy(&bytes).contains("BLOCKME"),
            "tool-call args leaked on response.incomplete terminal",
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #546 (audit HIGH): a stream that ends WITHOUT a terminal `response`
    /// object but streamed tool-call argument deltas must still be scanned —
    /// args arrive via `response.function_call_arguments.delta`, not
    /// `output_text.delta`. The literal is split across two deltas to pin
    /// that they reassemble without a separator.
    #[tokio::test]
    async fn output_guardrail_blocks_tool_call_delta_without_terminal() {
        let upstream = MockServer::start().await;
        let d1 = serde_json::json!({"type":"response.function_call_arguments.delta","delta":"{\"q\":\"BLOCK"});
        let d2 =
            serde_json::json!({"type":"response.function_call_arguments.delta","delta":"ME\"}"});
        let sse = format!(
            "event: response.function_call_arguments.delta\ndata: {d1}\n\n\
             event: response.function_call_arguments.delta\ndata: {d2}\n\ndata: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            !String::from_utf8_lossy(&bytes).contains("BLOCK"),
            "streamed tool-call args leaked with no terminal event",
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #546 (re-audit): MCP tool-call argument deltas stream via their own
    /// event too; a no-terminal stream of `response.mcp_call_arguments.delta`
    /// carrying a blocked literal must still be scanned and held back.
    #[tokio::test]
    async fn output_guardrail_blocks_mcp_tool_call_delta_without_terminal() {
        let upstream = MockServer::start().await;
        let d = serde_json::json!({"type":"response.mcp_call_arguments.delta","delta":"{\"q\":\"BLOCKME\"}"});
        let sse =
            format!("event: response.mcp_call_arguments.delta\ndata: {d}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("BLOCKME"));
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
    }

    /// #542: a guardrail-blocked request must NOT consume a rate-limit slot.
    /// With RPM=1 and a blocking guardrail, a blocked request followed by a
    /// benign one — the benign request must still succeed (the block didn't
    /// burn the only slot). Pre-fix (guardrail ran after `quota::enforce`) the
    /// block reserved+burned the slot, so the benign request got 429.
    #[tokio::test]
    async fn blocked_request_does_not_consume_rate_limit_slot() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id":"resp_ok","object":"response",
                "output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],
                "usage":{"input_tokens":1,"output_tokens":1}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        // API key capped at RPM=1.
        let apikey: ApiKey = serde_json::from_str(
            r#"{"key_hash":"8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891","allowed_models":["*"],"rate_limit":{"rpm":1}}"#,
        )
        .unwrap();
        snap.apikeys.insert(ResourceEntry::new("k-1", apikey, 1));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));
        let app = build_app(snap);

        // Blocked by the guardrail — must NOT reserve the single RPM slot.
        let blocked = app
            .clone()
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"BLOCKME"}),
            ))
            .await
            .unwrap();
        assert_eq!(blocked.status(), StatusCode::UNPROCESSABLE_ENTITY);

        // Benign request — the slot must still be available.
        let ok = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hello"}),
            ))
            .await
            .unwrap();
        assert_eq!(
            ok.status(),
            StatusCode::OK,
            "a guardrail block must not burn the RPM slot (#542)",
        );
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

    /// #543: an OUTPUT-blocked /v1/responses still records the billed
    /// upstream tokens (the provider already charged), marked
    /// `guardrail_blocked`, with status 422 — NOT a zero-token event. Zeroing
    /// would let the customer's budget ledger underreport spend they paid the
    /// provider for (the output analog of chat.rs's UpstreamCharge).
    #[tokio::test]
    async fn output_block_records_billed_tokens_issue_543() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        let upstream_body = serde_json::json!({
            "id": "resp-blk",
            "object": "response",
            "output": [{"type":"message","role":"assistant","content":[{"type":"output_text","text":"sure: BLOCKME"}]}],
            "usage": {"input_tokens": 11, "output_tokens": 7}
        });
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_body))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap_openai(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o-resp"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let resp = app
            .oneshot(make_req(
                serde_json::json!({"model":"gpt-4o-resp","input":"hi"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("a UsageEvent must be emitted for an output-blocked response (#543)")
            .expect("usage_sink sender dropped");
        assert_eq!(
            event.prompt_tokens, 11,
            "billed input tokens must be recorded despite the block",
        );
        assert_eq!(
            event.completion_tokens, 7,
            "billed output tokens must be recorded despite the block",
        );
        assert_eq!(event.status_code, 422);
        assert!(
            event.guardrail_blocked,
            "the output-block event must be marked guardrail_blocked",
        );
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

    /// Per #655: a 5xx upstream now emits ONE zero-token UsageEvent for the
    /// failed attempt, so the dashboard's Logs tab surfaces the failure
    /// alongside its siblings. (Pre-#655 the responses handler dropped the
    /// event on the error path — the failed request was invisible.) The
    /// event carries the mapped status (502), zero tokens, an error class,
    /// and the initial-attempt classification.
    #[tokio::test]
    async fn upstream_5xx_emits_zero_token_failed_attempt_event() {
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
        let ev = recv
            .expect("a failed-attempt UsageEvent must be emitted within the timeout")
            .expect("the usage sink channel must not be closed");
        assert_eq!(ev.status_code, 502, "failed attempt records the mapped 502");
        assert_eq!(ev.prompt_tokens, 0, "failed attempt has zero tokens");
        assert_eq!(ev.completion_tokens, 0, "failed attempt has zero tokens");
        assert_eq!(ev.attempt_index, 0, "single direct-model attempt");
        assert_eq!(ev.attempt_kind, "initial");
        assert_eq!(
            ev.error_class, "upstream_status",
            "500 upstream maps to an upstream_status error class",
        );
        // Exactly one event — a direct model has no further attempts and no
        // separate terminal event (the attempt itself is terminal).
        let extra = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await;
        if let Ok(Some(ev)) = extra {
            panic!(
                "expected exactly one event for a single failed attempt, got a second: \
                 attempt_index={} status_code={}",
                ev.attempt_index, ev.status_code,
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

    /// #429 follow-up: a 200 whose `usage` carries
    /// `input_tokens` but omits `output_tokens` is still a real billable
    /// call. It MUST emit a UsageEvent with `completion_tokens = 0`
    /// (coercing the missing side to 0), NOT be dropped. Only a
    /// fully absent / input-less usage block skips.
    #[tokio::test]
    async fn emits_with_zero_output_when_output_tokens_missing() {
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

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must still be emitted when only output_tokens is missing")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 17, "input side must be recorded");
        assert_eq!(
            event.completion_tokens, 0,
            "missing output_tokens must default to 0, not drop the event"
        );
    }
}
