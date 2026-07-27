//! `/v1/realtime` — OpenAI Realtime WebSocket relay (#721,
//! AISIX-Cloud#873 §⑤).
//!
//! Authenticates on connect, resolves the target Model from `?model=`,
//! opens the provider WebSocket and relays frames bidirectionally.
//!
//! ## Protocol scope
//!
//! v1 relays the **OpenAI Realtime wire protocol**: adapter `openai`
//! (api.openai.com and any OpenAI-compatible `api_base`, which covers
//! xAI-style vendors) and `azure-openai` (`{base}/openai/realtime
//! ?api-version=…&deployment=…`, `api-key` header). Gemini Live / AWS
//! Bedrock speak entirely different session/event models and need a
//! cross-protocol translation layer (LiteLLM ships those as dedicated
//! per-provider `transform_realtime_request/response` modules) — that is
//! a separate feature, not part of this endpoint.
//!
//! ## Auth
//!
//! Two credential channels, checked before the upgrade completes:
//!
//! 1. `Authorization: Bearer <key>` / `x-api-key` headers — server-side
//!    clients (LiteLLM parity: `user_api_key_auth_websocket`).
//! 2. The `sec-websocket-protocol` item `openai-insecure-api-key.<key>`
//!    — browser clients cannot set headers; this is the documented
//!    OpenAI browser flow. The gateway echoes the `realtime` subprotocol
//!    when offered.
//!
//! Auth/ACL/quota failures reject the HTTP upgrade itself (401/403/429
//! envelope) rather than accept-then-close-1008: same enforcement point,
//! observable to every WS client as a failed handshake.
//!
//! ## Usage
//!
//! The relay harvests `response.done` usage frames (and
//! `conversation.item.input_audio_transcription.completed` token usage)
//! from the upstream stream and emits ONE aggregated UsageEvent per
//! session (`inbound_protocol = "realtime"`), committing total tokens to
//! the rate-limit reservation like the other non-chat surfaces (#911
//! [21]).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use aisix_core::models::model::Adapter;
use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::extract::ws::{CloseFrame, Message as AxMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, Method};
use axum::response::{IntoResponse, Response};
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message as TgMessage;

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::ProxyError;
use crate::state::ProxyState;
use aisix_gateway::BridgeError;

/// Azure Realtime GA api-version (see the jobs surface twin constant).
const AZURE_REALTIME_API_VERSION: &str = "2024-10-01-preview";

/// Subprotocol item carrying the caller's API key in the browser flow.
const SUBPROTOCOL_KEY_PREFIX: &str = "openai-insecure-api-key.";

pub(crate) async fn realtime(
    State(state): State<ProxyState>,
    Query(params): Query<HashMap<String, String>>,
    headers: HeaderMap,
    client: ClientContext,
    ws: WebSocketUpgrade,
) -> Response {
    let request_id = client.request_id.clone();
    let started = Instant::now();

    match prepare(&state, &params, &headers, &client).await {
        Ok(prep) => {
            let state2 = state.clone();
            let client2 = client.clone();
            // `on_upgrade` runs the session on a detached task, so the
            // request span has to be attached to the future rather than
            // inherited — without it the session's guardrail checks log
            // without a `request_id` (AISIX-Cloud#1060).
            let span = tracing::Span::current();
            ws.protocols(["realtime"]).on_upgrade(move |socket| {
                use tracing::Instrument as _;
                async move {
                    run_session(state2, prep, socket, client2, request_id, started).await;
                }
                .instrument(span)
            })
        }
        Err(err) => {
            let status = err.status().as_u16();
            emit_access_log(
                &Method::GET,
                status,
                started.elapsed(),
                &request_id,
                None,
                Some(&err),
            );
            crate::usage_attr::emit_error_usage_event(
                &state,
                "realtime",
                "realtime",
                &request_id,
                params.get("model").map(String::as_str).unwrap_or(""),
                "",
                status,
                err.kind(),
                &client,
            );
            err.into_response()
        }
    }
}

/// Everything resolved before the upgrade is accepted.
struct Prepared {
    auth: AuthenticatedKey,
    model_entry: std::sync::Arc<aisix_core::ResourceEntry<aisix_core::Model>>,
    pk_id: String,
    upstream_request: tokio_tungstenite::tungstenite::handshake::client::Request,
    reservation: aisix_ratelimit::MultiReservation,
    requested_model: String,
    provider_label: String,
}

async fn prepare(
    state: &ProxyState,
    params: &HashMap<String, String>,
    headers: &HeaderMap,
    client: &ClientContext,
) -> Result<Prepared, ProxyError> {
    let auth = authenticate(state, headers)?;

    let requested_model = params
        .get("model")
        .map(String::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if requested_model.is_empty() {
        return Err(ProxyError::InvalidRequest(
            "`model` query parameter is required on /v1/realtime".into(),
        ));
    }

    let snapshot = state.snapshot.load();
    let model_entry = crate::model_resolve::resolve_model(&snapshot, &requested_model)
        .ok_or_else(|| ProxyError::ModelNotFound(format!("model {requested_model:?} not found")))?;
    if !auth.key().can_access(&requested_model) {
        return Err(ProxyError::ModelForbidden(format!(
            "api key is not authorized for model {requested_model:?}"
        )));
    }
    let model = &model_entry.value;
    if model.is_routing() || model.is_ensemble() || model.is_semantic() {
        return Err(ProxyError::InvalidRequest(format!(
            "model {requested_model:?} is a virtual router; /v1/realtime requires a direct model"
        )));
    }
    crate::dispatch::check_ip_access(model, &client.source_ip)?;

    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;
    let secret = crate::dispatch::require_api_key(&pk_entry.value, model)?.to_string();
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();

    let upstream_request = match pk_entry.value.adapter {
        Some(Adapter::Openai) => {
            let base = crate::dispatch::resolve_base_url(&pk_entry.value)?;
            let url = crate::dispatch::build_v1_url(&base, "/realtime");
            let url = format!(
                "{}?model={}",
                to_ws_scheme(&url)?,
                urlencode(&upstream_model)
            );
            let mut req = url.into_client_request().map_err(|e| {
                ProxyError::InvalidRequest(format!("invalid upstream realtime URL: {e}"))
            })?;
            req.headers_mut().insert(
                "authorization",
                format!("Bearer {secret}").parse().map_err(|_| {
                    ProxyError::InvalidRequest("provider secret is not header-safe".into())
                })?,
            );
            // LiteLLM parity (OpenAIRealtime.async_realtime): the beta
            // header is sent unconditionally; GA endpoints ignore it.
            req.headers_mut()
                .insert("openai-beta", "realtime=v1".parse().unwrap());
            req
        }
        Some(Adapter::AzureOpenai) => {
            let base = pk_entry
                .value
                .api_base
                .as_deref()
                .map(str::trim)
                .filter(|b| !b.is_empty())
                .ok_or_else(|| {
                    ProxyError::InvalidRequest(format!(
                        "azure provider_key {:?} has no api_base",
                        pk_entry.value.display_name
                    ))
                })?
                .trim_end_matches('/')
                .to_string();
            let url = format!(
                "{}/openai/realtime?api-version={AZURE_REALTIME_API_VERSION}&deployment={}",
                to_ws_scheme(&base)?,
                urlencode(&upstream_model)
            );
            let mut req = url.into_client_request().map_err(|e| {
                ProxyError::InvalidRequest(format!("invalid upstream realtime URL: {e}"))
            })?;
            req.headers_mut().insert(
                "api-key",
                secret.parse().map_err(|_| {
                    ProxyError::InvalidRequest("provider secret is not header-safe".into())
                })?,
            );
            req
        }
        _ => {
            return Err(ProxyError::InvalidRequest(format!(
                "model {requested_model:?} uses provider {:?} which does not speak the OpenAI \
                 Realtime protocol; /v1/realtime supports OpenAI-compatible and Azure OpenAI \
                 providers",
                pk_entry.value.provider
            )));
        }
    };
    let reservation = crate::quota::enforce(
        state,
        &auth,
        Some(&crate::quota::ModelRateLimit::from_model(
            &model_entry.value.display_name,
            &model_entry.id,
            &model_entry.value,
        )),
    )
    .await?;

    let provider_label = model.provider.clone().unwrap_or_default();
    Ok(Prepared {
        auth,
        pk_id: pk_entry.id.to_string(),
        model_entry,
        upstream_request,
        reservation,
        requested_model,
        provider_label,
    })
}

/// Header bearer (`Authorization` / `x-api-key`) first, then the browser
/// subprotocol credential.
fn authenticate(state: &ProxyState, headers: &HeaderMap) -> Result<AuthenticatedKey, ProxyError> {
    if let Some(auth) = headers.get(axum::http::header::AUTHORIZATION) {
        let s = auth.to_str().map_err(|_| ProxyError::MissingAuth)?;
        let token = s.strip_prefix("Bearer ").map(str::trim).unwrap_or("");
        if token.is_empty() {
            return Err(ProxyError::MissingAuth);
        }
        return crate::auth::authenticate_token(state, token);
    }
    if let Some(raw) = headers.get("x-api-key") {
        let token = raw.to_str().map_err(|_| ProxyError::MissingAuth)?.trim();
        if token.is_empty() {
            return Err(ProxyError::MissingAuth);
        }
        return crate::auth::authenticate_token(state, token);
    }
    if let Some(proto) = headers.get("sec-websocket-protocol") {
        let s = proto.to_str().map_err(|_| ProxyError::MissingAuth)?;
        for item in s.split(',') {
            if let Some(token) = item.trim().strip_prefix(SUBPROTOCOL_KEY_PREFIX) {
                if !token.is_empty() {
                    return crate::auth::authenticate_token(state, token);
                }
            }
        }
    }
    Err(ProxyError::MissingAuth)
}

fn to_ws_scheme(url: &str) -> Result<String, ProxyError> {
    if let Some(rest) = url.strip_prefix("https://") {
        Ok(format!("wss://{rest}"))
    } else if let Some(rest) = url.strip_prefix("http://") {
        Ok(format!("ws://{rest}"))
    } else if url.starts_with("ws://") || url.starts_with("wss://") {
        Ok(url.to_string())
    } else {
        Err(ProxyError::InvalidRequest(format!(
            "api_base {url:?} has no http(s) scheme"
        )))
    }
}

fn urlencode(s: &str) -> String {
    // Conservative percent-encoding for the query-value position.
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Accumulated session usage harvested from upstream frames.
#[derive(Default)]
struct SessionUsage {
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
    responses: u32,
}

impl SessionUsage {
    fn absorb(&mut self, text: &str) {
        // Fast path: only parse frames that can carry usage.
        if !text.contains("\"response.done\"")
            && !text.contains("\"conversation.item.input_audio_transcription.completed\"")
        {
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(text) else {
            return;
        };
        match v.get("type").and_then(Value::as_str) {
            Some("response.done") => {
                let usage = &v["response"]["usage"];
                self.input_tokens += usage["input_tokens"].as_u64().unwrap_or(0);
                self.output_tokens += usage["output_tokens"].as_u64().unwrap_or(0);
                self.cached_tokens += usage["input_token_details"]["cached_tokens"]
                    .as_u64()
                    .unwrap_or(0);
                self.responses += 1;
            }
            Some("conversation.item.input_audio_transcription.completed") => {
                // Transcription-intent sessions bill via this frame
                // (LiteLLM `_capture_transcription_usage`).
                let usage = &v["usage"];
                if usage.get("type").and_then(Value::as_str) == Some("tokens") {
                    self.input_tokens += usage["input_tokens"].as_u64().unwrap_or(0);
                    self.output_tokens += usage["output_tokens"].as_u64().unwrap_or(0);
                    self.responses += 1;
                }
            }
            _ => {}
        }
    }
}

async fn run_session(
    state: ProxyState,
    prep: Prepared,
    client_ws: WebSocket,
    client: ClientContext,
    request_id: String,
    started: Instant,
) {
    let Prepared {
        auth,
        model_entry,
        pk_id,
        upstream_request,
        reservation,
        requested_model,
        provider_label,
    } = prep;

    let guardrail_ctx = aisix_guardrails::RequestContext {
        model_id: &model_entry.id,
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let chain = state.guardrail_index.resolve(&guardrail_ctx);

    let (mut client_tx, mut client_rx) = client_ws.split();

    let upstream = match tokio_tungstenite::connect_async(upstream_request).await {
        Ok((ws, _resp)) => ws,
        Err(e) => {
            tracing::warn!(error = %e, model = %requested_model, "realtime upstream connect failed");
            let _ = client_tx
                .send(AxMessage::Text(
                    serde_json::json!({
                        "type": "error",
                        "error": {
                            "type": "upstream_error",
                            "message": "failed to connect to the upstream realtime endpoint"
                        }
                    })
                    .to_string(),
                ))
                .await;
            let _ = client_tx
                .send(AxMessage::Close(Some(CloseFrame {
                    code: 1011,
                    reason: "upstream connect failed".into(),
                })))
                .await;
            // `note_failure` hands the error back, so the same value that
            // drove the cooldown decision also names the failure in the
            // access log instead of being rebuilt.
            let connect_err = ProxyError::Bridge(crate::cooldown::note_failure(
                &state.runtime_status,
                &model_entry.id,
                model_entry.value.cooldown.as_ref(),
                aisix_gateway::BridgeError::Transport(aisix_gateway::error_with_causes(&e)),
            ));
            emit_access_log(
                &Method::GET,
                502,
                started.elapsed(),
                &request_id,
                Some((&provider_label, &requested_model)),
                Some(&connect_err),
            );
            crate::usage_attr::emit_error_usage_event(
                &state,
                "realtime",
                "realtime",
                &request_id,
                &requested_model,
                &auth.entry.id,
                502,
                "transport",
                &client,
            );
            return;
        }
    };
    let (mut up_tx, mut up_rx) = upstream.split();

    let mut usage = SessionUsage::default();
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    let mut close_status: u16 = 200;
    // Paired with `close_status`: every branch that sets a FAILING status
    // also names the failure, so the access log can say why a session
    // ended (AISIX-Cloud#1093).
    //
    // A client-side transport error (`Some(Err(_))` on the receive half)
    // is deliberately not one of them: it keeps `close_status` 200 and
    // stays `None`. Reclassifying it is a behaviour change, not a logging
    // one — `RequestOutcome::from_status` would flip that session from
    // `success` to `client_error` and move every operator's realtime
    // success rate. That belongs with the termination-reason taxonomy
    // (`downstream_remote_disconnect` and friends) the issue asks for
    // separately, which needs its own status decision.
    let mut session_error: Option<ProxyError> = None;
    // Operator-configured stream idle deadline (stream_timeout on the
    // Model). Absent → no idle cap; realtime sessions are long-lived by
    // design.
    let idle_cap = model_entry.value.stream_timeout_effective();

    loop {
        let next = async {
            tokio::select! {
                m = client_rx.next() => Dir::FromClient(m),
                m = up_rx.next() => Dir::FromUpstream(m),
            }
        };
        let event = match idle_cap {
            Some(cap) => match tokio::time::timeout(cap, next).await {
                Ok(r) => r,
                Err(_) => {
                    let _ = client_tx
                        .send(AxMessage::Close(Some(CloseFrame {
                            code: 1001,
                            reason: "idle timeout".into(),
                        })))
                        .await;
                    close_status = 504;
                    session_error = Some(ProxyError::Bridge(BridgeError::Timeout {
                        elapsed_ms: cap.as_millis() as u64,
                        cause: "no realtime frame within the stream idle budget".into(),
                    }));
                    break;
                }
            },
            None => next.await,
        };

        match event {
            Dir::FromClient(m) => match m {
                Some(Ok(AxMessage::Text(text))) => {
                    if !chain.is_empty() {
                        if let Some(resp) = guardrail_block_event(
                            &chain,
                            &model_entry.value.display_name,
                            &text,
                            true,
                            &mut monitor_hits,
                        )
                        .await
                        {
                            let _ = client_tx.send(AxMessage::Text(resp)).await;
                            let _ = client_tx
                                .send(AxMessage::Close(Some(CloseFrame {
                                    code: 1011,
                                    reason: "content policy".into(),
                                })))
                                .await;
                            close_status = 400;
                            session_error = Some(ProxyError::ContentFiltered(
                                "realtime frame blocked by a guardrail".into(),
                            ));
                            break;
                        }
                    }
                    if up_tx.send(TgMessage::Text(text)).await.is_err() {
                        break;
                    }
                }
                Some(Ok(AxMessage::Binary(b))) => {
                    if up_tx.send(TgMessage::Binary(b)).await.is_err() {
                        break;
                    }
                }
                Some(Ok(AxMessage::Close(_))) | None => {
                    let _ = up_tx.send(TgMessage::Close(None)).await;
                    break;
                }
                Some(Ok(_)) => {} // ping/pong handled by the transports
                Some(Err(_)) => {
                    let _ = up_tx.send(TgMessage::Close(None)).await;
                    break;
                }
            },
            Dir::FromUpstream(m) => match m {
                Some(Ok(TgMessage::Text(text))) => {
                    usage.absorb(&text);
                    if !chain.is_empty() {
                        if let Some(resp) = guardrail_block_event(
                            &chain,
                            &model_entry.value.display_name,
                            &text,
                            false,
                            &mut monitor_hits,
                        )
                        .await
                        {
                            let _ = client_tx.send(AxMessage::Text(resp)).await;
                            let _ = client_tx
                                .send(AxMessage::Close(Some(CloseFrame {
                                    code: 1011,
                                    reason: "content policy".into(),
                                })))
                                .await;
                            close_status = 400;
                            session_error = Some(ProxyError::ContentFiltered(
                                "realtime frame blocked by a guardrail".into(),
                            ));
                            break;
                        }
                    }
                    if client_tx.send(AxMessage::Text(text)).await.is_err() {
                        break;
                    }
                }
                Some(Ok(TgMessage::Binary(b))) => {
                    if client_tx.send(AxMessage::Binary(b)).await.is_err() {
                        break;
                    }
                }
                Some(Ok(TgMessage::Close(frame))) => {
                    let _ = client_tx
                        .send(AxMessage::Close(frame.map(|f| CloseFrame {
                            code: f.code.into(),
                            reason: f.reason.to_string().into(),
                        })))
                        .await;
                    break;
                }
                Some(Ok(_)) => {} // ping/pong/raw frames
                Some(Err(e)) => {
                    tracing::debug!(error = %e, "realtime upstream stream error");
                    let _ = client_tx
                        .send(AxMessage::Close(Some(CloseFrame {
                            code: 1011,
                            reason: "upstream error".into(),
                        })))
                        .await;
                    close_status = 502;
                    session_error = Some(ProxyError::Bridge(BridgeError::Transport(
                        aisix_gateway::error_with_causes(&e),
                    )));
                    break;
                }
                None => {
                    let _ = client_tx.send(AxMessage::Close(None)).await;
                    break;
                }
            },
        }
    }

    let elapsed = started.elapsed();
    let total_tokens = usage.input_tokens + usage.output_tokens;
    reservation.commit_tokens(total_tokens).await;

    emit_access_log(
        &Method::GET,
        close_status,
        elapsed,
        &request_id,
        Some((&provider_label, &requested_model)),
        session_error.as_ref(),
    );
    state.metrics.record_request(
        &provider_label,
        &model_entry.value.display_name,
        close_status,
        RequestOutcome::from_status(close_status),
        elapsed,
    );

    let snap = state.snapshot.load();
    let mut event = UsageEvent {
        request_id: request_id.clone(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_entry.id.clone(),
        api_key_id: auth.entry.id.clone(),
        requested_model: requested_model.clone(),
        prompt_tokens: usage.input_tokens.min(u32::MAX as u64) as u32,
        completion_tokens: usage.output_tokens.min(u32::MAX as u64) as u32,
        cached_prompt_tokens: usage.cached_tokens.min(u32::MAX as u64) as u32,
        status_code: close_status,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        cost_usd: model_entry
            .value
            .cost
            .as_ref()
            .map(|c| c.calculate(usage.input_tokens, usage.output_tokens))
            .unwrap_or(0.0),
        inbound_protocol: "realtime".to_string(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        guardrail_monitor_hits: monitor_hits,
        ..Default::default()
    };
    crate::usage_attr::apply_pk_telemetry(&mut event, &snap, &pk_id);
    state.usage_sink.try_emit("realtime", event.clone());
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
}

enum Dir {
    FromClient(Option<Result<AxMessage, axum::Error>>),
    FromUpstream(Option<Result<TgMessage, tokio_tungstenite::tungstenite::Error>>),
}

/// Whole-frame guardrail scan (the `/passthrough` blob precedent applied
/// per WS text frame). Returns the client-facing error event on Block.
async fn guardrail_block_event(
    chain: &aisix_guardrails::GuardrailChain,
    model_name: &str,
    text: &str,
    input_side: bool,
    monitor_hits: &mut Vec<aisix_core::GuardrailMonitorHit>,
) -> Option<String> {
    let (verdict, hits) = if input_side {
        let chat = aisix_gateway::ChatFormat::new(
            model_name,
            vec![aisix_gateway::ChatMessage::user(text.to_string())],
        );
        aisix_guardrails::Guardrail::check_input_observed(chain, &chat).await
    } else {
        let synth = aisix_gateway::ChatResponse {
            id: String::new(),
            model: model_name.to_string(),
            message: aisix_gateway::ChatMessage::assistant(text.to_string()),
            finish_reason: aisix_gateway::FinishReason::Stop,
            usage: aisix_gateway::UsageStats::default(),
        };
        aisix_guardrails::Guardrail::check_output_observed(chain, &synth).await
    };
    monitor_hits.extend(hits);
    if let aisix_guardrails::GuardrailVerdict::Block {
        reason,
        guardrail_name,
    } = verdict
    {
        let side = if input_side { "input" } else { "output" };
        tracing::warn!(
            guardrail_hook = side,
            reason = %reason,
            "guardrail blocked realtime frame",
        );
        let msg = crate::error::guardrail_block_message(
            if input_side { "request" } else { "response" },
            guardrail_name.as_deref(),
        );
        return Some(
            serde_json::json!({
                "type": "error",
                "error": {"type": "invalid_request_error", "code": "content_filtered", "message": msg}
            })
            .to_string(),
        );
    }
    None
}

fn emit_access_log(
    method: &Method,
    status: u16,
    elapsed: Duration,
    request_id: &str,
    target: Option<(&str, &str)>,
    error: Option<&ProxyError>,
) {
    let (error_kind, error) = match error {
        Some(e) => {
            let (kind, msg) = crate::attempt::access_log_error(e);
            (Some(kind), Some(msg))
        }
        None => (None, None),
    };
    AccessLog {
        method: method.as_str(),
        path: "/v1/realtime",
        status,
        latency: elapsed,
        provider: target.map(|(p, _)| p).filter(|p| !p.is_empty()),
        model: target.map(|(_, m)| m),
        api_key_id: None,
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
        error_kind,
        error: error.as_deref(),
    }
    .emit();
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use aisix_obs::{UsageEvent as ObsUsageEvent, UsageSink};
    use futures::{SinkExt, StreamExt};
    use std::sync::{Arc, Mutex};
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    const PK_ID: &str = "22222222-2222-2222-2222-222222222222";
    // sha256("sk-caller") — the plaintext used by all tests below.
    const CALLER_HASH: &str = "8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891";

    fn snapshot(api_base: &str, adapter: &str, provider: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        let pk_json = format!(
            r#"{{"display_name":"rt-pk","secret":"sk-up","api_base":"{api_base}","provider":"{provider}","adapter":"{adapter}"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        let m_json = format!(
            r#"{{"display_name":"rt-model","provider":"{provider}","model_name":"gpt-realtime","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&m_json).unwrap();
        snap.models.insert(ResourceEntry::new("m-rt", m, 1));
        let k_json = format!(r#"{{"key_hash":"{CALLER_HASH}","allowed_models":["*"]}}"#);
        let k: ApiKey = serde_json::from_str(&k_json).unwrap();
        snap.apikeys.insert(ResourceEntry::new("k-1", k, 1));
        snap
    }

    /// Bind the full proxy router on a real TCP port (WS handshakes need a
    /// live connection; `oneshot` can't upgrade).
    async fn serve(
        snap: AisixSnapshot,
    ) -> (
        std::net::SocketAddr,
        tokio::sync::mpsc::Receiver<ObsUsageEvent>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::channel::<ObsUsageEvent>(16);
        let hub = Arc::new(Hub::new());
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app.into_make_service())
                .await
                .unwrap();
        });
        (addr, rx)
    }

    /// Scripted mock upstream: accepts ONE WebSocket, records the request
    /// path + auth header, waits for one text frame, replies with a
    /// `response.done` usage frame, then closes.
    async fn spawn_upstream() -> (
        std::net::SocketAddr,
        Arc<Mutex<Option<(String, String)>>>,
        Arc<Mutex<Vec<String>>>,
    ) {
        let seen_handshake: Arc<Mutex<Option<(String, String)>>> = Arc::new(Mutex::new(None));
        let seen_frames: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hs = seen_handshake.clone();
        let frames = seen_frames.clone();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let hs2 = hs.clone();
            let ws = tokio_tungstenite::accept_hdr_async(
                stream,
                move |req: &tokio_tungstenite::tungstenite::handshake::server::Request,
                      resp: tokio_tungstenite::tungstenite::handshake::server::Response| {
                    let auth = req
                        .headers()
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("")
                        .to_string();
                    *hs2.lock().unwrap() = Some((req.uri().to_string(), auth));
                    Ok(resp)
                },
            )
            .await
            .unwrap();
            let (mut tx, mut rx) = ws.split();
            while let Some(Ok(msg)) = rx.next().await {
                if let TgMessage::Text(t) = msg {
                    frames.lock().unwrap().push(t.clone());
                    tx.send(TgMessage::Text(
                        serde_json::json!({
                            "type": "response.done",
                            "response": {"usage": {
                                "input_tokens": 7,
                                "output_tokens": 3,
                                "input_token_details": {"cached_tokens": 1}
                            }}
                        })
                        .to_string(),
                    ))
                    .await
                    .unwrap();
                    tx.send(TgMessage::Close(None)).await.ok();
                    break;
                }
            }
        });
        (addr, seen_handshake, seen_frames)
    }

    #[tokio::test]
    async fn relays_frames_and_emits_aggregated_usage_event() {
        let (up_addr, handshake, frames) = spawn_upstream().await;
        let snap = snapshot(&format!("http://{up_addr}/v1"), "openai", "openai");
        let (addr, mut rx) = serve(snap).await;

        let mut req = format!("ws://{addr}/v1/realtime?model=rt-model")
            .into_client_request()
            .unwrap();
        req.headers_mut()
            .insert("authorization", "Bearer sk-caller".parse().unwrap());
        let (ws, _) = tokio_tungstenite::connect_async(req)
            .await
            .expect("handshake");
        let (mut tx, mut client_rx) = ws.split();

        tx.send(TgMessage::Text(
            serde_json::json!({"type": "session.update", "session": {"instructions": "hi"}})
                .to_string(),
        ))
        .await
        .unwrap();

        // The upstream's response.done frame must reach the client verbatim.
        let mut got_done = false;
        while let Some(Ok(msg)) = client_rx.next().await {
            match msg {
                TgMessage::Text(t) if t.contains("response.done") => {
                    got_done = true;
                }
                TgMessage::Close(_) => break,
                _ => {}
            }
        }
        assert!(got_done, "client must receive the upstream response.done");

        // Upstream saw the relayed client frame + the gateway's provider auth.
        assert_eq!(frames.lock().unwrap().len(), 1);
        assert!(frames.lock().unwrap()[0].contains("session.update"));
        let (uri, auth) = handshake
            .lock()
            .unwrap()
            .clone()
            .expect("handshake recorded");
        assert!(
            uri.contains("/v1/realtime") && uri.contains("model=gpt-realtime"),
            "upstream URI must be the realtime path with the UPSTREAM model id, got {uri}"
        );
        assert_eq!(auth, "Bearer sk-up");

        // Session-aggregate usage event.
        let ev = tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("usage event expected")
            .expect("sink closed");
        assert_eq!(ev.inbound_protocol, "realtime");
        assert_eq!(ev.prompt_tokens, 7);
        assert_eq!(ev.completion_tokens, 3);
        assert_eq!(ev.cached_prompt_tokens, 1);
        assert_eq!(ev.requested_model, "rt-model");
        assert_eq!(ev.api_key_id, "k-1");
    }

    #[tokio::test]
    async fn subprotocol_key_authenticates_and_realtime_is_echoed() {
        let (up_addr, _handshake, _frames) = spawn_upstream().await;
        let snap = snapshot(&format!("http://{up_addr}/v1"), "openai", "openai");
        let (addr, _rx) = serve(snap).await;

        let mut req = format!("ws://{addr}/v1/realtime?model=rt-model")
            .into_client_request()
            .unwrap();
        // Browser flow: no headers, key rides the subprotocol list.
        req.headers_mut().insert(
            "sec-websocket-protocol",
            "realtime, openai-insecure-api-key.sk-caller, openai-beta.realtime-v1"
                .parse()
                .unwrap(),
        );
        let (ws, resp) = tokio_tungstenite::connect_async(req)
            .await
            .expect("subprotocol auth must be accepted");
        assert_eq!(
            resp.headers()
                .get("sec-websocket-protocol")
                .and_then(|v| v.to_str().ok()),
            Some("realtime"),
            "the gateway must echo the `realtime` subprotocol"
        );
        drop(ws);
    }

    #[tokio::test]
    async fn missing_auth_rejects_the_handshake() {
        let snap = snapshot("http://127.0.0.1:9/v1", "openai", "openai");
        let (addr, _rx) = serve(snap).await;

        let req = format!("ws://{addr}/v1/realtime?model=rt-model")
            .into_client_request()
            .unwrap();
        let err = tokio_tungstenite::connect_async(req)
            .await
            .expect_err("handshake must fail without credentials");
        let msg = err.to_string();
        assert!(msg.contains("401"), "expected 401 rejection, got: {msg}");
    }

    #[tokio::test]
    async fn missing_model_param_rejects_with_400() {
        let snap = snapshot("http://127.0.0.1:9/v1", "openai", "openai");
        let (addr, mut rx) = serve(snap).await;

        let mut req = format!("ws://{addr}/v1/realtime")
            .into_client_request()
            .unwrap();
        req.headers_mut()
            .insert("authorization", "Bearer sk-caller".parse().unwrap());
        let err = tokio_tungstenite::connect_async(req)
            .await
            .expect_err("handshake must fail without ?model=");
        assert!(err.to_string().contains("400"), "got: {err}");

        // The failure surfaces in Logs under the SAME protocol tag as a
        // successful realtime session, so protocol filtering catches both.
        let ev = tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("a failed realtime handshake must emit an error UsageEvent")
            .expect("sink closed");
        assert_eq!(ev.status_code, 400);
        assert_eq!(
            ev.inbound_protocol, "realtime",
            "error event must carry the realtime protocol tag, not \"openai\""
        );
    }

    #[tokio::test]
    async fn non_realtime_capable_adapter_is_rejected() {
        let snap = snapshot("http://127.0.0.1:9", "anthropic", "anthropic");
        let (addr, _rx) = serve(snap).await;

        let mut req = format!("ws://{addr}/v1/realtime?model=rt-model")
            .into_client_request()
            .unwrap();
        req.headers_mut()
            .insert("authorization", "Bearer sk-caller".parse().unwrap());
        let err = tokio_tungstenite::connect_async(req)
            .await
            .expect_err("handshake must fail on a non-OpenAI-protocol provider");
        assert!(err.to_string().contains("400"), "got: {err}");
    }

    #[tokio::test]
    async fn model_acl_rejects_unauthorized_key() {
        let (up_addr, _h, _f) = spawn_upstream().await;
        let snap = snapshot(&format!("http://{up_addr}/v1"), "openai", "openai");
        // Restrict the caller key to a different model.
        let k_json = format!(r#"{{"key_hash":"{CALLER_HASH}","allowed_models":["other-model"]}}"#);
        let k: ApiKey = serde_json::from_str(&k_json).unwrap();
        snap.apikeys.insert(ResourceEntry::new("k-1", k, 2));
        let (addr, _rx) = serve(snap).await;

        let mut req = format!("ws://{addr}/v1/realtime?model=rt-model")
            .into_client_request()
            .unwrap();
        req.headers_mut()
            .insert("authorization", "Bearer sk-caller".parse().unwrap());
        let err = tokio_tungstenite::connect_async(req)
            .await
            .expect_err("handshake must fail on model ACL");
        assert!(err.to_string().contains("403"), "got: {err}");
    }
}
