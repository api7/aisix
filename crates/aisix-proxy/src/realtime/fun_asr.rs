//! DashScope Fun-ASR names its model in run-task, after the client upgrade.
//! Each serial task therefore owns its authorization, upstream socket and usage row.

use super::{
    connect_failure, connect_upstream, guardrail_block_event, to_ws_scheme, HandedBack,
    REALTIME_HANDSHAKE_SLOTS,
};
use crate::{
    auth::AuthenticatedKey, client_ip::ClientContext, error::ProxyError, state::ProxyState,
};
use aisix_gateway::BridgeError;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio_tungstenite::tungstenite::{client::IntoClientRequest, Message as UpMessage};
use tracing::Instrument;

const PATH: &str = "/api-ws/v1/inference";

pub(crate) async fn inference(
    State(state): State<ProxyState>,
    method: axum::http::Method,
    auth: AuthenticatedKey,
    client: ClientContext,
    upgrade: Result<WebSocketUpgrade, axum::extract::ws::rejection::WebSocketUpgradeRejection>,
) -> Response {
    let ws = match upgrade {
        Ok(ws) => ws,
        Err(err) => {
            return crate::reject::reject_before_dispatch(
                &state,
                method.as_str(),
                PATH,
                &client.request_id,
                Some(&auth.entry.id),
                Instant::now(),
                crate::reject::Envelope::OpenAi,
                ProxyError::WebSocketUpgradeRequired {
                    status: err.status(),
                    detail: err.body_text(),
                },
            );
        }
    };
    let drain = crate::DrainGuard::new(state.livez.clone());
    let span = tracing::Span::current();
    ws.on_upgrade(move |mut ws| {
        async move {
            serve(&state, auth, client, &mut ws).await;
            drop(drain);
        }
        .instrument(span)
    })
}

fn invalid(message: &str) -> ProxyError {
    ProxyError::InvalidRequest(message.to_owned())
}

fn upstream_error(message: impl ToString) -> ProxyError {
    ProxyError::Bridge(BridgeError::Transport(message.to_string()))
}

fn failure_frame(id: &str, err: &ProxyError) -> String {
    let error = err.envelope().error;
    json!({"header": {"task_id": id, "event": "task-failed",
        "error_code": error.code.unwrap_or(error.kind), "error_message": error.message}, "payload": {}}).to_string()
}

async fn fail(ws: &mut WebSocket, id: &str, err: &ProxyError) {
    let _ = tokio::time::timeout(super::UPSTREAM_CLOSE_ACK, async {
        ws.send(Message::Text(failure_frame(id, err))).await?;
        ws.send(Message::Close(None)).await
    })
    .await;
}

async fn bounded<F: std::future::Future>(
    limit: Option<Duration>,
    finish_sent: Option<Instant>,
    f: F,
) -> Result<F::Output, ProxyError> {
    let Some(limit) = limit else {
        return Ok(f.await);
    };
    let timeout = || {
        ProxyError::Bridge(BridgeError::Timeout {
            elapsed_ms: limit.as_millis() as u64,
            cause: "DashScope WebSocket idle timeout".into(),
        })
    };
    let elapsed = finish_sent.map_or(Duration::ZERO, |start| start.elapsed());
    let remaining = limit.checked_sub(elapsed).ok_or_else(timeout)?;
    tokio::time::timeout(remaining, f)
        .await
        .map_err(|_| timeout())
}

async fn serve(
    state: &ProxyState,
    mut auth: AuthenticatedKey,
    mut client: ClientContext,
    ws: &mut WebSocket,
) {
    let mut handshake_span = Some(tracing::Span::current());
    loop {
        let Ok(Some(Ok(frame))) = bounded(state.default_timeouts.stream, None, ws.recv()).await
        else {
            return;
        };
        if matches!(frame, Message::Ping(_) | Message::Pong(_)) {
            continue;
        }
        if matches!(frame, Message::Close(_)) {
            return;
        }
        let span = handshake_span.take().unwrap_or_else(|| {
            client.request_id = crate::request_id::new_request_id();
            client.trace = client
                .trace
                .as_ref()
                .map(|trace| Arc::new(trace.next_request()));
            tracing::error_span!(parent: None, "request", request_id = %client.request_id)
        });
        let started = Instant::now();
        let reject = |err: &ProxyError| {
            span.in_scope(|| {
                crate::reject::record_rejection(
                    state,
                    "GET",
                    PATH,
                    &client.request_id,
                    Some(&auth.entry.id),
                    started,
                    err,
                )
            });
        };
        let command = match frame {
            Message::Text(text) => serde_json::from_str::<Value>(&text)
                .map(|run| (run, text))
                .map_err(|_| invalid("expected a JSON run-task command")),
            _ => Err(invalid("expected run-task before audio")),
        };
        let (mut run, text) = match command {
            Ok((run, text)) => (run, text),
            Err(err) => {
                reject(&err);
                fail(ws, "", &err).await;
                return;
            }
        };
        let id = run
            .pointer("/header/task_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let model = run
            .pointer("/payload/model")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        if run.pointer("/header/action").and_then(Value::as_str) != Some("run-task")
            || uuid::Uuid::parse_str(&id).is_err()
            || run.pointer("/payload/task_group").and_then(Value::as_str) != Some("audio")
            || run.pointer("/payload/task").and_then(Value::as_str) != Some("asr")
            || run.pointer("/payload/function").and_then(Value::as_str) != Some("recognition")
            || model.is_empty()
        {
            let err = invalid("expected an audio/asr/recognition run-task with task_id and model");
            reject(&err);
            fail(ws, &id, &err).await;
            return;
        }
        let snap = state.snapshot.load();
        client.request_headers = crate::client_ip::capture_request_headers(&client.headers, &snap);
        let cap = crate::usage_attr::content_cap(&snap);
        let attribution = Arc::new(crate::attribution::RequestAttribution::new(
            client.user_agent.clone(),
            crate::client_ip::exported_header_names(&snap, true),
            cap,
            client.request_id.clone(),
        ));
        drop(snap);
        let mut task = Task {
            id,
            model,
            started,
            dial_started: None,
            duration: 0.0,
            upstream_confirmed_id: false,
            capture: cap
                .map(|cap| cap as usize)
                .map(|cap| (aisix_obs::CapturedContent::new(&text, "", cap), cap)),
            audit: Default::default(),
            hits: Vec::new(),
        };
        let result = crate::attribution::scope(attribution, async {
            let result = task.run(state, &mut auth, &mut client, ws, &mut run).await;
            task.record(state, &auth, &client, &result);
            result
        })
        .instrument(span)
        .await;
        match result {
            Ok(End::Finished) => continue,
            Ok(End::Disconnected) => return,
            Ok(End::Failed(_)) => {
                let _ =
                    tokio::time::timeout(super::UPSTREAM_CLOSE_ACK, ws.send(Message::Close(None)))
                        .await;
                return;
            }
            Err(err) => {
                fail(ws, &task.id, &err).await;
                return;
            }
        }
    }
}

enum End {
    Finished,
    Failed(ProxyError),
    Disconnected,
}

impl End {
    fn after_write(reported: Option<Self>, delivered: bool) -> Option<Self> {
        if delivered || matches!(reported, Some(Self::Failed(_))) {
            reported
        } else {
            Some(Self::Disconnected)
        }
    }
}

struct Task {
    id: String,
    model: String,
    started: Instant,
    dial_started: Option<Instant>,
    duration: f64,
    upstream_confirmed_id: bool,
    capture: Option<(aisix_obs::CapturedContent, usize)>,
    audit: crate::usage_attr::GuardrailAudit,
    hits: Vec<aisix_core::GuardrailMonitorHit>,
}

impl Task {
    async fn run(
        &mut self,
        state: &ProxyState,
        auth: &mut AuthenticatedKey,
        client: &mut ClientContext,
        ws: &mut WebSocket,
        run: &mut Value,
    ) -> Result<End, ProxyError> {
        // A reused socket can outlive its caller's key or model grants.
        let token =
            crate::auth::header_credential(&client.headers).ok_or(ProxyError::InvalidApiKey)??;
        *auth = crate::auth::authenticate_token(
            state,
            &token,
            crate::auth::DenialContext {
                method: "GET",
                path: PATH,
                request_id: &client.request_id,
                source_ip: crate::auth::LazySourceIp::Ready(&client.source_ip),
            },
        )
        .await?;
        client.caller = auth.caller();
        let snap = state.snapshot.load();
        let entry = crate::model_resolve::resolve_model(&snap, &self.model)
            .ok_or_else(|| ProxyError::ModelNotFound(self.model.clone()))?;
        if !auth.key().can_access(&snap, &self.model) {
            return Err(ProxyError::ModelForbidden(self.model.clone()));
        }
        let model = &entry.value;
        crate::dispatch::check_ip_access(model, &client.source_ip)?;
        let provider = crate::dispatch::require_provider(model)?;
        if !["alibaba", "alibaba-cn"]
            .iter()
            .any(|p| provider.eq_ignore_ascii_case(p))
            || model.is_embedding()
        {
            return Err(invalid("Fun-ASR requires a direct Alibaba model"));
        }
        let pk = crate::dispatch::resolve_provider_key(&snap, model)?;
        if pk.value.upstream_connection().is_some() {
            return Err(ProxyError::Bridge(BridgeError::InvalidUpstreamConfig(
                "Fun-ASR WebSocket does not support provider key resolve_addresses or custom TLS settings".into(),
            )));
        }
        let secret = crate::dispatch::require_api_key(&pk.value, model)?;
        let upstream_model = crate::dispatch::require_upstream_model(model)?;
        let mut url = url::Url::parse(&to_ws_scheme(&crate::dispatch::resolve_base_url(
            &pk.value,
        )?)?)
        .map_err(|_| invalid("invalid DashScope API base URL"))?;
        let path = url.path().trim_end_matches('/');
        if !path.ends_with(PATH) {
            let root = crate::dispatch::dashscope_native_root(path);
            url.set_path(&format!("{root}{PATH}"));
        }
        url.set_fragment(None);
        let mut request = url.as_str().into_client_request().map_err(upstream_error)?;
        request.headers_mut().insert(
            "authorization",
            format!("Bearer {secret}")
                .parse()
                .map_err(|_| invalid("invalid provider credential"))?,
        );
        let header_ctx =
            crate::dispatch::upstream_header_ctx(&pk.value, &pk.id, model, &entry.id, client)
                .with_surface_blocked(REALTIME_HANDSHAKE_SLOTS);
        for (name, value) in aisix_gateway::resolve_default_headers(&header_ctx) {
            if !REALTIME_HANDSHAKE_SLOTS.contains(&name.as_str()) && value.to_str().is_ok() {
                request.headers_mut().entry(name).or_insert(value);
            }
        }
        let mut forwarded = aisix_gateway::ForwardedClientHeaders::resolve(&header_ctx);
        forwarded.drop_non_ascii_values();
        forwarded.apply(request.headers_mut());
        let chain = state
            .guardrail_index
            .resolve(&aisix_guardrails::RequestContext {
                passthrough_route_id: "",
                model_id: &entry.id,
                mcp_server_id: "",
                api_key_id: &auth.entry.id,
                team_id: auth.key().team_id.as_deref(),
            });
        self.audit = chain.audit_log();
        run["payload"]["model"] = json!(upstream_model);
        let start = run.to_string();
        self.scan(&chain, &start, None, true).await?;
        crate::budget::gate(state, &snap).check_provider_key(&pk.id)?;
        let reservation = crate::quota::enforce(
            state,
            &snap,
            auth,
            Some(&crate::quota::ModelRateLimit::from_model(
                &self.model,
                &entry.id,
                model,
            )),
        )
        .await?;
        let idle = crate::routing::effective_timeouts(model, None, state.default_timeouts).stream;
        drop(snap);
        let dial_started = Instant::now();
        self.dial_started = Some(dial_started);
        crate::attribution::note_upstream_wait_begin();
        forwarded.note_auth_source("authorization");
        let dial = connect_upstream(request);
        tokio::pin!(dial);
        let connected = loop {
            tokio::select! {
                result = &mut dial => break result,
                frame = ws.recv() => match frame {
                    Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                    None | Some(Err(_) | Ok(Message::Close(_))) => return Ok(End::Disconnected),
                    _ => return Err(invalid("expected task-started before audio or finish-task")),
                }
            }
        };
        let result = match connected {
            Ok((mut upstream, response)) => {
                aisix_gateway::note_upstream_response(response.headers());
                bounded(idle, None, upstream.send(UpMessage::Text(start)))
                    .await?
                    .map_err(upstream_error)?;
                let result = self.relay(ws, &mut upstream, idle, &chain).await;
                let _ = tokio::time::timeout(super::UPSTREAM_CLOSE_ACK, upstream.close(None)).await;
                result
            }
            Err(failure) => {
                let (error, refused) = connect_failure(
                    failure,
                    dial_started.elapsed(),
                    idle,
                    &client.request_id,
                    async {
                        while let Some(Ok(frame)) = ws.recv().await {
                            if matches!(frame, Message::Close(_)) {
                                break;
                            }
                        }
                    },
                )
                .await;
                if let Some(refused) = refused {
                    aisix_gateway::note_upstream_response(&refused.headers);
                    aisix_gateway::note_upstream_error_body(&refused.body);
                }
                Err(ProxyError::Bridge(error))
            }
        };
        reservation
            .commit_tokens(aisix_ratelimit::TokenCharge::reported(0))
            .await;
        result
    }

    async fn scan(
        &mut self,
        chain: &aisix_guardrails::GuardrailChain,
        frame: &str,
        transcript: Option<&str>,
        input: bool,
    ) -> Result<(), ProxyError> {
        if !chain.is_empty() {
            let decoded;
            let text = match transcript {
                Some(text) => text,
                None => {
                    decoded = crate::json_splice::collect_string_values(frame.as_bytes())
                        .map_err(|_| invalid("task frame exceeds guardrail scan limits"))?;
                    &decoded
                }
            };
            if let Some((_, err)) =
                guardrail_block_event(chain, &self.model, text, input, &mut self.hits).await
            {
                return Err(err);
            }
        }
        Ok(())
    }

    async fn relay(
        &mut self,
        ws: &mut WebSocket,
        upstream: &mut tokio_tungstenite::WebSocketStream<HandedBack>,
        idle: Option<Duration>,
        chain: &aisix_guardrails::GuardrailChain,
    ) -> Result<End, ProxyError> {
        let mut started = false;
        let mut finish_sent = None;
        loop {
            let next = bounded(idle, finish_sent, async {
                tokio::select! {
                    frame = ws.recv() => super::Dir::FromClient(frame),
                    frame = upstream.next() => super::Dir::FromUpstream(frame),
                }
            })
            .await
            .map_err(|err| {
                if started && finish_sent.is_none() {
                    invalid("DashScope task timed out waiting for client audio or finish-task")
                } else {
                    err
                }
            })?;
            match next {
                super::Dir::FromClient(Some(Ok(frame))) => {
                    let outgoing = match frame {
                        Message::Binary(audio) if started && finish_sent.is_none() => {
                            UpMessage::Binary(audio)
                        }
                        Message::Text(text) => {
                            let command: Value = serde_json::from_str(&text)
                                .map_err(|_| invalid("invalid task command"))?;
                            if !started
                                || finish_sent.is_some()
                                || command.pointer("/header/task_id").and_then(Value::as_str)
                                    != Some(&self.id)
                                || command.pointer("/header/action") != Some(&json!("finish-task"))
                            {
                                return Err(invalid("expected finish-task for the active task"));
                            }
                            self.scan(chain, &text, None, true).await?;
                            UpMessage::Text(text)
                        }
                        Message::Ping(_) | Message::Pong(_) => continue,
                        Message::Close(_) => return Ok(End::Disconnected),
                        _ => {
                            return Err(invalid(
                                "audio must follow task-started and precede finish-task",
                            ))
                        }
                    };
                    let is_finish = matches!(outgoing, UpMessage::Text(_));
                    bounded(idle, None, upstream.send(outgoing))
                        .await?
                        .map_err(upstream_error)?;
                    if is_finish {
                        finish_sent = Some(Instant::now());
                    }
                }
                super::Dir::FromClient(_) => return Ok(End::Disconnected),
                super::Dir::FromUpstream(Some(Ok(UpMessage::Text(text)))) => {
                    let event: Value = serde_json::from_str(&text)
                        .map_err(|_| upstream_error("invalid DashScope event"))?;
                    if event.pointer("/header/task_id").and_then(Value::as_str) != Some(&self.id) {
                        return Err(upstream_error(
                            "upstream event did not match the active task",
                        ));
                    }
                    self.upstream_confirmed_id = true;
                    let kind = event.pointer("/header/event").and_then(Value::as_str);
                    if let Some(duration) = event
                        .pointer("/payload/usage/duration")
                        .and_then(Value::as_f64)
                        .filter(|d| *d >= 0.0)
                    {
                        // DashScope reports cumulative seconds on result-generated frames.
                        self.duration = duration;
                    }
                    let transcript = if kind == Some("result-generated") {
                        event
                            .pointer("/payload/output/sentence/text")
                            .and_then(Value::as_str)
                    } else {
                        None
                    };
                    self.scan(chain, &text, transcript, false).await?;
                    let end = match kind {
                        Some("task-started") => {
                            started = true;
                            None
                        }
                        Some("task-finished") => Some(End::Finished),
                        Some("task-failed") => {
                            let code = event
                                .pointer("/header/error_code")
                                .and_then(Value::as_str)
                                .unwrap_or("task-failed");
                            let message = event
                                .pointer("/header/error_message")
                                .and_then(Value::as_str)
                                .unwrap_or("DashScope task failed");
                            Some(End::Failed(ProxyError::Bridge(
                                BridgeError::UpstreamInBand {
                                    status: match code {
                                        "BadRequest"
                                        | "CLIENT_ERROR"
                                        | "InvalidParameter"
                                        | "NO_INPUT_AUDIO_ERROR"
                                        | "NO_VALID_AUDIO_ERROR" => Some(400),
                                        _ => None,
                                    },
                                    message: format!("{code}: {message}"),
                                    parsed: None,
                                    wire: aisix_gateway::UpstreamWire::Unknown,
                                },
                            )))
                        }
                        _ => None,
                    };
                    if matches!(end, Some(End::Failed(_))) {
                        self.capture = self
                            .capture
                            .take()
                            .map(|(content, cap)| (content.with_response(&text, cap), cap));
                    } else if let Some((content, cap)) = &mut self.capture {
                        content.append_response_frame(&text, *cap);
                    }
                    let write = bounded(idle, None, ws.send(Message::Text(text))).await;
                    if let Some(end) = End::after_write(end, matches!(write, Ok(Ok(())))) {
                        return Ok(end);
                    }
                }
                super::Dir::FromUpstream(Some(Ok(UpMessage::Ping(_) | UpMessage::Pong(_)))) => {
                    continue
                }
                super::Dir::FromUpstream(Some(Err(err))) => return Err(upstream_error(err)),
                super::Dir::FromUpstream(_) => {
                    return Err(upstream_error("DashScope closed before the task ended"))
                }
            }
        }
    }

    fn record(
        &mut self,
        state: &ProxyState,
        auth: &AuthenticatedKey,
        client: &ClientContext,
        result: &Result<End, ProxyError>,
    ) {
        use crate::{request_metrics as metrics, usage_attr as usage};
        use aisix_obs::ErrorSource;
        let error = match result {
            Ok(End::Failed(err)) | Err(err) => Some(err),
            _ => None,
        };
        let status = match result {
            Ok(End::Finished) => 200,
            Ok(End::Disconnected) => crate::CLIENT_CLOSED_REQUEST,
            Ok(End::Failed(err)) | Err(err) => err.status().as_u16(),
        };
        let source = error.map_or(ErrorSource::None, crate::attempt::error_source);
        if self.dial_started.is_some() {
            crate::attribution::note_upstream_wait_end(result.is_ok());
        }
        if let Some((content, _)) = &mut self.capture {
            if matches!(result, Err(_) | Ok(End::Disconnected)) {
                content.response.clear();
                content.response_truncated = false;
                content.truncated = content.prompt_truncated;
            }
            if let Err(err) = result {
                crate::attribution::note_client_error_body(failure_frame(&self.id, err));
            }
        }
        let elapsed = self.started.elapsed();
        let snap = state.snapshot.load();
        let target = crate::attribution::current().unwrap_or_default();
        let pk = usage::ResolvedPk::resolve(&snap, &target.provider_key_id);
        let model_label = usage::usage_event_model_label(&snap, &self.model);
        let model_id = crate::attribution::unattempted_model_id(&self.model);
        if self.dial_started.is_some() {
            let deployment_status = if source == ErrorSource::Upstream {
                status
            } else {
                200
            };
            if let Some(ProxyError::Bridge(error)) = error {
                let model = snap.models.get_by_id(&model_id);
                if let Some((ttl, reason)) = crate::cooldown::decide_cooldown(
                    error,
                    model.as_ref().and_then(|e| e.value.cooldown.as_ref()),
                ) {
                    state.runtime_status.mark_cooldown(&model_id, ttl, reason);
                }
            }
            crate::attempt::RoutingTelemetry::record_deployment(
                state,
                &model_id,
                deployment_status,
            );
        }
        metrics::record(
            state,
            PATH,
            metrics::Caller::new(auth),
            metrics::LastTarget::new(&snap, &target).upstream(&model_label, true, false),
            status,
            source,
            elapsed,
        );
        let mut event = aisix_obs::UsageEvent {
            request_id: client.request_id.clone(),
            occurred_at: aisix_obs::UsageEvent::occurred_at_now(),
            provider_request_id: if self.upstream_confirmed_id {
                self.id.clone()
            } else {
                String::new()
            },
            requested_model: self.model.clone(),
            model_id,
            requested_model_id: target.requested_model_id,
            api_key_id: auth.entry.id.clone(),
            status_code: status,
            inbound_protocol: "dashscope".into(),
            audio_duration_seconds: self.duration,
            downstream_latency_ms: crate::attempt::measured_ms(elapsed),
            upstream_latency_ms: self
                .dial_started
                .map_or(0, |t| crate::attempt::measured_ms(t.elapsed())),
            client_source_ip: client.source_ip.clone(),
            client_user_agent: client.user_agent.value.clone(),
            request_headers: client.request_headers.clone(),
            error_class: error
                .map(crate::attempt::error_class)
                .unwrap_or_default()
                .into(),
            error_message: error
                .map(crate::attempt::gateway_error_message)
                .unwrap_or_default(),
            guardrail_blocked: result.as_ref().is_err_and(|e| e.is_guardrail_block()),
            applied_guardrails: usage::applied_guardrails(&self.audit),
            guardrail_monitor_hits: self.hits.clone(),
            guardrail_scores: usage::guardrail_scores(&self.audit),
            guardrail_enforced_hits: usage::enforced_hits(&self.audit),
            guardrail_bypassed_reason: usage::bypass_reason(&self.audit),
            ..Default::default()
        };
        usage::apply_pk_telemetry(&mut event, &pk);
        usage::apply_caller_identity(&mut event, auth.jwt.as_ref(), &auth.caller());
        usage::emit_usage(
            state,
            &snap,
            crate::operation::TRANSCRIPTION,
            event,
            usage::usage_attribution(&model_label, &pk),
            self.capture.as_ref().map(|(content, _)| content),
            client.trace.as_ref(),
            true,
            self.dial_started.is_some(),
        );
        let upstream_id = crate::attribution::terminal_provider_request_id();
        super::emit_access_log(
            PATH,
            &axum::http::Method::GET,
            status,
            elapsed,
            &client.request_id,
            Some(&auth.entry.id),
            Some((&target.provider, &self.model)),
            error,
            None,
            (!upstream_id.is_empty()).then_some(upstream_id.as_str()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undelivered_terminal_events_preserve_provider_failures() {
        let failure = End::Failed(upstream_error("recognition failed"));
        assert!(matches!(
            End::after_write(Some(failure), false),
            Some(End::Failed(_))
        ));
        assert!(matches!(
            End::after_write(Some(End::Finished), false),
            Some(End::Disconnected)
        ));
    }
}
