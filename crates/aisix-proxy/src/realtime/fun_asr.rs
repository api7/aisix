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

async fn fail(ws: &mut WebSocket, id: &str, err: &ProxyError) {
    let error = err.envelope().error;
    let frame = json!({"header": {"task_id": id, "event": "task-failed",
        "error_code": error.code.unwrap_or(error.kind), "error_message": error.message}, "payload": {}});
    let _ = ws.send(Message::Text(frame.to_string())).await;
    let _ = ws.send(Message::Close(None)).await;
}

async fn bounded<F: std::future::Future>(
    limit: Option<Duration>,
    f: F,
) -> Result<F::Output, ProxyError> {
    match limit {
        Some(limit) => tokio::time::timeout(limit, f).await.map_err(|_| {
            ProxyError::Bridge(BridgeError::Timeout {
                elapsed_ms: limit.as_millis() as u64,
                cause: "DashScope WebSocket idle timeout".into(),
            })
        }),
        None => Ok(f.await),
    }
}

async fn serve(
    state: &ProxyState,
    mut auth: AuthenticatedKey,
    mut client: ClientContext,
    ws: &mut WebSocket,
) {
    let mut handshake_span = Some(tracing::Span::current());
    loop {
        let Ok(Some(Ok(frame))) = bounded(state.default_timeouts.stream, ws.recv()).await else {
            return;
        };
        let text = match frame {
            Message::Text(text) => text,
            Message::Ping(_) | Message::Pong(_) => continue,
            Message::Close(_) => return,
            Message::Binary(_) => {
                fail(ws, "", &invalid("expected run-task before audio")).await;
                return;
            }
        };
        let mut run: Value = match serde_json::from_str(&text) {
            Ok(run) => run,
            Err(_) => {
                fail(ws, "", &invalid("expected a JSON run-task command")).await;
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
        let span = handshake_span.take().unwrap_or_else(|| {
            client.request_id = crate::request_id::new_request_id();
            client.trace = client
                .trace
                .as_ref()
                .map(|trace| Arc::new(trace.next_request()));
            tracing::error_span!(parent: None, "request", request_id = %client.request_id)
        });
        let mut task = Task {
            id,
            model,
            started: Instant::now(),
            dial_started: None,
            duration: 0.0,
            audit: Default::default(),
            hits: Vec::new(),
        };
        let result = crate::attribution::detached(async {
            let result = task.run(state, &mut auth, &client, ws, &mut run).await;
            task.record(state, &auth, &client, &result);
            result
        })
        .instrument(span)
        .await;
        match result {
            Ok(End::Finished | End::Failed(_)) => continue,
            Ok(End::Disconnected) => {
                let _ = ws.send(Message::Close(None)).await;
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

struct Task {
    id: String,
    model: String,
    started: Instant,
    dial_started: Option<Instant>,
    duration: f64,
    audit: crate::usage_attr::GuardrailAudit,
    hits: Vec<aisix_core::GuardrailMonitorHit>,
}

impl Task {
    async fn run(
        &mut self,
        state: &ProxyState,
        auth: &mut AuthenticatedKey,
        client: &ClientContext,
        ws: &mut WebSocket,
        run: &mut Value,
    ) -> Result<End, ProxyError> {
        if run.pointer("/header/action") != Some(&json!("run-task"))
            || uuid::Uuid::parse_str(&self.id).is_err()
            || run.pointer("/payload/task_group") != Some(&json!("audio"))
            || run.pointer("/payload/task") != Some(&json!("asr"))
            || run.pointer("/payload/function") != Some(&json!("recognition"))
            || self.model.is_empty()
        {
            return Err(invalid(
                "expected an audio/asr/recognition run-task with task_id and model",
            ));
        }
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
        let mut forwarded = aisix_gateway::ForwardedClientHeaders::resolve(
            &aisix_gateway::UpstreamHeaderContext::from_overrides(pk.value.request.as_ref())
                .with_client_headers(&client.headers)
                .with_surface_blocked(REALTIME_HANDSHAKE_SLOTS),
        );
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
        self.scan(&chain, &start, true).await?;
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
        crate::attribution::note_upstream_wait_unmeasured();
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
            Ok((mut upstream, _)) => {
                bounded(idle, upstream.send(UpMessage::Text(start)))
                    .await?
                    .map_err(upstream_error)?;
                let result = self.relay(ws, &mut upstream, idle, &chain).await;
                let _ = tokio::time::timeout(super::UPSTREAM_CLOSE_ACK, upstream.close(None)).await;
                result
            }
            Err(failure) => {
                let (error, _) = connect_failure(
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
        text: &str,
        input: bool,
    ) -> Result<(), ProxyError> {
        if !chain.is_empty() {
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
        let mut finishing = false;
        loop {
            let next = bounded(idle, async {
                tokio::select! {
                    frame = ws.recv() => super::Dir::FromClient(frame),
                    frame = upstream.next() => super::Dir::FromUpstream(frame),
                }
            })
            .await?;
            match next {
                super::Dir::FromClient(Some(Ok(frame))) => {
                    let outgoing = match frame {
                        Message::Binary(audio) if started && !finishing => UpMessage::Binary(audio),
                        Message::Text(text) => {
                            let command: Value = serde_json::from_str(&text)
                                .map_err(|_| invalid("invalid task command"))?;
                            if !started
                                || finishing
                                || command.pointer("/header/task_id").and_then(Value::as_str)
                                    != Some(&self.id)
                                || command.pointer("/header/action") != Some(&json!("finish-task"))
                            {
                                return Err(invalid("expected finish-task for the active task"));
                            }
                            finishing = true;
                            self.scan(chain, &text, true).await?;
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
                    bounded(idle, upstream.send(outgoing))
                        .await?
                        .map_err(upstream_error)?;
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
                    let kind = event.pointer("/header/event").and_then(Value::as_str);
                    if let Some(duration) = event
                        .pointer("/payload/usage/duration")
                        .and_then(Value::as_f64)
                        .filter(|d| *d >= 0.0)
                    {
                        // DashScope reports cumulative seconds on result-generated frames.
                        self.duration = duration;
                    }
                    self.scan(chain, &text, false).await?;
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
                                    status: None,
                                    message: format!("{code}: {message}"),
                                    parsed: None,
                                    wire: aisix_gateway::UpstreamWire::Unknown,
                                },
                            )))
                        }
                        _ => None,
                    };
                    let sent = bounded(idle, ws.send(Message::Text(text))).await;
                    if let Some(end) = end {
                        return Ok(end);
                    }
                    if !matches!(sent, Ok(Ok(()))) {
                        return Ok(End::Disconnected);
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
        &self,
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
            provider_request_id: if self.dial_started.is_some() {
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
            gateway_overhead_ms: self
                .dial_started
                .map(|t| crate::attempt::measured_ms(t.duration_since(self.started))),
            upstream_latency_ms: self
                .dial_started
                .map_or(0, |t| crate::attempt::measured_ms(t.elapsed())),
            client_source_ip: client.source_ip.clone(),
            client_user_agent: client.user_agent.clone(),
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
            usage::usage_event_labels(&model_label, &pk),
            None,
            client.trace.as_ref(),
            true,
            self.dial_started.is_some(),
        );
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
            self.dial_started.map(|_| self.id.as_str()),
        );
    }
}
