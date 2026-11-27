//! Gemini generateContent wire protocol (including Antigravity enterprise gateway URLs).
pub(crate) mod native;
mod request;
mod response;

use aisix_gateway::{BridgeError, ChatChunk, ChatFormat, ChatResponse};
use aisix_obs::{CapturedContent, OutputCollector, OutputWire};
use axum::response::sse::Event;
use axum::{
    extract::{FromRequestParts, State},
    http::Method,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

use crate::{auth::AuthenticatedKey, client_ip::ClientContext, ProxyError, ProxyState};

pub(crate) const ENDPOINT: &str = "/v1beta/models/:model:generateContent";
pub(crate) const STREAM_ENDPOINT: &str = "/v1beta/models/:model:streamGenerateContent";

pub(crate) fn unsupported_target() -> ProxyError {
    ProxyError::InvalidRequest("this model has no Gemini or OpenAI Chat Completions target".into())
}

pub(crate) fn path(path: &str) -> Option<(&str, bool)> {
    let model_action = if let Some(short) = path.strip_prefix("/v1beta/models/") {
        short
    } else {
        let mut segments = path.strip_prefix("/v1beta1/projects/")?.splitn(7, '/');
        if segments.next()?.is_empty()
            || segments.next()? != "locations"
            || segments.next()?.is_empty()
            || segments.next()? != "publishers"
            || segments.next()? != "google"
            || segments.next()? != "models"
        {
            return None;
        }
        segments.next()?
    };
    let (model, method) = model_action.rsplit_once(':')?;
    if model.is_empty() {
        return None;
    }
    match method {
        "generateContent" => Some((model, false)),
        "streamGenerateContent" => Some((model, true)),
        _ => None,
    }
}

pub(crate) async fn entry(
    State(state): State<ProxyState>,
    req: axum::extract::Request,
) -> Response {
    use axum::extract::FromRequest;
    let (mut parts, raw_body) = req.into_parts();
    let started = std::time::Instant::now();
    let route = path(parts.uri.path()).map(|(m, s)| (m.to_owned(), s));
    let Some((model, stream)) = route.filter(|_| parts.method == Method::POST) else {
        let client = ClientContext::from_request_parts(&mut parts, &state)
            .await
            .unwrap();
        return crate::passthrough_route::entry(
            State(state),
            client,
            axum::extract::Request::from_parts(parts, raw_body),
        )
        .await;
    };
    let auth = match AuthenticatedKey::from_request_parts(&mut parts, &state).await {
        Ok(auth) => auth,
        Err(e) => return e.gemini_response(),
    };
    let client = ClientContext::from_request_parts(&mut parts, &state)
        .await
        .unwrap();
    let reject = |err| {
        crate::reject::reject_before_dispatch(
            &state,
            "POST",
            client.endpoint,
            &client.request_id,
            Some(&auth.entry.id),
            started,
            crate::reject::Envelope::Gemini,
            err,
        )
    };
    if stream
        && !url::form_urlencoded::parse(parts.uri.query().unwrap_or_default().as_bytes())
            .filter_map(|(name, value)| (name == "alt").then_some(value))
            .eq(["sse"])
    {
        return reject(ProxyError::InvalidRequest(
            "streamGenerateContent requires alt=sse".into(),
        ));
    }
    let body = match Json::<Value>::from_request(
        axum::extract::Request::from_parts(parts, raw_body),
        &state,
    )
    .await
    {
        Ok(Json(body)) => body,
        Err(e) => {
            return reject(crate::error::proxy_error_from_json_rejection(
                e,
                state.request_body_limit_bytes,
            ))
        }
    };
    let model = match percent_encoding::percent_decode_str(&model).decode_utf8() {
        Ok(m) => m,
        Err(_) => {
            return reject(ProxyError::InvalidRequest(
                "invalid UTF-8 model name".into(),
            ))
        }
    };
    let chat = match request::translate(&model, stream, &body) {
        Ok(r) => r,
        Err(e) => return reject(e),
    };
    crate::chat::serve(&state, auth, client, chat, Some(body), started).await
}

pub(crate) fn google_status(code: u16) -> &'static str {
    match code {
        400 | 413 | 415 | 422 => "INVALID_ARGUMENT",
        401 => "UNAUTHENTICATED",
        403 => "PERMISSION_DENIED",
        404 => "NOT_FOUND",
        408 | 504 => "DEADLINE_EXCEEDED",
        409 => "ABORTED",
        429 => "RESOURCE_EXHAUSTED",
        499 => "CANCELLED",
        501 => "UNIMPLEMENTED",
        502 | 503 => "UNAVAILABLE",
        _ => "INTERNAL",
    }
}

pub(crate) fn error(code: u16, message: &str) -> Value {
    json!({"error":{"code":code,"status":google_status(code),"message":message}})
}

pub(crate) fn native_error(err: &BridgeError) -> Option<&Value> {
    match err {
        BridgeError::UpstreamStatus { parsed, .. } | BridgeError::UpstreamInBand { parsed, .. } => {
            parsed.as_ref()?.native_body.as_ref()
        }
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Wire {
    OpenAi,
    Converted { audio_mime: Option<&'static str> },
    Native,
}

impl Wire {
    pub(crate) fn for_target(client: &ClientContext, native: bool) -> Self {
        if !matches!(client.endpoint, ENDPOINT | STREAM_ENDPOINT) {
            Self::OpenAi
        } else if native {
            Self::Native
        } else {
            Self::Converted { audio_mime: None }
        }
    }

    pub(crate) fn with_chat_audio(self, ctx: &aisix_gateway::BridgeContext) -> Self {
        match self {
            Self::Converted { .. } => Self::Converted {
                audio_mime: ctx.chat_audio_mime.get().copied(),
            },
            wire => wire,
        }
    }

    pub(crate) fn unary(
        self,
        now: i64,
        resp: ChatResponse,
        req: &ChatFormat,
        content_cap: Option<u32>,
    ) -> (Response, Option<CapturedContent>) {
        let capture = content_cap.map(|cap| {
            let content = CapturedContent::new(
                &crate::chat::captured_request_json(req),
                &resp.message.content_str(),
                cap as usize,
            );
            (content, cap)
        });
        let body = match self {
            Self::OpenAi => {
                let content = capture.map(|(c, cap)| {
                    c.with_output(crate::chat::chat_message_output(&resp.message, cap))
                });
                return (
                    Json(crate::render::render_response(now, resp, &req.model)).into_response(),
                    content,
                );
            }
            Self::Native => native::render_response(&resp),
            Self::Converted { audio_mime } => {
                let openai = crate::render::render_response(now, resp, &req.model);
                response::unary(
                    serde_json::to_value(openai).expect("serializable chat completion"),
                    audio_mime,
                )
            }
        };
        let content = capture.map(|(c, cap)| {
            c.with_output(Some(OutputCollector::from_body(
                OutputWire::Gemini,
                cap as usize,
                &body,
            )))
        });
        (Json(body).into_response(), content)
    }
}

#[derive(Default)]
pub(crate) struct Encoder {
    converted: response::Stream,
    native: native::ChunkRenderer,
}

impl Encoder {
    pub(crate) fn chunk(
        &mut self,
        wire: Wire,
        now: i64,
        chunk: ChatChunk,
        alias: &str,
    ) -> Option<Event> {
        if wire == Wire::Native {
            return self
                .native
                .render(&chunk)
                .map(|v| Event::default().data(v.to_string()));
        }
        let openai = crate::render::render_chunk(now, chunk, alias);
        let Wire::Converted { audio_mime } = wire else {
            return Some(
                Event::default()
                    .json_data(openai)
                    .expect("serializable chat chunk"),
            );
        };
        self.converted
            .chunk(
                serde_json::to_value(openai).expect("serializable chat chunk"),
                audio_mime,
            )
            .map(|v| Event::default().data(v.to_string()))
    }

    pub(crate) fn finish(&mut self, wire: Wire) -> Option<Event> {
        match wire {
            Wire::Native => None,
            Wire::OpenAi => Some(Event::default().data("[DONE]")),
            Wire::Converted { .. } => self
                .converted
                .finish()
                .map(|v| Event::default().data(v.to_string())),
        }
    }
}

pub(crate) fn guardrail_event(wire: Wire, name: Option<&str>, unavailable: Option<&str>) -> Event {
    stream_error(
        wire,
        422,
        crate::error::guardrail_block_frame_payload(name, unavailable),
        None,
    )
}

pub(crate) fn stream_error(
    wire: Wire,
    code: u16,
    payload: String,
    native: Option<&Value>,
) -> Event {
    if wire == Wire::OpenAi {
        if aisix_gateway::wants_upstream_error_body() {
            crate::attribution::note_client_error_body(format!(
                "event: error\ndata: {payload}\n\n"
            ));
        }
        return Event::default().event("error").data(payload);
    }
    let body = native.cloned().unwrap_or_else(|| {
        let v = serde_json::from_str::<Value>(&payload).unwrap_or_default();
        error(
            code,
            v.pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("Stream failed"),
        )
    });
    let body = body.to_string();
    if aisix_gateway::wants_upstream_error_body() {
        crate::attribution::note_client_error_body(format!("data: {body}\n\n"));
    }
    Event::default().data(body)
}

pub(crate) use request::{has_execution, policy_input, redact, translate, with_model_effort};
