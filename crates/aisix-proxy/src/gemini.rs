//! Gemini generateContent wire protocol (including Antigravity enterprise gateway URLs).
pub(crate) mod native;
mod request;
mod response;

use aisix_gateway::{BridgeError, ChatChunk, ChatResponse};
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
        Err(e) => return e.into_gemini_response(),
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
    Converted,
    Native,
}

impl Wire {
    pub(crate) fn for_target(client: &ClientContext, native: bool) -> Self {
        if !matches!(client.endpoint, ENDPOINT | STREAM_ENDPOINT) {
            Self::OpenAi
        } else if native {
            Self::Native
        } else {
            Self::Converted
        }
    }

    pub(crate) fn unary(self, now: i64, resp: ChatResponse, alias: &str) -> Response {
        if self == Self::Native {
            return Json(native::render_response(&resp)).into_response();
        }
        let openai = crate::render::render_response(now, resp, alias);
        if self == Self::OpenAi {
            return Json(openai).into_response();
        }
        Json(response::unary(
            serde_json::to_value(openai).expect("serializable chat completion"),
        ))
        .into_response()
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
        if wire == Wire::OpenAi {
            return Some(
                Event::default()
                    .json_data(openai)
                    .expect("serializable chat chunk"),
            );
        }
        self.converted
            .chunk(serde_json::to_value(openai).expect("serializable chat chunk"))
            .map(|v| Event::default().data(v.to_string()))
    }

    pub(crate) fn finish(&mut self, wire: Wire) -> Option<Event> {
        match wire {
            Wire::Native => None,
            Wire::OpenAi => Some(Event::default().data("[DONE]")),
            Wire::Converted => self
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
    Event::default().data(body.to_string())
}

pub(crate) use request::{redact, translate, with_model_effort};
