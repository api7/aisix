//! Google AI Studio's generateContent wire, also used by streamGenerateContent.
//! <https://ai.google.dev/api/generate-content>

use std::{collections::HashMap, time::Instant};

use aisix_core::ApiSurface;
use aisix_gateway::{
    chat::{gemini_header_service_tier, ChatChoice},
    gemini::{finish_with_tool_calls, map_gemini_finish_reason, GeminiUsageMetadata},
    BridgeContext, BridgeError, ChatChunk, ChatChunkStream, ChatDelta, ChatMessage, ChatResponse,
    FinishReason, ServiceTier, SseDecoder, SseEvent, UpstreamErrorView, UpstreamWire, UsageStats,
};
use axum::http::{header, HeaderMap, HeaderValue};
use futures::StreamExt;
use serde_json::{json, Value};
use url::Url;

const FRAME: &str = "_aisix_gemini_frame";
const CANDIDATE: &str = "_aisix_gemini_candidate";

fn endpoint(ctx: &BridgeContext, streaming: bool) -> Result<Url, BridgeError> {
    let base = crate::dispatch::resolve_base_url_for(&ctx.provider_key, ApiSurface::Gemini)
        .map_err(|e| BridgeError::InvalidUpstreamConfig(e.to_string()))?;
    let model = ctx.model.upstream_model().ok_or_else(|| {
        BridgeError::InvalidUpstreamConfig("Gemini requires an upstream model name".into())
    })?;
    let method = if streaming {
        "streamGenerateContent"
    } else {
        "generateContent"
    };
    let mut url = Url::parse(&base)
        .map_err(|e| BridgeError::InvalidUpstreamConfig(format!("invalid Gemini base URL: {e}")))?;
    url.path_segments_mut()
        .map_err(|_| BridgeError::InvalidUpstreamConfig("invalid Gemini base URL".into()))?
        .pop_if_empty()
        .push("models")
        .push(&format!("{model}:{method}"));
    if streaming {
        url.query_pairs_mut().append_pair("alt", "sse");
    }
    Ok(url)
}

fn error_view(body: &[u8]) -> Option<UpstreamErrorView> {
    let raw: Value = serde_json::from_slice(body).ok()?;
    let e = raw.get("error")?.as_object()?;
    Some(UpstreamErrorView {
        kind: e.get("status").and_then(Value::as_str).map(str::to_owned),
        message: e.get("message").and_then(Value::as_str).map(str::to_owned),
        native_body: Some(raw),
        ..Default::default()
    })
}

async fn send(
    ctx: &BridgeContext,
    body: &Value,
    streaming: bool,
) -> Result<reqwest::Response, BridgeError> {
    let pk = &ctx.provider_key;
    let mut headers = HeaderMap::new();
    let key = crate::dispatch::require_api_key(pk, &ctx.model)
        .map_err(|e| BridgeError::InvalidUpstreamConfig(e.to_string()))?;
    let mut key = HeaderValue::from_str(key)
        .map_err(|_| BridgeError::InvalidUpstreamConfig("invalid Gemini API key header".into()))?;
    key.set_sensitive(true);
    headers.insert("x-goog-api-key", key);
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    headers.insert(
        "x-aisix-request-id",
        HeaderValue::from_str(&ctx.request_id)
            .map_err(|e| BridgeError::Config(format!("invalid request id: {e}")))?,
    );
    aisix_gateway::apply_request_headers(
        &mut headers,
        &ctx.header_ctx()
            .with_surface_blocked(aisix_core::CREDENTIAL_SLOT_HEADERS),
    );

    let client = crate::http_client::client_for(pk.upstream_connection().as_ref());
    let bytes = aisix_gateway::json_body(body).map_err(|e| BridgeError::Config(e.to_string()))?;
    let bytes = crate::decisions::apply_body_overrides(pk, bytes)
        .map_err(|e| BridgeError::Config(e.to_string()))?;
    let mut request = client
        .post(endpoint(ctx, streaming)?)
        .headers(headers)
        .body(bytes);
    if !streaming {
        if let Some(deadline) = ctx.deadline {
            request = request.timeout(deadline);
        }
    }
    let connect = if streaming { ctx.deadline } else { None };
    let resp = crate::stream_timeout::send_with_deadline(request, connect, Instant::now()).await?;
    if resp.status().is_success() {
        return Ok(resp);
    }
    Err(aisix_gateway::upstream_error_within(connect, resp, |resp| {
        aisix_gateway::upstream_status_parsed_from_response(resp, UpstreamWire::Vertex, error_view)
    })
    .await)
}

fn in_band_error(raw: &Value) -> Option<BridgeError> {
    let mut err = aisix_gateway::capture_in_band_error_value(raw, UpstreamWire::Vertex)?;
    if let BridgeError::UpstreamInBand {
        parsed: Some(view), ..
    } = &mut err
    {
        view.native_body = Some(raw.clone());
    }
    Some(err)
}

fn usage(raw: &Value) -> Result<(Option<UsageStats>, Option<ServiceTier>), BridgeError> {
    let Some(meta) = raw.get("usageMetadata").filter(|v| !v.is_null()) else {
        return Ok((None, None));
    };
    let meta: GeminiUsageMetadata = serde::Deserialize::deserialize(meta)
        .map_err(|e| BridgeError::UpstreamDecode(format!("Gemini usageMetadata: {e}")))?;
    let tier = meta.service_tier();
    Ok((meta.reports_counts().then(|| meta.into_usage_stats()), tier))
}

// The policy code owns the text and tool arguments from here on. Keep the
// rest of each part, including its signature, and only the original text
// lengths so rendering can restore its boundaries after a policy rewrite.
fn project(candidate: &mut Value) -> ChatMessage {
    let mut message = ChatMessage::assistant("");
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    if let Some(parts) = candidate
        .pointer_mut("/content/parts")
        .and_then(Value::as_array_mut)
    {
        for part in parts {
            let thought = part["thought"] == true;
            if let Some(slot) = part.get_mut("text").filter(|v| v.is_string()) {
                let Value::String(s) = slot.take() else {
                    unreachable!()
                };
                if thought {
                    reasoning.push_str(&s);
                } else {
                    text.push_str(&s);
                }
                *slot = Value::from(s.len());
            }
            if let Some(call) = part.get_mut("functionCall") {
                let args = call.get_mut("args").map(Value::take);
                calls.push(json!({"type":"function", "function":{
                    "name":call.get("name"), "arguments":args.as_ref().map(Value::to_string)
                }}));
            }
        }
    }
    message.content = (!text.is_empty()).then_some(text);
    if !reasoning.is_empty() {
        message
            .extra
            .insert("reasoning_content".into(), reasoning.into());
    }
    if !calls.is_empty() {
        message.extra.insert("tool_calls".into(), calls.into());
    }
    message
}

fn restore<'a>(candidate: &mut Value, mut text: &'a str, mut reasoning: &'a str, calls: &[Value]) {
    let Some(parts) = candidate
        .pointer_mut("/content/parts")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    let mut calls = calls.iter();
    for part in parts {
        let remaining = if part["thought"] == true {
            &mut reasoning
        } else {
            &mut text
        };
        if let Some(slot) = part.get_mut("text") {
            if let Some(len) = slot.as_u64() {
                let (head, tail) = remaining
                    .split_at_checked(len as usize)
                    .unwrap_or((remaining, ""));
                *slot = Value::from(head);
                *remaining = tail;
            }
        }
        if let Some(call) = part.get_mut("functionCall") {
            let projected = calls.next();
            if let Some(args) = call.get_mut("args") {
                if let Some(s) = projected
                    .and_then(|v| v.pointer("/function/arguments"))
                    .and_then(Value::as_str)
                {
                    *args = serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.to_owned()));
                }
            }
        }
    }
    // A masking policy can lengthen the text. It still belongs in the
    // last text part of its channel, including when a frame has tools too.
    for (thought, suffix) in [(false, text), (true, reasoning)] {
        if suffix.is_empty() {
            continue;
        }
        if let Some(part) = candidate["content"]["parts"].as_array_mut().and_then(|p| {
            p.iter_mut()
                .rev()
                .find(|p| (p["thought"] == true) == thought && p["text"].is_string())
        }) {
            if let Value::String(text) = &mut part["text"] {
                text.push_str(suffix);
            }
        }
    }
}

fn finish(candidate: &Value, has_calls: bool) -> FinishReason {
    let reason = candidate["finishReason"].as_str();
    if has_calls {
        finish_with_tool_calls(reason)
    } else {
        map_gemini_finish_reason(reason)
    }
}

pub(crate) async fn chat(ctx: &BridgeContext, body: &Value) -> Result<ChatResponse, BridgeError> {
    let resp = send(ctx, body, false).await?;
    let header_tier = gemini_header_service_tier(resp.headers());
    let raw: Value = resp
        .json()
        .await
        .map_err(|e| aisix_gateway::upstream_body_read_error(&e))?;
    response(raw).map(|r| r.with_fallback_service_tier(header_tier))
}

fn response(mut raw: Value) -> Result<ChatResponse, BridgeError> {
    if let Some(e) = in_band_error(&raw) {
        return Err(e);
    }
    let (usage, service_tier) = usage(&raw)?;
    let mut choices = raw
        .get_mut("candidates")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(position, candidate)| {
            let message = project(candidate);
            ChatChoice {
                index: candidate["index"].as_u64().unwrap_or(position as u64) as u32,
                finish_reason: finish(candidate, message.extra.contains_key("tool_calls")),
                message,
                logprobs: None,
            }
        });
    let first = choices.next();
    let more_choices = choices.collect();
    let (mut message, finish_reason) =
        first
            .map(|c| (c.message, c.finish_reason))
            .unwrap_or_else(|| {
                (
                    ChatMessage::assistant(""),
                    if raw.pointer("/promptFeedback/blockReason").is_some() {
                        FinishReason::ContentFilter
                    } else {
                        FinishReason::Stop
                    },
                )
            });
    let id = raw["responseId"].as_str().unwrap_or_default().to_owned();
    let model = raw["modelVersion"].as_str().unwrap_or_default().to_owned();
    message.extra.insert(FRAME.into(), raw);
    Ok(ChatResponse {
        id,
        model,
        message,
        more_choices,
        finish_reason,
        logprobs: None,
        usage: usage.unwrap_or_default(),
        service_tier,
    })
}

pub(crate) fn render_response(resp: &ChatResponse) -> Value {
    let mut frame = resp.message.extra[FRAME].clone();
    if let Some(candidates) = frame.get_mut("candidates").and_then(Value::as_array_mut) {
        for (candidate, msg) in candidates.iter_mut().zip(resp.messages()) {
            restore(
                candidate,
                msg.content.as_deref().unwrap_or_default(),
                msg.extra
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                msg.extra
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default(),
            );
        }
    }
    frame
}

pub(crate) async fn chat_stream(
    ctx: &BridgeContext,
    body: &Value,
) -> Result<ChatChunkStream, BridgeError> {
    let resp = send(ctx, body, true).await?;
    let tier = gemini_header_service_tier(resp.headers());
    let mut bytes = resp.bytes_stream();
    let stream = async_stream::try_stream! {
        let mut decoder = SseDecoder::new();
        let mut next_tool = HashMap::new();
        loop {
            let item = bytes.next().await;
            let eof = item.is_none();
            let events = match item {
                Some(item) => decoder.feed(item.map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?.as_ref()),
                None => decoder.finish().map(|e| e.into_iter().collect()),
            }.map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            for event in events {
                let SseEvent::Data(data) = event else {continue};
                let raw: Value = serde_json::from_str(&data)
                    .map_err(|e| BridgeError::UpstreamDecode(format!("Gemini SSE data: {e}")))?;
                for chunk in chunks(raw, &mut next_tool)? {
                    yield chunk.with_fallback_service_tier(&tier);
                }
            }
            if eof {break;}
        }
    };
    Ok(Box::pin(stream))
}

fn chunks(
    mut frame: Value,
    next_tool: &mut HashMap<u32, u64>,
) -> Result<Vec<ChatChunk>, BridgeError> {
    if let Some(err) = in_band_error(&frame) {
        return Err(err);
    }
    let (usage, tier) = usage(&frame)?;
    let empty = || ChatChunk {
        id: frame["responseId"].as_str().unwrap_or_default().to_owned(),
        model: frame["modelVersion"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        choice_index: 0,
        delta: ChatDelta::default(),
        finish_reason: None,
        logprobs: None,
        usage: None,
        accounting_usage: None,
        service_tier: tier.clone(),
    };
    let base = empty();
    let mut out = Vec::new();
    if let Some(candidates) = frame.get_mut("candidates").and_then(Value::as_array_mut) {
        for (position, mut candidate) in std::mem::take(candidates).into_iter().enumerate() {
            let mut chunk = base.clone();
            chunk.choice_index = candidate["index"].as_u64().unwrap_or(position as u64) as u32;
            let mut msg = project(&mut candidate);
            let calls = msg.extra.remove("tool_calls").and_then(|v| match v {
                Value::Array(calls) => Some(calls),
                _ => None,
            });
            chunk.finish_reason = candidate["finishReason"].as_str().map(|_| {
                finish(
                    &candidate,
                    calls.is_some() || next_tool.contains_key(&chunk.choice_index),
                )
            });
            chunk.delta.content = msg.content;
            chunk.delta.reasoning_content =
                msg.extra.remove("reasoning_content").and_then(|v| match v {
                    Value::String(s) => Some(s),
                    _ => None,
                });
            chunk.delta.tool_calls = calls.map(|mut calls| {
                let index = next_tool.entry(chunk.choice_index).or_default();
                for call in &mut calls {
                    call["index"] = (*index).into();
                    *index += 1;
                }
                calls
            });
            chunk.delta.extra.insert(CANDIDATE.into(), candidate);
            out.push(chunk);
        }
    }
    if out.is_empty() {
        out.push(base);
    }
    let last = out.last_mut().expect("at least one chunk per Gemini frame");
    last.usage = usage;
    last.delta.extra.insert(FRAME.into(), frame);
    Ok(out)
}

#[derive(Default)]
pub(crate) struct ChunkRenderer {
    candidates: Vec<Value>,
}

impl ChunkRenderer {
    pub(crate) fn render(&mut self, chunk: &ChatChunk) -> Option<Value> {
        let delta = &chunk.delta;
        if let Some(mut candidate) = delta.extra.get(CANDIDATE).cloned() {
            restore(
                &mut candidate,
                delta.content.as_deref().unwrap_or_default(),
                delta.reasoning_content.as_deref().unwrap_or_default(),
                delta.tool_calls.as_deref().unwrap_or_default(),
            );
            self.candidates.push(candidate);
        }
        let mut frame = delta.extra.get(FRAME)?.clone();
        if !self.candidates.is_empty() {
            frame["candidates"] = std::mem::take(&mut self.candidates).into();
        }
        Some(frame)
    }
}
