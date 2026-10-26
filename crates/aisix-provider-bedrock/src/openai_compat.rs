//! Bedrock Runtime's OpenAI-compatible Chat Completions endpoint
//! (`POST /openai/v1/chat/completions` on the bedrock-runtime host).
//!
//! AWS serves Grok and OpenAI GPT 5.6 and newer here, and the request keeps
//! its OpenAI shape instead of being rewritten to Converse. A request that
//! needs something only Converse offers — a Bedrock-native block, a JSON
//! mode without a schema, or tools on a GPT model while it reasons — is still
//! served by Converse ([`needs_converse`]).
//!
//! The endpoint authenticates with the provider key's AWS credentials,
//! signed with SigV4 for the `bedrock` service.

use std::borrow::Cow;
use std::time::{Duration, Instant};

use aisix_gateway::content_part::{parse_part, ContentPart, MediaSource};
use aisix_gateway::media_fetch::{fetch_media_urls, Encoding};
use aisix_gateway::{
    BridgeContext, BridgeError, ChatChunk, ChatChunkStream, ChatFormat, ChatResponse, Role,
    SseDecoder, SseEvent,
};
use aisix_provider_openai::wire::{
    build_request, messages_from, response_into_chat_response, stream_chunk_into_chat_chunk,
    DeveloperRoleMode, OpenAiResponse, OpenAiStreamChunk, ToolCallIndexer,
};
use futures::StreamExt;
use serde_json::Value;

use crate::bridge::{
    auth_headers, capability_id, is_arn, strip_region_prefix, BedrockSecret, RequestAuth,
};

const CHAT_COMPLETIONS_PATH: &str = "/openai/v1/chat/completions";

/// The `openai.gpt-<major>[.<minor>]` version a model id names, or `None`
/// for anything else — gpt-oss included, whose name has no version.
fn gpt_version(model_id: &str) -> Option<(u32, u32)> {
    let bytes = model_id.as_bytes();
    let needle = "openai.gpt-";
    let mut from = 0;
    while let Some(found) = model_id[from..].find(needle) {
        let at = from + found;
        from = at + needle.len();
        if at > 0 && !matches!(bytes[at - 1], b'.' | b'/') {
            continue;
        }
        let rest = &model_id[from..];
        let major_len = rest.bytes().take_while(u8::is_ascii_digit).count();
        if major_len == 0 || major_len > 3 {
            continue;
        }
        let major: u32 = rest[..major_len].parse().ok()?;
        let after = &rest[major_len..];
        let minor = after
            .strip_prefix('.')
            .map(|m| {
                let len = m.bytes().take_while(u8::is_ascii_digit).count();
                (len > 0 && len <= 3)
                    .then(|| m[..len].parse::<u32>().ok())
                    .flatten()
            })
            .unwrap_or(Some(0));
        return Some((major, minor.unwrap_or(0)));
    }
    None
}

fn is_grok(model_id: &str) -> bool {
    let bytes = model_id.as_bytes();
    model_id
        .match_indices("xai.grok-")
        .any(|(at, _)| at == 0 || matches!(bytes[at - 1], b'.' | b'/'))
}

/// A GPT-5.x or GPT-6.x model id, never gpt-oss.
pub(crate) fn is_openai_gpt(model_id: &str) -> bool {
    gpt_version(model_id).is_some()
}

/// Grok, and GPT 5.6 and newer: the families AWS serves on the Chat
/// Completions endpoint.
fn chat_completions_family(model_id: &str) -> bool {
    is_grok(model_id) || gpt_version(model_id).is_some_and(|v| v >= (5, 6))
}

/// Whether a model is served by the Chat Completions endpoint by default:
/// a family above, addressed through a cross-region inference profile
/// (`global.`, `us.`, `us-gov.`, …). An in-region id and an ARN stay on
/// Converse.
pub(crate) fn is_default_route(model_id: &str) -> bool {
    !is_arn(model_id)
        && strip_region_prefix(model_id) != model_id
        && chat_completions_family(model_id)
}

/// Bedrock-native request fields only Converse forwards.
const CONVERSE_ONLY_FIELDS: &[&str] = &[
    "guardrailConfig",
    "performanceConfig",
    "serviceTier",
    "requestMetadata",
    "outputConfig",
    "thinking",
    "additionalModelRequestFields",
    "top_k",
];

/// Whether a request on a Chat Completions model must still go to Converse.
pub(crate) fn needs_converse(req: &ChatFormat, model_id: &str) -> bool {
    if CONVERSE_ONLY_FIELDS
        .iter()
        .any(|k| req.extra.get(*k).is_some_and(|v| !v.is_null()))
    {
        return true;
    }
    // AWS enforces a `json_schema` response format for both families here;
    // a `json_object` one is answered with a 400 unless the prompt mentions
    // JSON, so Converse keeps handling it.
    match req.extra.get("response_format") {
        None | Some(Value::Null) => {}
        Some(Value::Object(rf)) => match rf.get("type").and_then(Value::as_str) {
            Some("text") => {}
            Some("json_schema") if rf.contains_key("json_schema") => {}
            _ => return true,
        },
        Some(_) => {}
    }
    let has_tools = ["tools", "functions"].iter().any(|k| {
        req.extra
            .get(*k)
            .is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()))
    });
    // GPT takes function tools here only while it does not reason, which is
    // every effort but "none" and the default; Converse serves tools at any
    // effort. Grok takes them here with any effort.
    has_tools
        && !is_grok(model_id)
        && req.extra.get("reasoning_effort").and_then(Value::as_str) != Some("none")
}

/// Whether this request is served by the Chat Completions endpoint.
pub(crate) fn serves(req: &ChatFormat, model_id: &str) -> bool {
    is_default_route(model_id) && !needs_converse(req, model_id)
}

/// Shape the OpenAI body for this endpoint. Every parameter goes out as
/// the caller sent it, including ones AWS refuses for the model, so the
/// caller gets AWS's own error; only `max_tokens` is sent as
/// `max_completion_tokens`, which every model here accepts and GPT 5.6
/// requires.
fn shape_body(body: &mut Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    if let Some(max_tokens) = obj.remove("max_tokens") {
        obj.entry("max_completion_tokens").or_insert(max_tokens);
    }
}

/// A user turn's remote image URLs, inlined as `data:` URLs from what the
/// gateway downloaded for them.
///
/// The second value is the request's deadline less the time the downloads
/// took, which is what the upstream call has left.
async fn inline_remote_images<'a>(
    req: Cow<'a, ChatFormat>,
    ctx: &BridgeContext,
) -> Result<(Cow<'a, ChatFormat>, Option<Duration>), BridgeError> {
    let fetched = fetch_media_urls(&req, ctx, Encoding::Base64, |m| {
        m.kind == aisix_gateway::content_part::MediaKind::Image
    })
    .await?;
    let deadline = fetched.charge(ctx).deadline;
    let mut req = req;
    let rewrites: Vec<(usize, usize, String)> = req
        .messages
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m.role, Role::User | Role::System | Role::Developer))
        .flat_map(|(mi, m)| {
            m.content_blocks
                .iter()
                .flatten()
                .enumerate()
                .filter_map(move |(pi, block)| {
                    let Ok(ContentPart::Media(media)) = parse_part(block) else {
                        return None;
                    };
                    let MediaSource::Uri { uri, .. } = media.source else {
                        return None;
                    };
                    Some((mi, pi, uri.to_string()))
                })
        })
        .collect();
    for (mi, pi, uri) in rewrites {
        let Some(download) = fetched.take(&uri) else {
            continue;
        };
        let data_url = format!(
            "data:{};base64,{}",
            download.mime_type,
            download.data.into_base64()
        );
        if let Some(block) = req.to_mut().messages[mi]
            .content_blocks
            .as_mut()
            .and_then(|b| b.get_mut(pi))
        {
            match block.get_mut("image_url") {
                Some(Value::Object(image_url)) => {
                    image_url.insert("url".into(), Value::String(data_url));
                }
                Some(slot) => *slot = Value::String(data_url),
                None => {}
            }
        }
    }
    Ok((req, deadline))
}

/// The endpoint URL: the provider key's `api_base` when set, else the
/// region's bedrock-runtime host, with the Chat Completions path appended
/// unless the base already carries it.
fn endpoint_url(base: Option<&str>, region: &str) -> String {
    let base = match base.map(str::trim).filter(|b| !b.is_empty()) {
        Some(b) => b.trim_end_matches('/').to_string(),
        None => format!("https://bedrock-runtime.{region}.amazonaws.com"),
    };
    if base.ends_with(CHAT_COMPLETIONS_PATH) {
        base
    } else if let Some(root) = base.strip_suffix("/openai/v1") {
        format!("{root}{CHAT_COMPLETIONS_PATH}")
    } else {
        format!("{base}{CHAT_COMPLETIONS_PATH}")
    }
}

/// The request headers plus the ones that authenticate them.
fn authenticated_headers(
    auth: &RequestAuth,
    region: &str,
    url: &str,
    mut headers: Vec<(String, String)>,
    body: &[u8],
) -> Result<http::HeaderMap, BridgeError> {
    let auth = auth_headers(auth, region, url, &headers, body)?;
    headers.extend(auth);
    let mut map = http::HeaderMap::new();
    for (name, value) in headers {
        if let (Ok(n), Ok(v)) = (
            http::HeaderName::from_bytes(name.as_bytes()),
            http::HeaderValue::from_str(&value),
        ) {
            map.append(n, v);
        }
    }
    Ok(map)
}

/// Positional tool-call ids (`call_0`, `call_1`) are numbered from zero in
/// every response, so a multi-turn history would hold the same id for
/// different calls; they are replaced with unique ones. Ids that are already
/// unique pass through.
fn is_positional_tool_call_id(id: &str) -> bool {
    id.strip_prefix("call_")
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

fn mint_tool_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

/// A streamed delta names its call by `index`, so one call keeps one minted
/// id across its deltas; a whole response's calls carry no `index` and are
/// told apart by position.
fn mint_positional_ids(
    tool_calls: &mut [Value],
    minted: &mut Vec<(u64, String)>,
    by_position: bool,
) {
    for (position, call) in tool_calls.iter_mut().enumerate() {
        let Some(obj) = call.as_object_mut() else {
            continue;
        };
        let Some(id) = obj.get("id").and_then(Value::as_str) else {
            continue;
        };
        if !is_positional_tool_call_id(id) {
            continue;
        }
        let index = if by_position {
            position as u64
        } else {
            obj.get("index").and_then(Value::as_u64).unwrap_or(0)
        };
        let id = match minted.iter().find(|(i, _)| *i == index) {
            Some((_, id)) => id.clone(),
            None => {
                let id = mint_tool_call_id();
                minted.push((index, id.clone()));
                id
            }
        };
        obj.insert("id".into(), Value::String(id));
    }
}

fn request_id_header(headers: &reqwest::header::HeaderMap) -> String {
    headers
        .get("x-amzn-requestid")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

/// An error answer from the endpoint, in AWS's shape (`x-amzn-errortype`
/// plus a `message` body) or OpenAI's (`{"error": {...}}`).
async fn http_error(resp: reqwest::Response, started: Instant) -> BridgeError {
    let status = resp.status().as_u16();
    let retry_after = aisix_gateway::parse_retry_after(resp.headers());
    let errortype = resp
        .headers()
        .get("x-amzn-errortype")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body =
        aisix_gateway::read_body_capped(resp, aisix_gateway::MAX_UPSTREAM_ERROR_BODY_BYTES).await;
    let parsed: Option<Value> = serde_json::from_slice(&body).ok();
    let field = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
    let error_obj = parsed
        .as_ref()
        .and_then(|v| v.get("error"))
        .filter(|e| e.is_object());
    let message = error_obj
        .and_then(|e| field(e, "message"))
        .or_else(|| {
            parsed
                .as_ref()
                .and_then(|v| field(v, "message").or_else(|| field(v, "Message")))
        })
        .or_else(|| parsed.as_ref().and_then(|v| field(v, "error")))
        .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
    let code = error_obj.and_then(|e| field(e, "code"));
    let kind = errortype.or_else(|| error_obj.and_then(|e| field(e, "type")).or(code.clone()));
    crate::errors::upstream_error(status, kind, code, message, retry_after, started)
}

async fn with_deadline<T, F>(
    deadline: Option<Duration>,
    started: Instant,
    fut: F,
) -> Result<T, BridgeError>
where
    F: std::future::Future<Output = Result<T, BridgeError>>,
{
    match deadline {
        None => fut.await,
        Some(d) => tokio::time::timeout(d, fut).await.unwrap_or_else(|_| {
            Err(BridgeError::Timeout {
                elapsed_ms: started.elapsed().as_millis() as u64,
                cause: String::new(),
            })
        }),
    }
}

/// What a dispatch on this endpoint needs from the bridge.
pub(crate) struct Endpoint {
    pub client: reqwest::Client,
    pub creds: BedrockSecret,
    pub base: Option<String>,
    pub extra_headers: Vec<(String, String)>,
}

async fn send(
    endpoint: &Endpoint,
    req: Cow<'_, ChatFormat>,
    ctx: &BridgeContext,
    model_id: &str,
    stream: bool,
) -> Result<(reqwest::Response, Instant, Option<Duration>), BridgeError> {
    let (req, deadline) = inline_remote_images(req, ctx).await?;
    let body = {
        let messages = messages_from(&req, DeveloperRoleMode::MapToSystem);
        let mut body = serde_json::to_value(build_request(
            &req,
            capability_id(model_id),
            &messages,
            stream,
        ))
        .map_err(|e| BridgeError::Config(format!("serialize request body: {e}")))?;
        shape_body(&mut body);
        bytes::Bytes::from(
            serde_json::to_vec(&body)
                .map_err(|e| BridgeError::Config(format!("serialize request body: {e}")))?,
        )
    };
    drop(req);
    let url = aisix_gateway::url_cache::cached_endpoint_url::<BridgeError>(
        &ctx.provider_key_id,
        "bedrock/chat_completions",
        &[
            endpoint.base.as_deref().unwrap_or(""),
            endpoint.creds.region(),
        ],
        || {
            Ok(endpoint_url(
                endpoint.base.as_deref(),
                endpoint.creds.region(),
            ))
        },
    )?;
    let url_text = match &url {
        aisix_gateway::url_cache::EndpointUrl::Parsed(u) => u.as_str().to_string(),
        aisix_gateway::url_cache::EndpointUrl::Unparsed(raw) => raw.clone(),
    };
    let mut headers = vec![
        ("content-type".to_string(), "application/json".to_string()),
        (
            "accept".to_string(),
            if stream {
                "text/event-stream"
            } else {
                "application/json"
            }
            .to_string(),
        ),
    ];
    headers.extend(endpoint.extra_headers.iter().cloned());
    let auth = endpoint.creds.request_auth(ctx.deadline).await?;
    let headers = authenticated_headers(&auth, endpoint.creds.region(), &url_text, headers, &body)?;
    let started = Instant::now();
    let client = endpoint.client.clone();
    let resp = with_deadline(deadline, started, async move {
        url.post_on(&client)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(aisix_gateway::send_error)
    })
    .await?;
    aisix_gateway::note_upstream_response(resp.headers());
    if !resp.status().is_success() {
        return Err(http_error(resp, started).await);
    }
    Ok((resp, started, deadline))
}

pub(crate) async fn chat(
    endpoint: Endpoint,
    req: Cow<'_, ChatFormat>,
    ctx: &BridgeContext,
    model_id: &str,
) -> Result<ChatResponse, BridgeError> {
    let (resp, started, deadline) = send(&endpoint, req, ctx, model_id, false).await?;
    let request_id = request_id_header(resp.headers());
    // The body read gets what is left of the one budget, not a fresh one.
    let left = deadline.map(|d| d.saturating_sub(started.elapsed()));
    let parsed: OpenAiResponse = with_deadline(left, started, async {
        resp.json()
            .await
            .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
    })
    .await?;
    let mut chat = response_into_chat_response(parsed);
    if chat.id.is_empty() {
        chat.id = request_id;
    }
    if let Some(Value::Array(calls)) = chat.message.extra.get_mut("tool_calls") {
        mint_positional_ids(calls, &mut Vec::new(), true);
    }
    Ok(chat)
}

pub(crate) async fn chat_stream(
    endpoint: Endpoint,
    req: Cow<'_, ChatFormat>,
    ctx: &BridgeContext,
    model_id: &str,
) -> Result<ChatChunkStream, BridgeError> {
    let (resp, _, _) = send(&endpoint, req, ctx, model_id, true).await?;
    let request_id = request_id_header(resp.headers());
    let byte_stream = resp.bytes_stream();
    let stream = async_stream::try_stream! {
        let mut decoder = SseDecoder::new();
        let mut indexer = ToolCallIndexer::default();
        let mut minted: Vec<(u64, String)> = Vec::new();
        let mut byte_stream = Box::pin(byte_stream);
        'outer: while let Some(next) = byte_stream.next().await {
            let bytes = next.map_err(|e| BridgeError::Transport(aisix_gateway::transport_error_message(&e)))?;
            for event in decoder.feed(bytes.as_ref()).map_err(|e| BridgeError::UpstreamDecode(e.to_string()))? {
                match event {
                    SseEvent::Done => break 'outer,
                    SseEvent::Data(payload) => {
                        if let Some(chunk) = chunk_from(&payload, &mut indexer, &mut minted, &request_id)? {
                            yield chunk;
                        }
                    }
                }
            }
        }
        if let Some(SseEvent::Data(payload)) = decoder.finish().map_err(|e| BridgeError::UpstreamDecode(e.to_string()))? {
            if let Some(chunk) = chunk_from(&payload, &mut indexer, &mut minted, &request_id)? {
                yield chunk;
            }
        }
    };
    Ok(Box::pin(stream))
}

fn chunk_from(
    payload: &str,
    indexer: &mut ToolCallIndexer,
    minted: &mut Vec<(u64, String)>,
    request_id: &str,
) -> Result<Option<ChatChunk>, BridgeError> {
    let parsed: OpenAiStreamChunk = serde_json::from_str(payload).map_err(|e| {
        aisix_gateway::capture_in_band_error(payload, aisix_gateway::UpstreamWire::OpenAI)
            .unwrap_or(BridgeError::UpstreamDecode(e.to_string()))
    })?;
    let Some(mut chunk) = stream_chunk_into_chat_chunk(parsed, indexer) else {
        return Ok(None);
    };
    if chunk.id.is_empty() {
        chunk.id = request_id.to_string();
    }
    if let Some(calls) = chunk.delta.tool_calls.as_mut() {
        mint_positional_ids(calls, minted, false);
    }
    Ok(Some(chunk))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_route_needs_the_family_and_a_cross_region_prefix() {
        for id in [
            "global.openai.gpt-6.1-sol",
            "us.openai.gpt-6-luna",
            "us.openai.gpt-5.6-sol",
            "global.xai.grok-4.6",
            "us-gov.xai.grok-4.7",
        ] {
            assert!(is_default_route(id), "{id}");
        }
        for id in [
            "openai.gpt-6-sol",
            "us.openai.gpt-5.5",
            "us.openai.gpt-oss-120b-1:0",
            "openai.gpt-oss-20b-1:0",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.openai.gpt-6-sol",
        ] {
            assert!(!is_default_route(id), "{id}");
        }
    }

    #[test]
    fn gpt_versions_parse_major_and_minor() {
        assert_eq!(gpt_version("global.openai.gpt-6.1-sol"), Some((6, 1)));
        assert_eq!(gpt_version("us.openai.gpt-6-luna"), Some((6, 0)));
        assert_eq!(gpt_version("openai.gpt-5.6-terra"), Some((5, 6)));
        assert_eq!(gpt_version("openai.gpt-oss-120b-1:0"), None);
    }

    #[test]
    fn positional_ids_are_recognised() {
        assert!(is_positional_tool_call_id("call_0"));
        assert!(is_positional_tool_call_id("call_12"));
        assert!(!is_positional_tool_call_id("call_0a9f"));
        assert!(!is_positional_tool_call_id("chatcmpl-tool-1"));
    }
}
