//! `/v1/messages` and `/v1/messages/count_tokens` for Claude on Bedrock,
//! served on Bedrock's own Anthropic Messages routes: `InvokeModel`,
//! `InvokeModelWithResponseStream` and `CountTokens`. The caller's body
//! reaches Claude as it was sent, apart from what
//! [`aisix_provider_anthropic::platform::shape_request`] adapts, and the
//! response comes back in the same wire.

use std::sync::OnceLock;
use std::time::Instant;

use aisix_gateway::media_fetch::{fetch_urls, Encoding};
use aisix_gateway::{BridgeContext, BridgeError, NativeMessages};
use aisix_provider_anthropic::platform::{self, Platform};
use aws_sdk_bedrockruntime::error::SdkError;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::types::error::ResponseStreamError;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, ConverseTokensRequest, CountTokensInput,
    InvokeModelTokensRequest, Message, ResponseStream, SystemContentBlock, Tool, ToolConfiguration,
    ToolInputSchema, ToolSpecification,
};
use serde_json::Value;

use crate::bridge::{
    strip_region_prefix, upstream_model, validate_model_id_chars, BedrockBridge, BedrockPublisher,
    BedrockSecret,
};

/// Whether `/v1/messages` for `upstream_id` takes Bedrock's own Anthropic
/// route: an `anthropic.` model id, region prefix or not, and an ARN that
/// names a Claude model. An ARN that hides its model (an application
/// inference profile) keeps the chat translation.
pub(crate) fn serves_anthropic_messages(upstream_id: &str) -> bool {
    if upstream_id.starts_with("arn:") {
        return upstream_id.contains("claude");
    }
    BedrockPublisher::from_model_id(upstream_id) == Some(BedrockPublisher::Anthropic)
}

impl BedrockBridge {
    pub(crate) async fn native_messages(
        &self,
        mut body: Value,
        ctx: &BridgeContext,
    ) -> Result<NativeMessages, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        validate_model_id_chars(upstream_id)?;
        let stream = body.get("stream").and_then(Value::as_bool) == Some(true);
        // Bedrock refuses a request with no turn on every rail; refused here
        // with the same message the chat rails give.
        if body
            .get("messages")
            .and_then(Value::as_array)
            .is_none_or(|m| m.is_empty())
        {
            return Err(BridgeError::InvalidRequest(
                "messages must include at least one user or assistant turn; \
                 a system-only request is not supported for this model"
                    .into(),
            ));
        }
        let fetched = fetch_urls(platform::media_urls(&body), ctx, Encoding::Base64).await?;
        let charged = fetched.charge(ctx);
        let ctx = &*charged;
        platform::inline_media(&mut body, Platform::Bedrock, &fetched)
            .map_err(BridgeError::InvalidRequest)?;
        drop(fetched);
        let betas = platform::header_betas(ctx.client_headers.as_deref());
        platform::shape_request(&mut body, Platform::Bedrock, upstream_id, betas);
        let bytes = Vec::from(
            aisix_gateway::json_body(&body)
                .map_err(|e| BridgeError::Config(format!("serialize request body: {e}")))?,
        );
        drop(body);

        let client = self.build_client_from_ctx(ctx).await?;
        let started = Instant::now();
        let deadline = ctx.deadline;
        if !stream {
            let resp = client
                .invoke_model()
                .model_id(upstream_id)
                .content_type("application/json")
                .accept("application/json")
                .body(Blob::new(bytes))
                .send()
                .await
                .map_err(|e| crate::errors::sdk_error(e, started, deadline))?;
            let json: Value = serde_json::from_slice(resp.body().as_ref())
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            return Ok(NativeMessages::Json(json));
        }

        let mut resp = client
            .invoke_model_with_response_stream()
            .model_id(upstream_id)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(bytes))
            .send()
            .await
            .map_err(|e| crate::errors::sdk_error(e, started, deadline))?;

        // The first event is awaited here, before the response is committed,
        // so an exception Bedrock raises before any content (a throttle is
        // the common one) fails this attempt and the caller can fail over.
        let first = match deadline {
            Some(d) => tokio::time::timeout(d.saturating_sub(started.elapsed()), resp.body.recv())
                .await
                .map_err(|_| BridgeError::Timeout {
                    elapsed_ms: started.elapsed().as_millis() as u64,
                    cause: String::new(),
                })?,
            None => resp.body.recv().await,
        };
        let first = match first {
            Ok(Some(event)) => event_bytes(event),
            Ok(None) => return Err(BridgeError::StreamAborted),
            Err(e) => return Err(stream_error(&e).into_bridge_error()),
        };
        let frames = async_stream::stream! {
            if let Some(frame) = first {
                yield Ok(frame);
            }
            loop {
                match resp.body.recv().await {
                    Ok(Some(event)) => {
                        if let Some(frame) = event_bytes(event) {
                            yield Ok(frame);
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let err = stream_error(&e);
                        yield Ok(platform::error_frame(err.kind, &err.message));
                        break;
                    }
                }
            }
        };
        Ok(NativeMessages::Stream(Box::pin(frames)))
    }

    pub(crate) async fn native_count_tokens(
        &self,
        body: Value,
        ctx: &BridgeContext,
    ) -> Result<Value, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        validate_model_id_chars(upstream_id)?;
        // CountTokens names the base model: no inference-profile prefix, no
        // ARN path.
        let tail = upstream_id.rsplit('/').next().unwrap_or(upstream_id);
        let model_id = strip_region_prefix(tail);
        let input = match platform::bedrock_count_tokens_invoke_body(&body, upstream_id) {
            Some(invoke) => {
                let bytes =
                    Vec::from(aisix_gateway::json_body(&invoke).map_err(|e| {
                        BridgeError::Config(format!("serialize request body: {e}"))
                    })?);
                CountTokensInput::InvokeModel(
                    InvokeModelTokensRequest::builder()
                        .body(Blob::new(bytes))
                        .build()
                        .map_err(|e| BridgeError::Config(e.to_string()))?,
                )
            }
            None => CountTokensInput::Converse(converse_count_input(&body)?),
        };
        let client = self.build_client_from_ctx(ctx).await?;
        let started = Instant::now();
        let runtime = client
            .count_tokens()
            .model_id(model_id)
            .input(input)
            .send()
            .await
            .map_err(|e| crate::errors::sdk_error(e, started, ctx.deadline));
        match runtime {
            Ok(resp) => Ok(serde_json::json!({"input_tokens": resp.input_tokens()})),
            // bedrock-runtime's CountTokens does not count every Claude model
            // (those Bedrock serves only through cross-region inference); AWS
            // counts them on Bedrock Mantle's Anthropic count_tokens instead.
            Err(BridgeError::UpstreamStatus {
                status: 400 | 404, ..
            }) if model_id.to_ascii_lowercase().contains("claude") => {
                mantle_count_tokens(&body, model_id, ctx).await
            }
            Err(e) => Err(e),
        }
    }
}

static MANTLE_ENDPOINT: OnceLock<Option<String>> = OnceLock::new();

/// Install `upstream.bedrock.mantle_endpoint`. Called once during boot.
pub fn init_mantle_endpoint(endpoint: Option<String>) {
    let _ = MANTLE_ENDPOINT.set(endpoint.filter(|e| !e.trim().is_empty()));
}

const MANTLE_MESSAGES_PATH: &str = "/anthropic/v1/messages";

/// `{mantle}/anthropic/v1/messages/count_tokens`, where `{mantle}` is the
/// configured endpoint (which may already end in the messages path) or the
/// public regional host.
fn mantle_count_tokens_url(region: &str) -> String {
    let base = match MANTLE_ENDPOINT.get().cloned().flatten() {
        Some(configured) => {
            let base = configured.trim().trim_end_matches('/').to_string();
            if base.ends_with(MANTLE_MESSAGES_PATH) {
                base
            } else {
                format!("{base}{MANTLE_MESSAGES_PATH}")
            }
        }
        None => format!("https://bedrock-mantle.{region}.api.aws{MANTLE_MESSAGES_PATH}"),
    };
    format!("{base}/count_tokens")
}

/// Count on Bedrock Mantle's Anthropic `count_tokens`: the base model id,
/// the request's messages, `system` and `tools`, and the options counting
/// reads (`thinking`, `tool_choice`, `output_config`), authenticated like
/// every other Bedrock call on the provider key: its bearer API key, or
/// SigV4 (service `bedrock`).
async fn mantle_count_tokens(
    body: &Value,
    model_id: &str,
    ctx: &BridgeContext,
) -> Result<Value, BridgeError> {
    let creds = BedrockSecret::parse(&ctx.provider_key.api_key)?;
    let mut count = serde_json::Map::new();
    count.insert("model".to_string(), Value::String(model_id.to_string()));
    for key in [
        "messages",
        "system",
        "tools",
        "thinking",
        "tool_choice",
        "output_config",
    ] {
        if let Some(v) = body.get(key).filter(|v| !v.is_null()) {
            count.insert(key.to_string(), v.clone());
        }
    }
    let bytes = Vec::from(
        aisix_gateway::json_body(&Value::Object(count))
            .map_err(|e| BridgeError::Config(format!("serialize request body: {e}")))?,
    );
    let url = mantle_count_tokens_url(creds.region());

    let unsigned = [
        ("content-type".to_string(), "application/json".to_string()),
        ("anthropic-version".to_string(), "2023-06-01".to_string()),
    ];
    let auth = creds.request_auth(ctx.deadline).await?;
    let signed = crate::bridge::auth_headers(&auth, creds.region(), &url, &unsigned, &bytes)?;

    let client = aisix_gateway::upstream_tls::raw_client_for_provider_key(
        ctx.provider_key.upstream_connection().as_ref(),
    );
    let endpoint = aisix_gateway::url_cache::cached_endpoint_url(
        &ctx.provider_key_id,
        "bedrock/mantle-count-tokens",
        &[&url],
        || Ok::<_, BridgeError>(url.clone()),
    )?;
    let mut req = endpoint.post_on(&client).body(bytes);
    for (name, value) in unsigned.iter().chain(signed.iter()) {
        req = req.header(name, value);
    }
    if let Some(d) = ctx.deadline {
        req = req.timeout(d);
    }
    let started = Instant::now();
    let resp = req.send().await.map_err(aisix_gateway::send_error)?;
    aisix_gateway::note_upstream_response(resp.headers());
    let status = resp.status();
    if !status.is_success() {
        let retry_after = aisix_gateway::parse_retry_after(resp.headers());
        let kind = resp
            .headers()
            .get("x-amzn-errortype")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let body =
            aisix_gateway::read_body_capped(resp, aisix_gateway::MAX_UPSTREAM_ERROR_BODY_BYTES)
                .await;
        return Err(crate::errors::upstream_error(
            status.as_u16(),
            kind,
            None,
            aisix_gateway::upstream_error_message(&body),
            retry_after,
            started,
        ));
    }
    let json: Value = resp
        .json()
        .await
        .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
    let tokens = json
        .get("input_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            BridgeError::UpstreamDecode("count_tokens answer carries no input_tokens".into())
        })?;
    Ok(serde_json::json!({"input_tokens": tokens}))
}

/// The `converse` input for a body whose messages are all plain strings:
/// each message as one text block, `system` as text blocks, and the tools
/// as tool specs under a Bedrock-valid name.
fn converse_count_input(body: &Value) -> Result<ConverseTokensRequest, BridgeError> {
    let mut req = ConverseTokensRequest::builder();
    for m in body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let role = match m.get("role").and_then(Value::as_str) {
            Some("assistant") => ConversationRole::Assistant,
            _ => ConversationRole::User,
        };
        let mut msg = Message::builder().role(role);
        if let Some(text) = m.get("content").and_then(Value::as_str) {
            msg = msg.content(ContentBlock::Text(text.to_string()));
        }
        req = req.messages(
            msg.build()
                .map_err(|e| BridgeError::Config(e.to_string()))?,
        );
    }
    match body.get("system") {
        Some(Value::String(s)) => req = req.system(SystemContentBlock::Text(s.clone())),
        Some(Value::Array(blocks)) => {
            for b in blocks {
                let text = b.get("text").and_then(Value::as_str).unwrap_or_default();
                req = req.system(SystemContentBlock::Text(text.to_string()));
            }
        }
        _ => {}
    }
    let tools: Vec<Tool> = body
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|t| {
            let name = converse_tool_name(t.get("name").and_then(Value::as_str).unwrap_or(""));
            let description = t
                .get("description")
                .and_then(Value::as_str)
                .filter(|d| !d.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| name.clone());
            let schema = t
                .get("input_schema")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
            ToolSpecification::builder()
                .name(name)
                .description(description)
                .input_schema(ToolInputSchema::Json(crate::convert::json_to_document(
                    &schema,
                )))
                .build()
                .ok()
                .map(Tool::ToolSpec)
        })
        .collect();
    if !tools.is_empty() {
        req = req.tool_config(
            ToolConfiguration::builder()
                .set_tools(Some(tools))
                .build()
                .map_err(|e| BridgeError::Config(e.to_string()))?,
        );
    }
    Ok(req.build())
}

/// A tool name Converse accepts: `[a-zA-Z0-9_]`, starting with a letter,
/// at most 64 characters.
fn converse_tool_name(raw: &str) -> String {
    let name: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let name = match name.chars().next() {
        Some(c) if !c.is_ascii_alphabetic() => format!("t_{name}"),
        _ => name,
    };
    name.chars().take(64).collect()
}

/// One stream event as an Anthropic SSE frame. A `chunk` carries the
/// Anthropic event JSON itself.
fn event_bytes(event: ResponseStream) -> Option<bytes::Bytes> {
    match event {
        ResponseStream::Chunk(part) => part.bytes.map(|b| platform::event_frame(b.as_ref())),
        _ => None,
    }
}

/// An exception Bedrock raised inside the event stream, in Anthropic's
/// error taxonomy so an Anthropic client's retry logic reads it.
struct StreamException {
    kind: &'static str,
    status: Option<u16>,
    message: String,
    code: Option<String>,
}

fn stream_error(
    e: &SdkError<ResponseStreamError, aws_smithy_types::event_stream::RawMessage>,
) -> StreamException {
    let Some(svc) = e.as_service_error() else {
        return StreamException {
            kind: "api_error",
            status: None,
            message: "upstream stream failed".to_string(),
            code: None,
        };
    };
    let (kind, status) = match svc {
        ResponseStreamError::ThrottlingException(_) => ("rate_limit_error", Some(429)),
        ResponseStreamError::ServiceUnavailableException(_) => ("overloaded_error", Some(529)),
        ResponseStreamError::ValidationException(_) => ("invalid_request_error", Some(400)),
        ResponseStreamError::ModelTimeoutException(_) => ("timeout_error", Some(504)),
        ResponseStreamError::InternalServerException(_) => ("api_error", Some(500)),
        // The model backend's own status, when Bedrock relays one.
        ResponseStreamError::ModelStreamErrorException(inner) => {
            match inner.original_status_code() {
                Some(429) => ("rate_limit_error", Some(429)),
                Some(503 | 529) => ("overloaded_error", Some(529)),
                Some(400) => ("invalid_request_error", Some(400)),
                Some(c) => ("api_error", u16::try_from(c).ok()),
                None => ("api_error", Some(502)),
            }
        }
        _ => ("api_error", None),
    };
    // The upstream's own text; a status-keyed phrase only when it sent none.
    let message = svc
        .meta()
        .message()
        .filter(|m| !m.is_empty())
        .map(|m| aisix_gateway::truncate_lossy(m, aisix_gateway::MAX_UPSTREAM_ERROR_MESSAGE_BYTES))
        .unwrap_or_else(|| match status {
            Some(s) => format!("upstream reported stream error {s}"),
            None => "upstream reported a stream error".to_string(),
        });
    StreamException {
        kind,
        status,
        message,
        code: svc.meta().code().map(str::to_string),
    }
}

impl StreamException {
    fn into_bridge_error(self) -> BridgeError {
        BridgeError::UpstreamInBand {
            status: self.status,
            message: self.message,
            parsed: Some(Box::new(aisix_gateway::UpstreamErrorView {
                kind: self.code.or(Some(self.kind.to_string())),
                message: None,
                code: None,
                param: None,
            })),
            wire: aisix_gateway::UpstreamWire::Bedrock,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_ids_take_the_anthropic_route() {
        for id in [
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-5-v1:0",
        ] {
            assert!(serves_anthropic_messages(id), "{id}");
        }
        for id in [
            "meta.llama3-3-70b-instruct-v1:0",
            "openai.gpt-oss-120b-1:0",
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abcd1234",
        ] {
            assert!(!serves_anthropic_messages(id), "{id}");
        }
    }

    #[test]
    fn converse_tool_names_are_made_valid() {
        assert_eq!(converse_tool_name("mcp__fs.read"), "mcp__fs_read");
        assert_eq!(converse_tool_name("1st"), "t_1st");
    }
}
