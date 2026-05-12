mod span_attributes;
mod types;

use axum::response::sse::Event as SseEvent;
use fastrace::Span;
use reqwest::Url;
use serde_json::Value;
use span_attributes::{
    StreamOutputCollector, chunk_span_properties, request_span_properties, response_span_properties,
};
pub use types::MessagesError;

use super::FormatHandlerAdapter;
use crate::{
    gateway::{
        error::GatewayError,
        formats::AnthropicMessagesFormat,
        traits::{ChatFormat, ProviderCapabilities},
        types::{
            anthropic::{
                AnthropicContent, AnthropicContentBlock, AnthropicMessage,
                AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicStreamEvent,
                ImageSource, SystemBlock, SystemPrompt,
            },
            common::Usage,
            openai::{ContentPart, FunctionCall, ImageUrl, MessageContent, ToolCall},
        },
    },
    proxy::guardrails::{
        input_guardrail_payload_from_chat_messages, input_payload_from_check_payload,
        input_payload_to_chat_messages, output_guardrail_payload_from_chat_messages,
        output_payload_from_check_payload, output_payload_to_chat_messages,
    },
};

fn anthropic_error_sse_event(message: String) -> SseEvent {
    SseEvent::default()
        .event("error")
        .data(anthropic_error_event_payload(message))
}

fn anthropic_error_event_payload(message: String) -> String {
    serde_json::json!({
        "type": "error",
        "error": {
            "type": "api_error",
            "message": message,
        }
    })
    .to_string()
}

fn serialize_stream_event(event: &AnthropicStreamEvent) -> SseEvent {
    let mut sse_event =
        SseEvent::default().data(AnthropicMessagesFormat::serialize_chunk_payload(event));

    if let Some(event_type) = AnthropicMessagesFormat::sse_event_type(event) {
        sse_event = sse_event.event(event_type);
    }

    sse_event
}

pub(crate) struct MessagesAdapter;

impl FormatHandlerAdapter for MessagesAdapter {
    type Format = AnthropicMessagesFormat;
    type Request = AnthropicMessagesRequest;
    type Response = AnthropicMessagesResponse;
    type StreamChunk = AnthropicStreamEvent;
    type Error = MessagesError;
    type Collector = StreamOutputCollector;
    type LifecycleState = ();

    fn span_name() -> &'static str {
        "aisix.llm.messages"
    }

    fn missing_model_error() -> Self::Error {
        MessagesError::MissingModelInContext
    }

    fn set_model(request: &mut Self::Request, model: String) {
        request.model = model;
    }

    fn request_span_properties(
        request: &Self::Request,
        provider: &dyn ProviderCapabilities,
        base_url: Option<&Url>,
    ) -> Vec<(String, String)> {
        request_span_properties(request, provider, base_url)
    }

    fn response_span_properties(response: &Self::Response, usage: &Usage) -> Vec<(String, String)> {
        response_span_properties(response, usage)
    }

    fn apply_chunk_span_properties(span: &Span, chunk: &Self::StreamChunk, _is_first_item: bool) {
        span.add_properties(|| chunk_span_properties(chunk));
    }

    fn starts_output(chunk: &Self::StreamChunk) -> bool {
        matches!(chunk, AnthropicStreamEvent::ContentBlockStart { .. })
    }

    fn record_stream_item(collector: &mut Self::Collector, chunk: &Self::StreamChunk) {
        collector.record_event(chunk);
    }

    fn output_message_span_properties(collector: &Self::Collector) -> Vec<(String, String)> {
        collector.output_message_span_properties()
    }

    fn guardrail_input_payload(
        _lifecycle_state: &Self::LifecycleState,
        request: &Self::Request,
    ) -> Result<Option<crate::guardrail::traits::GuardrailCheckPayload>, Self::Error> {
        let (hub_request, _) = AnthropicMessagesFormat::to_hub(request)?;
        let payload = input_guardrail_payload_from_chat_messages(&hub_request.messages)
            .map(crate::guardrail::traits::GuardrailCheckPayload::Input)
            .map_err(bridge_error)?;
        Ok(Some(payload))
    }

    fn apply_input_guardrail_rewrite(
        _lifecycle_state: &mut Self::LifecycleState,
        request: &mut Self::Request,
        rewrite: crate::guardrail::traits::GuardrailCheckPayload,
    ) -> Result<(), Self::Error> {
        let messages = input_payload_to_chat_messages(
            &input_payload_from_check_payload(rewrite).map_err(bridge_error)?,
        )
        .map_err(bridge_error)?;
        rewrite_anthropic_request_messages(request, &messages).map_err(MessagesError::from)
    }

    fn guardrail_output_payload(
        _lifecycle_state: &Self::LifecycleState,
        response: &Self::Response,
    ) -> Result<Option<crate::guardrail::traits::GuardrailCheckPayload>, Self::Error> {
        let messages =
            vec![anthropic_response_to_chat_message(response).map_err(MessagesError::from)?];
        let payload = output_guardrail_payload_from_chat_messages(&messages)
            .map(crate::guardrail::traits::GuardrailCheckPayload::Output)
            .map_err(bridge_error)?;
        Ok(Some(payload))
    }

    fn apply_output_guardrail_rewrite(
        _lifecycle_state: &mut Self::LifecycleState,
        response: &mut Self::Response,
        rewrite: crate::guardrail::traits::GuardrailCheckPayload,
    ) -> Result<(), Self::Error> {
        let messages = output_payload_to_chat_messages(
            &output_payload_from_check_payload(rewrite).map_err(bridge_error)?,
        )
        .map_err(bridge_error)?;
        let [message] = messages.as_slice() else {
            return Err(bridge_error(GatewayError::Bridge(format!(
                "anthropic output guardrail rewrite expected exactly 1 message, got {}",
                messages.len()
            ))));
        };

        response.role = message.role.clone();
        response.content =
            anthropic_blocks_from_chat_message(message).map_err(MessagesError::from)?;
        Ok(())
    }

    fn serialize_stream_item(chunk: &Self::StreamChunk) -> SseEvent {
        serialize_stream_event(chunk)
    }

    fn stream_error_event(error: &GatewayError) -> Option<SseEvent> {
        Some(anthropic_error_sse_event(error.to_string()))
    }
}

fn rewrite_anthropic_request_messages(
    request: &mut AnthropicMessagesRequest,
    messages: &[crate::gateway::types::openai::ChatMessage],
) -> Result<(), GatewayError> {
    let (system, anthropic_messages) = anthropic_request_parts_from_chat_messages(messages)?;
    request.system = system;
    request.messages = anthropic_messages;
    Ok(())
}

fn anthropic_request_parts_from_chat_messages(
    messages: &[crate::gateway::types::openai::ChatMessage],
) -> Result<(Option<SystemPrompt>, Vec<AnthropicMessage>), GatewayError> {
    let split_index = messages
        .iter()
        .position(|message| message.role != "system")
        .unwrap_or(messages.len());

    if messages[split_index..]
        .iter()
        .any(|message| message.role == "system")
    {
        return Err(GatewayError::Bridge(
            "Anthropic request rewrite requires system messages to remain at the front".into(),
        ));
    }

    let system = system_prompt_from_chat_messages(&messages[..split_index])?;
    let anthropic_messages = messages[split_index..]
        .iter()
        .map(chat_message_to_anthropic_message)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((system, anthropic_messages))
}

fn system_prompt_from_chat_messages(
    messages: &[crate::gateway::types::openai::ChatMessage],
) -> Result<Option<SystemPrompt>, GatewayError> {
    if messages.is_empty() {
        return Ok(None);
    }

    let mut blocks = Vec::new();
    for message in messages {
        for text in message_content_text_segments(message.content.as_ref())? {
            blocks.push(SystemBlock {
                r#type: "text".into(),
                text,
                cache_control: None,
            });
        }
    }

    match blocks.as_slice() {
        [] => Ok(None),
        [single] => Ok(Some(SystemPrompt::Text(single.text.clone()))),
        _ => Ok(Some(SystemPrompt::Blocks(blocks))),
    }
}

fn chat_message_to_anthropic_message(
    message: &crate::gateway::types::openai::ChatMessage,
) -> Result<AnthropicMessage, GatewayError> {
    match message.role.as_str() {
        "user" | "assistant" => Ok(AnthropicMessage {
            role: message.role.clone(),
            content: anthropic_content_from_chat_message(message)?,
        }),
        "tool" => Ok(AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![AnthropicContentBlock::ToolResult {
                tool_use_id: message.tool_call_id.clone().ok_or_else(|| {
                    GatewayError::Bridge(
                        "tool message rewrite requires tool_call_id for Anthropic tool_result"
                            .into(),
                    )
                })?,
                content: anthropic_content_from_optional_message_content(message.content.as_ref())?,
                is_error: None,
                cache_control: None,
            }]),
        }),
        other => Err(GatewayError::Bridge(format!(
            "unsupported role {} for Anthropic request rewrite",
            other
        ))),
    }
}

fn anthropic_content_from_chat_message(
    message: &crate::gateway::types::openai::ChatMessage,
) -> Result<AnthropicContent, GatewayError> {
    let mut blocks = anthropic_blocks_from_message_content(message.content.as_ref())?;

    if let Some(tool_calls) = &message.tool_calls {
        if message.role != "assistant" {
            return Err(GatewayError::Bridge(
                "only assistant messages can carry tool calls in Anthropic rewrite".into(),
            ));
        }

        for tool_call in tool_calls {
            if tool_call.r#type != "function" {
                return Err(GatewayError::Bridge(format!(
                    "Anthropic rewrite only supports function tool calls, got {}",
                    tool_call.r#type
                )));
            }

            let input = serde_json::from_str(&tool_call.function.arguments).map_err(|error| {
                GatewayError::Bridge(format!(
                    "assistant tool call arguments are not valid JSON: {}",
                    error
                ))
            })?;

            blocks.push(AnthropicContentBlock::ToolUse {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input,
                cache_control: None,
            });
        }
    }

    anthropic_content_from_blocks(blocks)
}

fn anthropic_content_from_optional_message_content(
    content: Option<&MessageContent>,
) -> Result<Option<AnthropicContent>, GatewayError> {
    let blocks = anthropic_blocks_from_message_content(content)?;
    if blocks.is_empty() {
        Ok(None)
    } else {
        anthropic_content_from_blocks(blocks).map(Some)
    }
}

fn anthropic_content_from_blocks(
    blocks: Vec<AnthropicContentBlock>,
) -> Result<AnthropicContent, GatewayError> {
    match blocks.as_slice() {
        [] => Err(GatewayError::Bridge(
            "Anthropic rewrite requires at least one content block".into(),
        )),
        [AnthropicContentBlock::Text { text, .. }] => Ok(AnthropicContent::Text(text.clone())),
        _ => Ok(AnthropicContent::Blocks(blocks)),
    }
}

fn anthropic_blocks_from_chat_message(
    message: &crate::gateway::types::openai::ChatMessage,
) -> Result<Vec<AnthropicContentBlock>, GatewayError> {
    if message.role != "assistant" {
        return Err(GatewayError::Bridge(format!(
            "Anthropic response rewrite requires an assistant message, got {}",
            message.role
        )));
    }

    let mut blocks = anthropic_blocks_from_message_content(message.content.as_ref())?;
    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            if tool_call.r#type != "function" {
                return Err(GatewayError::Bridge(format!(
                    "Anthropic response rewrite only supports function tool calls, got {}",
                    tool_call.r#type
                )));
            }
            let input: Value =
                serde_json::from_str(&tool_call.function.arguments).map_err(|error| {
                    GatewayError::Bridge(format!(
                        "assistant tool call arguments are not valid JSON: {}",
                        error
                    ))
                })?;
            blocks.push(AnthropicContentBlock::ToolUse {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input,
                cache_control: None,
            });
        }
    }
    Ok(blocks)
}

fn anthropic_blocks_from_message_content(
    content: Option<&MessageContent>,
) -> Result<Vec<AnthropicContentBlock>, GatewayError> {
    let Some(content) = content else {
        return Ok(vec![]);
    };

    match content {
        MessageContent::Text(text) => Ok(vec![AnthropicContentBlock::Text {
            text: text.clone(),
            cache_control: None,
        }]),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(AnthropicContentBlock::Text {
                    text: text.clone(),
                    cache_control: None,
                }),
                ContentPart::ImageUrl { image_url } => Ok(AnthropicContentBlock::Image {
                    source: image_url_to_source(&image_url.url)?,
                    cache_control: None,
                }),
            })
            .collect(),
    }
}

fn anthropic_response_to_chat_message(
    response: &AnthropicMessagesResponse,
) -> Result<crate::gateway::types::openai::ChatMessage, GatewayError> {
    let mut text_segments = Vec::new();
    let mut rich_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut has_non_text_part = false;

    for block in &response.content {
        match block {
            AnthropicContentBlock::Text { text, .. } => {
                text_segments.push(text.clone());
                rich_parts.push(ContentPart::Text { text: text.clone() });
            }
            AnthropicContentBlock::Image { source, .. } => {
                has_non_text_part = true;
                rich_parts.push(ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: format!("data:{};base64,{}", source.media_type, source.data),
                        detail: None,
                    },
                });
            }
            AnthropicContentBlock::ToolUse {
                id, name, input, ..
            } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    r#type: "function".into(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input)
                            .map_err(|error| GatewayError::Transform(error.to_string()))?,
                    },
                });
            }
            AnthropicContentBlock::ToolResult { .. } => {
                return Err(GatewayError::Bridge(
                    "assistant response contained unsupported tool_result block".into(),
                ));
            }
        }
    }

    Ok(crate::gateway::types::openai::ChatMessage {
        role: response.role.clone(),
        content: if has_non_text_part {
            Some(MessageContent::Parts(rich_parts))
        } else if !text_segments.is_empty() {
            Some(MessageContent::Text(text_segments.join("")))
        } else {
            None
        },
        name: None,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id: None,
    })
}

fn message_content_text_segments(
    content: Option<&MessageContent>,
) -> Result<Vec<String>, GatewayError> {
    let Some(content) = content else {
        return Ok(vec![]);
    };

    match content {
        MessageContent::Text(text) => Ok(vec![text.clone()]),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(text.clone()),
                ContentPart::ImageUrl { .. } => Err(GatewayError::Bridge(
                    "Anthropic system prompt rewrite does not support image content".into(),
                )),
            })
            .collect(),
    }
}

fn image_url_to_source(url: &str) -> Result<ImageSource, GatewayError> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Err(GatewayError::Bridge(
            "Anthropic rewrite only supports image_url data URLs for image content".into(),
        ));
    };
    let Some((metadata, data)) = payload.split_once(',') else {
        return Err(GatewayError::Bridge(
            "invalid data URL for Anthropic image content".into(),
        ));
    };
    let Some(media_type) = metadata.strip_suffix(";base64") else {
        return Err(GatewayError::Bridge(
            "Anthropic image content requires base64 data URLs".into(),
        ));
    };

    Ok(ImageSource {
        r#type: "base64".into(),
        media_type: media_type.into(),
        data: data.into(),
    })
}

fn bridge_error<E>(error: E) -> MessagesError
where
    E: std::fmt::Display,
{
    MessagesError::GatewayError(GatewayError::Bridge(error.to_string()))
}
