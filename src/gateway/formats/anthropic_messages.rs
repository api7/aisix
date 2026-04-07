use serde_json::{Value, json};

use crate::gateway::{
    error::{GatewayError, Result},
    traits::{AnthropicMessagesNativeStreamState, ChatFormat, NativeHandler, ProviderCapabilities},
    types::{
        anthropic::{
            AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicMessagesRequest,
            AnthropicMessagesResponse, AnthropicStreamEvent, AnthropicTool, AnthropicToolChoice,
            AnthropicUsage, CacheControl, ImageSource, SystemPrompt,
        },
        common::{AnthropicMessagesExtras, BridgeContext},
        openai::{
            ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage, ChatMessage,
            ContentPart, FunctionCall, FunctionDefinition, ImageUrl, MessageContent, StopCondition,
            Tool, ToolCall, ToolChoice, ToolChoiceFunction,
        },
    },
};

pub struct AnthropicMessagesFormat;

impl ChatFormat for AnthropicMessagesFormat {
    type Request = AnthropicMessagesRequest;
    type Response = AnthropicMessagesResponse;
    type StreamChunk = AnthropicStreamEvent;
    type BridgeState = ();
    type NativeStreamState = AnthropicMessagesNativeStreamState;

    fn name() -> &'static str {
        "anthropic_messages"
    }

    fn is_stream(req: &Self::Request) -> bool {
        req.stream.unwrap_or(false)
    }

    fn extract_model(req: &Self::Request) -> &str {
        &req.model
    }

    fn to_hub(req: &Self::Request) -> Result<(ChatCompletionRequest, BridgeContext)> {
        let (mut messages, system_cache_control) =
            system_prompt_to_hub_messages(req.system.as_ref())?;
        for message in &req.messages {
            messages.extend(anthropic_message_to_hub_messages(message)?);
        }

        let metadata = req
            .metadata
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| GatewayError::Transform(error.to_string()))?;

        let mut ctx = BridgeContext::default();
        if metadata.is_some() || system_cache_control.is_some() {
            ctx.anthropic_messages_extras = Some(AnthropicMessagesExtras {
                metadata,
                system_cache_control,
            });
        }
        if let Some(top_k) = req.top_k {
            ctx.passthrough.insert("top_k".into(), json!(top_k));
        }

        Ok((
            ChatCompletionRequest {
                messages,
                model: req.model.clone(),
                max_tokens: Some(req.max_tokens),
                stop: stop_sequences_to_openai(req.stop_sequences.as_ref()),
                stream: req.stream,
                temperature: req.temperature,
                top_p: req.top_p,
                tools: req
                    .tools
                    .as_ref()
                    .map(|tools| anthropic_tools_to_openai(tools))
                    .transpose()?,
                tool_choice: anthropic_tool_choice_to_openai(req.tool_choice.as_ref())?,
                user: req
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.user_id.clone()),
                ..Default::default()
            },
            ctx,
        ))
    }

    fn from_hub(resp: &ChatCompletionResponse, _ctx: &BridgeContext) -> Result<Self::Response> {
        openai_response_to_anthropic(resp)
    }

    fn from_hub_stream(
        _chunk: &crate::gateway::types::openai::ChatCompletionChunk,
        _state: &mut Self::BridgeState,
        _ctx: &BridgeContext,
    ) -> Result<Vec<Self::StreamChunk>> {
        Err(GatewayError::Bridge(
            "Anthropic messages hub streaming bridge is not implemented yet".into(),
        ))
    }

    fn native_support(provider: &dyn ProviderCapabilities) -> Option<NativeHandler<'_>>
    where
        Self: Sized,
    {
        provider
            .as_native_anthropic_messages()
            .map(NativeHandler::AnthropicMessages)
    }

    fn call_native(
        native: &NativeHandler<'_>,
        request: &Self::Request,
        _stream: bool,
    ) -> Result<(String, Value)>
    where
        Self: Sized,
    {
        match native {
            NativeHandler::AnthropicMessages(handler) => Ok((
                handler
                    .native_anthropic_messages_endpoint(&request.model)
                    .into_owned(),
                handler.transform_anthropic_messages_request(request)?,
            )),
            _ => Err(GatewayError::NativeNotSupported {
                provider: native.provider_name().into(),
            }),
        }
    }

    fn transform_native_stream_chunk(
        provider: &dyn ProviderCapabilities,
        raw: &str,
        state: &mut Self::NativeStreamState,
    ) -> Result<Vec<Self::StreamChunk>> {
        let Some(handler) = provider.as_native_anthropic_messages() else {
            return Err(GatewayError::NativeNotSupported {
                provider: provider.name().into(),
            });
        };

        handler.transform_anthropic_messages_stream_chunk(raw, state)
    }

    fn parse_native_response(native: &NativeHandler<'_>, body: Value) -> Result<Self::Response>
    where
        Self: Sized,
    {
        match native {
            NativeHandler::AnthropicMessages(handler) => {
                handler.transform_anthropic_messages_response(body)
            }
            _ => Err(GatewayError::NativeNotSupported {
                provider: native.provider_name().into(),
            }),
        }
    }

    fn serialize_chunk_payload(chunk: &Self::StreamChunk) -> String {
        serde_json::to_string(chunk).expect("anthropic stream event should serialize")
    }

    fn sse_event_type(chunk: &Self::StreamChunk) -> Option<&'static str> {
        Some(chunk.event_type())
    }
}

fn system_prompt_to_hub_messages(
    system: Option<&SystemPrompt>,
) -> Result<(Vec<ChatMessage>, Option<CacheControl>)> {
    match system {
        None => Ok((vec![], None)),
        Some(SystemPrompt::Text(text)) => Ok((
            vec![hub_message(
                "system",
                Some(MessageContent::Text(text.clone())),
            )],
            None,
        )),
        Some(SystemPrompt::Blocks(blocks)) => {
            let mut messages = Vec::with_capacity(blocks.len());
            let mut cache_control = None;

            for block in blocks {
                if let Some(block_cache_control) = &block.cache_control {
                    if cache_control.is_some() {
                        return Err(GatewayError::Bridge(
                            "Anthropic system prompts with multiple cache_control blocks are not supported by hub bridging"
                                .into(),
                        ));
                    }
                    cache_control = Some(block_cache_control.clone());
                }

                messages.push(hub_message(
                    "system",
                    Some(MessageContent::Text(block.text.clone())),
                ));
            }

            Ok((messages, cache_control))
        }
    }
}

fn anthropic_message_to_hub_messages(message: &AnthropicMessage) -> Result<Vec<ChatMessage>> {
    match message.role.as_str() {
        "user" => anthropic_user_message_to_hub_messages(message),
        "assistant" => Ok(vec![anthropic_assistant_message_to_hub(message)?]),
        other => Err(GatewayError::Bridge(format!(
            "unsupported Anthropic message role {} for hub bridging",
            other
        ))),
    }
}

fn anthropic_user_message_to_hub_messages(message: &AnthropicMessage) -> Result<Vec<ChatMessage>> {
    match &message.content {
        AnthropicContent::Text(text) => Ok(vec![hub_message(
            "user",
            Some(MessageContent::Text(text.clone())),
        )]),
        AnthropicContent::Blocks(blocks) => anthropic_user_blocks_to_hub_messages(blocks),
    }
}

fn anthropic_user_blocks_to_hub_messages(
    blocks: &[AnthropicContentBlock],
) -> Result<Vec<ChatMessage>> {
    let mut messages = Vec::new();
    let mut pending_blocks = Vec::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { .. } | AnthropicContentBlock::Image { .. } => {
                pending_blocks.push(block.clone());
            }
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                if !pending_blocks.is_empty() {
                    messages.push(hub_message(
                        "user",
                        anthropic_blocks_to_openai_content(&pending_blocks)?,
                    ));
                    pending_blocks.clear();
                }

                messages.push(ChatMessage {
                    role: "tool".into(),
                    content: anthropic_optional_content_to_openai(content.as_ref())?,
                    name: None,
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                });
            }
            AnthropicContentBlock::ToolUse { .. } => {
                return Err(GatewayError::Bridge(
                    "Anthropic user messages cannot contain tool_use blocks when bridging to hub format"
                        .into(),
                ));
            }
        }
    }

    if !pending_blocks.is_empty() {
        messages.push(hub_message(
            "user",
            anthropic_blocks_to_openai_content(&pending_blocks)?,
        ));
    }

    if messages.is_empty() {
        messages.push(hub_message(
            "user",
            Some(MessageContent::Text(String::new())),
        ));
    }

    Ok(messages)
}

fn anthropic_assistant_message_to_hub(message: &AnthropicMessage) -> Result<ChatMessage> {
    let (content, tool_calls) = match &message.content {
        AnthropicContent::Text(text) => (Some(MessageContent::Text(text.clone())), None),
        AnthropicContent::Blocks(blocks) => anthropic_assistant_blocks_to_hub(blocks)?,
    };

    Ok(ChatMessage {
        role: "assistant".into(),
        content,
        name: None,
        tool_calls,
        tool_call_id: None,
    })
}

fn anthropic_assistant_blocks_to_hub(
    blocks: &[AnthropicContentBlock],
) -> Result<(Option<MessageContent>, Option<Vec<ToolCall>>)> {
    let mut text_segments = Vec::new();
    let mut rich_parts = Vec::new();
    let mut has_non_text_part = false;
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text, .. } => {
                text_segments.push(text.clone());
                rich_parts.push(ContentPart::Text { text: text.clone() });
            }
            AnthropicContentBlock::Image { source } => {
                has_non_text_part = true;
                rich_parts.push(ContentPart::ImageUrl {
                    image_url: anthropic_source_to_openai_image_url(source)?,
                });
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
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
                    "Anthropic assistant messages cannot contain tool_result blocks when bridging to hub format"
                        .into(),
                ));
            }
        }
    }

    Ok((
        openai_message_content_from_segments(text_segments, rich_parts, has_non_text_part),
        (!tool_calls.is_empty()).then_some(tool_calls),
    ))
}

fn anthropic_optional_content_to_openai(
    content: Option<&AnthropicContent>,
) -> Result<Option<MessageContent>> {
    match content {
        None => Ok(None),
        Some(AnthropicContent::Text(text)) => Ok(Some(MessageContent::Text(text.clone()))),
        Some(AnthropicContent::Blocks(blocks)) => anthropic_blocks_to_openai_content(blocks),
    }
}

fn anthropic_blocks_to_openai_content(
    blocks: &[AnthropicContentBlock],
) -> Result<Option<MessageContent>> {
    let mut text_segments = Vec::new();
    let mut rich_parts = Vec::new();
    let mut has_non_text_part = false;

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text, .. } => {
                text_segments.push(text.clone());
                rich_parts.push(ContentPart::Text { text: text.clone() });
            }
            AnthropicContentBlock::Image { source } => {
                has_non_text_part = true;
                rich_parts.push(ContentPart::ImageUrl {
                    image_url: anthropic_source_to_openai_image_url(source)?,
                });
            }
            AnthropicContentBlock::ToolUse { .. } | AnthropicContentBlock::ToolResult { .. } => {
                return Err(GatewayError::Bridge(
                    "Anthropic content blocks cannot be represented as OpenAI message content"
                        .into(),
                ));
            }
        }
    }

    Ok(openai_message_content_from_segments(
        text_segments,
        rich_parts,
        has_non_text_part,
    ))
}

fn openai_message_content_from_segments(
    text_segments: Vec<String>,
    rich_parts: Vec<ContentPart>,
    has_non_text_part: bool,
) -> Option<MessageContent> {
    if has_non_text_part {
        Some(MessageContent::Parts(rich_parts))
    } else if !text_segments.is_empty() {
        Some(MessageContent::Text(text_segments.join("")))
    } else {
        None
    }
}

fn anthropic_tools_to_openai(tools: &[AnthropicTool]) -> Result<Vec<Tool>> {
    Ok(tools
        .iter()
        .map(|tool| Tool {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: Some(tool.input_schema.clone()),
                strict: None,
            },
        })
        .collect())
}

fn anthropic_tool_choice_to_openai(
    choice: Option<&AnthropicToolChoice>,
) -> Result<Option<ToolChoice>> {
    Ok(match choice {
        None => None,
        Some(AnthropicToolChoice::Auto) => Some(ToolChoice::Mode("auto".into())),
        Some(AnthropicToolChoice::Any) => Some(ToolChoice::Mode("required".into())),
        Some(AnthropicToolChoice::Tool { name }) => Some(ToolChoice::Function {
            r#type: "function".into(),
            function: ToolChoiceFunction { name: name.clone() },
        }),
    })
}

fn stop_sequences_to_openai(stop_sequences: Option<&Vec<String>>) -> Option<StopCondition> {
    match stop_sequences {
        Some(stop_sequences) if stop_sequences.len() == 1 => {
            Some(StopCondition::Single(stop_sequences[0].clone()))
        }
        Some(stop_sequences) if !stop_sequences.is_empty() => {
            Some(StopCondition::Multiple(stop_sequences.clone()))
        }
        _ => None,
    }
}

fn openai_response_to_anthropic(
    response: &ChatCompletionResponse,
) -> Result<AnthropicMessagesResponse> {
    let choice = match response.choices.as_slice() {
        [choice] => choice,
        [] => {
            return Err(GatewayError::Bridge(
                "OpenAI chat response did not include a choice for Anthropic conversion".into(),
            ));
        }
        _ => {
            return Err(GatewayError::Bridge(
                "Anthropic response format cannot represent multiple OpenAI choices".into(),
            ));
        }
    };

    if choice.message.role != "assistant" {
        return Err(GatewayError::Bridge(format!(
            "Anthropic response format requires an assistant message, got {}",
            choice.message.role
        )));
    }

    Ok(AnthropicMessagesResponse {
        id: response.id.clone(),
        r#type: "message".into(),
        role: "assistant".into(),
        content: openai_message_to_anthropic_blocks(&choice.message)?,
        model: response.model.clone(),
        stop_reason: openai_finish_reason_to_anthropic(choice.finish_reason.as_deref()),
        stop_sequence: None,
        usage: openai_usage_to_anthropic(response.usage.as_ref()),
    })
}

fn openai_message_to_anthropic_blocks(message: &ChatMessage) -> Result<Vec<AnthropicContentBlock>> {
    let mut blocks = openai_content_to_anthropic_blocks(message.content.as_ref())?;

    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            if tool_call.r#type != "function" {
                return Err(GatewayError::Bridge(format!(
                    "Anthropic response format only supports function tool calls, got {}",
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
            });
        }
    }

    Ok(blocks)
}

fn openai_content_to_anthropic_blocks(
    content: Option<&MessageContent>,
) -> Result<Vec<AnthropicContentBlock>> {
    match content {
        None => Ok(vec![]),
        Some(MessageContent::Text(text)) => Ok(vec![AnthropicContentBlock::Text {
            text: text.clone(),
            cache_control: None,
        }]),
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(AnthropicContentBlock::Text {
                    text: text.clone(),
                    cache_control: None,
                }),
                ContentPart::ImageUrl { image_url } => Ok(AnthropicContentBlock::Image {
                    source: openai_image_url_to_anthropic_source(&image_url.url)?,
                }),
            })
            .collect(),
    }
}

fn openai_usage_to_anthropic(usage: Option<&ChatCompletionUsage>) -> AnthropicUsage {
    let Some(usage) = usage else {
        return AnthropicUsage::default();
    };

    let cached_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .unwrap_or(0);

    AnthropicUsage {
        input_tokens: usage.prompt_tokens.saturating_sub(cached_tokens),
        output_tokens: usage.completion_tokens,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: cached_tokens,
    }
}

fn openai_finish_reason_to_anthropic(finish_reason: Option<&str>) -> Option<String> {
    finish_reason.map(|reason| match reason {
        "stop" => "end_turn".into(),
        "length" => "max_tokens".into(),
        "tool_calls" => "tool_use".into(),
        other => other.to_string(),
    })
}

fn anthropic_source_to_openai_image_url(source: &ImageSource) -> Result<ImageUrl> {
    if source.r#type != "base64" {
        return Err(GatewayError::Bridge(format!(
            "Anthropic image source type {} is not supported by hub bridging",
            source.r#type
        )));
    }

    Ok(ImageUrl {
        url: format!("data:{};base64,{}", source.media_type, source.data),
        detail: None,
    })
}

fn openai_image_url_to_anthropic_source(url: &str) -> Result<ImageSource> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Err(GatewayError::Bridge(
            "Anthropic format only supports image_url data URLs when bridging from OpenAI".into(),
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

fn hub_message(role: &str, content: Option<MessageContent>) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content,
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::AnthropicMessagesFormat;
    use crate::gateway::{
        error::GatewayError,
        traits::ChatFormat,
        types::{
            anthropic::AnthropicMessagesRequest, common::BridgeContext,
            openai::ChatCompletionResponse,
        },
    };

    #[test]
    fn request_to_hub_maps_system_metadata_tools_and_tool_results() {
        let request: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "system": [{
                "type": "text",
                "text": "You are helpful.",
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "get_weather",
                        "input": {"city": "SF"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "sunny"
                    }]
                }
            ],
            "metadata": {"user_id": "user-123"},
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "tool_choice": {"type": "auto"},
            "top_k": 5,
            "stop_sequences": ["DONE"],
            "stream": true
        }))
        .unwrap();

        let (hub, ctx) = AnthropicMessagesFormat::to_hub(&request).unwrap();

        assert_eq!(hub.model, "claude-3-5-sonnet-20241022");
        assert_eq!(hub.max_tokens, Some(1024));
        assert_eq!(hub.user.as_deref(), Some("user-123"));
        assert_eq!(hub.messages.len(), 3);
        assert_eq!(hub.messages[0].role, "system");
        assert_eq!(hub.messages[1].role, "assistant");
        assert_eq!(hub.messages[2].role, "tool");
        assert_eq!(hub.messages[2].tool_call_id.as_deref(), Some("tool_1"));
        assert!(
            matches!(hub.tool_choice, Some(crate::gateway::types::openai::ToolChoice::Mode(ref mode)) if mode == "auto")
        );
        assert_eq!(hub.tools.as_ref().unwrap()[0].function.name, "get_weather");
        assert_eq!(ctx.passthrough["top_k"], 5);

        let extras = ctx.anthropic_messages_extras.unwrap();
        assert_eq!(extras.metadata.unwrap()["user_id"], "user-123");
        assert_eq!(extras.system_cache_control.unwrap().r#type, "ephemeral");
    }

    #[test]
    fn response_from_hub_maps_text_tool_calls_and_usage() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Calling tool",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"SF\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19,
                "prompt_tokens_details": {"cached_tokens": 2}
            }
        }))
        .unwrap();

        let bridged =
            AnthropicMessagesFormat::from_hub(&response, &BridgeContext::default()).unwrap();

        assert_eq!(bridged.id, "chatcmpl-123");
        assert_eq!(bridged.r#type, "message");
        assert_eq!(bridged.role, "assistant");
        assert_eq!(bridged.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(bridged.usage.input_tokens, 10);
        assert_eq!(bridged.usage.output_tokens, 7);
        assert_eq!(bridged.usage.cache_read_input_tokens, 2);
        assert!(matches!(
            &bridged.content[0],
            crate::gateway::types::anthropic::AnthropicContentBlock::Text { text, .. }
                if text == "Calling tool"
        ));
        assert!(matches!(
            &bridged.content[1],
            crate::gateway::types::anthropic::AnthropicContentBlock::ToolUse { name, .. }
                if name == "get_weather"
        ));
    }

    #[test]
    fn hub_stream_bridge_is_not_implemented_yet() {
        let chunk: crate::gateway::types::openai::ChatCompletionChunk =
            serde_json::from_value(json!({
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-test",
                "choices": []
            }))
            .unwrap();

        let result =
            AnthropicMessagesFormat::from_hub_stream(&chunk, &mut (), &BridgeContext::default());

        assert!(matches!(
            result,
            Err(GatewayError::Bridge(message))
                if message.contains("hub streaming bridge")
        ));
    }
}
