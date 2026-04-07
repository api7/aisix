use std::time::{SystemTime, UNIX_EPOCH};

use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::gateway::{
    error::{GatewayError, Result},
    traits::{ChatStreamState, ToolCallAccumulator},
    types::{
        anthropic::{
            AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicMessagesRequest,
            AnthropicMessagesResponse, AnthropicMetadata, AnthropicStreamEvent, AnthropicTool,
            AnthropicToolChoice, AnthropicUsage, ContentDelta, ImageSource, SystemBlock,
            SystemPrompt,
        },
        openai::{
            ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
            ChatCompletionChunkDelta, ChatCompletionRequest, ChatCompletionResponse,
            ChatCompletionUsage, ChunkFunctionCall, ChunkToolCall, ContentPart, FunctionCall,
            ImageUrl, MessageContent, PromptTokensDetails, StopCondition, Tool, ToolCall,
            ToolChoice,
        },
    },
};

const DEFAULT_MAX_TOKENS: u32 = 4096;

pub(crate) fn openai_to_anthropic_request(
    request: &ChatCompletionRequest,
) -> Result<AnthropicMessagesRequest> {
    if let Some(n) = request.n
        && n != 1
    {
        return Err(GatewayError::Bridge(format!(
            "Anthropic provider only supports n=1, got {}",
            n
        )));
    }

    let (messages, system) = openai_messages_to_anthropic(&request.messages)?;
    let tools = request
        .tools
        .as_ref()
        .map(|tools| openai_tools_to_anthropic(tools))
        .transpose()?;
    let tool_choice = match (request.tool_choice.as_ref(), tools.as_ref()) {
        (Some(_), None) => {
            return Err(GatewayError::Bridge(
                "Anthropic provider requires tools when tool_choice is set".into(),
            ));
        }
        (Some(choice), Some(_)) => openai_tool_choice_to_anthropic(choice)?,
        (None, _) => None,
    };

    Ok(AnthropicMessagesRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request
            .max_tokens
            .or(request.max_completion_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS),
        system,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: None,
        stop_sequences: openai_stop_sequences(&request.stop),
        stream: request.stream,
        metadata: request.user.clone().map(|user_id| AnthropicMetadata {
            user_id: Some(user_id),
        }),
        tools,
        tool_choice,
    })
}

pub(crate) fn anthropic_to_openai_response(
    response: &AnthropicMessagesResponse,
) -> Result<ChatCompletionResponse> {
    let message = anthropic_blocks_to_openai_message(&response.content)?;
    let usage = anthropic_usage_to_openai_usage(&response.usage);

    Ok(ChatCompletionResponse {
        id: response.id.clone(),
        object: "chat.completion".into(),
        created: now_unix_secs(),
        model: response.model.clone(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message,
            finish_reason: map_anthropic_stop_reason(response.stop_reason.as_deref()),
        }],
        usage: Some(usage),
        system_fingerprint: None,
    })
}

pub(crate) fn parse_anthropic_sse_to_openai(
    raw: &str,
    state: &mut ChatStreamState,
) -> Result<Vec<ChatCompletionChunk>> {
    let Some(event) = parse_sse_data::<AnthropicStreamEvent>(raw)? else {
        return Ok(vec![]);
    };

    match event {
        AnthropicStreamEvent::MessageStart { message } => {
            state.response_id = Some(message.id.clone());
            state.response_model = Some(message.model.clone());
            state.response_created = Some(now_unix_secs());
            state.input_tokens = Some(message.usage.input_tokens);

            Ok(vec![ChatCompletionChunk {
                id: message.id,
                object: "chat.completion.chunk".into(),
                created: state.response_created.unwrap_or_default(),
                model: message.model,
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatCompletionChunkDelta {
                        role: Some("assistant".into()),
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
                system_fingerprint: None,
            }])
        }
        AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        } => match content_block {
            AnthropicContentBlock::ToolUse { id, name, input } => {
                let initial_arguments = initial_tool_arguments(&input)?;
                let accumulator = state
                    .tool_call_accumulators
                    .entry((0, index))
                    .or_insert_with(|| ToolCallAccumulator {
                        id: Some(id.clone()),
                        kind: Some("function".into()),
                        name: Some(name.clone()),
                        arguments: initial_arguments.clone().unwrap_or_default(),
                    });
                accumulator.id = Some(id.clone());
                accumulator.kind = Some("function".into());
                accumulator.name = Some(name.clone());

                Ok(vec![build_stream_chunk(
                    state,
                    ChatCompletionChunkDelta {
                        role: None,
                        content: None,
                        tool_calls: Some(vec![ChunkToolCall {
                            index,
                            id: Some(id),
                            r#type: Some("function".into()),
                            function: Some(ChunkFunctionCall {
                                name: Some(name),
                                arguments: initial_arguments,
                            }),
                        }]),
                    },
                    None,
                    None,
                )?])
            }
            AnthropicContentBlock::Text { .. }
            | AnthropicContentBlock::Image { .. }
            | AnthropicContentBlock::ToolResult { .. } => Ok(vec![]),
        },
        AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
            ContentDelta::TextDelta { text } => Ok(vec![build_stream_chunk(
                state,
                ChatCompletionChunkDelta {
                    role: None,
                    content: Some(text),
                    tool_calls: None,
                },
                None,
                None,
            )?]),
            ContentDelta::InputJsonDelta { partial_json } => {
                let accumulator = state.tool_call_accumulators.entry((0, index)).or_default();
                accumulator.arguments.push_str(&partial_json);

                Ok(vec![build_stream_chunk(
                    state,
                    ChatCompletionChunkDelta {
                        role: None,
                        content: None,
                        tool_calls: Some(vec![ChunkToolCall {
                            index,
                            id: None,
                            r#type: None,
                            function: Some(ChunkFunctionCall {
                                name: None,
                                arguments: Some(partial_json),
                            }),
                        }]),
                    },
                    None,
                    None,
                )?])
            }
        },
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            state.output_tokens = Some(usage.output_tokens);
            if usage.input_tokens > 0 {
                state.input_tokens = Some(usage.input_tokens);
            }

            let usage = match (state.input_tokens, state.output_tokens) {
                (Some(input_tokens), Some(output_tokens)) => {
                    Some(stream_usage_to_openai_usage(input_tokens, output_tokens))
                }
                _ => None,
            };

            Ok(vec![build_stream_chunk(
                state,
                ChatCompletionChunkDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                },
                map_anthropic_stop_reason(delta.stop_reason.as_deref()),
                usage,
            )?])
        }
        AnthropicStreamEvent::ContentBlockStop { .. }
        | AnthropicStreamEvent::MessageStop
        | AnthropicStreamEvent::Ping => Ok(vec![]),
        AnthropicStreamEvent::Error { error } => Err(GatewayError::Stream(format!(
            "anthropic error {}: {}",
            error.r#type, error.message
        ))),
    }
}

pub(crate) fn parse_anthropic_native_sse(raw: &str) -> Result<Vec<AnthropicStreamEvent>> {
    Ok(parse_sse_data::<AnthropicStreamEvent>(raw)?
        .into_iter()
        .collect())
}

fn openai_messages_to_anthropic(
    messages: &[crate::gateway::types::openai::ChatMessage],
) -> Result<(Vec<AnthropicMessage>, Option<SystemPrompt>)> {
    let mut anthropic_messages = Vec::new();
    let mut system_blocks = Vec::new();

    for message in messages {
        match message.role.as_str() {
            "system" | "developer" => {
                system_blocks.extend(system_message_to_blocks(message)?);
            }
            "user" => {
                anthropic_messages.push(AnthropicMessage {
                    role: "user".into(),
                    content: openai_message_content_to_anthropic(message.content.as_ref())?,
                });
            }
            "assistant" => {
                anthropic_messages.push(openai_assistant_message_to_anthropic(message)?);
            }
            "tool" => {
                anthropic_messages.push(openai_tool_message_to_anthropic(message)?);
            }
            other => {
                return Err(GatewayError::Bridge(format!(
                    "Anthropic provider does not support message role {}",
                    other
                )));
            }
        }
    }

    let system = if system_blocks.is_empty() {
        None
    } else if system_blocks.len() == 1 && system_blocks[0].cache_control.is_none() {
        Some(SystemPrompt::Text(system_blocks[0].text.clone()))
    } else {
        Some(SystemPrompt::Blocks(system_blocks))
    };

    Ok((anthropic_messages, system))
}

fn system_message_to_blocks(
    message: &crate::gateway::types::openai::ChatMessage,
) -> Result<Vec<SystemBlock>> {
    let Some(content) = message.content.as_ref() else {
        return Ok(vec![]);
    };

    match content {
        MessageContent::Text(text) => Ok(vec![SystemBlock {
            r#type: "text".into(),
            text: text.clone(),
            cache_control: None,
        }]),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(SystemBlock {
                    r#type: "text".into(),
                    text: text.clone(),
                    cache_control: None,
                }),
                ContentPart::ImageUrl { .. } => Err(GatewayError::Bridge(
                    "Anthropic provider does not support image content in system messages".into(),
                )),
            })
            .collect(),
    }
}

fn openai_assistant_message_to_anthropic(
    message: &crate::gateway::types::openai::ChatMessage,
) -> Result<AnthropicMessage> {
    let mut blocks = content_to_anthropic_blocks(message.content.as_ref())?;

    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            blocks.push(AnthropicContentBlock::ToolUse {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input: serde_json::from_str(&tool_call.function.arguments).map_err(|error| {
                    GatewayError::Bridge(format!(
                        "assistant tool call arguments are not valid JSON: {}",
                        error
                    ))
                })?,
            });
        }
    }

    let content = if blocks.is_empty() {
        AnthropicContent::Text(String::new())
    } else {
        normalize_anthropic_content(blocks)
    };

    Ok(AnthropicMessage {
        role: "assistant".into(),
        content,
    })
}

fn openai_tool_message_to_anthropic(
    message: &crate::gateway::types::openai::ChatMessage,
) -> Result<AnthropicMessage> {
    let tool_use_id = message.tool_call_id.clone().ok_or_else(|| {
        GatewayError::Bridge("tool message missing tool_call_id for Anthropic conversion".into())
    })?;

    Ok(AnthropicMessage {
        role: "user".into(),
        content: AnthropicContent::Blocks(vec![AnthropicContentBlock::ToolResult {
            tool_use_id,
            content: match message.content.as_ref() {
                Some(content) => Some(openai_message_content_to_anthropic(Some(content))?),
                None => None,
            },
            is_error: None,
        }]),
    })
}

fn openai_message_content_to_anthropic(
    content: Option<&MessageContent>,
) -> Result<AnthropicContent> {
    Ok(match content {
        None => AnthropicContent::Text(String::new()),
        Some(MessageContent::Text(text)) => AnthropicContent::Text(text.clone()),
        Some(MessageContent::Parts(_)) => {
            normalize_anthropic_content(content_to_anthropic_blocks(content)?)
        }
    })
}

fn content_to_anthropic_blocks(
    content: Option<&MessageContent>,
) -> Result<Vec<AnthropicContentBlock>> {
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
                }),
            })
            .collect(),
    }
}

fn normalize_anthropic_content(blocks: Vec<AnthropicContentBlock>) -> AnthropicContent {
    if let [
        AnthropicContentBlock::Text {
            text,
            cache_control,
        },
    ] = blocks.as_slice()
        && cache_control.is_none()
    {
        return AnthropicContent::Text(text.clone());
    }

    AnthropicContent::Blocks(blocks)
}

fn openai_tools_to_anthropic(tools: &[Tool]) -> Result<Vec<AnthropicTool>> {
    tools
        .iter()
        .map(|tool| {
            if tool.r#type != "function" {
                return Err(GatewayError::Bridge(format!(
                    "Anthropic provider does not support tool type {}",
                    tool.r#type
                )));
            }

            Ok(AnthropicTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                input_schema: tool
                    .function
                    .parameters
                    .clone()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            })
        })
        .collect()
}

fn openai_tool_choice_to_anthropic(choice: &ToolChoice) -> Result<Option<AnthropicToolChoice>> {
    match choice {
        ToolChoice::Mode(mode) => match mode.as_str() {
            "auto" => Ok(Some(AnthropicToolChoice::Auto)),
            "required" => Ok(Some(AnthropicToolChoice::Any)),
            "none" => Err(GatewayError::Bridge(
                "Anthropic provider cannot faithfully represent tool_choice=none when tools are present"
                    .into(),
            )),
            other => Err(GatewayError::Bridge(format!(
                "unsupported OpenAI tool_choice mode {} for Anthropic provider",
                other
            ))),
        },
        ToolChoice::Function { function, .. } => Ok(Some(AnthropicToolChoice::Tool {
            name: function.name.clone(),
        })),
    }
}

fn openai_stop_sequences(stop: &Option<StopCondition>) -> Option<Vec<String>> {
    match stop {
        Some(StopCondition::Single(stop)) => Some(vec![stop.clone()]),
        Some(StopCondition::Multiple(stops)) if !stops.is_empty() => Some(stops.clone()),
        _ => None,
    }
}

fn anthropic_blocks_to_openai_message(
    blocks: &[AnthropicContentBlock],
) -> Result<crate::gateway::types::openai::ChatMessage> {
    let mut text_segments = Vec::new();
    let mut rich_parts = Vec::new();
    let mut tool_calls = Vec::new();
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
                    image_url: ImageUrl {
                        url: format!("data:{};base64,{}", source.media_type, source.data),
                        detail: None,
                    },
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
                    "assistant response contained unsupported tool_result block".into(),
                ));
            }
        }
    }

    Ok(crate::gateway::types::openai::ChatMessage {
        role: "assistant".into(),
        content: if has_non_text_part {
            Some(MessageContent::Parts(rich_parts))
        } else if !text_segments.is_empty() {
            Some(MessageContent::Text(text_segments.join("")))
        } else {
            None
        },
        name: None,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
    })
}

fn anthropic_usage_to_openai_usage(usage: &AnthropicUsage) -> ChatCompletionUsage {
    let prompt_tokens =
        usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens;
    let cached_tokens = usage.cache_creation_input_tokens + usage.cache_read_input_tokens;

    ChatCompletionUsage {
        prompt_tokens,
        completion_tokens: usage.output_tokens,
        total_tokens: prompt_tokens + usage.output_tokens,
        prompt_tokens_details: (cached_tokens > 0).then_some(PromptTokensDetails {
            cached_tokens: Some(cached_tokens),
            audio_tokens: None,
        }),
        completion_tokens_details: None,
    }
}

fn stream_usage_to_openai_usage(input_tokens: u32, output_tokens: u32) -> ChatCompletionUsage {
    ChatCompletionUsage {
        prompt_tokens: input_tokens,
        completion_tokens: output_tokens,
        total_tokens: input_tokens + output_tokens,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    }
}

fn map_anthropic_stop_reason(stop_reason: Option<&str>) -> Option<String> {
    stop_reason.map(|reason| match reason {
        "end_turn" | "stop_sequence" => "stop".into(),
        "max_tokens" => "length".into(),
        "tool_use" => "tool_calls".into(),
        other => other.to_string(),
    })
}

fn parse_sse_data<T: DeserializeOwned>(raw: &str) -> Result<Option<T>> {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed == "[DONE]"
        || trimmed == "data: [DONE]"
        || trimmed.starts_with(':')
        || trimmed.starts_with("event:")
        || trimmed.starts_with("id:")
        || trimmed.starts_with("retry:")
    {
        return Ok(None);
    }

    let Some(line) = trimmed.strip_prefix("data:") else {
        return Ok(None);
    };

    let payload = line.trim_start();
    if payload.is_empty() || payload == "[DONE]" {
        return Ok(None);
    }

    serde_json::from_str(payload)
        .map(Some)
        .map_err(|error| GatewayError::Transform(error.to_string()))
}

fn build_stream_chunk(
    state: &ChatStreamState,
    delta: ChatCompletionChunkDelta,
    finish_reason: Option<String>,
    usage: Option<ChatCompletionUsage>,
) -> Result<ChatCompletionChunk> {
    Ok(ChatCompletionChunk {
        id: state.response_id.clone().ok_or_else(|| {
            GatewayError::Stream("anthropic stream emitted a delta before message_start".into())
        })?,
        object: "chat.completion.chunk".into(),
        created: state.response_created.ok_or_else(|| {
            GatewayError::Stream("anthropic stream missing response_created metadata".into())
        })?,
        model: state.response_model.clone().ok_or_else(|| {
            GatewayError::Stream("anthropic stream missing response_model metadata".into())
        })?,
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage,
        system_fingerprint: None,
    })
}

fn initial_tool_arguments(input: &Value) -> Result<Option<String>> {
    if input.is_null() {
        return Ok(None);
    }

    if let Some(object) = input.as_object()
        && object.is_empty()
    {
        return Ok(None);
    }

    if let Some(array) = input.as_array()
        && array.is_empty()
    {
        return Ok(None);
    }

    let serialized =
        serde_json::to_string(input).map_err(|error| GatewayError::Transform(error.to_string()))?;
    if serialized == "\"\"" {
        Ok(None)
    } else {
        Ok(Some(serialized))
    }
}

fn image_url_to_source(url: &str) -> Result<ImageSource> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Err(GatewayError::Bridge(
            "Anthropic provider only supports image_url data URLs for image content".into(),
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

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        anthropic_to_openai_response, openai_to_anthropic_request, parse_anthropic_native_sse,
        parse_anthropic_sse_to_openai,
    };
    use crate::gateway::{
        traits::ChatStreamState,
        types::{
            anthropic::{AnthropicMessagesResponse, AnthropicStreamEvent},
            openai::ChatCompletionRequest,
        },
    };

    #[test]
    fn transform_request_hoists_system_and_maps_tools() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the weather?"}
            ],
            "max_completion_tokens": 512,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }],
            "tool_choice": "auto",
            "user": "user-123"
        }))
        .unwrap();

        let anthropic = openai_to_anthropic_request(&request).unwrap();

        assert_eq!(anthropic.model, "claude-3-5-sonnet-20241022");
        assert_eq!(anthropic.max_tokens, 512);
        assert!(
            matches!(anthropic.system, Some(crate::gateway::types::anthropic::SystemPrompt::Text(ref s)) if s == "You are helpful.")
        );
        assert_eq!(anthropic.messages.len(), 1);
        assert_eq!(anthropic.messages[0].role, "user");
        assert_eq!(anthropic.tools.as_ref().unwrap()[0].name, "get_weather");
        assert!(matches!(
            anthropic.tool_choice,
            Some(crate::gateway::types::anthropic::AnthropicToolChoice::Auto)
        ));
        assert_eq!(
            anthropic.metadata.as_ref().unwrap().user_id.as_deref(),
            Some("user-123")
        );
    }

    #[test]
    fn transform_response_maps_tool_use_and_cached_tokens() {
        let response: AnthropicMessagesResponse = serde_json::from_value(json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "SF"}}
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 2
            }
        }))
        .unwrap();

        let openai = anthropic_to_openai_response(&response).unwrap();

        assert_eq!(openai.id, "msg_123");
        assert!(openai.created > 0);
        assert_eq!(
            openai.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        assert!(matches!(
            openai.choices[0].message.content.as_ref(),
            Some(crate::gateway::types::openai::MessageContent::Text(text)) if text == "Let me check."
        ));
        let tool_call = &openai.choices[0].message.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tool_call.id, "tu_1");
        assert_eq!(tool_call.function.name, "get_weather");
        assert_eq!(openai.usage.as_ref().unwrap().prompt_tokens, 15);
        assert_eq!(
            openai
                .usage
                .as_ref()
                .unwrap()
                .prompt_tokens_details
                .as_ref()
                .unwrap()
                .cached_tokens,
            Some(5)
        );
    }

    #[test]
    fn transform_stream_chunk_maps_text_and_tool_deltas() {
        let mut state = ChatStreamState::default();

        let start = parse_anthropic_sse_to_openai(
            r#"data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":7}}}"#,
            &mut state,
        )
        .unwrap();
        let tool_start = parse_anthropic_sse_to_openai(
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"get_weather","input":{}}}"#,
            &mut state,
        )
        .unwrap();
        let tool_delta = parse_anthropic_sse_to_openai(
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"SF\"}"}}"#,
            &mut state,
        )
        .unwrap();
        let finish = parse_anthropic_sse_to_openai(
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":11,"input_tokens":7}}"#,
            &mut state,
        )
        .unwrap();

        assert_eq!(start[0].choices[0].delta.role.as_deref(), Some("assistant"));
        let started_tool = tool_start[0].choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(started_tool[0].index, 1);
        assert_eq!(started_tool[0].id.as_deref(), Some("tu_1"));
        assert_eq!(
            started_tool[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(
            tool_delta[0].choices[0].delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .arguments
                .as_deref(),
            Some("{\"city\":\"SF\"}")
        );
        assert_eq!(
            finish[0].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        assert_eq!(finish[0].usage.as_ref().unwrap().total_tokens, 18);
    }

    #[test]
    fn native_stream_parser_ignores_control_lines() {
        assert!(
            parse_anthropic_native_sse("event: message_start")
                .unwrap()
                .is_empty()
        );
        assert!(
            parse_anthropic_native_sse("data: [DONE]")
                .unwrap()
                .is_empty()
        );

        let events = parse_anthropic_native_sse(r#"data: {"type":"ping"}"#).unwrap();
        assert!(matches!(events.as_slice(), [AnthropicStreamEvent::Ping]));
    }

    #[test]
    fn transform_request_rejects_unsupported_n_values() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 2
        }))
        .unwrap();

        let error = openai_to_anthropic_request(&request).unwrap_err();
        assert!(matches!(
            error,
            crate::gateway::error::GatewayError::Bridge(message)
                if message.contains("n=1")
                    && message.contains('2')
        ));
    }

    #[test]
    fn transform_request_rejects_tool_choice_without_tools() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": "auto"
        }))
        .unwrap();

        let error = openai_to_anthropic_request(&request).unwrap_err();
        assert!(matches!(
            error,
            crate::gateway::error::GatewayError::Bridge(message)
                if message.contains("requires tools")
        ));
    }
}
