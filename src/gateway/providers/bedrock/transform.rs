use std::time::{SystemTime, UNIX_EPOCH};

use serde::de::DeserializeOwned;
use serde_json::json;
use uuid::Uuid;

use crate::gateway::{
    error::{GatewayError, Result},
    traits::{ChatStreamState, ToolCallAccumulator},
    types::{
        bedrock::{
            BedrockContentBlock, BedrockContentBlockDeltaEvent, BedrockContentBlockStartEvent,
            BedrockContentBlockStopEvent, BedrockConverseRequest, BedrockConverseResponse,
            BedrockConverseStreamFrame, BedrockConverseStreamMetadataEvent, BedrockEmptyObject,
            BedrockInferenceConfig, BedrockMessage, BedrockMessageStartEvent,
            BedrockMessageStopEvent, BedrockRole, BedrockSpecificToolChoice,
            BedrockSystemContentBlock, BedrockTool, BedrockToolChoice, BedrockToolConfig,
            BedrockToolInputSchema, BedrockToolResultBlock, BedrockToolResultContentBlock,
            BedrockToolSpecification, BedrockToolUseBlock, BedrockUsage,
        },
        openai::{
            ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
            ChatCompletionChunkDelta, ChatCompletionRequest, ChatCompletionResponse,
            ChatCompletionUsage, ChatMessage, ChunkFunctionCall, ChunkToolCall, ContentPart,
            FunctionCall, MessageContent, StopCondition, Tool, ToolCall, ToolChoice,
        },
    },
};

pub(super) fn openai_to_bedrock_request(
    request: &ChatCompletionRequest,
) -> Result<BedrockConverseRequest> {
    if let Some(count) = request.n
        && count != 1
    {
        return Err(GatewayError::Bridge(
            "bedrock converse only supports n = 1".into(),
        ));
    }

    let mut messages = Vec::new();
    let mut system = Vec::new();

    for message in &request.messages {
        match message.role.as_str() {
            "system" | "developer" => {
                system.extend(system_blocks_from_content(
                    message.content.as_ref(),
                    message.role.as_str(),
                )?);
            }
            "user" => messages.push(BedrockMessage {
                role: BedrockRole::User,
                content: content_blocks_from_message_content(
                    message.content.as_ref(),
                    message.role.as_str(),
                    false,
                )?,
            }),
            "assistant" => messages.push(assistant_message_to_bedrock(message)?),
            "tool" => messages.push(tool_message_to_bedrock(message)?),
            other => {
                return Err(GatewayError::Bridge(format!(
                    "bedrock converse does not support message role {other}"
                )));
            }
        }
    }

    Ok(BedrockConverseRequest {
        messages,
        system: (!system.is_empty()).then_some(system),
        inference_config: build_inference_config(request),
        tool_config: build_tool_config(request.tools.as_ref(), request.tool_choice.as_ref())?,
    })
}

pub(super) fn bedrock_to_openai_response(
    request: &ChatCompletionRequest,
    response: BedrockConverseResponse,
) -> Result<ChatCompletionResponse> {
    let (role, content, tool_calls) = bedrock_message_to_openai(&response.output.message)?;
    let usage = response.usage.map(|usage| ChatCompletionUsage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });

    Ok(ChatCompletionResponse {
        id: format!("bedrock-{}", Uuid::new_v4()),
        object: "chat.completion".into(),
        created: current_unix_timestamp()?,
        model: request.model.clone(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role,
                content,
                name: None,
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                tool_call_id: None,
            },
            finish_reason: response.stop_reason.as_deref().map(map_finish_reason),
        }],
        usage,
        system_fingerprint: None,
    })
}

pub(super) fn parse_bedrock_stream_to_openai(
    raw: &str,
    state: &mut ChatStreamState,
) -> Result<Vec<ChatCompletionChunk>> {
    let frame: BedrockConverseStreamFrame =
        serde_json::from_str(raw).map_err(|error| GatewayError::Transform(error.to_string()))?;

    match frame.event_type.as_str() {
        "messageStart" => {
            let event: BedrockMessageStartEvent =
                deserialize_bedrock_stream_payload(frame.payload)?;
            state
                .response_id
                .get_or_insert_with(|| format!("bedrock-{}", Uuid::new_v4()));
            if state.response_created.is_none() {
                state.response_created = Some(current_unix_timestamp()?);
            }

            Ok(vec![build_bedrock_stream_chunk(
                state,
                ChatCompletionChunkDelta {
                    role: Some(
                        match event.role {
                            BedrockRole::Assistant => "assistant",
                            BedrockRole::User => "user",
                        }
                        .to_string(),
                    ),
                    content: None,
                    tool_calls: None,
                },
                None,
                None,
            )?])
        }
        "contentBlockStart" => {
            let event: BedrockContentBlockStartEvent =
                deserialize_bedrock_stream_payload(frame.payload)?;
            let Some(tool_use) = event.start.and_then(|start| start.tool_use) else {
                return Ok(vec![]);
            };

            let accumulator = state
                .tool_call_accumulators
                .entry((0, event.content_block_index))
                .or_insert_with(|| ToolCallAccumulator {
                    id: Some(tool_use.tool_use_id.clone()),
                    kind: Some("function".into()),
                    name: Some(tool_use.name.clone()),
                    arguments: String::new(),
                });
            accumulator.id = Some(tool_use.tool_use_id.clone());
            accumulator.kind = Some("function".into());
            accumulator.name = Some(tool_use.name.clone());

            Ok(vec![build_bedrock_stream_chunk(
                state,
                ChatCompletionChunkDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![ChunkToolCall {
                        index: event.content_block_index,
                        id: Some(tool_use.tool_use_id),
                        r#type: Some("function".into()),
                        function: Some(ChunkFunctionCall {
                            name: Some(tool_use.name),
                            arguments: None,
                        }),
                    }]),
                },
                None,
                None,
            )?])
        }
        "contentBlockDelta" => {
            let event: BedrockContentBlockDeltaEvent =
                deserialize_bedrock_stream_payload(frame.payload)?;

            if let Some(text) = event.delta.text {
                return Ok(vec![build_bedrock_stream_chunk(
                    state,
                    ChatCompletionChunkDelta {
                        role: None,
                        content: Some(text),
                        tool_calls: None,
                    },
                    None,
                    None,
                )?]);
            }

            let Some(tool_use) = event.delta.tool_use else {
                return Ok(vec![]);
            };

            let accumulator = state
                .tool_call_accumulators
                .entry((0, event.content_block_index))
                .or_default();
            accumulator.arguments.push_str(&tool_use.input);

            Ok(vec![build_bedrock_stream_chunk(
                state,
                ChatCompletionChunkDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![ChunkToolCall {
                        index: event.content_block_index,
                        id: None,
                        r#type: None,
                        function: Some(ChunkFunctionCall {
                            name: None,
                            arguments: Some(tool_use.input),
                        }),
                    }]),
                },
                None,
                None,
            )?])
        }
        "contentBlockStop" => {
            let _: BedrockContentBlockStopEvent =
                deserialize_bedrock_stream_payload(frame.payload)?;
            Ok(vec![])
        }
        "messageStop" => {
            let event: BedrockMessageStopEvent = deserialize_bedrock_stream_payload(frame.payload)?;
            let finish_reason = event.stop_reason.as_deref().map(map_finish_reason);
            if finish_reason.is_none() {
                return Ok(vec![]);
            }

            Ok(vec![build_bedrock_stream_chunk(
                state,
                ChatCompletionChunkDelta::default(),
                finish_reason,
                None,
            )?])
        }
        "metadata" => {
            let event: BedrockConverseStreamMetadataEvent =
                deserialize_bedrock_stream_payload(frame.payload)?;
            let Some(usage) = event.usage else {
                return Ok(vec![]);
            };

            state.input_tokens = Some(usage.input_tokens);
            state.output_tokens = Some(usage.output_tokens);

            Ok(vec![build_bedrock_stream_chunk(
                state,
                ChatCompletionChunkDelta::default(),
                None,
                Some(bedrock_usage_to_openai_usage(&usage)),
            )?])
        }
        _ => Ok(vec![]),
    }
}

fn build_inference_config(request: &ChatCompletionRequest) -> Option<BedrockInferenceConfig> {
    let stop_sequences = match request.stop.as_ref() {
        Some(StopCondition::Single(stop)) => Some(vec![stop.clone()]),
        Some(StopCondition::Multiple(stops)) => Some(stops.clone()),
        None => None,
    };

    let config = BedrockInferenceConfig {
        max_tokens: request.max_completion_tokens.or(request.max_tokens),
        temperature: request.temperature,
        top_p: request.top_p,
        stop_sequences,
    };

    (config.max_tokens.is_some()
        || config.temperature.is_some()
        || config.top_p.is_some()
        || config.stop_sequences.is_some())
    .then_some(config)
}

fn build_tool_config(
    tools: Option<&Vec<Tool>>,
    tool_choice: Option<&ToolChoice>,
) -> Result<Option<BedrockToolConfig>> {
    let Some(tools) = tools else {
        if tool_choice.is_some() {
            return Err(GatewayError::Bridge(
                "bedrock converse tool_choice requires tools".into(),
            ));
        }

        return Ok(None);
    };

    let tools = tools
        .iter()
        .map(openai_tool_to_bedrock)
        .collect::<Result<Vec<_>>>()?;
    let tool_choice = tool_choice.map(openai_tool_choice_to_bedrock).transpose()?;

    Ok(Some(BedrockToolConfig { tools, tool_choice }))
}

fn openai_tool_to_bedrock(tool: &Tool) -> Result<BedrockTool> {
    if tool.r#type != "function" {
        return Err(GatewayError::Bridge(format!(
            "bedrock converse only supports function tools, got {}",
            tool.r#type
        )));
    }

    Ok(BedrockTool {
        tool_spec: BedrockToolSpecification {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            input_schema: BedrockToolInputSchema {
                json: tool
                    .function
                    .parameters
                    .clone()
                    .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
            },
        },
    })
}

fn openai_tool_choice_to_bedrock(tool_choice: &ToolChoice) -> Result<BedrockToolChoice> {
    match tool_choice {
        ToolChoice::Mode(mode) => match mode.as_str() {
            "auto" => Ok(BedrockToolChoice::Auto {
                auto: BedrockEmptyObject::default(),
            }),
            "required" => Ok(BedrockToolChoice::Any {
                any: BedrockEmptyObject::default(),
            }),
            "none" => Err(GatewayError::Bridge(
                "bedrock converse does not support tool_choice = none".into(),
            )),
            other => Err(GatewayError::Bridge(format!(
                "unsupported tool_choice mode for bedrock converse: {other}"
            ))),
        },
        ToolChoice::Function { r#type, function } => {
            if r#type != "function" {
                return Err(GatewayError::Bridge(format!(
                    "bedrock converse only supports function tool choices, got {}",
                    r#type
                )));
            }

            Ok(BedrockToolChoice::Tool {
                tool: BedrockSpecificToolChoice {
                    name: function.name.clone(),
                },
            })
        }
    }
}

fn assistant_message_to_bedrock(message: &ChatMessage) -> Result<BedrockMessage> {
    let mut content =
        content_blocks_from_message_content(message.content.as_ref(), message.role.as_str(), true)?;

    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            content.push(BedrockContentBlock::ToolUse {
                tool_use: BedrockToolUseBlock {
                    tool_use_id: tool_call.id.clone(),
                    name: tool_call.function.name.clone(),
                    input: serde_json::from_str(&tool_call.function.arguments).map_err(
                        |error| {
                            GatewayError::Bridge(format!(
                                "invalid assistant tool call arguments for {}: {}",
                                tool_call.function.name, error
                            ))
                        },
                    )?,
                },
            });
        }
    }

    if content.is_empty() {
        return Err(GatewayError::Bridge(
            "bedrock converse assistant messages require text or tool_calls".into(),
        ));
    }

    Ok(BedrockMessage {
        role: BedrockRole::Assistant,
        content,
    })
}

fn tool_message_to_bedrock(message: &ChatMessage) -> Result<BedrockMessage> {
    let tool_use_id = message.tool_call_id.clone().ok_or_else(|| {
        GatewayError::Bridge("bedrock converse tool messages require tool_call_id".into())
    })?;

    Ok(BedrockMessage {
        role: BedrockRole::User,
        content: vec![BedrockContentBlock::ToolResult {
            tool_result: BedrockToolResultBlock {
                tool_use_id,
                content: tool_result_content_from_message(message.content.as_ref())?,
                status: None,
            },
        }],
    })
}

fn system_blocks_from_content(
    content: Option<&MessageContent>,
    role: &str,
) -> Result<Vec<BedrockSystemContentBlock>> {
    let texts = text_fragments_from_message_content(content, role, false)?;
    Ok(texts
        .into_iter()
        .map(|text| BedrockSystemContentBlock { text })
        .collect())
}

fn content_blocks_from_message_content(
    content: Option<&MessageContent>,
    role: &str,
    allow_empty: bool,
) -> Result<Vec<BedrockContentBlock>> {
    let texts = text_fragments_from_message_content(content, role, allow_empty)?;
    Ok(texts
        .into_iter()
        .map(|text| BedrockContentBlock::Text { text })
        .collect())
}

fn text_fragments_from_message_content(
    content: Option<&MessageContent>,
    role: &str,
    allow_empty: bool,
) -> Result<Vec<String>> {
    let Some(content) = content else {
        if allow_empty {
            return Ok(vec![]);
        }

        return Err(GatewayError::Bridge(format!(
            "bedrock converse {role} messages require content"
        )));
    };

    match content {
        MessageContent::Text(text) => Ok(vec![text.clone()]),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(text.clone()),
                ContentPart::ImageUrl { .. } => Err(GatewayError::Bridge(format!(
                    "bedrock converse {role} messages do not support image content yet"
                ))),
            })
            .collect(),
    }
}

fn tool_result_content_from_message(
    content: Option<&MessageContent>,
) -> Result<Vec<BedrockToolResultContentBlock>> {
    let Some(content) = content else {
        return Ok(vec![BedrockToolResultContentBlock::Text {
            text: String::new(),
        }]);
    };

    let text = match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(text.as_str()),
                ContentPart::ImageUrl { .. } => Err(GatewayError::Bridge(
                    "bedrock converse tool messages do not support image content".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?
            .join(""),
    };

    if let Ok(json) = serde_json::from_str(&text) {
        return Ok(vec![BedrockToolResultContentBlock::Json { json }]);
    }

    Ok(vec![BedrockToolResultContentBlock::Text { text }])
}

fn bedrock_message_to_openai(
    message: &BedrockMessage,
) -> Result<(String, Option<MessageContent>, Vec<ToolCall>)> {
    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &message.content {
        match block {
            BedrockContentBlock::Text { text } => texts.push(text.clone()),
            BedrockContentBlock::ToolUse { tool_use } => tool_calls.push(ToolCall {
                id: tool_use.tool_use_id.clone(),
                r#type: "function".into(),
                function: FunctionCall {
                    name: tool_use.name.clone(),
                    arguments: serde_json::to_string(&tool_use.input)
                        .map_err(|error| GatewayError::Transform(error.to_string()))?,
                },
            }),
            BedrockContentBlock::ToolResult { .. } => {
                return Err(GatewayError::Transform(
                    "bedrock assistant response unexpectedly contained toolResult".into(),
                ));
            }
        }
    }

    let content = (!texts.is_empty()).then_some(MessageContent::Text(texts.join("")));
    let role = match message.role {
        BedrockRole::Assistant => "assistant",
        BedrockRole::User => "user",
    }
    .to_string();

    Ok((role, content, tool_calls))
}

fn map_finish_reason(reason: &str) -> String {
    match reason {
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        other => other,
    }
    .to_string()
}

fn deserialize_bedrock_stream_payload<T: DeserializeOwned>(
    payload: serde_json::Value,
) -> Result<T> {
    serde_json::from_value(payload).map_err(|error| GatewayError::Transform(error.to_string()))
}

fn build_bedrock_stream_chunk(
    state: &ChatStreamState,
    delta: ChatCompletionChunkDelta,
    finish_reason: Option<String>,
    usage: Option<ChatCompletionUsage>,
) -> Result<ChatCompletionChunk> {
    Ok(ChatCompletionChunk {
        id: state.response_id.clone().ok_or_else(|| {
            GatewayError::Stream("bedrock stream emitted a delta before messageStart".into())
        })?,
        object: "chat.completion.chunk".into(),
        created: state.response_created.ok_or_else(|| {
            GatewayError::Stream("bedrock stream missing response_created metadata".into())
        })?,
        model: state.response_model.clone().ok_or_else(|| {
            GatewayError::Stream("bedrock stream missing response_model metadata".into())
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

fn bedrock_usage_to_openai_usage(usage: &BedrockUsage) -> ChatCompletionUsage {
    ChatCompletionUsage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    }
}

fn current_unix_timestamp() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| GatewayError::Internal(error.to_string()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        bedrock_to_openai_response, openai_to_bedrock_request, parse_bedrock_stream_to_openai,
    };
    use crate::gateway::types::{
        bedrock::{BedrockEmptyObject, BedrockToolChoice},
        openai::{ChatCompletionRequest, MessageContent},
    };

    #[test]
    fn openai_to_bedrock_request_maps_system_tools_and_tool_results() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "meta.llama3-2-90b-instruct-v1:0",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "toolu_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Shanghai\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "toolu_123",
                    "content": "{\"temperature\":24}"
                }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}}
                    }
                }
            }],
            "tool_choice": "required",
            "temperature": 0.2,
            "max_completion_tokens": 128
        }))
        .unwrap();

        let body = openai_to_bedrock_request(&request).unwrap();

        assert_eq!(body.system.as_ref().unwrap()[0].text, "You are helpful.");
        assert_eq!(body.messages.len(), 3);
        assert_eq!(
            body.tool_config.as_ref().unwrap().tool_choice,
            Some(BedrockToolChoice::Any {
                any: BedrockEmptyObject::default()
            })
        );
        assert_eq!(
            body.inference_config.as_ref().unwrap().max_tokens,
            Some(128)
        );
    }

    #[test]
    fn bedrock_to_openai_response_maps_tool_use_and_usage() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "meta.llama3-2-90b-instruct-v1:0",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let response = serde_json::from_value(json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Need a tool."},
                        {"toolUse": {"toolUseId": "toolu_123", "name": "get_weather", "input": {"city": "Shanghai"}}}
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 3, "outputTokens": 5, "totalTokens": 8}
        }))
        .unwrap();

        let mapped = bedrock_to_openai_response(&request, response).unwrap();

        assert_eq!(mapped.model, "meta.llama3-2-90b-instruct-v1:0");
        assert!(mapped.id.starts_with("bedrock-"));
        assert_eq!(
            mapped.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        assert!(matches!(
            mapped.choices[0].message.content.as_ref(),
            Some(MessageContent::Text(text)) if text == "Need a tool."
        ));
        assert_eq!(
            mapped.choices[0].message.tool_calls.as_ref().unwrap()[0]
                .function
                .arguments,
            "{\"city\":\"Shanghai\"}"
        );
        assert_eq!(mapped.usage.as_ref().unwrap().total_tokens, 8);
    }

    #[test]
    fn parse_bedrock_stream_to_openai_emits_role_text_finish_and_usage() {
        let mut state = crate::gateway::traits::ChatStreamState {
            response_model: Some("bedrock/test-model".into()),
            ..Default::default()
        };

        let message_start = parse_bedrock_stream_to_openai(
            r#"{"type":"messageStart","payload":{"role":"assistant"}}"#,
            &mut state,
        )
        .unwrap();
        let text_delta = parse_bedrock_stream_to_openai(
            r#"{"type":"contentBlockDelta","payload":{"contentBlockIndex":0,"delta":{"text":"hello from stream"}}}"#,
            &mut state,
        )
        .unwrap();
        let message_stop = parse_bedrock_stream_to_openai(
            r#"{"type":"messageStop","payload":{"stopReason":"end_turn"}}"#,
            &mut state,
        )
        .unwrap();
        let metadata = parse_bedrock_stream_to_openai(
            r#"{"type":"metadata","payload":{"usage":{"inputTokens":7,"outputTokens":9,"totalTokens":16}}}"#,
            &mut state,
        )
        .unwrap();

        assert_eq!(
            message_start[0].choices[0].delta.role.as_deref(),
            Some("assistant")
        );
        assert!(message_start[0].id.starts_with("bedrock-"));
        assert_eq!(
            text_delta[0].choices[0].delta.content.as_deref(),
            Some("hello from stream")
        );
        assert_eq!(
            message_stop[0].choices[0].finish_reason.as_deref(),
            Some("stop")
        );
        assert_eq!(metadata[0].usage.as_ref().unwrap().total_tokens, 16);
        assert_eq!(state.input_tokens, Some(7));
        assert_eq!(state.output_tokens, Some(9));
    }

    #[test]
    fn parse_bedrock_stream_to_openai_emits_tool_call_start_and_delta() {
        let mut state = crate::gateway::traits::ChatStreamState {
            response_model: Some("bedrock/test-model".into()),
            ..Default::default()
        };

        parse_bedrock_stream_to_openai(
            r#"{"type":"messageStart","payload":{"role":"assistant"}}"#,
            &mut state,
        )
        .unwrap();
        let tool_start = parse_bedrock_stream_to_openai(
            r#"{"type":"contentBlockStart","payload":{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"toolu_123","name":"get_weather"}}}}"#,
            &mut state,
        )
        .unwrap();
        let tool_delta = parse_bedrock_stream_to_openai(
            r#"{"type":"contentBlockDelta","payload":{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"city\":\"Shanghai\"}"}}}}"#,
            &mut state,
        )
        .unwrap();

        assert_eq!(
            tool_start[0].choices[0].delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .name
                .as_deref(),
            Some("get_weather")
        );
        assert_eq!(
            tool_delta[0].choices[0].delta.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .arguments
                .as_deref(),
            Some("{\"city\":\"Shanghai\"}")
        );
        assert_eq!(
            state
                .tool_call_accumulators
                .get(&(0, 0))
                .map(|accumulator| accumulator.arguments.as_str()),
            Some("{\"city\":\"Shanghai\"}")
        );
    }
}
