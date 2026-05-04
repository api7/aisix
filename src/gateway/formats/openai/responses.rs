#![allow(dead_code)]

use std::collections::BTreeMap;

use serde_json::Value;

use crate::gateway::{
    error::{GatewayError, Result},
    traits::{ChatFormat, NativeHandler, OpenAIResponsesNativeStreamState, ProviderCapabilities},
    types::{
        common::{BridgeContext, OpenAIResponsesExtras, Usage},
        openai::{
            ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
            ChatCompletionUsage, ChatMessage, ChunkToolCall, ContentPart, FunctionDefinition,
            ImageUrl, MessageContent, ResponseFormat, StreamOptions, Tool, ToolChoice,
            ToolChoiceFunction,
            responses::{
                ResponsesApiRequest, ResponsesApiResponse, ResponsesApiStreamEvent,
                ResponsesContent, ResponsesContentPart, ResponsesInput, ResponsesInputItem,
                ResponsesOutputContent, ResponsesOutputItem, ResponsesTool, ResponsesUsage,
                Truncation,
            },
        },
    },
};

pub struct ResponsesApiFormat;

#[derive(Debug, Clone, Default)]
pub struct ResponsesBridgeState {
    started: bool,
    response_id: Option<String>,
    response_model: Option<String>,
    created_at: Option<u64>,
    text_output: Option<StreamingTextOutput>,
    tool_calls: BTreeMap<usize, StreamingToolCall>,
    usage: Usage,
}

#[derive(Debug, Clone)]
struct StreamingTextOutput {
    text: String,
    output_index: usize,
}

#[derive(Debug, Clone, Default)]
struct StreamingToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    output_index: Option<usize>,
}

impl ChatFormat for ResponsesApiFormat {
    type Request = ResponsesApiRequest;
    type Response = ResponsesApiResponse;
    type StreamChunk = ResponsesApiStreamEvent;
    type BridgeState = ResponsesBridgeState;
    type NativeStreamState = OpenAIResponsesNativeStreamState;

    fn name() -> &'static str {
        "openai_responses"
    }

    fn is_stream(req: &Self::Request) -> bool {
        req.stream.unwrap_or(false)
    }

    fn extract_model(req: &Self::Request) -> &str {
        &req.model
    }

    fn to_hub(req: &Self::Request) -> Result<(ChatCompletionRequest, BridgeContext)> {
        ensure_request_is_bridgeable(req)?;

        let mut messages = Vec::new();
        if let Some(instructions) = req.instructions.as_ref().filter(|text| !text.is_empty()) {
            messages.push(ChatMessage {
                role: "system".into(),
                content: Some(MessageContent::Text(instructions.clone())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }

        match &req.input {
            ResponsesInput::Text(text) => messages.push(ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Text(text.clone())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }),
            ResponsesInput::Items(items) => {
                for item in items {
                    messages.push(responses_input_item_to_hub_message(item)?);
                }
            }
        }

        let tools = req
            .tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .map(responses_tool_to_hub_tool)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let tool_choice = responses_tool_choice_to_hub(req.tool_choice.as_ref())?;
        let response_format = responses_text_config_to_hub_response_format(req.text.as_ref())?;

        let hub_request = ChatCompletionRequest {
            messages,
            model: req.model.clone(),
            max_completion_tokens: req.max_output_tokens,
            logprobs: req.top_logprobs.map(|_| true),
            top_logprobs: req.top_logprobs,
            response_format,
            stream: req.stream,
            stream_options: req.stream.filter(|stream| *stream).map(|_| StreamOptions {
                include_usage: Some(true),
            }),
            temperature: req.temperature,
            top_p: req.top_p,
            tools,
            tool_choice,
            parallel_tool_calls: req.parallel_tool_calls,
            ..Default::default()
        };

        let mut ctx = BridgeContext::default();
        if let Some(extras) = responses_extras_from_request(req) {
            ctx.openai_responses_extras = Some(extras);
        }

        Ok((hub_request, ctx))
    }

    fn from_hub(resp: &ChatCompletionResponse, ctx: &BridgeContext) -> Result<Self::Response> {
        let mut output = Vec::new();
        let mut next_output_index = 0;

        for choice in &resp.choices {
            let choice_output =
                choice_message_to_response_output(&resp.id, next_output_index, &choice.message)?;
            next_output_index += choice_output.len();
            output.extend(choice_output);
        }

        let extras = ctx.openai_responses_extras.as_ref();
        Ok(ResponsesApiResponse {
            id: resp.id.clone(),
            object: "response".into(),
            created_at: resp.created,
            model: resp.model.clone(),
            output,
            status: "completed".into(),
            usage: responses_usage_from_chat_usage(resp.usage.as_ref()),
            metadata: extras.and_then(|extras| extras.metadata.clone()),
            previous_response_id: extras.and_then(|extras| extras.previous_response_id.clone()),
        })
    }

    fn from_hub_stream(
        chunk: &ChatCompletionChunk,
        state: &mut Self::BridgeState,
        ctx: &BridgeContext,
    ) -> Result<Vec<Self::StreamChunk>> {
        if chunk.choices.len() > 1 {
            return Err(GatewayError::Bridge(
                "Responses API stream bridge only supports a single chat choice".into(),
            ));
        }

        update_stream_metadata(state, chunk);
        update_stream_usage(state, chunk.usage.as_ref());

        let mut events = Vec::new();
        if !state.started && (!chunk.choices.is_empty() || chunk.usage.is_some()) {
            state.started = true;
            let response = partial_stream_response(state, ctx, "in_progress");
            events.push(ResponsesApiStreamEvent::ResponseCreated {
                response: response.clone(),
            });
            events.push(ResponsesApiStreamEvent::ResponseInProgress { response });
        }

        let Some(choice) = chunk.choices.first() else {
            return Ok(events);
        };
        if choice.index != 0 {
            return Err(GatewayError::Bridge(
                "Responses API stream bridge only supports choice index 0".into(),
            ));
        }

        if let Some(content) = choice
            .delta
            .content
            .as_ref()
            .filter(|content| !content.is_empty())
        {
            if state.text_output.is_none() {
                let output_index = next_stream_output_index(state);
                state.text_output = Some(StreamingTextOutput {
                    text: String::new(),
                    output_index,
                });
                events.push(ResponsesApiStreamEvent::OutputItemAdded {
                    output_index,
                    item: streaming_message_output_item(state, false),
                });
                events.push(ResponsesApiStreamEvent::ContentPartAdded {
                    output_index,
                    content_index: 0,
                    part: ResponsesOutputContent::OutputText {
                        text: String::new(),
                    },
                });
            }

            if let Some(text_output) = state.text_output.as_mut() {
                text_output.text.push_str(content);
            }

            let output_index = state
                .text_output
                .as_ref()
                .map(|text_output| text_output.output_index)
                .expect("text output state should exist before emitting text deltas");

            events.push(ResponsesApiStreamEvent::OutputTextDelta {
                output_index,
                content_index: 0,
                delta: content.clone(),
            });
        }

        if let Some(tool_calls) = choice.delta.tool_calls.as_ref() {
            for tool_call in tool_calls {
                let next_output_index = state
                    .tool_calls
                    .get(&tool_call.index)
                    .and_then(|tool_state| tool_state.output_index)
                    .unwrap_or_else(|| next_stream_output_index(state));
                let arguments_delta = tool_call
                    .function
                    .as_ref()
                    .and_then(|function| function.arguments.as_ref())
                    .filter(|arguments| !arguments.is_empty())
                    .cloned();
                let (is_new_tool, output_index) = {
                    let tool_state = state.tool_calls.entry(tool_call.index).or_default();
                    let is_new_tool = tool_state.output_index.is_none();
                    if is_new_tool {
                        tool_state.output_index = Some(next_output_index);
                    }
                    merge_streaming_tool_call(tool_state, tool_call);
                    (
                        is_new_tool,
                        tool_state
                            .output_index
                            .expect("streaming tool call should have a stable output index"),
                    )
                };

                if is_new_tool {
                    events.push(ResponsesApiStreamEvent::OutputItemAdded {
                        output_index,
                        item: streaming_function_call_output_item(state, tool_call.index, false),
                    });
                }

                if let Some(arguments) = arguments_delta {
                    events.push(ResponsesApiStreamEvent::FunctionCallArgumentsDelta {
                        output_index,
                        delta: arguments,
                    });
                }
            }
        }

        Ok(events)
    }

    fn stream_end_events(
        state: &mut Self::BridgeState,
        ctx: &BridgeContext,
    ) -> Vec<Self::StreamChunk> {
        if !state.started {
            return vec![];
        }

        let mut events = Vec::new();
        if let Some(text_output) = state.text_output.as_ref() {
            events.push(ResponsesApiStreamEvent::OutputTextDone {
                output_index: text_output.output_index,
                content_index: 0,
                text: text_output.text.clone(),
            });
            events.push(ResponsesApiStreamEvent::ContentPartDone {
                output_index: text_output.output_index,
                content_index: 0,
                part: ResponsesOutputContent::OutputText {
                    text: text_output.text.clone(),
                },
            });
            events.push(ResponsesApiStreamEvent::OutputItemDone {
                output_index: text_output.output_index,
                item: streaming_message_output_item(state, true),
            });
        }

        for tool_index in state.tool_calls.keys().copied().collect::<Vec<_>>() {
            let Some(tool_call) = state.tool_calls.get(&tool_index) else {
                continue;
            };
            let output_index = tool_call
                .output_index
                .expect("streaming tool call should have a stable output index");

            events.push(ResponsesApiStreamEvent::FunctionCallArgumentsDone {
                output_index,
                arguments: tool_call.arguments.clone(),
            });
            events.push(ResponsesApiStreamEvent::OutputItemDone {
                output_index,
                item: streaming_function_call_output_item(state, tool_index, true),
            });
        }

        events.push(ResponsesApiStreamEvent::ResponseCompleted {
            response: partial_stream_response(state, ctx, "completed"),
        });
        events
    }

    fn native_support(provider: &dyn ProviderCapabilities) -> Option<NativeHandler<'_>>
    where
        Self: Sized,
    {
        provider
            .as_native_openai_responses()
            .map(NativeHandler::OpenAIResponses)
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
            NativeHandler::OpenAIResponses(handler) => Ok((
                handler
                    .native_openai_responses_endpoint(&request.model)
                    .into_owned(),
                handler.transform_openai_responses_request(request)?,
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
        let Some(handler) = provider.as_native_openai_responses() else {
            return Err(GatewayError::NativeNotSupported {
                provider: provider.name().into(),
            });
        };

        let events = handler.transform_openai_responses_stream_chunk(raw, state)?;
        for event in &events {
            update_native_usage_from_event(event, state);
        }
        Ok(events)
    }

    fn native_usage(state: &Self::NativeStreamState) -> Usage {
        state.usage.clone()
    }

    fn response_usage(response: &Self::Response) -> Usage {
        responses_usage_to_common(&response.usage)
    }

    fn parse_native_response(native: &NativeHandler<'_>, body: Value) -> Result<Self::Response>
    where
        Self: Sized,
    {
        match native {
            NativeHandler::OpenAIResponses(handler) => {
                handler.transform_openai_responses_response(body)
            }
            _ => Err(GatewayError::NativeNotSupported {
                provider: native.provider_name().into(),
            }),
        }
    }

    fn serialize_chunk_payload(chunk: &Self::StreamChunk) -> String {
        serde_json::to_string(chunk).expect("responses stream event should serialize")
    }

    fn sse_event_type(chunk: &Self::StreamChunk) -> Option<&'static str> {
        Some(chunk.event_type())
    }
}

fn ensure_request_is_bridgeable(request: &ResponsesApiRequest) -> Result<()> {
    if request.background.unwrap_or(false) {
        return Err(GatewayError::Bridge(
            "Responses API background mode cannot be bridged through Chat Completions".into(),
        ));
    }

    if matches!(request.truncation, Some(Truncation::Auto)) {
        return Err(GatewayError::Bridge(
            "Responses API truncation=auto cannot be bridged through Chat Completions".into(),
        ));
    }

    if let Some(tools) = request.tools.as_ref() {
        for tool in tools {
            if !matches!(tool, ResponsesTool::Function { .. }) {
                return Err(GatewayError::Bridge(
                    "Responses API built-in tools cannot be bridged through Chat Completions"
                        .into(),
                ));
            }
        }
    }

    Ok(())
}

fn responses_input_item_to_hub_message(item: &ResponsesInputItem) -> Result<ChatMessage> {
    match item {
        ResponsesInputItem::Message { role, content } => Ok(ChatMessage {
            role: role.clone(),
            content: Some(responses_content_to_hub_message_content(content)?),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }),
        ResponsesInputItem::FunctionCallOutput { call_id, output } => Ok(ChatMessage {
            role: "tool".into(),
            content: Some(MessageContent::Text(output.clone())),
            name: None,
            tool_calls: None,
            tool_call_id: Some(call_id.clone()),
        }),
    }
}

fn responses_content_to_hub_message_content(content: &ResponsesContent) -> Result<MessageContent> {
    match content {
        ResponsesContent::Text(text) => Ok(MessageContent::Text(text.clone())),
        ResponsesContent::Parts(parts) => Ok(MessageContent::Parts(
            parts
                .iter()
                .map(response_content_part_to_hub_content_part)
                .collect::<Result<Vec<_>>>()?,
        )),
    }
}

fn response_content_part_to_hub_content_part(part: &ResponsesContentPart) -> Result<ContentPart> {
    match part {
        ResponsesContentPart::InputText { text } => Ok(ContentPart::Text { text: text.clone() }),
        ResponsesContentPart::InputImage {
            image_url,
            file_id,
            detail,
        } => {
            let Some(url) = image_url.as_ref() else {
                return Err(GatewayError::Bridge(format!(
                    "Responses API input_image with file_id {} cannot be bridged through Chat Completions",
                    file_id.as_deref().unwrap_or("<unknown>")
                )));
            };
            Ok(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: url.clone(),
                    detail: detail.clone(),
                },
            })
        }
    }
}

fn responses_tool_to_hub_tool(tool: &ResponsesTool) -> Result<Tool> {
    match tool {
        ResponsesTool::Function {
            name,
            description,
            parameters,
            strict,
        } => Ok(Tool {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: name.clone(),
                description: description.clone(),
                parameters: parameters.clone(),
                strict: *strict,
            },
        }),
        _ => Err(GatewayError::Bridge(
            "Responses API built-in tools cannot be bridged through Chat Completions".into(),
        )),
    }
}

fn responses_tool_choice_to_hub(tool_choice: Option<&Value>) -> Result<Option<ToolChoice>> {
    let Some(tool_choice) = tool_choice else {
        return Ok(None);
    };

    match tool_choice {
        Value::String(mode) => Ok(Some(ToolChoice::Mode(mode.clone()))),
        Value::Object(map) => {
            let Some(kind) = map.get("type").and_then(Value::as_str) else {
                return Err(GatewayError::Bridge(
                    "Responses API tool_choice object requires a type field".into(),
                ));
            };

            match kind {
                "none" | "auto" | "required" => Ok(Some(ToolChoice::Mode(kind.into()))),
                "function" => {
                    let name = map
                        .get("name")
                        .and_then(Value::as_str)
                        .or_else(|| {
                            map.get("function")
                                .and_then(Value::as_object)
                                .and_then(|function| function.get("name"))
                                .and_then(Value::as_str)
                        })
                        .ok_or_else(|| {
                            GatewayError::Bridge(
                                "Responses API function tool_choice requires a function name"
                                    .into(),
                            )
                        })?;
                    Ok(Some(ToolChoice::Function {
                        r#type: "function".into(),
                        function: ToolChoiceFunction { name: name.into() },
                    }))
                }
                other => Err(GatewayError::Bridge(format!(
                    "unsupported Responses API tool_choice type {} for hub bridging",
                    other
                ))),
            }
        }
        _ => Err(GatewayError::Bridge(
            "Responses API tool_choice must be a string or object".into(),
        )),
    }
}

fn responses_text_config_to_hub_response_format(
    text: Option<&crate::gateway::types::openai::responses::ResponseTextConfig>,
) -> Result<Option<ResponseFormat>> {
    let Some(text) = text else {
        return Ok(None);
    };
    let Some(format) = text.format.as_ref() else {
        return Ok(None);
    };

    serde_json::from_value(format.clone()).map(Some).map_err(|error| {
        GatewayError::Bridge(format!(
            "Responses API text.format cannot be bridged to Chat Completions response_format: {}",
            error
        ))
    })
}

fn responses_extras_from_request(request: &ResponsesApiRequest) -> Option<OpenAIResponsesExtras> {
    let extras = OpenAIResponsesExtras {
        previous_response_id: request.previous_response_id.clone(),
        instructions: request.instructions.clone(),
        store: request.store,
        metadata: request.metadata.clone(),
        background: request.background,
        context_management: request.context_management.clone(),
        conversation: request.conversation.clone(),
        include: request.include.clone(),
        max_tool_calls: request.max_tool_calls,
        prompt: request.prompt.clone(),
        prompt_cache_key: request.prompt_cache_key.clone(),
        prompt_cache_retention: request.prompt_cache_retention.clone(),
        reasoning: request.reasoning.clone(),
        safety_identifier: request.safety_identifier.clone(),
        service_tier: request.service_tier.clone(),
        stream_options: request.stream_options.clone(),
        text: request.text.clone(),
        top_logprobs: request.top_logprobs,
        truncation: request.truncation.clone(),
    };

    if extras.previous_response_id.is_none()
        && extras.instructions.is_none()
        && extras.store.is_none()
        && extras.metadata.is_none()
        && extras.background.is_none()
        && extras.context_management.is_none()
        && extras.conversation.is_none()
        && extras.include.is_none()
        && extras.max_tool_calls.is_none()
        && extras.prompt.is_none()
        && extras.prompt_cache_key.is_none()
        && extras.prompt_cache_retention.is_none()
        && extras.reasoning.is_none()
        && extras.safety_identifier.is_none()
        && extras.service_tier.is_none()
        && extras.stream_options.is_none()
        && extras.text.is_none()
        && extras.top_logprobs.is_none()
        && extras.truncation.is_none()
    {
        None
    } else {
        Some(extras)
    }
}

fn choice_message_to_response_output(
    response_id: &str,
    next_output_index: usize,
    message: &ChatMessage,
) -> Result<Vec<ResponsesOutputItem>> {
    let mut output = Vec::new();

    let content = chat_message_content_to_response_content(message.content.as_ref())?;
    if !content.is_empty() {
        output.push(ResponsesOutputItem::Message {
            id: response_message_output_id(response_id, next_output_index),
            role: message.role.clone(),
            content,
            status: "completed".into(),
        });
    }

    if let Some(tool_calls) = message.tool_calls.as_ref() {
        for tool_call in tool_calls {
            output.push(ResponsesOutputItem::FunctionCall {
                id: tool_call.id.clone(),
                call_id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone(),
                status: "completed".into(),
            });
        }
    }

    Ok(output)
}

fn chat_message_content_to_response_content(
    content: Option<&MessageContent>,
) -> Result<Vec<ResponsesOutputContent>> {
    let Some(content) = content else {
        return Ok(vec![]);
    };

    match content {
        MessageContent::Text(text) => Ok(vec![ResponsesOutputContent::OutputText {
            text: text.clone(),
        }]),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => {
                    Ok(ResponsesOutputContent::OutputText { text: text.clone() })
                }
                ContentPart::ImageUrl { .. } => Err(GatewayError::Bridge(
                    "assistant image output cannot be bridged to Responses API output_text".into(),
                )),
            })
            .collect(),
    }
}

fn responses_usage_from_chat_usage(usage: Option<&ChatCompletionUsage>) -> ResponsesUsage {
    let usage = usage.cloned().unwrap_or_default();
    ResponsesUsage {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    }
}

fn responses_usage_to_common(usage: &ResponsesUsage) -> Usage {
    Usage {
        input_tokens: Some(usage.input_tokens),
        output_tokens: Some(usage.output_tokens),
        total_tokens: Some(usage.total_tokens),
        ..Default::default()
    }
}

fn update_stream_metadata(state: &mut ResponsesBridgeState, chunk: &ChatCompletionChunk) {
    if state.response_id.is_none() && !chunk.id.is_empty() {
        state.response_id = Some(chunk.id.clone());
    }
    if state.response_model.is_none() && !chunk.model.is_empty() {
        state.response_model = Some(chunk.model.clone());
    }
    if state.created_at.is_none() {
        state.created_at = Some(chunk.created);
    }
}

fn update_stream_usage(state: &mut ResponsesBridgeState, usage: Option<&ChatCompletionUsage>) {
    let Some(usage) = usage else {
        return;
    };
    state.usage.merge(&Usage {
        input_tokens: Some(usage.prompt_tokens),
        output_tokens: Some(usage.completion_tokens),
        total_tokens: Some(usage.total_tokens),
        ..Default::default()
    });
}

fn merge_streaming_tool_call(state: &mut StreamingToolCall, tool_call: &ChunkToolCall) {
    if let Some(id) = tool_call.id.as_ref() {
        state.id = Some(id.clone());
    }
    if let Some(function) = tool_call.function.as_ref() {
        if let Some(name) = function.name.as_ref() {
            state.name = Some(name.clone());
        }
        if let Some(arguments) = function.arguments.as_ref() {
            state.arguments.push_str(arguments);
        }
    }
}

fn streaming_message_output_item(
    state: &ResponsesBridgeState,
    completed: bool,
) -> ResponsesOutputItem {
    let response_id = state.response_id.as_deref().unwrap_or("response");
    let text_output = state
        .text_output
        .as_ref()
        .expect("text output state should exist before building output item");

    ResponsesOutputItem::Message {
        id: response_message_output_id(response_id, text_output.output_index),
        role: "assistant".into(),
        content: vec![ResponsesOutputContent::OutputText {
            text: text_output.text.clone(),
        }],
        status: if completed {
            "completed".into()
        } else {
            "in_progress".into()
        },
    }
}

fn streaming_function_call_output_item(
    state: &ResponsesBridgeState,
    tool_index: usize,
    completed: bool,
) -> ResponsesOutputItem {
    let response_id = state.response_id.as_deref().unwrap_or("response");
    let tool_call = state
        .tool_calls
        .get(&tool_index)
        .expect("tool call state should exist before building output item");
    let id = tool_call
        .id
        .clone()
        .unwrap_or_else(|| format!("{}_call_{}", response_id, tool_index));

    ResponsesOutputItem::FunctionCall {
        id: id.clone(),
        call_id: id,
        name: tool_call
            .name
            .clone()
            .unwrap_or_else(|| format!("tool_{}", tool_index)),
        arguments: tool_call.arguments.clone(),
        status: if completed {
            "completed".into()
        } else {
            "in_progress".into()
        },
    }
}

fn next_stream_output_index(state: &ResponsesBridgeState) -> usize {
    state
        .text_output
        .as_ref()
        .map(|text_output| text_output.output_index)
        .into_iter()
        .chain(
            state
                .tool_calls
                .values()
                .filter_map(|tool_call| tool_call.output_index),
        )
        .max()
        .map_or(0, |output_index| output_index + 1)
}

fn partial_stream_response(
    state: &ResponsesBridgeState,
    ctx: &BridgeContext,
    status: &str,
) -> ResponsesApiResponse {
    let mut output = Vec::new();
    if let Some(text_output) = state.text_output.as_ref() {
        output.push((
            text_output.output_index,
            streaming_message_output_item(state, status == "completed"),
        ));
    }
    for tool_index in state.tool_calls.keys().copied() {
        let Some(tool_call) = state.tool_calls.get(&tool_index) else {
            continue;
        };
        let Some(output_index) = tool_call.output_index else {
            continue;
        };
        output.push((
            output_index,
            streaming_function_call_output_item(state, tool_index, status == "completed"),
        ));
    }
    output.sort_by_key(|(output_index, _)| *output_index);
    let output = output.into_iter().map(|(_, item)| item).collect();

    let extras = ctx.openai_responses_extras.as_ref();
    ResponsesApiResponse {
        id: state
            .response_id
            .clone()
            .unwrap_or_else(|| "response".into()),
        object: "response".into(),
        created_at: state.created_at.unwrap_or_default(),
        model: state.response_model.clone().unwrap_or_default(),
        output,
        status: status.into(),
        usage: ResponsesUsage {
            input_tokens: state.usage.input_tokens.unwrap_or_default(),
            output_tokens: state.usage.output_tokens.unwrap_or_default(),
            total_tokens: state.usage.resolved_total_tokens().unwrap_or_default(),
        },
        metadata: extras.and_then(|extras| extras.metadata.clone()),
        previous_response_id: extras.and_then(|extras| extras.previous_response_id.clone()),
    }
}

fn response_message_output_id(response_id: &str, output_index: usize) -> String {
    format!("{}_message_{}", response_id, output_index)
}

fn update_native_usage_from_event(
    event: &ResponsesApiStreamEvent,
    state: &mut OpenAIResponsesNativeStreamState,
) {
    let Some(usage) = (match event {
        ResponsesApiStreamEvent::ResponseCreated { response }
        | ResponsesApiStreamEvent::ResponseInProgress { response }
        | ResponsesApiStreamEvent::ResponseCompleted { response } => {
            Some(responses_usage_to_common(&response.usage))
        }
        _ => None,
    }) else {
        return;
    };

    state.usage.merge(&usage);
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use serde_json::{Value, json};

    use super::{ResponsesApiFormat, responses_usage_to_common};
    use crate::gateway::{
        error::GatewayError,
        provider_instance::ProviderAuth,
        traits::{
            ChatFormat, ChatTransform, NativeHandler, NativeOpenAIResponsesSupport,
            OpenAIResponsesNativeStreamState, ProviderCapabilities, ProviderMeta, StreamReaderKind,
        },
        types::{
            common::{BridgeContext, OpenAIResponsesExtras},
            openai::{
                ChatCompletionChunk, ChatCompletionResponse,
                responses::{
                    ResponsesApiRequest, ResponsesApiResponse, ResponsesApiStreamEvent,
                    ResponsesOutputItem,
                },
            },
        },
    };

    struct DummyResponsesNativeProvider;

    impl ProviderMeta for DummyResponsesNativeProvider {
        fn name(&self) -> &'static str {
            "dummy-responses-native"
        }

        fn default_base_url(&self) -> &'static str {
            "https://example.com"
        }

        fn stream_reader_kind(&self) -> StreamReaderKind {
            StreamReaderKind::Sse
        }

        fn build_auth_headers(
            &self,
            _auth: &ProviderAuth,
        ) -> crate::gateway::error::Result<http::HeaderMap> {
            Ok(http::HeaderMap::new())
        }
    }

    impl ChatTransform for DummyResponsesNativeProvider {}

    impl NativeOpenAIResponsesSupport for DummyResponsesNativeProvider {
        fn native_openai_responses_endpoint(&self, _model: &str) -> Cow<'static, str> {
            Cow::Borrowed("/v1/responses")
        }

        fn transform_openai_responses_request(
            &self,
            req: &ResponsesApiRequest,
        ) -> crate::gateway::error::Result<Value> {
            Ok(serde_json::to_value(req).unwrap())
        }

        fn transform_openai_responses_response(
            &self,
            body: Value,
        ) -> crate::gateway::error::Result<ResponsesApiResponse> {
            serde_json::from_value(body).map_err(|error| GatewayError::Transform(error.to_string()))
        }

        fn transform_openai_responses_stream_chunk(
            &self,
            raw: &str,
            _state: &mut OpenAIResponsesNativeStreamState,
        ) -> crate::gateway::error::Result<Vec<ResponsesApiStreamEvent>> {
            Ok(vec![serde_json::from_str(raw).map_err(|error| {
                GatewayError::Transform(error.to_string())
            })?])
        }
    }

    impl ProviderCapabilities for DummyResponsesNativeProvider {
        fn as_native_openai_responses(&self) -> Option<&dyn NativeOpenAIResponsesSupport> {
            Some(self)
        }
    }

    #[test]
    fn to_hub_maps_text_request_and_preserves_responses_extras() {
        let request: ResponsesApiRequest = serde_json::from_value(json!({
            "model": "gpt-4.1",
            "input": "Hello",
            "instructions": "Be concise",
            "max_output_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"}
            }],
            "tool_choice": {"type": "function", "name": "get_weather"},
            "stream": true,
            "store": true,
            "metadata": {"request_id": "req_1"},
            "previous_response_id": "resp_prev",
            "top_logprobs": 3,
            "text": {"format": {"type": "json_object"}}
        }))
        .unwrap();

        let (hub, ctx) = ResponsesApiFormat::to_hub(&request).unwrap();

        assert_eq!(hub.model, "gpt-4.1");
        assert_eq!(hub.messages.len(), 2);
        assert_eq!(hub.messages[0].role, "system");
        assert_eq!(hub.messages[1].role, "user");
        assert_eq!(hub.max_completion_tokens, Some(128));
        assert_eq!(hub.temperature, Some(0.2));
        assert_eq!(hub.top_p, Some(0.9));
        assert_eq!(hub.logprobs, Some(true));
        assert_eq!(hub.top_logprobs, Some(3));
        assert_eq!(hub.stream, Some(true));
        assert_eq!(
            hub.stream_options
                .as_ref()
                .and_then(|options| options.include_usage),
            Some(true)
        );
        assert_eq!(hub.tools.as_ref().unwrap().len(), 1);
        assert_eq!(hub.response_format.as_ref().unwrap().r#type, "json_object");

        let extras = ctx.openai_responses_extras.as_ref().unwrap();
        assert_eq!(extras.previous_response_id.as_deref(), Some("resp_prev"));
        assert_eq!(extras.store, Some(true));
        assert_eq!(extras.metadata.as_ref().unwrap()["request_id"], "req_1");
    }

    #[test]
    fn to_hub_rejects_builtin_tools_and_truncation_auto() {
        let built_in_tools_request: ResponsesApiRequest = serde_json::from_value(json!({
            "model": "gpt-4.1",
            "input": "Hello",
            "tools": [{"type": "web_search_preview"}]
        }))
        .unwrap();
        let truncation_request: ResponsesApiRequest = serde_json::from_value(json!({
            "model": "gpt-4.1",
            "input": "Hello",
            "truncation": "auto"
        }))
        .unwrap();

        assert_matches!(
            ResponsesApiFormat::to_hub(&built_in_tools_request),
            Err(GatewayError::Bridge(message)) if message.contains("built-in tools")
        );
        assert_matches!(
            ResponsesApiFormat::to_hub(&truncation_request),
            Err(GatewayError::Bridge(message)) if message.contains("truncation=auto")
        );
    }

    #[test]
    fn from_hub_maps_message_and_tool_calls_to_responses_response() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl_123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}
        }))
        .unwrap();
        let ctx = BridgeContext {
            openai_responses_extras: Some(OpenAIResponsesExtras {
                previous_response_id: Some("resp_prev".into()),
                instructions: None,
                store: None,
                metadata: Some(json!({"trace": "abc"})),
                background: None,
                context_management: None,
                conversation: None,
                include: None,
                max_tool_calls: None,
                prompt: None,
                prompt_cache_key: None,
                prompt_cache_retention: None,
                reasoning: None,
                safety_identifier: None,
                service_tier: None,
                stream_options: None,
                text: None,
                top_logprobs: None,
                truncation: None,
            }),
            ..Default::default()
        };

        let bridged = ResponsesApiFormat::from_hub(&response, &ctx).unwrap();

        assert_eq!(bridged.id, "chatcmpl_123");
        assert_eq!(bridged.object, "response");
        assert_eq!(bridged.status, "completed");
        assert_eq!(bridged.previous_response_id.as_deref(), Some("resp_prev"));
        assert_eq!(bridged.metadata.as_ref().unwrap()["trace"], "abc");
        assert_eq!(bridged.output.len(), 2);
        assert_matches!(&bridged.output[0], crate::gateway::types::openai::responses::ResponsesOutputItem::Message { role, .. } if role == "assistant");
        assert_matches!(&bridged.output[1], crate::gateway::types::openai::responses::ResponsesOutputItem::FunctionCall { name, .. } if name == "get_weather");
        assert_eq!(bridged.usage.total_tokens, 16);
        assert_eq!(
            responses_usage_to_common(&bridged.usage).total_tokens,
            Some(16)
        );
    }

    #[test]
    fn from_hub_stream_emits_text_delta_and_completion_events() {
        let first_chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl_123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "Hel"}
            }]
        }))
        .unwrap();
        let second_chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl_123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": {"content": "lo"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16}
        }))
        .unwrap();

        let mut state = super::ResponsesBridgeState::default();
        let ctx = BridgeContext::default();

        let start_events =
            ResponsesApiFormat::from_hub_stream(&first_chunk, &mut state, &ctx).unwrap();
        let next_events =
            ResponsesApiFormat::from_hub_stream(&second_chunk, &mut state, &ctx).unwrap();
        let end_events = ResponsesApiFormat::stream_end_events(&mut state, &ctx);

        assert_eq!(start_events.len(), 5);
        assert!(matches!(
            start_events[0],
            ResponsesApiStreamEvent::ResponseCreated { .. }
        ));
        assert!(matches!(
            start_events[1],
            ResponsesApiStreamEvent::ResponseInProgress { .. }
        ));
        assert!(matches!(
            start_events[2],
            ResponsesApiStreamEvent::OutputItemAdded { .. }
        ));
        assert!(matches!(
            start_events[3],
            ResponsesApiStreamEvent::ContentPartAdded { .. }
        ));
        assert_matches!(&start_events[4], ResponsesApiStreamEvent::OutputTextDelta { delta, .. } if delta == "Hel");

        assert_eq!(next_events.len(), 1);
        assert_matches!(&next_events[0], ResponsesApiStreamEvent::OutputTextDelta { delta, .. } if delta == "lo");

        assert_eq!(end_events.len(), 4);
        assert!(matches!(
            end_events[0],
            ResponsesApiStreamEvent::OutputTextDone { .. }
        ));
        assert!(matches!(
            end_events[1],
            ResponsesApiStreamEvent::ContentPartDone { .. }
        ));
        assert!(matches!(
            end_events[2],
            ResponsesApiStreamEvent::OutputItemDone { .. }
        ));
        if let ResponsesApiStreamEvent::ResponseCompleted { response } = &end_events[3] {
            assert_eq!(response.status, "completed");
            assert_eq!(response.usage.total_tokens, 16);
            assert_eq!(response.output.len(), 1);
        } else {
            panic!("expected response.completed event");
        }
    }

    #[test]
    fn stream_bridge_keeps_stable_output_indexes_when_text_follows_tool_call() {
        let first_chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl_123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"S"}
                    }]
                }
            }]
        }))
        .unwrap();
        let second_chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl_123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"}
            }]
        }))
        .unwrap();
        let third_chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl_123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "F\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .unwrap();

        let mut state = super::ResponsesBridgeState::default();
        let ctx = BridgeContext::default();

        let start_events =
            ResponsesApiFormat::from_hub_stream(&first_chunk, &mut state, &ctx).unwrap();
        let text_events =
            ResponsesApiFormat::from_hub_stream(&second_chunk, &mut state, &ctx).unwrap();
        let tool_events =
            ResponsesApiFormat::from_hub_stream(&third_chunk, &mut state, &ctx).unwrap();
        let end_events = ResponsesApiFormat::stream_end_events(&mut state, &ctx);

        assert_matches!(
            &start_events[2],
            ResponsesApiStreamEvent::OutputItemAdded { output_index, .. } if *output_index == 0
        );
        assert_matches!(
            &start_events[3],
            ResponsesApiStreamEvent::FunctionCallArgumentsDelta { output_index, delta }
                if *output_index == 0 && delta == "{\"city\":\"S"
        );

        assert_matches!(
            &text_events[0],
            ResponsesApiStreamEvent::OutputItemAdded { output_index, .. } if *output_index == 1
        );
        assert_matches!(
            &text_events[1],
            ResponsesApiStreamEvent::ContentPartAdded { output_index, .. } if *output_index == 1
        );
        assert_matches!(
            &text_events[2],
            ResponsesApiStreamEvent::OutputTextDelta { output_index, delta, .. }
                if *output_index == 1 && delta == "Hello"
        );

        assert_matches!(
            &tool_events[0],
            ResponsesApiStreamEvent::FunctionCallArgumentsDelta { output_index, delta }
                if *output_index == 0 && delta == "F\"}"
        );

        assert!(end_events.iter().any(|event| {
            matches!(
                event,
                ResponsesApiStreamEvent::FunctionCallArgumentsDone { output_index, arguments }
                    if *output_index == 0 && arguments == "{\"city\":\"SF\"}"
            )
        }));

        let completed = end_events
            .iter()
            .find_map(|event| match event {
                ResponsesApiStreamEvent::ResponseCompleted { response } => Some(response),
                _ => None,
            })
            .expect("expected response.completed event");
        assert!(matches!(
            completed.output[0],
            ResponsesOutputItem::FunctionCall { .. }
        ));
        assert!(matches!(
            completed.output[1],
            ResponsesOutputItem::Message { .. }
        ));
    }

    #[test]
    fn native_support_and_parse_native_response_delegate_to_provider() {
        let provider = DummyResponsesNativeProvider;
        let request: ResponsesApiRequest = serde_json::from_value(json!({
            "model": "gpt-4.1",
            "input": "Hello"
        }))
        .unwrap();
        let native = ResponsesApiFormat::native_support(&provider).unwrap();
        let response_body = json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4.1",
            "output": [],
            "status": "completed",
            "usage": {"input_tokens": 7, "output_tokens": 9, "total_tokens": 16}
        });
        let raw_event = json!({
            "type": "response.completed",
            "response": response_body
        });

        let (endpoint, body) = ResponsesApiFormat::call_native(&native, &request, false).unwrap();
        let parsed_response =
            ResponsesApiFormat::parse_native_response(&native, response_body.clone()).unwrap();
        let mut native_state = OpenAIResponsesNativeStreamState::default();
        let native_events = ResponsesApiFormat::transform_native_stream_chunk(
            &provider,
            &serde_json::to_string(&raw_event).unwrap(),
            &mut native_state,
        )
        .unwrap();

        assert_eq!(endpoint, "/v1/responses");
        assert_eq!(body["model"], "gpt-4.1");
        assert_eq!(parsed_response.id, "resp_123");
        assert_eq!(native_events.len(), 1);
        assert_eq!(
            ResponsesApiFormat::native_usage(&native_state).total_tokens,
            Some(16)
        );

        match native {
            NativeHandler::OpenAIResponses(_) => {}
            _ => panic!("expected OpenAI responses native handler"),
        }
    }
}
