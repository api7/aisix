use opentelemetry_semantic_conventions::attribute::{
    GEN_AI_OPERATION_NAME, GEN_AI_OUTPUT_TYPE, GEN_AI_REQUEST_MAX_TOKENS, GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE, GEN_AI_REQUEST_TOP_P, GEN_AI_RESPONSE_ID, GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS, SERVER_ADDRESS, SERVER_PORT, USER_ID,
};
use reqwest::Url;
use serde_json::{Map, Value};

use super::message_attributes::{
    append_openinference_message_properties, append_openinference_output_message_properties,
    append_openinference_tool_properties, gen_ai_input_messages_json, gen_ai_output_messages_json,
    gen_ai_tool_definitions_json, output_item_finish_reason, request_input_message_views,
    response_output_message_views,
};
use crate::{
    gateway::{
        traits::ProviderCapabilities,
        types::{
            common::Usage,
            openai::responses::{
                ConversationReference, ResponsesApiRequest, ResponsesApiResponse,
                ResponsesApiStreamEvent, ResponsesUsage,
            },
        },
    },
    proxy::utils::trace::span_attributes::{
        append_finish_reason_properties, append_usage_properties, collect_finish_reasons,
    },
};

pub(in crate::proxy::handlers::responses) fn request_span_properties(
    request: &ResponsesApiRequest,
    provider: &dyn ProviderCapabilities,
    base_url: Option<&Url>,
) -> Vec<(String, String)> {
    let provider_semantics = provider.semantic_conventions();
    let input_messages = request_input_message_views(request);
    let mut properties = vec![
        (GEN_AI_OPERATION_NAME.into(), "chat".into()),
        ("openinference.span.kind".into(), "LLM".into()),
        (
            "gen_ai.provider.name".into(),
            provider_semantics.gen_ai_provider_name.to_string(),
        ),
        (
            "llm.system".into(),
            provider_semantics.llm_system.to_string(),
        ),
        (GEN_AI_REQUEST_MODEL.into(), request.model.clone()),
    ];

    if let Some(llm_provider) = provider_semantics.llm_provider {
        properties.push(("llm.provider".into(), llm_provider.to_string()));
    }

    if let Some(max_output_tokens) = request.max_output_tokens {
        properties.push((
            GEN_AI_REQUEST_MAX_TOKENS.into(),
            max_output_tokens.to_string(),
        ));
    }

    if let Some(value) = request.temperature {
        properties.push((GEN_AI_REQUEST_TEMPERATURE.into(), value.to_string()));
    }

    if let Some(value) = request.top_p {
        properties.push((GEN_AI_REQUEST_TOP_P.into(), value.to_string()));
    }

    if let Some(value) = output_type(request) {
        properties.push((GEN_AI_OUTPUT_TYPE.into(), value.to_string()));
    }

    if let Some(value) = request_invocation_parameters(request) {
        properties.push(("llm.invocation_parameters".into(), value));
    }

    if let Some(user_id) = request_user_id(request) {
        properties.push((USER_ID.into(), user_id));
    }

    if let Some(previous_response_id) = request
        .previous_response_id
        .as_ref()
        .filter(|previous_response_id| !previous_response_id.is_empty())
    {
        properties.push((
            "aisix.responses.previous_response_id".into(),
            previous_response_id.clone(),
        ));
    }

    if let Some(conversation_id) = request_conversation_id(request.conversation.as_ref()) {
        properties.push(("aisix.responses.conversation_id".into(), conversation_id));
    }

    append_openinference_message_properties(&mut properties, "llm.input_messages", &input_messages);

    if let Some(value) = gen_ai_input_messages_json(&input_messages) {
        properties.push(("gen_ai.input.messages".into(), value));
    }

    if let Some(tools) = request.tools.as_deref() {
        append_openinference_tool_properties(&mut properties, tools);

        if let Some(value) = gen_ai_tool_definitions_json(tools) {
            properties.push(("gen_ai.tool.definitions".into(), value));
        }
    }

    if let Some(base_url) = base_url {
        if let Some(address) = base_url.host_str() {
            properties.push((SERVER_ADDRESS.into(), address.to_string()));
        }
        if let Some(port) = base_url.port_or_known_default() {
            properties.push((SERVER_PORT.into(), port.to_string()));
        }
    }

    properties
}

pub(in crate::proxy::handlers::responses) fn response_span_properties(
    response: &ResponsesApiResponse,
    usage: &Usage,
) -> Vec<(String, String)> {
    let output_messages = response_output_message_views(response);
    let mut properties = vec![
        (GEN_AI_RESPONSE_ID.into(), response.id.clone()),
        (GEN_AI_RESPONSE_MODEL.into(), response.model.clone()),
        ("llm.model_name".into(), response.model.clone()),
        ("aisix.responses.status".into(), response.status.clone()),
    ];

    append_finish_reason_properties(
        &mut properties,
        collect_finish_reasons(
            output_messages
                .iter()
                .map(|message| message.finish_reason.clone()),
        ),
    );
    append_response_usage_properties(&mut properties, usage, &response.usage);
    append_openinference_output_message_properties(
        &mut properties,
        "llm.output_messages",
        &output_messages,
    );

    if let Some(value) = gen_ai_output_messages_json(&output_messages) {
        properties.push(("gen_ai.output.messages".into(), value));
    }

    properties
}

pub(in crate::proxy::handlers::responses) fn chunk_span_properties(
    event: &ResponsesApiStreamEvent,
) -> Vec<(String, String)> {
    let mut properties = Vec::new();

    match event {
        ResponsesApiStreamEvent::ResponseCreated { response }
        | ResponsesApiStreamEvent::ResponseInProgress { response }
        | ResponsesApiStreamEvent::ResponseCompleted { response } => {
            if !response.id.is_empty() {
                properties.push((GEN_AI_RESPONSE_ID.into(), response.id.clone()));
            }

            if !response.model.is_empty() {
                properties.push((GEN_AI_RESPONSE_MODEL.into(), response.model.clone()));
                properties.push(("llm.model_name".into(), response.model.clone()));
            }

            if !response.status.is_empty() {
                properties.push(("aisix.responses.status".into(), response.status.clone()));
            }

            if matches!(event, ResponsesApiStreamEvent::ResponseCompleted { .. }) {
                append_finish_reason_properties(
                    &mut properties,
                    collect_finish_reasons(response.output.iter().map(output_item_finish_reason)),
                );

                if response.usage.input_tokens > 0
                    || response.usage.output_tokens > 0
                    || response.usage.total_tokens > 0
                {
                    append_response_usage_properties(
                        &mut properties,
                        &Usage::default(),
                        &response.usage,
                    );
                }
            }
        }
        ResponsesApiStreamEvent::OutputItemAdded { .. }
        | ResponsesApiStreamEvent::OutputItemDone { .. }
        | ResponsesApiStreamEvent::ContentPartAdded { .. }
        | ResponsesApiStreamEvent::ContentPartDone { .. }
        | ResponsesApiStreamEvent::OutputTextDelta { .. }
        | ResponsesApiStreamEvent::OutputTextDone { .. }
        | ResponsesApiStreamEvent::FunctionCallArgumentsDelta { .. }
        | ResponsesApiStreamEvent::FunctionCallArgumentsDone { .. }
        | ResponsesApiStreamEvent::Error { .. } => {}
    }

    properties
}

pub(in crate::proxy::handlers::responses) fn event_starts_output(
    event: &ResponsesApiStreamEvent,
) -> bool {
    match event {
        ResponsesApiStreamEvent::OutputTextDelta { delta, .. } => !delta.is_empty(),
        ResponsesApiStreamEvent::FunctionCallArgumentsDelta { delta, .. } => !delta.is_empty(),
        ResponsesApiStreamEvent::OutputTextDone { text, .. } => !text.is_empty(),
        ResponsesApiStreamEvent::FunctionCallArgumentsDone { arguments, .. } => {
            !arguments.is_empty()
        }
        _ => false,
    }
}

fn output_type(request: &ResponsesApiRequest) -> Option<&'static str> {
    let text_config = request.text.as_ref()?;
    let format_type = text_config
        .format
        .as_ref()
        .and_then(|format| format.get("type"))
        .and_then(Value::as_str);

    match format_type {
        Some("json_object") | Some("json_schema") => Some("json"),
        _ => Some("text"),
    }
}

fn request_user_id(request: &ResponsesApiRequest) -> Option<String> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("user_id").or_else(|| metadata.get("user")))
        .and_then(Value::as_str)
        .filter(|user_id| !user_id.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            request
                .safety_identifier
                .as_ref()
                .filter(|identifier| !identifier.is_empty())
                .cloned()
        })
}

fn request_conversation_id(conversation: Option<&ConversationReference>) -> Option<String> {
    match conversation? {
        ConversationReference::Id(id) => (!id.is_empty()).then(|| id.clone()),
        ConversationReference::Descriptor { id } => (!id.is_empty()).then(|| id.clone()),
    }
}

fn request_invocation_parameters(request: &ResponsesApiRequest) -> Option<String> {
    let mut params = Map::new();

    insert_bool(&mut params, "background", request.background);
    insert_value(
        &mut params,
        "context_management",
        request.context_management.as_ref(),
    );
    insert_value(&mut params, "conversation", request.conversation.as_ref());
    insert_value(&mut params, "include", request.include.as_ref());
    insert_u32(&mut params, "max_tool_calls", request.max_tool_calls);
    insert_value(&mut params, "tool_choice", request.tool_choice.as_ref());
    insert_bool(
        &mut params,
        "parallel_tool_calls",
        request.parallel_tool_calls,
    );
    insert_value(&mut params, "prompt", request.prompt.as_ref());
    insert_string(
        &mut params,
        "prompt_cache_key",
        request.prompt_cache_key.as_ref(),
    );
    insert_value(
        &mut params,
        "prompt_cache_retention",
        request.prompt_cache_retention.as_ref(),
    );
    insert_value(&mut params, "reasoning", request.reasoning.as_ref());
    insert_string(
        &mut params,
        "safety_identifier",
        request.safety_identifier.as_ref(),
    );
    insert_string(&mut params, "service_tier", request.service_tier.as_ref());
    insert_bool(&mut params, "stream", request.stream);
    insert_value(
        &mut params,
        "stream_options",
        request.stream_options.as_ref(),
    );
    insert_value(&mut params, "metadata", request.metadata.as_ref());
    insert_value(&mut params, "text", request.text.as_ref());
    insert_u8(&mut params, "top_logprobs", request.top_logprobs);
    insert_string(
        &mut params,
        "previous_response_id",
        request.previous_response_id.as_ref(),
    );
    insert_bool(&mut params, "store", request.store);
    insert_value(&mut params, "truncation", request.truncation.as_ref());

    (!params.is_empty())
        .then_some(Value::Object(params))
        .and_then(|value| serde_json::to_string(&value).ok())
}

fn append_response_usage_properties(
    properties: &mut Vec<(String, String)>,
    usage: &Usage,
    raw_usage: &ResponsesUsage,
) {
    append_usage_properties(properties, usage);

    if usage.input_tokens.is_none() {
        let input_tokens = raw_usage.input_tokens.to_string();
        properties.push((GEN_AI_USAGE_INPUT_TOKENS.into(), input_tokens.clone()));
        properties.push(("llm.token_count.prompt".into(), input_tokens));
    }

    if usage.output_tokens.is_none() {
        let output_tokens = raw_usage.output_tokens.to_string();
        properties.push((GEN_AI_USAGE_OUTPUT_TOKENS.into(), output_tokens.clone()));
        properties.push(("llm.token_count.completion".into(), output_tokens));
    }

    if usage.resolved_total_tokens().is_none() {
        properties.push((
            "llm.token_count.total".into(),
            raw_usage.total_tokens.to_string(),
        ));
    }
}

fn insert_bool(params: &mut Map<String, Value>, key: &str, value: Option<bool>) {
    if let Some(value) = value {
        params.insert(key.into(), Value::Bool(value));
    }
}

fn insert_u32(params: &mut Map<String, Value>, key: &str, value: Option<u32>) {
    if let Some(value) = value {
        params.insert(key.into(), Value::from(value));
    }
}

fn insert_u8(params: &mut Map<String, Value>, key: &str, value: Option<u8>) {
    if let Some(value) = value {
        params.insert(key.into(), Value::from(value));
    }
}

fn insert_string(params: &mut Map<String, Value>, key: &str, value: Option<&String>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        params.insert(key.into(), Value::String(value.clone()));
    }
}

fn insert_value<T>(params: &mut Map<String, Value>, key: &str, value: Option<&T>)
where
    T: serde::Serialize,
{
    if let Some(value) = value
        && let Ok(serialized) = serde_json::to_value(value)
    {
        params.insert(key.into(), serialized);
    }
}
