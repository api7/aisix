use reqwest::Url;
use serde_json::{Map, Value};

use super::message_attributes::{
    append_openinference_message_properties, append_openinference_output_message_properties,
    append_openinference_tool_properties, gen_ai_input_messages_json, gen_ai_output_messages_json,
    gen_ai_tool_definitions_json, request_input_message_views, response_output_message_views,
};
use crate::{
    gateway::{
        traits::ProviderCapabilities,
        types::{
            anthropic::{
                AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicStreamEvent,
                AnthropicToolChoice, AnthropicUsage, DeltaUsage, MessageStartUsage,
            },
            common::Usage,
        },
    },
    proxy::utils::trace::span_attributes::{
        append_finish_reason_properties, append_usage_properties, collect_finish_reasons,
    },
};

pub(in crate::proxy::handlers::messages) fn request_span_properties(
    request: &AnthropicMessagesRequest,
    provider: &dyn ProviderCapabilities,
    base_url: Option<&Url>,
) -> Vec<(String, String)> {
    let provider_semantics = provider.semantic_conventions();
    let input_messages = request_input_message_views(request);
    let mut properties = vec![
        ("gen_ai.operation.name".into(), "chat".into()),
        ("openinference.span.kind".into(), "LLM".into()),
        (
            "gen_ai.provider.name".into(),
            provider_semantics.gen_ai_provider_name.to_string(),
        ),
        (
            "llm.system".into(),
            provider_semantics.llm_system.to_string(),
        ),
        ("gen_ai.request.model".into(), request.model.clone()),
        (
            "gen_ai.request.max_tokens".into(),
            request.max_tokens.to_string(),
        ),
    ];

    if let Some(llm_provider) = provider_semantics.llm_provider {
        properties.push(("llm.provider".into(), llm_provider.to_string()));
    }

    if let Some(value) = request.temperature {
        properties.push(("gen_ai.request.temperature".into(), value.to_string()));
    }

    if let Some(value) = request.top_p {
        properties.push(("gen_ai.request.top_p".into(), value.to_string()));
    }

    if let Some(value) = request.top_k {
        properties.push(("gen_ai.request.top_k".into(), value.to_string()));
    }

    if let Some(value) = stop_sequences_json(request.stop_sequences.as_deref()) {
        properties.push(("gen_ai.request.stop_sequences".into(), value));
    }

    if let Some(value) = request_invocation_parameters(request) {
        properties.push(("llm.invocation_parameters".into(), value));
    }

    if let Some(user_id) = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.user_id.as_ref())
        .filter(|user_id| !user_id.is_empty())
    {
        properties.push(("user.id".into(), user_id.clone()));
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
            properties.push(("server.address".into(), address.to_string()));
        }
        if let Some(port) = base_url.port_or_known_default() {
            properties.push(("server.port".into(), port.to_string()));
        }
    }

    properties
}

pub(in crate::proxy::handlers::messages) fn response_span_properties(
    response: &AnthropicMessagesResponse,
    usage: &Usage,
) -> Vec<(String, String)> {
    let output_messages = response_output_message_views(response);
    let mut properties = vec![
        ("gen_ai.response.id".into(), response.id.clone()),
        ("gen_ai.response.model".into(), response.model.clone()),
        ("llm.model_name".into(), response.model.clone()),
    ];

    append_finish_reason_properties(
        &mut properties,
        collect_finish_reasons(std::iter::once(response.stop_reason.clone())),
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

pub(in crate::proxy::handlers::messages) fn chunk_span_properties(
    event: &AnthropicStreamEvent,
) -> Vec<(String, String)> {
    let mut properties = Vec::new();

    match event {
        AnthropicStreamEvent::MessageStart { message } => {
            properties.push(("gen_ai.response.id".into(), message.id.clone()));
            properties.push(("gen_ai.response.model".into(), message.model.clone()));
            properties.push(("llm.model_name".into(), message.model.clone()));
            append_message_start_usage_properties(&mut properties, &message.usage);
        }
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            append_finish_reason_properties(
                &mut properties,
                collect_finish_reasons(std::iter::once(delta.stop_reason.clone())),
            );
            append_delta_usage_properties(&mut properties, usage);
        }
        AnthropicStreamEvent::ContentBlockStart { .. }
        | AnthropicStreamEvent::ContentBlockDelta { .. }
        | AnthropicStreamEvent::ContentBlockStop { .. }
        | AnthropicStreamEvent::MessageStop
        | AnthropicStreamEvent::Ping
        | AnthropicStreamEvent::Error { .. } => {}
    }

    properties
}

fn stop_sequences_json(stop_sequences: Option<&[String]>) -> Option<String> {
    let stop_sequences = stop_sequences?;

    serde_json::to_string(stop_sequences).ok()
}

fn request_invocation_parameters(request: &AnthropicMessagesRequest) -> Option<String> {
    let mut params = Map::new();
    params.insert("max_tokens".into(), Value::from(request.max_tokens));

    if let Some(value) = request.temperature {
        params.insert("temperature".into(), Value::from(value));
    }
    if let Some(value) = request.top_p {
        params.insert("top_p".into(), Value::from(value));
    }
    if let Some(value) = request.top_k {
        params.insert("top_k".into(), Value::from(value));
    }
    if let Some(value) = request.stream {
        params.insert("stream".into(), Value::from(value));
    }
    if let Some(stop_sequences) = request.stop_sequences.as_ref() {
        params.insert(
            "stop_sequences".into(),
            Value::Array(stop_sequences.iter().cloned().map(Value::String).collect()),
        );
    }
    if let Some(value) = request.cache_control.as_ref()
        && let Ok(value) = serde_json::to_value(value)
    {
        params.insert("cache_control".into(), value);
    }
    if let Some(value) = request.tool_choice.as_ref()
        && let Some(value) = anthropic_tool_choice_to_value(value)
    {
        params.insert("tool_choice".into(), value);
    }

    serde_json::to_string(&Value::Object(params)).ok()
}

fn anthropic_tool_choice_to_value(tool_choice: &AnthropicToolChoice) -> Option<Value> {
    serde_json::to_value(tool_choice).ok()
}

fn append_response_usage_properties(
    properties: &mut Vec<(String, String)>,
    usage: &Usage,
    raw_usage: &AnthropicUsage,
) {
    append_usage_properties(properties, usage);

    let raw_input_tokens = raw_usage.input_tokens
        + raw_usage.cache_creation_input_tokens
        + raw_usage.cache_read_input_tokens;

    if usage.input_tokens.is_none() {
        let input_tokens = raw_input_tokens.to_string();
        properties.push(("gen_ai.usage.input_tokens".into(), input_tokens.clone()));
        properties.push(("llm.token_count.prompt".into(), input_tokens));
    }

    if usage.output_tokens.is_none() {
        let output_tokens = raw_usage.output_tokens.to_string();
        properties.push(("gen_ai.usage.output_tokens".into(), output_tokens.clone()));
        properties.push(("llm.token_count.completion".into(), output_tokens));
    }

    if usage.resolved_total_tokens().is_none() {
        properties.push((
            "llm.token_count.total".into(),
            (raw_input_tokens + raw_usage.output_tokens).to_string(),
        ));
    }

    if usage.cache_creation_input_tokens.is_none() && raw_usage.cache_creation_input_tokens > 0 {
        let cache_creation = raw_usage.cache_creation_input_tokens.to_string();
        properties.push((
            "gen_ai.usage.cache_creation.input_tokens".into(),
            cache_creation.clone(),
        ));
        properties.push((
            "llm.token_count.prompt_details.cache_write".into(),
            cache_creation,
        ));
    }

    if usage.cache_read_input_tokens.is_none() && raw_usage.cache_read_input_tokens > 0 {
        let cache_read = raw_usage.cache_read_input_tokens.to_string();
        properties.push((
            "gen_ai.usage.cache_read.input_tokens".into(),
            cache_read.clone(),
        ));
        properties.push((
            "llm.token_count.prompt_details.cache_read".into(),
            cache_read,
        ));
    }
}

fn append_message_start_usage_properties(
    properties: &mut Vec<(String, String)>,
    usage: &MessageStartUsage,
) {
    append_message_usage_values(
        properties,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_input_tokens,
        usage.cache_read_input_tokens,
    );
}

fn append_delta_usage_properties(properties: &mut Vec<(String, String)>, usage: &DeltaUsage) {
    append_message_usage_values(
        properties,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_input_tokens,
        usage.cache_read_input_tokens,
    );
}

fn append_message_usage_values(
    properties: &mut Vec<(String, String)>,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    cache_creation_input_tokens: Option<u32>,
    cache_read_input_tokens: Option<u32>,
) {
    let input_tokens = input_tokens.map(|input_tokens| {
        input_tokens
            + cache_creation_input_tokens.unwrap_or(0)
            + cache_read_input_tokens.unwrap_or(0)
    });

    if let Some(input_tokens) = input_tokens {
        let input_tokens = input_tokens.to_string();
        properties.push(("gen_ai.usage.input_tokens".into(), input_tokens.clone()));
        properties.push(("llm.token_count.prompt".into(), input_tokens));
    }

    if let Some(output_tokens) = output_tokens {
        let output_tokens = output_tokens.to_string();
        properties.push(("gen_ai.usage.output_tokens".into(), output_tokens.clone()));
        properties.push(("llm.token_count.completion".into(), output_tokens));
    }

    if let Some(input_tokens) = input_tokens
        && let Some(output_tokens) = output_tokens
    {
        properties.push((
            "llm.token_count.total".into(),
            (input_tokens + output_tokens).to_string(),
        ));
    }

    if let Some(cache_creation_input_tokens) = cache_creation_input_tokens {
        let cache_creation = cache_creation_input_tokens.to_string();
        properties.push((
            "gen_ai.usage.cache_creation.input_tokens".into(),
            cache_creation.clone(),
        ));
        properties.push((
            "llm.token_count.prompt_details.cache_write".into(),
            cache_creation,
        ));
    }

    if let Some(cache_read_input_tokens) = cache_read_input_tokens {
        let cache_read = cache_read_input_tokens.to_string();
        properties.push((
            "gen_ai.usage.cache_read.input_tokens".into(),
            cache_read.clone(),
        ));
        properties.push((
            "llm.token_count.prompt_details.cache_read".into(),
            cache_read,
        ));
    }
}
