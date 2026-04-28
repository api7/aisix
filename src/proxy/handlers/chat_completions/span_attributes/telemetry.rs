use fastrace::prelude::Span;
use reqwest::Url;
use serde_json::{Map, Value};

use crate::{
    gateway::types::{
        common::Usage,
        openai::{
            ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
            ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage,
            ResponseFormat, StopCondition,
        },
    },
    gateway::traits::ProviderCapabilities,
};

use super::{
    message_attributes::{
        append_openinference_message_properties,
        append_openinference_output_message_properties,
        append_openinference_tool_properties, gen_ai_input_messages_json,
        gen_ai_output_messages_json, gen_ai_tool_definitions_json,
        message_view_from_chat_message, response_output_message_views,
    },
};

pub(in crate::proxy::handlers::chat_completions) fn request_span_properties(
    request: &ChatCompletionRequest,
    provider: &dyn ProviderCapabilities,
    base_url: Option<&Url>,
) -> Vec<(String, String)> {
    let provider_semantics = provider.semantic_conventions();
    let input_messages: Vec<_> = request
        .messages
        .iter()
        .map(message_view_from_chat_message)
        .collect();
    let mut properties = vec![
        ("gen_ai.operation.name".into(), "chat".into()),
        ("openinference.span.kind".into(), "LLM".into()),
        (
            "gen_ai.provider.name".into(),
            provider_semantics.gen_ai_provider_name.to_string(),
        ),
        ("llm.system".into(), provider_semantics.llm_system.to_string()),
        ("gen_ai.request.model".into(), request.model.clone()),
    ];

    if let Some(llm_provider) = provider_semantics.llm_provider {
        properties.push(("llm.provider".into(), llm_provider.to_string()));
    }

    if let Some(choice_count) = request.n.filter(|count| *count != 1) {
        properties.push((
            "gen_ai.request.choice.count".into(),
            choice_count.to_string(),
        ));
    }

    if let Some(seed) = request.seed {
        properties.push(("gen_ai.request.seed".into(), seed.to_string()));
    }

    if let Some(max_tokens) = request.max_completion_tokens.or(request.max_tokens) {
        properties.push(("gen_ai.request.max_tokens".into(), max_tokens.to_string()));
    }

    if let Some(value) = request.frequency_penalty {
        properties.push(("gen_ai.request.frequency_penalty".into(), value.to_string()));
    }

    if let Some(value) = request.presence_penalty {
        properties.push(("gen_ai.request.presence_penalty".into(), value.to_string()));
    }

    if let Some(value) = request.temperature {
        properties.push(("gen_ai.request.temperature".into(), value.to_string()));
    }

    if let Some(value) = request.top_p {
        properties.push(("gen_ai.request.top_p".into(), value.to_string()));
    }

    if let Some(value) = numeric_extra_to_string(request.extra.get("top_k")) {
        properties.push(("gen_ai.request.top_k".into(), value));
    }

    if let Some(value) = stop_sequences_json(request.stop.as_ref()) {
        properties.push(("gen_ai.request.stop_sequences".into(), value));
    }

    if let Some(value) = response_format_output_type(request.response_format.as_ref()) {
        properties.push(("gen_ai.output.type".into(), value.to_string()));
    }

    if let Some(value) = request_invocation_parameters(request) {
        properties.push(("llm.invocation_parameters".into(), value));
    }

    if let Some(user_id) = request.user.as_ref().filter(|user_id| !user_id.is_empty()) {
        properties.push(("user.id".into(), user_id.clone()));
    }

    append_openinference_message_properties(
        &mut properties,
        "llm.input_messages",
        &input_messages,
    );

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

pub(in crate::proxy::handlers::chat_completions) fn response_span_properties(
    response: &ChatCompletionResponse,
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
        collect_finish_reasons(response.choices.iter().map(choice_finish_reason)),
    );
    append_response_usage_properties(&mut properties, usage, response.usage.as_ref());
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

pub(in crate::proxy::handlers::chat_completions) fn chunk_span_properties(
    chunk: &ChatCompletionChunk,
) -> Vec<(String, String)> {
    let mut properties = Vec::new();

    if !chunk.id.is_empty() {
        properties.push(("gen_ai.response.id".into(), chunk.id.clone()));
    }

    if !chunk.model.is_empty() {
        properties.push(("gen_ai.response.model".into(), chunk.model.clone()));
        properties.push(("llm.model_name".into(), chunk.model.clone()));
    }

    append_finish_reason_properties(
        &mut properties,
        collect_finish_reasons(chunk.choices.iter().map(chunk_choice_finish_reason)),
    );
    append_chunk_usage_properties(&mut properties, chunk.usage.as_ref());

    properties
}

pub(in crate::proxy::handlers::chat_completions) fn usage_span_properties(
    usage: &Usage,
) -> Vec<(String, String)> {
    let mut properties = Vec::new();
    append_usage_properties(&mut properties, usage);
    properties
}

pub(in crate::proxy::handlers::chat_completions) fn apply_span_properties(
    span: &Span,
    properties: Vec<(String, String)>,
) {
    if properties.is_empty() {
        return;
    }

    span.add_properties(move || properties);
}

fn response_format_output_type(response_format: Option<&ResponseFormat>) -> Option<&'static str> {
    match response_format?.r#type.as_str() {
        "json_object" | "json_schema" => Some("json"),
        "text" => Some("text"),
        "image" => Some("image"),
        "speech" => Some("speech"),
        _ => None,
    }
}

fn numeric_extra_to_string(value: Option<&Value>) -> Option<String> {
    let value = value?;

    if let Some(integer) = value.as_i64() {
        return Some(integer.to_string());
    }

    if let Some(integer) = value.as_u64() {
        return Some(integer.to_string());
    }

    value.as_f64().map(|float| float.to_string())
}

fn stop_sequences_json(stop: Option<&StopCondition>) -> Option<String> {
    let stop = stop?;
    let value = match stop {
        StopCondition::Single(value) => Value::Array(vec![Value::String(value.clone())]),
        StopCondition::Multiple(values) => {
            Value::Array(values.iter().cloned().map(Value::String).collect())
        }
    };

    serde_json::to_string(&value).ok()
}

fn request_invocation_parameters(request: &ChatCompletionRequest) -> Option<String> {
    let mut params = Map::new();

    if let Some(value) = request.frequency_penalty {
        params.insert("frequency_penalty".into(), Value::from(value));
    }
    if let Some(value) = request.logprobs {
        params.insert("logprobs".into(), Value::from(value));
    }
    if let Some(value) = request.top_logprobs {
        params.insert("top_logprobs".into(), Value::from(value));
    }
    if let Some(value) = request.max_tokens {
        params.insert("max_tokens".into(), Value::from(value));
    }
    if let Some(value) = request.max_completion_tokens {
        params.insert("max_completion_tokens".into(), Value::from(value));
    }
    if let Some(value) = request.n {
        params.insert("n".into(), Value::from(value));
    }
    if let Some(value) = request.presence_penalty {
        params.insert("presence_penalty".into(), Value::from(value));
    }
    if let Some(value) = request.seed {
        params.insert("seed".into(), Value::from(value));
    }
    if let Some(value) = stop_sequences_json(request.stop.as_ref()) {
        if let Ok(parsed) = serde_json::from_str(&value) {
            params.insert("stop".into(), parsed);
        }
    }
    if let Some(value) = request.stream {
        params.insert("stream".into(), Value::from(value));
    }
    if let Some(value) = request.temperature {
        params.insert("temperature".into(), Value::from(value));
    }
    if let Some(value) = request.top_p {
        params.insert("top_p".into(), Value::from(value));
    }
    if let Some(value) = request.parallel_tool_calls {
        params.insert("parallel_tool_calls".into(), Value::from(value));
    }
    if let Some(value) = request.response_format.as_ref() {
        if let Ok(value) = serde_json::to_value(value) {
            params.insert("response_format".into(), value);
        }
    }
    if let Some(value) = request.stream_options.as_ref() {
        if let Ok(value) = serde_json::to_value(value) {
            params.insert("stream_options".into(), value);
        }
    }
    if let Some(value) = request.tool_choice.as_ref() {
        if let Ok(value) = serde_json::to_value(value) {
            params.insert("tool_choice".into(), value);
        }
    }

    if params.is_empty() {
        return None;
    }

    serde_json::to_string(&Value::Object(params)).ok()
}

fn collect_finish_reasons<I>(finish_reasons: I) -> Vec<String>
where
    I: IntoIterator<Item = Option<String>>,
{
    let mut values = Vec::new();

    for finish_reason in finish_reasons.into_iter().flatten() {
        if !values.iter().any(|value| value == &finish_reason) {
            values.push(finish_reason);
        }
    }

    values
}

fn append_finish_reason_properties(
    properties: &mut Vec<(String, String)>,
    finish_reasons: Vec<String>,
) {
    if finish_reasons.is_empty() {
        return;
    }

    properties.push((
        "gen_ai.response.finish_reasons".into(),
        serde_json::to_string(&finish_reasons).unwrap_or_default(),
    ));

    if let Some(finish_reason) = finish_reasons.first() {
        properties.push(("llm.finish_reason".into(), finish_reason.clone()));
    }
}

fn append_usage_properties(properties: &mut Vec<(String, String)>, usage: &Usage) {
    if let Some(input_tokens) = usage.input_tokens {
        let input_tokens = input_tokens.to_string();
        properties.push(("gen_ai.usage.input_tokens".into(), input_tokens.clone()));
        properties.push(("llm.token_count.prompt".into(), input_tokens));
    }

    if let Some(output_tokens) = usage.output_tokens {
        let output_tokens = output_tokens.to_string();
        properties.push(("gen_ai.usage.output_tokens".into(), output_tokens.clone()));
        properties.push(("llm.token_count.completion".into(), output_tokens));
    }

    if let Some(total_tokens) = usage.resolved_total_tokens() {
        properties.push(("llm.token_count.total".into(), total_tokens.to_string()));
    }

    if let Some(cache_creation_input_tokens) = usage.cache_creation_input_tokens {
        let cache_creation_input_tokens = cache_creation_input_tokens.to_string();
        properties.push((
            "gen_ai.usage.cache_creation.input_tokens".into(),
            cache_creation_input_tokens.clone(),
        ));
        properties.push((
            "llm.token_count.prompt_details.cache_write".into(),
            cache_creation_input_tokens,
        ));
    }

    if let Some(cache_read_input_tokens) = usage.cache_read_input_tokens {
        let cache_read_input_tokens = cache_read_input_tokens.to_string();
        properties.push((
            "gen_ai.usage.cache_read.input_tokens".into(),
            cache_read_input_tokens.clone(),
        ));
        properties.push((
            "llm.token_count.prompt_details.cache_read".into(),
            cache_read_input_tokens,
        ));
    }

    if let Some(input_audio_tokens) = usage.input_audio_tokens {
        properties.push((
            "llm.token_count.prompt_details.audio".into(),
            input_audio_tokens.to_string(),
        ));
    }

    if let Some(output_audio_tokens) = usage.output_audio_tokens {
        properties.push((
            "llm.token_count.completion_details.audio".into(),
            output_audio_tokens.to_string(),
        ));
    }
}

fn append_response_usage_properties(
    properties: &mut Vec<(String, String)>,
    usage: &Usage,
    raw_usage: Option<&ChatCompletionUsage>,
) {
    append_usage_properties(properties, usage);

    let Some(raw_usage) = raw_usage else {
        return;
    };

    if usage.input_tokens.is_none() {
        let input_tokens = raw_usage.prompt_tokens.to_string();
        properties.push(("gen_ai.usage.input_tokens".into(), input_tokens.clone()));
        properties.push(("llm.token_count.prompt".into(), input_tokens));
    }

    if usage.output_tokens.is_none() {
        let output_tokens = raw_usage.completion_tokens.to_string();
        properties.push(("gen_ai.usage.output_tokens".into(), output_tokens.clone()));
        properties.push(("llm.token_count.completion".into(), output_tokens));
    }

    if usage.resolved_total_tokens().is_none() {
        properties.push((
            "llm.token_count.total".into(),
            raw_usage.total_tokens.to_string(),
        ));
    }

    if usage.cache_read_input_tokens.is_none() {
        if let Some(cached_tokens) = raw_usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
        {
            let cached_tokens = cached_tokens.to_string();
            properties.push((
                "gen_ai.usage.cache_read.input_tokens".into(),
                cached_tokens.clone(),
            ));
            properties.push((
                "llm.token_count.prompt_details.cache_read".into(),
                cached_tokens,
            ));
        }
    }

    if usage.input_audio_tokens.is_none() {
        if let Some(audio_tokens) = raw_usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens)
        {
            properties.push((
                "llm.token_count.prompt_details.audio".into(),
                audio_tokens.to_string(),
            ));
        }
    }

    if let Some(reasoning_tokens) = raw_usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens)
    {
        properties.push((
            "llm.token_count.completion_details.reasoning".into(),
            reasoning_tokens.to_string(),
        ));
    }

    if usage.output_audio_tokens.is_none() {
        if let Some(audio_tokens) = raw_usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens)
        {
            properties.push((
                "llm.token_count.completion_details.audio".into(),
                audio_tokens.to_string(),
            ));
        }
    }
}

fn append_chunk_usage_properties(
    properties: &mut Vec<(String, String)>,
    raw_usage: Option<&ChatCompletionUsage>,
) {
    if let Some(reasoning_tokens) = raw_usage
        .and_then(|usage| usage.completion_tokens_details.as_ref())
        .and_then(|details| details.reasoning_tokens)
    {
        properties.push((
            "llm.token_count.completion_details.reasoning".into(),
            reasoning_tokens.to_string(),
        ));
    }
}

fn choice_finish_reason(choice: &ChatCompletionChoice) -> Option<String> {
    choice.finish_reason.clone()
}

fn chunk_choice_finish_reason(choice: &ChatCompletionChunkChoice) -> Option<String> {
    choice.finish_reason.clone()
}
