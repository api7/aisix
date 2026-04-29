use fastrace::prelude::Span;

use crate::gateway::types::common::Usage;

pub(crate) fn collect_finish_reasons<I>(finish_reasons: I) -> Vec<String>
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

pub(crate) fn append_finish_reason_properties(
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

pub(crate) fn append_usage_properties(properties: &mut Vec<(String, String)>, usage: &Usage) {
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

pub(crate) fn usage_span_properties(usage: &Usage) -> Vec<(String, String)> {
    let mut properties = Vec::new();
    append_usage_properties(&mut properties, usage);
    properties
}

pub(crate) fn apply_span_properties(span: &Span, properties: Vec<(String, String)>) {
    if properties.is_empty() {
        return;
    }

    span.add_properties(move || properties);
}
