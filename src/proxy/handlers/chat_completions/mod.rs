mod span_attributes;
mod types;

use axum::response::sse::Event as SseEvent;
use fastrace::Span;
use opentelemetry_semantic_conventions::attribute::GEN_AI_RESPONSE_FINISH_REASONS;
use reqwest::Url;
use span_attributes::{
    StreamOutputCollector, chunk_span_properties, request_span_properties, response_span_properties,
};
pub use types::ChatCompletionError;

use super::FormatHandlerAdapter;
use crate::{
    gateway::{
        error::GatewayError,
        formats::OpenAIChatFormat,
        traits::{ChatFormat, ProviderCapabilities},
        types::{
            common::Usage,
            openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
        },
    },
    proxy::guardrails::{
        input_guardrail_payload_from_chat_messages, input_payload_from_check_payload,
        input_payload_to_chat_messages, output_guardrail_payload_from_chat_messages,
        output_payload_from_check_payload, output_payload_to_chat_messages,
    },
};

pub(crate) struct ChatCompletionsAdapter;

impl FormatHandlerAdapter for ChatCompletionsAdapter {
    type Format = OpenAIChatFormat;
    type Request = ChatCompletionRequest;
    type Response = ChatCompletionResponse;
    type StreamChunk = ChatCompletionChunk;
    type Error = ChatCompletionError;
    type Collector = StreamOutputCollector;
    type LifecycleState = ();

    fn span_name() -> &'static str {
        "aisix.llm.chat_completions"
    }

    fn missing_model_error() -> Self::Error {
        ChatCompletionError::MissingModelInContext
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

    fn apply_chunk_span_properties(span: &Span, chunk: &Self::StreamChunk, is_first_item: bool) {
        if is_first_item {
            span.add_properties(|| chunk_span_properties(chunk));
            return;
        }

        let properties = chunk_span_properties(chunk);
        properties
            .iter()
            .filter(|(key, _)| {
                key == GEN_AI_RESPONSE_FINISH_REASONS
                    || key == "llm.finish_reason"
                    || key == "llm.token_count.completion_details.reasoning"
            })
            .for_each(|item| span.add_property(|| item.clone()));
    }

    fn starts_output(_chunk: &Self::StreamChunk) -> bool {
        true
    }

    fn record_stream_item(collector: &mut Self::Collector, chunk: &Self::StreamChunk) {
        collector.record_chunk(chunk);
    }

    fn output_message_span_properties(collector: &Self::Collector) -> Vec<(String, String)> {
        collector.output_message_span_properties()
    }

    fn guardrail_input_payload(
        _lifecycle_state: &Self::LifecycleState,
        request: &Self::Request,
    ) -> Result<Option<crate::guardrail::traits::GuardrailCheckPayload>, Self::Error> {
        let (hub_request, _) = OpenAIChatFormat::to_hub(request)?;
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
        request.messages = input_payload_to_chat_messages(
            &input_payload_from_check_payload(rewrite).map_err(bridge_error)?,
        )
        .map_err(bridge_error)?;
        Ok(())
    }

    fn guardrail_output_payload(
        _lifecycle_state: &Self::LifecycleState,
        response: &Self::Response,
    ) -> Result<Option<crate::guardrail::traits::GuardrailCheckPayload>, Self::Error> {
        let messages = response
            .choices
            .iter()
            .map(|choice| choice.message.clone())
            .collect::<Vec<_>>();
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

        if messages.len() != response.choices.len() {
            return Err(bridge_error(GatewayError::Bridge(format!(
                "chat completion output guardrail rewrite expected {} messages, got {}",
                response.choices.len(),
                messages.len()
            ))));
        }

        for (choice, message) in response.choices.iter_mut().zip(messages) {
            choice.message = message;
        }

        Ok(())
    }

    fn end_of_stream_event(saw_item: bool) -> Option<SseEvent> {
        saw_item.then(|| SseEvent::default().data("[DONE]"))
    }
}

fn bridge_error<E>(error: E) -> ChatCompletionError
where
    E: std::fmt::Display,
{
    ChatCompletionError::GatewayError(GatewayError::Bridge(error.to_string()))
}
