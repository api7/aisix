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

use crate::{
    gateway::{
        formats::OpenAIChatFormat,
        traits::ProviderCapabilities,
        types::{
            common::Usage,
            openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
        },
    },
    proxy::handlers::format_handler::FormatHandlerAdapter,
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

    fn end_of_stream_event(saw_item: bool) -> Option<SseEvent> {
        saw_item.then(|| SseEvent::default().data("[DONE]"))
    }
}
