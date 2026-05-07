mod span_attributes;
mod types;

use axum::response::sse::Event as SseEvent;
use fastrace::Span;
use reqwest::Url;
use span_attributes::{
    StreamOutputCollector, chunk_span_properties, request_span_properties, response_span_properties,
};
pub use types::MessagesError;

use crate::{
    gateway::{
        error::GatewayError,
        formats::AnthropicMessagesFormat,
        traits::{ChatFormat, ProviderCapabilities},
        types::{
            anthropic::{
                AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicStreamEvent,
            },
            common::Usage,
        },
    },
    proxy::handlers::format_handler::FormatHandlerAdapter,
};

fn anthropic_error_sse_event(message: String) -> SseEvent {
    SseEvent::default()
        .event("error")
        .data(anthropic_error_event_payload(message))
}

fn anthropic_error_event_payload(message: String) -> String {
    serde_json::json!({
        "type": "error",
        "error": {
            "type": "api_error",
            "message": message,
        }
    })
    .to_string()
}

fn serialize_stream_event(event: &AnthropicStreamEvent) -> SseEvent {
    let mut sse_event =
        SseEvent::default().data(AnthropicMessagesFormat::serialize_chunk_payload(event));

    if let Some(event_type) = AnthropicMessagesFormat::sse_event_type(event) {
        sse_event = sse_event.event(event_type);
    }

    sse_event
}

pub(crate) struct MessagesAdapter;

impl FormatHandlerAdapter for MessagesAdapter {
    type Format = AnthropicMessagesFormat;
    type Request = AnthropicMessagesRequest;
    type Response = AnthropicMessagesResponse;
    type StreamChunk = AnthropicStreamEvent;
    type Error = MessagesError;
    type Collector = StreamOutputCollector;
    type LifecycleState = ();

    fn span_name() -> &'static str {
        "aisix.llm.messages"
    }

    fn missing_model_error() -> Self::Error {
        MessagesError::MissingModelInContext
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

    fn apply_chunk_span_properties(span: &Span, chunk: &Self::StreamChunk, _is_first_item: bool) {
        span.add_properties(|| chunk_span_properties(chunk));
    }

    fn starts_output(chunk: &Self::StreamChunk) -> bool {
        matches!(chunk, AnthropicStreamEvent::ContentBlockStart { .. })
    }

    fn record_stream_item(collector: &mut Self::Collector, chunk: &Self::StreamChunk) {
        collector.record_event(chunk);
    }

    fn output_message_span_properties(collector: &Self::Collector) -> Vec<(String, String)> {
        collector.output_message_span_properties()
    }

    fn serialize_stream_item(chunk: &Self::StreamChunk) -> SseEvent {
        serialize_stream_event(chunk)
    }

    fn stream_error_event(error: &GatewayError) -> Option<SseEvent> {
        Some(anthropic_error_sse_event(error.to_string()))
    }
}
