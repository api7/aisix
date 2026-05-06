mod span_attributes;
mod types;

use axum::response::sse::Event as SseEvent;
use fastrace::Span;
use reqwest::Url;
use span_attributes::{
    StreamOutputCollector, chunk_span_properties, event_starts_output, request_span_properties,
    response_span_properties,
};
pub use types::ResponsesError;

use crate::{
    gateway::{
        error::GatewayError,
        formats::ResponsesApiFormat,
        traits::{ChatFormat, ProviderCapabilities},
        types::{
            common::Usage,
            openai::responses::{
                ResponsesApiRequest, ResponsesApiResponse, ResponsesApiStreamEvent,
            },
        },
    },
    proxy::handlers::format_handler::FormatHandlerAdapter,
};

fn serialize_stream_event(event: &ResponsesApiStreamEvent) -> SseEvent {
    let mut sse_event =
        SseEvent::default().data(ResponsesApiFormat::serialize_chunk_payload(event));

    if let Some(event_type) = ResponsesApiFormat::sse_event_type(event) {
        sse_event = sse_event.event(event_type);
    }

    sse_event
}

pub(crate) struct ResponsesAdapter;

impl FormatHandlerAdapter for ResponsesAdapter {
    type Format = ResponsesApiFormat;
    type Request = ResponsesApiRequest;
    type Response = ResponsesApiResponse;
    type StreamChunk = ResponsesApiStreamEvent;
    type Error = ResponsesError;
    type Collector = StreamOutputCollector;

    fn span_name() -> &'static str {
        "aisix.llm.responses"
    }

    fn missing_model_error() -> Self::Error {
        ResponsesError::MissingModelInContext
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
        event_starts_output(chunk)
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
        Some(serialize_stream_event(&ResponsesApiStreamEvent::Error {
            message: error.to_string(),
        }))
    }
}
