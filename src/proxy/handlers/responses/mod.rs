mod runtime;
mod span_attributes;
mod types;

use async_trait::async_trait;
use axum::response::sse::Event as SseEvent;
use fastrace::Span;
use reqwest::Url;
use runtime::{
    ResponsesLifecycleState, accumulate_complete, accumulate_stream_event,
    accumulate_stream_success, build_merged_input_messages, init_lifecycle, load_previous_messages,
    persist_if_enabled,
};
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
                ResponsesOutputItem,
            },
        },
    },
    proxy::{AppState, handlers::format_handler::FormatHandlerAdapter, hooks::RequestContext},
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

#[async_trait]
impl FormatHandlerAdapter for ResponsesAdapter {
    type Format = ResponsesApiFormat;
    type Request = ResponsesApiRequest;
    type Response = ResponsesApiResponse;
    type StreamChunk = ResponsesApiStreamEvent;
    type Error = ResponsesError;
    type Collector = StreamOutputCollector;
    type LifecycleState = ResponsesLifecycleState;

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

    async fn prepare_lifecycle(
        state: &AppState,
        _request_ctx: &mut RequestContext,
        request: &mut Self::Request,
    ) -> Result<Self::LifecycleState, Self::Error> {
        let mut lifecycle_state = init_lifecycle(request);
        let storage = state.message_history_storage();
        let previous_messages =
            load_previous_messages(storage.as_ref(), request.previous_response_id.as_deref())
                .await?;
        lifecycle_state.merged_input_messages =
            build_merged_input_messages(request, &previous_messages)?;
        request.replay_messages = previous_messages;

        Ok(lifecycle_state)
    }

    async fn handle_complete_response(
        state: &AppState,
        _request_ctx: &mut RequestContext,
        lifecycle_state: &mut Self::LifecycleState,
        response: &mut Self::Response,
        usage: &Usage,
    ) -> Result<(), Self::Error> {
        let stored_history = accumulate_complete(
            lifecycle_state,
            response,
            usage.clone().with_derived_total(),
        )?;
        let storage = state.message_history_storage();
        persist_if_enabled(storage.as_ref(), lifecycle_state, &stored_history).await?;
        rewrite_response_ids(response, &stored_history.response_id);
        Ok(())
    }

    fn handle_stream_item(
        _state: &AppState,
        _request_ctx: &mut RequestContext,
        lifecycle_state: &mut Self::LifecycleState,
        chunk: &mut Self::StreamChunk,
    ) {
        accumulate_stream_event(lifecycle_state, chunk);
        rewrite_stream_event_ids(chunk, &lifecycle_state.response_id);
    }

    async fn handle_stream_success(
        state: &AppState,
        _request_ctx: &mut RequestContext,
        lifecycle_state: Self::LifecycleState,
        usage: Option<&Usage>,
    ) -> Result<(), Self::Error> {
        let stored_history = accumulate_stream_success(&lifecycle_state, usage)?;
        let storage = state.message_history_storage();
        persist_if_enabled(storage.as_ref(), &lifecycle_state, &stored_history).await?;
        Ok(())
    }

    async fn handle_stream_failure(
        _state: &AppState,
        _request_ctx: &mut RequestContext,
        _lifecycle_state: Self::LifecycleState,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn serialize_stream_item(chunk: &Self::StreamChunk) -> SseEvent {
        serialize_stream_event(chunk)
    }

    fn stream_error_event(error: &GatewayError) -> Option<SseEvent> {
        Some(serialize_stream_event(&ResponsesApiStreamEvent::Error {
            message: error.to_string(),
        }))
    }

    fn lifecycle_error_event(error: &Self::Error) -> Option<SseEvent> {
        Some(serialize_stream_event(&ResponsesApiStreamEvent::Error {
            message: error.to_string(),
        }))
    }
}

fn rewrite_response_ids(response: &mut ResponsesApiResponse, response_id: &str) {
    response.id = response_id.to_owned();
    for (output_index, item) in response.output.iter_mut().enumerate() {
        rewrite_output_item_ids(item, response_id, output_index);
    }
}

fn rewrite_stream_event_ids(event: &mut ResponsesApiStreamEvent, response_id: &str) {
    match event {
        ResponsesApiStreamEvent::ResponseCreated { response }
        | ResponsesApiStreamEvent::ResponseInProgress { response }
        | ResponsesApiStreamEvent::ResponseCompleted { response } => {
            rewrite_response_ids(response, response_id);
        }
        ResponsesApiStreamEvent::OutputItemAdded { output_index, item }
        | ResponsesApiStreamEvent::OutputItemDone { output_index, item } => {
            rewrite_output_item_ids(item, response_id, *output_index);
        }
        _ => {}
    }
}

fn rewrite_output_item_ids(item: &mut ResponsesOutputItem, response_id: &str, output_index: usize) {
    if let ResponsesOutputItem::Message { id, .. } = item {
        *id = format!("{}_message_{}", response_id, output_index);
    }
}
