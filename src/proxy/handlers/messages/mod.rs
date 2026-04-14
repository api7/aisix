mod types;

use std::{convert::Infallible, time::Duration};

use axum::{
    Json,
    extract::{Extension, State},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, Sse},
    },
};
use fastrace::prelude::{Event as TraceEvent, *};
use log::error;
use tokio::sync::oneshot;
pub use types::MessagesError;

use crate::{
    config::entities::{Model, ResourceEntry},
    gateway::{
        formats::AnthropicMessagesFormat,
        types::{
            anthropic::{
                AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicStreamEvent,
            },
            common::Usage,
            response::{ChatResponse, ChatResponseStream},
        },
    },
    proxy::{
        AppState,
        hooks::{self, RequestContext},
        provider::create_provider_instance,
    },
    utils::future::maybe_timeout,
};

/// Handles Anthropic Messages API requests on `/v1/messages`.
///
/// Accepts an `AnthropicMessagesRequest` JSON body, applies proxy auth and hook
/// context from `AppState`, `SpanContext`, and `RequestContext`, and returns
/// either a complete Anthropic Messages JSON response or an SSE stream of
/// Anthropic stream events. Failures are mapped into `MessagesError`.
#[fastrace::trace]
pub async fn messages(
    State(state): State<AppState>,
    Extension(span_ctx): Extension<SpanContext>,
    mut request_ctx: RequestContext,
    Json(mut request_data): Json<AnthropicMessagesRequest>,
) -> Result<Response, MessagesError> {
    hooks::observability::record_start_time(&mut request_ctx).await;
    hooks::authorization::check(&mut request_ctx, request_data.model.clone()).await?;
    hooks::rate_limit::pre_check(&mut request_ctx).await?;

    let model = request_ctx
        .extensions()
        .await
        .get::<ResourceEntry<Model>>()
        .cloned()
        .ok_or(MessagesError::MissingModelInContext)?;

    request_data.model = model.model.name.clone();
    let timeout = model.timeout.map(Duration::from_millis);

    let gateway = state.gateway();
    let provider_instance = create_provider_instance(gateway.as_ref(), &model)?;

    match maybe_timeout(timeout, gateway.messages(&request_data, &provider_instance)).await {
        Ok(response) => match response? {
            ChatResponse::Complete { response, usage } => {
                handle_regular_request(response, usage, &mut request_ctx).await
            }
            ChatResponse::Stream { stream, usage_rx } => {
                handle_stream_request(stream, usage_rx, &mut request_ctx, span_ctx).await
            }
        },
        Err(err) => Err(MessagesError::Timeout(err)),
    }
}

#[fastrace::trace]
async fn handle_regular_request(
    response: AnthropicMessagesResponse,
    usage: Usage,
    request_ctx: &mut RequestContext,
) -> Result<Response, MessagesError> {
    if let Err(err) = hooks::rate_limit::post_check(request_ctx, &usage).await {
        error!("Rate limit post_check error: {}", err);
    }

    let mut resp = Json(response).into_response();
    hooks::rate_limit::inject_response_headers(request_ctx, resp.headers_mut()).await;
    hooks::observability::record_usage(request_ctx, &usage).await;

    Ok(resp)
}

fn spawn_stream_usage_observer(request_ctx: RequestContext, usage_rx: oneshot::Receiver<Usage>) {
    tokio::spawn(async move {
        let mut request_ctx = request_ctx;

        match usage_rx.await {
            Ok(usage) => {
                if let Err(err) =
                    hooks::rate_limit::post_check_streaming(&mut request_ctx, &usage).await
                {
                    error!("Rate limit post_check_streaming error: {}", err);
                }
                hooks::observability::record_streaming_usage(&mut request_ctx, &usage).await;
            }
            Err(err) => {
                error!("Failed to receive streaming usage from gateway: {}", err);
            }
        }
    });
}

#[fastrace::trace]
async fn handle_stream_request(
    stream: ChatResponseStream<AnthropicMessagesFormat>,
    usage_rx: oneshot::Receiver<Usage>,
    request_ctx: &mut RequestContext,
    span_ctx: SpanContext,
) -> Result<Response, MessagesError> {
    use futures::stream::StreamExt;

    spawn_stream_usage_observer(request_ctx.clone(), usage_rx);

    let stream_request_ctx = request_ctx.clone();
    let stream_span = Span::root("messages_sse_connection", span_ctx);
    let sse_stream = futures::stream::unfold(
        (stream, stream_span, 0usize, stream_request_ctx, false),
        |(mut stream, span, idx, mut request_ctx, should_terminate)| async move {
            if should_terminate {
                drop(span);
                return None;
            }

            match stream.next().await {
                Some(Ok(event)) => {
                    if idx == 0 {
                        hooks::observability::record_first_token_latency(&mut request_ctx).await;
                        span.add_event(TraceEvent::new("first token arrived"));
                    }

                    let (sse_event, should_terminate) = match serialize_stream_event(&event) {
                        Ok(event) => (Ok::<SseEvent, Infallible>(event), false),
                        Err(err) => {
                            error!("Failed to serialize anthropic stream event: {}", err);
                            (Ok(anthropic_error_sse_event(err.to_string())), true)
                        }
                    };

                    Some((
                        sse_event,
                        (stream, span, idx + 1, request_ctx, should_terminate),
                    ))
                }
                Some(Err(err)) => {
                    error!("Gateway stream error: {}", err);
                    Some((
                        Ok(anthropic_error_sse_event(err.to_string())),
                        (stream, span, idx + 1, request_ctx, true),
                    ))
                }
                None => {
                    drop(span);
                    None
                }
            }
        },
    );

    let mut response = Sse::new(sse_stream).into_response();
    hooks::rate_limit::inject_response_headers(request_ctx, response.headers_mut()).await;
    Ok(response)
}

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

fn serialize_stream_event(event: &AnthropicStreamEvent) -> Result<SseEvent, serde_json::Error> {
    Ok(SseEvent::default()
        .event(event.event_type())
        .data(serde_json::to_string(event)?))
}
