mod span_attributes;
mod types;

use std::{convert::Infallible, time::Duration};

use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, Sse},
    },
};
use fastrace::prelude::{Event as TraceEvent, *};
use log::error;
use span_attributes::{
    StreamOutputCollector, apply_span_properties, chunk_span_properties, request_span_properties,
    response_span_properties, usage_span_properties,
};
use tokio::sync::{oneshot, oneshot::error::TryRecvError};
pub use types::MessagesError;

use crate::{
    config::entities::{Model, ResourceEntry},
    gateway::{
        error::GatewayError,
        formats::AnthropicMessagesFormat,
        traits::ChatFormat,
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
    utils::future::{WithSpan, maybe_timeout},
};

/// Handles Anthropic Messages API requests on `/v1/messages`.
///
/// Accepts an `AnthropicMessagesRequest` JSON body, applies proxy auth and hook
/// context from `AppState`, `SpanContext`, and `RequestContext`, and returns
/// either a complete Anthropic Messages JSON response or an SSE stream of
/// Anthropic stream events. Failures are mapped into `MessagesError`.
pub async fn messages(
    State(state): State<AppState>,
    mut request_ctx: RequestContext,
    Json(mut request_data): Json<AnthropicMessagesRequest>,
) -> Result<Response, MessagesError> {
    hooks::observability::record_start_time(&mut request_ctx).await;
    hooks::authorization::check(
        &mut request_ctx,
        AnthropicMessagesFormat::extract_model(&request_data).to_owned(),
    )
    .await?;
    hooks::rate_limit::pre_check(&mut request_ctx).await?;

    let model = request_ctx
        .extensions()
        .await
        .get::<ResourceEntry<Model>>()
        .cloned()
        .ok_or(MessagesError::MissingModelInContext)?;

    request_data.model = model.model.clone();
    let timeout = model.timeout.map(Duration::from_millis);

    let gateway = state.gateway();
    let resources = state.resources();
    let provider = model.provider(resources.as_ref()).ok_or_else(|| {
        GatewayError::Internal(format!("provider {} not found", model.provider_id))
    })?;
    let provider_instance = create_provider_instance(gateway.as_ref(), &provider)?;
    let provider_base_url = provider_instance.effective_base_url().ok();

    let span = Span::enter_with_local_parent("aisix.llm.messages");
    apply_span_properties(
        &span,
        request_span_properties(
            &request_data,
            provider_instance.def.as_ref(),
            provider_base_url.as_ref(),
        ),
    );

    let (response, span) = (WithSpan {
        inner: maybe_timeout(timeout, gateway.messages(&request_data, &provider_instance)),
        span: Some(span),
    })
    .await;

    match response {
        Ok(Ok(ChatResponse::Complete { response, usage })) => {
            span.add_properties(|| response_span_properties(&response, &usage));
            handle_regular_request(response, usage, &mut request_ctx).await
        }
        Ok(Ok(ChatResponse::Stream { stream, usage_rx })) => {
            handle_stream_request(stream, usage_rx, &mut request_ctx, span).await
        }
        Ok(Err(err)) => {
            span.add_property(|| ("error.type", "gateway_error"));
            Err(err.into())
        }
        Err(err) => {
            span.add_property(|| ("error.type", "timeout"));
            Err(MessagesError::Timeout(err))
        }
    }
}

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

async fn handle_stream_request(
    stream: ChatResponseStream<AnthropicMessagesFormat>,
    usage_rx: oneshot::Receiver<Usage>,
    request_ctx: &mut RequestContext,
    span: Span,
) -> Result<Response, MessagesError> {
    use futures::stream::StreamExt;

    let stream_request_ctx = request_ctx.clone();
    let sse_stream = futures::stream::unfold(
        (
            stream,
            span,
            0usize,
            stream_request_ctx,
            false,
            Some(usage_rx),
            StreamOutputCollector::default(),
            false,
        ),
        |(
            mut stream,
            span,
            idx,
            mut request_ctx,
            should_terminate,
            mut usage_rx,
            mut output_collector,
            mut first_token_arrived,
        )| async move {
            if should_terminate {
                drop(span);
                return None;
            }

            match stream.next().await {
                Some(Ok(event)) => {
                    output_collector.record_event(&event);

                    if let AnthropicStreamEvent::ContentBlockStart { .. } = event
                        && !first_token_arrived
                    {
                        first_token_arrived = true;
                        hooks::observability::record_first_token_latency(&mut request_ctx).await;
                        span.add_event(
                            TraceEvent::new("first token arrived")
                                .with_property(|| ("kind", "first_token_arrived")),
                        );
                    }

                    span.add_properties(|| chunk_span_properties(&event));

                    let sse_event = Ok::<SseEvent, Infallible>(serialize_stream_event(&event));

                    Some((
                        sse_event,
                        (
                            stream,
                            span,
                            idx + 1,
                            request_ctx,
                            false,
                            usage_rx,
                            output_collector,
                            first_token_arrived,
                        ),
                    ))
                }
                Some(Err(err)) => {
                    error!("Gateway stream error: {}", err);
                    span.add_property(|| ("error.type", "stream_error"));
                    span.add_properties(|| output_collector.output_message_span_properties());
                    if let Some(usage_rx) = usage_rx.take() {
                        spawn_stream_usage_observer(request_ctx.clone(), usage_rx);
                    }
                    Some((
                        Ok(anthropic_error_sse_event(err.to_string())),
                        (
                            stream,
                            span,
                            idx + 1,
                            request_ctx,
                            true,
                            usage_rx,
                            output_collector,
                            first_token_arrived,
                        ),
                    ))
                }
                None => {
                    span.add_properties(|| output_collector.output_message_span_properties());

                    if let Some(mut usage_rx) = usage_rx.take() {
                        match usage_rx.try_recv() {
                            Ok(usage) => {
                                if let Err(err) = hooks::rate_limit::post_check_streaming(
                                    &mut request_ctx,
                                    &usage,
                                )
                                .await
                                {
                                    error!("Rate limit post_check_streaming error: {}", err);
                                }
                                hooks::observability::record_streaming_usage(
                                    &mut request_ctx,
                                    &usage,
                                )
                                .await;
                                span.add_properties(|| usage_span_properties(&usage));
                            }
                            Err(TryRecvError::Empty) => {
                                spawn_stream_usage_observer(request_ctx.clone(), usage_rx);
                            }
                            Err(TryRecvError::Closed) => {
                                error!(
                                    "Failed to receive streaming usage from gateway: channel closed"
                                );
                            }
                        }
                    }

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

fn serialize_stream_event(event: &AnthropicStreamEvent) -> SseEvent {
    let mut sse_event =
        SseEvent::default().data(AnthropicMessagesFormat::serialize_chunk_payload(event));

    if let Some(event_type) = AnthropicMessagesFormat::sse_event_type(event) {
        sse_event = sse_event.event(event_type);
    }

    sse_event
}
