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
    StreamOutputCollector, apply_span_properties, chunk_span_properties, event_starts_output,
    request_span_properties, response_span_properties, usage_span_properties,
};
use tokio::sync::{oneshot, oneshot::error::TryRecvError};
pub use types::ResponsesError;

use crate::{
    config::entities::{Model, ResourceEntry},
    gateway::{
        error::GatewayError,
        formats::ResponsesApiFormat,
        traits::ChatFormat,
        types::{
            common::Usage,
            openai::responses::{
                ResponsesApiRequest, ResponsesApiResponse, ResponsesApiStreamEvent,
            },
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

pub async fn responses(
    State(state): State<AppState>,
    mut request_ctx: RequestContext,
    Json(mut request_data): Json<ResponsesApiRequest>,
) -> Result<Response, ResponsesError> {
    hooks::observability::record_start_time(&mut request_ctx).await;
    hooks::authorization::check(
        &mut request_ctx,
        ResponsesApiFormat::extract_model(&request_data).to_owned(),
    )
    .await?;
    hooks::rate_limit::pre_check(&mut request_ctx).await?;

    let model = request_ctx
        .extensions()
        .await
        .get::<ResourceEntry<Model>>()
        .cloned()
        .ok_or(ResponsesError::MissingModelInContext)?;

    request_data.model = model.model.clone();
    let timeout = model.timeout.map(Duration::from_millis);

    let gateway = state.gateway();
    let resources = state.resources();
    let provider = model.provider(resources.as_ref()).ok_or_else(|| {
        GatewayError::Internal(format!("provider {} not found", model.provider_id))
    })?;
    let provider_instance = create_provider_instance(gateway.as_ref(), &provider)?;
    let provider_base_url = provider_instance.effective_base_url().ok();

    let span = Span::enter_with_local_parent("aisix.llm.responses");
    apply_span_properties(
        &span,
        request_span_properties(
            &request_data,
            provider_instance.def.as_ref(),
            provider_base_url.as_ref(),
        ),
    );

    let (response, span) = (WithSpan {
        inner: maybe_timeout(
            timeout,
            gateway.chat::<ResponsesApiFormat>(&request_data, &provider_instance),
        ),
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
            Err(ResponsesError::Timeout(err))
        }
    }
}

async fn handle_regular_request(
    response: ResponsesApiResponse,
    usage: Usage,
    request_ctx: &mut RequestContext,
) -> Result<Response, ResponsesError> {
    if let Err(err) = hooks::rate_limit::post_check(request_ctx, &usage).await {
        error!("Rate limit post_check error: {}", err);
    }

    let mut response = Json(response).into_response();
    hooks::rate_limit::inject_response_headers(request_ctx, response.headers_mut()).await;
    hooks::observability::record_usage(request_ctx, &usage).await;

    Ok(response)
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
    stream: ChatResponseStream<ResponsesApiFormat>,
    usage_rx: oneshot::Receiver<Usage>,
    request_ctx: &mut RequestContext,
    span: Span,
) -> Result<Response, ResponsesError> {
    use futures::stream::StreamExt;

    let stream_request_ctx = request_ctx.clone();
    let sse_stream = futures::stream::unfold(
        (
            stream,
            span,
            stream_request_ctx,
            false,
            Some(usage_rx),
            StreamOutputCollector::default(),
            false,
        ),
        |(
            mut stream,
            span,
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

                    if event_starts_output(&event) && !first_token_arrived {
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

                    Some((
                        Ok(serialize_stream_event(&ResponsesApiStreamEvent::Error {
                            message: err.to_string(),
                        })),
                        (
                            stream,
                            span,
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

fn serialize_stream_event(event: &ResponsesApiStreamEvent) -> SseEvent {
    let mut sse_event =
        SseEvent::default().data(ResponsesApiFormat::serialize_chunk_payload(event));

    if let Some(event_type) = ResponsesApiFormat::sse_event_type(event) {
        sse_event = sse_event.event(event_type);
    }

    sse_event
}
