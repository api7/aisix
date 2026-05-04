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

    let span = Span::enter_with_local_parent("aisix.llm.responses");

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
            0usize,
            stream_request_ctx,
            false,
            Some(usage_rx),
        ),
        |(mut stream, span, idx, mut request_ctx, should_terminate, mut usage_rx)| async move {
            if should_terminate {
                drop(span);
                return None;
            }

            match stream.next().await {
                Some(Ok(event)) => {
                    if idx == 0 {
                        hooks::observability::record_first_token_latency(&mut request_ctx).await;
                        span.add_event(
                            TraceEvent::new("first token arrived")
                                .with_property(|| ("kind", "first_token_arrived")),
                        );
                    }

                    let sse_event = Ok::<SseEvent, Infallible>(serialize_stream_event(&event));

                    Some((
                        sse_event,
                        (stream, span, idx + 1, request_ctx, false, usage_rx),
                    ))
                }
                Some(Err(err)) => {
                    error!("Gateway stream error: {}", err);
                    span.add_property(|| ("error.type", "stream_error"));
                    if let Some(usage_rx) = usage_rx.take() {
                        spawn_stream_usage_observer(request_ctx.clone(), usage_rx);
                    }
                    Some((
                        Ok(serialize_stream_event(&ResponsesApiStreamEvent::Error {
                            message: err.to_string(),
                        })),
                        (stream, span, idx + 1, request_ctx, true, usage_rx),
                    ))
                }
                None => {
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
