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
use futures::stream::BoxStream;
use log::error;
pub use types::*;

use crate::{
    config::entities::{Model, ResourceEntry},
    providers::{Provider, ProviderError, create_provider},
    proxy::{
        AppState,
        hooks::{self, RequestContext, ResponseData, TokenUsage},
    },
    utils::future::maybe_timeout,
};

#[fastrace::trace]
pub async fn chat_completions(
    State(_state): State<AppState>,
    Extension(span_ctx): Extension<SpanContext>,
    mut request_ctx: RequestContext,
    Json(mut request_data): Json<ChatCompletionRequest>,
) -> Result<Response, ChatCompletionError> {
    hooks::observability::record_start_time(&mut request_ctx).await;
    hooks::authorization::check(&mut request_ctx, request_data.model.clone()).await?;
    hooks::rate_limit::pre_check(&mut request_ctx).await?;

    let model = request_ctx
        .extensions()
        .await
        .get::<ResourceEntry<Model>>()
        .cloned()
        .unwrap(); //TODO: safe unwrap

    let provider = create_provider(&model.provider_config);
    let timeout = model.timeout.map(Duration::from_millis);

    // Replace request model name with real model name
    request_data.model = model.model.name.clone();

    // Check if it's a streaming request
    let is_stream = request_data.stream.unwrap_or(false);

    let response = if is_stream {
        handle_stream_request(provider, request_data, &mut request_ctx, span_ctx, timeout).await?
    } else {
        handle_regular_request(provider, request_data, &mut request_ctx, timeout).await?
    };

    Ok(response)
}

async fn finalize_stream_request(request_ctx: &mut RequestContext) {
    if let Err(err) = hooks::rate_limit::post_check_streaming(request_ctx).await {
        error!("Rate limit post_check_streaming error: {}", err);
    }
    hooks::observability::record_streaming_usage(request_ctx).await;
}

#[fastrace::trace]
async fn handle_regular_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    request_ctx: &mut RequestContext,
    timeout: Option<Duration>,
) -> Result<Response, ChatCompletionError> {
    match maybe_timeout(timeout, provider.chat_completion(request)).await? {
        Ok(response) => {
            let response_data = ResponseData::ChatCompletion(response.clone());

            if let Err(err) = hooks::rate_limit::post_check(
                request_ctx,
                &response_data,
            )
            .await
            {
                error!("Rate limit post_check error: {}", err);
            }

            let mut resp = Json(response).into_response();
            hooks::rate_limit::inject_response_headers(request_ctx, resp.headers_mut()).await;
            hooks::observability::record_usage(request_ctx, &response_data).await;

            Ok(resp)
        }
        Err(err) => {
            error!("Provider request failed: {}", err);
            Err(ChatCompletionError::ProviderError(err))
        }
    }
}

#[fastrace::trace]
async fn handle_stream_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    request_ctx: &mut RequestContext,
    span_ctx: SpanContext,
    timeout: Option<Duration>,
) -> Result<Response, ChatCompletionError> {
    use futures::stream::StreamExt;

    let res: Result<BoxStream<'static, Result<ChatCompletionChunk, ProviderError>>, ProviderError> =
        match maybe_timeout(timeout, provider.chat_completion_stream(request)).await {
            Ok(res) => res,
            Err(err) => Err(ChatCompletionError::Timeout(err))?,
        };

    match res {
        Ok(stream) => {
            let stream_request_ctx = request_ctx.clone(); //TODO
            let stream_span = Span::root("sse_connection", span_ctx);

            let sse_stream = futures::stream::unfold(
                (stream, stream_span, 0, stream_request_ctx, false, false),
                |(mut stream, span, idx, mut request_ctx, done, saw_chunk)| async move {
                    if done {
                        finalize_stream_request(&mut request_ctx).await;

                        drop(span);
                        return None;
                    }
                    match stream.next().await {
                        Some(Ok(chunk)) => {
                            // Record first-token event
                            if idx == 0 {
                                hooks::observability::record_first_token_latency(&mut request_ctx)
                                    .await;
                                span.add_event(TraceEvent::new("first token arrived"));
                            }

                            // Record token usage
                            if let Some(usage) = &chunk.usage {
                                request_ctx
                                    .extensions_mut()
                                    .await
                                    .insert(TokenUsage::from_chat_completion(usage));
                            }

                            let event = match serde_json::to_string(&chunk) {
                                Ok(json) => {
                                    Ok::<SseEvent, Infallible>(SseEvent::default().data(json))
                                }
                                Err(err) => {
                                    error!("Failed to serialize chunk: {}", err);
                                    Ok(SseEvent::default().data(""))
                                }
                            };
                            Some((event, (stream, span, idx + 1, request_ctx, false, true)))
                        }
                        Some(Err(err)) => {
                            error!("Stream error: {}", err);
                            finalize_stream_request(&mut request_ctx).await;
                            drop(span);
                            None
                        }
                        None => {
                            if saw_chunk {
                                Some((
                                    Ok(SseEvent::default().data("[DONE]")),
                                    (stream, span, idx + 1, request_ctx, true, saw_chunk),
                                ))
                            } else {
                                // TODO: check why
                                finalize_stream_request(&mut request_ctx).await;

                                drop(span);
                                None
                            }
                        }
                    }
                },
            );
            Ok(Sse::new(sse_stream).into_response())
        }
        Err(err) => {
            error!("Provider stream request failed: {}", err);
            Err(ChatCompletionError::ProviderError(err))
        }
    }
}
