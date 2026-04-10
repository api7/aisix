mod types;

use std::{convert::Infallible, time::Duration};

use axum::{
    Json,
    body::Body,
    extract::{Extension, Request, State},
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
        hooks::{HOOK_FILTER_ALL, HOOK_MANAGER, HookContext, ResponseData, TokenUsage},
        hooks2::{RequestContext, authorization},
    },
    utils::future::maybe_timeout,
};

#[fastrace::trace]
pub async fn chat_completions(
    State(_state): State<AppState>,
    Extension(span_ctx): Extension<SpanContext>,
    mut request_ctx: RequestContext,
    mut hook_ctx: HookContext,
    Json(mut request_data): Json<ChatCompletionRequest>,
) -> Result<Response, ChatCompletionError> {
    authorization::check(&mut request_ctx, request_data.model.clone()).await?;

    let mut request = Request::new(Body::empty()); //TODO
    HOOK_MANAGER
        .pre_call(&mut hook_ctx, &mut request, HOOK_FILTER_ALL)
        .await?;

    let model = hook_ctx.get::<ResourceEntry<Model>>().cloned().unwrap(); //TODO: safe unwrap

    let provider = create_provider(&model.provider_config);
    let timeout = model.timeout.map(Duration::from_millis);

    // Replace request model name with real model name
    request_data.model = model.model.name.clone();

    // Check if it's a streaming request
    let is_stream = request_data.stream.unwrap_or(false);

    let response = if is_stream {
        handle_stream_request(provider, request_data, &mut hook_ctx, span_ctx, timeout).await?
    } else {
        handle_regular_request(provider, request_data, &mut hook_ctx, timeout).await?
    };

    Ok(response)
}

#[fastrace::trace]
async fn handle_regular_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    hook_ctx: &mut HookContext,
    timeout: Option<Duration>,
) -> Result<Response, ChatCompletionError> {
    match maybe_timeout(timeout, provider.chat_completion(request)).await? {
        Ok(response) => {
            HOOK_MANAGER
                .post_call_success(
                    hook_ctx,
                    &ResponseData::ChatCompletion(response.clone()),
                    HOOK_FILTER_ALL,
                )
                .await?;

            // Build response and add headers
            let mut resp = Json(response).into_response();
            HOOK_MANAGER
                .post_call_headers(hook_ctx, resp.headers_mut(), HOOK_FILTER_ALL)
                .await?;

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
    hook_ctx: &mut HookContext,
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
            let stream_hook_ctx = std::mem::take(hook_ctx);
            let stream_span = Span::root("sse_connection", span_ctx);

            let sse_stream = futures::stream::unfold(
                (stream, stream_span, 0, stream_hook_ctx, false, false),
                |(mut stream, span, idx, mut hook_ctx, done, saw_chunk)| async move {
                    if done {
                        if let Err(err) = HOOK_MANAGER
                            .post_call_streaming(&mut hook_ctx, HOOK_FILTER_ALL)
                            .await
                        {
                            error!("Hook post_call_streaming error: {}", err);
                        }

                        drop(span);
                        return None;
                    }
                    match stream.next().await {
                        Some(Ok(chunk)) => {
                            // Record first-token event
                            if idx == 0 {
                                span.add_event(TraceEvent::new("first token arrived"));
                            }

                            // Record token usage
                            if let Some(usage) = &chunk.usage {
                                hook_ctx.insert(TokenUsage::from_chat_completion(usage));
                            }

                            if let Err(err) = HOOK_MANAGER
                                .post_call_streaming_chunk(
                                    &mut hook_ctx,
                                    &chunk,
                                    idx,
                                    HOOK_FILTER_ALL,
                                )
                                .await
                            {
                                error!("Hook post_call_streaming_chunk error: {}", err);
                                drop(span);
                                return None;
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
                            Some((event, (stream, span, idx + 1, hook_ctx, false, true)))
                        }
                        Some(Err(err)) => {
                            error!("Stream error: {}", err);
                            drop(span);
                            None
                        }
                        None => {
                            if saw_chunk {
                                Some((
                                    Ok(SseEvent::default().data("[DONE]")),
                                    (stream, span, idx + 1, hook_ctx, true, saw_chunk),
                                ))
                            } else {
                                if let Err(err) = HOOK_MANAGER
                                    .post_call_streaming(&mut hook_ctx, HOOK_FILTER_ALL)
                                    .await
                                {
                                    error!("Hook post_call_streaming error: {}", err);
                                }

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
