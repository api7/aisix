mod types;

use std::convert::Infallible;

use axum::{
    Json,
    extract::{Extension, Request, State},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, Sse},
    },
};
use fastrace::prelude::{Event as TraceEvent, *};
use log::error;
pub use types::*;

use crate::{
    config::entities::{Model, ResourceEntry},
    providers::{Provider, create_provider},
    proxy::{
        AppState,
        hooks::{HOOK_FILTER_ALL, HOOK_MANAGER, HookContext, ResponseData, TokenUsage},
        middlewares::RequestModel,
    },
};

#[fastrace::trace]
pub async fn chat_completions(
    State(_state): State<AppState>,
    Extension(mut request_data): Extension<ChatCompletionRequest>,
    Extension(span_ctx): Extension<SpanContext>,
    mut hook_ctx: HookContext,
    mut request: Request,
) -> Result<Response, ChatCompletionError> {
    hook_ctx.insert(RequestModel(request_data.model));
    HOOK_MANAGER
        .pre_call(&mut hook_ctx, &mut request, HOOK_FILTER_ALL)
        .await?;

    let model = hook_ctx.get::<ResourceEntry<Model>>().cloned().unwrap(); //TODO: safe unwrap

    let provider = create_provider(&model.provider_config);

    // Replace request model name with real model name
    request_data.model = model.model.name.clone();

    // Check if it's a streaming request
    let is_stream = request_data.stream.unwrap_or(false);

    let response = if is_stream {
        handle_stream_request(provider, request_data, hook_ctx, span_ctx).await?
    } else {
        handle_regular_request(provider, request_data, hook_ctx).await?
    };

    Ok(response)
}

#[fastrace::trace]
async fn handle_regular_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    hook_ctx: HookContext,
) -> Result<Response, ChatCompletionError> {
    let mut hook_ctx = hook_ctx;
    match provider.chat_completion(request).await {
        Ok(response) => {
            HOOK_MANAGER
                .post_call_success(
                    &mut hook_ctx,
                    &ResponseData::ChatCompletion(response.clone()),
                    HOOK_FILTER_ALL,
                )
                .await?;

            // Build response and add headers
            let mut resp = Json(response).into_response();
            HOOK_MANAGER
                .post_call_headers(&mut hook_ctx, resp.headers_mut(), HOOK_FILTER_ALL)
                .await?;

            Ok(resp)
        }
        Err(err) => {
            error!("Provider request failed: {}", err);
            let err: anyhow::Error = err.into();
            HOOK_MANAGER
                .post_call_failure(&mut hook_ctx, &err, HOOK_FILTER_ALL)
                .await?;
            Ok(ChatCompletionError::ProviderError(err.to_string()).into_response())
        }
    }
}

#[fastrace::trace]
async fn handle_stream_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    hook_ctx: HookContext,
    span_ctx: SpanContext,
) -> Result<Response, ChatCompletionError> {
    use futures::stream::StreamExt;

    match provider.chat_completion_stream(request).await {
        Ok(stream) => {
            let stream_span = Span::root("sse_connection", span_ctx);

            let sse_stream = futures::stream::unfold(
                (stream, stream_span, 0, hook_ctx, false),
                |(mut stream, span, idx, mut hook_ctx, done)| async move {
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
                            Some((event, (stream, span, idx + 1, hook_ctx, false)))
                        }
                        Some(Err(err)) => {
                            error!("Stream error: {}", err);
                            drop(span);
                            None
                        }
                        None => Some((
                            Ok(SseEvent::default().data("[DONE]")),
                            (stream, span, idx + 1, hook_ctx, true),
                        )),
                    }
                },
            );
            Ok(Sse::new(sse_stream).into_response())
        }
        Err(err) => {
            error!("Provider stream request failed: {}", err);
            Ok(ChatCompletionError::ProviderError(err.to_string()).into_response())
        }
    }
}
