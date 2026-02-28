mod types;

use std::{convert::Infallible, time::Instant};

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

use crate::{
    config::entities::{Model, ResourceEntry},
    providers::{Provider, create_provider},
    proxy::{
        AppState,
        hooks::{Context, HOOK_MANAGER, HookAction, ResponseData, TokenUsage},
        middlewares::RequestModel,
    },
};

pub use types::*;

#[fastrace::trace]
pub async fn chat_completions(
    State(_state): State<AppState>,
    Extension(mut request_data): Extension<ChatCompletionRequest>,
    Extension(span_ctx): Extension<SpanContext>,
    mut hook_ctx: Context,
    mut request: Request,
) -> Response {
    // PRE CALL HOOKS START
    hook_ctx.insert(RequestModel(request_data.model));

    let action = HOOK_MANAGER
        .execute_pre_call(&mut hook_ctx, &mut request, None)
        .await;

    match action {
        Ok(HookAction::EarlyReturn(response)) => {
            return response;
        }
        Err(err) => {
            error!("Hook pre_call error: {}", err);
            return (ChatCompletionError::InternalError(err.to_string())).into_response();
        }
        _ => {}
    }

    // PRE CALL HOOKS END

    let model = hook_ctx.get::<ResourceEntry<Model>>().cloned().unwrap(); //TODO: safe unwrap

    let provider = create_provider(&model.provider_config);

    // Replace request model name with real model name
    //TODO safe unwrap
    request_data.model = model.model.name.clone();

    // Check if it's a streaming request
    let is_stream = request_data.stream.unwrap_or(false);

    let start_time = Instant::now();
    let response = if is_stream {
        handle_stream_request(provider, request_data, hook_ctx, start_time, span_ctx).await
    } else {
        let response = handle_regular_request(provider, request_data, hook_ctx).await;

        let duration = start_time.elapsed().as_millis() as u64;
        crate::utils::metrics::METRIC_LLM_LATENCY.record(
            duration,
            &[opentelemetry::KeyValue::new(
                "model",
                model.model.name.clone(),
            )],
        );

        response
    };

    response
}

#[fastrace::trace]
async fn handle_regular_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    hook_ctx: Context,
) -> Response {
    let mut hook_ctx = hook_ctx;
    match provider.chat_completion(request).await {
        Ok(response) => {
            // Execute post_call_success hooks
            let response_data = ResponseData::ChatCompletion(response.clone());
            if let Err(err) = HOOK_MANAGER
                .execute_post_call_success(&mut hook_ctx, &response_data)
                .await
            {
                error!("Hook post_call_success error: {}", err);
            }

            // Build response and add headers
            let mut resp = Json(response).into_response();
            if let Err(err) = HOOK_MANAGER
                .execute_post_call_headers(&mut hook_ctx, resp.headers_mut())
                .await
            {
                error!("Hook post_call_headers error: {}", err);
            }

            resp
        }
        Err(err) => {
            error!("Provider request failed: {}", err);
            ChatCompletionError::ProviderError(err.to_string()).into_response()
        }
    }
}

#[fastrace::trace]
async fn handle_stream_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    hook_ctx: Context,
    start_time: Instant,
    span_ctx: SpanContext,
) -> Response {
    use futures::stream::StreamExt;

    let model = hook_ctx
        .get::<ResourceEntry<Model>>()
        .unwrap() //TODO: safe unwrap
        .model
        .clone();
    match provider.chat_completion_stream(request).await {
        Ok(stream) => {
            let stream_span = Span::root("sse_connection", span_ctx);

            let sse_stream = futures::stream::unfold(
                (stream, stream_span, 0, model, hook_ctx, start_time, false),
                |(mut stream, span, idx, model, mut hook_ctx, start_time, done)| async move {
                    if done {
                        drop(span);
                        return None;
                    }
                    match stream.next().await {
                        Some(Ok(chunk)) => {
                            // Record first-token latency once
                            if idx == 0 {
                                let latency = start_time.elapsed().as_millis() as u64;
                                crate::utils::metrics::METRIC_LLM_FIRST_TOKEN_LATENCY.record(
                                    latency,
                                    &[opentelemetry::KeyValue::new("model", model.name.clone())],
                                );
                                span.add_event(TraceEvent::new("first token arrived"));
                            }

                            // Record token usage for last chunk
                            if let Some(usage) = &chunk.usage {
                                if let Err(err) = HOOK_MANAGER
                                    .execute_post_call_streaming(
                                        &mut hook_ctx,
                                        &TokenUsage::from_chat_completion(usage),
                                    )
                                    .await
                                {
                                    error!("Hook post_call_streaming error: {}", err);
                                }
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
                            Some((
                                event,
                                (stream, span, idx + 1, model, hook_ctx, start_time, false),
                            ))
                        }
                        Some(Err(err)) => {
                            error!("Stream error: {}", err);
                            // Drop span here too so it captures the correct end time
                            drop(span);
                            None
                        }
                        None => {
                            let duration = start_time.elapsed().as_millis() as u64;
                            crate::utils::metrics::METRIC_LLM_LATENCY.record(
                                duration,
                                &[opentelemetry::KeyValue::new("model", model.name.clone())],
                            );
                            Some((
                                Ok(SseEvent::default().data("[DONE]")),
                                (stream, span, idx + 1, model, hook_ctx, start_time, true),
                            ))
                        }
                    }
                },
            );
            Sse::new(sse_stream).into_response()
        }
        Err(err) => {
            error!("Provider stream request failed: {}", err);
            (ChatCompletionError::ProviderError(err.to_string())).into_response()
        }
    }
}
