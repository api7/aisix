use std::convert::Infallible;

use axum::{
    Json,
    extract::Extension,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use log::error;
use serde::{Deserialize, Serialize};

use crate::{
    providers::{Provider, create_provider},
    proxy::{
        hooks::{Context, HOOK_MANAGER, ResponseData, TokenUsage},
        middlewares::HasModelField,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: String,
    //TODO audio
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    //TODO logit_bias
    //TODO logprobs
    //TODO max_completion_tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    //TODO metadata
    //TODO modalities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    //TODO parallel_tool_calls
    //TODO prediction
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    //TODO prompt_cache_key
    //TODO prompt_cache_retention
    //TODO reasoning_effort
    //TODO response_format
    //TODO safety_identifier
    //TODO service_tier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    //TODO store
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    //TODO stream_options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    //TODO tool_choice
    //TODO tools
    //TODO top_logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    //TODO verbosity
    //TODO web_search_options
}

impl HasModelField for ChatCompletionRequest {
    fn model(&self) -> Option<String> {
        Some(self.model.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: ChatCompletionUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatCompletionChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatCompletionUsage>,
}

#[fastrace::trace]
pub async fn chat_completions(
    Extension(mut request): Extension<ChatCompletionRequest>,
    Extension(hook_ctx): Extension<Context>,
) -> Response {
    let model = hook_ctx.model.clone();
    let provider = create_provider(&model.provider_config);

    // Replace request model name with real model name
    //TODO safe unwrap
    request.model = model.model.split("/").nth(1).unwrap().to_string();

    // Check if it's a streaming request
    let is_stream = request.stream.unwrap_or(false);

    if is_stream {
        handle_stream_request(provider, request, hook_ctx).await
    } else {
        handle_regular_request(provider, request, hook_ctx).await
    }
}

#[fastrace::trace]
async fn handle_regular_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    hook_ctx: Context,
) -> Response {
    let mut hook_ctx = hook_ctx;
    match provider.chat_completion(request).await {
        Ok(mut response) => {
            response.model = hook_ctx.original_model.clone();

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
                .execute_post_call_headers(&hook_ctx, resp.headers_mut())
                .await
            {
                error!("Hook post_call_headers error: {}", err);
            }

            resp
        }
        Err(err) => {
            error!("Provider request failed: {}", err);
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Provider request failed: {}", err),
                        "type": "server_error",
                        "code": "provider_error"
                    }
                })),
            )
                .into_response()
        }
    }
}

async fn handle_stream_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    hook_ctx: Context,
) -> Response {
    use futures::stream::StreamExt;
    match provider.chat_completion_stream(request).await {
        Ok(stream) => {
            let original_model = hook_ctx.original_model.clone();
            let hook_ctx_clone = hook_ctx.clone();

            let (tx, rx) = tokio::sync::mpsc::channel(100);

            // Spawn task to process stream and execute hooks after completion
            tokio::spawn(async move {
                let mut last_usage: Option<TokenUsage> = None;
                let mut hook_ctx = hook_ctx_clone;

                futures::pin_mut!(stream);
                while let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(mut chunk) => {
                            chunk.model = original_model.clone();

                            // Check if this chunk has usage (typically the last real chunk)
                            if let Some(usage) = &chunk.usage {
                                last_usage = Some(TokenUsage {
                                    prompt_tokens: Some(usage.prompt_tokens as u64),
                                    completion_tokens: Some(usage.completion_tokens as u64),
                                    total_tokens: usage.total_tokens as u64,
                                });
                            }

                            // Send chunk to client
                            if tx.send(chunk).await.is_err() {
                                break;
                            }
                        }
                        Err(err) => {
                            error!("Stream error: {}", err);
                            break;
                        }
                    }
                }

                // Stream ended, execute post_call_streaming hook
                if let Some(usage) = last_usage {
                    if let Err(err) = HOOK_MANAGER
                        .execute_post_call_streaming(&mut hook_ctx, &usage)
                        .await
                    {
                        error!("Hook post_call_streaming error: {}", err);
                    }
                }
            });

            // Create SSE stream from receiver
            let sse_stream = futures::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|chunk| {
                    let event = match serde_json::to_string(&chunk) {
                        Ok(json) => Ok::<Event, Infallible>(Event::default().data(json)),
                        Err(err) => {
                            error!("Failed to serialize chunk: {}", err);
                            Ok(Event::default().data(""))
                        }
                    };
                    (event, rx)
                })
            })
            .chain(futures::stream::iter(vec![Ok::<Event, Infallible>(
                Event::default().data("[DONE]"),
            )]));

            // Build response and add headers (with pre-check values)
            let mut resp = Sse::new(sse_stream).into_response();
            if let Err(err) = HOOK_MANAGER
                .execute_post_call_headers(&hook_ctx, resp.headers_mut())
                .await
            {
                error!("Hook post_call_headers error: {}", err);
            }

            resp
        }
        Err(err) => {
            error!("Provider stream request failed: {}", err);
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Provider stream request failed: {}", err),
                        "type": "server_error",
                        "code": "provider_stream_error"
                    }
                })),
            )
                .into_response()
        }
    }
}
