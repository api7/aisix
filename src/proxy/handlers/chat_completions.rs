use std::convert::Infallible;

use axum::{
    Json,
    extract::{Extension, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use log::error;
use serde::{Deserialize, Serialize};

use crate::{
    config::entities,
    providers::{Provider, create_provider},
    proxy::{AppState, policies},
};

use super::extractors::{RateLimitGuards, ValidatedModel, validate_model::HasModelField};

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
    State(_state): State<AppState>,
    Extension(api_key): Extension<entities::ResourceEntry<entities::ApiKey>>,
    ValidatedModel(mut request, model): ValidatedModel<ChatCompletionRequest>,
) -> Response {
    // Check rate limits
    let _guards = match RateLimitGuards::check(&api_key, &model).await {
        Ok(guards) => guards,
        Err(err) => return err.into_response(),
    };

    let provider = create_provider(&model.provider_config);

    // Save original model value for response
    let original_model = request.model.clone();

    // Replace request model name with real model name
    //TODO safe unwrap
    request.model = model.model.split("/").nth(1).unwrap().to_string();

    // Check if it's a streaming request
    let is_stream = request.stream.unwrap_or(false);

    if is_stream {
        handle_stream_request(provider, request, original_model, api_key, model).await
    } else {
        handle_regular_request(provider, request, original_model, api_key, model).await
    }
}

#[fastrace::trace]
async fn handle_regular_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    original_model: String,
    api_key: entities::ResourceEntry<entities::ApiKey>,
    model: entities::ResourceEntry<entities::Model>,
) -> Response {
    match provider.chat_completion(request).await {
        Ok(mut response) => {
            let tokens = response.usage.total_tokens as u64;
            response.model = original_model;

            // Record token usage
            let limiter = policies::rate_limit::get_rate_limiter();

            // Check model token limits
            if let Some(ref rate_limit) = model.rate_limit {
                if let Err(err) = limiter
                    .record_usage("model", &model.name, rate_limit, tokens)
                    .await
                {
                    return (
                        StatusCode::TOO_MANY_REQUESTS,
                        Json(serde_json::json!({
                            "error": {
                                "message": err.to_string(),
                                "type": "rate_limit_error",
                                "code": "rate_limit_exceeded"
                            }
                        })),
                    )
                        .into_response();
                }
            }

            // Check apikey token limits
            if let Some(ref rate_limit) = api_key.rate_limit {
                if let Err(err) = limiter
                    .record_usage("apikey", &api_key.key, rate_limit, tokens)
                    .await
                {
                    return (
                        StatusCode::TOO_MANY_REQUESTS,
                        Json(serde_json::json!({
                            "error": {
                                "message": err.to_string(),
                                "type": "rate_limit_error",
                                "code": "rate_limit_exceeded"
                            }
                        })),
                    )
                        .into_response();
                }
            }

            Json::<ChatCompletionResponse>(response).into_response()
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
    original_model: String,
    api_key: entities::ResourceEntry<entities::ApiKey>,
    model: entities::ResourceEntry<entities::Model>,
) -> Response {
    use futures::stream::StreamExt;
    match provider.chat_completion_stream(request).await {
        Ok(stream) => {
            let limiter = policies::rate_limit::get_rate_limiter();
            let model_rate_limit = model.rate_limit.clone();
            let apikey_rate_limit = api_key.rate_limit.clone();
            let model_name = model.name.clone();
            let api_key_key = api_key.key.clone();

            let sse_stream = stream
                .filter_map(move |chunk_result| {
                    let original_model = original_model.clone();
                    let limiter = limiter.clone();
                    let model_rate_limit = model_rate_limit.clone();
                    let apikey_rate_limit = apikey_rate_limit.clone();
                    let model_name = model_name.clone();
                    let api_key_key = api_key_key.clone();

                    async move {
                        match chunk_result {
                            Ok(mut chunk) => {
                                chunk.model = original_model;

                                // Check if this chunk has usage (typically the last real chunk)
                                if let Some(usage) = &chunk.usage {
                                    let tokens = usage.total_tokens as u64;

                                    // Record usage for model
                                    if let Some(ref rate_limit) = model_rate_limit {
                                        let _ = limiter
                                            .record_usage("model", &model_name, rate_limit, tokens)
                                            .await;
                                    }

                                    // Record usage for apikey
                                    if let Some(ref rate_limit) = apikey_rate_limit {
                                        let _ = limiter
                                            .record_usage(
                                                "apikey",
                                                &api_key_key,
                                                rate_limit,
                                                tokens,
                                            )
                                            .await;
                                    }
                                }

                                match serde_json::to_string(&chunk) {
                                    Ok(json) => {
                                        Some(Ok::<Event, Infallible>(Event::default().data(json)))
                                    }
                                    Err(err) => {
                                        error!("Failed to serialize chunk: {}", err);
                                        None
                                    }
                                }
                            }
                            Err(err) => {
                                error!("Stream error: {}", err);
                                None
                            }
                        }
                    }
                })
                .chain(futures::stream::iter(vec![Ok::<Event, Infallible>(
                    Event::default().data("[DONE]"),
                )]));
            Sse::new(sse_stream).into_response()
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
