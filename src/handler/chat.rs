/// Chat completions handler
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use futures::stream::StreamExt;
use log::{error, warn};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

use super::AppState;
use crate::{
    config::entities::models::ProviderConfig,
    handler::validate_model::{HasModelField, ValidatedJson},
    providers::{Provider, deepseek::DeepSeekProvider},
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
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

#[fastrace::trace]
pub async fn chat_completions(
    State(state): State<AppState>,
    /* Extension(consumer): Extension<Option<ConsumerInfo>>, */
    ValidatedJson(mut request, model): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let provider: Box<dyn Provider> = {
        let _new_provider_span =
            fastrace::prelude::LocalSpan::enter_with_local_parent("create_provider_instance");
        match &model.provider_config.get().unwrap() {
            ProviderConfig::DeepSeek(deepseek_config) => {
                Box::new(DeepSeekProvider::new(deepseek_config.api_key.clone()))
            }
        }
    };

    // Save original model value for response
    let original_model = request.model.clone();

    // Replace request model name with real model name
    request.model = model.model.split("/").nth(1).unwrap().to_string();

    // Check if it's a streaming request
    let is_stream = request.stream.unwrap_or(false);

    if is_stream {
        // Streaming response
        handle_stream_request(provider, request, original_model).await
    } else {
        // Non-streaming response
        handle_regular_request(provider, request, original_model).await
    }
}

#[fastrace::trace]
async fn handle_regular_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    original_model: String,
) -> Response {
    /* if let Some(guardrail_config) = &provider_config.guardrail {
        if guardrail_config.enabled && guardrail_config.filter_input {
            let gr_config = guardrail::GuardrailConfig {
                guardrail_id: guardrail_config.guardrail_id.clone(),
                guardrail_version: guardrail_config.guardrail_version.clone(),
                aws_region: guardrail_config.aws_region.clone(),
                aws_access_key_id: guardrail_config.aws_access_key_id.clone(),
                aws_secret_access_key: guardrail_config.aws_secret_access_key.clone(),
            };

            match guardrail::check_input_messages(&gr_config, &request.messages).await {
                Ok(guardrail::GuardrailAction::None) => {
                    // Passed check, continue processing
                }
                Ok(guardrail::GuardrailAction::Detected) => {
                    // Detected potential issue but allowed through, logging done
                }
                Ok(guardrail::GuardrailAction::Blocked { reason }) => {
                    // Block request
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": {
                                "message": format!("Content blocked by guardrail: {}", reason),
                                "type": "content_policy_violation",
                                "code": "guardrail_blocked"
                            }
                        })),
                    )
                        .into_response();
                }
                Err(err) => {
                    warn!("Guardrail check failed, allowing request to continue: {}", err);
                    // If guardrail service fails, do not block normal requests
                }
            }
        }
    } */

    match provider.chat_completion(request).await {
        Ok(mut response) => {
            /* if let Some(guardrail_config) = &provider_config.guardrail {
                if guardrail_config.enabled && guardrail_config.filter_output {
                    let gr_config = guardrail::GuardrailConfig {
                        guardrail_id: guardrail_config.guardrail_id.clone(),
                        guardrail_version: guardrail_config.guardrail_version.clone(),
                        aws_region: guardrail_config.aws_region.clone(),
                        aws_access_key_id: guardrail_config.aws_access_key_id.clone(),
                        aws_secret_access_key: guardrail_config.aws_secret_access_key.clone(),
                    };

                    // Check content of first choice
                    if let Some(first_choice) = response.choices.first() {
                        let content = &first_choice.message.content;
                        match guardrail::check_output_content(&gr_config, content).await {
                            Ok(guardrail::GuardrailAction::None) => {
                                // Passed check, continue returning
                            }
                            Ok(guardrail::GuardrailAction::Detected) => {
                                // Detected potential issue but allowed through, logging done
                            }
                            Ok(guardrail::GuardrailAction::Blocked { reason }) => {
                                // Block response
                                return (
                                    StatusCode::OK,
                                    Json(serde_json::json!({
                                        "error": {
                                            "message": format!("Output blocked by guardrail: {}", reason),
                                            "type": "content_policy_violation",
                                            "code": "guardrail_blocked_output"
                                        }
                                    })),
                                )
                                    .into_response();
                            }
                            Err(err) => {
                                warn!("Guardrail output check failed, allowing response to continue: {}", err);
                                // If guardrail service fails, do not block normal response
                            }
                        }
                    }
                }
            } */

            response.model = original_model;
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

#[fastrace::trace]
async fn handle_stream_request(
    provider: Box<dyn Provider>,
    request: ChatCompletionRequest,
    original_model: String,
) -> Response {
    match provider.chat_completion_stream(request).await {
        Ok(response) => {
            let stream = response.bytes_stream();
            let sse_stream = stream.flat_map(move |result| {
                let original_model = original_model.clone();
                futures::stream::iter(match result {
                    Ok(chunk) => {
                        let text = String::from_utf8_lossy(&chunk);
                        // SSE format: data: {json}\n\n
                        // Collect all data lines from same chunk
                        text.lines()
                            .filter(|line| line.starts_with("data: "))
                            .map(move |line| {
                                let json_str = &line[6..];
                                if json_str == "[DONE]" {
                                    Ok::<_, Infallible>(Event::default().data("[DONE]"))
                                } else {
                                    // Parse JSON and modify model field
                                    match serde_json::from_str::<serde_json::Value>(json_str) {
                                        Ok(mut json_value) => {
                                            if let Some(obj) = json_value.as_object_mut() {
                                                obj.insert(
                                                    "model".to_string(),
                                                    serde_json::Value::String(
                                                        original_model.clone(),
                                                    ),
                                                );
                                                if let Ok(modified_json) =
                                                    serde_json::to_string(&json_value)
                                                {
                                                    return Ok(Event::default().data(modified_json));
                                                }
                                            }
                                            // If unable to modify model, return original data
                                            Ok(Event::default().data(json_str.to_string()))
                                        }
                                        Err(_) => {
                                            // If parsing fails, return original data
                                            Ok(Event::default().data(json_str.to_string()))
                                        }
                                    }
                                }
                            })
                            .collect::<Vec<_>>()
                    }
                    Err(err) => {
                        warn!("SSE stream error: {}", err);
                        vec![]
                    }
                })
            });

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
