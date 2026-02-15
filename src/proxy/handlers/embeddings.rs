use axum::{
    Json,
    extract::Extension,
    response::{IntoResponse, Response},
};
use log::error;
use serde::{Deserialize, Serialize};

use crate::{
    providers::create_provider,
    proxy::{
        hooks::{Context, ResponseData, HOOK_MANAGER},
        middlewares::HasModelField,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub input: OneOrMany<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl HasModelField for EmbeddingRequest {
    fn model(&self) -> Option<String> {
        Some(self.model.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Option<EmbeddingUsage>,
}

pub async fn embeddings(
    Extension(mut request): Extension<EmbeddingRequest>,
    Extension(hook_ctx): Extension<Context>,
) -> Response {
    let mut hook_ctx = hook_ctx;
    let model = hook_ctx.model.clone();
    let provider = create_provider(&model.provider_config);

    // Replace request model name with real model name
    //TODO safe unwrap
    request.model = model.model.split("/").nth(1).unwrap().to_string();

    match provider.embedding(request).await {
        Ok(mut response) => {
            response.model = hook_ctx.original_model.clone();

            // Execute post_call_success hooks
            let response_data = ResponseData::Embedding(response.clone());
            if let Err(err) = HOOK_MANAGER.execute_post_call_success(&mut hook_ctx, &response_data).await {
                error!("Hook post_call_success error: {}", err);
            }

            // Build response and add headers
            let mut resp = Json(response).into_response();
            if let Err(err) = HOOK_MANAGER.execute_post_call_headers(&hook_ctx, resp.headers_mut()).await {
                error!("Hook post_call_headers error: {}", err);
            }

            resp
        }
        Err(err) => {
            error!("Error generating embeddings: {}", err);
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error generating embeddings: {}", err),
            )
                .into_response()
        }
    }
}
