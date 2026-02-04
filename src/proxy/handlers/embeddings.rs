use axum::{
    Json,
    extract::Extension,
    response::{IntoResponse, Response},
};
use log::error;
use serde::{Deserialize, Serialize};

use crate::{
    config::entities,
    providers::create_provider,
    proxy::{
        middlewares::HasModelField,
        policies::rate_limit::{self, RateLimitError, RateLimitState},
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
    Extension(api_key): Extension<entities::ResourceEntry<entities::ApiKey>>,
    Extension(model): Extension<entities::ResourceEntry<entities::Model>>,
    Extension(mut rate_limit_state): Extension<RateLimitState>,
) -> Response {
    let provider = create_provider(&model.provider_config);

    // Save original model value for response
    let original_model = request.model.clone();

    // Replace request model name with real model name
    //TODO safe unwrap
    request.model = model.model.split("/").nth(1).unwrap().to_string();

    match provider.embedding(request).await {
        Ok(mut response) => {
            response.model = original_model;

            // Record token usage if available
            if let Some(ref usage) = response.usage {
                let tokens = usage.total_tokens as u64;

                // Record token usage with post_check for api_key
                match rate_limit::post_check(&api_key, tokens).await {
                    Ok(results) => {
                        rate_limit_state.store_post_check(results);
                    }
                    Err((metric, err)) => {
                        if let RateLimitError::Internal(msg) = &err {
                            log::error!(
                                "Post-check internal error for api_key: metric={:?}, error={}",
                                metric,
                                msg
                            );
                        }
                    }
                }

                // Use finalize_response to handle model post_check and add headers
                rate_limit_state
                    .finalize_response(Json(response), tokens, &model)
                    .await
            } else {
                // No usage info, just return response with pre-check headers
                let mut resp = Json::<EmbeddingResponse>(response).into_response();
                rate_limit_state.add_headers(resp.headers_mut());
                resp
            }
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
