use axum::{
    Json,
    extract::{Extension, State},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::config::entities::apikey::ApiKey;
use crate::{
    handlers::{AppState, extractors::ValidatedModel, extractors::validate_model::HasModelField},
    providers::create_provider,
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
    State(_state): State<AppState>,
    Extension(api_key): Extension<ApiKey>,
    ValidatedModel(mut request, model): ValidatedModel<EmbeddingRequest>,
) -> Response {
    // Check rate limits
    let _guards = match crate::handlers::extractors::RateLimitGuards::check(&api_key, &model).await
    {
        Ok(guards) => guards,
        Err(err) => return err.into_response(),
    };

    let provider = create_provider(&model.provider_config);

    // Save original model value for response
    let original_model = request.model.clone();

    // Replace request model name with real model name
    //TODO safe unwrap
    request.model = model.model.split("/").nth(1).unwrap().to_string();

    match provider.embedding(request).await {
        Ok(mut response) => {
            response.model = original_model;

            // Record token usage if available and check limits
            if let Some(ref usage) = response.usage {
                let tokens = usage.total_tokens as u64;
                let limiter = crate::policies::rate_limit::get_rate_limiter();

                // Check model token limits
                if let Some(ref rate_limit) = model.rate_limit {
                    if let Err(err) = limiter
                        .record_usage("model", &model.name, rate_limit, tokens)
                        .await
                    {
                        return (
                            axum::http::StatusCode::TOO_MANY_REQUESTS,
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
                            axum::http::StatusCode::TOO_MANY_REQUESTS,
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
            }

            Json::<EmbeddingResponse>(response).into_response()
        }
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error generating embeddings: {}", err),
        )
            .into_response(),
    }
}
