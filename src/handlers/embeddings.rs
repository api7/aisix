use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

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
    ValidatedModel(mut request, model): ValidatedModel<EmbeddingRequest>,
) -> Response {
    let provider = {
        let _new_provider_span =
            fastrace::prelude::LocalSpan::enter_with_local_parent("create_provider_instance");

        match model.provider_config.get() {
            Some(config) => create_provider(config),
            None => panic!("TODO: Provider config not set for model {}", model.name),
        }
    };

    // Save original model value for response
    let original_model = request.model.clone();

    // Replace request model name with real model name
    //TODO safe unwrap
    request.model = model.model.split("/").nth(1).unwrap().to_string();

    match provider.embedding(request).await {
        Ok(mut response) => {
            response.model = original_model;
            Json::<EmbeddingResponse>(response).into_response()
        }
        Err(err) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error generating embeddings: {}", err),
        )
            .into_response(),
    }
}
