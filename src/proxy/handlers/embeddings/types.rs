use axum::{Json, response::IntoResponse};
use http::StatusCode;
use serde::{Deserialize, Serialize};

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

pub enum EmbeddingError {
    ProviderError(String),
    InternalError(String), //TODO more specific error definitions
}

impl IntoResponse for EmbeddingError {
    fn into_response(self) -> axum::response::Response {
        match self {
            EmbeddingError::ProviderError(err) => (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Provider error: {}", err),
                        "type": "server_error",
                        "code": "provider_error"
                    }
                })),
            )
                .into_response(),
            EmbeddingError::InternalError(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Internal error: {}", err),
                        "type": "server_error",
                        "code": "internal_error"
                    }
                })),
            )
                .into_response(),
        }
    }
}
