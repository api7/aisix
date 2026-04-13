use axum::{
    Json,
    response::{IntoResponse, Response},
};
use http::StatusCode;
use thiserror::Error;
use tokio::time::error::Elapsed;

use crate::{
    gateway::error::GatewayError,
    proxy::hooks::{authorization::AuthorizationError, rate_limit::RateLimitError},
};

#[derive(Debug, Error)]
pub enum ChatCompletionError {
    #[error("Authorization error: {0}")]
    AuthorizationError(#[from] AuthorizationError),
    #[error("Rate limit error: {0}")]
    RateLimitError(#[from] RateLimitError),
    #[error("Gateway error: {0}")]
    GatewayError(#[from] GatewayError),
    #[error("Request timed out")]
    Timeout(#[from] Elapsed),
    #[error("Model was not inserted into request context after authorization check")]
    MissingModelInContext,
}

impl IntoResponse for ChatCompletionError {
    fn into_response(self) -> Response {
        match self {
            ChatCompletionError::AuthorizationError(err) => err.into_response(),
            ChatCompletionError::RateLimitError(RateLimitError::Raw(resp)) => resp,
            ChatCompletionError::GatewayError(err) => (
                err.status_code(),
                Json(serde_json::json!({
                    "error": {
                        "message": err.to_string(),
                        "type": if err.status_code().is_client_error() { "invalid_request_error" } else { "server_error" },
                        "code": "gateway_error"
                    }
                })),
            )
                .into_response(),
            ChatCompletionError::Timeout(_) => (
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({
                    "error": {
                        "message": "Provider request timed out",
                        "type": "server_error",
                        "code": "request_timeout"
                    }
                })),
            )
                .into_response(),
            ChatCompletionError::MissingModelInContext => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": "model missing in request context",
                        "type": "server_error",
                        "code": "internal_error"
                    }
                })),
            )
                .into_response(),
        }
    }
}
