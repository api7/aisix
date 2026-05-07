use axum::response::{IntoResponse, Response};
use thiserror::Error;
use tokio::time::error::Elapsed;

use crate::{
    gateway::error::GatewayError,
    proxy::{
        handlers::openai_error::{
            gateway_error_response, missing_model_response, timeout_response,
        },
        hooks::{authorization::AuthorizationError, rate_limit::RateLimitError},
    },
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
            ChatCompletionError::GatewayError(err) => {
                gateway_error_response("Chat Completions", &err, err.status_code())
            }
            ChatCompletionError::Timeout(_) => timeout_response("Chat Completions"),
            ChatCompletionError::MissingModelInContext => {
                missing_model_response("Chat Completions")
            }
        }
    }
}
