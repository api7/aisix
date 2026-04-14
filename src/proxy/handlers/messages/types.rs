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
pub enum MessagesError {
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

impl IntoResponse for MessagesError {
    fn into_response(self) -> Response {
        match self {
            MessagesError::AuthorizationError(err) => err.into_response(),
            MessagesError::RateLimitError(RateLimitError::Raw(resp)) => resp,
            MessagesError::GatewayError(err) => {
                let status = err.status_code();
                let (message, error_type, code) = match err {
                    GatewayError::Provider { .. }
                    | GatewayError::Http(_)
                    | GatewayError::Stream(_) => (
                        "Provider error".to_string(),
                        "server_error",
                        "provider_error",
                    ),
                    GatewayError::Internal(_) => (
                        "Gateway internal error".to_string(),
                        "server_error",
                        "internal_error",
                    ),
                    _ => (
                        err.to_string(),
                        if status.is_client_error() {
                            "invalid_request_error"
                        } else {
                            "server_error"
                        },
                        "gateway_error",
                    ),
                };

                (
                    status,
                    Json(serde_json::json!({
                        "error": {
                            "message": message,
                            "type": error_type,
                            "code": code
                        }
                    })),
                )
                    .into_response()
            }
            MessagesError::Timeout(_) => (
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
            MessagesError::MissingModelInContext => (
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
