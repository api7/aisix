use axum::{
    Json,
    response::{IntoResponse, Response},
};
use http::{HeaderMap, StatusCode};
use thiserror::Error;
use tokio::time::error::Elapsed;
use uuid::Uuid;

use crate::{
    gateway::error::GatewayError,
    proxy::hooks::{authorization::AuthorizationError, rate_limit::RateLimitError},
};

/// Errors that can occur while handling Anthropic Messages API requests.
#[derive(Debug, Error)]
pub enum MessagesError {
    /// The caller cannot access the requested model.
    #[error("Authorization error: {0}")]
    AuthorizationError(#[from] AuthorizationError),
    /// The request exceeded a configured rate or concurrency limit.
    #[error("Rate limit error: {0}")]
    RateLimitError(#[from] RateLimitError),
    /// The gateway failed while validating, dispatching, or bridging the request.
    #[error("Gateway error: {0}")]
    GatewayError(#[from] GatewayError),
    /// The upstream request did not complete before the model timeout.
    #[error("Request timed out")]
    Timeout(#[from] Elapsed),
    /// Authorization completed but the resolved model was not inserted into request context.
    #[error("Model was not inserted into request context after authorization check")]
    MissingModelInContext,
}

impl IntoResponse for MessagesError {
    fn into_response(self) -> Response {
        match self {
            MessagesError::AuthorizationError(err) => match err {
                AuthorizationError::ModelNotFound(message) => anthropic_error_response(
                    StatusCode::BAD_REQUEST,
                    "not_found_error",
                    format!("Model '{message}' not found"),
                    None,
                ),
                AuthorizationError::AccessForbidden(message) => anthropic_error_response(
                    StatusCode::FORBIDDEN,
                    "permission_error",
                    format!("Access to model '{message}' is forbidden"),
                    None,
                ),
                AuthorizationError::MissingApiKeyInContext => anthropic_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    "API key missing in request context".to_string(),
                    None,
                ),
            },
            MessagesError::RateLimitError(RateLimitError::Raw(resp)) => {
                let (parts, _) = resp.into_parts();
                let message = if parts.status == StatusCode::TOO_MANY_REQUESTS {
                    "Rate limit exceeded".to_string()
                } else {
                    "Internal server error".to_string()
                };
                anthropic_error_response(
                    parts.status,
                    if parts.status == StatusCode::TOO_MANY_REQUESTS {
                        "rate_limit_error"
                    } else {
                        "api_error"
                    },
                    message,
                    Some(parts.headers),
                )
            }
            MessagesError::GatewayError(err) => {
                let status = err.status_code();
                let (message, error_type) = match &err {
                    GatewayError::Provider { .. }
                    | GatewayError::Http(_)
                    | GatewayError::Stream(_) => {
                        ("Provider error".to_string(), gateway_error_type(&err))
                    }
                    GatewayError::Internal(_) => {
                        ("Gateway internal error".to_string(), "api_error")
                    }
                    _ => (err.to_string(), gateway_error_type(&err)),
                };

                anthropic_error_response(status, error_type, message, None)
            }
            MessagesError::Timeout(_) => anthropic_error_response(
                StatusCode::GATEWAY_TIMEOUT,
                "timeout_error",
                "Provider request timed out".to_string(),
                None,
            ),
            MessagesError::MissingModelInContext => anthropic_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "model missing in request context".to_string(),
                None,
            ),
        }
    }
}

fn anthropic_error_response(
    status: StatusCode,
    error_type: &'static str,
    message: String,
    headers: Option<HeaderMap>,
) -> Response {
    let mut response = (
        status,
        Json(serde_json::json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
            //TODO: A normalized request ID, which may reuse the trace ID.
            "request_id": format!("req_{}", Uuid::new_v4()),
        })),
    )
        .into_response();

    if let Some(headers) = headers {
        for (name, value) in &headers {
            if name.as_str().eq_ignore_ascii_case("content-length")
                || name.as_str().eq_ignore_ascii_case("content-type")
            {
                continue;
            }
            response.headers_mut().insert(name, value.clone());
        }
    }

    response
}

fn gateway_error_type(error: &GatewayError) -> &'static str {
    match error {
        GatewayError::Validation(_)
        | GatewayError::Bridge(_)
        | GatewayError::Transform(_)
        | GatewayError::NativeNotSupported { .. } => "invalid_request_error",
        GatewayError::Internal(_) => "api_error",
        GatewayError::Provider { status, .. } => match *status {
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::FORBIDDEN => "permission_error",
            StatusCode::NOT_FOUND => "not_found_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => "timeout_error",
            StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
            _ if status.is_server_error() => "api_error",
            _ => "invalid_request_error",
        },
        GatewayError::Http(_) | GatewayError::Stream(_) => "overloaded_error",
    }
}
