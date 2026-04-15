use axum::{
    Json,
    response::{IntoResponse, Response},
};
use http::{HeaderMap, StatusCode};
use log::error;
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
                    StatusCode::NOT_FOUND,
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
                    "Rate limit exceeded"
                } else {
                    "Internal server error"
                };

                anthropic_error_response(
                    parts.status,
                    if parts.status == StatusCode::TOO_MANY_REQUESTS {
                        "rate_limit_error"
                    } else {
                        "api_error"
                    },
                    message.to_string(),
                    Some(parts.headers),
                )
            }
            MessagesError::GatewayError(err) => {
                let status = err.status_code();
                let error_type = gateway_error_type(&err);
                error!("Messages gateway error: {}", err);

                anthropic_error_response(
                    status,
                    error_type,
                    gateway_error_message(&err, error_type).to_string(),
                    None,
                )
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

fn gateway_error_message(error: &GatewayError, error_type: &'static str) -> &'static str {
    match error {
        GatewayError::Validation(_) | GatewayError::Bridge(_) => "Invalid request",
        GatewayError::Transform(_) | GatewayError::NativeNotSupported { .. } => "Server error",
        GatewayError::Internal(_) => "Internal server error",
        GatewayError::Provider { status, .. } => match error_type {
            "authentication_error" => "Authentication failed",
            "billing_error" => "Payment required",
            "permission_error" => "Permission denied",
            "not_found_error" => "Requested resource not found",
            "rate_limit_error" => "Rate limit exceeded",
            "request_too_large" => "Request payload too large",
            "timeout_error" => "Upstream request timed out",
            "overloaded_error" => "Upstream service unavailable",
            _ if status.as_u16() == 529 => "overloaded_error", // provider-specific overload status code
            _ => "Provider error",
        },
        _ => "Upstream service unavailable",
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
            // TODO: Reuse a normalized request ID once proxy/request tracing exposes one here.
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
        GatewayError::Validation(_) | GatewayError::Bridge(_) => "invalid_request_error",
        GatewayError::Transform(_) | GatewayError::NativeNotSupported { .. } => "api_error",
        GatewayError::Internal(_) => "api_error",
        GatewayError::Provider { status, .. } => match *status {
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::PAYMENT_REQUIRED => "billing_error",
            StatusCode::FORBIDDEN => "permission_error",
            StatusCode::NOT_FOUND => "not_found_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
            StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => "timeout_error",
            StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
            _ if status.is_server_error() => "api_error",
            _ => "invalid_request_error",
        },
        _ => "overloaded_error",
    }
}

#[cfg(test)]
mod tests {
    use http::StatusCode;
    use serde_json::json;

    use super::{gateway_error_message, gateway_error_type};
    use crate::gateway::error::GatewayError;

    #[test]
    fn transform_errors_are_reported_as_api_errors() {
        let error = GatewayError::Transform("bad provider payload".into());

        let error_type = gateway_error_type(&error);
        assert_eq!(error_type, "api_error");
        assert_eq!(gateway_error_message(&error, error_type), "Server error");
    }

    #[test]
    fn native_not_supported_is_reported_as_api_error() {
        let error = GatewayError::NativeNotSupported {
            provider: "gemini".into(),
        };

        let error_type = gateway_error_type(&error);
        assert_eq!(error_type, "api_error");
        assert_eq!(gateway_error_message(&error, error_type), "Server error");
    }

    #[test]
    fn provider_billing_errors_are_reported_with_billing_type() {
        let error = GatewayError::Provider {
            status: StatusCode::PAYMENT_REQUIRED,
            body: json!({"error": "payment required"}),
            provider: "anthropic".into(),
            retryable: false,
        };

        let error_type = gateway_error_type(&error);
        assert_eq!(error_type, "billing_error");
        assert_eq!(
            gateway_error_message(&error, error_type),
            "Payment required"
        );
    }

    #[test]
    fn provider_large_payload_errors_are_reported_with_payload_type() {
        let error = GatewayError::Provider {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            body: json!({"error": "payload too large"}),
            provider: "anthropic".into(),
            retryable: false,
        };

        let error_type = gateway_error_type(&error);
        assert_eq!(error_type, "request_too_large");
        assert_eq!(
            gateway_error_message(&error, error_type),
            "Request payload too large"
        );
    }
}
