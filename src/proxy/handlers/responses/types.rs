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
pub enum ResponsesError {
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

impl IntoResponse for ResponsesError {
    fn into_response(self) -> Response {
        match self {
            ResponsesError::AuthorizationError(err) => err.into_response(),
            ResponsesError::RateLimitError(RateLimitError::Raw(resp)) => resp,
            ResponsesError::GatewayError(err) => {
                gateway_error_response("Responses", &err, err.status_code())
            }
            ResponsesError::Timeout(_) => timeout_response("Responses"),
            ResponsesError::MissingModelInContext => missing_model_response("Responses"),
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::response::IntoResponse;
    use http::StatusCode;
    use http_body_util::BodyExt;
    use pretty_assertions::assert_eq;
    use serde_json::{Value, json};

    use super::ResponsesError;
    use crate::gateway::error::GatewayError;

    #[tokio::test]
    async fn provider_errors_keep_provider_error_shape_but_surface_message_and_details() {
        let response = ResponsesError::GatewayError(GatewayError::Provider {
            status: StatusCode::BAD_REQUEST,
            body: json!({
                "error": {
                    "message": "unknown model",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }),
            provider: "openai".into(),
            retryable: false,
        })
        .into_response();

        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let payload: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            payload,
            json!({
                "error": {
                    "message": "unknown model",
                    "type": "server_error",
                    "code": "provider_error",
                    "details": {
                        "error": {
                            "message": "unknown model",
                            "type": "invalid_request_error",
                            "code": "model_not_found"
                        }
                    }
                }
            })
        );
    }
}
