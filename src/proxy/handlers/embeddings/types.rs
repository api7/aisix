use axum::{
    response::{IntoResponse, Response},
};
use http::StatusCode;
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
pub enum EmbeddingError {
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

impl IntoResponse for EmbeddingError {
    fn into_response(self) -> Response {
        match self {
            EmbeddingError::AuthorizationError(err) => err.into_response(),
            EmbeddingError::RateLimitError(RateLimitError::Raw(resp)) => resp,
            EmbeddingError::GatewayError(err) => {
                let status = match &err {
                    GatewayError::Provider { .. }
                    | GatewayError::Http(_)
                    | GatewayError::Stream(_) => StatusCode::BAD_GATEWAY,
                    GatewayError::EmbeddingsNotSupported { .. } => StatusCode::NOT_IMPLEMENTED,
                    _ => err.status_code(),
                };
                gateway_error_response("Embeddings", &err, status)
            }
            EmbeddingError::Timeout(_) => timeout_response("Embeddings"),
            EmbeddingError::MissingModelInContext => missing_model_response("Embeddings"),
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

    use super::EmbeddingError;
    use crate::gateway::error::GatewayError;

    #[tokio::test]
    async fn embeddings_not_supported_returns_not_implemented() {
        let response = EmbeddingError::GatewayError(GatewayError::EmbeddingsNotSupported {
            provider: "anthropic".into(),
        })
        .into_response();

        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let payload: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            payload,
            json!({
                "error": {
                    "message": "Provider error",
                    "type": "server_error",
                    "code": "provider_error"
                }
            })
        );
    }
}
