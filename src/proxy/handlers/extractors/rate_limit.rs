use axum::{
    Json,
    extract::FromRequestParts,
    http::{StatusCode, request::Parts},
    response::IntoResponse,
};
use serde_json::json;

use crate::{
    config::entities::{apikey::ApiKey, models::Model},
    proxy::policies::rate_limit::{self, ConcurrencyGuard},
};

pub enum RateLimitError {
    Unauthorized,
    ModelNotFound,
    RateLimitExceeded(String),
}

impl IntoResponse for RateLimitError {
    fn into_response(self) -> axum::response::Response {
        match self {
            RateLimitError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": {
                        "message": "Unauthorized",
                        "type": "authentication_error",
                        "code": "unauthorized"
                    }
                })),
            )
                .into_response(),
            RateLimitError::ModelNotFound => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": "Model not found in extensions",
                        "type": "invalid_request_error",
                        "code": "model_not_found"
                    }
                })),
            )
                .into_response(),
            RateLimitError::RateLimitExceeded(msg) => (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({
                    "error": {
                        "message": msg,
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                })),
            )
                .into_response(),
        }
    }
}

/// Rate limit guards that hold concurrency permits
pub struct RateLimitGuards {
    pub model_guard: Option<ConcurrencyGuard>,
    pub apikey_guard: Option<ConcurrencyGuard>,
}

impl RateLimitGuards {
    /// Perform rate limit check with model and apikey
    pub async fn check(api_key: &ApiKey, model: &Model) -> Result<Self, RateLimitError> {
        let limiter = rate_limit::get_rate_limiter();

        // Check model rate limits
        let model_guard = if let Some(ref rate_limit) = model.rate_limit {
            Some(
                limiter
                    .check_and_reserve("model", &model.name, rate_limit)
                    .await
                    .map_err(|err| {
                        log::warn!("Model rate limit exceeded: {}", err);
                        RateLimitError::RateLimitExceeded(err.to_string())
                    })?,
            )
            .flatten()
        } else {
            None
        };

        // Check apikey rate limits
        let apikey_guard = if let Some(ref rate_limit) = api_key.rate_limit {
            Some(
                limiter
                    .check_and_reserve("apikey", &api_key.key, rate_limit)
                    .await
                    .map_err(|err| {
                        log::warn!("ApiKey rate limit exceeded: {}", err);
                        RateLimitError::RateLimitExceeded(err.to_string())
                    })?,
            )
            .flatten()
        } else {
            None
        };

        Ok(Self {
            model_guard,
            apikey_guard,
        })
    }
}

impl<S> FromRequestParts<S> for RateLimitGuards
where
    S: Send + Sync,
{
    type Rejection = RateLimitError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Get API key from extensions (set by auth middleware)
        let api_key = parts
            .extensions
            .get::<ApiKey>()
            .ok_or(RateLimitError::Unauthorized)?;

        // Get model from extensions (needs to be set before this extractor runs)
        let model = parts
            .extensions
            .get::<Model>()
            .ok_or(RateLimitError::ModelNotFound)?;

        Self::check(api_key, model).await
    }
}
