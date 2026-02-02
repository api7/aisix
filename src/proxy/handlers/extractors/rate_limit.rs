use axum::{Json, http::StatusCode, response::IntoResponse};
use serde_json::json;

use crate::{
    config::entities::{ApiKey, Model},
    proxy::policies::rate_limit::{self, ConcurrencyGuard},
};

pub enum RateLimitError {
    RateLimitExceeded(String),
}

impl IntoResponse for RateLimitError {
    fn into_response(self) -> axum::response::Response {
        match self {
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
