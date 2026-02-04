use axum::{
    Json,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::{
    config::entities,
    proxy::policies::rate_limit::{
        self, Metric as RateLimitMetric, RateLimitError, RateLimitState,
    },
};

/// Error type for rate limit middleware
pub enum RateLimitMiddlewareError {
    /// API key not found in request extensions
    MissingApiKey,
    /// Model not found in request extensions
    MissingModel,
    /// Rate limit exceeded
    RateLimitExceeded {
        api_key: String,
        metric: RateLimitMetric,
        error: RateLimitError,
    },
}

impl IntoResponse for RateLimitMiddlewareError {
    fn into_response(self) -> Response {
        match self {
            Self::MissingApiKey => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": "API key not found in extensions",
                        "type": "internal_error",
                        "code": "missing_api_key"
                    }
                })),
            )
                .into_response(),
            Self::MissingModel => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": "Model not found in extensions",
                        "type": "internal_error",
                        "code": "missing_model"
                    }
                })),
            )
                .into_response(),
            Self::RateLimitExceeded {
                api_key,
                metric,
                error,
            } => rate_limit::RateLimitResponse::new(api_key, metric, error).into_response(),
        }
    }
}

/// Middleware to perform rate limit pre-check and store state in Extensions
///
/// Requires:
/// - API key in Extensions (from auth middleware)
/// - Validated model in Extensions (from validate_model middleware)
///
/// This middleware performs pre_check for both api_key and model:
/// - Request metrics (RPM/RPD): checks AND commits
/// - Token metrics (TPM/TPD): checks only, does NOT commit
///
/// The RateLimitState is stored in Extensions for the handler to use in post_check.
///
/// Usage:
/// ```rust
/// .layer(from_fn(rate_limit_check))
/// ```
pub async fn rate_limit_check(
    mut req: Request,
    next: Next,
) -> Result<Response, RateLimitMiddlewareError> {
    // Get API key from Extensions (should be set by auth middleware)
    let api_key = req
        .extensions()
        .get::<entities::ResourceEntry<entities::ApiKey>>()
        .cloned()
        .ok_or_else(|| RateLimitMiddlewareError::MissingApiKey)?;

    // Get model from Extensions (should be set by validate_model middleware)
    let model = req
        .extensions()
        .get::<entities::ResourceEntry<entities::Model>>()
        .cloned()
        .ok_or_else(|| RateLimitMiddlewareError::MissingModel)?;

    // Perform rate limit pre-check
    let mut rate_limit_state = RateLimitState::new();

    // Check api_key rate limits
    rate_limit_state.store_pre_check(rate_limit::pre_check(&api_key).await.map_err(
        |(metric, error)| RateLimitMiddlewareError::RateLimitExceeded {
            api_key: api_key.id.clone(),
            metric,
            error,
        },
    )?);

    // Check model rate limits
    rate_limit_state.store_pre_check(rate_limit::pre_check(&model).await.map_err(
        |(metric, error)| RateLimitMiddlewareError::RateLimitExceeded {
            api_key: api_key.id.clone(),
            metric,
            error,
        },
    )?);

    // Store rate limit state in Extensions for handler to use in post_check
    req.extensions_mut().insert(rate_limit_state);

    // Continue to handler
    Ok(next.run(req).await)
}
