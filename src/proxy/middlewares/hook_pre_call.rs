use axum::{
    Json,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use log::error;
use serde_json::json;

use crate::{
    config::entities,
    proxy::hooks::{Context, HOOK_MANAGER, HookAction},
};

use super::HasModelField;

/// Error type for hook pre-call middleware
pub enum HookPreCallError {
    /// API key not found in request extensions
    MissingApiKey,
    /// Model not found in request extensions
    MissingModel,
    /// Request body not found in extensions
    MissingRequestBody,
}

impl IntoResponse for HookPreCallError {
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
            Self::MissingRequestBody => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": "Request body not found in extensions",
                        "type": "internal_error",
                        "code": "missing_request_body"
                    }
                })),
            )
                .into_response(),
        }
    }
}

/// Middleware to execute pre-call hooks
///
/// This middleware:
/// 1. Extracts API key, model, and original model from request Extensions
/// 2. Creates a HookContext with this information
/// 3. Executes all registered pre_call hooks (e.g., rate limit pre-check)
/// 4. If any hook returns EarlyReturn (e.g., 429 error), returns immediately
/// 5. Otherwise, stores HookContext in Extensions for handlers to use in post_call hooks
///
/// Requires (from previous middlewares):
/// - API key in Extensions (from auth middleware)
/// - Validated model in Extensions (from validate_model middleware)
/// - Parsed request body in Extensions (from parse_body middleware)
///
/// Usage:
/// ```rust
/// .layer(from_fn(hook_pre_call))
/// ```
pub async fn hook_pre_call(mut req: Request, next: Next) -> Result<Response, HookPreCallError> {
    // Get API key from Extensions (should be set by auth middleware)
    let api_key = req
        .extensions()
        .get::<entities::ResourceEntry<entities::ApiKey>>()
        .cloned()
        .ok_or_else(|| HookPreCallError::MissingApiKey)?;

    // Get model from Extensions (should be set by validate_model middleware)
    let model = req
        .extensions()
        .get::<entities::ResourceEntry<entities::Model>>()
        .cloned()
        .ok_or_else(|| HookPreCallError::MissingModel)?;

    // Extract original_model from request body
    // Try to get it from any type that implements HasModelField
    let original_model = req
        .extensions()
        .get::<crate::proxy::handlers::chat_completions::ChatCompletionRequest>()
        .and_then(|r| r.model())
        .or_else(|| {
            req.extensions()
                .get::<crate::proxy::handlers::embeddings::EmbeddingRequest>()
                .and_then(|r| r.model())
        })
        .ok_or_else(|| HookPreCallError::MissingRequestBody)?;

    // Create HookContext
    let mut ctx = Context::new(api_key, model, original_model);

    // Execute pre_call hooks
    match HOOK_MANAGER.execute_pre_call(&mut ctx).await {
        Ok(HookAction::Continue) => {
            // Store context in Extensions for handler to use in post_call hooks
            req.extensions_mut().insert(ctx);
            // Continue to handler
            Ok(next.run(req).await)
        }
        Ok(HookAction::EarlyReturn(response)) => {
            // Hook returned early (e.g., 429 rate limit error)
            Ok(response)
        }
        Err(err) => {
            // Internal error in hook execution
            //TODO: better error handling/logging here
            error!("Hook execution error: {}", err);
            Ok((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": format!("Hook execution error"),
                        "type": "internal_error",
                        "code": "hook_error"
                    }
                })),
            )
                .into_response())
        }
    }
}
