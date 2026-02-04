use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::{config::entities, proxy::AppState};

/// Trait for request types that have a model field
pub trait HasModelField {
    fn model(&self) -> Option<String>;
}

pub enum ValidatedModelError {
    MissingRequest,
    MissingModelField,
    ModelNotFound(String),
    AccessForbidden(String),
    Unauthorized,
}

impl IntoResponse for ValidatedModelError {
    fn into_response(self) -> axum::response::Response {
        match self {
            ValidatedModelError::MissingRequest => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": "Request not found in extensions",
                        "type": "internal_error",
                        "code": "missing_request"
                    }
                })),
            )
                .into_response(),
            ValidatedModelError::MissingModelField => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": "Missing 'model' field in request body",
                        "type": "invalid_request_error",
                        "code": "missing_model_field"
                    }
                })),
            )
                .into_response(),
            ValidatedModelError::ModelNotFound(model) => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("Model '{}' not found", model),
                        "type": "invalid_request_error",
                        "code": "model_not_found"
                    }
                })),
            )
                .into_response(),
            ValidatedModelError::AccessForbidden(model) => (
                StatusCode::FORBIDDEN,
                Json(json!({
                    "error": {
                        "message": format!("Access to model '{}' is forbidden", model),
                        "type": "invalid_request_error",
                        "code": "model_access_forbidden"
                    }
                })),
            )
                .into_response(),
            ValidatedModelError::Unauthorized => (
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
        }
    }
}

/// Middleware to validate model and store in Extensions
///
/// Requires:
/// - Request body already parsed and stored in Extensions (use parse_body middleware first)
/// - API key in Extensions (from auth middleware)
///
/// Usage:
/// ```rust
/// .layer(from_fn_with_state(state, validate_model::<ChatCompletionRequest>))
/// ```
pub async fn validate_model<T>(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, ValidatedModelError>
where
    T: HasModelField + Clone + Send + Sync + 'static,
{
    // Get parsed body from Extensions
    let body = req
        .extensions()
        .get::<T>()
        .cloned()
        .ok_or_else(|| ValidatedModelError::MissingRequest)?;

    // Get model name from body
    let model_name = body
        .model()
        .ok_or_else(|| ValidatedModelError::MissingModelField)?;

    // Look up model in resources
    let res = state.resources();
    let model = res
        .models
        .get_by_name(&model_name)
        .ok_or_else(|| ValidatedModelError::ModelNotFound(model_name.clone()))?;

    // Get API key from Extensions (should be set by auth middleware)
    let api_key = req
        .extensions()
        .get::<entities::ResourceEntry<entities::ApiKey>>()
        .cloned()
        .ok_or_else(|| ValidatedModelError::Unauthorized)?;

    // Check if API key has access to this model
    if !api_key.allowed_models.contains(&model.name) {
        return Err(ValidatedModelError::AccessForbidden(model.name.clone()));
    }

    // Store validated model in Extensions
    req.extensions_mut().insert(model);

    // Continue
    Ok(next.run(req).await)
}
