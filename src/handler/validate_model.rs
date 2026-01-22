use axum::{
    Json,
    extract::{FromRequest, Request},
    http::StatusCode,
};
use log::error;
use serde_json::json;

use crate::{config::entities::models::Model, handler::AppState};

pub trait HasModelField {
    fn model(&self) -> Option<String>;
}

#[derive(Debug, Clone)]
pub struct ValidatedJson<T>(pub T, pub Model);

impl<T> FromRequest<AppState> for ValidatedJson<T>
where
    T: serde::de::DeserializeOwned + HasModelField,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    #[fastrace::trace]
    async fn from_request(req: Request, state: &AppState) -> Result<Self, Self::Rejection> {
        let Json(data) = Json::<T>::from_request(req, state).await.map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("Failed to parse JSON body: {}", e),
                        "type": "invalid_request_error",
                        "code": "invalid_json"
                    }
                })),
            )
        })?;

        let model = match data.model() {
            Some(m) => m,
            None => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": {
                            "message": "Missing 'model' field in request body",
                            "type": "invalid_request_error",
                            "code": "missing_model_field"
                        }
                    })),
                ));
            }
        };

        let res = state.resources();

        let model = match res.models.get_by_name(&model) {
            Some(m) => m,
            None => {
                error!("Model not found (id={})", model);
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": {
                            "message": format!("Model '{}' not found", model),
                            "type": "invalid_request_error",
                            "code": "model_not_found"
                        }
                    })),
                ));
            }
        };

        Ok(Self(data, model))
    }
}

/// Model validation middleware: Parse and validate model information in request
///
/// Features:
/// 1. Parse model name in @slug/model format
/// 2. Find corresponding provider and model instance
/// 3. Check if consumer has permission to access the model
/// 4. Store result in Extension for handler to use
/* pub async fn validate_model_middleware<T>(
    State(_state): State<AppState>,
    Extension(_consumer): Extension<Option<ConsumerInfo>>,
    Json(request): Json<T>,
    next: Next,
) -> Result<Response, Response> {
    // Extract model field from body (only for /v1/chat/completions route)
    // Since body cannot be read repeatedly, use a simplified approach here
    // The actual model name will be extracted from request body in handler
    // We'll validate it here

    // Skip validation for now, handle separately in handler
    // TODO: Optimize body extraction logic later
    Ok(next.run(request).await)
} */

/// Parse model name in @slug/model format
pub fn parse_model_format(
    model_name: &str,
) -> Result<(String, String), (StatusCode, Json<serde_json::Value>)> {
    if !model_name.starts_with('@') {
        error!(
            "Invalid model format: {}，must use @slug/model format",
            model_name
        );
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!("Invalid model format: '{}'. Must use @slug/model format (e.g., @deepseek/deepseek-chat)", model_name),
                    "type": "invalid_request_error",
                    "code": "invalid_model_format"
                }
            })),
        ));
    }

    let parts: Vec<&str> = model_name[1..].splitn(2, '/').collect();
    if parts.len() != 2 {
        error!("Model name format error: {}", model_name);
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!("Invalid model name format: '{}'. Expected format: @slug/model-name", model_name),
                    "type": "invalid_request_error",
                    "code": "invalid_model_format"
                }
            })),
        ));
    }

    Ok((parts[0].to_string(), parts[1].to_string()))
}
