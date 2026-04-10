use anyhow::Result;
use axum::{Json, response::IntoResponse};
use http::StatusCode;
use log::error;
use serde_json::json;
use thiserror::Error;

use crate::{
    config::entities::{ApiKey, ResourceEntry},
    proxy::{AppState, hooks2::RequestContext},
};

#[derive(Clone)]
pub struct RequestModel(#[allow(unused)] pub String);

#[derive(Debug, Clone, Error, PartialEq, Eq, Hash)]
pub enum AuthorizationError {
    #[error("Model '{0}' not found")]
    ModelNotFound(String),
    #[error("Access to model '{0}' is forbidden")]
    AccessForbidden(String),

    // INTERNAL ERROR
    #[error("Apikey not found in context")]
    MissingApiKeyInContext,
}

impl IntoResponse for AuthorizationError {
    fn into_response(self) -> axum::response::Response {
        match self {
            AuthorizationError::ModelNotFound(_) => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": self.to_string(),
                        "type": "invalid_request_error",
                        "code": "model_not_found"
                    }
                })),
            )
                .into_response(),
            AuthorizationError::AccessForbidden(_) => (
                StatusCode::FORBIDDEN,
                Json(json!({
                    "error": {
                        "message": self.to_string(),
                        "type": "invalid_request_error",
                        "code": "model_access_forbidden"
                    }
                })),
            )
                .into_response(),
            AuthorizationError::MissingApiKeyInContext => {
                (StatusCode::INTERNAL_SERVER_ERROR).into_response()
            }
        }
    }
}

pub async fn check(ctx: &mut RequestContext, model_name: String) -> Result<(), AuthorizationError> {
    let state = ctx
        .get::<AppState>()
        .cloned()
        .expect("AppState should be in context");

    let model = match state.resources().models.get_by_name(&model_name) {
        Some(model) => model,
        None => {
            return Err(AuthorizationError::ModelNotFound(model_name.clone()));
        }
    };

    let api_key = match ctx.get::<ResourceEntry<ApiKey>>().cloned() {
        Some(api_key) => api_key,
        None => {
            error!("API key not found in context");
            return Err(AuthorizationError::MissingApiKeyInContext);
        }
    };

    // Check if API key has access to this model
    if !api_key.allowed_models.contains(&model_name) {
        return Err(AuthorizationError::AccessForbidden(model_name.clone()));
    }

    ctx.insert(model);
    ctx.insert(RequestModel(model_name));

    Ok(())
}
