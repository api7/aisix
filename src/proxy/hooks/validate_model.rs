use anyhow::Result;
use async_trait::async_trait;
use axum::{Json, extract::Request, response::IntoResponse};
use http::StatusCode;
use log::error;
use serde_json::json;

use super::{HookContext, ProxyHook};
use crate::{
    config::entities::{ApiKey, ResourceEntry},
    proxy::{AppState, hooks::HookError, middlewares::RequestModel},
};

pub enum ValidatedModelError {
    MissingModelField,
    ModelNotFound(String),
    AccessForbidden(String),
}

impl IntoResponse for ValidatedModelError {
    fn into_response(self) -> axum::response::Response {
        match self {
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
        }
    }
}

pub struct ValidateModelHook;

#[async_trait]
impl ProxyHook for ValidateModelHook {
    fn name(&self) -> &str {
        "validate_model"
    }

    async fn pre_call(&self, ctx: &mut HookContext, _req: &mut Request) -> Result<(), HookError> {
        let request_model = match ctx.get::<RequestModel>().cloned() {
            Some(model) => model,
            None => {
                error!("Request model not found in context");
                return Err(HookError::RawResponse(
                    (ValidatedModelError::MissingModelField).into_response(),
                ));
            }
        };
        let model_name = request_model.0;

        let state = ctx
            .get::<AppState>()
            .cloned()
            .expect("AppState should be in context");

        let model = match state.resources().models.get_by_name(&model_name) {
            Some(model) => model,
            None => {
                return Err(HookError::RawResponse(
                    ValidatedModelError::ModelNotFound(model_name.clone()).into_response(),
                ));
            }
        };

        let api_key = match ctx.get::<ResourceEntry<ApiKey>>().cloned() {
            Some(api_key) => api_key,
            None => {
                error!("API key not found in context");
                return Err(HookError::RawResponse(
                    (StatusCode::INTERNAL_SERVER_ERROR).into_response(),
                ));
            }
        };

        // Check if API key has access to this model
        if !api_key.allowed_models.contains(&model_name) {
            return Err(HookError::RawResponse(
                ValidatedModelError::AccessForbidden(model_name.clone()).into_response(),
            ));
        }

        ctx.insert(model);

        Ok(())
    }
}
