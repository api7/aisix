use axum::{
    Json,
    extract::{FromRequest, Request},
    http::StatusCode,
    response::IntoResponse,
};
use serde_json::json;

use crate::config::entities::models::Model;

use super::super::AppState;

pub trait HasModelField {
    fn model(&self) -> Option<String>;
}

pub enum ValidatedModelError {
    BadRequest,
    MissingModelField,
    ModelNotFound(String),
    AccessForbidden(String),
    Unauthorized,
}

impl From<axum::extract::rejection::JsonRejection> for ValidatedModelError {
    fn from(_: axum::extract::rejection::JsonRejection) -> Self {
        ValidatedModelError::BadRequest
    }
}

impl IntoResponse for ValidatedModelError {
    fn into_response(self) -> axum::response::Response {
        match self {
            ValidatedModelError::BadRequest => (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": "Bad request",
                        "type": "invalid_request_error",
                        "code": "bad_request"
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

#[derive(Debug, Clone)]
pub struct ValidatedModel<T>(pub T, pub Model);

impl<T> FromRequest<AppState> for ValidatedModel<T>
where
    T: serde::de::DeserializeOwned + HasModelField,
{
    type Rejection = ValidatedModelError;

    async fn from_request(req: Request, state: &AppState) -> Result<Self, Self::Rejection> {
        let extensions = req.extensions().clone();

        let Json(data) = Json::<T>::from_request(req, state)
            .await
            .map_err(ValidatedModelError::from)?;

        let model = match data.model() {
            Some(m) => m,
            None => {
                return Err(ValidatedModelError::MissingModelField);
            }
        };

        let res = state.resources();

        let model = match res.models.get_by_name(&model) {
            Some(m) => m,
            None => {
                return Err(ValidatedModelError::ModelNotFound(model));
            }
        };

        let api_key = extensions
            .get::<crate::config::entities::apikey::ApiKey>()
            .cloned();
        if let None = api_key {
            return Err(ValidatedModelError::Unauthorized);
        }

        let api_key = api_key.unwrap();
        if !api_key.allowed_models.contains(&model.name) {
            return Err(ValidatedModelError::AccessForbidden(model.name.clone()));
        }

        Ok(Self(data, model))
    }
}
