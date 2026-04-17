//! Admin error envelope — spec §3 uses the simpler `{"error_msg": "..."}`
//! shape (distinct from the OpenAI-style proxy envelope).
//!
//! The `AdminError` enum is the internal taxonomy. `IntoResponse` lets
//! handlers `?`-propagate without touching JSON shape boilerplate.

use aisix_core::models::SchemaError;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

use crate::store::StoreError;

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error_msg: String,
}

#[derive(Debug, thiserror::Error)]
pub enum AdminError {
    #[error("missing or malformed admin authorization")]
    Unauthorized,
    #[error("{0}")]
    BadRequest(String),
    #[error("resource not found")]
    NotFound,
    #[error("name {0:?} already in use by another resource")]
    Conflict(String),
    #[error("schema validation failed at {path}: {message}")]
    Schema { path: String, message: String },
    #[error("store error: {0}")]
    Store(String),
}

impl AdminError {
    pub fn status(&self) -> StatusCode {
        match self {
            AdminError::Unauthorized => StatusCode::UNAUTHORIZED,
            AdminError::BadRequest(_) | AdminError::Schema { .. } => StatusCode::BAD_REQUEST,
            AdminError::NotFound => StatusCode::NOT_FOUND,
            AdminError::Conflict(_) => StatusCode::CONFLICT,
            AdminError::Store(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl From<SchemaError> for AdminError {
    fn from(e: SchemaError) -> Self {
        AdminError::Schema {
            path: e.path,
            message: e.message,
        }
    }
}

impl From<StoreError> for AdminError {
    fn from(e: StoreError) -> Self {
        AdminError::Store(e.to_string())
    }
}

impl IntoResponse for AdminError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = ErrorBody {
            error_msg: self.to_string(),
        };
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_match_spec_admin_envelope_rules() {
        assert_eq!(AdminError::Unauthorized.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            AdminError::BadRequest("x".into()).status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(AdminError::NotFound.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            AdminError::Conflict("n".into()).status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            AdminError::Store("boom".into()).status(),
            StatusCode::INTERNAL_SERVER_ERROR,
        );
    }

    #[test]
    fn error_body_uses_error_msg_field_not_openai_shape() {
        let body = ErrorBody {
            error_msg: "missing field".into(),
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["error_msg"], "missing field");
        // No top-level `error` object (that's the proxy envelope).
        assert!(json.get("error").is_none());
    }
}
