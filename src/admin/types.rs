use axum::{
    Json,
    response::{IntoResponse, Response},
};
use http::StatusCode;
use serde::Serialize;
use serde_json::json;
use utoipa::ToSchema;

#[derive(Serialize, ToSchema)]
pub struct ListResponse<T> {
    pub total: usize,
    pub list: Vec<T>,
}

impl<T: Serialize + ToSchema> IntoResponse for ListResponse<T> {
    fn into_response(self) -> Response {
        (StatusCode::OK, Json(self)).into_response()
    }
}

#[derive(Serialize, ToSchema)]
pub struct ItemResponse<T> {
    pub key: String,
    pub value: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_index: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_index: Option<i64>,
}

impl<T: Serialize + ToSchema> IntoResponse for ItemResponse<T> {
    fn into_response(self) -> Response {
        (StatusCode::OK, Json(self)).into_response()
    }
}

#[derive(Serialize, ToSchema)]
pub struct DeleteResponse {
    pub deleted: i64,
    pub key: String,
}

impl IntoResponse for DeleteResponse {
    fn into_response(self) -> Response {
        (StatusCode::OK, Json(self)).into_response()
    }
}

#[derive(Serialize, ToSchema)]
#[serde(tag = "error_msg")]
pub enum APIError {
    BadRequest(String),
    NotFound(String),
    InternalError(String),
}

impl IntoResponse for APIError {
    fn into_response(self) -> Response {
        match self {
            APIError::BadRequest(msg) => {
                (StatusCode::BAD_REQUEST, Json(json!({"error_msg": msg}))).into_response()
            }
            APIError::NotFound(msg) => {
                (StatusCode::NOT_FOUND, Json(json!({"error_msg": msg}))).into_response()
            }
            APIError::InternalError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error_msg": msg})),
            )
                .into_response(),
        }
    }
}

#[derive(Serialize, ToSchema)]
#[serde(tag = "error_msg")]
pub enum AuthError {
    MissingKey,
    InvalidKey,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        match self {
            AuthError::MissingKey => (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error_msg": "Missing API key"})),
            )
                .into_response(),
            AuthError::InvalidKey => (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error_msg": "Invalid API key"})),
            )
                .into_response(),
        }
    }
}
