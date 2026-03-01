use axum::{
    Json,
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::{
    config::entities::{ApiKey, ResourceEntry},
    proxy::AppState,
};

#[derive(Debug)]
pub enum AuthError {
    MissingApiKey,
    InvalidApiKey,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        match self {
            AuthError::MissingApiKey => (
                http::StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": {
                        "message": "Missing API key in request",
                        "type": "invalid_request_error",
                        "param": null,
                        "code": null
                    }
                })),
            )
                .into_response(),
            AuthError::InvalidApiKey => (
                axum::http::StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_request_error",
                        "param": null,
                        "code": null
                    }
                })),
            )
                .into_response(),
        }
    }
}

pub async fn auth(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let api_key = match req.headers().get(http::header::AUTHORIZATION) {
        Some(value) => {
            let header = value.to_str().unwrap_or("");
            let (prefix, rest) = header.split_at(7.min(header.len()));
            if prefix.eq_ignore_ascii_case("bearer ") {
                rest
            } else {
                header
            }
        }
        None => {
            return Err(AuthError::MissingApiKey);
        }
    };

    let api_key = match state.resources().apikeys.get_by_key(api_key) {
        Some(api_key) => api_key,
        None => {
            return Err(AuthError::InvalidApiKey);
        }
    };

    req.extensions_mut()
        .insert::<ResourceEntry<ApiKey>>(api_key.1);

    Ok(next.run(req).await)
}
