use axum::{
    Json,
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
};
use fastrace::Span;
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
    let span = Span::enter_with_local_parent("aisix.proxy.middleware.authn");

    let api_key = if let Some(value) = req.headers().get(http::header::AUTHORIZATION) {
        let header = value.to_str().unwrap_or("");
        let (prefix, rest) = header.split_at(7.min(header.len()));
        if prefix.eq_ignore_ascii_case("bearer ") {
            rest
        } else {
            header
        }
    } else if let Some(value) = req.headers().get("x-api-key") {
        value.to_str().unwrap_or("")
    } else {
        return Err(AuthError::MissingApiKey);
    };

    let api_key = match state.resources().apikeys.get_by_key(api_key) {
        Some(api_key) => api_key,
        None => {
            return Err(AuthError::InvalidApiKey);
        }
    };

    span.add_property(|| ("aisix.apikey_id", api_key.id.clone()));

    req.extensions_mut()
        .insert::<ResourceEntry<ApiKey>>(api_key);

    drop(span);

    Ok(next.run(req).await)
}
