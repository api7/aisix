use axum::{
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
};
use log::debug;

use crate::{config::entities::apikey::ApiKey, handlers::AppState};

#[derive(Debug)]
pub enum AuthError {
    InvalidApiKey,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        (axum::http::StatusCode::UNAUTHORIZED, "Unauthorized").into_response()
    }
}

pub async fn auth(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let api_key = match request.headers().get(http::header::AUTHORIZATION) {
        Some(value) => {
            let header = value.to_str().unwrap_or("");
            let (prefix, rest) = header.split_at(7.min(header.len()));
            if prefix.eq_ignore_ascii_case("bearer ") {
                rest
            } else {
                header
            }
        }
        None => return Err(AuthError::InvalidApiKey),
    };
    debug!("Authenticating request with API key: {}", api_key);

    let api_key = match state.resources().apikeys.get_by_key(api_key) {
        Some(api_key) => api_key,
        None => return Err(AuthError::InvalidApiKey),
    };

    request.extensions_mut().insert::<ApiKey>(api_key.1);

    Ok(next.run(request).await)
}
