use axum::{
    Json,
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;
use tower::util::ServiceExt;

use crate::admin::AppState;

pub async fn chat_completions(State(state): State<AppState>, mut req: Request<Body>) -> Response {
    match "/v1/chat/completions".parse() {
        Ok(uri) => {
            *req.uri_mut() = uri;
        }
        Err(_) => {
            return (StatusCode::INTERNAL_SERVER_ERROR).into_response();
        }
    }

    let Some(proxy_router) = state.proxy_router() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": {
                    "message": "Playground proxy router is not configured",
                    "type": "server_error",
                    "code": "proxy_unavailable",
                    "param": null
                }
            })),
        )
            .into_response();
    };

    match proxy_router.oneshot(req).await {
        Ok(response) => response,
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": format!("Failed to execute completions: {err}"),
                    "type": "server_error",
                    "code": "proxy_error",
                    "param": null
                }
            })),
        )
            .into_response(),
    }
}
