use axum::{
    Json,
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde_json::json;

#[allow(unused)]
#[derive(Clone)]
pub struct RawRequestBody {
    pub bytes: Bytes,
}

impl From<Bytes> for RawRequestBody {
    fn from(bytes: Bytes) -> Self {
        Self { bytes }
    }
}

#[derive(Clone)]
pub struct RequestModel(pub String);

/// Middleware to parse request body and store in Extensions
/// TODO: move back to extractor?
pub async fn parse_body<T>(req: Request, next: Next) -> Result<Response, Response>
where
    T: DeserializeOwned + Clone + Send + Sync + 'static,
{
    let (mut parts, body) = req.into_parts();

    // Read body bytes
    //TODO: limit size
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(err) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("Failed to read request body: {}", err),
                        "type": "invalid_request_error",
                        "code": "bad_request"
                    }
                })),
            )
                .into_response());
        }
    };

    parts.extensions.insert(RawRequestBody::from(bytes.clone()));

    // Deserialize
    let body: T = match serde_json::from_slice(&bytes) {
        Ok(val) => val,
        Err(err) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("Failed to parse JSON: {}", err),
                        "type": "invalid_request_error",
                        "code": "invalid_json"
                    }
                })),
            )
                .into_response());
        }
    };

    // Store in Extensions
    parts.extensions.insert(body);

    // Continue
    Ok(next
        .run(Request::from_parts(parts, Body::from(bytes)))
        .await)
}
