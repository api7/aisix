use axum::{
    Json,
    response::{IntoResponse, Response},
};
use http::StatusCode;
use log::error;
use serde_json::{Map, Value, json};

use crate::gateway::error::GatewayError;

pub(crate) fn gateway_error_response(
    handler_name: &'static str,
    err: &GatewayError,
    status: StatusCode,
) -> Response {
    error!("{handler_name} gateway error: {err}");

    match err {
        GatewayError::Provider { body, .. } => {
            provider_error_response(status, provider_error_message(body), Some(body.clone()))
        }
        GatewayError::Http(http_error) => provider_error_response(
            status,
            format!("Upstream HTTP error: {http_error}"),
            Some(Value::String(http_error.to_string())),
        ),
        GatewayError::Stream(stream_error) => provider_error_response(
            status,
            format!("Upstream stream error: {stream_error}"),
            Some(Value::String(stream_error.clone())),
        ),
        GatewayError::EmbeddingsNotSupported { .. } => {
            provider_error_response(status, "Provider error".into(), None)
        }
        GatewayError::Internal(_) => internal_error_response(status),
        _ => generic_gateway_error_response(status, err),
    }
}

pub(crate) fn timeout_response(handler_name: &'static str) -> Response {
    error!("{handler_name} request timed out");

    (
        StatusCode::GATEWAY_TIMEOUT,
        Json(json!({
            "error": {
                "message": "Provider request timed out",
                "type": "server_error",
                "code": "request_timeout"
            }
        })),
    )
        .into_response()
}

pub(crate) fn missing_model_response(handler_name: &'static str) -> Response {
    error!("{handler_name} model missing in request context");

    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": {
                "message": "model missing in request context",
                "type": "server_error",
                "code": "internal_error"
            }
        })),
    )
        .into_response()
}

fn provider_error_response(status: StatusCode, message: String, details: Option<Value>) -> Response {
    let mut error_object = Map::from_iter([
        ("message".to_string(), Value::String(message)),
        ("type".to_string(), Value::String("server_error".into())),
        ("code".to_string(), Value::String("provider_error".into())),
    ]);
    if let Some(details) = details.filter(|details| !details.is_null()) {
        error_object.insert("details".to_string(), details);
    }

    (
        status,
        Json(json!({
            "error": Value::Object(error_object),
        })),
    )
        .into_response()
}

fn internal_error_response(status: StatusCode) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": "Gateway internal error",
                "type": "server_error",
                "code": "internal_error"
            }
        })),
    )
        .into_response()
}

fn generic_gateway_error_response(status: StatusCode, err: &GatewayError) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": err.to_string(),
                "type": if status.is_client_error() {
                    "invalid_request_error"
                } else {
                    "server_error"
                },
                "code": "gateway_error"
            }
        })),
    )
        .into_response()
}

fn provider_error_message(body: &Value) -> String {
    provider_nested_message(body).unwrap_or_else(|| "Provider error".into())
}

fn provider_nested_message(body: &Value) -> Option<String> {
    match body {
        Value::Object(map) => match map.get("error") {
            Some(Value::Object(error_object)) => error_object.get("message").and_then(value_to_string),
            Some(Value::String(text)) => Some(text.clone()),
            _ => map.get("message").and_then(value_to_string),
        },
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        _ => None,
    }
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}
