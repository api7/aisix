use std::sync::LazyLock;

use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use http::StatusCode;
use serde_json::json;
use uuid::Uuid;

use crate::{
    admin::{
        AppState,
        types::{APIError, DeleteResponse, ItemResponse, ListResponse},
    },
    config::{PutEntry, entities::Model},
};

static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema#",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "model": {
                "type": "string",
                "pattern": "^(deepseek|gemini|openai|mock)/.+$"
            },
            "provider_config": {"type": "object"},
            "rate_limit": {"type": "object"}
        },
        "required": ["name", "model", "provider_config"],
        "allOf": [
            {
                "if": {
                    "properties": {
                        "model": { "pattern": "^deepseek/" }
                    },
                    "required": ["model"]
                },
                "then": {
                    "properties": {
                        "provider_config": { "$ref": "#/$defs/openai_compatible" }
                    }
                }
            },
            {
                "if": {
                    "properties": {
                        "model": { "pattern": "^gemini/" }
                    },
                    "required": ["model"]
                },
                "then": {
                    "properties": {
                        "provider_config": { "$ref": "#/$defs/openai_compatible" }
                    }
                }
            },
            {
                "if": {
                    "properties": {
                        "model": { "pattern": "^openai/" }
                    },
                    "required": ["model"]
                },
                "then": {
                    "properties": {
                        "provider_config": { "$ref": "#/$defs/openai_compatible" }
                    }
                }
            },
            {
                "if": {
                    "properties": {
                        "model": { "pattern": "^mock/" }
                    },
                    "required": ["model"]
                },
                "then": {
                    "properties": {
                        "provider_config": { "additionalProperties": false }
                    },
                }
            }
        ],
        "$defs": {
            "openai_compatible": {
                "type": "object",
                "required": ["api_key"],
                "properties": {
                    "api_key": {"type": "string"},
                    "api_base": {"type": "string"}
                },
                "additionalProperties": false
            }
        }
    })
});
static SCHEMA_VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| jsonschema::validator_for(&*SCHEMA).expect("Invalid JSON schema for Model"));
pub const OPENAPI_TAG: &str = "AI Models";

#[utoipa::path(
    get,
    context_path = crate::admin::PATH_PREFIX,
    path = "/models",
    tag = OPENAPI_TAG,
    responses(
        (status = StatusCode::OK, description = "Get model list success", body = ListResponse<ItemResponse<Model>>),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn list(State(state): State<AppState>) -> Response {
    let data = match state
        .config_provider
        .get_all::<serde_json::Value>("/models")
        .await
    {
        Ok(data) => data,
        Err(err) => {
            return APIError::InternalError(err).into_response();
        }
    };

    ListResponse {
        total: data.len(),
        list: data
            .into_iter()
            .map(|item| ItemResponse {
                key: item.key,
                value: item.value,
                created_index: Some(item.create_revision),
                modified_index: Some(item.mod_revision),
            })
            .collect(),
    }
    .into_response()
}

#[utoipa::path(
    get,
    context_path = crate::admin::PATH_PREFIX,
    path = "/models/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the model"),
    ),
    responses(
        (status = StatusCode::OK, description = "Get model success", body = ItemResponse<Model>),
        (status = StatusCode::NOT_FOUND, description = "Model not found", body = APIError),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn get(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let key = format!("/models/{}", id);
    let data = match state.config_provider.get::<serde_json::Value>(&key).await {
        Ok(opt) => match opt {
            Some(data) => data,
            None => {
                return APIError::NotFound(format!("Model with ID {} not found", id))
                    .into_response();
            }
        },
        Err(err) => {
            return APIError::InternalError(err).into_response();
        }
    };

    ItemResponse {
        key,
        value: data.value,
        created_index: Some(data.create_revision),
        modified_index: Some(data.mod_revision),
    }
    .into_response()
}

#[utoipa::path(
    post,
    context_path = crate::admin::PATH_PREFIX,
    path = "/models",
    tag = OPENAPI_TAG,
    request_body(content_type = "application/json", content = Model),
    responses(
        (status = StatusCode::CREATED, description = "Model created successfully", body = ItemResponse<Model>),
        (status = StatusCode::BAD_REQUEST, description = "Bad request", body = APIError),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn post(State(state): State<AppState>, body: Bytes) -> Response {
    update(
        state,
        &format!("/models/{}", Uuid::new_v4().to_string()),
        body,
    )
    .await
}

#[utoipa::path(
    put,
    context_path = crate::admin::PATH_PREFIX,
    path = "/models/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the model"),
    ),
    request_body(content_type = "application/json", content = Model),
    responses(
        (status = StatusCode::OK, description = "Model updated successfully", body = ItemResponse<Model>),
        (status = StatusCode::CREATED, description = "Model created successfully", body = ItemResponse<Model>),
        (status = StatusCode::BAD_REQUEST, description = "Bad request", body = APIError),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn put(State(state): State<AppState>, Path(id): Path<String>, body: Bytes) -> Response {
    update(state, &format!("/models/{}", id), body).await
}

#[utoipa::path(
    delete,
    context_path = crate::admin::PATH_PREFIX,
    path = "/models/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the model"),
    ),
    responses(
        (status = StatusCode::OK, description = "Model deleted successfully", body = DeleteResponse),
        (status = StatusCode::NOT_FOUND, description = "Model not found", body = APIError),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn delete(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let key = format!("/models/{}", id);
    match state.config_provider.delete(&key).await {
        Ok(deleted) if deleted > 0 => DeleteResponse { deleted, key }.into_response(),
        Ok(_) => APIError::NotFound(format!("Model with ID {} not found", id)).into_response(),
        Err(err) => APIError::InternalError(err).into_response(),
    }
}

async fn update(state: AppState, key: &str, body: Bytes) -> Response {
    let model = match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(value) => value,
        Err(err) => {
            return APIError::BadRequest(format!("Invalid JSON: {}", err)).into_response();
        }
    };

    match SCHEMA_VALIDATOR.validate(&model) {
        Ok(_) => {}
        Err(err) => {
            return APIError::BadRequest(format!("JSON schema validation error: {}", err))
                .into_response();
        }
    };

    match state.config_provider.put(&key, &model).await {
        Ok(res) => match res {
            PutEntry::Created => (
                StatusCode::CREATED,
                ItemResponse {
                    key: key.to_string(),
                    value: model,
                    created_index: None,
                    modified_index: None,
                },
            )
                .into_response(),
            PutEntry::Updated(_prev) => (
                StatusCode::OK,
                ItemResponse {
                    key: key.to_string(),
                    value: model,
                    created_index: None,
                    modified_index: None,
                },
            )
                .into_response(),
        },
        Err(err) => APIError::InternalError(err).into_response(),
    }
}
