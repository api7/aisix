use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use http::StatusCode;
use uuid::Uuid;

use crate::{
    admin::{
        AppState,
        types::{APIError, DeleteResponse, ItemResponse, ListResponse},
    },
    config::{
        PutEntry,
        entities::{Model, Provider, models::SCHEMA_VALIDATOR},
    },
    utils::jsonschema::format_evaluation_error,
};

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
    update(state, &Uuid::new_v4().to_string(), body).await
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
    update(state, &id, body).await
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

async fn update(state: AppState, id: &str, body: Bytes) -> Response {
    let key = format!("/models/{id}");

    let model = match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(value) => value,
        Err(err) => {
            return APIError::BadRequest(format!("Invalid JSON: {}", err)).into_response();
        }
    };

    let evaluation = SCHEMA_VALIDATOR.evaluate(&model);
    if !evaluation.flag().valid {
        return APIError::BadRequest(format!(
            "JSON schema validation error: {}",
            format_evaluation_error(&evaluation)
        ))
        .into_response();
    }

    let model = match serde_json::from_value::<Model>(model) {
        Ok(value) => value,
        Err(err) => {
            return APIError::BadRequest(format!("Invalid model data: {}", err)).into_response();
        }
    };

    let provider_key = format!("/providers/{}", model.provider_id);
    match state.config_provider.get::<Provider>(&provider_key).await {
        Ok(Some(_)) => {}
        Ok(None) => {
            return APIError::BadRequest(format!(
                "Provider with ID {} not found",
                model.provider_id
            ))
            .into_response();
        }
        Err(err) => {
            return APIError::InternalError(err).into_response();
        }
    }

    // Check if the model name already exists: fast path
    if let Some(found) = state.resources.models.get_by_name(&model.name)
        && found.id != id
    {
        return APIError::BadRequest("Model name already exists".to_string()).into_response();
    }

    // Check if the model name already exists: slow path
    match state.config_provider.get_all::<Model>("/models").await {
        Ok(data) => {
            if data
                .iter()
                .any(|item| item.value.name == model.name && item.key != key)
            {
                return APIError::BadRequest("Model name already exists".to_string())
                    .into_response();
            }
        }
        Err(err) => {
            return APIError::InternalError(err).into_response();
        }
    }

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
