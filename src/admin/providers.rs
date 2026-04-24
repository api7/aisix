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
        entities::{Model, Provider, providers::SCHEMA_VALIDATOR},
    },
    utils::jsonschema::format_evaluation_error,
};

pub const OPENAPI_TAG: &str = "Providers";

#[utoipa::path(
    get,
    context_path = crate::admin::PATH_PREFIX,
    path = "/providers",
    tag = OPENAPI_TAG,
    responses(
        (status = StatusCode::OK, description = "Get provider list success", body = ListResponse<ItemResponse<Provider>>),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn list(State(state): State<AppState>) -> Response {
    let data = match state
        .config_provider
        .get_all::<serde_json::Value>("/providers")
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
    path = "/providers/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the provider"),
    ),
    responses(
        (status = StatusCode::OK, description = "Get provider success", body = ItemResponse<Provider>),
        (status = StatusCode::NOT_FOUND, description = "Provider not found", body = APIError),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn get(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let key = format!("/providers/{id}");
    let data = match state.config_provider.get::<serde_json::Value>(&key).await {
        Ok(opt) => match opt {
            Some(data) => data,
            None => {
                return APIError::NotFound(format!("Provider with ID {} not found", id))
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
    path = "/providers",
    tag = OPENAPI_TAG,
    request_body(content_type = "application/json", content = Provider),
    responses(
        (status = StatusCode::CREATED, description = "Provider created successfully", body = ItemResponse<Provider>),
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
    path = "/providers/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the provider"),
    ),
    request_body(content_type = "application/json", content = Provider),
    responses(
        (status = StatusCode::OK, description = "Provider updated successfully", body = ItemResponse<Provider>),
        (status = StatusCode::CREATED, description = "Provider created successfully", body = ItemResponse<Provider>),
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
    path = "/providers/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the provider"),
    ),
    responses(
        (status = StatusCode::BAD_REQUEST, description = "Provider is still referenced by models", body = APIError),
        (status = StatusCode::OK, description = "Provider deleted successfully", body = DeleteResponse),
        (status = StatusCode::NOT_FOUND, description = "Provider not found", body = APIError),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn delete(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let key = format!("/providers/{id}");

    match state.config_provider.get::<serde_json::Value>(&key).await {
        Ok(Some(_)) => {}
        Ok(None) => {
            return APIError::NotFound(format!("Provider with ID {} not found", id))
                .into_response();
        }
        Err(err) => {
            return APIError::InternalError(err).into_response();
        }
    }

    match state.config_provider.get_all::<Model>("/models").await {
        Ok(models) => {
            if models.iter().any(|item| item.value.provider_id == id) {
                return APIError::BadRequest("provider is still referenced by models".to_string())
                    .into_response();
            }
        }
        Err(err) => {
            return APIError::InternalError(err).into_response();
        }
    }

    match state.config_provider.delete(&key).await {
        Ok(deleted) if deleted > 0 => DeleteResponse { deleted, key }.into_response(),
        Ok(_) => APIError::NotFound(format!("Provider with ID {} not found", id)).into_response(),
        Err(err) => APIError::InternalError(err).into_response(),
    }
}

async fn update(state: AppState, id: &str, body: Bytes) -> Response {
    let key = format!("/providers/{id}");

    let provider = match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(value) => value,
        Err(err) => {
            return APIError::BadRequest(format!("Invalid JSON: {}", err)).into_response();
        }
    };

    let evaluation = SCHEMA_VALIDATOR.evaluate(&provider);
    if !evaluation.flag().valid {
        return APIError::BadRequest(format!(
            "JSON schema validation error: {}",
            format_evaluation_error(&evaluation)
        ))
        .into_response();
    }

    let provider = match serde_json::from_value::<Provider>(provider) {
        Ok(value) => value,
        Err(err) => {
            return APIError::BadRequest(format!("Invalid provider data: {}", err)).into_response();
        }
    };

    match state.config_provider.put(&key, &provider).await {
        Ok(res) => match res {
            PutEntry::Created => (
                StatusCode::CREATED,
                ItemResponse {
                    key: key.to_string(),
                    value: provider,
                    created_index: None,
                    modified_index: None,
                },
            )
                .into_response(),
            PutEntry::Updated(_prev) => (
                StatusCode::OK,
                ItemResponse {
                    key: key.to_string(),
                    value: provider,
                    created_index: None,
                    modified_index: None,
                },
            )
                .into_response(),
        },
        Err(err) => APIError::InternalError(err).into_response(),
    }
}
