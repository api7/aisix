use std::collections::HashSet;

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
        entities::{
            Guardrail, Policy,
            policies::{SCHEMA_VALIDATOR, validate_policy_definition},
        },
    },
    utils::jsonschema::format_evaluation_error,
};

pub const OPENAPI_TAG: &str = "Policies";

#[utoipa::path(
    get,
    context_path = crate::admin::PATH_PREFIX,
    path = "/policies",
    tag = OPENAPI_TAG,
    responses(
        (status = StatusCode::OK, description = "Get policy list success", body = ListResponse<ItemResponse<Policy>>),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn list(State(state): State<AppState>) -> Response {
    let data = match state
        .config_provider
        .get_all::<serde_json::Value>("/policies")
        .await
    {
        Ok(data) => data,
        Err(err) => return APIError::InternalError(err).into_response(),
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
    path = "/policies/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the policy"),
    ),
    responses(
        (status = StatusCode::OK, description = "Get policy success", body = ItemResponse<Policy>),
        (status = StatusCode::NOT_FOUND, description = "Policy not found", body = APIError),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn get(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let key = format!("/policies/{id}");
    let data = match state.config_provider.get::<serde_json::Value>(&key).await {
        Ok(Some(data)) => data,
        Ok(None) => {
            return APIError::NotFound(format!("Policy with ID {id} not found")).into_response();
        }
        Err(err) => return APIError::InternalError(err).into_response(),
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
    path = "/policies",
    tag = OPENAPI_TAG,
    request_body(content_type = "application/json", content = Policy),
    responses(
        (status = StatusCode::CREATED, description = "Policy created successfully", body = ItemResponse<Policy>),
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
    path = "/policies/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the policy"),
    ),
    request_body(content_type = "application/json", content = Policy),
    responses(
        (status = StatusCode::OK, description = "Policy updated successfully", body = ItemResponse<Policy>),
        (status = StatusCode::CREATED, description = "Policy created successfully", body = ItemResponse<Policy>),
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
    path = "/policies/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the policy"),
    ),
    responses(
        (status = StatusCode::OK, description = "Policy deleted successfully", body = DeleteResponse),
        (status = StatusCode::NOT_FOUND, description = "Policy not found", body = APIError),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn delete(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let key = format!("/policies/{id}");
    match state.config_provider.delete(&key).await {
        Ok(deleted) if deleted > 0 => DeleteResponse { deleted, key }.into_response(),
        Ok(_) => APIError::NotFound(format!("Policy with ID {id} not found")).into_response(),
        Err(err) => APIError::InternalError(err).into_response(),
    }
}

async fn update(state: AppState, id: &str, body: Bytes) -> Response {
    let key = format!("/policies/{id}");

    let policy = match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(value) => value,
        Err(err) => return APIError::BadRequest(format!("Invalid JSON: {err}")).into_response(),
    };

    let evaluation = SCHEMA_VALIDATOR.evaluate(&policy);
    if !evaluation.flag().valid {
        return APIError::BadRequest(format!(
            "JSON schema validation error: {}",
            format_evaluation_error(&evaluation)
        ))
        .into_response();
    }

    let policy = match serde_json::from_value::<Policy>(policy) {
        Ok(value) => value,
        Err(err) => {
            return APIError::BadRequest(format!("Invalid policy data: {err}")).into_response();
        }
    };

    if let Err(err) = validate_policy_definition(id, &policy) {
        return APIError::BadRequest(err).into_response();
    }

    let mut seen_guardrails = HashSet::new();
    for guardrail_id in policy.referenced_guardrail_ids() {
        if !seen_guardrails.insert(guardrail_id.to_string()) {
            continue;
        }

        let guardrail_key = format!("/guardrails/{guardrail_id}");
        match state.config_provider.get::<Guardrail>(&guardrail_key).await {
            Ok(Some(_)) => {}
            Ok(None) => {
                return APIError::BadRequest(format!("Guardrail with ID {guardrail_id} not found"))
                    .into_response();
            }
            Err(err) => return APIError::InternalError(err).into_response(),
        }
    }

    if let Some(found) = state.resources.policies.get_by_name(&policy.name)
        && found.id != id
    {
        return APIError::BadRequest("Policy name already exists".to_string()).into_response();
    }

    match state.config_provider.get_all::<Policy>("/policies").await {
        Ok(data) => {
            if data
                .iter()
                .any(|item| item.value.name == policy.name && item.key != key)
            {
                return APIError::BadRequest("Policy name already exists".to_string())
                    .into_response();
            }
        }
        Err(err) => return APIError::InternalError(err).into_response(),
    }

    match state.config_provider.put(&key, &policy).await {
        Ok(res) => match res {
            PutEntry::Created => (
                StatusCode::CREATED,
                ItemResponse {
                    key: key.to_string(),
                    value: policy,
                    created_index: None,
                    modified_index: None,
                },
            )
                .into_response(),
            PutEntry::Updated(_prev) => (
                StatusCode::OK,
                ItemResponse {
                    key: key.to_string(),
                    value: policy,
                    created_index: None,
                    modified_index: None,
                },
            )
                .into_response(),
        },
        Err(err) => APIError::InternalError(err).into_response(),
    }
}
