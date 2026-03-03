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
        utils::format_jsonschema_error,
    },
    config::{PutEntry, entities::ApiKey},
};

static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema#",
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "allowed_models": {
                "type": "array",
                "items": { "type": "string" }
            },
            "rate_limit": {"type": "object"}
        },
        "required": ["key", "allowed_models"],
        "additionalProperties": false
    })
});
static SCHEMA_VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| jsonschema::validator_for(&*SCHEMA).expect("Invalid JSON schema for API Key"));
pub const OPENAPI_TAG: &str = "API Keys";

#[utoipa::path(
    get,
    context_path = crate::admin::PATH_PREFIX,
    path = "/apikeys",
    tag = OPENAPI_TAG,
    responses(
        (status = StatusCode::OK, description = "Get API key list success", body = ListResponse<ItemResponse<ApiKey>>),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn list(State(state): State<AppState>) -> Response {
    let data = match state
        .config_provider
        .get_all::<serde_json::Value>("/apikeys")
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
    path = "/apikeys/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the API key"),
    ),
    responses(
        (status = StatusCode::OK, description = "Get API key success", body = ItemResponse<ApiKey>),
        (status = StatusCode::NOT_FOUND, description = "API key not found", body = APIError),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn get(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let key = format!("/apikeys/{}", id);
    let data = match state.config_provider.get::<serde_json::Value>(&key).await {
        Ok(opt) => match opt {
            Some(data) => data,
            None => {
                return APIError::NotFound(format!("API key with ID {} not found", id))
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
    path = "/apikeys",
    tag = OPENAPI_TAG,
    request_body(content_type = "application/json", content = ApiKey),
    responses(
        (status = StatusCode::CREATED, description = "API key created successfully", body = ItemResponse<ApiKey>),
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
    path = "/apikeys/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the API key"),
    ),
    request_body(content_type = "application/json", content = ApiKey),
    responses(
        (status = StatusCode::OK, description = "API key updated successfully", body = ItemResponse<ApiKey>),
        (status = StatusCode::CREATED, description = "API key created successfully", body = ItemResponse<ApiKey>),
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
    path = "/apikeys/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = String, Path, description = "The ID of the API key"),
    ),
    responses(
        (status = StatusCode::OK, description = "API key deleted successfully", body = DeleteResponse),
        (status = StatusCode::NOT_FOUND, description = "API key not found", body = APIError),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Internal server error", body = APIError)
    )
)]
pub async fn delete(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let key = format!("/apikeys/{}", id);
    match state.config_provider.delete(&key).await {
        Ok(deleted) if deleted > 0 => DeleteResponse { deleted, key }.into_response(),
        Ok(_) => APIError::NotFound(format!("API key with ID {} not found", id)).into_response(),
        Err(err) => APIError::InternalError(err).into_response(),
    }
}

async fn update(state: AppState, id: &str, body: Bytes) -> Response {
    let key = format!("/apikeys/{}", id);

    let api_key = match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(value) => value,
        Err(err) => {
            return APIError::BadRequest(format!("Invalid JSON: {}", err)).into_response();
        }
    };

    let evaluation = SCHEMA_VALIDATOR.evaluate(&api_key);
    if !evaluation.flag().valid {
        return APIError::BadRequest(format!(
            "JSON schema validation error: {}",
            format_jsonschema_error(&evaluation)
        ))
        .into_response();
    }

    let api_key = match serde_json::from_value::<ApiKey>(api_key) {
        Ok(value) => value,
        Err(err) => {
            return APIError::BadRequest(format!("Invalid API key data: {}", err)).into_response();
        }
    };

    // Check if the API key already exists: fast path
    if let Some((found_id, _)) = state.resources.apikeys.get_by_key(&api_key.key)
        && found_id != id
    {
        return APIError::BadRequest("API key already exists".to_string()).into_response();
    }

    // Check if the API key already exists: slow path
    match state.config_provider.get_all::<ApiKey>("/apikeys").await {
        Ok(data) => {
            if data
                .iter()
                .any(|item| item.value.key == api_key.key && item.key != key)
            {
                return APIError::BadRequest("API key already exists".to_string()).into_response();
            }
        }
        Err(err) => {
            return APIError::InternalError(err).into_response();
        }
    }

    match state.config_provider.put(&key, &api_key).await {
        Ok(res) => match res {
            PutEntry::Created => (
                StatusCode::CREATED,
                ItemResponse {
                    key: key.to_string(),
                    value: api_key,
                    created_index: None,
                    modified_index: None,
                },
            )
                .into_response(),
            PutEntry::Updated(_prev) => (
                StatusCode::OK,
                ItemResponse {
                    key: key.to_string(),
                    value: api_key,
                    created_index: None,
                    modified_index: None,
                },
            )
                .into_response(),
        },
        Err(err) => APIError::InternalError(err).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{SCHEMA_VALIDATOR, format_jsonschema_error};

    #[rstest::rstest]
    #[case(json!({ // ok
        "key": "sk-test",
        "allowed_models": [],
    }), true, None)]
    #[case(json!({ // ok with rate_limit
        "key": "sk-test",
        "allowed_models": ["openai/gpt-4"],
        "rate_limit": {},
    }), true, None)]
    #[case(json!({ // missing key
        "allowed_models": [],
    }), false, Some(r#"property "/" validation failed: "key" is a required property"#))]
    #[case(json!({ // missing allowed_models
        "key": "sk-test",
    }), false, Some(r#"property "/" validation failed: "allowed_models" is a required property"#))]
    #[case(json!({ // invalid key type
        "key": 123,
        "allowed_models": [],
    }), false, Some(r#"property "/key" validation failed: 123 is not of type "string""#))]
    #[case(json!({ // invalid allowed_models type
        "key": "sk-test",
        "allowed_models": "not-an-array",
    }), false, Some(r#"property "/allowed_models" validation failed: "not-an-array" is not of type "array""#))]
    #[case(json!({ // invalid allowed_models element type
        "key": "sk-test",
        "allowed_models": [1],
    }), false, Some(r#"property "/allowed_models" validation failed: 1 at index 0 is not of type "string""#))]
    #[case(json!({ // invalid rate_limit type
        "key": "sk-test",
        "allowed_models": [],
        "rate_limit": 123,
    }), false, Some(r#"property "/rate_limit" validation failed: 123 is not of type "object""#))]
    #[case(json!({ // additional property not allowed
        "key": "sk-test",
        "allowed_models": [],
        "extra": "not allowed",
    }), false, Some(r#"property "/" validation failed: Additional properties are not allowed ('extra' was unexpected)"#))]
    fn schemas(
        #[case] input: serde_json::Value,
        #[case] ok: bool,
        #[case] expected_error: Option<&str>,
    ) {
        let evaluation = SCHEMA_VALIDATOR.evaluate(&input);

        assert_eq!(evaluation.flag().valid, ok, "unexpected evaluation result");
        if !ok {
            assert_eq!(
                format_jsonschema_error(&evaluation),
                expected_error.unwrap(),
                "unexpected error message"
            );
        }
    }
}
