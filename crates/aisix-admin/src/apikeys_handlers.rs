//! CRUD handlers for `/admin/v1/apikeys`.
//!
//! Same shape as [`crate::models_handlers`], operating on `ApiKey`
//! resources. Duplicate-name detection uses `ApiKey::key` (which is the
//! ApiKey's unique human-readable name from [`aisix_core::Resource`]),
//! matching the proxy auth lookup by `by_name` index.

use aisix_core::models::validate_apikey;
use aisix_core::resource::ResourceEntry;
use aisix_core::ApiKey;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

pub async fn list_apikeys(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<ApiKey>>>, AdminError> {
    let entries = state.store.list_apikeys().await?;
    Ok(Json(entries))
}

pub async fn get_apikey(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<ApiKey>>, AdminError> {
    let entry = state
        .store
        .get_apikey(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}

pub async fn create_apikey(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<ApiKey>>, AdminError> {
    let apikey = decode_apikey(&raw)?;
    let all = state.store.list_apikeys().await?;
    assert_unique_key(&all, &apikey.key, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, apikey, STARTING_REVISION);
    state.store.put_apikey(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn update_apikey(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<ApiKey>>, AdminError> {
    let existing = state
        .store
        .get_apikey(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let apikey = decode_apikey(&raw)?;

    let all = state.store.list_apikeys().await?;
    assert_unique_key(&all, &apikey.key, Some(&id))?;

    let entry = ResourceEntry::new(&id, apikey, existing.revision + 1);
    state.store.put_apikey(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn delete_apikey(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_apikey(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

fn decode_apikey(raw: &Value) -> Result<ApiKey, AdminError> {
    validate_apikey(raw)?;
    serde_json::from_value(raw.clone())
        .map_err(|e| AdminError::BadRequest(format!("malformed ApiKey payload: {e}")))
}

fn assert_unique_key(
    existing: &[ResourceEntry<ApiKey>],
    key: &str,
    self_id: Option<&str>,
) -> Result<(), AdminError> {
    for e in existing {
        if e.value.key == key && self_id.is_none_or(|sid| sid != e.id) {
            return Err(AdminError::Conflict(key.to_string()));
        }
    }
    Ok(())
}
