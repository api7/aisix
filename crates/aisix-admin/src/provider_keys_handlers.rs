//! CRUD handlers for `/admin/v1/provider_keys`.
//!
//! Same shape as the Models / ApiKeys handlers: validate against the
//! JSON schema, reject duplicate display_names (409), generate a uuid
//! v4 on POST, bump revision on PUT.

use aisix_core::models::validate_provider_key;
use aisix_core::resource::ResourceEntry;
use aisix_core::ProviderKey;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

pub async fn list_provider_keys(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<ProviderKey>>>, AdminError> {
    let entries = state.store.list_provider_keys().await?;
    Ok(Json(entries))
}

pub async fn get_provider_key(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<ProviderKey>>, AdminError> {
    let entry = state
        .store
        .get_provider_key(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}

pub async fn create_provider_key(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<ProviderKey>>, AdminError> {
    let provider_key = decode(&raw)?;
    let all = state.store.list_provider_keys().await?;
    assert_unique_display_name(&all, &provider_key.display_name, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, provider_key, STARTING_REVISION);
    state.store.put_provider_key(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn update_provider_key(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<ProviderKey>>, AdminError> {
    let existing = state
        .store
        .get_provider_key(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let provider_key = decode(&raw)?;

    let all = state.store.list_provider_keys().await?;
    assert_unique_display_name(&all, &provider_key.display_name, Some(&id))?;

    let entry = ResourceEntry::new(&id, provider_key, existing.revision + 1);
    state.store.put_provider_key(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn delete_provider_key(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_provider_key(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

fn decode(raw: &Value) -> Result<ProviderKey, AdminError> {
    validate_provider_key(raw)?;
    serde_json::from_value(raw.clone())
        .map_err(|e| AdminError::BadRequest(format!("malformed ProviderKey payload: {e}")))
}

fn assert_unique_display_name(
    existing: &[ResourceEntry<ProviderKey>],
    display_name: &str,
    self_id: Option<&str>,
) -> Result<(), AdminError> {
    for e in existing {
        if e.value.display_name == display_name && self_id.is_none_or(|sid| sid != e.id) {
            return Err(AdminError::Conflict(display_name.to_string()));
        }
    }
    Ok(())
}
