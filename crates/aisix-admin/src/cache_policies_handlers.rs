//! CRUD handlers for `/admin/v1/cache_policies`.
//!
//! Same shape as the Models / ApiKeys / ProviderKeys handlers:
//! validate against the JSON schema, reject duplicate names (409),
//! generate a uuid v4 on POST, bump revision on PUT.

use aisix_core::models::validate_cache_policy;
use aisix_core::resource::ResourceEntry;
use aisix_core::CachePolicy;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

pub async fn list_cache_policies(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<CachePolicy>>>, AdminError> {
    let entries = state.store.list_cache_policies().await?;
    Ok(Json(entries))
}

pub async fn get_cache_policy(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<CachePolicy>>, AdminError> {
    let entry = state
        .store
        .get_cache_policy(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}

pub async fn create_cache_policy(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<CachePolicy>>, AdminError> {
    let cache_policy = decode(&raw)?;
    let all = state.store.list_cache_policies().await?;
    assert_unique_name(&all, &cache_policy.name, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, cache_policy, STARTING_REVISION);
    state.store.put_cache_policy(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn update_cache_policy(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<CachePolicy>>, AdminError> {
    let existing = state
        .store
        .get_cache_policy(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let cache_policy = decode(&raw)?;

    let all = state.store.list_cache_policies().await?;
    assert_unique_name(&all, &cache_policy.name, Some(&id))?;

    let entry = ResourceEntry::new(&id, cache_policy, existing.revision + 1);
    state.store.put_cache_policy(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn delete_cache_policy(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_cache_policy(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

fn decode(raw: &Value) -> Result<CachePolicy, AdminError> {
    validate_cache_policy(raw)?;
    serde_json::from_value(raw.clone())
        .map_err(|e| AdminError::BadRequest(format!("malformed CachePolicy payload: {e}")))
}

fn assert_unique_name(
    existing: &[ResourceEntry<CachePolicy>],
    name: &str,
    self_id: Option<&str>,
) -> Result<(), AdminError> {
    for e in existing {
        if e.value.name == name && self_id.is_none_or(|sid| sid != e.id) {
            return Err(AdminError::Conflict(name.to_string()));
        }
    }
    Ok(())
}
