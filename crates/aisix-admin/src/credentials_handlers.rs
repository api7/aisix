//! CRUD handlers for `/admin/v1/credentials`.
//!
//! Same shape as the Models / ApiKeys handlers: validate against the
//! JSON schema, reject duplicate names (409), generate a uuid v4 on
//! POST, bump revision on PUT.

use aisix_core::models::validate_credential;
use aisix_core::resource::ResourceEntry;
use aisix_core::Credential;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

pub async fn list_credentials(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<Credential>>>, AdminError> {
    let entries = state.store.list_credentials().await?;
    Ok(Json(entries))
}

pub async fn get_credential(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<Credential>>, AdminError> {
    let entry = state
        .store
        .get_credential(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}

pub async fn create_credential(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<Credential>>, AdminError> {
    let credential = decode(&raw)?;
    let all = state.store.list_credentials().await?;
    assert_unique_name(&all, &credential.name, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, credential, STARTING_REVISION);
    state.store.put_credential(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn update_credential(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<Credential>>, AdminError> {
    let existing = state
        .store
        .get_credential(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let credential = decode(&raw)?;

    let all = state.store.list_credentials().await?;
    assert_unique_name(&all, &credential.name, Some(&id))?;

    let entry = ResourceEntry::new(&id, credential, existing.revision + 1);
    state.store.put_credential(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn delete_credential(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_credential(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

fn decode(raw: &Value) -> Result<Credential, AdminError> {
    validate_credential(raw)?;
    serde_json::from_value(raw.clone())
        .map_err(|e| AdminError::BadRequest(format!("malformed Credential payload: {e}")))
}

fn assert_unique_name(
    existing: &[ResourceEntry<Credential>],
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
