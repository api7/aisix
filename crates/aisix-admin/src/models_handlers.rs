//! CRUD handlers for `/admin/v1/models`.
//!
//! Every mutating endpoint:
//! 1. validates the JSON body against the Model schema (aisix-core),
//! 2. rejects duplicate `name` against other resources in the store,
//! 3. persists via `ConfigStore`,
//! 4. returns the full `ResourceEntry<Model>` as JSON.
//!
//! ids are UUID v4s generated on POST; PUT preserves the existing id.

use aisix_core::models::validate_model;
use aisix_core::resource::ResourceEntry;
use aisix_core::Model;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

pub async fn list_models(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<Model>>>, AdminError> {
    let entries = state.store.list_models().await?;
    Ok(Json(entries))
}

pub async fn get_model(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<Model>>, AdminError> {
    let entry = state
        .store
        .get_model(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}

pub async fn create_model(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<Model>>, AdminError> {
    let model = decode_model(&raw)?;
    let all = state.store.list_models().await?;
    assert_unique_name(&all, &model.name, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, model, STARTING_REVISION);
    state.store.put_model(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn update_model(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<Model>>, AdminError> {
    let existing = state
        .store
        .get_model(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let model = decode_model(&raw)?;

    let all = state.store.list_models().await?;
    assert_unique_name(&all, &model.name, Some(&id))?;

    let entry = ResourceEntry::new(&id, model, existing.revision + 1);
    state.store.put_model(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn delete_model(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_model(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

fn decode_model(raw: &Value) -> Result<Model, AdminError> {
    validate_model(raw)?;
    serde_json::from_value(raw.clone())
        .map_err(|e| AdminError::BadRequest(format!("malformed Model payload: {e}")))
}

fn assert_unique_name(
    existing: &[ResourceEntry<Model>],
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
