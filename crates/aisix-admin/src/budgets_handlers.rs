//! CRUD handlers for `/admin/v1/budgets`.

use aisix_core::models::validate_budget;
use aisix_core::resource::ResourceEntry;
use aisix_core::Budget;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

pub async fn list_budgets(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<Budget>>>, AdminError> {
    let entries = state.store.list_budgets().await?;
    Ok(Json(entries))
}

pub async fn get_budget(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<Budget>>, AdminError> {
    let entry = state
        .store
        .get_budget(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}

pub async fn create_budget(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<Budget>>, AdminError> {
    let budget = decode(&raw)?;
    let all = state.store.list_budgets().await?;
    assert_unique_name(&all, &budget.name, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, budget, STARTING_REVISION);
    state.store.put_budget(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn update_budget(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<Budget>>, AdminError> {
    let existing = state
        .store
        .get_budget(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let budget = decode(&raw)?;

    let all = state.store.list_budgets().await?;
    assert_unique_name(&all, &budget.name, Some(&id))?;

    let entry = ResourceEntry::new(&id, budget, existing.revision + 1);
    state.store.put_budget(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn delete_budget(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_budget(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

fn decode(raw: &Value) -> Result<Budget, AdminError> {
    validate_budget(raw)?;
    serde_json::from_value(raw.clone())
        .map_err(|e| AdminError::BadRequest(format!("malformed Budget payload: {e}")))
}

fn assert_unique_name(
    existing: &[ResourceEntry<Budget>],
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
