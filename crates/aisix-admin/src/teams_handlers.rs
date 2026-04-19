//! CRUD handlers for `/admin/v1/teams`.
//!
//! A Team groups one or more ApiKeys under a shared name, optional budget
//! reference, and optional rate-limit policy. The handlers follow the
//! same validation-then-store pattern as every other admin entity.

use aisix_core::models::validate_team;
use aisix_core::resource::ResourceEntry;
use aisix_core::Team;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

pub async fn list_teams(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<Team>>>, AdminError> {
    let entries = state.store.list_teams().await?;
    Ok(Json(entries))
}

pub async fn get_team(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<Team>>, AdminError> {
    let entry = state
        .store
        .get_team(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}

pub async fn create_team(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<Team>>, AdminError> {
    let team = decode_team(&raw)?;
    let all = state.store.list_teams().await?;
    assert_unique_name(&all, &team.name, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, team, STARTING_REVISION);
    state.store.put_team(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn update_team(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<Team>>, AdminError> {
    let existing = state
        .store
        .get_team(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let team = decode_team(&raw)?;

    let all = state.store.list_teams().await?;
    assert_unique_name(&all, &team.name, Some(&id))?;

    let entry = ResourceEntry::new(&id, team, existing.revision + 1);
    state.store.put_team(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn delete_team(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_team(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

fn decode_team(raw: &Value) -> Result<Team, AdminError> {
    validate_team(raw)?;
    serde_json::from_value(raw.clone())
        .map_err(|e| AdminError::BadRequest(format!("malformed Team payload: {e}")))
}

fn assert_unique_name(
    existing: &[ResourceEntry<Team>],
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
