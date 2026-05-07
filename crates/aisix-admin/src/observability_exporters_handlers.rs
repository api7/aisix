//! CRUD handlers for `/admin/v1/observability_exporters`.
//!
//! Same shape as the Models / ApiKeys / ProviderKeys handlers:
//! validate against the JSON schema, reject duplicate names (409),
//! generate a uuid v4 on POST, bump revision on PUT.

use aisix_core::models::validate_observability_exporter;
use aisix_core::resource::ResourceEntry;
use aisix_core::ObservabilityExporter;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

pub async fn list_observability_exporters(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<ObservabilityExporter>>>, AdminError> {
    let entries = state.store.list_observability_exporters().await?;
    Ok(Json(entries))
}

pub async fn get_observability_exporter(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<ObservabilityExporter>>, AdminError> {
    let entry = state
        .store
        .get_observability_exporter(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}

pub async fn create_observability_exporter(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<ObservabilityExporter>>, AdminError> {
    let exporter = decode(&raw)?;
    let all = state.store.list_observability_exporters().await?;
    assert_unique_name(&all, &exporter.name, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, exporter, STARTING_REVISION);
    state
        .store
        .put_observability_exporter(entry.clone())
        .await?;
    Ok(Json(entry))
}

pub async fn update_observability_exporter(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<ObservabilityExporter>>, AdminError> {
    let existing = state
        .store
        .get_observability_exporter(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let exporter = decode(&raw)?;

    let all = state.store.list_observability_exporters().await?;
    assert_unique_name(&all, &exporter.name, Some(&id))?;

    let entry = ResourceEntry::new(&id, exporter, existing.revision + 1);
    state
        .store
        .put_observability_exporter(entry.clone())
        .await?;
    Ok(Json(entry))
}

pub async fn delete_observability_exporter(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_observability_exporter(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

fn decode(raw: &Value) -> Result<ObservabilityExporter, AdminError> {
    validate_observability_exporter(raw)?;
    serde_json::from_value(raw.clone()).map_err(|e| {
        AdminError::BadRequest(format!("malformed ObservabilityExporter payload: {e}"))
    })
}

fn assert_unique_name(
    existing: &[ResourceEntry<ObservabilityExporter>],
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
