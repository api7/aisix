//! Read handlers for `/admin/v1/observability_exporters`: list and
//! get-by-id, same shape as [`crate::models_handlers`].

use aisix_core::resource::ResourceEntry;
use aisix_core::ObservabilityExporter;
use axum::extract::{Path, State};
use axum::Json;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

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
