//! Read handlers for `/admin/v1/passthrough_routes`: list and get-by-id,
//! same shape as [`crate::mcp_servers_handlers`]. Cross-field coupling
//! (match dimensions, target shape, per-mode required companions) lives in
//! the canonical schema, enforced on every declarative write path.

use aisix_core::resource::ResourceEntry;
use aisix_core::PassthroughRoute;
use axum::extract::{Path, State};
use axum::Json;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

pub async fn list_passthrough_routes(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<PassthroughRoute>>>, AdminError> {
    let entries = state.store.list_passthrough_routes().await?;
    Ok(Json(entries))
}

pub async fn get_passthrough_route(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<PassthroughRoute>>, AdminError> {
    let entry = state
        .store
        .get_passthrough_route(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}
