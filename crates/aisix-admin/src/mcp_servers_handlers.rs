//! Read handlers for `/admin/v1/mcp_servers`: list and get-by-id,
//! same shape as [`crate::models_handlers`]. Name constraints (no
//! reserved `__` tool-namespace separator, no trailing `_`) live in
//! the canonical schema, enforced on every declarative write path.

use aisix_core::resource::ResourceEntry;
use aisix_core::McpServer;
use axum::extract::{Path, State};
use axum::Json;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

pub async fn list_mcp_servers(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<McpServer>>>, AdminError> {
    let entries = state.store.list_mcp_servers().await?;
    Ok(Json(entries))
}

pub async fn get_mcp_server(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<McpServer>>, AdminError> {
    let entry = state
        .store
        .get_mcp_server(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}
