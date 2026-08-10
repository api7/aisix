//! CRUD handlers for `/admin/v1/mcp_servers`.
//!
//! Same shape as the ProviderKeys handlers: validate against the JSON schema,
//! reject duplicate names (409), generate a uuid v4 on POST, bump revision on
//! PUT. Additionally rejects a name containing the reserved tool-namespace
//! separator `__`, since the name prefixes the server's tools.

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
