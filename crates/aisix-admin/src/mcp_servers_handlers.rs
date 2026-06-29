//! CRUD handlers for `/admin/v1/mcp_servers`.
//!
//! Same shape as the ProviderKeys handlers: validate against the JSON schema,
//! reject duplicate display_names (409), generate a uuid v4 on POST, bump
//! revision on PUT. Additionally rejects a display_name containing the reserved
//! tool-namespace separator `__`, since the name prefixes the server's tools.

use aisix_core::models::validate_mcp_server;
use aisix_core::resource::ResourceEntry;
use aisix_core::McpServer;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

/// Reserved separator between a server's name and a tool name in the gateway's
/// aggregated namespace (`<display_name>__<tool>`). A server name must not
/// contain it.
const TOOL_NAMESPACE_SEPARATOR: &str = "__";

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

pub async fn create_mcp_server(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<McpServer>>, AdminError> {
    let mcp_server = decode(&raw)?;
    let all = state.store.list_mcp_servers().await?;
    assert_unique_display_name(&all, &mcp_server.display_name, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, mcp_server, STARTING_REVISION);
    state.store.put_mcp_server(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn update_mcp_server(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<McpServer>>, AdminError> {
    let existing = state
        .store
        .get_mcp_server(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let mcp_server = decode(&raw)?;

    let all = state.store.list_mcp_servers().await?;
    assert_unique_display_name(&all, &mcp_server.display_name, Some(&id))?;

    let entry = ResourceEntry::new(&id, mcp_server, existing.revision + 1);
    state.store.put_mcp_server(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn delete_mcp_server(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_mcp_server(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

fn decode(raw: &Value) -> Result<McpServer, AdminError> {
    validate_mcp_server(raw)?;
    let server: McpServer = serde_json::from_value(raw.clone())
        .map_err(|e| AdminError::BadRequest(format!("malformed McpServer payload: {e}")))?;
    if server.display_name.contains(TOOL_NAMESPACE_SEPARATOR) {
        return Err(AdminError::BadRequest(format!(
            "display_name must not contain the reserved separator `{TOOL_NAMESPACE_SEPARATOR}`"
        )));
    }
    Ok(server)
}

fn assert_unique_display_name(
    existing: &[ResourceEntry<McpServer>],
    display_name: &str,
    self_id: Option<&str>,
) -> Result<(), AdminError> {
    for e in existing {
        if e.value.display_name == display_name && self_id.is_none_or(|sid| sid != e.id) {
            return Err(AdminError::Conflict(display_name.to_string()));
        }
    }
    Ok(())
}
