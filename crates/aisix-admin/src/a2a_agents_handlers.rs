//! CRUD handlers for `/admin/v1/a2a_agents`.
//!
//! Same shape as the McpServers handlers: validate against the JSON schema,
//! reject duplicate names (409), generate a uuid v4 on POST, bump revision on
//! PUT. The name is the path segment under which the agent is exposed
//! (`/a2a/<name>`), so it must be a single URL path segment (no `/`). The
//! per-auth_type credential coupling is enforced by the canonical schema, so
//! every configuration path rejects an incomplete credential set; the checks
//! below are defense in depth.

use aisix_core::resource::ResourceEntry;
use aisix_core::A2aAgent;
use axum::extract::{Path, State};
use axum::Json;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

pub async fn list_a2a_agents(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<A2aAgent>>>, AdminError> {
    let entries = state.store.list_a2a_agents().await?;
    Ok(Json(entries))
}

pub async fn get_a2a_agent(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<A2aAgent>>, AdminError> {
    let entry = state
        .store
        .get_a2a_agent(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}
