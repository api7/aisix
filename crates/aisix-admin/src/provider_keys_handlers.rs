//! CRUD handlers for `/admin/v1/provider_keys`.
//!
//! Same shape as the Models / ApiKeys handlers: validate against the
//! JSON schema, reject duplicate display_names (409), generate a uuid
//! v4 on POST, bump revision on PUT.

use aisix_core::resource::ResourceEntry;
use aisix_core::ProviderKey;
use axum::extract::{Path, State};
use axum::Json;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

pub async fn list_provider_keys(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<ProviderKey>>>, AdminError> {
    let entries = state.store.list_provider_keys().await?;
    Ok(Json(entries))
}

pub async fn get_provider_key(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<ProviderKey>>, AdminError> {
    let entry = state
        .store
        .get_provider_key(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}
