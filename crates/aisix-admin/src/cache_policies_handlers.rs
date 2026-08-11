//! Read handlers for `/admin/v1/cache_policies`: list and get-by-id,
//! same shape as [`crate::models_handlers`].

use aisix_core::resource::ResourceEntry;
use aisix_core::CachePolicy;
use axum::extract::{Path, State};
use axum::Json;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

pub async fn list_cache_policies(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<CachePolicy>>>, AdminError> {
    let entries = state.store.list_cache_policies().await?;
    Ok(Json(entries))
}

pub async fn get_cache_policy(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<CachePolicy>>, AdminError> {
    let entry = state
        .store
        .get_cache_policy(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}
