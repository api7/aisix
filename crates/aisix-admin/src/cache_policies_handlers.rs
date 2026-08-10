//! CRUD handlers for `/admin/v1/cache_policies`.
//!
//! Same shape as the Models / ApiKeys / ProviderKeys handlers:
//! validate against the JSON schema, reject duplicate names (409),
//! generate a uuid v4 on POST, bump revision on PUT.

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
