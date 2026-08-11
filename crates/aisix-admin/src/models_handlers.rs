//! Read handlers for `/admin/v1/models`: list and get-by-id over the
//! configured [`crate::ConfigStore`], returning `ResourceEntry<Model>`
//! as JSON. Models are written declaratively (resources file or etcd),
//! not through this surface.

use aisix_core::resource::ResourceEntry;
use aisix_core::Model;
use axum::extract::{Path, State};
use axum::Json;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

pub async fn list_models(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<Model>>>, AdminError> {
    let entries = state.store.list_models().await?;
    Ok(Json(entries))
}

pub async fn get_model(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<Model>>, AdminError> {
    let entry = state
        .store
        .get_model(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}
