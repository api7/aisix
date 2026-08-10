//! CRUD handlers for `/admin/v1/models`.
//!
//! Every mutating endpoint:
//! 1. validates the JSON body against the Model schema (aisix-core),
//! 2. rejects duplicate `name` against other resources in the store,
//! 3. persists via `ConfigStore`,
//! 4. returns the full `ResourceEntry<Model>` as JSON.
//!
//! ids are UUID v4s generated on POST; PUT preserves the existing id.

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
