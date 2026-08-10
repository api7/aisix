//! CRUD handlers for `/admin/v1/guardrails`.
//!
//! Same shape as the Models / ApiKeys / ProviderKeys handlers:
//! validate against the JSON schema, reject duplicate names (409),
//! generate a uuid v4 on POST, bump revision on PUT.

use aisix_core::resource::ResourceEntry;
use aisix_core::Guardrail;
use axum::extract::{Path, State};
use axum::Json;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

pub async fn list_guardrails(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<Guardrail>>>, AdminError> {
    let entries = state.store.list_guardrails().await?;
    Ok(Json(entries))
}

pub async fn get_guardrail(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<Guardrail>>, AdminError> {
    let entry = state
        .store
        .get_guardrail(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}
