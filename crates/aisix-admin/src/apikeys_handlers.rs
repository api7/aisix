//! Read handlers for `/admin/v1/api_keys` (and the former `apikeys`
//! spelling — same handlers).
//!
//! Same shape as [`crate::models_handlers`], operating on `ApiKey`
//! resources, except the responses project through [`PublicApiKey`]:
//! an explicit read-safe allowlist of fields, manually mapped, so a
//! field newly added to `ApiKey` never leaks here by default.

use aisix_core::resource::ResourceEntry;
use aisix_core::ApiKey;
use axum::extract::{Path, State};
use axum::Json;
use serde::Serialize;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

#[derive(Debug, Clone, Serialize)]
pub struct PublicApiKey {
    pub key_hash: String,
    pub allowed_models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<aisix_core::models::RateLimit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_access: Option<aisix_core::models::McpAccess>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_agents: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub disabled: bool,
}

impl From<ApiKey> for PublicApiKey {
    fn from(value: ApiKey) -> Self {
        Self {
            key_hash: value.key_hash,
            allowed_models: value.allowed_models,
            rate_limit: value.rate_limit,
            mcp_access: value.mcp_access,
            allowed_agents: value.allowed_agents,
            expires_at: value.expires_at,
            disabled: value.disabled,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicApiKeyEntry {
    pub id: String,
    pub value: PublicApiKey,
    pub revision: i64,
}

impl From<ResourceEntry<ApiKey>> for PublicApiKeyEntry {
    fn from(value: ResourceEntry<ApiKey>) -> Self {
        Self {
            id: value.id,
            value: PublicApiKey::from(value.value),
            revision: value.revision,
        }
    }
}

fn public_entry(entry: ResourceEntry<ApiKey>) -> PublicApiKeyEntry {
    entry.into()
}

pub async fn list_apikeys(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<PublicApiKeyEntry>>, AdminError> {
    let entries = state.store.list_apikeys().await?;
    Ok(Json(entries.into_iter().map(public_entry).collect()))
}

pub async fn get_apikey(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<PublicApiKeyEntry>, AdminError> {
    let entry = state
        .store
        .get_apikey(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(public_entry(entry)))
}
