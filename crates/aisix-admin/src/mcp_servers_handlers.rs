//! CRUD handlers for `/admin/v1/mcp_servers`.
//!
//! Same shape as the ProviderKeys handlers: validate against the JSON schema,
//! reject duplicate display_names (409), generate a uuid v4 on POST, bump
//! revision on PUT. Additionally rejects a display_name containing the reserved
//! tool-namespace separator `__`, since the name prefixes the server's tools.

use aisix_core::models::validate_mcp_server;
use aisix_core::resource::ResourceEntry;
use aisix_core::{McpAuthType, McpServer};
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
    // Per-auth_type credential coupling. The JSON schema stays flat and
    // permissive on this (see the note on the McpServer struct); the write
    // path is where an incomplete credential set is rejected outright.
    let has_secret = !server.secret.as_deref().unwrap_or_default().is_empty();
    match server.auth_type {
        McpAuthType::None => {}
        McpAuthType::Bearer if !has_secret => {
            return Err(AdminError::BadRequest(
                "secret is required and must be non-empty when auth_type is `bearer`".to_string(),
            ));
        }
        McpAuthType::ApiKey if !has_secret => {
            return Err(AdminError::BadRequest(
                "secret is required and must be non-empty when auth_type is `api_key`".to_string(),
            ));
        }
        McpAuthType::OAuth2 => {
            let has_client_id = !server.client_id.as_deref().unwrap_or_default().is_empty();
            let has_token_url = !server.token_url.as_deref().unwrap_or_default().is_empty();
            if !has_secret || !has_client_id || !has_token_url {
                return Err(AdminError::BadRequest(
                    "client_id, token_url, and secret (the OAuth client secret) are required \
                     and must be non-empty when auth_type is `oauth2`"
                        .to_string(),
                ));
            }
        }
        McpAuthType::Bearer | McpAuthType::ApiKey => {}
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decode_rejects_separator_in_display_name() {
        let err = decode(&json!({"display_name": "a__b", "url": "https://x/mcp"}))
            .expect_err("`__` in display_name must be rejected");
        assert!(matches!(err, AdminError::BadRequest(_)));
    }

    #[test]
    fn decode_rejects_bearer_without_secret() {
        let err = decode(&json!({
            "display_name": "gh",
            "url": "https://x/mcp",
            "auth_type": "bearer"
        }))
        .expect_err("bearer auth without a secret must be rejected");
        assert!(matches!(err, AdminError::BadRequest(_)));
    }

    #[test]
    fn decode_rejects_api_key_without_secret() {
        let err = decode(&json!({
            "display_name": "gh",
            "url": "https://x/mcp",
            "auth_type": "api_key"
        }))
        .expect_err("api_key auth without a secret must be rejected");
        assert!(matches!(err, AdminError::BadRequest(_)));
    }

    #[test]
    fn decode_rejects_incomplete_oauth2() {
        // Each of client_id / token_url / secret is individually required.
        for missing in ["client_id", "token_url", "secret"] {
            let mut v = json!({
                "display_name": "gh",
                "url": "https://x/mcp",
                "auth_type": "oauth2",
                "client_id": "cid",
                "token_url": "https://auth.example.com/oauth/token",
                "secret": "cs"
            });
            v.as_object_mut().unwrap().remove(missing);
            let err = decode(&v).unwrap_err();
            assert!(
                matches!(err, AdminError::BadRequest(_)),
                "oauth2 without `{missing}` must be a BadRequest"
            );
        }
    }

    #[test]
    fn decode_accepts_api_key_and_oauth2_servers() {
        let api_key = decode(&json!({
            "display_name": "gh",
            "url": "https://x/mcp",
            "auth_type": "api_key",
            "secret": "k-1"
        }))
        .expect("valid api_key server should decode");
        assert_eq!(api_key.secret.as_deref(), Some("k-1"));

        let oauth2 = decode(&json!({
            "display_name": "gh2",
            "url": "https://x/mcp",
            "auth_type": "oauth2",
            "client_id": "cid",
            "token_url": "https://auth.example.com/oauth/token",
            "secret": "cs",
            "scopes": ["read"]
        }))
        .expect("valid oauth2 server should decode");
        assert_eq!(oauth2.client_id.as_deref(), Some("cid"));
        assert_eq!(
            oauth2.token_url.as_deref(),
            Some("https://auth.example.com/oauth/token")
        );
    }

    #[test]
    fn decode_accepts_valid_server() {
        let server = decode(&json!({
            "display_name": "github",
            "url": "https://api.example.com/mcp",
            "auth_type": "bearer",
            "secret": "tok"
        }))
        .expect("valid server should decode");
        assert_eq!(server.display_name, "github");
        assert_eq!(server.secret.as_deref(), Some("tok"));
    }
}
