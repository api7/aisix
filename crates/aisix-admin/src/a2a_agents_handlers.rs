//! CRUD handlers for `/admin/v1/a2a_agents`.
//!
//! Same shape as the McpServers handlers: validate against the JSON schema,
//! reject duplicate display_names (409), generate a uuid v4 on POST, bump
//! revision on PUT. The display_name is the path segment under which the agent
//! is exposed (`/a2a/<display_name>`), so it must be a single URL path segment
//! (no `/`). The per-auth_type credential coupling is enforced here too, since
//! the flat schema stays permissive on it.

use aisix_core::models::validate_a2a_agent;
use aisix_core::resource::ResourceEntry;
use aisix_core::{A2aAgent, A2aAuthType};
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

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

pub async fn create_a2a_agent(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<A2aAgent>>, AdminError> {
    let agent = decode(&raw)?;
    let all = state.store.list_a2a_agents().await?;
    assert_unique_display_name(&all, &agent.display_name, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, agent, STARTING_REVISION);
    state.store.put_a2a_agent(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn update_a2a_agent(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<A2aAgent>>, AdminError> {
    let existing = state
        .store
        .get_a2a_agent(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let agent = decode(&raw)?;

    let all = state.store.list_a2a_agents().await?;
    assert_unique_display_name(&all, &agent.display_name, Some(&id))?;

    let entry = ResourceEntry::new(&id, agent, existing.revision + 1);
    state.store.put_a2a_agent(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn delete_a2a_agent(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_a2a_agent(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

fn decode(raw: &Value) -> Result<A2aAgent, AdminError> {
    validate_a2a_agent(raw)?;
    let agent: A2aAgent = serde_json::from_value(raw.clone())
        .map_err(|e| AdminError::BadRequest(format!("malformed A2aAgent payload: {e}")))?;
    // The display_name is the path segment in `/a2a/<display_name>`, so it must
    // be a single URL path segment.
    if agent.display_name.contains('/') {
        return Err(AdminError::BadRequest(
            "display_name must not contain `/` — it is the agent's URL path segment".to_string(),
        ));
    }
    // Per-auth_type credential coupling. The JSON schema stays flat and
    // permissive on this (see the note on the A2aAgent struct); the write path
    // is where an incomplete credential set is rejected outright.
    let has_secret = !agent.secret.as_deref().unwrap_or_default().is_empty();
    match agent.auth_type {
        A2aAuthType::None => {}
        A2aAuthType::Bearer if !has_secret => {
            return Err(AdminError::BadRequest(
                "secret is required and must be non-empty when auth_type is `bearer`".to_string(),
            ));
        }
        A2aAuthType::ApiKey if !has_secret => {
            return Err(AdminError::BadRequest(
                "secret is required and must be non-empty when auth_type is `api_key`".to_string(),
            ));
        }
        A2aAuthType::Bearer | A2aAuthType::ApiKey => {}
    }
    Ok(agent)
}

fn assert_unique_display_name(
    existing: &[ResourceEntry<A2aAgent>],
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
    fn decode_rejects_slash_in_display_name() {
        let err = decode(&json!({"display_name": "a/b", "url": "https://x/a2a"}))
            .expect_err("`/` in display_name must be rejected");
        assert!(matches!(err, AdminError::BadRequest(_)));
    }

    #[test]
    fn decode_rejects_bearer_without_secret() {
        let err = decode(&json!({
            "display_name": "agent",
            "url": "https://x/a2a",
            "auth_type": "bearer"
        }))
        .expect_err("bearer auth without a secret must be rejected");
        assert!(matches!(err, AdminError::BadRequest(_)));
    }

    #[test]
    fn decode_rejects_oauth2_auth_type_as_schema_error() {
        // `oauth2` is not part of the a2a_agent resource model — `auth_type`
        // allows only `none` / `bearer` / `api_key` — so a payload carrying it
        // fails schema validation with a 400 before reaching the store.
        let err = decode(&json!({
            "display_name": "agent",
            "url": "https://x/a2a",
            "auth_type": "oauth2",
            "secret": "cs"
        }))
        .expect_err("oauth2 auth_type must be rejected by schema validation");
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn decode_accepts_valid_agent_with_pinned_version() {
        let agent = decode(&json!({
            "display_name": "invoice-processor",
            "url": "https://agents.example.com/a2a",
            "protocol_version": "0.3",
            "auth_type": "bearer",
            "secret": "tok"
        }))
        .expect("valid agent should decode");
        assert_eq!(agent.display_name, "invoice-processor");
        assert_eq!(agent.secret.as_deref(), Some("tok"));
    }
}
