//! CRUD handlers for `/admin/v1/apikeys`.
//!
//! Same shape as [`crate::models_handlers`], operating on `ApiKey`
//! resources. Duplicate-name detection uses `ApiKey::key` (which is the
//! ApiKey's unique human-readable name from [`aisix_core::Resource`]),
//! matching the proxy auth lookup by `by_name` index.
//!
//! Also provides key rotation: `POST /admin/v1/apikeys/:id/rotate`
//! replaces the `key` field with a freshly-generated `sk-*` value and
//! bumps the revision, invalidating the old credential.

use aisix_core::models::validate_apikey;
use aisix_core::resource::ResourceEntry;
use aisix_core::ApiKey;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

use crate::auth::AdminAuth;
use crate::error::AdminError;
use crate::state::AdminState;

const STARTING_REVISION: i64 = 1;

pub async fn list_apikeys(
    _auth: AdminAuth,
    State(state): State<AdminState>,
) -> Result<Json<Vec<ResourceEntry<ApiKey>>>, AdminError> {
    let entries = state.store.list_apikeys().await?;
    Ok(Json(entries))
}

pub async fn get_apikey(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<ResourceEntry<ApiKey>>, AdminError> {
    let entry = state
        .store
        .get_apikey(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    Ok(Json(entry))
}

pub async fn create_apikey(
    _auth: AdminAuth,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<ApiKey>>, AdminError> {
    let apikey = decode_apikey(&raw)?;
    let all = state.store.list_apikeys().await?;
    assert_unique_key(&all, &apikey.key_hash, None)?;

    let id = Uuid::new_v4().to_string();
    let entry = ResourceEntry::new(&id, apikey, STARTING_REVISION);
    state.store.put_apikey(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn update_apikey(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
    Json(raw): Json<Value>,
) -> Result<Json<ResourceEntry<ApiKey>>, AdminError> {
    let existing = state
        .store
        .get_apikey(&id)
        .await?
        .ok_or(AdminError::NotFound)?;
    let apikey = decode_apikey(&raw)?;

    let all = state.store.list_apikeys().await?;
    assert_unique_key(&all, &apikey.key_hash, Some(&id))?;

    let entry = ResourceEntry::new(&id, apikey, existing.revision + 1);
    state.store.put_apikey(entry.clone()).await?;
    Ok(Json(entry))
}

pub async fn delete_apikey(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let removed = state.store.delete_apikey(&id).await?;
    if !removed {
        return Err(AdminError::NotFound);
    }
    Ok(Json(serde_json::json!({"deleted": true, "id": id})))
}

/// `POST /admin/v1/apikeys/:id/rotate`
///
/// Generates a new plaintext bearer (`sk-<uuid>`) and replaces the
/// stored `key_hash` with its SHA-256. The old hash stops working as
/// soon as the etcd watch propagates the new snapshot (≤ 500 ms).
///
/// **Returns the new plaintext exactly once** in the response body
/// under `plaintext` — admin caller MUST capture it. Subsequent GETs
/// only expose the hash. (Mirrors the cp-api self-hosted behavior in
/// prd-09a §9A.7B.4.)
pub async fn rotate_apikey(
    _auth: AdminAuth,
    Path(id): Path<String>,
    State(state): State<AdminState>,
) -> Result<Json<Value>, AdminError> {
    let existing = state
        .store
        .get_apikey(&id)
        .await?
        .ok_or(AdminError::NotFound)?;

    // `sk-` prefix + first segment of a UUID v4 gives a 12-hex-char
    // suffix that's unguessable yet short.
    let new_plaintext = format!("sk-{}", Uuid::new_v4().as_simple());
    let new_hash = ApiKey::hash_bearer(&new_plaintext);

    let mut updated = existing.value.clone();
    updated.key_hash = new_hash;

    let entry = ResourceEntry::new(&id, updated, existing.revision + 1);
    state.store.put_apikey(entry.clone()).await?;
    Ok(Json(serde_json::json!({
        "entry":     entry,
        "plaintext": new_plaintext,
    })))
}

fn decode_apikey(raw: &Value) -> Result<ApiKey, AdminError> {
    validate_apikey(raw)?;
    serde_json::from_value(raw.clone())
        .map_err(|e| AdminError::BadRequest(format!("malformed ApiKey payload: {e}")))
}

fn assert_unique_key(
    existing: &[ResourceEntry<ApiKey>],
    key_hash: &str,
    self_id: Option<&str>,
) -> Result<(), AdminError> {
    for e in existing {
        if e.value.key_hash == key_hash && self_id.is_none_or(|sid| sid != e.id) {
            return Err(AdminError::Conflict(key_hash.to_string()));
        }
    }
    Ok(())
}
