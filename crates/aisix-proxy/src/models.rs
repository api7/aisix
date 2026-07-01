//! `GET /v1/models` — OpenAI-compatible model listing.
//!
//! Returns the subset of Models in the current snapshot that the
//! authenticated ApiKey is permitted to use. The response shape
//! matches the OpenAI `/v1/models` contract so any client that uses
//! `client.models.list()` sees the models available to it.
//!
//! Each Model surfaces as:
//! ```json
//! {
//!   "id":       "<model.name>",
//!   "object":   "model",
//!   "created":  <unix seconds>,
//!   "owned_by": "<provider>"
//! }
//! ```
//!
//! The wrapping list object follows OpenAI's `ListResponse` envelope:
//! ```json
//! { "object": "list", "data": [ ... ] }
//! ```

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::auth::AuthenticatedKey;
use crate::state::ProxyState;

/// A single model entry in the `/v1/models` response.
#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
}

/// OpenAI-style list envelope.
#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

pub async fn list_models(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
) -> impl IntoResponse {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let snapshot = state.snapshot.load();

    // Collect the names of all non-routing models (routing aliases are
    // implementation detail, not something callers PUT into requests).
    // Wildcard aliases (`provider/*`) are patterns, not concrete ids a caller
    // can request by name, so they're excluded too.
    // Then filter to what the authenticated key may access.
    let all_names: Vec<String> = snapshot
        .models
        .entries()
        .into_iter()
        .filter(|e| !e.value.is_routing() && !e.value.display_name.contains('*'))
        .map(|e| e.value.display_name.clone())
        .collect();

    let api_key = auth.key();
    let permitted: Vec<&str> =
        api_key.accessible_models(all_names.iter().map(|s: &String| s.as_str()));

    let mut data: Vec<ModelObject> = permitted
        .into_iter()
        .map(|name| {
            // owner = provider name if we can resolve it, otherwise "aisix".
            let owned_by = snapshot
                .models
                .get_by_name(name)
                .and_then(|e| e.value.provider.as_ref().cloned())
                .unwrap_or_else(|| "aisix".to_string());

            ModelObject {
                id: name.to_string(),
                object: "model",
                created: now,
                owned_by,
            }
        })
        .collect();

    // Stable ordering so clients see a deterministic list.
    data.sort_by(|a, b| a.id.cmp(&b.id));

    Json(ModelList {
        object: "list",
        data,
    })
}

#[cfg(test)]
mod tests {

    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use aisix_provider_openai::OpenAiBridge;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use axum::Router;
    use std::sync::Arc;

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    fn build_app(snapshot: AisixSnapshot) -> Router {
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snapshot);
        crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn model_entry(id: &str, name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, m, 1)
    }

    fn provider_key_entry() -> ResourceEntry<aisix_core::ProviderKey> {
        let pk: aisix_core::ProviderKey =
            serde_json::from_str(r#"{"display_name":"openai-up","secret":"sk-up","provider":"openai","adapter":"openai"}"#).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap() -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(provider_key_entry());
        snap
    }

    fn apikey_entry(key: &str, allowed: &[&str]) -> ResourceEntry<ApiKey> {
        // Plaintext in, hash on the wire (§9A.7B.4).
        let key_hash = ApiKey::hash_bearer(key);
        let json = format!(
            r#"{{"key_hash": "{key_hash}", "allowed_models": {}}}"#,
            serde_json::to_string(&allowed).unwrap()
        );
        let k: ApiKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("key-1", k, 1)
    }

    #[tokio::test]
    async fn unauthenticated_request_is_401() {
        let snap = new_snap();
        snap.models.insert(model_entry("m1", "gpt4"));
        snap.apikeys.insert(apikey_entry("sk-caller", &["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn wildcard_key_sees_all_non_routing_models() {
        let snap = new_snap();
        snap.models.insert(model_entry("m1", "gpt4"));
        snap.models.insert(model_entry("m2", "claude"));
        snap.apikeys.insert(apikey_entry("sk-caller", &["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
        let data = v["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        // sorted by id
        assert_eq!(data[0]["id"], "claude");
        assert_eq!(data[1]["id"], "gpt4");
    }

    #[tokio::test]
    async fn restricted_key_sees_only_allowed_models() {
        let snap = new_snap();
        snap.models.insert(model_entry("m1", "gpt4"));
        snap.models.insert(model_entry("m2", "claude"));
        snap.apikeys.insert(apikey_entry("sk-caller", &["gpt4"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let data = v["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "gpt4");
    }

    #[tokio::test]
    async fn empty_allowed_models_returns_empty_list() {
        let snap = new_snap();
        snap.models.insert(model_entry("m1", "gpt4"));
        snap.apikeys.insert(apikey_entry("sk-caller", &[]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let data = v["data"].as_array().unwrap();
        assert_eq!(data.len(), 0);
    }

    #[tokio::test]
    async fn routing_models_are_excluded_from_list() {
        let snap = new_snap();
        snap.models.insert(model_entry("m1", "gpt4"));
        // Insert a routing model.
        let routing_cfg = serde_json::json!({
            "display_name": "smart-router",
            "routing": {
                "strategy": "failover",
                "targets": [{"model": "gpt4"}]
            }
        });
        let routing: Model = serde_json::from_value(routing_cfg).unwrap();
        snap.models.insert(ResourceEntry::new("r1", routing, 1));
        snap.apikeys.insert(apikey_entry("sk-caller", &["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let data = v["data"].as_array().unwrap();
        // Only gpt4, not smart-router.
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "gpt4");
    }

    #[tokio::test]
    async fn wildcard_models_are_excluded_from_list() {
        let snap = new_snap();
        snap.models.insert(model_entry("m1", "gpt4"));
        // A `provider/*` wildcard alias is a pattern, not a concrete id.
        snap.models.insert(model_entry("w1", "openai/*"));
        snap.apikeys.insert(apikey_entry("sk-caller", &["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let data = v["data"].as_array().unwrap();
        // Only the concrete gpt4, not the `openai/*` pattern.
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "gpt4");
    }

    #[tokio::test]
    async fn response_shape_matches_openai_contract() {
        let snap = new_snap();
        snap.models.insert(model_entry("m1", "gpt4"));
        snap.apikeys.insert(apikey_entry("sk-caller", &["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let item = &v["data"][0];
        assert_eq!(item["object"], "model");
        assert!(item["created"].as_i64().unwrap() > 0);
        assert_eq!(item["owned_by"], "openai");
    }
}
