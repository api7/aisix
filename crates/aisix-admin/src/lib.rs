//! aisix-admin — Admin API + Playground + embedded UI (:3001).
//!
//! Mounts the admin surface behind admin-key bearer auth:
//! - `GET  /health`
//! - `GET|POST            /admin/v1/models`
//! - `GET|PUT|DELETE      /admin/v1/models/:id`
//! - `GET|POST            /admin/v1/apikeys`
//! - `GET|PUT|DELETE      /admin/v1/apikeys/:id`
//!
//! Writes validate against the JSON Schemas from `aisix-core` and reject
//! duplicate names (409). The storage layer is pluggable via the
//! [`ConfigStore`] trait; production wires an etcd-backed impl in a
//! follow-up PR, tests use [`InMemoryStore`].
//!
//! Errors follow the simple admin envelope: `{"error_msg": "..."}`,
//! distinct from the proxy's OpenAI-style envelope.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod apikeys_handlers;
mod auth;
mod error;
pub mod etcd_store;
mod models_handlers;
mod state;
pub mod store;

pub use auth::AdminAuth;
pub use error::{AdminError, ErrorBody};
pub use etcd_store::EtcdConfigStore;
pub use state::AdminState;
pub use store::{ConfigStore, InMemoryStore, StoreError};

use axum::routing::get;
use axum::{http::StatusCode, Json, Router};
use serde_json::json;

pub fn build_router(state: AdminState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route(
            "/admin/v1/models",
            get(models_handlers::list_models).post(models_handlers::create_model),
        )
        .route(
            "/admin/v1/models/:id",
            get(models_handlers::get_model)
                .put(models_handlers::update_model)
                .delete(models_handlers::delete_model),
        )
        .route(
            "/admin/v1/apikeys",
            get(apikeys_handlers::list_apikeys).post(apikeys_handlers::create_apikey),
        )
        .route(
            "/admin/v1/apikeys/:id",
            get(apikeys_handlers::get_apikey)
                .put(apikeys_handlers::update_apikey)
                .delete(apikeys_handlers::delete_apikey),
        )
        .with_state(state)
}

async fn health(
    axum::extract::State(state): axum::extract::State<AdminState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let snap = state.snapshot.load();
    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "models": snap.models.len(),
            "apikeys": snap.apikeys.len(),
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AdminConfig, AisixSnapshot};
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn cfg() -> AdminConfig {
        AdminConfig {
            addr: "127.0.0.1:0".into(),
            admin_keys: vec!["admin-secret".into()],
            tls: None,
        }
    }

    fn build_state() -> AdminState {
        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let store = InMemoryStore::new() as Arc<dyn ConfigStore>;
        AdminState::new(handle, store, &cfg())
    }

    fn model_payload(name: &str) -> Value {
        json!({
            "name": name,
            "model": "openai/gpt-4o",
            "provider_config": {"api_key": "sk-x"}
        })
    }

    fn apikey_payload(key: &str, allowed: &[&str]) -> Value {
        json!({"key": key, "allowed_models": allowed})
    }

    fn auth_req(method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
        let body = match body {
            Some(v) => Body::from(v.to_string()),
            None => Body::empty(),
        };
        Request::builder()
            .method(method)
            .uri(uri)
            .header("authorization", "Bearer admin-secret")
            .header("content-type", "application/json")
            .body(body)
            .unwrap()
    }

    async fn run(app: Router, req: Request<Body>) -> axum::http::Response<Body> {
        app.oneshot(req).await.unwrap()
    }

    async fn body_json(resp: axum::http::Response<Body>) -> Value {
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_reports_snapshot_counts() {
        let app = build_router(build_state());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["status"], "ok");
    }

    #[tokio::test]
    async fn create_model_returns_entry_with_generated_id() {
        let app = build_router(build_state());
        let resp = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("my-gpt4"))),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert!(!v["id"].as_str().unwrap().is_empty());
        assert_eq!(v["revision"], 1);
        assert_eq!(v["value"]["name"], "my-gpt4");
    }

    #[tokio::test]
    async fn create_model_without_auth_is_401() {
        let app = build_router(build_state());
        let req = Request::builder()
            .method("POST")
            .uri("/admin/v1/models")
            .header("content-type", "application/json")
            .body(Body::from(model_payload("m").to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let v = body_json(resp).await;
        // Spec §3 admin envelope — {"error_msg": "..."}.
        assert!(v["error_msg"].is_string());
        assert!(v.get("error").is_none());
    }

    #[tokio::test]
    async fn create_model_with_wrong_admin_key_is_401() {
        let app = build_router(build_state());
        let req = Request::builder()
            .method("POST")
            .uri("/admin/v1/models")
            .header("authorization", "Bearer wrong")
            .header("content-type", "application/json")
            .body(Body::from(model_payload("m").to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn create_model_with_invalid_provider_prefix_is_400_schema_error() {
        let app = build_router(build_state());
        let body = json!({
            "name": "x",
            "model": "mistral/large",
            "provider_config": {"api_key": "sk-x"}
        });
        let resp = run(app, auth_req("POST", "/admin/v1/models", Some(body))).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert!(v["error_msg"]
            .as_str()
            .unwrap()
            .contains("schema validation"));
    }

    #[tokio::test]
    async fn duplicate_model_name_on_create_is_409() {
        let state = build_state();
        let app = build_router(state.clone());
        let _ = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("dup"))),
        )
        .await;
        let app = build_router(state);
        let resp = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("dup"))),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn list_models_returns_created_entries() {
        let state = build_state();
        let app = build_router(state.clone());
        let _ = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("foo"))),
        )
        .await;
        let app = build_router(state.clone());
        let _ = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("bar"))),
        )
        .await;
        let app = build_router(state);
        let resp = run(app, auth_req("GET", "/admin/v1/models", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v.as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn get_model_round_trip() {
        let state = build_state();
        let app = build_router(state.clone());
        let resp = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("foo"))),
        )
        .await;
        let created = body_json(resp).await;
        let id = created["id"].as_str().unwrap();

        let app = build_router(state);
        let resp = run(
            app,
            auth_req("GET", &format!("/admin/v1/models/{id}"), None),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["value"]["name"], "foo");
    }

    #[tokio::test]
    async fn get_model_missing_is_404() {
        let app = build_router(build_state());
        let resp = run(app, auth_req("GET", "/admin/v1/models/nonexistent", None)).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn update_model_bumps_revision_and_persists_changes() {
        let state = build_state();
        let app = build_router(state.clone());
        let resp = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("foo"))),
        )
        .await;
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        // Change provider upstream.
        let updated_body = json!({
            "name": "foo",
            "model": "anthropic/claude-sonnet-4-5",
            "provider_config": {"api_key": "sk-ant"}
        });
        let app = build_router(state);
        let resp = run(
            app,
            auth_req("PUT", &format!("/admin/v1/models/{id}"), Some(updated_body)),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["revision"], 2);
        assert_eq!(v["value"]["model"], "anthropic/claude-sonnet-4-5");
    }

    #[tokio::test]
    async fn update_model_renaming_to_existing_name_is_409() {
        let state = build_state();
        let app = build_router(state.clone());
        let _ = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("foo"))),
        )
        .await;
        let app = build_router(state.clone());
        let resp = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("bar"))),
        )
        .await;
        let bar_id = body_json(resp).await["id"].as_str().unwrap().to_string();

        // Try to rename "bar" -> "foo".
        let app = build_router(state);
        let resp = run(
            app,
            auth_req(
                "PUT",
                &format!("/admin/v1/models/{bar_id}"),
                Some(model_payload("foo")),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn update_model_keeping_own_name_is_allowed() {
        let state = build_state();
        let app = build_router(state.clone());
        let resp = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("foo"))),
        )
        .await;
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        let app = build_router(state);
        let resp = run(
            app,
            auth_req(
                "PUT",
                &format!("/admin/v1/models/{id}"),
                Some(model_payload("foo")),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn delete_model_is_204_esque_and_subsequent_get_is_404() {
        let state = build_state();
        let app = build_router(state.clone());
        let resp = run(
            app,
            auth_req("POST", "/admin/v1/models", Some(model_payload("foo"))),
        )
        .await;
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        let app = build_router(state.clone());
        let resp = run(
            app,
            auth_req("DELETE", &format!("/admin/v1/models/{id}"), None),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);

        let app = build_router(state);
        let resp = run(
            app,
            auth_req("GET", &format!("/admin/v1/models/{id}"), None),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_missing_model_is_404() {
        let app = build_router(build_state());
        let resp = run(app, auth_req("DELETE", "/admin/v1/models/missing-id", None)).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn apikey_crud_follows_the_same_flow() {
        let state = build_state();

        // Create.
        let app = build_router(state.clone());
        let resp = run(
            app,
            auth_req(
                "POST",
                "/admin/v1/apikeys",
                Some(apikey_payload("sk-user-1", &["my-gpt4"])),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        // Duplicate key rejected.
        let app = build_router(state.clone());
        let resp = run(
            app,
            auth_req(
                "POST",
                "/admin/v1/apikeys",
                Some(apikey_payload("sk-user-1", &["*"])),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::CONFLICT);

        // List sees exactly one.
        let app = build_router(state.clone());
        let resp = run(app, auth_req("GET", "/admin/v1/apikeys", None)).await;
        assert_eq!(body_json(resp).await.as_array().unwrap().len(), 1);

        // Delete.
        let app = build_router(state);
        let resp = run(
            app,
            auth_req("DELETE", &format!("/admin/v1/apikeys/{id}"), None),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
