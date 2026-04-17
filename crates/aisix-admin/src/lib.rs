//! aisix-admin — Admin API + Playground + embedded UI (:3001).
//!
//! PR #5 provides only the startup-sequence building block: [`build_router`]
//! mounts `/health` so the server binary can bind and serve its admin
//! listener. The full CRUD surface for Models / ApiKeys, the OpenAPI
//! scalar, and the embedded UI are implemented in follow-up PRs.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AdminConfig, AisixSnapshot};
use axum::{http::StatusCode, routing::get, Json, Router};
use serde_json::json;
use std::sync::Arc;

/// Runtime state shared across admin handlers.
#[derive(Clone)]
pub struct AdminState {
    pub snapshot: SnapshotHandle<AisixSnapshot>,
    /// Admin keys are held as an Arc<[String]> so cloning the state is cheap.
    pub admin_keys: Arc<[String]>,
}

impl AdminState {
    pub fn new(snapshot: SnapshotHandle<AisixSnapshot>, cfg: &AdminConfig) -> Self {
        Self {
            snapshot,
            admin_keys: Arc::from(cfg.admin_keys.clone()),
        }
    }
}

/// Build the admin router. For PR #5 this is just `/health`.
pub fn build_router(state: AdminState) -> Router {
    Router::new()
        .route("/health", get(health))
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
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt;

    fn cfg() -> AdminConfig {
        AdminConfig {
            addr: "127.0.0.1:0".into(),
            admin_keys: vec!["k1".into()],
            tls: None,
        }
    }

    #[tokio::test]
    async fn health_returns_ok_and_counts() {
        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let state = AdminState::new(handle, &cfg());
        let app = build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["status"], "ok");
    }

    #[test]
    fn admin_state_clones_admin_keys_into_arc() {
        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let state = AdminState::new(handle, &cfg());
        let b = state.clone();
        assert_eq!(b.admin_keys.len(), 1);
        assert_eq!(&*b.admin_keys[0], "k1");
    }
}
