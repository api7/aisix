//! aisix-proxy — client-facing proxy router (:3000).
//!
//! This crate will host the full `/v1/*` OpenAI-compatible surface. For
//! PR #5 only the startup-sequence building block is here: [`build_router`]
//! returns a minimal router with `/health` so the server binary can bind,
//! serve, and be exercised by the startup-sequence e2e test.
//!
//! The full routes, middleware stack, and streaming bridges land in their
//! own follow-up PRs.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, ProxyConfig};
use axum::{http::StatusCode, routing::get, Json, Router};
use serde_json::json;

/// Runtime state the proxy router hands to handlers. Cheap to clone
/// (contains only an `Arc`-backed `SnapshotHandle`).
#[derive(Clone)]
pub struct ProxyState {
    pub snapshot: SnapshotHandle<AisixSnapshot>,
    pub request_body_limit_bytes: usize,
}

impl ProxyState {
    pub fn new(snapshot: SnapshotHandle<AisixSnapshot>, cfg: &ProxyConfig) -> Self {
        Self {
            snapshot,
            request_body_limit_bytes: cfg.request_body_limit_bytes,
        }
    }
}

/// Build the proxy router. For PR #5 this is just `/health`; subsequent
/// PRs will mount `/v1/chat/completions`, `/v1/models`, etc.
pub fn build_router(state: ProxyState) -> Router {
    Router::new()
        .route("/health", get(health))
        .with_state(state)
}

async fn health(
    axum::extract::State(state): axum::extract::State<ProxyState>,
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
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt;

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1024,
            tls: None,
        }
    }

    #[tokio::test]
    async fn health_returns_snapshot_counts() {
        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let state = ProxyState::new(handle, &cfg());
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
        assert_eq!(v["models"], 0);
        assert_eq!(v["apikeys"], 0);
    }
}
