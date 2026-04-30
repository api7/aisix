//! Shared axum state for every admin handler.
//!
//! Holds:
//! - the bootstrap-config-provided `admin_keys` (auth)
//! - the `ConfigStore` trait object (CRUD backend)
//! - a `SnapshotHandle` for the /health endpoint (snapshot counts)
//! - an optional `Metrics` handle — when present, `/metrics` renders
//!   the same Prometheus exposition that the proxy's middleware writes to
//!
//! The store is held behind an `Arc<dyn ConfigStore>` so production can
//! wire an etcd-backed impl and tests can use `InMemoryStore` via the
//! same type.

use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AdminConfig, AisixSnapshot};
use aisix_obs::Metrics;
use aisix_proxy::HealthTracker;
use axum::Router;
use std::sync::Arc;

use crate::store::ConfigStore;

#[derive(Clone)]
pub struct AdminState {
    pub snapshot: SnapshotHandle<AisixSnapshot>,
    pub admin_keys: Arc<[String]>,
    pub store: Arc<dyn ConfigStore>,
    pub metrics: Option<Arc<Metrics>>,
    /// Shared in-process health tracker from the proxy. Used by the
    /// `/admin/v1/health` endpoint to report per-model health status.
    pub health_tracker: Option<Arc<HealthTracker>>,
    /// Proxy router shared for the `/playground/chat/completions` endpoint.
    /// The playground handler calls `router.oneshot(req)` so the request
    /// goes through the full proxy middleware stack (auth, rate-limit, bridge)
    /// without an additional network hop.
    pub proxy_router: Option<Router>,
}

impl AdminState {
    pub fn new(
        snapshot: SnapshotHandle<AisixSnapshot>,
        store: Arc<dyn ConfigStore>,
        cfg: &AdminConfig,
    ) -> Self {
        Self {
            snapshot,
            admin_keys: Arc::from(cfg.admin_keys.clone()),
            store,
            metrics: None,
            health_tracker: None,
            proxy_router: None,
        }
    }

    /// Attach a metrics handle. Production wires the same handle that
    /// lives in `ProxyState` so a single call to `/metrics` reflects
    /// requests from both surfaces.
    pub fn with_metrics(mut self, metrics: Arc<Metrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Attach the in-process health tracker from the proxy. When set,
    /// `GET /admin/v1/health` reflects per-model upstream health.
    pub fn with_health_tracker(mut self, tracker: Arc<HealthTracker>) -> Self {
        self.health_tracker = Some(tracker);
        self
    }

    /// Wire the proxy router so the playground endpoint can forward
    /// requests to it in-process via `tower::ServiceExt::oneshot`.
    pub fn with_proxy_router(mut self, router: Router) -> Self {
        self.proxy_router = Some(router);
        self
    }
}
