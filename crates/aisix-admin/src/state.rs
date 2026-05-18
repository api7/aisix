//! Shared axum state for every admin handler.
//!
//! Holds:
//! - the bootstrap-config-provided `admin_keys` (auth)
//! - the `ConfigStore` trait object (CRUD backend)
//! - a `SnapshotHandle` for authenticated operator-health handlers
//! - an optional `Metrics` handle — when present, `/metrics` renders
//!   the same Prometheus exposition that the proxy's middleware writes to
//!
//! The store is held behind an `Arc<dyn ConfigStore>` so production can
//! wire an etcd-backed impl and tests can use `InMemoryStore` via the
//! same type.

use aisix_core::config::PrometheusConfig;
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AdminConfig, AisixSnapshot};
use aisix_etcd::WatchStatus;
use aisix_obs::Metrics;
use aisix_proxy::{HealthTracker, LivezState, ModelRuntimeStatusTracker};
use axum::Router;
use std::sync::Arc;

use crate::store::ConfigStore;

#[derive(Clone)]
pub struct AdminState {
    pub snapshot: SnapshotHandle<AisixSnapshot>,
    pub admin_keys: Arc<[String]>,
    pub store: Arc<dyn ConfigStore>,
    pub metrics: Option<Arc<Metrics>>,
    pub prometheus: PrometheusConfig,
    /// Shared in-process health tracker from the proxy. Used by the
    /// `/admin/v1/health` endpoint to report per-model health status.
    pub health_tracker: Option<Arc<HealthTracker>>,
    /// Shared in-process runtime status tracker from the proxy. Used by
    /// `/admin/v1/models/status` to report direct-model runtime state.
    pub runtime_status_tracker: Option<Arc<ModelRuntimeStatusTracker>>,
    /// Watch supervisor's freshness state. When wired, the
    /// `/admin/v1/health` endpoint includes etcd revision +
    /// snapshot age so operators can detect a frozen / wedged config
    /// stream that would otherwise let the gateway serve a stale
    /// snapshot indefinitely. See issue #114.
    pub watch_status: Option<WatchStatus>,
    /// Shared liveness state for the public `GET /livez` endpoint.
    pub livez_state: Arc<LivezState>,
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
            prometheus: PrometheusConfig::default(),
            health_tracker: None,
            runtime_status_tracker: None,
            watch_status: None,
            livez_state: Arc::new(LivezState::new()),
            proxy_router: None,
        }
    }

    /// Attach the watch supervisor's freshness status. When set, the
    /// `/admin/v1/health` response includes etcd revision + snapshot
    /// age so operators can spot a wedged config stream.
    pub fn with_watch_status(mut self, status: WatchStatus) -> Self {
        self.watch_status = Some(status);
        self
    }

    /// Attach a metrics handle. Production wires the same handle that
    /// lives in `ProxyState` so a single call to `/metrics` reflects
    /// requests from both surfaces.
    pub fn with_metrics(mut self, metrics: Arc<Metrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn with_prometheus_config(mut self, prometheus: PrometheusConfig) -> Self {
        self.prometheus = prometheus;
        self
    }

    /// Attach the in-process health tracker from the proxy. When set,
    /// `GET /admin/v1/health` reflects per-model upstream health.
    pub fn with_health_tracker(mut self, tracker: Arc<HealthTracker>) -> Self {
        self.health_tracker = Some(tracker);
        self
    }

    /// Share the public liveness state with the proxy so both listeners
    /// report the same shutdown signal.
    pub fn with_livez_state(mut self, livez_state: Arc<LivezState>) -> Self {
        self.livez_state = livez_state;
        self
    }

    /// Attach the in-process runtime status tracker from the proxy.
    pub fn with_runtime_status_tracker(mut self, tracker: Arc<ModelRuntimeStatusTracker>) -> Self {
        self.runtime_status_tracker = Some(tracker);
        self
    }

    /// Wire the proxy router so the playground endpoint can forward
    /// requests to it in-process via `tower::ServiceExt::oneshot`.
    pub fn with_proxy_router(mut self, router: Router) -> Self {
        self.proxy_router = Some(router);
        self
    }
}
