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
use std::sync::Arc;

use crate::store::ConfigStore;

#[derive(Clone)]
pub struct AdminState {
    pub snapshot: SnapshotHandle<AisixSnapshot>,
    pub admin_keys: Arc<[String]>,
    pub store: Arc<dyn ConfigStore>,
    pub metrics: Option<Arc<Metrics>>,
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
        }
    }

    /// Attach a metrics handle. Production wires the same handle that
    /// lives in `ProxyState` so a single call to `/metrics` reflects
    /// requests from both surfaces.
    pub fn with_metrics(mut self, metrics: Arc<Metrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }
}
