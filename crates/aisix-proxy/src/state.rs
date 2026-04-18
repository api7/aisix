//! Axum state shared across every proxy handler.
//!
//! `ProxyState` holds:
//! - the lock-free `SnapshotHandle<AisixSnapshot>` for looking up
//!   Models and ApiKeys on every request
//! - the `Hub` for resolving a `Provider` to the Bridge that serves it
//! - the per-key [`Limiter`] — queried before each upstream call and
//!   finalised after the response completes
//! - the configured request-body size limit
//!
//! Cheap to clone: every field is either an `Arc` or a small Copy scalar.

use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, ProxyConfig};
use aisix_gateway::Hub;
use aisix_obs::Metrics;
use aisix_ratelimit::Limiter;
use std::sync::Arc;

#[derive(Clone)]
pub struct ProxyState {
    pub snapshot: SnapshotHandle<AisixSnapshot>,
    pub hub: Arc<Hub>,
    pub limiter: Arc<Limiter>,
    pub metrics: Arc<Metrics>,
    pub request_body_limit_bytes: usize,
}

impl ProxyState {
    pub fn new(snapshot: SnapshotHandle<AisixSnapshot>, hub: Arc<Hub>, cfg: &ProxyConfig) -> Self {
        Self {
            snapshot,
            hub,
            limiter: Arc::new(Limiter::new()),
            metrics: Arc::new(Metrics::new(false)),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
        }
    }

    /// Alternative constructor for callers that want to share a preexisting
    /// limiter (e.g. tests with a deterministic clock).
    pub fn with_limiter(
        snapshot: SnapshotHandle<AisixSnapshot>,
        hub: Arc<Hub>,
        limiter: Arc<Limiter>,
        cfg: &ProxyConfig,
    ) -> Self {
        Self {
            snapshot,
            hub,
            limiter,
            metrics: Arc::new(Metrics::new(false)),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
        }
    }

    /// Full constructor used by the server bootstrap — lets the same
    /// Metrics handle be shared with the admin `/metrics` endpoint.
    pub fn with_components(
        snapshot: SnapshotHandle<AisixSnapshot>,
        hub: Arc<Hub>,
        limiter: Arc<Limiter>,
        metrics: Arc<Metrics>,
        cfg: &ProxyConfig,
    ) -> Self {
        Self {
            snapshot,
            hub,
            limiter,
            metrics,
            request_body_limit_bytes: cfg.request_body_limit_bytes,
        }
    }
}
