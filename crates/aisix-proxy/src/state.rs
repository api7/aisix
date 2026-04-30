//! Axum state shared across every proxy handler.
//!
//! `ProxyState` holds:
//! - the lock-free `SnapshotHandle<AisixSnapshot>` for looking up
//!   Models and ApiKeys on every request
//! - the `Hub` for resolving a `Provider` to the Bridge that serves it
//! - the per-key [`Limiter`] — queried before each upstream call and
//!   finalised after the response completes
//! - an `Arc<Metrics>` shared with the admin `/metrics` endpoint
//! - an `Arc<dyn Cache>` consulted before bridge dispatch (None disables
//!   caching for that ProxyState; tests use this to keep the cache off
//!   the hot path when they don't care about it)
//! - the configured request-body size limit
//!
//! Cheap to clone: every field is either an `Arc` or a small Copy scalar.

use aisix_cache::{Cache, MemoryCache};
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, ProxyConfig};
use aisix_gateway::Hub;
use aisix_guardrails::{Guardrail, GuardrailChain};
use aisix_obs::{LangfuseSender, Metrics, UsageSink};
use aisix_ratelimit::Limiter;
use std::sync::Arc;

use crate::budget::BudgetClient;
use crate::health::HealthTracker;
use crate::routing::RoutingRegistry;

#[derive(Clone)]
pub struct ProxyState {
    pub snapshot: SnapshotHandle<AisixSnapshot>,
    pub hub: Arc<Hub>,
    pub limiter: Arc<Limiter>,
    pub metrics: Arc<Metrics>,
    pub cache: Option<Arc<dyn Cache>>,
    pub routing: Arc<RoutingRegistry>,
    /// Content-policy hooks. Default is an empty chain (no-op); the
    /// server bootstrap loads a real chain from config.
    pub guardrails: Arc<dyn Guardrail>,
    /// Per-request budget gate. Asks cp-api whether the api_key may
    /// proceed; cached for 5s with sticky fallback on cp-api outage.
    pub budgets: Arc<BudgetClient>,
    /// Per-model health tracker. Updated on every upstream call outcome;
    /// read by `GET /admin/v1/health`.
    pub health: Arc<HealthTracker>,
    /// Optional Langfuse exporter. When `Some`, chat handlers emit one
    /// generation event at end-of-request. `None` disables emission.
    pub langfuse: Option<Arc<LangfuseSender>>,
    /// CP-side usage telemetry sink. Backed by an mpsc channel into the
    /// sender worker spawned in aisix-server (see `telemetry::spawn`).
    /// Defaults to a no-op sink when running outside managed mode so
    /// chat handlers don't have to special-case `Option`.
    pub usage_sink: UsageSink,
    pub request_body_limit_bytes: usize,
}

impl ProxyState {
    pub fn new(snapshot: SnapshotHandle<AisixSnapshot>, hub: Arc<Hub>, cfg: &ProxyConfig) -> Self {
        Self {
            snapshot,
            hub,
            limiter: Arc::new(Limiter::new()),
            metrics: Arc::new(Metrics::new(false)),
            cache: Some(Arc::new(MemoryCache::with_defaults())),
            routing: Arc::new(RoutingRegistry::new()),
            guardrails: Arc::new(GuardrailChain::empty()),
            budgets: Arc::new(BudgetClient::disabled()),
            health: Arc::new(HealthTracker::new()),
            langfuse: None,
            usage_sink: UsageSink::disabled(),
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
            cache: Some(Arc::new(MemoryCache::with_defaults())),
            routing: Arc::new(RoutingRegistry::new()),
            guardrails: Arc::new(GuardrailChain::empty()),
            budgets: Arc::new(BudgetClient::disabled()),
            health: Arc::new(HealthTracker::new()),
            langfuse: None,
            usage_sink: UsageSink::disabled(),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
        }
    }

    /// Full constructor used by the server bootstrap — lets the same
    /// Metrics handle be shared with the admin `/metrics` endpoint and
    /// lets the caller supply a configured Cache backend.
    pub fn with_components(
        snapshot: SnapshotHandle<AisixSnapshot>,
        hub: Arc<Hub>,
        limiter: Arc<Limiter>,
        metrics: Arc<Metrics>,
        cache: Option<Arc<dyn Cache>>,
        cfg: &ProxyConfig,
    ) -> Self {
        Self {
            snapshot,
            hub,
            limiter,
            metrics,
            cache,
            routing: Arc::new(RoutingRegistry::new()),
            guardrails: Arc::new(GuardrailChain::empty()),
            budgets: Arc::new(BudgetClient::disabled()),
            health: Arc::new(HealthTracker::new()),
            langfuse: None,
            usage_sink: UsageSink::disabled(),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
        }
    }

    /// Disable caching on an existing state. Used by tests that need
    /// every request to reach wiremock.
    pub fn without_cache(mut self) -> Self {
        self.cache = None;
        self
    }

    /// Replace the guardrail chain. Used by both the server bootstrap
    /// and tests that want a deterministic policy.
    pub fn with_guardrails(mut self, guardrails: Arc<dyn Guardrail>) -> Self {
        self.guardrails = guardrails;
        self
    }

    /// Attach a Langfuse sender. The exporter is opt-in; when absent
    /// no events are emitted regardless of request volume.
    pub fn with_langfuse(mut self, sender: Arc<LangfuseSender>) -> Self {
        self.langfuse = Some(sender);
        self
    }

    /// Attach a CP-side usage telemetry sink. Default is `disabled()`;
    /// the server bootstrap calls this in managed mode after spawning
    /// the sender worker.
    pub fn with_usage_sink(mut self, sink: UsageSink) -> Self {
        self.usage_sink = sink;
        self
    }

    /// Swap in a live `BudgetClient` that talks to cp-api. Default is
    /// the disabled (allow-all) client used in self-hosted dev.
    pub fn with_budget_client(mut self, client: Arc<BudgetClient>) -> Self {
        self.budgets = client;
        self
    }
}
