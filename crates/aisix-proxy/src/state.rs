//! Axum state shared across every proxy handler.
//!
//! `ProxyState` holds:
//! - the lock-free `SnapshotHandle<AisixSnapshot>` for looking up
//!   Models and ApiKeys on every request
//! - the `Hub` for resolving a `Provider` to the Bridge that serves it
//! - the per-key [`Limiter`] — queried before each upstream call and
//!   finalised after the response completes
//! - an `Arc<Metrics>` shared with the admin `/metrics` endpoint
//! - the [`CacheBackends`] consulted before bridge dispatch (None disables
//!   caching for that ProxyState; tests use this to keep the cache off
//!   the hot path when they don't care about it)
//! - the configured request-body size limit
//!
//! Cheap to clone: every field is either an `Arc` or a small Copy scalar.

use aisix_cache::{Cache, MemoryCache};
use aisix_core::models::CacheBackend;
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, ProxyConfig};
use aisix_gateway::Hub;
use aisix_guardrails::LiveGuardrailIndex;
use aisix_obs::{Metrics, OtlpHttpFanOut, UsageSink};
use aisix_ratelimit::Limiter;
use dashmap::DashSet;
use std::sync::Arc;

use crate::budget::BudgetClient;
use crate::client_ip::ResolvedRealIp;
use crate::health::{HealthTracker, LivezState, ModelRuntimeStatusTracker};
use crate::routing::RoutingRegistry;

/// The cache instances a DP deployment has available, selected per
/// request by the matched `CachePolicy.backend` (#519 B.8).
///
/// The memory cache is always built (in-process, no config needed);
/// the redis cache exists iff the boot config carries `cache.redis`.
/// A policy that asks for `redis` on a deployment without one gets NO
/// caching for its requests (`cache_status = disabled`) — never a
/// silent fallback to node-local memory, which would lie about the
/// sharing semantics the operator picked.
#[derive(Clone)]
pub struct CacheBackends {
    memory: Arc<dyn Cache>,
    redis: Option<Arc<dyn Cache>>,
    /// Policy ids already warned about an unavailable redis backend,
    /// so the gate logs once per policy instead of once per request.
    redis_warned: Arc<DashSet<String>>,
}

impl CacheBackends {
    pub fn new(memory: Arc<dyn Cache>, redis: Option<Arc<dyn Cache>>) -> Self {
        Self {
            memory,
            redis,
            redis_warned: Arc::new(DashSet::new()),
        }
    }

    /// Memory cache only — the default for self-hosted dev and tests.
    pub fn memory_only() -> Self {
        Self::new(Arc::new(MemoryCache::with_defaults()), None)
    }

    /// Resolve the cache instance for a matched policy's `backend`.
    ///
    /// `Memory` always resolves. `Redis` resolves iff the deployment
    /// configured one; otherwise caching is inactive for the request
    /// and we warn once per policy id.
    pub fn for_policy_backend(
        &self,
        backend: CacheBackend,
        policy_id: &str,
        policy_name: &str,
    ) -> Option<&Arc<dyn Cache>> {
        match backend {
            CacheBackend::Memory => Some(&self.memory),
            CacheBackend::Redis => {
                let redis = self.redis.as_ref();
                if redis.is_none() && self.redis_warned.insert(policy_id.to_string()) {
                    tracing::warn!(
                        target: "aisix::cache",
                        policy_id = %policy_id,
                        policy_name = %policy_name,
                        "cache policy requests backend=redis but this DP has no \
                         redis cache configured; caching is disabled for matching \
                         requests (set `cache.redis` in the gateway config)"
                    );
                }
                redis
            }
        }
    }
}

#[derive(Clone)]
pub struct ProxyState {
    pub snapshot: SnapshotHandle<AisixSnapshot>,
    pub hub: Arc<Hub>,
    pub limiter: Arc<Limiter>,
    pub metrics: Arc<Metrics>,
    pub cache: Option<CacheBackends>,
    pub routing: Arc<RoutingRegistry>,
    /// Per-request guardrail index. Resolves the applicable chain from
    /// attachment scope + priority on each request. Rebuilds lazily
    /// when the snapshot version changes. Default is an empty index
    /// (no-op); the server bootstrap wires a live handle at startup.
    pub guardrail_index: Arc<LiveGuardrailIndex>,
    /// Per-request budget gate. Asks cp-api whether the api_key may
    /// proceed; cached for 5s with sticky fallback on cp-api outage.
    pub budgets: Arc<BudgetClient>,
    /// Per-model health tracker. Updated on every upstream call outcome;
    /// read by `GET /admin/v1/health`.
    pub health: Arc<HealthTracker>,
    /// Public liveness state served on `GET /livez`.
    pub livez: Arc<LivezState>,
    /// Runtime model-status tracker keyed by resolved direct-model id.
    /// Used for request-path cooldown/background health exclusion and
    /// surfaced by `GET /admin/v1/models/status`.
    pub runtime_status: Arc<ModelRuntimeStatusTracker>,
    /// CP-side usage telemetry sink. Backed by an mpsc channel into the
    /// sender worker spawned in aisix-server (see `telemetry::spawn`).
    /// Defaults to a no-op sink when running outside managed mode so
    /// chat handlers don't have to special-case `Option`.
    pub usage_sink: UsageSink,
    /// Per-env OTLP/HTTP fan-out — POSTs one OTLP-encoded span per
    /// chat request to every enabled `ObservabilityExporter` in the
    /// snapshot. Cheap clonable handle holding a shared
    /// `reqwest::Client` connection pool. Always present (the
    /// no-exporters case = empty snapshot table = no spawned tasks).
    pub otlp_fan_out: OtlpHttpFanOut,
    pub request_body_limit_bytes: usize,
    /// Pre-parsed `proxy.real_ip` config for resolving the downstream
    /// client IP on each request (#492). Default = trust nothing → the
    /// logged source IP is the immediate TCP peer.
    pub real_ip: Arc<ResolvedRealIp>,
}

impl ProxyState {
    pub fn new(snapshot: SnapshotHandle<AisixSnapshot>, hub: Arc<Hub>, cfg: &ProxyConfig) -> Self {
        let guardrail_index = LiveGuardrailIndex::new(snapshot.clone(), None);
        Self {
            snapshot,
            hub,
            limiter: Arc::new(Limiter::new()),
            metrics: Arc::new(Metrics::new(false)),
            cache: Some(CacheBackends::memory_only()),
            routing: Arc::new(RoutingRegistry::new()),
            guardrail_index,
            budgets: Arc::new(BudgetClient::disabled()),
            health: Arc::new(HealthTracker::new()),
            livez: Arc::new(LivezState::new()),
            runtime_status: Arc::new(ModelRuntimeStatusTracker::new()),
            usage_sink: UsageSink::disabled(),
            otlp_fan_out: OtlpHttpFanOut::new(),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
            real_ip: Arc::new(ResolvedRealIp::from_config(&cfg.real_ip)),
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
        let guardrail_index = LiveGuardrailIndex::new(snapshot.clone(), None);
        Self {
            snapshot,
            hub,
            limiter,
            metrics: Arc::new(Metrics::new(false)),
            cache: Some(CacheBackends::memory_only()),
            routing: Arc::new(RoutingRegistry::new()),
            guardrail_index,
            budgets: Arc::new(BudgetClient::disabled()),
            health: Arc::new(HealthTracker::new()),
            livez: Arc::new(LivezState::new()),
            runtime_status: Arc::new(ModelRuntimeStatusTracker::new()),
            usage_sink: UsageSink::disabled(),
            otlp_fan_out: OtlpHttpFanOut::new(),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
            real_ip: Arc::new(ResolvedRealIp::from_config(&cfg.real_ip)),
        }
    }

    /// Full constructor used by the server bootstrap — lets the same
    /// Metrics handle be shared with the admin `/metrics` endpoint and
    /// lets the caller supply the configured cache backends.
    pub fn with_components(
        snapshot: SnapshotHandle<AisixSnapshot>,
        hub: Arc<Hub>,
        limiter: Arc<Limiter>,
        metrics: Arc<Metrics>,
        cache: Option<CacheBackends>,
        cfg: &ProxyConfig,
    ) -> Self {
        let guardrail_index = LiveGuardrailIndex::new(snapshot.clone(), None);
        Self {
            snapshot,
            hub,
            limiter,
            metrics,
            cache,
            routing: Arc::new(RoutingRegistry::new()),
            guardrail_index,
            budgets: Arc::new(BudgetClient::disabled()),
            health: Arc::new(HealthTracker::new()),
            livez: Arc::new(LivezState::new()),
            runtime_status: Arc::new(ModelRuntimeStatusTracker::new()),
            usage_sink: UsageSink::disabled(),
            otlp_fan_out: OtlpHttpFanOut::new(),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
            real_ip: Arc::new(ResolvedRealIp::from_config(&cfg.real_ip)),
        }
    }

    /// Disable caching on an existing state. Used by tests that need
    /// every request to reach wiremock.
    pub fn without_cache(mut self) -> Self {
        self.cache = None;
        self
    }

    /// Replace the guardrail index. Used by the server bootstrap to
    /// wire a live snapshot-backed index; tests can substitute a
    /// deterministic one via `LiveGuardrailIndex::new(stub_handle, None)`.
    pub fn with_guardrail_index(mut self, index: Arc<LiveGuardrailIndex>) -> Self {
        self.guardrail_index = index;
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
