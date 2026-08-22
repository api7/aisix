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

use aisix_cache::{Cache, MemoryCache, MemorySemanticCache, SemanticCacheStore};
use aisix_core::models::CacheBackend;
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, ProxyConfig};
use aisix_gateway::Hub;
use aisix_guardrails::LiveGuardrailIndex;
use aisix_obs::{ClientTypeClassifier, Metrics, OtlpHttpFanOut, UsageSink};
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
    /// Semantic (L2) store for `backend: memory` policies. Always
    /// built — in-process, no config needed, zero cost until a policy
    /// with a `semantic` block matches a request.
    semantic_memory: Arc<dyn SemanticCacheStore>,
    /// Semantic (L2) store for `backend: redis` policies. Wired by the
    /// bootstrap only when `cache.redis` is configured, is not cluster
    /// mode, AND the server passed the vector-search capability probe —
    /// so its absence here IS the degradation signal.
    semantic_redis: Option<Arc<dyn SemanticCacheStore>>,
    /// Policy ids already warned about an unavailable redis backend,
    /// so the gate logs once per policy instead of once per request.
    redis_warned: Arc<DashSet<String>>,
    /// Policy ids already warned about the redis semantic layer being
    /// unavailable (same warn-once discipline as `redis_warned`).
    semantic_redis_warned: Arc<DashSet<String>>,
    /// Policy ids already warned about a stable semantic config error
    /// (missing / non-embedding `embedding_model`). The per-request
    /// metric keeps counting; only the log line is deduplicated.
    semantic_resolve_warned: Arc<DashSet<String>>,
}

impl CacheBackends {
    pub fn new(memory: Arc<dyn Cache>, redis: Option<Arc<dyn Cache>>) -> Self {
        Self {
            memory,
            redis,
            semantic_memory: Arc::new(MemorySemanticCache::new()),
            semantic_redis: None,
            redis_warned: Arc::new(DashSet::new()),
            semantic_redis_warned: Arc::new(DashSet::new()),
            semantic_resolve_warned: Arc::new(DashSet::new()),
        }
    }

    /// Attach the shared semantic store for `backend: redis` policies.
    /// The bootstrap calls this only after the capability probe passed.
    pub fn with_semantic_redis(mut self, store: Arc<dyn SemanticCacheStore>) -> Self {
        self.semantic_redis = Some(store);
        self
    }

    /// True the FIRST time `policy_id` reports a stable semantic config
    /// error, so the gate logs once per policy instead of per request.
    pub fn semantic_resolve_warn_once(&self, policy_id: &str) -> bool {
        self.semantic_resolve_warned.insert(policy_id.to_string())
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

    /// Resolve the semantic (L2) store for a matched policy's
    /// `backend`. Same never-fall-back discipline as
    /// [`Self::for_policy_backend`]: a policy whose backend has no
    /// semantic store gets NO semantic matching (exact matching still
    /// works) rather than a silent per-node stand-in with different
    /// sharing semantics.
    pub fn semantic_for_policy_backend(
        &self,
        backend: CacheBackend,
        policy_id: &str,
        policy_name: &str,
    ) -> Option<&Arc<dyn SemanticCacheStore>> {
        match backend {
            CacheBackend::Memory => Some(&self.semantic_memory),
            CacheBackend::Redis => {
                let store = self.semantic_redis.as_ref();
                if store.is_none() && self.semantic_redis_warned.insert(policy_id.to_string()) {
                    tracing::warn!(
                        target: "aisix::cache",
                        policy_id = %policy_id,
                        policy_name = %policy_name,
                        "cache policy configures semantic matching on backend=redis but \
                         the configured cache.redis has no vector-search support \
                         (requires Redis 8+ or the search module; cluster mode is not \
                         supported yet); requests fall back to exact matching only"
                    );
                }
                store
            }
        }
    }
}

#[derive(Clone)]
pub struct ProxyState {
    // Axum clones state for several layers on every request. Keep the many
    // shared handles behind one refcount so each clone/drop uses one atomic.
    inner: Arc<ProxyStateInner>,
}

#[derive(Clone)]
pub struct ProxyStateInner {
    pub snapshot: SnapshotHandle<AisixSnapshot>,
    pub hub: Arc<Hub>,
    pub limiter: Arc<Limiter>,
    pub metrics: Arc<Metrics>,
    pub cache: Option<CacheBackends>,
    pub routing: Arc<RoutingRegistry>,
    /// Per-instance cache of semantic-router example embeddings, populated
    /// lazily on first use and reused across requests so semantic routing
    /// costs one embedding call (the prompt) in steady state.
    pub semantic_cache: Arc<crate::semantic::SemanticVectorCache>,
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
    /// Pre-parsed `proxy.request_id.accept_headers`: the inbound headers a
    /// caller may supply its own request id in, in priority order
    /// (AISIX-Cloud#1288). Default = `[x-aisix-request-id]`.
    pub request_id_accept: Arc<[axum::http::HeaderName]>,
    /// Boot-compiled `proxy.url_rewrites` rules, applied in order to every
    /// request before routing (first match wins). Empty = layer no-ops.
    pub url_rewrites: Arc<[crate::rewrite::CompiledRewrite]>,
    /// Optional config-freshness probe for `GET /readyz`: returns the time
    /// since the etcd watch last applied config (`None` = never applied).
    /// Wired from the watch supervisor in aisix-server; `None` here means
    /// no freshness signal, so readiness gates on shutdown only (#591).
    pub config_apply_age: Option<Arc<dyn Fn() -> Option<std::time::Duration> + Send + Sync>>,
    /// Batch ids whose completed output has already been attributed to
    /// UsageEvents by THIS process (#720). Process-local dedup only — the
    /// deterministic `request_id = "batch-<id>"` on the emitted events is
    /// what keeps cross-restart re-emission idempotent on the cp-api side.
    pub billed_batches: Arc<dashmap::DashSet<String>>,
    /// Boot-compiled User-Agent → `client_type` classifier: operator
    /// rules from `observability.metrics.client_type_rules` first, then
    /// the built-in allowlist (AISIX-Cloud#1045). Default = built-ins
    /// only; the server bootstrap swaps in the compiled config rules.
    pub client_classifier: Arc<ClientTypeClassifier>,
    /// Deployment-wide retry budget (`upstream.retries`) — the floor every
    /// dispatch falls back to when neither the target Model nor its model
    /// group sets one. See `routing::effective_retries`.
    pub default_retries: u32,
    /// Deployment-wide timeout defaults (`upstream.timeout_ms` /
    /// `upstream.stream_timeout_ms`) — the floor every dispatch falls back
    /// to when neither the target Model nor its model group sets one. See
    /// `routing::effective_timeouts`.
    pub default_timeouts: crate::routing::TimeoutDefaults,
}

impl std::ops::Deref for ProxyState {
    type Target = ProxyStateInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for ProxyState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.inner)
    }
}

/// Frozen `unix_secs` for unit-test limiters — an arbitrary mid-window
/// instant; the exact value only shapes reported retry-after seconds.
#[cfg(test)]
const TEST_RATE_LIMIT_CLOCK_SECS: u64 = 1_763_000_000;

impl ProxyState {
    pub fn new(snapshot: SnapshotHandle<AisixSnapshot>, hub: Arc<Hub>, cfg: &ProxyConfig) -> Self {
        let metrics = Arc::new(Metrics::new(false));
        let guardrail_index = LiveGuardrailIndex::new_with_sink(
            snapshot.clone(),
            None,
            Some(metrics.clone()),
            aisix_guardrails::SemanticRuntimeSlot::none(),
        );
        // Unit tests get a frozen rate-limit clock: on the wall clock, any
        // "the next request 429s" assertion silently races the fixed-window
        // minute boundary — a test that straddles :00 lands its two requests
        // in different windows and the 429 never comes (seen flaking in the
        // mcp per-server-limit tests under a loaded runner). Freezing the
        // clock puts every request of a test in one window by construction.
        // Only this crate's own test build is affected; other crates calling
        // `ProxyState::new` (e.g. aisix-admin's playground) compile the
        // system-clock arm.
        #[cfg(test)]
        let limiter = Arc::new(Limiter::local_with_clock(aisix_ratelimit::TestClock::new(
            TEST_RATE_LIMIT_CLOCK_SECS,
        )));
        #[cfg(not(test))]
        let limiter = Arc::new(Limiter::new());
        Self::from_inner(ProxyStateInner {
            snapshot,
            hub,
            limiter,
            metrics,
            cache: Some(CacheBackends::memory_only()),
            routing: Arc::new(RoutingRegistry::new()),
            semantic_cache: Arc::new(crate::semantic::SemanticVectorCache::default()),
            guardrail_index,
            budgets: Arc::new(BudgetClient::disabled()),
            health: Arc::new(HealthTracker::new()),
            livez: Arc::new(LivezState::new()),
            config_apply_age: None,
            runtime_status: Arc::new(ModelRuntimeStatusTracker::new()),
            usage_sink: UsageSink::disabled(),
            otlp_fan_out: OtlpHttpFanOut::new(),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
            real_ip: Arc::new(ResolvedRealIp::from_config(&cfg.real_ip)),
            request_id_accept: cfg
                .request_id
                .parse_accept_headers()
                .unwrap_or_default()
                .into(),
            url_rewrites: crate::rewrite::compile(&cfg.url_rewrites),
            billed_batches: Arc::new(dashmap::DashSet::new()),
            client_classifier: Arc::new(ClientTypeClassifier::builtin()),
            default_retries: aisix_core::config::DEFAULT_UPSTREAM_RETRIES,
            default_timeouts: crate::routing::TimeoutDefaults::default(),
        })
    }

    /// Alternative constructor for callers that want to share a preexisting
    /// limiter (e.g. tests with a deterministic clock).
    pub fn with_limiter(
        snapshot: SnapshotHandle<AisixSnapshot>,
        hub: Arc<Hub>,
        limiter: Arc<Limiter>,
        cfg: &ProxyConfig,
    ) -> Self {
        let metrics = Arc::new(Metrics::new(false));
        let guardrail_index = LiveGuardrailIndex::new_with_sink(
            snapshot.clone(),
            None,
            Some(metrics.clone()),
            aisix_guardrails::SemanticRuntimeSlot::none(),
        );
        Self::from_inner(ProxyStateInner {
            snapshot,
            hub,
            limiter,
            metrics,
            cache: Some(CacheBackends::memory_only()),
            routing: Arc::new(RoutingRegistry::new()),
            semantic_cache: Arc::new(crate::semantic::SemanticVectorCache::default()),
            guardrail_index,
            budgets: Arc::new(BudgetClient::disabled()),
            health: Arc::new(HealthTracker::new()),
            livez: Arc::new(LivezState::new()),
            config_apply_age: None,
            runtime_status: Arc::new(ModelRuntimeStatusTracker::new()),
            usage_sink: UsageSink::disabled(),
            otlp_fan_out: OtlpHttpFanOut::new(),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
            real_ip: Arc::new(ResolvedRealIp::from_config(&cfg.real_ip)),
            request_id_accept: cfg
                .request_id
                .parse_accept_headers()
                .unwrap_or_default()
                .into(),
            url_rewrites: crate::rewrite::compile(&cfg.url_rewrites),
            billed_batches: Arc::new(dashmap::DashSet::new()),
            client_classifier: Arc::new(ClientTypeClassifier::builtin()),
            default_retries: aisix_core::config::DEFAULT_UPSTREAM_RETRIES,
            default_timeouts: crate::routing::TimeoutDefaults::default(),
        })
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
        let guardrail_index = LiveGuardrailIndex::new_with_sink(
            snapshot.clone(),
            None,
            Some(metrics.clone()),
            aisix_guardrails::SemanticRuntimeSlot::none(),
        );
        // The bootstrap constructor is the one place the tracker gets a
        // metrics sink + snapshot handle, so cooldown transitions emit
        // `aisix_deployment_*`. Clone both before they are moved into the
        // struct below. Both trackers consult one shared BookkeepingFlags
        // so the "does any configured consumer read this?" answer can't
        // drift between them.
        let bookkeeping_flags = crate::health::BookkeepingFlags::new(snapshot.clone());
        let runtime_status = Arc::new(ModelRuntimeStatusTracker::with_observability(
            metrics.clone(),
            snapshot.clone(),
            Arc::clone(&bookkeeping_flags),
        ));
        Self::from_inner(ProxyStateInner {
            snapshot,
            hub,
            limiter,
            metrics,
            cache,
            routing: Arc::new(RoutingRegistry::new()),
            semantic_cache: Arc::new(crate::semantic::SemanticVectorCache::default()),
            guardrail_index,
            budgets: Arc::new(BudgetClient::disabled()),
            health: Arc::new(HealthTracker::with_flags(bookkeeping_flags)),
            livez: Arc::new(LivezState::new()),
            config_apply_age: None,
            runtime_status,
            usage_sink: UsageSink::disabled(),
            otlp_fan_out: OtlpHttpFanOut::new(),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
            real_ip: Arc::new(ResolvedRealIp::from_config(&cfg.real_ip)),
            request_id_accept: cfg
                .request_id
                .parse_accept_headers()
                .unwrap_or_default()
                .into(),
            url_rewrites: crate::rewrite::compile(&cfg.url_rewrites),
            billed_batches: Arc::new(dashmap::DashSet::new()),
            client_classifier: Arc::new(ClientTypeClassifier::builtin()),
            default_retries: aisix_core::config::DEFAULT_UPSTREAM_RETRIES,
            default_timeouts: crate::routing::TimeoutDefaults::default(),
        })
    }

    fn from_inner(inner: ProxyStateInner) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Disable caching on an existing state. Used by tests that need
    /// every request to reach wiremock.
    pub fn without_cache(mut self) -> Self {
        Arc::make_mut(&mut self.inner).cache = None;
        self
    }

    /// Replace the guardrail index. Used by the server bootstrap to
    /// wire a live snapshot-backed index; tests can substitute a
    /// deterministic one via `LiveGuardrailIndex::new(stub_handle, None)`.
    pub fn with_guardrail_index(mut self, index: Arc<LiveGuardrailIndex>) -> Self {
        Arc::make_mut(&mut self.inner).guardrail_index = index;
        self
    }

    /// Swap in the classifier compiled from
    /// `observability.metrics.client_type_rules` (AISIX-Cloud#1045).
    /// Default is built-ins only.
    pub fn with_client_classifier(mut self, classifier: Arc<ClientTypeClassifier>) -> Self {
        Arc::make_mut(&mut self.inner).client_classifier = classifier;
        self
    }

    /// Apply the deployment-wide `upstream.retries` budget. Default is
    /// [`aisix_core::config::DEFAULT_UPSTREAM_RETRIES`].
    pub fn with_default_retries(mut self, retries: u32) -> Self {
        Arc::make_mut(&mut self.inner).default_retries = retries;
        self
    }

    /// Apply the deployment-wide `upstream.timeout_ms` /
    /// `upstream.stream_timeout_ms` defaults, with `0` meaning "no
    /// default at that slot".
    pub fn with_default_timeouts(mut self, timeout_ms: u64, stream_timeout_ms: u64) -> Self {
        Arc::make_mut(&mut self.inner).default_timeouts = crate::routing::TimeoutDefaults {
            request: (timeout_ms > 0).then(|| std::time::Duration::from_millis(timeout_ms)),
            stream: (stream_timeout_ms > 0)
                .then(|| std::time::Duration::from_millis(stream_timeout_ms)),
        };
        self
    }

    /// Attach a CP-side usage telemetry sink. Default is `disabled()`;
    /// the server bootstrap calls this in managed mode after spawning
    /// the sender worker.
    pub fn with_usage_sink(mut self, sink: UsageSink) -> Self {
        Arc::make_mut(&mut self.inner).usage_sink = sink;
        self
    }

    /// Swap in a live `BudgetClient` that talks to cp-api. Default is
    /// the disabled (allow-all) client used in self-hosted dev.
    pub fn with_budget_client(mut self, client: Arc<BudgetClient>) -> Self {
        Arc::make_mut(&mut self.inner).budgets = client;
        self
    }

    /// Wire the config-freshness probe used by `GET /readyz` (#591). The
    /// closure returns the time since the etcd watch last applied config.
    pub fn with_config_apply_age(
        mut self,
        probe: Arc<dyn Fn() -> Option<std::time::Duration> + Send + Sync>,
    ) -> Self {
        Arc::make_mut(&mut self.inner).config_apply_age = Some(probe);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::ProxyState;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ProxyConfig};
    use aisix_gateway::Hub;
    use std::sync::Arc;

    fn test_state() -> ProxyState {
        ProxyState::new(
            SnapshotHandle::new(AisixSnapshot::new()),
            Arc::new(Hub::new()),
            &ProxyConfig {
                addr: "127.0.0.1:0".into(),
                request_body_limit_bytes: 1_048_576,
                tls: None,
                real_ip: Default::default(),
                request_id: Default::default(),
                thread_per_core: None,
                workers: None,
                url_rewrites: Vec::new(),
            },
        )
    }

    #[test]
    fn proxy_state_clone_shares_one_inner_and_mutation_is_copy_on_write() {
        assert_eq!(
            std::mem::size_of::<ProxyState>(),
            std::mem::size_of::<Arc<()>>()
        );

        let original = test_state();
        assert_eq!(Arc::strong_count(&original.inner), 1);

        let mut cloned = original.clone();
        assert!(Arc::ptr_eq(&original.inner, &cloned.inner));
        assert_eq!(Arc::strong_count(&original.inner), 2);

        cloned.cache = None;
        assert!(!Arc::ptr_eq(&original.inner, &cloned.inner));
        assert!(original.cache.is_some());
        assert!(cloned.cache.is_none());

        let configured = original.clone().with_default_retries(99);
        assert!(!Arc::ptr_eq(&original.inner, &configured.inner));
        assert_ne!(original.default_retries, 99);
        assert_eq!(configured.default_retries, 99);
    }
}
