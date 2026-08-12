//! Per-model health tracking for the admin `/admin/v1/health` endpoint.
//!
//! Tracks consecutive upstream failures per model name. The state machine
//! progresses as follows:
//!
//! ```text
//!  Healthy (0) ──[4+ failures]──► Degraded (1) ──[8+ failures]──► Down (2)
//!     ▲                               │                               │
//!     └─────────[any success]─────────┴───────────────────────────────┘
//! ```
//!
//! Thresholds are conservative — a temporary blip doesn't flip a model to
//! Down. Operators can query the health endpoint to see which models are
//! under stress without waiting for a full outage.

use dashmap::DashMap;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, RoutingStrategy};
use aisix_obs::{DeploymentLabels, DeploymentState, Metrics};
use axum::http::header::{HeaderName, HeaderValue, CONTENT_TYPE};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

static X_CONTENT_TYPE_OPTIONS: HeaderName = HeaderName::from_static("x-content-type-options");
static NOSNIFF: HeaderValue = HeaderValue::from_static("nosniff");
static TEXT_PLAIN_UTF8: HeaderValue = HeaderValue::from_static("text/plain; charset=utf-8");

#[derive(Debug, Default)]
pub struct LivezState {
    shutting_down: AtomicBool,
}

impl LivezState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_shutting_down(&self) {
        self.shutting_down.store(true, Ordering::Relaxed);
    }

    fn shutdown_check(&self) -> Result<(), &'static str> {
        if self.shutting_down.load(Ordering::Relaxed) {
            Err("process is shutting down")
        } else {
            Ok(())
        }
    }

    /// Whether graceful shutdown has been signalled. Used by `/readyz` to
    /// drain traffic before the process exits.
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Relaxed)
    }
}

/// Decide whether config availability blocks readiness. `last_apply_age` is
/// the time since the config watch last applied an event; `None` means no
/// apply yet — still starting up, or restarted with no usable snapshot
/// cache. Returns `Some(reason)` when the instance cannot serve, `None`
/// when it can.
///
/// The *age* is deliberately not an input. It measures time since the last
/// config **event**, and an environment whose resources are not changing
/// produces none — so a threshold on it reports a healthy gateway as
/// unready once the environment goes quiet, which for most deployments is
/// the steady state. `/readyz` previously blocked past 300s and took every
/// replica out of its Service five minutes after a deployment went idle.
///
/// Nor is readiness the right lever for the case that threshold was aimed
/// at. A genuinely wedged watch is a property of the config source, so it
/// hits every replica at once: withdrawing them all converts "serving the
/// last accepted config" into a total outage, with no healthy instance to
/// shift traffic to. Config freshness stays observable — and alertable —
/// on `/status/config` and the `aisix_config_*` metrics.
pub fn config_readiness_block(last_apply_age: Option<Duration>) -> Option<&'static str> {
    match last_apply_age {
        None => Some("config not yet applied"),
        Some(_) => None,
    }
}

pub fn livez_response(livez: &LivezState, verbose: bool) -> Response {
    let mut body = String::new();
    let mut failed = false;

    body.push_str("[+]ping ok\n");
    match livez.shutdown_check() {
        Ok(()) => body.push_str("[+]shutdown ok\n"),
        Err(_) => {
            failed = true;
            body.push_str("[-]shutdown failed: reason withheld\n");
        }
    }

    let headers = [
        (CONTENT_TYPE, TEXT_PLAIN_UTF8.clone()),
        (X_CONTENT_TYPE_OPTIONS.clone(), NOSNIFF.clone()),
    ];

    if failed {
        // Graceful shutdown is an expected drain, not an internal error —
        // 503 so Kubernetes stops routing without treating it as a crash
        // loop (#591).
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            headers,
            format!("{body}livez check failed"),
        )
            .into_response();
    }

    if !verbose {
        return (StatusCode::OK, headers, "ok").into_response();
    }

    (
        StatusCode::OK,
        headers,
        format!("{body}livez check passed\n"),
    )
        .into_response()
}

/// `GET /readyz` — readiness (traffic eligibility), distinct from `/livez`
/// (process liveness). Returns 503 while draining (graceful shutdown) or
/// while config isn't fresh (still starting up, or a wedged watch), so
/// Kubernetes keeps the instance out of the Service endpoints until it can
/// actually serve. `config_block` is the result of
/// [`config_readiness_block`]; pass `None` when no freshness signal is
/// wired (readiness then gates on shutdown only).
pub fn readyz_response(
    livez: &LivezState,
    config_block: Option<&'static str>,
    verbose: bool,
) -> Response {
    let mut body = String::new();
    let mut failed = false;

    match livez.shutdown_check() {
        Ok(()) => body.push_str("[+]shutdown ok\n"),
        Err(_) => {
            failed = true;
            body.push_str("[-]shutdown failed: draining\n");
        }
    }
    match config_block {
        None => body.push_str("[+]config ok\n"),
        Some(_) => {
            failed = true;
            body.push_str("[-]config failed: not ready\n");
        }
    }

    let headers = [
        (CONTENT_TYPE, TEXT_PLAIN_UTF8.clone()),
        (X_CONTENT_TYPE_OPTIONS.clone(), NOSNIFF.clone()),
    ];

    if failed {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            headers,
            format!("{body}readyz check failed"),
        )
            .into_response();
    }

    if !verbose {
        return (StatusCode::OK, headers, "ok").into_response();
    }

    (
        StatusCode::OK,
        headers,
        format!("{body}readyz check passed\n"),
    )
        .into_response()
}

/// Numeric health level reported by the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(into = "u8")]
pub enum HealthLevel {
    /// No recent failures — serving normally.
    Healthy,
    /// Between `DEGRADED_THRESHOLD` and `DOWN_THRESHOLD` consecutive failures.
    Degraded,
    /// At or beyond `DOWN_THRESHOLD` consecutive failures.
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatus {
    Healthy,
    Unhealthy,
    Cooldown,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeStatusSnapshot {
    pub status: RuntimeStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cooldown_until: Option<SystemTime>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_checked_at: Option<SystemTime>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_check_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_reason: Option<String>,
}

impl Default for RuntimeStatusSnapshot {
    fn default() -> Self {
        Self {
            status: RuntimeStatus::Healthy,
            cooldown_until: None,
            last_checked_at: None,
            last_check_status: None,
            status_reason: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct RuntimeEntry {
    unhealthy: bool,
    cooldown_until: Option<SystemTime>,
    last_checked_at: Option<SystemTime>,
    last_check_status: Option<u16>,
    status_reason: Option<String>,
    /// Exponentially-weighted moving average of recent observed upstream
    /// latency in milliseconds. `None` until the first sample. Drives the
    /// `least_latency` routing strategy; independent of health/cooldown.
    latency_ewma_ms: Option<f64>,
    /// Number of requests currently in flight to this target. Held in an
    /// `Arc` so an [`InFlightGuard`] can decrement it after the DashMap lock
    /// is released (and for the streaming path, after the handler returns).
    /// Drives the `least_busy` routing strategy.
    in_flight: Arc<AtomicUsize>,
    /// Last value published to the `aisix_deployment_state` gauge for this
    /// target, so [`ModelRuntimeStatusTracker::sync_deployment_state`] can
    /// skip a write when nothing changed. `None` = never published.
    emitted_state: Option<DeploymentState>,
}

impl RuntimeEntry {
    /// Serving state as the router sees it: a target that is cooling down,
    /// or that its background check has marked unhealthy, is out of
    /// rotation ([`ModelRuntimeStatusTracker::should_skip_for_routing`]).
    ///
    /// The gauge is derived from this — never from "did we just observe a
    /// transition". A cooldown lapses on its own with nothing calling back
    /// into the tracker, so an edge-triggered gauge misses the recovery and
    /// pins the target at Down forever.
    fn deployment_state(&self, now: SystemTime) -> DeploymentState {
        if self.unhealthy || self.cooldown_until.is_some_and(|until| until > now) {
            DeploymentState::Down
        } else {
            DeploymentState::Healthy
        }
    }

    fn snapshot(&self, now: SystemTime, stale_after: Option<Duration>) -> RuntimeStatusSnapshot {
        let cooldown_until = self.cooldown_until.filter(|until| *until > now);
        let unhealthy = self.unhealthy && !self.is_stale(now, stale_after);
        let status = if cooldown_until.is_some() {
            RuntimeStatus::Cooldown
        } else if unhealthy {
            RuntimeStatus::Unhealthy
        } else {
            RuntimeStatus::Healthy
        };
        RuntimeStatusSnapshot {
            status,
            cooldown_until,
            last_checked_at: self.last_checked_at,
            last_check_status: self.last_check_status,
            status_reason: self.status_reason.clone(),
        }
    }

    fn is_stale(&self, now: SystemTime, stale_after: Option<Duration>) -> bool {
        let Some(stale_after) = stale_after else {
            return false;
        };
        let Some(last_checked_at) = self.last_checked_at else {
            return false;
        };
        match now.duration_since(last_checked_at) {
            Ok(elapsed) => elapsed > stale_after,
            Err(_) => false,
        }
    }
}

impl From<HealthLevel> for u8 {
    fn from(h: HealthLevel) -> u8 {
        match h {
            HealthLevel::Healthy => 0,
            HealthLevel::Degraded => 1,
            HealthLevel::Down => 2,
        }
    }
}

/// Consecutive failures required to enter Degraded.
const DEGRADED_THRESHOLD: u32 = 4;
/// Consecutive failures required to enter Down.
const DOWN_THRESHOLD: u32 = 8;

struct Entry {
    consecutive_failures: AtomicU32,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
        }
    }
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry")
            .field(
                "consecutive_failures",
                &self.consecutive_failures.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Entry {
    fn level(&self) -> HealthLevel {
        let n = self.consecutive_failures.load(Ordering::Relaxed);
        if n >= DOWN_THRESHOLD {
            HealthLevel::Down
        } else if n >= DEGRADED_THRESHOLD {
            HealthLevel::Degraded
        } else {
            HealthLevel::Healthy
        }
    }

    fn on_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    fn on_failure(&self) {
        // Cap at DOWN_THRESHOLD + 1 so the counter doesn't overflow on long
        // outages while still being distinguishable from a down-threshold hit.
        let prev = self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        if prev > DOWN_THRESHOLD {
            self.consecutive_failures
                .store(DOWN_THRESHOLD + 1, Ordering::Relaxed);
        }
    }
}

/// Version-gated answer to "does any configured consumer depend on the
/// exact per-request bookkeeping write path?"
///
/// Three predicates, all derived from the live snapshot:
/// - a Model routes with `least_busy` (reads the in-flight counters,
///   `crate::routing::order_attempts_by_metric`)
/// - a Model routes with `least_latency` (reads the latency EWMA)
/// - a Model has background health checks enabled (the background
///   checker and the health surface observe tracker state)
///
/// While **any** predicate holds, every bookkeeping method below runs its
/// historical write path unchanged, so configured deployments keep
/// byte-identical behavior. Only when none holds do the trackers take the
/// cheap read-first paths — writes whose consumers provably don't exist.
///
/// The predicate set is recomputed at most once per snapshot version
/// (packed with the version into one atomic so the pair can never be
/// observed torn). A racing store between the version read and the table
/// walk can cache bits against a stale version; the next call detects the
/// mismatch and recomputes, so the value converges immediately.
#[derive(Debug)]
pub struct BookkeepingFlags {
    snapshot: SnapshotHandle<AisixSnapshot>,
    /// `(snapshot version << 3) | predicate bits`, or [`UNCOMPUTED`].
    packed: AtomicU64,
}

const FLAG_LEAST_BUSY: u64 = 1;
const FLAG_LEAST_LATENCY: u64 = 1 << 1;
const FLAG_HEALTH_CHECKS: u64 = 1 << 2;
const FLAG_BITS: u64 = 0b111;
const UNCOMPUTED: u64 = u64::MAX;

impl BookkeepingFlags {
    pub fn new(snapshot: SnapshotHandle<AisixSnapshot>) -> Arc<Self> {
        Arc::new(Self {
            snapshot,
            packed: AtomicU64::new(UNCOMPUTED),
        })
    }

    /// True when any predicate holds — the trackers then use their
    /// historical write paths.
    pub fn any_active(&self) -> bool {
        self.bits() != 0
    }

    fn bits(&self) -> u64 {
        let ver = self.snapshot.version();
        let packed = self.packed.load(Ordering::Relaxed);
        if packed != UNCOMPUTED && packed >> 3 == ver {
            return packed & FLAG_BITS;
        }
        let snap = self.snapshot.load();
        let mut bits = 0;
        for entry in snap.models.entries() {
            let m = &entry.value;
            if let Some(routing) = &m.routing {
                match routing.strategy {
                    RoutingStrategy::LeastBusy => bits |= FLAG_LEAST_BUSY,
                    RoutingStrategy::LeastLatency => bits |= FLAG_LEAST_LATENCY,
                    _ => {}
                }
            }
            if m.background_model_check.as_ref().is_some_and(|c| c.enabled) {
                bits |= FLAG_HEALTH_CHECKS;
            }
        }
        self.packed.store((ver << 3) | bits, Ordering::Relaxed);
        bits
    }
}

/// Shared tracker — one per `ProxyState`, cloned cheaply via `Arc`.
#[derive(Default, Debug)]
pub struct HealthTracker {
    entries: DashMap<String, Entry>,
    /// `None` (tests, lightweight constructors) means "assume active":
    /// the historical write path always runs. The production bootstrap
    /// wires the shared [`BookkeepingFlags`].
    flags: Option<Arc<BookkeepingFlags>>,
}

/// Smoothing factor for the per-target latency EWMA. Higher = more weight on
/// the most recent sample (faster reaction to a slowing upstream), lower =
/// smoother. 0.3 balances reacting to a real regression against per-request
/// jitter, roughly matching LiteLLM's last-10-samples moving average.
const LATENCY_EWMA_ALPHA: f64 = 0.3;

#[derive(Default, Debug)]
pub struct ModelRuntimeStatusTracker {
    entries: DashMap<String, RuntimeEntry>,
    /// Optional metrics sink. Wired only by the production
    /// [`crate::state::ProxyState::with_components`] bootstrap so cooldown
    /// transitions surface on the Prometheus scrape
    /// (`aisix_deployment_state` / `aisix_deployment_cooled_down_total`);
    /// `None` in tests and the lightweight constructors, where the tracker
    /// stays a pure state machine.
    metrics: Option<Arc<Metrics>>,
    /// Optional snapshot handle, used purely to resolve a cooled target's
    /// id into rich deployment labels (provider / upstream_model /
    /// provider_key_id) at emit time — a rare, O(1) `get_by_id` lookup
    /// only on a cooldown transition. `None` falls back to model-id-only
    /// labels.
    snapshot: Option<SnapshotHandle<AisixSnapshot>>,
    /// `None` (tests, lightweight constructors) means "assume active":
    /// every method runs its historical write path. See
    /// [`BookkeepingFlags`].
    flags: Option<Arc<BookkeepingFlags>>,
}

/// RAII guard that decrements a target's in-flight counter when dropped.
/// Created by [`ModelRuntimeStatusTracker::begin_in_flight`] before an
/// upstream attempt. For the streaming path the guard is moved into the
/// stream body so the count stays raised until the stream ends or is
/// cancelled, matching the request's true lifetime.
pub struct InFlightGuard {
    /// `None` is the no-op guard handed out while bookkeeping is
    /// inactive (no configured consumer); dropping it does nothing. A
    /// guard armed before a config change that deactivates bookkeeping
    /// still decrements the counter it incremented, so the count can
    /// never go negative.
    counter: Option<Arc<AtomicUsize>>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if let Some(counter) = &self.counter {
            counter.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl HealthTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Production constructor: consults the shared [`BookkeepingFlags`]
    /// so the per-request success write can take the read-first path
    /// when no configured consumer exists.
    pub fn with_flags(flags: Arc<BookkeepingFlags>) -> Self {
        Self {
            entries: DashMap::new(),
            flags: Some(flags),
        }
    }

    fn bookkeeping_active(&self) -> bool {
        self.flags.as_ref().is_none_or(|f| f.any_active())
    }

    /// Record a successful upstream response for `model`.
    pub fn record_success(&self, model: &str) {
        if self.bookkeeping_active() {
            self.entries
                .entry(model.to_string())
                .or_default()
                .on_success();
            return;
        }
        // Read-first path: a model with no tracked failures is already
        // Healthy — skip the key allocation and the shard write lock the
        // `entry()` API pays on every call. The counter is atomic, so
        // the reset happens under the read guard; a miss means the model
        // never failed and there is nothing to reset.
        if let Some(e) = self.entries.get(model) {
            if e.consecutive_failures.load(Ordering::Relaxed) != 0 {
                e.consecutive_failures.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Record a failed upstream call (any non-4xx bridge error) for `model`.
    pub fn record_failure(&self, model: &str) {
        self.entries
            .entry(model.to_string())
            .or_default()
            .on_failure();
    }

    /// Current [`HealthLevel`] for `model`. Returns `Healthy` if the model
    /// has never been seen (no prior calls, no failures tracked).
    pub fn level(&self, model: &str) -> HealthLevel {
        self.entries
            .get(model)
            .map(|e| e.level())
            .unwrap_or(HealthLevel::Healthy)
    }

    /// Snapshot of all (model_name, level) pairs seen so far.
    /// Models with no recorded calls are omitted — callers enumerate the
    /// snapshot's model table to include never-seen models as Healthy.
    pub fn all_levels(&self) -> Vec<(String, HealthLevel)> {
        self.entries
            .iter()
            .map(|e| (e.key().clone(), e.value().level()))
            .collect()
    }
}

impl ModelRuntimeStatusTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Production constructor: wires the metrics sink and snapshot handle
    /// so cooldown transitions emit `aisix_deployment_state` /
    /// `aisix_deployment_cooled_down_total`. Used by
    /// [`crate::state::ProxyState::with_components`]; the plain [`new`]
    /// (and `Default`) stay metrics-free for tests.
    pub fn with_observability(
        metrics: Arc<Metrics>,
        snapshot: SnapshotHandle<AisixSnapshot>,
        flags: Arc<BookkeepingFlags>,
    ) -> Self {
        Self {
            entries: DashMap::new(),
            metrics: Some(metrics),
            snapshot: Some(snapshot),
            flags: Some(flags),
        }
    }

    fn bookkeeping_active(&self) -> bool {
        self.flags.as_ref().is_none_or(|f| f.any_active())
    }

    pub fn mark_cooldown(&self, model_id: &str, ttl: Duration, reason: impl Into<String>) {
        let now = SystemTime::now();
        let until = now + ttl;
        let reason = reason.into();
        let mut entry = self.entries.entry(model_id.to_string()).or_default();
        // A fresh cooldown = the target was not already cooling (never
        // cooled, or a previous cooldown has since expired). Only that
        // transition is counted, so a burst of failures re-marking an
        // already-cooled target doesn't inflate the counter.
        let entered_cooldown = entry.cooldown_until.is_none_or(|u| u <= now);
        entry.cooldown_until = Some(until);
        entry.status_reason = Some(reason);
        // Hold the DashMap entry guard across the emit so concurrent
        // cooldown/recovery on the same model can't publish the gauge out of
        // order (which would leave it stale until the next transition). The
        // emit only reads `snapshot` and writes `metrics` — it never re-locks
        // `entries` — so holding the guard here is deadlock-free.
        if entered_cooldown {
            self.record_cooldown(model_id);
        }
        self.sync_deployment_state(model_id, &mut entry, now);
    }

    pub fn mark_healthy(&self, model_id: &str) {
        if self.bookkeeping_active() {
            if let Some(mut entry) = self.entries.get_mut(model_id) {
                entry.unhealthy = false;
                entry.cooldown_until = None;
                entry.status_reason = None;
                self.sync_deployment_state(model_id, &mut entry, SystemTime::now());
            }
            return;
        }
        // Read-first path (no configured bookkeeping consumer): in the
        // steady state — entry clean, gauge already Healthy — a read
        // guard and a few field loads replace the per-request shard
        // write lock + wall-clock read. The write path below still runs
        // whenever there is anything to do (cooldown early-recovery,
        // the first-success Healthy publish on an entry begin_in_flight
        // created), so the `aisix_deployment_state` series behaves
        // exactly as before. A MISSING entry stays a no-op, exactly as
        // on the historical path: the single-attempt handlers call
        // mark_healthy without begin_in_flight, and their targets must
        // not grow a series they never had.
        let needs_write = match self.entries.get(model_id) {
            Some(e) => {
                e.unhealthy
                    || e.cooldown_until.is_some()
                    || e.status_reason.is_some()
                    || e.emitted_state != Some(DeploymentState::Healthy)
            }
            None => false,
        };
        if !needs_write {
            return;
        }
        if let Some(mut entry) = self.entries.get_mut(model_id) {
            entry.unhealthy = false;
            entry.cooldown_until = None;
            entry.status_reason = None;
            self.sync_deployment_state(model_id, &mut entry, SystemTime::now());
        }
    }

    /// Publish `aisix_deployment_state` for `model_id` when the entry's
    /// serving state differs from what the gauge currently shows. Called
    /// after every mutation of `unhealthy` / `cooldown_until` — including
    /// the ones that merely *observe* a lapsed cooldown — so the gauge can
    /// never disagree with [`RuntimeEntry::deployment_state`]. The dedupe
    /// keeps already-healthy targets from writing the gauge on every
    /// successful request.
    ///
    /// Emitted while the caller still holds the DashMap entry guard, so
    /// concurrent cooldown/recovery on the same model can't publish out of
    /// order (see `mark_cooldown`).
    fn sync_deployment_state(&self, model_id: &str, entry: &mut RuntimeEntry, now: SystemTime) {
        let state = entry.deployment_state(now);
        if entry.emitted_state == Some(state) {
            return;
        }
        entry.emitted_state = Some(state);
        self.emit_deployment_state(model_id, state);
    }

    /// Bump `aisix_deployment_cooled_down_total` for `model_id`.
    fn record_cooldown(&self, model_id: &str) {
        self.with_deployment_labels(model_id, |metrics, labels| {
            metrics.record_deployment_cooldown(labels);
        });
    }

    /// Set the `aisix_deployment_state` gauge for `model_id`.
    fn emit_deployment_state(&self, model_id: &str, state: DeploymentState) {
        self.with_deployment_labels(model_id, |metrics, labels| {
            metrics.set_deployment_state(labels, state);
        });
    }

    /// Resolve `model_id`'s deployment labels and hand them to `f`. No-op
    /// unless a metrics sink is wired. Rich labels (provider /
    /// upstream_model / provider_key_id) come from the snapshot by id; a
    /// missing snapshot or unknown id falls back to a model-id-only set.
    fn with_deployment_labels(
        &self,
        model_id: &str,
        f: impl FnOnce(&Metrics, DeploymentLabels<'_>),
    ) {
        let Some(metrics) = self.metrics.as_ref() else {
            return;
        };
        let (provider, model, upstream_model, provider_key_id) = self
            .snapshot
            .as_ref()
            .and_then(|handle| {
                let snap = handle.load();
                let entry = snap.models.get_by_id(model_id)?;
                let m = &entry.value;
                Some((
                    m.provider.clone().unwrap_or_else(|| "unknown".to_string()),
                    m.display_name.clone(),
                    m.upstream_model().unwrap_or("unknown").to_string(),
                    m.provider_key_id
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string()),
                ))
            })
            .unwrap_or_else(|| {
                (
                    "unknown".to_string(),
                    model_id.to_string(),
                    "unknown".to_string(),
                    "unknown".to_string(),
                )
            });
        f(
            metrics,
            DeploymentLabels {
                provider: &provider,
                model: &model,
                upstream_model: &upstream_model,
                provider_key_id: &provider_key_id,
            },
        );
    }

    pub fn clear_unhealthy(&self, model_id: &str) {
        if let Some(mut entry) = self.entries.get_mut(model_id) {
            entry.unhealthy = false;
            if entry.status_reason.as_deref() == Some("background_check_failed") {
                entry.status_reason = None;
            }
            self.sync_deployment_state(model_id, &mut entry, SystemTime::now());
        }
    }

    pub fn mark_unhealthy(&self, model_id: &str, status: Option<u16>, reason: impl Into<String>) {
        let now = SystemTime::now();
        let reason = reason.into();
        let mut entry = self
            .entries
            .entry(model_id.to_string())
            .and_modify(|entry| {
                entry.unhealthy = true;
                entry.last_checked_at = Some(now);
                entry.last_check_status = status;
                entry.status_reason = Some(reason.clone());
            })
            .or_insert_with(|| RuntimeEntry {
                unhealthy: true,
                last_checked_at: Some(now),
                last_check_status: status,
                status_reason: Some(reason),
                ..RuntimeEntry::default()
            });
        self.sync_deployment_state(model_id, &mut entry, now);
    }

    pub fn record_ignored_check(&self, model_id: &str, status: u16, reason: impl Into<String>) {
        let now = SystemTime::now();
        let reason = reason.into();
        self.entries
            .entry(model_id.to_string())
            .and_modify(|entry| {
                entry.last_checked_at = Some(now);
                entry.last_check_status = Some(status);
                entry.status_reason = Some(reason.clone());
            })
            .or_insert_with(|| RuntimeEntry {
                last_checked_at: Some(now),
                last_check_status: Some(status),
                status_reason: Some(reason),
                ..RuntimeEntry::default()
            });
    }

    /// Fold a fresh latency sample (ms) into the target's EWMA. Called on
    /// each successful upstream attempt; drives the `least_latency` routing
    /// strategy. Independent of health/cooldown state.
    pub fn record_latency(&self, model_id: &str, latency_ms: u32) {
        // The EWMA's only reader is `least_latency` target ordering; with
        // no such strategy configured the sample has no consumer. When
        // the strategy is (re)configured the EWMA cold-starts, exactly as
        // it does on process start.
        if !self.bookkeeping_active() {
            return;
        }
        let sample = f64::from(latency_ms);
        self.entries
            .entry(model_id.to_string())
            .and_modify(|entry| {
                entry.latency_ewma_ms = Some(match entry.latency_ewma_ms {
                    Some(prev) => LATENCY_EWMA_ALPHA * sample + (1.0 - LATENCY_EWMA_ALPHA) * prev,
                    None => sample,
                });
            })
            .or_insert_with(|| RuntimeEntry {
                latency_ewma_ms: Some(sample),
                ..RuntimeEntry::default()
            });
    }

    /// Current latency EWMA (ms) for `model_id`, or `None` if never sampled.
    pub fn latency_ewma_ms(&self, model_id: &str) -> Option<f64> {
        self.entries.get(model_id).and_then(|e| e.latency_ewma_ms)
    }

    /// Mark one request as in flight to `model_id` and return a guard that
    /// decrements the count when dropped. Drives the `least_busy` strategy.
    pub fn begin_in_flight(&self, model_id: &str) -> InFlightGuard {
        // The counter's only reader is `least_busy` target ordering; with
        // no such strategy configured, hand out a no-op guard instead of
        // paying the counter RMWs and guard refcount per request. When
        // the strategy is (re)configured, counting resumes for new
        // requests; requests already in flight hold no-op guards, so the
        // count transiently underreads until they drain — the same
        // cold-start the counter has on process start.
        //
        // The ENTRY-CREATION side effect is preserved: `mark_healthy`'s
        // first-success Healthy publish keys off the entry this method
        // creates, and only the endpoints that call begin_in_flight may
        // publish that series (the single-attempt handlers never do).
        // Steady state downgrades to a read-guard existence check.
        if !self.bookkeeping_active() {
            if self.entries.get(model_id).is_none() {
                self.entries.entry(model_id.to_string()).or_default();
            }
            return InFlightGuard { counter: None };
        }
        let counter = Arc::clone(
            &self
                .entries
                .entry(model_id.to_string())
                .or_default()
                .in_flight,
        );
        counter.fetch_add(1, Ordering::Relaxed);
        InFlightGuard {
            counter: Some(counter),
        }
    }

    /// Current in-flight request count for `model_id`.
    pub fn in_flight(&self, model_id: &str) -> usize {
        self.entries
            .get(model_id)
            .map(|e| e.in_flight.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    pub fn status(&self, model_id: &str) -> RuntimeStatusSnapshot {
        self.status_with_stale(model_id, None)
    }

    pub fn status_with_stale(
        &self,
        model_id: &str,
        stale_after: Option<Duration>,
    ) -> RuntimeStatusSnapshot {
        let now = SystemTime::now();
        self.entries
            .get(model_id)
            .map(|entry| entry.snapshot(now, stale_after))
            .unwrap_or_default()
    }

    pub fn should_skip_for_routing(
        &self,
        model_id: &str,
        stale_after: Option<Duration>,
    ) -> RuntimeStatus {
        self.status_with_stale(model_id, stale_after).status
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use std::thread;

    #[test]
    fn new_model_is_healthy() {
        let t = HealthTracker::new();
        assert_eq!(t.level("m"), HealthLevel::Healthy);
    }

    #[test]
    fn consecutive_failures_transition_to_degraded_then_down() {
        let t = HealthTracker::new();
        for i in 1..=10 {
            t.record_failure("m");
            let expected = if i < DEGRADED_THRESHOLD {
                HealthLevel::Healthy
            } else if i < DOWN_THRESHOLD {
                HealthLevel::Degraded
            } else {
                HealthLevel::Down
            };
            assert_eq!(t.level("m"), expected, "wrong level after {i} failures");
        }
    }

    #[test]
    fn success_resets_to_healthy_regardless_of_prior_state() {
        let t = HealthTracker::new();
        for _ in 0..10 {
            t.record_failure("m");
        }
        assert_eq!(t.level("m"), HealthLevel::Down);
        t.record_success("m");
        assert_eq!(t.level("m"), HealthLevel::Healthy);
    }

    #[test]
    fn models_are_independent() {
        let t = HealthTracker::new();
        for _ in 0..10 {
            t.record_failure("bad");
        }
        assert_eq!(t.level("good"), HealthLevel::Healthy);
        assert_eq!(t.level("bad"), HealthLevel::Down);
    }

    #[test]
    fn all_levels_omits_never_seen_models() {
        let t = HealthTracker::new();
        assert!(t.all_levels().is_empty());
        t.record_success("m");
        assert_eq!(t.all_levels().len(), 1);
    }

    // -------------------------------------------------------------------
    // On-demand bookkeeping (BookkeepingFlags)
    // -------------------------------------------------------------------

    fn model_json(routing_strategy: Option<&str>, health_check: bool) -> aisix_core::Model {
        let mut v = serde_json::json!({ "name": "vg", "display_name": "vg" });
        if let Some(s) = routing_strategy {
            v["routing"] = serde_json::json!({
                "strategy": s,
                "targets": [{ "model": "d1" }],
            });
        }
        if health_check {
            v["background_model_check"] = serde_json::json!({
                "enabled": true,
                "interval_seconds": 5,
                "timeout_seconds": 1,
                "prompt": "ping",
                "max_tokens": 1,
                "stale_after_seconds": 60,
            });
        }
        serde_json::from_value(v).expect("test model json")
    }

    fn snapshot_with(model: Option<aisix_core::Model>) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        if let Some(m) = model {
            snap.models
                .insert(aisix_core::ResourceEntry::new("m-1", m, 1));
        }
        snap
    }

    fn inactive_tracker() -> (SnapshotHandle<AisixSnapshot>, ModelRuntimeStatusTracker) {
        let handle = SnapshotHandle::new(snapshot_with(None));
        let flags = BookkeepingFlags::new(handle.clone());
        let t = ModelRuntimeStatusTracker {
            entries: DashMap::new(),
            metrics: None,
            snapshot: None,
            flags: Some(flags),
        };
        (handle, t)
    }

    #[test]
    fn bookkeeping_flags_derive_from_snapshot() {
        for (model, expect) in [
            (None, false),
            (Some(model_json(Some("round_robin"), false)), false),
            (Some(model_json(Some("weighted"), false)), false),
            (Some(model_json(Some("failover"), false)), false),
            // least_cost ranks by static configured cost, not runtime
            // bookkeeping — it must NOT activate the write paths.
            (Some(model_json(Some("least_cost"), false)), false),
            (Some(model_json(Some("least_busy"), false)), true),
            (Some(model_json(Some("least_latency"), false)), true),
            (Some(model_json(None, true)), true),
        ] {
            let described = format!("{model:?}");
            let flags = BookkeepingFlags::new(SnapshotHandle::new(snapshot_with(model)));
            assert_eq!(flags.any_active(), expect, "for {described}");
        }
    }

    #[test]
    fn inactive_bookkeeping_skips_inflight_and_latency() {
        let (_handle, t) = inactive_tracker();
        let g = t.begin_in_flight("d1");
        assert_eq!(t.in_flight("d1"), 0, "no-op guard must not count");
        drop(g);
        assert_eq!(t.in_flight("d1"), 0, "no-op guard must not underflow");
        t.record_latency("d1", 100);
        assert_eq!(t.latency_ewma_ms("d1"), None);
    }

    #[test]
    fn bookkeeping_reactivates_on_snapshot_swap() {
        let (handle, t) = inactive_tracker();
        let g = t.begin_in_flight("d1");
        assert_eq!(t.in_flight("d1"), 0);
        drop(g);

        // Config change introduces a least_busy router → counting resumes
        // (version-gated recompute, no restart needed).
        handle.store(snapshot_with(Some(model_json(Some("least_busy"), false))));
        let g = t.begin_in_flight("d1");
        assert_eq!(t.in_flight("d1"), 1);
        // Any active predicate takes the WHOLE family back to the old
        // path — least_busy alone re-enables the EWMA write too.
        t.record_latency("d1", 100);
        assert_eq!(t.latency_ewma_ms("d1"), Some(100.0));
        drop(g);
        assert_eq!(t.in_flight("d1"), 0);
    }

    #[test]
    fn inactive_mark_healthy_publishes_healthy_once_then_reads() {
        let (_handle, t) = inactive_tracker();
        // Real multi-attempt-endpoint sequence: begin_in_flight creates
        // the entry (no-op guard, but the side effect is preserved),
        // then the first success publishes Healthy.
        drop(t.begin_in_flight("d1"));
        t.mark_healthy("d1");
        {
            let e = t
                .entries
                .get("d1")
                .expect("entry created by begin_in_flight");
            assert_eq!(e.emitted_state, Some(DeploymentState::Healthy));
        }
        // Steady state: read-only, entry untouched.
        t.mark_healthy("d1");
        assert_eq!(t.status("d1").status, RuntimeStatus::Healthy);
    }

    /// The single-attempt handlers (embeddings, images, completions,
    /// audio, rerank, count_tokens) call mark_healthy WITHOUT
    /// begin_in_flight. On the historical path their targets never got
    /// an entry — and therefore never published `aisix_deployment_state`
    /// — so the inactive fast path must stay a no-op for a missing
    /// entry, or zero-config deployments grow a series main never had.
    #[test]
    fn inactive_mark_healthy_without_begin_in_flight_stays_noop() {
        let (_handle, t) = inactive_tracker();
        t.mark_healthy("embeddings-only-target");
        assert!(
            t.entries.get("embeddings-only-target").is_none(),
            "mark_healthy on a never-seen id must not create an entry"
        );
    }

    #[test]
    fn inactive_mark_healthy_still_recovers_cooldown_early() {
        let (_handle, t) = inactive_tracker();
        t.mark_cooldown("d1", Duration::from_secs(3600), "upstream failure");
        assert_eq!(t.status("d1").status, RuntimeStatus::Cooldown);
        // A success during cooldown still recovers immediately — the
        // read-first path detects the dirty entry and takes the full
        // write path.
        t.mark_healthy("d1");
        assert_eq!(t.status("d1").status, RuntimeStatus::Healthy);
        assert_eq!(
            t.entries.get("d1").unwrap().emitted_state,
            Some(DeploymentState::Healthy)
        );
    }

    #[test]
    fn inactive_record_success_creates_no_entry_but_still_resets() {
        let flags = BookkeepingFlags::new(SnapshotHandle::new(snapshot_with(None)));
        let t = HealthTracker::with_flags(flags);
        // Happy path: no failures → no entry, no allocation.
        t.record_success("m");
        assert!(
            t.all_levels().is_empty(),
            "no entry for a never-failed model"
        );
        assert_eq!(t.level("m"), HealthLevel::Healthy);
        // Reset still works through the read guard.
        for _ in 0..10 {
            t.record_failure("m");
        }
        assert_eq!(t.level("m"), HealthLevel::Down);
        t.record_success("m");
        assert_eq!(t.level("m"), HealthLevel::Healthy);
    }

    #[tokio::test]
    async fn livez_default_success_is_plain_ok() {
        let state = LivezState::new();
        let resp = livez_response(&state, false);

        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(std::str::from_utf8(&body).unwrap(), "ok");
    }

    #[tokio::test]
    async fn livez_verbose_success_lists_checks() {
        let state = LivezState::new();
        let resp = livez_response(&state, true);

        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("[+]ping ok"));
        assert!(text.contains("[+]shutdown ok"));
        assert!(text.contains("livez check passed"));
    }

    #[tokio::test]
    async fn livez_failure_returns_503_with_reason_withheld() {
        let state = LivezState::new();
        state.mark_shutting_down();
        let resp = livez_response(&state, false);

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("[-]shutdown failed: reason withheld"));
        assert!(text.contains("livez check failed"));
    }

    #[test]
    fn config_readiness_block_logic() {
        // No apply yet → not ready (startup).
        assert!(config_readiness_block(None).is_some());
        // Applied → ready.
        assert!(config_readiness_block(Some(Duration::from_secs(5))).is_none());
    }

    #[test]
    fn an_idle_environment_stays_ready() {
        // The age is time since the last config EVENT, and an environment
        // whose resources are not changing produces none. Blocking on it
        // took every replica out of its Service five minutes after the
        // deployment went quiet, while the watch was healthy the whole
        // time. Ready must not depend on how long ago the last event was.
        for age in [
            Duration::from_secs(299),
            Duration::from_secs(301),
            Duration::from_secs(3600),
            Duration::from_secs(86_400 * 7),
        ] {
            assert!(
                config_readiness_block(Some(age)).is_none(),
                "config applied {age:?} ago must still be ready",
            );
        }
    }

    #[tokio::test]
    async fn readyz_ok_when_not_draining_and_config_fresh() {
        let state = LivezState::new();
        let resp = readyz_response(&state, None, false);
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn readyz_503_when_draining() {
        let state = LivezState::new();
        state.mark_shutting_down();
        let resp = readyz_response(&state, None, false);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn readyz_503_when_config_not_ready() {
        let state = LivezState::new();
        let resp = readyz_response(&state, Some("config not yet applied"), true);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("[-]config failed"));
    }

    #[test]
    fn runtime_tracker_defaults_to_healthy() {
        let t = ModelRuntimeStatusTracker::new();
        let s = t.status("m-1");
        assert_eq!(s.status, RuntimeStatus::Healthy);
        assert!(s.cooldown_until.is_none());
    }

    #[test]
    fn runtime_tracker_cooldown_expires() {
        let t = ModelRuntimeStatusTracker::new();
        t.mark_cooldown("m-1", Duration::from_millis(5), "retryable_failure");
        assert_eq!(t.status("m-1").status, RuntimeStatus::Cooldown);
        thread::sleep(Duration::from_millis(10));
        assert_eq!(t.status("m-1").status, RuntimeStatus::Healthy);
    }

    #[test]
    fn runtime_tracker_unhealthy_then_healthy() {
        let t = ModelRuntimeStatusTracker::new();
        t.mark_unhealthy("m-1", Some(500), "background_check_failed");
        let unhealthy = t.status("m-1");
        assert_eq!(unhealthy.status, RuntimeStatus::Unhealthy);
        assert_eq!(unhealthy.last_check_status, Some(500));
        t.mark_healthy("m-1");
        assert_eq!(t.status("m-1").status, RuntimeStatus::Healthy);
    }

    #[test]
    fn cooldown_transition_emits_deployment_metrics_once() {
        use aisix_core::{Model, ResourceEntry};

        // A snapshot with one direct model lets the tracker resolve rich
        // labels (provider / upstream_model / provider_key_id) for the
        // cooled target id instead of falling back to model-id-only.
        let model: Model = serde_json::from_value(serde_json::json!({
            "display_name": "cooldown-metrics-model",
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "provider_key_id": "pk-cooldown",
        }))
        .unwrap();
        let snapshot = AisixSnapshot::new();
        snapshot
            .models
            .insert(ResourceEntry::new("m-cool", model, 1));

        let metrics = Arc::new(Metrics::new(false));
        let handle = SnapshotHandle::new(snapshot);
        let tracker = ModelRuntimeStatusTracker::with_observability(
            metrics.clone(),
            handle.clone(),
            BookkeepingFlags::new(handle),
        );

        // First mark = a fresh transition (counter++, gauge → Down). The
        // second mark re-cools an already-cooled target and must NOT
        // double-count.
        tracker.mark_cooldown("m-cool", Duration::from_secs(30), "upstream_server_error");
        tracker.mark_cooldown("m-cool", Duration::from_secs(30), "upstream_server_error");

        let scrape = metrics.render();
        assert!(
            scrape.contains("aisix_deployment_cooled_down_total"),
            "cooldown counter missing from scrape:\n{scrape}"
        );
        // Labels came from the snapshot, not the model-id-only fallback.
        assert!(
            scrape.contains("provider=\"openai\"")
                && scrape.contains("upstream_model=\"gpt-4o-mini\"")
                && scrape.contains("provider_key_id=\"pk-cooldown\""),
            "expected resolved deployment labels in scrape:\n{scrape}"
        );
        let cooled = scrape
            .lines()
            .find(|l| l.starts_with("aisix_deployment_cooled_down_total{"))
            .expect("cooldown counter line");
        let count: f64 = cooled.rsplit(' ').next().unwrap().parse().unwrap();
        assert_eq!(count, 1.0, "cooldown counted once per transition: {cooled}");

        // Recovery flips the gauge back to Healthy(0).
        tracker.mark_healthy("m-cool");
        assert_eq!(
            deployment_state_gauge(&metrics),
            Some(0.0),
            "state gauge is Healthy(0) after recovery"
        );
    }

    /// A cooldown that lapses on its own is the *ordinary* recovery: the
    /// router filters cooled targets out of rotation, so nothing calls back
    /// into the tracker while the TTL runs down, and the first success
    /// arrives only after it has already expired. The old edge-triggered
    /// gauge could not see a transition at that point and left the target
    /// pinned at Down(2) forever.
    #[test]
    fn gauge_returns_to_healthy_after_a_cooldown_expires_naturally() {
        let metrics = Arc::new(Metrics::new(false));
        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let tracker = ModelRuntimeStatusTracker::with_observability(
            metrics.clone(),
            handle.clone(),
            BookkeepingFlags::new(handle),
        );

        tracker.mark_cooldown(
            "m-expiry",
            Duration::from_millis(5),
            "upstream_server_error",
        );
        assert_eq!(deployment_state_gauge(&metrics), Some(2.0), "cooled → Down");

        thread::sleep(Duration::from_millis(15));
        assert_eq!(
            tracker.status("m-expiry").status,
            RuntimeStatus::Healthy,
            "the cooldown has lapsed, so the target is back in rotation"
        );

        tracker.mark_healthy("m-expiry");
        assert_eq!(
            deployment_state_gauge(&metrics),
            Some(0.0),
            "gauge follows the target back into rotation"
        );
    }

    /// A background check failure takes the target out of rotation exactly
    /// like a cooldown does (`should_skip_for_routing` → Unhealthy), so the
    /// gauge has to say Down — and come back on the next passing check.
    #[test]
    fn gauge_tracks_background_check_failures_and_recovery() {
        let metrics = Arc::new(Metrics::new(false));
        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let tracker = ModelRuntimeStatusTracker::with_observability(
            metrics.clone(),
            handle.clone(),
            BookkeepingFlags::new(handle),
        );

        tracker.mark_unhealthy("m-bg", Some(503), "background_check_failed");
        assert_eq!(deployment_state_gauge(&metrics), Some(2.0));

        tracker.clear_unhealthy("m-bg");
        assert_eq!(deployment_state_gauge(&metrics), Some(0.0));
    }

    /// The gauge is level-triggered but not chatty: a target that is already
    /// healthy must not re-publish on every successful request, and the
    /// cooldown counter must not move when nothing entered cooldown.
    #[test]
    fn repeated_success_neither_churns_the_gauge_nor_the_cooldown_counter() {
        let metrics = Arc::new(Metrics::new(false));
        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let tracker = ModelRuntimeStatusTracker::with_observability(
            metrics.clone(),
            handle.clone(),
            BookkeepingFlags::new(handle),
        );

        // With bookkeeping active begin_in_flight creates the entry; with
        // it inactive (this tracker: empty snapshot) the first mark_healthy
        // does. Either way the assertions below must hold.
        drop(tracker.begin_in_flight("m-ok"));
        tracker.mark_healthy("m-ok");
        tracker.mark_healthy("m-ok");
        tracker.mark_healthy("m-ok");

        assert_eq!(deployment_state_gauge(&metrics), Some(0.0));
        assert!(
            !metrics
                .render()
                .contains("aisix_deployment_cooled_down_total"),
            "a never-cooled target must not emit the cooldown counter"
        );
    }

    /// Value of the single `aisix_deployment_state` series in the scrape.
    fn deployment_state_gauge(metrics: &Metrics) -> Option<f64> {
        metrics
            .render()
            .lines()
            .find(|l| l.starts_with("aisix_deployment_state{"))
            .and_then(|l| l.rsplit(' ').next()?.parse().ok())
    }

    #[test]
    fn runtime_tracker_ignored_status_does_not_mark_unhealthy() {
        let t = ModelRuntimeStatusTracker::new();
        t.record_ignored_check("m-1", 429, "ignored_transient_error");
        let s = t.status("m-1");
        assert_eq!(s.status, RuntimeStatus::Healthy);
        assert_eq!(s.last_check_status, Some(429));
        assert_eq!(s.status_reason.as_deref(), Some("ignored_transient_error"));
    }

    #[test]
    fn runtime_tracker_unhealthy_becomes_healthy_after_stale_window() {
        let t = ModelRuntimeStatusTracker::new();
        t.mark_unhealthy("m-1", Some(503), "background_check_failed");
        assert_eq!(
            t.status_with_stale("m-1", Some(Duration::from_secs(60)))
                .status,
            RuntimeStatus::Unhealthy
        );
        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(
            t.status_with_stale("m-1", Some(Duration::from_millis(1)))
                .status,
            RuntimeStatus::Healthy
        );
    }
}
