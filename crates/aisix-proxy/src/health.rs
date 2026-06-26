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
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime};

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

/// A config snapshot older than this — or never applied — means the etcd
/// watch isn't delivering fresh config, so the instance shouldn't be
/// counted ready for traffic (#591). Matches the freshness threshold the
/// admin health aggregate uses.
pub const READYZ_STALE_AFTER: Duration = Duration::from_secs(300);

/// Decide whether config freshness blocks readiness. `last_apply_age` is
/// the time since the config watch last applied an event: `None` means no
/// apply yet (still starting up / disconnected), `Some(age)` past the
/// stale threshold means a wedged watch. Returns `Some(reason)` when the
/// instance is not ready, `None` when config is fresh enough.
pub fn config_readiness_block(last_apply_age: Option<Duration>) -> Option<&'static str> {
    match last_apply_age {
        None => Some("config not yet applied"),
        Some(age) if age > READYZ_STALE_AFTER => Some("config watch is stale"),
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
}

impl RuntimeEntry {
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

/// Shared tracker — one per `ProxyState`, cloned cheaply via `Arc`.
#[derive(Default, Debug)]
pub struct HealthTracker {
    entries: DashMap<String, Entry>,
}

#[derive(Default, Debug)]
pub struct ModelRuntimeStatusTracker {
    entries: DashMap<String, RuntimeEntry>,
}

impl HealthTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful upstream response for `model`.
    pub fn record_success(&self, model: &str) {
        self.entries
            .entry(model.to_string())
            .or_default()
            .on_success();
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

    pub fn mark_cooldown(&self, model_id: &str, ttl: Duration, reason: impl Into<String>) {
        let now = SystemTime::now();
        let until = now + ttl;
        let reason = reason.into();
        self.entries
            .entry(model_id.to_string())
            .and_modify(|entry| {
                entry.cooldown_until = Some(until);
                entry.status_reason = Some(reason.clone());
            })
            .or_insert_with(|| RuntimeEntry {
                cooldown_until: Some(until),
                status_reason: Some(reason),
                ..RuntimeEntry::default()
            });
    }

    pub fn mark_healthy(&self, model_id: &str) {
        if let Some(mut entry) = self.entries.get_mut(model_id) {
            entry.unhealthy = false;
            entry.cooldown_until = None;
            entry.status_reason = None;
        }
    }

    pub fn clear_unhealthy(&self, model_id: &str) {
        if let Some(mut entry) = self.entries.get_mut(model_id) {
            entry.unhealthy = false;
            if entry.status_reason.as_deref() == Some("background_check_failed") {
                entry.status_reason = None;
            }
        }
    }

    pub fn mark_unhealthy(&self, model_id: &str, status: Option<u16>, reason: impl Into<String>) {
        let now = SystemTime::now();
        let reason = reason.into();
        self.entries
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
        // Fresh apply → ready.
        assert!(config_readiness_block(Some(Duration::from_secs(5))).is_none());
        // Beyond the stale threshold → not ready (wedged watch).
        assert!(
            config_readiness_block(Some(READYZ_STALE_AFTER + Duration::from_secs(1))).is_some()
        );
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
