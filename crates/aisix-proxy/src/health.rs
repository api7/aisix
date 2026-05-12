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
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

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
    async fn livez_failure_returns_500_with_reason_withheld() {
        let state = LivezState::new();
        state.mark_shutting_down();
        let resp = livez_response(&state, false);

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("[-]shutdown failed: reason withheld"));
        assert!(text.contains("livez check failed"));
    }
}
