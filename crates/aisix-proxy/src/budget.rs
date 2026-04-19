//! In-process budget tracker.
//!
//! Tracks accumulated USD spend per (api_key_id, calendar month) tuple.
//! Lookup is O(1); the tracker resets a key's counter automatically
//! when the calendar month rolls over. State is process-local for V1
//! — operators who need cross-restart durability swap in a future
//! Redis-backed tracker behind the same trait shape.
//!
//! The clock is injectable so unit tests can step time without
//! sleeping wall-clock.

use chrono::{DateTime, Datelike, Utc};
use dashmap::DashMap;

/// Wall-clock seam.
pub trait BudgetClock: Send + Sync + 'static {
    fn now(&self) -> DateTime<Utc>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemBudgetClock;

impl BudgetClock for SystemBudgetClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// (year, month_of_year) — the monthly bucket key.
type MonthKey = (i32, u32);

#[derive(Debug, Default)]
struct Entry {
    bucket: MonthKey,
    spend_usd: f64,
}

pub struct BudgetTracker<C: BudgetClock = SystemBudgetClock> {
    inner: DashMap<String, Entry>,
    clock: C,
}

impl<C: BudgetClock> std::fmt::Debug for BudgetTracker<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetTracker")
            .field("tracked_keys", &self.inner.len())
            .finish()
    }
}

impl Default for BudgetTracker<SystemBudgetClock> {
    fn default() -> Self {
        Self {
            inner: DashMap::new(),
            clock: SystemBudgetClock,
        }
    }
}

impl BudgetTracker<SystemBudgetClock> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<C: BudgetClock> BudgetTracker<C> {
    pub fn with_clock(clock: C) -> Self {
        Self {
            inner: DashMap::new(),
            clock,
        }
    }

    /// Current month's spend for an ApiKey. Auto-resets if the bucket
    /// is stale.
    pub fn spend(&self, api_key_id: &str) -> f64 {
        let now = self.clock.now();
        let bucket = month_key(&now);
        match self.inner.get(api_key_id) {
            Some(e) if e.bucket == bucket => e.spend_usd,
            _ => 0.0,
        }
    }

    /// Add `usd` to the current month's running total. Resets the
    /// bucket if the month rolled over since the last call.
    pub fn add(&self, api_key_id: &str, usd: f64) {
        let now = self.clock.now();
        let bucket = month_key(&now);
        let mut entry = self.inner.entry(api_key_id.to_string()).or_default();
        if entry.bucket != bucket {
            entry.bucket = bucket;
            entry.spend_usd = 0.0;
        }
        entry.spend_usd += usd;
    }

    /// True if `(current spend + projected_cost) > cap`. The check
    /// excludes the projected request itself — used for pre-commit
    /// short-circuit when the *previous* month's tail already
    /// over-shot the cap.
    pub fn would_exceed(&self, api_key_id: &str, cap_usd: f64) -> bool {
        self.spend(api_key_id) >= cap_usd
    }

    /// Snapshot of all (api_key_id, spend_usd) pairs for the current
    /// calendar month. Entries from previous months are omitted (they
    /// will auto-reset on next write). Used by the admin spend endpoint.
    pub fn all_entries(&self) -> Vec<(String, f64)> {
        let now = self.clock.now();
        let bucket = month_key(&now);
        self.inner
            .iter()
            .filter_map(|e| {
                if e.value().bucket == bucket {
                    Some((e.key().clone(), e.value().spend_usd))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Total spend across all api-keys for the current month.
    pub fn total_spend(&self) -> f64 {
        self.all_entries().iter().map(|(_, v)| v).sum()
    }
}

fn month_key(t: &DateTime<Utc>) -> MonthKey {
    (t.year(), t.month())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::Arc;

    /// Test clock that returns whatever epoch second is set on it.
    struct TestClock {
        epoch_secs: AtomicI64,
    }

    impl TestClock {
        fn new(t: DateTime<Utc>) -> Self {
            Self {
                epoch_secs: AtomicI64::new(t.timestamp()),
            }
        }
        fn set(&self, t: DateTime<Utc>) {
            self.epoch_secs.store(t.timestamp(), Ordering::SeqCst);
        }
    }

    impl BudgetClock for TestClock {
        fn now(&self) -> DateTime<Utc> {
            Utc.timestamp_opt(self.epoch_secs.load(Ordering::SeqCst), 0)
                .single()
                .unwrap()
        }
    }

    fn jan(day: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, day, 12, 0, 0).single().unwrap()
    }
    fn feb(day: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 2, day, 12, 0, 0).single().unwrap()
    }

    #[test]
    fn empty_tracker_reports_zero_spend() {
        let t = BudgetTracker::with_clock(TestClock::new(jan(1)));
        assert_eq!(t.spend("k1"), 0.0);
        assert!(!t.would_exceed("k1", 10.0));
    }

    #[test]
    fn add_accumulates_within_the_same_month() {
        let t = BudgetTracker::with_clock(TestClock::new(jan(1)));
        t.add("k1", 1.5);
        t.add("k1", 2.5);
        assert!((t.spend("k1") - 4.0).abs() < 1e-9);
    }

    #[test]
    fn would_exceed_fires_only_when_cap_reached() {
        let t = BudgetTracker::with_clock(TestClock::new(jan(1)));
        t.add("k1", 9.0);
        assert!(!t.would_exceed("k1", 10.0));
        t.add("k1", 1.0);
        assert!(t.would_exceed("k1", 10.0));
    }

    #[test]
    fn month_rollover_resets_bucket_automatically() {
        let clock = Arc::new(TestClock::new(jan(15)));
        let t = BudgetTracker::with_clock(ClockHandle(clock.clone()));
        t.add("k1", 50.0);
        assert!((t.spend("k1") - 50.0).abs() < 1e-9);

        // Roll into February.
        clock.set(feb(1));
        // Reading first auto-resets the bucket.
        assert_eq!(t.spend("k1"), 0.0);
        // And subsequent adds start fresh.
        t.add("k1", 1.0);
        assert!((t.spend("k1") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn keys_are_independent_of_each_other() {
        let t = BudgetTracker::with_clock(TestClock::new(jan(1)));
        t.add("k1", 5.0);
        t.add("k2", 10.0);
        assert!((t.spend("k1") - 5.0).abs() < 1e-9);
        assert!((t.spend("k2") - 10.0).abs() < 1e-9);
    }

    /// Handle wrapper so we can share a clock between the test's
    /// BudgetTracker and the test scope without `&` lifetime juggling.
    struct ClockHandle(Arc<TestClock>);
    impl BudgetClock for ClockHandle {
        fn now(&self) -> DateTime<Utc> {
            self.0.now()
        }
    }
}
