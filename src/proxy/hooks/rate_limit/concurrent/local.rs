use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use async_trait::async_trait;
use dashmap::DashMap;

use super::{
    ConcurrencyError, ConcurrencyGuard, ConcurrencyInfo, ConcurrencyLimiter, ConcurrencyPermit,
};

/// RAII guard that decrements the counter on drop and evicts the zero-count
/// entry from the map to reclaim memory.
struct CounterGuard {
    counter: Arc<AtomicU64>,
    counters: Arc<DashMap<String, Arc<AtomicU64>>>,
    key: String,
}

impl ConcurrencyGuard for CounterGuard {}

impl Drop for CounterGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Release);

        // Evict the entry when the counter reaches zero to prevent unbounded
        // map growth.
        //
        // Safety against races with try_acquire:
        //   try_acquire calls fetch_add *while holding the DashMap shard lock*
        //   (via the RefMut from entry().or_insert_with()).  remove_if also
        //   acquires the shard lock before evaluating the predicate, so the
        //   two operations are mutually exclusive for the same key:
        //
        //   - If remove_if sees count == 0, no try_acquire is mid-increment.
        //   - Arc::ptr_eq guards against evicting a freshly-inserted entry
        //     that replaced ours after a previous eviction.
        if self.counter.load(Ordering::Acquire) == 0 {
            self.counters.remove_if(&self.key, |_, v| {
                Arc::ptr_eq(v, &self.counter) && v.load(Ordering::Acquire) == 0
            });
        }
    }
}

/// Local in-memory concurrency limiter using AtomicU64 counters.
///
/// Each key gets an independent `AtomicU64` counter stored in a `DashMap`.
/// On `try_acquire`, the counter is atomically incremented **while holding the
/// DashMap shard lock** so that the drop-time cleanup (which also acquires
/// the shard lock via `remove_if`) cannot observe a spurious zero between our
/// `entry().or_insert_with()` and `fetch_add`.  Config hot-reloads take
/// effect immediately because `max_concurrent` is read at call time.
pub struct LocalConcurrencyLimiter {
    counters: Arc<DashMap<String, Arc<AtomicU64>>>,
}

impl LocalConcurrencyLimiter {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(DashMap::new()),
        }
    }
}

#[async_trait]
impl ConcurrencyLimiter for LocalConcurrencyLimiter {
    async fn try_acquire(
        &self,
        key: &str,
        max_concurrent: u64,
    ) -> Result<ConcurrencyPermit, ConcurrencyError> {
        // Increment while the DashMap shard lock is held (RefMut from
        // or_insert_with keeps the lock alive until dropped).  This ensures
        // CounterGuard::drop's remove_if cannot race with our increment.
        let entry = self
            .counters
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)));
        let prev = entry.fetch_add(1, Ordering::Acquire);
        let counter = entry.clone();
        drop(entry); // release shard lock before the limit check

        if prev >= max_concurrent {
            // Over limit — roll back immediately.
            counter.fetch_sub(1, Ordering::Release);
            return Err(ConcurrencyError::Exceeded {
                limit: max_concurrent,
                // `prev` was the count before our increment; since we rolled
                // back, the "current" occupancy is at least `prev`.
                current: prev,
            });
        }

        // `prev` was the value before increment, so current occupancy
        // (including this request) is `prev + 1`.
        Ok(ConcurrencyPermit::new(
            ConcurrencyInfo {
                limit: max_concurrent,
                current: prev + 1,
            },
            CounterGuard {
                counter,
                counters: self.counters.clone(),
                key: key.to_string(),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_acquire_within_limit() {
        let limiter = LocalConcurrencyLimiter::new();
        let permit = limiter.try_acquire("k1", 5).await;
        assert!(permit.is_ok());
        let p = permit.unwrap();
        assert_eq!(p.info.limit, 5);
        assert_eq!(p.info.current, 1);
        assert_eq!(p.info.remaining(), 4);
    }

    #[tokio::test]
    async fn test_acquire_releases_on_drop() {
        let limiter = LocalConcurrencyLimiter::new();
        {
            let _p = limiter.try_acquire("k2", 1).await.unwrap();
            // Slot is occupied — next acquire should fail
            let err = limiter.try_acquire("k2", 1).await;
            assert!(err.is_err());
        }
        // After drop, slot is freed
        let permit = limiter.try_acquire("k2", 1).await;
        assert!(permit.is_ok());
    }

    #[tokio::test]
    async fn test_acquire_exceeds_limit() {
        let limiter = LocalConcurrencyLimiter::new();
        let _p1 = limiter.try_acquire("k3", 2).await.unwrap();
        let _p2 = limiter.try_acquire("k3", 2).await.unwrap();

        let err = limiter.try_acquire("k3", 2).await;
        assert!(matches!(
            err,
            Err(ConcurrencyError::Exceeded {
                limit: 2,
                current: 2,
            })
        ));
    }

    #[tokio::test]
    async fn test_key_isolation() {
        let limiter = LocalConcurrencyLimiter::new();
        let _p1 = limiter.try_acquire("a", 1).await.unwrap();

        // Different key should still work
        let p2 = limiter.try_acquire("b", 1).await;
        assert!(p2.is_ok());
    }

    #[tokio::test]
    async fn test_hot_reload_increase() {
        let limiter = LocalConcurrencyLimiter::new();
        let _p1 = limiter.try_acquire("hr", 1).await.unwrap();

        // With limit=1, second request fails
        assert!(limiter.try_acquire("hr", 1).await.is_err());

        // "Hot reload" — increase limit to 2
        let p2 = limiter.try_acquire("hr", 2).await;
        assert!(p2.is_ok());
    }

    #[tokio::test]
    async fn test_hot_reload_decrease() {
        let limiter = LocalConcurrencyLimiter::new();
        // Acquire 2 slots with limit=3
        let _p1 = limiter.try_acquire("hrd", 3).await.unwrap();
        let _p2 = limiter.try_acquire("hrd", 3).await.unwrap();

        // "Hot reload" — decrease limit to 1; existing permits stay, new ones rejected
        let err = limiter.try_acquire("hrd", 1).await;
        assert!(matches!(err, Err(ConcurrencyError::Exceeded { .. })));

        // After existing permits drop, should work again with new limit
        drop(_p1);
        drop(_p2);
        let p3 = limiter.try_acquire("hrd", 1).await;
        assert!(p3.is_ok());
    }

    #[tokio::test]
    async fn test_concurrent_counter_accuracy() {
        let limiter = Arc::new(LocalConcurrencyLimiter::new());
        let max = 10u64;

        let mut permits = Vec::new();
        for i in 0..max {
            let p = limiter.try_acquire("acc", max).await.unwrap();
            assert_eq!(p.info.current, i + 1);
            permits.push(p);
        }

        // All slots occupied
        assert!(limiter.try_acquire("acc", max).await.is_err());

        // Drop all
        permits.clear();

        // Should be able to acquire again
        let p = limiter.try_acquire("acc", max).await.unwrap();
        assert_eq!(p.info.current, 1);
    }

    /// Verify that map entries are evicted once all permits for a key are dropped.
    #[tokio::test]
    async fn test_entry_evicted_after_all_permits_dropped() {
        let limiter = LocalConcurrencyLimiter::new();

        assert_eq!(limiter.counters.len(), 0);

        let p1 = limiter.try_acquire("evict", 5).await.unwrap();
        let p2 = limiter.try_acquire("evict", 5).await.unwrap();
        assert_eq!(limiter.counters.len(), 1); // one entry for "evict"

        // First permit dropped — entry still present (counter = 1)
        drop(p1);
        assert_eq!(limiter.counters.len(), 1);

        // Second permit dropped — counter reaches 0, entry evicted
        drop(p2);
        assert_eq!(limiter.counters.len(), 0);

        // Re-acquiring creates a fresh entry
        let p3 = limiter.try_acquire("evict", 5).await.unwrap();
        assert_eq!(p3.info.current, 1);
        assert_eq!(limiter.counters.len(), 1);
        drop(p3);
        assert_eq!(limiter.counters.len(), 0);
    }

    /// Verify key isolation: dropping all permits for one key does not affect others.
    #[tokio::test]
    async fn test_eviction_does_not_affect_other_keys() {
        let limiter = LocalConcurrencyLimiter::new();

        let p_a = limiter.try_acquire("alpha", 5).await.unwrap();
        let p_b = limiter.try_acquire("beta", 5).await.unwrap();
        assert_eq!(limiter.counters.len(), 2);

        drop(p_a); // "alpha" evicted
        assert_eq!(limiter.counters.len(), 1);
        assert!(limiter.counters.contains_key("beta"));

        drop(p_b); // "beta" evicted
        assert_eq!(limiter.counters.len(), 0);
    }
}
