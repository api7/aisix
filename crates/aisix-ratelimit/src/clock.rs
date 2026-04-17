//! Injected clock so the fixed-window counters can be tested
//! deterministically. The production [`SystemClock`] delegates to
//! `SystemTime::now()`; [`TestClock`] is a thread-safe stepper a test
//! can advance by hand.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Second-resolution wall-clock. That's all the fixed-window counters
/// need — they bucket by minute and by day boundaries.
pub trait Clock: Send + Sync + 'static {
    fn unix_secs(&self) -> u64;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn unix_secs(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// Simple test double. Call [`TestClock::advance`] between operations
/// to jump to the next window without spinning wall-clock time.
#[derive(Debug, Clone, Default)]
pub struct TestClock {
    now: Arc<AtomicU64>,
}

impl TestClock {
    pub fn new(initial_secs: u64) -> Self {
        Self {
            now: Arc::new(AtomicU64::new(initial_secs)),
        }
    }

    pub fn advance(&self, secs: u64) {
        self.now.fetch_add(secs, Ordering::SeqCst);
    }

    pub fn set(&self, secs: u64) {
        self.now.store(secs, Ordering::SeqCst);
    }
}

impl Clock for TestClock {
    fn unix_secs(&self) -> u64 {
        self.now.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_clock_returns_positive_now() {
        assert!(SystemClock.unix_secs() > 0);
    }

    #[test]
    fn test_clock_advances_and_sets() {
        let c = TestClock::new(100);
        assert_eq!(c.unix_secs(), 100);
        c.advance(30);
        assert_eq!(c.unix_secs(), 130);
        c.set(500);
        assert_eq!(c.unix_secs(), 500);
    }
}
