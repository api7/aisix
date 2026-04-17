//! Exponential backoff used by the watch supervisor when reconnecting to
//! etcd (spec §2: 1s → 2 → 4 → 8 → 16 → 32 → 60s max).
//!
//! This is a pure data structure — it returns durations. The calling task is
//! responsible for actually sleeping.

use std::time::Duration;

pub const BASE_MS: u64 = 1_000;
pub const MAX_MS: u64 = 60_000;

#[derive(Debug, Clone)]
pub struct ExpBackoff {
    current_ms: u64,
    base_ms: u64,
    max_ms: u64,
}

impl Default for ExpBackoff {
    fn default() -> Self {
        Self::new(BASE_MS, MAX_MS)
    }
}

impl ExpBackoff {
    pub const fn new(base_ms: u64, max_ms: u64) -> Self {
        Self {
            current_ms: base_ms,
            base_ms,
            max_ms,
        }
    }

    /// Return the current delay and advance for the next call. Saturates at
    /// `max_ms` so long-running reconnect loops don't balloon past 60s.
    pub fn next_delay(&mut self) -> Duration {
        let d = Duration::from_millis(self.current_ms);
        self.current_ms = (self.current_ms.saturating_mul(2)).min(self.max_ms);
        d
    }

    /// Reset the backoff after a successful reconnect so the next failure
    /// restarts at `base_ms`.
    pub fn reset(&mut self) {
        self.current_ms = self.base_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_then_saturates_at_max() {
        let mut b = ExpBackoff::new(1_000, 60_000);
        let seq: Vec<u64> = (0..8).map(|_| b.next_delay().as_millis() as u64).collect();
        assert_eq!(
            seq,
            vec![1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 60_000, 60_000]
        );
    }

    #[test]
    fn reset_returns_to_base() {
        let mut b = ExpBackoff::new(500, 8_000);
        b.next_delay();
        b.next_delay();
        b.reset();
        assert_eq!(b.next_delay().as_millis() as u64, 500);
    }

    #[test]
    fn default_matches_spec_1s_to_60s() {
        let b = ExpBackoff::default();
        assert_eq!(b.base_ms, BASE_MS);
        assert_eq!(b.max_ms, MAX_MS);
    }
}
