//! Fixed-window counter.
//!
//! A single counter that resets at the start of every new window. Used
//! for both the per-minute and per-day dimensions; callers instantiate
//! one per dimension with the matching `window_secs`.
//!
//! Not thread-safe on its own — the caller is expected to hold the
//! `KeyState` guard before touching it. That keeps the hot path lock-
//! cheap: one `DashMap` shard + one `parking_lot::Mutex` per key.

/// Result of attempting to reserve capacity.
#[derive(Debug, PartialEq, Eq)]
pub enum WindowCheck {
    Ok,
    Full { retry_after_secs: u64 },
}

#[derive(Debug)]
pub struct FixedWindowCounter {
    window_secs: u64,
    window_start: u64,
    count: u64,
}

impl FixedWindowCounter {
    pub fn new(window_secs: u64) -> Self {
        assert!(window_secs > 0, "window_secs must be positive");
        Self {
            window_secs,
            window_start: 0,
            count: 0,
        }
    }

    pub fn window_secs(&self) -> u64 {
        self.window_secs
    }

    fn roll_if_stale(&mut self, now_secs: u64) {
        let bucket_start = (now_secs / self.window_secs) * self.window_secs;
        if bucket_start != self.window_start {
            self.window_start = bucket_start;
            self.count = 0;
        }
    }

    /// Check whether `delta` more units would fit under `limit`. If yes,
    /// commit them (increment counter) and return `Ok`. If no, return
    /// `Full` with seconds until the window rolls over.
    pub fn check_and_increment(&mut self, now_secs: u64, delta: u64, limit: u64) -> WindowCheck {
        self.roll_if_stale(now_secs);
        let would_be = self.count.saturating_add(delta);
        if would_be > limit {
            let remainder = self
                .window_secs
                .saturating_sub(now_secs.saturating_sub(self.window_start));
            return WindowCheck::Full {
                retry_after_secs: remainder.max(1),
            };
        }
        self.count = would_be;
        WindowCheck::Ok
    }

    /// Add to the counter without a check. Used on the post-deduct side
    /// for TPM/TPD — the token count isn't known until the upstream
    /// response has completed, so we record after the fact.
    pub fn add(&mut self, now_secs: u64, delta: u64) {
        self.roll_if_stale(now_secs);
        self.count = self.count.saturating_add(delta);
    }

    pub fn current(&mut self, now_secs: u64) -> u64 {
        self.roll_if_stale(now_secs);
        self.count
    }

    /// Peek at whether the current count already exceeds the limit. Used
    /// on the *next* request's pre-commit to short-circuit before
    /// increment: TPM is checked-but-not-incremented at pre-commit, then
    /// incremented on post-deduct by the actual token usage.
    pub fn is_exceeded(&mut self, now_secs: u64, limit: u64) -> Option<u64> {
        self.roll_if_stale(now_secs);
        if self.count > limit {
            let remainder = self
                .window_secs
                .saturating_sub(now_secs.saturating_sub(self.window_start));
            Some(remainder.max(1))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_increments_fit_then_subsequent_block() {
        let mut w = FixedWindowCounter::new(60);
        assert_eq!(w.check_and_increment(100, 1, 3), WindowCheck::Ok);
        assert_eq!(w.check_and_increment(100, 1, 3), WindowCheck::Ok);
        assert_eq!(w.check_and_increment(100, 1, 3), WindowCheck::Ok);
        // 4th attempt overflows the limit.
        match w.check_and_increment(100, 1, 3) {
            WindowCheck::Full { retry_after_secs } => {
                assert!(retry_after_secs > 0);
                assert!(retry_after_secs <= 60);
            }
            _ => panic!("expected Full"),
        }
    }

    #[test]
    fn counter_rolls_over_at_window_boundary() {
        let mut w = FixedWindowCounter::new(60);
        for _ in 0..3 {
            w.check_and_increment(100, 1, 3);
        }
        // Cross into the next minute.
        assert_eq!(w.check_and_increment(161, 1, 3), WindowCheck::Ok);
        assert_eq!(w.current(161), 1);
    }

    #[test]
    fn retry_after_reflects_time_remaining_in_window() {
        let mut w = FixedWindowCounter::new(60);
        // Fill the bucket at second 100 (bucket starts at 60, ends at 120).
        for _ in 0..3 {
            w.check_and_increment(100, 1, 3);
        }
        match w.check_and_increment(110, 1, 3) {
            WindowCheck::Full { retry_after_secs } => {
                assert_eq!(retry_after_secs, 10); // 60+60 - 110 = 10
            }
            _ => panic!("expected Full"),
        }
    }

    #[test]
    fn add_records_post_deduct_usage_and_is_checkable() {
        let mut w = FixedWindowCounter::new(60);
        w.add(100, 1_000);
        w.add(101, 500);
        assert_eq!(w.current(101), 1_500);

        assert!(w.is_exceeded(101, 2_000).is_none()); // 1500 <= 2000
        assert!(w.is_exceeded(101, 1_000).is_some()); // 1500 > 1000
    }

    #[test]
    fn check_with_zero_delta_is_a_read_only_peek_that_succeeds() {
        let mut w = FixedWindowCounter::new(60);
        assert_eq!(w.check_and_increment(100, 0, 5), WindowCheck::Ok);
        assert_eq!(w.current(100), 0);
    }

    #[test]
    fn retry_after_is_at_least_one_second() {
        let mut w = FixedWindowCounter::new(60);
        // Window is [60, 120). Fill it past the cap.
        w.add(100, 1_000);
        // Query right at the last second of the same window — remainder
        // would be 0, but we clamp to 1 so clients don't spin.
        let hint = w.is_exceeded(119, 100).unwrap();
        assert!(hint >= 1);
    }
}
