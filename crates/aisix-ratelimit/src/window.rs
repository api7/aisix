//! Window counters.
//!
//! [`FixedWindowCounter`] is a single counter that resets at the start of
//! every new window; callers instantiate one per dimension with the
//! matching `window_secs`. [`SlidingWindowCounter`] is the
//! `window_type: sliding` variant: epoch-aligned sub-buckets of `step_secs`
//! summed over a rolling window.
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

    /// Decrement the current window's count by `delta`, saturating at
    /// zero. No-op when `now_secs` falls outside the current window —
    /// the increment we want to undo has already been wiped by the
    /// roll-over, and applying the decrement to the fresh window would
    /// destroy unrelated counts. Used by the limiter to compensate for
    /// the RPM increment when a sibling RPD cap rejects the same
    /// request: only that one increment must roll back, not any
    /// concurrent ones from other in-flight reservations.
    pub fn decrement(&mut self, now_secs: u64, delta: u64) {
        let bucket_start = (now_secs / self.window_secs) * self.window_secs;
        if bucket_start == self.window_start {
            self.count = self.count.saturating_sub(delta);
        }
    }

    pub fn current(&mut self, now_secs: u64) -> u64 {
        self.roll_if_stale(now_secs);
        self.count
    }

    /// Peek at whether the window's budget is already consumed. Used on
    /// the *next* request's pre-commit to short-circuit before
    /// increment: TPM is checked-but-not-incremented at pre-commit, then
    /// incremented on post-deduct by the actual token usage.
    ///
    /// A count that has landed exactly ON the limit is consumed, not
    /// still admissible. Token limiting is post-paid — the usage is
    /// committed after the response — so every request admitted before the
    /// window filled still completes and counts: the overshoot is bounded by
    /// the tokens of all in-flight responses, not one. Admitting at equality
    /// on top of that overshot by a further whole response (#950). The Redis
    /// backend's acquire script draws the same line, so swapping
    /// backends cannot change the observable limit.
    pub fn is_exceeded(&mut self, now_secs: u64, limit: u64) -> Option<u64> {
        self.roll_if_stale(now_secs);
        if self.count >= limit {
            let remainder = self
                .window_secs
                .saturating_sub(now_secs.saturating_sub(self.window_start));
            Some(remainder.max(1))
        } else {
            None
        }
    }
}

/// Rolling-window counter built from epoch-aligned sub-buckets.
///
/// A sub-bucket starts at a multiple of `step_secs`. The usage at `now` is
/// the sum of every sub-bucket whose start is at or after
/// `floor(now / step) * step - window`: the current sub-bucket plus the
/// `window / step` whole ones before it. Every admission inside any
/// `window`-long interval ending at `now` lands in one of those, so no such
/// interval admits more than the limit. A unit recorded at `t` leaves the
/// sum between `window` and `window + step` after `t`.
///
/// The ring holds exactly the sub-buckets that can be in the sum, so memory
/// is bounded by `window / step + 1` and does not grow with the limit. The
/// Redis backend's scripts implement the same arithmetic.
#[derive(Debug)]
pub struct SlidingWindowCounter {
    window_secs: u64,
    step_secs: u64,
    /// `(sub-bucket start, count)`, indexed by `(start / step) % len`.
    slots: Vec<(u64, u64)>,
}

impl SlidingWindowCounter {
    pub fn new(window_secs: u64, step_secs: u64) -> Self {
        assert!(
            step_secs > 0 && window_secs.is_multiple_of(step_secs),
            "window must be a positive multiple of step"
        );
        let len = (window_secs / step_secs + 1) as usize;
        Self {
            window_secs,
            step_secs,
            slots: vec![(0, 0); len],
        }
    }

    fn current_start(&self, now_secs: u64) -> u64 {
        (now_secs / self.step_secs) * self.step_secs
    }

    /// Count recorded in the sub-bucket starting at `start`, or zero when
    /// its slot has since been reused for a newer one.
    fn count_at(&self, start: u64) -> u64 {
        let (slot_start, count) =
            self.slots[((start / self.step_secs) as usize) % self.slots.len()];
        if slot_start == start {
            count
        } else {
            0
        }
    }

    /// Starts of the sub-buckets in the sum at `now`, oldest first.
    fn starts_in_window(&self, now_secs: u64) -> impl Iterator<Item = u64> {
        let cur = self.current_start(now_secs);
        let step = self.step_secs as usize;
        (cur.saturating_sub(self.window_secs)..=cur).step_by(step)
    }

    fn current_slot(&mut self, now_secs: u64) -> &mut u64 {
        let cur = self.current_start(now_secs);
        let len = self.slots.len();
        let slot = &mut self.slots[((cur / self.step_secs) as usize) % len];
        if slot.0 != cur {
            *slot = (cur, 0);
        }
        &mut slot.1
    }

    /// Seconds from `now` until enough of the oldest sub-buckets have left
    /// the sum to drop it by at least `need`. At least 1.
    fn secs_until_released(&self, now_secs: u64, need: u64) -> u64 {
        let mut released = 0u64;
        for start in self.starts_in_window(now_secs) {
            released = released.saturating_add(self.count_at(start));
            if released >= need {
                let expiry = start + self.window_secs + self.step_secs;
                return expiry.saturating_sub(now_secs).max(1);
            }
        }
        // Unreachable while `need` never exceeds the sum; the whole window
        // plus a step is the longest any recorded unit can stay counted.
        (self.window_secs + self.step_secs).max(1)
    }

    /// Check whether `delta` more units fit under `limit`. If yes, record
    /// them in the current sub-bucket; if not, report how long until enough
    /// old sub-buckets expire for them to fit.
    pub fn check_and_increment(&mut self, now_secs: u64, delta: u64, limit: u64) -> WindowCheck {
        let used = self.current(now_secs);
        let would_be = used.saturating_add(delta);
        if would_be > limit {
            return WindowCheck::Full {
                retry_after_secs: self.secs_until_released(now_secs, would_be - limit),
            };
        }
        *self.current_slot(now_secs) += delta;
        WindowCheck::Ok
    }

    /// Record `delta` in the current sub-bucket without a check.
    pub fn add(&mut self, now_secs: u64, delta: u64) {
        let slot = self.current_slot(now_secs);
        *slot = slot.saturating_add(delta);
    }

    /// Undo `delta` from the current sub-bucket, saturating at zero. Only
    /// used to roll back an increment made at the same `now`.
    pub fn decrement(&mut self, now_secs: u64, delta: u64) {
        let slot = self.current_slot(now_secs);
        *slot = slot.saturating_sub(delta);
    }

    /// Undo `delta` recorded at `at_secs`, from the sub-bucket it landed
    /// in, saturating at zero. No-op once that sub-bucket's slot has been
    /// reused for a newer one: the unit already left the sum, and the slot
    /// now holds other requests' counts.
    pub fn refund(&mut self, at_secs: u64, delta: u64) {
        let start = self.current_start(at_secs);
        let len = self.slots.len();
        let slot = &mut self.slots[((start / self.step_secs) as usize) % len];
        if slot.0 == start {
            slot.1 = slot.1.saturating_sub(delta);
        }
    }

    /// Rolling usage at `now`.
    pub fn current(&self, now_secs: u64) -> u64 {
        self.starts_in_window(now_secs)
            .map(|start| self.count_at(start))
            .fold(0u64, u64::saturating_add)
    }

    /// Like [`FixedWindowCounter::is_exceeded`]: usage at or past `limit`
    /// is spent. The hint is how long until the sum drops below `limit`.
    pub fn is_exceeded(&self, now_secs: u64, limit: u64) -> Option<u64> {
        let used = self.current(now_secs);
        (used >= limit).then(|| self.secs_until_released(now_secs, used - limit + 1))
    }

    /// Seconds until everything counted now has left the window, which is
    /// when the counter is back to its initial state. Zero when nothing is
    /// counted.
    pub fn reset_secs(&self, now_secs: u64) -> u64 {
        self.starts_in_window(now_secs)
            .filter(|&start| self.count_at(start) > 0)
            .last()
            .map(|start| (start + self.window_secs + self.step_secs).saturating_sub(now_secs))
            .unwrap_or(0)
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

        assert!(w.is_exceeded(101, 2_000).is_none()); // 1500 < 2000
        assert!(w.is_exceeded(101, 1_000).is_some()); // 1500 > 1000
    }

    /// #950: usage landing exactly on the cap means the budget is gone.
    /// Admitting there let a caller whose committed usage was already the
    /// whole allowance spend one more full response.
    #[test]
    fn a_count_exactly_on_the_limit_is_exhausted() {
        let mut w = FixedWindowCounter::new(60);
        w.add(100, 10);
        assert_eq!(w.current(100), 10);
        assert!(
            w.is_exceeded(100, 10).is_some(),
            "committed usage == limit must stop admitting",
        );
        assert!(
            w.is_exceeded(100, 11).is_none(),
            "one token of budget left still admits",
        );
    }

    #[test]
    fn check_with_zero_delta_is_a_read_only_peek_that_succeeds() {
        let mut w = FixedWindowCounter::new(60);
        assert_eq!(w.check_and_increment(100, 0, 5), WindowCheck::Ok);
        assert_eq!(w.current(100), 0);
    }

    #[test]
    fn decrement_reduces_current_window_count_only() {
        let mut w = FixedWindowCounter::new(60);
        // Window is [60, 120). Two increments at second 100.
        w.check_and_increment(100, 1, 100);
        w.check_and_increment(100, 1, 100);
        assert_eq!(w.current(100), 2);

        // Roll back one. Concurrent siblings' counts should survive.
        w.decrement(100, 1);
        assert_eq!(w.current(100), 1);

        // Saturating: undoing more than was counted goes to zero,
        // never wraps.
        w.decrement(100, 100);
        assert_eq!(w.current(100), 0);
    }

    #[test]
    fn decrement_is_noop_after_window_rollover() {
        let mut w = FixedWindowCounter::new(60);
        // Counter is at 5 in window [60, 120).
        for _ in 0..5 {
            w.check_and_increment(100, 1, 100);
        }
        // Caller observed at second 100 then tries to roll back at
        // second 200 (window [180, 240)). The increment they'd undo
        // is in the *previous* window, which has already rolled. We
        // must NOT subtract from the new window — that would corrupt
        // unrelated counts.
        w.decrement(200, 1);
        assert_eq!(w.current(200), 0); // fresh window untouched
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

    // --- SlidingWindowCounter ------------------------------------------

    /// An epoch-aligned base well past zero, so `cur - window` never
    /// saturates and every sub-bucket in the sum is a real one.
    const T0: u64 = 1_700_000_000 - 1_700_000_000 % 3_600;

    #[test]
    fn sliding_sums_the_current_and_preceding_window_of_sub_buckets() {
        let mut w = SlidingWindowCounter::new(60, 10);
        w.add(T0 + 5, 1); // sub-bucket T0
        w.add(T0 + 25, 2); // sub-bucket T0+20
        w.add(T0 + 59, 4); // sub-bucket T0+50
        assert_eq!(w.current(T0 + 59), 7);
        // At T0+60..69 the sum starts at T0: all three still count.
        assert_eq!(w.current(T0 + 69), 7);
        // At T0+70 the sum starts at T0+10: the T0 sub-bucket left.
        assert_eq!(w.current(T0 + 70), 6);
        // At T0+90 it starts at T0+30: only T0+50 remains.
        assert_eq!(w.current(T0 + 90), 4);
        assert_eq!(w.current(T0 + 120), 0);
    }

    #[test]
    fn sliding_refuses_a_burst_across_the_fixed_boundary() {
        // A fixed window resets at T0+60 and admits 3 more at once; the
        // sliding one keeps counting the 3 admitted just before it.
        let mut fixed = FixedWindowCounter::new(60);
        let mut sliding = SlidingWindowCounter::new(60, 10);
        for _ in 0..3 {
            assert_eq!(fixed.check_and_increment(T0 + 58, 1, 3), WindowCheck::Ok);
            assert_eq!(sliding.check_and_increment(T0 + 58, 1, 3), WindowCheck::Ok);
        }
        assert_eq!(fixed.check_and_increment(T0 + 61, 1, 3), WindowCheck::Ok);
        assert_eq!(
            sliding.check_and_increment(T0 + 61, 1, 3),
            // The three sit in sub-bucket T0+50, which leaves the sum at
            // T0+50+60+10 = T0+120.
            WindowCheck::Full {
                retry_after_secs: 59
            }
        );
    }

    #[test]
    fn sliding_retry_is_the_wait_for_just_enough_old_sub_buckets() {
        let mut w = SlidingWindowCounter::new(60, 10);
        w.add(T0 + 3, 1); // leaves at T0+70
        w.add(T0 + 14, 2); // leaves at T0+80
        w.add(T0 + 31, 2); // leaves at T0+100
                           // Limit 5, used 5: one more needs 1 released → the T0 sub-bucket.
        assert_eq!(
            w.check_and_increment(T0 + 40, 1, 5),
            WindowCheck::Full {
                retry_after_secs: 30
            }
        );
        // Limit 3: one more needs 3 released → T0 and T0+10 together.
        assert_eq!(
            w.check_and_increment(T0 + 40, 1, 3),
            WindowCheck::Full {
                retry_after_secs: 40
            }
        );
        // Token form: used 5 against limit 4 needs the sum below 4, i.e.
        // 2 released → T0 and T0+10.
        assert_eq!(w.is_exceeded(T0 + 40, 4), Some(40));
        assert_eq!(w.is_exceeded(T0 + 40, 6), None);
        // Refusals record nothing.
        assert_eq!(w.current(T0 + 40), 5);
        // And the predicted instants are when it really admits again.
        assert_eq!(
            w.check_and_increment(T0 + 69, 1, 5),
            WindowCheck::Full {
                retry_after_secs: 1
            }
        );
        assert_eq!(w.check_and_increment(T0 + 70, 1, 5), WindowCheck::Ok);
    }

    #[test]
    fn sliding_release_is_never_before_the_window_and_at_most_a_step_after() {
        for offset in 0..10 {
            let mut w = SlidingWindowCounter::new(60, 10);
            let at = T0 + offset;
            assert_eq!(w.check_and_increment(at, 1, 1), WindowCheck::Ok);
            let mut t = at + 1;
            let mut hints = Vec::new();
            while let WindowCheck::Full { retry_after_secs } = w.check_and_increment(t, 1, 1) {
                hints.push((t, retry_after_secs));
                t += 1;
            }
            // Every refusal's hint named exactly the second it admits again.
            for (refused_at, hint) in hints {
                assert_eq!(
                    refused_at + hint,
                    t,
                    "offset {offset}, refused at {refused_at}"
                );
            }
            let released_after = t - at;
            assert!(
                (60..=70).contains(&released_after),
                "offset {offset}: released after {released_after}s"
            );
        }
    }

    #[test]
    fn sliding_decrement_undoes_only_the_current_sub_bucket() {
        let mut w = SlidingWindowCounter::new(60, 10);
        w.add(T0, 3);
        w.add(T0 + 10, 1);
        w.decrement(T0 + 10, 1);
        assert_eq!(w.current(T0 + 10), 3);
        w.decrement(T0 + 10, 5);
        assert_eq!(w.current(T0 + 10), 3, "saturates in its own sub-bucket");
    }

    #[test]
    fn sliding_reset_is_when_the_newest_usage_leaves() {
        let mut w = SlidingWindowCounter::new(60, 10);
        assert_eq!(w.reset_secs(T0), 0, "nothing counted, nothing to wait for");
        w.add(T0 + 2, 1);
        w.add(T0 + 33, 1); // sub-bucket T0+30 leaves at T0+100
        assert_eq!(w.reset_secs(T0 + 40), 60);
        assert_eq!(w.reset_secs(T0 + 99), 1);
        assert_eq!(w.reset_secs(T0 + 100), 0);
    }

    #[test]
    fn sliding_storage_does_not_grow_with_use() {
        let mut w = SlidingWindowCounter::new(3_600, 300);
        for t in (0..20_000).step_by(7) {
            w.add(T0 + t, 1_000);
        }
        assert_eq!(w.slots.len(), 13);
        // Only the last 13 sub-buckets are in the sum.
        let now = T0 + 19_999;
        let cur = now - now % 300;
        let expected: u64 = (0..20_000)
            .step_by(7)
            .filter(|t| T0 + t >= cur - 3_600)
            .count() as u64
            * 1_000;
        assert_eq!(w.current(now), expected);
    }
}
