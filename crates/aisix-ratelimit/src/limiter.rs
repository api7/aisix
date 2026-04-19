//! Two-phase limiter keyed on an opaque `key` (the caller's ApiKey id
//! in production).
//!
//! Phase 1 — **pre-commit**, called before the upstream request fires:
//! - check concurrency (acquire a permit or fail)
//! - check + increment RPM / RPD counters
//! - *check-only* TPM / TPD (we don't know the token cost yet)
//!
//! Phase 2 — **post-deduct**, called after the upstream response
//! completes:
//! - add actual `prompt_tokens + completion_tokens` to TPM / TPD
//! - release the concurrency permit
//!
//! The returned [`Reservation`] handle wraps the concurrency permit so
//! callers cannot forget to release on the error path — the permit is
//! released on drop if `commit_tokens` / `abort` isn't called.

use aisix_core::{RateLimit, RateLimitScope};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::clock::{Clock, SystemClock};
use crate::error::RateLimitError;
use crate::window::{FixedWindowCounter, WindowCheck};

const MINUTE_SECS: u64 = 60;
const DAY_SECS: u64 = 24 * 60 * 60;

/// Per-key state guarded by a single mutex. Hot path locks once per
/// request; each operation inside is O(1).
#[derive(Debug)]
struct KeyState {
    rpm: FixedWindowCounter,
    rpd: FixedWindowCounter,
    tpm: FixedWindowCounter,
    tpd: FixedWindowCounter,
    in_flight: u32,
}

impl KeyState {
    fn new() -> Self {
        Self {
            rpm: FixedWindowCounter::new(MINUTE_SECS),
            rpd: FixedWindowCounter::new(DAY_SECS),
            tpm: FixedWindowCounter::new(MINUTE_SECS),
            tpd: FixedWindowCounter::new(DAY_SECS),
            in_flight: 0,
        }
    }
}

/// Current window state for a single key, returned by [`Limiter::peek`].
/// Used by the proxy handlers to inject the `x-ratelimit-*` response
/// headers that OpenAI SDK clients expect.
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    pub rpm_limit: Option<u64>,
    pub rpm_used: u64,
    pub rpm_reset_secs: u64,
    pub tpm_limit: Option<u64>,
    pub tpm_used: u64,
    pub tpm_reset_secs: u64,
    pub concurrency_limit: Option<u32>,
    pub in_flight: u32,
}

impl RateLimitStatus {
    pub fn rpm_remaining(&self) -> Option<u64> {
        self.rpm_limit
            .map(|lim| lim.saturating_sub(self.rpm_used))
    }
    pub fn tpm_remaining(&self) -> Option<u64> {
        self.tpm_limit
            .map(|lim| lim.saturating_sub(self.tpm_used))
    }
}

pub struct Limiter<C: Clock = SystemClock> {
    states: DashMap<String, Arc<Mutex<KeyState>>>,
    clock: C,
}

impl Limiter<SystemClock> {
    pub fn new() -> Self {
        Self::with_clock(SystemClock)
    }
}

impl Default for Limiter<SystemClock> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Clock> Limiter<C> {
    pub fn with_clock(clock: C) -> Self {
        Self {
            states: DashMap::new(),
            clock,
        }
    }

    /// Snapshot of the current rate-limit state for a key, used to inject
    /// `x-ratelimit-*` response headers. Returns `None` if the key has
    /// never been seen (i.e. no counters yet — headers are meaningless).
    ///
    /// This is a **read-only** operation; it does not affect any counters.
    pub fn peek(&self, key: &str, limits: &aisix_core::RateLimit) -> Option<RateLimitStatus> {
        let now = self.clock.unix_secs();
        let state = self.states.get(key)?;
        let mut s = state.lock();

        // Roll counters so we're looking at the current window.
        let rpm_used = s.rpm.current(now);
        let tpm_used = s.tpm.current(now);
        let in_flight = s.in_flight;

        // Seconds remaining in the current minute-window. Zero if the
        // window just started or has already rolled.
        let minute_reset = MINUTE_SECS - (now % MINUTE_SECS);

        Some(RateLimitStatus {
            rpm_limit: limits.rpm,
            rpm_used,
            rpm_reset_secs: minute_reset,
            tpm_limit: limits.tpm,
            tpm_used,
            tpm_reset_secs: minute_reset,
            concurrency_limit: limits.concurrency,
            in_flight,
        })
    }

    fn state_for(&self, key: &str) -> Arc<Mutex<KeyState>> {
        if let Some(entry) = self.states.get(key) {
            return entry.clone();
        }
        self.states
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(KeyState::new())))
            .clone()
    }

    /// Pre-commit phase. Returns a [`Reservation`] that must be finalised
    /// via [`Limiter::commit_tokens`] or dropped to release the
    /// concurrency permit automatically.
    pub fn pre_commit(
        &self,
        key: &str,
        limits: &RateLimit,
    ) -> Result<Reservation<'_, C>, RateLimitError> {
        let now = self.clock.unix_secs();
        let state = self.state_for(key);
        let mut s = state.lock();

        // Concurrency first — cheapest and never consumes a window slot.
        if let Some(max) = limits.concurrency {
            if s.in_flight >= max {
                return Err(RateLimitError::Concurrency);
            }
        }

        // Token limits — checked but not incremented. We refuse new
        // requests if the previous minute/day already overran the cap.
        if let Some(max) = limits.tpm {
            if let Some(retry) = s.tpm.is_exceeded(now, max) {
                return Err(RateLimitError::Tokens {
                    scope: RateLimitScope::Tokens,
                    retry_after_secs: retry,
                });
            }
        }
        if let Some(max) = limits.tpd {
            if let Some(retry) = s.tpd.is_exceeded(now, max) {
                return Err(RateLimitError::Tokens {
                    scope: RateLimitScope::Tokens,
                    retry_after_secs: retry,
                });
            }
        }

        // Request limits — checked AND incremented.
        if let Some(max) = limits.rpm {
            if let WindowCheck::Full { retry_after_secs } = s.rpm.check_and_increment(now, 1, max) {
                return Err(RateLimitError::Requests {
                    scope: RateLimitScope::Requests,
                    retry_after_secs,
                });
            }
        }
        if let Some(max) = limits.rpd {
            if let WindowCheck::Full { retry_after_secs } = s.rpd.check_and_increment(now, 1, max) {
                // Compensate: we already incremented RPM above. Decrement
                // it so the caller's retry on a different day still
                // counts correctly. RPM would have rolled by then, so
                // this is primarily defensive.
                if s.rpm.current(now) > 0 {
                    // Roll back the increment we just made.
                    s.rpm = FixedWindowCounter::new(MINUTE_SECS);
                    if let Some(max) = limits.rpm {
                        let _ = s.rpm.check_and_increment(now, 0, max);
                    }
                }
                return Err(RateLimitError::Requests {
                    scope: RateLimitScope::Requests,
                    retry_after_secs,
                });
            }
        }

        s.in_flight += 1;
        drop(s);

        Ok(Reservation {
            limiter: self,
            key: key.to_string(),
            committed: false,
        })
    }
}

/// Reservation guard. Dropping without a `commit_tokens` call is still
/// safe — the concurrency permit is released, just no tokens are
/// counted.
pub struct Reservation<'a, C: Clock> {
    limiter: &'a Limiter<C>,
    key: String,
    committed: bool,
}

impl<'a, C: Clock> std::fmt::Debug for Reservation<'a, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reservation")
            .field("key", &self.key)
            .field("committed", &self.committed)
            .finish()
    }
}

impl<'a, C: Clock> Reservation<'a, C> {
    /// Post-deduct phase. Records the actual token cost against TPM/TPD
    /// and releases the concurrency permit.
    pub fn commit_tokens(mut self, tokens: u64) {
        let now = self.limiter.clock.unix_secs();
        let state = self.limiter.state_for(&self.key);
        let mut s = state.lock();
        s.tpm.add(now, tokens);
        s.tpd.add(now, tokens);
        s.in_flight = s.in_flight.saturating_sub(1);
        self.committed = true;
    }
}

impl<'a, C: Clock> Drop for Reservation<'a, C> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let state = self.limiter.state_for(&self.key);
        let mut s = state.lock();
        s.in_flight = s.in_flight.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::TestClock;

    fn limits(rpm: Option<u64>, tpm: Option<u64>, concurrency: Option<u32>) -> RateLimit {
        RateLimit {
            rpm,
            rpd: None,
            tpm,
            tpd: None,
            concurrency,
        }
    }

    #[test]
    fn rpm_caps_request_count_in_window() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits(Some(2), None, None);

        let _r1 = limiter.pre_commit("k1", &l).unwrap();
        let _r2 = limiter.pre_commit("k1", &l).unwrap();
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        match err {
            RateLimitError::Requests {
                retry_after_secs, ..
            } => {
                assert!(retry_after_secs > 0);
            }
            other => panic!("expected Requests, got {other:?}"),
        }
    }

    #[test]
    fn rpm_resets_after_window_rollover() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits(Some(1), None, None);

        let _r1 = limiter.pre_commit("k1", &l).unwrap();
        assert!(limiter.pre_commit("k1", &l).is_err());

        // Jump past the minute boundary.
        clock.advance(61);
        let _r2 = limiter.pre_commit("k1", &l).unwrap();
    }

    #[test]
    fn concurrency_limit_blocks_new_reservations() {
        let clock = TestClock::new(0);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits(None, None, Some(2));

        let r1 = limiter.pre_commit("k1", &l).unwrap();
        let r2 = limiter.pre_commit("k1", &l).unwrap();
        assert!(matches!(
            limiter.pre_commit("k1", &l).unwrap_err(),
            RateLimitError::Concurrency,
        ));

        // Drop r1 — concurrency should free up.
        drop(r1);
        let _r3 = limiter.pre_commit("k1", &l).unwrap();
        drop(r2);
    }

    #[test]
    fn token_commit_updates_post_deduct_counters() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits(Some(10), Some(1_000), None);

        let r1 = limiter.pre_commit("k1", &l).unwrap();
        r1.commit_tokens(600);

        // TPM now at 600. Next pre_commit with a strict TPM should still
        // succeed because 600 <= 1000.
        let _r2 = limiter.pre_commit("k1", &l).unwrap();
    }

    #[test]
    fn tpm_blocks_next_request_once_previous_exhausted_the_window() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits(Some(10), Some(1_000), None);

        let r1 = limiter.pre_commit("k1", &l).unwrap();
        r1.commit_tokens(1_500); // overshoot — allowed for the in-flight request

        // Next pre_commit sees tpm > 1000 and refuses.
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(matches!(err, RateLimitError::Tokens { .. }));

        clock.advance(61); // roll the window
        let _r2 = limiter.pre_commit("k1", &l).unwrap();
    }

    #[test]
    fn reservations_for_different_keys_do_not_collide() {
        let clock = TestClock::new(0);
        let limiter = Limiter::with_clock(clock);
        let l = limits(Some(1), None, None);

        let _r_a = limiter.pre_commit("alpha", &l).unwrap();
        let _r_b = limiter.pre_commit("beta", &l).unwrap();
    }

    #[test]
    fn drop_without_commit_still_releases_concurrency_permit() {
        let clock = TestClock::new(0);
        let limiter = Limiter::with_clock(clock);
        let l = limits(None, None, Some(1));

        {
            let _r = limiter.pre_commit("k1", &l).unwrap();
        } // dropped
        let _r2 = limiter.pre_commit("k1", &l).unwrap();
    }

    #[test]
    fn peek_returns_none_for_unknown_key() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock);
        assert!(limiter.peek("unknown", &RateLimit::default()).is_none());
    }

    #[test]
    fn peek_reports_current_window_counts() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits(Some(60), Some(100_000), Some(10));

        let r = limiter.pre_commit("k1", &l).unwrap();
        r.commit_tokens(500);

        let status = limiter.peek("k1", &l).unwrap();
        assert_eq!(status.rpm_limit, Some(60));
        assert_eq!(status.rpm_used, 1);
        assert_eq!(status.rpm_remaining(), Some(59));
        assert_eq!(status.tpm_limit, Some(100_000));
        assert_eq!(status.tpm_used, 500);
        assert_eq!(status.tpm_remaining(), Some(99_500));
        assert_eq!(status.in_flight, 0); // committed → released
    }

    #[test]
    fn peek_reflects_in_flight_count_during_dispatch() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock);
        let l = limits(None, None, Some(5));

        let _r1 = limiter.pre_commit("k1", &l).unwrap();
        let _r2 = limiter.pre_commit("k1", &l).unwrap();
        let status = limiter.peek("k1", &l).unwrap();
        assert_eq!(status.in_flight, 2);
        assert_eq!(status.concurrency_limit, Some(5));
    }

    #[test]
    fn no_limits_means_no_rejections() {
        let clock = TestClock::new(0);
        let limiter = Limiter::with_clock(clock);
        let l = RateLimit::default();

        for _ in 0..100 {
            let r = limiter.pre_commit("k1", &l).unwrap();
            r.commit_tokens(1_000);
        }
    }
}
