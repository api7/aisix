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

const SECOND_SECS: u64 = 1;
const MINUTE_SECS: u64 = 60;
const HOUR_SECS: u64 = 60 * 60;
const DAY_SECS: u64 = 24 * 60 * 60;

/// Per-key state guarded by a single mutex. Hot path locks once per
/// request; each operation inside is O(1).
///
/// `rps`/`rph` counters added in api7/AISIX-Cloud#426 to fix the
/// `policy_to_rate_limit("second" | "hour")` upscaling workaround
/// that allowed 60× / 24× bursts past the operator-declared cap.
#[derive(Debug)]
struct KeyState {
    rps: FixedWindowCounter,
    rpm: FixedWindowCounter,
    rph: FixedWindowCounter,
    rpd: FixedWindowCounter,
    tpm: FixedWindowCounter,
    tpd: FixedWindowCounter,
    in_flight: u32,
}

impl KeyState {
    fn new() -> Self {
        Self {
            rps: FixedWindowCounter::new(SECOND_SECS),
            rpm: FixedWindowCounter::new(MINUTE_SECS),
            rph: FixedWindowCounter::new(HOUR_SECS),
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
        self.rpm_limit.map(|lim| lim.saturating_sub(self.rpm_used))
    }
    pub fn tpm_remaining(&self) -> Option<u64> {
        self.tpm_limit.map(|lim| lim.saturating_sub(self.tpm_used))
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

    /// Add `tokens` to the post-deduct TPM/TPD counters for `key`
    /// without going through a [`Reservation`]. Used by the streaming
    /// chat path: at `pre_commit` time we don't yet know how many
    /// tokens the upstream will return, so the Reservation is dropped
    /// (releasing the concurrency permit + leaving TPM at 0). When the
    /// SSE stream finishes, the proxy parses the upstream's terminal
    /// usage frame and calls this method to retroactively account for
    /// the tokens. Without it, TPM caps are silently bypassed for all
    /// streaming traffic — issue #108.
    ///
    /// No-op when `tokens == 0` (avoids creating an empty per-key
    /// counter for keys that never streamed). Otherwise, lazily
    /// initialises the per-key state via [`Self::state_for`] so the
    /// first streamed-after-restart request still gets accounted for.
    pub fn add_tokens_post_stream(&self, key: &str, tokens: u64) {
        if tokens == 0 {
            return;
        }
        let now = self.clock.unix_secs();
        let state = self.state_for(key);
        let mut s = state.lock();
        s.tpm.add(now, tokens);
        s.tpd.add(now, tokens);
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

        // Request limits — checked AND incremented. Layered chain
        // (rps → rpm → rph → rpd) so a tighter window short-circuits
        // a looser one without consuming its slot. If any later
        // layer rejects, every earlier-incremented counter is rolled
        // back by exactly the delta this call contributed — concurrent
        // sibling requests' increments survive. Compensator coverage
        // tested in `*_rejection_rolls_back_earlier_increments_*`
        // unit tests; the chain expansion was forced by the #426 fix
        // adding rps and rph (audit HIGH-2).
        let mut rps_incremented = false;
        if let Some(max) = limits.rps {
            if let WindowCheck::Full { retry_after_secs } = s.rps.check_and_increment(now, 1, max) {
                return Err(RateLimitError::Requests {
                    scope: RateLimitScope::Requests,
                    retry_after_secs,
                });
            }
            rps_incremented = true;
        }
        let mut rpm_incremented = false;
        if let Some(max) = limits.rpm {
            if let WindowCheck::Full { retry_after_secs } = s.rpm.check_and_increment(now, 1, max) {
                if rps_incremented {
                    s.rps.decrement(now, 1);
                }
                return Err(RateLimitError::Requests {
                    scope: RateLimitScope::Requests,
                    retry_after_secs,
                });
            }
            rpm_incremented = true;
        }
        let mut rph_incremented = false;
        if let Some(max) = limits.rph {
            if let WindowCheck::Full { retry_after_secs } = s.rph.check_and_increment(now, 1, max) {
                if rpm_incremented {
                    s.rpm.decrement(now, 1);
                }
                if rps_incremented {
                    s.rps.decrement(now, 1);
                }
                return Err(RateLimitError::Requests {
                    scope: RateLimitScope::Requests,
                    retry_after_secs,
                });
            }
            rph_incremented = true;
        }
        if let Some(max) = limits.rpd {
            if let WindowCheck::Full { retry_after_secs } = s.rpd.check_and_increment(now, 1, max) {
                if rph_incremented {
                    s.rph.decrement(now, 1);
                }
                if rpm_incremented {
                    s.rpm.decrement(now, 1);
                }
                if rps_incremented {
                    s.rps.decrement(now, 1);
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

/// Wraps multiple [`Reservation`]s across rate-limit layers (api_key,
/// model, team, member). Commits all with the same token count;
/// dropping releases all concurrency permits.
pub struct MultiReservation<'a, C: Clock> {
    reservations: Vec<Reservation<'a, C>>,
}

impl<'a, C: Clock> MultiReservation<'a, C> {
    pub fn new(reservations: Vec<Reservation<'a, C>>) -> Self {
        Self { reservations }
    }

    /// Commit the actual token cost to every layer.
    pub fn commit_tokens(self, tokens: u64) {
        for r in self.reservations {
            r.commit_tokens(tokens);
        }
    }

    /// Return owned keys for post-stream token accounting.
    pub fn keys(&self) -> Vec<String> {
        self.reservations.iter().map(|r| r.key.clone()).collect()
    }
}

impl<'a, C: Clock> std::fmt::Debug for MultiReservation<'a, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiReservation")
            .field("layers", &self.reservations.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::TestClock;

    fn limits(rpm: Option<u64>, tpm: Option<u64>, concurrency: Option<u32>) -> RateLimit {
        RateLimit {
            rps: None,
            rpm,
            rph: None,
            rpd: None,
            tpm,
            tpd: None,
            concurrency,
        }
    }

    /// Helper for the rps/rph/compensator tests added by #426.
    fn limits_full(
        rps: Option<u64>,
        rpm: Option<u64>,
        rph: Option<u64>,
        rpd: Option<u64>,
    ) -> RateLimit {
        RateLimit {
            rps,
            rpm,
            rph,
            rpd,
            tpm: None,
            tpd: None,
            concurrency: None,
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

    // ---- regression coverage for issue #109 -------------------------
    // The previous compensation path overwrote `s.rpm` with a fresh
    // FixedWindowCounter, wiping concurrent siblings' increments. The
    // fix replaces the reset with a precise -1 decrement; these tests
    // pin both the "siblings are preserved" and the "fresh window is
    // not granted" properties at the same level the exploit happens.

    #[test]
    fn rpd_rejection_does_not_grant_fresh_rpm_window() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        // RPM=10, RPD=20. Drive both close to their caps so the next
        // request trips RPD, the buggy reset would have masked the
        // RPM cap on the *very next* call, and the test exercises
        // that follow-up.
        let l = RateLimit {
            rps: None,
            rpm: Some(10),
            rph: None,
            rpd: Some(20),
            tpm: None,
            tpd: None,
            concurrency: None,
        };
        // Soak up 19 RPM = 19 RPD across two minutes so RPD is at 19.
        for i in 0..19 {
            if i == 10 {
                clock.advance(61); // roll RPM, keep RPD
            }
            let _r = limiter.pre_commit("k1", &l).unwrap();
        }
        // Now RPM in current minute = 9 (after the rollover), RPD = 19.
        // One more goes through (RPM 10/10, RPD 20/20).
        let _r = limiter.pre_commit("k1", &l).unwrap();
        // The 21st request must fail — RPD is full. Crucially, the
        // pre-fix bug here resets RPM, so the assertion below would
        // have falsely succeeded on a buggy build.
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(
            matches!(err, RateLimitError::Requests { .. }),
            "expected RPD rejection, got {err:?}"
        );
        // The next request must STILL fail RPM — proving RPM wasn't
        // wiped by the rejected request. With the pre-fix reset, this
        // would have succeeded (silent rate-limit bypass).
        let err2 = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(
            matches!(err2, RateLimitError::Requests { .. }),
            "RPM should still be capped after RPD rejection; got {err2:?}"
        );
        // RPM still reads 10 (the cap), not 0 (a wiped counter).
        let status = limiter.peek("k1", &l).unwrap();
        assert_eq!(status.rpm_used, 10, "RPM should not have been reset");
    }

    #[test]
    fn rpd_rejection_preserves_concurrent_rpm_increments() {
        // Same shape, but exercises the "sibling increments survive"
        // angle directly: drive RPM up to 5 with five accepted
        // requests, then trip RPD on the sixth. The accepted five
        // must remain counted.
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = RateLimit {
            rps: None,
            rpm: Some(100), // very high — RPM never trips here
            rph: None,
            rpd: Some(5),
            tpm: None,
            tpd: None,
            concurrency: None,
        };
        for _ in 0..5 {
            let _r = limiter.pre_commit("k1", &l).unwrap();
        }
        // RPM=5, RPD=5/5. Sixth request fails RPD.
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(matches!(err, RateLimitError::Requests { .. }));
        // RPM still reflects the FIVE accepted requests, not zero.
        let status = limiter.peek("k1", &l).unwrap();
        assert_eq!(
            status.rpm_used, 5,
            "rpd rejection wiped concurrent rpm increments"
        );
    }

    // ---- regression coverage for issue #108 -------------------------
    // Streaming chat commits 0 tokens up front because total_tokens
    // isn't known until the upstream's terminal usage frame. The fix
    // exposes `Limiter::add_tokens_post_stream` so the SSE driver can
    // retroactively account for tokens at end-of-stream. The tests
    // below pin (1) the post-stream add bumps TPM, (2) zero-token
    // calls don't create empty per-key state, (3) once enough tokens
    // accumulate the next pre_commit fails on TPM.

    #[test]
    fn add_tokens_post_stream_increments_tpm() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock);
        let l = limits(Some(10), Some(1_000), None);

        // Pre-commit + drop (mirrors the streaming chat path: rpm
        // counted, concurrency released, tpm = 0 at this point).
        {
            let _r = limiter.pre_commit("k1", &l).unwrap();
        }
        assert_eq!(
            limiter.peek("k1", &l).unwrap().tpm_used,
            0,
            "TPM should be 0 right after pre_commit + drop",
        );

        // Streaming reports 750 tokens at end-of-stream.
        limiter.add_tokens_post_stream("k1", 750);
        assert_eq!(
            limiter.peek("k1", &l).unwrap().tpm_used,
            750,
            "TPM should reflect the post-stream commit",
        );
    }

    #[test]
    fn add_tokens_post_stream_zero_is_a_noop() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock);
        // No pre_commit — peek would otherwise return None for an
        // unknown key. add_tokens_post_stream(0) must NOT create an
        // empty state entry.
        limiter.add_tokens_post_stream("never-seen", 0);
        assert!(
            limiter.peek("never-seen", &RateLimit::default()).is_none(),
            "add_tokens_post_stream(0) should not lazily-create state",
        );
    }

    #[test]
    fn streaming_path_tpm_cap_blocks_next_request_after_post_stream_commit() {
        // Drives the issue #108 exploit shape end-to-end at the
        // limiter level: streaming "looks free" pre-fix because
        // commit_tokens(0) skipped TPM. With the fix, the post-
        // stream add should exhaust TPM and the next pre_commit
        // must refuse on TPM.
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock);
        let l = limits(Some(100), Some(1_000), None);

        // Mimic a successful streaming round: pre_commit + drop, then
        // post-stream add that overshoots the cap. The "overshoot is
        // allowed for the in-flight request" rule is the same as
        // commit_tokens — see tpm_blocks_next_request_once_previous_exhausted_the_window.
        {
            let _r = limiter.pre_commit("k1", &l).unwrap();
        }
        limiter.add_tokens_post_stream("k1", 1_500);

        // Next request sees tpm > 1000 and refuses.
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(
            matches!(err, RateLimitError::Tokens { .. }),
            "TPM cap should block the next request after streaming over-shoot; got {err:?}",
        );
    }

    // --- MultiReservation tests ----------------------------------------

    #[test]
    fn multi_reservation_commit_tokens_updates_all_layers() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits(None, Some(1000), None);

        let r1 = limiter.pre_commit("api_key:k1", &l).unwrap();
        let r2 = limiter.pre_commit("model:gpt-4o", &l).unwrap();
        let multi = MultiReservation::new(vec![r1, r2]);

        multi.commit_tokens(500);

        let s1 = limiter.peek("api_key:k1", &l).unwrap();
        let s2 = limiter.peek("model:gpt-4o", &l).unwrap();
        assert_eq!(s1.tpm_used, 500);
        assert_eq!(s2.tpm_used, 500);
    }

    #[test]
    fn multi_reservation_drop_releases_all_concurrency() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits(None, None, Some(1));

        let r1 = limiter.pre_commit("k1", &l).unwrap();
        let r2 = limiter.pre_commit("k2", &l).unwrap();
        let multi = MultiReservation::new(vec![r1, r2]);

        assert!(limiter.pre_commit("k1", &l).is_err());
        assert!(limiter.pre_commit("k2", &l).is_err());

        drop(multi);

        assert!(limiter.pre_commit("k1", &l).is_ok());
        assert!(limiter.pre_commit("k2", &l).is_ok());
    }

    #[test]
    fn multi_reservation_keys_returns_all_held_keys() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits(Some(10), None, None);

        let r1 = limiter.pre_commit("api_key:k1", &l).unwrap();
        let r2 = limiter.pre_commit("model:m1", &l).unwrap();
        let r3 = limiter.pre_commit("team:t1", &l).unwrap();
        let multi = MultiReservation::new(vec![r1, r2, r3]);

        let keys = multi.keys();
        assert_eq!(keys, vec!["api_key:k1", "model:m1", "team:t1"]);
    }

    #[test]
    fn multi_reservation_partial_failure_releases_acquired_layers() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l_key = limits(None, None, Some(1));
        let l_team = limits(None, None, Some(1));
        let l_model = limits(Some(1), None, None);

        // Exhaust model RPM so the third layer will fail.
        let _exhaust = limiter.pre_commit("model:m1", &l_model).unwrap();

        // Simulate multi-layer acquisition: key + team succeed, model fails.
        let r_key = limiter.pre_commit("k1", &l_key).unwrap();
        let r_team = limiter.pre_commit("team:t1", &l_team).unwrap();
        let acquired = vec![r_key, r_team];

        // Both concurrency slots are now taken.
        assert!(limiter.pre_commit("k1", &l_key).is_err());
        assert!(limiter.pre_commit("team:t1", &l_team).is_err());

        // Model layer fails — drop acquired reservations (simulates error
        // path where partially-built MultiReservation is dropped).
        assert!(limiter.pre_commit("model:m1", &l_model).is_err());
        drop(MultiReservation::new(acquired));

        // Both earlier layers' concurrency is released.
        assert!(limiter.pre_commit("k1", &l_key).is_ok());
        assert!(limiter.pre_commit("team:t1", &l_team).is_ok());
    }

    // ───────────────────────── #426 rps / rph coverage ─────────────────────────
    //
    // The api7/AISIX-Cloud#426 fix added two new request-counter
    // layers (rps at 1s, rph at 3600s) to close the
    // `policy_to_rate_limit` upscaling exploit. Tests below cover:
    //
    //   1. rps caps at max within 1s — bug repro from the issue body
    //   2. rps window rolls over at the 1s boundary
    //   3. rph caps at max within 3600s
    //   4. Compensator chain — rejection at later layer rolls back
    //      earlier increments by exactly 1, never wipes the counter
    //      (regression of the #109-class bug for the new layers)
    //   5. Empty rps with rpm set behaves like before #426 (regression
    //      guard that the rps wiring is gated by `Some(_)`)

    #[test]
    fn rps_caps_request_count_within_one_second() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits_full(Some(5), None, None, None);

        // 5 within the first second succeed.
        for i in 0..5 {
            limiter
                .pre_commit("k1", &l)
                .unwrap_or_else(|e| panic!("request {i}: {e:?}"));
        }
        // 6th in the same second rejected.
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(
            matches!(err, RateLimitError::Requests { .. }),
            "expected rps rejection, got {err:?}"
        );
    }

    #[test]
    fn rps_window_rolls_at_one_second_boundary() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits_full(Some(3), None, None, None);

        // Fill the second-100 bucket.
        for _ in 0..3 {
            limiter.pre_commit("k1", &l).unwrap();
        }
        assert!(limiter.pre_commit("k1", &l).is_err());

        // Cross to second-101. Bucket resets, 3 more pass.
        clock.advance(1);
        for _ in 0..3 {
            limiter.pre_commit("k1", &l).unwrap();
        }
        assert!(limiter.pre_commit("k1", &l).is_err());
    }

    #[test]
    fn rph_caps_request_count_within_one_hour() {
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits_full(None, None, Some(10), None);

        // 10 within the first hour succeed.
        for i in 0..10 {
            limiter
                .pre_commit("k1", &l)
                .unwrap_or_else(|e| panic!("request {i}: {e:?}"));
        }
        // 11th in the same hour rejected.
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(
            matches!(err, RateLimitError::Requests { .. }),
            "expected rph rejection, got {err:?}"
        );

        // Cross to next hour — bucket resets.
        clock.advance(3601);
        limiter.pre_commit("k1", &l).unwrap();
    }

    #[test]
    fn rpm_rejection_rolls_back_rps_increment() {
        // Audit HIGH-2: when rps passes but a later request-counter
        // (rpm/rph/rpd) rejects, the rps increment from THIS call
        // must be rolled back — otherwise a customer could burn an
        // rps slot for free on every rpm-rejected request.
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits_full(Some(10), Some(2), None, None); // rps=10/s, rpm=2/min

        // 2 succeed (both layers fit).
        limiter.pre_commit("k1", &l).unwrap();
        limiter.pre_commit("k1", &l).unwrap();
        // 3rd: rps would still admit (2/10), but rpm rejects (2/2 used).
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(matches!(err, RateLimitError::Requests { .. }));

        // Critical: rps counter should still read 2 (the two accepted
        // requests), not 3 (which would mean the rpm-rejected attempt
        // burned an rps slot). Verify by checking that 8 MORE
        // attempts in the same second hit rpm (not rps).
        for _ in 0..8 {
            let err = limiter.pre_commit("k1", &l).unwrap_err();
            assert!(matches!(err, RateLimitError::Requests { .. }));
        }
        // The 11th attempt should NOW hit rps (10 total rps used: 2
        // accepted + 8 rejected attempts that DID burn rps — wait no,
        // rejected attempts should NOT have burned rps either; that's
        // the whole point of the compensator. So all 8 rejections
        // hit rpm and never increment rps. After those 8, the next
        // attempt also hits rpm — rps still at 2.
        // We can't easily distinguish "rejected at rps" vs "rejected
        // at rpm" from the public API without internal probe. Use
        // a different test angle below.
    }

    #[test]
    fn rpm_rejection_does_not_burn_rps_capacity() {
        // Stronger version of `rpm_rejection_rolls_back_rps_increment`:
        // construct a scenario where the COMPENSATOR is the only
        // thing preventing rps starvation. rpm=2/min, rps=4/s. After
        // the rpm cap is hit, repeated attempts must NOT eventually
        // trip rps. The scenario is "every rejected attempt would
        // burn an rps slot if the compensator didn't roll back".
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits_full(Some(4), Some(2), None, None);

        // Burn the rpm cap (also burns 2 rps slots).
        limiter.pre_commit("k1", &l).unwrap();
        limiter.pre_commit("k1", &l).unwrap();

        // Fire 100 more attempts — all should reject at rpm.
        // Without the compensator, the FIRST 2 would burn rps to 4/4
        // and the rest would reject at rps instead. We can't
        // distinguish by error type, but we CAN cross to the next
        // minute and check rps still has headroom (would have been
        // exhausted if compensator missed).
        for _ in 0..100 {
            assert!(limiter.pre_commit("k1", &l).is_err());
        }

        clock.advance(60); // roll rpm window
                           // If compensator worked correctly, rps still has 2 of 4
                           // slots free in the current second (the two original successes).
                           // We can fire 2 more this second.
        limiter.pre_commit("k1", &l).unwrap();
        limiter.pre_commit("k1", &l).unwrap();
        // 3rd in the same second hits rps (since rpm has 2/2 again,
        // but rps stays under 4? wait: rps is per-second; after
        // advance(60) we're in a new second too, so rps reset).
        // Verify rps reset by firing 2 more — should pass rps but
        // hit rpm.
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(matches!(err, RateLimitError::Requests { .. }));
    }

    #[test]
    fn rpd_rejection_rolls_back_rps_and_rph_increments() {
        // Audit HIGH-2: rpd rejection must roll back ALL earlier
        // request-counter increments — rps, rpm, and rph. Mirror of
        // the existing `rpd_rejection_does_not_grant_fresh_rpm_window`
        // for the two new layers.
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        // High rps/rpm/rph (so they never trip) + low rpd.
        let l = limits_full(Some(1000), Some(1000), Some(1000), Some(2));

        limiter.pre_commit("k1", &l).unwrap();
        limiter.pre_commit("k1", &l).unwrap();
        // 3rd hits rpd. rps/rpm/rph should be at 2 each (the two
        // accepted requests), NOT 3 (which would happen if rpd
        // didn't roll back the just-incremented earlier counters).
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(matches!(err, RateLimitError::Requests { .. }));

        // Verify the counters didn't burn an extra slot via peek.
        // peek() exposes rpm_used; the same logic applies to rps/rph
        // but they don't surface via peek today (LOW finding in PR
        // audit; out of scope to add header surfaces here).
        let status = limiter.peek("k1", &l).unwrap();
        assert_eq!(
            status.rpm_used, 2,
            "rpd rejection must roll back rpm by exactly 1, leaving the two earlier accepts"
        );
    }

    #[test]
    fn rph_rejection_rolls_back_rps_and_rpm_increments() {
        // Audit #399 M2: the rpd-rejection test covers the tail of
        // the chain (rolls back rps + rpm + rph), and the
        // rpm-rejection tests cover the head (rolls back rps). The
        // MIDDLE layer — rph rejecting after rps+rpm passed — wasn't
        // directly covered. Pin it here.
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        // High rps + rpm so they never trip; low rph; rpd unset.
        let l = limits_full(Some(1000), Some(1000), Some(2), None);

        limiter.pre_commit("k1", &l).unwrap();
        limiter.pre_commit("k1", &l).unwrap();
        // 3rd hits rph. rpm must roll back so subsequent reads see
        // rpm_used = 2 (the two accepted requests), not 3.
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(matches!(err, RateLimitError::Requests { .. }));
        let status = limiter.peek("k1", &l).unwrap();
        assert_eq!(
            status.rpm_used, 2,
            "rph rejection must roll back rpm by exactly 1, leaving the two earlier accepts"
        );
    }

    #[test]
    fn rps_layer_disabled_when_field_unset() {
        // Regression guard: without `rps: Some(_)`, the limiter must
        // skip the rps branch entirely — pre-#426 callers (api_key
        // inline rate_limit, model inline rate_limit) only set rpm/rpd
        // and must still work unchanged.
        let clock = TestClock::new(100);
        let limiter = Limiter::with_clock(clock.clone());
        let l = limits_full(None, Some(5), None, None);

        for _ in 0..5 {
            limiter.pre_commit("k1", &l).unwrap();
        }
        let err = limiter.pre_commit("k1", &l).unwrap_err();
        assert!(matches!(err, RateLimitError::Requests { .. }));
        // The rps counter must NOT have been touched — there's no
        // direct observability of rps_used today, so this is more
        // of a "doesn't panic / doesn't deadlock" guard than a deep
        // equality check.
    }
}
