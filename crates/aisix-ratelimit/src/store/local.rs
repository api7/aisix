//! In-process counter store — the historical, default backend.
//!
//! Behaviour-identical to the pre-#798 limiter: a `DashMap` of per-key
//! window counters guarded by one `parking_lot::Mutex` each. State
//! is per-replica and not shared, so a multi-replica cluster multiplies
//! every limit by the replica count — exactly what [`super::redis`]
//! exists to fix. `member` is ignored here (concurrency is a plain
//! `in_flight` counter).

use aisix_core::{RateLimit, RateLimitScope, RateLimitWindowType};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::Mutex;
use std::sync::Arc;

use super::{
    request_dims, sliding_step_secs, token_dims, Dim, RateStore, DAY_SECS, DIM_RPD, DIM_RPH,
    DIM_RPM, DIM_RPS, DIM_TPD, DIM_TPM, HOUR_SECS, MINUTE_SECS, SECOND_SECS,
};
use crate::clock::{Clock, SystemClock};
use crate::error::{LimitDetail, RateLimitError};
use crate::limiter::RateLimitStatus;
use crate::window::{FixedWindowCounter, SlidingWindowCounter, WindowCheck};

/// Per-key state guarded by a single mutex. Hot path locks once per
/// request; each operation inside is O(1) for a fixed window and
/// O(window / step) for a sliding one.
#[derive(Debug)]
struct KeyState {
    rps: FixedWindowCounter,
    rpm: FixedWindowCounter,
    rph: FixedWindowCounter,
    rpd: FixedWindowCounter,
    tpm: FixedWindowCounter,
    tpd: FixedWindowCounter,
    /// Allocated on first sliding use, so a bucket that only ever counts
    /// fixed windows carries no sub-buckets. Separate from the fixed
    /// counters: switching a limit's `window_type` starts from zero.
    sliding: Option<Box<SlidingState>>,
    in_flight: u32,
}

#[derive(Debug)]
struct SlidingState {
    rpm: SlidingWindowCounter,
    rph: SlidingWindowCounter,
    rpd: SlidingWindowCounter,
    tpm: SlidingWindowCounter,
    tpd: SlidingWindowCounter,
}

impl SlidingState {
    fn new() -> Self {
        let counter = |window| SlidingWindowCounter::new(window, sliding_step_secs(window));
        Self {
            rpm: counter(MINUTE_SECS),
            rph: counter(HOUR_SECS),
            rpd: counter(DAY_SECS),
            tpm: counter(MINUTE_SECS),
            tpd: counter(DAY_SECS),
        }
    }

    fn counter(&mut self, name: &str) -> &mut SlidingWindowCounter {
        match name {
            DIM_RPM => &mut self.rpm,
            DIM_RPH => &mut self.rph,
            DIM_RPD => &mut self.rpd,
            DIM_TPM => &mut self.tpm,
            DIM_TPD => &mut self.tpd,
            other => unreachable!("dimension {other} has no sliding window"),
        }
    }
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
            sliding: None,
            in_flight: 0,
        }
    }

    fn fixed(&mut self, name: &str) -> &mut FixedWindowCounter {
        match name {
            DIM_RPS => &mut self.rps,
            DIM_RPM => &mut self.rpm,
            DIM_RPH => &mut self.rph,
            DIM_RPD => &mut self.rpd,
            DIM_TPM => &mut self.tpm,
            DIM_TPD => &mut self.tpd,
            other => unreachable!("unknown dimension {other}"),
        }
    }

    fn sliding(&mut self, name: &str) -> &mut SlidingWindowCounter {
        self.sliding
            .get_or_insert_with(|| Box::new(SlidingState::new()))
            .counter(name)
    }

    fn check_and_increment(&mut self, d: &Dim, now: u64) -> WindowCheck {
        match d.step_secs {
            None => self.fixed(d.name).check_and_increment(now, 1, d.limit),
            Some(_) => self.sliding(d.name).check_and_increment(now, 1, d.limit),
        }
    }

    fn decrement(&mut self, d: &Dim, now: u64) {
        match d.step_secs {
            None => self.fixed(d.name).decrement(now, 1),
            Some(_) => self.sliding(d.name).decrement(now, 1),
        }
    }

    fn current(&mut self, d: &Dim, now: u64) -> u64 {
        match d.step_secs {
            None => self.fixed(d.name).current(now),
            Some(_) => self.sliding(d.name).current(now),
        }
    }

    fn is_exceeded(&mut self, d: &Dim, now: u64) -> Option<u64> {
        match d.step_secs {
            None => self.fixed(d.name).is_exceeded(now, d.limit),
            Some(_) => self.sliding(d.name).is_exceeded(now, d.limit),
        }
    }

    fn add_tokens(&mut self, now: u64, tokens: u64, window_type: RateLimitWindowType) {
        for name in [DIM_TPM, DIM_TPD] {
            match window_type {
                RateLimitWindowType::Fixed => self.fixed(name).add(now, tokens),
                RateLimitWindowType::Sliding => self.sliding(name).add(now, tokens),
            }
        }
    }
}

/// Per-process window store.
pub struct LocalStore<C: Clock = SystemClock> {
    states: DashMap<String, Arc<Mutex<KeyState>>>,
    clock: C,
}

impl LocalStore<SystemClock> {
    pub fn new() -> Self {
        Self::with_clock(SystemClock)
    }
}

impl Default for LocalStore<SystemClock> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Clock> LocalStore<C> {
    pub fn with_clock(clock: C) -> Self {
        Self {
            states: DashMap::new(),
            clock,
        }
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
}

#[async_trait]
impl<C: Clock> RateStore for LocalStore<C> {
    async fn acquire(
        &self,
        key: &str,
        limits: &RateLimit,
        _member: &str,
    ) -> Result<(), RateLimitError> {
        let now = self.clock.unix_secs();
        let state = self.state_for(key);
        let mut s = state.lock();

        // Concurrency first — cheapest and never consumes a window slot.
        if let Some(max) = limits.concurrency {
            if s.in_flight >= max {
                return Err(RateLimitError::Concurrency {
                    detail: LimitDetail::concurrency(u64::from(max), u64::from(s.in_flight)),
                });
            }
        }

        // Token limits — checked but not incremented. We refuse new
        // requests if the window already reached the cap.
        for d in token_dims(limits) {
            if let Some(retry) = s.is_exceeded(&d, now) {
                return Err(RateLimitError::Tokens {
                    scope: RateLimitScope::Tokens,
                    detail: LimitDetail::window(d.name, d.limit, s.current(&d, now), retry),
                });
            }
        }

        // Request limits — checked AND incremented. Layered chain
        // (rps → rpm → rph → rpd) so a tighter window short-circuits a
        // looser one without consuming its slot. If any later layer
        // rejects, every earlier-incremented counter is rolled back by
        // exactly the delta this call contributed — concurrent sibling
        // requests' increments survive.
        //
        // Each rejection first reads the refused window's own state for
        // the 429's `x-ratelimit-*` headers, before the roll-backs.
        let dims = request_dims(limits);
        for (i, d) in dims.iter().enumerate() {
            if let WindowCheck::Full { retry_after_secs } = s.check_and_increment(d, now) {
                let detail =
                    LimitDetail::window(d.name, d.limit, s.current(d, now), retry_after_secs);
                for earlier in &dims[..i] {
                    s.decrement(earlier, now);
                }
                return Err(RateLimitError::Requests {
                    scope: RateLimitScope::Requests,
                    detail,
                });
            }
        }

        s.in_flight += 1;
        Ok(())
    }

    async fn commit(
        &self,
        key: &str,
        tokens: u64,
        _member: &str,
        window_type: RateLimitWindowType,
    ) {
        let now = self.clock.unix_secs();
        let state = self.state_for(key);
        let mut s = state.lock();
        s.add_tokens(now, tokens, window_type);
        s.in_flight = s.in_flight.saturating_sub(1);
    }

    fn release(&self, key: &str, _member: &str) {
        // Non-inserting: a release for a never-acquired bucket is a no-op,
        // so the Redis store's belt-and-suspenders local release on the
        // happy path doesn't pollute the local map with empty state.
        if let Some(state) = self.states.get(key) {
            let mut s = state.lock();
            s.in_flight = s.in_flight.saturating_sub(1);
        }
    }

    fn add_tokens(&self, key: &str, tokens: u64, window_type: RateLimitWindowType) {
        if tokens == 0 {
            return;
        }
        let now = self.clock.unix_secs();
        let state = self.state_for(key);
        let mut s = state.lock();
        s.add_tokens(now, tokens, window_type);
    }

    /// Never shrinks: a key's state is kept for the life of the process.
    fn local_key_count(&self) -> Option<u64> {
        Some(self.states.len() as u64)
    }

    async fn peek(&self, key: &str, limits: &RateLimit) -> Option<RateLimitStatus> {
        let now = self.clock.unix_secs();
        let state = self.states.get(key)?;
        let mut s = state.lock();

        let in_flight = s.in_flight;

        let (rpm_used, rpm_reset_secs, tpm_used, tpm_reset_secs) =
            if limits.window_type.is_sliding() {
                // Reset is when everything counted now has left the window.
                let (rpm, tpm) = match s.sliding.as_deref() {
                    Some(sl) => (
                        (sl.rpm.current(now), sl.rpm.reset_secs(now)),
                        (sl.tpm.current(now), sl.tpm.reset_secs(now)),
                    ),
                    None => ((0, 0), (0, 0)),
                };
                (rpm.0, rpm.1, tpm.0, tpm.1)
            } else {
                // Seconds remaining in the current minute-window.
                let minute_reset = MINUTE_SECS - (now % MINUTE_SECS);
                (
                    s.rpm.current(now),
                    minute_reset,
                    s.tpm.current(now),
                    minute_reset,
                )
            };

        Some(RateLimitStatus {
            rpm_limit: limits.rpm,
            rpm_used,
            rpm_reset_secs,
            tpm_limit: limits.tpm,
            tpm_used,
            tpm_reset_secs,
            concurrency_limit: limits.concurrency,
            in_flight,
        })
    }
}
