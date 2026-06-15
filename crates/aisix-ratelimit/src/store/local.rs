//! In-process counter store — the historical, default backend.
//!
//! Behaviour-identical to the pre-#798 limiter: a `DashMap` of per-key
//! fixed-window counters guarded by one `parking_lot::Mutex` each. State
//! is per-replica and not shared, so a multi-replica cluster multiplies
//! every limit by the replica count — exactly what [`super::redis`]
//! exists to fix. `member` is ignored here (concurrency is a plain
//! `in_flight` counter).

use aisix_core::{RateLimit, RateLimitScope};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::Mutex;
use std::sync::Arc;

use super::{RateStore, DAY_SECS, HOUR_SECS, MINUTE_SECS, SECOND_SECS};
use crate::clock::{Clock, SystemClock};
use crate::error::RateLimitError;
use crate::limiter::RateLimitStatus;
use crate::window::{FixedWindowCounter, WindowCheck};

/// Per-key state guarded by a single mutex. Hot path locks once per
/// request; each operation inside is O(1).
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

/// Per-process fixed-window store.
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
        // (rps → rpm → rph → rpd) so a tighter window short-circuits a
        // looser one without consuming its slot. If any later layer
        // rejects, every earlier-incremented counter is rolled back by
        // exactly the delta this call contributed — concurrent sibling
        // requests' increments survive.
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
        Ok(())
    }

    async fn commit(&self, key: &str, tokens: u64, _member: &str) {
        let now = self.clock.unix_secs();
        let state = self.state_for(key);
        let mut s = state.lock();
        s.tpm.add(now, tokens);
        s.tpd.add(now, tokens);
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

    fn add_tokens(&self, key: &str, tokens: u64) {
        if tokens == 0 {
            return;
        }
        let now = self.clock.unix_secs();
        let state = self.state_for(key);
        let mut s = state.lock();
        s.tpm.add(now, tokens);
        s.tpd.add(now, tokens);
    }

    async fn peek(&self, key: &str, limits: &RateLimit) -> Option<RateLimitStatus> {
        let now = self.clock.unix_secs();
        let state = self.states.get(key)?;
        let mut s = state.lock();

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
}
