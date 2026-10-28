//! Whether usage events are reaching the control plane.
//!
//! The control plane's budget counters are fed only by the usage events this
//! gateway delivers, so a budget it reports as not exhausted is only as
//! current as the last delivery. The telemetry sender records here when
//! events start waiting and when a POST reaches the control plane; strict
//! budget enforcement and `aisix_usage_delivery_unhealthy_seconds` read it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Delivery is unhealthy since T when usage events have been waiting since T
/// and no POST has reached the control plane since.
#[derive(Debug, Default)]
pub struct UsageDeliveryHealth {
    /// Milliseconds of T on a monotonic clock (see [`mono_ms`]); `0` while
    /// nothing is waiting. Monotonic so a wall-clock step cannot move the
    /// strict-mode decision.
    waiting_since_ms: AtomicU64,
}

/// Milliseconds since a process-wide origin placed a day in the past, so an
/// explicit T up to a day back still has a positive reading.
fn mono_ms(at: Instant) -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    let origin = *ORIGIN.get_or_init(|| {
        let now = Instant::now();
        now.checked_sub(Duration::from_secs(86_400)).unwrap_or(now)
    });
    (at.saturating_duration_since(origin).as_millis() as u64).max(1)
}

impl UsageDeliveryHealth {
    /// Events are waiting to be delivered. Keeps an earlier T.
    pub fn note_waiting(&self) {
        self.note_waiting_since(Instant::now());
    }

    /// [`Self::note_waiting`] with an explicit T.
    pub fn note_waiting_since(&self, at: Instant) {
        let _ = self.waiting_since_ms.compare_exchange(
            0,
            mono_ms(at),
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
    }

    /// A POST reached the control plane, and nothing is waiting.
    pub fn note_reached(&self) {
        self.waiting_since_ms.store(0, Ordering::Release);
    }

    /// Whether events are waiting with no POST having reached the control
    /// plane since they started to.
    pub fn is_waiting(&self) -> bool {
        self.waiting_since_ms.load(Ordering::Acquire) != 0
    }

    /// How long delivery has been unhealthy; zero while healthy.
    pub fn unhealthy_for(&self) -> Duration {
        match self.waiting_since_ms.load(Ordering::Acquire) {
            0 => Duration::ZERO,
            since => Duration::from_millis(mono_ms(Instant::now()).saturating_sub(since)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waiting_keeps_the_earliest_start_until_a_post_reaches() {
        let health = UsageDeliveryHealth::default();
        assert_eq!(health.unhealthy_for(), Duration::ZERO);
        health.note_waiting_since(Instant::now() - Duration::from_secs(120));
        health.note_waiting();
        assert!(health.unhealthy_for() >= Duration::from_secs(119));
        health.note_reached();
        assert_eq!(health.unhealthy_for(), Duration::ZERO);
    }
}
