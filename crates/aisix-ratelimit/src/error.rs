//! Limiter error taxonomy.
//!
//! Uses [`aisix_core::RateLimitScope`] so the proxy layer can plug the
//! error straight into its OpenAI-style envelope without translation.

use aisix_core::RateLimitScope;

/// Retry hint reported for a concurrency rejection.
///
/// A concurrency slot frees when some in-flight request finishes, which
/// the gateway cannot predict — unlike a windowed dimension there is no
/// reset instant to report. One minute is the hint the established
/// ecosystem's proxy limiters emit for the same rejection, so clients
/// tuned against those back off identically here. It is an interop
/// choice, not a measurement: do not "tune" it without redoing that
/// comparison.
pub const CONCURRENCY_RETRY_AFTER_SECS: u64 = 60;

/// Dimension name reported for a concurrency rejection. The windowed
/// dimensions report the `Dim::name` the store already keys them by
/// (`rps` / `rpm` / `rph` / `rpd` / `tpm` / `tpd`).
pub const CONCURRENCY_DIMENSION: &str = "concurrency";

/// The single limit that produced a rejection, and its state at that
/// moment. Carries what the caller-facing `x-ratelimit-*` headers on a
/// 429 report, so the response layer never has to re-derive which of a
/// key's limits actually fired.
///
/// `dimension` is finer-grained than [`RateLimitScope`]: the scope only
/// separates request-counting from token-counting (it labels the
/// rejection metric), while this names the exact window — `rpm` and
/// `rpd` share a scope but are different limits with different reset
/// instants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitDetail {
    /// `rps` | `rpm` | `rph` | `rpd` | `tpm` | `tpd` | `concurrency`.
    pub dimension: &'static str,
    /// The configured cap for that dimension. Requests for `rp*`,
    /// tokens for `tp*`, in-flight slots for `concurrency` — the unit
    /// follows the dimension, which is why the wire reports both.
    pub limit: u64,
    /// Headroom left under `limit`. Zero on every rejection except a
    /// concurrency one racing a slot release.
    pub remaining: u64,
    /// Seconds until the caller may retry: to the window boundary for a
    /// windowed dimension, [`CONCURRENCY_RETRY_AFTER_SECS`] otherwise.
    pub reset_secs: u64,
}

impl LimitDetail {
    /// Detail for a windowed dimension that has just refused a request.
    /// `used` is the counter's value *before* the refused request, so
    /// `remaining` saturates to zero exactly when the cap is met or
    /// overrun.
    pub(crate) fn window(dimension: &'static str, limit: u64, used: u64, reset_secs: u64) -> Self {
        Self {
            dimension,
            limit,
            remaining: limit.saturating_sub(used),
            reset_secs,
        }
    }

    /// Detail for a concurrency gate that has just refused a request.
    pub(crate) fn concurrency(limit: u64, in_flight: u64) -> Self {
        Self {
            dimension: CONCURRENCY_DIMENSION,
            limit,
            remaining: limit.saturating_sub(in_flight),
            reset_secs: CONCURRENCY_RETRY_AFTER_SECS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RateLimitError {
    #[error("request limit exceeded ({scope})")]
    Requests {
        scope: RateLimitScope,
        detail: LimitDetail,
    },
    #[error("token limit exceeded ({scope})")]
    Tokens {
        scope: RateLimitScope,
        detail: LimitDetail,
    },
    #[error("concurrency limit exceeded")]
    Concurrency { detail: LimitDetail },
}

impl RateLimitError {
    pub fn scope(&self) -> RateLimitScope {
        match self {
            RateLimitError::Requests { scope, .. } => *scope,
            RateLimitError::Tokens { scope, .. } => *scope,
            RateLimitError::Concurrency { .. } => RateLimitScope::Requests,
        }
    }

    /// The limit that fired, for the `x-ratelimit-*` headers on the 429.
    pub fn detail(&self) -> LimitDetail {
        match self {
            RateLimitError::Requests { detail, .. }
            | RateLimitError::Tokens { detail, .. }
            | RateLimitError::Concurrency { detail } => *detail,
        }
    }

    pub fn retry_after_secs(&self) -> Option<u64> {
        Some(self.detail().reset_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_scope_preserved_on_access() {
        let e = RateLimitError::Requests {
            scope: RateLimitScope::Requests,
            detail: LimitDetail::window("rpm", 10, 10, 42),
        };
        assert_eq!(e.scope(), RateLimitScope::Requests);
        assert_eq!(e.retry_after_secs(), Some(42));
        assert_eq!(e.detail().dimension, "rpm");
        assert_eq!(e.detail().limit, 10);
        assert_eq!(e.detail().remaining, 0);
    }

    #[test]
    fn concurrency_reports_a_fixed_retry_hint() {
        // A concurrency slot has no window, so there is nothing to
        // compute — but a 429 with no retry hint at all leaves a client
        // with only "retry immediately", which is the behaviour the
        // hint exists to prevent.
        let e = RateLimitError::Concurrency {
            detail: LimitDetail::concurrency(8, 8),
        };
        assert_eq!(e.retry_after_secs(), Some(CONCURRENCY_RETRY_AFTER_SECS));
        assert_eq!(e.detail().dimension, "concurrency");
        assert_eq!(e.detail().limit, 8);
        assert_eq!(e.detail().remaining, 0);
        // Concurrency rejections still label the rejection metric with
        // the request scope — the metric's label set is unchanged.
        assert_eq!(e.scope(), RateLimitScope::Requests);
    }

    #[test]
    fn window_remaining_saturates_when_the_counter_overran_the_cap() {
        // Token windows are checked-but-not-incremented, so a commit can
        // push the counter past the cap; `remaining` must floor at zero
        // rather than wrap.
        let d = LimitDetail::window("tpm", 1_000, 1_750, 17);
        assert_eq!(d.remaining, 0);
        assert_eq!(d.limit, 1_000);
        assert_eq!(d.reset_secs, 17);
    }
}
