//! Rate-limit configuration attached to Models and ApiKeys.
//!
//! All fields are optional; absence means "no limit on that dimension".
//! Windows per spec §3:
//! - `rps` — 1s fixed window (request count only — see api7/ai-gateway#396
//!   for the deferred per-second token-rate counter)
//! - `tpm`/`rpm` — 60s fixed window
//! - `rph` — 3600s fixed window (request count only — see ai-gateway#396)
//! - `tpd`/`rpd` — 86400s fixed window
//! - `concurrency` — semaphore capacity (not windowed)
//!
//! `rps`/`rph` were added in api7/AISIX-Cloud#426 to fix the upscaling
//! workaround in `policy_to_rate_limit` where `window=second` was
//! converted to `rpm = max_requests * 60` (allowing 60× bursts) and
//! `window=hour` was converted to `rpd = max_requests * 24` (same
//! exploit at 24× scale). Token-rate counters at sub-minute windows
//! (`tps`/`tph`) intentionally deferred because the existing
//! post-deduct `FixedWindowCounter::add` racing window roll-over
//! makes sub-minute token windows unsound; see ai-gateway#396.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RateLimit {
    /// Tokens per minute (60s window).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<u64>,

    /// Tokens per day (86400s window).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpd: Option<u64>,

    /// Requests per second (1s window). Added in #426 — see module
    /// docstring. Per-second tokens (`tps`) intentionally NOT shipped;
    /// see api7/ai-gateway#396 for the design tracking issue.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rps: Option<u64>,

    /// Requests per minute (60s window).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<u64>,

    /// Requests per hour (3600s window). Added in #426 — see module
    /// docstring. Per-hour tokens (`tph`) intentionally NOT shipped;
    /// see ai-gateway#396.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rph: Option<u64>,

    /// Requests per day (86400s window).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpd: Option<u64>,

    /// Max concurrent in-flight requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<u32>,
}

impl RateLimit {
    pub const fn is_unrestricted(&self) -> bool {
        self.tpm.is_none()
            && self.tpd.is_none()
            && self.rps.is_none()
            && self.rpm.is_none()
            && self.rph.is_none()
            && self.rpd.is_none()
            && self.concurrency.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_unrestricted() {
        assert!(RateLimit::default().is_unrestricted());
    }

    #[test]
    fn omits_none_fields_on_serialise() {
        let rl = RateLimit {
            rpm: Some(60),
            ..Default::default()
        };
        let json = serde_json::to_value(&rl).unwrap();
        assert_eq!(json["rpm"], 60);
        assert!(json.get("tpm").is_none());
        assert!(json.get("concurrency").is_none());
    }

    #[test]
    fn rejects_unknown_fields() {
        let r: Result<RateLimit, _> = serde_json::from_str(r#"{"rpm": 10, "extra": 1}"#);
        assert!(r.is_err());
    }
}
