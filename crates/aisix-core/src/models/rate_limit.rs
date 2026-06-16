//! Rate-limit configuration attached to Models and ApiKeys.
//!
//! All fields are optional; absence means "no limit on that dimension".
//! Windows per spec §3:
//! - `rps` — 1s fixed window (request count only)
//! - `tpm`/`rpm` — 60s fixed window
//! - `rph` — 3600s fixed window (request count only)
//! - `tpd`/`rpd` — 86400s fixed window
//! - `concurrency` — semaphore capacity (not windowed)
//!
//! Token-rate counters are minute/day only; there is no `tps` or `tph`
//! field.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RateLimit {
    /// Tokens per 60-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<u64>,

    /// Tokens per 86,400-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpd: Option<u64>,

    /// Requests per 1-second window. There is no per-second token limit field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rps: Option<u64>,

    /// Requests per 60-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<u64>,

    /// Requests per 3,600-second window. There is no per-hour token limit field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rph: Option<u64>,

    /// Requests per 86,400-second window.
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
