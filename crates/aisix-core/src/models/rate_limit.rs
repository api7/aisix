//! Rate-limit configuration attached to Models and ApiKeys.
//!
//! All fields are optional; absence means "no limit on that dimension".
//! Windows per spec §3:
//! - `tpm`/`rpm` — 60s fixed window
//! - `tpd`/`rpd` — 86400s fixed window
//! - `concurrency` — semaphore capacity (not windowed)

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

    /// Requests per minute (60s window).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<u64>,

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
            && self.rpm.is_none()
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
