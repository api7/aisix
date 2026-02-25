use std::fmt::Display;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitMetric {
    TPM,
    TPD,
    RPM,
    RPD,
    //TODO concurrency
}

impl Display for RateLimitMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLimitMetric::TPM => write!(f, "tpm"),
            RateLimitMetric::TPD => write!(f, "tpd"),
            RateLimitMetric::RPM => write!(f, "rpm"),
            RateLimitMetric::RPD => write!(f, "rpd"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RateLimit {
    #[serde(rename = "tpm")]
    pub token_per_minute: Option<u64>,
    #[serde(rename = "tpd")]
    pub token_per_day: Option<u64>,
    #[serde(rename = "rpm")]
    pub request_per_minute: Option<u64>,
    #[serde(rename = "rpd")]
    pub request_per_day: Option<u64>,
    #[serde(rename = "concurrency")]
    pub request_concurrency: Option<u64>,
}

pub trait HasRateLimit {
    fn rate_limit(&self) -> Option<RateLimit>;

    fn rate_limit_key(&self, metric: RateLimitMetric) -> String;
}
