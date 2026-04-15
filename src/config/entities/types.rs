use std::fmt::Display;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitMetric {
    TPM,
    TPD,
    RPM,
    RPD,
    Concurrency,
}

impl Display for RateLimitMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLimitMetric::TPM => write!(f, "tpm"),
            RateLimitMetric::TPD => write!(f, "tpd"),
            RateLimitMetric::RPM => write!(f, "rpm"),
            RateLimitMetric::RPD => write!(f, "rpd"),
            RateLimitMetric::Concurrency => write!(f, "concurrency"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RateLimit {
    #[serde(rename = "tpm", skip_serializing_if = "Option::is_none")]
    pub token_per_minute: Option<u64>,
    #[serde(rename = "tpd", skip_serializing_if = "Option::is_none")]
    pub token_per_day: Option<u64>,
    #[serde(rename = "rpm", skip_serializing_if = "Option::is_none")]
    pub request_per_minute: Option<u64>,
    #[serde(rename = "rpd", skip_serializing_if = "Option::is_none")]
    pub request_per_day: Option<u64>,
    #[serde(rename = "concurrency", skip_serializing_if = "Option::is_none")]
    pub request_concurrency: Option<u64>,
}

pub trait HasRateLimit {
    fn rate_limit(&self) -> Option<RateLimit>;

    fn rate_limit_key(&self, metric: RateLimitMetric) -> String;
}
