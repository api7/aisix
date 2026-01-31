use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
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
