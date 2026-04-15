use serde::{Deserialize, Serialize};

use crate::gateway::providers::macros::provider;

pub const IDENTIFIER: &str = "deepseek";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct DeepSeekProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

provider!(DeepSeek {
    display_name: "deepseek",
    base_url: "https://api.deepseek.com",
    auth: bearer,
});
