use serde::{Deserialize, Serialize};

use crate::gateway::providers::macros::provider;

/// Provider identifier string used to look up OpenRouter in the gateway registry.
pub const IDENTIFIER: &str = "openrouter";

/// Configuration for an OpenRouter provider deployment.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct OpenRouterProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

provider!(OpenRouter {
    display_name: "openrouter",
    base_url: "https://openrouter.ai/api/v1",
    chat_path: "/chat/completions",
    auth: bearer,
});

#[cfg(test)]
mod tests {
    use super::OpenRouter;
    use crate::gateway::traits::ProviderMeta;

    #[test]
    fn provider_macro_expands_correctly() {
        let provider = OpenRouter;

        pretty_assertions::assert_eq!(provider.name(), "openrouter");
        pretty_assertions::assert_eq!(provider.default_base_url(), "https://openrouter.ai/api/v1");
        pretty_assertions::assert_eq!(provider.chat_endpoint_path("ignored"), "/chat/completions");

        pretty_assertions::assert_eq!(
            provider.build_url(provider.default_base_url(), "ignored"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }
}
