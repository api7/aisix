use serde::{Deserialize, Serialize};

use crate::gateway::providers::macros::provider;

/// Provider identifier string used to look up Mistral in the gateway registry.
pub const IDENTIFIER: &str = "mistral";

/// Configuration for a Mistral provider deployment.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MistralProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

provider!(Mistral {
    display_name: "mistral",
    base_url: "https://api.mistral.ai",
    auth: bearer,
    quirks: {
        tool_args_may_be_object: true,
    }
});

#[cfg(test)]
mod tests {
    use super::Mistral;
    use crate::gateway::traits::{ChatTransform, ProviderMeta};

    #[test]
    fn provider_macro_expands_correctly() {
        let provider = Mistral;

        pretty_assertions::assert_eq!(provider.name(), "mistral");
        pretty_assertions::assert_eq!(provider.default_base_url(), "https://api.mistral.ai");

        pretty_assertions::assert_eq!(
            provider.build_url(provider.default_base_url(), "ignored"),
            "https://api.mistral.ai/v1/chat/completions"
        );

        pretty_assertions::assert_eq!(provider.default_quirks().tool_args_may_be_object, true);
    }
}
