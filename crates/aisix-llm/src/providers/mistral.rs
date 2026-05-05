use serde::{Deserialize, Serialize};

use crate::providers::macros::provider;

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
    use pretty_assertions::assert_eq;

    use super::Mistral;
    use crate::traits::{ChatTransform, ProviderMeta};

    #[test]
    fn provider_macro_expands_correctly() {
        let provider = Mistral;

        assert_eq!(provider.name(), "mistral");
        assert_eq!(provider.default_base_url(), "https://api.mistral.ai");

        assert_eq!(
            provider.build_url(provider.default_base_url(), "ignored"),
            "https://api.mistral.ai/v1/chat/completions"
        );

        assert!(provider.default_quirks().tool_args_may_be_object);
    }
}
