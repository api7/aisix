use serde::{Deserialize, Serialize};

use crate::gateway::providers::macros::provider;

/// Provider identifier string used to look up xAI in the gateway registry.
pub const IDENTIFIER: &str = "xai";

/// Configuration for an xAI provider deployment.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct XaiProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

provider!(Xai {
    display_name: "xai",
    base_url: "https://api.x.ai/v1",
    auth: bearer,
    quirks: {
        unsupported_params: &["logit_bias"],
        inject_stream_usage: true,
    }
});

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::Xai;
    use crate::gateway::{
        traits::{ChatTransform, ProviderMeta},
        types::openai::ChatCompletionRequest,
    };

    #[test]
    fn provider_macro_expands_correctly() {
        let provider = Xai;

        assert_eq!(provider.name(), "xai");
        assert_eq!(provider.default_base_url(), "https://api.x.ai/v1");

        assert_eq!(
            provider.build_url(provider.default_base_url(), "ignored"),
            "https://api.x.ai/v1/chat/completions"
        );
    }

    #[test]
    fn transform_request_applies_xai_quirks() {
        let provider = Xai;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "grok-4.3",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "max_tokens": 128,
            "logit_bias": {"42": 100}
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed.get("logit_bias"), None);
        assert_eq!(transformed["max_tokens"], 128);
        assert_eq!(transformed["stream_options"]["include_usage"], true);
    }
}
