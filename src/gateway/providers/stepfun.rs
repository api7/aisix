//! StepFun documents an OpenAI-compatible chat endpoint.
//!
//! The standard OpenAI-compatible path is /v1/chat/completions, so a single
//! stepfun provider is enough here.
//!
//! Docs:
//! - https://platform.stepfun.com/docs/zh/quickstart/overview
//! - https://platform.stepfun.com/docs/zh/api-reference/chat/chat-completion-create

use serde::{Deserialize, Serialize};

use crate::gateway::providers::macros::provider;

pub const IDENTIFIER: &str = "stepfun";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct StepFunProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

provider!(StepFun {
    display_name: "stepfun",
    base_url: "https://api.stepfun.com/v1",
    auth: bearer,
});

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::StepFun;
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{ChatTransform, ProviderMeta},
        types::openai::ChatCompletionRequest,
    };

    #[test]
    fn provider_metadata_and_url_are_correct() {
        let provider = StepFun;
        let headers = provider
            .build_auth_headers(&ProviderAuth::ApiKey("stepfun-com-key".into()))
            .unwrap();

        assert_eq!(provider.name(), "stepfun");
        assert_eq!(provider.default_base_url(), "https://api.stepfun.com/v1");
        assert_eq!(headers["authorization"], "Bearer stepfun-com-key");
        assert_eq!(
            provider.build_url(provider.default_base_url(), "ignored"),
            "https://api.stepfun.com/v1/chat/completions"
        );
    }

    #[test]
    fn transform_request_passes_openai_compatible_payload_through() {
        let provider = StepFun;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "step-3.5-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_format": {"type": "deepseek-style"},
            "reasoning_effort": "high"
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed["model"], "step-3.5-flash");
        assert_eq!(transformed["reasoning_format"]["type"], "deepseek-style");
        assert_eq!(transformed["reasoning_effort"], "high");
    }
}
