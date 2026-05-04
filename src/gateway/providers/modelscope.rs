//! ModelScope currently documents OpenAI-compatible chat access for both its
//! global and CN API-Inference services.
//!
//! The docs explicitly point developers at per-model sample code as the
//! authoritative source, especially for reasoning models, so modelscope and
//! modelscope-cn intentionally share a conservative chat-only passthrough
//! implementation here instead of encoding undocumented provider-wide quirks.
//!
//! Docs:
//! - https://modelscope.ai/docs/model-service/API-Inference/intro
//! - https://modelscope.cn/docs/model-service/API-Inference/intro

use http::{HeaderMap, HeaderValue, header::AUTHORIZATION};
use serde::{Deserialize, Serialize};

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{ChatTransform, ProviderCapabilities, ProviderMeta},
};

pub const IDENTIFIER: &str = "modelscope";
pub const CN_IDENTIFIER: &str = "modelscope-cn";

const DEFAULT_BASE_URL: &str = "https://api-inference.modelscope.ai/v1";
const DEFAULT_CN_BASE_URL: &str = "https://api-inference.modelscope.cn/v1";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelScopeProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelScopeCnProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct ModelScope;
pub struct ModelScopeCn;

impl ProviderMeta for ModelScope {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        DEFAULT_BASE_URL
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        build_auth_headers(self.name(), auth)
    }
}

impl ProviderMeta for ModelScopeCn {
    fn name(&self) -> &'static str {
        CN_IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        DEFAULT_CN_BASE_URL
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        build_auth_headers(self.name(), auth)
    }
}

impl ChatTransform for ModelScope {}

impl ChatTransform for ModelScopeCn {}

impl ProviderCapabilities for ModelScope {}

impl ProviderCapabilities for ModelScopeCn {}

fn build_auth_headers(identifier: &str, auth: &ProviderAuth) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let value = HeaderValue::from_str(&format!("Bearer {}", auth.api_key_for(identifier)?))
        .map_err(|error| GatewayError::Validation(error.to_string()))?;
    headers.insert(AUTHORIZATION, value);
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::{ModelScope, ModelScopeCn};
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{ChatTransform, ProviderMeta},
        types::openai::ChatCompletionRequest,
    };

    #[test]
    fn provider_metadata_and_urls_are_correct() {
        let global = ModelScope;
        let cn = ModelScopeCn;
        let global_headers = global
            .build_auth_headers(&ProviderAuth::ApiKey("modelscope-global-key".into()))
            .unwrap();
        let cn_headers = cn
            .build_auth_headers(&ProviderAuth::ApiKey("modelscope-cn-key".into()))
            .unwrap();

        assert_eq!(global.name(), "modelscope");
        assert_eq!(global.default_base_url(), "https://api-inference.modelscope.ai/v1");
        assert_eq!(global_headers["authorization"], "Bearer modelscope-global-key");
        assert_eq!(
            global.build_url(global.default_base_url(), "ignored"),
            "https://api-inference.modelscope.ai/v1/chat/completions"
        );

        assert_eq!(cn.name(), "modelscope-cn");
        assert_eq!(cn.default_base_url(), "https://api-inference.modelscope.cn/v1");
        assert_eq!(cn_headers["authorization"], "Bearer modelscope-cn-key");
        assert_eq!(
            cn.build_url(cn.default_base_url(), "ignored"),
            "https://api-inference.modelscope.cn/v1/chat/completions"
        );
    }

    #[test]
    fn transform_request_passes_openai_payload_through() {
        let provider = ModelScope;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen3.5-35B-A3B",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "stream": true
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed["model"], "Qwen/Qwen3.5-35B-A3B");
        assert_eq!(transformed["temperature"], 0.7);
        assert_eq!(transformed["stream"], true);
        assert_eq!(transformed["messages"][0]["content"], "hello");
    }
}