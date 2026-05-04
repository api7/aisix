//! SiliconFlow documents OpenAI-compatible chat and embeddings APIs for both
//! its global and CN endpoints.
//!
//! The implementation here keeps the general OpenAI request shape intact,
//! while encoding the one documented model-specific compatibility quirk:
//! deepseek-ai/DeepSeek-V3.1 requires enable_thinking=false when tools are
//! used.
//!
//! Docs:
//! - https://docs.siliconflow.com/en/userguide/quickstart
//! - https://docs.siliconflow.cn/cn/userguide/quickstart
//! - https://docs.siliconflow.com/en/api-reference/chat-completions/chat-completions.md
//! - https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions.md
//! - https://docs.siliconflow.com/en/api-reference/embeddings/create-embeddings.md
//! - https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings.md

use http::{HeaderMap, HeaderValue, header::AUTHORIZATION};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{ChatTransform, EmbedTransform, ProviderCapabilities, ProviderMeta},
    types::{
        embed::{EmbedRequestBody, EmbeddingRequest},
        openai::ChatCompletionRequest,
    },
};

pub const IDENTIFIER: &str = "siliconflow";
pub const CN_IDENTIFIER: &str = "siliconflow-cn";

const DEFAULT_BASE_URL: &str = "https://api.siliconflow.com/v1";
const DEFAULT_CN_BASE_URL: &str = "https://api.siliconflow.cn/v1";
const DEEPSEEK_V31_MODEL: &str = "deepseek-ai/DeepSeek-V3.1";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SiliconFlowProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SiliconFlowCnProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct SiliconFlow;
pub struct SiliconFlowCn;

impl ProviderMeta for SiliconFlow {
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

impl ProviderMeta for SiliconFlowCn {
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

impl ChatTransform for SiliconFlow {
    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        transform_request(request)
    }
}

impl ChatTransform for SiliconFlowCn {
    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        transform_request(request)
    }
}

impl EmbedTransform for SiliconFlow {
    fn transform_embeddings_request(&self, request: &EmbeddingRequest) -> Result<EmbedRequestBody> {
        transform_embeddings_request(request)
    }
}

impl EmbedTransform for SiliconFlowCn {
    fn transform_embeddings_request(&self, request: &EmbeddingRequest) -> Result<EmbedRequestBody> {
        transform_embeddings_request(request)
    }
}

impl ProviderCapabilities for SiliconFlow {
    fn as_embed_transform(&self) -> Option<&dyn EmbedTransform> {
        Some(self)
    }
}

impl ProviderCapabilities for SiliconFlowCn {
    fn as_embed_transform(&self) -> Option<&dyn EmbedTransform> {
        Some(self)
    }
}

fn build_auth_headers(identifier: &str, auth: &ProviderAuth) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let value = HeaderValue::from_str(&format!("Bearer {}", auth.api_key_for(identifier)?))
        .map_err(|error| GatewayError::Validation(error.to_string()))?;
    headers.insert(AUTHORIZATION, value);
    Ok(headers)
}

fn transform_request(request: &ChatCompletionRequest) -> Result<Value> {
    let mut body = serde_json::to_value(request)
        .map_err(|error| GatewayError::Transform(error.to_string()))?;

    let Value::Object(map) = &mut body else {
        return Ok(body);
    };

    let model = map.get("model").and_then(Value::as_str).ok_or_else(|| {
        GatewayError::Validation("siliconflow providers require a string model".into())
    })?;

    let has_tools = map
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());

    if model == DEEPSEEK_V31_MODEL && has_tools {
        match map.get("enable_thinking").and_then(Value::as_bool) {
            Some(true) => {
                return Err(GatewayError::Validation(
                    "siliconflow providers require enable_thinking=false when using tools with deepseek-ai/DeepSeek-V3.1".into(),
                ));
            }
            Some(false) => {}
            None => {
                map.insert("enable_thinking".into(), Value::Bool(false));
            }
        }
    }

    Ok(body)
}

fn transform_embeddings_request(request: &EmbeddingRequest) -> Result<EmbedRequestBody> {
    let mut body = serde_json::to_value(request)
        .map_err(|error| GatewayError::Transform(error.to_string()))?;

    if let Value::Object(map) = &mut body {
        // The typed gateway embeddings request only models the classic OpenAI
        // text embeddings shape, so omit SiliconFlow's VL-only user field.
        map.remove("user");
    }

    Ok(EmbedRequestBody::Json(body))
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::{SiliconFlow, SiliconFlowCn};
    use crate::gateway::{
        error::GatewayError,
        provider_instance::ProviderAuth,
        traits::{ChatTransform, EmbedTransform, ProviderCapabilities, ProviderMeta},
        types::{
            embed::{EmbedRequestBody, EmbeddingRequest},
            openai::ChatCompletionRequest,
        },
    };

    #[test]
    fn provider_metadata_and_urls_are_correct() {
        let global = SiliconFlow;
        let cn = SiliconFlowCn;
        let global_headers = global
            .build_auth_headers(&ProviderAuth::ApiKey("siliconflow-global-key".into()))
            .unwrap();
        let cn_headers = cn
            .build_auth_headers(&ProviderAuth::ApiKey("siliconflow-cn-key".into()))
            .unwrap();

        assert_eq!(global.name(), "siliconflow");
        assert_eq!(global.default_base_url(), "https://api.siliconflow.com/v1");
        assert_eq!(
            global_headers["authorization"],
            "Bearer siliconflow-global-key"
        );
        assert_eq!(
            global.build_url(global.default_base_url(), "ignored"),
            "https://api.siliconflow.com/v1/chat/completions"
        );
        assert_eq!(
            global.build_url_for_endpoint(global.default_base_url(), "/v1/embeddings"),
            "https://api.siliconflow.com/v1/embeddings"
        );
        assert!(global.as_embed_transform().is_some());

        assert_eq!(cn.name(), "siliconflow-cn");
        assert_eq!(cn.default_base_url(), "https://api.siliconflow.cn/v1");
        assert_eq!(cn_headers["authorization"], "Bearer siliconflow-cn-key");
        assert_eq!(
            cn.build_url(cn.default_base_url(), "ignored"),
            "https://api.siliconflow.cn/v1/chat/completions"
        );
        assert_eq!(
            cn.build_url_for_endpoint(cn.default_base_url(), "/v1/embeddings"),
            "https://api.siliconflow.cn/v1/embeddings"
        );
        assert!(cn.as_embed_transform().is_some());
    }

    #[test]
    fn transform_request_injects_enable_thinking_false_for_deepseek_v31_tool_calls() {
        let provider = SiliconFlow;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "deepseek-ai/DeepSeek-V3.1",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        }
                    }
                }
            }]
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed["enable_thinking"], false);
    }

    #[test]
    fn transform_request_rejects_enable_thinking_true_for_deepseek_v31_tool_calls() {
        let provider = SiliconFlow;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "deepseek-ai/DeepSeek-V3.1",
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": true,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        }
                    }
                }
            }]
        }))
        .unwrap();

        let error = provider.transform_request(&request).unwrap_err();

        assert_matches!(
            error,
            GatewayError::Validation(message)
                if message.contains("enable_thinking=false")
        );
    }

    #[test]
    fn transform_request_preserves_explicit_extensions_for_other_models() {
        let provider = SiliconFlow;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen3-32B",
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": true,
            "thinking_budget": 2048,
            "min_p": 0.1
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed["enable_thinking"], true);
        assert_eq!(transformed["thinking_budget"], 2048);
        assert_eq!(transformed["min_p"], 0.1);
    }

    #[test]
    fn transform_embeddings_request_strips_user_but_keeps_openai_fields() {
        let provider = SiliconFlow;
        let request: EmbeddingRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen3-Embedding-8B",
            "input": ["hello"],
            "dimensions": 1024,
            "encoding_format": "base64",
            "user": "user-123"
        }))
        .unwrap();

        let body = provider.transform_embeddings_request(&request).unwrap();

        match body {
            EmbedRequestBody::Json(value) => {
                assert_eq!(value["dimensions"], 1024);
                assert_eq!(value["encoding_format"], "base64");
                assert_eq!(value.get("user"), None);
            }
        }
    }
}
