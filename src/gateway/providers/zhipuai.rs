use std::borrow::Cow;

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

/// ZhipuAI currently exposes an OpenAI-compatible base URL that already ends in
/// /api/paas/v4, so chat and embeddings both override their endpoint suffixes.
/// Docs:
/// - https://docs.bigmodel.cn/cn/guide/develop/openai/introduction.md
/// - https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E6%96%87%E6%9C%AC%E5%B5%8C%E5%85%A5.md
pub const IDENTIFIER: &str = "zhipuai";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ZhipuAiProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct ZhipuAi;

impl ProviderMeta for ZhipuAi {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        "https://open.bigmodel.cn/api/paas/v4"
    }

    fn chat_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
        Cow::Borrowed("/chat/completions")
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(&format!("Bearer {}", auth.api_key_for(self.name())?))
            .map_err(|error| GatewayError::Validation(error.to_string()))?;
        headers.insert(AUTHORIZATION, value);
        Ok(headers)
    }
}

impl ChatTransform for ZhipuAi {
    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        let body = serde_json::to_value(request)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;

        validate_temperature(&body)?;

        Ok(body)
    }
}

impl EmbedTransform for ZhipuAi {
    fn embeddings_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
        Cow::Borrowed("/embeddings")
    }

    fn transform_embeddings_request(&self, request: &EmbeddingRequest) -> Result<EmbedRequestBody> {
        let mut body = serde_json::to_value(request)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;

        if let Value::Object(map) = &mut body {
            validate_dimensions(map)?;

            // ZhipuAI embeddings document model, input, and dimensions, but not
            // OpenAI's encoding_format or user fields.
            map.remove("encoding_format");
            map.remove("user");
        }

        Ok(EmbedRequestBody::Json(body))
    }
}

impl ProviderCapabilities for ZhipuAi {
    fn as_embed_transform(&self) -> Option<&dyn EmbedTransform> {
        Some(self)
    }
}

fn validate_temperature(body: &Value) -> Result<()> {
    let Value::Object(map) = body else {
        return Ok(());
    };

    let Some(temperature) = map.get("temperature").and_then(Value::as_f64) else {
        return Ok(());
    };

    if temperature <= 0.0 || temperature > 1.0 {
        return Err(GatewayError::Validation(format!(
            "zhipuai requires temperature to be within (0, 1], got {temperature}"
        )));
    }

    Ok(())
}

fn validate_dimensions(map: &serde_json::Map<String, Value>) -> Result<()> {
    let Some(model) = map.get("model").and_then(Value::as_str) else {
        return Ok(());
    };
    let Some(dimensions) = map.get("dimensions").and_then(Value::as_u64) else {
        return Ok(());
    };

    match model {
        "embedding-3" if !matches!(dimensions, 256 | 512 | 1024 | 2048) => {
            return Err(GatewayError::Validation(format!(
                "zhipuai embedding-3 only supports dimensions 256, 512, 1024, or 2048, got {dimensions}"
            )));
        }
        "embedding-2" if dimensions != 1024 => {
            return Err(GatewayError::Validation(format!(
                "zhipuai embedding-2 uses fixed dimension 1024, got {dimensions}"
            )));
        }
        _ => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::ZhipuAi;
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{ChatTransform, EmbedTransform, ProviderCapabilities, ProviderMeta},
        types::{
            embed::{EmbedRequestBody, EmbeddingRequest},
            openai::ChatCompletionRequest,
        },
    };

    #[test]
    fn provider_metadata_and_urls_are_correct() {
        let provider = ZhipuAi;
        let headers = provider
            .build_auth_headers(&ProviderAuth::ApiKey("zhipu-key".into()))
            .unwrap();

        assert_eq!(provider.name(), "zhipuai");
        assert_eq!(
            provider.default_base_url(),
            "https://open.bigmodel.cn/api/paas/v4"
        );
        assert_eq!(headers["authorization"], "Bearer zhipu-key");
        assert_eq!(
            provider.build_url(provider.default_base_url(), "glm-5.1"),
            "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        );
        assert_eq!(
            provider.embeddings_endpoint_path("embedding-3"),
            "/embeddings"
        );
        assert!(provider.as_embed_transform().is_some());
    }

    #[test]
    fn transform_request_rejects_zero_temperature() {
        let provider = ZhipuAi;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "glm-5.1",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.0,
            "thinking": {"type": "enabled"},
            "tool_stream": true
        }))
        .unwrap();

        let error = provider.transform_request(&request).unwrap_err();

        assert_matches!(
            error,
            crate::gateway::error::GatewayError::Validation(message)
                if message.contains("temperature") && message.contains("(0, 1]")
        );
    }

    #[test]
    fn transform_request_preserves_zhipu_extensions() {
        let provider = ZhipuAi;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.8,
            "thinking": {"type": "enabled"},
            "tool_stream": true
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed["temperature"], 0.8);
        assert_eq!(transformed["thinking"]["type"], "enabled");
        assert_eq!(transformed["tool_stream"], true);
    }

    #[test]
    fn transform_embeddings_request_strips_openai_only_fields() {
        let provider = ZhipuAi;
        let request: EmbeddingRequest = serde_json::from_value(json!({
            "model": "embedding-3",
            "input": ["hello"],
            "dimensions": 512,
            "encoding_format": "float",
            "user": "user-123"
        }))
        .unwrap();

        let body = provider.transform_embeddings_request(&request).unwrap();

        match body {
            EmbedRequestBody::Json(value) => {
                assert_eq!(value["dimensions"], 512);
                assert_eq!(value.get("encoding_format"), None);
                assert_eq!(value.get("user"), None);
            }
        }
    }

    #[test]
    fn transform_embeddings_request_rejects_invalid_embedding_dimensions() {
        let provider = ZhipuAi;
        let request: EmbeddingRequest = serde_json::from_value(json!({
            "model": "embedding-3",
            "input": ["hello"],
            "dimensions": 768
        }))
        .unwrap();

        let error = provider.transform_embeddings_request(&request).unwrap_err();

        assert_matches!(
            error,
            crate::gateway::error::GatewayError::Validation(message)
                if message.contains("embedding-3") && message.contains("256")
        );
    }
}
