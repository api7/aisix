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

/// Fireworks AI currently uses its OpenAI-compatible inference API.
/// Docs: https://docs.fireworks.ai/tools-sdks/openai-compatibility
pub const IDENTIFIER: &str = "fireworks-ai";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct FireworksAiProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct FireworksAi;

impl ProviderMeta for FireworksAi {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        "https://api.fireworks.ai/inference/v1"
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(&format!("Bearer {}", auth.api_key_for(self.name())?))
            .map_err(|error| GatewayError::Validation(error.to_string()))?;
        headers.insert(AUTHORIZATION, value);
        Ok(headers)
    }
}

impl ChatTransform for FireworksAi {
    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        let mut body = serde_json::to_value(request)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;

        if let Value::Object(map) = &mut body {
            // Fireworks defaults to truncating max_tokens on context overflow.
            // Set the documented override so requests keep OpenAI-style error semantics.
            map.entry("context_length_exceeded_behavior")
                .or_insert_with(|| Value::String("error".into()));
        }

        Ok(body)
    }
}

impl EmbedTransform for FireworksAi {
    fn transform_embeddings_request(&self, request: &EmbeddingRequest) -> Result<EmbedRequestBody> {
        let mut body = serde_json::to_value(request)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;

        if let Value::Object(map) = &mut body {
            // Fireworks documents dimensions, return_logits, normalize, and prompt_template,
            // but not OpenAI's encoding_format or user fields on /embeddings.
            map.remove("encoding_format");
            map.remove("user");
        }

        Ok(EmbedRequestBody::Json(body))
    }
}

impl ProviderCapabilities for FireworksAi {
    fn as_embed_transform(&self) -> Option<&dyn EmbedTransform> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::FireworksAi;
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
        let provider = FireworksAi;
        let headers = provider
            .build_auth_headers(&ProviderAuth::ApiKey("fw-key".into()))
            .unwrap();

        assert_eq!(provider.name(), "fireworks-ai");
        assert_eq!(
            provider.default_base_url(),
            "https://api.fireworks.ai/inference/v1"
        );
        assert_eq!(headers["authorization"], "Bearer fw-key");
        assert_eq!(
            provider.build_url(provider.default_base_url(), "ignored"),
            "https://api.fireworks.ai/inference/v1/chat/completions"
        );
        assert!(provider.as_embed_transform().is_some());
    }

    #[test]
    fn transform_request_defaults_to_openai_context_length_behavior() {
        let provider = FireworksAi;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "accounts/fireworks/models/kimi-k2-instruct-0905",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed["context_length_exceeded_behavior"], "error");
    }

    #[test]
    fn transform_request_preserves_explicit_context_length_behavior() {
        let provider = FireworksAi;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "accounts/fireworks/models/kimi-k2-instruct-0905",
            "messages": [{"role": "user", "content": "hello"}],
            "context_length_exceeded_behavior": "truncate"
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed["context_length_exceeded_behavior"], "truncate");
    }

    #[test]
    fn transform_embeddings_request_strips_unsupported_fields() {
        let provider = FireworksAi;
        let request: EmbeddingRequest = serde_json::from_value(json!({
            "model": "fireworks/qwen3-embedding-8b",
            "input": ["hello"],
            "dimensions": 128,
            "encoding_format": "float",
            "user": "user-123"
        }))
        .unwrap();

        let body = provider.transform_embeddings_request(&request).unwrap();

        match body {
            EmbedRequestBody::Json(value) => {
                assert_eq!(value["dimensions"], 128);
                assert_eq!(value.get("encoding_format"), None);
                assert_eq!(value.get("user"), None);
            }
        }
    }
}
