use http::{HeaderMap, HeaderValue, header::AUTHORIZATION};
use serde::{Deserialize, Serialize};

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{ChatTransform, CompatQuirks, EmbedTransform, ProviderCapabilities, ProviderMeta},
};

pub const IDENTIFIER: &str = "openai";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct OpenAIProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct OpenAIDef;

impl ProviderMeta for OpenAIDef {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        "https://api.openai.com"
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(&format!("Bearer {}", auth.api_key_for(self.name())?))
            .map_err(|error| GatewayError::Validation(error.to_string()))?;
        headers.insert(AUTHORIZATION, value);
        Ok(headers)
    }
}

impl ChatTransform for OpenAIDef {
    fn default_quirks(&self) -> CompatQuirks {
        CompatQuirks {
            inject_stream_usage: true,
            ..CompatQuirks::NONE
        }
    }
}

impl EmbedTransform for OpenAIDef {}

impl ProviderCapabilities for OpenAIDef {
    fn as_embed_transform(&self) -> Option<&dyn EmbedTransform> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::OpenAIDef;
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{ChatTransform, ProviderCapabilities, ProviderMeta},
        types::{
            embed::{EmbedRequestBody, EmbedResponseBody, EmbeddingRequest},
            openai::ChatCompletionRequest,
        },
    };

    #[test]
    fn openai_def_builds_bearer_auth_headers() {
        let provider = OpenAIDef;
        let headers = provider
            .build_auth_headers(&ProviderAuth::ApiKey("sk-openai".into()))
            .unwrap();

        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.default_base_url(), "https://api.openai.com");
        assert_eq!(headers["authorization"], "Bearer sk-openai");
    }

    #[test]
    fn openai_def_injects_stream_usage_in_default_transform() {
        let provider = OpenAIDef;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        }))
        .unwrap();

        let body = provider.transform_request(&request).unwrap();

        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn openai_def_reports_provider_name_for_missing_api_key() {
        let provider = OpenAIDef;
        let error = provider
            .build_auth_headers(&ProviderAuth::None)
            .unwrap_err();

        assert!(matches!(
            error,
            crate::gateway::error::GatewayError::Validation(message)
                if message.contains("openai")
                    && message.contains("ProviderAuth::ApiKey")
        ));
    }

    #[test]
    fn openai_def_exposes_embedding_transform() {
        let provider = OpenAIDef;
        let transform = provider.as_embed_transform().unwrap();
        let request: EmbeddingRequest = serde_json::from_value(json!({
            "model": "text-embedding-3-large",
            "input": ["hello", "world"]
        }))
        .unwrap();

        let body = transform.transform_embeddings_request(&request).unwrap();
        match body {
            EmbedRequestBody::Json(value) => {
                assert_eq!(value["model"], "text-embedding-3-large");
                assert_eq!(value["input"][0], "hello");
            }
        }

        let response = transform
            .transform_embeddings_response(EmbedResponseBody::Json(json!({
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "embedding": [0.1, 0.2],
                    "index": 0
                }],
                "model": "text-embedding-3-large",
                "usage": {"prompt_tokens": 2, "total_tokens": 2}
            })))
            .unwrap();

        assert_eq!(response.model, "text-embedding-3-large");
        assert_eq!(response.data.len(), 1);
    }
}
