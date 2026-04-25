use std::borrow::Cow;

use http::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{ChatTransform, CompatQuirks, EmbedTransform, ProviderCapabilities, ProviderMeta},
    types::{
        embed::{EmbedRequestBody, EmbeddingRequest},
        openai::ChatCompletionRequest,
    },
};

pub const IDENTIFIER: &str = "azure";
pub const DEFAULT_API_VERSION: &str = "2024-10-21";
const DEFAULT_BASE_URL: &str = "https://example.openai.azure.com";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AzureProviderConfig {
    pub api_key: String,
    pub api_base: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
}

pub struct AzureDef;

impl ProviderMeta for AzureDef {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        DEFAULT_BASE_URL
    }

    fn chat_endpoint_path(&self, model: &str) -> Cow<'static, str> {
        Cow::Owned(format!(
            "/openai/deployments/{}/chat/completions",
            model.replace('/', "%2F")
        ))
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        const HEADER_NAME: http::header::HeaderName =
            http::header::HeaderName::from_static("api-key");

        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(auth.api_key_for(self.name())?)
            .map_err(|error| GatewayError::Validation(error.to_string()))?;
        headers.insert(HEADER_NAME, value);
        Ok(headers)
    }
}

impl ChatTransform for AzureDef {
    fn default_quirks(&self) -> CompatQuirks {
        CompatQuirks {
            inject_stream_usage: true,
            ..CompatQuirks::NONE
        }
    }

    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        let mut body = serde_json::to_value(request)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;
        self.default_quirks().apply_to_request(&mut body);
        remove_model_field(&mut body);
        Ok(body)
    }
}

impl EmbedTransform for AzureDef {
    fn embeddings_endpoint_path(&self, model: &str) -> Cow<'static, str> {
        Cow::Owned(format!(
            "/openai/deployments/{}/embeddings",
            model.replace('/', "%2F")
        ))
    }

    fn transform_embeddings_request(&self, request: &EmbeddingRequest) -> Result<EmbedRequestBody> {
        let mut body = serde_json::to_value(request)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;
        remove_model_field(&mut body);
        Ok(EmbedRequestBody::Json(body))
    }
}

impl ProviderCapabilities for AzureDef {
    fn as_embed_transform(&self) -> Option<&dyn EmbedTransform> {
        Some(self)
    }
}

fn remove_model_field(body: &mut Value) {
    if let Value::Object(map) = body {
        map.remove("model");
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::{AzureDef, DEFAULT_API_VERSION};
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{ChatTransform, EmbedTransform, ProviderCapabilities, ProviderMeta},
        types::{embed::EmbedRequestBody, openai::ChatCompletionRequest},
    };

    #[test]
    fn azure_def_uses_deployment_paths_and_api_key_auth() {
        let provider = AzureDef;
        let headers = provider
            .build_auth_headers(&ProviderAuth::ApiKey("azure-key".into()))
            .unwrap();
        let chat_url = provider.build_url(
            &format!(
                "https://example-resource.openai.azure.com/?api-version={}",
                DEFAULT_API_VERSION
            ),
            "gpt-4o-prod",
        );
        let embed_url = provider.build_url_for_endpoint(
            &format!(
                "https://example-resource.openai.azure.com/?api-version={}",
                DEFAULT_API_VERSION
            ),
            provider
                .embeddings_endpoint_path("text-embedding-3-large")
                .as_ref(),
        );

        let chat_url = reqwest::Url::parse(&chat_url).unwrap();
        let embed_url = reqwest::Url::parse(&embed_url).unwrap();

        assert_eq!(provider.name(), "azure");
        assert_eq!(
            provider.default_base_url(),
            "https://example.openai.azure.com"
        );
        assert_eq!(headers["api-key"], "azure-key");
        assert_eq!(
            chat_url.path(),
            "/openai/deployments/gpt-4o-prod/chat/completions"
        );
        assert_eq!(chat_url.query(), Some("api-version=2024-10-21"));
        assert_eq!(
            embed_url.path(),
            "/openai/deployments/text-embedding-3-large/embeddings"
        );
        assert_eq!(embed_url.query(), Some("api-version=2024-10-21"));
        assert!(provider.as_embed_transform().is_some());
    }

    #[test]
    fn azure_def_transforms_chat_request_without_model_and_injects_stream_usage() {
        let provider = AzureDef;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-4o-prod",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        }))
        .unwrap();

        let body = provider.transform_request(&request).unwrap();

        assert_eq!(body["messages"][0]["content"], "Hello");
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body.get("model"), None);
    }

    #[test]
    fn azure_def_transforms_embeddings_request_without_model() {
        let provider = AzureDef;
        let request = serde_json::from_value(json!({
            "model": "text-embedding-3-large",
            "input": ["hello", "world"]
        }))
        .unwrap();

        let body = provider.transform_embeddings_request(&request).unwrap();

        match body {
            EmbedRequestBody::Json(value) => {
                assert_eq!(value["input"][0], "hello");
                assert_eq!(value.get("model"), None);
            }
        }
    }
}
