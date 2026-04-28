use std::borrow::Cow;

use http::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{
        ChatTransform, CompatQuirks, EmbedTransform, ProviderCapabilities, ProviderMeta,
        ProviderSemanticConventions, provider::encode_path_segment,
    },
    types::{
        embed::{EmbedRequestBody, EmbeddingRequest},
        openai::ChatCompletionRequest,
    },
};

/// Provider identifier string used to look up this provider in the gateway registry.
pub const IDENTIFIER: &str = "azure";

/// Default Azure OpenAI REST API version sent as the `api-version` query parameter.
/// Overridable per deployment via [`AzureProviderConfig::api_version`].
/// See https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#api-versioning for details.
pub const DEFAULT_API_VERSION: &str = "v1";
const DEFAULT_BASE_URL: &str = "https://example.openai.azure.com";

/// Configuration for an Azure OpenAI provider deployment.
///
/// `api_key` authenticates requests, `api_base` identifies the Azure resource,
/// and `api_version` optionally overrides the default REST API version.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AzureProviderConfig {
    pub api_key: String,
    pub api_base: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
}

/// Provider definition for Azure OpenAI compatible deployments.
pub struct AzureDef;

impl ProviderMeta for AzureDef {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        DEFAULT_BASE_URL
    }

    fn semantic_conventions(&self) -> ProviderSemanticConventions {
        ProviderSemanticConventions {
            gen_ai_provider_name: "azure.ai.openai",
            llm_system: "openai",
            llm_provider: Some("azure"),
        }
    }

    fn chat_endpoint_path(&self, model: &str) -> Cow<'static, str> {
        Cow::Owned(format!(
            "/openai/deployments/{}/chat/completions",
            encode_path_segment(model)
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
            encode_path_segment(model)
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
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::{AzureDef, DEFAULT_API_VERSION};
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{
            ChatTransform, EmbedTransform, ProviderCapabilities, ProviderMeta,
            ProviderSemanticConventions,
        },
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
        assert_eq!(chat_url.query(), Some("api-version=v1"));
        assert_eq!(
            embed_url.path(),
            "/openai/deployments/text-embedding-3-large/embeddings"
        );
        assert_eq!(embed_url.query(), Some("api-version=v1"));
        assert!(provider.as_embed_transform().is_some());
        assert_eq!(
            provider.semantic_conventions(),
            ProviderSemanticConventions {
                gen_ai_provider_name: "azure.ai.openai",
                llm_system: "openai",
                llm_provider: Some("azure"),
            }
        );
    }

    #[test]
    fn azure_def_percent_encodes_reserved_deployment_characters() {
        let provider = AzureDef;
        let deployment = "prod slot/50%?blue#canary";

        let chat_url = provider.build_url(
            &format!(
                "https://example-resource.openai.azure.com/?api-version={}",
                DEFAULT_API_VERSION
            ),
            deployment,
        );
        let embed_url = provider.build_url_for_endpoint(
            &format!(
                "https://example-resource.openai.azure.com/?api-version={}",
                DEFAULT_API_VERSION
            ),
            provider.embeddings_endpoint_path(deployment).as_ref(),
        );

        let chat_url = reqwest::Url::parse(&chat_url).unwrap();
        let embed_url = reqwest::Url::parse(&embed_url).unwrap();

        assert_eq!(
            chat_url.path(),
            "/openai/deployments/prod%20slot%2F50%25%3Fblue%23canary/chat/completions"
        );
        assert_eq!(
            embed_url.path(),
            "/openai/deployments/prod%20slot%2F50%25%3Fblue%23canary/embeddings"
        );
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

        assert_matches!(body, EmbedRequestBody::Json(value) => {
            assert_eq!(value["input"][0], "hello");
            assert_eq!(value.get("model"), None);
        });
    }
}
