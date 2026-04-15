use std::borrow::Cow;

use http::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{ChatTransform, EmbedTransform, ProviderCapabilities, ProviderMeta},
};

pub const IDENTIFIER: &str = "gemini";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GeminiProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct GoogleDef;

impl ProviderMeta for GoogleDef {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        "https://generativelanguage.googleapis.com/v1beta/openai"
    }

    fn chat_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
        Cow::Borrowed("/chat/completions")
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        const HEADER_NAME: http::header::HeaderName =
            http::header::HeaderName::from_static("x-goog-api-key");

        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(auth.api_key_for(self.name())?)
            .map_err(|error| GatewayError::Validation(error.to_string()))?;
        headers.insert(HEADER_NAME, value);
        Ok(headers)
    }
}

impl ChatTransform for GoogleDef {}

impl EmbedTransform for GoogleDef {
    fn embeddings_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
        Cow::Borrowed("/embeddings")
    }
}

impl ProviderCapabilities for GoogleDef {
    fn as_embed_transform(&self) -> Option<&dyn EmbedTransform> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::GoogleDef;
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{EmbedTransform, ProviderCapabilities, ProviderMeta},
    };

    #[test]
    fn google_def_uses_compatible_gemini_endpoint_and_auth_header() {
        let provider = GoogleDef;
        let headers = provider
            .build_auth_headers(&ProviderAuth::ApiKey("gemini-key".into()))
            .unwrap();

        assert_eq!(provider.name(), "gemini");
        assert_eq!(
            provider.default_base_url(),
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(
            provider.build_url(provider.default_base_url(), "gemini-2.5-flash"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        );
        assert_eq!(
            provider.embeddings_endpoint_path("gemini-embedding-001"),
            "/embeddings"
        );
        assert_eq!(headers["x-goog-api-key"], "gemini-key");
        assert!(provider.as_embed_transform().is_some());
    }
}
