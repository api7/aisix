use http::{HeaderMap, HeaderValue, header::AUTHORIZATION};

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{ChatTransform, CompatQuirks, ProviderCapabilities, ProviderMeta},
};

pub struct OpenAIDef;

impl ProviderMeta for OpenAIDef {
    fn name(&self) -> &'static str {
        "openai"
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

impl ProviderCapabilities for OpenAIDef {}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::OpenAIDef;
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{ChatTransform, ProviderMeta},
        types::openai::ChatCompletionRequest,
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
}
