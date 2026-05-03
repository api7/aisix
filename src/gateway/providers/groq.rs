use http::{HeaderMap, HeaderValue, header::AUTHORIZATION};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{ChatTransform, CompatQuirks, ProviderCapabilities, ProviderMeta},
    types::openai::ChatCompletionRequest,
};

/// Provider identifier string used to look up Groq in the gateway registry.
pub const IDENTIFIER: &str = "groq";

/// Configuration for a Groq provider deployment.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GroqProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct Groq;

impl ProviderMeta for Groq {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        "https://api.groq.com/openai"
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(&format!("Bearer {}", auth.api_key_for(self.name())?))
            .map_err(|error| GatewayError::Validation(error.to_string()))?;
        headers.insert(AUTHORIZATION, value);
        Ok(headers)
    }
}

impl ChatTransform for Groq {
    fn default_quirks(&self) -> CompatQuirks {
        CompatQuirks {
            unsupported_params: &["logprobs", "logit_bias", "top_logprobs"],
            ..CompatQuirks::NONE
        }
    }

    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        let mut body = serde_json::to_value(request)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;
        self.default_quirks().apply_to_request(&mut body);

        let Value::Object(map) = &mut body else {
            return Ok(body);
        };

        if let Some(Value::Array(messages)) = map.get_mut("messages") {
            for message in messages {
                if let Value::Object(message_map) = message {
                    message_map.remove("name");
                }
            }
        }

        if map.get("n").is_some() {
            map.insert("n".into(), Value::from(1));
        }

        Ok(body)
    }
}

impl ProviderCapabilities for Groq {}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::Groq;
    use crate::gateway::{
        traits::{ChatTransform, ProviderMeta},
        types::openai::ChatCompletionRequest,
    };

    #[test]
    fn provider_metadata_and_url_are_correct() {
        let provider = Groq;

        pretty_assertions::assert_eq!(provider.name(), "groq");
        pretty_assertions::assert_eq!(provider.default_base_url(), "https://api.groq.com/openai");

        pretty_assertions::assert_eq!(
            provider.build_url(provider.default_base_url(), "ignored"),
            "https://api.groq.com/openai/v1/chat/completions"
        );
    }

    #[test]
    fn transform_request_strips_groq_unsupported_parameters() {
        let provider = Groq;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "hi", "name": "alice"}],
            "n": 3,
            "logprobs": true,
            "top_logprobs": 5,
            "logit_bias": {"42": 100}
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed.get("logprobs"), None);
        assert_eq!(transformed.get("top_logprobs"), None);
        assert_eq!(transformed.get("logit_bias"), None);
        assert_eq!(transformed["messages"][0].get("name"), None);
        assert_eq!(transformed["n"], 1);
    }
}
