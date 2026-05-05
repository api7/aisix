//! Cohere is currently wired through its OpenAI Compatibility API rather than
//! the native `v2/chat` API surface.
//!
//! Docs:
//! - https://docs.cohere.com/v2/docs/compatibility-api
//! - https://docs.cohere.com/v2/reference/chat

use http::{HeaderMap, HeaderValue, header::AUTHORIZATION};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{ChatTransform, CompatQuirks, EmbedTransform, ProviderCapabilities, ProviderMeta},
    types::{
        embed::{EmbedRequestBody, EmbeddingRequest},
        openai::ChatCompletionRequest,
    },
};

/// Provider identifier string used to look up Cohere in the gateway registry.
pub const IDENTIFIER: &str = "cohere";

/// Configuration for a Cohere provider deployment.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CohereProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct Cohere;

impl ProviderMeta for Cohere {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        "https://api.cohere.ai/compatibility/v1"
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(&format!("Bearer {}", auth.api_key_for(self.name())?))
            .map_err(|error| GatewayError::Validation(error.to_string()))?;
        headers.insert(AUTHORIZATION, value);
        Ok(headers)
    }
}

impl ChatTransform for Cohere {
    fn default_quirks(&self) -> CompatQuirks {
        // Compatibility mode documents these Chat Completions fields as
        // unsupported, so we strip them before forwarding upstream.
        CompatQuirks {
            unsupported_params: &[
                "store",
                "metadata",
                "logit_bias",
                "top_logprobs",
                "n",
                "modalities",
                "prediction",
                "audio",
                "service_tier",
                "parallel_tool_calls",
            ],
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

        // Compatibility mode currently accepts only reasoning_effort="none"
        // or "high" and maps that to Cohere's native thinking behavior.
        enum ReasoningEffortAction {
            Keep,
            Remove,
            Error(String),
        }

        let reasoning_effort_action = match map.get("reasoning_effort") {
            Some(Value::Null) => ReasoningEffortAction::Remove,
            Some(Value::String(value)) if matches!(value.as_str(), "none" | "high") => {
                ReasoningEffortAction::Keep
            }
            Some(Value::String(value)) => ReasoningEffortAction::Error(format!(
                "cohere compatibility API only supports reasoning_effort values \"none\" and \"high\", got \"{value}\""
            )),
            Some(_) => ReasoningEffortAction::Error(
                "cohere compatibility API expects reasoning_effort to be a string".into(),
            ),
            None => ReasoningEffortAction::Keep,
        };

        match reasoning_effort_action {
            ReasoningEffortAction::Keep => {}
            ReasoningEffortAction::Remove => {
                map.remove("reasoning_effort");
            }
            ReasoningEffortAction::Error(message) => {
                return Err(GatewayError::Validation(message));
            }
        }

        Ok(body)
    }
}

impl EmbedTransform for Cohere {
    fn transform_embeddings_request(&self, request: &EmbeddingRequest) -> Result<EmbedRequestBody> {
        let mut body = serde_json::to_value(request)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;

        if let Value::Object(map) = &mut body {
            // Compatibility embeddings keep encoding_format but do not support
            // dimensions or user.
            map.remove("dimensions");
            map.remove("user");
        }

        Ok(EmbedRequestBody::Json(body))
    }
}

impl ProviderCapabilities for Cohere {
    fn as_embed_transform(&self) -> Option<&dyn EmbedTransform> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::Cohere;
    use crate::{
        traits::{ChatTransform, EmbedTransform, ProviderCapabilities, ProviderMeta},
        types::{
            embed::{EmbedRequestBody, EmbeddingRequest},
            openai::ChatCompletionRequest,
        },
    };

    #[test]
    fn provider_metadata_and_urls_are_correct() {
        let provider = Cohere;

        assert_eq!(provider.name(), "cohere");
        assert_eq!(
            provider.default_base_url(),
            "https://api.cohere.ai/compatibility/v1"
        );
        assert_eq!(
            provider.build_url(provider.default_base_url(), "ignored"),
            "https://api.cohere.ai/compatibility/v1/chat/completions"
        );
        assert!(provider.as_embed_transform().is_some());
    }

    #[test]
    fn transform_request_applies_cohere_quirks() {
        let provider = Cohere;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "command-a-03-2025",
            "messages": [{"role": "user", "content": "hello"}],
            "logprobs": true,
            "top_logprobs": 5,
            "n": 3,
            "parallel_tool_calls": true,
            "reasoning_effort": "high",
            "metadata": {"trace_id": "abc"},
            "store": true,
            "logit_bias": {"42": 100},
            "modalities": ["text"],
            "prediction": {"type": "content", "content": "hint"},
            "audio": {"voice": "alloy"},
            "service_tier": "auto"
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed["logprobs"], true);
        assert_eq!(transformed.get("top_logprobs"), None);
        assert_eq!(transformed.get("n"), None);
        assert_eq!(transformed.get("parallel_tool_calls"), None);
        assert_eq!(transformed.get("metadata"), None);
        assert_eq!(transformed.get("store"), None);
        assert_eq!(transformed.get("logit_bias"), None);
        assert_eq!(transformed.get("modalities"), None);
        assert_eq!(transformed.get("prediction"), None);
        assert_eq!(transformed.get("audio"), None);
        assert_eq!(transformed.get("service_tier"), None);
        assert_eq!(transformed["reasoning_effort"], "high");
    }

    #[test]
    fn transform_request_rejects_unsupported_reasoning_effort_values() {
        let provider = Cohere;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "command-a-03-2025",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "medium"
        }))
        .unwrap();

        let error = provider.transform_request(&request).unwrap_err();

        assert_matches!(
            error,
            crate::error::GatewayError::Validation(message)
                if message.contains("reasoning_effort")
                    && message.contains("none")
                    && message.contains("high")
        );
    }

    #[test]
    fn transform_embeddings_request_strips_unsupported_fields() {
        let provider = Cohere;
        let request: EmbeddingRequest = serde_json::from_value(json!({
            "model": "embed-v4.0",
            "input": ["hello"],
            "dimensions": 256,
            "encoding_format": "float",
            "user": "user-123"
        }))
        .unwrap();

        let body = provider.transform_embeddings_request(&request).unwrap();

        match body {
            EmbedRequestBody::Json(value) => {
                assert_eq!(value["encoding_format"], "float");
                assert_eq!(value.get("dimensions"), None);
                assert_eq!(value.get("user"), None);
            }
        }
    }
}
