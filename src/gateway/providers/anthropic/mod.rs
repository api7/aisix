pub mod transform;

use std::borrow::Cow;

use http::{HeaderMap, HeaderValue};
use serde_json::Value;

use self::transform::{
    anthropic_to_openai_response, openai_to_anthropic_request, parse_anthropic_native_sse,
    parse_anthropic_sse_to_openai,
};
use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{
        AnthropicMessagesNativeStreamState, ChatStreamState, ChatTransform,
        NativeAnthropicMessagesSupport, ProviderCapabilities, ProviderMeta,
    },
    types::{
        anthropic::{AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicStreamEvent},
        openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
    },
};

pub struct AnthropicDef;

impl ProviderMeta for AnthropicDef {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn default_base_url(&self) -> &'static str {
        "https://api.anthropic.com"
    }

    fn chat_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
        Cow::Borrowed("/v1/messages")
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(auth.api_key_for(self.name())?)
                .map_err(|error| GatewayError::Validation(error.to_string()))?,
        );
        headers.insert(
            http::header::HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static("2023-06-01"),
        );
        Ok(headers)
    }
}

impl ChatTransform for AnthropicDef {
    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        serde_json::to_value(openai_to_anthropic_request(request)?)
            .map_err(|error| GatewayError::Transform(error.to_string()))
    }

    fn transform_response(&self, body: Value) -> Result<ChatCompletionResponse> {
        let response: AnthropicMessagesResponse = serde_json::from_value(body)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;
        anthropic_to_openai_response(&response)
    }

    fn transform_stream_chunk(
        &self,
        raw: &str,
        state: &mut ChatStreamState,
    ) -> Result<Vec<ChatCompletionChunk>> {
        parse_anthropic_sse_to_openai(raw, state)
    }
}

impl ProviderCapabilities for AnthropicDef {
    fn as_native_anthropic_messages(&self) -> Option<&dyn NativeAnthropicMessagesSupport> {
        Some(self)
    }
}

impl NativeAnthropicMessagesSupport for AnthropicDef {
    fn native_anthropic_messages_endpoint(&self, _model: &str) -> Cow<'static, str> {
        Cow::Borrowed("/v1/messages")
    }

    fn transform_anthropic_messages_request(
        &self,
        req: &AnthropicMessagesRequest,
    ) -> Result<Value> {
        serde_json::to_value(req).map_err(|error| GatewayError::Transform(error.to_string()))
    }

    fn transform_anthropic_messages_response(
        &self,
        body: Value,
    ) -> Result<AnthropicMessagesResponse> {
        serde_json::from_value(body).map_err(|error| GatewayError::Transform(error.to_string()))
    }

    fn transform_anthropic_messages_stream_chunk(
        &self,
        raw: &str,
        _state: &mut AnthropicMessagesNativeStreamState,
    ) -> Result<Vec<AnthropicStreamEvent>> {
        parse_anthropic_native_sse(raw)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::AnthropicDef;
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{
            AnthropicMessagesNativeStreamState, ChatTransform, NativeAnthropicMessagesSupport,
            ProviderCapabilities, ProviderMeta,
        },
        types::anthropic::{AnthropicMessagesRequest, AnthropicStreamEvent},
    };

    #[test]
    fn anthropic_def_builds_expected_headers_and_registers_native_support() {
        let provider = AnthropicDef;
        let headers = provider
            .build_auth_headers(&ProviderAuth::ApiKey("sk-ant".into()))
            .unwrap();

        assert_eq!(provider.name(), "anthropic");
        assert_eq!(provider.default_base_url(), "https://api.anthropic.com");
        assert_eq!(provider.chat_endpoint_path("ignored"), "/v1/messages");
        assert_eq!(headers["x-api-key"], "sk-ant");
        assert_eq!(headers["anthropic-version"], "2023-06-01");
        assert!(provider.as_native_anthropic_messages().is_some());
    }

    #[test]
    fn native_anthropic_passthrough_serializes_and_parses() {
        let provider = AnthropicDef;
        let request: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();

        let body = provider
            .transform_anthropic_messages_request(&request)
            .unwrap();
        let parsed = provider
            .transform_anthropic_messages_response(json!({
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 2}
            }))
            .unwrap();
        let events = provider
            .transform_anthropic_messages_stream_chunk(
                r#"data: {"type":"ping"}"#,
                &mut AnthropicMessagesNativeStreamState::default(),
            )
            .unwrap();

        assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(parsed.id, "msg_123");
        assert!(matches!(events.as_slice(), [AnthropicStreamEvent::Ping]));
    }

    #[test]
    fn transform_request_and_response_bridge_through_anthropic_def() {
        let provider = AnthropicDef;
        let body = provider
            .transform_request(
                &serde_json::from_value(json!({
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hello"}]
                }))
                .unwrap(),
            )
            .unwrap();
        let response = provider
            .transform_response(json!({
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 2}
            }))
            .unwrap();

        assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(response.choices[0].message.role, "assistant");
        assert_eq!(
            response.choices[0].message.content.as_ref().map(|_| true),
            Some(true)
        );
    }
}
