use std::borrow::Cow;

use http::HeaderMap;
use serde_json::{Map, Value};

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{
        chat_format::ChatStreamState,
        native::{NativeAnthropicMessagesSupport, NativeOpenAIResponsesSupport},
    },
    types::openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
};

/// Provider metadata with no data transformation logic.
pub trait ProviderMeta: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn default_base_url(&self) -> &'static str;

    /// Chat endpoint path for the provider. Implementations may use `model`
    /// for providers whose route shape depends on the model name.
    fn chat_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
        Cow::Borrowed("/v1/chat/completions")
    }

    fn stream_reader_kind(&self) -> StreamReaderKind {
        StreamReaderKind::Sse
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap>;

    /// Build the final request URL for the chat endpoint.
    fn build_url(&self, base_url: &str, model: &str) -> String {
        format!(
            "{}{}",
            base_url.trim_end_matches('/'),
            self.chat_endpoint_path(model)
        )
    }
}

/// OpenAI Chat to provider-native data conversion.
pub trait ChatTransform: ProviderMeta {
    fn default_quirks(&self) -> CompatQuirks {
        CompatQuirks::NONE
    }

    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        let mut body = serde_json::to_value(request)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;
        self.default_quirks().apply_to_request(&mut body);
        Ok(body)
    }

    fn transform_response(&self, body: Value) -> Result<ChatCompletionResponse> {
        serde_json::from_value(body).map_err(|error| GatewayError::Transform(error.to_string()))
    }

    fn transform_stream_chunk(
        &self,
        raw: &str,
        _state: &mut ChatStreamState,
    ) -> Result<Vec<ChatCompletionChunk>> {
        let quirks = self.default_quirks();
        let trimmed = raw.trim();
        let done_signal = quirks.stream_done_signal.trim();
        let normalized_done_signal = done_signal
            .strip_prefix("data:")
            .map(str::trim_start)
            .unwrap_or(done_signal);

        if trimmed.is_empty()
            || trimmed == done_signal
            || trimmed == normalized_done_signal
            || trimmed.starts_with(':')
            || trimmed.starts_with("event:")
            || trimmed.starts_with("id:")
            || trimmed.starts_with("retry:")
        {
            return Ok(vec![]);
        }

        let Some(line) = trimmed.strip_prefix("data:") else {
            return Ok(vec![]);
        };

        let payload = line.trim_start();
        if payload.is_empty() || payload == done_signal || payload == normalized_done_signal {
            return Ok(vec![]);
        }

        let chunk = serde_json::from_str(payload)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;
        Ok(vec![chunk])
    }
}

/// Capability discovery for optional provider extensions.
pub trait ProviderCapabilities: ChatTransform {
    fn as_native_anthropic_messages(&self) -> Option<&dyn NativeAnthropicMessagesSupport> {
        None
    }

    fn as_native_openai_responses(&self) -> Option<&dyn NativeOpenAIResponsesSupport> {
        None
    }

    fn as_embed_transform(&self) -> Option<&dyn EmbedTransform> {
        None
    }

    fn as_tts_transform(&self) -> Option<&dyn TtsTransform> {
        None
    }

    fn as_stt_transform(&self) -> Option<&dyn SttTransform> {
        None
    }

    fn as_image_gen_transform(&self) -> Option<&dyn ImageGenTransform> {
        None
    }
}

/// Stream decoding mode used by a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamReaderKind {
    Sse,
    AwsEventStream,
    JsonArrayStream,
}

/// Small declarative differences across OpenAI-compatible providers.
#[derive(Debug, Clone)]
pub struct CompatQuirks {
    pub unsupported_params: &'static [&'static str],
    pub param_renames: &'static [(&'static str, &'static str)],
    pub tool_args_may_be_object: bool,
    pub inject_stream_usage: bool,
    pub stream_done_signal: &'static str,
}

impl CompatQuirks {
    pub const NONE: Self = Self {
        unsupported_params: &[],
        param_renames: &[],
        tool_args_may_be_object: false,
        inject_stream_usage: false,
        stream_done_signal: "data: [DONE]",
    };

    /// Apply provider quirks to a serialized request body.
    pub fn apply_to_request(&self, body: &mut Value) {
        let Value::Object(map) = body else {
            return;
        };

        for param in self.unsupported_params {
            map.remove(*param);
        }

        for (from, to) in self.param_renames {
            if let Some(value) = map.remove(*from)
                && !map.contains_key(*to)
            {
                map.insert((*to).to_string(), value);
            }
        }

        if self.inject_stream_usage && map.get("stream").and_then(Value::as_bool) == Some(true) {
            let stream_options = map
                .entry("stream_options".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if !stream_options.is_object() {
                *stream_options = Value::Object(Map::new());
            }
            if let Value::Object(stream_options_map) = stream_options {
                stream_options_map.insert("include_usage".into(), Value::Bool(true));
            }
        }
    }
}

/// Placeholder trait for embeddings until multimodal traits arrive.
pub trait EmbedTransform: Send + Sync + 'static {}

/// Placeholder trait for text-to-speech until multimodal traits arrive.
pub trait TtsTransform: Send + Sync + 'static {}

/// Placeholder trait for speech-to-text until multimodal traits arrive.
pub trait SttTransform: Send + Sync + 'static {}

/// Placeholder trait for image generation until multimodal traits arrive.
pub trait ImageGenTransform: Send + Sync + 'static {}

#[cfg(test)]
mod tests {
    use http::HeaderMap;
    use serde_json::json;

    use super::{ChatTransform, CompatQuirks, ProviderMeta, StreamReaderKind};
    use crate::gateway::{provider_instance::ProviderAuth, traits::chat_format::ChatStreamState};

    struct DummyProvider;

    impl ProviderMeta for DummyProvider {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn default_base_url(&self) -> &'static str {
            "https://example.com"
        }

        fn stream_reader_kind(&self) -> StreamReaderKind {
            StreamReaderKind::Sse
        }

        fn build_auth_headers(
            &self,
            _auth: &ProviderAuth,
        ) -> crate::gateway::error::Result<HeaderMap> {
            Ok(HeaderMap::new())
        }
    }

    impl ChatTransform for DummyProvider {}

    #[test]
    fn apply_to_request_removes_and_renames_fields() {
        let quirks = CompatQuirks {
            unsupported_params: &["seed"],
            param_renames: &[("max_tokens", "max_completion_tokens")],
            ..CompatQuirks::NONE
        };
        let mut body = json!({
            "seed": 7,
            "max_tokens": 256,
            "temperature": 0.2
        });

        quirks.apply_to_request(&mut body);

        assert_eq!(body.get("seed"), None);
        assert_eq!(body["max_completion_tokens"], 256);
        assert_eq!(body["temperature"], 0.2);
    }

    #[test]
    fn apply_to_request_preserves_explicit_destination_value() {
        let quirks = CompatQuirks {
            param_renames: &[("max_tokens", "max_completion_tokens")],
            ..CompatQuirks::NONE
        };
        let mut body = json!({
            "max_tokens": 256,
            "max_completion_tokens": 128
        });

        quirks.apply_to_request(&mut body);

        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], 128);
    }

    #[test]
    fn apply_to_request_injects_stream_usage_when_enabled() {
        let quirks = CompatQuirks {
            inject_stream_usage: true,
            ..CompatQuirks::NONE
        };
        let mut body = json!({
            "stream": true
        });

        quirks.apply_to_request(&mut body);

        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn apply_to_request_skips_stream_usage_for_non_streaming_requests() {
        let quirks = CompatQuirks {
            inject_stream_usage: true,
            ..CompatQuirks::NONE
        };
        let mut body = json!({
            "stream": false
        });

        quirks.apply_to_request(&mut body);

        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn transform_stream_chunk_ignores_sse_control_lines() {
        let provider = DummyProvider;
        let mut state = ChatStreamState::default();

        assert!(
            provider
                .transform_stream_chunk(": keep-alive", &mut state)
                .unwrap()
                .is_empty()
        );
        assert!(
            provider
                .transform_stream_chunk("event: message", &mut state)
                .unwrap()
                .is_empty()
        );
        assert!(
            provider
                .transform_stream_chunk("id: 123", &mut state)
                .unwrap()
                .is_empty()
        );
        assert!(
            provider
                .transform_stream_chunk("retry: 5000", &mut state)
                .unwrap()
                .is_empty()
        );
        assert!(
            provider
                .transform_stream_chunk("not-an-sse-payload", &mut state)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn transform_stream_chunk_ignores_done_signals() {
        let provider = DummyProvider;
        let mut state = ChatStreamState::default();

        assert!(
            provider
                .transform_stream_chunk("data: [DONE]", &mut state)
                .unwrap()
                .is_empty()
        );
        assert!(
            provider
                .transform_stream_chunk("[DONE]", &mut state)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn transform_stream_chunk_parses_only_data_payload() {
        let provider = DummyProvider;
        let mut state = ChatStreamState::default();
        let payload = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"}
            }]
        });

        let chunks = provider
            .transform_stream_chunk(&format!("data: {}", payload), &mut state)
            .unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, "chatcmpl-123");
        assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("Hello"));
    }
}
