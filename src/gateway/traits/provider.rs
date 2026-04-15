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
    types::{
        embed::{EmbedRequestBody, EmbedResponseBody, EmbeddingRequest, EmbeddingResponse},
        openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
    },
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

    /// Build the final request URL for an arbitrary provider endpoint.
    fn build_url_for_endpoint(&self, base_url: &str, endpoint_path: &str) -> String {
        let Ok(mut parsed) = reqwest::Url::parse(base_url) else {
            return format!("{}{}", base_url.trim_end_matches('/'), endpoint_path);
        };

        let base_segments = parsed
            .path_segments()
            .map(|segments| {
                segments
                    .filter(|segment| !segment.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let endpoint_segments = endpoint_path
            .split('/')
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>();

        let max_overlap = base_segments.len().min(endpoint_segments.len());
        let overlap = (1..=max_overlap)
            .rev()
            .find(|count| {
                base_segments[base_segments.len() - count..] == endpoint_segments[..*count]
            })
            .unwrap_or(0);

        let mut joined_segments = base_segments;
        joined_segments.extend_from_slice(&endpoint_segments[overlap..]);

        parsed.set_path(&format!("/{}", joined_segments.join("/")));
        parsed.to_string()
    }

    /// Build the final request URL for the chat endpoint.
    fn build_url(&self, base_url: &str, model: &str) -> String {
        let endpoint_path = self.chat_endpoint_path(model);
        self.build_url_for_endpoint(base_url, endpoint_path.as_ref())
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

    #[allow(dead_code)]
    fn as_native_openai_responses(&self) -> Option<&dyn NativeOpenAIResponsesSupport> {
        None
    }

    fn as_embed_transform(&self) -> Option<&dyn EmbedTransform> {
        None
    }

    #[allow(dead_code)]
    fn as_tts_transform(&self) -> Option<&dyn TtsTransform> {
        None
    }

    #[allow(dead_code)]
    fn as_stt_transform(&self) -> Option<&dyn SttTransform> {
        None
    }

    #[allow(dead_code)]
    fn as_image_gen_transform(&self) -> Option<&dyn ImageGenTransform> {
        None
    }
}

/// Stream decoding mode used by a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamReaderKind {
    Sse,
    #[allow(dead_code)]
    AwsEventStream,
    #[allow(dead_code)]
    JsonArrayStream,
}

/// Small declarative differences across OpenAI-compatible providers.
#[derive(Debug, Clone)]
pub struct CompatQuirks {
    pub unsupported_params: &'static [&'static str],
    pub param_renames: &'static [(&'static str, &'static str)],
    #[allow(dead_code)]
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

/// Provider-specific embeddings request and response conversion.
pub trait EmbedTransform: Send + Sync + 'static {
    fn embeddings_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
        Cow::Borrowed("/v1/embeddings")
    }

    fn transform_embeddings_request(&self, request: &EmbeddingRequest) -> Result<EmbedRequestBody> {
        let body = serde_json::to_value(request)
            .map_err(|error| GatewayError::Transform(error.to_string()))?;
        Ok(EmbedRequestBody::Json(body))
    }

    fn transform_embeddings_response(&self, body: EmbedResponseBody) -> Result<EmbeddingResponse> {
        let EmbedResponseBody::Json(body) = body;

        serde_json::from_value(body).map_err(|error| GatewayError::Transform(error.to_string()))
    }
}

/// Placeholder trait for text-to-speech until multimodal traits arrive.
#[allow(dead_code)]
pub trait TtsTransform: Send + Sync + 'static {}

/// Placeholder trait for speech-to-text until multimodal traits arrive.
#[allow(dead_code)]
pub trait SttTransform: Send + Sync + 'static {}

/// Placeholder trait for image generation until multimodal traits arrive.
#[allow(dead_code)]
pub trait ImageGenTransform: Send + Sync + 'static {}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use http::HeaderMap;
    use serde_json::json;

    use super::{ChatTransform, CompatQuirks, EmbedTransform, ProviderMeta, StreamReaderKind};
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::chat_format::ChatStreamState,
        types::embed::{EmbedRequestBody, EmbedResponseBody, EmbeddingRequest},
    };

    struct DummyProvider;

    struct VersionedDummyProvider;

    impl ProviderMeta for DummyProvider {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn default_base_url(&self) -> &'static str {
            "https://example.com"
        }

        fn chat_endpoint_path(&self, _model: &str) -> Cow<'static, str> {
            Cow::Borrowed("/chat/completions")
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

    impl EmbedTransform for DummyProvider {}

    impl ProviderMeta for VersionedDummyProvider {
        fn name(&self) -> &'static str {
            "versioned-dummy"
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

    impl ChatTransform for VersionedDummyProvider {}

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
    fn embed_transform_defaults_to_json_round_trip() {
        let provider = DummyProvider;
        let request: EmbeddingRequest = serde_json::from_value(json!({
            "model": "text-embedding-3-large",
            "input": "hello"
        }))
        .unwrap();

        let body = provider.transform_embeddings_request(&request).unwrap();
        match body {
            EmbedRequestBody::Json(value) => {
                pretty_assertions::assert_eq!(value["model"], "text-embedding-3-large");
                pretty_assertions::assert_eq!(value["input"], "hello");
            }
        }

        let response = provider
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

        pretty_assertions::assert_eq!(response.data.len(), 1);

        let usage = match response.usage {
            Some(usage) => usage,
            None => panic!("expected usage in embedding response"),
        };

        pretty_assertions::assert_eq!(usage.total_tokens, 2);
    }

    #[test]
    fn build_url_avoids_duplicate_version_prefixes() {
        let provider = VersionedDummyProvider;

        assert_eq!(
            provider.build_url("https://example.com/v1", "ignored"),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn build_url_appends_chat_path_without_overlap() {
        let provider = DummyProvider;

        assert_eq!(
            provider.build_url("https://example.com", "ignored"),
            "https://example.com/chat/completions"
        );
    }

    #[test]
    fn build_url_handles_trailing_slash_in_base_url() {
        let provider = DummyProvider;

        assert_eq!(
            provider.build_url("https://example.com/v1/", "ignored"),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn build_url_falls_back_for_invalid_urls() {
        let provider = DummyProvider;

        assert_eq!(
            provider.build_url("not-a-valid-url", "ignored"),
            "not-a-valid-url/chat/completions"
        );
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
