use std::borrow::Cow;

use http::HeaderMap;
use serde_json::{Map, Value};

use crate::gateway::{
    error::{GatewayError, Result},
    traits::{
        chat_format::ChatStreamState,
        native::{NativeAnthropicMessagesSupport, NativeOpenAIResponsesSupport},
    },
    types::openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
};

/// Authentication material used by provider definitions.
#[derive(Debug, Clone, Default)]
pub enum ProviderAuth {
    ApiKey(String),
    #[default]
    None,
}

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
        if raw.trim().is_empty() || raw.starts_with(quirks.stream_done_signal) {
            return Ok(vec![]);
        }

        let line = raw.strip_prefix("data: ").unwrap_or(raw);
        let chunk = serde_json::from_str(line)
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
            if let Some(value) = map.remove(*from) {
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
    use serde_json::json;

    use super::CompatQuirks;

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
}
