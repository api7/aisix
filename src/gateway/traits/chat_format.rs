use std::collections::HashMap;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::gateway::{
    error::{GatewayError, Result},
    traits::{NativeHandler, ProviderCapabilities},
    types::{
        common::BridgeContext,
        openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
    },
};

/// A complete chat API format contract and its bridge rules to the hub format.
pub trait ChatFormat: Send + Sync + 'static {
    /// Request type for this format.
    type Request: DeserializeOwned + Serialize + Send + Sync;
    /// Non-streaming response type for this format.
    type Response: Serialize + Send + Sync;
    /// Streaming chunk type for this format.
    type StreamChunk: Serialize + Send + Sync;
    /// Stateful bridge data used while converting hub chunks.
    type BridgeState: Default + Send + Unpin;
    /// Stateful bridge data used on native streaming paths.
    type NativeStreamState: Default + Send + Unpin;

    /// Stable format name used for logs and diagnostics.
    fn name() -> &'static str;

    /// Whether the request expects a streaming response.
    fn is_stream(req: &Self::Request) -> bool;

    /// Extract the model identifier from the request.
    fn extract_model(req: &Self::Request) -> &str;

    /// Convert this request into the hub request plus side-channel bridge data.
    fn to_hub(req: &Self::Request) -> Result<(ChatCompletionRequest, BridgeContext)>;

    /// Convert a hub response back into this format.
    fn from_hub(resp: &ChatCompletionResponse, ctx: &BridgeContext) -> Result<Self::Response>;

    /// Convert a hub streaming chunk into zero or more chunks of this format.
    fn from_hub_stream(
        chunk: &ChatCompletionChunk,
        state: &mut Self::BridgeState,
        ctx: &BridgeContext,
    ) -> Result<Vec<Self::StreamChunk>>;

    /// Emit any format-specific end-of-stream events.
    fn stream_end_events(
        _state: &mut Self::BridgeState,
        _ctx: &BridgeContext,
    ) -> Vec<Self::StreamChunk> {
        vec![]
    }

    /// Return a native handler when the provider can bypass the hub format.
    fn native_support(_provider: &dyn ProviderCapabilities) -> Option<NativeHandler<'_>>
    where
        Self: Sized,
    {
        None
    }

    /// Prepare a native request body for providers that support this format directly.
    fn call_native(
        native: &NativeHandler<'_>,
        request: &Self::Request,
        stream: bool,
    ) -> Result<(String, Value)>
    where
        Self: Sized,
    {
        let _ = (request, stream);
        Err(GatewayError::NativeNotSupported {
            provider: native.provider_name().into(),
        })
    }

    /// Convert a native streaming chunk into zero or more chunks of this format.
    fn transform_native_stream_chunk(
        provider: &dyn ProviderCapabilities,
        raw: &str,
        state: &mut Self::NativeStreamState,
    ) -> Result<Vec<Self::StreamChunk>>;

    /// Parse a native non-streaming response into this format.
    fn parse_native_response(native: &NativeHandler<'_>, body: Value) -> Result<Self::Response>
    where
        Self: Sized,
    {
        let _ = body;
        Err(GatewayError::Bridge(format!(
            "parse_native_response called on a non-native format for provider {}",
            native.provider_name()
        )))
    }

    /// Serialize a chunk into the JSON payload used by SSE framing.
    fn serialize_chunk_payload(chunk: &Self::StreamChunk) -> String;

    /// Optional SSE event type for this chunk.
    fn sse_event_type(_chunk: &Self::StreamChunk) -> Option<&'static str> {
        None
    }
}

/// Incremental state for reconstructing tool calls across hub chunks.
#[derive(Debug, Clone, Default)]
pub struct ToolCallAccumulator {
    pub id: Option<String>,
    pub kind: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
}

/// Key for partially assembled tool calls: (choice_index, tool_call_index).
pub type ToolCallAccumulatorKey = (u32, usize);

/// Stateful data used while transforming provider chunks into hub chunks.
#[derive(Debug, Clone, Default)]
pub struct ChatStreamState {
    pub chunk_index: usize,
    pub tool_call_accumulators: HashMap<ToolCallAccumulatorKey, ToolCallAccumulator>,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use http::HeaderMap;
    use serde_json::json;

    use super::{ChatFormat, ChatStreamState, ToolCallAccumulator};
    use crate::gateway::{
        error::GatewayError,
        traits::{
            NativeHandler, NativeOpenAIResponsesSupport, ProviderAuth, ProviderMeta,
            StreamReaderKind, provider::ChatTransform,
        },
        types::{
            common::BridgeContext,
            openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
        },
    };

    struct DummyNativeProvider;

    impl ProviderMeta for DummyNativeProvider {
        fn name(&self) -> &'static str {
            "dummy-native-provider"
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

    impl ChatTransform for DummyNativeProvider {}

    impl NativeOpenAIResponsesSupport for DummyNativeProvider {
        fn native_openai_responses_endpoint(&self, _model: &str) -> Cow<'static, str> {
            Cow::Borrowed("/v1/responses")
        }

        fn transform_openai_responses_request(
            &self,
            _req: &crate::gateway::types::openai::responses::ResponsesApiRequest,
        ) -> crate::gateway::error::Result<serde_json::Value> {
            Ok(json!({}))
        }

        fn transform_openai_responses_response(
            &self,
            _body: serde_json::Value,
        ) -> crate::gateway::error::Result<
            crate::gateway::types::openai::responses::ResponsesApiResponse,
        > {
            unreachable!("not used in this test")
        }

        fn transform_openai_responses_stream_chunk(
            &self,
            _raw: &str,
            _state: &mut crate::gateway::traits::OpenAIResponsesNativeStreamState,
        ) -> crate::gateway::error::Result<
            Vec<crate::gateway::types::openai::responses::ResponsesApiStreamEvent>,
        > {
            Ok(vec![])
        }
    }

    struct DummyFormat;

    impl ChatFormat for DummyFormat {
        type Request = serde_json::Value;
        type Response = serde_json::Value;
        type StreamChunk = serde_json::Value;
        type BridgeState = ();
        type NativeStreamState = ();

        fn name() -> &'static str {
            "dummy"
        }

        fn is_stream(_req: &Self::Request) -> bool {
            false
        }

        fn extract_model(_req: &Self::Request) -> &str {
            "dummy-model"
        }

        fn to_hub(
            _req: &Self::Request,
        ) -> crate::gateway::error::Result<(ChatCompletionRequest, BridgeContext)> {
            unreachable!("not used in this test")
        }

        fn from_hub(
            _resp: &ChatCompletionResponse,
            _ctx: &BridgeContext,
        ) -> crate::gateway::error::Result<Self::Response> {
            unreachable!("not used in this test")
        }

        fn from_hub_stream(
            _chunk: &ChatCompletionChunk,
            _state: &mut Self::BridgeState,
            _ctx: &BridgeContext,
        ) -> crate::gateway::error::Result<Vec<Self::StreamChunk>> {
            Ok(vec![])
        }

        fn transform_native_stream_chunk(
            _provider: &dyn crate::gateway::traits::ProviderCapabilities,
            _raw: &str,
            _state: &mut Self::NativeStreamState,
        ) -> crate::gateway::error::Result<Vec<Self::StreamChunk>> {
            Ok(vec![])
        }

        fn serialize_chunk_payload(chunk: &Self::StreamChunk) -> String {
            serde_json::to_string(chunk).unwrap()
        }
    }

    #[test]
    fn default_call_native_uses_provider_name() {
        let provider = DummyNativeProvider;
        let native = NativeHandler::OpenAIResponses(&provider);

        let error = DummyFormat::call_native(&native, &json!({}), false).unwrap_err();
        assert!(matches!(
            error,
            GatewayError::NativeNotSupported { provider } if provider == "dummy-native-provider"
        ));
    }

    #[test]
    fn default_parse_native_response_returns_bridge_error() {
        let provider = DummyNativeProvider;
        let native = NativeHandler::OpenAIResponses(&provider);

        let error = DummyFormat::parse_native_response(&native, json!({})).unwrap_err();
        assert!(matches!(
            error,
            GatewayError::Bridge(message)
            if message.contains("parse_native_response called on a non-native format")
                && message.contains("dummy-native-provider")
        ));
    }

    #[test]
    fn stream_state_separates_tool_call_accumulators_by_choice_and_index() {
        let mut state = ChatStreamState::default();
        state.tool_call_accumulators.insert(
            (0, 0),
            ToolCallAccumulator {
                arguments: "first".into(),
                ..Default::default()
            },
        );
        state.tool_call_accumulators.insert(
            (1, 0),
            ToolCallAccumulator {
                arguments: "second".into(),
                ..Default::default()
            },
        );

        assert_eq!(state.tool_call_accumulators.len(), 2);
        assert_eq!(
            state.tool_call_accumulators.get(&(0, 0)).unwrap().arguments,
            "first"
        );
        assert_eq!(
            state.tool_call_accumulators.get(&(1, 0)).unwrap().arguments,
            "second"
        );
    }
}
