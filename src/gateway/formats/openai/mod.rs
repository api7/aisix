use crate::gateway::{
    error::{GatewayError, Result},
    traits::{ChatFormat, NativeHandler, ProviderCapabilities},
    types::{
        common::BridgeContext,
        openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
    },
};

pub struct OpenAIChatFormat;

impl ChatFormat for OpenAIChatFormat {
    type Request = ChatCompletionRequest;
    type Response = ChatCompletionResponse;
    type StreamChunk = ChatCompletionChunk;
    type BridgeState = ();
    type NativeStreamState = ();

    fn name() -> &'static str {
        "openai_chat"
    }

    fn is_stream(req: &Self::Request) -> bool {
        req.stream.unwrap_or(false)
    }

    fn extract_model(req: &Self::Request) -> &str {
        &req.model
    }

    fn to_hub(req: &Self::Request) -> Result<(ChatCompletionRequest, BridgeContext)> {
        Ok((req.clone(), BridgeContext::default()))
    }

    fn from_hub(resp: &Self::Response, _ctx: &BridgeContext) -> Result<Self::Response> {
        Ok(resp.clone())
    }

    fn from_hub_stream(
        chunk: &ChatCompletionChunk,
        _state: &mut Self::BridgeState,
        _ctx: &BridgeContext,
    ) -> Result<Vec<Self::StreamChunk>> {
        Ok(vec![chunk.clone()])
    }

    fn native_support(_provider: &dyn ProviderCapabilities) -> Option<NativeHandler<'_>>
    where
        Self: Sized,
    {
        None
    }

    fn transform_native_stream_chunk(
        _provider: &dyn ProviderCapabilities,
        _raw: &str,
        _state: &mut Self::NativeStreamState,
    ) -> Result<Vec<Self::StreamChunk>> {
        Err(GatewayError::Bridge(
            "OpenAIChatFormat has no native stream path".into(),
        ))
    }

    fn serialize_chunk_payload(chunk: &Self::StreamChunk) -> String {
        serde_json::to_string(chunk).expect("chat completion chunk should serialize")
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::OpenAIChatFormat;
    use crate::gateway::{
        error::GatewayError,
        provider_instance::ProviderAuth,
        traits::{ChatFormat, ProviderCapabilities, ProviderMeta, StreamReaderKind},
        types::{common::BridgeContext, openai::*},
    };

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
        ) -> crate::gateway::error::Result<http::HeaderMap> {
            Ok(http::HeaderMap::new())
        }
    }

    impl crate::gateway::traits::ChatTransform for DummyProvider {}

    impl ProviderCapabilities for DummyProvider {}

    #[test]
    fn request_round_trips_through_hub_identity() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "custom_provider_field": "value"
        }))
        .unwrap();

        let (hub, ctx) = OpenAIChatFormat::to_hub(&request).unwrap();

        assert_eq!(
            serde_json::to_value(&hub).unwrap(),
            serde_json::to_value(&request).unwrap()
        );
        assert!(ctx.anthropic_messages_extras.is_none());
        assert!(ctx.openai_responses_extras.is_none());
        assert!(ctx.passthrough.is_empty());
    }

    #[test]
    fn response_round_trips_from_hub_identity() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }))
        .unwrap();

        let bridged = OpenAIChatFormat::from_hub(&response, &BridgeContext::default()).unwrap();

        assert_eq!(
            serde_json::to_value(&bridged).unwrap(),
            serde_json::to_value(&response).unwrap()
        );
    }

    #[test]
    fn stream_chunk_round_trips_and_serializes_payload() {
        let chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"loc"
                        }
                    }]
                }
            }]
        }))
        .unwrap();

        let emitted =
            OpenAIChatFormat::from_hub_stream(&chunk, &mut (), &BridgeContext::default()).unwrap();

        assert_eq!(emitted.len(), 1);
        assert_eq!(
            serde_json::to_value(&emitted[0]).unwrap(),
            serde_json::to_value(&chunk).unwrap()
        );
        assert_eq!(
            OpenAIChatFormat::serialize_chunk_payload(&emitted[0]),
            serde_json::to_string(&chunk).unwrap()
        );
        assert!(OpenAIChatFormat::stream_end_events(&mut (), &BridgeContext::default()).is_empty());
    }

    #[test]
    fn is_stream_and_extract_model_use_request_fields() {
        let streaming_request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        }))
        .unwrap();
        let non_streaming_request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();

        assert!(OpenAIChatFormat::is_stream(&streaming_request));
        assert!(!OpenAIChatFormat::is_stream(&non_streaming_request));
        assert_eq!(
            OpenAIChatFormat::extract_model(&streaming_request),
            "gpt-4o-mini"
        );
    }

    #[test]
    fn native_stream_path_returns_bridge_error() {
        let provider = DummyProvider;
        let error = OpenAIChatFormat::transform_native_stream_chunk(&provider, "data: {}", &mut ())
            .unwrap_err();

        assert!(matches!(
            error,
            GatewayError::Bridge(message) if message.contains("no native stream path")
        ));
        assert!(OpenAIChatFormat::native_support(&provider).is_none());
    }
}
