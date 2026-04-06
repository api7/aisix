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
        let _ = (native, request, stream);
        Err(GatewayError::NativeNotSupported {
            provider: "unknown".into(),
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
        let _ = (native, body);
        unreachable!("parse_native_response called on a non-native format")
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

/// Stateful data used while transforming provider chunks into hub chunks.
#[derive(Debug, Clone, Default)]
pub struct ChatStreamState {
    pub chunk_index: usize,
    pub tool_call_accumulators: HashMap<usize, ToolCallAccumulator>,
    pub input_tokens: u32,
    pub output_tokens: u32,
}
