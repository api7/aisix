use std::borrow::Cow;

use serde_json::Value;

use crate::gateway::{
    error::Result,
    traits::provider::ChatTransform,
    types::{
        anthropic::{AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicStreamEvent},
        common::Usage,
        openai::responses::{ResponsesApiRequest, ResponsesApiResponse, ResponsesApiStreamEvent},
    },
};

/// Stateful data for native Anthropic Messages streaming transforms.
#[derive(Debug, Clone, Default)]
pub struct AnthropicMessagesNativeStreamState {
    pub usage: Usage,
}

/// Stateful data for native OpenAI Responses streaming transforms.
#[derive(Debug, Clone, Default)]
pub struct OpenAIResponsesNativeStreamState;

/// Native Anthropic Messages support for providers that can bypass the hub format.
pub trait NativeAnthropicMessagesSupport: ChatTransform {
    fn native_anthropic_messages_endpoint(&self, model: &str) -> Cow<'static, str>;
    fn transform_anthropic_messages_request(&self, req: &AnthropicMessagesRequest)
    -> Result<Value>;
    fn transform_anthropic_messages_response(
        &self,
        body: Value,
    ) -> Result<AnthropicMessagesResponse>;
    fn transform_anthropic_messages_stream_chunk(
        &self,
        raw: &str,
        state: &mut AnthropicMessagesNativeStreamState,
    ) -> Result<Vec<AnthropicStreamEvent>>;
}

/// Native OpenAI Responses support for providers that can bypass the hub format.
pub trait NativeOpenAIResponsesSupport: ChatTransform {
    fn native_openai_responses_endpoint(&self, model: &str) -> Cow<'static, str>;
    fn transform_openai_responses_request(&self, req: &ResponsesApiRequest) -> Result<Value>;
    fn transform_openai_responses_response(&self, body: Value) -> Result<ResponsesApiResponse>;
    fn transform_openai_responses_stream_chunk(
        &self,
        raw: &str,
        state: &mut OpenAIResponsesNativeStreamState,
    ) -> Result<Vec<ResponsesApiStreamEvent>>;
}

/// Type-erased native handler returned by format implementations.
pub enum NativeHandler<'a> {
    AnthropicMessages(&'a dyn NativeAnthropicMessagesSupport),
    OpenAIResponses(&'a dyn NativeOpenAIResponsesSupport),
}

impl NativeHandler<'_> {
    /// Returns the provider name behind this native handler.
    pub fn provider_name(&self) -> &'static str {
        match self {
            Self::AnthropicMessages(handler) => handler.name(),
            Self::OpenAIResponses(handler) => handler.name(),
        }
    }
}
