pub mod types;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt, stream::BoxStream};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use types::{
    AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicStreamEvent, StreamState,
};

use crate::{
    providers::{Provider, ProviderError, URLFormatter},
    proxy::types::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
};

pub const IDENTIFIER: &str = "anthropic";
const DEFAULT_API_BASE: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AnthropicProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct AnthropicProvider {
    config: AnthropicProviderConfig,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(client: Client, api_key: String) -> Self {
        Self {
            config: AnthropicProviderConfig {
                api_key: api_key.clone(),
                api_base: None,
            },
            client,
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.config.api_base = Some(base_url);
        self
    }

    async fn send_request<T: Serialize>(
        &self,
        request: &T,
    ) -> Result<reqwest::Response, ProviderError> {
        let response = self
            .client
            .post(self.format_url("messages"))
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(ProviderError::ServiceError(status, error_text));
        }

        Ok(response)
    }
}

impl URLFormatter for AnthropicProvider {
    fn format_url(&self, endpoint: &str) -> String {
        format!(
            "{}/{}",
            self.config.api_base.as_deref().unwrap_or(DEFAULT_API_BASE),
            endpoint
        )
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    #[fastrace::trace]
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let anthropic_request = AnthropicMessagesRequest::from(request);
        let response = self.send_request(&anthropic_request).await?;
        let anthropic_response = response.json::<AnthropicMessagesResponse>().await?;
        Ok(ChatCompletionResponse::from(anthropic_response))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, ProviderError>>, ProviderError> {
        let mut anthropic_request = AnthropicMessagesRequest::from(request);
        anthropic_request.stream = Some(true);

        let response = self.send_request(&anthropic_request).await?;
        Ok(Box::pin(parse_anthropic_sse_stream(
            response.bytes_stream(),
        )))
    }
}

/// Parses Anthropic's SSE stream format.
///
/// Anthropic uses typed SSE events (`event: <type>\ndata: <json>\n\n`),
/// unlike OpenAI which uses only `data:` lines with a `[DONE]` sentinel.
/// The `type` field is embedded inside the JSON data payload via serde tag,
/// so we only need to parse `data:` lines and deserialize into `AnthropicStreamEvent`.
fn parse_anthropic_sse_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>>,
) -> impl Stream<Item = Result<ChatCompletionChunk, ProviderError>> {
    stream
        .chain(futures::stream::once(async {
            Ok(Bytes::from_static(b"\n"))
        }))
        .scan(BytesMut::new(), |buffer, result| match result {
            Ok(chunk) => {
                buffer.extend_from_slice(&chunk);

                let mut lines = Vec::new();
                if let Some(last_newline) = buffer.iter().rposition(|&b| b == b'\n') {
                    let complete_data = buffer.split_to(last_newline + 1);
                    let text = String::from_utf8_lossy(&complete_data);
                    for line in text.lines() {
                        lines.push(Ok(line.to_string()));
                    }
                }

                futures::future::ready(Some(futures::stream::iter(lines)))
            }
            Err(err) => futures::future::ready(Some(futures::stream::iter(vec![Err(
                ProviderError::RequestError(err),
            )]))),
        })
        .flatten()
        .scan(StreamState::new(), |state, line| {
            let result = match line {
                Ok(line) => {
                    if let Some(json_str) = line.strip_prefix("data:") {
                        let json_str = json_str.trim_start();
                        match serde_json::from_str::<AnthropicStreamEvent>(json_str) {
                            Ok(event) => state.process_event(event).map(Ok),
                            Err(err) => Some(Err(ProviderError::CodecError(err))),
                        }
                    } else {
                        // Skip `event:` lines, empty lines, and any other non-data lines.
                        None
                    }
                }
                Err(e) => Some(Err(e)),
            };
            futures::future::ready(Some(result))
        })
        .filter_map(futures::future::ready)
}
