use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt, stream::BoxStream};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;

use super::Provider;
use crate::handlers::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekProviderConfig {
    pub api_key: String,
    pub api_base: Option<String>,
}

pub struct DeepSeekProvider {
    config: DeepSeekProviderConfig,
    client: Client,
}

impl DeepSeekProvider {
    #[fastrace::trace]
    pub fn new(client: Client, api_key: String) -> Self {
        Self {
            config: DeepSeekProviderConfig {
                api_key: api_key.clone(),
                api_base: Some("https://api.deepseek.com/v1".to_string()),
            },
            client,
        }
    }

    #[allow(dead_code)]
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.config.api_base = Some(base_url);
        self
    }
}

#[async_trait]
impl Provider for DeepSeekProvider {
    #[fastrace::trace(properties = { "request": "{request:?}" })]
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Box<dyn Error + Send + Sync>> {
        let url = format!(
            "{}/chat/completions",
            self.config.api_base.as_ref().unwrap()
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("DeepSeek API error {}: {}", status, error_text).into());
        }

        let completion = response.json::<ChatCompletionResponse>().await?;
        Ok(completion)
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<
        BoxStream<'static, Result<ChatCompletionChunk, Box<dyn Error + Send + Sync>>>,
        Box<dyn Error + Send + Sync>,
    > {
        let url = format!(
            "{}/chat/completions",
            self.config.api_base.as_ref().unwrap()
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("DeepSeek API error {}: {}", status, error_text).into());
        }

        Ok(Box::pin(parse_sse_stream(response.bytes_stream())))
    }
}

fn parse_sse_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>>,
) -> impl Stream<Item = Result<ChatCompletionChunk, Box<dyn Error + Send + Sync>>> {
    stream
        .chain(futures::stream::once(async {
            Ok(Bytes::from_static(b"\n"))
        }))
        .scan(BytesMut::new(), |buffer, result| {
            match result {
                Ok(chunk) => {
                    buffer.extend_from_slice(&chunk);

                    let mut lines = Vec::new();

                    // If there are incomplete lines, then this chunk will not terminate with a line break.
                    // We accumulate these remaining bytes and append them during the next extraction.
                    if let Some(last_newline) = buffer.iter().rposition(|&b| b == b'\n') {
                        let complete_data = buffer.split_to(last_newline + 1);

                        let text = String::from_utf8_lossy(&complete_data);
                        for line in text.lines() {
                            lines.push(Ok(line.to_string()));
                        }
                    }

                    futures::future::ready(Some(futures::stream::iter(lines)))
                }
                Err(e) => {
                    let err: Box<dyn Error + Send + Sync> = Box::new(e);
                    futures::future::ready(Some(futures::stream::iter(vec![Err(err)])))
                }
            }
        })
        .flatten()
        .filter_map(|line_result| async move {
            match line_result {
                Ok(line) => {
                    // Only process lines starting with "data: "
                    if let Some(json_str) = line.strip_prefix("data: ") {
                        // Skip [DONE] events
                        // Stream termination signifies completion; we need not explicitly propagate them.
                        if json_str == "[DONE]" {
                            return None;
                        }

                        match serde_json::from_str::<ChatCompletionChunk>(json_str) {
                            Ok(chunk) => Some(Ok(chunk)),
                            Err(e) => {
                                // Propagate parse errors instead of skipping
                                let err: Box<dyn Error + Send + Sync> =
                                    format!("Failed to parse SSE chunk: {}", e).into();
                                Some(Err(err))
                            }
                        }
                    } else {
                        None
                    }
                }
                Err(e) => Some(Err(e)),
            }
        })
}
