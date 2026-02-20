use std::error::Error;

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::providers::Provider;
use crate::proxy::types::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};

use super::openai_compatible::{chat_completion, chat_completion_stream};

pub const IDENTIFIER: &str = "deepseek";
const DEFAULT_API_BASE: &str = "https://api.deepseek.com/v1";

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
                api_base: Some(DEFAULT_API_BASE.to_string()),
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
    #[fastrace::trace]
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Box<dyn Error + Send + Sync>> {
        let url = format!(
            "{}/chat/completions",
            self.config.api_base.as_deref().unwrap_or(DEFAULT_API_BASE)
        );
        chat_completion(self.client.clone(), &url, &self.config.api_key, request).await
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
            self.config.api_base.as_deref().unwrap_or(DEFAULT_API_BASE)
        );
        chat_completion_stream(self.client.clone(), &url, &self.config.api_key, request).await
    }
}
