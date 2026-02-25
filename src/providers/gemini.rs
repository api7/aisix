use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
    providers::{
        Provider, ProviderError,
        openai_compatible::{chat_completion, chat_completion_stream, embedding},
    },
    proxy::types::{
        ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
        EmbeddingResponse,
    },
};

pub const IDENTIFIER: &str = "gemini";
const DEFAULT_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/openai";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GeminiProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct GeminiProvider {
    config: GeminiProviderConfig,
    client: Client,
}

impl GeminiProvider {
    #[fastrace::trace]
    pub fn new(client: Client, api_key: String) -> Self {
        Self {
            config: GeminiProviderConfig {
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

// TODO custom input/output struct definition and transformer
#[async_trait]
impl Provider for GeminiProvider {
    #[fastrace::trace(properties = { "request": "{request:?}" })]
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let url = format!(
            "{}/chat/completions",
            self.config.api_base.as_deref().unwrap_or(DEFAULT_API_BASE)
        );
        chat_completion(self.client.clone(), &url, &self.config.api_key, request).await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, ProviderError>>, ProviderError> {
        let url = format!(
            "{}/chat/completions",
            self.config.api_base.as_deref().unwrap_or(DEFAULT_API_BASE)
        );
        chat_completion_stream(self.client.clone(), &url, &self.config.api_key, request).await
    }

    async fn embedding(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ProviderError> {
        let url = format!(
            "{}/embeddings",
            self.config.api_base.as_deref().unwrap_or(DEFAULT_API_BASE)
        );
        embedding(self.client.clone(), &url, &self.config.api_key, request).await
    }
}
