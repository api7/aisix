use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;

use super::Provider;
use crate::handler::chat::{ChatCompletionRequest, ChatCompletionResponse};

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
    pub fn new(api_key: String) -> Self {
        Self {
            config: DeepSeekProviderConfig {
                api_key: api_key.clone(),
                api_base: Some("https://api.deepseek.com/v1".to_string()),
            },
            client: Client::new(),
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

    #[fastrace::trace]
    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<reqwest::Response, Box<dyn Error + Send + Sync>> {
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

        Ok(response)
    }
}
