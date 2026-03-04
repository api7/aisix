use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize, de::Error};
use utoipa::ToSchema;

use super::{ConfigProvider, EntityStore};
use crate::{
    config::entities::{
        ResourceEntry,
        types::{HasRateLimit, RateLimit, RateLimitMetric},
    },
    providers::{configs, identifiers},
};

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(untagged)]
pub enum ProviderConfig {
    DeepSeek(configs::DeepSeekProviderConfig),
    Gemini(configs::GeminiProviderConfig),
    OpenAI(configs::OpenAIProviderConfig),
    Mock(configs::MockProviderConfig),
}

impl ProviderConfig {
    pub fn from_json(
        provider: &str,
        json_value: &serde_json::Value,
    ) -> Result<Self, serde_json::Error> {
        match provider {
            identifiers::DEEPSEEK => {
                let config =
                    serde_json::from_value::<configs::DeepSeekProviderConfig>(json_value.clone())?;
                Ok(ProviderConfig::DeepSeek(config))
            }
            identifiers::GEMINI => {
                let config =
                    serde_json::from_value::<configs::GeminiProviderConfig>(json_value.clone())?;
                Ok(ProviderConfig::Gemini(config))
            }
            identifiers::MOCK => Ok(ProviderConfig::Mock(configs::MockProviderConfig {})),
            identifiers::OPENAI => {
                let config =
                    serde_json::from_value::<configs::OpenAIProviderConfig>(json_value.clone())?;
                Ok(ProviderConfig::OpenAI(config))
            }
            _ => Err(serde_json::Error::custom(format!(
                "Unknown provider type: {}",
                provider
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ProviderModel {
    #[serde(skip)]
    pub provider: String,
    #[serde(skip)]
    pub name: String,

    #[serde(rename = "model")]
    #[schema(pattern = "^(deepseek|gemini|openai|mock)/.+$")]
    pub original_model: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Model {
    pub name: String,

    #[serde(flatten)]
    #[schema(inline)]
    pub model: ProviderModel,
    pub provider_config: ProviderConfig,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimit>,
}

impl<'de> Deserialize<'de> for Model {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ModelRaw {
            name: String,
            model: String,
            provider_config: serde_json::Value,
            rate_limit: Option<RateLimit>,
        }

        let raw = ModelRaw::deserialize(deserializer)?;

        let mut model_parts = raw.model.split('/');
        let provider = model_parts.next().unwrap_or("").to_lowercase();
        let provider_model = model_parts.next().unwrap_or("").to_string();
        if provider.is_empty() || provider_model.is_empty() {
            return Err(D::Error::custom(format!(
                "Invalid model format for {}: {}",
                raw.name, raw.model
            )));
        }

        let provider_config = match ProviderConfig::from_json(&provider, &raw.provider_config) {
            Ok(config) => config,
            Err(err) => {
                return Err(D::Error::custom(format!(
                    "Failed to parse provider_config for model {}: {}",
                    raw.name, err
                )));
            }
        };

        Ok(Model {
            name: raw.name,
            model: ProviderModel {
                provider,
                name: provider_model,
                original_model: raw.model,
            },
            provider_config,
            rate_limit: raw.rate_limit,
        })
    }
}

impl HasRateLimit for ResourceEntry<Model> {
    fn rate_limit(&self) -> Option<RateLimit> {
        self.rate_limit.clone()
    }

    fn rate_limit_key(&self, metric: RateLimitMetric) -> String {
        format!("model:{}:{}", self.name, metric)
    }
}

#[derive(Clone)]
pub struct ModelsStore {
    store: EntityStore<Model>,
}

impl ModelsStore {
    pub async fn new(provider: Arc<dyn ConfigProvider + Send + Sync>) -> Self {
        Self {
            store: EntityStore::new(provider, "/models/", "models", None).await,
        }
    }

    pub fn list(&self) -> HashMap<String, ResourceEntry<Model>> {
        self.store.list()
    }

    pub fn get(&self, key: &str) -> Option<ResourceEntry<Model>> {
        self.store.get(key)
    }

    pub fn get_by_name(&self, name: &str) -> Option<ResourceEntry<Model>> {
        for model in self.store.list().values() {
            if model.name == name {
                return Some(model.clone());
            }
        }
        None
    }

    pub fn latest_mod_revision(&self) -> i64 {
        self.store.latest_mod_revision()
    }
}
