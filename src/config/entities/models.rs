use super::{ConfigProvider, EntityStore};
use crate::providers::{configs, identifiers};
use serde::{Deserialize, Serialize, de::Error};
use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderConfig {
    DeepSeek(configs::DeepSeekProviderConfig),
    Gemini(configs::GeminiProviderConfig),
    OpenAI(configs::OpenAIProviderConfig),
    Mock,
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
            identifiers::MOCK => Ok(ProviderConfig::Mock),
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

#[derive(Debug, Clone, Serialize)]
pub struct Model {
    pub name: String,
    pub model: String,

    #[serde(skip)]
    provider: OnceLock<String>,
    #[serde(skip)]
    provider_model: OnceLock<String>,
    #[serde(skip)]
    pub provider_config: OnceLock<ProviderConfig>,
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
            model: raw.model,

            provider: OnceLock::from(provider),
            provider_model: OnceLock::from(provider_model),
            provider_config: OnceLock::from(provider_config),
        })
    }
}

impl Model {
    pub fn get_provider(&self) -> &str {
        self.provider.get().unwrap()
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

    pub fn list(&self) -> HashMap<String, Model> {
        self.store.list()
    }

    pub fn get(&self, key: &str) -> Option<Model> {
        self.store.get(key)
    }

    pub fn get_by_name(&self, name: &str) -> Option<Model> {
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
