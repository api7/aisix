use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use serde::{Deserialize, Serialize};
use serde_json::json;
use utoipa::ToSchema;

use super::{EntityStore, IndexFn, ResourceEntry};
use crate::{
    config::{
        ConfigProvider,
        entities::types::{HasRateLimit, RateLimit, RateLimitMetric},
    },
    utils::jsonschema::format_evaluation_error,
};

static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema#",
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "allowed_models": {
                "type": "array",
                "items": { "type": "string" }
            },
            "rate_limit": {"type": "object"}
        },
        "required": ["key", "allowed_models"],
        "additionalProperties": false
    })
});
pub static SCHEMA_VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| jsonschema::validator_for(&SCHEMA).expect("Invalid JSON schema for API Key"));

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ApiKey {
    pub key: String,
    pub allowed_models: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimit>,
}

impl HasRateLimit for ResourceEntry<ApiKey> {
    fn rate_limit(&self) -> Option<RateLimit> {
        self.rate_limit.clone()
    }

    fn rate_limit_key(&self, metric: RateLimitMetric) -> String {
        format!("apikey:{}:{}", self.key, metric)
    }
}

fn validate(key: &str, value: &ApiKey) -> Result<(), String> {
    let evaluation = SCHEMA_VALIDATOR.evaluate(
        &serde_json::to_value(value)
            .map_err(|e| format!("Failed to serialize API key for validation: {}", e))?,
    );
    if !evaluation.flag().valid {
        return Err(format!(
            r#"JSON schema validation error on apikey "{key}": {}"#,
            format_evaluation_error(&evaluation)
        ));
    }

    Ok(())
}

#[derive(Clone)]
pub struct ApiKeysStore {
    store: EntityStore<ApiKey>,
}

static INDEX_FNS: &[IndexFn<ApiKey>] = &[("by_key", |k: &ApiKey| Some(k.key.clone()))];

impl ApiKeysStore {
    pub async fn new(provider: Arc<dyn ConfigProvider + Send + Sync>) -> Self {
        Self {
            store: EntityStore::new(provider, "/apikeys/", "apikeys", Some(validate), INDEX_FNS)
                .await,
        }
    }

    pub fn list(&self) -> Arc<HashMap<String, ResourceEntry<ApiKey>>> {
        self.store.list()
    }

    pub fn get(&self, username: &str) -> Option<ResourceEntry<ApiKey>> {
        self.store.get(username)
    }

    pub fn get_by_key(&self, api_key: &str) -> Option<ResourceEntry<ApiKey>> {
        self.store.get_by_secondary("by_key", api_key)
    }

    #[fastrace::trace]
    pub fn is_model_allowed(&self, username: &str, model: &str) -> bool {
        if let Some(consumer) = self.get(username) {
            consumer.allowed_models.contains(&model.to_string())
        } else {
            false
        }
    }

    pub fn latest_mod_revision(&self) -> i64 {
        self.store.latest_mod_revision()
    }
}
