use crate::config::{
    ConfigProvider,
    entities::types::{HasRateLimit, RateLimit, RateLimitMetric},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use utoipa::ToSchema;

use super::{EntityStore, ResourceEntry};

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ApiKey {
    pub key: String,
    pub allowed_models: Vec<String>,

    pub rate_limit: Option<RateLimit>,
}

impl HasRateLimit for ResourceEntry<ApiKey> {
    fn rate_limit(&self) -> Option<RateLimit> {
        self.rate_limit.clone()
    }

    fn rate_limit_key(&self, metric: RateLimitMetric) -> String {
        format!("apikey:{}:{}", self.key, metric.to_string())
    }
}

#[derive(Clone)]
pub struct ApiKeysStore {
    store: EntityStore<ApiKey>,
}

impl ApiKeysStore {
    pub async fn new(provider: Arc<dyn ConfigProvider + Send + Sync>) -> Self {
        Self {
            store: EntityStore::new(provider, "/apikeys/", "apikeys", None).await,
        }
    }

    pub fn list(&self) -> HashMap<String, ResourceEntry<ApiKey>> {
        self.store.list()
    }

    pub fn get(&self, username: &str) -> Option<ResourceEntry<ApiKey>> {
        self.store.get(username)
    }

    pub fn get_by_key(&self, api_key: &str) -> Option<(String, ResourceEntry<ApiKey>)> {
        //TODO pre-computed map
        for (username, apikey) in self.store.list().into_iter() {
            if apikey.key == api_key {
                return Some((username, apikey));
            }
        }
        None
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
