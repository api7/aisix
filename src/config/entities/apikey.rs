use crate::config::ConfigProvider;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use super::EntityStore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub allowed_models: Vec<String>,
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

    pub fn list(&self) -> HashMap<String, ApiKey> {
        self.store.list()
    }

    pub fn get(&self, username: &str) -> Option<ApiKey> {
        self.store.get(username)
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
