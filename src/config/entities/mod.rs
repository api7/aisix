use std::{collections::HashMap, ops::Deref, sync::Arc};

use arc_swap::ArcSwap;
use log::{info, warn};
use serde::de::DeserializeOwned;
use tokio::sync::mpsc::Receiver;

use crate::config::{ConfigEvent, ConfigProvider};

mod apikey;
pub mod models;
pub mod types;

pub use apikey::ApiKey;
pub use models::Model;

#[derive(Clone)]
pub struct ResourceRegistry {
    pub models: models::ModelsStore,
    pub apikeys: apikey::ApiKeysStore,
}

impl ResourceRegistry {
    pub async fn new(provider: Arc<dyn ConfigProvider + Send + Sync>) -> Self {
        let models = models::ModelsStore::new(provider.clone()).await;
        let apikeys = apikey::ApiKeysStore::new(provider).await;

        Self { models, apikeys }
    }
}

#[derive(Clone, Debug)]
pub struct ResourceEntry<T> {
    value: T,
    revision: i64,
}

impl<T> ResourceEntry<T> {
    pub fn revision(&self) -> i64 {
        self.revision
    }
}

impl<T> Deref for ResourceEntry<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[derive(Clone)]
pub struct ResourceStore<T> {
    data: Arc<ArcSwap<HashMap<String, ResourceEntry<T>>>>,
    latest_mod_revision: Arc<ArcSwap<i64>>,
}

impl<T: Clone> ResourceStore<T> {
    pub fn new() -> Self {
        Self {
            data: Arc::new(ArcSwap::from_pointee(HashMap::new())),
            latest_mod_revision: Arc::new(ArcSwap::from_pointee(0i64)),
        }
    }

    pub fn upsert(&self, key: String, value: T, revision: i64) {
        // Use load-modify-store pattern
        let current: arc_swap::Guard<Arc<HashMap<String, ResourceEntry<T>>>> = self.data.load();
        let mut new_map = (**current).clone();
        new_map.insert(key, ResourceEntry { value, revision });
        self.data.store(Arc::new(new_map));

        // Update latest mod_revision
        if revision > self.latest_mod_revision() {
            self.latest_mod_revision.store(Arc::new(revision));
        }
    }

    pub fn delete(&self, key: &str, mod_revision: i64) -> bool {
        let current = self.data.load();
        let mut new_map = (**current).clone();
        let deleted = new_map.remove(key).is_some();
        self.data.store(Arc::new(new_map));

        // Update latest mod_revision
        if mod_revision > self.latest_mod_revision() {
            self.latest_mod_revision.store(Arc::new(mod_revision));
        }

        deleted
    }

    pub fn get(&self, key: &str) -> Option<ResourceEntry<T>> {
        let current: arc_swap::Guard<Arc<HashMap<String, ResourceEntry<T>>>> = self.data.load();
        current.get(key).map(|entry| entry.clone())
    }

    pub fn snapshot(&self) -> HashMap<String, ResourceEntry<T>> {
        let current = self.data.load();
        current
            .iter()
            .map(|(k, entry)| (k.clone(), entry.clone()))
            .collect()
    }

    pub fn latest_mod_revision(&self) -> i64 {
        let current = self.latest_mod_revision.load();
        **current
    }
}

/// Generic Entity Store that automatically subscribes to config prefixes and handles JSON deserialization
#[derive(Clone)]
pub struct EntityStore<T> {
    store: ResourceStore<T>,
}

impl<T: DeserializeOwned + Clone + Send + Sync + 'static> EntityStore<T> {
    /// Create and start an entity store
    ///
    /// # Parameters
    /// - `provider`: ConfigProvider instance
    /// - `prefix`: Listening path prefix (e.g., "/models/")
    /// - `entity_name`: Entity name for logging
    /// - `validator`: Optional validation function called when loading or updating entities, skips entity if returns Err
    pub async fn new(
        provider: Arc<dyn ConfigProvider + Send + Sync>,
        prefix: &str,
        entity_name: &str,
        validator: Option<Arc<dyn Fn(&str, &T) -> Result<(), String> + Send + Sync>>,
    ) -> Self {
        let store = ResourceStore::new();

        // Full load of existing data at startup
        info!("{} Starting full load, prefix={}", entity_name, prefix);
        match provider.get_all(Some(prefix)).await {
            Ok(kvs) => {
                for (key, value, mod_revision) in kvs {
                    // Extract relative path
                    let base_prefix = if let Some(idx) = key.find(prefix) {
                        &key[..idx + prefix.len()]
                    } else {
                        key.as_str()
                    };

                    let relative_key = key
                        .strip_prefix(base_prefix)
                        .unwrap_or(&key)
                        .trim_start_matches('/')
                        .to_string();

                    // Parse and store
                    match serde_json::from_slice::<T>(&value) {
                        Ok(parsed) => {
                            // Apply validator check
                            if let Some(ref v) = validator {
                                match v(&relative_key, &parsed) {
                                    Ok(_) => {
                                        store.upsert(relative_key.clone(), parsed, mod_revision);
                                    }
                                    Err(err) => {
                                        warn!(
                                            "{} validation failed, key={}: {}",
                                            entity_name, relative_key, err
                                        );
                                    }
                                }
                            } else {
                                store.upsert(relative_key.clone(), parsed, mod_revision);
                            }
                        }
                        Err(err) => {
                            warn!(
                                "{} full load parsing failed, key={}: {}",
                                entity_name, relative_key, err
                            );
                        }
                    }
                }
                info!("{} full load completed", entity_name);
            }
            Err(err) => {
                warn!(
                    "{} full load failed: {}, will only rely on watch events",
                    entity_name, err
                );
            }
        }

        // Subscribe to incremental updates
        match provider.watch(Some(prefix)).await {
            Ok(mut rx) => {
                let store_clone = store.clone();
                let entity_name = entity_name.to_string();
                let prefix = prefix.to_string();
                let validator_clone = validator.clone();

                tokio::spawn(async move {
                    Self::consume_events(
                        store_clone,
                        &mut rx,
                        &entity_name,
                        &prefix,
                        validator_clone,
                    )
                    .await;
                });
            }
            Err(_) => {
                warn!(
                    "Duplicate registration of {} prefix watch ignored: {}",
                    entity_name, prefix
                );
            }
        }

        Self { store }
    }

    /// Get the value of the specified key
    pub fn get(&self, key: &str) -> Option<ResourceEntry<T>> {
        self.store.get(key)
    }

    /// Get snapshot of all entities
    pub fn list(&self) -> HashMap<String, ResourceEntry<T>> {
        self.store.snapshot()
    }

    /// Get the latest mod_revision of this resource type
    pub fn latest_mod_revision(&self) -> i64 {
        self.store.latest_mod_revision()
    }

    async fn consume_events(
        store: ResourceStore<T>,
        rx: &mut Receiver<ConfigEvent>,
        entity_name: &str,
        prefix: &str,
        validator: Option<Arc<dyn Fn(&str, &T) -> Result<(), String> + Send + Sync>>,
    ) {
        info!("{} Watch started, prefix={}", entity_name, prefix);

        let normalize_key = |key: String| {
            let base_prefix = if let Some(idx) = key.find(prefix) {
                &key[..idx + prefix.len()]
            } else {
                key.as_str()
            };

            key.strip_prefix(base_prefix)
                .unwrap_or(&key)
                .trim_start_matches('/')
                .to_string()
        };

        while let Some(event) = rx.recv().await {
            match event {
                ConfigEvent::Put((key, value, mod_revision)) => {
                    let relative_key = normalize_key(key.clone());

                    match serde_json::from_slice::<T>(&value) {
                        Ok(parsed) => {
                            if let Some(ref v) = validator {
                                match v(&relative_key, &parsed) {
                                    Ok(_) => {
                                        store.upsert(relative_key.clone(), parsed, mod_revision);
                                    }
                                    Err(err) => {
                                        warn!(
                                            "{} validation failed, key={}: {}",
                                            entity_name, relative_key, err
                                        );
                                    }
                                }
                            } else {
                                store.upsert(relative_key.clone(), parsed, mod_revision);
                            }
                        }
                        Err(err) => {
                            warn!(
                                "{} JSON parsing failed, key={}: {}",
                                entity_name, relative_key, err
                            );
                        }
                    }
                }
                ConfigEvent::Delete((key, mod_revision)) => {
                    let relative_key = normalize_key(key.clone());

                    if !store.delete(relative_key.as_str(), mod_revision) {
                        info!(
                            "{} Delete event missed cache, key={}",
                            entity_name, relative_key
                        );
                    }
                }
            }
        }

        warn!("{} Watch ended, waiting to be restarted", entity_name);
    }
}
