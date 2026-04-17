//! [`ConfigStore`] — the storage abstraction every admin handler reads
//! and writes through.
//!
//! Production wires an etcd-backed implementation (follow-up PR); tests
//! use [`InMemoryStore`]. The trait keeps CRUD minimal — orchestration
//! (schema validation, duplicate-name detection, uuid generation) belongs
//! in the handler layer so the store stays dumb and fast.

use aisix_core::resource::ResourceEntry;
use aisix_core::{ApiKey, Model};
use dashmap::DashMap;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("store backend failure: {0}")]
    Backend(String),
}

// `async_trait` macro is what makes `dyn ConfigStore` trait objects
// compile — bare `async fn` in traits isn't dyn-compatible today.
#[async_trait::async_trait]
pub trait ConfigStore: Send + Sync + 'static {
    async fn put_model(&self, entry: ResourceEntry<Model>) -> Result<(), StoreError>;
    async fn get_model(&self, id: &str) -> Result<Option<ResourceEntry<Model>>, StoreError>;
    async fn list_models(&self) -> Result<Vec<ResourceEntry<Model>>, StoreError>;
    async fn delete_model(&self, id: &str) -> Result<bool, StoreError>;

    async fn put_apikey(&self, entry: ResourceEntry<ApiKey>) -> Result<(), StoreError>;
    async fn get_apikey(&self, id: &str) -> Result<Option<ResourceEntry<ApiKey>>, StoreError>;
    async fn list_apikeys(&self) -> Result<Vec<ResourceEntry<ApiKey>>, StoreError>;
    async fn delete_apikey(&self, id: &str) -> Result<bool, StoreError>;
}

/// In-memory store. Thread-safe via DashMap; mainly used by tests, but
/// also a viable fallback for single-process development runs.
#[derive(Debug, Default)]
pub struct InMemoryStore {
    models: DashMap<String, ResourceEntry<Model>>,
    apikeys: DashMap<String, ResourceEntry<ApiKey>>,
}

impl InMemoryStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

#[async_trait::async_trait]
impl ConfigStore for InMemoryStore {
    async fn put_model(&self, entry: ResourceEntry<Model>) -> Result<(), StoreError> {
        self.models.insert(entry.id.clone(), entry);
        Ok(())
    }

    async fn get_model(&self, id: &str) -> Result<Option<ResourceEntry<Model>>, StoreError> {
        Ok(self.models.get(id).map(|r| r.clone()))
    }

    async fn list_models(&self) -> Result<Vec<ResourceEntry<Model>>, StoreError> {
        Ok(self.models.iter().map(|r| r.clone()).collect())
    }

    async fn delete_model(&self, id: &str) -> Result<bool, StoreError> {
        Ok(self.models.remove(id).is_some())
    }

    async fn put_apikey(&self, entry: ResourceEntry<ApiKey>) -> Result<(), StoreError> {
        self.apikeys.insert(entry.id.clone(), entry);
        Ok(())
    }

    async fn get_apikey(&self, id: &str) -> Result<Option<ResourceEntry<ApiKey>>, StoreError> {
        Ok(self.apikeys.get(id).map(|r| r.clone()))
    }

    async fn list_apikeys(&self) -> Result<Vec<ResourceEntry<ApiKey>>, StoreError> {
        Ok(self.apikeys.iter().map(|r| r.clone()).collect())
    }

    async fn delete_apikey(&self, id: &str) -> Result<bool, StoreError> {
        Ok(self.apikeys.remove(id).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model(name: &str) -> Model {
        let cfg = format!(
            r#"{{
                "name": "{name}",
                "model": "openai/gpt-4o",
                "provider_config": {{"api_key": "sk-x"}}
            }}"#
        );
        serde_json::from_str(&cfg).unwrap()
    }

    #[tokio::test]
    async fn in_memory_put_get_roundtrips() {
        let store = InMemoryStore::new();
        let entry = ResourceEntry::new("m-1", sample_model("foo"), 1);
        store.put_model(entry.clone()).await.unwrap();
        let got = store.get_model("m-1").await.unwrap().unwrap();
        assert_eq!(got.id, "m-1");
        assert_eq!(got.value.name, "foo");
    }

    #[tokio::test]
    async fn in_memory_list_returns_all_entries() {
        let store = InMemoryStore::new();
        store
            .put_model(ResourceEntry::new("m-1", sample_model("foo"), 1))
            .await
            .unwrap();
        store
            .put_model(ResourceEntry::new("m-2", sample_model("bar"), 2))
            .await
            .unwrap();
        let all = store.list_models().await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn in_memory_delete_returns_false_when_absent() {
        let store = InMemoryStore::new();
        assert!(!store.delete_model("missing").await.unwrap());
        store
            .put_model(ResourceEntry::new("m-1", sample_model("foo"), 1))
            .await
            .unwrap();
        assert!(store.delete_model("m-1").await.unwrap());
        assert!(store.get_model("m-1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn models_and_apikeys_share_store_without_collision() {
        let store = InMemoryStore::new();
        store
            .put_model(ResourceEntry::new("shared-id", sample_model("m"), 1))
            .await
            .unwrap();

        let apikey: ApiKey =
            serde_json::from_str(r#"{"key":"sk-k","allowed_models":["m"]}"#).unwrap();
        store
            .put_apikey(ResourceEntry::new("shared-id", apikey, 1))
            .await
            .unwrap();

        assert!(store.get_model("shared-id").await.unwrap().is_some());
        assert!(store.get_apikey("shared-id").await.unwrap().is_some());
    }
}
