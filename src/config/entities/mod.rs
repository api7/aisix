pub mod apikeys;
pub mod models;
pub mod types;

use std::{collections::HashMap, ops::Deref, sync::Arc};

pub use apikeys::ApiKey;
use arc_swap::ArcSwap;
use log::{info, warn};
pub use models::Model;
use serde::de::DeserializeOwned;
use tokio::sync::mpsc::Receiver;

use crate::config::{ConfigEvent, ConfigProvider, GetEntry};

#[derive(Clone)]
pub struct ResourceRegistry {
    pub models: models::ModelsStore,
    pub apikeys: apikeys::ApiKeysStore,
}

impl ResourceRegistry {
    pub async fn new(provider: Arc<dyn ConfigProvider + Send + Sync>) -> Self {
        let models = models::ModelsStore::new(provider.clone()).await;
        let apikeys = apikeys::ApiKeysStore::new(provider).await;

        Self { models, apikeys }
    }
}

#[derive(Clone, Debug)]
pub struct ResourceEntry<T> {
    pub id: String,
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

/// Named secondary index function: (index_name, fn(&T) -> Option<secondary_key>).
/// Returns `None` to skip indexing a particular entry under that index.
pub type IndexFn<T> = (&'static str, fn(&T) -> Option<String>);

/// Atomic snapshot of the store: primary map, all secondary indexes, and the
/// latest etcd mod_revision — replaced as a single unit on every write.
struct StoreData<T> {
    /// Primary key → entry
    primary: Arc<HashMap<String, ResourceEntry<T>>>,
    /// index_name → (secondary_key → primary_key)
    secondary: HashMap<&'static str, HashMap<String, String>>,
    mod_revision: i64,
}

impl<T: Clone> StoreData<T> {
    fn empty(index_fns: &'static [IndexFn<T>]) -> Self {
        let secondary = index_fns
            .iter()
            .map(|(name, _)| (*name, HashMap::new()))
            .collect();
        Self {
            primary: Arc::new(HashMap::new()),
            secondary,
            mod_revision: 0,
        }
    }
}

#[derive(Clone)]
struct ResourceStore<T: 'static> {
    data: Arc<ArcSwap<StoreData<T>>>,
    index_fns: &'static [IndexFn<T>],
}

impl<T: Clone + 'static> ResourceStore<T> {
    fn new(index_fns: &'static [IndexFn<T>]) -> Self {
        Self {
            data: Arc::new(ArcSwap::from_pointee(StoreData::empty(index_fns))),
            index_fns,
        }
    }

    fn upsert(&self, key: String, value: T, revision: i64) {
        let current = self.data.load();

        let mut new_primary = (*current.primary).clone();
        let mut new_secondary = current.secondary.clone();

        // Update secondary indexes: remove stale key, insert new key
        for (name, key_fn) in self.index_fns {
            let index = new_secondary.entry(name).or_default();
            if let Some(old_entry) = new_primary.get(&key)
                && let Some(old_sec_key) = key_fn(old_entry)
            {
                index.remove(&old_sec_key);
            }
            if let Some(new_sec_key) = key_fn(&value) {
                index.insert(new_sec_key, key.clone());
            }
        }

        new_primary.insert(
            key.clone(),
            ResourceEntry {
                id: key,
                value,
                revision,
            },
        );

        self.data.store(Arc::new(StoreData {
            primary: Arc::new(new_primary),
            secondary: new_secondary,
            mod_revision: revision.max(current.mod_revision),
        }));
    }

    fn delete(&self, key: &str, mod_revision: i64) -> bool {
        let current = self.data.load();

        let mut new_primary = (*current.primary).clone();
        let mut new_secondary = current.secondary.clone();

        // Remove from secondary indexes before removing from primary
        if let Some(old_entry) = new_primary.get(key) {
            for (name, key_fn) in self.index_fns {
                if let Some(old_sec_key) = key_fn(old_entry)
                    && let Some(index) = new_secondary.get_mut(name)
                {
                    index.remove(&old_sec_key);
                }
            }
        }

        let deleted = new_primary.remove(key).is_some();

        self.data.store(Arc::new(StoreData {
            primary: Arc::new(new_primary),
            secondary: new_secondary,
            mod_revision: mod_revision.max(current.mod_revision),
        }));

        deleted
    }

    fn get(&self, key: &str) -> Option<ResourceEntry<T>> {
        self.data.load().primary.get(key).cloned()
    }

    /// Get an entry via a secondary index
    fn get_by_secondary(&self, index: &str, sec_key: &str) -> Option<ResourceEntry<T>> {
        let snapshot = self.data.load();
        let primary_key = snapshot.secondary.get(index)?.get(sec_key)?;
        snapshot.primary.get(primary_key).cloned()
    }

    /// Returns an Arc to the primary map
    fn primary_snapshot(&self) -> Arc<HashMap<String, ResourceEntry<T>>> {
        Arc::clone(&self.data.load().primary)
    }

    fn latest_mod_revision(&self) -> i64 {
        self.data.load().mod_revision
    }
}

pub type EntityValidator<T> = fn(&str, &T) -> Result<(), String>;

/// Generic Entity Store that automatically subscribes to config prefixes and handles JSON deserialization
#[derive(Clone)]
pub struct EntityStore<T: 'static> {
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
    /// - `index_fns`: Named secondary index functions for O(1) secondary lookups
    pub async fn new(
        provider: Arc<dyn ConfigProvider>,
        prefix: &str,
        entity_name: &str,
        validator: Option<EntityValidator<T>>,
        index_fns: &'static [IndexFn<T>],
    ) -> Self {
        let store = ResourceStore::new(index_fns);

        // Full load of existing data at startup
        info!("{} starting full load, prefix={}", entity_name, prefix);
        match provider.get_all::<T>(prefix).await {
            Ok(kvs) => {
                for GetEntry {
                    key,
                    value,
                    create_revision: _,
                    mod_revision,
                } in kvs
                {
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

                    // Apply validator check and store (value is already deserialized by get_all)
                    if let Some(ref v) = validator {
                        match v(&relative_key, &value) {
                            Ok(_) => {
                                store.upsert(relative_key.clone(), value, mod_revision);
                            }
                            Err(err) => {
                                warn!(
                                    "{} validation failed, key={}: {}",
                                    entity_name, relative_key, err
                                );
                            }
                        }
                    } else {
                        store.upsert(relative_key.clone(), value, mod_revision);
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

                tokio::spawn(async move {
                    Self::consume_events(store_clone, &mut rx, &entity_name, &prefix, validator)
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

    pub fn list(&self) -> Arc<HashMap<String, ResourceEntry<T>>> {
        self.store.primary_snapshot()
    }

    /// Get an entry via a secondary index
    fn get_by_secondary(&self, index: &str, sec_key: &str) -> Option<ResourceEntry<T>> {
        self.store.get_by_secondary(index, sec_key)
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
        validator: Option<EntityValidator<T>>,
    ) {
        info!("{} watch started, prefix={}", entity_name, prefix);

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

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, time::Duration};

    use anyhow::Result;
    use async_trait::async_trait;
    use tokio::sync::mpsc;

    use super::*;
    use crate::config::{ConfigEvent, ConfigEventReceiver, GetEntry, PutEntry};

    // ── Shared test fixture ───────────────────────────────────────────────────

    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Item {
        name: String,
        tag: String,
    }

    fn item(name: &str, tag: &str) -> Item {
        Item {
            name: name.into(),
            tag: tag.into(),
        }
    }

    static ITEM_INDEXES: &[IndexFn<Item>] = &[
        ("by_name", |i: &Item| Some(i.name.clone())),
        ("by_tag", |i: &Item| Some(i.tag.clone())),
    ];

    // ── ResourceStore unit tests ──────────────────────────────────────────────

    #[test]
    fn resource_store_upsert_and_get() {
        let store: ResourceStore<Item> = ResourceStore::new(&[]);
        store.upsert("k1".into(), item("alice", "admin"), 1);

        let entry = store.get("k1").unwrap();
        assert_eq!(entry.id, "k1");
        assert_eq!(entry.name, "alice");
    }

    #[test]
    fn resource_store_get_missing_returns_none() {
        let store: ResourceStore<Item> = ResourceStore::new(&[]);
        assert!(store.get("nope").is_none());
    }

    #[test]
    fn resource_store_delete() {
        let store: ResourceStore<Item> = ResourceStore::new(&[]);
        store.upsert("k1".into(), item("alice", "admin"), 1);

        assert!(store.delete("k1", 2));
        assert!(store.get("k1").is_none());
    }

    #[test]
    fn resource_store_delete_missing_returns_false() {
        let store: ResourceStore<Item> = ResourceStore::new(&[]);
        assert!(!store.delete("ghost", 1));
    }

    #[test]
    fn resource_store_secondary_lookup() {
        let store: ResourceStore<Item> = ResourceStore::new(ITEM_INDEXES);
        store.upsert("k1".into(), item("alice", "admin"), 1);

        assert!(store.get_by_secondary("by_name", "alice").is_some());
        assert!(store.get_by_secondary("by_tag", "admin").is_some());
        assert!(store.get_by_secondary("by_name", "bob").is_none());
    }

    #[test]
    fn resource_store_secondary_stale_key_removed_on_update() {
        let store: ResourceStore<Item> = ResourceStore::new(ITEM_INDEXES);
        store.upsert("k1".into(), item("alice", "admin"), 1);
        // Rename: old secondary key "alice" should be gone, "bob" should map to k1
        store.upsert("k1".into(), item("bob", "admin"), 2);

        assert!(store.get_by_secondary("by_name", "alice").is_none());
        assert_eq!(store.get_by_secondary("by_name", "bob").unwrap().id, "k1");
    }

    #[test]
    fn resource_store_multiple_indexes_independent() {
        let store: ResourceStore<Item> = ResourceStore::new(ITEM_INDEXES);
        store.upsert("k1".into(), item("alice", "alpha"), 1);
        store.upsert("k2".into(), item("bob", "beta"), 2);

        assert_eq!(store.get_by_secondary("by_name", "alice").unwrap().id, "k1");
        assert_eq!(store.get_by_secondary("by_name", "bob").unwrap().id, "k2");
        assert_eq!(store.get_by_secondary("by_tag", "alpha").unwrap().id, "k1");
        assert_eq!(store.get_by_secondary("by_tag", "beta").unwrap().id, "k2");
    }

    #[test]
    fn resource_store_delete_clears_secondary_entries() {
        let store: ResourceStore<Item> = ResourceStore::new(ITEM_INDEXES);
        store.upsert("k1".into(), item("alice", "admin"), 1);
        store.delete("k1", 2);

        assert!(store.get_by_secondary("by_name", "alice").is_none());
        assert!(store.get_by_secondary("by_tag", "admin").is_none());
    }

    #[test]
    fn resource_store_unknown_index_returns_none() {
        let store: ResourceStore<Item> = ResourceStore::new(ITEM_INDEXES);
        store.upsert("k1".into(), item("alice", "admin"), 1);
        assert!(store.get_by_secondary("by_nonexistent", "alice").is_none());
    }

    #[test]
    fn resource_store_mod_revision_tracks_max() {
        let store: ResourceStore<Item> = ResourceStore::new(&[]);
        assert_eq!(store.latest_mod_revision(), 0);

        store.upsert("k1".into(), item("a", "x"), 5);
        assert_eq!(store.latest_mod_revision(), 5);

        // Lower revision must not overwrite the tracked max
        store.upsert("k2".into(), item("b", "y"), 3);
        assert_eq!(store.latest_mod_revision(), 5);

        store.delete("k1", 9);
        assert_eq!(store.latest_mod_revision(), 9);
    }

    #[test]
    fn resource_store_primary_snapshot_is_zero_copy() {
        let store: ResourceStore<Item> = ResourceStore::new(&[]);
        store.upsert("k1".into(), item("a", "x"), 1);

        let snap1 = store.primary_snapshot();
        let snap2 = store.primary_snapshot();
        // Both Arcs point to the same allocation — no data was cloned
        assert!(Arc::ptr_eq(&snap1, &snap2));
    }

    #[test]
    fn resource_store_no_indexes_has_empty_secondary() {
        let store: ResourceStore<Item> = ResourceStore::new(&[]);
        store.upsert("k1".into(), item("alice", "admin"), 1);

        // No indexes registered → any secondary lookup returns None
        assert!(store.get_by_secondary("by_name", "alice").is_none());
    }

    // ── MockConfigProvider for EntityStore tests ──────────────────────────────

    struct MockProvider {
        /// Pre-loaded (etcd_key, raw_json_bytes) pairs
        data: Vec<(String, Vec<u8>)>,
        watch_rx: Mutex<Option<ConfigEventReceiver>>,
    }

    impl MockProvider {
        fn new(data: Vec<(&str, Vec<u8>)>, rx: ConfigEventReceiver) -> Arc<Self> {
            Arc::new(Self {
                data: data.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
                watch_rx: Mutex::new(Some(rx)),
            })
        }
    }

    #[async_trait]
    impl crate::config::ConfigProvider for MockProvider {
        async fn get_all_raw(
            &self,
            _prefix: Option<&str>,
        ) -> Result<Vec<GetEntry<Vec<u8>>>, String> {
            Ok(self
                .data
                .iter()
                .enumerate()
                .map(|(i, (k, v))| GetEntry {
                    key: k.clone(),
                    value: v.clone(),
                    create_revision: i as i64 + 1,
                    mod_revision: i as i64 + 1,
                })
                .collect())
        }

        async fn get_raw(&self, _key: &str) -> Result<Option<GetEntry<Vec<u8>>>, String> {
            Ok(None)
        }

        async fn put_raw(&self, _key: &str, _value: Vec<u8>) -> Result<PutEntry<Vec<u8>>, String> {
            Ok(PutEntry::Created)
        }

        async fn delete(&self, _key: &str) -> Result<i64, String> {
            Ok(0)
        }

        async fn watch(&self, _prefix: Option<&str>) -> Result<ConfigEventReceiver> {
            Ok(self
                .watch_rx
                .lock()
                .unwrap()
                .take()
                .expect("MockProvider::watch called more than once"))
        }
    }

    fn raw(v: &impl serde::Serialize) -> Vec<u8> {
        serde_json::to_vec(v).unwrap()
    }

    fn put_event(key: &str, v: &impl serde::Serialize, rev: i64) -> ConfigEvent {
        ConfigEvent::Put((key.into(), serde_json::to_vec(v).unwrap(), rev))
    }

    fn delete_event(key: &str, rev: i64) -> ConfigEvent {
        ConfigEvent::Delete((key.into(), rev))
    }

    // ── EntityStore integration tests ─────────────────────────────────────────

    #[tokio::test]
    async fn entity_store_full_load() {
        let (tx, rx) = mpsc::channel(8);
        let provider = MockProvider::new(
            vec![
                ("/items/a", raw(&item("alice", "admin"))),
                ("/items/b", raw(&item("bob", "user"))),
            ],
            rx,
        );

        let store = EntityStore::new(provider, "/items/", "test", None, ITEM_INDEXES).await;

        assert!(store.get("a").is_some());
        assert!(store.get("b").is_some());
        assert!(store.get("c").is_none());
        assert_eq!(store.list().len(), 2);

        drop(tx);
    }

    #[tokio::test]
    async fn entity_store_watch_put_adds_entry() {
        let (tx, rx) = mpsc::channel(8);
        let provider = MockProvider::new(vec![], rx);

        let store = EntityStore::new(provider, "/items/", "test", None, ITEM_INDEXES).await;

        tx.send(put_event("/items/c", &item("carol", "mod"), 10))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        let entry = store
            .get("c")
            .expect("entry c should exist after Put event");
        assert_eq!(entry.name, "carol");
        assert_eq!(store.latest_mod_revision(), 10);
    }

    #[tokio::test]
    async fn entity_store_watch_delete_removes_entry() {
        let (tx, rx) = mpsc::channel(8);
        let provider = MockProvider::new(vec![("/items/a", raw(&item("alice", "admin")))], rx);

        let store = EntityStore::new(provider, "/items/", "test", None, ITEM_INDEXES).await;
        assert!(store.get("a").is_some());

        tx.send(delete_event("/items/a", 5)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(store.get("a").is_none());
        assert_eq!(store.latest_mod_revision(), 5);
    }

    #[tokio::test]
    async fn entity_store_validator_skips_invalid_on_full_load() {
        let (tx, rx) = mpsc::channel(8);
        let provider = MockProvider::new(
            vec![
                ("/items/good", raw(&item("alice", "admin"))),
                ("/items/bad", raw(&item("", "admin"))), // empty name is invalid
            ],
            rx,
        );

        fn reject_empty_name(_key: &str, v: &Item) -> Result<(), String> {
            if v.name.is_empty() {
                Err("name must not be empty".into())
            } else {
                Ok(())
            }
        }

        let store = EntityStore::new(
            provider,
            "/items/",
            "test",
            Some(reject_empty_name),
            ITEM_INDEXES,
        )
        .await;

        assert!(store.get("good").is_some());
        assert!(store.get("bad").is_none());

        drop(tx);
    }

    #[tokio::test]
    async fn entity_store_validator_skips_invalid_on_watch_put() {
        let (tx, rx) = mpsc::channel(8);
        let provider = MockProvider::new(vec![], rx);

        fn reject_empty_name(_key: &str, v: &Item) -> Result<(), String> {
            if v.name.is_empty() {
                Err("name must not be empty".into())
            } else {
                Ok(())
            }
        }

        let store = EntityStore::new(
            provider,
            "/items/",
            "test",
            Some(reject_empty_name),
            ITEM_INDEXES,
        )
        .await;

        tx.send(put_event("/items/bad", &item("", "x"), 1))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(store.get("bad").is_none());
    }

    #[tokio::test]
    async fn entity_store_secondary_index_available_after_full_load() {
        let (tx, rx) = mpsc::channel(8);
        let provider = MockProvider::new(vec![("/items/a", raw(&item("alice", "admin")))], rx);

        let store = EntityStore::new(provider, "/items/", "test", None, ITEM_INDEXES).await;

        assert!(store.get_by_secondary("by_name", "alice").is_some());
        assert!(store.get_by_secondary("by_tag", "admin").is_some());

        drop(tx);
    }

    #[tokio::test]
    async fn entity_store_secondary_index_updated_via_watch() {
        let (tx, rx) = mpsc::channel(8);
        let provider = MockProvider::new(vec![], rx);

        let store = EntityStore::new(provider, "/items/", "test", None, ITEM_INDEXES).await;

        tx.send(put_event("/items/a", &item("alice", "admin"), 1))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(store.get_by_secondary("by_name", "alice").is_some());

        // Update: rename alice → bob (stale secondary key should be removed)
        tx.send(put_event("/items/a", &item("bob", "admin"), 2))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(store.get_by_secondary("by_name", "alice").is_none());
        assert!(store.get_by_secondary("by_name", "bob").is_some());
    }

    #[tokio::test]
    async fn entity_store_secondary_index_cleared_on_delete() {
        let (tx, rx) = mpsc::channel(8);
        let provider = MockProvider::new(vec![("/items/a", raw(&item("alice", "admin")))], rx);

        let store = EntityStore::new(provider, "/items/", "test", None, ITEM_INDEXES).await;

        assert!(store.get_by_secondary("by_name", "alice").is_some());

        tx.send(delete_event("/items/a", 3)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(store.get_by_secondary("by_name", "alice").is_none());
    }
}
