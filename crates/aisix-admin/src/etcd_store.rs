//! etcd-backed [`ConfigStore`].
//!
//! Writes serialise just the entity value (not the full
//! `ResourceEntry`) to `{prefix}/{kind}/{id}` so the read path in
//! `aisix-etcd::loader` — which already parses value-only JSON — works
//! unchanged. The ResourceEntry wrapper is reconstructed on read from
//! etcd's own `mod_revision`.
//!
//! Data layout:
//! ```text
//! /aisix/
//!   models/
//!     <uuid>  → { "name": "...", "model": "...", "provider_config": {...}, ... }
//!   apikeys/
//!     <uuid>  → { "key_hash": "...", "allowed_models": [...], ... }
//! ```
//!
//! Production wires this in `aisix-server`'s bootstrap; tests that want
//! deterministic behaviour continue to use [`crate::InMemoryStore`].

use aisix_core::resource::ResourceEntry;
use aisix_core::{ApiKey, Budget, Credential, Model, Team};
use etcd_client::{Client, DeleteOptions, GetOptions};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::sync::Mutex;

use crate::store::{ConfigStore, StoreError};

/// Subkey segments used under the configured prefix. Mirrored in
/// `aisix-etcd`'s loader so the two paths agree at the byte level.
pub const MODELS_SUBKEY: &str = "models";
pub const APIKEYS_SUBKEY: &str = "api_keys";
pub const CREDENTIALS_SUBKEY: &str = "credentials";
pub const BUDGETS_SUBKEY: &str = "budgets";
pub const TEAMS_SUBKEY: &str = "teams";

pub struct EtcdConfigStore {
    client: Mutex<Client>,
    prefix: String,
}

impl std::fmt::Debug for EtcdConfigStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EtcdConfigStore")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl EtcdConfigStore {
    pub fn new(client: Client, prefix: impl Into<String>) -> Self {
        let prefix = prefix.into().trim_end_matches('/').to_string();
        Self {
            client: Mutex::new(client),
            prefix,
        }
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Full key for a single entity: `{prefix}/{kind}/{id}`.
    pub(crate) fn key_for(&self, kind: &str, id: &str) -> String {
        format!("{}/{}/{}", self.prefix, kind, id)
    }

    /// Trailing-slash form used on prefix scans.
    pub(crate) fn range_prefix(&self, kind: &str) -> String {
        format!("{}/{}/", self.prefix, kind)
    }

    /// Extract the id segment given we already know which kind-prefix was used.
    pub(crate) fn id_from_key<'a>(&self, full_key: &'a str, kind: &str) -> Option<&'a str> {
        let needle = format!("{}/{}/", self.prefix, kind);
        full_key.strip_prefix(&needle)
    }

    async fn put_json<T: Serialize>(&self, key: &str, value: &T) -> Result<(), StoreError> {
        let bytes = serde_json::to_vec(value).map_err(|e| StoreError::Backend(e.to_string()))?;
        self.client
            .lock()
            .await
            .put(key.as_bytes().to_vec(), bytes, None)
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn get_one<T: DeserializeOwned>(
        &self,
        key: &str,
    ) -> Result<Option<(T, i64)>, StoreError> {
        let resp = self
            .client
            .lock()
            .await
            .get(key.as_bytes().to_vec(), None)
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let kv = match resp.kvs().first() {
            Some(kv) => kv,
            None => return Ok(None),
        };
        let value: T = serde_json::from_slice(kv.value())
            .map_err(|e| StoreError::Backend(format!("decode {key}: {e}")))?;
        Ok(Some((value, kv.mod_revision())))
    }

    async fn list_range<T: DeserializeOwned>(
        &self,
        kind: &str,
    ) -> Result<Vec<(String, T, i64)>, StoreError> {
        let prefix = self.range_prefix(kind);
        let resp = self
            .client
            .lock()
            .await
            .get(
                prefix.as_bytes().to_vec(),
                Some(GetOptions::new().with_prefix()),
            )
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;

        let mut out = Vec::with_capacity(resp.kvs().len());
        for kv in resp.kvs() {
            let key_str = String::from_utf8_lossy(kv.key()).into_owned();
            let id = match self.id_from_key(&key_str, kind) {
                Some(id) if !id.is_empty() => id.to_string(),
                _ => continue, // stray key — skip rather than abort the list
            };
            let value: T = match serde_json::from_slice(kv.value()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(key = %key_str, error = %err, "skipping malformed etcd value");
                    continue;
                }
            };
            out.push((id, value, kv.mod_revision()));
        }
        Ok(out)
    }

    async fn delete_one(&self, key: &str) -> Result<bool, StoreError> {
        let resp = self
            .client
            .lock()
            .await
            .delete(key.as_bytes().to_vec(), Some(DeleteOptions::new()))
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(resp.deleted() > 0)
    }
}

#[async_trait::async_trait]
impl ConfigStore for EtcdConfigStore {
    async fn put_model(&self, entry: ResourceEntry<Model>) -> Result<(), StoreError> {
        let key = self.key_for(MODELS_SUBKEY, &entry.id);
        self.put_json(&key, &entry.value).await
    }

    async fn get_model(&self, id: &str) -> Result<Option<ResourceEntry<Model>>, StoreError> {
        let key = self.key_for(MODELS_SUBKEY, id);
        Ok(self
            .get_one::<Model>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_models(&self) -> Result<Vec<ResourceEntry<Model>>, StoreError> {
        Ok(self
            .list_range::<Model>(MODELS_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn delete_model(&self, id: &str) -> Result<bool, StoreError> {
        self.delete_one(&self.key_for(MODELS_SUBKEY, id)).await
    }

    async fn put_apikey(&self, entry: ResourceEntry<ApiKey>) -> Result<(), StoreError> {
        let key = self.key_for(APIKEYS_SUBKEY, &entry.id);
        self.put_json(&key, &entry.value).await
    }

    async fn get_apikey(&self, id: &str) -> Result<Option<ResourceEntry<ApiKey>>, StoreError> {
        let key = self.key_for(APIKEYS_SUBKEY, id);
        Ok(self
            .get_one::<ApiKey>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_apikeys(&self) -> Result<Vec<ResourceEntry<ApiKey>>, StoreError> {
        Ok(self
            .list_range::<ApiKey>(APIKEYS_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn delete_apikey(&self, id: &str) -> Result<bool, StoreError> {
        self.delete_one(&self.key_for(APIKEYS_SUBKEY, id)).await
    }

    async fn put_credential(&self, entry: ResourceEntry<Credential>) -> Result<(), StoreError> {
        let key = self.key_for(CREDENTIALS_SUBKEY, &entry.id);
        self.put_json(&key, &entry.value).await
    }

    async fn get_credential(
        &self,
        id: &str,
    ) -> Result<Option<ResourceEntry<Credential>>, StoreError> {
        let key = self.key_for(CREDENTIALS_SUBKEY, id);
        Ok(self
            .get_one::<Credential>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_credentials(&self) -> Result<Vec<ResourceEntry<Credential>>, StoreError> {
        Ok(self
            .list_range::<Credential>(CREDENTIALS_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn delete_credential(&self, id: &str) -> Result<bool, StoreError> {
        self.delete_one(&self.key_for(CREDENTIALS_SUBKEY, id)).await
    }

    async fn put_budget(&self, entry: ResourceEntry<Budget>) -> Result<(), StoreError> {
        let key = self.key_for(BUDGETS_SUBKEY, &entry.id);
        self.put_json(&key, &entry.value).await
    }

    async fn get_budget(&self, id: &str) -> Result<Option<ResourceEntry<Budget>>, StoreError> {
        let key = self.key_for(BUDGETS_SUBKEY, id);
        Ok(self
            .get_one::<Budget>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_budgets(&self) -> Result<Vec<ResourceEntry<Budget>>, StoreError> {
        Ok(self
            .list_range::<Budget>(BUDGETS_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn delete_budget(&self, id: &str) -> Result<bool, StoreError> {
        self.delete_one(&self.key_for(BUDGETS_SUBKEY, id)).await
    }

    async fn put_team(&self, entry: ResourceEntry<Team>) -> Result<(), StoreError> {
        let key = self.key_for(TEAMS_SUBKEY, &entry.id);
        self.put_json(&key, &entry.value).await
    }

    async fn get_team(&self, id: &str) -> Result<Option<ResourceEntry<Team>>, StoreError> {
        let key = self.key_for(TEAMS_SUBKEY, id);
        Ok(self
            .get_one::<Team>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_teams(&self) -> Result<Vec<ResourceEntry<Team>>, StoreError> {
        Ok(self
            .list_range::<Team>(TEAMS_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn delete_team(&self, id: &str) -> Result<bool, StoreError> {
        self.delete_one(&self.key_for(TEAMS_SUBKEY, id)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a store *without* a real client so pure helper tests don't
    // pay a Docker tax. The client is never used by these tests.
    fn dummy_store() -> EtcdConfigStore {
        // We can't construct `etcd_client::Client` without connecting, so
        // build a "real" one pointing at a bogus endpoint — the connect
        // is lazy and these tests never issue a request.
        let client_fut = etcd_client::Client::connect(["http://127.0.0.1:59999"], None);
        let client = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(client_fut)
            .expect("lazy connect never fails synchronously");
        EtcdConfigStore::new(client, "/aisix")
    }

    #[test]
    fn key_for_matches_spec_layout() {
        let store = dummy_store();
        assert_eq!(store.key_for("models", "abc-1"), "/aisix/models/abc-1");
        assert_eq!(store.key_for("api_keys", "xyz"), "/aisix/api_keys/xyz");
    }

    #[test]
    fn range_prefix_includes_trailing_slash() {
        let store = dummy_store();
        assert_eq!(store.range_prefix("models"), "/aisix/models/");
    }

    #[test]
    fn id_from_key_extracts_id_segment() {
        let store = dummy_store();
        assert_eq!(
            store.id_from_key("/aisix/models/abc-1", "models"),
            Some("abc-1"),
        );
        // Wrong kind prefix → None.
        assert!(store.id_from_key("/aisix/api_keys/x", "models").is_none());
        // Outside the configured prefix → None.
        assert!(store.id_from_key("/other/models/x", "models").is_none());
    }

    #[test]
    fn prefix_trailing_slash_is_trimmed_at_construction() {
        let client = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(etcd_client::Client::connect(
                ["http://127.0.0.1:59999"],
                None,
            ))
            .expect("lazy connect never fails synchronously");
        let store = EtcdConfigStore::new(client, "/aisix/");
        assert_eq!(store.prefix(), "/aisix");
        assert_eq!(store.key_for("models", "a"), "/aisix/models/a");
    }

    // Real end-to-end tests against a live etcd. Ignored by default so
    // CI without Docker still passes; run locally with:
    //   cargo test -p aisix-admin -- --ignored --test-threads=1
    #[tokio::test]
    #[ignore = "requires a running etcd container via testcontainers"]
    async fn put_get_list_delete_roundtrip_against_real_etcd() {
        use testcontainers::runners::AsyncRunner;
        use testcontainers::{GenericImage, ImageExt};

        let container = GenericImage::new("bitnami/etcd", "3.5")
            .with_env_var("ALLOW_NONE_AUTHENTICATION", "yes")
            .with_env_var("ETCD_LISTEN_CLIENT_URLS", "http://0.0.0.0:2379")
            .with_env_var("ETCD_ADVERTISE_CLIENT_URLS", "http://0.0.0.0:2379")
            .start()
            .await
            .expect("etcd container");
        let port = container
            .get_host_port_ipv4(2379)
            .await
            .expect("container port");
        let endpoint = format!("http://127.0.0.1:{port}");

        let client = etcd_client::Client::connect([endpoint], None)
            .await
            .expect("etcd client");
        let store = EtcdConfigStore::new(client, "/aisix-it");

        let model: Model = serde_json::from_str(
            r#"{
                "name": "it-gpt4",
                "model": "openai/gpt-4o",
                "provider_config": {"api_key": "sk-x"}
            }"#,
        )
        .unwrap();
        let entry = ResourceEntry::new("m-it-1", model, 0);
        store.put_model(entry.clone()).await.unwrap();

        let got = store.get_model("m-it-1").await.unwrap().unwrap();
        assert_eq!(got.id, "m-it-1");
        assert_eq!(got.value.name, "it-gpt4");
        assert!(got.revision > 0, "etcd should return a real mod_revision");

        let listed = store.list_models().await.unwrap();
        assert_eq!(listed.len(), 1);

        assert!(store.delete_model("m-it-1").await.unwrap());
        assert!(store.get_model("m-it-1").await.unwrap().is_none());
        assert!(!store.delete_model("m-it-1").await.unwrap());
    }
}
