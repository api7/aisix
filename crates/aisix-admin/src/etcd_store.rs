//! etcd-backed [`ConfigStore`] — read-only.
//!
//! Resources reach etcd through the declarative paths (the control
//! plane or direct etcd writes); this store only reads them back for
//! the admin GET surface. Values are entity-value JSON (not the full
//! `ResourceEntry`) at `{prefix}/{kind}/{id}` — the same layout
//! `aisix-etcd::loader` parses. The ResourceEntry wrapper is
//! reconstructed on read from etcd's own `mod_revision`.
//!
//! Data layout:
//! ```text
//! /aisix/
//!   models/
//!     <uuid>  → { "name": "...", "model": "...", "provider_config": {...}, ... }
//!   api_keys/
//!     <uuid>  → { "key_hash": "...", "allowed_models": [...], ... }
//! ```
//!
//! Production wires this in `aisix-server`'s bootstrap; tests that want
//! deterministic behaviour continue to use [`crate::InMemoryStore`].

use aisix_core::resource::ResourceEntry;
use aisix_core::{
    A2aAgent, ApiKey, CachePolicy, Guardrail, McpServer, Model, ObservabilityExporter,
    PassthroughRoute, ProviderKey,
};
use aisix_etcd::kv_client;
use etcd_client::{Client, GetOptions, KvClient};
use serde::de::DeserializeOwned;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::store::{ConfigStore, StoreError};

/// Subkey segments used under the configured prefix. Mirrored in
/// `aisix-etcd`'s loader so the two paths agree at the byte level.
pub const MODELS_SUBKEY: &str = "models";
pub const APIKEYS_SUBKEY: &str = "api_keys";
pub const PROVIDER_KEYS_SUBKEY: &str = "provider_keys";
pub const GUARDRAILS_SUBKEY: &str = "guardrails";
pub const CACHE_POLICIES_SUBKEY: &str = "cache_policies";
pub const OBSERVABILITY_EXPORTERS_SUBKEY: &str = "observability_exporters";
pub const MCP_SERVERS_SUBKEY: &str = "mcp_servers";
pub const A2A_AGENTS_SUBKEY: &str = "a2a_agents";
pub const PASSTHROUGH_ROUTES_SUBKEY: &str = "passthrough_routes";

pub struct EtcdConfigStore {
    /// A KV client carrying the gateway's raised gRPC decode limit, not the
    /// `Client` handed in: `Client::get` reads through a sub-client that
    /// keeps tonic's 4 MiB default, which a full configuration set outgrows.
    client: Mutex<KvClient>,
    prefix: String,
    /// `etcd.request_timeout_ms`, applied per call. `None` — the default
    /// — leaves the reads unbounded, which is what they were before the
    /// key was wired to anything.
    request_timeout: Option<Duration>,
}

impl std::fmt::Debug for EtcdConfigStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EtcdConfigStore")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl EtcdConfigStore {
    pub fn new(
        client: Client,
        prefix: impl Into<String>,
        request_timeout: Option<Duration>,
    ) -> Self {
        let prefix = prefix.into().trim_end_matches('/').to_string();
        Self {
            client: Mutex::new(kv_client(&client)),
            prefix,
            request_timeout,
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

    /// Apply `request_timeout`, when set, to one in-flight read.
    async fn bound<T>(&self, fut: impl std::future::Future<Output = T>) -> Result<T, StoreError> {
        match self.request_timeout {
            None => Ok(fut.await),
            Some(d) => tokio::time::timeout(d, fut).await.map_err(|_| {
                StoreError::Backend(format!(
                    "etcd read exceeded etcd.request_timeout_ms ({} ms)",
                    d.as_millis()
                ))
            }),
        }
    }

    async fn get_one<T: DeserializeOwned>(
        &self,
        key: &str,
    ) -> Result<Option<(T, i64)>, StoreError> {
        let mut client = self.client.lock().await;
        let resp = self
            .bound(client.get(key.as_bytes().to_vec(), None))
            .await?
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
        let mut client = self.client.lock().await;
        let resp = self
            .bound(client.get(
                prefix.as_bytes().to_vec(),
                Some(GetOptions::new().with_prefix()),
            ))
            .await?
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
}

#[async_trait::async_trait]
impl ConfigStore for EtcdConfigStore {
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

    async fn get_provider_key(
        &self,
        id: &str,
    ) -> Result<Option<ResourceEntry<ProviderKey>>, StoreError> {
        let key = self.key_for(PROVIDER_KEYS_SUBKEY, id);
        Ok(self
            .get_one::<ProviderKey>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_provider_keys(&self) -> Result<Vec<ResourceEntry<ProviderKey>>, StoreError> {
        Ok(self
            .list_range::<ProviderKey>(PROVIDER_KEYS_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn get_guardrail(
        &self,
        id: &str,
    ) -> Result<Option<ResourceEntry<Guardrail>>, StoreError> {
        let key = self.key_for(GUARDRAILS_SUBKEY, id);
        Ok(self
            .get_one::<Guardrail>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_guardrails(&self) -> Result<Vec<ResourceEntry<Guardrail>>, StoreError> {
        Ok(self
            .list_range::<Guardrail>(GUARDRAILS_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn get_cache_policy(
        &self,
        id: &str,
    ) -> Result<Option<ResourceEntry<CachePolicy>>, StoreError> {
        let key = self.key_for(CACHE_POLICIES_SUBKEY, id);
        Ok(self
            .get_one::<CachePolicy>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_cache_policies(&self) -> Result<Vec<ResourceEntry<CachePolicy>>, StoreError> {
        Ok(self
            .list_range::<CachePolicy>(CACHE_POLICIES_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn get_observability_exporter(
        &self,
        id: &str,
    ) -> Result<Option<ResourceEntry<ObservabilityExporter>>, StoreError> {
        let key = self.key_for(OBSERVABILITY_EXPORTERS_SUBKEY, id);
        Ok(self
            .get_one::<ObservabilityExporter>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_observability_exporters(
        &self,
    ) -> Result<Vec<ResourceEntry<ObservabilityExporter>>, StoreError> {
        Ok(self
            .list_range::<ObservabilityExporter>(OBSERVABILITY_EXPORTERS_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn get_mcp_server(
        &self,
        id: &str,
    ) -> Result<Option<ResourceEntry<McpServer>>, StoreError> {
        let key = self.key_for(MCP_SERVERS_SUBKEY, id);
        Ok(self
            .get_one::<McpServer>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_mcp_servers(&self) -> Result<Vec<ResourceEntry<McpServer>>, StoreError> {
        Ok(self
            .list_range::<McpServer>(MCP_SERVERS_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn get_a2a_agent(&self, id: &str) -> Result<Option<ResourceEntry<A2aAgent>>, StoreError> {
        let key = self.key_for(A2A_AGENTS_SUBKEY, id);
        Ok(self
            .get_one::<A2aAgent>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_a2a_agents(&self) -> Result<Vec<ResourceEntry<A2aAgent>>, StoreError> {
        Ok(self
            .list_range::<A2aAgent>(A2A_AGENTS_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
    }

    async fn get_passthrough_route(
        &self,
        id: &str,
    ) -> Result<Option<ResourceEntry<PassthroughRoute>>, StoreError> {
        let key = self.key_for(PASSTHROUGH_ROUTES_SUBKEY, id);
        Ok(self
            .get_one::<PassthroughRoute>(&key)
            .await?
            .map(|(v, rev)| ResourceEntry::new(id, v, rev)))
    }

    async fn list_passthrough_routes(
        &self,
    ) -> Result<Vec<ResourceEntry<PassthroughRoute>>, StoreError> {
        Ok(self
            .list_range::<PassthroughRoute>(PASSTHROUGH_ROUTES_SUBKEY)
            .await?
            .into_iter()
            .map(|(id, v, rev)| ResourceEntry::new(id, v, rev))
            .collect())
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
        EtcdConfigStore::new(client, "/aisix", None)
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
        let store = EtcdConfigStore::new(client, "/aisix/", None);
        assert_eq!(store.prefix(), "/aisix");
        assert_eq!(store.key_for("models", "a"), "/aisix/models/a");
    }

    // Real end-to-end tests against a live etcd. Ignored by default so
    // CI without Docker still passes; run locally with:
    //   cargo test -p aisix-admin -- --ignored --test-threads=1
    #[tokio::test]
    #[ignore = "requires a running etcd container via testcontainers"]
    async fn reads_serve_directly_written_etcd_keys() {
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

        let mut client = etcd_client::Client::connect([endpoint], None)
            .await
            .expect("etcd client");
        let store = EtcdConfigStore::new(client.clone(), "/aisix-it", None);

        // Resources reach etcd by direct writes (the declarative path);
        // the store is the read side. Seed a model the way an operator
        // or the control plane would: a raw JSON value at
        // `{prefix}/{kind}/{id}`.
        let model_json = r#"{
                "display_name": "it-gpt4",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }"#;
        client
            .put("/aisix-it/models/m-it-1", model_json, None)
            .await
            .expect("direct etcd put");

        let got = store.get_model("m-it-1").await.unwrap().unwrap();
        assert_eq!(got.id, "m-it-1");
        assert_eq!(got.value.display_name, "it-gpt4");
        assert!(got.revision > 0, "etcd should return a real mod_revision");

        let listed = store.list_models().await.unwrap();
        assert_eq!(listed.len(), 1);

        client
            .delete("/aisix-it/models/m-it-1", None)
            .await
            .expect("direct etcd delete");
        assert!(store.get_model("m-it-1").await.unwrap().is_none());
        assert!(store.list_models().await.unwrap().is_empty());
    }
}
