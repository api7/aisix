//! [`FileManagedStore`] — the [`ConfigStore`] the admin listener uses
//! when the gateway loads its resources from a file
//! (`resources_file` in config.yaml) instead of etcd.
//!
//! Reads are served from the live snapshot (the same one the proxy
//! reads), so `GET` lists / gets reflect the loaded file — including
//! SIGHUP reloads — without a second storage backend. The admin surface
//! is read-only by construction: the resource write endpoints were
//! removed together with the Admin API write path, so this store only
//! ever answers gets and lists. Resources change by editing the file
//! and sending SIGHUP.

use aisix_core::resource::Resource;
use aisix_core::resource::ResourceEntry;
use aisix_core::snapshot::{ResourceTable, SnapshotHandle};
use aisix_core::{
    A2aAgent, AisixSnapshot, ApiKey, CachePolicy, Guardrail, McpServer, Model,
    ObservabilityExporter, ProviderKey,
};

use crate::store::{ConfigStore, StoreError};

/// Read-only [`ConfigStore`] over the file-loaded snapshot.
pub struct FileManagedStore {
    snapshot: SnapshotHandle<AisixSnapshot>,
}

impl FileManagedStore {
    pub fn new(snapshot: SnapshotHandle<AisixSnapshot>) -> Self {
        Self { snapshot }
    }

    fn get_from<T: Resource + Clone>(
        &self,
        table: fn(&AisixSnapshot) -> &ResourceTable<T>,
        id: &str,
    ) -> Option<ResourceEntry<T>> {
        table(&self.snapshot.load())
            .get_by_id(id)
            .map(|e| (*e).clone())
    }

    fn list_from<T: Resource + Clone>(
        &self,
        table: fn(&AisixSnapshot) -> &ResourceTable<T>,
    ) -> Vec<ResourceEntry<T>> {
        table(&self.snapshot.load())
            .entries()
            .into_iter()
            .map(|e| (*e).clone())
            .collect()
    }
}

macro_rules! impl_file_managed_store {
    ($( { $ty:ty, $table:ident, $get:ident, $list:ident } )+) => {
        #[async_trait::async_trait]
        impl ConfigStore for FileManagedStore {
            $(
                async fn $get(&self, id: &str) -> Result<Option<ResourceEntry<$ty>>, StoreError> {
                    Ok(self.get_from(|s| &s.$table, id))
                }

                async fn $list(&self) -> Result<Vec<ResourceEntry<$ty>>, StoreError> {
                    Ok(self.list_from(|s| &s.$table))
                }
            )+
        }
    };
}

impl_file_managed_store! {
    { Model, models, get_model, list_models }
    { ApiKey, apikeys, get_apikey, list_apikeys }
    { ProviderKey, provider_keys, get_provider_key, list_provider_keys }
    { Guardrail, guardrails, get_guardrail, list_guardrails }
    { CachePolicy, cache_policies, get_cache_policy, list_cache_policies }
    { ObservabilityExporter, observability_exporters, get_observability_exporter, list_observability_exporters }
    { McpServer, mcp_servers, get_mcp_server, list_mcp_servers }
    { A2aAgent, a2a_agents, get_a2a_agent, list_a2a_agents }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_with_model() -> SnapshotHandle<AisixSnapshot> {
        let snap = AisixSnapshot::new();
        let model: Model = serde_json::from_str(
            r#"{
              "display_name": "file-model",
              "provider": "openai",
              "model_name": "gpt-4o",
              "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }"#,
        )
        .unwrap();
        snap.models.insert(ResourceEntry::new("m-1", model, 1));
        SnapshotHandle::new(snap)
    }

    #[tokio::test]
    async fn reads_serve_the_live_snapshot() {
        let handle = snapshot_with_model();
        let store = FileManagedStore::new(handle.clone());

        let listed = store.list_models().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].value.display_name, "file-model");

        let got = store.get_model("m-1").await.unwrap().unwrap();
        assert_eq!(got.id, "m-1");
        assert!(store.get_model("missing").await.unwrap().is_none());

        // A snapshot swap (SIGHUP reload) is immediately visible.
        handle.store(AisixSnapshot::new());
        assert!(store.list_models().await.unwrap().is_empty());
    }
}
