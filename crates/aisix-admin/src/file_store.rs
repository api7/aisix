//! [`FileManagedStore`] — the [`ConfigStore`] the admin listener uses
//! when the gateway loads its resources from a file
//! (`resources_file` in config.yaml) instead of etcd.
//!
//! Reads are served from the live snapshot (the same one the proxy
//! reads), so `GET` lists / gets reflect the loaded file — including
//! SIGHUP reloads — without a second storage backend. Every write
//! returns [`StoreError::ReadOnly`], which the HTTP layer maps to a
//! 409 telling the operator to edit the file and reload. A router-level
//! guard in `build_router` rejects write requests before handler logic
//! runs; these store errors are the defense-in-depth backstop for any
//! non-HTTP caller.

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
    resources_path: String,
}

impl FileManagedStore {
    pub fn new(snapshot: SnapshotHandle<AisixSnapshot>, resources_path: impl Into<String>) -> Self {
        Self {
            snapshot,
            resources_path: resources_path.into(),
        }
    }

    /// The message every write path returns. Includes the file path so
    /// the operator knows exactly what to edit.
    pub fn read_only_message(resources_path: &str) -> String {
        format!(
            "resources are file-managed: this gateway loads its resources from \
             {resources_path}; edit that file and send SIGHUP to reload instead of \
             using the resource write API"
        )
    }

    fn read_only(&self) -> StoreError {
        StoreError::ReadOnly(Self::read_only_message(&self.resources_path))
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
    ($( { $ty:ty, $table:ident, $put:ident, $get:ident, $list:ident, $delete:ident } )+) => {
        #[async_trait::async_trait]
        impl ConfigStore for FileManagedStore {
            $(
                async fn $put(&self, _entry: ResourceEntry<$ty>) -> Result<(), StoreError> {
                    Err(self.read_only())
                }

                async fn $get(&self, id: &str) -> Result<Option<ResourceEntry<$ty>>, StoreError> {
                    Ok(self.get_from(|s| &s.$table, id))
                }

                async fn $list(&self) -> Result<Vec<ResourceEntry<$ty>>, StoreError> {
                    Ok(self.list_from(|s| &s.$table))
                }

                async fn $delete(&self, _id: &str) -> Result<bool, StoreError> {
                    Err(self.read_only())
                }
            )+
        }
    };
}

impl_file_managed_store! {
    { Model, models, put_model, get_model, list_models, delete_model }
    { ApiKey, apikeys, put_apikey, get_apikey, list_apikeys, delete_apikey }
    { ProviderKey, provider_keys, put_provider_key, get_provider_key, list_provider_keys, delete_provider_key }
    { Guardrail, guardrails, put_guardrail, get_guardrail, list_guardrails, delete_guardrail }
    { CachePolicy, cache_policies, put_cache_policy, get_cache_policy, list_cache_policies, delete_cache_policy }
    { ObservabilityExporter, observability_exporters, put_observability_exporter, get_observability_exporter, list_observability_exporters, delete_observability_exporter }
    { McpServer, mcp_servers, put_mcp_server, get_mcp_server, list_mcp_servers, delete_mcp_server }
    { A2aAgent, a2a_agents, put_a2a_agent, get_a2a_agent, list_a2a_agents, delete_a2a_agent }
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
        let store = FileManagedStore::new(handle.clone(), "/etc/aisix/resources.yaml");

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

    #[tokio::test]
    async fn writes_and_deletes_are_read_only_errors_naming_the_file() {
        let store = FileManagedStore::new(snapshot_with_model(), "/etc/aisix/resources.yaml");
        let entry = store.get_model("m-1").await.unwrap().unwrap();

        let err = store.put_model(entry).await.unwrap_err();
        match &err {
            StoreError::ReadOnly(msg) => {
                assert!(msg.contains("file-managed"), "{msg}");
                assert!(msg.contains("/etc/aisix/resources.yaml"), "{msg}");
                assert!(msg.contains("SIGHUP"), "{msg}");
            }
            other => panic!("expected ReadOnly, got {other:?}"),
        }
        assert!(matches!(
            store.delete_model("m-1").await.unwrap_err(),
            StoreError::ReadOnly(_)
        ));
        // Spot-check a second kind so the macro expansion is covered.
        assert!(matches!(
            store.delete_guardrail("g-1").await.unwrap_err(),
            StoreError::ReadOnly(_)
        ));
    }
}
