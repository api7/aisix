//! The concrete snapshot shape for aisix — one table per entity kind.
//!
//! The etcd watch supervisor builds a fresh [`AisixSnapshot`] on every
//! coherent rebuild (compaction, initial load) and atomically swaps it into
//! a [`SnapshotHandle<AisixSnapshot>`]. The data plane only sees the handle.
//!
//! Later entities (`Team`, `Budget`, `Guardrail`, …) will be added here as
//! their feature PRs land.

use super::apikey::ApiKey;
use super::model::Model;
use crate::snapshot::ResourceTable;

/// Composite of every typed [`ResourceTable`] the gateway reads on the hot
/// path. Cheap to construct empty; populated by the loader.
#[derive(Debug, Default)]
pub struct AisixSnapshot {
    pub models: ResourceTable<Model>,
    pub apikeys: ResourceTable<ApiKey>,
}

impl AisixSnapshot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience: total entry count across all tables. Handy for debug /
    /// readiness checks.
    pub fn total_entries(&self) -> usize {
        self.models.len() + self.apikeys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourceEntry;

    fn sample_model() -> Model {
        serde_json::from_str(
            r#"{
              "name": "my-gpt4",
              "model": "openai/gpt-4o",
              "provider_config": {"api_key": "sk-x"}
            }"#,
        )
        .unwrap()
    }

    fn sample_apikey() -> ApiKey {
        serde_json::from_str(r#"{"key": "sk-my-api-key-123", "allowed_models": ["my-gpt4"]}"#)
            .unwrap()
    }

    #[test]
    fn empty_snapshot_has_no_entries() {
        let s = AisixSnapshot::new();
        assert_eq!(s.total_entries(), 0);
        assert!(s.models.is_empty());
        assert!(s.apikeys.is_empty());
    }

    #[test]
    fn tables_are_independent() {
        let s = AisixSnapshot::new();
        s.models
            .insert(ResourceEntry::new("m-1", sample_model(), 1));
        s.apikeys
            .insert(ResourceEntry::new("k-1", sample_apikey(), 1));

        assert_eq!(s.total_entries(), 2);
        // Note: `.id` is the wrapper field (etcd key uuid); the inner
        // `Resource::id()` would read the private `runtime_id` which the
        // loader fills in separately.
        assert_eq!(s.models.get_by_name("my-gpt4").unwrap().id, "m-1");
        assert_eq!(
            s.apikeys.get_by_name("sk-my-api-key-123").unwrap().id,
            "k-1"
        );
    }
}
