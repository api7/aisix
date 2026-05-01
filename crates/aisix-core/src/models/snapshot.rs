//! The concrete snapshot shape for aisix — one table per entity kind.
//!
//! The etcd watch supervisor builds a fresh [`AisixSnapshot`] on every
//! coherent rebuild (compaction, initial load) and atomically swaps it into
//! a [`SnapshotHandle<AisixSnapshot>`]. The data plane only sees the handle.
//!
//! Tables grow as feature PRs add entities: Model + ApiKey landed first;
//! Credential + Budget arrived with PR #19; Team / Guardrail will follow.

use super::apikey::ApiKey;
use super::cache_policy::CachePolicy;
use super::credential::Credential;
use super::guardrail::Guardrail;
use super::model::Model;
use super::observability_exporter::ObservabilityExporter;
use super::team::Team;
use crate::snapshot::ResourceTable;

/// Composite of every typed [`ResourceTable`] the gateway reads on the hot
/// path. Cheap to construct empty; populated by the loader.
#[derive(Debug, Default)]
pub struct AisixSnapshot {
    pub models: ResourceTable<Model>,
    pub apikeys: ResourceTable<ApiKey>,
    pub credentials: ResourceTable<Credential>,
    pub teams: ResourceTable<Team>,
    pub guardrails: ResourceTable<Guardrail>,
    /// Per-env cache policies. Stage 2 honors only the existence of an
    /// enabled row to gate the cache; Stage 3 will parse `applies_to`
    /// + per-policy `ttl_seconds`. See `aisix-core::CachePolicy`.
    pub cache_policies: ResourceTable<CachePolicy>,
    /// Per-env observability exporters. Each enabled row receives a
    /// fan-out POST per chat completion (see `aisix-obs::OtlpHttpFanOut`).
    pub observability_exporters: ResourceTable<ObservabilityExporter>,
}

impl AisixSnapshot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience: total entry count across all tables. Handy for debug /
    /// readiness checks.
    pub fn total_entries(&self) -> usize {
        self.models.len()
            + self.apikeys.len()
            + self.credentials.len()
            + self.teams.len()
            + self.guardrails.len()
            + self.cache_policies.len()
            + self.observability_exporters.len()
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
        serde_json::from_str(r#"{"key_hash": "91ed2dbc407561556f3e7be98ba0bd2a57986d6a868c482d867d19c6d40d201c", "allowed_models": ["my-gpt4"]}"#)
            .unwrap()
    }

    fn sample_credential() -> Credential {
        serde_json::from_str(r#"{"name":"openai-prod","api_key":"sk-prod"}"#).unwrap()
    }

    #[test]
    fn empty_snapshot_has_no_entries() {
        let s = AisixSnapshot::new();
        assert_eq!(s.total_entries(), 0);
        assert!(s.models.is_empty());
        assert!(s.apikeys.is_empty());
        assert!(s.credentials.is_empty());
    }

    #[test]
    fn all_three_tables_are_independent() {
        let s = AisixSnapshot::new();
        s.models
            .insert(ResourceEntry::new("m-1", sample_model(), 1));
        s.apikeys
            .insert(ResourceEntry::new("k-1", sample_apikey(), 1));
        s.credentials
            .insert(ResourceEntry::new("c-1", sample_credential(), 1));

        assert_eq!(s.total_entries(), 3);
        assert_eq!(s.models.get_by_name("my-gpt4").unwrap().id, "m-1");
        assert_eq!(
            // Snapshot's by_name index for ApiKey is keyed by key_hash
            // (§9A.7B.4) — the SHA-256 of the bearer plaintext.
            s.apikeys
                .get_by_name("91ed2dbc407561556f3e7be98ba0bd2a57986d6a868c482d867d19c6d40d201c")
                .unwrap()
                .id,
            "k-1",
        );
        assert_eq!(s.credentials.get_by_name("openai-prod").unwrap().id, "c-1");
    }
}
