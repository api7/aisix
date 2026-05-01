//! Turn raw etcd entries into a typed [`AisixSnapshot`].
//!
//! Flow:
//! 1. parse the key → `(kind, id)`
//! 2. validate the value against the kind's JSON Schema
//! 3. deserialise into the typed struct via serde (cheap after schema
//!    passes)
//! 4. insert into the appropriate [`ResourceTable`]
//!
//! Malformed payloads are logged at WARN level and skipped, not fatal —
//! this matches spec §2: "the gateway does not abort on a single bad
//! entry; it serves the rest."

use aisix_core::models::{
    validate_apikey, validate_cache_policy, validate_credential, validate_guardrail, validate_model,
    ApiKey, CachePolicy, Credential, Guardrail, Model, SchemaError,
};
use aisix_core::resource::ResourceEntry;
use aisix_core::AisixSnapshot;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::key::{self, ResourceKey};
use crate::provider::RawEntry;

/// Counts of rejected entries during a build, useful for metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BuildStats {
    pub accepted: usize,
    pub schema_rejected: usize,
    pub parse_rejected: usize,
    pub unknown_kind: usize,
    pub key_rejected: usize,
}

/// Build a fresh snapshot from raw entries. Never fails — bad rows are
/// counted in [`BuildStats`] and skipped. The prefix lets us strip it
/// before key parsing.
pub fn build_snapshot(prefix: &str, entries: &[RawEntry]) -> (AisixSnapshot, BuildStats) {
    let snapshot = AisixSnapshot::new();
    let mut stats = BuildStats::default();

    for raw in entries {
        let parsed = match key::parse(prefix, &raw.key) {
            Ok(k) => k,
            Err(err) => {
                tracing::warn!(key = %raw.key, error = %err, "skipping etcd entry with bad key");
                stats.key_rejected += 1;
                continue;
            }
        };

        let value: Value = match serde_json::from_slice(&raw.value) {
            Ok(v) => v,
            Err(err) => {
                tracing::warn!(key = %raw.key, error = %err, "skipping non-JSON etcd entry");
                stats.parse_rejected += 1;
                continue;
            }
        };

        match parsed.kind {
            "models" => {
                if let Some(entry) = validate_and_parse::<Model>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_model,
                    &mut stats,
                ) {
                    snapshot.models.insert(entry);
                }
            }
            "api_keys" => {
                if let Some(entry) = validate_and_parse::<ApiKey>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_apikey,
                    &mut stats,
                ) {
                    snapshot.apikeys.insert(entry);
                }
            }
            "credentials" => {
                if let Some(entry) = validate_and_parse::<Credential>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_credential,
                    &mut stats,
                ) {
                    snapshot.credentials.insert(entry);
                }
            }
            "guardrails" => {
                if let Some(entry) = validate_and_parse::<Guardrail>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_guardrail,
                    &mut stats,
                ) {
                    snapshot.guardrails.insert(entry);
                }
            }
            "cache_policies" => {
                if let Some(entry) = validate_and_parse::<CachePolicy>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_cache_policy,
                    &mut stats,
                ) {
                    snapshot.cache_policies.insert(entry);
                }
            }
            other => {
                tracing::debug!(key = %raw.key, kind = %other, "unknown etcd kind; skipping");
                stats.unknown_kind += 1;
            }
        }
    }

    (snapshot, stats)
}

fn validate_and_parse<T>(
    key: &str,
    revision: i64,
    parsed: ResourceKey<'_>,
    value: &Value,
    validate: fn(&Value) -> Result<(), SchemaError>,
    stats: &mut BuildStats,
) -> Option<ResourceEntry<T>>
where
    T: DeserializeOwned,
{
    if let Err(err) = validate(value) {
        tracing::warn!(key = %key, error = %err, "schema validation failed; skipping");
        stats.schema_rejected += 1;
        return None;
    }

    match serde_json::from_value::<T>(value.clone()) {
        Ok(t) => {
            stats.accepted += 1;
            Some(ResourceEntry::new(parsed.id, t, revision))
        }
        Err(err) => {
            // Schema passed but serde refused — usually a deny_unknown_fields
            // mismatch. Treat as schema-rejected for stats purposes.
            tracing::warn!(key = %key, error = %err, "serde parse failed after schema pass");
            stats.parse_rejected += 1;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(key: &str, value: &[u8], rev: i64) -> RawEntry {
        RawEntry {
            key: key.into(),
            value: value.to_vec(),
            revision: rev,
        }
    }

    const VALID_MODEL: &[u8] = br#"{
        "name": "my-gpt4",
        "model": "openai/gpt-4o",
        "provider_config": {"api_key": "sk-x"}
    }"#;

    const VALID_APIKEY: &[u8] = br#"{
        "key_hash": "1460db1b6902f8b1fc2a40d9381a24d0fd22c3bc1b2c6f999c521da73776fbe0",
        "allowed_models": ["my-gpt4"]
    }"#;

    #[test]
    fn builds_snapshot_for_happy_entries() {
        let entries = vec![
            raw("/aisix/models/m-1", VALID_MODEL, 2),
            raw("/aisix/api_keys/k-1", VALID_APIKEY, 3),
        ];
        let (snap, stats) = build_snapshot("/aisix", &entries);

        assert_eq!(stats.accepted, 2);
        assert_eq!(snap.models.len(), 1);
        assert_eq!(snap.apikeys.len(), 1);
        assert_eq!(snap.models.get_by_name("my-gpt4").unwrap().id, "m-1");
        // by_name index for ApiKey is keyed by key_hash (§9A.7B.4).
        assert_eq!(
            snap.apikeys
                .get_by_name("1460db1b6902f8b1fc2a40d9381a24d0fd22c3bc1b2c6f999c521da73776fbe0")
                .unwrap()
                .id,
            "k-1"
        );
    }

    #[test]
    fn malformed_json_is_skipped_not_fatal() {
        let entries = vec![
            raw("/aisix/models/bad", b"not-json", 1),
            raw("/aisix/models/good", VALID_MODEL, 2),
        ];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.parse_rejected, 1);
        assert_eq!(stats.accepted, 1);
        assert_eq!(snap.models.len(), 1);
    }

    #[test]
    fn schema_failure_is_counted() {
        let entries = vec![raw(
            "/aisix/models/bad-provider",
            br#"{"name":"x","model":"mistral/large","provider_config":{"api_key":"k"}}"#,
            1,
        )];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.schema_rejected, 1);
        assert_eq!(stats.accepted, 0);
    }

    #[test]
    fn unknown_kinds_are_skipped() {
        let entries = vec![raw("/aisix/teams/t-1", b"{}", 1)];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.unknown_kind, 1);
        assert!(snap.models.is_empty());
        assert!(snap.apikeys.is_empty());
    }

    #[test]
    fn bad_key_shape_is_counted_separately() {
        let entries = vec![raw("/other/models/a", VALID_MODEL, 1)];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.key_rejected, 1);
    }

    #[test]
    fn one_bad_entry_does_not_abort_the_batch() {
        let entries = vec![
            raw("/aisix/models/m-1", VALID_MODEL, 1),
            raw("/aisix/models/bad", b"not-json", 2),
            raw("/aisix/models/m-2", VALID_MODEL, 3), // same name -> update in place
            raw("/aisix/api_keys/k-1", VALID_APIKEY, 4),
        ];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 3);
        assert_eq!(stats.parse_rejected, 1);
        // m-1 and m-2 share the same name; the second insert rebinds the
        // name to m-2, but both id entries are present in the table.
        assert_eq!(snap.models.len(), 2);
        assert_eq!(snap.apikeys.len(), 1);
    }
}
