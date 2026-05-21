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
    validate_apikey, validate_cache_policy, validate_guardrail, validate_model,
    validate_observability_exporter, validate_provider_key, validate_rate_limit_policy, ApiKey,
    CachePolicy, Guardrail, Model, ObservabilityExporter, ProviderKey, RateLimitPolicy,
    SchemaError,
};
use aisix_core::resource::ResourceEntry;
use aisix_core::AisixSnapshot;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::key::{self, ResourceKey};
use crate::provider::RawEntry;

/// Why the loader skipped an entry. Surfaced in [`RejectedEntry`] so
/// the heartbeat / health surface can tell operators what kind of
/// problem hit each row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionKind {
    /// Key didn't match the `<prefix>/<kind>/<id>` shape.
    BadKey,
    /// Value bytes didn't parse as JSON.
    NonJson,
    /// JSON parsed but failed the kind's JSON Schema.
    SchemaFailed,
    /// JSON Schema passed but `serde_json::from_value` refused — usually
    /// a `deny_unknown_fields` mismatch between schema and Rust struct.
    ParseFailed,
    /// Key referenced a `kind` segment we don't know about. Logged at
    /// debug normally but counted here so unknown kinds show up too.
    UnknownKind,
}

impl RejectionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BadKey => "bad_key",
            Self::NonJson => "non_json",
            Self::SchemaFailed => "schema_failed",
            Self::ParseFailed => "parse_failed",
            Self::UnknownKind => "unknown_kind",
        }
    }
}

/// One rejected etcd entry. Captured by the loader on every skip path
/// so the data plane can report back to the control plane via heartbeat
/// — without this signal a user who saved an invalid row in the
/// dashboard sees "Saved successfully" but has no way to learn the DP
/// dropped it. See issue #115.
///
/// `timestamp_unix_secs` is wall-clock seconds-since-epoch so the
/// heartbeat / dashboard can age-out old rejections without parsing
/// a [`SystemTime`] across the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedEntry {
    pub key: String,
    pub kind: RejectionKind,
    pub error: String,
    pub timestamp_unix_secs: u64,
}

impl RejectedEntry {
    fn new(key: impl Into<String>, kind: RejectionKind, error: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            key: key.into(),
            kind,
            error: error.into(),
            timestamp_unix_secs: now,
        }
    }
}

/// Counts of rejected entries during a build, plus the rejection
/// list itself. The counts stay handy for metrics; the list is what
/// the heartbeat sends upstream so the dashboard can show "your DP
/// rejected these resources, here's why."
///
/// `Copy` is dropped because the rejections vec can be large; existing
/// call sites that took `BuildStats` by value continue to work via
/// the auto-derived `Clone` (only invoked explicitly when needed).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BuildStats {
    pub accepted: usize,
    pub schema_rejected: usize,
    pub parse_rejected: usize,
    pub unknown_kind: usize,
    pub key_rejected: usize,
    /// Detailed reject list. One entry per skipped row, in the order
    /// the loader processed them. Capacity is whatever the caller's
    /// upstream provider feeds in; the supervisor caps its retained
    /// buffer separately.
    pub rejections: Vec<RejectedEntry>,
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
                stats.rejections.push(RejectedEntry::new(
                    raw.key.clone(),
                    RejectionKind::BadKey,
                    err.to_string(),
                ));
                continue;
            }
        };

        let value: Value = match serde_json::from_slice(&raw.value) {
            Ok(v) => v,
            Err(err) => {
                tracing::warn!(key = %raw.key, error = %err, "skipping non-JSON etcd entry");
                stats.parse_rejected += 1;
                stats.rejections.push(RejectedEntry::new(
                    raw.key.clone(),
                    RejectionKind::NonJson,
                    err.to_string(),
                ));
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
            "provider_keys" => {
                if let Some(entry) = validate_and_parse::<ProviderKey>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_provider_key,
                    &mut stats,
                ) {
                    snapshot.provider_keys.insert(entry);
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
            "observability_exporters" => {
                if let Some(entry) = validate_and_parse::<ObservabilityExporter>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_observability_exporter,
                    &mut stats,
                ) {
                    snapshot.observability_exporters.insert(entry);
                }
            }
            "rate_limit_policies" => {
                if let Some(entry) = validate_and_parse::<RateLimitPolicy>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_rate_limit_policy,
                    &mut stats,
                ) {
                    snapshot.rate_limit_policies.insert(entry);
                }
            }
            other => {
                tracing::debug!(key = %raw.key, kind = %other, "unknown etcd kind; skipping");
                stats.unknown_kind += 1;
                stats.rejections.push(RejectedEntry::new(
                    raw.key.clone(),
                    RejectionKind::UnknownKind,
                    format!("unknown kind {other:?}"),
                ));
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
        stats.rejections.push(RejectedEntry::new(
            key,
            RejectionKind::SchemaFailed,
            err.to_string(),
        ));
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
            stats.rejections.push(RejectedEntry::new(
                key,
                RejectionKind::ParseFailed,
                err.to_string(),
            ));
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
        "display_name": "my-gpt4",
        "provider": "openai",
        "model_name": "gpt-4o",
        "provider_key_id": "11111111-1111-1111-1111-111111111111"
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
            br#"{"display_name":"x","provider":"this-is-not-a-provider-id","model_name":"large","provider_key_id":"pk-1"}"#,
            1,
        )];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.schema_rejected, 1);
        assert_eq!(stats.accepted, 0);
    }

    #[test]
    fn unknown_kinds_are_skipped() {
        let entries = vec![raw("/aisix/unknown_kind/x-1", b"{}", 1)];
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

    // ---- regression coverage for issue #115 -------------------------
    // The loader used to log a warning and silently skip invalid rows.
    // Customers who saved an invalid resource in the dashboard saw
    // "Saved" but the DP dropped the row — no signal back. The fix
    // attaches a `RejectedEntry` per skip path so the heartbeat can
    // surface the failure to cp-api.

    #[test]
    fn rejection_records_bad_key_with_kind_and_error_message() {
        let entries = vec![raw("/wrong/models/x", VALID_MODEL, 1)];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.rejections.len(), 1);
        assert_eq!(stats.rejections[0].kind, RejectionKind::BadKey);
        assert_eq!(stats.rejections[0].key, "/wrong/models/x");
        assert!(!stats.rejections[0].error.is_empty());
    }

    #[test]
    fn rejection_records_non_json_payload() {
        let entries = vec![raw("/aisix/models/m1", b"not-json", 1)];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.rejections.len(), 1);
        assert_eq!(stats.rejections[0].kind, RejectionKind::NonJson);
    }

    #[test]
    fn rejection_records_schema_failure() {
        let entries = vec![raw(
            "/aisix/models/bad",
            br#"{"display_name":"x","provider":"this-is-not-a-provider-id","model_name":"l","provider_key_id":"pk"}"#,
            1,
        )];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.rejections.len(), 1);
        assert_eq!(stats.rejections[0].kind, RejectionKind::SchemaFailed);
    }

    #[test]
    fn rejection_records_unknown_kind() {
        let entries = vec![raw("/aisix/unknown_kind/x-1", b"{}", 1)];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.rejections.len(), 1);
        assert_eq!(stats.rejections[0].kind, RejectionKind::UnknownKind);
    }

    #[test]
    fn happy_entries_have_no_rejections() {
        let entries = vec![raw("/aisix/models/m-1", VALID_MODEL, 1)];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert!(stats.rejections.is_empty());
    }

    // ---- provider_key schema coverage (issue api7/AISIX-Cloud#398
    //      Tier 3 "DP loader schema check") ----------------------
    //
    // Pins the ProviderKey loader path. The existing tests above
    // cover only `models` and `api_keys` shapes — the
    // `provider_keys` branch at L175 was unverified for either
    // happy-path acceptance or for Adapter family extra-config
    // rejection (the gap the audit on #398 originally flagged).

    const VALID_PROVIDER_KEY: &[u8] = br#"{
        "display_name": "openai-mock",
        "secret": "sk-test-not-real",
        "api_base": "http://mock-llm:8000/v1",
        "provider": "openai"
    }"#;

    #[test]
    fn provider_key_happy_path_accepts() {
        let entries = vec![raw(
            "/aisix/provider_keys/pk-1",
            VALID_PROVIDER_KEY,
            1,
        )];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1);
        assert_eq!(snap.provider_keys.len(), 1);
        assert!(stats.rejections.is_empty());
    }

    #[test]
    fn provider_key_aws_region_payload_currently_rejected() {
        // Documents the current gap: an `amazon-bedrock`
        // provider_key payload must carry `aws_region` per
        // adapter_map.yaml:30, but the ProviderKey struct in
        // `aisix-core::models::provider_key` is
        // `#[serde(deny_unknown_fields)]` and has no `aws_region`
        // field. Today the loader REJECTS the entry, so a customer
        // creating a Bedrock provider_key via cp-api never sees
        // the row reach the DP. Tracked as a follow-up to Adapter
        // family e2e coverage (Tier 3 + Tier 4-7 in #398).
        //
        // When ProviderKey gains adapter-family extra-config fields
        // (or an `extra: HashMap<String,Value>` escape hatch), this
        // test should flip to assert `accepted=1` instead.
        let entries = vec![raw(
            "/aisix/provider_keys/pk-bedrock",
            br#"{"display_name":"bedrock-pk","secret":"x","provider":"amazon-bedrock","aws_region":"us-east-1"}"#,
            1,
        )];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 0);
        assert_eq!(stats.schema_rejected, 1);
        assert_eq!(stats.rejections.len(), 1);
        assert_eq!(stats.rejections[0].kind, RejectionKind::SchemaFailed);
    }

    #[test]
    fn provider_key_gcp_project_payload_currently_rejected() {
        // Same gap as aws_region: `google-vertex` needs
        // gcp_project + gcp_region per adapter_map.yaml. Today the
        // loader rejects.
        let entries = vec![raw(
            "/aisix/provider_keys/pk-vertex",
            br#"{"display_name":"vertex-pk","secret":"x","provider":"google-vertex","gcp_project":"my-proj","gcp_region":"us-central1"}"#,
            1,
        )];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 0);
        assert_eq!(stats.schema_rejected, 1);
    }

    #[test]
    fn provider_key_azure_resource_payload_currently_rejected() {
        // Same gap as aws_region: `azure` needs
        // azure_resource_name + api_version per adapter_map.yaml.
        // Today the loader rejects.
        let entries = vec![raw(
            "/aisix/provider_keys/pk-azure",
            br#"{"display_name":"azure-pk","secret":"x","provider":"azure","azure_resource_name":"my-azure","api_version":"2024-02-01"}"#,
            1,
        )];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 0);
        assert_eq!(stats.schema_rejected, 1);
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

    const VALID_RATE_LIMIT_POLICY: &[u8] = br#"{
        "name": "team-quota",
        "scope": "team",
        "scope_ref": "team-uuid-1",
        "window": "minute",
        "max_requests": 100
    }"#;

    #[test]
    fn rate_limit_policy_loads_into_snapshot() {
        let entries = vec![raw(
            "/aisix/rate_limit_policies/rlp-1",
            VALID_RATE_LIMIT_POLICY,
            5,
        )];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1);
        assert_eq!(snap.rate_limit_policies.len(), 1);
        let entry = snap.rate_limit_policies.get_by_id("rlp-1").unwrap();
        assert_eq!(entry.value.name, "team-quota");
        assert_eq!(entry.value.scope, "team");
        assert_eq!(entry.value.scope_ref, "team-uuid-1");
        assert_eq!(entry.value.max_requests, Some(100));
    }
}
