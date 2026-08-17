//! Turn raw etcd entries into a typed [`AisixSnapshot`].
//!
//! Flow:
//! 1. parse the key → `(kind, id)`
//! 2. validate the value against the kind's **lenient** JSON Schema —
//!    types, required fields, ranges and closed enums are enforced;
//!    unknown fields are not
//! 3. deserialise into the typed struct via `serde_ignored`, collecting
//!    the paths of any fields serde had to ignore
//! 4. insert into the appropriate [`ResourceTable`]
//!
//! Every row lands in one of three compatibility states (issue #871):
//!
//! - **incompatible** (RED): step 2 or 3 fails — the row is skipped and
//!   logged at ERROR, as today's contract genuinely cannot represent it;
//! - **partially compatible** (YELLOW): the row loaded but carried fields
//!   this build does not know — typically written by a newer control
//!   plane. It serves with those fields ignored, and the ignored paths
//!   are reported through [`BuildStats::partially_compatible`];
//! - fully compatible (GREEN): exact match, no signal.
//!
//! Rejected payloads are skipped, not fatal — this matches spec §2:
//! "the gateway does not abort on a single bad entry; it serves the
//! rest."

use aisix_core::models::{
    validate_a2a_agent_lenient, validate_apikey_lenient, validate_cache_policy_lenient,
    validate_claim_mapping_lenient, validate_guardrail_attachment_lenient,
    validate_guardrail_lenient, validate_mcp_policy_lenient, validate_mcp_server_lenient,
    validate_model_lenient, validate_observability_exporter_lenient,
    validate_oidc_provider_lenient, validate_passthrough_route_lenient,
    validate_provider_key_lenient, validate_rate_limit_policy_lenient, A2aAgent, ApiKey,
    CachePolicy, ClaimMapping, Guardrail, GuardrailAttachment, McpPolicy, McpServer, Model,
    ObservabilityExporter, OidcProvider, PassthroughRoute, ProviderKey, RateLimitPolicy,
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
    /// JSON Schema passed but serde deserialization refused — e.g. a
    /// duplicate field via a rename alias, or an unknown field inside a
    /// tagged enum whose shape stays closed.
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
    /// Unix seconds since when this key has been serving its last known
    /// good value instead of the rejected bytes (#871). Always `None` as
    /// produced by the loader — the supervisor joins its retained
    /// stale-serving state in on read, so the heartbeat reports the
    /// staleness age next to the rejection.
    pub stale_serving_since_unix_secs: Option<u64>,
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
            stale_serving_since_unix_secs: None,
        }
    }
}

/// One unknown-field observation from a row that still loaded — the
/// YELLOW ("partially compatible") state of issue #871. Aggregated per
/// (kind, field path) with a row count, never per row, so a fleet-wide
/// additive CP change produces one entry per field instead of one per
/// resource. Kept apart from [`RejectedEntry`] so YELLOW volume can
/// never evict RED entries from the retained rejection buffer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PartialCompatEntry {
    /// Resource kind segment as it appears in the etcd key
    /// (e.g. `api_keys`).
    pub kind: String,
    /// Dotted path of the ignored field inside the document
    /// (e.g. `quota_profile` or `rate_limit.burst`). Array indices are
    /// normalized to `[]` (`routing.targets[].priority`) so the entry
    /// count is bounded by the document shape, not the data volume.
    pub field: String,
    /// Number of rows of this kind carrying this unknown field in the
    /// build.
    pub count: usize,
}

/// The unknown fields one loaded row carried, keyed by its full etcd
/// key. This is the per-row form the supervisor merges into its
/// retained YELLOW state on incremental watch events (a re-put of the
/// same key replaces its entry; a delete removes it); the aggregated
/// [`PartialCompatEntry`] form is derived from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialCompatRow {
    /// Full etcd key of the row.
    pub key: String,
    /// Resource kind segment (e.g. `api_keys`).
    pub kind: String,
    /// Ignored field paths, index-normalized and deduplicated, sorted.
    pub fields: Vec<String>,
}

/// Aggregate per-row unknown-field observations into the reporting form:
/// one entry per (kind, field path) with the number of rows carrying it,
/// sorted by kind then field.
pub fn aggregate_partial_compat(rows: &[PartialCompatRow]) -> Vec<PartialCompatEntry> {
    let mut counts: std::collections::BTreeMap<(&str, &str), usize> = Default::default();
    for row in rows {
        for field in &row.fields {
            *counts
                .entry((row.kind.as_str(), field.as_str()))
                .or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .map(|((kind, field), count)| PartialCompatEntry {
            kind: kind.to_string(),
            field: field.to_string(),
            count,
        })
        .collect()
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
    /// Unknown-field observations from rows that loaded anyway,
    /// aggregated per (kind, field path). Empty when every row matched
    /// its schema exactly.
    pub partially_compatible: Vec<PartialCompatEntry>,
    /// The same observations in per-row form (etcd key → ignored
    /// fields), for the supervisor's incremental watch-event merging.
    pub partial_rows: Vec<PartialCompatRow>,
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
                let row_kind = parsed.kind;
                if let Some(mut entry) = validate_and_parse::<Model>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_model_lenient,
                    &mut stats,
                ) {
                    // Known-but-kind-inapplicable knobs load stripped and
                    // report through the same partially-compatible channel
                    // as unknown fields — the strict write path rejects
                    // these shapes, stored rows must keep loading.
                    let stripped = entry.value.strip_kind_inapplicable();
                    if !stripped.is_empty() {
                        let fields: Vec<String> = stripped
                            .iter()
                            .map(|f| format!("inapplicable:{f}"))
                            .collect();
                        warn_partial_compat_deduped(&raw.key, row_kind, &fields);
                        // MERGE into this key's existing partial-compat
                        // row rather than pushing a second one: a row can
                        // carry BOTH unknown fields (pushed inside
                        // validate_and_parse) and inapplicable knobs, and
                        // the supervisor keys its retained report by etcd
                        // key — two rows for one key would drop one half
                        // (resync overwrites, watch takes the first).
                        merge_partial_compat_fields(&mut stats, &raw.key, row_kind, fields);
                    }
                    snapshot.models.insert(entry);
                }
            }
            "api_keys" => {
                if let Some(entry) = validate_and_parse::<ApiKey>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_apikey_lenient,
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
                    validate_provider_key_lenient,
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
                    validate_guardrail_lenient,
                    &mut stats,
                ) {
                    snapshot.guardrails.insert(entry);
                }
            }
            "guardrail_attachments" => {
                if let Some(entry) = validate_and_parse::<GuardrailAttachment>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_guardrail_attachment_lenient,
                    &mut stats,
                ) {
                    snapshot.guardrail_attachments.insert(entry);
                }
            }
            "cache_policies" => {
                if let Some(entry) = validate_and_parse::<CachePolicy>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_cache_policy_lenient,
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
                    validate_observability_exporter_lenient,
                    &mut stats,
                ) {
                    snapshot.observability_exporters.insert(entry);
                }
            }
            "rate_limit_policies" => {
                // The condition-tree caps, operator×dimension matrix and
                // regex compilability are beyond the JSON Schema — the
                // semantic hook rejects such rows inside
                // `validate_and_parse`, BEFORE any accept accounting, so a
                // failing row is indistinguishable from a schema failure
                // everywhere downstream (`apply_put` gates success on
                // `stats.accepted`, partial-compat state is never
                // recorded for a row that does not serve).
                if let Some(entry) = validate_and_parse_with_semantics::<RateLimitPolicy>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_rate_limit_policy_lenient,
                    |p| p.validate_semantics(),
                    &mut stats,
                ) {
                    snapshot.rate_limit_policies.insert(entry);
                }
            }
            "mcp_servers" => {
                if let Some(entry) = validate_and_parse::<McpServer>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_mcp_server_lenient,
                    &mut stats,
                ) {
                    snapshot.mcp_servers.insert(entry);
                }
            }
            "mcp_policies" => {
                if let Some(entry) = validate_and_parse::<McpPolicy>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_mcp_policy_lenient,
                    &mut stats,
                ) {
                    snapshot.mcp_policies.insert(entry);
                }
            }
            "passthrough_routes" => {
                if let Some(entry) = validate_and_parse::<PassthroughRoute>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_passthrough_route_lenient,
                    &mut stats,
                ) {
                    snapshot.passthrough_routes.insert(entry);
                }
            }
            "a2a_agents" => {
                if let Some(entry) = validate_and_parse::<A2aAgent>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_a2a_agent_lenient,
                    &mut stats,
                ) {
                    snapshot.a2a_agents.insert(entry);
                }
            }
            "oidc_providers" => {
                if let Some(entry) = validate_and_parse::<OidcProvider>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_oidc_provider_lenient,
                    &mut stats,
                ) {
                    snapshot.oidc_providers.insert(entry);
                }
            }
            "claim_mappings" => {
                if let Some(entry) = validate_and_parse::<ClaimMapping>(
                    &raw.key,
                    raw.revision,
                    parsed,
                    &value,
                    validate_claim_mapping_lenient,
                    &mut stats,
                ) {
                    snapshot.claim_mappings.insert(entry);
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

    stats.partially_compatible = aggregate_partial_compat(&stats.partial_rows);
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
    validate_and_parse_with_semantics(key, revision, parsed, value, validate, |_| Ok(()), stats)
}

/// [`validate_and_parse`] plus a typed semantic hook, run after serde
/// succeeds but BEFORE any accept accounting. A semantic failure is
/// recorded exactly like a schema failure (RED, `schema_rejected`,
/// [`RejectionKind::SchemaFailed`]) and the row contributes nothing to
/// `accepted`/`partial_rows` — so `Supervisor::apply_put`, which gates
/// success on `stats.accepted`, retains the last-good value and
/// surfaces the rejection, instead of reporting a silent no-op apply.
fn validate_and_parse_with_semantics<T>(
    key: &str,
    revision: i64,
    parsed: ResourceKey<'_>,
    value: &Value,
    validate: fn(&Value) -> Result<(), SchemaError>,
    semantic: fn(&T) -> Result<(), String>,
    stats: &mut BuildStats,
) -> Option<ResourceEntry<T>>
where
    T: DeserializeOwned,
{
    // RED: the lenient schema still enforces types, required fields,
    // ranges and closed enums — a failure here means this build cannot
    // represent the row at all, so it is skipped. ERROR, not warn: under
    // the supported CP-before-DP upgrade order this is the signal that a
    // resource stopped applying on this instance.
    if let Err(err) = validate(value) {
        tracing::error!(key = %key, error = %err, "schema validation failed; skipping (incompatible row)");
        stats.schema_rejected += 1;
        stats.rejections.push(RejectedEntry::new(
            key,
            RejectionKind::SchemaFailed,
            err.to_string(),
        ));
        return None;
    }

    let mut ignored: Vec<String> = Vec::new();
    match serde_ignored::deserialize::<_, _, T>(value, |path| {
        ignored.push(normalize_ignored_path(&path.to_string()));
    }) {
        Ok(t) => {
            if let Err(err) = semantic(&t) {
                tracing::error!(key = %key, error = %err, "semantic validation failed; skipping (incompatible row)");
                stats.schema_rejected += 1;
                stats
                    .rejections
                    .push(RejectedEntry::new(key, RejectionKind::SchemaFailed, err));
                return None;
            }
            stats.accepted += 1;
            if !ignored.is_empty() {
                // YELLOW: loaded, but fields this build does not know were
                // ignored — typically written by a newer control plane.
                ignored.sort_unstable();
                ignored.dedup();
                // A single document can carry an arbitrary number of
                // unknown fields with arbitrary-length names; everything
                // captured here flows into logs, the retained map, the
                // status JSON and the heartbeat body. Cap per row — the
                // sentinel keeps the truncation visible in every report.
                if ignored.len() > MAX_REPORTED_FIELDS_PER_ROW {
                    ignored.truncate(MAX_REPORTED_FIELDS_PER_ROW);
                    ignored.push("...truncated".to_string());
                }
                warn_partial_compat_deduped(key, parsed.kind, &ignored);
                stats.partial_rows.push(PartialCompatRow {
                    key: key.to_string(),
                    kind: parsed.kind.to_string(),
                    fields: ignored,
                });
            }
            Some(ResourceEntry::new(parsed.id, t, revision))
        }
        Err(err) => {
            // RED: schema passed but serde refused — a duplicate field via
            // a rename alias, or an unknown field inside a tagged enum
            // whose shape stays closed.
            tracing::error!(key = %key, error = %err, "serde parse failed after schema pass (incompatible row)");
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

/// Cap on ignored-field paths reported per row. A row over the cap keeps
/// its first entries (sorted) plus a `...truncated` sentinel, so the
/// truncation shows up in every downstream report instead of silently
/// under-counting.
const MAX_REPORTED_FIELDS_PER_ROW: usize = 64;

/// Normalize a `serde_ignored` path into document terms: array indices
/// become `[]` so the aggregated report stays bounded by the document
/// shape (`targets.0.x` and `targets.1.x` are one field, not two), and
/// the `?` segments serde_ignored emits for `Option` wrapping layers —
/// invisible in the JSON — are dropped (`rate_limit.?.burst` →
/// `rate_limit.burst`).
///
/// Lossy by design: a map key that is itself all digits (or literally
/// `?`) aliases with the normalized forms and merges counts, in the
/// aggregate and in the WARN line alike. The actionable signal is the
/// field name per kind, not which array element carried it.
fn normalize_ignored_path(path: &str) -> String {
    path.split('.')
        .filter(|seg| *seg != "?")
        .map(|seg| {
            if seg.parse::<usize>().is_ok() {
                "[]"
            } else {
                seg
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Add `fields` to the partial-compat row already recorded for `key`
/// this build, or start one if none exists. Exactly one row per etcd key
/// so the supervisor's key-addressed retained report never drops a half
/// (a row with both unknown AND inapplicable fields). `rfind` because
/// `validate_and_parse` pushes the unknown-fields row for this same key
/// moments earlier — it is at or near the tail.
fn merge_partial_compat_fields(stats: &mut BuildStats, key: &str, kind: &str, fields: Vec<String>) {
    match stats.partial_rows.iter_mut().rfind(|r| r.key == key) {
        Some(row) => {
            row.fields.extend(fields);
            row.fields.sort_unstable();
            row.fields.dedup();
        }
        None => stats.partial_rows.push(PartialCompatRow {
            key: key.to_string(),
            kind: kind.to_string(),
            fields,
        }),
    }
}

/// WARN once per (kind, field-set) for the process lifetime. Resyncs
/// rebuild the whole snapshot on a cadence; without dedup every cycle
/// would re-log every YELLOW row. The set is capped: past the cap new
/// combinations keep logging (never silently dropped) but are no longer
/// remembered, so a pathological fleet re-logs on each resync instead
/// of growing memory without bound.
fn warn_partial_compat_deduped(key: &str, kind: &str, fields: &[String]) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    const MAX_REMEMBERED: usize = 1024;
    static WARNED: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();

    let fields_joined = fields.join(",");
    let entry = (kind.to_string(), fields_joined);
    // Poison-tolerant: this set only dedupes log lines, so a panic while
    // the lock was held (e.g. inside a tracing subscriber) must not wedge
    // every subsequent snapshot build in the supervisor task.
    let mut warned = WARNED
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if warned.contains(&entry) {
        return;
    }
    tracing::warn!(
        key = %key,
        kind = %kind,
        ignored_fields = %entry.1,
        "row loaded with unknown fields ignored (partially compatible; \
         likely written by a newer control plane)"
    );
    if warned.len() < MAX_REMEMBERED {
        warned.insert(entry);
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
        // After #302 Phase A `provider` is a free-form string. Use a
        // genuine schema violation (empty `display_name`) to keep the
        // rejection-path test meaningful.
        let entries = vec![raw(
            "/aisix/models/bad-shape",
            br#"{"display_name":"","provider":"openai","model_name":"large","provider_key_id":"pk-1"}"#,
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
            br#"{"display_name":"","provider":"openai","model_name":"l","provider_key_id":"pk"}"#,
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
        let entries = vec![raw("/aisix/provider_keys/pk-1", VALID_PROVIDER_KEY, 1)];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1);
        assert_eq!(snap.provider_keys.len(), 1);
        assert!(stats.rejections.is_empty());
    }

    #[test]
    fn provider_key_aws_region_payload_loads_partially_compatible() {
        // Flip of the pre-#871 `*_currently_rejected` pin. cp-api's
        // adapter_map admits Bedrock provider_key payloads carrying a
        // top-level `aws_region`; the DP adapter never reads that field
        // — the Bedrock bridge takes its region from the credential JSON
        // inside `api_key` (`aisix-provider-bedrock/src/bridge.rs`,
        // `BedrockSecret.region`). So the field stays off the model
        // (a struct field nothing consumes would be dead config) and the
        // row loads with the field ignored and reported, instead of the
        // pre-#871 whole-row rejection that silently kept the key from
        // ever reaching dispatch.
        let entries = vec![raw(
            "/aisix/provider_keys/pk-bedrock",
            br#"{"display_name":"bedrock-pk","secret":"x","provider":"amazon-bedrock","aws_region":"us-east-1"}"#,
            1,
        )];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1, "rejections: {:?}", stats.rejections);
        assert!(stats.rejections.is_empty());
        assert!(snap.provider_keys.get_by_id("pk-bedrock").is_some());
        assert_eq!(
            stats.partially_compatible,
            vec![PartialCompatEntry {
                kind: "provider_keys".into(),
                field: "aws_region".into(),
                count: 1,
            }]
        );
    }

    #[test]
    fn provider_key_gcp_project_payload_loads_partially_compatible() {
        // Same decision as aws_region: the Vertex bridge reads project
        // and region from the credential JSON inside `api_key`
        // (`VertexSecret.project` / `.region`), never from top-level
        // fields — YELLOW load, fields reported.
        let entries = vec![raw(
            "/aisix/provider_keys/pk-vertex",
            br#"{"display_name":"vertex-pk","secret":"x","provider":"google-vertex","gcp_project":"my-proj","gcp_region":"us-central1"}"#,
            1,
        )];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1, "rejections: {:?}", stats.rejections);
        assert!(snap.provider_keys.get_by_id("pk-vertex").is_some());
        assert_eq!(
            stats.partially_compatible,
            vec![
                PartialCompatEntry {
                    kind: "provider_keys".into(),
                    field: "gcp_project".into(),
                    count: 1,
                },
                PartialCompatEntry {
                    kind: "provider_keys".into(),
                    field: "gcp_region".into(),
                    count: 1,
                },
            ]
        );
    }

    #[test]
    fn provider_key_azure_resource_payload_loads_partially_compatible() {
        // Same decision as aws_region: the Azure bridge derives the
        // resource name from `api_base` and pins its own API version —
        // neither top-level field is consumed — YELLOW load, fields
        // reported.
        let entries = vec![raw(
            "/aisix/provider_keys/pk-azure",
            br#"{"display_name":"azure-pk","secret":"x","provider":"azure","azure_resource_name":"my-azure","api_version":"2024-02-01"}"#,
            1,
        )];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1, "rejections: {:?}", stats.rejections);
        assert!(snap.provider_keys.get_by_id("pk-azure").is_some());
        assert_eq!(
            stats.partially_compatible,
            vec![
                PartialCompatEntry {
                    kind: "provider_keys".into(),
                    field: "api_version".into(),
                    count: 1,
                },
                PartialCompatEntry {
                    kind: "provider_keys".into(),
                    field: "azure_resource_name".into(),
                    count: 1,
                },
            ]
        );
    }

    // ---- forward-compat: lenient parse + tri-state (issue #871) ----
    //
    // Under the supported rolling-upgrade order (CP first, DPs
    // behind), an upgraded CP adds a field to a resource and an older
    // DP receives the document. The DP must load the row with the
    // unknown field ignored (YELLOW / partially compatible) and report
    // exactly which field it ignored — not whole-row reject, which
    // silently drops the resource on the next resync/restart.

    #[test]
    fn api_key_unknown_field_is_accepted_and_reported_partially_compatible() {
        let entries = vec![raw(
            "/aisix/api_keys/k-forward",
            br#"{
                "key_hash": "1460db1b6902f8b1fc2a40d9381a24d0fd22c3bc1b2c6f999c521da73776fbe0",
                "allowed_models": ["my-gpt4"],
                "quota_profile": "gold"
            }"#,
            1,
        )];
        let (snap, stats) = build_snapshot("/aisix", &entries);

        // YELLOW: the row loads and serves with the unknown field ignored.
        assert_eq!(stats.accepted, 1, "rejections: {:?}", stats.rejections);
        assert_eq!(stats.schema_rejected, 0);
        assert_eq!(stats.parse_rejected, 0);
        assert!(stats.rejections.is_empty());
        assert_eq!(snap.apikeys.len(), 1);
        let entry = snap.apikeys.get_by_id("k-forward").unwrap();
        assert_eq!(entry.value.allowed_models, vec!["my-gpt4"]);

        // ...and the ignored field is reported, aggregated per
        // (kind, field path) with a row count.
        assert_eq!(
            stats.partially_compatible,
            vec![PartialCompatEntry {
                kind: "api_keys".into(),
                field: "quota_profile".into(),
                count: 1,
            }]
        );
    }

    #[test]
    fn nested_unknown_field_reports_dotted_path() {
        let entries = vec![raw(
            "/aisix/api_keys/k-nested",
            br#"{
                "key_hash": "1460db1b6902f8b1fc2a40d9381a24d0fd22c3bc1b2c6f999c521da73776fbe0",
                "allowed_models": ["m"],
                "rate_limit": {"rpm": 60, "burst": 10}
            }"#,
            1,
        )];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1, "rejections: {:?}", stats.rejections);
        assert_eq!(
            stats.partially_compatible,
            vec![PartialCompatEntry {
                kind: "api_keys".into(),
                field: "rate_limit.burst".into(),
                count: 1,
            }]
        );
    }

    #[test]
    fn unknown_enum_value_stays_incompatible() {
        // A value this build cannot interpret has no lenient fallback:
        // a routing strategy from a newer CP is RED, not YELLOW.
        let entries = vec![raw(
            "/aisix/models/m-newstrat",
            br#"{"display_name":"r","routing":{"strategy":"quantum","targets":[{"model":"a"}]}}"#,
            1,
        )];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 0);
        assert_eq!(stats.schema_rejected, 1);
        assert_eq!(stats.rejections[0].kind, RejectionKind::SchemaFailed);
        assert!(stats.partially_compatible.is_empty());
    }

    #[test]
    fn partial_compat_aggregates_across_rows_of_a_kind() {
        let doc = br#"{
            "key_hash": "1460db1b6902f8b1fc2a40d9381a24d0fd22c3bc1b2c6f999c521da73776fbe0",
            "allowed_models": ["m"],
            "quota_profile": "gold"
        }"#;
        let entries = vec![
            raw("/aisix/api_keys/k-1", doc, 1),
            raw("/aisix/api_keys/k-2", doc, 2),
        ];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 2);
        assert_eq!(
            stats.partially_compatible,
            vec![PartialCompatEntry {
                kind: "api_keys".into(),
                field: "quota_profile".into(),
                count: 2,
            }]
        );
        // The per-row form keeps one record per etcd key.
        assert_eq!(stats.partial_rows.len(), 2);
        assert_eq!(stats.partial_rows[0].key, "/aisix/api_keys/k-1");
        assert_eq!(stats.partial_rows[1].key, "/aisix/api_keys/k-2");
    }

    #[test]
    fn guardrail_attachment_in_cp_projection_shape_is_fully_compatible() {
        // The managed control plane writes `env_id` on every attachment
        // document (its own tenancy scoping; the gateway does not read
        // it). The field is declared on the model as known-and-ignored,
        // so a same-version managed fleet reports ZERO partially
        // compatible rows — a standing false version-skew alarm here
        // would train operators to ignore the YELLOW signal entirely.
        let entries = vec![raw(
            "/aisix/guardrail_attachments/ga-1",
            br#"{
                "guardrail_id": "11111111-1111-1111-1111-111111111111",
                "scope_type": "env",
                "scope_id": null,
                "priority": 0,
                "enabled": true,
                "env_id": "22222222-2222-2222-2222-222222222222"
            }"#,
            1,
        )];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1, "rejections: {:?}", stats.rejections);
        assert_eq!(snap.guardrail_attachments.len(), 1);
        assert!(
            stats.partially_compatible.is_empty(),
            "env_id is a registered cross-plane field, not version skew: {:?}",
            stats.partially_compatible
        );
    }

    #[test]
    fn model_dead_knob_strips_and_reports_inapplicable() {
        // A routing group with a top-level `retries` (dead — the group
        // slot is routing.retries) loads with the field stripped and
        // reports it via the partial-compat channel.
        let entries = vec![raw(
            "/aisix/models/m-dead",
            br#"{
                "display_name": "grp",
                "routing": {"targets": [{"model": "m"}]},
                "retries": 3
            }"#,
            1,
        )];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1, "rejections: {:?}", stats.rejections);
        let m = snap.models.get_by_name("grp").expect("row loaded");
        assert!(m.value.retries.is_none(), "dead knob stripped from struct");
        assert_eq!(
            stats.partial_rows[0].fields,
            vec!["inapplicable:retries".to_string()]
        );
    }

    #[test]
    fn model_unknown_and_inapplicable_fields_merge_into_one_row() {
        // A row carrying BOTH an unknown field (future CP) and a dead
        // knob must report BOTH under ONE key — two rows would let the
        // supervisor's key-addressed retained map drop a half (M1).
        let entries = vec![raw(
            "/aisix/models/m-both",
            br#"{
                "display_name": "grp2",
                "routing": {"targets": [{"model": "m"}]},
                "retries": 3,
                "zz_future_field": true
            }"#,
            1,
        )];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1, "rejections: {:?}", stats.rejections);
        let rows: Vec<_> = stats
            .partial_rows
            .iter()
            .filter(|r| r.key == "/aisix/models/m-both")
            .collect();
        assert_eq!(rows.len(), 1, "one row per key: {:?}", stats.partial_rows);
        assert_eq!(
            rows[0].fields,
            vec![
                "inapplicable:retries".to_string(),
                "zz_future_field".to_string()
            ],
            "both signals present and sorted"
        );
    }

    #[test]
    fn per_row_unknown_field_report_is_capped_with_a_visible_sentinel() {
        // One document can carry arbitrarily many unknown fields with
        // arbitrary-length names; everything captured flows into logs,
        // the retained map, the status JSON and the heartbeat body.
        let mut doc = serde_json::json!({
            "key_hash": "1460db1b6902f8b1fc2a40d9381a24d0fd22c3bc1b2c6f999c521da73776fbe0",
            "allowed_models": ["m"]
        });
        for i in 0..200 {
            doc.as_object_mut()
                .unwrap()
                .insert(format!("unknown_field_{i:03}"), serde_json::json!(1));
        }
        let entries = vec![raw(
            "/aisix/api_keys/k-flood",
            doc.to_string().as_bytes(),
            1,
        )];
        let (_snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1);
        let fields = &stats.partial_rows[0].fields;
        assert_eq!(fields.len(), 65, "64 fields + the truncation sentinel");
        assert_eq!(fields.last().map(String::as_str), Some("...truncated"));
    }

    // ---- renamed-field dual acceptance ----
    //
    // provider_key `secret`→`api_key` and mcp_server / a2a_agent
    // `display_name`→`name`: documents written under the former names
    // (existing etcd data, current control-plane writes) and documents
    // written under the canonical names must BOTH pass this loader's
    // schema gate and serde parse. A regression here silently drops
    // resources from the snapshot in managed mode.

    #[test]
    fn provider_key_accepts_both_credential_spellings() {
        let entries = vec![
            raw(
                "/aisix/provider_keys/pk-former",
                br#"{"display_name":"pk-former","secret":"sk-x"}"#,
                1,
            ),
            raw(
                "/aisix/provider_keys/pk-canonical",
                br#"{"display_name":"pk-canonical","api_key":"sk-x"}"#,
                2,
            ),
        ];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 2, "rejections: {:?}", stats.rejections);
        assert_eq!(snap.provider_keys.len(), 2);
        // Both spellings land on the same typed field.
        for id in ["pk-former", "pk-canonical"] {
            let entry = snap.provider_keys.get_by_id(id).unwrap();
            assert_eq!(entry.value.api_key, "sk-x");
        }
    }

    #[test]
    fn provider_key_with_both_spellings_is_rejected_as_parse_failure() {
        // The schema's `anyOf` admits a document carrying both spellings;
        // serde then rejects it as a duplicate field (both names map to
        // the same field), so the ambiguous row is skipped — not loaded
        // with one value silently winning — and the batch continues.
        let entries = vec![
            raw(
                "/aisix/provider_keys/pk-ambiguous",
                br#"{"display_name":"pk-a","api_key":"sk-new","secret":"sk-old"}"#,
                1,
            ),
            raw(
                "/aisix/provider_keys/pk-good",
                br#"{"display_name":"pk-good","api_key":"sk-x"}"#,
                2,
            ),
        ];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.schema_rejected, 0);
        assert_eq!(stats.parse_rejected, 1);
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.rejections.len(), 1);
        assert_eq!(stats.rejections[0].kind, RejectionKind::ParseFailed);
        assert!(snap.provider_keys.get_by_id("pk-good").is_some());
        assert!(snap.provider_keys.get_by_id("pk-ambiguous").is_none());
    }

    #[test]
    fn mcp_server_and_a2a_agent_accept_both_label_spellings() {
        let entries = vec![
            raw(
                "/aisix/mcp_servers/mcp-former",
                br#"{"display_name":"gh-former","url":"https://x/mcp"}"#,
                1,
            ),
            raw(
                "/aisix/mcp_servers/mcp-canonical",
                br#"{"name":"gh-canonical","url":"https://x/mcp"}"#,
                2,
            ),
            raw(
                "/aisix/a2a_agents/a2a-former",
                br#"{"display_name":"agent-former","url":"https://x/a2a"}"#,
                3,
            ),
            raw(
                "/aisix/a2a_agents/a2a-canonical",
                br#"{"name":"agent-canonical","url":"https://x/a2a"}"#,
                4,
            ),
        ];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 4, "rejections: {:?}", stats.rejections);
        // The name index is fed by the renamed field for both spellings.
        assert!(snap.mcp_servers.get_by_name("gh-former").is_some());
        assert!(snap.mcp_servers.get_by_name("gh-canonical").is_some());
        assert!(snap.a2a_agents.get_by_name("agent-former").is_some());
        assert!(snap.a2a_agents.get_by_name("agent-canonical").is_some());
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
        assert_eq!(
            entry.value.scope,
            Some(aisix_core::models::PolicyScope::Team)
        );
        assert_eq!(entry.value.scope_ref.as_deref(), Some("team-uuid-1"));
        assert_eq!(entry.value.max_requests, Some(100));
    }

    #[test]
    fn conditional_rate_limit_policy_loads_into_snapshot() {
        let entries = vec![raw(
            "/aisix/rate_limit_policies/rlp-2",
            br#"{
                "name": "premium-family",
                "conditions": [
                    { "dimension": "team", "operator": "in", "value": ["t-1"] },
                    { "logic": "or", "children": [
                        { "dimension": "model_name", "operator": "~~", "value": "^gpt-4" },
                        { "dimension": "provider", "operator": "==", "value": "anthropic" }
                    ]}
                ],
                "group_by": ["member"],
                "limits": { "rpm": 20 }
            }"#,
            6,
        )];
        let (snap, stats) = build_snapshot("/aisix", &entries);
        assert_eq!(stats.accepted, 1);
        let entry = snap.rate_limit_policies.get_by_id("rlp-2").unwrap();
        assert!(entry.value.is_conditional());
    }

    #[test]
    fn semantically_invalid_policy_is_rejected_like_a_schema_failure() {
        // Passes the JSON Schema (shape is fine) but fails the semantic
        // gate: the regex does not compile. The row must contribute to
        // schema_rejected — NOT accepted — so `Supervisor::apply_put`
        // (which gates success on `stats.accepted`) retains the
        // last-good value and surfaces the rejection; and no
        // partial-compat state may be recorded for a row that does not
        // serve.
        let bad = raw(
            "/aisix/rate_limit_policies/rlp-bad",
            br#"{
                "name": "bad-regex",
                "conditions": [
                    { "dimension": "model_name", "operator": "~~", "value": "(unclosed" }
                ],
                "limits": { "rpm": 5 },
                "future_field": true
            }"#,
            7,
        );
        let (snap, stats) = build_snapshot("/aisix", std::slice::from_ref(&bad));
        assert_eq!(stats.accepted, 0);
        assert_eq!(stats.schema_rejected, 1);
        assert_eq!(snap.rate_limit_policies.len(), 0);
        assert!(
            stats.partial_rows.is_empty(),
            "a rejected row must not leave partial-compat state behind"
        );
        let rej = &stats.rejections[0];
        assert_eq!(rej.key, "/aisix/rate_limit_policies/rlp-bad");
        assert_eq!(rej.kind, RejectionKind::SchemaFailed);
        assert!(rej.error.contains("does not compile"), "{}", rej.error);
    }
}
