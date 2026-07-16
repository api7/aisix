//! Load-observability state for the data plane's configuration source.
//!
//! Answers one operator question — *did my configuration take effect, and
//! if not, why?* — for both load paths the gateway supports:
//!
//! - the **etcd** watch source (managed mode), and
//! - the standalone **file** source (`resources_file`).
//!
//! [`ConfigStatus`] is a cheap-to-clone shared handle the load paths update
//! as they observe and apply snapshots. It is read by the non-admin
//! metrics/status listener to serve `GET /status/config`, `GET /status/ready`,
//! and the `aisix_config_*` Prometheus series.
//!
//! Nothing here does new *tracking* — the etcd supervisor and the file source
//! already collect rejected entries and revisions. This module gives that
//! internal state one canonical, source-agnostic shape and derives the
//! reported [`ConfigState`] from it.
//!
//! ## Reported state
//!
//! [`ConfigState`] is server-derived from the last observed and applied
//! snapshots (see [`ConfigStatusInner::derive_state`]).
//!
//! ## Hash definition
//!
//! `source_hash` and `config_hash` are deterministic and reproducible by a
//! caller that knows what it wrote:
//!
//! - **etcd**: `sha256` over the entries, each rendered as
//!   `key '\0' canonical_json_value '\n'`, concatenated in ascending key
//!   order. `canonical_json_value` recursively sorts object keys and drops
//!   insignificant whitespace ([`canonical_json`]); a value that is not JSON
//!   is hashed as its raw bytes. `source_hash` covers every entry in the last
//!   observed full snapshot; `config_hash` covers only the accepted subset
//!   (the entries actually served). When a full snapshot is applied with no
//!   rejections the two are equal. A resource rejected on a *live watch event*
//!   is surfaced via `rejected[]` and does not enter `source_hash` until the
//!   next full resync (the watch delta never carries the bad bytes into the
//!   served entry map) — so `source_hash == config_hash` can hold while the
//!   state is `degraded`; `rejected[]` is the authoritative partial-rejection
//!   signal, not a hash diff.
//! - **file**: `sha256` over the raw file bytes. On a clean load the applied
//!   `config_hash` equals `source_hash` (the whole file is applied); on a
//!   rejected reload the applied hash stays at the last-good file's hash.

use chrono::{DateTime, SecondsFormat, Utc};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Which source the data plane reads configuration from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Etcd,
    File,
}

impl SourceKind {
    fn is_etcd(self) -> bool {
        matches!(self, SourceKind::Etcd)
    }
}

/// Server-derived configuration state reported by `GET /status/config`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigState {
    /// Applied config matches the latest observed snapshot and nothing was
    /// rejected.
    Synced,
    /// Applied config is serving, but the latest snapshot carried entries the
    /// gateway rejected.
    Degraded,
    /// The latest observed snapshot was wholly rejected; the gateway is
    /// serving the last-good snapshot (or nothing usable from the latest).
    OutOfSync,
    /// A valid configuration was applied but it holds zero resources.
    Empty,
    /// No valid configuration has ever been applied this boot.
    NeverLoaded,
}

/// Coarse reason bucket for `aisix_config_reload_failures_total{reason}`.
///
/// Deliberately low-cardinality — the per-resource detail lives in
/// `rejected[]`, this is the aggregate "why did a reload not fully succeed".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadReason {
    /// The source could not be fetched (etcd unreachable, file unreadable).
    Fetch,
    /// The source was fetched but could not be parsed (non-JSON value, bad
    /// YAML).
    Parse,
    /// The source parsed but a resource failed schema / shape / reference
    /// validation.
    Validate,
}

impl ReloadReason {
    pub fn as_str(self) -> &'static str {
        match self {
            ReloadReason::Fetch => "fetch",
            ReloadReason::Parse => "parse",
            ReloadReason::Validate => "validate",
        }
    }

    /// Map a rejected entry's `last_error_kind` (the loader's `RejectionKind`
    /// rendered snake_case, or a file classification) to a coarse reason.
    ///
    /// `non_json` is the only source-format ("parse") kind; every other
    /// resource-level rejection is a shape/identity ("validate") failure.
    pub fn from_error_kind(last_error_kind: &str) -> Self {
        match last_error_kind {
            "non_json" => ReloadReason::Parse,
            _ => ReloadReason::Validate,
        }
    }
}

/// A resource the gateway rejected, as reported on the wire. Field names
/// match the control plane's `rejected_resources` surface so an operator sees
/// the same vocabulary on both planes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RejectedResource {
    /// Plural resource kind (`models`, `provider_keys`, …); empty when the
    /// source identifier could not be parsed into `kind`/`id`.
    pub resource_kind: String,
    /// Resource id; empty when unparseable.
    pub resource_id: String,
    /// Snake-case failure kind: `bad_key | non_json | schema_failed |
    /// parse_failed | unknown_kind` for etcd; a file classification otherwise.
    pub last_error_kind: String,
    /// Human-readable error message. Schema-validation messages are
    /// credential-masked at the schema layer (instance values redacted before
    /// they reach this buffer); parse / decode / key messages are positional
    /// and carry no instance values.
    pub last_error: String,
    /// RFC3339 UTC timestamp the rejection was first observed this boot.
    pub first_seen_at: String,
    /// RFC3339 UTC timestamp the rejection was most recently observed.
    pub last_seen_at: String,
}

/// One rejected entry handed to [`ConfigStatus`] by a load path. `identity`
/// (the etcd key or file scope) is the stable merge key used to preserve
/// `first_seen_at` across reloads; it is never serialized.
#[derive(Debug, Clone)]
pub struct IncomingRejection {
    pub identity: String,
    pub resource_kind: String,
    pub resource_id: String,
    pub last_error_kind: String,
    pub last_error: String,
    pub seen_at: DateTime<Utc>,
}

/// The result of a snapshot the gateway actually applied (served).
#[derive(Debug, Clone)]
pub struct AppliedSnapshot {
    /// Hash of the accepted (served) entry set.
    pub config_hash: String,
    /// etcd revision the applied snapshot reflects; `None` in file mode.
    pub revision: Option<i64>,
    /// Per-kind counts of served resources.
    pub resource_counts: BTreeMap<String, usize>,
}

/// A completed load observation handed to [`ConfigStatus::record_load`].
#[derive(Debug, Clone)]
pub struct LoadObservation {
    /// Hash of the full raw snapshot observed from the source.
    pub source_hash: String,
    /// etcd revision the observed snapshot reflects; `None` in file mode.
    pub observed_revision: Option<i64>,
    /// The applied snapshot, when this load produced/kept a served snapshot.
    /// `None` only when a reload was wholly rejected and the last-good
    /// snapshot is retained (file reload failure) — the caller then sets
    /// [`Self::wholly_rejected`].
    pub applied: Option<AppliedSnapshot>,
    /// Rejected entries observed in this snapshot.
    pub rejected: Vec<IncomingRejection>,
    /// Whether this load counts as a config reload for
    /// `aisix_config_reloads_total` (full (re)syncs and file loads do;
    /// incremental etcd events do not).
    pub is_reload: bool,
    /// True when the latest observed snapshot was rejected as a whole and the
    /// gateway kept serving a previous snapshot (file reload failure).
    pub wholly_rejected: bool,
}

/// Cheap-to-clone shared handle to the config load state.
#[derive(Debug, Clone)]
pub struct ConfigStatus {
    inner: Arc<Mutex<ConfigStatusInner>>,
}

#[derive(Debug)]
struct RetainedRejection {
    resource_kind: String,
    resource_id: String,
    last_error_kind: String,
    last_error: String,
    first_seen_at: DateTime<Utc>,
    last_seen_at: DateTime<Utc>,
}

#[derive(Debug)]
struct ConfigStatusInner {
    source_kind: SourceKind,

    // Observed (latest raw snapshot seen from the source).
    connected: bool,
    observed_revision: Option<i64>,
    source_hash: Option<String>,
    observed_at: Option<DateTime<Utc>>,

    // Applied (last snapshot actually served).
    ever_applied: bool,
    config_hash: Option<String>,
    applied_revision: Option<i64>,
    applied_at: Option<DateTime<Utc>>,
    apply_seq: u64,
    resource_counts: BTreeMap<String, usize>,

    // Latest observed snapshot rejected as a whole (last-good retained).
    latest_wholly_rejected: bool,

    // Reload signals.
    last_reload_successful: bool,
    last_reload_at: Option<DateTime<Utc>>,
    last_reload_success_at: Option<DateTime<Utc>>,

    // Sticky failure (until next boot).
    last_failure: Option<StickyFailure>,

    // Retained rejections, keyed by source identity to keep first_seen stable.
    rejected: BTreeMap<String, RetainedRejection>,

    // Metric counters.
    reloads_total: u64,
    reload_failures: BTreeMap<&'static str, u64>,
}

#[derive(Debug, Clone)]
struct StickyFailure {
    at: DateTime<Utc>,
    last_error_kind: String,
    last_error: String,
}

impl ConfigStatus {
    /// Construct a status handle for a source. Starts in `never_loaded`.
    pub fn new(source_kind: SourceKind) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ConfigStatusInner {
                source_kind,
                connected: false,
                observed_revision: None,
                source_hash: None,
                observed_at: None,
                ever_applied: false,
                config_hash: None,
                applied_revision: None,
                applied_at: None,
                apply_seq: 0,
                resource_counts: BTreeMap::new(),
                latest_wholly_rejected: false,
                last_reload_successful: false,
                last_reload_at: None,
                last_reload_success_at: None,
                last_failure: None,
                rejected: BTreeMap::new(),
                reloads_total: 0,
                reload_failures: BTreeMap::new(),
            })),
        }
    }

    /// Record a completed load observation. Idempotent enough to be called
    /// on every apply: `apply_seq` and `applied_at` only advance when the
    /// applied `config_hash` changes.
    pub fn record_load(&self, obs: LoadObservation) {
        let now = Utc::now();
        let mut inner = self.inner.lock().unwrap();

        inner.connected = true;
        inner.observed_at = Some(now);
        inner.source_hash = Some(obs.source_hash);
        inner.observed_revision = obs.observed_revision;
        inner.latest_wholly_rejected = obs.wholly_rejected;

        if let Some(applied) = obs.applied {
            let changed = inner.config_hash.as_deref() != Some(applied.config_hash.as_str());
            if changed || !inner.ever_applied {
                inner.apply_seq += 1;
                inner.applied_at = Some(now);
            }
            inner.ever_applied = true;
            inner.config_hash = Some(applied.config_hash);
            inner.applied_revision = applied.revision;
            inner.resource_counts = applied.resource_counts;
        }

        // Merge rejections, preserving first_seen for identities still present.
        let mut merged: BTreeMap<String, RetainedRejection> = BTreeMap::new();
        for r in obs.rejected {
            let first_seen_at = inner
                .rejected
                .get(&r.identity)
                .map(|prev| prev.first_seen_at)
                .unwrap_or(r.seen_at);
            merged.insert(
                r.identity,
                RetainedRejection {
                    resource_kind: r.resource_kind,
                    resource_id: r.resource_id,
                    last_error_kind: r.last_error_kind,
                    last_error: r.last_error,
                    first_seen_at,
                    last_seen_at: r.seen_at,
                },
            );
        }
        inner.rejected = merged;

        let clean = inner.rejected.is_empty();
        inner.last_reload_successful = clean;
        inner.last_reload_at = Some(now);
        if clean {
            inner.last_reload_success_at = Some(now);
        } else if let Some((kind, err)) = inner
            .rejected
            .values()
            .next()
            .map(|r| (r.last_error_kind.clone(), r.last_error.clone()))
        {
            inner.last_failure = Some(StickyFailure {
                at: now,
                last_error_kind: kind,
                last_error: err,
            });
        }

        if obs.is_reload {
            inner.reloads_total += 1;
            // One increment per reason category present in this reload.
            let mut reasons: BTreeMap<&'static str, ()> = BTreeMap::new();
            for r in inner.rejected.values() {
                reasons.insert(
                    ReloadReason::from_error_kind(&r.last_error_kind).as_str(),
                    (),
                );
            }
            for reason in reasons.keys() {
                *inner.reload_failures.entry(reason).or_insert(0) += 1;
            }
        }
    }

    /// Record that the source became unreachable (etcd connect / load failed).
    /// Counts a fetch-reason reload failure and marks the source disconnected;
    /// leaves the last-good applied state intact.
    pub fn record_fetch_failure(&self) {
        let now = Utc::now();
        let mut inner = self.inner.lock().unwrap();
        inner.connected = false;
        inner.last_reload_successful = false;
        inner.last_reload_at = Some(now);
        inner.reloads_total += 1;
        *inner
            .reload_failures
            .entry(ReloadReason::Fetch.as_str())
            .or_insert(0) += 1;
        inner.last_failure = Some(StickyFailure {
            at: now,
            last_error_kind: "fetch".to_string(),
            last_error: "configuration source unreachable".to_string(),
        });
    }

    /// Whether a valid configuration has ever been applied. Gates
    /// `GET /status/ready`.
    pub fn is_ready(&self) -> bool {
        self.inner.lock().unwrap().ever_applied
    }

    /// The hash of the last applied (served) config snapshot, or `None`
    /// when no snapshot has been applied yet this boot. A cheap targeted
    /// read for callers that need only the hash — the heartbeat's
    /// per-node config-verification field — without building the full
    /// [`Self::view`] / [`Self::metrics`] snapshot.
    pub fn applied_config_hash(&self) -> Option<String> {
        self.inner.lock().unwrap().config_hash.clone()
    }

    /// Point-in-time JSON view for `GET /status/config`.
    pub fn view(&self) -> ConfigStatusView {
        self.inner.lock().unwrap().view()
    }

    /// Point-in-time numeric view for the `aisix_config_*` Prometheus series.
    pub fn metrics(&self) -> ConfigMetricsView {
        self.inner.lock().unwrap().metrics()
    }
}

impl ConfigStatusInner {
    fn applied_total(&self) -> usize {
        self.resource_counts.values().sum()
    }

    fn derive_state(&self) -> ConfigState {
        if !self.ever_applied {
            return ConfigState::NeverLoaded;
        }
        if self.latest_wholly_rejected {
            return ConfigState::OutOfSync;
        }
        let total = self.applied_total();
        if !self.rejected.is_empty() {
            // A whole-snapshot rejection that stored an empty snapshot still
            // reads as out-of-sync; a partial rejection is degraded.
            if total == 0 {
                ConfigState::OutOfSync
            } else {
                ConfigState::Degraded
            }
        } else if total == 0 {
            ConfigState::Empty
        } else {
            ConfigState::Synced
        }
    }

    fn view(&self) -> ConfigStatusView {
        let etcd = self.source_kind.is_etcd();
        let source = SourceView {
            source_type: self.source_kind,
            connected: etcd.then_some(self.connected),
            observed_revision: if etcd { self.observed_revision } else { None },
            source_hash: self.source_hash.clone(),
            observed_at: self.observed_at.map(rfc3339),
        };
        let applied = if self.ever_applied {
            Some(AppliedView {
                applied_revision: if etcd { self.applied_revision } else { None },
                config_hash: self.config_hash.clone().unwrap_or_default(),
                apply_seq: self.apply_seq,
                applied_at: self.applied_at.map(rfc3339).unwrap_or_default(),
                resource_counts: self.resource_counts.clone(),
            })
        } else {
            None
        };
        let last_reload = self.last_reload_at.map(|at| LastReloadView {
            successful: self.last_reload_successful,
            at: rfc3339(at),
        });
        let last_failure = self.last_failure.as_ref().map(|f| FailureView {
            at: rfc3339(f.at),
            last_error_kind: f.last_error_kind.clone(),
            last_error: f.last_error.clone(),
        });
        let mut rejected: Vec<RejectedResource> = self
            .rejected
            .values()
            .map(|r| RejectedResource {
                resource_kind: r.resource_kind.clone(),
                resource_id: r.resource_id.clone(),
                last_error_kind: r.last_error_kind.clone(),
                last_error: r.last_error.clone(),
                first_seen_at: rfc3339(r.first_seen_at),
                last_seen_at: rfc3339(r.last_seen_at),
            })
            .collect();
        rejected.sort_by(|a, b| {
            (&a.resource_kind, &a.resource_id).cmp(&(&b.resource_kind, &b.resource_id))
        });
        ConfigStatusView {
            state: self.derive_state(),
            source,
            applied,
            last_reload,
            last_failure,
            rejected,
        }
    }

    fn metrics(&self) -> ConfigMetricsView {
        let etcd = self.source_kind.is_etcd();
        let mut rejected_by_kind: BTreeMap<String, usize> = BTreeMap::new();
        for r in self.rejected.values() {
            *rejected_by_kind.entry(r.resource_kind.clone()).or_insert(0) += 1;
        }
        ConfigMetricsView {
            source_kind: self.source_kind,
            last_reload_successful: self.last_reload_successful,
            last_reload_success_ts: self.last_reload_success_at.map(|t| t.timestamp()),
            reloads_total: self.reloads_total,
            reload_failures: self.reload_failures.iter().map(|(k, v)| (*k, *v)).collect(),
            rejected_by_kind,
            observed_revision: if etcd { self.observed_revision } else { None },
            applied_revision: if etcd { self.applied_revision } else { None },
            config_hash: self.config_hash.clone(),
            connected: etcd.then_some(self.connected),
        }
    }
}

/// `GET /status/config` response body.
#[derive(Debug, Clone, Serialize)]
pub struct ConfigStatusView {
    pub state: ConfigState,
    pub source: SourceView,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied: Option<AppliedView>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_reload: Option<LastReloadView>,
    /// `null` when no failure has occurred this boot.
    pub last_failure: Option<FailureView>,
    pub rejected: Vec<RejectedResource>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceView {
    #[serde(rename = "type")]
    pub source_type: SourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connected: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_revision: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AppliedView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied_revision: Option<i64>,
    pub config_hash: String,
    pub apply_seq: u64,
    pub applied_at: String,
    pub resource_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LastReloadView {
    pub successful: bool,
    pub at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FailureView {
    pub at: String,
    pub last_error_kind: String,
    pub last_error: String,
}

/// Numeric view consumed by the metrics exporter.
#[derive(Debug, Clone)]
pub struct ConfigMetricsView {
    pub source_kind: SourceKind,
    pub last_reload_successful: bool,
    pub last_reload_success_ts: Option<i64>,
    pub reloads_total: u64,
    pub reload_failures: BTreeMap<&'static str, u64>,
    pub rejected_by_kind: BTreeMap<String, usize>,
    pub observed_revision: Option<i64>,
    pub applied_revision: Option<i64>,
    pub config_hash: Option<String>,
    pub connected: Option<bool>,
}

fn rfc3339(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Secs, true)
}

/// Hash an etcd entry set: `sha256` over `key '\0' canonical_value '\n'` for
/// each entry, in ascending key order. See the module docs for the exact
/// definition. `entries` is `(key, raw_value_bytes)`.
pub fn hash_entries<'a, I>(entries: I) -> String
where
    I: IntoIterator<Item = (&'a str, &'a [u8])>,
{
    let mut sorted: Vec<(&str, &[u8])> = entries.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut hasher = Sha256::new();
    for (key, value) in sorted {
        hasher.update(key.as_bytes());
        hasher.update([0u8]);
        match serde_json::from_slice::<serde_json::Value>(value) {
            Ok(v) => hasher.update(canonical_json(&v).as_bytes()),
            // Not JSON (rejected as non_json): hash the raw bytes so the
            // observed hash still changes deterministically with the input.
            Err(_) => hasher.update(value),
        }
        hasher.update([b'\n']);
    }
    hex(hasher.finalize().as_slice())
}

/// Hash raw file bytes: `sha256` hex.
pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(hasher.finalize().as_slice())
}

/// Serialize a JSON value with object keys sorted recursively and no
/// insignificant whitespace, so two structurally-equal documents hash the
/// same regardless of key order or spacing.
fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            out.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                // A JSON string key is itself canonical via serde_json.
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                write_canonical(&map[*k], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        // Scalars round-trip deterministically through serde_json.
        other => out.push_str(&serde_json::to_string(other).unwrap_or_default()),
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0x0f) as u32, 16).unwrap());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn incoming(
        identity: &str,
        kind: &str,
        id: &str,
        error_kind: &str,
        error: &str,
    ) -> IncomingRejection {
        IncomingRejection {
            identity: identity.to_string(),
            resource_kind: kind.to_string(),
            resource_id: id.to_string(),
            last_error_kind: error_kind.to_string(),
            last_error: error.to_string(),
            seen_at: Utc::now(),
        }
    }

    fn applied(hash: &str, counts: &[(&str, usize)]) -> AppliedSnapshot {
        AppliedSnapshot {
            config_hash: hash.to_string(),
            revision: Some(7),
            resource_counts: counts.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    #[test]
    fn never_loaded_before_any_load() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        assert_eq!(cs.view().state, ConfigState::NeverLoaded);
        assert!(!cs.is_ready());
    }

    #[test]
    fn synced_when_clean_load_with_resources() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        cs.record_load(LoadObservation {
            source_hash: "h".into(),
            observed_revision: Some(7),
            applied: Some(applied("h", &[("models", 2)])),
            rejected: vec![],
            is_reload: true,
            wholly_rejected: false,
        });
        let v = cs.view();
        assert_eq!(v.state, ConfigState::Synced);
        assert!(cs.is_ready());
        let applied = v.applied.unwrap();
        assert_eq!(applied.applied_revision, Some(7));
        assert_eq!(applied.resource_counts.get("models"), Some(&2));
        assert_eq!(applied.apply_seq, 1);
        assert!(v.last_failure.is_none());
    }

    #[test]
    fn applied_config_hash_is_none_until_applied_then_tracks_latest() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        assert_eq!(cs.applied_config_hash(), None);
        // Distinct source vs applied hashes: a regression reading the
        // observed source_hash instead of the served config_hash would
        // pass with equal values, so keep them apart.
        cs.record_load(LoadObservation {
            source_hash: "src1".into(),
            observed_revision: Some(1),
            applied: Some(applied("applied1", &[("models", 1)])),
            rejected: vec![],
            is_reload: true,
            wholly_rejected: false,
        });
        // Must be the APPLIED (served) hash, never the observed source_hash.
        assert_eq!(cs.applied_config_hash().as_deref(), Some("applied1"));
        // A later apply carrying a new hash is reflected.
        cs.record_load(LoadObservation {
            source_hash: "src2".into(),
            observed_revision: Some(2),
            applied: Some(applied("applied2", &[("models", 1)])),
            rejected: vec![],
            is_reload: true,
            wholly_rejected: false,
        });
        assert_eq!(cs.applied_config_hash().as_deref(), Some("applied2"));
        // A wholly-rejected reload keeps the last-good applied hash even
        // as source_hash advances — we report what we serve, not what we
        // observed.
        cs.record_load(LoadObservation {
            source_hash: "src3".into(),
            observed_revision: Some(3),
            applied: None,
            rejected: vec![],
            is_reload: true,
            wholly_rejected: true,
        });
        assert_eq!(cs.applied_config_hash().as_deref(), Some("applied2"));
    }

    #[test]
    fn empty_when_clean_load_with_zero_resources() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        cs.record_load(LoadObservation {
            source_hash: "h".into(),
            observed_revision: Some(3),
            applied: Some(applied("h", &[])),
            rejected: vec![],
            is_reload: true,
            wholly_rejected: false,
        });
        assert_eq!(cs.view().state, ConfigState::Empty);
        assert!(cs.is_ready());
    }

    #[test]
    fn degraded_when_partial_rejection() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        cs.record_load(LoadObservation {
            source_hash: "src".into(),
            observed_revision: Some(9),
            applied: Some(applied("applied", &[("models", 1)])),
            rejected: vec![incoming(
                "/aisix/models/bad",
                "models",
                "bad",
                "schema_failed",
                "schema validation failed at `/display_name`",
            )],
            is_reload: true,
            wholly_rejected: false,
        });
        let v = cs.view();
        assert_eq!(v.state, ConfigState::Degraded);
        assert_eq!(v.rejected.len(), 1);
        assert_eq!(v.rejected[0].resource_kind, "models");
        assert_eq!(v.rejected[0].last_error_kind, "schema_failed");
        assert!(v.last_failure.is_some());
    }

    #[test]
    fn out_of_sync_when_whole_snapshot_rejected_and_last_good_retained() {
        let cs = ConfigStatus::new(SourceKind::File);
        // First a clean load establishes last-good.
        cs.record_load(LoadObservation {
            source_hash: "good".into(),
            observed_revision: None,
            applied: Some(applied("good", &[("models", 1)])),
            rejected: vec![],
            is_reload: true,
            wholly_rejected: false,
        });
        // A reload that fails wholesale keeps last-good but flags wholly-rejected.
        cs.record_load(LoadObservation {
            source_hash: "bad".into(),
            observed_revision: None,
            applied: None,
            rejected: vec![incoming(
                "models[0] (\"x\")",
                "models",
                "x",
                "schema_failed",
                "boom",
            )],
            is_reload: true,
            wholly_rejected: true,
        });
        assert_eq!(cs.view().state, ConfigState::OutOfSync);
    }

    #[test]
    fn out_of_sync_when_etcd_resync_wholly_rejected_to_empty() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        cs.record_load(LoadObservation {
            source_hash: "src".into(),
            observed_revision: Some(4),
            applied: Some(applied("empty", &[])), // zero accepted
            rejected: vec![incoming(
                "/aisix/models/bad",
                "models",
                "bad",
                "non_json",
                "not json",
            )],
            is_reload: true,
            wholly_rejected: false,
        });
        assert_eq!(cs.view().state, ConfigState::OutOfSync);
    }

    #[test]
    fn apply_seq_advances_only_on_config_change() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        let obs = |hash: &str| LoadObservation {
            source_hash: hash.into(),
            observed_revision: Some(1),
            applied: Some(applied(hash, &[("models", 1)])),
            rejected: vec![],
            is_reload: false,
            wholly_rejected: false,
        };
        cs.record_load(obs("a"));
        assert_eq!(cs.view().applied.unwrap().apply_seq, 1);
        cs.record_load(obs("a")); // unchanged
        assert_eq!(cs.view().applied.unwrap().apply_seq, 1);
        cs.record_load(obs("b")); // changed
        assert_eq!(cs.view().applied.unwrap().apply_seq, 2);
    }

    #[test]
    fn last_failure_is_sticky_across_a_later_clean_reload() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        cs.record_load(LoadObservation {
            source_hash: "s1".into(),
            observed_revision: Some(1),
            applied: Some(applied("a1", &[("models", 1)])),
            rejected: vec![incoming(
                "/aisix/models/bad",
                "models",
                "bad",
                "schema_failed",
                "boom",
            )],
            is_reload: true,
            wholly_rejected: false,
        });
        assert!(cs.view().last_failure.is_some());
        // Clean reload: last_reload flips to successful but last_failure stays.
        cs.record_load(LoadObservation {
            source_hash: "s2".into(),
            observed_revision: Some(2),
            applied: Some(applied("a2", &[("models", 1)])),
            rejected: vec![],
            is_reload: true,
            wholly_rejected: false,
        });
        let v = cs.view();
        assert_eq!(v.state, ConfigState::Synced);
        assert!(v.last_reload.unwrap().successful);
        assert!(v.last_failure.is_some(), "last_failure must be sticky");
        assert!(v.rejected.is_empty());
    }

    #[test]
    fn first_seen_is_preserved_across_reloads() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        let mut r = incoming(
            "/aisix/models/bad",
            "models",
            "bad",
            "schema_failed",
            "boom",
        );
        r.seen_at = "2026-07-14T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        cs.record_load(LoadObservation {
            source_hash: "s".into(),
            observed_revision: Some(1),
            applied: Some(applied("a", &[])),
            rejected: vec![r],
            is_reload: true,
            wholly_rejected: false,
        });
        let first = cs.view().rejected[0].first_seen_at.clone();

        let mut r2 = incoming(
            "/aisix/models/bad",
            "models",
            "bad",
            "schema_failed",
            "boom again",
        );
        r2.seen_at = "2026-07-14T01:00:00Z".parse::<DateTime<Utc>>().unwrap();
        cs.record_load(LoadObservation {
            source_hash: "s".into(),
            observed_revision: Some(2),
            applied: Some(applied("a", &[])),
            rejected: vec![r2],
            is_reload: true,
            wholly_rejected: false,
        });
        let v = cs.view();
        assert_eq!(
            v.rejected[0].first_seen_at, first,
            "first_seen must be stable"
        );
        assert_eq!(v.rejected[0].last_seen_at, "2026-07-14T01:00:00Z");
    }

    #[test]
    fn file_mode_omits_etcd_only_fields() {
        let cs = ConfigStatus::new(SourceKind::File);
        cs.record_load(LoadObservation {
            source_hash: "f".into(),
            observed_revision: None,
            applied: Some(AppliedSnapshot {
                config_hash: "f".into(),
                revision: None,
                resource_counts: [("models".to_string(), 1)].into_iter().collect(),
            }),
            rejected: vec![],
            is_reload: true,
            wholly_rejected: false,
        });
        let json = serde_json::to_value(cs.view()).unwrap();
        assert_eq!(json["source"]["type"], "file");
        assert!(json["source"].get("connected").is_none());
        assert!(json["source"].get("observed_revision").is_none());
        assert!(json["applied"].get("applied_revision").is_none());
    }

    #[test]
    fn etcd_mode_includes_connected_and_revisions() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        cs.record_load(LoadObservation {
            source_hash: "e".into(),
            observed_revision: Some(11),
            applied: Some(applied("e", &[("models", 1)])),
            rejected: vec![],
            is_reload: true,
            wholly_rejected: false,
        });
        let json = serde_json::to_value(cs.view()).unwrap();
        assert_eq!(json["source"]["type"], "etcd");
        assert_eq!(json["source"]["connected"], true);
        assert_eq!(json["source"]["observed_revision"], 11);
        assert_eq!(json["applied"]["applied_revision"], 7);
    }

    #[test]
    fn reload_failure_counters_bucket_by_reason() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        cs.record_load(LoadObservation {
            source_hash: "s".into(),
            observed_revision: Some(1),
            applied: Some(applied("a", &[("models", 1)])),
            rejected: vec![
                incoming("/aisix/models/a", "models", "a", "non_json", "x"),
                incoming(
                    "/aisix/provider_keys/b",
                    "provider_keys",
                    "b",
                    "schema_failed",
                    "y",
                ),
            ],
            is_reload: true,
            wholly_rejected: false,
        });
        let m = cs.metrics();
        assert_eq!(m.reloads_total, 1);
        assert_eq!(m.reload_failures.get("parse"), Some(&1)); // non_json
        assert_eq!(m.reload_failures.get("validate"), Some(&1)); // schema_failed
        assert_eq!(m.rejected_by_kind.get("models"), Some(&1));
        assert_eq!(m.rejected_by_kind.get("provider_keys"), Some(&1));
    }

    #[test]
    fn fetch_failure_marks_disconnected_and_counts_fetch_reason() {
        let cs = ConfigStatus::new(SourceKind::Etcd);
        cs.record_fetch_failure();
        let m = cs.metrics();
        assert_eq!(m.connected, Some(false));
        assert_eq!(m.reload_failures.get("fetch"), Some(&1));
        // Still never_loaded — a fetch failure never applied anything.
        assert_eq!(cs.view().state, ConfigState::NeverLoaded);
    }

    #[test]
    fn hash_is_order_independent_over_keys_and_whitespace() {
        let a = hash_entries([
            ("/aisix/models/m1", br#"{"b":1,"a":2}"#.as_slice()),
            ("/aisix/models/m2", br#"{"x":  "y"}"#.as_slice()),
        ]);
        // Same entries, different insertion order + key order + whitespace.
        let b = hash_entries([
            ("/aisix/models/m2", br#"{"x":"y"}"#.as_slice()),
            ("/aisix/models/m1", br#"{ "a":2, "b":1 }"#.as_slice()),
        ]);
        assert_eq!(a, b, "hash must be canonical over key order and whitespace");
    }

    #[test]
    fn hash_changes_when_a_value_changes() {
        let a = hash_entries([("/aisix/models/m1", br#"{"a":1}"#.as_slice())]);
        let b = hash_entries([("/aisix/models/m1", br#"{"a":2}"#.as_slice())]);
        assert_ne!(a, b);
    }

    #[test]
    fn accepted_subset_hash_equals_source_hash_when_nothing_rejected() {
        let entries: [(&str, &[u8]); 2] = [
            ("/aisix/models/m1", br#"{"a":1}"#),
            ("/aisix/models/m2", br#"{"b":2}"#),
        ];
        let source = hash_entries(entries.iter().map(|(k, v)| (*k, *v)));
        let accepted = hash_entries(entries.iter().map(|(k, v)| (*k, *v)));
        assert_eq!(source, accepted);
    }

    #[test]
    fn non_json_value_still_hashes_deterministically() {
        let a = hash_entries([("/aisix/models/m1", b"not-json".as_slice())]);
        let b = hash_entries([("/aisix/models/m1", b"not-json".as_slice())]);
        assert_eq!(a, b);
        let c = hash_entries([("/aisix/models/m1", b"other".as_slice())]);
        assert_ne!(a, c);
    }

    #[test]
    fn reload_reason_maps_error_kinds() {
        assert_eq!(
            ReloadReason::from_error_kind("non_json"),
            ReloadReason::Parse
        );
        assert_eq!(
            ReloadReason::from_error_kind("schema_failed"),
            ReloadReason::Validate
        );
        assert_eq!(
            ReloadReason::from_error_kind("parse_failed"),
            ReloadReason::Validate
        );
        assert_eq!(
            ReloadReason::from_error_kind("bad_key"),
            ReloadReason::Validate
        );
        assert_eq!(
            ReloadReason::from_error_kind("unknown_kind"),
            ReloadReason::Validate
        );
    }
}
