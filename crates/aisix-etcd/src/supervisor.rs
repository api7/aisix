//! Watch supervisor — the single long-running task that owns the
//! [`ConfigProvider`] and keeps an [`AisixSnapshot`] current in a
//! [`SnapshotHandle`].
//!
//! Responsibilities (spec §2):
//! 1. Initial `load_all` + publish first snapshot
//! 2. Open a watch stream from the load revision
//! 3. Apply Put/Delete events incrementally on top of the current
//!    snapshot (building a *new* snapshot each time so reads stay
//!    lock-free)
//! 4. On compaction or stream error, full-reload + resync
//! 5. Reconnect with exponential backoff (1→60s) on transport failure
//!
//! The apply step is *copy-on-write* per batch: we clone the current
//! snapshot into a new one, mutate, and `store` it. That keeps the
//! read path reading a fully-formed `Arc<Snapshot>` the whole time.

use aisix_core::config_status::{
    hash_record, hash_records, AppliedSnapshot, ConfigStatus, IncomingRejection, LoadObservation,
    PartialCompatResource, SourceKind,
};
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::AisixSnapshot;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use ring::digest::{Context, SHA256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Bound::{Excluded, Unbounded};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

use crate::backoff::ExpBackoff;
use crate::key::{PrefixScope, PrefixSet, WatchedPrefix};
use crate::loader::{self, BuildStats, PartialCompatEntry, PartialCompatRow, RejectedEntry};
use crate::provider::{ConfigProvider, ProviderError, RawEntry, WatchEvent};
use crate::snapshot_cache::{encode_entry, encode_stale, SnapshotCache};
use std::sync::atomic::AtomicBool;

/// Cheap clonable handle for the watch supervisor's freshness state —
/// the etcd revision the snapshot reflects, and how long ago the
/// supervisor last applied an event. Read by `/admin/v1/health` so
/// operators can tell at a glance whether the gateway is serving from
/// a frozen snapshot (etcd partition or watch supervisor wedged) vs
/// from a live config stream. See issue #114. Also read by the managed-
/// mode heartbeat, which reports the revision as `applied_revision` so
/// cp-api can compare it against the kine revision of its own writes
/// (#519 B.3).
///
/// The previous health endpoint only reported per-model upstream
/// connectivity; it was silent on the gateway's own freshness, so a
/// dead etcd watch could go unnoticed for hours while the proxy kept
/// serving the last-known config.
#[derive(Debug, Default, Clone)]
pub struct WatchStatus {
    inner: Arc<WatchStatusInner>,
}

#[derive(Debug)]
struct WatchStatusInner {
    /// Highest revision the supervisor has applied to its snapshot.
    /// Atomically updated on every load_once / apply_put / apply_delete /
    /// apply_resync. Zero before first apply.
    revision: AtomicI64,
    /// Wall-clock instant of the most recent apply. `None` means the
    /// supervisor has not yet completed its first cycle — boot state.
    /// `Mutex<Option<Instant>>` over `parking_lot` would be marginally
    /// cheaper, but std::sync::Mutex is uncontended here (one writer,
    /// multiple readers) so the overhead is irrelevant.
    last_apply_at: Mutex<Option<Instant>>,
}

impl Default for WatchStatusInner {
    fn default() -> Self {
        Self {
            revision: AtomicI64::new(0),
            last_apply_at: Mutex::new(None),
        }
    }
}

impl WatchStatus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that the supervisor just applied an event at `revision`.
    /// `revision` is the etcd revision the resulting snapshot reflects;
    /// caller stamps the highest revision it's seen so concurrent /
    /// out-of-order updates don't downgrade the published view.
    pub(crate) fn record_apply(&self, revision: i64) {
        let prev = self.inner.revision.load(Ordering::Relaxed);
        if revision > prev {
            self.inner.revision.store(revision, Ordering::Relaxed);
        }
        *self.inner.last_apply_at.lock().unwrap() = Some(Instant::now());
    }

    /// [`Self::record_apply`] for a completed READ, which knows the exact
    /// point its result is consistent as of and therefore ASSIGNS rather
    /// than raising.
    ///
    /// A read of several prefixes can land below what the resync just
    /// stamped from the entry set — a row in the later-read prefix may
    /// have been written after the earlier prefix was read, while a write
    /// to the earlier prefix in that interval is missing from the union
    /// altogether. Keeping the higher number would go on claiming a write
    /// the gateway has not seen.
    pub(crate) fn record_read(&self, revision: i64) {
        self.inner.revision.store(revision, Ordering::Relaxed);
        *self.inner.last_apply_at.lock().unwrap() = Some(Instant::now());
    }

    /// Snapshot the current freshness state. Returns the revision and
    /// the age (wall-clock duration since last apply); `None` for age
    /// means the supervisor has not yet successfully completed a cycle.
    pub fn snapshot(&self) -> WatchStatusSnapshot {
        let revision = self.inner.revision.load(Ordering::Relaxed);
        let last_apply_age = self
            .inner
            .last_apply_at
            .lock()
            .unwrap()
            .map(|t| t.elapsed());
        WatchStatusSnapshot {
            revision,
            last_apply_age,
        }
    }
}

/// Point-in-time read of [`WatchStatus`].
#[derive(Debug, Clone, Copy)]
pub struct WatchStatusSnapshot {
    /// Highest etcd revision currently reflected in the snapshot. Zero
    /// before first apply.
    pub revision: i64,
    /// How long ago the supervisor last applied an event. `None` means
    /// no apply has happened yet (boot, or DP started in disconnected
    /// mode without a usable snapshot cache).
    pub last_apply_age: Option<Duration>,
}

/// Maximum rejected entries the supervisor retains in memory. The
/// heartbeat path drains and re-fills this on each tick, but if the
/// CP is unreachable for a while we don't want to leak unbounded
/// memory. Newest rejection wins on overflow (drops the oldest).
const MAX_RETAINED_REJECTIONS: usize = 256;

/// Maximum partially-compatible rows the supervisor retains, in its own
/// buffer so YELLOW volume can never evict RED entries from the rejection
/// buffer above (#871). Bounded by the number of config rows in practice;
/// past the cap new rows are still logged by the loader but drop out of
/// the aggregated report, with a WARN so the truncation is never silent.
const MAX_RETAINED_PARTIAL_ROWS: usize = 1024;

/// How long shutdown waits for an in-flight snapshot-cache write to
/// reach disk before abandoning it.
///
/// Sized as a backstop against a wedged disk, not as a budget: the write
/// is one local file and completes in milliseconds on any healthy one.
/// Deliberately not configurable — an operator has nothing to trade off
/// here, and a knob would only offer a way to make shutdown hang longer.
const CACHE_WRITE_DRAIN: Duration = Duration::from_secs(5);

/// One key whose latest etcd bytes are rejected while its last
/// successfully loaded value keeps serving (#871, xDS-NACK style).
/// `entry` pins the last-known-good raw document with the revision it
/// was accepted at; `since_unix_secs` is the instant stale serving began
/// (the first rejected replacement observed for the key), reported as
/// the staleness age and persisted in the snapshot cache so the age
/// stays continuous across restarts.
///
/// Deliberately uncapped, unlike the rejection and partial-compat
/// buffers: dropping an entry here would take a served resource offline,
/// not truncate a report. The map is bounded by the number of rows that
/// ever loaded successfully — a subset of the served snapshot, which is
/// itself uncapped.
#[derive(Debug, Clone)]
pub struct StaleServing {
    pub entry: RawEntry,
    pub since_unix_secs: u64,
    cache_record: OnceLock<Arc<[u8]>>,
}

impl StaleServing {
    pub(crate) fn new(entry: RawEntry, since_unix_secs: u64) -> Self {
        Self {
            entry,
            since_unix_secs,
            cache_record: OnceLock::new(),
        }
    }

    fn cache_record(&self) -> Arc<[u8]> {
        self.cache_record.get_or_init(|| encode_stale(self)).clone()
    }
}

impl PartialEq for StaleServing {
    fn eq(&self, other: &Self) -> bool {
        self.entry == other.entry && self.since_unix_secs == other.since_unix_secs
    }
}

impl Eq for StaleServing {}

/// One observed etcd entry plus the bytes it contributes to the config
/// digests, computed once when the entry is stored.
#[derive(Debug, Clone)]
struct StateEntry {
    entry: RawEntry,
    /// [`aisix_core::config_status::hash_record`] of `entry`. A pure
    /// function of `(key, value)`, so it is valid for exactly as long as
    /// this map holds these bytes.
    record: Arc<[u8]>,
    cache_record: OnceLock<Arc<[u8]>>,
}

impl StateEntry {
    fn new(entry: RawEntry) -> Self {
        let record: Arc<[u8]> = hash_record(&entry.key, &entry.value).into();
        Self {
            entry,
            record,
            cache_record: OnceLock::new(),
        }
    }

    fn cache_record(&self) -> Arc<[u8]> {
        self.cache_record
            .get_or_init(|| encode_entry(&self.entry))
            .clone()
    }
}

// Checkpoints contain the digest state AFTER each group of records. Any
// insertion, replacement or deletion invalidates the checkpoint at its key
// and all later checkpoints. Keep the map and its cache under the same lock.
const HASH_CHECKPOINT_RECORDS: usize = 64;

#[derive(Default)]
struct ObservedState {
    entries: BTreeMap<String, StateEntry>,
    checkpoints: Vec<(String, Context)>,
    digest: Option<String>,
    #[cfg(test)]
    last_hashed_records: usize,
}

impl ObservedState {
    fn invalidate(&mut self, key: &str) {
        let keep = self.checkpoints.partition_point(|(k, _)| k.as_str() < key);
        self.checkpoints.truncate(keep);
        self.digest = None;
    }

    fn insert(&mut self, entry: RawEntry) {
        let row = StateEntry::new(entry);
        if self
            .entries
            .get(&row.entry.key)
            .is_none_or(|old| old.record != row.record)
        {
            self.invalidate(&row.entry.key);
        }
        self.entries.insert(row.entry.key.clone(), row);
    }

    fn remove(&mut self, key: &str) -> Option<StateEntry> {
        let removed = self.entries.remove(key);
        if removed.is_some() {
            self.invalidate(key);
        }
        removed
    }

    fn source_hash(&mut self) -> String {
        #[cfg(test)]
        {
            self.last_hashed_records = 0;
        }
        if let Some(digest) = &self.digest {
            return digest.clone();
        }
        let (start, mut hash) = match self.checkpoints.last() {
            Some((key, hash)) => (Excluded(key.clone()), hash.clone()),
            None => (Unbounded, Context::new(&SHA256)),
        };
        for (index, (key, row)) in self.entries.range((start, Unbounded)).enumerate() {
            hash.update(&row.record);
            #[cfg(test)]
            {
                self.last_hashed_records += 1;
            }
            if (index + 1) % HASH_CHECKPOINT_RECORDS == 0 {
                self.checkpoints.push((key.clone(), hash.clone()));
            }
        }
        let digest = hex::encode(hash.finish());
        self.digest = Some(digest.clone());
        digest
    }
}

/// One supervisor instance. Consumers call [`Supervisor::run`] once and
/// drop the returned handle on shutdown.
pub struct Supervisor<P: ConfigProvider> {
    /// One source per watched prefix: the environment's, and — in a
    /// managed deployment — the shared `<base>/global/` catalog. Each is
    /// range-read and watched separately; all of them land in the ONE
    /// snapshot this supervisor publishes, so a request never sees half
    /// a configuration.
    sources: Vec<PrefixSource<P>>,
    /// The same prefixes, in the form key parsing resolves against.
    prefixes: PrefixSet,
    handle: SnapshotHandle<AisixSnapshot>,

    // Cached records and SHA-256 prefixes share the authoritative entry
    // map's lock, so status publication cannot reuse a prefix invalidated
    // by a concurrent Put/Delete/resync. Disk snapshot encoding is lazy.
    state: Mutex<ObservedState>,
    revision: Mutex<i64>,
    cache: SnapshotCache,

    /// Freshness signal exposed to /admin/v1/health. Updated on every
    /// successful apply path (load_once / apply_put / apply_delete /
    /// apply_resync). `Clone` produces a cheap read handle for the
    /// admin handler.
    status: WatchStatus,

    /// Load-observability signal exposed on the metrics/status listener
    /// (`/status/config`, `/status/ready`, `aisix_config_*` series).
    /// Recomputed from `state` / `rejections` / the published snapshot after
    /// every apply so operators can answer "did my config take effect".
    config_status: ConfigStatus,

    /// Most recent loader rejections, capped at
    /// [`MAX_RETAINED_REJECTIONS`]. Read by the heartbeat path so the
    /// CP can surface "your DP rejected these resources" in the
    /// dashboard. Newest at the back; on overflow the oldest entries
    /// are dropped — see issue #115. The buffer is replaced (not
    /// appended-to) on every load_once / apply_resync because those
    /// re-process the full entry set; apply_put / apply_delete append
    /// per-event because they only see one row.
    rejections: Mutex<Vec<RejectedEntry>>,

    /// Rows currently served with unknown fields ignored (partially
    /// compatible, #871), keyed by etcd key so incremental watch events
    /// merge cleanly: a Put replaces (or clears) the key's entry, a
    /// Delete removes it, a resync replaces the map wholesale. Reported
    /// aggregated per (kind, field) on `/status/config` and the
    /// heartbeat. Separate from `rejections` by design — see
    /// [`MAX_RETAINED_PARTIAL_ROWS`].
    partial_compat: Mutex<HashMap<String, PartialCompatRow>>,

    /// Last-known-good state, keyed by etcd key: exactly the keys whose
    /// latest etcd bytes are rejected while a previously accepted value
    /// keeps serving (#871). A rejected put pins the serving bytes here;
    /// a successful put or a delete removes the key; a resync drops keys
    /// that now load or left etcd and re-injects the rest into the fresh
    /// snapshot. Persisted via [`SnapshotCache`] so retention survives
    /// restarts. Independent of the capped `rejections` buffer — buffer
    /// overflow must never take a served resource offline.
    ///
    /// Locking: never held together with another supervisor lock; every
    /// user snapshots or mutates it in its own scope.
    stale_serving: Mutex<HashMap<String, StaleServing>>,

    // JoinHandles for in-flight `flush_cache` writes. [`Self::run`]
    // drains them before returning so a gateway stopped shortly after an
    // apply comes back with a persisted snapshot rather than none. WHICH
    // apply is a separate question: `SnapshotCache::store` serialises its
    // snapshot before taking the cache's write lock, so two writes racing
    // that work can commit in the opposite order to the applies that
    // produced them. Draining does not change that, and does not claim
    // to. Tests use
    // [`Self::await_pending_cache_writes`] to order against a write
    // without relying on a wall-clock sleep, which proved flaky on slow
    // CI runners. Kept to the writes actually in flight: `flush_cache`
    // drops finished handles as it pushes, so a long-lived gateway does
    // not accumulate one per apply.
    pending_writes: Mutex<Vec<JoinHandle<()>>>,
}

/// One watched prefix and the provider that reads it.
struct PrefixSource<P: ConfigProvider> {
    prefix: WatchedPrefix,
    provider: Arc<P>,
    /// Set once the tolerated refusal below has been logged, cleared the
    /// next time the prefix reads successfully — so an old control plane
    /// produces one WARN, not one per reconnect.
    refusal_logged: AtomicBool,
}

impl<P: ConfigProvider> PrefixSource<P> {
    /// Whether this prefix may be treated as empty when etcd answers
    /// `err`, instead of failing the cycle.
    ///
    /// Only the shared catalog, and only for a refusal. A control plane
    /// older than the catalog denies reads outside the environment's own
    /// prefix, and a gateway that failed its cycle over that would serve
    /// nothing at all rather than serve without prices. Every other
    /// error, and every error on the environment prefix, is handled as
    /// before: the cycle fails and the backoff loop retries.
    ///
    /// A refusal that is really about credentials cannot hide here — it
    /// refuses the environment prefix too, which is not tolerated.
    fn tolerates(&self, err: &ProviderError) -> bool {
        self.prefix.scope == PrefixScope::Global && matches!(err, ProviderError::Rejected(_))
    }

    fn log_refusal(&self, err: &ProviderError) {
        if self.refusal_logged.swap(true, Ordering::Relaxed) {
            return;
        }
        tracing::warn!(
            prefix = %self.prefix.prefix,
            error = %err,
            "etcd refused the shared pricing catalog; keeping the prices last read from it, \
             or serving without prices if it was never read — a model with `pricing_key` and \
             no catalog entry falls back to its inline `cost` until the control plane grants \
             read access to this prefix",
        );
    }

    fn clear_refusal(&self) {
        self.refusal_logged.store(false, Ordering::Relaxed);
    }
}

impl<P: ConfigProvider> Supervisor<P> {
    /// Construct without on-disk persistence. Equivalent to
    /// [`Self::with_cache(provider, prefix, SnapshotCache::disabled())`].
    pub fn new(provider: Arc<P>, prefix: impl Into<String>) -> Self {
        Self::with_cache(provider, prefix, SnapshotCache::disabled())
    }

    /// Construct with a snapshot cache. After every successful
    /// resync / put / delete the supervisor flushes the current entry
    /// set to the cache so a restart that can't reach etcd still has
    /// configuration to serve from.
    ///
    /// Watches the environment prefix alone. Use [`Self::with_sources`]
    /// to add the shared catalog.
    pub fn with_cache(provider: Arc<P>, prefix: impl Into<String>, cache: SnapshotCache) -> Self {
        Self::with_sources(vec![(WatchedPrefix::environment(prefix), provider)], cache)
    }

    /// Construct over several prefixes, each with its own provider.
    ///
    /// All of them feed ONE snapshot and one revision floor: the applied
    /// revision is the MINIMUM across prefixes, because sequential reads
    /// make the union consistent only as far as the earliest of them (see
    /// [`PrefixLoad::applied_revision`]), and readiness waits for every
    /// prefix's initial range read (a tolerated refusal on the shared
    /// catalog counts as read-and-kept — see [`PrefixSource::tolerates`]).
    pub fn with_sources(mut sources: Vec<(WatchedPrefix, Arc<P>)>, cache: SnapshotCache) -> Self {
        // Environment prefixes are read FIRST, and that ordering is load
        // bearing rather than cosmetic: [`PrefixSource::tolerates`] is
        // only safe because credentials etcd genuinely refuses are
        // refused for the environment prefix too, and that refusal has
        // to be the one that surfaces. Reading the tolerant prefix first
        // would let a wrong password come back as a tolerated catalog
        // refusal followed by a second, redundant failure.
        sources.sort_by_key(|(p, _)| match p.scope {
            PrefixScope::Environment => 0,
            PrefixScope::Global => 1,
        });
        let prefixes = PrefixSet::new(sources.iter().map(|(p, _)| p.clone()).collect());
        let sources = sources
            .into_iter()
            .map(|(prefix, provider)| PrefixSource {
                prefix,
                provider,
                refusal_logged: AtomicBool::new(false),
            })
            .collect();
        Self {
            sources,
            prefixes,
            handle: SnapshotHandle::new(AisixSnapshot::new()),
            state: Mutex::new(ObservedState::default()),
            revision: Mutex::new(0),
            cache,
            status: WatchStatus::new(),
            config_status: ConfigStatus::new(SourceKind::Etcd),
            rejections: Mutex::new(Vec::new()),
            partial_compat: Mutex::new(HashMap::new()),
            stale_serving: Mutex::new(HashMap::new()),
            pending_writes: Mutex::new(Vec::new()),
        }
    }

    /// Cheap clonable handle to the supervisor's freshness state.
    /// Read by /admin/v1/health to surface "etcd watch alive" /
    /// "snapshot age" metrics. See [`WatchStatus`].
    pub fn watch_status(&self) -> WatchStatus {
        self.status.clone()
    }

    /// Cheap clonable handle to the load-observability state. Read by the
    /// metrics/status listener to serve `/status/config`, `/status/ready`,
    /// and the `aisix_config_*` series.
    pub fn config_status(&self) -> ConfigStatus {
        self.config_status.clone()
    }

    /// Recompute the load-observability view from the supervisor's
    /// authoritative in-memory state (the raw entry map, the retained
    /// rejection buffer, the published snapshot, and the revision floor) and
    /// publish it to [`Self::config_status`]. Idempotent: safe to call after
    /// every apply. `is_reload` counts a config reload for
    /// `aisix_config_reloads_total` — set only on full (re)syncs, not on
    /// incremental watch events.
    fn sync_config_status(&self, is_reload: bool) {
        // Snapshot the stale-serving state first, in its own lock scope
        // (see the `stale_serving` field docs for the locking rule).
        let stale: HashMap<String, StaleServing> = self.stale_serving.lock().unwrap().clone();
        let source_hash;
        let config_hash;
        let rejected: Vec<IncomingRejection>;
        {
            let mut state = self.state.lock().unwrap();
            let rejections = self.rejections.lock().unwrap();
            // Every observed write, including a rejected one, contributes
            // to the source hash. Unchanged prefixes keep their SHA state.
            source_hash = state.source_hash();
            let rejected_keys: HashSet<&str> = rejections.iter().map(|r| r.key.as_str()).collect();
            // config_hash covers the bytes each key ACTUALLY serves: the
            // observed etcd bytes for accepted keys, the pinned last-known-
            // good bytes for stale-serving keys (#871), and nothing for a
            // rejected key with no last good (it doesn't serve). The stale
            // map — not the capped rejection buffer — decides which keys
            // substitute, so buffer overflow can never flip a served key's
            // hash contribution to the rejected bytes.
            //
            // With nothing rejected and nothing pinned the filter admits
            // every key and the chain adds none, so the two digests are
            // over the identical record sequence — the overwhelmingly
            // common case, and one full SHA-256 pass to skip.
            config_hash = if rejected_keys.is_empty() && stale.is_empty() {
                source_hash.clone()
            } else {
                hash_records(served_records(&state.entries, &rejected_keys, &stale))
            };
            rejected = rejections
                .iter()
                .map(|r| self.map_rejection(r, &stale))
                .collect();
        }
        let revision = *self.revision.lock().unwrap();
        let resource_counts = resource_counts(&self.handle.load());
        let (partially_compatible, partially_compatible_rows_by_kind) =
            self.partial_compat_observation();
        let mut stale_served_rows_by_kind: BTreeMap<String, usize> = BTreeMap::new();
        for key_str in stale.keys() {
            if let Ok(parsed) = self.prefixes.resolve(key_str) {
                *stale_served_rows_by_kind
                    .entry(parsed.kind.to_string())
                    .or_insert(0) += 1;
            }
        }

        self.config_status.record_load(LoadObservation {
            source_hash,
            observed_revision: Some(revision),
            applied: Some(AppliedSnapshot {
                config_hash,
                revision: Some(revision),
                resource_counts,
            }),
            rejected,
            partially_compatible,
            partially_compatible_rows_by_kind,
            stale_served_rows_by_kind,
            is_reload,
            // etcd always publishes the accepted subset (even an empty one);
            // it never retains a previous snapshot wholesale, so a wholly-
            // rejected resync is captured by the empty accepted set instead.
            wholly_rejected: false,
        });
    }

    /// Map a loader [`RejectedEntry`] to the source-agnostic wire shape. The
    /// key is split into `<kind>/<id>` via [`key::parse`]; an unparseable key
    /// (the `bad_key` path) reports empty kind/id, mirroring the control
    /// plane's rejected-resources surface. `stale` joins in the instant the
    /// key began serving its last known good value, if it is (#871).
    fn map_rejection(
        &self,
        r: &RejectedEntry,
        stale: &HashMap<String, StaleServing>,
    ) -> IncomingRejection {
        let (kind, id) = match self.prefixes.resolve(&r.key) {
            Ok(parsed) => (parsed.kind.to_string(), parsed.id.to_string()),
            Err(_) => (String::new(), String::new()),
        };
        IncomingRejection {
            identity: r.key.clone(),
            resource_kind: kind,
            resource_id: id,
            last_error_kind: r.kind.as_str().to_string(),
            last_error: r.error.clone(),
            seen_at: DateTime::from_timestamp(r.timestamp_unix_secs as i64, 0)
                .unwrap_or_else(Utc::now),
            serving_stale_since: stale
                .get(&r.key)
                .and_then(|s| DateTime::from_timestamp(s.since_unix_secs as i64, 0)),
        }
    }

    /// Snapshot of the most recent loader rejections (capped), with the
    /// stale-serving instant joined in per key (#871). Used by the
    /// heartbeat path to forward "DP rejected these resources" to
    /// cp-api. Returns a clone so the caller doesn't hold the lock
    /// across the heartbeat HTTP call.
    pub fn recent_rejections(&self) -> Vec<RejectedEntry> {
        let stale: HashMap<String, u64> = {
            let guard = self.stale_serving.lock().unwrap();
            guard
                .iter()
                .map(|(k, s)| (k.clone(), s.since_unix_secs))
                .collect()
        };
        let mut out = self.rejections.lock().unwrap().clone();
        for r in &mut out {
            r.stale_serving_since_unix_secs = stale.get(&r.key).copied();
        }
        out
    }

    /// Replace the retained rejection buffer wholesale. Called by the
    /// resync paths (load_once / apply_resync) which re-process every
    /// entry — old per-key rejections are no longer accurate.
    fn set_rejections(&self, mut new: Vec<RejectedEntry>) {
        if new.len() > MAX_RETAINED_REJECTIONS {
            // Keep the *newest* entries; tail of the vec is freshest.
            new.drain(..new.len() - MAX_RETAINED_REJECTIONS);
        }
        *self.rejections.lock().unwrap() = new;
    }

    /// Append one rejection from a per-event apply path (apply_put).
    /// Drops the oldest on overflow. Existing entries for the same
    /// key are replaced so heartbeat reports the latest error once.
    fn push_rejection(&self, r: RejectedEntry) {
        let mut guard = self.rejections.lock().unwrap();
        guard.retain(|existing| existing.key != r.key);
        if guard.len() >= MAX_RETAINED_REJECTIONS {
            guard.remove(0);
        }
        guard.push(r);
    }

    /// Remove retained rejection signal for a key that was either
    /// successfully applied or deleted.
    fn remove_rejection_for_key(&self, key: &str) -> bool {
        let mut guard = self.rejections.lock().unwrap();
        let before = guard.len();
        guard.retain(|existing| existing.key != key);
        guard.len() != before
    }

    /// Aggregated partially-compatible observations for the currently
    /// served snapshot: one entry per (kind, field) with the number of
    /// rows carrying it, sorted. Read by the heartbeat path (cloned, no
    /// lock held across the HTTP call).
    pub fn recent_partial_compat(&self) -> Vec<PartialCompatEntry> {
        let guard = self.partial_compat.lock().unwrap();
        let rows: Vec<PartialCompatRow> = guard.values().cloned().collect();
        drop(guard);
        loader::aggregate_partial_compat(&rows)
    }

    /// Replace the retained partially-compatible state wholesale. Called
    /// by the resync paths, which re-process every entry.
    fn set_partial_rows(&self, rows: Vec<PartialCompatRow>) {
        let mut guard = self.partial_compat.lock().unwrap();
        guard.clear();
        for row in rows {
            if guard.len() >= MAX_RETAINED_PARTIAL_ROWS {
                tracing::warn!(
                    cap = MAX_RETAINED_PARTIAL_ROWS,
                    "partially-compatible rows exceed the retention cap; \
                     the aggregated report is truncated"
                );
                break;
            }
            guard.insert(row.key.clone(), row);
        }
    }

    /// Merge one apply_put outcome into the retained partially-compatible
    /// state: the row's new unknown-field set replaces its previous one,
    /// and a row that now matches exactly clears its entry.
    fn update_partial_row(&self, key: &str, row: Option<PartialCompatRow>) {
        let mut guard = self.partial_compat.lock().unwrap();
        match row {
            Some(row) => {
                if !guard.contains_key(key) && guard.len() >= MAX_RETAINED_PARTIAL_ROWS {
                    tracing::warn!(
                        key = %key,
                        cap = MAX_RETAINED_PARTIAL_ROWS,
                        "partially-compatible rows exceed the retention cap; \
                         this row is served but missing from the aggregated report"
                    );
                    return;
                }
                guard.insert(key.to_string(), row);
            }
            None => {
                guard.remove(key);
            }
        }
    }

    /// The retained partially-compatible state in the two wire shapes
    /// [`LoadObservation`] carries: the per-(kind, field) aggregate and
    /// the per-kind row counts.
    fn partial_compat_observation(&self) -> (Vec<PartialCompatResource>, BTreeMap<String, usize>) {
        let guard = self.partial_compat.lock().unwrap();
        let rows: Vec<PartialCompatRow> = guard.values().cloned().collect();
        drop(guard);
        let aggregated = loader::aggregate_partial_compat(&rows)
            .into_iter()
            .map(|e| PartialCompatResource {
                resource_kind: e.kind,
                field: e.field,
                count: e.count,
            })
            .collect();
        let mut rows_by_kind: BTreeMap<String, usize> = BTreeMap::new();
        for row in &rows {
            *rows_by_kind.entry(row.kind.clone()).or_insert(0) += 1;
        }
        (aggregated, rows_by_kind)
    }

    /// Drain the JoinHandles for any in-flight cache writes spawned
    /// by [`Self::flush_cache`] and await them, without a bound.
    ///
    /// [`Self::run`] calls this through [`Self::drain_pending_cache_writes`],
    /// which supplies the shutdown bound. Tests call it directly to order
    /// deterministically against the disk read that follows.
    async fn await_pending_cache_writes(&self) {
        let handles: Vec<JoinHandle<()>> = {
            let mut pending = self.pending_writes.lock().unwrap();
            std::mem::take(&mut *pending)
        };
        for handle in handles {
            // A write that panicked is its own bug, surfaced separately;
            // there is nothing useful to do about it here.
            let _ = handle.await;
        }
    }

    /// Shutdown drain: give any cache write still in flight up to
    /// [`CACHE_WRITE_DRAIN`] to reach disk, then give up.
    ///
    /// [`Self::flush_cache`] spawns the write detached so the apply path
    /// stays sync, so a gateway stopped shortly after an apply used to
    /// exit with that write unfinished and come back without its
    /// last-known-good snapshot — which since the proxy listener started
    /// gating on a first applied configuration means it refuses to bind
    /// at all until it reaches etcd, where it previously bound and served
    /// 401s.
    ///
    /// Bounded because shutdown must not be able to hang on a stuck disk.
    /// The bound is fixed rather than configurable: it is a backstop on a
    /// local file write, not a tuning knob.
    ///
    /// Expiry abandons the WAIT, not the write: dropping a `JoinHandle`
    /// detaches its task rather than cancelling it, so a write that was
    /// merely slow can still commit before the process exits. That is
    /// left alone deliberately — the supervisor loop has stopped, so no
    /// fresher state exists to lose the race to, and
    /// [`SnapshotCache::store`] renames a fsynced temporary over the
    /// destination, so a late write commits whole or not at all. Which
    /// apply ends up on disk is therefore unknown at this point, and the
    /// warning says so rather than promising one.
    async fn drain_pending_cache_writes(&self) {
        if tokio::time::timeout(CACHE_WRITE_DRAIN, self.await_pending_cache_writes())
            .await
            .is_err()
        {
            tracing::warn!(
                bound_ms = CACHE_WRITE_DRAIN.as_millis() as u64,
                "snapshot-cache write did not finish inside the shutdown drain; \
                 no longer waiting for it. It is abandoned rather than cancelled, \
                 so the on-disk cache ends up at one of the recent applies — \
                 never partway between them",
            );
        }
    }

    /// Try to seed the snapshot from the on-disk cache. Called once at
    /// boot before the etcd cycle starts so the proxy can serve traffic
    /// from cached config even if etcd is briefly unreachable.
    /// No-op when the cache is disabled or the file is missing /
    /// unparseable.
    pub fn restore_from_cache(&self) {
        let Some(cached) = self.cache.load() else {
            return;
        };
        // Seed the stale-serving state BEFORE the resync so keys whose
        // cached bytes are rejected recover their pinned last-known-good
        // values (#871) — apply_resync then re-validates each seed and
        // drops any whose key now loads cleanly or left the entry set.
        {
            let mut stale = self.stale_serving.lock().unwrap();
            stale.clear();
            for s in cached.stale {
                stale.insert(s.entry.key.clone(), s);
            }
        }
        let stats = self.apply_resync(&cached.entries);
        // Track the last cached revision so the first live cycle's
        // resync reflects the right "from where" in logs. We don't
        // try to use it as the watch start revision — the etcd server
        // may have compacted past it; load_all + watch from latest is
        // always safer.
        *self.revision.lock().unwrap() = cached.revision;
        // Reflect the cached revision on the status view (apply_resync above
        // synced with the entry-max revision).
        self.sync_config_status(false);
        tracing::info!(
            accepted = stats.accepted,
            revision = cached.revision,
            "snapshot restored from on-disk cache (offline-resilient boot)",
        );
    }

    /// Clone of the public snapshot handle. Axum state / request handlers
    /// hold this; calls to `.load()` are cheap atomic reads.
    pub fn handle(&self) -> SnapshotHandle<AisixSnapshot> {
        self.handle.clone()
    }

    /// Run one full reload + watch cycle and publish the resulting
    /// snapshot. Returns the stats from the build for observability.
    /// Stops after the first watch error — the outer [`Self::run`] loop
    /// decides whether to backoff and retry.
    pub async fn load_once(&self) -> Result<BuildStats, ProviderError> {
        let load = self.load_all_prefixes().await?;
        let revision = load.applied_revision();
        let stats = self.apply_resync_at(&load.entries, Some(revision));
        // apply_resync uses max(entry revisions); bump to the etcd
        // load_all revision so the cache file records the true "as
        // of" point, not just the max entry write.
        self.record_read_revision(revision);
        tracing::info!(
            accepted = stats.accepted,
            rejected = stats.schema_rejected + stats.parse_rejected,
            revision,
            "initial snapshot built",
        );
        Ok(stats)
    }

    /// Range-read every watched prefix and return the union of their
    /// entries, plus the revision each one's own read was consistent as of.
    ///
    /// The reads run in sequence, so the union as a whole is consistent
    /// only as far as the EARLIEST of them — see
    /// [`PrefixLoad::applied_revision`].
    async fn load_all_prefixes(&self) -> Result<PrefixLoad, ProviderError> {
        // Deduplicated by key, because the prefixes can nest: when the
        // gateway has no `env_id` the environment prefix is the bare base
        // and `<base>/global/` sits inside it, so both range reads return
        // the catalog's rows. The snapshot and the observed-state map are
        // both keyed and would absorb the repeat, but the BuildStats are
        // not — `accepted` and the per-field partially-compatible row
        // counts are sums, and those numbers are reported.
        let mut all: BTreeMap<String, RawEntry> = BTreeMap::new();
        // Per source, and not just the maximum: each prefix's watch has
        // to start from the revision ITS OWN range read was consistent
        // as of. The reads run in sequence, so a later prefix reports a
        // higher revision, and starting an earlier prefix's watch there
        // would skip every write to it in between — absent from that
        // read and never delivered to that watch.
        let mut revisions: Vec<Option<i64>> = Vec::with_capacity(self.sources.len());
        // Retention (below) runs in a second pass so it can see the WHOLE
        // read, not just the prefixes that came before the refusal.
        let mut read_ok: Vec<&str> = Vec::new();
        let mut refused: Vec<&PrefixSource<P>> = Vec::new();
        for source in &self.sources {
            match source.provider.load_all().await {
                Ok((entries, rev)) => {
                    source.clear_refusal();
                    read_ok.push(&source.prefix.prefix);
                    for entry in entries {
                        match all.entry(entry.key.clone()) {
                            std::collections::btree_map::Entry::Occupied(mut slot) => {
                                // Two reads of one key are two points in
                                // time; keep the later write.
                                if entry.revision > slot.get().revision {
                                    slot.insert(entry);
                                }
                            }
                            std::collections::btree_map::Entry::Vacant(slot) => {
                                slot.insert(entry);
                            }
                        }
                    }
                    revisions.push(Some(rev));
                }
                Err(err) if source.tolerates(&err) => {
                    // Read as refused: the prefix contributes no revision,
                    // has no revision to resume its watch from, and does not
                    // hold readiness back — as before. Its rows are carried
                    // over in the pass below.
                    source.log_refusal(&err);
                    refused.push(source);
                    revisions.push(None);
                }
                Err(err) => return Err(err),
            }
        }

        // A refusal is not a deletion. The rows a refused prefix
        // contributed to the last successful union are carried into this
        // one unchanged, and replaced only when a read of it succeeds
        // again — so a control plane that starts refusing the shared
        // catalog leaves the gateway pricing on the last catalog it was
        // allowed to read instead of dropping every price at once. Losing
        // them is not neutral: `least_cost` stops ranking and realtime /
        // batch usage events lose `cost_usd`, and the WARN above is the
        // only sign either happened.
        for source in refused {
            for entry in self.entries_last_read_from(&source.prefix) {
                // Only where nothing else answered for the key. On a
                // deployment with no `env_id` the environment prefix is the
                // bare base, so a SUCCESSFUL read of it already covers the
                // catalog — and that read is authoritative about the key
                // being gone, which carrying the old row over would undo.
                // A deleted price would come back to life, priced at
                // whatever it used to cost.
                if read_ok.iter().any(|p| entry.key.starts_with(p)) {
                    continue;
                }
                all.entry(entry.key.clone()).or_insert(entry);
            }
        }
        Ok(PrefixLoad {
            entries: all.into_values().collect(),
            revisions,
        })
    }

    /// The rows the last applied union carried under `prefix`.
    ///
    /// Attribution is by key prefix, which is exact for the only caller:
    /// [`PrefixSource::tolerates`] admits the shared catalog alone, and the
    /// catalog prefix is the longest one a supervisor watches, so no key
    /// under it belongs to another source.
    fn entries_last_read_from(&self, prefix: &WatchedPrefix) -> Vec<RawEntry> {
        let state = self.state.lock().unwrap();
        state
            .entries
            .iter()
            .filter(|(key, _)| key.starts_with(&prefix.prefix))
            .map(|(_, held)| held.entry.clone())
            .collect()
    }

    /// Record the revision a completed read was consistent as of, replacing
    /// whatever the preceding `apply_resync` derived from the entry set.
    /// Used by the cycle path so the cache and the heartbeat reflect when
    /// the DP last successfully reached the CP even when the resulting
    /// entry set is empty. Also stamps `WatchStatus.last_apply_at` so
    /// `/admin/v1/health` reflects the successful round-trip with etcd.
    ///
    /// **Assigns, rather than raising a floor.** `apply_resync` stamps
    /// `max(entry revision)`, which across several prefixes read in
    /// sequence can exceed the point the union as a whole is consistent as
    /// of: a row written to the later-read prefix after the earlier one was
    /// read carries a revision above the earlier read's header, while a
    /// write to the earlier prefix in that same interval is missing from
    /// the union entirely. Raising a floor would keep that overstatement
    /// and go on claiming a write the gateway has not seen — the one thing
    /// `applied_revision` exists to answer. The caller's value is the
    /// authoritative one and is never lower than honest.
    fn record_read_revision(&self, revision: i64) {
        *self.revision.lock().unwrap() = revision;
        self.status.record_read(revision);
        // Reflect the finalised revision on the status view (the preceding
        // apply_resync synced with the entry-max revision; this corrects it to
        // the load/watch header revision). Not itself a reload event.
        self.sync_config_status(false);
    }

    /// Pin the currently served bytes for `key` as its last known good
    /// (#871). Called when a watch put for the key is rejected. No-op
    /// when the key is already stale-tracked (the original pin and its
    /// `since` stand) or when nothing serves for the key (it never
    /// loaded successfully — there is no good value to pin).
    ///
    /// The serving bytes are read from `state[key]`: the invariant is
    /// that a key present in the served snapshot and NOT stale-tracked
    /// has its served bytes in `state` (a rejected put pins here BEFORE
    /// mirroring the rejected bytes into `state`, and a resync that
    /// rejects a key either stale-tracks it or drops it from the
    /// snapshot).
    fn capture_last_good(&self, key_str: &str, base: &AisixSnapshot, view: &BatchView) {
        if self.stale_serving.lock().unwrap().contains_key(key_str) {
            return;
        }
        let Ok(parsed) = self.prefixes.resolve(key_str) else {
            return;
        };
        // The presence probe reads through the batch's own staged
        // mutations: within one coalesced apply a row put earlier in the
        // batch is serving as far as this decision is concerned, even
        // though the snapshot carrying it has not been published yet.
        if !view.present(base, parsed.table_kind(), parsed.id) {
            return;
        }
        let Some(good) = self
            .state
            .lock()
            .unwrap()
            .entries
            .get(key_str)
            .map(|e| e.entry.clone())
        else {
            return;
        };
        // entry().or_insert_with keeps the original pin (and its `since`)
        // if a concurrent caller won the race after the check above.
        self.stale_serving
            .lock()
            .unwrap()
            .entry(key_str.to_string())
            .or_insert_with(|| StaleServing::new(good, now_unix_secs()));
    }

    /// Apply a single Put event on top of the current snapshot.
    /// Returns `true` if the apply succeeded (schema + parse passed).
    pub fn apply_put(&self, entry: &RawEntry) -> bool {
        self.apply_events(&[PendingEvent::Put(entry)])[0]
    }

    /// Apply a Delete event. Returns `true` if anything was actually
    /// removed (the kind/id was present).
    pub fn apply_delete(&self, key_str: &str) -> bool {
        self.apply_events(&[PendingEvent::Delete {
            key: key_str,
            revision: None,
        }])[0]
    }

    /// Apply a coalesced run of Put/Delete events as ONE copy-on-write
    /// cycle: one snapshot clone, one publish, one configuration-status
    /// sync, one cache flush. Returns each event's outcome, in order.
    ///
    /// Per-event work — schema validation, rejection bookkeeping,
    /// last-known-good pinning, the partial-compatibility signal, the
    /// observed-state map and the revision floor — still runs once per
    /// event and in order, so a batch decides exactly what the same
    /// events decided one at a time. What is shared is the work whose
    /// cost is the size of the WHOLE configuration rather than the size
    /// of the event: a bulk edit that lands 14 writes a second used to
    /// pay all of it 14 times a second (AISIX-Cloud#1542).
    fn apply_events(&self, events: &[PendingEvent<'_>]) -> Vec<bool> {
        let base = self.handle.load();
        let mut view = BatchView::default();
        let mut mutations: Vec<SnapshotMutation> = Vec::new();
        let mut outcomes = Vec::with_capacity(events.len());
        // The revision to stamp freshness with, if any event would have
        // stamped one. `record_apply` keeps the max, so one call with the
        // batch's max is what N calls would have left behind.
        let mut apply_stamp: Option<i64> = None;
        // What the batch has to publish. Split, because the two are not
        // the same set of events: a delete's revision-floor bump moves
        // what `/status/config` reports without changing a byte of the
        // observed state the cache file holds, and used to write nothing.
        let mut dirty = Dirty::default();

        for event in events {
            let outcome = match event {
                PendingEvent::Put(entry) => self.stage_put(
                    entry,
                    &base,
                    &mut view,
                    &mut mutations,
                    &mut apply_stamp,
                    &mut dirty,
                ),
                PendingEvent::Delete { key, revision } => self.stage_delete(
                    key,
                    *revision,
                    &base,
                    &mut view,
                    &mut mutations,
                    &mut apply_stamp,
                    &mut dirty,
                ),
            };
            outcomes.push(outcome);
        }
        drop(base);

        if !mutations.is_empty() {
            // RCU: load → clone → mutate → CAS, retrying the closure if a
            // concurrent apply raced our CAS. The previous implementation
            // used a bare load-mutate-store sequence which silently
            // dropped events under concurrency (see issue #112). The
            // closure body must be idempotent w.r.t. its input — the
            // staged mutations are a fixed ordered list replayed onto a
            // fresh clone each attempt.
            self.handle.rcu(|current| {
                let new = clone_snapshot(current);
                for mutation in &mutations {
                    mutation.apply_to(&new);
                }
                new
            });
        }
        if let Some(revision) = apply_stamp {
            // /admin/v1/health reads this — record the apply so
            // `last_apply_age` resets on every event we process.
            self.status.record_apply(revision);
        }
        if dirty.status {
            self.sync_config_status(false);
        }
        if dirty.cache {
            self.flush_cache();
        }
        outcomes
    }

    /// Stage one Put. See [`Self::apply_events`] for what is per-event
    /// and what is shared.
    fn stage_put(
        &self,
        entry: &RawEntry,
        base: &AisixSnapshot,
        view: &mut BatchView,
        mutations: &mut Vec<SnapshotMutation>,
        apply_stamp: &mut Option<i64>,
        dirty: &mut Dirty,
    ) -> bool {
        // Build a tiny snapshot out of just the new entry, then merge.
        let (tiny, mut stats) = loader::build_snapshot(&self.prefixes, std::slice::from_ref(entry));
        if stats.accepted == 0 {
            // The previous good value keeps serving. Pin it now (#871):
            // the next resync rebuilds from the rejected etcd bytes and
            // needs the pinned bytes to keep this row alive. Must run
            // BEFORE the state-map update below, which overwrites the
            // serving bytes with the rejected ones.
            self.capture_last_good(&entry.key, base, view);
            // Mirror the rejected bytes into the observed-state map and
            // the cache like any other observed etcd write: source_hash
            // reflects the observed etcd state immediately, and a
            // restart inside this window restores the same
            // rejected-bytes + pinned-value shape a post-resync restart
            // would — keeping the staleness clock continuous instead of
            // resetting it at the next boot.
            self.store_observed(entry);
            // Note: a previously retained partially-compatible entry for
            // this key is deliberately kept — the row's last-good value
            // (loaded with those fields ignored) is still what serves.
            // The loader already attached a RejectedEntry for whatever
            // path failed (bad key / non-JSON / schema / parse). Move
            // them into the supervisor's retained buffer so the next
            // heartbeat surfaces the failure to cp-api. See issue #115.
            for r in stats.rejections.drain(..) {
                self.push_rejection(r);
            }
            // A rejected watch event still changes the reported state
            // (rejected[] gains this entry; last_reload flips unsuccessful),
            // and its bytes are now part of the observed state on disk.
            dirty.status = true;
            dirty.cache = true;
            return false;
        }

        view.record_put(&self.prefixes, &entry.key);
        // Fold a run of puts into ONE staged snapshot rather than keeping a
        // `Box<AisixSnapshot>` per event alive until the commit: the loader
        // hands back a full fifteen-table snapshot for the single row it
        // parsed, and every one of those tables eagerly allocates a DashMap
        // shard array sized by the core count — hundreds of kilobytes per
        // event on a large host, for one row. Merging is order-preserving:
        // a later insert of the same id replaces the earlier one, which is
        // what applying the two puts in sequence does. A delete breaks the
        // run, so the ordering across kinds is kept too.
        match mutations.last_mut() {
            Some(SnapshotMutation::Merge(accumulated)) => merge_snapshot(accumulated, &tiny),
            _ => mutations.push(SnapshotMutation::Merge(Box::new(tiny))),
        }
        self.remove_rejection_for_key(&entry.key);
        // The key's latest bytes load again — retention ends (#871).
        self.stale_serving.lock().unwrap().remove(&entry.key);
        // Refresh this key's partially-compatible signal: replaced when
        // the new value still carries unknown fields, cleared when it now
        // matches the schema exactly.
        let partial = stats
            .partial_rows
            .drain(..)
            .find(|row| row.key == entry.key);
        self.update_partial_row(&entry.key, partial);
        // Mirror the put into the cache-tracking map. Track the highest
        // revision we've observed so the cache file records something
        // monotonic.
        self.store_observed(entry);
        stamp_max(apply_stamp, entry.revision);
        dirty.status = true;
        dirty.cache = true;
        true
    }

    /// Stage one Delete. `revision` carries the watch header revision the
    /// cycle loop would otherwise pass to [`Self::record_read_revision`]
    /// immediately afterwards; `None` (the standalone
    /// [`Self::apply_delete`] entry point) leaves the floor alone.
    #[allow(clippy::too_many_arguments)]
    fn stage_delete(
        &self,
        key_str: &str,
        revision: Option<i64>,
        base: &AisixSnapshot,
        view: &mut BatchView,
        mutations: &mut Vec<SnapshotMutation>,
        apply_stamp: &mut Option<i64>,
        dirty: &mut Dirty,
    ) -> bool {
        let parsed = match self.prefixes.resolve(key_str) {
            Ok(k) => k,
            Err(err) => {
                tracing::warn!(key = %key_str, error = %err, "ignoring delete with bad key");
                self.raise_revision_floor(revision, apply_stamp, dirty);
                return false;
            }
        };

        // Probe first — if the key isn't present we have nothing to
        // remove and don't want to stage a no-op mutation. The probe
        // reads through the batch's staged mutations as well as the
        // published snapshot, so a delete that follows a put of the same
        // key inside one batch sees the row the put staged.
        let present = view.present(base, parsed.table_kind(), parsed.id);
        let removed_rejection = self.remove_rejection_for_key(key_str);
        // A deleted key no longer serves, so its partially-compatible
        // signal (if any) goes with it — and so does its last-known-good
        // retention (#871): the pin must never outlive the etcd key.
        self.update_partial_row(key_str, None);
        self.stale_serving.lock().unwrap().remove(key_str);
        // The observed-state map drops the key on BOTH branches below.
        // A key can be absent from the snapshot yet present in `state`:
        // a rejected put mirrors its bytes there even when the row never
        // served. Leaving those bytes behind would keep the deleted key
        // in source_hash until the next resync and persist the deleted
        // document in the cache file.
        let removed_state = self.state.lock().unwrap().remove(key_str).is_some();
        if !present {
            if removed_rejection || removed_state {
                // Clearing a rejected key changes the reported state.
                // No per-event revision rides a wire delete, so stamp
                // freshness with the current floor.
                stamp_max(apply_stamp, *self.revision.lock().unwrap());
                dirty.status = true;
                dirty.cache = true;
            }
            self.raise_revision_floor(revision, apply_stamp, dirty);
            return removed_rejection;
        }

        view.record_delete(parsed.table_kind(), parsed.id);
        mutations.push(SnapshotMutation::Remove {
            kind: parsed.table_kind().to_string(),
            id: parsed.id.to_string(),
        });
        stamp_max(apply_stamp, *self.revision.lock().unwrap());
        dirty.status = true;
        dirty.cache = true;
        self.raise_revision_floor(revision, apply_stamp, dirty);
        true
    }

    /// Record an observed etcd write in the state map, with its digest
    /// record, and advance the revision floor to it.
    fn store_observed(&self, entry: &RawEntry) {
        {
            let mut state = self.state.lock().unwrap();
            state.insert(entry.clone());
        }
        let mut rev = self.revision.lock().unwrap();
        if entry.revision > *rev {
            *rev = entry.revision;
        }
    }

    /// In-batch form for the watch-event path: raise the floor and mark
    /// the batch as needing a status publish, deferring the single
    /// `record_apply` / `sync_config_status` to the end of the batch.
    ///
    /// A floor here, not an assignment as in [`Self::record_read_revision`]:
    /// these are individual watch events, delivered in revision order and
    /// each carrying only its own write's revision, so the highest seen is
    /// the point the configuration reflects. Only a completed READ knows a
    /// consistent-as-of point that can be lower than what came before.
    ///
    /// Status only — the floor is not part of the observed entry set, and
    /// `record_read_revision` never wrote the cache file either.
    fn raise_revision_floor(
        &self,
        revision: Option<i64>,
        apply_stamp: &mut Option<i64>,
        dirty: &mut Dirty,
    ) {
        let Some(revision) = revision else {
            return;
        };
        {
            let mut rev = self.revision.lock().unwrap();
            if revision > *rev {
                *rev = revision;
            }
        }
        stamp_max(apply_stamp, revision);
        dirty.status = true;
    }

    /// Replace the current snapshot with a freshly loaded set (resync).
    ///
    /// Rejected keys don't simply vanish (#871): a key whose latest bytes
    /// are rejected but whose previous good value was serving keeps
    /// serving that value — the pre-existing "cliff" where a routine
    /// resync/restart silently took a resource offline days after the
    /// write that broke it. Retention ends when the key loads cleanly
    /// again or leaves etcd.
    pub fn apply_resync(&self, entries: &[RawEntry]) -> BuildStats {
        self.apply_resync_at(entries, None)
    }

    /// [`Self::apply_resync`] told the revision the entry set was read at.
    ///
    /// A completed read knows the exact point its result is consistent as
    /// of, and that point can be BELOW the highest revision in the entry
    /// set — see [`Self::record_read_revision`]. The caller's value has to
    /// land BEFORE the cache flush at the end of this function, because the
    /// cache refuses a write below the revision it has already committed:
    /// commit the entry maximum here and every apply between the two
    /// numbers is silently dropped from the cache — precisely the events
    /// carrying the writes the read could not see. A restart during an etcd
    /// outage would then serve a cache claiming a revision whose changes it
    /// does not contain.
    pub fn apply_resync_at(&self, entries: &[RawEntry], read_revision: Option<i64>) -> BuildStats {
        let (snap, mut stats) = loader::build_snapshot(&self.prefixes, entries);

        // Reconcile the last-known-good state against this build, then
        // inject the retained values into the fresh snapshot.
        let rejected_keys: HashSet<&str> =
            stats.rejections.iter().map(|r| r.key.as_str()).collect();
        let entry_keys: HashSet<&str> = entries.iter().map(|e| e.key.as_str()).collect();
        // Serving bytes for newly rejected keys come from the PRE-resync
        // state map (see `capture_last_good` for the invariant). Collect
        // them before `state` is replaced below.
        let prev_state: HashMap<String, RawEntry> = {
            let state = self.state.lock().unwrap();
            stats
                .rejections
                .iter()
                .filter_map(|r| {
                    state
                        .entries
                        .get(&r.key)
                        .map(|e| (r.key.clone(), e.entry.clone()))
                })
                .collect()
        };
        let prev_snap = self.handle.load();
        let injected: Vec<RawEntry> = {
            let mut stale = self.stale_serving.lock().unwrap();
            // Retention ends for keys that now load cleanly or left etcd
            // entirely — the delete-side guarantee that a pinned value
            // never outlives its key.
            stale.retain(|k, _| {
                entry_keys.contains(k.as_str()) && rejected_keys.contains(k.as_str())
            });
            // Newly rejected keys that were serving up to this resync:
            // pin their serving bytes now.
            for r in &stats.rejections {
                if stale.contains_key(&r.key) {
                    continue;
                }
                let Ok(parsed) = self.prefixes.resolve(&r.key) else {
                    continue;
                };
                if !snapshot_has(&prev_snap, parsed.table_kind(), parsed.id) {
                    continue;
                }
                if let Some(good) = prev_state.get(&r.key) {
                    stale.insert(
                        r.key.clone(),
                        StaleServing::new(good.clone(), now_unix_secs()),
                    );
                }
            }
            stale.values().map(|s| s.entry.clone()).collect()
        };
        drop(prev_snap);

        // Re-build each pinned value from its bytes so every derived
        // signal (typed value, YELLOW ignored-field paths) stays
        // consistent with what actually serves. A pinned value this
        // build can no longer parse (e.g. after a DP downgrade) drops
        // its retention with an ERROR — same contract as any RED row.
        if !injected.is_empty() {
            let (lkg_snap, lkg_stats) = loader::build_snapshot(&self.prefixes, &injected);
            if !lkg_stats.rejections.is_empty() {
                let mut stale = self.stale_serving.lock().unwrap();
                for r in &lkg_stats.rejections {
                    tracing::error!(
                        key = %r.key,
                        error = %r.error,
                        "pinned last-known-good value no longer parses; dropping retention",
                    );
                    stale.remove(&r.key);
                }
            }
            merge_snapshot(&snap, &lkg_snap);
            stats.partial_rows.extend(lkg_stats.partial_rows);
            stats.partially_compatible = loader::aggregate_partial_compat(&stats.partial_rows);
            if lkg_stats.accepted > 0 {
                tracing::info!(
                    count = lkg_stats.accepted,
                    "serving last-known-good values for rejected keys",
                );
            }
        }

        self.handle.store(snap);

        // Replace the cache-tracking map wholesale and flush.
        {
            let mut state = self.state.lock().unwrap();
            *state = ObservedState::default();
            for e in entries {
                state.insert(e.clone());
            }
        }
        match read_revision {
            // The caller read the entry set and knows what it is
            // consistent as of. Assigned, not raised, for the reason in
            // this function's doc comment.
            Some(revision) => *self.revision.lock().unwrap() = revision,
            // No read behind this one — the standalone entry point, used
            // by tests and by callers replaying an entry set. The max of
            // any entry is the best available answer.
            None => {
                if let Some(rev_val) = entries.iter().map(|e| e.revision).max() {
                    let mut rev = self.revision.lock().unwrap();
                    if rev_val > *rev {
                        *rev = rev_val;
                    }
                }
            }
        }
        // /admin/v1/health: stamp freshness on every resync, even when the
        // resulting entry set is empty (record_apply with the current
        // revision floor so the operator sees recent activity).
        let cur_rev = *self.revision.lock().unwrap();
        self.status.record_apply(cur_rev);
        // Resync re-processes the entire entry set so the prior
        // per-key rejection list is no longer accurate — replace it
        // wholesale with what this build produced (issue #115). Same for
        // the partially-compatible state (#871).
        self.set_rejections(stats.rejections.clone());
        self.set_partial_rows(stats.partial_rows.clone());
        // A full resync is a config reload — publish the observability view
        // (source/config hashes, counts, rejected list) and count it.
        self.sync_config_status(true);
        self.flush_cache();
        stats
    }

    /// Snapshot the current cache-tracking map and write it to disk.
    /// Called from the apply paths; safe to invoke from sync code
    /// because the cache writer lives behind a tokio runtime detected
    /// via `tokio::spawn` — when called outside a runtime (tests that
    /// don't drive the cache), the write is silently dropped which is
    /// the desired no-op.
    fn flush_cache(&self) {
        if !self.cache.is_enabled() {
            return;
        }
        let entries: Vec<Arc<[u8]>> = {
            let state = self.state.lock().unwrap();
            state
                .entries
                .values()
                .map(StateEntry::cache_record)
                .collect()
        };
        let stale: Vec<Arc<[u8]>> = {
            let guard = self.stale_serving.lock().unwrap();
            guard.values().map(StaleServing::cache_record).collect()
        };
        let revision = *self.revision.lock().unwrap();
        let cache = self.cache.clone();
        // Spawn the actual write so the apply path stays sync. If we
        // aren't inside a runtime, just skip.
        // Track the JoinHandle so [`Self::run`] can drain it at shutdown,
        // and so tests can deterministically await the write via
        // [`Self::await_pending_cache_writes`] instead of leaning on
        // `tokio::time::sleep`, which under CI load raced the spawn
        // (~50ms wasn't enough on heavily loaded GitHub Actions runners).
        if let Ok(rt_handle) = tokio::runtime::Handle::try_current() {
            let join = rt_handle
                .spawn(async move { cache.store_encoded(&entries, revision, &stale).await });
            let mut pending = self.pending_writes.lock().unwrap();
            // Only writes still in flight are worth draining, and only
            // those may be retained: the list is appended to on every
            // apply and would otherwise grow for the life of the process.
            pending.retain(|handle| !handle.is_finished());
            pending.push(join);
        }
    }

    /// Long-running loop. Handles exp-backoff reconnects and resync on
    /// compaction. Runs until cancelled via the cancellation token, then
    /// drains any snapshot-cache write still in flight.
    ///
    /// The drain lives here rather than in the server's shutdown
    /// coordinator so it cannot be forgotten by a caller, and so it lands
    /// in the one place where no further write can be spawned: the loop
    /// that owns every call to [`Self::flush_cache`] has just exited.
    /// Callers already await this task after the connection drain, so
    /// nothing about the graceful-drain sequence changes.
    pub async fn run(self: Arc<Self>, cancel: tokio::sync::watch::Receiver<bool>) {
        self.watch_loop(cancel).await;
        self.drain_pending_cache_writes().await;
    }

    async fn watch_loop(&self, mut cancel: tokio::sync::watch::Receiver<bool>) {
        let mut backoff = ExpBackoff::default();
        loop {
            if *cancel.borrow() {
                return;
            }

            match self.cycle(&cancel).await {
                Ok(()) => {
                    // Graceful stream end (compaction or server-initiated
                    // close). Reset backoff, but still yield a short
                    // interval before reconnecting so we never spin.
                    backoff.reset();
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                        _ = cancel.changed() => {
                            if *cancel.borrow() { return; }
                        }
                    }
                }
                Err(SupervisorError::Cancelled) => return,
                Err(SupervisorError::Provider(err)) => {
                    // Surface the source outage on /status/config: connected
                    // flips false and a fetch-reason reload failure is counted.
                    // The last-good applied snapshot keeps serving.
                    self.config_status.record_fetch_failure();
                    let delay = backoff.next_delay();
                    // A refusal is not a transport hiccup: etcd answered
                    // and said no, and no amount of backing off changes
                    // that. The boot path exits on one, but by here the
                    // process is already up (etcd was unreachable when it
                    // started), so the loop keeps going and says loudly
                    // why nothing is being applied.
                    if matches!(err, ProviderError::Rejected(_)) {
                        tracing::error!(
                            error = %err,
                            backoff_ms = delay.as_millis() as u64,
                            "etcd refused this gateway's credentials — no configuration can be \
                             read until they are fixed; still retrying",
                        );
                    } else if matches!(err, ProviderError::TokenRefused(_)) {
                        // Deliberately NOT the line above. etcd accepted
                        // the credentials — it issued the token being
                        // refused — so pointing at the username and
                        // password sends an operator somewhere there is
                        // nothing to find. The causes are the auth store
                        // changing faster than the gateway can
                        // re-authenticate, and, under `--auth-token jwt`,
                        // a token that is already outside its `ttl` when
                        // it arrives because the two clocks disagree.
                        tracing::error!(
                            error = %err,
                            backoff_ms = delay.as_millis() as u64,
                            "etcd would not accept an auth token it had just issued to this \
                             gateway — the credentials are fine; check whether etcd's auth \
                             store is being changed continuously, and whether this host's \
                             clock agrees with etcd's; still retrying",
                        );
                    } else {
                        tracing::warn!(
                            error = %err,
                            backoff_ms = delay.as_millis() as u64,
                            "etcd watch failed; backing off before reconnect",
                        );
                    }
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = cancel.changed() => {
                            if *cancel.borrow() { return; }
                        }
                    }
                }
            }
        }
    }

    /// One attempt at load + watch. Any error returns without retrying —
    /// [`Self::run`] owns the backoff loop.
    async fn cycle(
        &self,
        cancel: &tokio::sync::watch::Receiver<bool>,
    ) -> Result<(), SupervisorError> {
        let load = self
            .load_all_prefixes()
            .await
            .map_err(SupervisorError::Provider)?;
        let revision = load.applied_revision();

        // ONE resync over the union: the snapshot is published only after
        // every prefix has been read, so readiness means every prefix's
        // initial load completed and no request can observe the
        // environment loaded and the catalog not.
        self.apply_resync_at(&load.entries, Some(revision));
        self.record_read_revision(revision);

        let mut streams = Vec::with_capacity(self.sources.len());
        for (source, from) in self.sources.iter().zip(&load.revisions) {
            // Each prefix resumes from its own read, never from the
            // maximum across prefixes. A prefix whose read was refused
            // has no revision to resume from and no stream this cycle.
            let Some(from) = *from else { continue };
            match source.provider.watch(from + 1).await {
                Ok(stream) => streams.push(stream),
                Err(err) if source.tolerates(&err) => {
                    // No stream for this prefix this cycle. It is retried
                    // whenever the cycle restarts — which is what a
                    // control-plane upgrade causes, since it takes the
                    // kine connections behind the surviving watch with it.
                    source.log_refusal(&err);
                }
                Err(err) => return Err(SupervisorError::Provider(err)),
            }
        }
        // Each stream is tagged so its END is delivered as an item
        // rather than absorbed by `select_all`. See [`Watched`].
        let mut stream = futures::stream::select_all(streams.into_iter().map(|s| {
            s.map(Watched::Event)
                .chain(futures::stream::iter([Watched::Ended]))
        }));

        // An event the drain below pulled off the stream but could not
        // add to its batch (a resync, a stream error, the end of the
        // stream). Held here so the next loop iteration handles it
        // exactly as if it had just arrived.
        let mut pushed_back: Option<Option<Watched>> = None;
        let mut apply_timing = ApplyTiming::default();

        loop {
            if *cancel.borrow() {
                return Err(SupervisorError::Cancelled);
            }

            let next = match pushed_back.take() {
                Some(item) => item,
                None => tokio::select! {
                    item = stream.next() => item,
                    _ = wait_for_cancel(cancel.clone()) => {
                        return Err(SupervisorError::Cancelled);
                    }
                },
            };

            match next {
                // Every stream exhausted, or any ONE of them ended:
                // either way this cycle is over and the next one re-reads
                // and re-watches every prefix.
                None | Some(Watched::Ended) => return Ok(()),
                Some(Watched::Event(Err(ProviderError::Compacted))) => {
                    tracing::warn!("etcd compaction detected — resyncing");
                    // Break out so `run` re-enters `cycle` cleanly; the
                    // next iteration re-loads from scratch. We don't want
                    // to treat compaction as a backoff-worthy failure.
                    return Ok(());
                }
                Some(Watched::Event(Err(err))) => return Err(SupervisorError::Provider(err)),
                Some(Watched::Event(Ok(WatchEvent::Resync { entries, revision }))) => {
                    self.apply_resync_at(&entries, Some(revision));
                    // The resync's header revision is the "consistent as
                    // of" point even when the entry set is empty or only
                    // contains older mod_revisions.
                    self.record_read_revision(revision);
                }
                // Explicit over the two batchable variants rather than a
                // catch-all: a new `WatchEvent` must fail to compile here
                // instead of falling into a batch that cannot carry it.
                Some(Watched::Event(Ok(
                    first @ (WatchEvent::Put(_) | WatchEvent::Delete { .. }),
                ))) => {
                    // Coalesce: hold the batch open for a short bounded
                    // window and apply the whole run as one
                    // copy-on-write cycle. The window closes on the first
                    // of three conditions — the quiet period with no new
                    // event, COALESCE_MAX_WAIT since the first event,
                    // or MAX_APPLY_BATCH events — so a bulk edit costs a
                    // bounded number of whole-configuration passes no
                    // matter how the source spaces its deliveries
                    // (AISIX-Cloud#1542). Draining only what was already
                    // buffered made that number depend on how the etcd
                    // endpoint happened to batch its watch deliveries and
                    // on how long the previous apply took.
                    let mut batch = vec![first];
                    let quiet_period = apply_timing.quiet_period();
                    let deadline = tokio::time::Instant::now() + COALESCE_MAX_WAIT;
                    while batch.len() < MAX_APPLY_BATCH {
                        // Checked here as well as in the select: with a
                        // stream that always has an event ready the
                        // `sleep_until` branch is never polled (`biased`),
                        // and the window would then be bounded by
                        // MAX_APPLY_BATCH alone rather than by time. That
                        // is a real shape, not a hypothetical — the etcd
                        // stream flattens one `WatchResponse` into many
                        // consecutively ready items.
                        if tokio::time::Instant::now() >= deadline {
                            break;
                        }
                        let item = tokio::select! {
                            biased;
                            // FIRST, ahead of the stream: `biased` gives
                            // the window to whichever branch is ready
                            // earliest in this list, and during a bulk
                            // edit the stream has an event ready on every
                            // poll — so ordered after it, this branch
                            // would never be reached and shutdown would
                            // keep consuming the backlog until the max
                            // wait or MAX_APPLY_BATCH ended the window.
                            // The batch collected so far is still applied
                            // below: those events are already off the
                            // stream, so that apply is the only thing
                            // that can still serve and persist them. The
                            // ones still on the stream are untouched and
                            // come back from etcd on the next start.
                            _ = wait_for_cancel(cancel.clone()) => break,
                            item = stream.next() => item,
                            _ = tokio::time::sleep_until(deadline) => break,
                            _ = tokio::time::sleep(quiet_period) => break,
                        };
                        match item {
                            Some(Watched::Event(Ok(
                                event @ (WatchEvent::Put(_) | WatchEvent::Delete { .. }),
                            ))) => batch.push(event),
                            other => {
                                pushed_back = Some(other);
                                break;
                            }
                        }
                    }
                    if batch.len() > 1 {
                        tracing::debug!(
                            events = batch.len(),
                            "coalescing watch events into one snapshot publish",
                        );
                    }
                    let staged: Vec<PendingEvent<'_>> = batch
                        .iter()
                        .map(|event| match event {
                            WatchEvent::Put(raw) => PendingEvent::Put(raw),
                            // Advance the applied-revision floor to the
                            // delete's mod_revision even when the key
                            // wasn't present — "processed everything up
                            // to rev X" must cover deletes, otherwise the
                            // heartbeat-reported applied_revision (#519
                            // B.3) stalls after a CP delete until the next
                            // put arrives.
                            WatchEvent::Delete { key, revision } => PendingEvent::Delete {
                                key,
                                revision: Some(*revision),
                            },
                            // Unreachable by the arm above and the drain's
                            // own filter, both of which are exhaustive over
                            // `WatchEvent`.
                            WatchEvent::Resync { .. } => {
                                unreachable!("a resync is never added to a coalesced batch")
                            }
                        })
                        .collect();
                    let started = std::time::Instant::now();
                    self.apply_events(&staged);
                    apply_timing.record(started.elapsed());
                }
            }
        }
    }
}

/// One pass of [`Supervisor::load_all_prefixes`]: the union of every
/// watched prefix's rows, plus the revision each prefix's own read was
/// consistent as of (`None` for a refusal that was tolerated).
struct PrefixLoad {
    entries: Vec<RawEntry>,
    revisions: Vec<Option<i64>>,
}

impl PrefixLoad {
    /// The revision the snapshot as a whole reflects, and what the
    /// heartbeat reports as `applied_revision`.
    ///
    /// The LOWEST any prefix reported, not the highest. kine revisions are
    /// cluster-global, but the range reads run in sequence: a write landing
    /// between the first prefix's read and the last one's is below the last
    /// read's header revision and absent from the union all the same. So
    /// the union is consistent as of the EARLIEST point any part of it was
    /// read at, and claiming the highest would tell the control plane the
    /// gateway had applied a write it has not seen — the one thing
    /// `applied_revision` exists to answer.
    ///
    /// A prefix whose read was refused reports no revision and does not
    /// lower the answer: it contributed no rows this cycle, so nothing in
    /// the union came from it. With no prefix reporting one at all the
    /// answer is 0, which the caller treats as a floor and never lowers an
    /// earlier one with.
    fn applied_revision(&self) -> i64 {
        self.revisions.iter().flatten().copied().min().unwrap_or(0)
    }
}

/// One item off the merged watch.
///
/// `select_all` drops an exhausted stream and keeps polling the
/// survivors, so a stream that ends cleanly is otherwise invisible — and
/// a cleanly ended watch is precisely the signal [`Supervisor::cycle`]
/// runs on: it returns, and `watch_loop` re-reads and re-opens both
/// prefixes. With one stream that came for free. With two, on two
/// connections, swallowing it would leave the cycle polling the
/// survivor while the ended prefix's configuration froze for the life
/// of the process — no resync, no error, and `/status/config` still
/// reporting connected.
enum Watched {
    Event(Result<WatchEvent, ProviderError>),
    /// The stream this item came from has ended.
    Ended,
}

#[derive(Debug)]
enum SupervisorError {
    Cancelled,
    Provider(ProviderError),
}

async fn wait_for_cancel(mut rx: tokio::sync::watch::Receiver<bool>) {
    loop {
        if *rx.borrow() {
            return;
        }
        if rx.changed().await.is_err() {
            // Sender dropped: treat as cancellation.
            return;
        }
    }
}

/// Largest number of watch events one coalesced apply may cover.
///
/// Bounds how much one apply can hold off the publish when the control
/// plane pushes a bulk edit, so a slow batch cannot make the served
/// snapshot arbitrarily stale.
const MAX_APPLY_BATCH: usize = 512;

/// Quiet period for cheap applies and the first write after an idle interval.
const COALESCE_QUIET_PERIOD: Duration = Duration::from_millis(20);

/// Hard cap on how long the first event of a batch waits for company.
///
/// Without it a control plane writing faster than the quiet period would
/// hold a batch open for the whole edit and the served snapshot would
/// never catch up, so this — not the quiet period — is what bounds
/// config-change visibility during a sustained burst. Everything above
/// roughly a tenth of a second is indistinguishable to an operator
/// watching a save land, and each further step up buys proportionally
/// less: a 1500-row burst costs 12 applies here and would cost 9 at the
/// 200 ms this was picked under.
const COALESCE_MAX_WAIT: Duration = Duration::from_millis(150);

#[derive(Default)]
struct ApplyTiming {
    previous: Option<(tokio::time::Instant, Duration)>,
}

impl ApplyTiming {
    fn quiet_period(&self) -> Duration {
        match self.previous {
            // Expensive applies need to amortize their configuration-wide
            // work across spaced writes too. Forget that cost after an idle
            // interval so a later isolated write keeps its short wait.
            Some((finished, cost)) if finished.elapsed() < COALESCE_MAX_WAIT => {
                cost.clamp(COALESCE_QUIET_PERIOD, COALESCE_MAX_WAIT)
            }
            _ => COALESCE_QUIET_PERIOD,
        }
    }

    fn record(&mut self, cost: Duration) {
        self.previous = Some((tokio::time::Instant::now(), cost));
    }
}

/// One watch event staged for a coalesced apply.
enum PendingEvent<'a> {
    Put(&'a RawEntry),
    Delete {
        key: &'a str,
        /// The watch header revision. `None` leaves the revision floor
        /// alone, which is what the standalone `apply_delete` entry point
        /// has always done.
        revision: Option<i64>,
    },
}

/// One staged change to the snapshot, replayed onto a fresh structural
/// clone inside the RCU closure.
enum SnapshotMutation {
    /// Merge an accepted put's single-entry snapshot. Boxed: an
    /// `AisixSnapshot` is fifteen tables wide and the delete variant is
    /// two strings.
    Merge(Box<AisixSnapshot>),
    Remove {
        kind: String,
        id: String,
    },
}

impl SnapshotMutation {
    fn apply_to(&self, dst: &AisixSnapshot) {
        match self {
            // `merge_snapshot` must cover every ResourceTable on
            // AisixSnapshot — a missing kind there means a watch event
            // silently drops on the floor and the snapshot never sees the
            // new entry, even though the loader and the proxy both know
            // about it.
            Self::Merge(src) => merge_snapshot(dst, src),
            Self::Remove { kind, id } => remove_from_snapshot(dst, kind, id),
        }
    }
}

/// The staged mutations of a batch, as a presence overlay on the snapshot
/// the batch started from. Lets the per-event decisions that ask "does
/// this row serve right now?" see the batch's own earlier events, which
/// have not been published yet.
#[derive(Debug, Default)]
struct BatchView {
    added: HashSet<(String, String)>,
    removed: HashSet<(String, String)>,
}

impl BatchView {
    fn record_put(&mut self, prefixes: &PrefixSet, key_str: &str) {
        let Ok(parsed) = prefixes.resolve(key_str) else {
            return;
        };
        let id = (parsed.table_kind().to_string(), parsed.id.to_string());
        self.removed.remove(&id);
        self.added.insert(id);
    }

    fn record_delete(&mut self, kind: &str, id: &str) {
        let id = (kind.to_string(), id.to_string());
        self.added.remove(&id);
        self.removed.insert(id);
    }

    fn present(&self, base: &AisixSnapshot, kind: &str, id: &str) -> bool {
        let probe = (kind.to_string(), id.to_string());
        if self.removed.contains(&probe) {
            return false;
        }
        if self.added.contains(&probe) {
            return true;
        }
        snapshot_has(base, kind, id)
    }
}

/// What one coalesced apply still owes when its events are staged.
#[derive(Debug, Default)]
struct Dirty {
    /// `/status/config` and the `aisix_config_*` series.
    status: bool,
    /// The on-disk snapshot cache, whose content is the observed entry
    /// set — so a bump of the revision floor alone does not dirty it.
    cache: bool,
}

/// Keep the larger of `slot` and `revision`.
fn stamp_max(slot: &mut Option<i64>, revision: i64) {
    *slot = Some(match *slot {
        Some(current) if current >= revision => current,
        _ => revision,
    });
}

/// The digest records for the bytes each key ACTUALLY serves, in the
/// ascending-key order `hash_records` is defined over: the observed etcd
/// bytes for an accepted key, the pinned last-known-good bytes for a
/// stale-serving one (#871), and nothing for a key rejected with no last
/// good. `state` is already sorted, so the pinned records are merged into
/// the walk rather than sorted with it.
fn served_records(
    state: &BTreeMap<String, StateEntry>,
    rejected_keys: &HashSet<&str>,
    stale: &HashMap<String, StaleServing>,
) -> Vec<Arc<[u8]>> {
    let mut pinned: Vec<(&str, Arc<[u8]>)> = stale
        .values()
        .map(|s| {
            let record: Arc<[u8]> = hash_record(&s.entry.key, &s.entry.value).into();
            (s.entry.key.as_str(), record)
        })
        .collect();
    pinned.sort_by(|a, b| a.0.cmp(b.0));

    let mut out: Vec<Arc<[u8]>> = Vec::with_capacity(state.len() + pinned.len());
    let mut next_pinned = 0usize;
    for (key, entry) in state {
        // A pinned key etcd no longer reports still serves — emit it in
        // its own place in the ordering rather than at the end.
        while next_pinned < pinned.len() && pinned[next_pinned].0 < key.as_str() {
            out.push(pinned[next_pinned].1.clone());
            next_pinned += 1;
        }
        if next_pinned < pinned.len() && pinned[next_pinned].0 == key.as_str() {
            out.push(pinned[next_pinned].1.clone());
            next_pinned += 1;
            continue;
        }
        if rejected_keys.contains(key.as_str()) {
            continue;
        }
        out.push(entry.record.clone());
    }
    for (_, record) in &pinned[next_pinned..] {
        out.push(record.clone());
    }
    out
}

/// Structural clone: the new snapshot shares every
/// [`Arc<ResourceEntry>`][aisix_core::ResourceEntry] with `src` and copies
/// each table's generation, so the copy-on-write cycle behind one watch
/// event costs a per-row `Arc` bump rather than a deep copy of every
/// payload in the configuration, and leaves every derived cache valid.
fn clone_snapshot(src: &AisixSnapshot) -> AisixSnapshot {
    src.clone()
}

/// Insert every entry of `src` into `dst` (replacing same-id entries),
/// sharing the `Arc` rather than copying the payload.
/// The exhaustive destructuring makes adding a ResourceTable to
/// [`AisixSnapshot`] a compile error here — a missing kind would mean
/// entries silently drop on the floor when a watch put merges or a
/// last-known-good row is re-injected on resync.
fn merge_snapshot(dst: &AisixSnapshot, src: &AisixSnapshot) {
    let AisixSnapshot {
        models,
        apikeys,
        provider_keys,
        guardrails,
        guardrail_attachments,
        cache_policies,
        observability_exporters,
        rate_limit_policies,
        mcp_servers,
        mcp_policies,
        a2a_agents,
        oidc_providers,
        claim_mappings,
        passthrough_routes,
        mcp_auth_settings,
        pricing,
        global_pricing,
    } = src;
    for e in models.entries() {
        dst.models.insert_arc(e);
    }
    for e in apikeys.entries() {
        dst.apikeys.insert_arc(e);
    }
    for e in provider_keys.entries() {
        dst.provider_keys.insert_arc(e);
    }
    for e in guardrails.entries() {
        dst.guardrails.insert_arc(e);
    }
    for e in guardrail_attachments.entries() {
        dst.guardrail_attachments.insert_arc(e);
    }
    for e in cache_policies.entries() {
        dst.cache_policies.insert_arc(e);
    }
    for e in observability_exporters.entries() {
        dst.observability_exporters.insert_arc(e);
    }
    for e in rate_limit_policies.entries() {
        dst.rate_limit_policies.insert_arc(e);
    }
    for e in mcp_servers.entries() {
        dst.mcp_servers.insert_arc(e);
    }
    for e in mcp_policies.entries() {
        dst.mcp_policies.insert_arc(e);
    }
    for e in a2a_agents.entries() {
        dst.a2a_agents.insert_arc(e);
    }
    for e in oidc_providers.entries() {
        dst.oidc_providers.insert_arc(e);
    }
    for e in claim_mappings.entries() {
        dst.claim_mappings.insert_arc(e);
    }
    for e in passthrough_routes.entries() {
        dst.passthrough_routes.insert_arc(e);
    }
    for e in mcp_auth_settings.entries() {
        dst.mcp_auth_settings.insert_arc(e);
    }
    for e in pricing.entries() {
        dst.pricing.insert_arc(e);
    }
    for e in global_pricing.entries() {
        dst.global_pricing.insert_arc(e);
    }
}

/// Remove `(kind, id)` from `snap`.
///
/// Exhaustively destructured for the same drift-guard reason as
/// [`merge_snapshot`] and [`snapshot_has`]: a kind added to the snapshot
/// but missed here would be found present by `snapshot_has`, stage a
/// removal, and then silently no-op — the row would never be deletable.
fn remove_from_snapshot(snap: &AisixSnapshot, kind: &str, id: &str) {
    let AisixSnapshot {
        models,
        apikeys,
        provider_keys,
        guardrails,
        guardrail_attachments,
        cache_policies,
        observability_exporters,
        rate_limit_policies,
        mcp_servers,
        mcp_policies,
        a2a_agents,
        oidc_providers,
        claim_mappings,
        passthrough_routes,
        mcp_auth_settings,
        pricing,
        global_pricing,
    } = snap;
    match kind {
        "models" => {
            models.remove(id);
        }
        "api_keys" => {
            apikeys.remove(id);
        }
        "provider_keys" => {
            provider_keys.remove(id);
        }
        "guardrails" => {
            guardrails.remove(id);
        }
        "guardrail_attachments" => {
            guardrail_attachments.remove(id);
        }
        "cache_policies" => {
            cache_policies.remove(id);
        }
        "observability_exporters" => {
            observability_exporters.remove(id);
        }
        "rate_limit_policies" => {
            rate_limit_policies.remove(id);
        }
        "mcp_servers" => {
            mcp_servers.remove(id);
        }
        "mcp_policies" => {
            mcp_policies.remove(id);
        }
        "a2a_agents" => {
            a2a_agents.remove(id);
        }
        "oidc_providers" => {
            oidc_providers.remove(id);
        }
        "claim_mappings" => {
            claim_mappings.remove(id);
        }
        "passthrough_routes" => {
            passthrough_routes.remove(id);
        }
        "mcp_auth_settings" => {
            mcp_auth_settings.remove(id);
        }
        // Not the `pricing` kind but the table selector — a global-prefix
        // pricing row arrives here as `global_pricing`. See
        // [`crate::key::ScopedKey::table_kind`].
        "pricing" => {
            pricing.remove(id);
        }
        "global_pricing" => {
            global_pricing.remove(id);
        }
        _ => {}
    }
}

/// Whether the snapshot holds an entry for `(kind, id)`. An unknown
/// kind reads as absent. Exhaustively destructured for the same
/// drift-guard reason as [`merge_snapshot`]: a kind added to the
/// snapshot but missed here would silently never pin a last known good.
fn snapshot_has(snap: &AisixSnapshot, kind: &str, id: &str) -> bool {
    let AisixSnapshot {
        models,
        apikeys,
        provider_keys,
        guardrails,
        guardrail_attachments,
        cache_policies,
        observability_exporters,
        rate_limit_policies,
        mcp_servers,
        mcp_policies,
        a2a_agents,
        oidc_providers,
        claim_mappings,
        passthrough_routes,
        mcp_auth_settings,
        pricing,
        global_pricing,
    } = snap;
    match kind {
        "models" => models.get_by_id(id).is_some(),
        "api_keys" => apikeys.get_by_id(id).is_some(),
        "provider_keys" => provider_keys.get_by_id(id).is_some(),
        "guardrails" => guardrails.get_by_id(id).is_some(),
        "guardrail_attachments" => guardrail_attachments.get_by_id(id).is_some(),
        "cache_policies" => cache_policies.get_by_id(id).is_some(),
        "observability_exporters" => observability_exporters.get_by_id(id).is_some(),
        "rate_limit_policies" => rate_limit_policies.get_by_id(id).is_some(),
        "mcp_servers" => mcp_servers.get_by_id(id).is_some(),
        "mcp_policies" => mcp_policies.get_by_id(id).is_some(),
        "a2a_agents" => a2a_agents.get_by_id(id).is_some(),
        "oidc_providers" => oidc_providers.get_by_id(id).is_some(),
        "claim_mappings" => claim_mappings.get_by_id(id).is_some(),
        "passthrough_routes" => passthrough_routes.get_by_id(id).is_some(),
        "mcp_auth_settings" => mcp_auth_settings.get_by_id(id).is_some(),
        "pricing" => pricing.get_by_id(id).is_some(),
        "global_pricing" => global_pricing.get_by_id(id).is_some(),
        _ => false,
    }
}

/// Wall-clock seconds since the Unix epoch; zero on a pre-epoch clock.
fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Per-kind counts of the served snapshot, keyed by the plural etcd resource
/// kind (matching the `<prefix>/<kind>/<id>` key segment). Only non-empty
/// kinds are included, so an empty snapshot yields an empty map.
fn resource_counts(snap: &AisixSnapshot) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (kind, n) in [
        ("models", snap.models.len()),
        ("api_keys", snap.apikeys.len()),
        ("provider_keys", snap.provider_keys.len()),
        ("guardrails", snap.guardrails.len()),
        ("guardrail_attachments", snap.guardrail_attachments.len()),
        ("cache_policies", snap.cache_policies.len()),
        (
            "observability_exporters",
            snap.observability_exporters.len(),
        ),
        ("rate_limit_policies", snap.rate_limit_policies.len()),
        ("mcp_servers", snap.mcp_servers.len()),
        ("mcp_policies", snap.mcp_policies.len()),
        ("a2a_agents", snap.a2a_agents.len()),
        ("oidc_providers", snap.oidc_providers.len()),
        ("claim_mappings", snap.claim_mappings.len()),
        ("passthrough_routes", snap.passthrough_routes.len()),
        ("mcp_auth_settings", snap.mcp_auth_settings.len()),
        ("pricing", snap.pricing.len()),
        ("global_pricing", snap.global_pricing.len()),
    ] {
        if n > 0 {
            counts.insert(kind.to_string(), n);
        }
    }
    counts
}

/// Total time the supervisor will wait across its full 1→60s backoff
/// ladder before saturating. Exposed as a constant for tests and docs.
pub const BACKOFF_SATURATE_AFTER: Duration = Duration::from_secs(63);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{RawEntry, WatchEvent};
    use async_trait::async_trait;
    use futures::stream;
    use std::sync::Mutex;

    struct FakeProvider {
        entries: Mutex<Vec<RawEntry>>,
        revision: i64,
        events: Mutex<Vec<Result<WatchEvent, ProviderError>>>,
    }

    impl FakeProvider {
        fn new(entries: Vec<RawEntry>, revision: i64) -> Self {
            Self {
                entries: Mutex::new(entries),
                revision,
                events: Mutex::new(Vec::new()),
            }
        }

        fn with_events(mut self, events: Vec<Result<WatchEvent, ProviderError>>) -> Self {
            self.events = Mutex::new(events);
            self
        }
    }

    #[async_trait]
    impl ConfigProvider for FakeProvider {
        async fn load_all(&self) -> Result<(Vec<RawEntry>, i64), ProviderError> {
            Ok((self.entries.lock().unwrap().clone(), self.revision))
        }

        async fn watch(
            &self,
            _start_revision: i64,
        ) -> Result<
            Box<dyn futures::Stream<Item = Result<WatchEvent, ProviderError>> + Send + Unpin>,
            ProviderError,
        > {
            let events: Vec<_> = self.events.lock().unwrap().drain(..).collect();
            Ok(Box::new(stream::iter(events)))
        }
    }

    const VALID_MODEL: &[u8] = br#"{
        "display_name": "my-gpt4",
        "provider": "openai",
        "model_name": "gpt-4o",
        "provider_key_id": "11111111-1111-1111-1111-111111111111"
    }"#;

    fn assert_observed_hash(state: &mut ObservedState) {
        let expected = aisix_core::config_status::hash_entries(
            state
                .entries
                .iter()
                .map(|(key, row)| (key.as_str(), row.entry.value.as_slice())),
        );
        assert_eq!(state.source_hash(), expected);
        assert!(state.checkpoints.len() <= state.entries.len() / HASH_CHECKPOINT_RECORDS);
    }

    #[test]
    fn observed_hash_reuses_unchanged_prefix_and_identical_records() {
        let mut state = ObservedState::default();
        for i in 0..256 {
            state.insert(entry(
                &format!("/aisix/models/{i:04}"),
                br#"{"a":1,"b":2}"#,
                1,
            ));
        }
        assert_observed_hash(&mut state);
        assert_eq!(state.last_hashed_records, 256);
        state.insert(entry("/aisix/models/0240", br#"{"a":2,"b":2}"#, 2));
        assert_observed_hash(&mut state);
        assert!(
            state.last_hashed_records < 128,
            "a late change must skip the unchanged prefix"
        );
        let digest = state.source_hash();
        state.insert(entry("/aisix/models/0240", br#"{ "b": 2, "a": 2 }"#, 3));
        assert_eq!(state.source_hash(), digest);
        assert_eq!(
            state.last_hashed_records, 0,
            "canonical-equivalent updates do not hash again"
        );
        assert_eq!(state.entries["/aisix/models/0240"].entry.revision, 3);
        assert!(state.remove("/aisix/models/missing").is_none());
        assert_observed_hash(&mut state);
        assert_eq!(state.last_hashed_records, 0);
    }

    #[test]
    fn observed_hash_matches_full_digest_across_mutation_boundaries() {
        let mut state = ObservedState::default();
        assert_observed_hash(&mut state);
        for i in 0..259 {
            state.insert(entry(
                &format!("/aisix/models/{i:04}"),
                br#"{"nested":{"z":1,"a":[]}}"#,
                1,
            ));
        }
        assert_observed_hash(&mut state);
        // Delete checkpoint keys, either edge, and an interior row; insert
        // both before an existing checkpoint and after the previous tail.
        for key in ["0063", "0127", "0000", "0258", "0190"] {
            state.remove(&format!("/aisix/models/{key}"));
            assert_observed_hash(&mut state);
        }
        for key in ["0000", "0062a", "0127", "0259"] {
            state.insert(entry(
                &format!("/aisix/models/{key}"),
                b"invalid JSON\0\xff",
                2,
            ));
            assert_observed_hash(&mut state);
        }
        // Multiple changes between publications, with varying raw lengths
        // and update order, exercise the first-dirty-key boundary.
        let mut seed = 6443u64;
        for batch in 0..80 {
            for op in 0..1 + batch % 8 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let key = format!("/aisix/models/{:04}", (seed >> 32) % 320);
                if op % 3 == 0 {
                    state.remove(&key);
                } else {
                    state.insert(entry(
                        &key,
                        &vec![b'x'; (seed as usize % 257) + 1],
                        batch + 3,
                    ));
                }
            }
            assert_observed_hash(&mut state);
        }
        for key in state.entries.keys().cloned().collect::<Vec<_>>() {
            state.remove(&key);
            assert_observed_hash(&mut state);
        }
        assert!(state.checkpoints.is_empty());
    }

    #[test]
    fn observed_hash_is_rebuilt_on_full_resync() {
        let sup = Supervisor::new(Arc::new(FakeProvider::new(vec![], 0)), "/aisix");
        let mut rows: Vec<_> = (0..256)
            .map(|i| entry(&format!("/aisix/models/{i:04}"), VALID_MODEL, 1))
            .collect();
        sup.apply_resync_at(&rows, Some(1));
        assert!(!sup.state.lock().unwrap().checkpoints.is_empty());
        rows.remove(240);
        rows[200].value = b"invalid JSON".to_vec();
        rows.push(entry("/aisix/models/0256", VALID_MODEL, 2));
        sup.apply_resync_at(&rows, Some(2));
        let expected = aisix_core::config_status::hash_entries(
            rows.iter()
                .map(|row| (row.key.as_str(), row.value.as_slice())),
        );
        let status = sup.config_status().view();
        assert_eq!(status.source.source_hash, Some(expected));
        assert_eq!(status.source.observed_revision, Some(2));
        assert_observed_hash(&mut sup.state.lock().unwrap());
        sup.apply_resync_at(&[], Some(3));
        assert_observed_hash(&mut sup.state.lock().unwrap());
        assert!(sup.state.lock().unwrap().checkpoints.is_empty());
    }

    fn entry(key: &str, v: &[u8], rev: i64) -> RawEntry {
        RawEntry {
            key: key.into(),
            value: v.to_vec(),
            revision: rev,
        }
    }

    const VALID_APIKEY: &[u8] = br#"{
        "key_hash": "0000000000000000000000000000000000000000000000000000000000000000",
        "allowed_models": ["my-gpt4"]
    }"#;

    const VALID_GUARDRAIL: &[u8] = br#"{
        "name": "kw",
        "kind": "keyword",
        "patterns": [{"kind": "literal", "value": "AKIA"}]
    }"#;

    // ── multi-prefix supervision (AISIX-Cloud#1546) ───────────────

    const ENV_PREFIX: &str = "/aisix/env-1/";
    const GLOBAL_PREFIX: &str = "/aisix/global/";
    const VALID_PRICE: &[u8] =
        br#"{"key":"openai/gpt-4o","input_per_1k":0.005,"output_per_1k":0.015}"#;

    /// One scripted `load_all` answer: `Some((rows, revision))` serves,
    /// `None` refuses.
    type ScriptedRead = Option<(Vec<RawEntry>, i64)>;

    /// A provider that either serves entries or refuses every call the
    /// way a control plane predating the shared catalog does — its kine
    /// ACL answers `PermissionDenied` for a Range outside the
    /// environment's own prefix, which reaches the supervisor as
    /// [`ProviderError::Rejected`].
    struct ScopedProvider {
        entries: Vec<RawEntry>,
        revision: i64,
        refuse: bool,
        /// The `start_revision` this provider's watch was opened with.
        watched_from: Mutex<Option<i64>>,
        /// Hand back a stream that never ends and never yields, the way
        /// a healthy watch on a prefix nobody is writing behaves.
        never_ends: bool,
        /// Successive `load_all` answers, one popped per call, `None`
        /// refusing. Empty — every constructor but `scripted` — means the
        /// provider answers the same way every time.
        script: Mutex<std::collections::VecDeque<ScriptedRead>>,
    }

    impl ScopedProvider {
        fn serving(entries: Vec<RawEntry>, revision: i64) -> Arc<Self> {
            Arc::new(Self {
                entries,
                revision,
                refuse: false,
                watched_from: Mutex::new(None),
                never_ends: false,
                script: Mutex::new(std::collections::VecDeque::new()),
            })
        }

        /// Serves its rows, then holds a watch open forever — the
        /// catalog's normal steady state, since nobody writes prices
        /// most of the time.
        fn quiet(entries: Vec<RawEntry>, revision: i64) -> Arc<Self> {
            Arc::new(Self {
                entries,
                revision,
                refuse: false,
                watched_from: Mutex::new(None),
                never_ends: true,
                script: Mutex::new(std::collections::VecDeque::new()),
            })
        }

        fn refusing() -> Arc<Self> {
            Arc::new(Self {
                entries: Vec::new(),
                revision: 0,
                refuse: true,
                watched_from: Mutex::new(None),
                never_ends: false,
                script: Mutex::new(std::collections::VecDeque::new()),
            })
        }

        /// Answers each `load_all` from `steps` in order — `Some((rows,
        /// revision))` serves, `None` refuses — so one test can walk a
        /// prefix through success, refusal and success again.
        fn scripted(steps: Vec<ScriptedRead>) -> Arc<Self> {
            Arc::new(Self {
                entries: Vec::new(),
                revision: 0,
                refuse: true,
                watched_from: Mutex::new(None),
                never_ends: false,
                script: Mutex::new(steps.into()),
            })
        }
    }

    #[async_trait]
    impl ConfigProvider for ScopedProvider {
        async fn load_all(&self) -> Result<(Vec<RawEntry>, i64), ProviderError> {
            let refusal = || {
                ProviderError::Rejected(
                    "etcdserver: permission denied: outside env env-1 prefix".into(),
                )
            };
            let step = self.script.lock().unwrap().pop_front();
            if let Some(step) = step {
                return step.map(Ok).unwrap_or_else(|| Err(refusal()));
            }
            if self.refuse {
                return Err(refusal());
            }
            Ok((self.entries.clone(), self.revision))
        }

        async fn watch(
            &self,
            start_revision: i64,
        ) -> Result<
            Box<dyn futures::Stream<Item = Result<WatchEvent, ProviderError>> + Send + Unpin>,
            ProviderError,
        > {
            *self.watched_from.lock().unwrap() = Some(start_revision);
            if self.refuse {
                return Err(ProviderError::Rejected(
                    "etcdserver: permission denied: outside env env-1 prefix".into(),
                ));
            }
            if self.never_ends {
                return Ok(Box::new(stream::pending()));
            }
            Ok(Box::new(stream::iter(Vec::new())))
        }
    }

    fn scoped_supervisor(
        env: Arc<ScopedProvider>,
        global: Arc<ScopedProvider>,
    ) -> Arc<Supervisor<ScopedProvider>> {
        Arc::new(Supervisor::with_sources(
            vec![
                (WatchedPrefix::environment(ENV_PREFIX), env),
                (WatchedPrefix::global(GLOBAL_PREFIX), global),
            ],
            SnapshotCache::disabled(),
        ))
    }

    #[tokio::test]
    async fn both_prefixes_land_in_one_snapshot() {
        let sup = scoped_supervisor(
            ScopedProvider::serving(
                vec![
                    entry("/aisix/env-1/models/m-1", VALID_MODEL, 7),
                    entry("/aisix/env-1/pricing/env-p", VALID_PRICE, 8),
                ],
                8,
            ),
            ScopedProvider::serving(
                vec![entry("/aisix/global/pricing/global-p", VALID_PRICE, 5)],
                5,
            ),
        );
        sup.load_once().await.unwrap();
        let snap = sup.handle().load();
        assert_eq!(snap.models.len(), 1);
        assert_eq!(snap.pricing.len(), 1);
        assert_eq!(snap.global_pricing.len(), 1);
    }

    #[tokio::test]
    async fn a_row_both_prefixes_return_is_counted_once() {
        // The nesting case: with no `env_id` the environment prefix is
        // the bare base, so `<base>/global/` is inside it and the
        // catalog's rows come back from BOTH range reads. The snapshot
        // absorbs the repeat because it is keyed; `accepted` is a sum and
        // would report two rows where etcd holds one.
        let shared = entry("/aisix/global/pricing/p-1", VALID_PRICE, 4);
        let sup = Arc::new(Supervisor::with_sources(
            vec![
                (
                    WatchedPrefix::environment("/aisix"),
                    ScopedProvider::serving(
                        vec![entry("/aisix/models/m-1", VALID_MODEL, 3), shared.clone()],
                        4,
                    ),
                ),
                (
                    WatchedPrefix::global("/aisix/global/"),
                    ScopedProvider::serving(vec![shared], 4),
                ),
            ],
            SnapshotCache::disabled(),
        ));

        let stats = sup.load_once().await.unwrap();
        assert_eq!(
            stats.accepted, 2,
            "one model and one price, counted once each"
        );
        let snap = sup.handle().load();
        assert_eq!(snap.models.len(), 1);
        // Resolved to the catalog table by the longer prefix, not to the
        // environment's, even though the outer prefix also returned it.
        assert_eq!(snap.global_pricing.len(), 1);
        assert_eq!(snap.pricing.len(), 0);
    }

    #[tokio::test]
    async fn applied_revision_is_the_minimum_across_prefixes() {
        // The prefixes are read in sequence, so the union is consistent
        // only as far as the EARLIEST read: a write that landed between
        // the two reads is missing from the union while sitting below the
        // later read's header revision. Reporting the higher one would
        // claim a write the gateway has not applied. Both orders, because
        // taking the LAST prefix's revision (or the FIRST, or the
        // environment's) passes one of them by accident.
        //
        // Every entry is written at revision 1 so the header revisions
        // are the only thing that can produce the expected value: the
        // resync raises the floor to the highest entry revision first,
        // and entries at 5 / 7 / 42 / 99 would supply the answer by
        // themselves — which is how the first version of this case
        // stayed green against a last-prefix-wins mutation.
        let global_ahead = scoped_supervisor(
            ScopedProvider::serving(vec![entry("/aisix/env-1/models/m-1", VALID_MODEL, 1)], 7),
            ScopedProvider::serving(vec![entry("/aisix/global/pricing/g", VALID_PRICE, 1)], 42),
        );
        global_ahead.load_once().await.unwrap();
        assert_eq!(global_ahead.watch_status().snapshot().revision, 7);

        let env_ahead = scoped_supervisor(
            ScopedProvider::serving(vec![entry("/aisix/env-1/models/m-1", VALID_MODEL, 1)], 99),
            ScopedProvider::serving(vec![entry("/aisix/global/pricing/g", VALID_PRICE, 1)], 5),
        );
        env_ahead.load_once().await.unwrap();
        assert_eq!(env_ahead.watch_status().snapshot().revision, 5);
    }

    #[tokio::test]
    async fn a_later_prefixs_newer_row_does_not_raise_the_applied_revision() {
        // The resync stamps `max(entry revision)` from the union, and the
        // later-read prefix can legitimately hold a row written AFTER the
        // earlier prefix was read — while a write to the earlier prefix in
        // that same interval is missing from the union entirely. Letting
        // the entry-max stand would keep claiming a revision the gateway
        // has not seen, which is exactly what taking the minimum header is
        // for. Distinct from the case above: here the overstatement comes
        // from an ENTRY, not from a header.
        let sup = scoped_supervisor(
            ScopedProvider::serving(vec![entry("/aisix/env-1/models/m-1", VALID_MODEL, 90)], 100),
            ScopedProvider::serving(
                vec![entry("/aisix/global/pricing/g", VALID_PRICE, 103)],
                105,
            ),
        );
        sup.load_once().await.unwrap();
        assert_eq!(
            sup.watch_status().snapshot().revision,
            100,
            "the union is consistent as of the earliest read, whatever revision \
             a later prefix's rows carry"
        );
    }

    #[tokio::test]
    async fn a_refused_prefix_does_not_lower_the_applied_revision() {
        // A refusal contributes no rows, so nothing in the union came
        // from that prefix and it has no earliest-read point to impose.
        // Reading its missing revision as 0 would freeze
        // `applied_revision` at the floor for the life of the deployment.
        let sup = scoped_supervisor(
            ScopedProvider::serving(vec![entry("/aisix/env-1/models/m-1", VALID_MODEL, 1)], 31),
            ScopedProvider::refusing(),
        );
        sup.load_once().await.unwrap();
        assert_eq!(sup.watch_status().snapshot().revision, 31);
    }

    #[tokio::test]
    async fn a_refused_catalog_keeps_the_prices_it_last_read() {
        // A denial is not a deletion. Once the catalog HAS been read, a
        // later refusal must leave those prices in place: dropping them
        // silently stops `least_cost` ranking and empties `cost_usd` on
        // realtime and batch usage events, with only the WARN to show for
        // it. A successful read replaces them; only a successful read does.
        const OTHER_PRICE: &[u8] =
            br#"{"key":"openai/gpt-4o","input_per_1k":0.001,"output_per_1k":0.002}"#;
        let sup = scoped_supervisor(
            ScopedProvider::serving(vec![entry("/aisix/env-1/models/m-1", VALID_MODEL, 1)], 5),
            ScopedProvider::scripted(vec![
                Some((vec![entry("/aisix/global/pricing/g-1", VALID_PRICE, 3)], 3)),
                None,
                Some((vec![entry("/aisix/global/pricing/g-2", OTHER_PRICE, 9)], 9)),
            ]),
        );

        sup.load_once().await.unwrap();
        let priced = sup.handle().load();
        assert_eq!(priced.global_pricing.len(), 1);
        assert!(priced.global_pricing.get_by_id("g-1").is_some());

        // Denied: the rows stay, and the environment is untouched.
        sup.load_once().await.unwrap();
        let denied = sup.handle().load();
        assert_eq!(
            denied.global_pricing.len(),
            1,
            "a refusal must not clear the catalog"
        );
        assert!(denied.global_pricing.get_by_id("g-1").is_some());
        assert_eq!(denied.models.len(), 1);

        // Allowed again with different content: the retained rows are
        // replaced by what the read returned, not merged with it.
        sup.load_once().await.unwrap();
        let replaced = sup.handle().load();
        assert_eq!(replaced.global_pricing.len(), 1);
        assert!(
            replaced.global_pricing.get_by_id("g-1").is_none(),
            "a successful read replaces the retained rows"
        );
        assert!(replaced.global_pricing.get_by_id("g-2").is_some());
    }

    #[tokio::test]
    async fn a_broader_successful_read_still_deletes_a_retained_row() {
        // The nesting case again: with no `env_id` the environment prefix
        // is the bare base, so a SUCCESSFUL read of it already answers for
        // the catalog's keys — including answering that one is gone. The
        // catalog prefix being refused must not undo that: carrying the
        // old row over would bring a deleted price back to life, at
        // whatever it used to cost.
        let priced = entry("/aisix/global/pricing/p-1", VALID_PRICE, 4);
        let sup = Arc::new(Supervisor::with_sources(
            vec![
                (
                    WatchedPrefix::environment("/aisix"),
                    ScopedProvider::scripted(vec![
                        Some((
                            vec![entry("/aisix/models/m-1", VALID_MODEL, 3), priced.clone()],
                            4,
                        )),
                        // The price is deleted; the environment read that
                        // covers it comes back without it.
                        Some((vec![entry("/aisix/models/m-1", VALID_MODEL, 3)], 9)),
                    ]),
                ),
                (
                    WatchedPrefix::global("/aisix/global/"),
                    ScopedProvider::scripted(vec![Some((vec![priced], 4)), None]),
                ),
            ],
            SnapshotCache::disabled(),
        ));

        sup.load_once().await.unwrap();
        assert_eq!(sup.handle().load().global_pricing.len(), 1);

        sup.load_once().await.unwrap();
        assert_eq!(
            sup.handle().load().global_pricing.len(),
            0,
            "a deleted price must stay deleted when a read that covers it succeeded"
        );
    }

    #[tokio::test]
    async fn a_refused_catalog_leaves_the_environment_serving() {
        // The old-control-plane case: the catalog prefix is denied, the
        // environment's is not. The gateway must come up on the
        // environment's configuration rather than fail its cycle — the
        // alternative is serving nothing at all instead of serving
        // without prices.
        let sup = scoped_supervisor(
            ScopedProvider::serving(
                vec![
                    entry("/aisix/env-1/models/m-1", VALID_MODEL, 11),
                    entry("/aisix/env-1/api_keys/k-1", VALID_APIKEY, 12),
                ],
                12,
            ),
            ScopedProvider::refusing(),
        );

        // Completes rather than erroring — readiness is not held back by
        // the refused prefix.
        let stats = sup.load_once().await.expect("environment load succeeds");
        assert_eq!(stats.accepted, 2);
        let snap = sup.handle().load();
        assert_eq!(snap.models.len(), 1);
        assert_eq!(snap.apikeys.len(), 1);
        // Read as empty, not as an error: no rejection is reported for a
        // prefix the gateway was never allowed to read.
        assert_eq!(snap.global_pricing.len(), 0);
        assert!(sup.recent_rejections().is_empty());
        // The revision floor comes from the prefix that did answer.
        assert_eq!(sup.watch_status().snapshot().revision, 12);
    }

    #[tokio::test]
    async fn one_watch_ending_ends_the_cycle() {
        // The reconnect trigger is a watch stream ending; `watch_loop`
        // re-enters `cycle`, which re-reads and re-opens every prefix.
        // `select_all` drops an exhausted stream and keeps polling the
        // survivors, so without the end being delivered as an item the
        // cycle would sit on the catalog's idle stream forever while the
        // environment's configuration froze — no resync, no error, and
        // nothing in `/status/config` to show it.
        //
        // The catalog stream here never yields and never ends, which is
        // its normal state: nobody writes prices most of the time. That
        // is what makes the bug reachable rather than theoretical.
        let sup = scoped_supervisor(
            ScopedProvider::serving(vec![entry("/aisix/env-1/models/m-1", VALID_MODEL, 1)], 7),
            ScopedProvider::quiet(vec![entry("/aisix/global/pricing/g", VALID_PRICE, 1)], 7),
        );
        let (_tx, rx) = tokio::sync::watch::channel(false);

        let cycle = tokio::time::timeout(Duration::from_secs(5), sup.cycle(&rx)).await;
        assert!(
            matches!(cycle, Ok(Ok(()))),
            "the environment watch ended, so the cycle must return and let \
             watch_loop reconnect; got {cycle:?}",
        );
        // Both prefixes did load before the cycle ended.
        let snap = sup.handle().load();
        assert_eq!(snap.models.len(), 1);
        assert_eq!(snap.global_pricing.len(), 1);
    }

    #[tokio::test]
    async fn each_prefix_resumes_from_its_own_read() {
        // The range reads run in sequence, so the later prefix reports a
        // higher revision. Resuming BOTH watches from the maximum would
        // skip every write to the earlier prefix made in between: absent
        // from its read, and before the point its watch begins. The
        // window is one range read wide and the loss is silent until the
        // next resync.
        let env =
            ScopedProvider::serving(vec![entry("/aisix/env-1/models/m-1", VALID_MODEL, 1)], 7);
        let global =
            ScopedProvider::serving(vec![entry("/aisix/global/pricing/g", VALID_PRICE, 1)], 42);
        let sup = scoped_supervisor(Arc::clone(&env), Arc::clone(&global));

        let (_tx, rx) = tokio::sync::watch::channel(false);
        sup.cycle(&rx).await.expect("cycle completes");

        assert_eq!(
            *env.watched_from.lock().unwrap(),
            Some(8),
            "the environment watch must resume from its OWN read (7), not from the maximum (42)",
        );
        assert_eq!(*global.watched_from.lock().unwrap(), Some(43));
        // The reported applied revision is the MINIMUM, for the same
        // reason the resume points differ: the union is only consistent
        // as far as the earliest read (`applied_revision_is_the_minimum_
        // across_prefixes`).
        assert_eq!(sup.watch_status().snapshot().revision, 7);
    }

    #[tokio::test]
    async fn a_refused_catalog_does_not_stop_the_watch_cycle() {
        // The watch half of the case above: the catalog's watch create is
        // refused too, and the cycle still runs to a clean end on the
        // environment's stream alone.
        let sup = scoped_supervisor(
            ScopedProvider::serving(vec![entry("/aisix/env-1/models/m-1", VALID_MODEL, 3)], 3),
            ScopedProvider::refusing(),
        );
        let (_tx, rx) = tokio::sync::watch::channel(false);
        // The environment stream is empty, so the cycle drains it and
        // returns Ok — a refusal on the catalog would have surfaced here
        // as SupervisorError::Provider.
        sup.cycle(&rx).await.expect("cycle survives the refusal");
        assert_eq!(sup.handle().load().models.len(), 1);
    }

    #[tokio::test]
    async fn the_environment_prefix_is_always_read_first() {
        // `tolerates` swallows a refusal on the catalog, so a refusal
        // that is really about credentials must reach the environment
        // prefix first — that one is not tolerated and is what fails the
        // cycle. Constructed catalog-first to prove the supervisor
        // reorders rather than trusting its caller.
        let sup = Arc::new(Supervisor::with_sources(
            vec![
                (
                    WatchedPrefix::global(GLOBAL_PREFIX),
                    ScopedProvider::refusing(),
                ),
                (
                    WatchedPrefix::environment(ENV_PREFIX),
                    ScopedProvider::refusing(),
                ),
            ],
            SnapshotCache::disabled(),
        ));
        assert!(
            matches!(sup.load_once().await, Err(ProviderError::Rejected(_))),
            "a refusal on both prefixes must surface as the environment's",
        );
    }

    #[tokio::test]
    async fn a_refused_environment_prefix_still_fails_the_cycle() {
        // The tolerance is scoped to the catalog. Credentials etcd
        // refuses outright must still stop the cycle and reach the
        // backoff loop's error path, or a misconfigured gateway would
        // quietly serve an empty configuration forever.
        let sup = scoped_supervisor(ScopedProvider::refusing(), ScopedProvider::refusing());
        assert!(matches!(
            sup.load_once().await,
            Err(ProviderError::Rejected(_))
        ));
    }

    /// The two config digests must stay byte-identical to what the
    /// unchanged `hash_entries` produces over the same served set: cp-api
    /// stores `config_hash` verbatim and never recomputes it, and the
    /// algorithm is documented in the public Admin API reference. The
    /// fixture drives the three shapes the cached-record path has to get
    /// right at once — an accepted row, a row rejected with nothing to
    /// serve, and a row serving its pinned last known good while etcd
    /// holds bytes that do not load.
    #[tokio::test]
    async fn config_digests_match_the_reference_algorithm() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        assert!(sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 2)));
        assert!(sup.apply_put(&entry("/aisix/api_keys/k-1", VALID_APIKEY, 3)));
        assert!(sup.apply_put(&entry("/aisix/guardrails/g-1", VALID_GUARDRAIL, 4)));
        // Rejected with no previous good value: serves nothing.
        assert!(!sup.apply_put(&entry("/aisix/models/m-2", b"{\"display_name\": 7}", 5)));
        // Rejected over a row that WAS serving: keeps serving the pin.
        // Deliberately a key that sorts in the MIDDLE of the served set —
        // the pinned record has to take the key's own place in the
        // ordering, and a digest over the right bytes in the wrong order
        // is a different digest.
        assert!(!sup.apply_put(&entry("/aisix/guardrails/g-1", b"not json at all", 6)));

        let view = sup.config_status().view();
        let state: Vec<(String, Vec<u8>)> = sup
            .state
            .lock()
            .unwrap()
            .entries
            .values()
            .map(|e| (e.entry.key.clone(), e.entry.value.clone()))
            .collect();
        let stale = sup.stale_serving.lock().unwrap().clone();
        assert_eq!(
            stale.len(),
            1,
            "the pinned row is what makes the two differ"
        );
        let rejected: HashSet<String> = sup
            .rejections
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.key.clone())
            .collect();
        assert_eq!(rejected.len(), 2);

        let expected_source = aisix_core::config_status::hash_entries(
            state.iter().map(|(k, v)| (k.as_str(), v.as_slice())),
        );
        let expected_config = aisix_core::config_status::hash_entries(
            state
                .iter()
                .filter(|(k, _)| !rejected.contains(k) && !stale.contains_key(k))
                .map(|(k, v)| (k.as_str(), v.as_slice()))
                .chain(
                    stale
                        .values()
                        .map(|s| (s.entry.key.as_str(), s.entry.value.as_slice())),
                ),
        );
        assert_eq!(
            view.source.source_hash.as_deref(),
            Some(expected_source.as_str())
        );
        assert_eq!(view.applied.as_ref().unwrap().config_hash, expected_config);
        assert_ne!(expected_source, expected_config);

        // The STALE map, not the capped rejection buffer, decides which
        // keys substitute their pinned bytes. Drop the pinned key's
        // rejection — what a buffer overflow does — and the served bytes
        // must not flip to the ones that do not load.
        assert!(sup.remove_rejection_for_key("/aisix/guardrails/g-1"));
        sup.sync_config_status(false);
        let expected_overflow = aisix_core::config_status::hash_entries(
            state
                .iter()
                .filter(|(k, _)| k != "/aisix/models/m-2" && !stale.contains_key(k))
                .map(|(k, v)| (k.as_str(), v.as_slice()))
                .chain(
                    stale
                        .values()
                        .map(|s| (s.entry.key.as_str(), s.entry.value.as_slice())),
                ),
        );
        assert_eq!(expected_overflow, expected_config);
        assert_eq!(
            sup.config_status()
                .view()
                .applied
                .as_ref()
                .unwrap()
                .config_hash,
            expected_overflow,
        );
    }

    /// A watch event must not deep-copy the configuration. The published
    /// snapshot shares every row it did not change with its predecessor —
    /// which is also what makes a derived cache able to tell an unchanged
    /// row from a rewritten one by `Arc` identity.
    #[tokio::test]
    async fn a_put_shares_every_untouched_row_with_the_previous_snapshot() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        assert!(sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 2)));
        assert!(sup.apply_put(&entry("/aisix/guardrails/g-1", VALID_GUARDRAIL, 3)));

        let before = sup.handle().load();
        let model_before = before.models.get_by_id("m-1").unwrap();
        let guardrail_before = before.guardrails.get_by_id("g-1").unwrap();

        assert!(sup.apply_put(&entry("/aisix/api_keys/k-1", VALID_APIKEY, 4)));

        let after = sup.handle().load();
        assert!(
            !Arc::ptr_eq(&before, &after),
            "a new snapshot was published"
        );
        assert!(Arc::ptr_eq(
            &model_before,
            &after.models.get_by_id("m-1").unwrap()
        ));
        assert!(Arc::ptr_eq(
            &guardrail_before,
            &after.guardrails.get_by_id("g-1").unwrap()
        ));
    }

    /// A write bumps the generation of the table it touched and of no
    /// other, across the copy-on-write publish. This is the invalidation
    /// key every derived cache reads (AISIX-Cloud#1542).
    #[tokio::test]
    async fn only_the_written_table_changes_generation() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        assert!(sup.apply_put(&entry("/aisix/guardrails/g-1", VALID_GUARDRAIL, 2)));

        let before = sup.handle().load();
        let (g0, m0) = (before.guardrails.generation(), before.models.generation());

        assert!(sup.apply_put(&entry("/aisix/api_keys/k-1", VALID_APIKEY, 3)));
        let after = sup.handle().load();
        assert!(after.apikeys.generation() > before.apikeys.generation());
        assert_eq!(after.guardrails.generation(), g0);
        assert_eq!(after.models.generation(), m0);
        assert!(
            sup.handle().version() > 0,
            "the global version still moves on every publish"
        );

        assert!(sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 4)));
        let latest = sup.handle().load();
        assert!(latest.models.generation() > m0);
        assert_eq!(latest.guardrails.generation(), g0);

        // A delete moves it too.
        assert!(sup.apply_delete("/aisix/guardrails/g-1"));
        assert!(sup.handle().load().guardrails.generation() > g0);
    }

    /// A run of watch events that is already queued is applied as ONE
    /// copy-on-write cycle: one publish, one status sync, one cache
    /// flush. Per-event decisions still run per event and in order.
    #[tokio::test]
    async fn a_queued_run_of_events_publishes_once() {
        let events: Vec<Result<WatchEvent, ProviderError>> = (1..=6)
            .map(|i| {
                Ok(WatchEvent::Put(entry(
                    &format!("/aisix/models/m-{i}"),
                    VALID_MODEL,
                    i,
                )))
            })
            .collect();
        let provider = Arc::new(FakeProvider::new(vec![], 0).with_events(events));
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        let version_before = sup.handle().version();
        let reloads_before = sup.config_status().metrics().reloads_total;

        let (_tx, rx) = tokio::sync::watch::channel(false);
        // One cycle: load_all, then drain the whole prepared stream.
        let _ = sup.cycle(&rx).await;

        let snap = sup.handle().load();
        assert_eq!(snap.models.len(), 6);
        // The initial resync publishes once; the six puts publish once
        // between them. Before the coalescing they published six times.
        assert_eq!(
            sup.handle().version() - version_before,
            2,
            "the queued run should be one publish on top of the resync",
        );
        assert_eq!(
            sup.config_status().metrics().reloads_total - reloads_before,
            1,
            "only the resync counts as a reload",
        );
    }

    /// A put after a delete of the same id must not be folded into the
    /// staged snapshot that preceded the delete — the delete would then
    /// remove the row the later put re-added, and it would be gone.
    #[tokio::test]
    async fn a_delete_breaks_the_run_of_puts_it_sits_between() {
        let events: Vec<Result<WatchEvent, ProviderError>> = vec![
            Ok(WatchEvent::Put(entry("/aisix/models/m-1", VALID_MODEL, 2))),
            Ok(WatchEvent::Put(entry(
                "/aisix/api_keys/k-1",
                VALID_APIKEY,
                3,
            ))),
            Ok(WatchEvent::Delete {
                key: "/aisix/models/m-1".into(),
                revision: 4,
            }),
            Ok(WatchEvent::Put(entry("/aisix/models/m-1", VALID_MODEL, 5))),
            Ok(WatchEvent::Put(entry(
                "/aisix/guardrails/g-1",
                VALID_GUARDRAIL,
                6,
            ))),
        ];
        let provider = Arc::new(FakeProvider::new(vec![], 0).with_events(events));
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        let (_tx, rx) = tokio::sync::watch::channel(false);
        let _ = sup.cycle(&rx).await;

        let snap = sup.handle().load();
        assert_eq!(snap.models.len(), 1, "the put after the delete survived");
        assert_eq!(snap.apikeys.len(), 1);
        assert_eq!(snap.guardrails.len(), 1);
    }

    #[tokio::test]
    async fn disabled_cache_skips_encoding_and_background_writes() {
        let provider = Arc::new(FakeProvider::new(
            vec![entry("/aisix/models/m-1", VALID_MODEL, 1)],
            1,
        ));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        assert!(sup.pending_writes.lock().unwrap().is_empty());
        assert!(sup
            .state
            .lock()
            .unwrap()
            .entries
            .values()
            .all(|row| row.cache_record.get().is_none()));

        sup.apply_put(&entry("/aisix/models/m-1", b"not json", 2));
        assert_eq!(sup.handle().load().models.len(), 1);
        assert_eq!(sup.stale_serving.lock().unwrap().len(), 1);
        assert!(sup
            .stale_serving
            .lock()
            .unwrap()
            .values()
            .all(|row| row.cache_record.get().is_none()));
        sup.apply_delete("/aisix/models/m-1");
        assert_eq!(sup.handle().load().models.len(), 0);
        assert!(sup.pending_writes.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn cache_records_are_shared_until_their_entry_changes() {
        let dir = tempfile::tempdir().unwrap();
        let cache = SnapshotCache::new(dir.path().join("snap.json"));
        let provider = Arc::new(FakeProvider::new(
            vec![
                entry("/aisix/models/m-1", VALID_MODEL, 1),
                entry("/aisix/api_keys/k-1", VALID_APIKEY, 1),
            ],
            1,
        ));
        let sup = Supervisor::with_cache(provider, "/aisix", cache.clone());
        sup.load_once().await.unwrap();
        let records = || {
            let state = sup.state.lock().unwrap();
            (
                state.entries["/aisix/models/m-1"].cache_record(),
                state.entries["/aisix/api_keys/k-1"].cache_record(),
            )
        };
        let before = records();
        sup.apply_put(&entry("/aisix/models/m-1", b"not json", 2));
        let after = records();
        assert!(!Arc::ptr_eq(&before.0, &after.0));
        assert!(Arc::ptr_eq(&before.1, &after.1));
        let pinned = sup.stale_serving.lock().unwrap()["/aisix/models/m-1"].cache_record();
        sup.apply_put(&entry("/aisix/api_keys/k-1", VALID_APIKEY, 3));
        assert!(Arc::ptr_eq(
            &pinned,
            &sup.stale_serving.lock().unwrap()["/aisix/models/m-1"].cache_record()
        ));
        sup.await_pending_cache_writes().await;
        let cached = cache.load().unwrap();
        assert_eq!(cached.revision, 3);
        assert_eq!(cached.entries.len(), 2);
        assert_eq!(cached.stale.len(), 1);
        assert_eq!(cached.stale[0].entry.value, VALID_MODEL);
    }

    /// A delete for a key the gateway never held changes what
    /// `/status/config` reports (the revision floor moved) and nothing
    /// about the observed entry set — so it must not rewrite the on-disk
    /// snapshot cache, which is what it did before the two were split.
    #[tokio::test]
    async fn a_delete_of_an_unknown_key_publishes_status_without_writing_the_cache() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let dir = tempfile::tempdir().unwrap();
        let sup = Supervisor::with_cache(
            provider,
            "/aisix",
            SnapshotCache::new(dir.path().join("snap.json")),
        );
        sup.load_once().await.unwrap();
        sup.await_pending_cache_writes().await;

        assert!(
            !sup.apply_events(&[PendingEvent::Delete {
                key: "/aisix/models/never-existed",
                revision: Some(99),
            }])[0]
        );

        assert!(
            sup.pending_writes.lock().unwrap().is_empty(),
            "a cache write was spawned for a key that changed no state",
        );
        // Status WAS published: the revision floor the delete carried is
        // what `/status/config` and the heartbeat now report.
        let view = sup.config_status().view();
        assert_eq!(view.source.observed_revision, Some(99));
        assert_eq!(view.applied.unwrap().applied_revision, Some(99));

        // A delete that DOES change state still writes.
        assert!(sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 100)));
        sup.await_pending_cache_writes().await;
        assert!(
            sup.apply_events(&[PendingEvent::Delete {
                key: "/aisix/models/m-1",
                revision: Some(101),
            }])[0]
        );
        assert!(!sup.pending_writes.lock().unwrap().is_empty());
    }

    /// A decision that asks "does this row serve right now?" must see the
    /// batch's own earlier events, which have not been published yet —
    /// otherwise coalescing silently changes what the same events decide.
    #[tokio::test]
    async fn a_rejected_put_pins_the_accepted_put_before_it_in_the_same_batch() {
        let events: Vec<Result<WatchEvent, ProviderError>> = vec![
            Ok(WatchEvent::Put(entry("/aisix/models/m-1", VALID_MODEL, 2))),
            Ok(WatchEvent::Put(entry(
                "/aisix/api_keys/k-1",
                VALID_APIKEY,
                3,
            ))),
            // Rejected AFTER the accepted put of the same key: the row is
            // serving as far as this decision goes, even though the
            // snapshot carrying it has not been published yet, so its
            // bytes are pinned as the last known good.
            Ok(WatchEvent::Put(entry(
                "/aisix/models/m-1",
                b"not json at all",
                4,
            ))),
            Ok(WatchEvent::Delete {
                key: "/aisix/api_keys/k-1".into(),
                revision: 5,
            }),
        ];
        let provider = Arc::new(FakeProvider::new(vec![], 0).with_events(events));
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        let (_tx, rx) = tokio::sync::watch::channel(false);
        let _ = sup.cycle(&rx).await;

        let snap = sup.handle().load();
        assert_eq!(snap.models.len(), 1, "the pinned last known good serves");
        assert_eq!(snap.apikeys.len(), 0, "the delete landed");
        let stale = sup.stale_serving.lock().unwrap();
        assert!(
            stale.contains_key("/aisix/models/m-1"),
            "the earlier put in the same batch counted as serving",
        );
    }

    /// A provider whose watch stream stays OPEN and delivers whatever the
    /// test feeds it, one event at a time. `FakeProvider` hands over a
    /// finished `stream::iter`, so every event on it is ready at once and
    /// no coalescing decision is ever exercised — which is exactly the
    /// shape a real etcd/kine watch does NOT have.
    struct LiveProvider {
        entries: Vec<RawEntry>,
        revision: i64,
        rx: Mutex<Option<futures::channel::mpsc::UnboundedReceiver<WatchItem>>>,
    }

    type WatchItem = Result<WatchEvent, ProviderError>;

    impl LiveProvider {
        fn new(
            revision: i64,
        ) -> (
            Arc<Self>,
            futures::channel::mpsc::UnboundedSender<WatchItem>,
        ) {
            let (tx, rx) = futures::channel::mpsc::unbounded();
            (
                Arc::new(Self {
                    entries: Vec::new(),
                    revision,
                    rx: Mutex::new(Some(rx)),
                }),
                tx,
            )
        }
    }

    #[async_trait]
    impl ConfigProvider for LiveProvider {
        async fn load_all(&self) -> Result<(Vec<RawEntry>, i64), ProviderError> {
            Ok((self.entries.clone(), self.revision))
        }

        async fn watch(
            &self,
            _start_revision: i64,
        ) -> Result<
            Box<dyn futures::Stream<Item = Result<WatchEvent, ProviderError>> + Send + Unpin>,
            ProviderError,
        > {
            Ok(Box::new(
                self.rx.lock().unwrap().take().expect("watched twice"),
            ))
        }
    }

    #[tokio::test(start_paused = true)]
    async fn coalescing_cost_expires_and_never_extends_the_maximum_wait() {
        let mut timing = ApplyTiming::default();
        assert_eq!(timing.quiet_period(), Duration::from_millis(20));

        timing.record(Duration::from_millis(90));
        tokio::time::advance(Duration::from_millis(70)).await;
        assert_eq!(timing.quiet_period(), Duration::from_millis(90));

        timing.record(Duration::from_secs(1));
        assert_eq!(timing.quiet_period(), Duration::from_millis(150));
        tokio::time::advance(Duration::from_millis(150)).await;
        assert_eq!(timing.quiet_period(), Duration::from_millis(20));

        timing.record(Duration::from_millis(90));
        timing.record(Duration::from_millis(1));
        assert_eq!(timing.quiet_period(), Duration::from_millis(20));
    }

    /// A burst that arrives one event at a time — how the control plane's
    /// outbox relay actually reaches a data plane, since it writes each
    /// row as its own transaction — must still cost a bounded number of
    /// whole-configuration passes. The window, not the endpoint's
    /// buffering, is what decides how many.
    ///
    /// Virtual time (`start_paused`) is what makes the count exact: the
    /// runtime advances the clock only when every task is parked on a
    /// timer, so the 1 ms spacing below is honoured precisely and the
    /// assertion is not a race against a loaded CI box.
    #[tokio::test(start_paused = true)]
    async fn a_burst_delivered_one_event_at_a_time_still_coalesces() {
        const N: i64 = 250;
        const SPACING: Duration = Duration::from_millis(1);

        let (provider, tx) = LiveProvider::new(0);
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        let version_before = sup.handle().version();
        let (_cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

        let watcher = tokio::spawn({
            let sup = sup.clone();
            async move {
                let _ = sup.cycle(&cancel_rx).await;
            }
        });

        for i in 1..=N {
            tokio::time::sleep(SPACING).await;
            tx.unbounded_send(Ok(WatchEvent::Put(entry(
                &format!("/aisix/models/m-{i}"),
                VALID_MODEL,
                i + 1,
            ))))
            .unwrap();
        }
        drop(tx);
        watcher.await.unwrap();

        let snap = sup.handle().load();
        assert_eq!(snap.models.len(), N as usize, "every event was applied");
        let version_after = sup.handle().version();
        assert!(
            version_after > version_before,
            "the resync alone should have published once",
        );
        // One publish for the initial resync, then one per coalesced
        // apply. Un-coalesced this burst is 250 of them.
        let publishes = version_after - version_before - 1;
        let burst_ms = (N as u128) * SPACING.as_millis();
        assert!(
            publishes <= 3,
            "a {N}-event burst spread over {burst_ms}ms should coalesce into at most 3 applies, \
             got {publishes}",
        );
        // Both ends matter, and the lower one is what pins
        // COALESCE_MAX_WAIT. Every gap here is under the quiet period, so
        // if the max wait stopped bounding the batch the whole burst
        // would collapse into a single apply and the upper bound above
        // would happily pass — while config-change visibility during a
        // sustained burst grew without limit. The burst spans more than
        // one max wait, so at least one window has to close mid-burst.
        assert!(
            burst_ms > COALESCE_MAX_WAIT.as_millis(),
            "the burst has to outlast one window or the bound below proves nothing",
        );
        assert!(
            publishes >= 2,
            "the maximum wait did not bound the batch: {publishes} apply(s) for a {burst_ms}ms \
             burst of events spaced under the quiet period",
        );
    }

    /// The window must not strand a lone write waiting for company that
    /// never comes: a single event on an otherwise idle stream is applied
    /// well inside the maximum wait, because the QUIET period is what
    /// releases it.
    #[tokio::test(start_paused = true)]
    async fn a_lone_event_is_applied_within_the_quiet_period() {
        /// Slack for the 1 ms poll below, which can only overshoot the
        /// moment the apply landed by one of its own ticks.
        const POLL_SLACK: Duration = Duration::from_millis(5);

        let (provider, tx) = LiveProvider::new(0);
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        let (_cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

        let watcher = tokio::spawn({
            let sup = sup.clone();
            async move {
                let _ = sup.cycle(&cancel_rx).await;
            }
        });

        // Let the initial load land before starting the clock, so what is
        // measured is the coalescing window and nothing else. Bounded:
        // virtual time auto-advances forever, so an unbounded spin here
        // would hang rather than fail if the resync stopped publishing.
        let load_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while sup.handle().version() == 0 {
            assert!(
                tokio::time::Instant::now() < load_deadline,
                "the initial resync never published",
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        let sent_at = tokio::time::Instant::now();
        tx.unbounded_send(Ok(WatchEvent::Put(entry(
            "/aisix/models/m-1",
            VALID_MODEL,
            2,
        ))))
        .unwrap();

        // The sender stays alive, so the stream never ends and only the
        // window can trigger the apply. Bounded by the quiet period
        // rather than by the maximum wait: a lone event has no company
        // coming, so the quiet period is the branch that has to release
        // it, and a bound of COALESCE_MAX_WAIT would still pass with that
        // branch deleted — at 7x the latency on the commonest path of
        // all.
        while sup.handle().load().models.is_empty() {
            assert!(
                sent_at.elapsed() <= COALESCE_QUIET_PERIOD + POLL_SLACK,
                "a lone event waited {:?} — past the quiet period, so only the maximum wait \
                 released it",
                sent_at.elapsed(),
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(sent_at.elapsed() >= COALESCE_QUIET_PERIOD);

        drop(tx);
        watcher.await.unwrap();
    }

    /// Shutdown must not throw away what the window already took off the
    /// stream. Those events are gone from the watch, so the apply that
    /// follows the cancelled window is the only thing that can still put
    /// them in the served snapshot and the on-disk cache — the same
    /// invariant `an_apply_immediately_before_shutdown_reaches_the_on_disk_cache`
    /// protects, reached by the second path the window opened.
    #[tokio::test(start_paused = true)]
    async fn a_cancel_inside_the_window_still_applies_what_it_collected() {
        let (provider, tx) = LiveProvider::new(0);
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

        let watcher = tokio::spawn({
            let sup = sup.clone();
            async move { sup.cycle(&cancel_rx).await }
        });
        let load_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while sup.handle().version() == 0 {
            assert!(
                tokio::time::Instant::now() < load_deadline,
                "the initial resync never published",
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        tx.unbounded_send(Ok(WatchEvent::Put(entry(
            "/aisix/models/m-1",
            VALID_MODEL,
            2,
        ))))
        .unwrap();
        // Long enough for the watcher to take the event into its batch,
        // far short of the quiet period that would close the window.
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert!(
            sup.handle().load().models.is_empty(),
            "the window must still be open here or this test proves nothing",
        );

        cancel_tx.send(true).unwrap();
        assert!(matches!(
            watcher.await.unwrap(),
            Err(SupervisorError::Cancelled)
        ));
        assert_eq!(
            sup.handle().load().models.len(),
            1,
            "the batch already taken off the stream was dropped on shutdown",
        );
    }

    /// ...and it must stop taking new ones. A bulk edit leaves an event
    /// ready on the stream at every poll, which is the one case where
    /// `biased` ordering decides whether the cancel branch is ever
    /// reached at all: ordered after the stream it never is, and
    /// shutdown keeps draining the backlog until the max wait or
    /// MAX_APPLY_BATCH closes the window.
    #[tokio::test(start_paused = true)]
    async fn a_cancel_stops_the_window_from_draining_a_ready_backlog() {
        const BACKLOG: i64 = 60;

        let (provider, tx) = LiveProvider::new(0);
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

        let watcher = tokio::spawn({
            let sup = sup.clone();
            async move { sup.cycle(&cancel_rx).await }
        });
        let load_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while sup.handle().version() == 0 {
            assert!(
                tokio::time::Instant::now() < load_deadline,
                "the initial resync never published",
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // Open a window with one event, then queue the rest without ever
        // yielding, so they are all ready when the watcher next polls.
        tx.unbounded_send(Ok(WatchEvent::Put(entry(
            "/aisix/models/m-1",
            VALID_MODEL,
            2,
        ))))
        .unwrap();
        tokio::time::sleep(Duration::from_millis(1)).await;
        for i in 2..=BACKLOG {
            tx.unbounded_send(Ok(WatchEvent::Put(entry(
                &format!("/aisix/models/m-{i}"),
                VALID_MODEL,
                i + 1,
            ))))
            .unwrap();
        }
        cancel_tx.send(true).unwrap();

        assert!(matches!(
            watcher.await.unwrap(),
            Err(SupervisorError::Cancelled)
        ));
        let applied = sup.handle().load().models.len();
        assert!(
            applied >= 1,
            "the event already taken off the stream was dropped on shutdown",
        );
        assert!(
            (applied as i64) < BACKLOG,
            "shutdown drained the whole ready backlog ({applied} events) instead of leaving the \
             window at the first cancel check",
        );
    }

    #[tokio::test]
    async fn load_once_publishes_initial_snapshot() {
        let provider = Arc::new(FakeProvider::new(
            vec![entry("/aisix/models/m-1", VALID_MODEL, 1)],
            5,
        ));
        let sup = Supervisor::new(provider, "/aisix");
        let stats = sup.load_once().await.unwrap();
        assert_eq!(stats.accepted, 1);
        let snap = sup.handle().load();
        assert_eq!(snap.models.len(), 1);
    }

    #[tokio::test]
    async fn apply_put_adds_to_snapshot() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        assert!(sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 2)));
        assert_eq!(sup.handle().load().models.len(), 1);
    }

    /// Regression for the supervisor `apply_put` / `clone_snapshot`
    /// drift: every kind on `AisixSnapshot` must be mergeable on a
    /// watch event, otherwise admin writes for those resources land
    /// in etcd but never reach the proxy snapshot. Smoke test #102
    /// hit this for ProviderKey — the proxy saw the Model fine but
    /// `dispatch::resolve_provider_key` blew up because the PK was
    /// invisible to the watch path.
    #[tokio::test]
    async fn apply_put_propagates_every_resource_kind() {
        const VALID_PROVIDER_KEY: &[u8] = br#"{
            "display_name": "watch-pk",
            "secret": "sk-watch"
        }"#;
        const VALID_GUARDRAIL: &[u8] = br#"{
            "name": "watch-block",
            "kind": "keyword",
            "patterns": [{"kind": "literal", "value": "x"}]
        }"#;
        const VALID_CACHE_POLICY: &[u8] = br#"{
            "name": "watch-cache",
            "enabled": true
        }"#;
        const VALID_OBSERVABILITY_EXPORTER: &[u8] = br#"{
            "name": "watch-otel",
            "kind": "otlp_http",
            "endpoint": "https://otel.example.com/v1/traces"
        }"#;
        // A guardrail attachment created mid-run (the #826 model-scope
        // path). Before the fix this kind was missing from apply_put's
        // merge loop, so the row was parsed but dropped — the proxy then
        // fell back to implicit-env scope and enforced the guardrail on
        // EVERY model instead of the scoped one.
        const VALID_GUARDRAIL_ATTACHMENT: &[u8] = br#"{
            "guardrail_id": "g-1",
            "scope_type": "model",
            "scope_id": "m-1",
            "priority": 100
        }"#;
        // An OIDC trust provider created mid-run (AISIX-Cloud#1080).
        // Same trap as #826: the kind existed in the loader but was
        // initially missing from apply_put's merge loop, so enabling
        // JWT auth via watch silently never took effect until resync.
        const VALID_OIDC_PROVIDER: &[u8] = br#"{
            "name": "watch-idp",
            "issuer": "https://idp.example.com/realms/agents",
            "audiences": ["aisix-gateway"]
        }"#;
        // The MCP OAuth discovery settings row (AISIX-Cloud#1143) —
        // configuring the resource URL mid-run must activate the
        // discovery surface without a resync.
        const VALID_MCP_AUTH_SETTINGS: &[u8] = br#"{
            "resource_url": "https://gw.example.com/mcp"
        }"#;
        // A claim mapping created mid-run (AISIX-Cloud#564) — same
        // guard: a rule added via watch must be live without a resync.
        const VALID_CLAIM_MAPPING: &[u8] = br#"{
            "name": "watch-rule",
            "jwt_provider": "watch-idp",
            "match": [{"claim": "department", "op": "exact", "values": ["finance"]}],
            "resolve": {"api_key_id": "ak-1"}
        }"#;

        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        for (key, body, _kind) in [
            ("/aisix/provider_keys/pk-1", VALID_PROVIDER_KEY, "PK"),
            ("/aisix/guardrails/g-1", VALID_GUARDRAIL, "Guardrail"),
            (
                "/aisix/guardrail_attachments/ga-1",
                VALID_GUARDRAIL_ATTACHMENT,
                "GuardrailAttachment",
            ),
            (
                "/aisix/cache_policies/cp-1",
                VALID_CACHE_POLICY,
                "CachePolicy",
            ),
            (
                "/aisix/observability_exporters/oe-1",
                VALID_OBSERVABILITY_EXPORTER,
                "ObservabilityExporter",
            ),
            (
                "/aisix/oidc_providers/op-1",
                VALID_OIDC_PROVIDER,
                "OidcProvider",
            ),
            (
                "/aisix/mcp_auth_settings/env-1",
                VALID_MCP_AUTH_SETTINGS,
                "McpAuthSettings",
            ),
            (
                "/aisix/claim_mappings/cm-1",
                VALID_CLAIM_MAPPING,
                "ClaimMapping",
            ),
        ] {
            assert!(
                sup.apply_put(&entry(key, body, 2)),
                "apply_put returned false for {key}"
            );
        }

        let snap = sup.handle().load();
        assert_eq!(snap.provider_keys.len(), 1, "ProviderKey not merged");
        assert_eq!(snap.guardrails.len(), 1, "Guardrail not merged");
        assert_eq!(
            snap.guardrail_attachments.len(),
            1,
            "GuardrailAttachment not merged"
        );
        assert_eq!(snap.cache_policies.len(), 1, "CachePolicy not merged");
        assert_eq!(
            snap.observability_exporters.len(),
            1,
            "ObservabilityExporter not merged"
        );
        assert_eq!(snap.oidc_providers.len(), 1, "OidcProvider not merged");
        assert_eq!(snap.claim_mappings.len(), 1, "ClaimMapping not merged");
        assert_eq!(
            snap.mcp_auth_settings.len(),
            1,
            "McpAuthSettings not merged"
        );
    }

    #[tokio::test]
    async fn apply_delete_removes_every_resource_kind() {
        let provider = Arc::new(FakeProvider::new(
            vec![
                entry(
                    "/aisix/provider_keys/pk-1",
                    br#"{"display_name":"x","secret":"y"}"#,
                    1,
                ),
                // #826: a watch delete for a guardrail attachment must
                // also reach the snapshot, or detaching a model-scope
                // never takes effect on the proxy.
                entry(
                    "/aisix/guardrail_attachments/ga-1",
                    br#"{"guardrail_id":"g-1","scope_type":"model","scope_id":"m-1","priority":100}"#,
                    1,
                ),
                // AISIX-Cloud#1080: deleting a trust provider must reach
                // the snapshot, or revoking JWT auth never takes effect.
                entry(
                    "/aisix/oidc_providers/op-1",
                    br#"{"name":"idp","issuer":"https://idp.example.com","audiences":["aisix"]}"#,
                    1,
                ),
                // AISIX-Cloud#1143: clearing the resource URL projects a
                // delete; it must deactivate the discovery surface.
                entry(
                    "/aisix/mcp_auth_settings/env-1",
                    br#"{"resource_url":"https://gw.example.com/mcp"}"#,
                    1,
                ),
                // AISIX-Cloud#564: deleting a claim mapping must reach
                // the snapshot, or revoking a rule never takes effect.
                entry(
                    "/aisix/claim_mappings/cm-1",
                    br#"{"name":"r","jwt_provider":"idp","match":[{"claim":"d","op":"exact","values":["v"]}],"resolve":{"api_key_id":"ak-1"}}"#,
                    1,
                ),
            ],
            1,
        ));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        assert_eq!(sup.handle().load().provider_keys.len(), 1);
        assert_eq!(sup.handle().load().guardrail_attachments.len(), 1);
        assert_eq!(sup.handle().load().oidc_providers.len(), 1);
        assert!(sup.apply_delete("/aisix/provider_keys/pk-1"));
        assert!(sup.handle().load().provider_keys.is_empty());
        assert!(sup.apply_delete("/aisix/guardrail_attachments/ga-1"));
        assert!(sup.handle().load().guardrail_attachments.is_empty());
        assert!(sup.apply_delete("/aisix/oidc_providers/op-1"));
        assert!(sup.handle().load().oidc_providers.is_empty());
        assert_eq!(sup.handle().load().claim_mappings.len(), 1);
        assert!(sup.apply_delete("/aisix/claim_mappings/cm-1"));
        assert!(sup.handle().load().claim_mappings.is_empty());
        assert!(sup.apply_delete("/aisix/mcp_auth_settings/env-1"));
        assert!(sup.handle().load().mcp_auth_settings.is_empty());
    }

    #[tokio::test]
    async fn apply_put_rejects_bad_payload_without_mutating() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        assert!(!sup.apply_put(&entry("/aisix/models/bad", b"not-json", 1)));
        assert!(sup.handle().load().models.is_empty());
    }

    #[tokio::test]
    async fn apply_put_rejects_semantically_invalid_policy_and_keeps_last_good() {
        // A conditional policy row that passes the JSON Schema but fails
        // the semantic gate (uncompilable regex) must behave exactly
        // like a schema failure on the watch path: apply_put returns
        // false, the previously-served row keeps serving, and the
        // rejection lands in the retained buffer for the heartbeat
        // (AISIX-Cloud#892 + #115).
        let good = br#"{
            "name": "premium",
            "conditions": [
                { "dimension": "model_name", "operator": "~~", "value": "^gpt-4" }
            ],
            "limits": { "rpm": 5 }
        }"#;
        let provider = Arc::new(FakeProvider::new(
            vec![entry("/aisix/rate_limit_policies/rlp-1", good, 1)],
            1,
        ));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        let bad = br#"{
            "name": "premium",
            "conditions": [
                { "dimension": "model_name", "operator": "~~", "value": "(unclosed" }
            ],
            "limits": { "rpm": 5 }
        }"#;
        assert!(!sup.apply_put(&entry("/aisix/rate_limit_policies/rlp-1", bad, 2)));

        // Last-good value keeps serving with its original tree.
        let snap = sup.handle().load();
        let served = snap.rate_limit_policies.get_by_id("rlp-1").unwrap();
        let tree = serde_json::to_value(served.value.conditions.as_ref().unwrap()).unwrap();
        assert_eq!(tree[0]["value"], "^gpt-4");
        // The rejection is retained for the next heartbeat.
        let rejected = sup.recent_rejections();
        assert!(
            rejected
                .iter()
                .any(|r| r.key == "/aisix/rate_limit_policies/rlp-1"
                    && r.error.contains("does not compile")),
            "{rejected:?}"
        );
    }

    #[tokio::test]
    async fn apply_delete_removes_entry() {
        let provider = Arc::new(FakeProvider::new(
            vec![entry("/aisix/models/m-1", VALID_MODEL, 1)],
            1,
        ));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        assert!(sup.apply_delete("/aisix/models/m-1"));
        assert!(sup.handle().load().models.is_empty());
    }

    #[tokio::test]
    async fn apply_resync_replaces_snapshot() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        sup.apply_resync(&[entry("/aisix/models/m-1", VALID_MODEL, 1)]);
        assert_eq!(sup.handle().load().models.len(), 1);
    }

    #[tokio::test]
    async fn run_loop_applies_put_then_exits_on_cancel() {
        let provider = Arc::new(FakeProvider::new(vec![], 0).with_events(vec![Ok(
            WatchEvent::Put(entry("/aisix/models/m-1", VALID_MODEL, 2)),
        )]));
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        let handle = sup.handle();
        let (tx, rx) = tokio::sync::watch::channel(false);

        let join = tokio::spawn(sup.clone().run(rx));

        // Let the supervisor drain its finite event stream and reach the
        // "stream ended" branch. The load + event apply both happen
        // synchronously relative to the event stream being in-memory.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(handle.load().models.len(), 1);

        tx.send(true).unwrap();
        join.await.unwrap();
    }

    /// #519 B.3: the cycle's Delete arm must advance the applied-
    /// revision floor to the delete event's mod_revision — without it
    /// the heartbeat-reported `applied_revision` stalls after a CP
    /// delete and the dashboard shows "propagating…" until an unrelated
    /// put arrives.
    #[tokio::test]
    async fn run_loop_advances_revision_on_delete_event() {
        let provider = Arc::new(FakeProvider::new(vec![], 2).with_events(vec![
            Ok(WatchEvent::Put(entry("/aisix/models/m-1", VALID_MODEL, 5))),
            Ok(WatchEvent::Delete {
                key: "/aisix/models/m-1".into(),
                revision: 9,
            }),
        ]));
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        let status = sup.watch_status();
        let (tx, rx) = tokio::sync::watch::channel(false);

        let join = tokio::spawn(sup.clone().run(rx));

        // Poll until the finite event stream drains (bounded — the
        // revision floor never decreases once it reaches 9).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while status.snapshot().revision < 9 && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(
            status.snapshot().revision,
            9,
            "delete event's mod_revision must advance the applied revision",
        );

        tx.send(true).unwrap();
        join.await.unwrap();
    }

    /// Rows enough to make persisting the configuration set real work
    /// rather than one scheduler tick. The write the shutdown drain has
    /// to catch is the serialisation and fsync of this much data; a
    /// handful of small rows finishes inside the first poll, and an
    /// assertion built on that would hold with or without the drain.
    const PADDING_ROWS: usize = 128;
    const PADDING_NAME_BYTES: usize = 96 * 1024;

    /// A valid model whose display name carries the padding, so the
    /// loader accepts every row and the test produces no rejection noise.
    fn padding_entry(i: usize) -> RawEntry {
        let name = format!("pad-{i}-{}", "x".repeat(PADDING_NAME_BYTES));
        let value = format!(
            r#"{{"display_name":"{name}","provider":"openai","model_name":"gpt-4o","provider_key_id":"11111111-1111-1111-1111-111111111111"}}"#
        );
        RawEntry {
            key: format!("/aisix/models/pad-{i}"),
            value: value.into_bytes(),
            revision: 1,
        }
    }

    /// `flush_cache` spawns the cache write detached so the apply path can
    /// stay sync, and nothing used to wait for it: a gateway stopped
    /// shortly after an apply exited with that write unfinished and came
    /// back without its last-known-good snapshot. That costs more since
    /// the proxy listener started gating on a first applied configuration
    /// — a gateway with no usable cache now refuses to bind at all until
    /// it reaches etcd, where it previously bound and served 401s.
    ///
    /// Shaped like the production sequence rather than around the fix: the
    /// supervisor's own `run` task applies the change, the runtime then
    /// goes away exactly as it does when `main` returns — dropping every
    /// spawned task at its await point — and the assertion is on what a
    /// restart would read off disk.
    #[test]
    fn an_apply_immediately_before_shutdown_reaches_the_on_disk_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("snap.json");

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            // A boot resync followed by a put: two writes, so the second
            // is still queued behind the first when shutdown arrives.
            // That backlog is what keeps a write genuinely IN FLIGHT —
            // a single write finishes on its own during the shutdown
            // path, and an assertion built on one would hold with the
            // drain removed. It is also the realistic shape: the control
            // plane writes several rows and the replica is stopped
            // straight after.
            let entries: Vec<RawEntry> = (0..PADDING_ROWS).map(padding_entry).collect();
            let provider = Arc::new(FakeProvider::new(entries, 1).with_events(vec![Ok(
                WatchEvent::Put(entry("/aisix/models/late", VALID_MODEL, 2)),
            )]));
            let sup = Arc::new(Supervisor::with_cache(
                provider,
                "/aisix",
                SnapshotCache::new(&cache_path),
            ));
            let (tx, rx) = tokio::sync::watch::channel(false);
            let join = tokio::spawn(sup.clone().run(rx));

            // Cancel the moment the put is visible in the served
            // snapshot. The flush it spawned is then still on its way to
            // disk, which is exactly the window this pins.
            for _ in 0..5_000 {
                if sup.handle().load().models.get_by_id("late").is_some() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            assert!(
                sup.handle().load().models.get_by_id("late").is_some(),
                "the put must reach the served snapshot before shutdown",
            );
            tx.send(true).unwrap();
            join.await.unwrap();
        });

        // What `main` returning does to a spawned write nobody waited for.
        drop(rt);

        let survived = SnapshotCache::new(&cache_path)
            .load()
            .is_some_and(|cached| cached.entries.iter().any(|e| e.key == "/aisix/models/late"));
        assert!(
            survived,
            "the last apply before shutdown must reach disk; undrained, its \
             write is still in flight when the runtime goes away",
        );
    }

    /// The drain must not become a way for shutdown to hang: a write that
    /// never finishes is abandoned, not waited on forever. Time is paused,
    /// so the bound is asserted without spending it.
    #[tokio::test(start_paused = true)]
    async fn the_shutdown_drain_abandons_a_write_that_never_finishes() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        // A stuck disk cannot be produced on demand through `flush_cache`,
        // so the handle is planted directly. The subject is the drain's
        // bound, not how the write got stuck.
        sup.pending_writes
            .lock()
            .unwrap()
            .push(tokio::spawn(std::future::pending::<()>()));

        // Both sides of the assertion below are the same constant, so it
        // pins the behaviour and not the budget. The budget is part of
        // what a deployment is promised at shutdown, so it is pinned here
        // in its own right.
        assert_eq!(
            CACHE_WRITE_DRAIN,
            Duration::from_secs(5),
            "the shutdown drain's budget is part of the contract",
        );

        let started = tokio::time::Instant::now();
        tokio::time::timeout(CACHE_WRITE_DRAIN * 4, sup.drain_pending_cache_writes())
            .await
            .expect("shutdown must not be able to hang on a stuck disk write");
        assert!(
            started.elapsed() >= CACHE_WRITE_DRAIN,
            "the drain must actually wait out its bound before giving up",
        );
    }

    #[tokio::test]
    async fn an_event_below_a_later_prefixs_entry_still_reaches_the_cache() {
        // The multi-prefix load reports the EARLIEST read as its revision
        // (`applied_revision_is_the_minimum_across_prefixes`), and that can
        // be below the highest revision in the entry set — a row in the
        // later-read prefix may have been written after the earlier prefix
        // was read. The resync's cache flush has to carry the read's number
        // rather than the entry maximum, because the cache refuses a write
        // below what it has committed: commit 103 here and every apply
        // between 100 and 103 is silently dropped from the cache — exactly
        // the events carrying the writes the read could not see. A restart
        // during an etcd outage would then serve a cache claiming 103 whose
        // changes it does not contain.
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("snap.json");

        {
            let sup = Arc::new(Supervisor::with_sources(
                vec![
                    (
                        WatchedPrefix::environment(ENV_PREFIX),
                        ScopedProvider::serving(
                            vec![entry("/aisix/env-1/models/m-1", VALID_MODEL, 90)],
                            100,
                        ),
                    ),
                    (
                        WatchedPrefix::global(GLOBAL_PREFIX),
                        ScopedProvider::serving(
                            vec![entry("/aisix/global/pricing/g", VALID_PRICE, 103)],
                            105,
                        ),
                    ),
                ],
                SnapshotCache::new(&cache_path),
            ));
            sup.load_once().await.unwrap();
            sup.await_pending_cache_writes().await;

            // The watch on the environment prefix resumes at 101 and
            // delivers a write the load could not have seen.
            assert!(sup.apply_put(&entry("/aisix/env-1/models/m-2", VALID_MODEL, 101)));
            sup.await_pending_cache_writes().await;
        }

        // A restart that cannot reach etcd serves the cache: both models
        // must be there. Same prefixes, so the cached keys resolve to the
        // same kinds they were stored under.
        let restarted = Supervisor::with_sources(
            vec![
                (
                    WatchedPrefix::environment(ENV_PREFIX),
                    ScopedProvider::refusing(),
                ),
                (
                    WatchedPrefix::global(GLOBAL_PREFIX),
                    ScopedProvider::refusing(),
                ),
            ],
            SnapshotCache::new(&cache_path),
        );
        restarted.restore_from_cache();
        assert_eq!(
            restarted.handle().load().models.len(),
            2,
            "the applied write at revision 101 must be in the cache"
        );
    }

    #[tokio::test]
    async fn resync_writes_to_disk_cache_then_restore_replays_it() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("snap.json");

        // First lifecycle: load with one entry, supervisor flushes to
        // disk on the resync.
        {
            let provider = Arc::new(FakeProvider::new(
                vec![entry("/aisix/models/m-1", VALID_MODEL, 7)],
                7,
            ));
            let sup = Supervisor::with_cache(provider, "/aisix", SnapshotCache::new(&cache_path));
            sup.load_once().await.unwrap();
            // Deterministically wait for the spawned cache write to
            // complete before we drop the supervisor. Replaces an
            // earlier 50ms sleep that flaked on slow CI runners.
            sup.await_pending_cache_writes().await;
        }

        // Second lifecycle: provider returns nothing, but restore_from_cache
        // populates the snapshot from disk so the proxy is ready.
        {
            let provider = Arc::new(FakeProvider::new(vec![], 0));
            let sup = Supervisor::with_cache(provider, "/aisix", SnapshotCache::new(&cache_path));
            // Snapshot is empty before restore.
            assert_eq!(sup.handle().load().models.len(), 0);
            sup.restore_from_cache();
            assert_eq!(
                sup.handle().load().models.len(),
                1,
                "restore_from_cache should re-publish the cached entry",
            );
        }
    }

    /// Regression for issue #112: concurrent `apply_put` calls used to
    /// race on the bare load-mutate-store sequence inside the
    /// supervisor, silently losing entries when both calls loaded the
    /// same Arc<Snapshot> and the second `store` overwrote the first.
    /// The fix replaces it with `SnapshotHandle::rcu`, which retries
    /// the closure until the CAS succeeds. With N=200 concurrent puts
    /// across distinct keys, every entry must end up in the snapshot.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn apply_put_concurrent_does_not_lose_events() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        sup.load_once().await.unwrap();

        const N: usize = 200;
        let mut tasks = tokio::task::JoinSet::new();
        for i in 0..N {
            let sup = Arc::clone(&sup);
            tasks.spawn(async move {
                let key = format!("/aisix/models/m-{i}");
                assert!(
                    sup.apply_put(&entry(&key, VALID_MODEL, (i + 1) as i64)),
                    "apply_put returned false for {key}"
                );
            });
        }
        while let Some(res) = tasks.join_next().await {
            res.unwrap();
        }
        let snap = sup.handle().load();
        assert_eq!(
            snap.models.len(),
            N,
            "concurrent apply_put lost entries (got {} of {})",
            snap.models.len(),
            N,
        );
    }

    /// Same regression shape for `apply_delete`: under concurrency the
    /// previous load-mutate-store path would have lost a sibling
    /// delete by overwriting it with a stale clone. With RCU, deleting
    /// every entry concurrently must leave the snapshot empty.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn apply_delete_concurrent_drains_snapshot() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Arc::new(Supervisor::new(provider, "/aisix"));
        sup.load_once().await.unwrap();

        const N: usize = 200;
        for i in 0..N {
            sup.apply_put(&entry(
                &format!("/aisix/models/m-{i}"),
                VALID_MODEL,
                (i + 1) as i64,
            ));
        }
        assert_eq!(sup.handle().load().models.len(), N);

        let mut tasks = tokio::task::JoinSet::new();
        for i in 0..N {
            let sup = Arc::clone(&sup);
            tasks.spawn(async move {
                sup.apply_delete(&format!("/aisix/models/m-{i}"));
            });
        }
        while let Some(res) = tasks.join_next().await {
            res.unwrap();
        }
        assert_eq!(
            sup.handle().load().models.len(),
            0,
            "concurrent apply_delete left orphaned entries",
        );
    }

    #[tokio::test]
    async fn put_and_delete_keep_cache_in_sync() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("snap.json");

        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::with_cache(provider, "/aisix", SnapshotCache::new(&cache_path));
        sup.load_once().await.unwrap();

        sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 5));
        sup.apply_put(&entry("/aisix/models/m-2", VALID_MODEL, 6));
        // Wait for both spawned cache writes to flush before reading.
        sup.await_pending_cache_writes().await;

        let cache = SnapshotCache::new(&cache_path);
        let cached = cache.load().expect("cache file present");
        assert_eq!(cached.entries.len(), 2);

        sup.apply_delete("/aisix/models/m-1");
        sup.await_pending_cache_writes().await;

        let cached = cache.load().expect("cache file present");
        assert_eq!(cached.entries.len(), 1);
        assert_eq!(cached.entries[0].key, "/aisix/models/m-2");
    }

    // ---- regression coverage for issue #114 -------------------------
    // /admin/v1/health needs to surface "etcd watch staleness". The
    // tests below pin: (1) WatchStatus reflects each apply path, and
    // (2) without an apply, last_apply_age stays None so the handler
    // can mark the supervisor as not-yet-warmed-up rather than
    // reporting age 0.

    #[tokio::test]
    async fn watch_status_starts_as_unset_before_any_apply() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        let snap = sup.watch_status().snapshot();
        assert_eq!(snap.revision, 0);
        assert!(
            snap.last_apply_age.is_none(),
            "last_apply_age should be None pre-first-apply; got {:?}",
            snap.last_apply_age,
        );
    }

    #[tokio::test]
    async fn watch_status_records_apply_on_load_and_put_and_delete() {
        let provider = Arc::new(FakeProvider::new(
            vec![entry("/aisix/models/m-init", VALID_MODEL, 4)],
            7,
        ));
        let sup = Supervisor::new(provider, "/aisix");

        // load_once → record_read_revision(7) → record_apply(7)
        sup.load_once().await.unwrap();
        let snap = sup.watch_status().snapshot();
        assert_eq!(
            snap.revision, 7,
            "load_once should advance revision to load_all's revision",
        );
        assert!(snap.last_apply_age.is_some());

        // apply_put with a higher revision advances the recorded one.
        assert!(sup.apply_put(&entry("/aisix/models/m-2", VALID_MODEL, 12)));
        let snap = sup.watch_status().snapshot();
        assert_eq!(snap.revision, 12);

        // apply_delete keeps the revision (no per-event revision on
        // the wire) but resets the apply timestamp.
        assert!(sup.apply_delete("/aisix/models/m-2"));
        let snap = sup.watch_status().snapshot();
        assert!(snap.last_apply_age.is_some());
        assert_eq!(snap.revision, 12);
    }

    #[tokio::test]
    async fn watch_status_age_grows_when_no_events_arrive() {
        // Pin the freshness signal: after an apply, the age is small;
        // wait briefly and observe it has grown. This is what the
        // /admin/v1/health reads this to detect a wedged watch — without
        // this signal the proxy could serve stale config indefinitely.
        let provider = Arc::new(FakeProvider::new(vec![], 5));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        let first = sup.watch_status().snapshot().last_apply_age.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        let later = sup.watch_status().snapshot().last_apply_age.unwrap();
        assert!(
            later > first,
            "last_apply_age should monotonically grow without new events; \
             first={first:?} later={later:?}",
        );
    }

    // ---- regression coverage for issue #115 -------------------------
    // The supervisor now retains the loader's rejected-entry list so
    // the heartbeat path can forward "DP rejected these resources" to
    // cp-api. Tests pin (1) apply_resync replaces the buffer wholesale,
    // (2) apply_put with a bad row appends to the buffer, (3) a
    // different successful apply_put does not hide an unrelated
    // rejection, and (4) fixing/deleting the rejected key clears it.

    // Schema rejection bait: empty `display_name` violates the
    // `minLength: 1` invariant. After #302 Phase A the `provider`
    // field is free-form string, so we trigger rejection via a
    // different required-field shape.
    const BAD_PROVIDER_MODEL: &[u8] = br#"{
        "display_name":"",
        "provider":"openai",
        "model_name":"l",
        "provider_key_id":"pk"
    }"#;

    #[tokio::test]
    async fn recent_rejections_replaced_by_apply_resync() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");

        // Seed the buffer with a bad apply_put.
        assert!(!sup.apply_put(&entry("/aisix/models/m-bad", BAD_PROVIDER_MODEL, 1)));
        assert_eq!(sup.recent_rejections().len(), 1);

        // A clean apply_resync should wipe the buffer.
        sup.apply_resync(&[entry("/aisix/models/m-good", VALID_MODEL, 2)]);
        assert!(
            sup.recent_rejections().is_empty(),
            "apply_resync with a clean entry set must reset the rejection buffer",
        );
    }

    #[tokio::test]
    async fn recent_rejections_accumulates_across_apply_puts() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");

        assert!(!sup.apply_put(&entry("/aisix/models/m-bad-1", BAD_PROVIDER_MODEL, 1)));
        assert!(!sup.apply_put(&entry("/aisix/models/m-bad-2", b"not-json", 2)));
        let rejections = sup.recent_rejections();
        assert_eq!(rejections.len(), 2);
        assert_eq!(rejections[0].kind, loader::RejectionKind::SchemaFailed);
        assert_eq!(rejections[1].kind, loader::RejectionKind::NonJson);
    }

    #[tokio::test]
    async fn recent_rejections_replaces_existing_key() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");

        assert!(!sup.apply_put(&entry("/aisix/models/m-bad", BAD_PROVIDER_MODEL, 1)));
        assert!(!sup.apply_put(&entry("/aisix/models/m-bad", b"not-json", 2)));

        let rejections = sup.recent_rejections();
        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].kind, loader::RejectionKind::NonJson);
    }

    #[tokio::test]
    async fn recent_rejections_survives_a_successful_put_for_different_key() {
        // A different key succeeding must not hide an unrelated
        // rejection; only the rejected key being fixed or deleted
        // should clear the heartbeat signal.
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        assert!(!sup.apply_put(&entry("/aisix/models/m-bad", BAD_PROVIDER_MODEL, 1)));
        assert_eq!(sup.recent_rejections().len(), 1);

        // A different model succeeds.
        assert!(sup.apply_put(&entry("/aisix/models/m-good", VALID_MODEL, 2)));
        assert_eq!(
            sup.recent_rejections().len(),
            1,
            "successful put must not silently drop earlier rejections",
        );
    }

    #[tokio::test]
    async fn recent_rejections_clears_when_same_key_becomes_valid() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");

        assert!(!sup.apply_put(&entry("/aisix/models/m-bad", BAD_PROVIDER_MODEL, 1)));
        assert_eq!(sup.recent_rejections().len(), 1);

        assert!(sup.apply_put(&entry("/aisix/models/m-bad", VALID_MODEL, 2)));
        assert!(
            sup.recent_rejections().is_empty(),
            "valid put for the same key must clear the retained rejection",
        );
    }

    #[tokio::test]
    async fn recent_rejections_clears_when_rejected_key_is_deleted() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");

        assert!(!sup.apply_put(&entry("/aisix/models/m-bad", BAD_PROVIDER_MODEL, 1)));
        assert_eq!(sup.recent_rejections().len(), 1);

        assert!(sup.apply_delete("/aisix/models/m-bad"));
        assert!(
            sup.recent_rejections().is_empty(),
            "delete must clear a rejection even when the bad row never entered the snapshot",
        );
    }

    // ---- partially-compatible retention (issue #871) ----

    /// A model document carrying a field this build does not know: loads
    /// (YELLOW) with the field reported.
    const YELLOW_MODEL: &[u8] = br#"{
        "display_name": "my-gpt4",
        "provider": "openai",
        "model_name": "gpt-4o",
        "provider_key_id": "11111111-1111-1111-1111-111111111111",
        "future_knob": true
    }"#;

    #[tokio::test]
    async fn partial_compat_tracked_on_put_and_cleared_on_exact_match() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        assert!(sup.apply_put(&entry("/aisix/models/m-1", YELLOW_MODEL, 1)));
        assert_eq!(sup.handle().load().models.len(), 1, "YELLOW row serves");
        let agg = sup.recent_partial_compat();
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].kind, "models");
        assert_eq!(agg[0].field, "future_knob");
        assert_eq!(agg[0].count, 1);
        // The status view carries the companion list next to rejected[].
        let view = sup.config_status().view();
        assert_eq!(view.partially_compatible.len(), 1);
        assert_eq!(view.partially_compatible[0].resource_kind, "models");
        assert_eq!(view.partially_compatible[0].field, "future_knob");
        assert!(view.rejected.is_empty());

        // Re-put with an exact-match document: the signal clears.
        assert!(sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 2)));
        assert!(sup.recent_partial_compat().is_empty());
        assert!(sup.config_status().view().partially_compatible.is_empty());
    }

    #[tokio::test]
    async fn partial_compat_cleared_on_delete() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        assert!(sup.apply_put(&entry("/aisix/models/m-1", YELLOW_MODEL, 1)));
        assert_eq!(sup.recent_partial_compat().len(), 1);
        assert!(sup.apply_delete("/aisix/models/m-1"));
        assert!(sup.recent_partial_compat().is_empty());
    }

    #[tokio::test]
    async fn partial_compat_replaced_wholesale_on_resync() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        assert!(sup.apply_put(&entry("/aisix/models/m-1", YELLOW_MODEL, 1)));
        assert_eq!(sup.recent_partial_compat().len(), 1);

        // Resync to a clean entry set: prior per-key YELLOW state is
        // no longer accurate and must be dropped, mirroring rejections.
        sup.apply_resync(&[entry("/aisix/models/m-2", VALID_MODEL, 2)]);
        assert!(sup.recent_partial_compat().is_empty());

        // Resync back to a YELLOW set repopulates it.
        sup.apply_resync(&[entry("/aisix/models/m-3", YELLOW_MODEL, 3)]);
        let agg = sup.recent_partial_compat();
        assert_eq!(agg.len(), 1);
        assert_eq!(agg[0].count, 1);
    }

    #[tokio::test]
    async fn partial_compat_kept_when_update_for_same_key_is_rejected() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        assert!(sup.apply_put(&entry("/aisix/models/m-1", YELLOW_MODEL, 1)));
        // A rejected update keeps the previous (YELLOW-loaded) value
        // serving, so the partially-compatible signal must survive too.
        assert!(!sup.apply_put(&entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)));
        assert_eq!(sup.handle().load().models.len(), 1);
        assert_eq!(sup.recent_partial_compat().len(), 1);
        assert_eq!(sup.recent_rejections().len(), 1);
    }

    // ---- RED last-known-good retention across resync/restart (#871 PR2) ----
    //
    // A watch put that is rejected already leaves the previous good value
    // serving (pinned above). But the retention used to end at the next
    // full resync: `apply_resync` rebuilt the snapshot from accepted rows
    // only, so a key whose latest etcd bytes are rejected VANISHED — an
    // api_key would 401 byte-identically to "no such key", days after the
    // write that caused it. The tests below pin the xDS-NACK-style fix:
    // the last known good value keeps serving for as long as the etcd key
    // exists, across resync and restart, with the staleness reported.

    #[tokio::test]
    async fn rejected_update_keeps_last_good_serving_across_resync() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        assert!(sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 1)));
        assert!(!sup.apply_put(&entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)));
        assert_eq!(sup.handle().load().models.len(), 1);

        // The next resync re-reads the full etcd state — which still
        // holds the rejected bytes for this key.
        sup.apply_resync(&[entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)]);
        assert_eq!(
            sup.handle().load().models.len(),
            1,
            "resync must keep serving the last known good value for a rejected key",
        );

        // The rejection signal persists every cycle, and the row is
        // reported as serving-stale with its age.
        assert_eq!(sup.recent_rejections().len(), 1);
        let view = serde_json::to_value(sup.config_status().view()).unwrap();
        assert_eq!(view["rejected"].as_array().unwrap().len(), 1);
        assert!(
            view["rejected"][0]["serving_stale_since"].is_string(),
            "rejected[] must carry the stale-serving timestamp: {view}",
        );
        assert!(
            view["rejected"][0]["serving_stale_age_seconds"].is_u64(),
            "rejected[] must carry the staleness age: {view}",
        );
        // The served row keeps counting.
        assert_eq!(view["applied"]["resource_counts"]["models"], 1);
    }

    #[tokio::test]
    async fn rejected_update_keeps_last_good_serving_across_restart() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("snap.json");

        // First lifecycle: a good row loads, then a resync observes the
        // rejected replacement bytes (the etcd state after a newer CP
        // wrote an update this DP cannot represent). The flushed cache
        // must carry enough to survive a restart.
        {
            let provider = Arc::new(FakeProvider::new(
                vec![entry("/aisix/models/m-1", VALID_MODEL, 1)],
                1,
            ));
            let sup = Supervisor::with_cache(provider, "/aisix", SnapshotCache::new(&cache_path));
            sup.load_once().await.unwrap();
            assert_eq!(sup.handle().load().models.len(), 1);
            sup.apply_resync(&[entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)]);
            assert_eq!(
                sup.handle().load().models.len(),
                1,
                "pre-restart: the last known good value serves through the resync",
            );
            sup.await_pending_cache_writes().await;
        }

        // Second lifecycle (process restart, etcd unreachable): restore
        // from disk. The last known good value must come back — without
        // it the restart is the cliff where the resource silently dies.
        {
            let provider = Arc::new(FakeProvider::new(vec![], 0));
            let sup = Supervisor::with_cache(provider, "/aisix", SnapshotCache::new(&cache_path));
            sup.restore_from_cache();
            assert_eq!(
                sup.handle().load().models.len(),
                1,
                "restart must restore the last known good value for a rejected key",
            );
            assert_eq!(
                sup.recent_rejections().len(),
                1,
                "the rejection signal must survive the restart too",
            );
        }
    }

    #[tokio::test]
    async fn deleting_a_rejected_never_serving_key_clears_observed_state() {
        // Audit finding on #871 PR2: a rejected put now mirrors its
        // bytes into the observed-state map even when the row never
        // served (no pin). Deleting that key takes the `!present` early
        // return in apply_delete, which must still drop the bytes from
        // `state` — otherwise the deleted key haunts source_hash until
        // the next resync and its document persists in the cache file.
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        let clean_hash = sup.config_status().view().source.source_hash;

        // Never served: the very first put for the key is rejected.
        assert!(!sup.apply_put(&entry("/aisix/models/m-bad", BAD_PROVIDER_MODEL, 1)));
        assert!(sup.handle().load().models.is_empty());
        assert_ne!(
            sup.config_status().view().source.source_hash,
            clean_hash,
            "the rejected bytes are part of the observed etcd state",
        );

        // The delete finds nothing in the snapshot but must still clear
        // the observed-state entry (and the rejection — clearing it is
        // "something removed", so the call reports true).
        assert!(sup.apply_delete("/aisix/models/m-bad"));
        assert!(sup.recent_rejections().is_empty());
        assert_eq!(
            sup.config_status().view().source.source_hash,
            clean_hash,
            "a deleted key must leave the observed etcd state immediately",
        );
    }

    #[tokio::test]
    async fn rejected_put_persists_pin_for_immediate_restart() {
        // A restart INSIDE the rejected-put window (before any resync
        // fixed the state to disk) must behave like a post-resync
        // restart: the rejected bytes and the pinned last-good ride the
        // cache together, so the value keeps serving AND the staleness
        // clock stays continuous instead of resetting at boot.
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("snap.json");
        let since_before;

        {
            let provider = Arc::new(FakeProvider::new(
                vec![entry("/aisix/models/m-1", VALID_MODEL, 1)],
                1,
            ));
            let sup = Supervisor::with_cache(provider, "/aisix", SnapshotCache::new(&cache_path));
            sup.load_once().await.unwrap();
            assert!(!sup.apply_put(&entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)));
            since_before = sup.recent_rejections()[0]
                .stale_serving_since_unix_secs
                .expect("rejected put with a serving value must report stale-since");
            sup.await_pending_cache_writes().await;
        }

        {
            let provider = Arc::new(FakeProvider::new(vec![], 0));
            let sup = Supervisor::with_cache(provider, "/aisix", SnapshotCache::new(&cache_path));
            sup.restore_from_cache();
            assert_eq!(
                sup.handle().load().models.len(),
                1,
                "restart in the rejected-put window must restore the pinned value",
            );
            let rejections = sup.recent_rejections();
            assert_eq!(rejections.len(), 1);
            assert_eq!(
                rejections[0].stale_serving_since_unix_secs,
                Some(since_before),
                "the staleness clock must be continuous across the restart",
            );
        }
    }

    #[tokio::test]
    async fn stale_served_row_dies_with_etcd_delete() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        assert!(sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 1)));
        assert!(!sup.apply_put(&entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)));
        sup.apply_resync(&[entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)]);
        assert_eq!(sup.handle().load().models.len(), 1);

        // The admin deletes the resource: the last known good goes with
        // it — retention must never outlive the etcd key.
        assert!(sup.apply_delete("/aisix/models/m-1"));
        assert!(sup.handle().load().models.is_empty());
        assert!(sup.recent_rejections().is_empty());
        // A later resync confirming the key's absence keeps it gone.
        sup.apply_resync(&[]);
        assert!(sup.handle().load().models.is_empty());
    }

    #[tokio::test]
    async fn stale_served_row_dies_when_resync_no_longer_carries_the_key() {
        // Same zombie guard for the resync-observed deletion: a key that
        // disappears from the full etcd read (no watch Delete seen, e.g.
        // reconnect after compaction) must drop its last known good.
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        assert!(sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 1)));
        assert!(!sup.apply_put(&entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)));
        sup.apply_resync(&[entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)]);
        assert_eq!(sup.handle().load().models.len(), 1);

        sup.apply_resync(&[]);
        assert!(
            sup.handle().load().models.is_empty(),
            "a key absent from the resynced etcd state must not keep serving",
        );
        assert!(sup.recent_rejections().is_empty());
    }

    #[tokio::test]
    async fn stale_last_good_that_was_yellow_keeps_its_partial_compat_signal() {
        // The value actually serving is itself YELLOW (unknown field
        // ignored), and the newer update is RED-rejected. Both signals
        // must coexist across a resync: rejected[] describes the new
        // bytes, partially_compatible[] describes the served old value.
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        assert!(sup.apply_put(&entry("/aisix/models/m-1", YELLOW_MODEL, 1)));
        assert!(!sup.apply_put(&entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)));
        sup.apply_resync(&[entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)]);

        assert_eq!(sup.handle().load().models.len(), 1);
        assert_eq!(sup.recent_rejections().len(), 1);
        let agg = sup.recent_partial_compat();
        assert_eq!(
            agg.len(),
            1,
            "the served YELLOW last-good keeps reporting its ignored fields",
        );
        assert_eq!(agg[0].field, "future_knob");
    }

    #[tokio::test]
    async fn config_hash_reflects_served_bytes_not_rejected_bytes() {
        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        assert!(sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 1)));
        let good_hash = sup.config_status().view().applied.unwrap().config_hash;

        sup.apply_resync(&[entry("/aisix/models/m-1", BAD_PROVIDER_MODEL, 2)]);
        let view = sup.config_status().view();
        let applied = view.applied.unwrap();
        // What's served didn't change, so the served-config hash must not
        // change either: the rejected bytes never enter config_hash (the
        // hash must not claim the new value applied), and the row must
        // not silently drop out of it (the hash must not claim the row
        // stopped serving).
        assert_eq!(
            applied.config_hash, good_hash,
            "config_hash must cover the bytes actually served (the last known good)",
        );
        // source_hash reflects the observed etcd state (the rejected
        // bytes), so the two hashes diverge — the honest "not converged"
        // signal, explained by rejected[].
        assert_ne!(
            Some(applied.config_hash.as_str()),
            view.source.source_hash.as_deref(),
        );
    }
}
