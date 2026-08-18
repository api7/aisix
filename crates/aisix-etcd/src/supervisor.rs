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
    hash_entries, AppliedSnapshot, ConfigStatus, IncomingRejection, LoadObservation,
    PartialCompatResource, SourceKind,
};
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::AisixSnapshot;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

use crate::backoff::ExpBackoff;
use crate::key;
use crate::loader::{self, BuildStats, PartialCompatEntry, PartialCompatRow, RejectedEntry};
use crate::provider::{ConfigProvider, ProviderError, RawEntry, WatchEvent};
use crate::snapshot_cache::SnapshotCache;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleServing {
    pub entry: RawEntry,
    pub since_unix_secs: u64,
}

/// One supervisor instance. Consumers call [`Supervisor::run`] once and
/// drop the returned handle on shutdown.
pub struct Supervisor<P: ConfigProvider> {
    provider: Arc<P>,
    prefix: String,
    handle: SnapshotHandle<AisixSnapshot>,

    // Last-known etcd state, kept in `key → RawEntry` form so deltas
    // (Put/Delete) can update it incrementally and the whole map can
    // be flushed to disk via `cache.store`.
    state: Mutex<HashMap<String, RawEntry>>,
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

    // JoinHandles for in-flight `flush_cache` writes. Tests use
    // [`Self::await_pending_cache_writes`] to deterministically wait
    // for these without relying on a wall-clock sleep, which proved
    // flaky on slow CI runners. Production code does not read this
    // field; if a handle is dropped (e.g. during shutdown), the
    // underlying write either completed or was cancelled — either
    // way the on-disk cache is best-effort and the next live cycle
    // re-publishes from etcd.
    pending_writes: Mutex<Vec<JoinHandle<()>>>,
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
    pub fn with_cache(provider: Arc<P>, prefix: impl Into<String>, cache: SnapshotCache) -> Self {
        Self {
            provider,
            prefix: prefix.into(),
            handle: SnapshotHandle::new(AisixSnapshot::new()),
            state: Mutex::new(HashMap::new()),
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
            let state = self.state.lock().unwrap();
            let rejections = self.rejections.lock().unwrap();
            // `state` is the raw entry map the DP holds — every observed
            // etcd write lands here, including rejected ones (a resync
            // inserts them wholesale; a rejected live Put mirrors its
            // bytes in too, #871), so source_hash always covers the
            // observed etcd state.
            source_hash =
                hash_entries(state.values().map(|e| (e.key.as_str(), e.value.as_slice())));
            let rejected_keys: HashSet<&str> = rejections.iter().map(|r| r.key.as_str()).collect();
            // config_hash covers the bytes each key ACTUALLY serves: the
            // observed etcd bytes for accepted keys, the pinned last-known-
            // good bytes for stale-serving keys (#871), and nothing for a
            // rejected key with no last good (it doesn't serve). The stale
            // map — not the capped rejection buffer — decides which keys
            // substitute, so buffer overflow can never flip a served key's
            // hash contribution to the rejected bytes.
            config_hash = hash_entries(
                state
                    .values()
                    .filter(|e| {
                        !rejected_keys.contains(e.key.as_str()) && !stale.contains_key(&e.key)
                    })
                    .map(|e| (e.key.as_str(), e.value.as_slice()))
                    .chain(
                        stale
                            .values()
                            .map(|s| (s.entry.key.as_str(), s.entry.value.as_slice())),
                    ),
            );
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
            if let Ok(parsed) = key::parse(&self.prefix, key_str) {
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
        let (kind, id) = match key::parse(&self.prefix, &r.key) {
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
    /// by [`Self::flush_cache`] and await them. Test-only synchroniser:
    /// production code never needs to block on disk persistence.
    #[cfg(test)]
    pub async fn await_pending_cache_writes(&self) {
        let handles: Vec<JoinHandle<()>> = {
            let mut pending = self.pending_writes.lock().unwrap();
            std::mem::take(&mut *pending)
        };
        for handle in handles {
            // Failures here are not test failures — a write that
            // panicked is its own bug surfaced separately. We only
            // need the await to deterministically order against the
            // disk read that follows.
            let _ = handle.await;
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
        let (entries, revision) = self.provider.load_all().await?;
        let stats = self.apply_resync(&entries);
        // apply_resync uses max(entry revisions); bump to the etcd
        // load_all revision so the cache file records the true "as
        // of" point, not just the max entry write.
        self.set_revision_floor(revision);
        tracing::info!(
            accepted = stats.accepted,
            rejected = stats.schema_rejected + stats.parse_rejected,
            revision,
            "initial snapshot built",
        );
        Ok(stats)
    }

    /// Bump the recorded revision floor. Used by the cycle path to
    /// stamp the cache with the etcd `load_all` revision even when the
    /// resulting entry set is empty (so the file still reflects when
    /// the DP last successfully reached the CP). Also stamps
    /// `WatchStatus.last_apply_at` so `/admin/v1/health` reflects the
    /// successful round-trip with etcd even on an empty config.
    fn set_revision_floor(&self, revision: i64) {
        {
            let mut rev = self.revision.lock().unwrap();
            if revision > *rev {
                *rev = revision;
            }
        }
        self.status.record_apply(revision);
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
    fn capture_last_good(&self, key_str: &str) {
        if self.stale_serving.lock().unwrap().contains_key(key_str) {
            return;
        }
        let Ok(parsed) = key::parse(&self.prefix, key_str) else {
            return;
        };
        if !snapshot_has(&self.handle.load(), parsed.kind, parsed.id) {
            return;
        }
        let Some(good) = self.state.lock().unwrap().get(key_str).cloned() else {
            return;
        };
        // entry().or_insert_with keeps the original pin (and its `since`)
        // if a concurrent caller won the race after the check above.
        self.stale_serving
            .lock()
            .unwrap()
            .entry(key_str.to_string())
            .or_insert_with(|| StaleServing {
                entry: good,
                since_unix_secs: now_unix_secs(),
            });
    }

    /// Apply a single Put event on top of the current snapshot.
    /// Returns `true` if the apply succeeded (schema + parse passed).
    pub fn apply_put(&self, entry: &RawEntry) -> bool {
        // Build a tiny snapshot out of just the new entry, then merge.
        let (tiny, mut stats) = loader::build_snapshot(&self.prefix, std::slice::from_ref(entry));
        if stats.accepted == 0 {
            // The previous good value keeps serving. Pin it now (#871):
            // the next resync rebuilds from the rejected etcd bytes and
            // needs the pinned bytes to keep this row alive. Must run
            // BEFORE the state-map update below, which overwrites the
            // serving bytes with the rejected ones.
            self.capture_last_good(&entry.key);
            // Mirror the rejected bytes into the observed-state map and
            // the cache like any other observed etcd write: source_hash
            // reflects the observed etcd state immediately, and a
            // restart inside this window restores the same
            // rejected-bytes + pinned-value shape a post-resync restart
            // would — keeping the staleness clock continuous instead of
            // resetting it at the next boot.
            {
                let mut state = self.state.lock().unwrap();
                state.insert(entry.key.clone(), entry.clone());
            }
            {
                let mut rev = self.revision.lock().unwrap();
                if entry.revision > *rev {
                    *rev = entry.revision;
                }
            }
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
            // (rejected[] gains this entry; last_reload flips unsuccessful).
            self.sync_config_status(false);
            self.flush_cache();
            return false;
        }

        // RCU: load → clone → mutate → CAS, retrying the closure if a
        // concurrent apply_put / apply_delete / apply_resync raced our
        // CAS. The previous implementation used a bare load-mutate-
        // store sequence which silently dropped events under
        // concurrency (see issue #112). The closure body must be
        // idempotent w.r.t. its input — `tiny` is captured by reference
        // and the same delta is applied each retry, which is fine
        // because the operation is "merge tiny into current".
        self.handle.rcu(|current| {
            let new = clone_snapshot(current);
            // Move any entries from `tiny` into `new`. `merge_snapshot`
            // must cover every ResourceTable on AisixSnapshot — a
            // missing kind there means a watch event silently drops on
            // the floor and the snapshot never sees the new entry, even
            // though the loader and the proxy both know about it.
            merge_snapshot(&new, &tiny);
            new
        });
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

        // Mirror the put into the cache-tracking map and flush.
        // Track the highest revision we've observed so the cache file
        // records something monotonic.
        {
            let mut state = self.state.lock().unwrap();
            state.insert(entry.key.clone(), entry.clone());
        }
        {
            let mut rev = self.revision.lock().unwrap();
            if entry.revision > *rev {
                *rev = entry.revision;
            }
        }
        // /admin/v1/health reads this — record the apply so `last_apply_age`
        // resets on every event we successfully process.
        self.status.record_apply(entry.revision);
        self.sync_config_status(false);
        self.flush_cache();
        true
    }

    /// Apply a Delete event. Returns `true` if anything was actually
    /// removed (the kind/id was present).
    pub fn apply_delete(&self, key_str: &str) -> bool {
        let parsed = match key::parse(&self.prefix, key_str) {
            Ok(k) => k,
            Err(err) => {
                tracing::warn!(key = %key_str, error = %err, "ignoring delete with bad key");
                return false;
            }
        };

        // Probe first — if the key isn't present in the current
        // snapshot we have nothing to do and don't want to take the
        // RCU CAS path (which would still publish a no-op clone and
        // race against concurrent applies). The probe + RCU cycle
        // produces an idempotent "removed" return value: a parallel
        // delete that wins the race observes the same key already
        // gone, so this caller returns false (nothing left to remove).
        let snap = self.handle.load();
        let present = snapshot_has(&snap, parsed.kind, parsed.id);
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
        drop(snap);
        if !present {
            if removed_rejection || removed_state {
                let cur_rev = *self.revision.lock().unwrap();
                self.status.record_apply(cur_rev);
                // Clearing a rejected key changes the reported state.
                self.sync_config_status(false);
                self.flush_cache();
            }
            return removed_rejection;
        }

        // RCU: load → clone → remove → CAS, retrying under contention.
        // The closure body re-checks `removed` from its own clone so
        // the eventual CAS reflects the latest snapshot's state — if a
        // sibling apply_delete won the race, the kind.remove on our
        // clone returns None and we still publish a coherent (no-op)
        // result.
        self.handle.rcu(|current| {
            let new = clone_snapshot(current);
            match parsed.kind {
                "models" => {
                    new.models.remove(parsed.id);
                }
                "api_keys" => {
                    new.apikeys.remove(parsed.id);
                }
                "provider_keys" => {
                    new.provider_keys.remove(parsed.id);
                }
                "guardrails" => {
                    new.guardrails.remove(parsed.id);
                }
                "guardrail_attachments" => {
                    new.guardrail_attachments.remove(parsed.id);
                }
                "cache_policies" => {
                    new.cache_policies.remove(parsed.id);
                }
                "observability_exporters" => {
                    new.observability_exporters.remove(parsed.id);
                }
                "rate_limit_policies" => {
                    new.rate_limit_policies.remove(parsed.id);
                }
                "mcp_servers" => {
                    new.mcp_servers.remove(parsed.id);
                }
                "mcp_policies" => {
                    new.mcp_policies.remove(parsed.id);
                }
                "a2a_agents" => {
                    new.a2a_agents.remove(parsed.id);
                }
                "oidc_providers" => {
                    new.oidc_providers.remove(parsed.id);
                }
                "claim_mappings" => {
                    new.claim_mappings.remove(parsed.id);
                }
                "passthrough_routes" => {
                    new.passthrough_routes.remove(parsed.id);
                }
                "mcp_auth_settings" => {
                    new.mcp_auth_settings.remove(parsed.id);
                }
                _ => {}
            }
            new
        });
        // Stamp /admin/v1/health freshness on a successful delete. We
        // don't have a per-event revision on the wire delete
        // (the etcd watch revision is held at the cycle level);
        // call record_apply with the current revision so age
        // resets even if the revision number doesn't move.
        let cur_rev = *self.revision.lock().unwrap();
        self.status.record_apply(cur_rev);
        self.sync_config_status(false);
        self.flush_cache();
        true
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
        let (snap, mut stats) = loader::build_snapshot(&self.prefix, entries);

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
                .filter_map(|r| state.get(&r.key).map(|e| (r.key.clone(), e.clone())))
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
                let Ok(parsed) = key::parse(&self.prefix, &r.key) else {
                    continue;
                };
                if !snapshot_has(&prev_snap, parsed.kind, parsed.id) {
                    continue;
                }
                if let Some(good) = prev_state.get(&r.key) {
                    stale.insert(
                        r.key.clone(),
                        StaleServing {
                            entry: good.clone(),
                            since_unix_secs: now_unix_secs(),
                        },
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
            let (lkg_snap, lkg_stats) = loader::build_snapshot(&self.prefix, &injected);
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
            state.clear();
            for e in entries {
                state.insert(e.key.clone(), e.clone());
            }
        }
        // Resync revision is the max of any entry; if the caller has a
        // separate "load_all revision" they pass it via the cycle path
        // (see `cycle`), this branch just covers the watch Resync event.
        let max_rev = entries.iter().map(|e| e.revision).max();
        if let Some(rev_val) = max_rev {
            let mut rev = self.revision.lock().unwrap();
            if rev_val > *rev {
                *rev = rev_val;
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
        let entries: Vec<RawEntry> = {
            let state = self.state.lock().unwrap();
            state.values().cloned().collect()
        };
        let stale: Vec<StaleServing> = {
            let guard = self.stale_serving.lock().unwrap();
            guard.values().cloned().collect()
        };
        let revision = *self.revision.lock().unwrap();
        let cache = self.cache.clone();
        // Spawn the actual write so the apply path stays sync. If we
        // aren't inside a runtime (cache::disabled() tests), just skip.
        // Track the JoinHandle so tests can deterministically await
        // the write via [`Self::await_pending_cache_writes`] instead
        // of leaning on `tokio::time::sleep`, which under CI load
        // raced the spawn (~50ms wasn't enough on heavily loaded
        // GitHub Actions runners).
        if let Ok(rt_handle) = tokio::runtime::Handle::try_current() {
            let join =
                rt_handle.spawn(async move { cache.store(&entries, revision, &stale).await });
            self.pending_writes.lock().unwrap().push(join);
        }
    }

    /// Long-running loop. Handles exp-backoff reconnects and resync on
    /// compaction. Runs until cancelled via the cancellation token.
    pub async fn run(self: Arc<Self>, mut cancel: tokio::sync::watch::Receiver<bool>) {
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
                    tracing::warn!(
                        error = %err,
                        backoff_ms = delay.as_millis() as u64,
                        "etcd watch failed; backing off before reconnect",
                    );
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
        let (entries, revision) = self
            .provider
            .load_all()
            .await
            .map_err(SupervisorError::Provider)?;

        self.apply_resync(&entries);
        self.set_revision_floor(revision);

        let mut stream = self
            .provider
            .watch(revision + 1)
            .await
            .map_err(SupervisorError::Provider)?;

        loop {
            if *cancel.borrow() {
                return Err(SupervisorError::Cancelled);
            }

            let next = tokio::select! {
                item = stream.next() => item,
                _ = wait_for_cancel(cancel.clone()) => {
                    return Err(SupervisorError::Cancelled);
                }
            };

            match next {
                None => return Ok(()),
                Some(Err(ProviderError::Compacted)) => {
                    tracing::warn!("etcd compaction detected — resyncing");
                    // Break out so `run` re-enters `cycle` cleanly; the
                    // next iteration re-loads from scratch. We don't want
                    // to treat compaction as a backoff-worthy failure.
                    return Ok(());
                }
                Some(Err(err)) => return Err(SupervisorError::Provider(err)),
                Some(Ok(WatchEvent::Put(raw))) => {
                    self.apply_put(&raw);
                }
                Some(Ok(WatchEvent::Delete { key, revision })) => {
                    self.apply_delete(&key);
                    // Advance the applied-revision floor to the delete's
                    // mod_revision even when the key wasn't present —
                    // "processed everything up to rev X" must cover
                    // deletes, otherwise the heartbeat-reported
                    // applied_revision (#519 B.3) stalls after a CP
                    // delete until the next put arrives.
                    self.set_revision_floor(revision);
                }
                Some(Ok(WatchEvent::Resync { entries, revision })) => {
                    self.apply_resync(&entries);
                    // Same rationale: the resync's header revision is the
                    // "consistent as of" point even when the entry set
                    // is empty or only contains older mod_revisions.
                    self.set_revision_floor(revision);
                }
            }
        }
    }
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

/// Shallow clone of every [`Arc<ResourceEntry>`] — fast and, importantly,
/// it doesn't materialise a deep copy of the `T` payload.
fn clone_snapshot(src: &AisixSnapshot) -> AisixSnapshot {
    let out = AisixSnapshot::new();
    merge_snapshot(&out, src);
    out
}

/// Insert every entry of `src` into `dst` (replacing same-id entries).
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
    } = src;
    for e in models.entries() {
        dst.models.insert(clone_entry(&e));
    }
    for e in apikeys.entries() {
        dst.apikeys.insert(clone_entry(&e));
    }
    for e in provider_keys.entries() {
        dst.provider_keys.insert(clone_entry(&e));
    }
    for e in guardrails.entries() {
        dst.guardrails.insert(clone_entry(&e));
    }
    for e in guardrail_attachments.entries() {
        dst.guardrail_attachments.insert(clone_entry(&e));
    }
    for e in cache_policies.entries() {
        dst.cache_policies.insert(clone_entry(&e));
    }
    for e in observability_exporters.entries() {
        dst.observability_exporters.insert(clone_entry(&e));
    }
    for e in rate_limit_policies.entries() {
        dst.rate_limit_policies.insert(clone_entry(&e));
    }
    for e in mcp_servers.entries() {
        dst.mcp_servers.insert(clone_entry(&e));
    }
    for e in mcp_policies.entries() {
        dst.mcp_policies.insert(clone_entry(&e));
    }
    for e in a2a_agents.entries() {
        dst.a2a_agents.insert(clone_entry(&e));
    }
    for e in oidc_providers.entries() {
        dst.oidc_providers.insert(clone_entry(&e));
    }
    for e in claim_mappings.entries() {
        dst.claim_mappings.insert(clone_entry(&e));
    }
    for e in passthrough_routes.entries() {
        dst.passthrough_routes.insert(clone_entry(&e));
    }
    for e in mcp_auth_settings.entries() {
        dst.mcp_auth_settings.insert(clone_entry(&e));
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
    ] {
        if n > 0 {
            counts.insert(kind.to_string(), n);
        }
    }
    counts
}

fn clone_entry<T: Clone>(src: &Arc<aisix_core::ResourceEntry<T>>) -> aisix_core::ResourceEntry<T> {
    aisix_core::ResourceEntry {
        id: src.id.clone(),
        value: src.value.clone(),
        revision: src.revision,
    }
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

    fn entry(key: &str, v: &[u8], rev: i64) -> RawEntry {
        RawEntry {
            key: key.into(),
            value: v.to_vec(),
            revision: rev,
        }
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

        // load_once → set_revision_floor(7) → record_apply(7)
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
