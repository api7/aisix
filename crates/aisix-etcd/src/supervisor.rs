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

use aisix_core::snapshot::SnapshotHandle;
use aisix_core::AisixSnapshot;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

use crate::backoff::ExpBackoff;
use crate::key;
use crate::loader::{self, BuildStats, RejectedEntry};
use crate::provider::{ConfigProvider, ProviderError, RawEntry, WatchEvent};
use crate::snapshot_cache::SnapshotCache;

/// Cheap clonable handle for the watch supervisor's freshness state —
/// the etcd revision the snapshot reflects, and how long ago the
/// supervisor last applied an event. Read by `/admin/v1/health` so
/// operators can tell at a glance whether the gateway is serving from
/// a frozen snapshot (etcd partition or watch supervisor wedged) vs
/// from a live config stream. See issue #114.
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

    /// Most recent loader rejections, capped at
    /// [`MAX_RETAINED_REJECTIONS`]. Read by the heartbeat path so the
    /// CP can surface "your DP rejected these resources" in the
    /// dashboard. Newest at the back; on overflow the oldest entries
    /// are dropped — see issue #115. The buffer is replaced (not
    /// appended-to) on every load_once / apply_resync because those
    /// re-process the full entry set; apply_put / apply_delete append
    /// per-event because they only see one row.
    rejections: Mutex<Vec<RejectedEntry>>,

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
            rejections: Mutex::new(Vec::new()),
            pending_writes: Mutex::new(Vec::new()),
        }
    }

    /// Cheap clonable handle to the supervisor's freshness state.
    /// Read by /admin/v1/health to surface "etcd watch alive" /
    /// "snapshot age" metrics. See [`WatchStatus`].
    pub fn watch_status(&self) -> WatchStatus {
        self.status.clone()
    }

    /// Snapshot of the most recent loader rejections (capped). Used by
    /// the heartbeat path to forward "DP rejected these resources" to
    /// cp-api. Returns a clone so the caller doesn't hold the lock
    /// across the heartbeat HTTP call.
    pub fn recent_rejections(&self) -> Vec<RejectedEntry> {
        self.rejections.lock().unwrap().clone()
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
        let Some((entries, revision)) = self.cache.load() else {
            return;
        };
        let stats = self.apply_resync(&entries);
        // Track the last cached revision so the first live cycle's
        // resync reflects the right "from where" in logs. We don't
        // try to use it as the watch start revision — the etcd server
        // may have compacted past it; load_all + watch from latest is
        // always safer.
        *self.revision.lock().unwrap() = revision;
        tracing::info!(
            accepted = stats.accepted,
            revision,
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
        let mut rev = self.revision.lock().unwrap();
        if revision > *rev {
            *rev = revision;
        }
        self.status.record_apply(revision);
    }

    /// Apply a single Put event on top of the current snapshot.
    /// Returns `true` if the apply succeeded (schema + parse passed).
    pub fn apply_put(&self, entry: &RawEntry) -> bool {
        // Build a tiny snapshot out of just the new entry, then merge.
        let (tiny, mut stats) = loader::build_snapshot(&self.prefix, std::slice::from_ref(entry));
        if stats.accepted == 0 {
            // The loader already attached a RejectedEntry for whatever
            // path failed (bad key / non-JSON / schema / parse). Move
            // them into the supervisor's retained buffer so the next
            // heartbeat surfaces the failure to cp-api. See issue #115.
            for r in stats.rejections.drain(..) {
                self.push_rejection(r);
            }
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

            // Move any entries from `tiny` into `new`. Must cover every
            // ResourceTable on AisixSnapshot — a missing kind here
            // means a watch event silently drops on the floor and the
            // snapshot never sees the new entry, even though the loader
            // and the proxy both know about it.
            for e in tiny.models.entries() {
                new.models.insert(clone_entry(&e));
            }
            for e in tiny.apikeys.entries() {
                new.apikeys.insert(clone_entry(&e));
            }
            for e in tiny.provider_keys.entries() {
                new.provider_keys.insert(clone_entry(&e));
            }
            for e in tiny.guardrails.entries() {
                new.guardrails.insert(clone_entry(&e));
            }
            for e in tiny.cache_policies.entries() {
                new.cache_policies.insert(clone_entry(&e));
            }
            for e in tiny.observability_exporters.entries() {
                new.observability_exporters.insert(clone_entry(&e));
            }
            for e in tiny.rate_limit_policies.entries() {
                new.rate_limit_policies.insert(clone_entry(&e));
            }
            new
        });
        self.remove_rejection_for_key(&entry.key);

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
        let present = match parsed.kind {
            "models" => snap.models.get_by_id(parsed.id).is_some(),
            "api_keys" => snap.apikeys.get_by_id(parsed.id).is_some(),
            "provider_keys" => snap.provider_keys.get_by_id(parsed.id).is_some(),
            "guardrails" => snap.guardrails.get_by_id(parsed.id).is_some(),
            "cache_policies" => snap.cache_policies.get_by_id(parsed.id).is_some(),
            "observability_exporters" => {
                snap.observability_exporters.get_by_id(parsed.id).is_some()
            }
            "rate_limit_policies" => snap.rate_limit_policies.get_by_id(parsed.id).is_some(),
            _ => false,
        };
        let removed_rejection = self.remove_rejection_for_key(key_str);
        drop(snap);
        if !present {
            if removed_rejection {
                let cur_rev = *self.revision.lock().unwrap();
                self.status.record_apply(cur_rev);
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
                "cache_policies" => {
                    new.cache_policies.remove(parsed.id);
                }
                "observability_exporters" => {
                    new.observability_exporters.remove(parsed.id);
                }
                "rate_limit_policies" => {
                    new.rate_limit_policies.remove(parsed.id);
                }
                _ => {}
            }
            new
        });
        self.state.lock().unwrap().remove(key_str);
        // Stamp /admin/v1/health freshness on a successful delete. We
        // don't have a per-event revision on the wire delete
        // (the etcd watch revision is held at the cycle level);
        // call record_apply with the current revision so age
        // resets even if the revision number doesn't move.
        let cur_rev = *self.revision.lock().unwrap();
        self.status.record_apply(cur_rev);
        self.flush_cache();
        true
    }

    /// Replace the current snapshot with a freshly loaded set (resync).
    pub fn apply_resync(&self, entries: &[RawEntry]) -> BuildStats {
        let (snap, stats) = loader::build_snapshot(&self.prefix, entries);
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
        // wholesale with what this build produced (issue #115).
        self.set_rejections(stats.rejections.clone());
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
            let join = rt_handle.spawn(async move { cache.store(&entries, revision).await });
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
                Some(Ok(WatchEvent::Delete { key, .. })) => {
                    self.apply_delete(&key);
                }
                Some(Ok(WatchEvent::Resync { entries, .. })) => {
                    self.apply_resync(&entries);
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
    for e in src.models.entries() {
        out.models.insert(clone_entry(&e));
    }
    for e in src.apikeys.entries() {
        out.apikeys.insert(clone_entry(&e));
    }
    for e in src.provider_keys.entries() {
        out.provider_keys.insert(clone_entry(&e));
    }
    for e in src.guardrails.entries() {
        out.guardrails.insert(clone_entry(&e));
    }
    for e in src.cache_policies.entries() {
        out.cache_policies.insert(clone_entry(&e));
    }
    for e in src.observability_exporters.entries() {
        out.observability_exporters.insert(clone_entry(&e));
    }
    for e in src.rate_limit_policies.entries() {
        out.rate_limit_policies.insert(clone_entry(&e));
    }
    out
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

        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();

        for (key, body, _kind) in [
            ("/aisix/provider_keys/pk-1", VALID_PROVIDER_KEY, "PK"),
            ("/aisix/guardrails/g-1", VALID_GUARDRAIL, "Guardrail"),
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
        ] {
            assert!(
                sup.apply_put(&entry(key, body, 2)),
                "apply_put returned false for {key}"
            );
        }

        let snap = sup.handle().load();
        assert_eq!(snap.provider_keys.len(), 1, "ProviderKey not merged");
        assert_eq!(snap.guardrails.len(), 1, "Guardrail not merged");
        assert_eq!(snap.cache_policies.len(), 1, "CachePolicy not merged");
        assert_eq!(
            snap.observability_exporters.len(),
            1,
            "ObservabilityExporter not merged"
        );
    }

    #[tokio::test]
    async fn apply_delete_removes_every_resource_kind() {
        let provider = Arc::new(FakeProvider::new(
            vec![entry(
                "/aisix/provider_keys/pk-1",
                br#"{"display_name":"x","secret":"y"}"#,
                1,
            )],
            1,
        ));
        let sup = Supervisor::new(provider, "/aisix");
        sup.load_once().await.unwrap();
        assert_eq!(sup.handle().load().provider_keys.len(), 1);
        assert!(sup.apply_delete("/aisix/provider_keys/pk-1"));
        assert!(sup.handle().load().provider_keys.is_empty());
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
        let (entries, _) = cache.load().expect("cache file present");
        assert_eq!(entries.len(), 2);

        sup.apply_delete("/aisix/models/m-1");
        sup.await_pending_cache_writes().await;

        let (entries, _) = cache.load().expect("cache file present");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "/aisix/models/m-2");
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
}
