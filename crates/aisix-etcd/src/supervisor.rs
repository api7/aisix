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
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use crate::backoff::ExpBackoff;
use crate::key;
use crate::loader::{self, BuildStats};
use crate::provider::{ConfigProvider, ProviderError, RawEntry, WatchEvent};
use crate::snapshot_cache::SnapshotCache;

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
    /// the DP last successfully reached the CP).
    fn set_revision_floor(&self, revision: i64) {
        let mut rev = self.revision.lock().unwrap();
        if revision > *rev {
            *rev = revision;
        }
    }

    /// Apply a single Put event on top of the current snapshot.
    /// Returns `true` if the apply succeeded (schema + parse passed).
    pub fn apply_put(&self, entry: &RawEntry) -> bool {
        // Build a tiny snapshot out of just the new entry, then merge.
        let (tiny, stats) = loader::build_snapshot(&self.prefix, std::slice::from_ref(entry));
        if stats.accepted == 0 {
            return false;
        }

        let new = clone_snapshot(&self.handle.load());

        // Move any entries from `tiny` into `new`.
        for e in tiny.models.entries() {
            new.models.insert(clone_entry(&e));
        }
        for e in tiny.apikeys.entries() {
            new.apikeys.insert(clone_entry(&e));
        }

        self.handle.store(new);

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

        let new = clone_snapshot(&self.handle.load());
        let removed = match parsed.kind {
            "models" => new.models.remove(parsed.id).is_some(),
            "apikeys" => new.apikeys.remove(parsed.id).is_some(),
            _ => false,
        };
        if removed {
            self.handle.store(new);
            self.state.lock().unwrap().remove(key_str);
            self.flush_cache();
        }
        removed
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
        if let Some(max_rev) = entries.iter().map(|e| e.revision).max() {
            let mut rev = self.revision.lock().unwrap();
            if max_rev > *rev {
                *rev = max_rev;
            }
        }
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
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move { cache.store(&entries, revision).await });
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
        "name": "my-gpt4",
        "model": "openai/gpt-4o",
        "provider_config": {"api_key": "sk-x"}
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
            // Yield so the spawned cache write has a chance to complete
            // before we drop the supervisor.
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
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

    #[tokio::test]
    async fn put_and_delete_keep_cache_in_sync() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("snap.json");

        let provider = Arc::new(FakeProvider::new(vec![], 0));
        let sup = Supervisor::with_cache(provider, "/aisix", SnapshotCache::new(&cache_path));
        sup.load_once().await.unwrap();

        sup.apply_put(&entry("/aisix/models/m-1", VALID_MODEL, 5));
        sup.apply_put(&entry("/aisix/models/m-2", VALID_MODEL, 6));
        // Yield so the spawned cache writes have a chance to complete.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let cache = SnapshotCache::new(&cache_path);
        let (entries, _) = cache.load().expect("cache file present");
        assert_eq!(entries.len(), 2);

        sup.apply_delete("/aisix/models/m-1");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let (entries, _) = cache.load().expect("cache file present");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "/aisix/models/m-2");
    }
}
