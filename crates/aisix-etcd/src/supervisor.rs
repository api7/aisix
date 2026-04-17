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
use std::sync::Arc;
use std::time::Duration;

use crate::backoff::ExpBackoff;
use crate::key;
use crate::loader::{self, BuildStats};
use crate::provider::{ConfigProvider, ProviderError, RawEntry, WatchEvent};

/// One supervisor instance. Consumers call [`Supervisor::run`] once and
/// drop the returned handle on shutdown.
pub struct Supervisor<P: ConfigProvider> {
    provider: Arc<P>,
    prefix: String,
    handle: SnapshotHandle<AisixSnapshot>,
}

impl<P: ConfigProvider> Supervisor<P> {
    pub fn new(provider: Arc<P>, prefix: impl Into<String>) -> Self {
        Self {
            provider,
            prefix: prefix.into(),
            handle: SnapshotHandle::new(AisixSnapshot::new()),
        }
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
        let (snapshot, stats) = loader::build_snapshot(&self.prefix, &entries);
        tracing::info!(
            accepted = stats.accepted,
            rejected = stats.schema_rejected + stats.parse_rejected,
            revision,
            "initial snapshot built",
        );
        self.handle.store(snapshot);
        Ok(stats)
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
        }
        removed
    }

    /// Replace the current snapshot with a freshly loaded set (resync).
    pub fn apply_resync(&self, entries: &[RawEntry]) -> BuildStats {
        let (snap, stats) = loader::build_snapshot(&self.prefix, entries);
        self.handle.store(snap);
        stats
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
}
