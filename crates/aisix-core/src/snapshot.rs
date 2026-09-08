//! Lock-free configuration snapshot.
//!
//! The data plane holds an `ArcSwap<Arc<Snapshot>>`. Reads are a single atomic
//! load — no mutex, no RCU dance in user code. Writes build a fresh snapshot
//! off the etcd watch thread and atomically replace the pointer (spec §2:
//! "no mutex on the read path, atomic replace on write").
//!
//! A [`Snapshot`] holds a [`ResourceTable<T>`] per entity kind. Each table
//! provides:
//! - O(1) `get_by_id` via a primary `DashMap<id, Arc<ResourceEntry<T>>>`
//! - O(1) `get_by_name` via a secondary `DashMap<name, id>` index
//! - `len()` / `iter()` for listing
//!
//! Concrete Snapshot shape (which tables it holds) lives closer to the
//! business types in `models::AisixSnapshot`. This crate provides the
//! primitive only.

use crate::resource::{Resource, ResourceEntry};
use arc_swap::ArcSwap;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Process-wide source of [`ResourceTable::generation`] stamps.
///
/// One counter for every table of every snapshot, so a stamp is unique
/// across kinds and across rebuilds. That is what lets a derived cache
/// compare two generations for equality and conclude "same rows": a
/// table rebuilt from scratch (resync, cache restore) takes fresh stamps
/// rather than replaying the previous snapshot's, so it can never
/// coincide with the value a cache is holding.
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Per-kind table with primary id-index and secondary name-index.
///
/// Both indices point at the same `Arc<ResourceEntry<T>>` so there is no
/// duplicate storage — the name map just holds ids.
#[derive(Debug)]
pub struct ResourceTable<T: Resource> {
    by_id: DashMap<String, Arc<ResourceEntry<T>>>,
    by_name: DashMap<String, String>,
    /// Cached entry count, maintained by [`ResourceTable::insert`] /
    /// [`ResourceTable::remove`]. DashMap's own `len()` / `is_empty()`
    /// visit every shard (a CAS pair per shard), so per-request
    /// emptiness checks on the hot path go through this counter
    /// instead — one relaxed load, O(1) regardless of shard count.
    count: AtomicUsize,
    /// Stamp bumped on every mutation of THIS table and carried across
    /// [`Clone`]. See [`ResourceTable::generation`].
    generation: AtomicU64,
}

/// Manual impl: the atomics are not `Clone`. The count is re-seeded
/// from the cloned map's length, which the etcd watch supervisor's
/// clone-then-mutate update cycle relies on being exact. The generation
/// is COPIED, not re-stamped: a clone holds the same rows, so a cache
/// keyed on it must not be invalidated by the copy-on-write cycle that
/// publishes an unrelated table's change.
impl<T: Resource> Clone for ResourceTable<T> {
    fn clone(&self) -> Self {
        let by_id = self.by_id.clone();
        let count = AtomicUsize::new(by_id.len());
        Self {
            by_id,
            by_name: self.by_name.clone(),
            count,
            generation: AtomicU64::new(self.generation.load(Ordering::Acquire)),
        }
    }
}

impl<T: Resource> Default for ResourceTable<T> {
    fn default() -> Self {
        Self {
            by_id: DashMap::new(),
            by_name: DashMap::new(),
            count: AtomicUsize::new(0),
            // Empty and never mutated: an unconfigured kind keeps
            // generation 0 forever, so two empty tables compare equal
            // and no consumer rebuilds anything for them.
            generation: AtomicU64::new(0),
        }
    }
}

impl<T: Resource> ResourceTable<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Invalidation key for anything DERIVED from this table's rows.
    ///
    /// Monotonic and unique process-wide: it changes whenever this table
    /// is mutated and is carried unchanged across [`Clone`], so a cache
    /// that keys on it rebuilds when — and only when — the rows it reads
    /// actually changed.
    ///
    /// **Key derived caches on this, never on
    /// [`SnapshotHandle::version`].** The snapshot version moves on every
    /// published write of any kind, so keying on it makes an API-key edit
    /// invalidate (say) the guardrail index, which is rebuilt
    /// synchronously on the next request that resolves one. That is the
    /// shape of AISIX-Cloud#1542. The snapshot version remains the right
    /// tool for "has ANY configuration changed", such as an install guard
    /// that must not publish an older build over a newer one.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Stamped AFTER the mutation lands, and released, so a reader that
    /// observes a new generation also observes the rows behind it.
    fn bump_generation(&self) {
        self.generation.store(
            NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
            Ordering::Release,
        );
    }

    /// Insert or replace an entry, updating both indices.
    ///
    /// If an entry with the same id already exists, the old name index entry
    /// is removed first (handles rename on update).
    pub fn insert(&self, entry: ResourceEntry<T>) {
        self.insert_arc(Arc::new(entry));
    }

    /// [`ResourceTable::insert`] for an entry already behind an `Arc` —
    /// the copy-on-write path, where the new table shares the previous
    /// snapshot's rows instead of deep-copying every payload.
    pub fn insert_arc(&self, entry: Arc<ResourceEntry<T>>) {
        let id = entry.id.clone();
        let name = entry.value.name().to_string();

        if let Some(old) = self.by_id.get(&id) {
            let old_name = old.value.name().to_string();
            if old_name != name {
                // Only clear the old mapping if it still points at us.
                self.by_name.remove_if(&old_name, |_, v| v == &id);
            }
        }

        self.by_name.insert(name, id.clone());
        // Provisional increment BEFORE the map insert, corrected after a
        // replace. Orders the count so it can only ever read high during
        // a mutation window, never low: the empty fast paths may take
        // one redundant full scan, but can never skip an entry that is
        // already visible in the map.
        self.count.fetch_add(1, Ordering::Relaxed);
        if self.by_id.insert(id, entry).is_some() {
            self.count.fetch_sub(1, Ordering::Relaxed);
        }
        self.bump_generation();
    }

    /// Remove by id; also removes the matching name index entry.
    pub fn remove(&self, id: &str) -> Option<Arc<ResourceEntry<T>>> {
        let (_, entry) = self.by_id.remove(id)?;
        self.count.fetch_sub(1, Ordering::Relaxed);
        let name = entry.value.name().to_string();
        self.by_name.remove_if(&name, |_, v| v == id);
        self.bump_generation();
        Some(entry)
    }

    pub fn get_by_id(&self, id: &str) -> Option<Arc<ResourceEntry<T>>> {
        self.by_id.get(id).map(|r| r.clone())
    }

    pub fn get_by_name(&self, name: &str) -> Option<Arc<ResourceEntry<T>>> {
        let id = self.by_name.get(name)?.clone();
        self.get_by_id(&id)
    }

    /// True if a different id already owns `name`. Used for duplicate-name
    /// detection on admin create/update (`self_id` = the id being updated,
    /// None for create).
    pub fn name_conflicts(&self, name: &str, self_id: Option<&str>) -> bool {
        match self.by_name.get(name) {
            Some(existing_id) => match self_id {
                Some(me) => existing_id.as_str() != me,
                None => true,
            },
            None => false,
        }
    }

    /// Snapshot of all entries. Callers get owned `Arc` clones, so iteration
    /// does not hold DashMap shards. O(1) when the table is empty — the
    /// per-request callers (exporter fan-out, policy scans) skip the
    /// all-shards walk on unconfigured deployments.
    pub fn entries(&self) -> Vec<Arc<ResourceEntry<T>>> {
        if self.is_empty() {
            return Vec::new();
        }
        self.by_id.iter().map(|kv| kv.value().clone()).collect()
    }

    /// True when any entry satisfies `pred`, without materialising the
    /// table into a `Vec`. Cheaper than `entries().iter().any(...)` on the
    /// hot path (no allocation, no per-row `Arc` clone). A DashMap shard
    /// guard is held during the scan, so `pred` must not call back into
    /// this table.
    pub fn any(&self, pred: impl Fn(&ResourceEntry<T>) -> bool) -> bool {
        !self.is_empty() && self.by_id.iter().any(|kv| pred(kv.value()))
    }

    /// The single entry satisfying `pred`, without materialising the
    /// table. Returns `(None, true)` when more than one entry matches so
    /// the caller can fail closed on an ambiguous lookup and log the
    /// misconfiguration — a security-sensitive resolver must never pick
    /// one of several matches silently. `(Some(_), false)` on exactly
    /// one match; `(None, false)` on none.
    pub fn find_unique_by(
        &self,
        pred: impl Fn(&ResourceEntry<T>) -> bool,
    ) -> (Option<Arc<ResourceEntry<T>>>, bool) {
        if self.is_empty() {
            return (None, false);
        }
        let mut found: Option<Arc<ResourceEntry<T>>> = None;
        for kv in self.by_id.iter() {
            if !pred(kv.value()) {
                continue;
            }
            if found.is_some() {
                return (None, true);
            }
            found = Some(kv.value().clone());
        }
        (found, false)
    }
}

/// Handle consumers clone to reach the current snapshot.
///
/// `SnapshotHandle<S>` is the type actually stored in axum state — consumers
/// call [`SnapshotHandle::load`] on every request to get the current `Arc<S>`
/// without any locking.
///
/// The manual `Clone` impl deliberately does *not* require `S: Clone` — the
/// handle only clones its inner `Arc`, the `S` is never duplicated.
#[derive(Debug)]
pub struct SnapshotHandle<S> {
    inner: Arc<ArcSwap<S>>,
    version: Arc<AtomicU64>,
}

impl<S> Clone for SnapshotHandle<S> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            version: Arc::clone(&self.version),
        }
    }
}

impl<S> SnapshotHandle<S> {
    pub fn new(initial: S) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(initial)),
            version: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Atomic load. Cheap (one Acquire load).
    pub fn load(&self) -> Arc<S> {
        self.inner.load_full()
    }

    /// Monotonic version counter. Incremented on every `store` / `rcu`.
    /// Consumers can compare this to detect snapshot changes without
    /// relying on `Arc` pointer identity (which suffers from the ABA
    /// problem when the allocator reuses addresses).
    ///
    /// This answers "has ANY configuration changed", which is almost
    /// never the question a cache of something DERIVED from the snapshot
    /// is asking. Key those on [`ResourceTable::generation`] of the
    /// tables they read instead: one API-key edit moves this counter, and
    /// a cache keyed on it is then rebuilt on the request path of every
    /// worker thread for a write it does not read (AISIX-Cloud#1542).
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Atomic store. Called by the etcd watch supervisor after building a
    /// fresh snapshot.
    pub fn store(&self, new: S) {
        self.inner.store(Arc::new(new));
        self.version.fetch_add(1, Ordering::Release);
    }

    /// Read-copy-update. Runs `f(current)` to produce a new snapshot,
    /// then commits the result with a CAS. If a concurrent `store` /
    /// `rcu` ran between the load and the CAS, the closure runs again
    /// against the latest snapshot. This is the only safe way to do a
    /// load-mutate-store on `ArcSwap`: the bare load + store sequence
    /// silently loses concurrent updates (see arc-swap::ArcSwap::rcu
    /// docs).
    ///
    /// `f` may be called more than once under contention, so it must
    /// be idempotent w.r.t. its input — clone the current snapshot and
    /// apply the same delta each time, do not pull side data from
    /// outside the closure that depends on a single observation.
    pub fn rcu<F>(&self, mut f: F)
    where
        F: FnMut(&S) -> S,
    {
        self.inner.rcu(|current| f(current.as_ref()));
        self.version.fetch_add(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Item {
        id: String,
        name: String,
    }

    impl Resource for Item {
        fn id(&self) -> &str {
            &self.id
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn kind() -> &'static str {
            "items"
        }
    }

    fn entry(id: &str, name: &str) -> ResourceEntry<Item> {
        ResourceEntry::new(
            id,
            Item {
                id: id.into(),
                name: name.into(),
            },
            1,
        )
    }

    #[test]
    fn insert_lookup_by_id_and_name() {
        let t = ResourceTable::<Item>::new();
        t.insert(entry("a-1", "alpha"));
        t.insert(entry("b-2", "beta"));

        assert_eq!(t.len(), 2);
        assert_eq!(t.get_by_id("a-1").unwrap().name(), "alpha");
        assert_eq!(t.get_by_name("beta").unwrap().id(), "b-2");
        assert!(t.get_by_name("missing").is_none());
    }

    #[test]
    fn rename_on_update_cleans_old_name_index() {
        let t = ResourceTable::<Item>::new();
        t.insert(entry("a-1", "alpha"));

        // Rename a-1 from alpha → aleph.
        t.insert(entry("a-1", "aleph"));

        assert_eq!(t.len(), 1);
        assert!(t.get_by_name("alpha").is_none());
        assert_eq!(t.get_by_name("aleph").unwrap().id(), "a-1");
    }

    #[test]
    fn duplicate_name_creates_conflict() {
        let t = ResourceTable::<Item>::new();
        t.insert(entry("a-1", "alpha"));
        assert!(t.name_conflicts("alpha", None));
        assert!(!t.name_conflicts("alpha", Some("a-1"))); // updating self is fine
        assert!(t.name_conflicts("alpha", Some("other")));
    }

    #[test]
    fn remove_clears_both_indices() {
        let t = ResourceTable::<Item>::new();
        t.insert(entry("a-1", "alpha"));
        assert!(t.remove("a-1").is_some());
        assert!(t.get_by_id("a-1").is_none());
        assert!(t.get_by_name("alpha").is_none());
    }

    /// The cached count must stay exact through every mutation shape:
    /// fresh insert, same-id replace, remove, remove-miss, and clone.
    #[test]
    fn cached_count_tracks_all_mutations() {
        let t = ResourceTable::<Item>::new();
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());

        t.insert(entry("a-1", "alpha"));
        t.insert(entry("b-2", "beta"));
        assert_eq!(t.len(), 2);
        assert!(!t.is_empty());

        // Same-id replace (update, incl. rename) must not double-count.
        t.insert(entry("a-1", "aleph"));
        assert_eq!(t.len(), 2);

        // Remove-miss must not decrement.
        assert!(t.remove("missing").is_none());
        assert_eq!(t.len(), 2);

        assert!(t.remove("a-1").is_some());
        assert_eq!(t.len(), 1);

        // Clone re-seeds the counter from the cloned map.
        let c = t.clone();
        assert_eq!(c.len(), 1);
        c.insert(entry("c-3", "gamma"));
        assert_eq!(c.len(), 2);
        assert_eq!(t.len(), 1); // original untouched

        assert!(t.remove("b-2").is_some());
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn snapshot_handle_atomic_swap() {
        let handle: SnapshotHandle<u64> = SnapshotHandle::new(0);
        assert_eq!(*handle.load(), 0);
        assert_eq!(handle.version(), 0);
        handle.store(42);
        assert_eq!(*handle.load(), 42);
        assert_eq!(handle.version(), 1);
    }

    #[test]
    fn version_increments_on_rcu() {
        let handle: SnapshotHandle<u64> = SnapshotHandle::new(0);
        assert_eq!(handle.version(), 0);
        handle.rcu(|v| v + 1);
        assert_eq!(handle.version(), 1);
        handle.rcu(|v| v + 1);
        assert_eq!(handle.version(), 2);
        assert_eq!(*handle.load(), 2);
    }

    #[test]
    fn handle_is_clone_and_share_the_same_cell() {
        let a: SnapshotHandle<u64> = SnapshotHandle::new(1);
        let b = a.clone();
        a.store(99);
        // b sees a's write — same underlying ArcSwap.
        assert_eq!(*b.load(), 99);
    }
}
