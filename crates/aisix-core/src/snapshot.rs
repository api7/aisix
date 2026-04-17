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
//! business types and will be filled in by PR #4 once `Model`, `ApiKey`,
//! `Team`, etc. exist. This crate provides the primitive only.

use crate::resource::{Resource, ResourceEntry};
use arc_swap::ArcSwap;
use dashmap::DashMap;
use std::sync::Arc;

/// Per-kind table with primary id-index and secondary name-index.
///
/// Both indices point at the same `Arc<ResourceEntry<T>>` so there is no
/// duplicate storage — the name map just holds ids.
#[derive(Debug)]
pub struct ResourceTable<T: Resource> {
    by_id: DashMap<String, Arc<ResourceEntry<T>>>,
    by_name: DashMap<String, String>,
}

impl<T: Resource> Default for ResourceTable<T> {
    fn default() -> Self {
        Self {
            by_id: DashMap::new(),
            by_name: DashMap::new(),
        }
    }
}

impl<T: Resource> ResourceTable<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Insert or replace an entry, updating both indices.
    ///
    /// If an entry with the same id already exists, the old name index entry
    /// is removed first (handles rename on update).
    pub fn insert(&self, entry: ResourceEntry<T>) {
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
        self.by_id.insert(id, Arc::new(entry));
    }

    /// Remove by id; also removes the matching name index entry.
    pub fn remove(&self, id: &str) -> Option<Arc<ResourceEntry<T>>> {
        let (_, entry) = self.by_id.remove(id)?;
        let name = entry.value.name().to_string();
        self.by_name.remove_if(&name, |_, v| v == id);
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
    /// does not hold DashMap shards.
    pub fn entries(&self) -> Vec<Arc<ResourceEntry<T>>> {
        self.by_id.iter().map(|kv| kv.value().clone()).collect()
    }
}

/// Handle consumers clone to reach the current snapshot.
///
/// `SnapshotHandle<S>` is the type actually stored in axum state — consumers
/// call [`SnapshotHandle::load`] on every request to get the current `Arc<S>`
/// without any locking.
#[derive(Debug, Clone)]
pub struct SnapshotHandle<S> {
    inner: Arc<ArcSwap<S>>,
}

impl<S> SnapshotHandle<S> {
    pub fn new(initial: S) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(initial)),
        }
    }

    /// Atomic load. Cheap (one Acquire load).
    pub fn load(&self) -> Arc<S> {
        self.inner.load_full()
    }

    /// Atomic store. Called by the etcd watch supervisor after building a
    /// fresh snapshot.
    pub fn store(&self, new: S) {
        self.inner.store(Arc::new(new));
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

    #[test]
    fn snapshot_handle_atomic_swap() {
        let handle: SnapshotHandle<u64> = SnapshotHandle::new(0);
        assert_eq!(*handle.load(), 0);
        handle.store(42);
        assert_eq!(*handle.load(), 42);
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
