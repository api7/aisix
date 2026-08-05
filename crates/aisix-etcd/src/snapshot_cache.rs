//! On-disk snapshot cache for offline resilience.
//!
//! Goal (prd-09 §9.7.2): the DP keeps serving `/v1/chat/completions`
//! from the last known etcd contents when the control plane is
//! unreachable — including across full container restarts.
//!
//! Approach:
//!
//! - The supervisor calls [`SnapshotCache::store`] after each
//!   successful apply (resync, put, delete). The on-disk file is the
//!   serialised list of [`RawEntry`] that produced the current
//!   in-memory snapshot, plus the etcd revision they came from.
//! - At boot, the supervisor calls [`SnapshotCache::load`] before
//!   touching etcd. If the file exists and parses, the entries are
//!   handed to `apply_resync` and the proxy starts serving traffic
//!   immediately. The first successful etcd `load_all` then
//!   overwrites the cache with fresh state.
//! - If etcd never comes back, the cached snapshot keeps serving
//!   forever — the DP is degraded (no new models / keys appear) but
//!   not down.
//!
//! Atomicity: `store` writes to `<path>.tmp` first, fsyncs, then
//! renames over the destination. A torn write never corrupts the
//! committed file. Disabled when [`SnapshotCache::disabled`] is used
//! (passed when `managed.snapshot_cache_path` is empty so operators
//! can opt out).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use crate::provider::RawEntry;
use crate::supervisor::StaleServing;

/// File-format version. Bumped whenever the wire shape of [`CachedFile`]
/// changes incompatibly so old DPs ignore future caches instead of
/// crashing on a stale-format upgrade. The `stale` section added for
/// #871 is additive (defaulted on read, ignored by older DPs), so it
/// stays at 1.
const FORMAT_VERSION: u32 = 1;

/// Owned, sync-or-async-safe snapshot cache. Cheap to clone — internally
/// holds an `Arc<Inner>` that serialises writes through a `Mutex`.
#[derive(Clone)]
pub struct SnapshotCache {
    inner: Arc<Inner>,
}

struct Inner {
    /// Some(path) → enabled; None → no-op.
    path: Option<PathBuf>,
    /// Serialise concurrent writes so the tmp-file rename dance can't
    /// race with itself. Reads are unguarded — they go through OS
    /// caches and the rename is atomic.
    write_lock: Mutex<()>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedFile {
    version: u32,
    revision: i64,
    entries: Vec<CachedEntry>,
    /// Last-known-good values pinned for keys whose current bytes (in
    /// `entries`) are rejected (#871). Restored before the boot resync so
    /// stale-serving rows survive a restart. Defaulted so pre-#871 cache
    /// files keep loading.
    #[serde(default)]
    stale: Vec<CachedStaleEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedEntry {
    key: String,
    /// Base64 because etcd values are byte arrays — typically JSON, but
    /// the cache must round-trip whatever the supervisor saw.
    value_b64: String,
    revision: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedStaleEntry {
    key: String,
    /// Base64 of the pinned last-known-good value bytes.
    value_b64: String,
    /// Revision the pinned value was accepted at.
    revision: i64,
    /// Unix seconds when stale serving began — persisted so the reported
    /// staleness age stays continuous across restarts.
    since_unix_secs: u64,
}

/// The parsed contents of a cache file: the raw entry set, the revision
/// it reflects, and the pinned last-known-good values (#871).
#[derive(Debug, Clone)]
pub struct CachedSnapshot {
    pub entries: Vec<RawEntry>,
    pub revision: i64,
    pub stale: Vec<StaleServing>,
}

impl SnapshotCache {
    /// Construct a cache backed by `path`. The file is created on the
    /// first successful [`Self::store`]; missing path on [`Self::load`]
    /// is treated as "no cache yet".
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            inner: Arc::new(Inner {
                path: Some(path.into()),
                write_lock: Mutex::new(()),
            }),
        }
    }

    /// No-op cache. Returned when persistence is disabled (e.g. when
    /// `managed.snapshot_cache_path` is empty). [`Self::load`] always
    /// returns `None` and [`Self::store`] is a quiet success.
    pub fn disabled() -> Self {
        Self {
            inner: Arc::new(Inner {
                path: None,
                write_lock: Mutex::new(()),
            }),
        }
    }

    /// Read the cached snapshot, or `None` if no usable file exists.
    /// "Usable" means: the file is present, parses as JSON, declares a
    /// recognised [`FORMAT_VERSION`], and every entry's value decodes.
    /// Anything else is logged and treated as cache-miss so a corrupt
    /// file can never wedge the DP.
    pub fn load(&self) -> Option<CachedSnapshot> {
        let path = self.inner.path.as_ref()?;
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "snapshot cache read failed");
                return None;
            }
        };
        let cached: CachedFile = match serde_json::from_slice(&bytes) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "snapshot cache parse failed; ignoring");
                return None;
            }
        };
        if cached.version != FORMAT_VERSION {
            tracing::warn!(
                got = cached.version,
                want = FORMAT_VERSION,
                "snapshot cache format mismatch; ignoring",
            );
            return None;
        }
        let entries: Result<Vec<_>, _> = cached
            .entries
            .into_iter()
            .map(|e| {
                B64.decode(&e.value_b64).map(|value| RawEntry {
                    key: e.key,
                    value,
                    revision: e.revision,
                })
            })
            .collect();
        let stale: Result<Vec<_>, _> = cached
            .stale
            .into_iter()
            .map(|s| {
                B64.decode(&s.value_b64).map(|value| StaleServing {
                    entry: RawEntry {
                        key: s.key,
                        value,
                        revision: s.revision,
                    },
                    since_unix_secs: s.since_unix_secs,
                })
            })
            .collect();
        match (entries, stale) {
            (Ok(entries), Ok(stale)) => Some(CachedSnapshot {
                entries,
                revision: cached.revision,
                stale,
            }),
            (Err(e), _) | (_, Err(e)) => {
                tracing::warn!(error = %e, "snapshot cache entry decode failed; ignoring");
                None
            }
        }
    }

    /// Write the given entries + revision + pinned last-known-good values
    /// atomically. Errors are logged-and-swallowed because losing the
    /// cache is not worth blowing up an otherwise-healthy DP — at worst
    /// the next restart rebuilds from etcd.
    pub async fn store(&self, entries: &[RawEntry], revision: i64, stale: &[StaleServing]) {
        let Some(path) = self.inner.path.clone() else {
            return;
        };
        let cached = CachedFile {
            version: FORMAT_VERSION,
            revision,
            entries: entries
                .iter()
                .map(|e| CachedEntry {
                    key: e.key.clone(),
                    value_b64: B64.encode(&e.value),
                    revision: e.revision,
                })
                .collect(),
            stale: stale
                .iter()
                .map(|s| CachedStaleEntry {
                    key: s.entry.key.clone(),
                    value_b64: B64.encode(&s.entry.value),
                    revision: s.entry.revision,
                    since_unix_secs: s.since_unix_secs,
                })
                .collect(),
        };
        let bytes = match serde_json::to_vec(&cached) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "snapshot cache serialise failed");
                return;
            }
        };
        let _guard = self.inner.write_lock.lock().await;
        if let Err(e) = atomic_write(&path, &bytes).await {
            tracing::warn!(error = %e, path = %path.display(), "snapshot cache write failed");
        }
    }
}

/// Write `bytes` to `path` atomically: write to a sibling tmp file,
/// fsync, rename. Survives crashes between any two of those steps.
async fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await?;
        }
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = tokio::fs::File::create(&tmp).await?;
        f.write_all(bytes).await?;
        f.sync_all().await?;
    }
    tokio::fs::rename(&tmp, path).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn entry(key: &str, value: &[u8], rev: i64) -> RawEntry {
        RawEntry {
            key: key.into(),
            value: value.to_vec(),
            revision: rev,
        }
    }

    #[tokio::test]
    async fn round_trips_entries() {
        let dir = tempdir().unwrap();
        let cache = SnapshotCache::new(dir.path().join("snap.json"));
        let entries = vec![
            entry("/aisix/models/m-1", br#"{"name":"m1"}"#, 7),
            entry("/aisix/api_keys/k-1", b"\xff\x00\x01raw", 8),
        ];
        cache.store(&entries, 42, &[]).await;

        let cached = cache.load().expect("cache file exists");
        assert_eq!(cached.revision, 42);
        assert_eq!(cached.entries, entries);
        assert!(cached.stale.is_empty());
    }

    #[tokio::test]
    async fn round_trips_stale_entries() {
        let dir = tempdir().unwrap();
        let cache = SnapshotCache::new(dir.path().join("snap.json"));
        let entries = vec![entry("/aisix/models/m-1", br#"{"bad":true}"#, 9)];
        let stale = vec![StaleServing {
            entry: entry("/aisix/models/m-1", br#"{"name":"last-good"}"#, 7),
            since_unix_secs: 1_770_000_000,
        }];
        cache.store(&entries, 9, &stale).await;

        let cached = cache.load().expect("cache file exists");
        assert_eq!(cached.stale, stale);
    }

    /// A pre-#871 cache file (no `stale` section) must keep loading.
    #[tokio::test]
    async fn legacy_file_without_stale_section_loads() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("snap.json");
        let legacy = serde_json::json!({
            "version": 1,
            "revision": 5,
            "entries": [
                {"key": "/aisix/models/m-1", "value_b64": B64.encode(br#"{"a":1}"#), "revision": 5}
            ],
        });
        tokio::fs::write(&path, serde_json::to_vec(&legacy).unwrap())
            .await
            .unwrap();
        let cached = SnapshotCache::new(&path).load().expect("legacy file loads");
        assert_eq!(cached.entries.len(), 1);
        assert!(cached.stale.is_empty());
    }

    #[tokio::test]
    async fn missing_file_returns_none() {
        let dir = tempdir().unwrap();
        let cache = SnapshotCache::new(dir.path().join("never-written.json"));
        assert!(cache.load().is_none());
    }

    #[tokio::test]
    async fn disabled_cache_is_a_noop() {
        let cache = SnapshotCache::disabled();
        cache.store(&[entry("/a", b"x", 1)], 1, &[]).await;
        assert!(cache.load().is_none());
    }

    #[tokio::test]
    async fn corrupt_file_is_treated_as_cache_miss() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("snap.json");
        tokio::fs::write(&path, b"this is not json").await.unwrap();
        let cache = SnapshotCache::new(&path);
        assert!(cache.load().is_none());
    }

    #[tokio::test]
    async fn unknown_format_version_is_ignored() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("snap.json");
        let bogus = serde_json::json!({
            "version": 99,
            "revision": 1,
            "entries": [],
        });
        tokio::fs::write(&path, serde_json::to_vec(&bogus).unwrap())
            .await
            .unwrap();
        let cache = SnapshotCache::new(&path);
        assert!(cache.load().is_none());
    }

    #[tokio::test]
    async fn store_overwrites_atomically() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("snap.json");
        let cache = SnapshotCache::new(&path);
        cache.store(&[entry("/a", b"v1", 1)], 1, &[]).await;
        cache.store(&[entry("/a", b"v2", 2)], 2, &[]).await;
        let cached = cache.load().unwrap();
        assert_eq!(cached.revision, 2);
        assert_eq!(cached.entries[0].value, b"v2".to_vec());
    }
}
