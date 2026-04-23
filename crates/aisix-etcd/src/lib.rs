//! aisix-etcd — etcd-backed [`ConfigProvider`] + watch supervisor.
//!
//! The gateway's hot read path talks to a `SnapshotHandle<AisixSnapshot>`
//! (see `aisix-core`). This crate is what *populates* that handle, running
//! a single supervisor task that:
//!
//! 1. Connects to etcd (5s × 5 retries on bootstrap — spec §2)
//! 2. Performs a full range read under the configured prefix
//! 3. Opens a watch stream from the next revision
//! 4. Applies Put / Delete events by copy-on-write replacing the snapshot
//! 5. Triggers a full resync on compaction
//! 6. Reconnects with exponential backoff (1→60s) on transport failure
//!
//! The [`ConfigProvider`] trait is the seam tests use to plug in an
//! in-memory provider and avoid a container dependency for unit testing.
//!
//! Spec references: §2 (config system), §3 (data models).

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod backoff;
pub mod etcd_provider;
pub mod key;
pub mod loader;
pub mod provider;
pub mod snapshot_cache;
pub mod supervisor;

pub use backoff::{ExpBackoff, BASE_MS, MAX_MS};
pub use etcd_provider::{
    ConnectPolicy, EtcdConfigProvider, EtcdWatchStream, CONNECT_MAX_ATTEMPTS,
    CONNECT_RETRY_INTERVAL,
};
pub use key::{parse as parse_key, KeyError, ResourceKey};
pub use loader::{build_snapshot, BuildStats};
pub use provider::{ConfigProvider, ProviderError, RawEntry, WatchEvent};
pub use snapshot_cache::SnapshotCache;
pub use supervisor::Supervisor;
