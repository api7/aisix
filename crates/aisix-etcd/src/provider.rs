//! The [`ConfigProvider`] abstraction the supervisor runs against.
//!
//! The real implementation is etcd (see [`crate::etcd_provider`]). Tests
//! plug in an in-memory provider so the supervisor can be exercised
//! deterministically without a container.
//!
//! Decoupling the supervisor from the concrete client also means the
//! future Admin API can write through the same trait on the happy path.

use async_trait::async_trait;
use std::sync::Arc;

/// Raw (key, value, revision) triple as returned by etcd ranges / watches.
/// Values are `serde_json::Value` so callers can run schema validation
/// before typed deserialisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub revision: i64,
}

/// Events surfaced to the watch consumer. `Resync` is emitted when the
/// supervisor has detected compaction or a reconnect, forcing a full
/// snapshot rebuild rather than delta application.
#[derive(Debug, Clone)]
pub enum WatchEvent {
    Put(RawEntry),
    Delete {
        key: String,
        revision: i64,
    },
    /// Full reload: supervisor has reloaded all entries under the prefix
    /// and is handing the whole set to the consumer in one atomic batch.
    Resync {
        entries: Arc<Vec<RawEntry>>,
        revision: i64,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("etcd connection failed: {0}")]
    Connect(String),
    #[error("etcd range request failed: {0}")]
    Range(String),
    #[error("etcd watch stream failed: {0}")]
    Watch(String),
    #[error("etcd revision was compacted — caller should resync")]
    Compacted,
}

/// Abstraction the supervisor depends on. Methods are async so the etcd
/// implementation can perform its gRPC I/O; test doubles can use channels.
#[async_trait]
pub trait ConfigProvider: Send + Sync + 'static {
    /// Full range read under the configured prefix. Returns the current
    /// entries plus the etcd revision at which the read was consistent;
    /// the supervisor starts its watch from `revision + 1`.
    async fn load_all(&self) -> Result<(Vec<RawEntry>, i64), ProviderError>;

    /// Open a watch stream starting from `start_revision`. The stream's
    /// items are individual events; `Resync` is *not* emitted on this
    /// channel — the supervisor is responsible for detecting compaction
    /// (via [`ProviderError::Compacted`]) and triggering a fresh
    /// `load_all` + `Resync` dispatch itself.
    async fn watch(
        &self,
        start_revision: i64,
    ) -> Result<
        Box<dyn futures::Stream<Item = Result<WatchEvent, ProviderError>> + Send + Unpin>,
        ProviderError,
    >;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_entry_is_cheap_to_clone() {
        let e = RawEntry {
            key: "/aisix/models/a".into(),
            value: b"{}".to_vec(),
            revision: 1,
        };
        let c = e.clone();
        assert_eq!(e, c);
    }

    #[test]
    fn provider_error_compacted_is_distinct() {
        let err = ProviderError::Compacted;
        assert_eq!(
            err.to_string(),
            "etcd revision was compacted — caller should resync"
        );
    }
}
