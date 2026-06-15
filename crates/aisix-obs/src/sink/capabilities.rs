//! Capability descriptors for observability sinks.
//!
//! Each [`super::ObservabilitySink`] declares a [`SinkCapabilities`] so the
//! shared delivery pipeline (and control-plane validation) can adapt to the
//! sink's wire contract. At-least-once HTTP posts, stable-filename object
//! writes, and offset-token exactly-once streaming differ in how a batch is
//! sized, ordered, deduplicated and retried — the capability matrix makes
//! those differences data, not a fork in the pipeline.

use serde::{Deserialize, Serialize};

/// How a sink deduplicates re-delivered data after a producer restart.
///
/// The producer owns the marker; the sink interprets it. See
/// [`IdempotencyMarker`] for the matching runtime value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdempotencyScheme {
    /// Snowpipe-Streaming-style channel-scoped offset token: the sink
    /// commits a monotonic offset and resumes after the last committed one.
    SnowpipeOffsetToken,
    /// Object-store-style stable, monotonic file name: a retried upload
    /// reuses the same key so downstream (e.g. Snowpipe) dedups by filename.
    FileSequence {
        /// Template for the object key, e.g. `{dp_node}-{worker}-{seq}.ndjson`.
        filename_template: String,
    },
    /// No server-side dedup. Delivery is at-least-once; a stable per-record
    /// sequence field lets a downstream system dedup if it chooses to.
    None,
}

/// The runtime value matching an [`IdempotencyScheme`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdempotencyMarker {
    /// Monotonic offset for [`IdempotencyScheme::SnowpipeOffsetToken`].
    OffsetToken(u64),
    /// File sequence number for [`IdempotencyScheme::FileSequence`].
    FileSeq(u64),
    /// No marker — at-least-once delivery.
    None,
}

/// Ordering guarantee the sink relies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderingScope {
    /// Records must be delivered in order within a channel (warehouse stream).
    PerChannel,
    /// No cross-record ordering requirement (independent HTTP posts).
    None,
}

/// What unit the pipeline uses to bound a batch handed to this sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchUnit {
    /// Bound by record count.
    Records,
    /// Bound by encoded byte size.
    Bytes,
    /// Bound by whichever limit is hit first.
    Both,
}

/// Identifies one ordered delivery channel for a sink.
///
/// Stable across DP restarts so offset-token / file-sequence resume works —
/// keyed by a control-plane-persisted node id, never an ephemeral
/// IP/hostname. At-least-once sinks (every `http_batch` target) never key
/// by channel and can ignore it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChannelKey {
    pub org_id: String,
    pub env_id: String,
    pub worker_idx: u32,
    /// CP-persisted DP node id, reused across pod restarts.
    pub dp_node_uid: String,
    /// Sink-specific destination (logstore / table / bucket-prefix).
    pub target: String,
}

/// Static description of how a sink wants to be driven by the shared pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SinkCapabilities {
    /// Dedup/replay scheme the sink implements.
    pub idempotency: IdempotencyScheme,
    /// Ordering the sink relies on.
    pub ordering: OrderingScope,
    /// How the pipeline sizes a batch for this sink.
    pub batch_unit: BatchUnit,
    /// Hard ceiling on a single batch's encoded size, in bytes. The pipeline
    /// never hands the sink a batch larger than this. `None` means the sink
    /// self-limits (no pipeline-enforced byte ceiling).
    pub max_batch_bytes: Option<u64>,
    /// True when the sink can accept part of a batch and report which records
    /// failed (e.g. Elasticsearch `_bulk`), so the pipeline retries only the
    /// failed tail. False = whole-batch all-or-nothing.
    pub supports_partial_batch: bool,
    /// True when the sink ingests row-by-row with its own commit protocol
    /// (e.g. Snowpipe Streaming) rather than discrete request/response posts.
    pub supports_streaming_ingest: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_least_once_capabilities_are_constructible() {
        // The shape an `http_batch` sink (e.g. Aliyun SLS) declares.
        let caps = SinkCapabilities {
            idempotency: IdempotencyScheme::None,
            ordering: OrderingScope::None,
            batch_unit: BatchUnit::Both,
            max_batch_bytes: Some(10 * 1024 * 1024),
            supports_partial_batch: false,
            supports_streaming_ingest: false,
        };
        assert_eq!(caps.idempotency, IdempotencyScheme::None);
        assert!(!caps.supports_streaming_ingest);
    }

    #[test]
    fn idempotency_scheme_and_marker_pair_up() {
        let scheme = IdempotencyScheme::FileSequence {
            filename_template: "{dp_node}-{worker}-{seq}.ndjson".into(),
        };
        // round-trips through serde so the CP can persist it
        let json = serde_json::to_string(&scheme).unwrap();
        let back: IdempotencyScheme = serde_json::from_str(&json).unwrap();
        assert_eq!(scheme, back);

        let marker = IdempotencyMarker::FileSeq(42);
        assert_ne!(marker, IdempotencyMarker::None);
    }
}
