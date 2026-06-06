//! Pluggable observability-sink framework — the capability-typed adapter
//! contract that the shared delivery pipeline drives.
//!
//! One implementation per sink *family* (`http_batch`, `object_store`,
//! `warehouse_stream`, `otlp`); per-vendor behaviour (Aliyun SLS, Datadog,
//! …) is configuration plus pluggable encoder/signer traits, not a new
//! sink. The shared pipeline owns batching, retry/backoff, backpressure and
//! per-sink delivery state; a sink only encodes a batch and reports the
//! outcome.
//!
//! This module is the framework foundation (AISIX-Cloud#692, phase F1): the
//! trait + capability matrix + idempotency types. The shared pipeline (F2)
//! and the concrete sinks (SLS, …) build on it.

mod capabilities;
mod manager;
mod object_store;
mod pipeline;
mod record;
mod sls;

pub use capabilities::{
    BatchUnit, ChannelKey, IdempotencyMarker, IdempotencyScheme, OrderingScope, SinkCapabilities,
};
pub use manager::ExporterPipelines;
pub use object_store::{build_object_store_sink, ObjectStoreSink};
pub use pipeline::{PipelineConfig, SinkHandle, SinkPipeline, SinkStatsSnapshot};
pub use record::{CapturedContent, EventBatch, SinkContent, SinkRecord, SCHEMA_VERSION};
pub use sls::{resolve_sls_credential, AliyunSlsSink};

use async_trait::async_trait;

/// Health snapshot a sink reports for the circuit-breaker and the dashboard.
#[derive(Debug, Clone)]
pub struct SinkHealth {
    pub healthy: bool,
    /// Masked, human-readable reason when unhealthy. Never contains secrets.
    pub detail: Option<String>,
}

impl SinkHealth {
    pub fn healthy() -> Self {
        Self {
            healthy: true,
            detail: None,
        }
    }

    pub fn unhealthy(detail: impl Into<String>) -> Self {
        Self {
            healthy: false,
            detail: Some(detail.into()),
        }
    }
}

/// Acknowledgement returned by a successful (possibly partial) delivery.
#[derive(Debug, Clone, Default)]
pub struct SinkAck {
    /// Number of records the sink accepted on this call.
    pub accepted: usize,
    /// For partial-success sinks, the index of the first record that failed
    /// — the pipeline retries from there. `None` = whole batch accepted.
    pub first_failed: Option<usize>,
    /// The idempotency marker now durably committed, if any (offset-token /
    /// file-sequence sinks). `None` for at-least-once sinks.
    pub committed: Option<IdempotencyMarker>,
}

/// Why a delivery failed. Drives the pipeline's retry/backoff/drop logic.
#[derive(Debug, thiserror::Error)]
pub enum SinkError {
    /// Transient — worth retrying with backoff (network, timeout, 5xx,
    /// throttle/429).
    #[error("transient sink error: {0}")]
    Transient(String),
    /// Permanent for this batch — retrying it unchanged will fail again
    /// (auth/403, malformed payload, oversize). The pipeline stops hammering
    /// and surfaces a masked health error instead.
    #[error("permanent sink error: {0}")]
    Permanent(String),
}

impl SinkError {
    /// Whether the pipeline should retry this batch with backoff.
    pub fn is_transient(&self) -> bool {
        matches!(self, SinkError::Transient(_))
    }
}

/// Result of one [`ObservabilitySink::append_batch`] call.
pub type SinkResult = Result<SinkAck, SinkError>;

/// A pluggable delivery target for observability events.
///
/// Implementors are held as `Arc<dyn ObservabilitySink>` and driven by the
/// shared pipeline. The trait is deliberately small: encode-and-deliver one
/// batch, plus the hooks streaming sinks need (committed-marker resume) and
/// everyone needs (health). HTTP/object-store sinks get sensible defaults so
/// they stay simple.
#[async_trait]
pub trait ObservabilitySink: Send + Sync + 'static {
    /// Stable name used in logs/metrics labels and health reporting.
    fn name(&self) -> &str;

    /// How this sink wants to be driven (idempotency, ordering, batch
    /// sizing, partial-success).
    fn capabilities(&self) -> SinkCapabilities;

    /// Deliver one batch. The pipeline retains ownership of `batch` and
    /// retries on [`SinkError::Transient`]; `marker` is the idempotency
    /// marker for this batch (or [`IdempotencyMarker::None`] for
    /// at-least-once sinks).
    async fn append_batch(&self, batch: &EventBatch, marker: &IdempotencyMarker) -> SinkResult;

    /// On startup, the last marker this channel durably committed, so the
    /// pipeline can resume after a restart. Defaults to `None` —
    /// at-least-once sinks (every `http_batch` / `object_store` target)
    /// don't track one.
    async fn last_committed_marker(&self, _channel: &ChannelKey) -> Option<IdempotencyMarker> {
        None
    }

    /// Cheap liveness/connectivity probe for the circuit-breaker and the
    /// control-plane "test connection" affordance.
    async fn healthcheck(&self) -> SinkHealth;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usage::UsageEvent;
    use std::sync::Arc;

    /// A trivial at-least-once sink that records what it was handed —
    /// proves the trait is object-safe (`Arc<dyn ObservabilitySink>`) and
    /// usable with the default `last_committed_marker`.
    struct CountingSink {
        accepted: parking_lot::Mutex<usize>,
    }

    #[async_trait]
    impl ObservabilitySink for CountingSink {
        fn name(&self) -> &str {
            "counting"
        }

        fn capabilities(&self) -> SinkCapabilities {
            SinkCapabilities {
                idempotency: IdempotencyScheme::None,
                ordering: OrderingScope::None,
                batch_unit: BatchUnit::Both,
                max_batch_bytes: Some(10 * 1024 * 1024),
                supports_partial_batch: false,
                supports_streaming_ingest: false,
            }
        }

        async fn append_batch(
            &self,
            batch: &EventBatch,
            _marker: &IdempotencyMarker,
        ) -> SinkResult {
            *self.accepted.lock() += batch.len();
            Ok(SinkAck {
                accepted: batch.len(),
                ..SinkAck::default()
            })
        }

        async fn healthcheck(&self) -> SinkHealth {
            SinkHealth::healthy()
        }
    }

    #[tokio::test]
    async fn dyn_sink_accepts_a_batch_and_defaults_marker_to_none() {
        let sink: Arc<dyn ObservabilitySink> = Arc::new(CountingSink {
            accepted: parking_lot::Mutex::new(0),
        });
        let batch = EventBatch::new(vec![Arc::new(SinkRecord::metadata_only(
            UsageEvent::default(),
        ))]);

        let ack = sink
            .append_batch(&batch, &IdempotencyMarker::None)
            .await
            .expect("delivery succeeds");
        assert_eq!(ack.accepted, 1);

        let channel = ChannelKey {
            org_id: "o".into(),
            env_id: "e".into(),
            worker_idx: 0,
            dp_node_uid: "node-1".into(),
            target: "request-events".into(),
        };
        assert_eq!(sink.last_committed_marker(&channel).await, None);
        assert!(sink.healthcheck().await.healthy);
    }

    #[test]
    fn sink_error_transience_drives_retry() {
        assert!(SinkError::Transient("429".into()).is_transient());
        assert!(!SinkError::Permanent("403".into()).is_transient());
    }

    #[test]
    fn health_helpers_round_trip() {
        assert!(SinkHealth::healthy().healthy);
        let bad = SinkHealth::unhealthy("endpoint unreachable");
        assert!(!bad.healthy);
        assert_eq!(bad.detail.as_deref(), Some("endpoint unreachable"));
    }
}
