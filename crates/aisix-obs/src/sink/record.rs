//! The canonical event a sink delivers, plus the batch wrapper.
//!
//! A [`SinkRecord`] wraps the existing per-request [`UsageEvent`] (the
//! canonical metadata, already mirrored to cp-api) and adds optional,
//! opt-in [`SinkContent`]. Reusing `UsageEvent` keeps the metadata schema
//! single-sourced; content lives in a separate field so the default
//! metadata-only path can never carry a prompt.

use std::sync::Arc;

use serde::Serialize;

use crate::usage::UsageEvent;

/// Current canonical event schema version, emitted on every [`SinkRecord`]
/// so downstream consumers can evolve safely.
pub const SCHEMA_VERSION: &str = "1.0";

/// Captured request/response content.
///
/// Populated ONLY when an exporter opts into full-content capture
/// (`content_mode = full`); absent by default so the metadata path cannot
/// leak prompts. Size caps / truncation are applied before this is built.
#[derive(Debug, Clone, Serialize)]
pub struct SinkContent {
    /// The request prompt — serialized chat messages (JSON) or raw text.
    pub prompt: String,
    /// The assembled response text (full, post-stream).
    pub response: String,
    /// True when either field was truncated to a configured size cap.
    pub truncated: bool,
}

/// One canonical observability event handed to sinks.
///
/// Sink body-encoders read fields off `usage` (and optionally `content`) to
/// build their wire payload.
#[derive(Debug, Clone, Serialize)]
pub struct SinkRecord {
    /// Canonical schema version. See [`SCHEMA_VERSION`].
    pub schema_version: &'static str,
    /// The per-request metadata (flattened into the record on the wire).
    #[serde(flatten)]
    pub usage: UsageEvent,
    /// Opt-in captured content; omitted entirely under `metadata_only`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<SinkContent>,
}

impl SinkRecord {
    /// Build a metadata-only record (no content captured).
    pub fn metadata_only(usage: UsageEvent) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            usage,
            content: None,
        }
    }

    /// Attach captured content (`content_mode = full`).
    pub fn with_content(mut self, content: SinkContent) -> Self {
        self.content = Some(content);
        self
    }
}

/// A batch of records the pipeline hands to a sink in one delivery.
///
/// Records are `Arc`-shared so the same record can fan out to several sinks
/// without copying the payload.
#[derive(Debug, Clone, Default)]
pub struct EventBatch {
    pub records: Vec<Arc<SinkRecord>>,
}

impl EventBatch {
    pub fn new(records: Vec<Arc<SinkRecord>>) -> Self {
        Self { records }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_only_record_omits_content() {
        let rec = SinkRecord::metadata_only(UsageEvent {
            request_id: "req-1".into(),
            ..UsageEvent::default()
        });
        let json = serde_json::to_value(&rec).unwrap();
        assert_eq!(json["schema_version"], "1.0");
        // content key is absent, and metadata is flattened (no `usage` nesting)
        assert!(json.get("content").is_none());
        assert_eq!(json["request_id"], "req-1");
    }

    #[test]
    fn full_content_record_carries_prompt_and_response() {
        let rec = SinkRecord::metadata_only(UsageEvent::default()).with_content(SinkContent {
            prompt: "hi".into(),
            response: "hello".into(),
            truncated: false,
        });
        let json = serde_json::to_value(&rec).unwrap();
        assert_eq!(json["content"]["prompt"], "hi");
        assert_eq!(json["content"]["response"], "hello");
    }

    #[test]
    fn batch_len_tracks_records() {
        let batch = EventBatch::new(vec![Arc::new(SinkRecord::metadata_only(
            UsageEvent::default(),
        ))]);
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
        assert!(EventBatch::default().is_empty());
    }
}
