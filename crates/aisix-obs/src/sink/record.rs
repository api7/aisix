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

impl SinkContent {
    /// Build captured content, truncating `prompt` and `response` to at most
    /// `max_bytes` each (on a UTF-8 char boundary so the result stays valid).
    /// `truncated` is set when either field was cut.
    pub fn capture(prompt: &str, response: &str, max_bytes: usize) -> Self {
        let (prompt, p_cut) = truncate_on_char_boundary(prompt, max_bytes);
        let (response, r_cut) = truncate_on_char_boundary(response, max_bytes);
        Self {
            prompt: prompt.to_owned(),
            response: response.to_owned(),
            truncated: p_cut || r_cut,
        }
    }
}

/// Truncate `s` to at most `max_bytes`, backing up to the previous UTF-8 char
/// boundary so the slice stays valid. Returns the slice and whether it was cut.
fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> (&str, bool) {
    if s.len() <= max_bytes {
        return (s, false);
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    (&s[..end], true)
}

/// Raw captured request/response content, handed from a request handler to the
/// exporter fan-out. The handler captures up to the largest `content_max_bytes`
/// among the env's content-capturing exporters (bounding the hot-path buffer);
/// each exporter then re-truncates to its own cap at delivery. `truncated`
/// records whether the *source* already exceeded the handler's capture cap, so
/// a per-exporter [`SinkContent`] reflects truncation from either stage.
///
/// This type exists ONLY on the exporter path — it never touches the
/// `UsageEvent` that feeds CP telemetry, so prompt/response content cannot
/// reach the control plane.
#[derive(Debug, Clone)]
pub struct CapturedContent {
    pub prompt: String,
    pub response: String,
    pub truncated: bool,
}

impl CapturedContent {
    /// Capture `prompt` and `response`, truncating each to `max_bytes` (on a
    /// UTF-8 char boundary) to bound the held buffer at the largest cap any
    /// content-capturing exporter requested. `truncated` is set if either was
    /// cut here; a per-exporter `SinkContent` ORs in its own (smaller) cap.
    pub fn new(prompt: &str, response: &str, max_bytes: usize) -> Self {
        let (prompt, p_cut) = truncate_on_char_boundary(prompt, max_bytes);
        let (response, r_cut) = truncate_on_char_boundary(response, max_bytes);
        Self {
            prompt: prompt.to_owned(),
            response: response.to_owned(),
            truncated: p_cut || r_cut,
        }
    }
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
    fn capture_keeps_short_content_untruncated() {
        let c = SinkContent::capture("hello", "world", 128);
        assert_eq!(c.prompt, "hello");
        assert_eq!(c.response, "world");
        assert!(!c.truncated);
    }

    #[test]
    fn capture_truncates_and_flags_oversize_content() {
        let big = "a".repeat(1000);
        let c = SinkContent::capture(&big, "ok", 100);
        assert_eq!(c.prompt.len(), 100);
        assert_eq!(c.response, "ok");
        assert!(c.truncated, "oversize prompt must set the truncated flag");
    }

    #[test]
    fn captured_content_new_bounds_each_field() {
        let cc = CapturedContent::new("hello", "world", 128);
        assert_eq!(cc.prompt, "hello");
        assert_eq!(cc.response, "world");
        assert!(!cc.truncated);

        let cc = CapturedContent::new(&"x".repeat(50), "ok", 10);
        assert_eq!(cc.prompt.len(), 10);
        assert!(cc.truncated);
    }

    #[test]
    fn capture_truncates_on_a_utf8_char_boundary() {
        // "你好" is 6 bytes (3 each); a 4-byte cap must back up to the 3-byte
        // boundary rather than split the second char, keeping valid UTF-8.
        let c = SinkContent::capture("你好", "", 4);
        assert_eq!(c.prompt, "你");
        assert!(c.truncated);
        // The result is shorter than the cap because it backed up to a boundary.
        assert_eq!(c.prompt.len(), 3);
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
