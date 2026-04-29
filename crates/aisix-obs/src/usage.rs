//! Per-request usage events the proxy emits at end-of-request.
//!
//! The wire shape mirrors cp-api's `dpmgr_usage_events` table 1:1 —
//! see `aisix-cloud:internal/dpmgr/api/telemetry.go` and
//! `migrations/009_dpmgr_usage_events.up.sql` for the receiving end.
//!
//! Lifecycle:
//!
//! ```text
//! chat_completions handler --emit()--> [ mpsc::channel ] --drain--> sender worker --POST--> /dp/telemetry
//!         (proxy crate)                                                 (server crate)              (cp-api)
//! ```
//!
//! Why split sink + worker:
//!
//! - The PROXY crate (which calls `emit()` from request handlers) only
//!   needs a cheap clonable handle. It can't depend on the SERVER crate
//!   without creating a cycle (server already depends on proxy).
//! - The SERVER crate owns the worker because telemetry batching, the
//!   mTLS reqwest client, and graceful-shutdown wiring naturally live
//!   alongside `register::register_and_persist` and `heartbeat::spawn`.
//! - This module sits in `aisix-obs` (proxy already depends on it for
//!   metrics + access_log + langfuse), exposes the data type and the
//!   sink wrapper, and lets server-side wire up the consumer.
//!
//! See prd-09a §9A.7B Phase 1 for the upstream protocol; the DP-side
//! batch contract (5s interval / 100-event ceiling) lives in the worker
//! (aisix-server), not here.

use serde::Serialize;

/// One usage event. Emitted at end-of-request (success / upstream error /
/// guardrail block) per chat completion. Field shape pinned to the
/// cp-api wire (snake_case via serde).
///
/// All fields are Copy / String / `Option<String>` so the event is
/// cheap to construct on the request hot path. `costed in USD` per
/// the DP's pricing snapshot at request time — provider prices can
/// change post-hoc, but we record what was current when the request
/// ran.
///
/// `model_id` and `api_key_id` are optional: a guardrail-rejected
/// request may have neither (rejection runs before model resolution).
/// cp-api stores empty strings as SQL NULL; the field is `Option`
/// here so the JSON serialiser emits `""` (not `null`) — matches what
/// the cp-api parser expects.
#[derive(Debug, Clone, Serialize)]
pub struct UsageEvent {
    /// DP-supplied request id (idempotency key). Use the same id the
    /// `x-aisix-call-id` response header carries so logs join.
    pub request_id: String,

    /// Wall-clock time the upstream call completed, RFC 3339 (UTC).
    pub occurred_at: String,

    /// UUID of the v3 Model row this request resolved to. Empty when
    /// the request never reached model resolution.
    #[serde(default)]
    pub model_id: String,

    /// UUID of the v3 ApiKey row that authenticated this request.
    /// Empty when auth failed before resolution.
    #[serde(default)]
    pub api_key_id: String,

    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub latency_ms: u32,

    /// HTTP status code the proxy returned to the downstream caller.
    pub status_code: u16,

    /// Cost the DP computed for this request in US dollars. Zero when
    /// the request never reached cost calculation (e.g. blocked by a
    /// guardrail before dispatch).
    pub cost_usd: f64,

    /// True when a guardrail rejected the request (input or output).
    pub guardrail_blocked: bool,
}

/// Cheap clonable handle the proxy hands to request handlers. Backed
/// by an mpsc::Sender; `try_emit` is non-blocking and silently drops
/// the event if the worker's queue is full (avoids back-pressuring
/// the request hot path on a wedged CP). Drops are counted via
/// tracing::warn! so observers can see a wedge.
///
/// In a deployment without CP-side telemetry (legacy / dev), the
/// handle's `tx` is `None` and `try_emit` is a no-op.
#[derive(Debug, Clone)]
pub struct UsageSink {
    tx: Option<tokio::sync::mpsc::Sender<UsageEvent>>,
}

impl UsageSink {
    /// Build a real sink backed by an mpsc::Sender. The receiving end
    /// is owned by the worker spawned in aisix-server.
    pub fn new(tx: tokio::sync::mpsc::Sender<UsageEvent>) -> Self {
        Self { tx: Some(tx) }
    }

    /// Build a no-op sink. `try_emit` drops events silently — used
    /// when the DP runs without a configured CP (dev / standalone
    /// modes) so handlers don't have to special-case Optional fields.
    pub fn disabled() -> Self {
        Self { tx: None }
    }

    /// Non-blocking emit. Returns immediately:
    /// - `Ok(())` on enqueue success;
    /// - `Ok(())` and a tracing::warn! on `try_send` failure (queue
    ///   full / receiver dropped). We deliberately don't propagate the
    ///   error to the caller — request handlers must NOT fail because
    ///   telemetry can't keep up.
    /// - `Ok(())` no-op on a `disabled()` sink.
    pub fn try_emit(&self, event: UsageEvent) {
        let Some(tx) = &self.tx else { return };
        if let Err(err) = tx.try_send(event) {
            // Both `Full` and `Closed` end up here. Full = worker is
            // overloaded; Closed = worker shut down cleanly. Either
            // way the event is gone — log once per drop and move on.
            tracing::warn!(error = %err, "usage event dropped (sink full or closed)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_sink_is_a_noop() {
        let sink = UsageSink::disabled();
        // Doesn't panic; doesn't allocate a worker. Two emits in a
        // row also fine.
        sink.try_emit(sample_event("req-1"));
        sink.try_emit(sample_event("req-2"));
    }

    #[tokio::test]
    async fn emit_into_real_channel_arrives_in_order() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let sink = UsageSink::new(tx);
        sink.try_emit(sample_event("req-a"));
        sink.try_emit(sample_event("req-b"));
        let a = rx.recv().await.unwrap();
        let b = rx.recv().await.unwrap();
        assert_eq!(a.request_id, "req-a");
        assert_eq!(b.request_id, "req-b");
    }

    #[test]
    fn full_channel_drop_does_not_panic() {
        // Capacity-1 channel; the second emit can't enqueue. The drop
        // must not propagate — handlers can't tolerate a panic on the
        // hot path.
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let sink = UsageSink::new(tx);
        sink.try_emit(sample_event("req-1"));
        sink.try_emit(sample_event("req-2")); // dropped, logged
    }

    #[test]
    fn serialises_with_snake_case_field_names() {
        let ev = UsageEvent {
            request_id: "req-1".into(),
            occurred_at: "2026-04-29T12:00:00Z".into(),
            model_id: "mod-uuid".into(),
            api_key_id: "ak-uuid".into(),
            prompt_tokens: 12,
            completion_tokens: 34,
            latency_ms: 56,
            status_code: 200,
            cost_usd: 0.0012,
            guardrail_blocked: false,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains(r#""request_id":"req-1""#));
        assert!(json.contains(r#""api_key_id":"ak-uuid""#));
        assert!(json.contains(r#""prompt_tokens":12"#));
        assert!(json.contains(r#""completion_tokens":34"#));
        assert!(json.contains(r#""guardrail_blocked":false"#));
    }

    fn sample_event(id: &str) -> UsageEvent {
        UsageEvent {
            request_id: id.into(),
            occurred_at: "2026-04-29T12:00:00Z".into(),
            model_id: String::new(),
            api_key_id: String::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            latency_ms: 0,
            status_code: 200,
            cost_usd: 0.0,
            guardrail_blocked: false,
        }
    }
}
