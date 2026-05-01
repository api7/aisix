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
#[derive(Debug, Clone, Default, Serialize)]
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

    /// OpenAI prompt-cache hit count. Subset of `prompt_tokens`.
    /// Defaults to 0 for providers that don't expose prompt caching.
    /// Serialised with `omitempty`-equivalent behaviour: cp-api accepts
    /// the absent-or-zero case identically.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cached_prompt_tokens: u32,

    /// OpenAI o1/o3 reasoning tokens. Subset of `completion_tokens`.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub reasoning_tokens: u32,

    /// Anthropic cache_creation_input_tokens. Separate counter on top
    /// of input_tokens; bills at ~1.25× prompt rate (per-model rate
    /// resolved by cp-api from model_pricing).
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_creation_tokens: u32,

    /// Anthropic cache_read_input_tokens. Separate counter on top of
    /// input_tokens; bills at ~0.10× prompt rate.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_read_tokens: u32,

    pub latency_ms: u32,

    /// HTTP status code the proxy returned to the downstream caller.
    pub status_code: u16,

    /// Provider response `id` — OpenAI's `chat.completion.id` or
    /// Anthropic's message `id`. Empty when the request never reached
    /// the upstream (guardrail block / pre-dispatch error).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider_request_id: String,

    /// Resolved model the provider actually billed (e.g.
    /// `gpt-4o-2024-08-06` when the request said `gpt-4o`). Differs
    /// from cp-api's `model_id` which points at the dashboard alias.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider_model_version: String,

    /// finish_reason / stop_reason from the upstream response. Empty
    /// for upstream errors and guardrail blocks.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub finish_reason: String,

    /// Cost the DP computed for this request in US dollars. Zero when
    /// the request never reached cost calculation (e.g. blocked by a
    /// guardrail before dispatch). cp-api recomputes this server-side
    /// from its pricing catalog; the DP-supplied value is dropped.
    pub cost_usd: f64,

    /// True when a guardrail rejected the request (input or output).
    pub guardrail_blocked: bool,

    /// Set when at least one remote-API guardrail (today: kind=bedrock)
    /// failed open: its upstream was unreachable but the operator
    /// configured `fail_open=true`, so the request went through. The
    /// reason ("bedrock_5xx" / "bedrock_timeout" / "bedrock_throttled")
    /// is what gets recorded so a compliance audit can identify
    /// requests that slipped past the policy. Empty string = no
    /// bypass (the normal Allow / Block paths). cp-api persists this
    /// to `dpmgr_usage_events.guardrail_bypassed_reason`; on the
    /// wire empty maps to NULL via `skip_serializing_if`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub guardrail_bypassed_reason: String,
}

#[inline]
fn is_zero_u32(n: &u32) -> bool {
    *n == 0
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
            ..Default::default()
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains(r#""request_id":"req-1""#));
        assert!(json.contains(r#""api_key_id":"ak-uuid""#));
        assert!(json.contains(r#""prompt_tokens":12"#));
        assert!(json.contains(r#""completion_tokens":34"#));
        assert!(json.contains(r#""guardrail_blocked":false"#));
    }

    #[test]
    fn cache_and_reasoning_fields_are_omitted_when_zero() {
        // Older DP builds and providers without cache support emit
        // events with these counters at 0. They must NOT appear in
        // the JSON — cp-api treats absent and 0 identically, but a
        // wire-compat regression here would inflate the request size
        // for every event.
        let ev = UsageEvent {
            request_id: "req-1".into(),
            ..Default::default()
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(!json.contains("cached_prompt_tokens"));
        assert!(!json.contains("reasoning_tokens"));
        assert!(!json.contains("cache_creation_tokens"));
        assert!(!json.contains("cache_read_tokens"));
        assert!(!json.contains("provider_request_id"));
        assert!(!json.contains("provider_model_version"));
        assert!(!json.contains("finish_reason"));
    }

    #[test]
    fn cache_and_reasoning_fields_serialise_when_set() {
        let ev = UsageEvent {
            request_id: "req-2".into(),
            prompt_tokens: 1000,
            completion_tokens: 200,
            cached_prompt_tokens: 500,
            reasoning_tokens: 50,
            cache_creation_tokens: 100,
            cache_read_tokens: 80,
            provider_request_id: "chatcmpl-abc".into(),
            provider_model_version: "gpt-4o-2024-08-06".into(),
            finish_reason: "stop".into(),
            ..Default::default()
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains(r#""cached_prompt_tokens":500"#));
        assert!(json.contains(r#""reasoning_tokens":50"#));
        assert!(json.contains(r#""cache_creation_tokens":100"#));
        assert!(json.contains(r#""cache_read_tokens":80"#));
        assert!(json.contains(r#""provider_request_id":"chatcmpl-abc""#));
        assert!(json.contains(r#""provider_model_version":"gpt-4o-2024-08-06""#));
        assert!(json.contains(r#""finish_reason":"stop""#));
    }

    fn sample_event(id: &str) -> UsageEvent {
        UsageEvent {
            request_id: id.into(),
            occurred_at: "2026-04-29T12:00:00Z".into(),
            status_code: 200,
            ..Default::default()
        }
    }
}
