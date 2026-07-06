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
//!   alongside cert-bundle provisioning and `heartbeat::spawn`.
//! - This module sits in `aisix-obs` (proxy already depends on it for
//!   metrics + access_log + otlp_http_sink), exposes the data type and
//!   the sink wrapper, and lets server-side wire up the consumer.
//!
//! See prd-09a §9A.7B Phase 1 for the upstream protocol; the DP-side
//! batch contract (5s interval / 100-event ceiling) lives in the worker
//! (aisix-server), not here.

use aisix_core::{AppliedGuardrail, GuardrailMonitorHit};
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

    /// The model alias exactly as the client sent it in the request
    /// body (`model` field) — a Model-Group name for routed requests,
    /// a direct model's display name otherwise. `model_id` records the
    /// resolved TARGET model, so without this field the group a caller
    /// asked for appears nowhere in telemetry (AISIX-Cloud#790). Empty
    /// when the request never carried a resolvable model name.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub requested_model: String,

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

    /// Time to first token in milliseconds. Only meaningful on the
    /// streaming path — measures elapsed time from request entry to
    /// the first upstream SSE chunk. 0 on non-streaming, error, and
    /// cache-hit paths (omitted from the wire via skip_serializing_if).
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub ttft_ms: u32,

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

    /// The guardrails that governed this request, captured at chain-build
    /// time: each entry is the guardrail `kind` (e.g. `keyword`,
    /// `aliyun_text_moderation`) plus the `hook` it's configured for
    /// (`input` / `output` / `both`). Lets the dashboard show *which*
    /// guardrails ran — not just the boolean `guardrail_blocked`. v1 records
    /// the attached set, not per-guardrail verdicts (#379).
    ///
    /// Empty (the dominant guardrail-free deployment, or a request rejected
    /// before guardrail resolution) is omitted from the wire via
    /// `skip_serializing_if`; cp-api stores absent as an empty set. cp-api's
    /// `/dp/telemetry` binds JSON leniently, so older CP images that don't
    /// know this field ignore it — the DP can ship it ahead of the CP.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub applied_guardrails: Vec<AppliedGuardrail>,

    /// Per-detector count of spans a `kind: "pii"` guardrail masked on this
    /// request (input + output merged), e.g. `{"email": 2}`. Detector names
    /// only — masked values are never captured (#932), so a compliance audit
    /// sees THAT redaction happened without the event becoming a PII sink.
    /// Empty (no redaction) is omitted from the wire; cp-api's `/dp/telemetry`
    /// binds JSON leniently, so older CP images ignore the unknown field.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub redacted_entity_counts: std::collections::BTreeMap<String, u32>,

    /// What each `enforcement_mode: monitor` guardrail WOULD have done to
    /// this request (AISIX-Cloud#562): one entry per suppressed Block
    /// (`would_block`, with the operator-facing reason) or suppressed mask
    /// (`would_mask`, with per-detector counts). Names only — never matched
    /// content (#153). Lets operators stage a policy and audit its hit rate
    /// in the dashboard before flipping it to `block`. Empty (no
    /// monitor-mode guardrail fired) is omitted from the wire; cp-api's
    /// `/dp/telemetry` binds JSON leniently, so older CP images ignore the
    /// unknown field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guardrail_monitor_hits: Vec<GuardrailMonitorHit>,

    /// Cache outcome on this request. One of:
    ///
    /// - `"hit"` — cached response served the request without a
    ///   round-trip to the upstream
    /// - `"miss"` — cache was consulted, no entry matched; the
    ///   upstream response was just stored
    /// - `"disabled"` — no enabled `cache_policy` in snapshot for
    ///   this env, the cache gate was closed
    ///
    /// Empty string = cache state unknown / not applicable (error
    /// paths that fail before the cache lookup). cp-api persists
    /// this to `dpmgr_usage_events.cache_status`; on the wire empty
    /// maps to NULL via `skip_serializing_if`. Source of truth is
    /// `aisix_proxy::chat::CacheStatus::as_str`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cache_status: String,

    /// On a cache HIT, the prompt tokens of the cached response — i.e.
    /// the input tokens the request *would* have spent on the upstream
    /// if the cache hadn't served it. Zero on miss / disabled / error.
    ///
    /// cp-api derives `cost_saved_usd` server-side by multiplying these
    /// counters by the model's pricing (same pattern as `cost_usd` on
    /// non-cache rows — the DP doesn't own the pricing catalog).
    /// Surfacing tokens (not USD) here keeps pricing changes a cp-api-
    /// only deploy and lets the dashboard show "tokens saved" too.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_hit_saved_input_tokens: u32,

    /// On a cache HIT, the completion tokens of the cached response.
    /// Zero on miss / disabled / error. See `cache_hit_saved_input_tokens`
    /// for the full pricing-derivation story.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_hit_saved_output_tokens: u32,

    /// Which client-facing protocol the request used:
    ///
    /// - `"openai"` — `/v1/chat/completions` / `/v1/responses` /
    ///   `/v1/embeddings` / `/v1/audio/*` / `/v1/images/*` / `/v1/rerank`
    ///   (every OpenAI-shape endpoint family)
    /// - `"anthropic"` — `/v1/messages` (Anthropic SDK)
    ///
    /// Disambiguates the `provider` label which today reflects the
    /// **upstream** provider only — an Anthropic-SDK call routed at a
    /// non-Anthropic Model used to log `provider=openai` with no
    /// indication the inbound protocol was Anthropic. Empty string on
    /// the wire = legacy DP image; cp-api stores empty as NULL.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub inbound_protocol: String,

    // ─── Per-attempt telemetry (#655) ───
    //
    // Each UsageEvent now represents ONE upstream attempt. A request
    // that fails over emits multiple events sharing `request_id` (the
    // grouping/trace key); they are ordered by `attempt_index`. This
    // mirrors a per-call logging model — `status_code`,
    // `latency_ms`, and `ttft_ms` are scoped to THIS attempt, so the
    // user-perceived total is reconstructed by summing the attempts of
    // one `request_id`. Direct (non-routing) requests emit a single
    // event with attempt_index=0, attempt_kind="initial".
    /// 0-based index of this attempt within the request. Together with
    /// `request_id` it uniquely identifies one attempt.
    #[serde(default)]
    pub attempt_index: u32,

    /// What kind of attempt this is: `"initial"` (first try of the
    /// first target), `"retry"` (same target, after a retryable
    /// failure), or `"fallback"` (a different target than the previous
    /// attempt). Defaults to `"initial"`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub attempt_kind: String,

    /// Display name of the routing target this attempt actually used —
    /// the target that served (success) or failed (failure). Empty for
    /// direct-model requests and cache hits, where `model_id` already
    /// identifies the single model. Replaces the old `served_by_model`,
    /// which only carried the winning target.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub attempt_model: String,

    /// Error class for a FAILED attempt — a bounded, low-sensitivity
    /// label (e.g. `"upstream_status"`, `"timeout"`, `"transport"`).
    /// Empty on the successful attempt.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error_class: String,

    /// Short human-readable error message for a FAILED attempt
    /// (length-capped). Empty on the successful attempt.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error_message: String,

    // ─── ProviderKey telemetry attribution (#302 M17 / AISIX-Cloud#436) ───
    //
    // Mirrors `aisix_core::models::provider_key::TelemetryTags` 1:1 so
    // cp-api can slice usage events by who-paid-what (catalog vs BYO),
    // featured / community attribution, and operator-defined per-PK
    // labels. Sourced at request dispatch time from the resolved
    // `ProviderKey.telemetry_tags`; all five default to empty / false
    // for backward compat with legacy PK rows that pre-date Phase A.
    //
    // Empty / false on the wire maps to NULL on the cp-api side via
    // `skip_serializing_if` — `dpmgr_usage_events` columns are
    // nullable so legacy events written by older DP images don't
    // require a migration.
    /// `"catalog"` for first-party curated providers, `"byo"` for
    /// bring-your-own. Empty when the resolved ProviderKey predates
    /// telemetry attribution.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider_kind: String,

    /// Whether this ProviderKey is in the dashboard's "Featured"
    /// surface. Defaults to false; cp-api treats false as "not
    /// featured OR unknown" — slicing should not rely on this single
    /// bit alone for catalog/community segmentation.
    #[serde(default, skip_serializing_if = "is_false")]
    pub provider_featured: bool,

    /// Branded provider slug for catalog entries (e.g. `"openai"`,
    /// `"anthropic"`). Empty for BYO and legacy rows.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub branded_provider: String,

    /// Operator-defined label for this provider key (e.g.
    /// `"production"`, `"shared-test"`). Catalog-side only — BYO
    /// rows use `byo_label`. Empty when the operator did not set one.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub pk_label: String,

    /// Operator-defined label for BYO entries (e.g. an internal team
    /// name). Empty for catalog rows. Mutually exclusive with
    /// `pk_label` by convention; cp-api projection emits one or the
    /// other based on `provider_kind`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub byo_label: String,

    // ─── Client attribution (#492) ───
    /// Source IP of the downstream caller as resolved by the proxy's
    /// real-ip chain: the TCP peer, or the first untrusted address found
    /// walking the configured forwarded header (default `x-forwarded-for`)
    /// right-to-left when the peer is a trusted proxy (nginx
    /// `set_real_ip_from` + `real_ip_recursive` parity). Empty when the
    /// peer address was unavailable. cp-api stores empty as NULL.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub client_source_ip: String,

    /// Client `User-Agent` header verbatim (control chars stripped,
    /// length-capped). Surfaces the client type (e.g. `codex-cli/1.2`).
    /// Empty when the client sent none. cp-api stores empty as NULL.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub client_user_agent: String,

    // ─── MCP gateway attribution ───
    /// Registered name of the upstream MCP server a `tools/call` was routed
    /// to (the namespace prefix of the requested tool). Empty for non-MCP
    /// events; cp-api stores empty as NULL. Older cp-api images that predate
    /// this field ignore it (DP-first rollout).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub mcp_server_name: String,

    /// Name of the MCP tool invoked, without the server prefix. Empty for
    /// non-MCP events; cp-api stores empty as NULL.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub mcp_tool_name: String,

    // ─── A2A gateway attribution ───
    /// Registered name of the upstream A2A agent a request was routed to.
    /// Empty for non-A2A events; cp-api stores empty as NULL. Older cp-api
    /// images that predate this field ignore it (DP-first rollout).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub a2a_agent_name: String,

    /// The JSON-RPC method invoked on the A2A agent (such as `message/send`).
    /// Empty for non-A2A events; cp-api stores empty as NULL.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub a2a_method: String,
}

#[inline]
fn is_zero_u32(n: &u32) -> bool {
    *n == 0
}

#[inline]
fn is_false(b: &bool) -> bool {
    !*b
}

/// Cheap clonable handle the proxy hands to request handlers. Backed
/// by an mpsc::Sender; `try_emit` is non-blocking and silently drops
/// the event if the worker's queue is full (avoids back-pressuring
/// the request hot path on a wedged CP). Drops are counted via
/// tracing::warn! and (when a `Metrics` handle is attached) the
/// `aisix_usage_event_drops_total{reason}` prometheus counter so
/// observers can see a wedge.
///
/// In a deployment without CP-side telemetry (legacy / dev), the
/// handle's `tx` is `None` and `try_emit` is a no-op.
///
/// Issue #408: the sink also bumps `aisix_usage_events_emitted_total
/// {handler, status_code, inbound_protocol}` on every call so e2e
/// can externally assert emission without a cp-api receiver in the
/// loop. Counter is bumped on emission *intent* (i.e. every call to
/// `try_emit`); drops counter is the subset that failed to enqueue.
/// Invariant (audit HIGH-1): `emitted == delivered + dropped`. Every
/// emit increments exactly one of these:
/// - delivered: channel accepted the event
/// - dropped (reason=sink_full): worker overloaded
/// - dropped (reason=sink_closed): worker shut down
/// - dropped (reason=sink_disabled): no sink wired (legacy / dev mode)
#[derive(Debug, Clone)]
pub struct UsageSink {
    tx: Option<tokio::sync::mpsc::Sender<UsageEvent>>,
    metrics: Option<crate::metrics::Metrics>,
}

impl UsageSink {
    /// Build a real sink backed by an mpsc::Sender. The receiving end
    /// is owned by the worker spawned in aisix-server. No prometheus
    /// counter wiring until `with_metrics` is also called.
    pub fn new(tx: tokio::sync::mpsc::Sender<UsageEvent>) -> Self {
        Self {
            tx: Some(tx),
            metrics: None,
        }
    }

    /// Build a no-op sink. `try_emit` drops events silently — used
    /// when the DP runs without a configured CP (dev / standalone
    /// modes) so handlers don't have to special-case Optional fields.
    pub fn disabled() -> Self {
        Self {
            tx: None,
            metrics: None,
        }
    }

    /// Attach a Metrics handle so `try_emit` bumps the #408 emission
    /// and drops counters. Optional — without it, the sink behaves
    /// exactly as the pre-#408 sink (channel send only, no counters).
    /// The server bootstrap calls this in managed mode after building
    /// the shared `Metrics` instance.
    pub fn with_metrics(mut self, metrics: crate::metrics::Metrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Non-blocking emit. Returns immediately:
    /// - `Ok(())` on enqueue success;
    /// - `Ok(())` and a tracing::warn! on `try_send` failure (queue
    ///   full / receiver dropped). We deliberately don't propagate the
    ///   error to the caller — request handlers must NOT fail because
    ///   telemetry can't keep up.
    /// - `Ok(())` no-op on a `disabled()` sink.
    ///
    /// `handler` is a fixed-set label for the prometheus
    /// `aisix_usage_events_emitted_total` counter (#408): `"chat"`,
    /// `"embeddings"`, `"messages"`, `"responses"`, etc. Keep it
    /// `&'static str` so cardinality stays bounded.
    pub fn try_emit(&self, handler: &'static str, event: UsageEvent) {
        // Normalise inbound_protocol to a fixed `&'static str` set at
        // the boundary (audit MEDIUM-3). This both kills the heap
        // alloc per call AND pins prometheus cardinality at the type
        // level: a future caller that sets `event.inbound_protocol`
        // to user-controlled data still produces a bounded label.
        let bounded_protocol: &'static str = match event.inbound_protocol.as_str() {
            "openai" => "openai",
            "anthropic" => "anthropic",
            "mcp" => "mcp",
            _ => "other",
        };

        // Bump the emit counter on *intent* — handler tried to emit.
        // Audit HIGH-1: paired with a drops counter bump on every
        // failure path (including `sink_disabled`) so the invariant
        // `emitted == delivered + dropped` holds strictly.
        if let Some(m) = &self.metrics {
            m.record_usage_event_emit(handler, event.status_code, bounded_protocol);
        }

        let Some(tx) = &self.tx else {
            // No sink wired (legacy / dev mode). Counted as a drop
            // with reason=sink_disabled so operators can see "DP
            // intended to emit but no sink was wired" — silent
            // zeros would otherwise hide the misconfiguration.
            if let Some(m) = &self.metrics {
                m.record_usage_event_drop("sink_disabled");
            }
            return;
        };

        if let Err(err) = tx.try_send(event) {
            // `Full` = worker is overloaded; `Closed` = worker shut
            // down cleanly. Either way the event is gone — record the
            // distinction so the operator knows *why* the wedge.
            let reason = match err {
                tokio::sync::mpsc::error::TrySendError::Full(_) => "sink_full",
                tokio::sync::mpsc::error::TrySendError::Closed(_) => "sink_closed",
            };
            if let Some(m) = &self.metrics {
                m.record_usage_event_drop(reason);
            }
            tracing::warn!(reason = reason, "usage event dropped");
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
        sink.try_emit("test", sample_event("req-1"));
        sink.try_emit("test", sample_event("req-2"));
    }

    #[tokio::test]
    async fn emit_into_real_channel_arrives_in_order() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let sink = UsageSink::new(tx);
        sink.try_emit("test", sample_event("req-a"));
        sink.try_emit("test", sample_event("req-b"));
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
        sink.try_emit("test", sample_event("req-1"));
        sink.try_emit("test", sample_event("req-2")); // dropped, logged
    }

    /// Issue #408: a `try_emit` call with a Metrics handle attached
    /// must bump `aisix_usage_events_emitted_total` exactly once per
    /// call. The status_code label is bucketed (2xx / 4xx / 5xx)
    /// rather than raw to keep prometheus cardinality bounded.
    #[tokio::test]
    async fn emits_counter_increments_per_call() {
        let metrics = crate::metrics::Metrics::new(false);
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let sink = UsageSink::new(tx).with_metrics(metrics.clone());

        sink.try_emit(
            "chat",
            UsageEvent {
                status_code: 200,
                inbound_protocol: "openai".into(),
                ..Default::default()
            },
        );
        sink.try_emit(
            "embeddings",
            UsageEvent {
                status_code: 200,
                inbound_protocol: "openai".into(),
                ..Default::default()
            },
        );

        let rendered = metrics.render();
        // The exact text format is metrics-rs's choice; assert both
        // the metric name and label combinations are present, and
        // value reaches 1 per (handler, status_code, inbound_protocol).
        assert!(
            rendered.contains("aisix_usage_events_emitted_total"),
            "counter must appear in scrape:\n{rendered}",
        );
        assert!(
            rendered.contains("handler=\"chat\"") && rendered.contains("handler=\"embeddings\""),
            "per-handler labels must be present:\n{rendered}",
        );
        assert!(
            rendered.contains("status_code=\"2xx\""),
            "status_code must be bucketed (2xx), not raw 200:\n{rendered}",
        );
        assert!(
            rendered.contains("inbound_protocol=\"openai\""),
            "inbound_protocol label must be present:\n{rendered}",
        );
    }

    /// Issue #408 audit HIGH-2: when `try_send` fails the emit
    /// counter still bumps for *intent* and the drops counter
    /// bumps for *outcome*. Strict numeric invariant pinned:
    /// `emit_total == delivered + drops_total`. After two calls
    /// (one delivered, one dropped on a capacity-1 channel) the
    /// scrape must show `emit_total == 2` and `drops_total == 1`.
    /// A regression that double-bumped emit on drop, or skipped
    /// emit when the channel rejected, would fail here.
    #[tokio::test]
    async fn dropped_event_records_reason_and_keeps_emit_count() {
        let metrics = crate::metrics::Metrics::new(false);
        let (tx, _rx) = tokio::sync::mpsc::channel(1); // capacity 1
        let sink = UsageSink::new(tx).with_metrics(metrics.clone());

        let event = || UsageEvent {
            status_code: 200,
            inbound_protocol: "openai".into(),
            ..Default::default()
        };

        sink.try_emit("chat", event()); // delivered
        sink.try_emit("chat", event()); // dropped (channel full)

        let rendered = metrics.render();
        // Numeric assertions (audit HIGH-2): name-only checks let a
        // double-bump regression through. Pin exact values.
        let emit_value = parse_counter_value(
            &rendered,
            "aisix_usage_events_emitted_total",
            &[("handler", "chat"), ("status_code", "2xx")],
        );
        assert_eq!(emit_value, 2, "emit must be exactly 2:\n{rendered}");

        let drop_value = parse_counter_value(
            &rendered,
            "aisix_usage_event_drops_total",
            &[("reason", "sink_full")],
        );
        assert_eq!(
            drop_value, 1,
            "drops_total{{reason=sink_full}} must be exactly 1:\n{rendered}",
        );
    }

    /// Issue #408 audit MEDIUM-1: the `sink_closed` reason path must
    /// be covered independently of `sink_full`. A future refactor that
    /// swapped the labels would only be caught by exercising both
    /// arms of the match.
    #[tokio::test]
    async fn dropped_event_with_closed_receiver_records_sink_closed_reason() {
        let metrics = crate::metrics::Metrics::new(false);
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        drop(rx); // close the receiver before any send
        let sink = UsageSink::new(tx).with_metrics(metrics.clone());

        sink.try_emit(
            "chat",
            UsageEvent {
                status_code: 200,
                inbound_protocol: "openai".into(),
                ..Default::default()
            },
        );

        let rendered = metrics.render();
        let drop_value = parse_counter_value(
            &rendered,
            "aisix_usage_event_drops_total",
            &[("reason", "sink_closed")],
        );
        assert_eq!(
            drop_value, 1,
            "drops_total{{reason=sink_closed}} must be 1 when receiver was dropped:\n{rendered}",
        );
    }

    /// Issue #408 audit HIGH-1: a `disabled()` sink with metrics
    /// attached must still preserve `emitted == delivered + dropped`.
    /// The drop is recorded with `reason=sink_disabled` so operators
    /// can see "DP intended to emit but no sink was wired" rather
    /// than silent zeros.
    #[tokio::test]
    async fn disabled_sink_with_metrics_records_sink_disabled_drop() {
        let metrics = crate::metrics::Metrics::new(false);
        let sink = UsageSink::disabled().with_metrics(metrics.clone());

        sink.try_emit(
            "chat",
            UsageEvent {
                status_code: 200,
                inbound_protocol: "openai".into(),
                ..Default::default()
            },
        );
        sink.try_emit(
            "chat",
            UsageEvent {
                status_code: 200,
                inbound_protocol: "openai".into(),
                ..Default::default()
            },
        );

        let rendered = metrics.render();
        // Both calls bump emit; both calls bump drop with sink_disabled.
        // Invariant: emit (2) == delivered (0) + drops (2).
        let emit_value = parse_counter_value(
            &rendered,
            "aisix_usage_events_emitted_total",
            &[("handler", "chat"), ("status_code", "2xx")],
        );
        assert_eq!(
            emit_value, 2,
            "emit must be 2 even on disabled sink:\n{rendered}"
        );

        let drop_value = parse_counter_value(
            &rendered,
            "aisix_usage_event_drops_total",
            &[("reason", "sink_disabled")],
        );
        assert_eq!(
            drop_value, 2,
            "drops_total{{reason=sink_disabled}} must be 2:\n{rendered}",
        );
    }

    /// Issue #408 audit MEDIUM-3: a wire-level `inbound_protocol`
    /// outside the documented set must be normalised to `"other"`
    /// rather than landing on the metric as a user-controlled
    /// cardinality vector. This pins the boundary defence.
    #[tokio::test]
    async fn unknown_inbound_protocol_buckets_into_other() {
        let metrics = crate::metrics::Metrics::new(false);
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let sink = UsageSink::new(tx).with_metrics(metrics.clone());

        sink.try_emit(
            "chat",
            UsageEvent {
                status_code: 200,
                // A future / out-of-spec value that some future
                // caller might set. Must NOT land on the wire as a
                // label value.
                inbound_protocol: "evil-cardinality-bomb".into(),
                ..Default::default()
            },
        );

        let rendered = metrics.render();
        assert!(
            !rendered.contains("evil-cardinality-bomb"),
            "user-controlled inbound_protocol must not leak into the label:\n{rendered}",
        );
        assert!(
            rendered.contains("inbound_protocol=\"other\""),
            "unknown inbound_protocol must bucket to \"other\":\n{rendered}",
        );
    }

    /// Helper: pull the integer counter value from a prometheus
    /// scrape, matching by metric name and all required label pairs.
    /// Returns 0 if no matching line. Robust to label ordering in
    /// the scrape output.
    #[cfg(test)]
    fn parse_counter_value(scrape: &str, name: &str, labels: &[(&str, &str)]) -> u64 {
        for line in scrape.lines() {
            if !line.starts_with(&format!("{name}{{")) {
                continue;
            }
            let all_match = labels
                .iter()
                .all(|(k, v)| line.contains(&format!("{k}=\"{v}\"")));
            if !all_match {
                continue;
            }
            // Format: `metric{labels} <value>`
            if let Some(value_str) = line.rsplit_once(' ').map(|(_, v)| v.trim()) {
                if let Ok(v) = value_str.parse::<u64>() {
                    return v;
                }
            }
        }
        0
    }

    #[test]
    fn serialises_with_snake_case_field_names() {
        let ev = UsageEvent {
            request_id: "req-1".into(),
            occurred_at: "2026-04-29T12:00:00Z".into(),
            model_id: "mod-uuid".into(),
            api_key_id: "ak-uuid".into(),
            requested_model: "smart-group".into(),
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
        // AISIX-Cloud#790: the client-sent alias rides next to model_id
        // so the dashboard can show the group a routed request used.
        assert!(json.contains(r#""requested_model":"smart-group""#));
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
        assert!(!json.contains("ttft_ms"));
        // ProviderKey telemetry tag wire-compat (#302 M17 /
        // AISIX-Cloud#436). Pre-attribution DP images would emit
        // empty / false defaults, which must NOT appear on the wire.
        assert!(!json.contains("provider_kind"));
        assert!(!json.contains("provider_featured"));
        assert!(!json.contains("branded_provider"));
        assert!(!json.contains("pk_label"));
        assert!(!json.contains("byo_label"));
        // Client attribution (#492): absent when the proxy couldn't
        // resolve a peer / the client sent no User-Agent.
        assert!(!json.contains("client_source_ip"));
        assert!(!json.contains("client_user_agent"));
        // Requested alias (AISIX-Cloud#790): absent when the request
        // never carried a resolvable model name.
        assert!(!json.contains("requested_model"));
        // Applied guardrails (#379): absent when no guardrail governed the
        // request (the dominant guardrail-free deployment). Empty must not
        // appear on the wire — cp-api treats absent as the empty set.
        assert!(!json.contains("applied_guardrails"));
    }

    #[test]
    fn applied_guardrails_serialise_when_set() {
        // #379: a request governed by guardrails carries the attached set
        // (kind + hook) so the dashboard can show which guardrails ran.
        let ev = UsageEvent {
            request_id: "req-guarded".into(),
            guardrail_blocked: true,
            applied_guardrails: vec![
                AppliedGuardrail {
                    kind: "keyword".into(),
                    hook: "input".into(),
                },
                AppliedGuardrail {
                    kind: "aliyun_text_moderation".into(),
                    hook: "both".into(),
                },
            ],
            ..Default::default()
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains(r#""applied_guardrails""#));
        assert!(json.contains(r#""kind":"keyword""#));
        assert!(json.contains(r#""hook":"input""#));
        assert!(json.contains(r#""kind":"aliyun_text_moderation""#));
        assert!(json.contains(r#""hook":"both""#));

        // Empty set stays off the wire entirely.
        let empty = serde_json::to_string(&UsageEvent::default()).unwrap();
        assert!(!empty.contains("applied_guardrails"));
    }

    #[test]
    fn client_attribution_fields_serialise_when_set() {
        let ev = UsageEvent {
            request_id: "req-client".into(),
            client_source_ip: "203.0.113.7".into(),
            client_user_agent: "codex-cli/1.2".into(),
            ..Default::default()
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains(r#""client_source_ip":"203.0.113.7""#));
        assert!(json.contains(r#""client_user_agent":"codex-cli/1.2""#));
    }

    #[test]
    fn telemetry_tag_fields_serialise_when_set() {
        // Catalog PK with operator-defined pk_label. Mirrors the
        // shape cp-api projects via mustMarshalProviderKeyKV (kind +
        // featured + branded_provider + pk_label, byo_label empty).
        let ev = UsageEvent {
            request_id: "req-tags-catalog".into(),
            provider_kind: "catalog".into(),
            provider_featured: true,
            branded_provider: "openai".into(),
            pk_label: "production".into(),
            ..Default::default()
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"provider_kind\":\"catalog\""));
        assert!(json.contains("\"provider_featured\":true"));
        assert!(json.contains("\"branded_provider\":\"openai\""));
        assert!(json.contains("\"pk_label\":\"production\""));
        // byo_label stays out — catalog PK doesn't use it.
        assert!(!json.contains("byo_label"));
    }

    #[test]
    fn telemetry_tag_fields_byo_variant_serialises() {
        // BYO PK with operator-defined byo_label. Per cp-api's
        // mustMarshalProviderKeyKV the catalog/BYO branches are
        // mutually exclusive — pk_label stays empty here.
        let ev = UsageEvent {
            request_id: "req-tags-byo".into(),
            provider_kind: "byo".into(),
            provider_featured: false,
            byo_label: "internal-vllm".into(),
            ..Default::default()
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"provider_kind\":\"byo\""));
        assert!(json.contains("\"byo_label\":\"internal-vllm\""));
        // featured=false skipped, branded_provider+pk_label stay empty.
        assert!(!json.contains("provider_featured"));
        assert!(!json.contains("branded_provider"));
        assert!(!json.contains("pk_label"));
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
            ttft_ms: 123,
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
        assert!(json.contains(r#""ttft_ms":123"#));
    }

    #[test]
    fn per_attempt_fields_serialise_only_when_present() {
        // A failed fallback attempt: zero tokens, error info, target name.
        let failed = UsageEvent {
            request_id: "req-routing".into(),
            attempt_index: 0,
            attempt_kind: "initial".into(),
            attempt_model: "primary".into(),
            status_code: 502,
            error_class: "upstream_status".into(),
            error_message: "upstream returned 502".into(),
            latency_ms: 2000,
            ..Default::default()
        };
        let json = serde_json::to_string(&failed).unwrap();
        assert!(json.contains(r#""attempt_index":0"#));
        assert!(json.contains(r#""attempt_kind":"initial""#));
        assert!(json.contains(r#""attempt_model":"primary""#));
        assert!(json.contains(r#""error_class":"upstream_status""#));
        assert!(json.contains(r#""error_message":"upstream returned 502""#));

        // A winning fallback attempt of the same request shares request_id.
        let won = UsageEvent {
            request_id: "req-routing".into(),
            attempt_index: 1,
            attempt_kind: "fallback".into(),
            attempt_model: "secondary".into(),
            status_code: 200,
            ..Default::default()
        };
        let json = serde_json::to_string(&won).unwrap();
        assert!(json.contains(r#""attempt_kind":"fallback""#));
        assert!(json.contains(r#""attempt_model":"secondary""#));
        // No error fields on the winner.
        assert!(!json.contains("error_class"));
        assert!(!json.contains("error_message"));

        // A direct (non-routing) request omits the routing-only fields.
        let empty = serde_json::to_string(&UsageEvent::default()).unwrap();
        assert!(!empty.contains("attempt_kind"));
        assert!(!empty.contains("attempt_model"));
        assert!(!empty.contains("error_class"));
        assert!(!empty.contains("error_message"));
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
