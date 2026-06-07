//! Per-attempt routing telemetry shared by the Model-Group dispatch
//! endpoints (#655).
//!
//! Each upstream attempt — the initial try, a same-target retry, or a
//! fallback to a different target — becomes its own `UsageEvent`. Failed
//! attempts carry zero tokens + error info; the winning attempt carries
//! the real tokens/cost. All attempts of one request share `request_id`
//! (the trace/group key) and are ordered by `index`. This mirrors
//! LiteLLM's per-call logging model.
//!
//! The type lives in its own module so `/v1/chat/completions`,
//! `/v1/messages`, and `/v1/responses` cannot drift apart on how they
//! classify and emit attempts.

use std::time::Instant;

use aisix_gateway::BridgeError;

use crate::error::ProxyError;

/// One recorded upstream attempt. See module docs.
#[derive(Clone)]
pub(crate) struct AttemptRecord {
    /// 0-based attempt index within the request.
    pub index: u32,
    /// `"initial"` (first try of the first target), `"retry"` (same
    /// target after a retryable failure), or `"fallback"` (a different
    /// target than the previous attempt).
    pub kind: &'static str,
    /// Routing target display name. Empty for direct (non-routing)
    /// models, where `model_id` already identifies the single model.
    pub target_model: String,
    /// Resolved ProviderKey UUID for this attempt's target — feeds the
    /// per-PK attribution tags on the emitted event. Empty when unknown.
    pub provider_key_id: String,
    /// This attempt's status (mapped upstream status / timeout on
    /// failure, 200 on success).
    pub status: u16,
    pub success: bool,
    /// Bounded error class (`routing_error_class`); empty on success.
    pub error_class: String,
    /// Short error message (length-capped); empty on success.
    pub error_message: String,
    /// This attempt's own wall-clock duration in ms.
    pub latency_ms: u32,
}

/// Per-attempt telemetry accumulated while serving one request. Direct
/// (non-routing) models record a single attempt with `target_model`
/// empty; routing groups record one entry per try.
#[derive(Clone, Default)]
pub(crate) struct RoutingTelemetry {
    pub attempts: Vec<AttemptRecord>,
    /// Display name of the most recently attempted target — drives the
    /// initial/retry/fallback classification in [`Self::begin_attempt`].
    last_target: Option<String>,
}

impl RoutingTelemetry {
    /// Build a telemetry log holding a single already-classified attempt
    /// (the streaming path, which only ever tries the first target).
    pub fn single(rec: AttemptRecord) -> Self {
        let last_target = Some(rec.target_model.clone());
        Self {
            attempts: vec![rec],
            last_target,
        }
    }

    /// Classify the next attempt against `display_name` and advance the
    /// last-target tracker. Returns `(index, kind)` to stamp onto the
    /// `AttemptRecord` the caller pushes once the attempt resolves. Call
    /// once per attempt, before dispatch.
    pub fn begin_attempt(&mut self, display_name: &str) -> (u32, &'static str) {
        let index = self.attempts.len() as u32;
        let kind = if self.attempts.is_empty() {
            "initial"
        } else if self.last_target.as_deref() != Some(display_name) {
            "fallback"
        } else {
            "retry"
        };
        self.last_target = Some(display_name.to_string());
        (index, kind)
    }

    pub fn attempt_count(&self) -> u32 {
        self.attempts.len() as u32
    }

    /// Number of attempts that moved to a different target than the
    /// previous one. Drives the access log's `routing_fallback_count`.
    pub fn fallback_count(&self) -> u32 {
        self.attempts
            .iter()
            .filter(|a| a.kind == "fallback")
            .count() as u32
    }

    /// The winning (successful) attempt, if any. None for all-failed and
    /// pre-dispatch-error requests.
    pub fn winner(&self) -> Option<&AttemptRecord> {
        self.attempts.iter().rfind(|a| a.success)
    }
}

/// Winning-attempt / failed-attempt classification stamped onto an
/// emitted `UsageEvent` (#655). Used by the `/v1/messages` and
/// `/v1/responses` emit helpers, which (unlike chat's `UsageExtras`)
/// carry the attempt fields as a small standalone bundle.
#[derive(Default, Clone)]
pub(crate) struct AttemptInfo {
    pub index: u32,
    /// `"initial"` / `"retry"` / `"fallback"`. Empty → wire default
    /// `"initial"`.
    pub kind: String,
    /// Routing target display name; empty for direct models.
    pub model: String,
    /// Bounded error class for a failed attempt; empty on success.
    pub error_class: String,
    /// Short error message for a failed attempt; empty on success.
    pub error_message: String,
}

impl AttemptInfo {
    pub fn from_record(rec: &AttemptRecord) -> Self {
        Self {
            index: rec.index,
            kind: rec.kind.to_string(),
            model: rec.target_model.clone(),
            error_class: rec.error_class.clone(),
            error_message: rec.error_message.clone(),
        }
    }
}

/// Bounded, low-sensitivity error class for the per-attempt `error_class`
/// telemetry field (#655).
pub(crate) fn routing_error_class(err: &BridgeError) -> &'static str {
    match err {
        BridgeError::Timeout { .. } => "timeout",
        BridgeError::UpstreamStatus { .. } => "upstream_status",
        BridgeError::UpstreamDecode(_) => "upstream_decode",
        BridgeError::Config(_) => "config",
        BridgeError::InvalidUpstreamConfig(_) => "invalid_config",
        BridgeError::InvalidUpstreamCredentials(_) => "invalid_credentials",
        BridgeError::Transport(_) => "transport",
        BridgeError::StreamAborted => "stream_aborted",
    }
}

/// Short, control-char-stripped error string for the per-attempt
/// `error_message` telemetry field (#655). Capped like `sanitize_tag`.
pub(crate) fn attempt_error_message(err: &BridgeError) -> String {
    err.to_string()
        .chars()
        .filter(|c| !c.is_control())
        .take(256)
        .collect()
}

/// Bounded error class + short message for a per-attempt record, derived
/// from a `ProxyError`. Bridge errors carry the upstream-mapped class +
/// message; everything else uses the DP-stable `ProxyError::kind`. Shared
/// by the `/v1/messages` and `/v1/responses` dispatch loops.
pub(crate) fn attempt_error_from_proxy(err: &ProxyError) -> (String, String) {
    match err {
        ProxyError::Bridge(be) => (
            routing_error_class(be).to_string(),
            attempt_error_message(be),
        ),
        other => (other.kind().to_string(), String::new()),
    }
}

/// Milliseconds elapsed since `started`, saturating at `u32::MAX`.
pub(crate) fn ms_since(started: Instant) -> u32 {
    started.elapsed().as_millis().min(u32::MAX as u128) as u32
}
