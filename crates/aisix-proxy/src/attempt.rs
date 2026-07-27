//! Per-attempt routing telemetry shared by the Model-Group dispatch
//! endpoints (#655).
//!
//! Each upstream attempt — the initial try, a same-target retry, or a
//! fallback to a different target — becomes its own `UsageEvent`. Failed
//! attempts carry zero tokens + error info; the winning attempt carries
//! the real tokens/cost. All attempts of one request share `request_id`
//! (the trace/group key) and are ordered by `index` — each attempt
//! is its own per-call log event.
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
    /// UUID of the concrete Model row this attempt dispatched to. For a
    /// direct (non-routing) request this equals the requested entry's
    /// id. Feeds the emitted event's `model_id` so pricing resolves
    /// against the TARGET, not the group (AISIX-Cloud#790) — group ids
    /// have no pricing rows.
    pub target_model_id: String,
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

/// Upper bound on the per-attempt `error_message` telemetry field.
///
/// Sized as a backstop, not as the real limit: an `UpstreamStatus`
/// message is already bounded to [`aisix_gateway::MAX_UPSTREAM_ERROR_MESSAGE_BYTES`]
/// (1 KiB) by the bridge, so a cap above that byte budget leaves the
/// bridge's bound as the only one that ever fires and the operator sees
/// the whole message the bridge kept. A tighter cap silently clipped it
/// a second time (AISIX-Cloud#1065).
const MAX_ATTEMPT_ERROR_MESSAGE_CHARS: usize = 2048;

/// Control-char-stripped, capped rendering of an error's `Display`.
///
/// Anything that a log reader would treat as a line break is dropped, so a
/// multi-line upstream body can't split the one-line-per-record shape of
/// the telemetry field and the access log. U+2028/U+2029 are listed
/// explicitly: they are `Zl`/`Zp`, not `Cc`, so `is_control()` lets them
/// through even though plenty of viewers break lines on them.
fn sanitize_error_message(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control() && !matches!(c, '\u{2028}' | '\u{2029}'))
        .take(MAX_ATTEMPT_ERROR_MESSAGE_CHARS)
        .collect()
}

/// Control-char-stripped error string for the per-attempt
/// `error_message` telemetry field (#655), capped at
/// [`MAX_ATTEMPT_ERROR_MESSAGE_CHARS`].
pub(crate) fn attempt_error_message(err: &BridgeError) -> String {
    sanitize_error_message(&err.to_string())
}

/// Failure class + reason for the access log's `error_kind` / `error`
/// fields.
///
/// Deliberately NOT [`attempt_error_from_proxy`]: that one leaves the
/// message empty for every non-bridge variant, which is fine for a
/// per-attempt record (the class is the point) but would put a failed
/// request back to carrying no reason at all — the gap this exists to
/// close. Here every variant contributes its `Display`, because the access
/// log is the one line an operator gets per request.
pub(crate) fn access_log_error(err: &ProxyError) -> (&'static str, String) {
    (err.kind(), sanitize_error_message(&err.to_string()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{UpstreamWire, MAX_UPSTREAM_ERROR_MESSAGE_BYTES};

    /// AISIX-Cloud#1093: the access log is the one line an operator gets
    /// per request, so EVERY failure has to name itself there — including
    /// the variants `attempt_error_from_proxy` deliberately leaves
    /// message-less because a per-attempt record only needs the class.
    #[test]
    fn access_log_error_names_every_variant_not_just_bridge_ones() {
        // The cause added for #1093 has to survive into the access log —
        // it is what separates "the upstream is slow" from "we never
        // reached it", which render identically without it.
        let (kind, msg) = access_log_error(&ProxyError::Bridge(BridgeError::Timeout {
            elapsed_ms: 7167,
            cause: "tcp connect error: Connection timed out (os error 110)".into(),
        }));
        assert_eq!(kind, "timeout");
        assert_eq!(
            msg,
            "upstream request timed out after 7167ms: \
             tcp connect error: Connection timed out (os error 110)"
        );

        // A non-bridge variant: `attempt_error_from_proxy` yields "" here,
        // which would put the access log right back to naming no cause.
        let not_found = ProxyError::ModelNotFound("model \"ghost\" not found".into());
        let (kind, msg) = access_log_error(&not_found);
        assert_eq!(kind, "model_not_found");
        assert!(msg.contains("ghost"), "{msg}");
        assert!(
            attempt_error_from_proxy(&not_found).1.is_empty(),
            "per-attempt records intentionally carry no message here — \
             that is why the access log needs its own helper"
        );
    }

    /// Control chars would break the one-line-per-request shape that makes
    /// the access log greppable.
    #[test]
    fn access_log_error_strips_control_chars_and_caps_length() {
        // U+2028/U+2029 are Zl/Zp rather than Cc, so `is_control()` alone
        // would forward them and a log viewer would break the record.
        let (_, msg) = access_log_error(&ProxyError::InvalidRequest(
            "bad\nrequest\tbody\u{2028}split\u{2029}again\r\n".into(),
        ));
        assert!(
            !msg.contains(['\n', '\r', '\t', '\u{2028}', '\u{2029}']),
            "{msg:?}"
        );
        assert!(msg.ends_with("badrequestbodysplitagain"), "{msg}");

        let long = ProxyError::InvalidRequest("x".repeat(MAX_ATTEMPT_ERROR_MESSAGE_CHARS * 2));
        let (_, msg) = access_log_error(&long);
        assert_eq!(msg.chars().count(), MAX_ATTEMPT_ERROR_MESSAGE_CHARS);
    }

    fn upstream_status(message: &str) -> BridgeError {
        BridgeError::UpstreamStatus {
            status: 400,
            message: message.to_string(),
            parsed: None,
            wire: UpstreamWire::OpenAI,
            retry_after: None,
        }
    }

    /// AISIX-Cloud#1065: an upstream error long enough to matter must
    /// survive into telemetry whole. A content-filter refusal — the
    /// shape that provoked the issue — runs past 256 chars, and the old
    /// cap clipped its tail, which is exactly where the actionable part
    /// (the link explaining the policy) sits. Hence a fixture that is
    /// prose ending in a URL, not a run of filler: what has to survive
    /// is the END of a realistically long message.
    #[test]
    fn long_upstream_message_is_not_clipped() {
        let upstream = "The response was filtered because the prompt triggered \
             the provider's content management policy. Please modify your prompt \
             and retry. To learn more about the content filtering policies that \
             apply here, read the documentation at \
             https://upstream.example/docs/content-filtering";
        assert!(
            upstream.len() > 256,
            "fixture must exceed the old cap to be a regression test"
        );

        let got = attempt_error_message(&upstream_status(upstream));

        assert!(
            got.ends_with("https://upstream.example/docs/content-filtering"),
            "message tail was clipped: {got}"
        );
        assert!(got.contains(upstream), "message body was altered: {got}");
    }

    /// The cap sits above the bridge's own byte bound, so anything the
    /// bridge already truncated passes through untouched — the bridge
    /// stays the single limit that fires.
    #[test]
    fn cap_clears_the_bridge_message_bound() {
        let bridge_capped = "x".repeat(MAX_UPSTREAM_ERROR_MESSAGE_BYTES);
        let got = attempt_error_message(&upstream_status(&bridge_capped));
        assert!(
            got.contains(&bridge_capped),
            "a bridge-bounded message must reach telemetry whole"
        );
    }

    /// The cap is still a backstop: a bridge variant carrying an
    /// unbounded string (`Config`, here) can't write unbounded telemetry.
    #[test]
    fn pathological_message_still_hits_the_backstop() {
        let got = attempt_error_message(&BridgeError::Config("y".repeat(9000)));
        assert_eq!(got.chars().count(), MAX_ATTEMPT_ERROR_MESSAGE_CHARS);
    }

    /// Control characters stay stripped — a multi-line upstream body
    /// must not break the single-string telemetry field.
    #[test]
    fn control_chars_are_stripped() {
        let got = attempt_error_message(&upstream_status("line one\nline\ttwo"));
        assert!(got.ends_with("line onelinetwo"), "got: {got}");
    }
}
