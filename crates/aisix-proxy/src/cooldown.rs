//! Cooldown decision helper shared across every dispatch path.
//!
//! The proxy exposes more than one upstream endpoint family:
//!  - `/v1/chat/completions` (chat.rs)
//!  - `/v1/messages` (messages.rs — Anthropic-shape)
//!  - `/v1/responses` (responses.rs — OpenAI Responses API passthrough)
//!  - `/v1/audio/{speech,transcriptions,translations}` (audio.rs)
//!  - `/v1/rerank` (rerank.rs)
//!
//! Every one of those paths surfaces upstream failures as
//! [`BridgeError`]. The runtime-status layer (see [`crate::health`])
//! and the cross-request cooldown contract pinned by issue #264 apply
//! to **all** of them: a 401 from Anthropic via `/v1/messages` should
//! take that direct model out of rotation for the next request just
//! as a 401 via `/v1/chat/completions` does.
//!
//! This module owns the per-attempt cooldown decision. Each dispatch
//! path calls [`decide_cooldown`] after building a `BridgeError` and
//! before returning the error to the client. If a cooldown is
//! warranted, the caller hands the result to
//! [`crate::health::ModelRuntimeStatusTracker::mark_cooldown`].
//!
//! Keeping this logic in one place (rather than per-dispatch) is what
//! prevents the H-1 audit class of bug: a new dispatch path silently
//! forgets to cool down because the routing-loop in chat.rs is where
//! cooldown lived historically.

use std::time::Duration;

use aisix_core::CooldownConfig;
use aisix_gateway::BridgeError;

use crate::health::ModelRuntimeStatusTracker;

/// Decide whether a bridge error should trigger cooldown on the
/// failing direct model, and for how long.
///
/// Cooldown is **independent** of `is_retryable`. A 401 (auth failure)
/// is non-retryable — retrying the same target in the current request
/// is pointless — but it should still cooldown because every
/// subsequent request that lands on the same target will see the same
/// 401. Conversely, a transient timeout may be retryable AND should
/// also cooldown.
///
/// `Retry-After` from upstream is honored when `honor_retry_after`
/// is set (default true), clamped to `max_seconds`. A configured
/// `default_seconds: 0` is treated as "do not cool down on this
/// category" — matches the operator's likely intent of "disable
/// cooldown TTL" (M-2 audit on PR #268).
pub fn decide_cooldown(
    err: &BridgeError,
    cfg: Option<&CooldownConfig>,
) -> Option<(Duration, &'static str)> {
    let default_cfg = CooldownConfig::default();
    let cfg = cfg.unwrap_or(&default_cfg);
    if !cfg.enabled_or_default() {
        return None;
    }

    let default_secs = cfg.default_seconds_or_default();
    // A configured `default_seconds: 0` is a per-category disable.
    // The schema allows `minimum: 0` for this reason — operators that
    // want NO cooldown on any failure of this model set it to 0.
    if default_secs == 0 {
        return None;
    }

    let max = Duration::from_secs(cfg.max_seconds_or_default());
    let default_ttl = Duration::from_secs(default_secs);
    let clamp = |d: Duration| -> Duration { d.min(max) };

    match err {
        BridgeError::UpstreamStatus {
            status,
            retry_after,
            ..
        } => {
            let triggers = cfg.effective_trigger_statuses();
            if !triggers.contains(status) {
                return None;
            }
            let ttl = if cfg.honor_retry_after_or_default() {
                retry_after.map(clamp).unwrap_or(default_ttl)
            } else {
                default_ttl
            };
            Some((ttl, reason_for_status(*status)))
        }
        BridgeError::Timeout { .. } if cfg.trigger_on_timeout_or_default() => {
            Some((default_ttl, "request_timeout"))
        }
        BridgeError::Transport(_) | BridgeError::StreamAborted
            if cfg.trigger_on_transport_or_default() =>
        {
            Some((default_ttl, "transport_error"))
        }
        BridgeError::UpstreamDecode(_) if cfg.trigger_on_transport_or_default() => {
            Some((default_ttl, "upstream_decode_error"))
        }
        // Config errors mean WE are misconfigured (missing provider, bad
        // bridge registration). Cooling down doesn't help; let it
        // surface and operator fixes the snapshot.
        BridgeError::Config(_) => None,
        _ => None,
    }
}

/// Record a failed dispatch attempt against `model_id`: run the
/// cooldown decision and, if it fires, mark the runtime tracker.
/// Returns the error unchanged so call sites can keep using `?`.
///
/// This is the right entry point for every `.map_err` on the proxy's
/// upstream call paths — including the early `.send().await` and
/// body-decode failures that bypass the `!status.is_success()` branch.
/// The audit on PR #268 (round 2) caught exactly that gap:
/// transport / decode errors were threading through `?` without ever
/// hitting `mark_cooldown`, so a TCP-reset against Anthropic via
/// `/v1/messages` would leave the target healthy in routing.
pub fn note_failure(
    tracker: &ModelRuntimeStatusTracker,
    model_id: &str,
    cfg: Option<&CooldownConfig>,
    err: BridgeError,
) -> BridgeError {
    if let Some((ttl, reason)) = decide_cooldown(&err, cfg) {
        tracker.mark_cooldown(model_id, ttl, reason);
    }
    err
}

/// Map an HTTP status to a stable `status_reason` token surfaced on
/// `/admin/v1/models/status`. Kept narrow and operator-friendly —
/// callers should not synthesize their own reason strings.
pub fn reason_for_status(status: u16) -> &'static str {
    match status {
        401 => "upstream_auth_failure",
        408 => "upstream_request_timeout",
        429 => "upstream_rate_limited",
        500..=599 => "upstream_server_error",
        _ => "upstream_status_failure",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn upstream(status: u16) -> BridgeError {
        BridgeError::upstream_status(status, "boom")
    }

    #[test]
    fn default_seconds_zero_disables_cooldown_entirely() {
        // M-2 contract: 0 = off for any error category. Previously
        // (pre-audit fix) 0 silently re-derived as the max_seconds
        // cap, which surprised operators who expected disable.
        let cfg = CooldownConfig {
            default_seconds: Some(0),
            ..Default::default()
        };
        assert!(decide_cooldown(&upstream(429), Some(&cfg)).is_none());
        assert!(decide_cooldown(&upstream(503), Some(&cfg)).is_none());
        assert!(decide_cooldown(
            &BridgeError::Timeout {
                cause: String::new(),
                elapsed_ms: 1
            },
            Some(&cfg)
        )
        .is_none());
    }

    #[test]
    fn config_disabled_skips_cooldown() {
        let cfg = CooldownConfig {
            enabled: Some(false),
            ..Default::default()
        };
        assert!(decide_cooldown(&upstream(429), Some(&cfg)).is_none());
    }

    #[test]
    fn note_failure_marks_cooldown_for_transport_errors() {
        // Round-2 audit contract: a Transport error (TCP reset, DNS
        // failure, …) must mark cooldown when trigger_on_transport
        // is on (default). The non-status `?` paths in messages.rs /
        // responses.rs / audio.rs / rerank.rs all route through here.
        let tracker = ModelRuntimeStatusTracker::new();
        let err = BridgeError::Transport("connection refused".into());
        let returned = note_failure(&tracker, "m-1", None, err);
        // Error returned unchanged.
        assert!(matches!(returned, BridgeError::Transport(_)));
        // Tracker now reports cooldown for this target.
        assert_eq!(
            tracker.status("m-1").status,
            crate::health::RuntimeStatus::Cooldown
        );
        assert_eq!(
            tracker.status("m-1").status_reason.as_deref(),
            Some("transport_error")
        );
    }

    #[test]
    fn note_failure_marks_cooldown_for_decode_errors() {
        let tracker = ModelRuntimeStatusTracker::new();
        let err = BridgeError::UpstreamDecode("bad json".into());
        let _ = note_failure(&tracker, "m-1", None, err);
        assert_eq!(
            tracker.status("m-1").status,
            crate::health::RuntimeStatus::Cooldown
        );
    }

    #[test]
    fn note_failure_no_op_when_cooldown_disabled() {
        let tracker = ModelRuntimeStatusTracker::new();
        let cfg = CooldownConfig {
            enabled: Some(false),
            ..Default::default()
        };
        let err = BridgeError::Transport("nope".into());
        let _ = note_failure(&tracker, "m-1", Some(&cfg), err);
        assert_eq!(
            tracker.status("m-1").status,
            crate::health::RuntimeStatus::Healthy
        );
    }

    #[test]
    fn honor_retry_after_clamps_to_max_seconds() {
        let cfg = CooldownConfig {
            max_seconds: Some(60),
            ..Default::default()
        };
        let err = BridgeError::upstream_status_with_retry_after(
            429,
            "rl",
            Some(Duration::from_secs(100_000)),
        );
        let (ttl, _) = decide_cooldown(&err, Some(&cfg)).unwrap();
        assert_eq!(ttl, Duration::from_secs(60));
    }
}
