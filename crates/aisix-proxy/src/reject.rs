//! Terminal handling for requests rejected BEFORE dispatch.
//!
//! Every dispatching handler ends by emitting one access-log line plus the
//! request metrics for whatever it did. The paths that give up *before*
//! dispatch — the body-cap middleware's `Content-Length` short-circuit and
//! the body-extractor rejection each handler unwraps at its top — used to
//! `return` a bare response instead, so an oversize request was invisible:
//! a caller saw `413`, the operator saw nothing in the access log and no
//! `aisix_proxy_requests_total` sample. "Client reports 413, gateway has no
//! record of the request" was indistinguishable from the request never
//! arriving.
//!
//! Route every pre-dispatch rejection through [`reject_before_dispatch`] so
//! the family can't drift again: the rendered envelope and the telemetry are
//! produced by the same call.

use std::time::Instant;

use aisix_obs::{AccessLog, RequestOutcome};
use axum::response::{IntoResponse, Response};

use crate::error::ProxyError;
use crate::state::ProxyState;
use crate::usage_attr::UNRESOLVED_MODEL_LABEL;

/// Metric `provider` label for a rejection that never reached routing.
/// Matches what the handlers' own pre-dispatch error paths (auth, 404)
/// already record, so the series doesn't fork.
const UNRESOLVED_PROVIDER_LABEL: &str = "unknown";

/// Which wire envelope the caller expects. The Anthropic-protocol routes
/// (`/v1/messages`, `/v1/messages/count_tokens`) must answer in Anthropic
/// shape or the Claude SDK can't parse the error (#336) — the rejection
/// path is no exception.
#[derive(Clone, Copy)]
pub(crate) enum Envelope {
    OpenAi,
    Anthropic,
}

/// Emit the access log + request metrics for a request refused before
/// dispatch, and render `err` into the caller's envelope.
///
/// `api_key_id` is `None` for the middleware short-circuit, which runs
/// ahead of authentication — the request is refused on its declared size
/// alone, before any credential is read.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reject_before_dispatch(
    state: &ProxyState,
    method: &str,
    path: &str,
    request_id: &str,
    api_key_id: Option<&str>,
    started: Instant,
    envelope: Envelope,
    err: ProxyError,
) -> Response {
    let status = err.status().as_u16();
    let elapsed = started.elapsed();
    let (error_kind, error) = crate::attempt::access_log_error(&err);
    AccessLog {
        method,
        path,
        status,
        latency: elapsed,
        // Nothing is resolved this early: no upstream was picked, and the
        // body naming the model is exactly what we refused to read.
        provider: None,
        model: None,
        api_key_id,
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
        error_kind: Some(error_kind),
        error: Some(&error),
    }
    .emit();
    state.metrics.record_request(
        UNRESOLVED_PROVIDER_LABEL,
        UNRESOLVED_MODEL_LABEL,
        status,
        RequestOutcome::from_status(status),
        elapsed,
    );
    match envelope {
        Envelope::OpenAi => err.into_response(),
        Envelope::Anthropic => err.into_anthropic_response(),
    }
}
