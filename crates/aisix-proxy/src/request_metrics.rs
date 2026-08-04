//! The one chokepoint for the per-request outcome metrics every handler
//! emits at the end of dispatch.
//!
//! Three families ride on a single call:
//!
//! - `aisix_requests_total` / `aisix_request_duration_seconds` — the legacy
//!   compatibility series, four labels, every endpoint.
//! - `aisix_proxy_requests_total` / `aisix_proxy_failed_requests_total` /
//!   `aisix_proxy_request_duration_seconds` — the detailed series over ALL
//!   proxied traffic.
//! - `aisix_llm_requests_total` / `aisix_llm_request_duration_seconds` — the
//!   subset of the above that is a model-inference call, per
//!   [`LLM_ENDPOINTS`].
//!
//! Splitting the two tiers is the point: an MCP tool call, a batch-file
//! upload and a 413 are all proxy requests, but counting them as LLM
//! requests would corrupt every per-request token/cost average and the LLM
//! success rate. What is NOT a judgement call is that both tiers must cover
//! every endpoint — before AISIX-Cloud#1234 only chat + messages emitted the
//! detailed families at all, so ten endpoints were absent from the
//! success-rate and request-count queries built on them while still showing
//! up in the legacy series.
//!
//! Handlers call [`record`] instead of touching `Metrics` directly, and the
//! tier is decided from the endpoint rather than by the caller, so a new
//! endpoint cannot land with a half-wired label set — the same anti-drift
//! move `usage_attr` makes for the UsageEvent side.

use std::time::Duration;

use aisix_obs::{RequestLabels, RequestOutcome};

use crate::auth::AuthenticatedKey;
use crate::state::ProxyState;
use crate::usage_attr::provider_key_metric_name;

/// Label value every `RequestLabels` field falls back to when the path
/// never resolved it. Matches `RequestLabels::default()`.
const UNKNOWN: &str = "unknown";

/// Caller identity for the detailed label set.
#[derive(Clone, Copy)]
pub(crate) struct Caller<'a> {
    pub api_key_id: &'a str,
    pub team_id: &'a str,
    pub user_id: &'a str,
    pub user_name: &'a str,
}

impl<'a> Caller<'a> {
    pub(crate) fn new(auth: &'a AuthenticatedKey) -> Self {
        let key = auth.key();
        Self {
            api_key_id: &auth.entry.id,
            team_id: key.team_id.as_deref().unwrap_or(UNKNOWN),
            user_id: key.user_id.as_deref().unwrap_or(UNKNOWN),
            user_name: key.user_name.as_deref().unwrap_or(UNKNOWN),
        }
    }

    /// A path that gave up before it could attribute the request to a team
    /// or user — the pre-dispatch rejections. `api_key_id` is `Some` once
    /// the auth extractor has run and `None` for the middleware
    /// short-circuits that precede it (see `reject`).
    pub(crate) fn unattributed(api_key_id: Option<&'a str>) -> Self {
        Self {
            api_key_id: api_key_id.unwrap_or(UNKNOWN),
            team_id: UNKNOWN,
            user_id: UNKNOWN,
            user_name: UNKNOWN,
        }
    }
}

/// What the handler resolved about the upstream it reached, or tried to.
/// [`Upstream::default()`] is the shape of a request that failed before
/// resolution; a handler fills in only the fields its endpoint has.
#[derive(Clone, Copy)]
pub(crate) struct Upstream<'a> {
    pub provider: &'a str,
    /// MUST be bounded: a name that already resolved against the snapshot,
    /// or `usage_attr::metric_model_label()` output on any path that can
    /// fire before resolution. The raw client-supplied `model` is
    /// attacker-controlled cardinality (#451).
    pub model: &'a str,
    pub upstream_model: &'a str,
    pub provider_key_id: &'a str,
    pub stream: bool,
    pub is_fallback: bool,
}

impl Default for Upstream<'_> {
    fn default() -> Self {
        Self {
            provider: UNKNOWN,
            model: UNKNOWN,
            upstream_model: UNKNOWN,
            provider_key_id: UNKNOWN,
            stream: false,
            is_fallback: false,
        }
    }
}

/// Endpoints whose requests belong in the `aisix_llm_*` families on top of
/// the `aisix_proxy_*` ones — the model-inference routes.
///
/// Values are `normalize_endpoint_label` outputs; `llm_endpoints_are_reachable`
/// pins that, because a typo here fails silently (the entry simply never
/// matches, and the endpoint quietly drops out of every LLM query).
///
/// Deliberately absent, and why:
/// - `/mcp`, `/mcp/{server}`, `/a2a` — tool and agent calls, no model.
/// - `/passthrough/:provider/*rest` — an opaque tunnel; the gateway parses
///   nothing and cannot attribute a model.
/// - `/v1/files`, `/v1/batches`, `/v1/fine_tuning/jobs` — management calls.
/// - `/v1/realtime` — does reach a model, but feeds none of the
///   `aisix_llm_*_tokens_total` families (still chat + messages only), so
///   counting it here would inflate the denominator of every
///   tokens-per-request query.
const LLM_ENDPOINTS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/images/generations",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/rerank",
    "/v1/responses",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/audio/speech",
    "/v1/videos",
    "/v1/videos/:id",
];

/// Whether this endpoint's requests are model inference.
///
/// Keyed off the route, not the call site, so a request lands in the same
/// families however it ended — a 413 refused before dispatch has to sit in
/// the same denominator as the model-not-found 404 the handler itself
/// records, or a success rate over the endpoint silently omits one of them.
///
/// Anything unlisted is proxy-only, the safe default: a wrong `false` loses
/// a row from an LLM query, a wrong `true` corrupts every per-request token
/// and cost average built on these counters.
fn is_llm_endpoint(endpoint: &str) -> bool {
    LLM_ENDPOINTS.contains(&endpoint)
}

/// Terminal request-metric emit, shared by every handler.
///
/// `endpoint` must be a bounded route template — a literal for the fixed
/// routes, or [`crate::normalize_endpoint_label`] output for the `:param` /
/// wildcard ones. Never a raw request path (#451).
pub(crate) fn record(
    state: &ProxyState,
    endpoint: &'static str,
    caller: Caller<'_>,
    upstream: Upstream<'_>,
    status: u16,
    elapsed: Duration,
) {
    let outcome = RequestOutcome::from_status(status);
    state
        .metrics
        .record_request(upstream.provider, upstream.model, status, outcome, elapsed);
    // Held in a binding: `RequestLabels` borrows it.
    let provider_key_name = {
        let snap = state.snapshot.load();
        provider_key_metric_name(&snap, upstream.provider_key_id)
    };
    let labels = RequestLabels {
        endpoint,
        // Derived from the endpoint rather than passed in, so the detailed
        // families can't disagree with `aisix_proxy_in_flight_requests`
        // about which protocol a route speaks.
        inbound_protocol: crate::inbound_protocol_for_endpoint(endpoint),
        provider: upstream.provider,
        model: upstream.model,
        upstream_model: upstream.upstream_model,
        provider_key_id: upstream.provider_key_id,
        provider_key_name: &provider_key_name,
        api_key_id: caller.api_key_id,
        team_id: caller.team_id,
        user_id: caller.user_id,
        user_name: caller.user_name,
        stream: upstream.stream,
        is_fallback: upstream.is_fallback,
        status,
        outcome,
    };
    if is_llm_endpoint(endpoint) {
        state.metrics.record_proxy_and_llm_request(labels, elapsed);
    } else {
        state.metrics.record_proxy_request(labels, elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every registered proxy route, as its raw request path. Adding a route
    /// to `build_router` without adding it here leaves the tests below
    /// unable to see it — which is the point: the two assertions that follow
    /// are what force a new endpoint's `endpoint` label and LLM-vs-proxy
    /// tier to be decided rather than defaulted.
    const ROUTES: &[&str] = &[
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/images/generations",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/rerank",
        "/v1/responses",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/audio/speech",
        "/v1/videos",
        "/v1/videos/vid_abc123",
        "/v1/videos/vid_abc123/content",
        "/v1/realtime",
        "/v1/files",
        "/v1/files/file_abc123",
        "/v1/files/file_abc123/content",
        "/v1/batches",
        "/v1/batches/batch_abc123",
        "/v1/batches/batch_abc123/cancel",
        "/v1/fine_tuning/jobs",
        "/v1/fine_tuning/jobs/ft_abc123",
        "/mcp",
        "/mcp/some-server",
        "/a2a/some-agent",
        "/passthrough/openai/v1/anything",
    ];

    /// No proxy route may fall through to the `"other"` bucket. A route that
    /// does is invisible per-endpoint in every request series — which is how
    /// `/v1/videos` shipped (AISIX-Cloud#1234): it was registered in
    /// `build_router` but missing from the normalizer's allowlist, so all
    /// video traffic reported `endpoint="other"`.
    #[test]
    fn every_route_has_its_own_endpoint_label() {
        for route in ROUTES {
            assert_ne!(
                crate::normalize_endpoint_label(route),
                "other",
                "route {route} is missing from normalize_endpoint_label"
            );
        }
    }

    /// Guards against a typo in [`LLM_ENDPOINTS`]. An entry that no route
    /// normalizes to can never match, and the failure is silent: the
    /// endpoint just stops appearing in `aisix_llm_requests_total`, which is
    /// indistinguishable from having no traffic.
    #[test]
    fn llm_endpoints_are_reachable() {
        let reachable: Vec<&str> = ROUTES
            .iter()
            .map(|r| crate::normalize_endpoint_label(r))
            .collect();
        for endpoint in LLM_ENDPOINTS {
            assert!(
                reachable.contains(endpoint),
                "no route normalizes to {endpoint} — dead entry in LLM_ENDPOINTS"
            );
        }
    }

    /// The tier split itself: the inference routes carry the LLM series, the
    /// tool / management / tunnel surfaces carry only the proxy series.
    #[test]
    fn tiers_split_inference_from_the_rest() {
        for route in [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages/count_tokens",
            "/v1/embeddings",
            "/v1/audio/speech",
            "/v1/videos/vid_abc123/content",
        ] {
            assert!(
                is_llm_endpoint(crate::normalize_endpoint_label(route)),
                "{route} should count as an LLM request"
            );
        }
        for route in [
            "/mcp/some-server",
            "/a2a/some-agent",
            "/v1/realtime",
            "/v1/batches/batch_abc123",
            "/passthrough/openai/v1/anything",
            "/livez",
        ] {
            assert!(
                !is_llm_endpoint(crate::normalize_endpoint_label(route)),
                "{route} must not count as an LLM request"
            );
        }
    }
}
