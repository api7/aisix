//! The one chokepoint for the per-request outcome metrics every handler
//! emits once dispatch has produced a response.
//!
//! # What `elapsed` measures, and why it is not end-to-end
//!
//! Handlers call [`record`] on their way out, so for a **streamed** response
//! the `elapsed` they pass is time to response START, not the full
//! generation — the SSE body has not been polled yet. Every duration series
//! fed from here therefore mixes two scopes: full request time for
//! non-streamed traffic, time-to-response-start for streamed. `chat.rs`
//! guards the SLO histogram against exactly this
//! (`record_request_e2e_latency` is called with the stream's own duration at
//! completion instead); nothing guards the three families below.
//!
//! Read a streaming p99 off `aisix_request_e2e_latency_seconds`, which is
//! recorded at stream completion. Do not read one off
//! `aisix_llm_request_duration_seconds` and expect end-to-end.
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
//! [`record_usage`] is the companion emit for what the request consumed —
//! the token and spend families, which had the same coverage problem, in
//! three different shapes (see its own docs).
//!
//! Handlers call [`record`] and [`record_usage`] instead of touching
//! `Metrics` directly, and the tier is decided from the endpoint rather than
//! by the caller, so a new endpoint cannot land with a half-wired label set —
//! the same anti-drift move `usage_attr` makes for the UsageEvent side.
//!
//! Two endpoints are model inference but report no tokens by nature —
//! `/v1/audio/speech` (billed per input character) and `/v1/videos` (per
//! video). They count as requests and contribute nothing to the token
//! families, so aggregate tokens-per-request is only meaningful per
//! `endpoint`, never summed across all of them.

use std::borrow::Cow;
use std::time::Duration;

use aisix_core::AisixSnapshot;
use aisix_obs::{LlmUsage, RequestLabels, RequestOutcome, UsageLabels};

use crate::auth::AuthenticatedKey;
use crate::state::ProxyState;
use crate::usage_attr::PkLabels;

/// Label value every `RequestLabels` field falls back to when the path
/// never resolved it. Matches `RequestLabels::default()`.
pub(crate) const UNKNOWN: &str = "unknown";

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

    /// Recover the caller from an api-key id alone.
    ///
    /// The streaming emits run from a detached task or a Drop guard that was
    /// handed an `api_key_id: &str` rather than the key itself, and threading
    /// the team / user / name triple down every dispatch signature to reach
    /// them would be a lot of plumbing for three labels. The id resolves back
    /// to the same row the auth extractor matched, so the labels come out
    /// identical to [`Caller::new`]; an id that no longer resolves (the key
    /// was deleted mid-stream) degrades to `unknown` rather than dropping the
    /// sample.
    pub(crate) fn from_api_key_id(
        snap: &aisix_core::AisixSnapshot,
        api_key_id: &'a str,
    ) -> Owned<'a> {
        let entry = snap.apikeys.get_by_id(api_key_id);
        let key = entry.as_ref().map(|e| &e.value);
        Owned {
            api_key_id,
            team_id: key.and_then(|k| k.team_id.clone()),
            user_id: key.and_then(|k| k.user_id.clone()),
            user_name: key.and_then(|k| k.user_name.clone()),
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

/// Owning form of [`Caller`], for the snapshot lookup whose strings cannot
/// outlive the guard. Call [`Owned::as_caller`] at the emit.
pub(crate) struct Owned<'a> {
    api_key_id: &'a str,
    team_id: Option<String>,
    user_id: Option<String>,
    user_name: Option<String>,
}

impl<'a> Owned<'a> {
    pub(crate) fn as_caller(&'a self) -> Caller<'a> {
        Caller {
            api_key_id: self.api_key_id,
            team_id: self.team_id.as_deref().unwrap_or(UNKNOWN),
            user_id: self.user_id.as_deref().unwrap_or(UNKNOWN),
            user_name: self.user_name.as_deref().unwrap_or(UNKNOWN),
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
    /// The attempt's ProviderKey id AND its readable name, resolved
    /// together by `usage_attr::ResolvedPk` (#941). Taking the pair rather
    /// than a bare id is deliberate: the name used to be looked up inside
    /// each emit, so a request paid one snapshot read per emit and a new
    /// call site could not tell it was doing so.
    pub pk: PkLabels<'a>,
    pub stream: bool,
    pub is_fallback: bool,
}

impl Default for Upstream<'_> {
    fn default() -> Self {
        Self {
            provider: UNKNOWN,
            model: UNKNOWN,
            upstream_model: UNKNOWN,
            pk: PkLabels::default(),
            stream: false,
            is_fallback: false,
        }
    }
}

/// The upstream a request had committed to when it FAILED, recovered from
/// the request's attribution cell.
///
/// A handler's failure branch holds a `ProxyError`, which carries no
/// upstream identity, so every failed request used to emit
/// [`Upstream::default()`] — `provider="unknown"`, no ProviderKey — even
/// when it had reached a real provider and been answered 5xx. That split
/// one ProviderKey's successes and failures across two label sets, so a
/// failure rate grouped by `provider` reported 0% for every real provider
/// and 100% for `unknown` (AISIX-Cloud#1325).
///
/// Names the LAST target the request selected. Under retry / fallback that
/// is the attempt whose error the caller was actually served, which is the
/// same attempt the access log's routing telemetry ends on.
///
/// Still `unknown` for a request that failed BEFORE selecting a target —
/// model-not-found, an input guardrail block, a budget refusal. Those
/// never reached a provider, so there is nothing to attribute and the
/// placeholder is the honest answer.
pub(crate) struct LastTarget<'a> {
    provider: String,
    upstream_model: &'a str,
    pk: Option<crate::usage_attr::ResolvedPk<'a>>,
}

impl<'a> LastTarget<'a> {
    /// `resolved` has to outlive the emit — read it into a local first
    /// (`let resolved = attribution::current().unwrap_or_default();`).
    pub(crate) fn new(snap: &AisixSnapshot, resolved: &'a crate::attribution::Resolved) -> Self {
        Self {
            // Lowercased here because the success path lowercases at its
            // own emit; a failure that spelled the vendor differently
            // would land on a second series for the same provider.
            provider: if resolved.provider.is_empty() {
                UNKNOWN.to_string()
            } else {
                resolved.provider.to_ascii_lowercase()
            },
            upstream_model: if resolved.upstream_model.is_empty() {
                UNKNOWN
            } else {
                &resolved.upstream_model
            },
            // An empty id must fall back to `PkLabels::default()`, NOT to
            // `ResolvedPk::resolve(snap, "")` — that one reports the id
            // verbatim, and an empty `provider_key_id` label would be a
            // third value alongside `unknown` and the real ids.
            pk: (!resolved.provider_key_id.is_empty())
                .then(|| crate::usage_attr::ResolvedPk::resolve(snap, &resolved.provider_key_id)),
        }
    }

    /// For the SLO latency histogram, which carries `provider` but no
    /// ProviderKey (per-key dimensions multiply every bucket).
    pub(crate) fn provider(&self) -> &str {
        &self.provider
    }

    /// The failure's labels, keeping the handler's own `model` / `stream`
    /// / `is_fallback` decisions — those are request-level facts the cell
    /// has no better answer for.
    pub(crate) fn upstream<'b>(
        &'b self,
        model: &'b str,
        stream: bool,
        is_fallback: bool,
    ) -> Upstream<'b> {
        Upstream {
            provider: &self.provider,
            model,
            upstream_model: self.upstream_model,
            pk: self.pk.as_ref().map(|p| p.labels()).unwrap_or_default(),
            stream,
            is_fallback,
        }
    }

    /// The model the caller addressed, bounded to the configured set —
    /// for the failure paths that could not recover it locally (the
    /// multipart audio routes parse it inside the dispatch that failed).
    pub(crate) fn requested_model<'b>(
        snap: &AisixSnapshot,
        resolved: &'b crate::attribution::Resolved,
    ) -> Cow<'b, str> {
        if resolved.requested_model.is_empty() {
            Cow::Borrowed(UNKNOWN)
        } else {
            crate::usage_attr::metric_model_label(snap, &resolved.requested_model)
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
/// - `/passthrough_route` — an opaque relay; even a `protocol`-aware route
///   resolves no configured Model to attribute.
/// - `/v1/files`, `/v1/batches`, `/v1/fine_tuning/jobs` — management calls.
///
/// `/v1/realtime` was in that list until the token families reached it too.
/// It was held out because it fed none of them, so counting it here would
/// have inflated the denominator of every tokens-per-request query; now that
/// a session reports its tokens and cost, that reason is gone and it belongs
/// with the rest.
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
    "/v1/realtime",
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

/// The one request-metric emit, shared by every handler.
///
/// Called on the handler's way out — see the module docs for why `elapsed`
/// is NOT the end-to-end figure on a streamed response.
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
    // Emit-chokepoint label bounding (#451 class): success paths hand in
    // the caller's requested string, which for a wildcard-served alias
    // is caller-minted — collapse it to the configured row's name here
    // so no handler-family member can mint unbounded series.
    let snap = state.snapshot.load();
    let (model_label, upstream_label) =
        crate::usage_attr::metric_model_label_pair(&snap, upstream.model, upstream.upstream_model);
    let upstream = Upstream {
        model: model_label.as_ref(),
        upstream_model: upstream_label.as_ref(),
        ..upstream
    };
    state
        .metrics
        .record_request(upstream.provider, upstream.model, status, outcome, elapsed);
    let labels = RequestLabels {
        endpoint,
        // Derived from the endpoint rather than passed in, so the detailed
        // families can't disagree with `aisix_proxy_in_flight_requests`
        // about which protocol a route speaks.
        inbound_protocol: crate::inbound_protocol_for_endpoint(endpoint),
        provider: upstream.provider,
        model: upstream.model,
        upstream_model: upstream.upstream_model,
        provider_key_id: upstream.pk.id(),
        provider_key_name: upstream.pk.name(),
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

/// What one request consumed. Every counter below no-ops on an all-zero
/// value, so the zero-token paths — a failed attempt, a 501, `/v1/files` —
/// cost nothing and create no series.
#[derive(Clone, Copy, Default)]
pub(crate) struct Tokens<'a> {
    pub input: u32,
    pub output: u32,
    /// The canonical, CACHE-INCLUSIVE total. Use
    /// `usage_attr::total_tokens_with_cache` wherever cache counters exist
    /// (#740/#1002) — a bare input+output silently undercounts cached
    /// traffic, and this value is what the by-client series reports as
    /// `token_type="total"`.
    pub total: u32,
    pub spend_usd: f64,
    /// Normalised inbound client for the by-client series
    /// (`state.client_classifier.classify(&client.user_agent)`).
    pub client_type: &'a str,
}

/// Terminal token/spend emit, shared by every handler — the companion to
/// [`record`].
///
/// Three families ride along, and each had a DIFFERENT endpoint coverage
/// before AISIX-Cloud#1234's follow-up: the `aisix_llm_*_tokens_total` and
/// `aisix_llm_spend_micro_usd_total` families were chat and messages only,
/// `aisix_llm_tokens_by_client_total` was chat, messages and responses, and
/// the legacy `aisix_tokens_consumed_total` was chat ALONE. A gateway that
/// billed a customer for `/v1/embeddings` reported none of those tokens.
///
/// Labels come from the same [`Caller`] / [`Upstream`] pair [`record`] uses,
/// so a query joining requests to tokens lines up by construction, and this
/// adds no series dimension the request families don't already carry.
pub(crate) fn record_usage(
    state: &ProxyState,
    endpoint: &'static str,
    caller: Caller<'_>,
    upstream: Upstream<'_>,
    tokens: Tokens<'_>,
) {
    // Same emit-chokepoint bounding as `record` — see the note there.
    let snap = state.snapshot.load();
    let (model_label, upstream_label) =
        crate::usage_attr::metric_model_label_pair(&snap, upstream.model, upstream.upstream_model);
    let upstream = Upstream {
        model: model_label.as_ref(),
        upstream_model: upstream_label.as_ref(),
        ..upstream
    };
    // Legacy compatibility series (provider × model).
    state
        .metrics
        .record_tokens(upstream.provider, upstream.model, u64::from(tokens.total));
    state.metrics.record_llm_usage(
        UsageLabels {
            endpoint,
            inbound_protocol: crate::inbound_protocol_for_endpoint(endpoint),
            provider: upstream.provider,
            model: upstream.model,
            upstream_model: upstream.upstream_model,
            provider_key_id: upstream.pk.id(),
            provider_key_name: upstream.pk.name(),
            api_key_id: caller.api_key_id,
            team_id: caller.team_id,
            user_id: caller.user_id,
            user_name: caller.user_name,
        },
        LlmUsage {
            input_tokens: tokens.input,
            output_tokens: tokens.output,
            total_tokens: tokens.total,
            spend_usd: tokens.spend_usd,
        },
    );
    // Deliberately NOT keyed on the labels above: this family is
    // client_type × model × token_type only, so the per-key dimensions
    // never multiply it (#890 req-4).
    state.metrics.record_llm_tokens_by_client(
        tokens.client_type,
        upstream.model,
        u64::from(tokens.input),
        u64::from(tokens.output),
        u64::from(tokens.total),
    );
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
            // Moved in once realtime started reporting its tokens + cost.
            "/v1/realtime",
        ] {
            assert!(
                is_llm_endpoint(crate::normalize_endpoint_label(route)),
                "{route} should count as an LLM request"
            );
        }
        for route in [
            "/mcp/some-server",
            "/a2a/some-agent",
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
