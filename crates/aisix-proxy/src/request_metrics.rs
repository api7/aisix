//! The one chokepoint for the per-request outcome metrics every handler
//! emits once dispatch has produced a response.
//!
//! # When a request is recorded, and what `elapsed` measures
//!
//! A request is recorded once, at its end, with the status it ended on and
//! the time from arrival to that end. For a non-streamed response that is
//! the handler's return. For a STREAMED response it is the end of the
//! stream: the handler parks a [`PendingRequestMetrics`] beside its
//! access-log line, and it is recorded with the request's terminal usage
//! event — `200` for a stream delivered to its end, `499` for one the caller
//! abandoned (including a body dropped before it was ever read), the
//! failure's status (`502`, `504`, an in-stream `4xx`) for one an upstream
//! error ended. So for streams the duration families measure the whole
//! stream, not time to response start, and a stream that fails after its
//! head counts as a failure.
//!
//! One open-ended body is still recorded at its head: the
//! `/v1/videos/:id/content` relay files no usage event (the submission
//! was metered), so there is no terminal emit to park on.
//!
//! A caller that leaves before the response head is written is NOT
//! recorded here — it never had a response. It is counted by
//! `aisix_proxy_client_cancelled_requests_total` alone.
//!
//! Time to first token is `aisix_request_ttft_seconds`; the SLO view of the same
//! whole-request duration is `aisix_request_e2e_latency_seconds`.
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

use aisix_core::{AisixSnapshot, ApiKey};
use aisix_obs::{LatencySide, LlmUsage, RequestLabels, RequestOutcome, UsageLabels};

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
    pub api_key_name: &'a str,
    pub team_id: &'a str,
    pub team_name: &'a str,
    pub user_id: &'a str,
    pub user_name: &'a str,
}

impl<'a> Caller<'a> {
    pub(crate) fn new(auth: &'a AuthenticatedKey) -> Self {
        let key = auth.key();
        Self {
            api_key_id: &auth.entry.id,
            api_key_name: auth.api_key_name().unwrap_or(UNKNOWN),
            team_id: key.team_id.as_deref().unwrap_or(UNKNOWN),
            team_name: auth.team_name().unwrap_or(UNKNOWN),
            user_id: key.user_id.as_deref().unwrap_or(UNKNOWN),
            user_name: auth.user_name().unwrap_or(UNKNOWN),
        }
    }

    /// Recover an owned caller from the request's dispatch snapshot.
    /// Capture this before starting a stream so key deletion or reassignment
    /// cannot change the caller attributed to the completed request.
    pub(crate) fn from_api_key_id(snap: &aisix_core::AisixSnapshot, api_key_id: &str) -> Owned {
        let entry = snap.apikeys.get_by_id(api_key_id);
        let key = entry.as_ref().map(|e| &e.value);
        let owner = key.map(|k| aisix_core::KeyOwnerNames::resolve(snap, k));
        let owned = |v: Option<&str>| v.map(str::to_owned);
        Owned {
            api_key_id: api_key_id.to_owned(),
            api_key_name: owned(key.and_then(ApiKey::telemetry_name)),
            team_id: key.and_then(|k| k.team_id.clone()),
            team_name: owned(owner.as_ref().and_then(|o| o.team_name())),
            user_id: key.and_then(|k| k.user_id.clone()),
            user_name: owned(key.zip(owner.as_ref()).and_then(|(k, o)| o.user_name(k))),
        }
    }

    /// A path that gave up before it could attribute the request to a team
    /// or user — the pre-dispatch rejections. `api_key_id` is `Some` once
    /// the auth extractor has run and `None` for the middleware
    /// short-circuits that precede it (see `reject`).
    pub(crate) fn unattributed(api_key_id: Option<&'a str>) -> Self {
        Self {
            api_key_id: api_key_id.unwrap_or(UNKNOWN),
            api_key_name: UNKNOWN,
            team_id: UNKNOWN,
            team_name: UNKNOWN,
            user_id: UNKNOWN,
            user_name: UNKNOWN,
        }
    }
}

/// Owning form of [`Caller`], for the snapshot lookup whose strings cannot
/// outlive the guard. Call [`Owned::as_caller`] at the emit.
#[derive(Clone)]
pub(crate) struct Owned {
    api_key_id: String,
    api_key_name: Option<String>,
    team_id: Option<String>,
    team_name: Option<String>,
    user_id: Option<String>,
    user_name: Option<String>,
}

impl Owned {
    /// The owning form of [`Caller::new`].
    pub(crate) fn from_auth(auth: &AuthenticatedKey) -> Self {
        Self::from_caller(Caller::new(auth))
    }

    pub(crate) fn from_caller(c: Caller<'_>) -> Self {
        let known = |v: &str| (v != UNKNOWN).then(|| v.to_owned());
        Self {
            api_key_id: c.api_key_id.to_owned(),
            api_key_name: known(c.api_key_name),
            team_id: known(c.team_id),
            team_name: known(c.team_name),
            user_id: known(c.user_id),
            user_name: known(c.user_name),
        }
    }

    pub(crate) fn as_caller(&self) -> Caller<'_> {
        Caller {
            api_key_id: &self.api_key_id,
            api_key_name: self.api_key_name.as_deref().unwrap_or(UNKNOWN),
            team_id: self.team_id.as_deref().unwrap_or(UNKNOWN),
            team_name: self.team_name.as_deref().unwrap_or(UNKNOWN),
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

/// Whether the caller addressed an ensemble model. See [`LastTarget::new`].
fn is_ensemble(snap: &AisixSnapshot, requested_model: &str) -> bool {
    !requested_model.is_empty()
        && crate::model_resolve::resolve_model(snap, requested_model)
            .is_some_and(|entry| entry.value.is_ensemble())
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
    upstream_model: Cow<'a, str>,
    pk: Option<crate::usage_attr::ResolvedPk<'a>>,
}

impl<'a> LastTarget<'a> {
    /// `resolved` has to outlive the emit — read it into a local first
    /// (`let resolved = attribution::current().unwrap_or_default();`).
    pub(crate) fn new(snap: &AisixSnapshot, resolved: &'a crate::attribution::Resolved) -> Self {
        // An ensemble has no single terminal target: its panel members run
        // concurrently and all of them are attempted, so the cell just holds
        // whichever resolved last. Naming that one would read as "this key
        // is what failed" — a plausible-looking wrong answer, which is worse
        // than the placeholder. Suppressed rather than deferred to the
        // ensemble design pass, because it is this change that would
        // otherwise introduce it.
        if is_ensemble(snap, &resolved.requested_model) {
            return Self {
                provider: UNKNOWN.to_string(),
                upstream_model: Cow::Borrowed(UNKNOWN),
                pk: None,
            };
        }
        Self {
            // Lowercased here because the success path lowercases at its
            // own emit; a failure that spelled the vendor differently
            // would land on a second series for the same provider.
            provider: if resolved.provider.is_empty() {
                UNKNOWN.to_string()
            } else {
                resolved.provider.to_ascii_lowercase()
            },
            // A wildcard row resolves to a SUBSTITUTED upstream id — the
            // caller's own suffix — so it goes through the same collapse the
            // success path's emit applies, or a failed request could mint one
            // series per made-up suffix (#451).
            upstream_model: if resolved.upstream_model.is_empty() {
                Cow::Borrowed(UNKNOWN)
            } else {
                crate::usage_attr::metric_model_label_pair(
                    snap,
                    &resolved.requested_model,
                    &resolved.upstream_model,
                )
                .1
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
            upstream_model: &self.upstream_model,
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
    "/v1/images/edits",
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
/// Called on the handler's way out, which is the request's end for every
/// response except a stream — a streamed response defers to
/// [`PendingRequestMetrics`] instead (see the module docs).
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
    // Emit-chokepoint label bounding (#451 class): success paths hand in
    // the caller's requested string, which for a wildcard-served alias
    // is caller-minted — collapse it to the configured row's name here
    // so no handler-family member can mint unbounded series.
    let snap = state.snapshot.load();
    let (model, upstream_model) =
        crate::usage_attr::metric_model_label_pair(&snap, upstream.model, upstream.upstream_model);
    let upstream = Upstream {
        model: model.as_ref(),
        upstream_model: upstream_model.as_ref(),
        ..upstream
    };
    emit(&state.metrics, endpoint, caller, upstream, status, elapsed);
}

/// Write one request's samples. `upstream` must already be bounded.
fn emit(
    metrics: &aisix_obs::Metrics,
    endpoint: &'static str,
    caller: Caller<'_>,
    upstream: Upstream<'_>,
    status: u16,
    elapsed: Duration,
) {
    let outcome = RequestOutcome::from_status(status);
    metrics.record_request(upstream.provider, upstream.model, status, outcome, elapsed);
    let labels = RequestLabels {
        endpoint,
        // Derived from the endpoint rather than passed in, so the detailed
        // families can't disagree with `aisix_proxy_in_flight_requests`
        // about which protocol a route speaks.
        inbound_protocol: crate::inbound_protocol_for_endpoint(endpoint),
        // Read off the SAME ProviderKey row that produced the id and name
        // beside it (`usage_attr::ResolvedPk`), so a request can never be
        // labelled with one key's identity and another's protocol.
        upstream_protocol: upstream.pk.protocol(),
        provider: upstream.provider,
        model: upstream.model,
        upstream_model: upstream.upstream_model,
        provider_key_id: upstream.pk.id(),
        provider_key_name: upstream.pk.name(),
        api_key_id: caller.api_key_id,
        api_key_name: caller.api_key_name,
        team_id: caller.team_id,
        team_name: caller.team_name,
        user_id: caller.user_id,
        user_name: caller.user_name,
        stream: upstream.stream,
        is_fallback: upstream.is_fallback,
        status,
        outcome,
    };
    if is_llm_endpoint(endpoint) {
        metrics.record_proxy_and_llm_request(labels, elapsed);
    } else {
        metrics.record_proxy_request(labels, elapsed);
    }
}

/// The request families of a STREAMED response, held until the stream ends.
///
/// A streaming handler returns when the response head goes out, so the
/// status it holds then is the `200` of a stream that has not delivered
/// anything yet, and its clock stops at time-to-response-start. Recording
/// there counted a stream the caller abandoned, or the upstream broke
/// mid-way, as a success, and kept it out of
/// `aisix_proxy_failed_requests_total`.
///
/// So the handler parks this beside its access-log line
/// (`attribution::defer_access_log` takes both), and it is recorded with
/// the status of the request's TERMINAL usage event, when that event goes
/// out — the same status the line and the usage row carry, whichever of the
/// stream's endings produced it. Taking it out of the cell is what makes it
/// exactly once.
///
/// Labels are bounded when it is built, against the snapshot the request
/// dispatched on: the model row may be gone by the time the stream ends.
pub(crate) struct PendingRequestMetrics {
    metrics: std::sync::Arc<aisix_obs::Metrics>,
    endpoint: &'static str,
    upstream_protocol: &'static str,
    provider: String,
    model: String,
    upstream_model: String,
    provider_key_id: String,
    provider_key_name: String,
    caller: Owned,
    stream: bool,
    is_fallback: bool,
    /// The request clock: a stream's duration is the whole stream.
    started: std::time::Instant,
}

impl PendingRequestMetrics {
    /// A streamed request's samples, to be recorded when its terminal usage
    /// event goes out. `started` is the REQUEST clock.
    pub(crate) fn at_stream_end(
        state: &ProxyState,
        endpoint: &'static str,
        caller: Caller<'_>,
        upstream: Upstream<'_>,
        started: std::time::Instant,
    ) -> Self {
        let snap = state.snapshot.load();
        let (model, upstream_model) = crate::usage_attr::metric_model_label_pair(
            &snap,
            upstream.model,
            upstream.upstream_model,
        );
        Self {
            metrics: state.metrics.clone(),
            endpoint,
            upstream_protocol: upstream.pk.protocol(),
            provider: upstream.provider.to_owned(),
            model: model.into_owned(),
            upstream_model: upstream_model.into_owned(),
            provider_key_id: upstream.pk.id().to_owned(),
            provider_key_name: upstream.pk.name().to_owned(),
            caller: Owned::from_caller(caller),
            stream: upstream.stream,
            is_fallback: upstream.is_fallback,
            started,
        }
    }

    /// Record the parked samples under the request's terminal status.
    pub(crate) fn record_terminal(self, status: u16) {
        let upstream = Upstream {
            provider: &self.provider,
            model: &self.model,
            upstream_model: &self.upstream_model,
            pk: PkLabels::from_parts(
                &self.provider_key_id,
                &self.provider_key_name,
                self.upstream_protocol,
            ),
            stream: self.stream,
            is_fallback: self.is_fallback,
        };
        emit(
            &self.metrics,
            self.endpoint,
            self.caller.as_caller(),
            upstream,
            status,
            self.started.elapsed(),
        );
    }
}

/// Stream callbacks must capture bounded model labels from their dispatch
/// snapshot: the model row may be gone by the time the stream ends.
///
/// Records the `downstream` side and arranges the `upstream` one — see
/// [`record_e2e_downstream`]. `/a2a` calls [`record_e2e_latency_downstream_only`].
pub(crate) fn record_e2e_latency(
    state: &ProxyState,
    endpoint: &'static str,
    caller: Caller<'_>,
    upstream: Upstream<'_>,
    status: u16,
    elapsed: Duration,
) {
    record_e2e_latency_as(state, endpoint, caller, upstream, status, elapsed, true);
}

/// [`record_e2e_latency`] for a route with no model dispatch to time on its
/// own: `/a2a` observes the `downstream` side alone.
pub(crate) fn record_e2e_latency_downstream_only(
    state: &ProxyState,
    endpoint: &'static str,
    caller: Caller<'_>,
    upstream: Upstream<'_>,
    status: u16,
    elapsed: Duration,
) {
    record_e2e_latency_as(state, endpoint, caller, upstream, status, elapsed, false);
}

fn record_e2e_latency_as(
    state: &ProxyState,
    endpoint: &'static str,
    caller: Caller<'_>,
    upstream: Upstream<'_>,
    status: u16,
    elapsed: Duration,
    with_upstream: bool,
) {
    let snap = state.snapshot.load();
    let (model, upstream_model) =
        crate::usage_attr::metric_model_label_pair(&snap, upstream.model, upstream.upstream_model);
    let labels = aisix_obs::LatencyLabels {
        endpoint,
        model: model.as_ref(),
        provider: upstream.provider,
        status,
        streaming: upstream.stream,
        details: UsageLabels {
            endpoint,
            inbound_protocol: crate::inbound_protocol_for_endpoint(endpoint),
            upstream_protocol: upstream.pk.protocol(),
            provider: upstream.provider,
            model: model.as_ref(),
            upstream_model: upstream_model.as_ref(),
            provider_key_id: upstream.pk.id(),
            provider_key_name: upstream.pk.name(),
            api_key_id: caller.api_key_id,
            api_key_name: caller.api_key_name,
            team_id: caller.team_id,
            team_name: caller.team_name,
            user_id: caller.user_id,
            user_name: caller.user_name,
        },
    };
    // An ensemble's response comes from no single attempt — its terminal
    // event is the judge's own call, only the last step — so it has no
    // upstream side to observe, on any path.
    let ensemble =
        crate::attribution::current().is_some_and(|r| is_ensemble(&snap, &r.requested_model));
    if with_upstream && !ensemble {
        record_e2e_downstream(&state.metrics, labels, elapsed);
    } else {
        state
            .metrics
            .record_request_e2e_latency(labels, LatencySide::Downstream, elapsed);
    }
}

/// Observe a request's `downstream` end-to-end latency, and note its labels
/// on the request's cell so that its `upstream` sibling — the terminal
/// attempt's own duration, which the terminal usage event carries — is
/// recorded under the same labels. Recorded only when the request's
/// terminal attempt was dispatched to an upstream; see
/// `attribution::note_e2e_downstream`.
pub(crate) fn record_e2e_downstream(
    metrics: &std::sync::Arc<aisix_obs::Metrics>,
    labels: aisix_obs::LatencyLabels<'_>,
    elapsed: Duration,
) {
    metrics.record_request_e2e_latency(labels, LatencySide::Downstream, elapsed);
    crate::attribution::note_e2e_downstream(
        metrics.clone(),
        OwnedLatencyLabels::from(labels),
        elapsed,
    );
}

/// A finished stream's request and attempt durations, both read at one
/// instant. Read apart, the attempt's clock — taken after the request's
/// `downstream` observation — would count the gateway's own bookkeeping in
/// between, and the `upstream` side could come out longer than the
/// `downstream` side that contains it.
pub(crate) fn stream_end_elapsed(
    started: std::time::Instant,
    attempt_started: std::time::Instant,
) -> (Duration, Duration) {
    let now = std::time::Instant::now();
    (
        now.saturating_duration_since(started),
        now.saturating_duration_since(attempt_started),
    )
}

/// An owned [`aisix_obs::LatencyLabels`], held on the request's cell
/// between the two sides of its end-to-end observation.
pub(crate) struct OwnedLatencyLabels {
    endpoint: String,
    model: String,
    provider: String,
    status: u16,
    streaming: bool,
    details: [String; 14],
}

impl From<aisix_obs::LatencyLabels<'_>> for OwnedLatencyLabels {
    fn from(l: aisix_obs::LatencyLabels<'_>) -> Self {
        let d = l.details;
        Self {
            endpoint: l.endpoint.to_owned(),
            model: l.model.to_owned(),
            provider: l.provider.to_owned(),
            status: l.status,
            streaming: l.streaming,
            details: [
                d.endpoint,
                d.inbound_protocol,
                d.upstream_protocol,
                d.provider,
                d.model,
                d.upstream_model,
                d.provider_key_id,
                d.provider_key_name,
                d.api_key_id,
                d.api_key_name,
                d.team_id,
                d.team_name,
                d.user_id,
                d.user_name,
            ]
            .map(str::to_owned),
        }
    }
}

impl OwnedLatencyLabels {
    pub(crate) fn as_labels(&self) -> aisix_obs::LatencyLabels<'_> {
        let d = &self.details;
        aisix_obs::LatencyLabels {
            endpoint: &self.endpoint,
            model: &self.model,
            provider: &self.provider,
            status: self.status,
            streaming: self.streaming,
            details: UsageLabels {
                endpoint: &d[0],
                inbound_protocol: &d[1],
                upstream_protocol: &d[2],
                provider: &d[3],
                model: &d[4],
                upstream_model: &d[5],
                provider_key_id: &d[6],
                provider_key_name: &d[7],
                api_key_id: &d[8],
                api_key_name: &d[9],
                team_id: &d[10],
                team_name: &d[11],
                user_id: &d[12],
                user_name: &d[13],
            },
        }
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
    /// Upstream prompt-cache detail (AISIX-Cloud#1404), each field
    /// forwarded exactly as the upstream reported it and never folded
    /// into `input` / `total` — the two accounting shapes are
    /// incompatible, so the split is what makes a cross-protocol cache
    /// ratio computable at all. `cached` is `UsageStats::
    /// cached_prompt_tokens` (already inside `input`); `cache_read` and
    /// `cache_creation` are the Anthropic-shape counters that sit
    /// outside `input` and inside `total`. All three are 0 on the
    /// endpoints and providers that report no cache detail, which emits
    /// no series.
    pub cached: u32,
    pub cache_read: u32,
    pub cache_creation: u32,
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
            upstream_protocol: upstream.pk.protocol(),
            provider: upstream.provider,
            model: upstream.model,
            upstream_model: upstream.upstream_model,
            provider_key_id: upstream.pk.id(),
            provider_key_name: upstream.pk.name(),
            api_key_id: caller.api_key_id,
            api_key_name: caller.api_key_name,
            team_id: caller.team_id,
            team_name: caller.team_name,
            user_id: caller.user_id,
            user_name: caller.user_name,
        },
        LlmUsage {
            input_tokens: tokens.input,
            output_tokens: tokens.output,
            total_tokens: tokens.total,
            cached_input_tokens: tokens.cached,
            cache_read_input_tokens: tokens.cache_read,
            cache_creation_input_tokens: tokens.cache_creation,
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

    use aisix_core::{Model, ResourceEntry};

    fn snapshot_with(display_name: &str, body: serde_json::Value) -> AisixSnapshot {
        let model: Model = serde_json::from_value(body).unwrap();
        let snap = AisixSnapshot::new();
        snap.models
            .insert(ResourceEntry::new(display_name, model, 1));
        snap
    }

    fn resolved(requested: &str) -> crate::attribution::Resolved {
        crate::attribution::Resolved {
            requested_model: requested.to_string(),
            provider: "OpenAI".to_string(),
            upstream_model: "gpt-4o-mini".to_string(),
            provider_key_id: "pk-1".to_string(),
            cache_hit_layer: None,
        }
    }

    /// The vendor id reaches the label lowercased, because the success path
    /// lowercases at its own emit — a failure spelling it differently would
    /// put one provider on two series, which is the bug this recovers from.
    #[test]
    fn a_recovered_target_matches_the_success_paths_spelling() {
        let snap = snapshot_with(
            "direct",
            serde_json::json!({
                "display_name": "direct",
                "provider": "openai",
                "model_name": "gpt-4o-mini",
                "provider_key_id": "pk-1",
            }),
        );
        let r = resolved("direct");
        let target = LastTarget::new(&snap, &r);
        let upstream = target.upstream("direct", false, false);
        assert_eq!(target.provider(), "openai");
        assert_eq!(upstream.provider, "openai");
        assert_eq!(upstream.upstream_model, "gpt-4o-mini");
        assert_eq!(upstream.pk.id(), "pk-1");
    }

    /// An ensemble's panel members all run, so the cell holds whichever
    /// resolved last. Naming it would read as "this key is what failed" —
    /// a plausible-looking wrong answer. The placeholder is the honest one.
    #[test]
    fn an_ensemble_failure_is_not_attributed_to_one_of_its_members() {
        let snap = snapshot_with(
            "panel",
            serde_json::json!({
                "display_name": "panel",
                "ensemble": {
                    "panel": [{"model": "gpt4"}],
                    "judge": {"model": "gpt4"},
                },
            }),
        );
        let r = resolved("panel");
        let target = LastTarget::new(&snap, &r);
        let upstream = target.upstream("panel", false, false);
        assert_eq!(upstream.provider, UNKNOWN);
        assert_eq!(upstream.upstream_model, UNKNOWN);
        assert_eq!(upstream.pk.id(), UNKNOWN);
        assert_eq!(upstream.pk.name(), UNKNOWN);
    }

    /// A request that never selected a target keeps the placeholder, and the
    /// ProviderKey id must be `unknown` rather than the empty string an
    /// unresolved `ResolvedPk` reports verbatim — an empty label value would
    /// be a second "nothing resolved" series alongside it.
    #[test]
    fn nothing_resolved_collapses_to_one_placeholder() {
        let snap = AisixSnapshot::new();
        let nothing = crate::attribution::Resolved::default();
        let target = LastTarget::new(&snap, &nothing);
        let upstream = target.upstream(crate::usage_attr::UNRESOLVED_MODEL_LABEL, false, false);
        assert_eq!(upstream.provider, UNKNOWN);
        assert_eq!(upstream.upstream_model, UNKNOWN);
        assert_eq!(upstream.pk.id(), UNKNOWN);
        assert_eq!(upstream.pk.name(), UNKNOWN);
    }

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
        "/v1/images/edits",
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
            "/v1/images/edits",
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
