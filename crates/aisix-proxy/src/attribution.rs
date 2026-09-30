//! What the request has committed to upstream, for the terminal emitters
//! that cannot see it.
//!
//! Two of them run without the values the rest of the request already
//! resolved:
//!
//! - the client-cancel guard in `record_request_telemetry` is armed in
//!   middleware — before auth, body parsing and model resolution — and
//!   fires from `Drop` once the handler future has been dropped, so it
//!   could only ever label `endpoint` (AISIX-Cloud#1317);
//! - every handler's failure branch holds a `ProxyError`, which carries
//!   no upstream identity. A request that reached a real provider and
//!   came back 502 therefore reported `provider="unknown"` on the same
//!   series its successes report the real provider, which puts successes
//!   and failures of one ProviderKey in different time series and makes
//!   a per-provider failure rate meaningless (AISIX-Cloud#1325).
//!
//! Both read the cell the telemetry middleware installs for the request.
//! Writes ride the two resolution chokepoints every endpoint already goes
//! through — [`crate::model_resolve::resolve_model`] for the model the
//! caller addressed and [`crate::dispatch::resolve_provider_key`] for the
//! target about to be dispatched to — so a new endpoint is attributed
//! without opting in, and a retry / fallback loop leaves the LAST target
//! it selected behind, which is the attempt the client's error came from.
//!
//! It is a task-local rather than a threaded parameter because the cancel
//! guard reads it from outside the handler entirely. A write with no
//! scope installed (background health checks, unit tests) is dropped.
//!
//! [`CancelContext`] is the same idea carried one step further. The cancel
//! guard does not only LABEL a cancelled request any more — it emits the
//! request's usage events, because the handler code that would have done so
//! is continuation code that a dropped future never runs (AISIX-Cloud#1571).
//! That needs more than labels: the caller's identity, the attempt that was
//! in flight, and the attempts that already failed. All of it is written at
//! chokepoints the handlers already pass through — the [`ClientContext`]
//! extractor, [`crate::model_resolve::resolve_model`], and
//! [`crate::attempt::RoutingTelemetry`] — so no endpoint has to opt in and
//! none of them can drift out.
//!
//! The cell carries one more thing for the same reason: a streamed
//! response's access-log LINE ([`PendingAccessLog`]). Its handler returns
//! when the head goes out, minutes before the request ends, so the line is
//! parked here and completed by whichever terminal emitter ends the request.
//!
//! And every request's line, streamed or not, is WRITTEN from here: the
//! telemetry middleware counts the request and response body bytes where
//! they cross its outermost layer, and those counts are only final once
//! the response body is finished with. So [`emit_access_log`] holds the
//! completed line on the cell, and the middleware's guard writes it with
//! both sizes when it finishes the request ([`RequestAttribution::finish`]).
//!
//! It is kept BESIDE [`Resolved`] rather than inside it because every
//! failed request reads `Resolved` back by value for its metric labels
//! ([`current`]); folding a `Vec<AttemptRecord>` and a `ClientContext` into
//! that clone would put the cancel path's cost on every error path.

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aisix_core::Model;

use crate::attempt::AttemptRecord;
use crate::client_ip::ClientContext;

/// What the request resolved, as it resolved it. An empty field means
/// "never got there", which readers render as the `unknown` label value
/// the request families already use for the same condition.
#[derive(Clone, Default)]
pub(crate) struct Resolved {
    /// The model name the CALLER addressed — raw, so it must be bounded
    /// through `usage_attr::metric_model_label` before it becomes a
    /// label (#451). Kept raw here because the bounding needs a snapshot
    /// and only the readers have one.
    pub requested_model: String,
    /// Vendor id of the last target the request selected.
    pub provider: String,
    /// That target's upstream model id.
    pub upstream_model: String,
    /// That target's ProviderKey id. The readable name is resolved from
    /// it at read time, so the pair is byte-identical to the one the
    /// success path emits.
    pub provider_key_id: String,
    /// Which cache layer answered this request, once one has — `Some`
    /// exactly when the response came out of the cache.
    ///
    /// It sits beside the target fields because it is the same fact from
    /// the other side: nothing was dispatched to BECAUSE the cache
    /// answered. Both exits from a hit need it and only one of them can
    /// see the cache — the output guardrail can still refuse a stored
    /// body, and that request leaves through the handler's error branch,
    /// which holds a `ProxyError` and no cache verdict at all
    /// (AISIX-Cloud#1571).
    pub cache_hit_layer: Option<&'static str>,
}

/// One upstream attempt that has begun and not yet settled.
///
/// The winning-or-failing outcome is what turns this into an
/// [`AttemptRecord`]; until then it is all a cancelled request can say about
/// the target it was waiting on.
#[derive(Clone)]
pub(crate) struct InFlightAttempt {
    /// 0-based attempt index within the request.
    pub index: u32,
    /// `"initial"` / `"retry"` / `"fallback"` — see
    /// [`crate::attempt::RoutingTelemetry::begin_attempt`].
    pub kind: &'static str,
    /// Routing target display name; empty for a direct model, exactly as
    /// [`AttemptRecord::target_model`] spells it.
    pub target_model: String,
    /// UUID of the concrete Model row this attempt dispatched to.
    pub model_id: String,
}

/// The attempt that WON, once it has settled.
///
/// Its own event is written by the handler after the response is processed
/// — so on the cancel path it was never written at all, and unlike a failed
/// attempt there is no row of this request already carrying its index. That
/// is what lets the terminal event name it in full.
#[derive(Clone)]
pub(crate) struct SettledWinner {
    pub index: u32,
    pub kind: &'static str,
    pub target_model: String,
    pub model_id: String,
    /// The attempt's own measured duration, as recorded. Taken from the
    /// record rather than measured here, so it does not absorb the
    /// post-dispatch work (the output scan, the cache write) the cancel
    /// actually landed in.
    pub latency_ms: u32,
}

/// What a route that never names a model reports in place of one.
///
/// `/mcp`, `/a2a/:agent` and the passthrough namespace tunnel to an upstream
/// the caller addressed by route rather than by model, and their own usage
/// events are attributed by those names instead. A cancelled request on one
/// of them has to carry the same ones, or its row is the only one of that
/// family an operator cannot tell apart from any other (AISIX-Cloud#1571).
/// Every field is empty on the model-named families.
#[derive(Clone, Default)]
pub(crate) struct RouteIdentity {
    pub passthrough_route: String,
    pub mcp_server: String,
    /// The tool the JSON-RPC body named, once it has been parsed — a cancel
    /// landing before that leaves it empty rather than guessing.
    pub mcp_tool: String,
    pub a2a_agent: String,
    /// The caller's raw JSON-RPC method, unbounded by nature.
    pub a2a_method: String,
    /// The canonical operation that method names, from a fixed set — what a
    /// per-operation figure groups by, and what the completed row carries.
    pub a2a_operation: String,
}

/// What a cancelled request needs to emit its own usage events. See the
/// module docs for why it does not live in [`Resolved`].
#[derive(Default)]
pub(crate) struct CancelContext {
    /// The caller, as the [`ClientContext`] extractor resolved it. `None`
    /// on the routes that resolve their principal themselves — `/mcp` and
    /// `/a2a` never run that extractor — where [`auth`](Self::auth) carries
    /// the identity instead and the two IP-derived fields stay empty,
    /// exactly as those families' own events leave them.
    pub client: Option<ClientContext>,
    /// The authenticated principal, from the one extractor every
    /// authenticated route passes through. Carries the caller identity and
    /// the anonymous flag for the families that build no [`ClientContext`].
    pub auth: Option<crate::auth::AuthenticatedKey>,
    /// See [`RouteIdentity`].
    pub route: RouteIdentity,
    /// The surface the request RESOLVED, when the path cannot say.
    ///
    /// The cancel guard otherwise reads the surface off the normalized
    /// endpoint label, which is right for every typed route and wrong for a
    /// passthrough route that is not mounted under `/passthrough/`: a
    /// custom `path_prefix` normalizes to `other`, and a HOST-matched route
    /// keeps whatever path the caller sent — `/v1/chat/completions` on a
    /// forward-proxied upstream normalizes to the CHAT surface. The row
    /// would then be filed as an abandoned chat call with no model.
    pub surface: Option<crate::operation::Surface>,
    /// Set by a route that files no usage row at ANY outcome but shares a
    /// normalized endpoint label with a metering sibling — the A2A agent
    /// card, which normalizes to `/a2a` like the calls do. The guard has
    /// only the label, so the route has to say so itself or a cancelled
    /// discovery fetch is filed as an agent call.
    pub unmetered: bool,
    /// The authenticated `api_key` row's id. Empty on an unauthenticated
    /// path, which the cancel emitter treats as "nothing to attribute".
    pub api_key_id: String,
    /// The Model uuid of the entry the caller addressed — but only when
    /// that entry is itself dispatchable. A routing / ensemble / semantic
    /// entry leaves this empty: its id prices nothing, and writing a group
    /// into `model_id` is the AISIX-Cloud#790 bug in reverse.
    pub entry_model_id: String,
    /// The attempt begun and not yet settled, if the cancel landed inside
    /// one.
    pub in_flight: Option<InFlightAttempt>,
    /// The winning attempt, once it settled. See [`SettledWinner`].
    pub won: Option<SettledWinner>,
    /// The Model uuid of the last target that SETTLED, win or lose.
    ///
    /// Covers the gap `won` does not: a cancel during the retry backoff
    /// between two FAILED attempts. A target had been selected — the
    /// access-log line names it — so the event must too, or a routing
    /// request reports a row naming no target at all, which is the thing
    /// this whole path exists to remove. Only the identity is carried:
    /// repeating that attempt's index would put two rows of one request
    /// under the same `attempt_index`, since the guard emits its event.
    pub last_settled_model_id: String,
    /// Attempts that settled as failures. On the cancel path these never
    /// reach the handler's own `emit_failed_attempts`, so the guard emits
    /// them. A SUCCESSFUL attempt is not kept: its event is built from the
    /// response the handler was still processing, of which the attempt
    /// record holds nothing, so there would be nothing to bill it with —
    /// and skipping it also keeps the happy path allocation-free here.
    pub failed_attempts: Vec<AttemptRecord>,
    /// Whether a usage event has already been emitted for this request, and
    /// whether one of them was the terminal one.
    ///
    /// Two windows write it. On the head phase it is microseconds wide —
    /// between an emission and the middleware advancing the guard past it.
    /// On the body phase it is the whole interlock: the body is dropped
    /// inside this cell's scope ([`sync_scope`]), so a family whose stream
    /// emitter fires there says so here. Either way a second, contradicting
    /// set of rows for one request is worse than the row the cancel path
    /// would have added, so the guard defers to whatever was written.
    pub emitted_any: bool,
    pub emitted_terminal: bool,
}

/// A streamed response's access-log line, parked by its handler until the
/// request has an outcome to report.
///
/// A streaming handler returns the moment the response HEAD exists — before
/// a single frame has been polled, and often minutes before the request
/// ends. Writing the line there is what made a stream the caller abandoned
/// log `200` beside its own `499` usage event, and left every streamed line
/// without the token counts and `provider_request_id` that arrive with the
/// response (AISIX-Cloud#1571).
///
/// So the handler parks here the fields only it can resolve, and the line
/// is completed beside the request's TERMINAL usage event — from
/// [`crate::usage_attr::emit_usage`], the one point every terminal event of
/// every family passes through — then written with the body sizes once the
/// response body is done with, like every other line ([`emit_access_log`]). That is what makes the two agree on
/// `status`, `error_kind` and `error` by construction rather than by each
/// family remembering to.
///
/// Taking it out of the cell is also the interlock: whichever completion
/// point gets there first emits, and the rest find nothing. A request can
/// therefore never write two lines, however many of its terminal emitters
/// race (see `crate::GuardPhase`).
pub(crate) struct PendingAccessLog {
    method: String,
    /// Bounded route template, never the caller's raw path (#451).
    path: String,
    provider: String,
    /// The model name the CALLER addressed. The dispatched target is read
    /// off [`Resolved`] at emit time instead, exactly as the handler's own
    /// line reads it through [`AccessLogTarget`].
    model: String,
    api_key_id: String,
    request_id: String,
    served_by_model: String,
    routing_attempt_count: Option<u32>,
    routing_fallback_count: Option<u32>,
    /// The request clock, so the line can report how long the whole request
    /// occupied the gateway — which on a stream is not what the caller
    /// waited for (see `AccessLog::duration`).
    started: Instant,
}

impl PendingAccessLog {
    /// What every family can say: who called, where, and when the request
    /// started. `started` is the REQUEST clock, not the attempt's.
    pub(crate) fn new(
        method: &str,
        path: &str,
        request_id: &str,
        api_key_id: &str,
        started: Instant,
    ) -> Self {
        Self {
            method: method.to_string(),
            path: path.to_string(),
            provider: String::new(),
            model: String::new(),
            api_key_id: api_key_id.to_string(),
            request_id: request_id.to_string(),
            served_by_model: String::new(),
            routing_attempt_count: None,
            routing_fallback_count: None,
            started,
        }
    }

    /// The vendor and the model name the CALLER addressed.
    pub(crate) fn with_model(mut self, provider: &str, model: &str) -> Self {
        self.provider = provider.to_string();
        self.model = model.to_string();
        self
    }

    /// The routing summary, for the families that dispatch through a group.
    /// Same shape their own inline line carries: the winner's display name,
    /// and the two counts, each absent when zero.
    pub(crate) fn with_routing(mut self, routing: &crate::attempt::RoutingTelemetry) -> Self {
        let summary = routing.access_log_summary();
        self.served_by_model = summary.served_by_model.unwrap_or_default().to_string();
        self.routing_attempt_count = summary.attempt_count;
        self.routing_fallback_count = summary.fallback_count;
        self
    }

    /// Complete the line, taking its outcome from the terminal usage event,
    /// and hand it to `cell` to be written once the body sizes are known.
    fn emit(self, cell: &RequestAttribution, target: &Resolved, event: &aisix_obs::UsageEvent) {
        let duration = self.started.elapsed();
        // What the CALLER waited for, taken VERBATIM off the event: the
        // first token forwarded downstream on a stream that delivered one.
        // Not re-derived, and no sentinel handling — the line and the row
        // are one record of one request, so they must not be able to report
        // two different waits, and a sub-millisecond first frame is a real
        // `0` rather than a missing stamp. It reads `0` exactly where the
        // event says the caller received nothing, which the `499` beside it
        // explains. Deliberately NOT the length of the stream, which
        // `duration` is (AISIX-Cloud#1394).
        let latency = Duration::from_millis(u64::from(event.downstream_latency_ms));
        let prompt = u64::from(event.prompt_tokens);
        // The gateway's own completion count, as the non-streamed line and
        // Prometheus report it — not the record's raw one, which may keep
        // an upstream's separately-counted reasoning apart.
        let completion_tokens = event.folded_completion_tokens();
        let completion = u64::from(completion_tokens);
        // Cache-inclusive, the way every emitter in this crate computes a
        // request's total: `prompt_tokens` excludes the cache dimensions on
        // the Anthropic-shaped paths, so summing the two visible columns
        // would put a cached request's line an order of magnitude under the
        // row it was emitted beside.
        let total = crate::usage_attr::total_tokens_with_cache(
            event.prompt_tokens,
            completion_tokens,
            event.cache_creation_tokens,
            event.cache_read_tokens,
        );
        // Keep a token-less outcome out of the token columns entirely,
        // rather than logging an abandoned stream as a zero-token success —
        // the same rule the rest of this line follows for `error_kind` and
        // `provider_request_id`.
        let counted = total > 0;
        let line = aisix_obs::AccessLog {
            method: &self.method,
            path: &self.path,
            status: event.status_code,
            latency,
            duration,
            provider: (!self.provider.is_empty()).then_some(self.provider.as_str()),
            model: (!self.model.is_empty()).then_some(self.model.as_str()),
            upstream_model: (!target.upstream_model.is_empty())
                .then_some(target.upstream_model.as_str()),
            provider_key_id: (!target.provider_key_id.is_empty())
                .then_some(target.provider_key_id.as_str()),
            api_key_id: (!self.api_key_id.is_empty()).then_some(self.api_key_id.as_str()),
            prompt_tokens: counted.then_some(prompt),
            completion_tokens: counted.then_some(completion),
            total_tokens: counted.then_some(total),
            request_id: &self.request_id,
            provider_request_id: (!event.provider_request_id.is_empty())
                .then_some(event.provider_request_id.as_str()),
            served_by_model: (!self.served_by_model.is_empty())
                .then_some(self.served_by_model.as_str()),
            routing_attempt_count: self.routing_attempt_count,
            routing_fallback_count: self.routing_fallback_count,
            error_kind: (!event.error_class.is_empty()).then_some(event.error_class.as_str()),
            error: (!event.error_message.is_empty()).then_some(event.error_message.as_str()),
            mcp: None,
            // Off the EVENT, not the cell: this emitter writes the line
            // beside the row cp-api stores, and the two must not be able to
            // disagree about the same request. Empty means the surface had
            // no cache decision (`/v1/realtime`, a cancel before the gate).
            cache: (!event.cache_status.is_empty()).then(|| aisix_obs::CacheAccessLog {
                status: event.cache_status.as_str(),
                hit_layer: (!event.cache_hit_layer.is_empty())
                    .then_some(event.cache_hit_layer.as_str()),
            }),
            request_body_bytes: None,
            response_body_bytes: None,
        };
        cell.park_access_log(&line, None);
    }
}

#[derive(Default)]
struct Cell {
    resolved: Resolved,
    cancel: CancelContext,
    /// See [`PendingAccessLog`]. `Some` only between a streaming handler
    /// returning and the request's terminal usage event going out.
    pending_log: Option<PendingAccessLog>,
    /// See [`note_stream_owns_access_log`].
    stream_owns_log: bool,
    /// See [`RequestAttribution::track`]. `None` on a cell no telemetry
    /// middleware installed — a throwaway [`detached`] one, a unit test —
    /// where a line is written at once because nothing would ever flush it.
    request_span: Option<tracing::Span>,
    /// The response head has been handed to the server.
    head_written: bool,
    /// [`RequestAttribution::finish`] has run: the body sizes are final and
    /// nothing will flush a line parked after this, so it is written at
    /// once.
    finished: bool,
    /// The request's completed access-log line, waiting for the body sizes.
    ready_line: Option<ReadyLine>,
    /// See [`note_e2e_downstream`].
    e2e_labels: Option<(
        Arc<aisix_obs::Metrics>,
        crate::request_metrics::OwnedLatencyLabels,
        Duration,
    )>,
    /// See [`note_terminal_upstream`].
    terminal_upstream: Option<Duration>,
    /// See [`note_semantic_decision`].
    semantic: Option<aisix_obs::SemanticAccessLog>,
    /// The `upstream` end-to-end observation has been recorded.
    e2e_upstream_recorded: bool,
}

impl Cell {
    /// The `upstream` end-to-end observation, once both of its halves are
    /// known and it has not been recorded yet.
    fn take_e2e_upstream(
        &mut self,
    ) -> Option<(
        Arc<aisix_obs::Metrics>,
        crate::request_metrics::OwnedLatencyLabels,
        Duration,
    )> {
        if self.e2e_upstream_recorded {
            return None;
        }
        let upstream = self.terminal_upstream?;
        let (metrics, labels, downstream) = self.e2e_labels.take()?;
        self.e2e_upstream_recorded = true;
        // The attempt ran inside the request, but the two durations are read
        // off the clock a moment apart — the downstream one first on some
        // stream ends — so the later read can come out a millisecond longer.
        Some((metrics, labels, upstream.min(downstream)))
    }
}

/// A completed access-log line held until the response body is finished
/// with, because both body sizes are only known then.
struct ReadyLine {
    record: aisix_obs::AccessLogRecord,
    /// Set for a line whose `duration` must span the body relay rather than
    /// end where the line was built — see [`emit_access_log_spanning_body`].
    duration_from: Option<Instant>,
}

/// Body sizes, counted where the bytes cross the gateway's outermost
/// layer. Atomics rather than the cell's lock because the request body is
/// read on whatever task consumes it, possibly a client's connection task.
#[derive(Default)]
struct BodyCounters {
    request_bytes: AtomicU64,
    /// The request body was read to its end.
    request_complete: AtomicBool,
    /// The request body was read only to be thrown away, by a refusal that
    /// decided before the body was read.
    request_discarded: AtomicBool,
    response_bytes: AtomicU64,
}

/// The maximum child embedding bridge invocations retained per parent event.
///
/// The parent event keeps this ordered prefix and a saturating dropped count,
/// rather than allowing an attacker-controlled semantic configuration to make
/// telemetry allocation unbounded.
pub(crate) const MAX_GATEWAY_EMBEDDING_CALLS: usize = 64;

/// The child embedding calls a request caused while it was being served.
///
/// This sits outside [`Cell`]: [`detached`] must isolate target attribution,
/// but semantic routing, the semantic cache, and semantic guardrails still
/// need to append their gateway-initiated calls to the real parent's terminal
/// usage event. A detached child therefore receives a fresh `Cell` and shares
/// only this ledger with its parent.
#[derive(Default)]
struct GatewayEmbeddingLedger {
    state: Mutex<GatewayEmbeddingLedgerState>,
}

#[derive(Default)]
struct GatewayEmbeddingLedgerState {
    calls: Vec<aisix_obs::GatewayEmbeddingCall>,
    dropped: u32,
}

/// The bounded child-work audit transferred to a terminal parent event.
#[derive(Default)]
pub(crate) struct GatewayEmbeddingAudit {
    pub(crate) calls: Vec<aisix_obs::GatewayEmbeddingCall>,
    pub(crate) dropped: u32,
}

impl GatewayEmbeddingLedger {
    fn record(&self, call: aisix_obs::GatewayEmbeddingCall) {
        let mut ledger = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if ledger.calls.len() < MAX_GATEWAY_EMBEDDING_CALLS {
            ledger.calls.push(call);
            return;
        }
        ledger.dropped = ledger.dropped.saturating_add(1);
    }
}

/// Records an in-flight gateway-owned embedding call if its future is
/// cancelled before the bridge returns.
///
/// The guard owns the ledger rather than consulting [`CURRENT`] from
/// [`Drop`]. A handler cancellation drops the detached child scope before the
/// parent cancel guard emits its terminal event, but task-local state is not a
/// safe dependency for that cleanup path. The shared ledger is still alive at
/// that point, so the terminal 499 event retains the failed child call.
#[must_use = "an in-flight embedding call must stay guarded until it settles"]
pub(crate) struct GatewayEmbeddingCallGuard {
    ledger: Option<Arc<GatewayEmbeddingLedger>>,
    purpose: aisix_obs::GatewayEmbeddingPurpose,
    embedding_model_id: String,
    started: Instant,
    recorded: bool,
}

impl GatewayEmbeddingCallGuard {
    /// Disarm the cancellation fallback and return this bridge invocation's
    /// elapsed time for the ordinary success/error record.
    pub(crate) fn settle(&mut self) -> u32 {
        self.recorded = true;
        u32::try_from(self.started.elapsed().as_millis()).unwrap_or(u32::MAX)
    }
}

impl Drop for GatewayEmbeddingCallGuard {
    fn drop(&mut self) {
        if !self.recorded {
            if let Some(ledger) = &self.ledger {
                ledger.record(aisix_obs::GatewayEmbeddingCall {
                    count: 1,
                    purpose: self.purpose,
                    embedding_model_id: self.embedding_model_id.clone(),
                    prompt_tokens: 0,
                    total_tokens: 0,
                    usage_source: aisix_obs::GatewayEmbeddingUsageSource::Unavailable,
                    latency_ms: u32::try_from(self.started.elapsed().as_millis())
                        .unwrap_or(u32::MAX),
                    outcome: aisix_obs::GatewayEmbeddingOutcome::Failed,
                });
            }
        }
    }
}

/// The per-request cell. Attempts within a request are sequential, so the
/// lock is uncontended; it exists because the cancel guard may read the
/// cell from a different point in the stack than the writer.
#[derive(Default)]
pub(crate) struct RequestAttribution {
    cell: Mutex<Cell>,
    bodies: BodyCounters,
    gateway_embedding_ledger: Arc<GatewayEmbeddingLedger>,
}

impl RequestAttribution {
    /// A sub-call scope that cannot change the parent request's target
    /// attribution but can still account for gateway-initiated embeddings on
    /// the parent's eventual terminal usage event.
    fn detached_child(&self) -> Self {
        Self {
            cell: Mutex::new(Cell::default()),
            bodies: BodyCounters::default(),
            gateway_embedding_ledger: Arc::clone(&self.gateway_embedding_ledger),
        }
    }

    fn note_gateway_embedding_call(&self, call: aisix_obs::GatewayEmbeddingCall) {
        self.gateway_embedding_ledger.record(call);
    }

    /// Move the bounded child-call prefix and its dropped count into the one
    /// terminal parent event. Taking rather than cloning makes a duplicate
    /// terminal emitter unable to report the same bridge call twice.
    fn take_gateway_embedding_audit(&self) -> GatewayEmbeddingAudit {
        let mut ledger = self
            .gateway_embedding_ledger
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        GatewayEmbeddingAudit {
            calls: std::mem::take(&mut ledger.calls),
            dropped: std::mem::take(&mut ledger.dropped),
        }
    }

    /// Mark this cell as installed by the telemetry middleware, which will
    /// call [`Self::finish`] once the response body is done with. From here
    /// an access-log line is held until then. `span` is the request span
    /// the line is written in, since it is written from `Drop`.
    pub(crate) fn track(&self, span: tracing::Span) {
        self.lock().request_span = Some(span);
    }

    pub(crate) fn note_head_written(&self) {
        self.lock().head_written = true;
    }

    pub(crate) fn add_request_bytes(&self, n: u64) {
        self.bodies.request_bytes.fetch_add(n, Ordering::Relaxed);
    }

    pub(crate) fn note_request_complete(&self) {
        self.bodies.request_complete.store(true, Ordering::Relaxed);
    }

    pub(crate) fn add_response_bytes(&self, n: u64) {
        self.bodies.response_bytes.fetch_add(n, Ordering::Relaxed);
    }

    /// `(request_body_bytes, response_body_bytes)` as the line reports them.
    fn body_sizes(&self, head_written: bool) -> (Option<u64>, Option<u64>) {
        let b = &self.bodies;
        let request = (b.request_complete.load(Ordering::Relaxed)
            && !b.request_discarded.load(Ordering::Relaxed))
        .then(|| b.request_bytes.load(Ordering::Relaxed));
        let response = head_written.then(|| b.response_bytes.load(Ordering::Relaxed));
        (request, response)
    }

    /// Hold `line` until the body sizes are known, or write it now when
    /// nothing will ever flush it (see [`Cell::request_span`] and
    /// [`Cell::finished`]).
    pub(crate) fn park_access_log(
        &self,
        line: &aisix_obs::AccessLog<'_>,
        duration_from: Option<Instant>,
    ) {
        let mut cell = self.lock();
        let mut record = line.to_record();
        record.semantic = cell.semantic.clone();
        let ready = ReadyLine {
            record,
            duration_from,
        };
        if cell.request_span.is_none() {
            drop(cell);
            ready.record.emit();
            return;
        }
        if cell.finished {
            let sizes = self.body_sizes(cell.head_written);
            let span = cell.request_span.clone();
            drop(cell);
            write_line(ready, sizes, span);
            return;
        }
        // One line per request is the invariant; should a path ever build
        // two, the first is written now rather than lost.
        if let Some(earlier) = cell.ready_line.replace(ready) {
            let sizes = self.body_sizes(false);
            let span = cell.request_span.clone();
            drop(cell);
            write_line(earlier, sizes, span);
        }
    }

    /// The request is over: its response body was fully handed to the
    /// server or dropped, or no response head was ever produced. Writes the
    /// held line with its final body sizes. Called once, by the telemetry
    /// middleware's guard.
    pub(crate) fn finish(&self) {
        let mut cell = self.lock();
        cell.finished = true;
        let line = cell.ready_line.take();
        let sizes = self.body_sizes(cell.head_written);
        let span = cell.request_span.clone();
        drop(cell);
        if let Some(line) = line {
            write_line(line, sizes, span);
        }
    }

    pub(crate) fn get(&self) -> Resolved {
        self.lock().resolved.clone()
    }

    /// Move the cancel context out of the cell. Only the cancel guard calls
    /// this, once, from `Drop` — taking rather than cloning keeps the
    /// `Vec<AttemptRecord>` off every other read of this cell.
    pub(crate) fn take_cancel_context(&self) -> CancelContext {
        std::mem::take(&mut self.lock().cancel)
    }

    /// Whether this request still has a parked line — i.e. whether a
    /// terminal emitter is still owed one. Read by the cancel guard before
    /// it builds a line of its own.
    pub(crate) fn has_pending_access_log(&self) -> bool {
        self.lock().pending_log.is_some()
    }

    /// Emit the request's deferred line, if it still has one, against the
    /// outcome `event` reports. Returns whether a line went out.
    ///
    /// Read through the cell handle rather than the task-local because the
    /// cancel guard holds the handle and runs from `Drop`, outside every
    /// scope.
    pub(crate) fn emit_deferred_access_log(&self, event: &aisix_obs::UsageEvent) -> bool {
        let mut cell = self.lock();
        let Some(pending) = cell.pending_log.take() else {
            return false;
        };
        let target = cell.resolved.clone();
        drop(cell);
        pending.emit(self, &target, event);
        true
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Cell> {
        self.cell
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn write_line(line: ReadyLine, sizes: (Option<u64>, Option<u64>), span: Option<tracing::Span>) {
    let mut record = line.record;
    if let Some(from) = line.duration_from {
        record.duration = from.elapsed();
    }
    (record.request_body_bytes, record.response_body_bytes) = sizes;
    match span {
        Some(span) => span.in_scope(|| record.emit()),
        None => record.emit(),
    }
}

tokio::task_local! {
    static CURRENT: Arc<RequestAttribution>;
}

/// Install `attribution` as the cell for everything `fut` does.
pub(crate) fn scope<F: Future>(
    attribution: Arc<RequestAttribution>,
    fut: F,
) -> impl Future<Output = F::Output> {
    CURRENT.scope(attribution, fut)
}

/// Install `cell` for the duration of one synchronous call.
///
/// Used to drop the response body inside the request's own cell
/// (`TelemetryBody`): two families build their stream's terminal emitter
/// OUTSIDE the generator, so it fires even on a body that was never polled
/// — and the cancel guard must be able to see that it did, or it writes a
/// second, contradicting row for the same request.
pub(crate) fn sync_scope<T>(cell: &Arc<RequestAttribution>, f: impl FnOnce() -> T) -> T {
    CURRENT.sync_scope(cell.clone(), f)
}

/// Run `fut` against a throwaway cell, so a sub-call the request makes ON
/// ITS OWN BEHALF cannot be mistaken for the target the CALLER addressed.
///
/// The cell records the last target `resolve_provider_key` committed to,
/// which is right for every dispatch the caller asked for and wrong for
/// every one the gateway decided to make: a semantic guardrail and the
/// semantic cache both call an embedding model, and the guardrail's output
/// hook and the cache write both run AFTER the winning attempt. Without
/// this, the request's own access-log line would name the embedding model
/// as the upstream it dispatched to — a field that is wrong rather than
/// merely absent, on exactly the line an operator reads to find out which
/// member of a routing group served them (AISIX-Cloud#1571).
pub(crate) fn detached<F: Future>(fut: F) -> impl Future<Output = F::Output> {
    let child = CURRENT
        .try_with(|parent| Arc::new(parent.detached_child()))
        .unwrap_or_else(|_| Arc::new(RequestAttribution::default()));
    CURRENT.scope(child, fut)
}

/// Note the model name the caller addressed. Called once per request from
/// the resolution chokepoint, and only when the name resolved — an
/// unresolvable name never reaches a target and its request fails before
/// anything can read the cell.
pub(crate) fn note_requested_model(requested: &str) {
    with(|r| {
        if r.requested_model.is_empty() {
            r.requested_model = requested.to_string();
        }
    });
}

/// Note the target this request is about to dispatch to. Called for every
/// attempt, so the cell holds the last one — the target whose failure the
/// caller was ultimately served.
pub(crate) fn note_target(model: &Model, provider_key_id: &str) {
    with(|r| {
        r.provider = model.provider.clone().unwrap_or_default();
        r.upstream_model = model.upstream_model().unwrap_or_default().to_string();
        r.provider_key_id = provider_key_id.to_string();
    });
}

/// Overwrite the target half with what a CACHE HIT may honestly claim.
///
/// A hit contacts no upstream, so nothing was dispatched to and the line
/// must not report one. What survives is the entry's own static mapping: a
/// DIRECT model's `model_name` and `provider_key_id` are properties of the
/// row the caller addressed, true whether or not a request ever left the
/// gateway. A Model Group has neither of its own, and which of its targets
/// produced the stored entry is recorded nowhere, so its line names no
/// target at all rather than the candidate this request happened to rank
/// first (AISIX-Cloud#1571).
///
/// It OVERWRITES rather than fills, and that is the whole point: the
/// pre-flight in [`crate::dispatch::resolve_provider_key`] runs before the
/// cache is consulted and has already written a target for every entry
/// that resolved to a SINGLE candidate — including a one-target routing
/// group, whose candidate is no more the producer than any other.
pub(crate) fn note_cache_hit_entry(entry: &Model, hit_layer: &'static str) {
    note_target(entry, entry.provider_key_id.as_deref().unwrap_or_default());
    with(|r| r.cache_hit_layer = Some(hit_layer));
}

/// What the current request has resolved, or `None` outside a request.
pub(crate) fn current() -> Option<Resolved> {
    CURRENT.try_with(|a| a.get()).ok()
}

/// Record one actual gateway-initiated embedding bridge call. This is safe
/// inside [`detached`]: that scope keeps a throwaway target cell but shares
/// the parent's [`GatewayEmbeddingLedger`]. Calls outside an HTTP request are
/// deliberately dropped, just like every other attribution write.
pub(crate) fn note_gateway_embedding_call(call: aisix_obs::GatewayEmbeddingCall) {
    let _ = CURRENT.try_with(|a| a.note_gateway_embedding_call(call));
}

/// Start accounting for one gateway-initiated embedding bridge call.
///
/// Call this immediately before `bridge.embed(...).await`, then call
/// [`GatewayEmbeddingCallGuard::settle`] before emitting the ordinary
/// success/error record. If the enclosing request future is dropped while
/// awaiting the bridge, [`Drop`] records a bounded failed call against the
/// parent ledger before cancellation emits the parent event.
pub(crate) fn begin_gateway_embedding_call(
    purpose: aisix_obs::GatewayEmbeddingPurpose,
    embedding_model_id: &str,
) -> GatewayEmbeddingCallGuard {
    GatewayEmbeddingCallGuard {
        ledger: CURRENT
            .try_with(|attribution| Arc::clone(&attribution.gateway_embedding_ledger))
            .ok(),
        purpose,
        embedding_model_id: embedding_model_id.to_string(),
        started: Instant::now(),
        recorded: false,
    }
}

/// Move this request's bounded gateway-initiated embedding audit into its
/// terminal usage event. Outside a request scope this is empty; cancellation
/// installs the original cell with [`sync_scope`] before it emits, so it
/// follows the same path as ordinary and streaming terminal emitters.
pub(crate) fn take_gateway_embedding_audit() -> GatewayEmbeddingAudit {
    CURRENT
        .try_with(|a| a.take_gateway_embedding_audit())
        .unwrap_or_default()
}

/// The dispatched-target half of an access-log line (AISIX-Cloud#1571).
///
/// The line's `model=` is the entry the CALLER addressed — for a routing
/// group, the group. That leaves the target that was actually selected
/// unnamed anywhere an operator can reach by request id, which is exactly
/// what a 499 line needs to say. Both fields are `None` until a target was
/// selected, so a line written before dispatch carries no target-derived
/// field at all.
///
/// Read off the request's attribution cell, so it is empty on the emitters
/// that run detached from the request task (`/v1/realtime`'s session
/// close) — there the same values ride the session's own usage event.
pub(crate) struct AccessLogTarget(Resolved);

impl AccessLogTarget {
    /// The target this request last selected.
    pub(crate) fn current() -> Self {
        Self(current().unwrap_or_default())
    }

    /// For a caller that already has the cell's contents in hand.
    pub(crate) fn from_resolved(resolved: Resolved) -> Self {
        Self(resolved)
    }

    pub(crate) fn upstream_model(&self) -> Option<&str> {
        (!self.0.upstream_model.is_empty()).then_some(self.0.upstream_model.as_str())
    }

    pub(crate) fn provider_key_id(&self) -> Option<&str> {
        (!self.0.provider_key_id.is_empty()).then_some(self.0.provider_key_id.as_str())
    }

    /// The cache verdict for an emitter that has none of its own. Only a
    /// HIT is ever recorded on the cell, so this is `None` on every other
    /// outcome — including a miss, which the buffered success exit reports
    /// from the verdict it holds directly.
    pub(crate) fn cache(&self) -> Option<aisix_obs::CacheAccessLog<'_>> {
        self.0
            .cache_hit_layer
            .map(|layer| aisix_obs::CacheAccessLog {
                status: "hit",
                hit_layer: Some(layer),
            })
    }
}

fn with(f: impl FnOnce(&mut Resolved)) {
    let _ = CURRENT.try_with(|a| f(&mut a.lock().resolved));
}

/// Same, for the cancel half of the cell.
fn with_cancel(f: impl FnOnce(&mut CancelContext)) {
    let _ = CURRENT.try_with(|a| f(&mut a.lock().cancel));
}

/// Note the caller behind this request. Called once, from the
/// [`ClientContext`] extractor — the one place every client-facing handler
/// resolves its caller, and which by construction runs after the `auth`
/// extractor that publishes the api_key row.
pub(crate) fn note_client(client: &ClientContext, api_key_id: &str) {
    with_cancel(|c| {
        // Never blank an id the auth chokepoint already resolved: this
        // extractor reads the api_key EXTENSION, which a route that
        // authenticates inside its own handler (passthrough, `/mcp`) has
        // not set yet, and an empty write here would undo it.
        if !api_key_id.is_empty() {
            c.api_key_id = api_key_id.to_string();
        }
        c.client = Some(client.clone());
    });
}

/// Note the authenticated principal. Called from the `AuthenticatedKey`
/// extractor — the one place every authenticated route resolves who is
/// calling, including the two that build no [`ClientContext`] and would
/// otherwise reach the cancel emitter with nothing to attribute the row to.
///
/// `api_key_id` is written here as well as by [`note_client`]. Both assign
/// unconditionally, so the later writer wins — which changes nothing,
/// because both read the id off the same `api_key` row extension.
pub(crate) fn note_authenticated(auth: &crate::auth::AuthenticatedKey) {
    with_cancel(|c| {
        c.api_key_id = auth.entry.id.clone();
        c.auth = Some(auth.clone());
    });
}

/// Note that this route files no usage row whatever the outcome. See
/// [`CancelContext::unmetered`]; the census in [`crate::operation`] is where
/// the routes that need it are named.
pub(crate) fn note_unmetered_route() {
    with_cancel(|c| c.unmetered = true);
}

/// Note the passthrough route this request matched, for the row a cancelled
/// one files. Called once, after `match_route`.
pub(crate) fn note_passthrough_route(route: &str) {
    with_cancel(|c| {
        c.route.passthrough_route = route.to_string();
        // The one point that KNOWS this is passthrough traffic. See
        // [`CancelContext::surface`] for the two route shapes whose path
        // says something else entirely.
        c.surface = Some(crate::operation::PASSTHROUGH);
    });
}

/// Note the MCP server the request addressed, and the tool its JSON-RPC body
/// named. Called twice: once with an empty `tool` before the body is read,
/// so a cancel during the upload still reports the server a scoped entry
/// named in its path, and again once the body has been parsed. Empty
/// arguments never overwrite what is already there.
pub(crate) fn note_mcp_call(server: &str, tool: &str) {
    with_cancel(|c| {
        if !server.is_empty() {
            c.route.mcp_server = server.to_string();
        }
        if !tool.is_empty() {
            c.route.mcp_tool = tool.to_string();
        }
    });
}

/// Note the A2A agent the request addressed and the JSON-RPC method it
/// called.
pub(crate) fn note_a2a_call(agent: &str, method: &str, operation: &str) {
    with_cancel(|c| {
        c.route.a2a_agent = agent.to_string();
        if !method.is_empty() {
            c.route.a2a_method = method.to_string();
        }
        if !operation.is_empty() {
            c.route.a2a_operation = operation.to_string();
        }
    });
}

/// Note the Model row the caller's name resolved to, when that row is a
/// dispatch target in its own right. Called from the resolution chokepoint
/// beside [`note_requested_model`]; a composite entry passes an empty id,
/// because its own uuid must never be reported as the model that served.
pub(crate) fn note_resolved_entry(entry_model_id: &str) {
    with_cancel(|c| {
        if c.entry_model_id.is_empty() {
            c.entry_model_id = entry_model_id.to_string();
        }
    });
}

/// Note that an attempt has begun. Paired with [`note_attempt_settled`],
/// both driven from [`crate::attempt::RoutingTelemetry`] so the three
/// Model-Group dispatch endpoints cannot record an attempt the cancel path
/// cannot see.
pub(crate) fn note_attempt_started(attempt: InFlightAttempt) {
    with_cancel(|c| c.in_flight = Some(attempt));
}

/// Note that the attempt in flight resolved, one way or the other.
///
/// A settled attempt is kept, not discarded — a cancel lands in the gaps too
/// (the retry backoff; the output scan and cache write after the winner), and
/// a target HAD been selected there. How much of it is kept depends on
/// whether its own event exists yet: a FAILED attempt's does, so only its
/// model id survives (`last_settled_model_id`) and repeating its index would
/// put two rows under one `attempt_index`; a WINNING attempt's does not,
/// because the handler writes that one after processing the response, so it
/// is kept whole (`won`).
pub(crate) fn note_attempt_settled(rec: &AttemptRecord) {
    with_cancel(|c| {
        c.in_flight = None;
        c.last_settled_model_id = rec.target_model_id.clone();
        if rec.success {
            c.won = Some(SettledWinner {
                index: rec.index,
                kind: rec.kind,
                target_model: rec.target_model.clone(),
                model_id: rec.target_model_id.clone(),
                latency_ms: rec.latency_ms,
            });
        } else {
            c.failed_attempts.push(rec.clone());
        }
    });
}

/// Park this request's access-log line until its outcome is known. Called
/// by a streaming handler in place of writing the line itself — see
/// [`PendingAccessLog`].
pub(crate) fn defer_access_log(pending: PendingAccessLog) {
    let _ = CURRENT.try_with(|a| a.lock().pending_log = Some(pending));
}

/// Note that this request answered with a stream whose own terminal
/// emitter will write the access-log line.
///
/// For the handlers that wrap their whole dispatch and log the wrapper's
/// status (`/a2a`): the streaming branch is several frames below the tail
/// that owns the line's fields, so it raises a flag there and the tail
/// parks the line instead of writing it.
pub(crate) fn note_stream_owns_access_log() {
    let _ = CURRENT.try_with(|a| a.lock().stream_owns_log = true);
}

/// Whether [`note_stream_owns_access_log`] was raised on this request.
pub(crate) fn stream_owns_access_log() -> bool {
    CURRENT
        .try_with(|a| a.lock().stream_owns_log)
        .unwrap_or(false)
}

/// Write a request's access-log line. Every emitter in this crate goes
/// through here rather than `AccessLog::emit`: inside a request the line is
/// held until the response body is done with, so it can carry both body
/// sizes; outside one it is written at once.
pub(crate) fn emit_access_log(line: aisix_obs::AccessLog<'_>) {
    park(line, None);
}

/// [`emit_access_log`] for a line built at the response head of a route
/// that then relays an open-ended body (`/v1/videos/{id}/content`): its
/// `duration` is measured from `started` to the end of that relay, as it is
/// on every surface whose line is completed at the request's end.
pub(crate) fn emit_access_log_spanning_body(line: aisix_obs::AccessLog<'_>, started: Instant) {
    park(line, Some(started));
}

fn park(line: aisix_obs::AccessLog<'_>, duration_from: Option<Instant>) {
    if CURRENT
        .try_with(|a| a.park_access_log(&line, duration_from))
        .is_err()
    {
        line.emit();
    }
}

/// Note how the semantic router this request addressed decided, for its
/// access-log line.
///
/// Published here rather than threaded to the line, because the line has
/// several writers — the handler's buffered exit, a stream's terminal
/// emitter, the cancel guard — and the decision is made before any of them
/// exists. Every one of them parks its line on this cell, which is where
/// it is attached ([`RequestAttribution::park_access_log`]).
pub(crate) fn note_semantic_decision(decision: aisix_obs::SemanticAccessLog) {
    let _ = CURRENT.try_with(|a| a.lock().semantic = Some(decision));
}

/// Note that the request body was read only to be discarded by a refusal
/// decided before it was read, so the line reports no request size.
pub(crate) fn note_request_body_discarded() {
    let _ = CURRENT.try_with(|a| a.bodies.request_discarded.store(true, Ordering::Relaxed));
}

/// Note the labels of the request's `downstream` end-to-end latency
/// observation, so its `upstream` sibling is recorded under the same ones.
///
/// The two halves of that sibling — these labels, and the terminal
/// attempt's duration ([`note_terminal_upstream`]) — arrive in either
/// order depending on the family, so whichever comes second records it.
pub(crate) fn note_e2e_downstream(
    metrics: Arc<aisix_obs::Metrics>,
    labels: crate::request_metrics::OwnedLatencyLabels,
    downstream: Duration,
) {
    record_e2e_upstream(|cell| cell.e2e_labels = Some((metrics, labels, downstream)));
}

/// Note the upstream duration of the attempt behind the request's terminal
/// usage event, when that attempt was dispatched. Called from the emission
/// chokepoint, so it is the same figure as the event's
/// `upstream_latency_ms`.
pub(crate) fn note_terminal_upstream(upstream: Duration) {
    record_e2e_upstream(|cell| cell.terminal_upstream = Some(upstream));
}

fn record_e2e_upstream(note: impl FnOnce(&mut Cell)) {
    let ready = CURRENT
        .try_with(|a| {
            let mut cell = a.lock();
            note(&mut cell);
            cell.take_e2e_upstream()
        })
        .ok()
        .flatten();
    if let Some((metrics, labels, upstream)) = ready {
        metrics.record_request_e2e_latency(
            labels.as_labels(),
            aisix_obs::LatencySide::Upstream,
            upstream,
        );
    }
}

/// Emit the deferred line of the request running on this task, if it has
/// one. Called from the terminal-usage-event chokepoint.
pub(crate) fn emit_deferred_access_log(event: &aisix_obs::UsageEvent) {
    let _ = CURRENT.try_with(|a| a.emit_deferred_access_log(event));
}

/// Note that a usage event has just left the emission chokepoint on this
/// request's own task. See [`CancelContext::emitted_any`].
pub(crate) fn note_usage_emitted(terminal: bool) {
    with_cancel(|c| {
        c.emitted_any = true;
        c.emitted_terminal |= terminal;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(provider: &str, upstream: &str) -> Model {
        serde_json::from_value(serde_json::json!({
            "display_name": "m",
            "provider": provider,
            "model_name": upstream,
            "provider_key_id": "pk-1",
        }))
        .unwrap()
    }

    fn embedding_call(index: usize) -> aisix_obs::GatewayEmbeddingCall {
        aisix_obs::GatewayEmbeddingCall {
            count: 1,
            purpose: aisix_obs::GatewayEmbeddingPurpose::SemanticRoute,
            embedding_model_id: format!("embed-{index}"),
            prompt_tokens: index as u32,
            total_tokens: index as u32,
            usage_source: aisix_obs::GatewayEmbeddingUsageSource::Reported,
            latency_ms: index as u32,
            outcome: aisix_obs::GatewayEmbeddingOutcome::Succeeded,
        }
    }

    /// The two sides are read off the clock a moment apart, downstream
    /// first on some stream ends; the recorded upstream side must still
    /// never exceed the request it ran inside.
    #[tokio::test]
    async fn the_upstream_side_never_exceeds_the_downstream_side() {
        let metrics = Arc::new(aisix_obs::Metrics::new(false));
        let labels = aisix_obs::LatencyLabels {
            endpoint: "/v1/messages",
            model: "m",
            provider: "anthropic",
            status: 200,
            streaming: true,
            ..Default::default()
        };
        scope(Arc::new(RequestAttribution::default()), async {
            crate::request_metrics::record_e2e_downstream(
                &metrics,
                labels,
                Duration::from_millis(306),
            );
            note_terminal_upstream(Duration::from_millis(307));
        })
        .await;
        let text = metrics.render();
        let sum = |side: &str| -> f64 {
            text.lines()
                .find(|l| {
                    l.starts_with("aisix_request_e2e_latency_seconds_sum")
                        && l.contains(&format!("side=\"{side}\""))
                })
                .and_then(|l| l.rsplit(' ').next())
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("no {side} sum in:\n{text}"))
        };
        assert!(
            sum("upstream") <= sum("downstream"),
            "upstream {} > downstream {}",
            sum("upstream"),
            sum("downstream")
        );
    }

    /// A write with no request scope must not panic — `resolve_provider_key`
    /// also runs from the background health checker.
    #[test]
    fn writes_outside_a_request_are_dropped() {
        note_requested_model("gpt-4o");
        note_target(&model("openai", "gpt-4o"), "pk-1");
        assert!(current().is_none());
    }

    /// A gateway-owned embedding must retain the caller's actual target while
    /// still reaching that caller's one terminal usage event. This is the
    /// reason the detached cell shares only the child-call ledger, not its
    /// resolved target state.
    #[tokio::test]
    async fn detached_embedding_call_reaches_the_parent_ledger_once() {
        let parent = Arc::new(RequestAttribution::default());
        scope(parent.clone(), async {
            note_target(&model("anthropic", "claude-sonnet"), "pk-chat");
            detached(async {
                note_target(&model("openai", "text-embedding-3-small"), "pk-embed");
                note_gateway_embedding_call(aisix_obs::GatewayEmbeddingCall {
                    count: 1,
                    purpose: aisix_obs::GatewayEmbeddingPurpose::Guardrail,
                    embedding_model_id: "embed-model-id".into(),
                    prompt_tokens: 4,
                    total_tokens: 4,
                    usage_source: aisix_obs::GatewayEmbeddingUsageSource::Reported,
                    latency_ms: 9,
                    outcome: aisix_obs::GatewayEmbeddingOutcome::Succeeded,
                });
            })
            .await;

            let target = current().expect("parent request remains in scope");
            assert_eq!(target.upstream_model, "claude-sonnet");
            assert_eq!(target.provider_key_id, "pk-chat");

            let audit = take_gateway_embedding_audit();
            assert_eq!(audit.calls.len(), 1);
            assert_eq!(audit.dropped, 0);
            assert_eq!(
                audit.calls[0].purpose,
                aisix_obs::GatewayEmbeddingPurpose::Guardrail
            );
            assert!(take_gateway_embedding_audit().calls.is_empty());
        })
        .await;
    }

    /// The guard is created immediately before `bridge.embed(...).await`.
    /// Poll this child through that point, then drop it as a cancelled handler
    /// would; the parent terminal emitter must still see the in-flight call.
    #[tokio::test]
    async fn cancelled_child_embedding_bridge_call_reaches_the_parent_ledger() {
        let parent = Arc::new(RequestAttribution::default());
        scope(parent, async {
            let mut bridge_await = Box::pin(detached(async {
                let _call = begin_gateway_embedding_call(
                    aisix_obs::GatewayEmbeddingPurpose::SemanticRoute,
                    "embed-model-id",
                );
                std::future::pending::<()>().await;
            }));
            std::future::poll_fn(|cx| {
                assert!(std::future::Future::poll(bridge_await.as_mut(), cx).is_pending());
                std::task::Poll::Ready(())
            })
            .await;
            drop(bridge_await);

            let audit = take_gateway_embedding_audit();
            assert_eq!(audit.calls.len(), 1);
            assert_eq!(audit.dropped, 0);
            let call = &audit.calls[0];
            assert_eq!(
                call.purpose,
                aisix_obs::GatewayEmbeddingPurpose::SemanticRoute
            );
            assert_eq!(call.embedding_model_id, "embed-model-id");
            assert_eq!(
                call.usage_source,
                aisix_obs::GatewayEmbeddingUsageSource::Unavailable
            );
            assert_eq!(call.outcome, aisix_obs::GatewayEmbeddingOutcome::Failed);
            assert_eq!(call.prompt_tokens, 0);
            assert_eq!(call.total_tokens, 0);
        })
        .await;
    }

    #[tokio::test]
    async fn child_embedding_ledger_keeps_the_ordered_prefix_and_reports_all_drops() {
        let parent = Arc::new(RequestAttribution::default());
        scope(parent, async {
            let total = MAX_GATEWAY_EMBEDDING_CALLS + 257;
            for index in 0..total {
                note_gateway_embedding_call(embedding_call(index));
            }

            let audit = take_gateway_embedding_audit();
            assert_eq!(audit.calls.len(), MAX_GATEWAY_EMBEDDING_CALLS);
            assert_eq!(audit.calls[0].embedding_model_id, "embed-0");
            assert_eq!(
                audit.calls[MAX_GATEWAY_EMBEDDING_CALLS - 1].embedding_model_id,
                format!("embed-{}", MAX_GATEWAY_EMBEDDING_CALLS - 1),
            );
            assert_eq!(audit.dropped, 257);

            let second_take = take_gateway_embedding_audit();
            assert!(second_take.calls.is_empty());
            assert_eq!(second_take.dropped, 0);
        })
        .await;
    }

    #[test]
    fn child_embedding_ledger_drop_counter_saturates() {
        let attribution = RequestAttribution::default();
        for index in 0..MAX_GATEWAY_EMBEDDING_CALLS {
            attribution.note_gateway_embedding_call(embedding_call(index));
        }
        attribution
            .gateway_embedding_ledger
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .dropped = u32::MAX;

        attribution.note_gateway_embedding_call(embedding_call(MAX_GATEWAY_EMBEDDING_CALLS));
        let audit = attribution.take_gateway_embedding_audit();
        assert_eq!(audit.calls.len(), MAX_GATEWAY_EMBEDDING_CALLS);
        assert_eq!(audit.dropped, u32::MAX);
    }

    #[tokio::test]
    async fn the_last_attempted_target_is_what_a_failure_reads() {
        scope(Arc::new(RequestAttribution::default()), async {
            note_requested_model("my-group");
            note_target(&model("openai", "gpt-4o"), "pk-openai");
            // The group fell back; the caller's error came from the second
            // target, so that is the one the terminal emit must name.
            note_target(&model("anthropic", "claude-3-5-sonnet"), "pk-anthropic");
            let r = current().expect("in scope");
            // Nothing but a cache hit writes this, so a dispatched request
            // must not look like one.
            assert_eq!(r.cache_hit_layer, None);
            assert!(AccessLogTarget::from_resolved(r.clone()).cache().is_none());
            assert_eq!(r.requested_model, "my-group");
            assert_eq!(r.provider, "anthropic");
            assert_eq!(r.upstream_model, "claude-3-5-sonnet");
            assert_eq!(r.provider_key_id, "pk-anthropic");
        })
        .await;
    }

    fn group_model() -> Model {
        serde_json::from_value(serde_json::json!({
            "display_name": "my-group",
            "routing": {
                "targets": [{"model": "a"}, {"model": "b"}],
            },
        }))
        .unwrap()
    }

    /// A cache hit on a GROUP must leave no target behind, even though the
    /// pre-flight already wrote one: a routing group with a single eligible
    /// candidate reaches `resolve_provider_key` before the cache is
    /// consulted, and that candidate did not serve the stored answer
    /// (AISIX-Cloud#1571).
    #[tokio::test]
    async fn a_group_cache_hit_erases_the_preflight_target() {
        scope(Arc::new(RequestAttribution::default()), async {
            note_requested_model("my-group");
            note_target(&model("openai", "gpt-4o"), "pk-openai");
            note_cache_hit_entry(&group_model(), "exact");
            let r = current().expect("in scope");
            assert_eq!(r.requested_model, "my-group");
            assert_eq!(r.provider, "");
            assert_eq!(r.upstream_model, "");
            assert_eq!(r.provider_key_id, "");
            let target = AccessLogTarget::from_resolved(r);
            assert_eq!(target.upstream_model(), None);
            assert_eq!(target.provider_key_id(), None);
            // ...and the same call says WHY there is no target, for the
            // emitters that hold a `ProxyError` and never saw the cache.
            let cache = target.cache().expect("a hit was recorded");
            assert_eq!(cache.status, "hit");
            assert_eq!(cache.hit_layer, Some("exact"));
        })
        .await;
    }

    /// A direct entry keeps its own mapping across the same call: those
    /// fields are properties of the row the caller addressed, not evidence
    /// that anything was dispatched.
    #[tokio::test]
    async fn a_direct_cache_hit_keeps_the_entrys_own_mapping() {
        scope(Arc::new(RequestAttribution::default()), async {
            let direct = model("openai", "gpt-4o");
            note_requested_model("m");
            note_cache_hit_entry(&direct, "semantic");
            let r = current().expect("in scope");
            assert_eq!(r.provider, "openai");
            assert_eq!(r.upstream_model, "gpt-4o");
            assert_eq!(r.provider_key_id, "pk-1");
            assert_eq!(r.cache_hit_layer, Some("semantic"));
        })
        .await;
    }

    struct EmitOnDrop;
    impl Drop for EmitOnDrop {
        fn drop(&mut self) {
            note_usage_emitted(true);
        }
    }

    /// The double-emission interlock rests on the cell being writable for as
    /// long as the handler's own code can still run — which includes the
    /// scoped future being DROPPED, since a `Drop` emitter inside the
    /// handler runs there. tokio installs the value during that drop, so
    /// `note_usage_emitted` reaches the cell and the guard stays quiet.
    ///
    /// What the interlock covers on this side is the handler emitting on
    /// its own task and then awaiting again (a cache write, a content
    /// capture) before returning. The streaming families reach the same
    /// cell from the body phase instead — see [`sync_scope`].
    /// This test pins the drop half anyway, because it is the half nothing
    /// else would notice losing: were tokio to stop installing the value, a
    /// future `Drop` emitter would silently write nowhere and one cancelled
    /// request would report two contradicting terminal rows.
    #[tokio::test]
    async fn the_cell_is_writable_while_the_scoped_future_is_dropped() {
        let cell = Arc::new(RequestAttribution::default());
        let mut fut = Box::pin(scope(cell.clone(), async {
            let _emitter = EmitOnDrop;
            std::future::pending::<()>().await;
        }));
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        assert!(
            std::future::Future::poll(fut.as_mut(), &mut cx).is_pending(),
            "premise: the future must be parked mid-flight, not finished",
        );
        // Cancellation, exactly as axum performs it.
        drop(fut);
        assert!(
            cell.take_cancel_context().emitted_terminal,
            "a Drop-time emitter inside the handler could not reach the cell — \
             the cancel guard would now emit a second terminal event for the \
             same request",
        );
    }

    /// The caller-addressed name is the FIRST one noted: a routed request
    /// resolves its group, then each target, and the `model` label belongs
    /// to what the client asked for.
    #[tokio::test]
    async fn the_requested_model_is_not_overwritten_by_a_target() {
        scope(Arc::new(RequestAttribution::default()), async {
            note_requested_model("my-group");
            note_requested_model("gpt-4o");
            assert_eq!(current().unwrap().requested_model, "my-group");
        })
        .await;
    }
}
