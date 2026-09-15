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
//! It is kept BESIDE [`Resolved`] rather than inside it because every
//! failed request reads `Resolved` back by value for its metric labels
//! ([`current`]); folding a `Vec<AttemptRecord>` and a `ClientContext` into
//! that clone would put the cancel path's cost on every error path.

use std::future::Future;
use std::sync::{Arc, Mutex};

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

#[derive(Default)]
struct Cell {
    resolved: Resolved,
    cancel: CancelContext,
}

/// The per-request cell. Attempts within a request are sequential, so the
/// lock is uncontended; it exists because the cancel guard may read the
/// cell from a different point in the stack than the writer.
#[derive(Default)]
pub(crate) struct RequestAttribution(Mutex<Cell>);

impl RequestAttribution {
    pub(crate) fn get(&self) -> Resolved {
        self.lock().resolved.clone()
    }

    /// Move the cancel context out of the cell. Only the cancel guard calls
    /// this, once, from `Drop` — taking rather than cloning keeps the
    /// `Vec<AttemptRecord>` off every other read of this cell.
    pub(crate) fn take_cancel_context(&self) -> CancelContext {
        std::mem::take(&mut self.lock().cancel)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Cell> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
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
    CURRENT.scope(Arc::new(RequestAttribution::default()), fut)
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

/// What the current request has resolved, or `None` outside a request.
pub(crate) fn current() -> Option<Resolved> {
    CURRENT.try_with(|a| a.get()).ok()
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

    /// A write with no request scope must not panic — `resolve_provider_key`
    /// also runs from the background health checker.
    #[test]
    fn writes_outside_a_request_are_dropped() {
        note_requested_model("gpt-4o");
        note_target(&model("openai", "gpt-4o"), "pk-1");
        assert!(current().is_none());
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
            assert_eq!(r.requested_model, "my-group");
            assert_eq!(r.provider, "anthropic");
            assert_eq!(r.upstream_model, "claude-3-5-sonnet");
            assert_eq!(r.provider_key_id, "pk-anthropic");
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
