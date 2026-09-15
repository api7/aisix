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
    /// When the attempt began, so a cancelled request can say how long the
    /// caller waited on THIS target rather than on the whole request.
    pub started: std::time::Instant,
}

/// What a cancelled request needs to emit its own usage events. See the
/// module docs for why it does not live in [`Resolved`].
#[derive(Default)]
pub(crate) struct CancelContext {
    /// The caller, as the [`ClientContext`] extractor resolved it. `None`
    /// until that extractor has run, which is how the cancel path tells a
    /// request that authenticated from one that hung up during body upload.
    pub client: Option<ClientContext>,
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
    /// Attempts that settled as failures. On the cancel path these never
    /// reach the handler's own `emit_failed_attempts`, so the guard emits
    /// them. A SUCCESSFUL attempt is not kept: its event is built from the
    /// response the handler was still processing, of which the attempt
    /// record holds nothing, so there would be nothing to bill it with —
    /// and skipping it also keeps the happy path allocation-free here.
    pub failed_attempts: Vec<AttemptRecord>,
    /// Whether the handler already emitted a usage event for this request,
    /// and whether one of them was the terminal one.
    ///
    /// The window is microseconds wide — between an emission and the
    /// middleware disarming the guard — but a second, contradicting set of
    /// rows for one request is worse than the row the cancel path would
    /// have added, so the guard defers to whatever the handler managed to
    /// write.
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
        c.api_key_id = api_key_id.to_string();
        c.client = Some(client.clone());
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
/// Clearing `in_flight` is deliberate: the settled attempt now has its own
/// record, so a cancel landing in the gap before the next attempt begins
/// reports no target rather than naming one whose event already went out
/// under the same attempt index.
pub(crate) fn note_attempt_settled(rec: &AttemptRecord) {
    with_cancel(|c| {
        c.in_flight = None;
        if !rec.success {
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

    /// The double-emission interlock rests on one property of tokio's
    /// task-local scope: the value is still installed while the scoped
    /// future is being DROPPED. That is what lets an emitter living inside
    /// the handler — `chat::build_sse_stream`'s `CompleteOnDrop` and its
    /// siblings — record its terminal event on the cell as cancellation
    /// unwinds, so [`CancelContext::emitted_terminal`] is already set by
    /// the time the cancel guard reads it and the guard stays quiet.
    ///
    /// Nothing else pins it. If tokio stopped installing the value during
    /// drop, `note_usage_emitted` would silently become a no-op on exactly
    /// the path the interlock exists for, and one cancelled stream would
    /// report two contradicting terminal rows.
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
