//! The usage telemetry of a request whose caller hung up before the
//! response head was written (AISIX-Cloud#1571).
//!
//! Every endpoint emits its usage events from the tail of its own handler.
//! When the downstream client disconnects first, axum drops the handler
//! future and none of that code runs — the emission is continuation code,
//! and a dropped future has no continuation. The request then reached an
//! upstream, possibly spent a provider's tokens, and left no row in the
//! usage log at all, which is exactly the case an operator most needs to
//! see: the usual reason a caller gives up is a long time to first token.
//!
//! So the emission moves to the one thing cancellation cannot skip: `Drop`.
//! `ClientCancelGuard` hands what the request managed to resolve — see
//! [`crate::attribution`] — to [`emit`], which writes the same events the
//! handler would have:
//!
//! - one non-terminal event per attempt that had already FAILED, byte-for-
//!   byte what `emit_failed_attempts` would have written for it;
//! - one terminal event for the request itself, `499` /
//!   `client_disconnected`, zero tokens and zero cost, naming the target
//!   that was in flight when the caller went away.
//!
//! Exactly one of the two paths ever fires for a given request: the guard
//! emits only while armed, and it is disarmed the moment the inner service
//! yields a response.
//!
//! Two things this path deliberately does NOT carry. A cancelled request's
//! events have no guardrail attribution (`applied_guardrails`,
//! `guardrail_enforced_hits`, `guardrail_scores`, the bypass reason): those
//! come off a chain each handler resolves for itself, with no chokepoint a
//! guard could read, and ten opt-in call sites is the drift this design
//! exists to avoid. And an attempt that had already SUCCEEDED is not
//! re-emitted: its token counts live in the handler's response processing,
//! not in the attempt record, so there is nothing to bill it with.

use aisix_obs::UsageEvent;

use crate::attribution::{CancelContext, Resolved};
use crate::client_ip::ClientContext;
use crate::state::ProxyState;
use crate::usage_attr::{self, ResolvedPk};

/// `error_message` of the terminal event, and of the access-log line the
/// guard writes beside it — one sentence, the same on both, so a row in the
/// usage log can be joined to the line that explains it.
pub(crate) const CANCELLED_BEFORE_HEAD: &str =
    "client closed the request before the response head was written";

/// `error_message` of the mid-stream shape: the response head went out and
/// the caller left while the body was still streaming. Stamped at the
/// emission chokepoint rather than by each streaming family, so both 499
/// shapes speak one vocabulary (see [`usage_attr::emit_usage`]).
pub(crate) const CANCELLED_MID_STREAM: &str =
    "client closed the request while the response was streaming";

/// Emit a cancelled request's usage events. Called once, from
/// `ClientCancelGuard::drop`.
///
/// Silent unless the request got far enough to be worth a row: it must have
/// authenticated AND named a model the gateway resolved. A caller that hangs
/// up while its body is still uploading has done neither — the auth and
/// `ClientContext` extractors run before the body one, so the api_key alone
/// would let an aborted upload mint a row with no model and no cost, which
/// is both unattributable and a cheap way to fill the usage log. That leaves
/// it where the pre-dispatch rejections in [`crate::reject`] already sit,
/// which emit no usage event either. Silent, too, for a route whose surface
/// reports no cancel (see [`crate::operation::surface_for_endpoint`]).
pub(crate) fn emit(
    state: &ProxyState,
    endpoint: &'static str,
    request_id: &str,
    resolved: &Resolved,
    ctx: CancelContext,
) {
    let Some(surface) = crate::operation::surface_for_endpoint(endpoint) else {
        return;
    };
    let Some(client) = ctx.client.as_ref() else {
        return;
    };
    if ctx.api_key_id.is_empty() || resolved.requested_model.is_empty() {
        return;
    }
    let snap = state.snapshot.load();
    let inbound_protocol = crate::inbound_protocol_for_endpoint(endpoint);

    // The attempts that had already failed. Non-terminal, exactly as the
    // handler's own emitter marks them — the terminal event below is what
    // ends this request. Skipped whole if the handler had already started
    // emitting: whatever it wrote stands, rather than being doubled.
    for rec in ctx.failed_attempts.iter().filter(|_| !ctx.emitted_any) {
        let pk = ResolvedPk::resolve(&snap, &rec.provider_key_id);
        let event = UsageEvent {
            // NO-GUARDRAIL-CHAIN: this emitter runs from `Drop` with no
            // handler. `applied_guardrails`, the enforced hits, the scores
            // and the bypass reason all come off a chain each handler
            // resolves for itself, inside a crate this side cannot call
            // back into — there is no chokepoint a guard could read, and
            // ten opt-in call sites is the drift this whole design exists
            // to avoid. A cancelled request therefore reports no guardrail
            // attribution at all rather than a wrong one.
            model_id: rec.target_model_id.clone(),
            status_code: rec.status,
            upstream_latency_ms: rec.latency_ms,
            attempt_index: rec.index,
            attempt_kind: rec.kind.to_string(),
            attempt_model: rec.target_model.clone(),
            error_class: rec.error_class.clone(),
            error_message: rec.error_message.clone(),
            ..base_event(request_id, resolved, &ctx, client, inbound_protocol, &pk)
        };
        emit_one(
            state,
            &snap,
            surface,
            event,
            &pk,
            client,
            false,
            rec.dispatched,
        );
    }

    // The request's own terminal event — unless the handler got its own
    // terminal event out before the future was dropped.
    if ctx.emitted_terminal {
        return;
    }
    // The attempt this event speaks for: the one in flight, else the one
    // that WON and was still being processed when the caller left. A failed
    // attempt is never chosen — the loop above already emitted its event,
    // under this same index.
    let attempt = ctx
        .in_flight
        .as_ref()
        .map(|a| Attempt {
            index: a.index,
            kind: a.kind,
            target_model: a.target_model.as_str(),
            model_id: a.model_id.as_str(),
            // `begin_attempt` publishes an attempt BEFORE the target's own
            // rate-limit reservation and before the bridge assembles the
            // request, so an attempt in flight has not necessarily reached
            // a provider — `AttemptRecord::dispatched` records exactly that
            // case as false. Claiming it would make `emit_usage` fabricate
            // an upstream span for a call nobody made, and there is no
            // latency to report that would mean what an attempt's normally
            // means.
            dispatched: false,
            latency_ms: 0,
        })
        .or_else(|| {
            ctx.won.as_ref().map(|w| Attempt {
                index: w.index,
                kind: w.kind,
                target_model: w.target_model.as_str(),
                model_id: w.model_id.as_str(),
                // This one did reach the provider, and its own measured
                // duration is what the winner's event would have carried.
                dispatched: true,
                latency_ms: w.latency_ms,
            })
        });
    let pk = ResolvedPk::resolve(&snap, &resolved.provider_key_id);
    let event = UsageEvent {
        // NO-GUARDRAIL-CHAIN: this emitter runs from `Drop` with no
        // handler. `applied_guardrails`, the enforced hits, the scores
        // and the bypass reason all come off a chain each handler
        // resolves for itself, inside a crate this side cannot call
        // back into — there is no chokepoint a guard could read, and
        // ten opt-in call sites is the drift this whole design exists
        // to avoid. A cancelled request therefore reports no guardrail
        // attribution at all rather than a wrong one.
        // The target this request had committed to, in narrowing order: the
        // attempt above, else the last attempt that settled (the cancel
        // landed in the retry backoff between two failures), else the entry
        // the caller addressed, but only if that entry dispatches itself. A
        // routing group's own id prices nothing, so it stays empty there,
        // the same convention a `model_not_found` event uses.
        model_id: attempt
            .as_ref()
            .map(|a| a.model_id.to_string())
            .filter(|id| !id.is_empty())
            .or_else(|| Some(ctx.last_settled_model_id.clone()).filter(|id| !id.is_empty()))
            .unwrap_or_else(|| ctx.entry_model_id.clone()),
        status_code: crate::CLIENT_CLOSED_REQUEST,
        upstream_latency_ms: attempt.as_ref().map(|a| a.latency_ms).unwrap_or(0),
        attempt_index: attempt.as_ref().map(|a| a.index).unwrap_or(0),
        attempt_kind: attempt
            .as_ref()
            .map(|a| a.kind.to_string())
            .unwrap_or_default(),
        attempt_model: attempt
            .as_ref()
            .map(|a| a.target_model.to_string())
            .unwrap_or_default(),
        error_class: crate::CLIENT_DISCONNECTED_KIND.to_string(),
        error_message: CANCELLED_BEFORE_HEAD.to_string(),
        ..base_event(request_id, resolved, &ctx, client, inbound_protocol, &pk)
    };
    let dispatched = attempt.as_ref().is_some_and(|a| a.dispatched);
    emit_one(state, &snap, surface, event, &pk, client, true, dispatched);
}

/// The attempt the terminal event speaks for, once the two shapes that can
/// supply one are collapsed.
struct Attempt<'a> {
    index: u32,
    kind: &'static str,
    target_model: &'a str,
    model_id: &'a str,
    /// Whether this attempt is KNOWN to have reached a provider. See the
    /// two construction sites — they answer it differently, and that is the
    /// whole reason this type exists rather than a bare `Option<&…>`.
    dispatched: bool,
    latency_ms: u32,
}

/// The fields every event on this path shares: who called, what they asked
/// for, and the ProviderKey attribution tags of the key it names.
fn base_event(
    request_id: &str,
    resolved: &Resolved,
    ctx: &CancelContext,
    client: &ClientContext,
    inbound_protocol: &'static str,
    pk: &ResolvedPk<'_>,
) -> UsageEvent {
    let tags = pk.telemetry_tags();
    let mut event = UsageEvent {
        // NO-GUARDRAIL-CHAIN: this emitter runs from `Drop` with no
        // handler. `applied_guardrails`, the enforced hits, the scores
        // and the bypass reason all come off a chain each handler
        // resolves for itself, inside a crate this side cannot call
        // back into — there is no chokepoint a guard could read, and
        // ten opt-in call sites is the drift this whole design exists
        // to avoid. A cancelled request therefore reports no guardrail
        // attribution at all rather than a wrong one.
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        api_key_id: ctx.api_key_id.clone(),
        // The name the CALLER addressed — the group for a routing request,
        // matching every other event of the same request.
        requested_model: resolved.requested_model.clone(),
        inbound_protocol: inbound_protocol.to_string(),
        provider_kind: crate::chat::sanitize_tag(
            tags.kind.map(|k| k.as_str().to_owned()).unwrap_or_default(),
        ),
        provider_featured: tags.featured,
        branded_provider: crate::chat::sanitize_tag(tags.branded_provider.unwrap_or_default()),
        pk_label: crate::chat::sanitize_tag(tags.pk_label.unwrap_or_default()),
        byo_label: crate::chat::sanitize_tag(tags.byo_label.unwrap_or_default()),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        ..Default::default()
    };
    usage_attr::apply_caller_identity(
        &mut event,
        client.jwt.as_ref(),
        client.caller.user_id.as_deref(),
        client.caller.user_name.as_deref(),
    );
    event
}

/// Emit one built event.
///
/// Takes the caller's `snap` rather than loading its own: the ProviderKey
/// tags and the model label on a single event have to come from ONE
/// generation, and the two events of one cancelled request from the same
/// one — a config update landing between two loads would otherwise make
/// them disagree.
#[allow(clippy::too_many_arguments)]
fn emit_one(
    state: &ProxyState,
    snap: &aisix_core::AisixSnapshot,
    surface: crate::operation::Surface,
    event: UsageEvent,
    pk: &ResolvedPk<'_>,
    client: &ClientContext,
    terminal: bool,
    dispatched: bool,
) {
    let model = usage_attr::usage_event_model_label(snap, &event.requested_model).into_owned();
    usage_attr::emit_usage(
        state,
        snap,
        surface,
        event,
        usage_attr::usage_event_labels(&model, pk),
        None,
        client.trace.as_ref(),
        terminal,
        dispatched,
    );
}
