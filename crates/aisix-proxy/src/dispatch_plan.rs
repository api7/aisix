//! What a caller-addressed model dispatches to, for the endpoints whose
//! handlers walk the attempts themselves (`/v1/chat/completions`,
//! `/v1/messages`, `/v1/messages/count_tokens`, `/v1/responses`).
//!
//! A routing group expands into its members, a semantic router into the
//! target it picks — and when that target is itself a routing group, into
//! that group's members, governed by that group's knobs. [`plan`] is the
//! one entry point for both, so the four endpoints take the same decision;
//! the knob accessors on [`DispatchPlan`] are what each attempt loop reads
//! in place of the addressed entry's own `routing` block.

use std::sync::Arc;

use aisix_core::resource::ResourceEntry;
use aisix_core::{AisixSnapshot, Model};

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::routing::{AttemptModel, RoutingRequest, TimeoutBudget, TimeoutDefaults};
use crate::state::ProxyState;

pub(crate) struct DispatchPlan {
    pub attempts: Vec<AttemptModel>,
    /// The semantic route that served the request (`x-aisix-route`).
    pub semantic_route: Option<String>,
    /// The model group a semantic router's chosen target is.
    pub group: Option<Arc<ResourceEntry<Model>>>,
}

/// Resolve the attempts for `entry`, the model the caller addressed as
/// `model_name`. `prompt` and `session` are read only for a semantic router:
/// the request's classification text, built by the endpoint's protocol
/// reader (`semantic::chat_prompt` and its siblings), and what its session
/// affinity reads off the body.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn plan(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    model_name: &str,
    entry: &ResourceEntry<Model>,
    request_id: &str,
    prompt: impl FnOnce() -> String,
    routing: RoutingRequest<'_>,
    session: crate::semantic::SessionRequest<'_>,
) -> Result<DispatchPlan, ProxyError> {
    if entry.value.is_semantic() {
        let dispatch = crate::semantic::resolve(
            state,
            snapshot,
            entry,
            &prompt(),
            request_id,
            routing,
            session,
        )
        .await?;
        return Ok(DispatchPlan {
            attempts: dispatch.attempts,
            semantic_route: dispatch.route,
            group: dispatch.group,
        });
    }
    let attempts = crate::routing::resolve_attempt_models(
        &state.routing,
        &state.runtime_status,
        &state.pricing,
        snapshot,
        model_name,
        &entry.id,
        &entry.value,
        routing,
    )?;
    Ok(DispatchPlan {
        attempts,
        semantic_route: None,
        group: None,
    })
}

/// Whether `entry` dispatches through targets it selects, so its attempts
/// carry per-target gates and attribution.
pub(crate) fn is_routing_request(entry: &Model) -> bool {
    entry.routing.is_some() || entry.is_semantic()
}

impl DispatchPlan {
    /// The `routing` block whose failover knobs govern the attempts: the
    /// group target's under a semantic router, else the entry's own.
    fn routing<'a>(&'a self, entry: &'a Model) -> Option<&'a aisix_core::Routing> {
        match &self.group {
            Some(group) => group.value.routing.as_ref(),
            None => entry.routing.as_ref(),
        }
    }

    pub fn retry_on_429(&self, entry: &Model) -> bool {
        self.routing(entry)
            .map(|r| r.retry_on_429_or_default())
            .unwrap_or(false)
    }

    pub fn fallback_statuses<'a>(&'a self, entry: &'a Model) -> &'a [u16] {
        self.routing(entry)
            .map(|r| r.fallback_on_statuses_or_default())
            .unwrap_or(&[])
    }

    /// The parent levels of the member → parent → deployment-default
    /// retries chain, nearest first.
    pub fn group_retries(&self, entry: &Model) -> Option<u32> {
        self.group
            .as_ref()
            .and_then(|g| crate::routing::group_retries_of(&g.value))
            .or_else(|| crate::routing::group_retries_of(entry))
    }

    /// Deadlines for `target`: target, group target, addressed entry,
    /// deployment default.
    pub fn timeouts(
        &self,
        target: &Model,
        entry: &Model,
        defaults: TimeoutDefaults,
    ) -> TimeoutBudget {
        match &self.group {
            Some(group) => {
                crate::routing::effective_timeouts_nested(target, &[&group.value, entry], defaults)
            }
            None => crate::routing::effective_timeouts(target, Some(entry), defaults),
        }
    }

    /// The parent identity a member's per-target rate-limit pass matches
    /// policies against; `None` when `entry` dispatches to itself.
    pub fn quota_parent<'a>(
        &'a self,
        entry: &'a ResourceEntry<Model>,
    ) -> Option<crate::quota::RoutingParent<'a>> {
        is_routing_request(&entry.value).then(|| crate::quota::RoutingParent {
            name: &entry.value.display_name,
            entry_id: &entry.id,
            group: self.group.as_ref().map(|g| crate::quota::RoutingGroup {
                name: &g.value.display_name,
                entry_id: &g.id,
            }),
        })
    }

    /// Reserve the group target's own rate-limit layers once for the
    /// request ([`crate::quota::reserve_semantic_group`]) into `request`,
    /// the request gate's reservation they extend, so they are committed and
    /// refunded with it. A refusal turns the request away at its gate:
    /// `request` is refunded too. Nothing to do without a group target.
    pub async fn reserve_group(
        &self,
        state: &ProxyState,
        snapshot: &AisixSnapshot,
        auth: &AuthenticatedKey,
        request: &mut aisix_ratelimit::MultiReservation,
    ) -> Result<(), ProxyError> {
        let Some(group) = &self.group else {
            return Ok(());
        };
        match crate::quota::reserve_semantic_group(state, snapshot, auth, &group.id, &group.value)
            .await
        {
            Ok(reserved) => {
                request.merge(reserved);
                Ok(())
            }
            Err(e) => {
                std::mem::take(request).refund().await;
                Err(e)
            }
        }
    }
}

/// Set `x-aisix-route` to the semantic route that served the request.
/// Debug-only, like every `x-aisix-*` routing header: call it only under
/// `proxy.expose_debug_headers`.
pub(crate) fn insert_route_header(headers: &mut axum::http::HeaderMap, route: Option<&str>) {
    let Some(route) = route else {
        return;
    };
    // `HeaderValue::try_from` rejects CR/LF and non-visible ASCII
    // (RFC 7230); surface the rejection so the route can be renamed.
    match axum::http::HeaderValue::try_from(route) {
        Ok(v) => {
            headers.insert("x-aisix-route", v);
        }
        Err(err) => {
            tracing::warn!(
                route_name = %route,
                error = %err,
                "semantic route name is not a valid HTTP header value; \
                 omitting x-aisix-route — rename the route to use only \
                 visible ASCII (no CR/LF, no non-ASCII characters)"
            );
        }
    }
}
