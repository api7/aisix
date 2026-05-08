//! Pre-dispatch quota gate shared by every LLM endpoint.
//!
//! Before this gate landed, only `/v1/chat/completions` ran budget +
//! rate-limit checks (`chat::dispatch`). Every other LLM endpoint —
//! `/v1/embeddings`, `/v1/messages`, `/v1/audio/*`,
//! `/v1/images/generations`, `/v1/responses`, `/v1/rerank`,
//! `/v1/completions`, the `/passthrough/...` family — went straight
//! from auth into the upstream Bridge, silently bypassing both. A
//! customer running the gateway as their org's LLM proxy expected
//! RPM/TPM caps and budget cutoffs to apply uniformly across
//! endpoints; the gap was visible to anyone using `/v1/messages`
//! (Anthropic API-shape) or `/v1/embeddings`. See issue #107.
//!
//! This module hosts the minimum check every non-chat handler now
//! performs: a budget pre-check via cp-api, then a rate-limit
//! reservation. Guardrails are *not* applied here — they need
//! per-handler text extraction (chat reads messages, embeddings reads
//! `input` strings, audio reads transcripts) and that wiring is a
//! larger, separate change. The chat handler still has its own
//! guardrail path; this gate runs in parallel for every other
//! endpoint.
//!
//! Returning a [`Reservation`] (not just a permit) lets the caller
//! commit token usage post-dispatch. Non-chat handlers that don't
//! track upstream tokens uniformly call
//! [`aisix_ratelimit::Reservation::commit_tokens`] with `0`, which
//! still releases the concurrency permit and counts the request
//! against RPM but skips TPM. Future work can plumb per-endpoint
//! token totals through.

use aisix_ratelimit::Reservation;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

/// Apply budget + rate-limit checks for one request. Call this before
/// touching the Bridge in every LLM endpoint handler. The returned
/// [`Reservation`] is alive until the caller commits or drops it; on
/// commit, RPM is finalised and TPM accounted for the supplied total.
pub(crate) async fn enforce<'a>(
    state: &'a ProxyState,
    auth: &AuthenticatedKey,
) -> Result<Reservation<'a, aisix_ratelimit::SystemClock>, ProxyError> {
    // Budget pre-check via cp-api. Mirrors chat::dispatch — the DP no
    // longer owns budget state; cp-api returns a cached/live decision
    // per api_key.
    let decision = state.budgets.check(&auth.entry.id).await;
    if !decision.allowed {
        return Err(ProxyError::BudgetExceeded(
            decision.reason.unwrap_or_else(|| auth.entry.id.clone()),
        ));
    }

    // Rate-limit reservation. The reservation holds a concurrency
    // permit until it's committed (or dropped). Commit at the end of
    // dispatch with whatever token count the upstream returned (0 if
    // the handler doesn't track tokens — RPM still counts).
    let rl_key = auth.entry.id.clone();
    let rl_limits = auth.key().rate_limit.clone().unwrap_or_default();
    state
        .limiter
        .pre_commit(&rl_key, &rl_limits)
        .map_err(ProxyError::from)
}
