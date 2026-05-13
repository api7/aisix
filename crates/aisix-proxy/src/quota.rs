//! Pre-dispatch quota gate shared by every LLM endpoint.
//!
//! Applies budget + multi-layer rate limiting:
//! 1. Budget pre-check (cp-api cached decision)
//! 2. API-key rate limit (`auth.entry.id`)
//! 3. Model rate limit (`model:<name>`) — when the resolved Model has one
//! 4. Team rate limit (`team:<id>`) — when the ApiKey carries team info
//! 5. Member rate limit (`member:<id>`) — when the ApiKey carries owner info
//!
//! All layers use AND logic — every layer must pass or the request gets
//! 429. The returned [`MultiReservation`] commits token usage to all
//! layers and releases all concurrency permits on drop.

use aisix_core::RateLimit;
use aisix_ratelimit::MultiReservation;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

/// Optional model rate-limit info resolved by the caller before enforce.
pub(crate) struct ModelRateLimit {
    pub name: String,
    pub limits: RateLimit,
}

impl ModelRateLimit {
    /// Build from a resolved model entry. Returns `None` when the model
    /// has no rate limit configured or has an unrestricted one (all fields
    /// are `None`).
    pub fn from_model(model_name: &str, model: &aisix_core::Model) -> Option<Self> {
        model
            .rate_limit
            .as_ref()
            .filter(|rl| !rl.is_unrestricted())
            .map(|rl| Self {
                name: model_name.to_owned(),
                limits: rl.clone(),
            })
    }
}

/// Reserve across all applicable rate-limit layers (api_key, model, team, member).
fn reserve_layers<'a>(
    state: &'a ProxyState,
    auth: &AuthenticatedKey,
    model_rl: Option<ModelRateLimit>,
) -> Result<MultiReservation<'a, aisix_ratelimit::SystemClock>, ProxyError> {
    let mut reservations = Vec::with_capacity(4);

    // Layer 1: API key rate limit.
    let key_limits = auth.key().rate_limit.clone().unwrap_or_default();
    if !key_limits.is_unrestricted() {
        let r = state
            .limiter
            .pre_commit(&auth.entry.id, &key_limits)
            .map_err(ProxyError::from)?;
        reservations.push(r);
    }

    // Layer 2: Model rate limit.
    if let Some(mrl) = model_rl {
        if !mrl.limits.is_unrestricted() {
            let key = format!("model:{}", mrl.name);
            let r = state
                .limiter
                .pre_commit(&key, &mrl.limits)
                .map_err(ProxyError::from)?;
            reservations.push(r);
        }
    }

    // Layer 3: Team rate limit.
    if let (Some(tid), Some(trl)) = (&auth.key().team_id, &auth.key().team_rate_limit) {
        if !tid.is_empty() && !trl.is_unrestricted() {
            let key = format!("team:{tid}");
            let r = state
                .limiter
                .pre_commit(&key, trl)
                .map_err(ProxyError::from)?;
            reservations.push(r);
        }
    }

    // Layer 4: Member/owner rate limit.
    if let (Some(oid), Some(orl)) = (&auth.key().owner_id, &auth.key().owner_rate_limit) {
        if !oid.is_empty() && !orl.is_unrestricted() {
            let key = format!("member:{oid}");
            let r = state
                .limiter
                .pre_commit(&key, orl)
                .map_err(ProxyError::from)?;
            reservations.push(r);
        }
    }

    Ok(MultiReservation::new(reservations))
}

/// Apply budget + multi-layer rate-limit checks for one request.
/// `model_rl` is the resolved Model's rate_limit (if any). Pass `None`
/// for endpoints that don't resolve a model (e.g. passthrough).
pub(crate) async fn enforce<'a>(
    state: &'a ProxyState,
    auth: &AuthenticatedKey,
    model_rl: Option<ModelRateLimit>,
) -> Result<MultiReservation<'a, aisix_ratelimit::SystemClock>, ProxyError> {
    let decision = state.budgets.check(&auth.entry.id).await;
    if !decision.allowed {
        return Err(ProxyError::BudgetExceeded(
            decision.reason.unwrap_or_else(|| auth.entry.id.clone()),
        ));
    }

    reserve_layers(state, auth, model_rl)
}

/// Rate-limit-only enforcement (no budget check). Used by `chat.rs`
/// which handles budget separately.
pub(crate) fn enforce_rate_limit<'a>(
    state: &'a ProxyState,
    auth: &AuthenticatedKey,
    model_rl: Option<ModelRateLimit>,
) -> Result<MultiReservation<'a, aisix_ratelimit::SystemClock>, ProxyError> {
    reserve_layers(state, auth, model_rl)
}
