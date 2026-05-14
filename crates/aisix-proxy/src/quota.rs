//! Pre-dispatch quota gate shared by every LLM endpoint.
//!
//! Applies budget + multi-layer rate limiting:
//! 1. Budget pre-check (cp-api cached decision)
//! 2. API-key inline rate limit (`auth.entry.id`)
//! 3. Model inline rate limit (`model:<name>`) — when the resolved Model has one
//! 4. Policy-based rate limits — looked up from the snapshot's
//!    `rate_limit_policies` table, matched by scope (api_key/model/team/member)
//!
//! All layers use AND logic — every layer must pass or the request gets
//! 429. The returned [`MultiReservation`] commits token usage to all
//! layers and releases all concurrency permits on drop.

use aisix_core::models::RateLimitPolicy;
use aisix_core::RateLimit;
use aisix_ratelimit::MultiReservation;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

/// Optional model rate-limit info resolved by the caller before enforce.
pub(crate) struct ModelRateLimit {
    pub name: String,
    pub entry_id: String,
    pub limits: Option<RateLimit>,
}

impl ModelRateLimit {
    /// Build from a resolved model entry. Always returns a
    /// `ModelRateLimit` carrying the model identity (name + entry ID)
    /// needed for model-scope policy matching. The inline rate limit
    /// is `None` when the model has no configured limit.
    pub fn from_model(model_name: &str, model_entry_id: &str, model: &aisix_core::Model) -> Self {
        let limits = model
            .rate_limit
            .as_ref()
            .filter(|rl| !rl.is_unrestricted())
            .cloned();
        Self {
            name: model_name.to_owned(),
            entry_id: model_entry_id.to_owned(),
            limits,
        }
    }
}

fn policy_to_rate_limit(policy: &RateLimitPolicy) -> RateLimit {
    let mut rl = RateLimit::default();
    match policy.window.as_str() {
        "second" => {
            rl.rpm = policy.max_requests.map(|r| r.saturating_mul(60));
            rl.tpm = policy.max_tokens.map(|t| t.saturating_mul(60));
        }
        "minute" => {
            rl.rpm = policy.max_requests;
            rl.tpm = policy.max_tokens;
        }
        "hour" => {
            rl.rpd = policy.max_requests.map(|r| r.saturating_mul(24));
            rl.tpd = policy.max_tokens.map(|t| t.saturating_mul(24));
        }
        _ => {}
    }
    rl
}

/// Reserve across all applicable rate-limit layers (api_key, model, policies).
fn reserve_layers<'a>(
    state: &'a ProxyState,
    auth: &AuthenticatedKey,
    model_rl: Option<&ModelRateLimit>,
) -> Result<MultiReservation<'a, aisix_ratelimit::SystemClock>, ProxyError> {
    let mut reservations = Vec::with_capacity(8);

    // Layer 1: API key inline rate limit.
    let key_limits = auth.key().rate_limit.clone().unwrap_or_default();
    if !key_limits.is_unrestricted() {
        let r = state
            .limiter
            .pre_commit(&auth.entry.id, &key_limits)
            .map_err(ProxyError::from)?;
        reservations.push(r);
    }

    // Layer 2: Model inline rate limit.
    if let Some(mrl) = model_rl {
        if let Some(ref limits) = mrl.limits {
            let key = format!("model:{}", mrl.name);
            let r = state
                .limiter
                .pre_commit(&key, limits)
                .map_err(ProxyError::from)?;
            reservations.push(r);
        }
    }

    // Layer 3+: Rate limit policies from snapshot.
    let snap = state.snapshot.load();
    for entry in snap.rate_limit_policies.entries() {
        let policy = &entry.value;
        let applies = match policy.scope.as_str() {
            "api_key" => policy.scope_ref == auth.entry.id,
            "model" => model_rl.is_some_and(|m| policy.scope_ref == m.entry_id),
            "team" => auth.key().team_id.as_deref() == Some(policy.scope_ref.as_str()),
            "member" => auth.key().owner_id.as_deref() == Some(policy.scope_ref.as_str()),
            _ => false,
        };
        if !applies {
            continue;
        }
        let rl = policy_to_rate_limit(policy);
        if rl.is_unrestricted() {
            continue;
        }
        let bucket_key = format!("policy:{}:{}:{}", policy.scope, policy.scope_ref, entry.id);
        let r = state
            .limiter
            .pre_commit(&bucket_key, &rl)
            .map_err(ProxyError::from)?;
        reservations.push(r);
    }

    Ok(MultiReservation::new(reservations))
}

/// Apply budget + multi-layer rate-limit checks for one request.
/// `model_rl` carries the resolved model identity for policy matching
/// and optional inline limits. Pass `None` only for endpoints that
/// don't resolve a model (e.g. passthrough).
pub(crate) async fn enforce<'a>(
    state: &'a ProxyState,
    auth: &AuthenticatedKey,
    model_rl: Option<&ModelRateLimit>,
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
    model_rl: Option<&ModelRateLimit>,
) -> Result<MultiReservation<'a, aisix_ratelimit::SystemClock>, ProxyError> {
    reserve_layers(state, auth, model_rl)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy(window: &str, max_req: Option<u64>, max_tok: Option<u64>) -> RateLimitPolicy {
        serde_json::from_value(serde_json::json!({
            "name": "test",
            "scope": "team",
            "scope_ref": "ref",
            "window": window,
            "max_requests": max_req,
            "max_tokens": max_tok,
        }))
        .unwrap()
    }

    #[test]
    fn minute_maps_to_rpm_tpm() {
        let rl = policy_to_rate_limit(&make_policy("minute", Some(100), Some(50000)));
        assert_eq!(rl.rpm, Some(100));
        assert_eq!(rl.tpm, Some(50000));
        assert!(rl.rpd.is_none());
        assert!(rl.tpd.is_none());
    }

    #[test]
    fn second_scales_to_per_minute() {
        let rl = policy_to_rate_limit(&make_policy("second", Some(10), Some(1000)));
        assert_eq!(rl.rpm, Some(600));
        assert_eq!(rl.tpm, Some(60000));
    }

    #[test]
    fn hour_scales_to_per_day() {
        let rl = policy_to_rate_limit(&make_policy("hour", Some(1000), Some(500000)));
        assert_eq!(rl.rpd, Some(24000));
        assert_eq!(rl.tpd, Some(12000000));
        assert!(rl.rpm.is_none());
    }

    #[test]
    fn unknown_window_produces_unrestricted() {
        let rl = policy_to_rate_limit(&make_policy("week", Some(100), Some(100)));
        assert!(rl.is_unrestricted());
    }

    #[test]
    fn partial_fields_only_set_relevant_dimension() {
        let rl = policy_to_rate_limit(&make_policy("minute", Some(60), None));
        assert_eq!(rl.rpm, Some(60));
        assert!(rl.tpm.is_none());
    }
}
