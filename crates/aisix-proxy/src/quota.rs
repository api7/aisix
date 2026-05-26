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
            // Pre-fix (api7/AISIX-Cloud#426): `rl.rpm = max * 60` — a
            // 5/second policy was upscaled to 300/minute, allowing
            // 60× bursts past the operator-declared cap inside any
            // single 1-second window.
            // Post-fix: native rps via `FixedWindowCounter::new(1)`.
            //
            // Tokens (`tps`) intentionally NOT wired — at a 1s window
            // the post-deduct add pattern races the window roll-over
            // and silently grants freebies on every cross-boundary
            // request. Tracked separately in api7/ai-gateway#396.
            rl.rps = policy.max_requests;
            // Audit M1 (#399): warn loudly when an operator set
            // `max_tokens` on a sub-minute window. Without the warn,
            // the policy looks accepted at cp-api but the token cap
            // is silently inert until ai-gateway#396 lands.
            if policy.max_tokens.is_some() {
                tracing::warn!(
                    policy_name = %policy.name,
                    window = %policy.window,
                    "max_tokens ignored: per-second token-rate counter not yet implemented; \
                     see api7/ai-gateway#396"
                );
            }
        }
        "minute" => {
            rl.rpm = policy.max_requests;
            rl.tpm = policy.max_tokens;
        }
        "hour" => {
            // Pre-fix (api7/AISIX-Cloud#426): `rl.rpd = max * 24` —
            // a 1000/hour policy was upscaled to 24000/day, allowing
            // the entire hourly cap to be burned in any single hour
            // with no enforcement (24× exploit shape, slower-window
            // counterpart of the "second" bug).
            // Post-fix: native rph via `FixedWindowCounter::new(3600)`.
            //
            // Tokens (`tph`) intentionally NOT wired — see ai-gateway#396.
            rl.rph = policy.max_requests;
            if policy.max_tokens.is_some() {
                tracing::warn!(
                    policy_name = %policy.name,
                    window = %policy.window,
                    "max_tokens ignored: per-hour token-rate counter not yet implemented; \
                     see api7/ai-gateway#396"
                );
            }
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
    let budget_labels = aisix_obs::BudgetLabels {
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref().unwrap_or("unknown"),
        owner_id: auth.key().owner_id.as_deref().unwrap_or("unknown"),
    };
    if let Some(budget) = decision.budget.as_ref() {
        state.metrics.set_budget_gauges(
            budget_labels,
            aisix_obs::BudgetGauges {
                limit_usd: budget.limit_usd,
                spent_usd: budget.spent_usd,
                remaining_usd: budget.remaining_usd,
                reset_seconds: budget.reset_seconds,
            },
        );
    } else {
        state.metrics.clear_budget_gauges(budget_labels);
    }
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

    // Regression guard for api7/AISIX-Cloud#426. Pre-fix these tests
    // asserted the BUG: `second` → `rpm = max * 60` and `hour` →
    // `rpd = max * 24`. The upscaling allowed 60× and 24× bursts past
    // the operator-declared cap. Post-fix asserts the new contract:
    // `second` produces a native rps and `hour` produces a native rph.
    #[test]
    fn second_maps_to_rps_not_rpm_times_sixty() {
        let rl = policy_to_rate_limit(&make_policy("second", Some(10), Some(1000)));
        assert_eq!(
            rl.rps,
            Some(10),
            "second window must populate rps natively, not rpm*60"
        );
        // No upscale into rpm/tpm — that was the #426 bug.
        assert!(
            rl.rpm.is_none(),
            "second window MUST NOT populate rpm (would 60× the cap)"
        );
        assert!(
            rl.tpm.is_none(),
            "second window MUST NOT populate tpm (would 60× the cap)"
        );
        // tps intentionally deferred — see ai-gateway#396.
    }

    #[test]
    fn hour_maps_to_rph_not_rpd_times_twentyfour() {
        let rl = policy_to_rate_limit(&make_policy("hour", Some(1000), Some(500000)));
        assert_eq!(
            rl.rph,
            Some(1000),
            "hour window must populate rph natively, not rpd*24"
        );
        // No upscale into rpd/tpd — that was the parallel #426 bug.
        assert!(
            rl.rpd.is_none(),
            "hour window MUST NOT populate rpd (would 24× the cap)"
        );
        assert!(
            rl.tpd.is_none(),
            "hour window MUST NOT populate tpd (would 24× the cap)"
        );
        // tph intentionally deferred — see ai-gateway#396.
    }

    #[test]
    fn minute_window_unchanged_by_426() {
        // Regression guard: the minute branch was always correct
        // (rpm/tpm map 1:1). #426 must not have touched it.
        let rl = policy_to_rate_limit(&make_policy("minute", Some(60), Some(30000)));
        assert_eq!(rl.rpm, Some(60));
        assert_eq!(rl.tpm, Some(30000));
        assert!(rl.rps.is_none());
        assert!(rl.rph.is_none());
        assert!(rl.rpd.is_none());
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
