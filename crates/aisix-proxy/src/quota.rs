//! Pre-dispatch quota gate shared by every LLM endpoint.
//!
//! Applies budget + multi-layer rate limiting:
//! 1. Budget pre-check (cp-api cached decision)
//! 2. API-key inline rate limit (`auth.entry.id`)
//! 3. Model inline rate limit (`model:<name>`) — when the resolved Model has one
//! 4. MCP-server rate limit (`mcp:<api_key_id>:<server>`) — on an MCP
//!    `tools/call`, the key's `mcp_rate_limits` entry for the server the
//!    call targets
//! 5. Policy-based rate limits — looked up from the snapshot's
//!    `rate_limit_policies` table. Classic rows match by scope
//!    (api_key/model/team/member/team_member); `team_member` is a
//!    per-member default for a team: it matches every key in the team
//!    but buckets the counter per `user_id`, so each member gets an
//!    independent identical quota (vs. `team`, one shared bucket).
//!    Conditional rows (AISIX-Cloud#892) match by their `conditions`
//!    tree and bucket by `group_by` — see [`match_policy_layer`] for
//!    the phase split that decides whether a row reserves at the
//!    request gate or per routing target.
//!
//! All layers use AND logic — every layer must pass or the request gets
//! 429. The returned [`MultiReservation`] commits token usage to all
//! layers and releases all concurrency permits on drop.

use aisix_core::models::{
    ConditionInput, GroupByDimension, PolicyScope, PolicyWindow, RateLimitPolicy,
};
use aisix_core::RateLimit;
use aisix_ratelimit::MultiReservation;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

/// Optional model rate-limit info resolved by the caller before enforce.
pub(crate) struct ModelRateLimit {
    pub name: String,
    /// The resolved entry's `display_name` — the bucket identity. Equal
    /// to `name` everywhere except a wildcard-served alias, where every
    /// caller-minted concrete name must land in the wildcard row's ONE
    /// bucket: keying on `name` there let any caller mint fresh
    /// full-size buckets per suffix, unbounded-multiplying the declared
    /// cap (the passthrough gate already keyed on `display_name`).
    pub bucket_name: String,
    pub entry_id: String,
    pub limits: Option<RateLimit>,
    /// The model's `provider` — the value of the `provider` condition
    /// dimension. `None` on routing/ensemble/semantic parents.
    pub provider: Option<String>,
    /// Whether the entry is a virtual parent (routing / ensemble /
    /// semantic): its concrete targets reserve their own model layers
    /// per attempt, so the request gate defers model-property
    /// conditional policies to the per-target phase.
    pub routing_parent: bool,
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
            bucket_name: model.display_name.clone(),
            entry_id: model_entry_id.to_owned(),
            limits,
            provider: model.provider.clone(),
            routing_parent: model.is_routing() || model.is_ensemble() || model.is_semantic(),
        }
    }
}

/// Identity of the caller-addressed virtual parent (routing group /
/// ensemble / semantic router), forwarded by the dispatch loops into the
/// per-target condition input so `model` / `model_name` leaves match the
/// parent as well as the concrete target (AISIX-Cloud#1267).
#[derive(Clone, Copy)]
pub(crate) struct RoutingParent<'a> {
    /// The parent Model's `display_name` (the alias the caller sent).
    pub name: &'a str,
    /// The parent's resource entry id.
    pub entry_id: &'a str,
}

/// The request's condition-dimension values at this gate point. Model
/// dimensions are absent when no model is resolved (MCP, A2A) — leaves
/// on them evaluate false while OR siblings can still match.
/// `routing_parent` is set only at the per-target gate of a routing
/// dispatch; the request gate passes `None` (there the primary values
/// already ARE the requested entry).
fn condition_input<'a>(
    auth: &'a AuthenticatedKey,
    model_rl: Option<&'a ModelRateLimit>,
    routing_parent: Option<RoutingParent<'a>>,
) -> ConditionInput<'a> {
    ConditionInput {
        team: auth.key().team_id.as_deref(),
        member: auth.key().user_id.as_deref(),
        api_key: Some(&auth.entry.id),
        model: model_rl.map(|m| m.entry_id.as_str()),
        model_name: model_rl.map(|m| m.name.as_str()),
        provider: model_rl.and_then(|m| m.provider.as_deref()),
        routing_parent_model: routing_parent.map(|p| p.entry_id),
        routing_parent_model_name: routing_parent.map(|p| p.name),
    }
}

/// Which scan of the policy table this is. Every policy reserves at
/// exactly one phase per attempt:
///
/// - classic rows: `model` scope rows follow the model (per target on a
///   routing dispatch), every other scope reserves at the request gate;
/// - conditional rows: rows referencing a model property reserve where
///   the concrete model is known — the request gate for a direct
///   dispatch, the per-target gate when the request entry is a
///   routing/ensemble parent (`defer_model_properties`); rows touching
///   no model property reserve once at the request gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolicyPhase {
    Request { defer_model_properties: bool },
    ModelTarget,
}

/// One policy layer this request must reserve at the current phase.
struct PolicyLayer {
    bucket_key: String,
    limits: RateLimit,
}

/// Decide whether `policy` applies at this phase and build its bucket
/// key + effective limits. `None` = not applicable here (wrong phase,
/// unmatched, suspended is checked by the caller, or a `group_by`
/// dimension the request does not carry).
fn match_policy_layer(
    policy: &RateLimitPolicy,
    policy_entry_id: &str,
    input: &ConditionInput<'_>,
    phase: PolicyPhase,
) -> Option<PolicyLayer> {
    if policy.is_conditional() {
        return match_conditional_layer(policy, policy_entry_id, input, phase);
    }
    // —— classic form ——
    let (Some(scope), Some(scope_ref)) = (policy.scope, policy.scope_ref.as_deref()) else {
        // Load-time validation rejects formless rows; nothing to enforce.
        return None;
    };
    if matches!(phase, PolicyPhase::ModelTarget) && scope != PolicyScope::Model {
        // Request-level scopes were reserved at the request gate; only
        // the model scope follows the target.
        return None;
    }
    let applies = match scope {
        PolicyScope::ApiKey => input.api_key == Some(scope_ref),
        PolicyScope::Model => input.model == Some(scope_ref),
        PolicyScope::Team => input.team == Some(scope_ref),
        PolicyScope::Member => input.member == Some(scope_ref),
        // Per-member default for a team: matches every key whose
        // team_id == scope_ref, but only when the key carries a
        // user_id (the bucket is keyed per member below).
        PolicyScope::TeamMember => input.team == Some(scope_ref) && input.member.is_some(),
    };
    if !applies {
        return None;
    }
    let limits = classic_rate_limit(policy)?;
    if limits.is_unrestricted() {
        return None;
    }
    // Most scopes share one counter across every key the policy matches
    // (`policy:<scope>:<scope_ref>:<id>`). `team_member` appends the
    // request's `user_id` so each member of the team counts against an
    // independent identical bucket (LiteLLM's `{team_id}:{user_id}`).
    let mut bucket_key = format!("policy:{scope}:{scope_ref}:{policy_entry_id}");
    if scope == PolicyScope::TeamMember {
        if let Some(member) = input.member {
            bucket_key = format!("{bucket_key}:{member}");
        }
    }
    Some(PolicyLayer { bucket_key, limits })
}

fn match_conditional_layer(
    policy: &RateLimitPolicy,
    policy_entry_id: &str,
    input: &ConditionInput<'_>,
    phase: PolicyPhase,
) -> Option<PolicyLayer> {
    let follows_model = policy.references_model_property();
    let due_here = match phase {
        PolicyPhase::Request {
            defer_model_properties,
        } => !(follows_model && defer_model_properties),
        PolicyPhase::ModelTarget => follows_model,
    };
    if !due_here {
        return None;
    }
    if !aisix_core::models::eval_condition_nodes(
        policy.conditions.as_deref().unwrap_or_default(),
        input,
    ) {
        return None;
    }
    // Bucket: `policy:v2:<policy_id>` plus one `:<dim>=<value>` segment
    // per `group_by` dimension, in canonical order so the declared
    // order never changes the bucket identity. A matched request
    // missing a split dimension is not subject to the policy (mirrors
    // `team_member` only applying to keys carrying a `user_id`).
    let group_by = policy.group_by.as_deref().unwrap_or_default();
    let mut bucket_key = format!("policy:v2:{policy_entry_id}");
    for dim in GroupByDimension::CANONICAL_ORDER {
        if !group_by.contains(&dim) {
            continue;
        }
        let value = input.get_group_by(dim)?;
        bucket_key = format!("{bucket_key}:{dim}={}", escape_bucket_segment(value));
    }
    let limits = policy.limits.clone()?;
    if limits.is_unrestricted() {
        return None;
    }
    Some(PolicyLayer { bucket_key, limits })
}

/// Escape a `group_by` segment value for the bucket key. CP-written
/// values are UUIDs/catalog ids and pass through untouched; the file
/// source lets operators pick arbitrary team/member id strings, where
/// an embedded `:` or `=` could otherwise alias two distinct value
/// tuples onto one bucket (`team="t:member=x"` vs `member="x"`).
fn escape_bucket_segment(value: &str) -> std::borrow::Cow<'_, str> {
    if value.contains([':', '=', '%']) {
        std::borrow::Cow::Owned(
            value
                .replace('%', "%25")
                .replace(':', "%3A")
                .replace('=', "%3D"),
        )
    } else {
        std::borrow::Cow::Borrowed(value)
    }
}

/// Convert a classic row's `window` + `max_*` into the 7-field
/// [`RateLimit`]. `None` when the row carries no window (formless rows
/// are rejected at load; this is the total fallback).
fn classic_rate_limit(policy: &RateLimitPolicy) -> Option<RateLimit> {
    let mut rl = RateLimit::default();
    match policy.window? {
        PolicyWindow::Second => {
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
                    window = "second",
                    "max_tokens ignored: per-second token-rate counter not yet implemented; \
                     see api7/ai-gateway#396"
                );
            }
        }
        PolicyWindow::Minute => {
            rl.rpm = policy.max_requests;
            rl.tpm = policy.max_tokens;
        }
        PolicyWindow::Hour => {
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
                    window = "hour",
                    "max_tokens ignored: per-hour token-rate counter not yet implemented; \
                     see api7/ai-gateway#396"
                );
            }
        }
        PolicyWindow::Day => {
            // Both counters are native (`token_dims`, `KeyState`, and
            // the Redis scripts handle `tpd` already) — day is the one
            // window where a token cap on a team-capable policy was
            // previously inexpressible (#771).
            rl.rpd = policy.max_requests;
            rl.tpd = policy.max_tokens;
        }
    }
    Some(rl)
}

/// Reserve across all applicable rate-limit layers (api_key, model,
/// mcp_server, policies). `mcp_server` is the MCP server an in-flight
/// `tools/call` targets; `None` for every non-MCP endpoint.
async fn reserve_layers(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    model_rl: Option<&ModelRateLimit>,
    mcp_server: Option<&str>,
) -> Result<MultiReservation, ProxyError> {
    // Starts empty so the common no-limits request never allocates;
    // the first reservation (if any) grows it on demand.
    let mut reservations = Vec::new();

    // Layer 1: API key inline rate limit.
    let key_limits = auth.key().rate_limit.clone().unwrap_or_default();
    if !key_limits.is_unrestricted() {
        let r = state
            .limiter
            .pre_commit(&auth.entry.id, &key_limits)
            .await
            .map_err(|e| reject(state, e, "api_key", None))?;
        reservations.push(r);
    }

    // Layer 2: Model inline rate limit.
    if let Some(mrl) = model_rl {
        if let Some(ref limits) = mrl.limits {
            let key = format!("model:{}", mrl.bucket_name);
            let r = state
                .limiter
                .pre_commit(&key, limits)
                .await
                .map_err(|e| reject(state, e, "model", None))?;
            reservations.push(r);
        }
    }

    // Layer 3: per-MCP-server limit carried by this key. Bucketed on
    // `mcp:<api_key_id>:<server>` so each server the key reaches counts
    // independently — of the other servers, and of every other key.
    if let Some(server) = mcp_server {
        if let Some(limits) = auth.key().mcp_rate_limit(server) {
            let rl = RateLimit::from(limits);
            if !rl.is_unrestricted() {
                let key = format!("mcp:{}:{}", auth.entry.id, server);
                let r = state
                    .limiter
                    .pre_commit(&key, &rl)
                    .await
                    .map_err(|e| reject(state, e, "mcp", None))?;
                reservations.push(r);
            }
        }
    }

    // Layer 4+: Rate limit policies from snapshot.
    let input = condition_input(auth, model_rl, None);
    let phase = PolicyPhase::Request {
        defer_model_properties: model_rl.is_some_and(|m| m.routing_parent),
    };
    reserve_policy_layers(state, snapshot, &input, phase, &mut reservations).await?;

    Ok(MultiReservation::new(reservations))
}

/// Scan the policy table once for the given phase and reserve every
/// applicable layer. Shared by the request gate ([`reserve_layers`])
/// and the per-target gate ([`reserve_model_only`]) so the two scans
/// cannot drift (the schedules gate had to be patched into both loops
/// once already — AISIX-Cloud#1104).
async fn reserve_policy_layers(
    state: &ProxyState,
    // The caller's request snapshot (#941). This gate used to load its
    // own, which on a zero-policy deployment was the entire cost of the
    // call — the emptiness check below is O(1).
    snap: &aisix_core::AisixSnapshot,
    input: &ConditionInput<'_>,
    phase: PolicyPhase,
    reservations: &mut Vec<aisix_ratelimit::Reservation>,
) -> Result<(), ProxyError> {
    // O(1) empty check before anything else: deployments with no
    // rate-limit policies (the default) skip the wall-clock read and
    // the per-shard table scan below entirely. Covers both callers —
    // the request gate and the per-target gate.
    if snap.rate_limit_policies.is_empty() {
        return Ok(());
    }
    let now = chrono::Utc::now();
    for entry in snap.rate_limit_policies.entries() {
        let policy = &entry.value;
        // Inside a scheduled suspension window the policy reserves
        // nothing; enforcement resumes automatically when the window
        // closes, on the unchanged bucket (AISIX-Cloud#1104).
        if policy.suspended_at(now) {
            continue;
        }
        let Some(layer) = match_policy_layer(policy, &entry.id, input, phase) else {
            continue;
        };
        let r = state
            .limiter
            .pre_commit(&layer.bucket_key, &layer.limits)
            .await
            .map_err(|e| reject(state, e, "policy", Some((&entry.id, &policy.name))))?;
        reservations.push(r);
    }
    Ok(())
}

/// Convert a store-level rejection into the surfaced [`ProxyError`],
/// counting it under `aisix_ratelimit_rejections_total{scope,layer}` —
/// the gate is the one point every endpoint funnels through, so the
/// counter covers them all. Policy-layer rejections carry the policy
/// identity for 429 attribution (`error.policy`, AISIX-Cloud#892).
fn reject(
    state: &ProxyState,
    err: aisix_ratelimit::RateLimitError,
    layer: &'static str,
    policy: Option<(&str, &str)>,
) -> ProxyError {
    state.metrics.record_ratelimit_rejection(
        &err.scope().to_string(),
        layer,
        policy.map(|(id, _)| id),
    );
    match policy {
        Some((id, name)) => ProxyError::PolicyRateLimit {
            source: err,
            policy_id: id.to_string(),
            policy_name: name.to_string(),
        },
        None => ProxyError::from(err),
    }
}

/// Apply budget + multi-layer rate-limit checks for one request.
/// `model_rl` carries the resolved model identity for policy matching
/// and optional inline limits. Pass `None` only for endpoints that
/// don't resolve a model (e.g. passthrough).
pub(crate) async fn enforce(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    model_rl: Option<&ModelRateLimit>,
) -> Result<MultiReservation, ProxyError> {
    check_budget(state, auth).await?;
    reserve_layers(state, snapshot, auth, model_rl, None).await
}

/// Apply budget + multi-layer rate-limit checks for one MCP `tools/call`
/// against `mcp_server`, the server the called tool belongs to. Same gate
/// as [`enforce`] plus the key's per-server layer; MCP resolves no model,
/// so the model layers are never engaged.
pub(crate) async fn enforce_mcp(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    mcp_server: &str,
) -> Result<MultiReservation, ProxyError> {
    check_budget(state, auth).await?;
    reserve_layers(state, snapshot, auth, None, Some(mcp_server)).await
}

/// Budget pre-check shared by the enforce entry points: refreshes the
/// budget gauges from the cached cp-api decision and rejects the request
/// when the key is over budget.
async fn check_budget(state: &ProxyState, auth: &AuthenticatedKey) -> Result<(), ProxyError> {
    let decision = state.budgets.check(&auth.entry.id).await;
    let budget_labels = aisix_obs::BudgetLabels {
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref().unwrap_or("unknown"),
        user_id: auth.key().user_id.as_deref().unwrap_or("unknown"),
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
        return Err(ProxyError::BudgetExceeded(Box::new(
            decision.reason.unwrap_or_else(|| {
                crate::budget::BudgetReason::message_only(auth.entry.id.clone())
            }),
        )));
    }
    Ok(())
}

/// Rate-limit-only enforcement (no budget check). Used by `chat.rs`
/// which handles budget separately.
pub(crate) async fn enforce_rate_limit(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    model_rl: Option<&ModelRateLimit>,
) -> Result<MultiReservation, ProxyError> {
    reserve_layers(state, snapshot, auth, model_rl, None).await
}

/// Reserve ONLY the model-scoped layers for one model, identified by its
/// display name + entry id: the model's inline `rate_limit`, `model`-scope
/// classic `RateLimitPolicy` rows, and conditional rows referencing a model
/// property (their request-level twin reserved at the request gate).
///
/// The ensemble fan-out uses this per sub-call: each panel member and the
/// judge is a separate upstream call that must honor its own model limits,
/// even though the request-level layers (api_key / team / member) are reserved
/// once on the entry alias and committed with the aggregate (#620). It
/// deliberately omits those request-level layers so they are not double-counted
/// per member. Returns an empty [`MultiReservation`] (zero overhead, no
/// `pre_commit` calls) when the model carries no limits, so unlimited members
/// pay nothing. On a partial failure the already-acquired layers release on the
/// dropped `Vec`, same as [`reserve_layers`].
///
/// `auth` supplies the identity dimensions a conditional row's tree may
/// combine with its model condition (e.g. `team ∈ {T} AND model_name ~~
/// ^gpt-4`); classic model-scope rows keep ignoring it (their bucket
/// never splits per user).
pub(crate) async fn reserve_model_only(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    model_name: &str,
    model_entry_id: &str,
    model: &aisix_core::Model,
    routing_parent: Option<RoutingParent<'_>>,
) -> Result<MultiReservation, ProxyError> {
    let mut reservations = Vec::new();

    // Inline model rate limit.
    let mrl = ModelRateLimit::from_model(model_name, model_entry_id, model);
    if let Some(ref limits) = mrl.limits {
        let key = format!("model:{}", mrl.bucket_name);
        let r = state
            .limiter
            .pre_commit(&key, limits)
            .await
            .map_err(|e| reject(state, e, "model", None))?;
        reservations.push(r);
    }

    // Policies that follow the model to this target. The condition
    // input carries the {target, caller-addressed parent} pair so a
    // policy pinning the parent's id or alias matches here too
    // (AISIX-Cloud#1267).
    let input = condition_input(auth, Some(&mrl), routing_parent);
    reserve_policy_layers(
        state,
        snapshot,
        &input,
        PolicyPhase::ModelTarget,
        &mut reservations,
    )
    .await?;

    Ok(MultiReservation::new(reservations))
}

/// Reserve the model-scoped layers for one routing-dispatch target (Model
/// Group / semantic-router member), mirroring the ensemble per-sub-call
/// reservation (#620). Returns `Ok(None)` for a direct (non-routing)
/// dispatch: there the target IS the requested entry, whose model layers
/// were already reserved pre-dispatch by [`enforce`]/[`enforce_rate_limit`],
/// so reserving again would double-count the request (AISIX-Cloud#1087).
///
/// An `Err` means this target is over one of its own limits right now —
/// the dispatch loops treat that as a failed 429 attempt and continue with
/// the remaining targets (matching LiteLLM, which filters rate-limited
/// deployments out of the candidate set).
pub(crate) async fn reserve_routing_target(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    routing_parent: Option<RoutingParent<'_>>,
    target_name: &str,
    target_entry_id: &str,
    target: &aisix_core::Model,
) -> Result<Option<MultiReservation>, ProxyError> {
    let Some(parent) = routing_parent else {
        return Ok(None);
    };
    reserve_model_only(
        state,
        snapshot,
        auth,
        target_name,
        target_entry_id,
        target,
        Some(parent),
    )
    .await
    .map(Some)
}

/// Seconds until the offending window reopens, for a
/// [`reserve_routing_target`] rejection. `chat.rs` funnels its rejection
/// through a `BridgeError`, which would otherwise drop the hint the
/// `/v1/messages` and `/v1/responses` loops keep by carrying the
/// `ProxyError::RateLimit` itself — so every endpoint's all-targets-exhausted
/// 429 lands with the same `Retry-After`.
pub(crate) fn retry_after_of(err: &ProxyError) -> Option<u64> {
    err.retry_after_secs()
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

    fn make_scoped_policy(scope: &str, scope_ref: &str) -> RateLimitPolicy {
        serde_json::from_value(serde_json::json!({
            "name": "test",
            "scope": scope,
            "scope_ref": scope_ref,
            "window": "minute",
            "max_requests": 10,
        }))
        .unwrap()
    }

    fn make_auth(team_id: Option<&str>, user_id: Option<&str>) -> AuthenticatedKey {
        let key: aisix_core::ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": "h",
            "allowed_models": [],
            "team_id": team_id,
            "user_id": user_id,
        }))
        .unwrap();
        AuthenticatedKey {
            anonymous: false,
            entry: std::sync::Arc::new(aisix_core::resource::ResourceEntry::new(
                "key-entry-1",
                key,
                1,
            )),
            jwt: None,
        }
    }

    fn make_conditional_policy(body: serde_json::Value) -> RateLimitPolicy {
        serde_json::from_value(body).unwrap()
    }

    fn make_model_rl(name: &str, entry_id: &str, provider: Option<&str>) -> ModelRateLimit {
        ModelRateLimit {
            name: name.to_owned(),
            bucket_name: name.to_owned(),
            entry_id: entry_id.to_owned(),
            limits: None,
            provider: provider.map(str::to_owned),
            routing_parent: false,
        }
    }

    const REQUEST: PolicyPhase = PolicyPhase::Request {
        defer_model_properties: false,
    };
    const REQUEST_DEFERRING: PolicyPhase = PolicyPhase::Request {
        defer_model_properties: true,
    };

    /// Classic-row bucket key via the unified matcher, at the request
    /// phase with no model resolved.
    fn classic_layer_key(
        policy: &RateLimitPolicy,
        entry_id: &str,
        auth: &AuthenticatedKey,
    ) -> String {
        match_policy_layer(
            policy,
            entry_id,
            &condition_input(auth, None, None),
            REQUEST,
        )
        .expect("policy applies")
        .bucket_key
    }

    #[test]
    fn team_member_bucket_key_is_per_user() {
        let policy = make_scoped_policy("team_member", "team-1");
        let auth_a = make_auth(Some("team-1"), Some("user-a"));
        let auth_b = make_auth(Some("team-1"), Some("user-b"));

        let key_a = classic_layer_key(&policy, "pol-1", &auth_a);
        let key_b = classic_layer_key(&policy, "pol-1", &auth_b);

        // Same team + same policy, but distinct members → distinct buckets,
        // so member A exhausting the default never throttles member B.
        assert_eq!(key_a, "policy:team_member:team-1:pol-1:user-a");
        assert_eq!(key_b, "policy:team_member:team-1:pol-1:user-b");
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn team_bucket_key_is_shared_across_members() {
        // Contrast with `team`: one bucket for the whole team regardless
        // of which member sends the request (pooled quota).
        let policy = make_scoped_policy("team", "team-1");
        let key_a = classic_layer_key(&policy, "pol-1", &make_auth(Some("team-1"), Some("user-a")));
        let key_b = classic_layer_key(&policy, "pol-1", &make_auth(Some("team-1"), Some("user-b")));
        assert_eq!(key_a, "policy:team:team-1:pol-1");
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn minute_maps_to_rpm_tpm() {
        let rl = classic_rate_limit(&make_policy("minute", Some(100), Some(50000))).unwrap();
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
        let rl = classic_rate_limit(&make_policy("second", Some(10), Some(1000))).unwrap();
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
        let rl = classic_rate_limit(&make_policy("hour", Some(1000), Some(500000))).unwrap();
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
    fn day_maps_to_rpd_tpd() {
        // #771: day is the one window whose token counter (`tpd`) is
        // already native end to end, so both limits wire through.
        let rl = classic_rate_limit(&make_policy("day", Some(10000), Some(2000000))).unwrap();
        assert_eq!(rl.rpd, Some(10000));
        assert_eq!(rl.tpd, Some(2000000));
        assert!(rl.rpm.is_none());
        assert!(rl.tpm.is_none());
        assert!(rl.rps.is_none());
        assert!(rl.rph.is_none());
    }

    #[test]
    fn minute_window_unchanged_by_426() {
        // Regression guard: the minute branch was always correct
        // (rpm/tpm map 1:1). #426 must not have touched it.
        let rl = classic_rate_limit(&make_policy("minute", Some(60), Some(30000))).unwrap();
        assert_eq!(rl.rpm, Some(60));
        assert_eq!(rl.tpm, Some(30000));
        assert!(rl.rps.is_none());
        assert!(rl.rph.is_none());
        assert!(rl.rpd.is_none());
    }

    #[test]
    fn unknown_window_is_rejected_at_deserialize() {
        // `PolicyWindow` is a closed enum, so an unknown window is rejected at
        // deserialize rather than silently producing an unrestricted limit.
        let r: Result<RateLimitPolicy, _> = serde_json::from_value(serde_json::json!({
            "name": "test",
            "scope": "team",
            "scope_ref": "ref",
            "window": "week",
            "max_requests": 100,
        }));
        assert!(r.is_err());
    }

    #[test]
    fn partial_fields_only_set_relevant_dimension() {
        let rl = classic_rate_limit(&make_policy("minute", Some(60), None)).unwrap();
        assert_eq!(rl.rpm, Some(60));
        assert!(rl.tpm.is_none());
    }

    // ---- conditional form (AISIX-Cloud#892) ----

    #[test]
    fn conditional_shared_bucket_and_limits() {
        let policy = make_conditional_policy(serde_json::json!({
            "name": "team-pool",
            "conditions": [
                { "dimension": "team", "operator": "in", "value": ["team-1"] }
            ],
            "limits": { "rpm": 100 },
        }));
        let auth = make_auth(Some("team-1"), Some("user-a"));
        let layer = match_policy_layer(
            &policy,
            "pol-1",
            &condition_input(&auth, None, None),
            REQUEST,
        )
        .expect("matches");
        // No group_by → one shared bucket for every matched request.
        assert_eq!(layer.bucket_key, "policy:v2:pol-1");
        assert_eq!(layer.limits.rpm, Some(100));
    }

    #[test]
    fn group_by_segments_follow_canonical_order() {
        // Declared [model, team]; the bucket key must order team before
        // model so declaration order never changes the bucket identity.
        let policy = make_conditional_policy(serde_json::json!({
            "name": "per-team-per-model",
            "conditions": [],
            "group_by": ["model", "team"],
            "limits": { "rpm": 5 },
        }));
        let auth = make_auth(Some("team-1"), None);
        let mrl = make_model_rl("gpt-4.1-prod", "model-1", Some("openai"));
        let layer = match_policy_layer(
            &policy,
            "pol-2",
            &condition_input(&auth, Some(&mrl), None),
            REQUEST,
        )
        .expect("matches");
        assert_eq!(
            layer.bucket_key,
            "policy:v2:pol-2:team=team-1:model=model-1"
        );
    }

    #[test]
    fn group_by_missing_dimension_skips_policy() {
        // Mirrors team_member semantics: a per-member split cannot apply
        // to a key that carries no user_id.
        let policy = make_conditional_policy(serde_json::json!({
            "name": "per-member",
            "conditions": [
                { "dimension": "team", "operator": "==", "value": "team-1" }
            ],
            "group_by": ["member"],
            "limits": { "rpm": 20 },
        }));
        let auth = make_auth(Some("team-1"), None);
        assert!(match_policy_layer(
            &policy,
            "pol-3",
            &condition_input(&auth, None, None),
            REQUEST
        )
        .is_none());
    }

    #[test]
    fn model_property_policy_defers_to_target_phase_on_routing() {
        let policy = make_conditional_policy(serde_json::json!({
            "name": "gpt4-family",
            "conditions": [
                { "dimension": "model_name", "operator": "~~", "value": "^gpt-4" }
            ],
            "limits": { "rpm": 10 },
        }));
        let auth = make_auth(Some("team-1"), None);
        let parent = make_model_rl("gpt4-group", "group-1", None);
        let input = condition_input(&auth, Some(&parent), None);
        // Request gate of a routing dispatch: deferred even though the
        // parent's name would match — the concrete target decides.
        assert!(match_policy_layer(&policy, "pol-4", &input, REQUEST_DEFERRING).is_none());
        // Per-target gate: matches the concrete target.
        let target = make_model_rl("gpt-4.1-prod", "model-1", Some("openai"));
        let target_input = condition_input(&auth, Some(&target), None);
        let layer = match_policy_layer(&policy, "pol-4", &target_input, PolicyPhase::ModelTarget)
            .expect("target matches");
        assert_eq!(layer.bucket_key, "policy:v2:pol-4");
    }

    #[test]
    fn group_referencing_policy_matches_at_target_phase_via_parent() {
        // AISIX-Cloud#1267: `model in [group uuid]` reserves at the
        // per-target gate because the condition input carries the
        // {target, parent} pair — previously the parent id was compared
        // nowhere and the policy never fired.
        let policy = make_conditional_policy(serde_json::json!({
            "name": "group-cap",
            "conditions": [
                { "dimension": "model", "operator": "in", "value": ["group-1"] }
            ],
            "group_by": ["member"],
            "limits": { "rph": 3 },
        }));
        let auth = make_auth(Some("team-1"), Some("user-a"));
        // Request gate of the routing dispatch: still deferred.
        let gate = make_model_rl("chat-group", "group-1", None);
        let gate_input = condition_input(&auth, Some(&gate), None);
        assert!(match_policy_layer(&policy, "pol-9", &gate_input, REQUEST_DEFERRING).is_none());
        // Per-target gate: the parent pair makes it match, bucketed per
        // member.
        let target = make_model_rl("gpt-4.1-prod", "model-1", Some("openai"));
        let parent = RoutingParent {
            name: "chat-group",
            entry_id: "group-1",
        };
        let input = condition_input(&auth, Some(&target), Some(parent));
        let layer = match_policy_layer(&policy, "pol-9", &input, PolicyPhase::ModelTarget)
            .expect("group-referencing policy matches via the parent");
        assert_eq!(layer.bucket_key, "policy:v2:pol-9:member=user-a");
        // Direct dispatch to the member (no parent): must NOT match.
        let direct = condition_input(&auth, Some(&target), None);
        assert!(match_policy_layer(&policy, "pol-9", &direct, REQUEST).is_none());
    }

    #[test]
    fn group_by_model_buckets_on_target_not_parent() {
        // The pair extends MATCHING only: a group-referencing policy
        // splitting by model still buckets on the dispatched target id,
        // so per-member counters stay per concrete model.
        let policy = make_conditional_policy(serde_json::json!({
            "name": "group-per-model",
            "conditions": [
                { "dimension": "model", "operator": "in", "value": ["group-1"] }
            ],
            "group_by": ["model"],
            "limits": { "rpm": 1 },
        }));
        let auth = make_auth(Some("team-1"), None);
        let target = make_model_rl("gpt-4.1-prod", "model-1", Some("openai"));
        let parent = RoutingParent {
            name: "chat-group",
            entry_id: "group-1",
        };
        let input = condition_input(&auth, Some(&target), Some(parent));
        let layer = match_policy_layer(&policy, "pol-10", &input, PolicyPhase::ModelTarget)
            .expect("matches via parent");
        assert_eq!(layer.bucket_key, "policy:v2:pol-10:model=model-1");
    }

    #[test]
    fn non_model_policy_not_rereserved_at_target_phase() {
        let policy = make_conditional_policy(serde_json::json!({
            "name": "team-pool",
            "conditions": [
                { "dimension": "team", "operator": "in", "value": ["team-1"] }
            ],
            "limits": { "rpm": 100 },
        }));
        let auth = make_auth(Some("team-1"), None);
        let target = make_model_rl("gpt-4.1-prod", "model-1", Some("openai"));
        let input = condition_input(&auth, Some(&target), None);
        // Reserved once at the request gate; the per-target scan must
        // not double-count it.
        assert!(match_policy_layer(&policy, "pol-5", &input, PolicyPhase::ModelTarget).is_none());
    }

    #[test]
    fn or_branch_matches_model_less_request() {
        // §3.3 rule 3: a missing dimension only fails its own leaf. An
        // MCP/A2A request (no model) still matches through the team
        // branch of an OR group — and, carrying a model-property leaf,
        // the policy is evaluated at the request gate because a
        // model-less request has no target phase.
        let policy = make_conditional_policy(serde_json::json!({
            "name": "team-or-provider",
            "conditions": [
                { "logic": "or", "children": [
                    { "dimension": "team", "operator": "==", "value": "team-1" },
                    { "dimension": "provider", "operator": "==", "value": "anthropic" }
                ]}
            ],
            "limits": { "rpm": 50 },
        }));
        let auth = make_auth(Some("team-1"), None);
        let layer = match_policy_layer(
            &policy,
            "pol-6",
            &condition_input(&auth, None, None),
            REQUEST,
        )
        .expect("matches via team branch");
        assert_eq!(layer.bucket_key, "policy:v2:pol-6");
    }

    #[test]
    fn classic_scope_rows_ignored_at_target_phase_except_model() {
        let team_policy = make_scoped_policy("team", "team-1");
        let auth = make_auth(Some("team-1"), Some("user-a"));
        let target = make_model_rl("gpt-4.1-prod", "model-1", Some("openai"));
        let input = condition_input(&auth, Some(&target), None);
        assert!(
            match_policy_layer(&team_policy, "pol-7", &input, PolicyPhase::ModelTarget).is_none()
        );

        let model_policy = make_scoped_policy("model", "model-1");
        let layer = match_policy_layer(&model_policy, "pol-8", &input, PolicyPhase::ModelTarget)
            .expect("model scope follows the target");
        assert_eq!(layer.bucket_key, "policy:model:model-1:pol-8");
    }
}
