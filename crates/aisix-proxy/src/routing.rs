//! Per-virtual-model routing state + target selection.
//!
//! When a request lands on a Model with `routing` configured, the proxy
//! asks the [`RoutingRegistry`] for an iterator of underlying target
//! Model names in attempt-order. The registry owns the per-virtual-
//! model state (round-robin counter, weighted PRNG seed); selection
//! itself is pure given that state.
//!
//! Positional strategies (spec §3.5) pick a starting target, then walk
//! forward on failure:
//! - **failover**: always start at `targets[0]`, walk forward on failure.
//! - **round_robin**: each *new* request advances a per-model counter
//!   so callers spread evenly across targets.
//! - **weighted**: pick a starting target with probability proportional
//!   to `weight`, then walk forward on failure (weights only affect the
//!   *first* target choice — once we're falling back, order is positional).
//!
//! Metric-ordered strategies rank the whole target set by a runtime signal
//! (attempted best-first, then falling forward). They can't be ordered from
//! `pick_targets` because the ranking key lives on the resolved target
//! Models / runtime state, so `resolve_attempt_models` ranks them instead:
//! - **least_cost**: cheapest target first, by combined input+output per-1K
//!   price; targets without a `cost` rank last.
//! - **least_latency**: fastest target first, by an EWMA of observed upstream
//!   latency; targets with no samples yet rank first (probe, then exploit).
//! - **least_busy**: least-loaded target first, by in-flight request count.

use aisix_core::{
    AisixSnapshot, Model, Routing, RoutingStrategy, RoutingTarget, WhenAllUnavailablePolicy,
};
use aisix_gateway::BridgeError;
use dashmap::DashMap;
use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::error::ProxyError;

/// Default Retry-After (in seconds) returned to the client when every
/// candidate is background-unhealthy and no cooldown timer is available
/// to derive a more precise hint. Operators tune per-model cooldown
/// TTLs via `cooldown.default_seconds`; this is only the all-unhealthy
/// fallback for the `when_all_unavailable: fail` path.
const FALLBACK_ALL_UNHEALTHY_RETRY_AFTER: Duration = Duration::from_secs(30);

/// Whether a Bridge error is retryable at all, optionally treating 429
/// as retryable. Non-429 4xx is the caller's mistake — retrying won't
/// help and may amplify damage. Everything else (5xx, timeout,
/// transport, decode, config, stream abort) gets the retry/failover path.
///
/// `fallback_on_statuses` (AISIX-Cloud#1012) is the routing model's
/// explicit opt-in list for providers that use 4xx codes for transient
/// conditions (overload, queue full, quota): a status in the list is
/// retryable regardless of the default classification. Empty by default,
/// which preserves the historical behavior exactly.
pub fn is_retryable(err: &BridgeError, retry_on_429: bool, fallback_on_statuses: &[u16]) -> bool {
    match err {
        BridgeError::UpstreamStatus { status, .. } => {
            if fallback_on_statuses.contains(status) {
                return true;
            }
            if *status == 429 {
                return retry_on_429;
            }
            !(400..500).contains(status)
        }
        // Customer-fixable config / credentials (#367) is the caller's
        // mistake — retrying or failing over won't help, same as a
        // non-429 4xx.
        BridgeError::InvalidUpstreamConfig(_) | BridgeError::InvalidUpstreamCredentials(_) => false,
        BridgeError::Timeout { .. }
        | BridgeError::Transport(_)
        | BridgeError::UpstreamDecode(_)
        | BridgeError::Config(_)
        | BridgeError::StreamAborted => true,
    }
}

/// Base delay before the first same-target retry. Each subsequent retry
/// doubles it, capped at [`RETRY_BACKOFF_MAX_MS`].
const RETRY_BACKOFF_BASE_MS: u64 = 250;
/// Ceiling for the exponential term — bounds the worst-case added latency.
const RETRY_BACKOFF_MAX_MS: u64 = 2_000;
/// Additive jitter ceiling, sampled uniformly in `[0, this]` and added on
/// top of the exponential term.
const RETRY_BACKOFF_JITTER_MS: u64 = 250;

/// Backoff before retrying the **same** target, for 1-based retry number
/// `retry` (`retry == 0` → no wait). Exponential term `base * 2^(retry-1)`
/// capped at [`RETRY_BACKOFF_MAX_MS`], plus uniform additive jitter in
/// `[0, RETRY_BACKOFF_JITTER_MS]`.
///
/// Same strategy as LiteLLM's router (`_calculate_retry_after`: capped
/// exponential floor + additive jitter — not full-jitter-to-zero, so a
/// struggling upstream always gets a real pause), with bounds tightened
/// from LiteLLM's library defaults (0.5s base / 8s cap) to suit an inline
/// proxy where the retry runs inside a single request's latency budget.
/// Cross-target fallover is deliberately NOT backed off — a different,
/// presumably healthy target should be tried immediately (LiteLLM's
/// healthy-deployment fast-path).
pub fn retry_backoff(retry: u32) -> Duration {
    if retry == 0 {
        return Duration::ZERO;
    }
    let exp = RETRY_BACKOFF_BASE_MS.saturating_mul(1u64 << (retry - 1).min(20));
    let base = exp.min(RETRY_BACKOFF_MAX_MS);
    let jitter = rand::thread_rng().gen_range(0..=RETRY_BACKOFF_JITTER_MS);
    Duration::from_millis(base + jitter)
}

#[derive(Default)]
pub struct RoutingRegistry {
    // virtual model name → atomic round-robin cursor
    cursors: DashMap<String, AtomicUsize>,
}

impl RoutingRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pick the target order for one request. The first element is the
    /// initial target; subsequent elements are later fallback targets (in
    /// declaration order, wrapping if needed). Length is bounded by the
    /// initial target plus `routing.max_fallbacks_or_default()`.
    pub fn pick_targets(
        &self,
        virtual_name: &str,
        routing: &Routing,
        stability_key: Option<&str>,
    ) -> Vec<String> {
        if routing.targets.is_empty() {
            return Vec::new();
        }
        // Metric-ordered strategies (least_cost, …) can't be ranked here:
        // the ranking key lives on the resolved target Models / runtime
        // state, which `resolve_attempt_models` has and this does not. Hand
        // back the full declaration-order list; ranking and `max_fallbacks`
        // truncation happen there instead.
        if routing.strategy.is_metric_based() {
            return routing.targets.iter().map(|t| t.model.clone()).collect();
        }
        let start = self.starting_index(virtual_name, routing, stability_key);
        attempt_order(
            &routing.targets,
            start,
            routing.max_fallbacks_or_default() + 1,
        )
    }

    fn starting_index(
        &self,
        virtual_name: &str,
        routing: &Routing,
        stability_key: Option<&str>,
    ) -> usize {
        match routing.strategy {
            RoutingStrategy::Failover => 0,
            RoutingStrategy::RoundRobin => self.advance_cursor(virtual_name, routing.targets.len()),
            RoutingStrategy::Weighted => {
                // Sticky (A/B / canary) routing makes the weighted pick
                // deterministic in the request's stability key; otherwise each
                // request samples the weight distribution independently.
                let sticky_key = routing
                    .sticky_or_default()
                    .then_some(stability_key)
                    .flatten();
                weighted_pick(&routing.targets, sticky_key)
            }
            // Metric-ordered strategies never reach here — `pick_targets`
            // short-circuits them before computing a start index.
            RoutingStrategy::LeastCost
            | RoutingStrategy::LeastLatency
            | RoutingStrategy::LeastBusy => 0,
        }
    }

    fn advance_cursor(&self, virtual_name: &str, modulo: usize) -> usize {
        let entry = self.cursors.entry(virtual_name.to_string()).or_default();
        let prev = entry.fetch_add(1, Ordering::Relaxed);
        prev % modulo
    }
}

impl std::fmt::Debug for RoutingRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutingRegistry")
            .field("virtual_models_seen", &self.cursors.len())
            .finish()
    }
}

/// Narrow a routing model's targets to those eligible for this request's
/// routing tags, mirroring LiteLLM's tag-based routing:
///   * No target is tagged → tag routing isn't in use; every target eligible.
///   * Request carries tags → targets whose tags intersect it (match-any); if
///     none match, fall back to `"default"`-tagged targets.
///   * Request has no tags → `"default"`-tagged targets if any, else all.
///
/// Returns owned clones so the caller runs the normal strategy over the
/// surviving subset. An empty result means the request asked for a tag tier
/// with no matching target and no default — the caller turns that into an error.
fn eligible_targets(targets: &[RoutingTarget], request_tags: &[String]) -> Vec<RoutingTarget> {
    if !targets.iter().any(RoutingTarget::has_tags) {
        return targets.to_vec();
    }
    let defaults = || -> Vec<RoutingTarget> {
        targets
            .iter()
            .filter(|t| t.is_default_target())
            .cloned()
            .collect()
    };
    if request_tags.is_empty() {
        let d = defaults();
        return if d.is_empty() { targets.to_vec() } else { d };
    }
    let matched: Vec<RoutingTarget> = targets
        .iter()
        .filter(|t| t.matches_request_tags(request_tags))
        .cloned()
        .collect();
    if matched.is_empty() {
        defaults()
    } else {
        matched
    }
}

/// Build the target-order vector starting at `start_idx`, walking forward
/// (wrap-around) for `limit` distinct entries.
fn attempt_order(targets: &[RoutingTarget], start_idx: usize, limit: usize) -> Vec<String> {
    let n = targets.len();
    let mut order = Vec::with_capacity(limit);
    for i in 0..limit {
        let t = &targets[(start_idx + i) % n];
        order.push(t.model.clone());
    }
    order
}

/// Pick an index by weighted-random. Ignores zero weights; a fully-zero
/// list falls back to index 0 deterministically.
///
/// Per #197: each call must draw an INDEPENDENT sample from the weight
/// distribution. The prior implementation used
/// `SystemTime::now().subsec_nanos() + Instant::now().elapsed().as_nanos()`
/// as entropy, which has two correctness bugs that compound:
///   1. `Instant::now().elapsed()` always returns ~0 (the Instant was
///      just created), so the mix is effectively just subsec_nanos.
///   2. Under rapid-fire requests (e2e fires N=100 in tight loop),
///      consecutive subsec_nanos values differ by a near-constant
///      step (≈1 µs of wall-clock per request). Modular reduction
///      `entropy() % total_weight` against that step pattern aliases
///      to a single bin — every request lands on the same target.
///      Empirical observation: 200/0 split on a configured 70/30.
///
/// Use `rand::thread_rng()` instead. The thread-local PRNG is seeded
/// from OS entropy on first use and is independent across calls; the
/// distribution converges to the configured weights over a finite
/// sample (per the spec the e2e pins).
///
/// With a `sticky_key` (A/B / canary routing) the pick is instead a
/// deterministic function of that key, so the same key always resolves to the
/// same target while the aggregate split still honors the weights.
fn weighted_pick(targets: &[RoutingTarget], sticky_key: Option<&str>) -> usize {
    let total: u64 = targets.iter().map(|t| t.weight_or_default() as u64).sum();
    if total == 0 {
        return 0;
    }
    let pick = match sticky_key {
        Some(key) => stable_hash(key) % total,
        None => rand::thread_rng().gen_range(0..total),
    };
    let mut acc: u64 = 0;
    for (i, t) in targets.iter().enumerate() {
        acc += t.weight_or_default() as u64;
        if pick < acc {
            return i;
        }
    }
    targets.len() - 1
}

/// Stable 64-bit FNV-1a hash used to map a sticky-routing key into the weight
/// distribution. Deterministic across processes and toolchains by design (the
/// std hasher is not), so a given key always resolves to the same target.
fn stable_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
    }
    h
}

/// Combined per-1K unit price used to rank `least_cost` targets. A target
/// Model without a configured `cost` sorts last (treated as +∞) so a
/// misconfigured target is deprioritised rather than silently preferred.
fn cost_key(model: &Model) -> f64 {
    model
        .cost
        .as_ref()
        .map(|c| c.input_per_1k + c.output_per_1k)
        .unwrap_or(f64::INFINITY)
}

/// Observed-latency key used to rank `least_latency` targets. A target with
/// no latency samples yet sorts first (treated as −∞) so it gets probed;
/// once it has an EWMA it ranks by that.
fn latency_key(runtime_status: &crate::ModelRuntimeStatusTracker, id: &str) -> f64 {
    runtime_status
        .latency_ewma_ms(id)
        .unwrap_or(f64::NEG_INFINITY)
}

/// Rank the resolved attempt list by the strategy's runtime metric,
/// best-first (ascending). Stable, so equal-metric targets keep their
/// declaration order. Only metric-based strategies reach here; positional
/// strategies are ordered in [`RoutingRegistry::pick_targets`].
fn order_attempts_by_metric(
    strategy: RoutingStrategy,
    attempts: &mut [AttemptModel],
    runtime_status: &crate::ModelRuntimeStatusTracker,
) {
    match strategy {
        RoutingStrategy::LeastCost => {
            attempts.sort_by(|a, b| cost_key(&a.model).total_cmp(&cost_key(&b.model)));
        }
        RoutingStrategy::LeastLatency => {
            attempts.sort_by(|a, b| {
                latency_key(runtime_status, &a.id).total_cmp(&latency_key(runtime_status, &b.id))
            });
        }
        RoutingStrategy::LeastBusy => {
            attempts.sort_by_key(|a| runtime_status.in_flight(&a.id));
        }
        RoutingStrategy::Failover | RoutingStrategy::RoundRobin | RoutingStrategy::Weighted => {}
    }
}

/// One concrete (non-routing) Model the dispatch loop will attempt, paired
/// with its snapshot id so health/cooldown tracking can key on it.
#[derive(Clone)]
pub(crate) struct AttemptModel {
    pub id: String,
    pub model: Model,
}

/// Outcome of routing-candidate filtering. Lifts the "all candidates
/// excluded" case out into a typed result so the dispatch loop can
/// short-circuit to a 503 + Retry-After instead of sending traffic to
/// a target we just confirmed is bad.
pub(crate) enum FilterOutcome {
    /// At least one candidate survived the filter. The returned vector
    /// is the filtered attempt list, in the original strategy order
    /// minus the excluded entries.
    Selected(Vec<AttemptModel>),
    /// Every candidate is currently background-unhealthy and the
    /// routing model is configured with `when_all_unavailable: fail`. The
    /// caller should surface a 503 with the supplied Retry-After hint
    /// (in seconds), if any.
    AllUnhealthy { retry_after_secs: Option<u64> },
}

pub(crate) fn filter_attempt_models(
    runtime_status: &crate::ModelRuntimeStatusTracker,
    attempts: Vec<AttemptModel>,
    policy: WhenAllUnavailablePolicy,
) -> FilterOutcome {
    let mut healthy = Vec::new();
    let mut cooldown_only = Vec::new();
    let mut unhealthy_count = 0usize;

    for attempt in attempts.iter().cloned() {
        let stale_after = attempt
            .model
            .background_model_check
            .as_ref()
            .map(|cfg| Duration::from_secs(cfg.stale_after_seconds));
        let snapshot = runtime_status.status_with_stale(&attempt.id, stale_after);
        match snapshot.status {
            crate::RuntimeStatus::Unhealthy => unhealthy_count += 1,
            crate::RuntimeStatus::Cooldown => cooldown_only.push(attempt),
            crate::RuntimeStatus::Healthy | crate::RuntimeStatus::NotApplicable => {
                healthy.push(attempt)
            }
        }
    }

    if !healthy.is_empty() {
        return FilterOutcome::Selected(healthy);
    }
    // No healthy candidates — prefer cooldown over unhealthy when
    // some non-unhealthy candidates exist. Sending to a target whose
    // cooldown timer hasn't expired is still better than sending to
    // a target that an active probe just confirmed is broken.
    //
    // Reuse the single status read from the classification loop above:
    // with `healthy` empty here, the non-unhealthy candidates are
    // exactly the `cooldown_only` ones. Re-reading runtime_status to
    // re-filter would add a redundant per-candidate query and open a
    // race window — a candidate flipping to unhealthy between the two
    // reads could yield an empty `Selected`, which streaming callers
    // turn into a panic by indexing `attempt_models[0]`.
    if unhealthy_count < attempts.len() && !cooldown_only.is_empty() {
        return FilterOutcome::Selected(cooldown_only);
    }
    // All candidates are excluded. Policy decides.
    //
    // Retry-After for the fail path is a coarse fallback (30s by
    // default — see FALLBACK_ALL_UNHEALTHY_RETRY_AFTER). We could
    // try to derive it from per-candidate cooldown timers, but the
    // categorisation above routes cooldown candidates into
    // `cooldown_only` (returned via the Selected branch above), so
    // by construction every candidate that reaches here is in the
    // background-unhealthy state and has no cooldown timer to read.
    match policy {
        WhenAllUnavailablePolicy::Fail => FilterOutcome::AllUnhealthy {
            retry_after_secs: Some(FALLBACK_ALL_UNHEALTHY_RETRY_AFTER.as_secs()),
        },
        WhenAllUnavailablePolicy::TryAnyway => FilterOutcome::Selected(attempts),
    }
}

/// Per-request routing inputs threaded into [`resolve_attempt_models`]: the
/// tags that gate tag/metadata routing, the stability key for sticky
/// (A/B / canary) weighted selection, and the caller's resolved source IP
/// for the per-target client-IP allowlist. Tags come from request headers;
/// the stability key is the routing-key header when present, otherwise the
/// caller's API key id.
///
/// `source_ip` defaults to the empty string, which
/// [`aisix_core::Model::ip_allowed`] treats as "not in range" — so a caller
/// that forgets to thread it fails closed on restricted targets rather than
/// silently disabling the allowlist.
#[derive(Clone, Copy, Default)]
pub(crate) struct RoutingRequest<'a> {
    pub tags: &'a [String],
    pub stability_key: Option<&'a str>,
    pub source_ip: &'a str,
}

/// Drop the targets whose own `allowed_cidrs` excludes `source_ip`.
///
/// Deliberately NOT folded into [`filter_attempt_models`]: that filter's
/// `when_all_unavailable: try_anyway` policy hands back the *unfiltered*
/// candidate list, which would send a request to a target the operator just
/// declared off-limits for this caller. An allowlist has no "try anyway".
fn targets_allowed_for_ip(
    snapshot: &AisixSnapshot,
    targets: Vec<RoutingTarget>,
    source_ip: &str,
) -> Vec<RoutingTarget> {
    targets
        .into_iter()
        .filter(|t| {
            // An unresolvable name is left in place so the resolution loop
            // below still reports it as a config error, rather than being
            // silently swallowed here as an IP rejection.
            snapshot
                .models
                .get_by_name(&t.model)
                .is_none_or(|entry| entry.value.ip_allowed(source_ip))
        })
        .collect()
}

/// Resolve the ordered list of concrete Models a request will attempt.
///
/// For a routing model (Model Group), walk `routing.targets` per the
/// configured strategy, resolve each target name to a Model in the
/// snapshot, then apply the health/cooldown filter. For a direct
/// (non-routing) model, the list is just the model itself.
///
/// Shared by `/v1/chat/completions` and `/v1/messages` so both endpoints
/// dispatch Model Groups identically (ai-gateway#471).
pub(crate) fn resolve_attempt_models(
    routing_registry: &RoutingRegistry,
    runtime_status: &crate::ModelRuntimeStatusTracker,
    snapshot: &AisixSnapshot,
    virtual_name: &str,
    virtual_id: &str,
    virtual_model: &Model,
    req: RoutingRequest<'_>,
) -> Result<Vec<AttemptModel>, ProxyError> {
    let Some(routing) = virtual_model.routing.as_ref() else {
        return Ok(vec![AttemptModel {
            id: virtual_id.to_string(),
            model: virtual_model.clone(),
        }]);
    };

    // Tag/metadata pre-filter: narrow the targets to those eligible for this
    // request's routing tags, then let the configured strategy order whatever
    // survives. A no-op when no target is tagged.
    let eligible = eligible_targets(&routing.targets, req.tags);
    if eligible.is_empty() {
        return Err(ProxyError::InvalidRequest(format!(
            "no routing target matches request tags {:?}",
            req.tags
        )));
    }
    // Client-IP pre-filter (AISIX-Cloud#1087 follow-up): a target whose own
    // `allowed_cidrs` excludes this caller is not a candidate. Applied BEFORE
    // the strategy picks, so `max_fallbacks` budgets attempts across the
    // targets this caller may actually reach, and a metric-based strategy
    // ranks only those. The group's own `allowed_cidrs` is separately enforced
    // pre-dispatch by `dispatch::check_ip_access`; this adds the member tier
    // that a group previously bypassed entirely.
    let eligible = targets_allowed_for_ip(snapshot, eligible, req.source_ip);
    if eligible.is_empty() {
        // Report the name the caller asked for, not the excluded members —
        // matching `ModelForbidden`, and without disclosing group internals.
        return Err(ProxyError::ModelIpRestricted(virtual_name.to_string()));
    }
    let filtered_routing = Routing {
        targets: eligible,
        ..routing.clone()
    };
    let routing = &filtered_routing;

    let names = routing_registry.pick_targets(virtual_name, routing, req.stability_key);
    if names.is_empty() {
        return Err(ProxyError::InvalidRequest(
            "routing model has no targets".into(),
        ));
    }
    let mut resolved = Vec::with_capacity(names.len());
    for name in &names {
        let target_entry = snapshot.models.get_by_name(name).ok_or_else(|| {
            ProxyError::InvalidRequest(format!(
                "routing target {name:?} does not resolve to a Model"
            ))
        })?;
        resolved.push(AttemptModel {
            id: target_entry.id.clone(),
            model: target_entry.value.clone(),
        });
    }
    // Metric-ordered strategies get the full target set from `pick_targets`;
    // rank it best-first here (target Models are now resolved) and cap it to
    // the same attempt budget the positional strategies apply upstream.
    if routing.strategy.is_metric_based() {
        order_attempts_by_metric(routing.strategy, &mut resolved, runtime_status);
        resolved.truncate(routing.max_fallbacks_or_default() + 1);
    }
    match filter_attempt_models(
        runtime_status,
        resolved,
        routing.when_all_unavailable_or_default(),
    ) {
        FilterOutcome::Selected(list) => Ok(list),
        FilterOutcome::AllUnhealthy { retry_after_secs } => {
            tracing::warn!(
                virtual_model = %virtual_name,
                retry_after_secs,
                "all routing candidates are unavailable; failing fast",
            );
            Err(ProxyError::AllCandidatesUnavailable { retry_after_secs })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::{Routing, RoutingStrategy, RoutingTarget};

    fn r(
        strategy: RoutingStrategy,
        targets: Vec<RoutingTarget>,
        max_fallbacks: Option<u32>,
    ) -> Routing {
        Routing {
            strategy,
            targets,
            retries: None,
            max_fallbacks,
            retry_on_429: None,
            fallback_on_statuses: None,
            when_all_unavailable: None,
            sticky: None,
        }
    }

    fn tagged(model: &str, tags: &[&str]) -> RoutingTarget {
        RoutingTarget::new(model).with_tags(tags.iter().map(|s| s.to_string()).collect())
    }

    fn model_names(targets: &[RoutingTarget]) -> Vec<&str> {
        targets.iter().map(|t| t.model.as_str()).collect()
    }

    #[test]
    fn stable_hash_is_deterministic() {
        assert_eq!(stable_hash("session-abc"), stable_hash("session-abc"));
        assert_ne!(stable_hash("a"), stable_hash("b"));
    }

    #[test]
    fn sticky_weighted_pick_is_deterministic_per_key() {
        let targets = vec![
            RoutingTarget::new("a").with_weight(50),
            RoutingTarget::new("b").with_weight(50),
        ];
        let first = weighted_pick(&targets, Some("session-1"));
        for _ in 0..50 {
            assert_eq!(weighted_pick(&targets, Some("session-1")), first);
        }
    }

    #[test]
    fn sticky_weighted_pick_spreads_distinct_keys() {
        // Distinct keys shouldn't all funnel to one target.
        let targets = vec![
            RoutingTarget::new("a").with_weight(50),
            RoutingTarget::new("b").with_weight(50),
        ];
        let mut seen = [false; 2];
        for i in 0..200 {
            seen[weighted_pick(&targets, Some(&format!("k{i}")))] = true;
        }
        assert!(seen[0] && seen[1]);
    }

    #[test]
    fn sticky_weighted_pick_honors_extreme_weights() {
        // A 100/0 canary split lands every key on the weighted target.
        let targets = vec![
            RoutingTarget::new("stable").with_weight(100),
            RoutingTarget::new("canary").with_weight(0),
        ];
        for i in 0..50 {
            assert_eq!(weighted_pick(&targets, Some(&format!("k{i}"))), 0);
        }
    }

    #[test]
    fn sticky_routing_pins_a_key_to_one_target() {
        let reg = RoutingRegistry::new();
        let mut routing = r(
            RoutingStrategy::Weighted,
            vec![
                RoutingTarget::new("stable").with_weight(90),
                RoutingTarget::new("canary").with_weight(10),
            ],
            Some(0), // only the chosen start target
        );
        routing.sticky = Some(true);
        let first = reg.pick_targets("v", &routing, Some("user-42"));
        assert_eq!(first.len(), 1);
        for _ in 0..20 {
            assert_eq!(reg.pick_targets("v", &routing, Some("user-42")), first);
        }
    }

    #[test]
    fn eligible_no_tagged_target_returns_all() {
        // No target is tagged → tag routing isn't in use, even with request tags.
        let targets = vec![RoutingTarget::new("a"), RoutingTarget::new("b")];
        assert_eq!(
            model_names(&eligible_targets(&targets, &["x".into()])),
            vec!["a", "b"]
        );
    }

    #[test]
    fn eligible_matches_any_overlapping_tag() {
        let targets = vec![tagged("eu", &["eu"]), tagged("us", &["us"])];
        assert_eq!(
            model_names(&eligible_targets(&targets, &["eu".into()])),
            vec!["eu"]
        );
    }

    #[test]
    fn eligible_tagged_no_match_falls_back_to_default() {
        let targets = vec![tagged("eu", &["eu"]), tagged("fallback", &["default"])];
        assert_eq!(
            model_names(&eligible_targets(&targets, &["apac".into()])),
            vec!["fallback"]
        );
    }

    #[test]
    fn eligible_untagged_request_prefers_default() {
        let targets = vec![tagged("eu", &["eu"]), tagged("fallback", &["default"])];
        assert_eq!(
            model_names(&eligible_targets(&targets, &[])),
            vec!["fallback"]
        );
    }

    #[test]
    fn eligible_untagged_request_without_default_returns_all() {
        let targets = vec![tagged("eu", &["eu"]), tagged("us", &["us"])];
        assert_eq!(
            model_names(&eligible_targets(&targets, &[])),
            vec!["eu", "us"]
        );
    }

    // ───────────────── per-target client-IP allowlist ─────────────────

    fn ip_snapshot(models: &[(&str, Option<Vec<&str>>)]) -> AisixSnapshot {
        let table = aisix_core::snapshot::ResourceTable::default();
        for (i, (name, cidrs)) in models.iter().enumerate() {
            let model: Model = serde_json::from_value(serde_json::json!({
                "display_name": name,
                "provider": "openai",
                "model_name": "up",
                "provider_key_id": "pk-1",
                "allowed_cidrs": cidrs,
            }))
            .unwrap();
            table.insert(aisix_core::ResourceEntry::new(format!("m-{i}"), model, 1));
        }
        AisixSnapshot {
            models: table,
            ..Default::default()
        }
    }

    #[test]
    fn ip_filter_drops_only_the_out_of_range_target() {
        let snap = ip_snapshot(&[("restricted", Some(vec!["10.0.0.0/8"])), ("open", None)]);
        let targets = vec![tagged("restricted", &[]), tagged("open", &[])];

        // In range → both stay candidates.
        assert_eq!(
            model_names(&targets_allowed_for_ip(&snap, targets.clone(), "10.1.2.3")),
            vec!["restricted", "open"]
        );
        // Out of range → the restricted member drops out, the group still serves.
        assert_eq!(
            model_names(&targets_allowed_for_ip(&snap, targets, "8.8.8.8")),
            vec!["open"]
        );
    }

    #[test]
    fn ip_filter_empties_when_every_target_excludes_the_caller() {
        // The caller turns an empty result into a 403 rather than dispatching.
        let snap = ip_snapshot(&[
            ("a", Some(vec!["10.0.0.0/8"])),
            ("b", Some(vec!["192.168.0.0/16"])),
        ]);
        let targets = vec![tagged("a", &[]), tagged("b", &[])];
        assert!(targets_allowed_for_ip(&snap, targets, "8.8.8.8").is_empty());
    }

    #[test]
    fn ip_filter_fails_closed_on_an_unattributable_source_ip() {
        // Mirrors `Model::ip_allowed`: an empty/unparseable IP can never
        // satisfy a configured allowlist, so a request whose peer address
        // was lost must not reach a restricted target.
        let snap = ip_snapshot(&[("restricted", Some(vec!["10.0.0.0/8"]))]);
        let targets = vec![tagged("restricted", &[])];
        assert!(targets_allowed_for_ip(&snap, targets, "").is_empty());
    }

    #[test]
    fn ip_filter_keeps_unresolvable_names_for_the_config_error_path() {
        // A target naming a Model that isn't in the snapshot must surface as
        // the existing "does not resolve to a Model" config error, not be
        // silently swallowed here as an IP rejection.
        let snap = ip_snapshot(&[("known", None)]);
        let targets = vec![tagged("ghost", &[])];
        assert_eq!(
            model_names(&targets_allowed_for_ip(&snap, targets, "8.8.8.8")),
            vec!["ghost"]
        );
    }

    #[test]
    fn ip_filter_is_a_noop_when_no_target_restricts() {
        let snap = ip_snapshot(&[("a", None), ("b", None)]);
        let targets = vec![tagged("a", &[]), tagged("b", &[])];
        assert_eq!(
            model_names(&targets_allowed_for_ip(&snap, targets, "8.8.8.8")),
            vec!["a", "b"]
        );
    }

    #[test]
    fn eligible_tagged_no_match_no_default_is_empty() {
        // The caller turns an empty result into a "no target matches tags" error.
        let targets = vec![tagged("eu", &["eu"]), tagged("us", &["us"])];
        assert!(eligible_targets(&targets, &["apac".into()]).is_empty());
    }

    #[test]
    fn failover_always_starts_at_index_zero() {
        let reg = RoutingRegistry::new();
        let routing = r(
            RoutingStrategy::Failover,
            vec![
                RoutingTarget::new("primary"),
                RoutingTarget::new("secondary"),
                RoutingTarget::new("tertiary"),
            ],
            None,
        );
        for _ in 0..5 {
            let order = reg.pick_targets("v", &routing, None);
            assert_eq!(order, vec!["primary", "secondary", "tertiary"]);
        }
    }

    #[test]
    fn round_robin_cycles_through_targets_per_call() {
        let reg = RoutingRegistry::new();
        let routing = r(
            RoutingStrategy::RoundRobin,
            vec![
                RoutingTarget::new("a"),
                RoutingTarget::new("b"),
                RoutingTarget::new("c"),
            ],
            Some(1), // only the first attempt — easier to assert ordering
        );
        let mut firsts = Vec::new();
        for _ in 0..6 {
            let order = reg.pick_targets("v", &routing, None);
            firsts.push(order[0].clone());
        }
        // Two full cycles of a→b→c.
        assert_eq!(firsts, vec!["a", "b", "c", "a", "b", "c"]);
    }

    #[test]
    fn round_robin_state_is_per_virtual_model() {
        let reg = RoutingRegistry::new();
        let routing = r(
            RoutingStrategy::RoundRobin,
            vec![RoutingTarget::new("a"), RoutingTarget::new("b")],
            Some(1),
        );
        // Two distinct virtual models advance independently.
        assert_eq!(reg.pick_targets("v1", &routing, None)[0], "a");
        assert_eq!(reg.pick_targets("v2", &routing, None)[0], "a");
        assert_eq!(reg.pick_targets("v1", &routing, None)[0], "b");
        assert_eq!(reg.pick_targets("v2", &routing, None)[0], "b");
    }

    #[test]
    fn fallback_walks_forward_with_wraparound() {
        let reg = RoutingRegistry::new();
        let routing = r(
            RoutingStrategy::RoundRobin,
            vec![
                RoutingTarget::new("a"),
                RoutingTarget::new("b"),
                RoutingTarget::new("c"),
            ],
            Some(2),
        );
        // First call starts at a → a, b, c
        assert_eq!(reg.pick_targets("v", &routing, None), vec!["a", "b", "c"]);
        // Second call starts at b → b, c, a
        assert_eq!(reg.pick_targets("v", &routing, None), vec!["b", "c", "a"]);
    }

    #[test]
    fn weighted_picks_from_targets_and_falls_back_in_order() {
        let reg = RoutingRegistry::new();
        let routing = r(
            RoutingStrategy::Weighted,
            vec![
                RoutingTarget::new("a").with_weight(99),
                RoutingTarget::new("b").with_weight(1),
            ],
            Some(1),
        );
        // We just assert correctness of the *order* shape:
        // exactly two attempts, distinct targets, both targets covered.
        // (Aggregate distribution is pinned by the dedicated tests
        // below.)
        let order = reg.pick_targets("v", &routing, None);
        assert_eq!(order.len(), 2);
        assert!(order.iter().any(|t| t == "a"));
        assert!(order.iter().any(|t| t == "b"));
    }

    #[test]
    fn weighted_with_all_zero_weights_picks_index_zero_deterministically() {
        let targets = vec![
            RoutingTarget::new("a").with_weight(0),
            RoutingTarget::new("b").with_weight(0),
        ];
        assert_eq!(weighted_pick(&targets, None), 0);
    }

    /// Aggregate-distribution property: across many trials, a 100/1
    /// weight bias must converge to ~99% on the heavy target. Pre-#197
    /// the threshold sat at ≥ 60% to absorb the weak nanos-clock entropy
    /// — that gate would also pass a weight-half-sensitivity regression
    /// (~75% would slip through). With proper PRNG entropy in
    /// `weighted_pick`, the empirical bin should land within ~1% of
    /// the analytic 100/(100+1) = 99.0% expectation; we assert ≥ 95%
    /// (≈4σ band for n=5000, rejects half-sensitivity AND weight-blind).
    #[test]
    fn weighted_pick_aggregate_distribution_favors_heavier_weight() {
        let targets = vec![
            RoutingTarget::new("a").with_weight(100),
            RoutingTarget::new("b").with_weight(1),
        ];
        let n = 5_000;
        let a_count = (0..n)
            .filter(|_| weighted_pick(&targets, None) == 0)
            .count();
        // Uniform 50/50 → ~2500. Weighted 100/1 → ~4950 in theory.
        // 95% threshold (4750) rejects both a weight-blind impl
        // (~50%) AND a half-sensitivity regression (~75% would also
        // fail). With proper PRNG entropy this gate has ~5σ margin;
        // CI-flake risk is negligible.
        assert!(
            a_count * 100 / n >= 95,
            "weight=100 target should dominate aggregate picks; got {a_count}/{n}",
        );
    }

    /// Companion to the above: that test passes both for a correctly
    /// weighted impl AND for an "always pick index 0" regression (since
    /// the heavy weight is at index 0). Swap the weights so the heavy
    /// target sits at index 1 — a weight-blind impl that always picks
    /// the first target would now fail this test, while a correct
    /// weighted impl still favors index 1.
    #[test]
    fn weighted_pick_aggregate_distribution_respects_index_swap() {
        let targets = vec![
            RoutingTarget::new("a").with_weight(1),
            RoutingTarget::new("b").with_weight(100),
        ];
        let n = 5_000;
        let b_count = (0..n)
            .filter(|_| weighted_pick(&targets, None) == 1)
            .count();
        assert!(
            b_count * 100 / n >= 95,
            "weight=100 target at index 1 should dominate aggregate picks; got {b_count}/{n}",
        );
    }

    /// Issue #197 regression: a 70/30 weighted split must land near
    /// 70/30 over a finite sample. The pre-fix nanos-clock entropy
    /// collapsed to a single bin under rapid-fire calls (observed
    /// 200/0 in e2e on a configured 70/30); a proper PRNG converges
    /// to the analytic distribution.
    ///
    /// Tolerance: n=1000 with p=0.7 has σ=√(np(1-p))=√210≈14.49. A ±50
    /// absolute window is ~3.45σ → P(false positive) ≈ 0.056%. The
    /// pre-fix collapse-to-one-bin failure produces 1000/0 which is
    /// ~33σ outside the window — caught with overwhelming margin.
    #[test]
    fn weighted_pick_70_30_split_converges_to_configured_ratio() {
        let targets = vec![
            RoutingTarget::new("a").with_weight(70),
            RoutingTarget::new("b").with_weight(30),
        ];
        let n = 1_000;
        let a_count = (0..n)
            .filter(|_| weighted_pick(&targets, None) == 0)
            .count();
        // Expected ~700; tolerance window [650, 750] (≈±3.45σ).
        assert!(
            (650..=750).contains(&a_count),
            "70/30 weighted split must land near 700/1000; got {a_count}/{n} on heavy target",
        );
    }

    /// 3-target coverage: a weight-blind impl that only ever picks
    /// `targets[0]` if `pick < sum/n` (and `targets[1]` otherwise)
    /// would pass every 2-target test in this module but fail with
    /// 3+ targets — the third bin would starve. Pin a 50/30/20 split
    /// and assert each bin lands within a generous tolerance window.
    ///
    /// n=2000 chosen so the smallest bin (20% → ~400) has σ ≈ 17.9;
    /// ±100 window ≈ 5.6σ for that bin, larger margins for the other
    /// two.
    #[test]
    fn weighted_pick_50_30_20_split_distributes_to_all_three_bins() {
        let targets = vec![
            RoutingTarget::new("a").with_weight(50),
            RoutingTarget::new("b").with_weight(30),
            RoutingTarget::new("c").with_weight(20),
        ];
        let n = 2_000;
        let mut counts = [0_usize; 3];
        for _ in 0..n {
            counts[weighted_pick(&targets, None)] += 1;
        }
        // Expected 1000/600/400. ±100 window catches a weight-blind
        // 2-target collapse (where the 3rd bin would be 0) AND
        // sample noise.
        assert!(
            (900..=1100).contains(&counts[0]),
            "50%-weighted bin should land near 1000/2000; got {counts:?}",
        );
        assert!(
            (500..=700).contains(&counts[1]),
            "30%-weighted bin should land near 600/2000; got {counts:?}",
        );
        assert!(
            (300..=500).contains(&counts[2]),
            "20%-weighted bin should land near 400/2000; got {counts:?}",
        );
    }

    /// Zero-weight-in-the-middle: a weight=0 target between two
    /// non-zero targets must NEVER be picked. The CDF predicate
    /// `pick < acc` (strict less-than) is what enforces this — a
    /// weight-0 segment doesn't widen `acc` so the predicate skips
    /// past it. A regression that used `<=` would incidentally pick
    /// the zero-weight bin on the boundary value of `pick`.
    #[test]
    fn weighted_pick_zero_weight_target_in_middle_is_never_picked() {
        let targets = vec![
            RoutingTarget::new("a").with_weight(10),
            RoutingTarget::new("b").with_weight(0),
            RoutingTarget::new("c").with_weight(10),
        ];
        let n = 2_000;
        let b_count = (0..n)
            .filter(|_| weighted_pick(&targets, None) == 1)
            .count();
        assert_eq!(
            b_count, 0,
            "weight=0 target must never be picked; got {b_count}/{n}",
        );
    }

    #[test]
    fn max_fallbacks_zero_disables_failover() {
        let reg = RoutingRegistry::new();
        let routing = r(
            RoutingStrategy::Failover,
            vec![RoutingTarget::new("a"), RoutingTarget::new("b")],
            Some(0),
        );
        let order = reg.pick_targets("v", &routing, None);
        assert_eq!(order, vec!["a"]);
    }

    #[test]
    fn empty_targets_yields_empty_order() {
        let reg = RoutingRegistry::new();
        let routing = r(RoutingStrategy::Failover, vec![], None);
        assert!(reg.pick_targets("v", &routing, None).is_empty());
    }

    #[test]
    fn is_retryable_distinguishes_4xx_from_other_failures() {
        assert!(!is_retryable(
            &BridgeError::upstream_status(400, "bad request"),
            false,
            &[]
        ));
        assert!(!is_retryable(
            &BridgeError::upstream_status(429, "rate limited"),
            false,
            &[]
        ));
        assert!(is_retryable(
            &BridgeError::upstream_status(429, "rate limited"),
            true,
            &[]
        ));
        assert!(is_retryable(
            &BridgeError::upstream_status(502, "bad gateway"),
            false,
            &[]
        ));
        assert!(is_retryable(
            &BridgeError::Timeout {
                cause: String::new(),
                elapsed_ms: 1
            },
            false,
            &[]
        ));
        assert!(is_retryable(
            &BridgeError::Transport("conn".into()),
            false,
            &[]
        ));
        assert!(is_retryable(
            &BridgeError::UpstreamDecode("x".into()),
            false,
            &[]
        ));
        assert!(is_retryable(
            &BridgeError::Config("bad key".into()),
            false,
            &[]
        ));
        assert!(is_retryable(&BridgeError::StreamAborted, false, &[]));
        // #367: customer-fixable config is a 4xx — not retryable.
        assert!(!is_retryable(
            &BridgeError::InvalidUpstreamConfig("no api_base".into()),
            false,
            &[]
        ));
    }

    /// AISIX-Cloud#1012: `fallback_on_statuses` opts specific upstream
    /// status codes into retry/failover. The list is additive — codes not
    /// listed keep the default classification — and it never resurrects
    /// non-status failures (customer-fixable config stays terminal).
    #[test]
    fn fallback_on_statuses_opts_specific_codes_into_retry() {
        // A listed 4xx becomes retryable.
        assert!(is_retryable(
            &BridgeError::upstream_status(408, "request timeout"),
            false,
            &[408, 409]
        ));
        assert!(is_retryable(
            &BridgeError::upstream_status(409, "conflict"),
            false,
            &[408, 409]
        ));
        // Codes NOT in the list keep the default: terminal.
        assert!(!is_retryable(
            &BridgeError::upstream_status(422, "unprocessable"),
            false,
            &[408, 409]
        ));
        assert!(!is_retryable(
            &BridgeError::upstream_status(400, "bad request"),
            false,
            &[408, 409]
        ));
        // 429 in the list works without retry_on_429.
        assert!(is_retryable(
            &BridgeError::upstream_status(429, "rate limited"),
            false,
            &[429]
        ));
        // 5xx stays retryable whether or not listed.
        assert!(is_retryable(
            &BridgeError::upstream_status(503, "unavailable"),
            false,
            &[408]
        ));
        // The list is status-scoped: it never affects non-status errors.
        assert!(!is_retryable(
            &BridgeError::InvalidUpstreamConfig("no api_base".into()),
            false,
            &[400, 401, 403]
        ));
    }

    // ── retry_backoff ─────────────────────────────────────────────
    #[test]
    fn retry_backoff_zero_is_no_wait() {
        assert_eq!(retry_backoff(0), Duration::ZERO);
    }

    #[test]
    fn retry_backoff_grows_exponentially_and_caps() {
        // The exponential FLOOR (delay minus the additive jitter) must be
        // base*2^(retry-1), capped. Sample many times: the minimum observed
        // delay tracks the floor and never exceeds floor + jitter ceiling.
        let cases = [
            (1u32, 250u64), // 250 * 2^0
            (2, 500),       // 250 * 2^1
            (3, 1000),      // 250 * 2^2
            (4, 2000),      // 250 * 2^3 = 2000 (== cap)
            (5, 2000),      // capped
            (50, 2000),     // capped, no overflow
        ];
        for (retry, floor) in cases {
            let mut min = u64::MAX;
            let mut max = 0u64;
            for _ in 0..2000 {
                let ms = retry_backoff(retry).as_millis() as u64;
                min = min.min(ms);
                max = max.max(ms);
            }
            assert!(min >= floor, "retry {retry}: min {min} < floor {floor}");
            assert!(
                max <= floor + 250,
                "retry {retry}: max {max} > floor {floor} + jitter 250",
            );
        }
    }

    // ── filter_attempt_models ─────────────────────────────────────
    fn am(id: &str) -> AttemptModel {
        let model: Model = serde_json::from_str(&format!(
            r#"{{
              "display_name": "{id}",
              "provider": "openai",
              "model_name": "gpt-4o-mini",
              "provider_key_id": "pk-{id}"
            }}"#
        ))
        .unwrap();
        AttemptModel {
            id: id.to_string(),
            model,
        }
    }

    // ── order_attempts_by_metric (least_cost) ─────────────────────
    fn am_with_cost(id: &str, input_per_1k: f64, output_per_1k: f64) -> AttemptModel {
        let model: Model = serde_json::from_str(&format!(
            r#"{{
              "display_name": "{id}",
              "provider": "openai",
              "model_name": "gpt-4o-mini",
              "provider_key_id": "pk-{id}",
              "cost": {{ "input_per_1k": {input_per_1k}, "output_per_1k": {output_per_1k} }}
            }}"#
        ))
        .unwrap();
        AttemptModel {
            id: id.to_string(),
            model,
        }
    }

    #[test]
    fn least_cost_orders_cheapest_first() {
        let t = crate::ModelRuntimeStatusTracker::new();
        let mut attempts = vec![
            am_with_cost("pricey", 10.0, 20.0), // 30 / 1K
            am_with_cost("cheap", 1.0, 2.0),    // 3 / 1K
            am_with_cost("mid", 5.0, 5.0),      // 10 / 1K
        ];
        order_attempts_by_metric(RoutingStrategy::LeastCost, &mut attempts, &t);
        let ids: Vec<&str> = attempts.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["cheap", "mid", "pricey"]);
    }

    #[test]
    fn least_cost_ranks_missing_cost_last_and_stably() {
        let t = crate::ModelRuntimeStatusTracker::new();
        let mut attempts = vec![
            am("no-cost-a"),                 // +∞
            am_with_cost("cheap", 1.0, 1.0), // 2 / 1K
            am("no-cost-b"),                 // +∞
        ];
        order_attempts_by_metric(RoutingStrategy::LeastCost, &mut attempts, &t);
        let ids: Vec<&str> = attempts.iter().map(|a| a.id.as_str()).collect();
        // Priced target first; equal (missing-cost) targets keep their
        // declaration order thanks to the stable sort.
        assert_eq!(ids, vec!["cheap", "no-cost-a", "no-cost-b"]);
    }

    #[test]
    fn non_metric_strategy_leaves_order_untouched() {
        let t = crate::ModelRuntimeStatusTracker::new();
        let mut attempts = vec![am_with_cost("b", 9.0, 9.0), am_with_cost("a", 1.0, 1.0)];
        order_attempts_by_metric(RoutingStrategy::Failover, &mut attempts, &t);
        let ids: Vec<&str> = attempts.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "a"]);
    }

    // ── order_attempts_by_metric (least_latency) ──────────────────
    #[test]
    fn least_latency_orders_fastest_first() {
        let t = crate::ModelRuntimeStatusTracker::new();
        t.record_latency("slow", 900);
        t.record_latency("fast", 50);
        t.record_latency("mid", 300);
        let mut attempts = vec![am("slow"), am("fast"), am("mid")];
        order_attempts_by_metric(RoutingStrategy::LeastLatency, &mut attempts, &t);
        let ids: Vec<&str> = attempts.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["fast", "mid", "slow"]);
    }

    #[test]
    fn least_latency_probes_unmeasured_targets_first() {
        let t = crate::ModelRuntimeStatusTracker::new();
        t.record_latency("measured", 100);
        // "unseen-a"/"unseen-b" have no samples → rank first (−∞), keeping
        // their declaration order via the stable sort.
        let mut attempts = vec![am("measured"), am("unseen-a"), am("unseen-b")];
        order_attempts_by_metric(RoutingStrategy::LeastLatency, &mut attempts, &t);
        let ids: Vec<&str> = attempts.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["unseen-a", "unseen-b", "measured"]);
    }

    #[test]
    fn record_latency_ewma_tracks_recent_samples() {
        let t = crate::ModelRuntimeStatusTracker::new();
        assert_eq!(t.latency_ewma_ms("m"), None);
        t.record_latency("m", 100);
        assert_eq!(t.latency_ewma_ms("m"), Some(100.0)); // first sample seeds
        t.record_latency("m", 200);
        // 0.3*200 + 0.7*100 = 130
        assert!((t.latency_ewma_ms("m").unwrap() - 130.0).abs() < 1e-9);
    }

    // ── order_attempts_by_metric (least_busy) ─────────────────────
    #[test]
    fn least_busy_orders_least_loaded_first() {
        let t = crate::ModelRuntimeStatusTracker::new();
        let _b1 = t.begin_in_flight("busy");
        let _b2 = t.begin_in_flight("busy"); // 2 in-flight
        let _m1 = t.begin_in_flight("mid"); // 1 in-flight
                                            // "idle" has 0 in-flight.
        let mut attempts = vec![am("busy"), am("idle"), am("mid")];
        order_attempts_by_metric(RoutingStrategy::LeastBusy, &mut attempts, &t);
        let ids: Vec<&str> = attempts.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["idle", "mid", "busy"]);
    }

    #[test]
    fn least_busy_cold_start_keeps_declaration_order() {
        let t = crate::ModelRuntimeStatusTracker::new();
        // All idle (0 in-flight) → stable sort preserves declaration order.
        let mut attempts = vec![am("a"), am("b"), am("c")];
        order_attempts_by_metric(RoutingStrategy::LeastBusy, &mut attempts, &t);
        let ids: Vec<&str> = attempts.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn in_flight_guard_increments_then_decrements_on_drop() {
        let t = crate::ModelRuntimeStatusTracker::new();
        assert_eq!(t.in_flight("m"), 0);
        let g1 = t.begin_in_flight("m");
        assert_eq!(t.in_flight("m"), 1);
        let g2 = t.begin_in_flight("m");
        assert_eq!(t.in_flight("m"), 2);
        drop(g1);
        assert_eq!(t.in_flight("m"), 1);
        drop(g2);
        assert_eq!(t.in_flight("m"), 0);
    }

    #[test]
    fn metric_strategy_pick_targets_returns_full_declaration_order() {
        let reg = RoutingRegistry::new();
        let routing = r(
            RoutingStrategy::LeastCost,
            vec![
                RoutingTarget::new("a"),
                RoutingTarget::new("b"),
                RoutingTarget::new("c"),
            ],
            Some(1), // truncation is deferred to resolve_attempt_models
        );
        // Ranking needs resolved Models, so pick_targets hands back every
        // target untouched regardless of max_fallbacks.
        assert_eq!(reg.pick_targets("v", &routing, None), vec!["a", "b", "c"]);
    }

    #[test]
    fn healthy_only_returns_all_healthy() {
        let t = crate::ModelRuntimeStatusTracker::new();
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, WhenAllUnavailablePolicy::Fail) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 2);
            }
            other => panic!(
                "expected Selected, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn cooldown_skipped_when_healthy_present() {
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_cooldown("a", Duration::from_secs(30), "retryable_failure");
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, WhenAllUnavailablePolicy::Fail) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 1);
                assert_eq!(list[0].id, "b");
            }
            _ => panic!("expected Selected"),
        }
    }

    #[test]
    fn all_unhealthy_fail_policy_returns_retry_after_hint() {
        // H3 contract: every candidate background-unhealthy, no
        // cooldown timer → return 503 + fallback Retry-After (30s
        // default). The dispatch loop converts this to a
        // ProxyError::AllCandidatesUnavailable.
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_unhealthy("a", Some(503), "background_check_failed");
        t.mark_unhealthy("b", Some(503), "background_check_failed");
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, WhenAllUnavailablePolicy::Fail) {
            FilterOutcome::AllUnhealthy { retry_after_secs } => {
                assert_eq!(retry_after_secs, Some(30));
            }
            _ => panic!("expected AllUnhealthy"),
        }
    }

    #[test]
    fn one_cooldown_with_all_else_unhealthy_keeps_the_cooldown_candidate() {
        // Mixed scenario: candidates a/b are background-unhealthy, c
        // is in cooldown. The filter should pick c (cooldown beats
        // unhealthy), not fail.
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_unhealthy("a", Some(503), "background_check_failed");
        t.mark_unhealthy("b", Some(503), "background_check_failed");
        t.mark_cooldown("c", Duration::from_secs(30), "x");
        let attempts = vec![am("a"), am("b"), am("c")];
        match filter_attempt_models(&t, attempts, WhenAllUnavailablePolicy::Fail) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 1);
                assert_eq!(list[0].id, "c");
            }
            _ => panic!("expected Selected with cooldown candidate"),
        }
    }

    #[test]
    fn all_unhealthy_try_anyway_policy_returns_full_list() {
        // Legacy opt-in: send to all candidates regardless.
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_unhealthy("a", Some(503), "background_check_failed");
        t.mark_unhealthy("b", Some(503), "background_check_failed");
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, WhenAllUnavailablePolicy::TryAnyway) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 2);
            }
            _ => panic!("expected Selected under TryAnyway policy"),
        }
    }

    #[test]
    fn cooldown_no_unhealthy_returns_cooldown_candidates() {
        // No healthy, no unhealthy — all candidates have a cooldown
        // timer set. Routing should still pick from them (better than
        // erroring out when we don't have evidence anyone is *broken*).
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_cooldown("a", Duration::from_secs(30), "x");
        t.mark_cooldown("b", Duration::from_secs(30), "x");
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, WhenAllUnavailablePolicy::Fail) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 2);
            }
            _ => panic!("expected Selected for cooldown-only"),
        }
    }
}
