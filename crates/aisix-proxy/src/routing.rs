//! Per-virtual-model routing state + target selection.
//!
//! When a request lands on a Model with `routing` configured, the proxy
//! asks the [`RoutingRegistry`] for an iterator of underlying target
//! Model names in attempt-order. The registry owns the per-virtual-
//! model state (round-robin counter, weighted PRNG seed); selection
//! itself is pure given that state.
//!
//! Strategies (spec §3.5):
//! - **failover**: always start at `targets[0]`, walk forward on failure.
//! - **round_robin**: each *new* request advances a per-model counter
//!   so callers spread evenly across targets.
//! - **weighted**: pick a starting target with probability proportional
//!   to `weight`, then walk forward on failure (weights only affect the
//!   *first* target choice — once we're falling back, order is positional).

use aisix_core::{
    AisixSnapshot, Model, OnAllFilteredPolicy, Routing, RoutingStrategy, RoutingTarget,
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
/// fallback for the `on_all_filtered: fail` path.
const FALLBACK_ALL_UNHEALTHY_RETRY_AFTER: Duration = Duration::from_secs(30);

/// Whether a Bridge error is retryable at all, optionally treating 429
/// as retryable. Non-429 4xx is the caller's mistake — retrying won't
/// help and may amplify damage. Everything else (5xx, timeout,
/// transport, decode, config, stream abort) gets the retry/failover path.
pub fn is_retryable(err: &BridgeError, retry_on_429: bool) -> bool {
    match err {
        BridgeError::UpstreamStatus { status, .. } => {
            if *status == 429 {
                return retry_on_429;
            }
            !(400..500).contains(status)
        }
        // Customer-fixable config (#367) is the caller's mistake —
        // retrying or failing over won't help, same as a non-429 4xx.
        BridgeError::InvalidUpstreamConfig(_) => false,
        BridgeError::Timeout { .. }
        | BridgeError::Transport(_)
        | BridgeError::UpstreamDecode(_)
        | BridgeError::Config(_)
        | BridgeError::StreamAborted => true,
    }
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
    pub fn pick_targets(&self, virtual_name: &str, routing: &Routing) -> Vec<String> {
        if routing.targets.is_empty() {
            return Vec::new();
        }
        let start = self.starting_index(virtual_name, routing);
        attempt_order(
            &routing.targets,
            start,
            routing.max_fallbacks_or_default() + 1,
        )
    }

    fn starting_index(&self, virtual_name: &str, routing: &Routing) -> usize {
        match routing.strategy {
            RoutingStrategy::Failover => 0,
            RoutingStrategy::RoundRobin => self.advance_cursor(virtual_name, routing.targets.len()),
            RoutingStrategy::Weighted => weighted_pick(&routing.targets),
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
fn weighted_pick(targets: &[RoutingTarget]) -> usize {
    let total: u64 = targets.iter().map(|t| t.weight_or_default() as u64).sum();
    if total == 0 {
        return 0;
    }
    let pick = rand::thread_rng().gen_range(0..total);
    let mut acc: u64 = 0;
    for (i, t) in targets.iter().enumerate() {
        acc += t.weight_or_default() as u64;
        if pick < acc {
            return i;
        }
    }
    targets.len() - 1
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
    /// routing model is configured with `on_all_filtered: fail`. The
    /// caller should surface a 503 with the supplied Retry-After hint
    /// (in seconds), if any.
    AllUnhealthy { retry_after_secs: Option<u64> },
}

pub(crate) fn filter_attempt_models(
    runtime_status: &crate::ModelRuntimeStatusTracker,
    attempts: Vec<AttemptModel>,
    policy: OnAllFilteredPolicy,
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
        OnAllFilteredPolicy::Fail => FilterOutcome::AllUnhealthy {
            retry_after_secs: Some(FALLBACK_ALL_UNHEALTHY_RETRY_AFTER.as_secs()),
        },
        OnAllFilteredPolicy::OriginalOrder => FilterOutcome::Selected(attempts),
    }
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
) -> Result<Vec<AttemptModel>, ProxyError> {
    let Some(routing) = virtual_model.routing.as_ref() else {
        return Ok(vec![AttemptModel {
            id: virtual_id.to_string(),
            model: virtual_model.clone(),
        }]);
    };

    let names = routing_registry.pick_targets(virtual_name, routing);
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
    match filter_attempt_models(
        runtime_status,
        resolved,
        routing.on_all_filtered_or_default(),
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
            on_all_filtered: None,
        }
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
            let order = reg.pick_targets("v", &routing);
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
            let order = reg.pick_targets("v", &routing);
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
        assert_eq!(reg.pick_targets("v1", &routing)[0], "a");
        assert_eq!(reg.pick_targets("v2", &routing)[0], "a");
        assert_eq!(reg.pick_targets("v1", &routing)[0], "b");
        assert_eq!(reg.pick_targets("v2", &routing)[0], "b");
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
        assert_eq!(reg.pick_targets("v", &routing), vec!["a", "b", "c"]);
        // Second call starts at b → b, c, a
        assert_eq!(reg.pick_targets("v", &routing), vec!["b", "c", "a"]);
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
        let order = reg.pick_targets("v", &routing);
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
        assert_eq!(weighted_pick(&targets), 0);
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
        let a_count = (0..n).filter(|_| weighted_pick(&targets) == 0).count();
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
        let b_count = (0..n).filter(|_| weighted_pick(&targets) == 1).count();
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
        let a_count = (0..n).filter(|_| weighted_pick(&targets) == 0).count();
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
            counts[weighted_pick(&targets)] += 1;
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
        let b_count = (0..n).filter(|_| weighted_pick(&targets) == 1).count();
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
        let order = reg.pick_targets("v", &routing);
        assert_eq!(order, vec!["a"]);
    }

    #[test]
    fn empty_targets_yields_empty_order() {
        let reg = RoutingRegistry::new();
        let routing = r(RoutingStrategy::Failover, vec![], None);
        assert!(reg.pick_targets("v", &routing).is_empty());
    }

    #[test]
    fn is_retryable_distinguishes_4xx_from_other_failures() {
        assert!(!is_retryable(
            &BridgeError::upstream_status(400, "bad request"),
            false
        ));
        assert!(!is_retryable(
            &BridgeError::upstream_status(429, "rate limited"),
            false
        ));
        assert!(is_retryable(
            &BridgeError::upstream_status(429, "rate limited"),
            true
        ));
        assert!(is_retryable(
            &BridgeError::upstream_status(502, "bad gateway"),
            false
        ));
        assert!(is_retryable(&BridgeError::Timeout { elapsed_ms: 1 }, false));
        assert!(is_retryable(&BridgeError::Transport("conn".into()), false));
        assert!(is_retryable(
            &BridgeError::UpstreamDecode("x".into()),
            false
        ));
        assert!(is_retryable(&BridgeError::Config("bad key".into()), false));
        assert!(is_retryable(&BridgeError::StreamAborted, false));
        // #367: customer-fixable config is a 4xx — not retryable.
        assert!(!is_retryable(
            &BridgeError::InvalidUpstreamConfig("no api_base".into()),
            false
        ));
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

    #[test]
    fn healthy_only_returns_all_healthy() {
        let t = crate::ModelRuntimeStatusTracker::new();
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::Fail) {
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
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::Fail) {
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
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::Fail) {
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
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::Fail) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 1);
                assert_eq!(list[0].id, "c");
            }
            _ => panic!("expected Selected with cooldown candidate"),
        }
    }

    #[test]
    fn all_unhealthy_original_order_policy_returns_full_list() {
        // Legacy opt-in: send to all candidates regardless.
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_unhealthy("a", Some(503), "background_check_failed");
        t.mark_unhealthy("b", Some(503), "background_check_failed");
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::OriginalOrder) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 2);
            }
            _ => panic!("expected Selected under OriginalOrder policy"),
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
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::Fail) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 2);
            }
            _ => panic!("expected Selected for cooldown-only"),
        }
    }
}
