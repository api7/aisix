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

use aisix_core::{Routing, RoutingStrategy, RoutingTarget};
use aisix_gateway::BridgeError;
use dashmap::DashMap;
use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering};

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
            &BridgeError::UpstreamStatus {
                status: 400,
                message: "bad request".into(),
            },
            false
        ));
        assert!(!is_retryable(
            &BridgeError::UpstreamStatus {
                status: 429,
                message: "rate limited".into(),
            },
            false
        ));
        assert!(is_retryable(
            &BridgeError::UpstreamStatus {
                status: 429,
                message: "rate limited".into(),
            },
            true
        ));
        assert!(is_retryable(
            &BridgeError::UpstreamStatus {
                status: 502,
                message: "bad gateway".into(),
            },
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
    }
}
