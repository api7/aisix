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
//!   *first* attempt — once we're falling back, order is positional).

use aisix_core::{Routing, RoutingStrategy, RoutingTarget};
use aisix_gateway::BridgeError;
use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Whether a Bridge error is retryable across routing targets.
/// 4xx is the caller's mistake — retrying to a different target won't
/// help and may amplify damage. Everything else (5xx, timeout, transport,
/// decode, config, stream abort) gets the fallback path.
pub fn is_retryable(err: &BridgeError) -> bool {
    match err {
        BridgeError::UpstreamStatus { status, .. } => !(400..500).contains(status),
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

    /// Pick the attempt order for one request. The first element is the
    /// initial target; subsequent elements are the fallback chain (in
    /// declaration order, wrapping if needed). Length is bounded by
    /// `routing.retry_budget_or_default()`.
    pub fn pick_order(&self, virtual_name: &str, routing: &Routing) -> Vec<String> {
        let budget = routing.retry_budget_or_default();
        if budget == 0 || routing.targets.is_empty() {
            return Vec::new();
        }
        let start = self.starting_index(virtual_name, routing);
        attempt_order(&routing.targets, start, budget)
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

/// Build the attempt-order vector starting at `start_idx`, walking forward
/// (wrap-around) for `budget` distinct entries.
fn attempt_order(targets: &[RoutingTarget], start_idx: usize, budget: usize) -> Vec<String> {
    let n = targets.len();
    let mut order = Vec::with_capacity(budget);
    for i in 0..budget {
        let t = &targets[(start_idx + i) % n];
        order.push(t.model.clone());
    }
    order
}

/// Pick an index by weighted-random. Ignores zero weights; a fully-zero
/// list falls back to index 0 deterministically.
fn weighted_pick(targets: &[RoutingTarget]) -> usize {
    let total: u64 = targets.iter().map(|t| t.weight_or_default() as u64).sum();
    if total == 0 {
        return 0;
    }
    // We don't need a strong PRNG here — just enough entropy to spread
    // requests roughly per weights. Use the system clock's nanos as a
    // cheap entropy source so we avoid pulling in `rand` crate just
    // for one call site.
    let pick = entropy() % total;
    let mut acc: u64 = 0;
    for (i, t) in targets.iter().enumerate() {
        acc += t.weight_or_default() as u64;
        if pick < acc {
            return i;
        }
    }
    targets.len() - 1
}

fn entropy() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0)
        // Mix in something monotonic so two calls in the same nanosecond
        // diverge — `Instant` provides that without `unsafe`.
        .wrapping_add(std::time::Instant::now().elapsed().as_nanos() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::{Routing, RoutingStrategy, RoutingTarget};

    fn r(strategy: RoutingStrategy, targets: Vec<RoutingTarget>, budget: Option<u32>) -> Routing {
        Routing {
            strategy,
            targets,
            retry_budget: budget,
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
            let order = reg.pick_order("v", &routing);
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
            let order = reg.pick_order("v", &routing);
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
        assert_eq!(reg.pick_order("v1", &routing)[0], "a");
        assert_eq!(reg.pick_order("v2", &routing)[0], "a");
        assert_eq!(reg.pick_order("v1", &routing)[0], "b");
        assert_eq!(reg.pick_order("v2", &routing)[0], "b");
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
            Some(0), // 0 → use full target count (3 attempts)
        );
        // First call starts at a → a, b, c
        assert_eq!(reg.pick_order("v", &routing), vec!["a", "b", "c"]);
        // Second call starts at b → b, c, a
        assert_eq!(reg.pick_order("v", &routing), vec!["b", "c", "a"]);
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
            Some(0),
        );
        // Across many trials, "a" should dominate as the starting pick.
        // We don't assert exact distribution (entropy is sub-second
        // wall-clock); we just assert correctness of the *order* shape:
        // exactly two attempts, distinct targets, both targets covered.
        let order = reg.pick_order("v", &routing);
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

    #[test]
    fn budget_one_disables_fallback() {
        let reg = RoutingRegistry::new();
        let routing = r(
            RoutingStrategy::Failover,
            vec![RoutingTarget::new("a"), RoutingTarget::new("b")],
            Some(1),
        );
        let order = reg.pick_order("v", &routing);
        assert_eq!(order, vec!["a"]);
    }

    #[test]
    fn empty_targets_yields_empty_order() {
        let reg = RoutingRegistry::new();
        let routing = r(RoutingStrategy::Failover, vec![], None);
        assert!(reg.pick_order("v", &routing).is_empty());
    }

    #[test]
    fn is_retryable_distinguishes_4xx_from_other_failures() {
        assert!(!is_retryable(&BridgeError::UpstreamStatus {
            status: 400,
            message: "bad request".into(),
        }));
        assert!(!is_retryable(&BridgeError::UpstreamStatus {
            status: 429,
            message: "rate limited".into(),
        }));
        assert!(is_retryable(&BridgeError::UpstreamStatus {
            status: 502,
            message: "bad gateway".into(),
        }));
        assert!(is_retryable(&BridgeError::Timeout { elapsed_ms: 1 }));
        assert!(is_retryable(&BridgeError::Transport("conn".into())));
        assert!(is_retryable(&BridgeError::UpstreamDecode("x".into())));
        assert!(is_retryable(&BridgeError::Config("bad key".into())));
        assert!(is_retryable(&BridgeError::StreamAborted));
    }
}
