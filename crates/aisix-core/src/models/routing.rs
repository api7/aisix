//! Virtual-routing config attached to a [`Model`](super::Model).
//!
//! When a Model carries a `routing` block, the proxy treats it as a
//! pointer to other Models. Per-request the proxy picks one target via
//! the configured strategy and dispatches through that target's bridge.
//! Failures may retry the current target and then fall back to later
//! targets.
//!
//! Three strategies (spec §3):
//! - `round_robin`: cycle through targets in declaration order.
//! - `weighted`: pick a target with probability proportional to its
//!   `weight`; falls back to round-robin when weights are missing.
//! - `failover`: always start at the first target; only move down the
//!   list on failure.

use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    /// Cycle through targets in declaration order.
    RoundRobin,
    /// Pick targets by configured weight. Missing target weights fall back to 1.
    Weighted,
    /// Always start with the first target and move to later targets only
    /// after failure.
    #[default]
    Failover,
}

/// One destination in a routing configuration. `model` references a direct model alias.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RoutingTarget {
    /// Model alias for a direct model that can receive routed traffic.
    pub model: String,
    /// Target weight for `weighted` routing. Other strategies ignore this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<u32>,
}

impl RoutingTarget {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            weight: None,
        }
    }

    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = Some(weight);
        self
    }

    pub fn weight_or_default(&self) -> u32 {
        self.weight.unwrap_or(1)
    }
}

/// Behavior when every routing target is filtered out by runtime health or cooldown state.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum OnAllFilteredPolicy {
    /// Return `503` with a fixed `Retry-After` hint.
    #[default]
    Fail,
    /// Route to the original candidate list in declaration order even
    /// when all targets were filtered by health or cooldown status. Use
    /// only when maintaining availability is preferred over avoiding
    /// recently unhealthy targets.
    OriginalOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Routing {
    /// Strategy used to select a target for each request.
    #[serde(default)]
    pub strategy: RoutingStrategy,
    /// Ordered set of direct models available to this routing model.
    pub targets: Vec<RoutingTarget>,
    /// Retry attempts on the current target before failing over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retries: Option<u32>,
    /// Max number of later targets to attempt after the initial target fails permanently. When omitted, all later targets may be attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fallbacks: Option<u32>,
    /// Whether upstream 429 participates in retries and failover.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_on_429: Option<bool>,
    /// Policy to apply when runtime status filtering removes every candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_all_filtered: Option<OnAllFilteredPolicy>,
}

impl Routing {
    pub fn retries_or_default(&self) -> usize {
        self.retries.unwrap_or(0) as usize
    }

    pub fn max_fallbacks_or_default(&self) -> usize {
        let later_targets = self.targets.len().saturating_sub(1);
        match self.max_fallbacks {
            Some(n) => (n as usize).min(later_targets),
            None => later_targets,
        }
    }

    pub fn retry_on_429_or_default(&self) -> bool {
        self.retry_on_429.unwrap_or(false)
    }

    pub fn on_all_filtered_or_default(&self) -> OnAllFilteredPolicy {
        self.on_all_filtered.unwrap_or_default()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_full_routing_block() {
        let json = r#"{
            "strategy": "weighted",
            "targets": [
                {"model": "primary", "weight": 90},
                {"model": "backup",  "weight": 10}
            ],
            "retries": 2,
            "max_fallbacks": 1,
            "retry_on_429": true
        }"#;
        let r: Routing = serde_json::from_str(json).unwrap();
        assert_eq!(r.strategy, RoutingStrategy::Weighted);
        assert_eq!(r.targets.len(), 2);
        assert_eq!(r.targets[0].model, "primary");
        assert_eq!(r.targets[0].weight_or_default(), 90);
        assert_eq!(r.retries_or_default(), 2);
        assert_eq!(r.max_fallbacks_or_default(), 1);
        assert!(r.retry_on_429_or_default());
    }

    #[test]
    fn strategy_defaults_to_failover() {
        let r: Routing =
            serde_json::from_str(r#"{"targets":[{"model":"a"},{"model":"b"}]}"#).unwrap();
        assert_eq!(r.strategy, RoutingStrategy::Failover);
        assert_eq!(r.retries_or_default(), 0);
        assert_eq!(r.max_fallbacks_or_default(), 1);
        assert!(!r.retry_on_429_or_default());
    }

    #[test]
    fn max_fallbacks_zero_disables_failover() {
        let r = Routing {
            strategy: RoutingStrategy::RoundRobin,
            targets: vec![RoutingTarget::new("a"), RoutingTarget::new("b")],
            retries: Some(0),
            max_fallbacks: Some(0),
            retry_on_429: None,
            on_all_filtered: None,
        };
        assert_eq!(r.max_fallbacks_or_default(), 0);
    }

    #[test]
    fn max_fallbacks_clamps_to_later_targets() {
        let r = Routing {
            strategy: RoutingStrategy::Failover,
            targets: vec![RoutingTarget::new("a")],
            retries: None,
            max_fallbacks: Some(99),
            retry_on_429: None,
            on_all_filtered: None,
        };
        assert_eq!(r.max_fallbacks_or_default(), 0);
    }

    #[test]
    fn on_all_filtered_defaults_to_fail() {
        let r: Routing = serde_json::from_str(r#"{"targets":[{"model":"a"}]}"#).unwrap();
        assert_eq!(r.on_all_filtered_or_default(), OnAllFilteredPolicy::Fail);
    }

    #[test]
    fn on_all_filtered_parses_original_order() {
        let r: Routing = serde_json::from_str(
            r#"{"targets":[{"model":"a"}],"on_all_filtered":"original_order"}"#,
        )
        .unwrap();
        assert_eq!(
            r.on_all_filtered_or_default(),
            OnAllFilteredPolicy::OriginalOrder
        );
    }

    #[test]
    fn on_all_filtered_rejects_unknown_value() {
        let r: Result<Routing, _> =
            serde_json::from_str(r#"{"targets":[{"model":"a"}],"on_all_filtered":"explode"}"#);
        assert!(r.is_err());
    }

    #[test]
    fn missing_weight_defaults_to_one() {
        let t = RoutingTarget::new("x");
        assert_eq!(t.weight_or_default(), 1);
    }

    #[test]
    fn rejects_unknown_routing_fields() {
        let r: Result<Routing, _> =
            serde_json::from_str(r#"{"strategy":"failover","targets":[{"model":"a"}],"foo":1}"#);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_unknown_target_fields() {
        let r: Result<RoutingTarget, _> =
            serde_json::from_str(r#"{"model":"a","weight":2,"extra":true}"#);
        assert!(r.is_err());
    }
}
