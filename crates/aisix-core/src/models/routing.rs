//! Virtual-routing config attached to a [`Model`](super::Model).
//!
//! When a Model carries a `routing` block, the proxy treats it as a
//! pointer to other Models. Per-request the proxy picks one target via
//! the configured strategy and dispatches through that target's bridge.
//! Failures may retry the current target and then fall back to later
//! targets.
//!
//! Positional strategies (spec §3) pick a *starting* target, then walk
//! forward on failure:
//! - `round_robin`: cycle through targets in declaration order.
//! - `weighted`: pick a target with probability proportional to its
//!   `weight`; falls back to round-robin when weights are missing.
//! - `failover`: always start at the first target; only move down the
//!   list on failure.
//!
//! Metric-ordered strategies rank *all* targets by a runtime signal and
//! attempt them best-first, falling forward down the ranked order:
//! - `least_cost`: cheapest target first, by the target model's `cost`
//!   (combined input+output per-1K price). Targets without a `cost` rank
//!   last.
//! - `least_latency`: fastest target first, by a moving average of recent
//!   observed upstream latency (time-to-first-token for streaming). Targets
//!   with no latency samples yet rank first so they get probed.
//! - `least_busy`: least-loaded target first, by the number of in-flight
//!   requests currently dispatched to each target.
//!
//! See [`RoutingStrategy::is_metric_based`].

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
    /// Rank targets cheapest-first by the target model's `cost` (combined
    /// input+output per-1K price), then fall forward. Targets without a
    /// configured `cost` rank last.
    LeastCost,
    /// Rank targets fastest-first by a moving average of recent observed
    /// upstream latency (time-to-first-token for streaming), then fall
    /// forward. Targets with no samples yet rank first so they get probed.
    LeastLatency,
    /// Rank targets least-loaded-first by the number of in-flight requests
    /// currently dispatched to each target, then fall forward.
    LeastBusy,
}

impl RoutingStrategy {
    /// Whether the strategy ranks the full target set by a runtime metric
    /// (rather than picking a start index and walking positionally). These
    /// strategies are ordered after target resolution, where each target's
    /// Model and runtime state are available.
    pub fn is_metric_based(&self) -> bool {
        matches!(
            self,
            RoutingStrategy::LeastCost | RoutingStrategy::LeastLatency | RoutingStrategy::LeastBusy
        )
    }
}

/// One destination in a routing configuration. `model` references a direct model alias.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RoutingTarget {
    /// Model alias for a direct model that can receive routed traffic.
    #[schemars(length(min = 1))]
    pub model: String,
    /// Target weight for `weighted` routing. Other strategies ignore this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<u32>,
    /// Tags for tag/metadata-conditional routing. When a request carries
    /// routing tags, only targets whose tags intersect the request's are
    /// eligible; a target tagged `"default"` is the fallback used when nothing
    /// matches and for untagged requests. Absent/empty means the target opts
    /// out of tag filtering (eligible only via the default fallback once any
    /// sibling target is tagged). The configured strategy then orders whatever
    /// set survives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(length(min = 1)))]
    pub tags: Option<Vec<String>>,
}

/// Reserved tag marking a target as the fallback when no tag matches.
pub const DEFAULT_ROUTING_TAG: &str = "default";

impl RoutingTarget {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            weight: None,
            tags: None,
        }
    }

    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = Some(weight);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    pub fn weight_or_default(&self) -> u32 {
        self.weight.unwrap_or(1)
    }

    /// True if this target carries at least one tag.
    pub fn has_tags(&self) -> bool {
        self.tags.as_ref().is_some_and(|t| !t.is_empty())
    }

    /// True if this target is the `"default"` fallback.
    pub fn is_default_target(&self) -> bool {
        self.tags
            .as_ref()
            .is_some_and(|t| t.iter().any(|tag| tag == DEFAULT_ROUTING_TAG))
    }

    /// True if any of this target's tags appears in `request_tags` (match-any).
    pub fn matches_request_tags(&self, request_tags: &[String]) -> bool {
        self.tags
            .as_ref()
            .is_some_and(|t| t.iter().any(|tag| request_tags.iter().any(|r| r == tag)))
    }
}

/// Behavior when every routing target is unavailable because of runtime health or cooldown state.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum WhenAllUnavailablePolicy {
    /// Return `503` with a fixed `Retry-After` hint.
    #[default]
    Fail,
    /// Try every target in declaration order even when all of them are
    /// currently unavailable because of health or cooldown status. Use
    /// only when maintaining availability is preferred over avoiding
    /// recently unhealthy targets.
    TryAnyway,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Routing {
    /// Strategy used to select a target for each request.
    #[serde(default)]
    pub strategy: RoutingStrategy,
    /// Ordered set of direct models available to this routing model.
    #[schemars(length(min = 1))]
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
    /// Policy to apply when every target is unavailable because of runtime health or cooldown state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when_all_unavailable: Option<WhenAllUnavailablePolicy>,
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

    pub fn when_all_unavailable_or_default(&self) -> WhenAllUnavailablePolicy {
        self.when_all_unavailable.unwrap_or_default()
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
            when_all_unavailable: None,
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
            when_all_unavailable: None,
        };
        assert_eq!(r.max_fallbacks_or_default(), 0);
    }

    #[test]
    fn when_all_unavailable_defaults_to_fail() {
        let r: Routing = serde_json::from_str(r#"{"targets":[{"model":"a"}]}"#).unwrap();
        assert_eq!(
            r.when_all_unavailable_or_default(),
            WhenAllUnavailablePolicy::Fail
        );
    }

    #[test]
    fn when_all_unavailable_parses_try_anyway() {
        let r: Routing = serde_json::from_str(
            r#"{"targets":[{"model":"a"}],"when_all_unavailable":"try_anyway"}"#,
        )
        .unwrap();
        assert_eq!(
            r.when_all_unavailable_or_default(),
            WhenAllUnavailablePolicy::TryAnyway
        );
    }

    #[test]
    fn when_all_unavailable_rejects_unknown_value() {
        let r: Result<Routing, _> =
            serde_json::from_str(r#"{"targets":[{"model":"a"}],"when_all_unavailable":"explode"}"#);
        assert!(r.is_err());
    }

    #[test]
    fn missing_weight_defaults_to_one() {
        let t = RoutingTarget::new("x");
        assert_eq!(t.weight_or_default(), 1);
    }

    #[test]
    fn target_tags_parse_and_predicates() {
        let r: Routing = serde_json::from_str(
            r#"{"targets":[{"model":"a","tags":["eu","premium"]},{"model":"b","tags":["default"]},{"model":"c"}]}"#,
        )
        .unwrap();
        assert!(r.targets[0].has_tags());
        assert!(!r.targets[0].is_default_target());
        assert!(r.targets[0].matches_request_tags(&["premium".into()]));
        assert!(!r.targets[0].matches_request_tags(&["apac".into()]));
        assert!(r.targets[1].is_default_target());
        assert!(!r.targets[2].has_tags());
        assert!(!r.targets[2].matches_request_tags(&["eu".into()]));
    }

    #[test]
    fn parses_metric_strategies() {
        let cost: Routing = serde_json::from_str(
            r#"{"strategy":"least_cost","targets":[{"model":"a"},{"model":"b"}]}"#,
        )
        .unwrap();
        assert_eq!(cost.strategy, RoutingStrategy::LeastCost);
        let latency: Routing = serde_json::from_str(
            r#"{"strategy":"least_latency","targets":[{"model":"a"},{"model":"b"}]}"#,
        )
        .unwrap();
        assert_eq!(latency.strategy, RoutingStrategy::LeastLatency);
        let busy: Routing = serde_json::from_str(
            r#"{"strategy":"least_busy","targets":[{"model":"a"},{"model":"b"}]}"#,
        )
        .unwrap();
        assert_eq!(busy.strategy, RoutingStrategy::LeastBusy);
    }

    #[test]
    fn is_metric_based_classification() {
        assert!(RoutingStrategy::LeastCost.is_metric_based());
        assert!(RoutingStrategy::LeastLatency.is_metric_based());
        assert!(RoutingStrategy::LeastBusy.is_metric_based());
        assert!(!RoutingStrategy::Failover.is_metric_based());
        assert!(!RoutingStrategy::RoundRobin.is_metric_based());
        assert!(!RoutingStrategy::Weighted.is_metric_based());
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
