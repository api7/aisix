//! Virtual-routing config attached to a [`Model`](super::Model).
//!
//! When a Model carries a `routing` block, the proxy treats it as a
//! pointer to other Models. Per-request the proxy picks one target via
//! the configured strategy and dispatches through that target's bridge.
//! Failures may retry the current target and then fall back to later
//! targets.
//!
//! Targets partition into **priority tiers** first (`priority`, higher
//! value preferred — the APISIX node-priority convention): the strategy
//! orders targets *within* each tier, tiers concatenate best-first, and a
//! lower tier is only reached when every higher-tier target failed or is
//! unavailable. All targets default to priority `0`, so priority is inert
//! unless configured.
//!
//! Positional strategies pick a *starting* target per tier, then walk
//! forward on failure:
//! - `round_robin`: smooth weighted round-robin over target `weight`s
//!   (equal weights degrade to a plain declaration-order cycle).
//! - `consistent_hash`: ketama-style consistent hashing of the request's
//!   hash key (see [`HashOnSource`]) over the tier's targets, `weight`
//!   scaling each target's share of the ring. The same key keeps landing
//!   on the same target; on failure the walk follows the ring, so only
//!   the failed target's keys move.
//! - `failover`: always start at the first target; only move down the
//!   list on failure. Declaration order is the priority order.
//!
//! Metric-ordered strategies rank targets by a runtime signal within each
//! tier and attempt them best-first, falling forward down the ranked order:
//! - `least_cost`: cheapest target first, by the target model's `cost`
//!   (combined input+output per-1K price). Targets without a `cost` rank
//!   last.
//! - `least_latency`: fastest target first, by a moving average of recent
//!   observed upstream latency (time-to-first-token for streaming). Targets
//!   with no latency samples yet rank first so they get probed.
//! - `least_busy`: least-loaded target first, by in-flight requests
//!   divided by target `weight` (the APISIX least_conn score).
//!
//! See [`RoutingStrategy::is_metric_based`].

use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    /// Smooth weighted round-robin over target `weight`s. Equal (or absent)
    /// weights degrade to a plain declaration-order cycle.
    RoundRobin,
    /// Ketama-style consistent hashing of the request's hash key (see
    /// `hash_on`) over the targets, `weight` scaling each target's share of
    /// the ring. The same key keeps landing on the same target while it is
    /// healthy; on failure the walk follows the ring so only the failed
    /// target's keys move.
    ConsistentHash,
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
    /// Rank targets least-loaded-first by in-flight requests divided by
    /// target `weight` (the APISIX least_conn score), then fall forward.
    LeastBusy,
}

/// Which request attribute a [`HashOnSource`] reads the hash key from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HashOnType {
    /// A request header, named by `name`.
    Header,
    /// A cookie from the request's `Cookie` header, named by `name`.
    Cookie,
    /// The caller's API key id.
    ApiKey,
    /// The caller's resolved client IP (honouring the trusted-proxy
    /// configuration).
    ClientIp,
}

/// One source for the `consistent_hash` hash key. Sources are tried in
/// order; the first one that yields a non-empty value wins.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
pub struct HashOnSource {
    /// Which request attribute supplies the hash key.
    #[serde(rename = "type")]
    pub source_type: HashOnType,
    /// The header or cookie name to read. Required for `header` and
    /// `cookie` sources; not accepted for `api_key` or `client_ip`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub name: Option<String>,
}

impl HashOnSource {
    pub fn header(name: impl Into<String>) -> Self {
        Self {
            source_type: HashOnType::Header,
            name: Some(name.into()),
        }
    }

    pub fn cookie(name: impl Into<String>) -> Self {
        Self {
            source_type: HashOnType::Cookie,
            name: Some(name.into()),
        }
    }

    pub fn api_key() -> Self {
        Self {
            source_type: HashOnType::ApiKey,
            name: None,
        }
    }

    pub fn client_ip() -> Self {
        Self {
            source_type: HashOnType::ClientIp,
            name: None,
        }
    }
}

/// Default hash-key chain when `hash_on` is not configured: the
/// `x-aisix-routing-key` request header, falling back to the caller's API
/// key id.
pub fn default_hash_on() -> Vec<HashOnSource> {
    vec![
        HashOnSource::header("x-aisix-routing-key"),
        HashOnSource::api_key(),
    ]
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
pub struct RoutingTarget {
    /// Model alias for a direct model that can receive routed traffic.
    #[schemars(length(min = 1))]
    pub model: String,
    /// Target weight, default `1`. Used by `round_robin` (rotation share),
    /// `consistent_hash` (share of the hash ring), and `least_busy`
    /// (in-flight divided by weight). `failover`, `least_cost`, and
    /// `least_latency` accept the field but do not use it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<u32>,
    /// Priority tier, default `0`; a higher value is preferred (the APISIX
    /// node-priority convention — give backup targets `-1`). The strategy
    /// orders targets within each tier; a lower tier is only tried when
    /// every higher-tier target failed or is unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
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
            priority: None,
            tags: None,
        }
    }

    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = Some(weight);
        self
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = Some(priority);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    pub fn weight_or_default(&self) -> u32 {
        self.weight.unwrap_or(1)
    }

    pub fn priority_or_default(&self) -> i32 {
        self.priority.unwrap_or(0)
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
pub struct Routing {
    /// Strategy used to select a target for each request.
    #[serde(default)]
    pub strategy: RoutingStrategy,
    /// Ordered set of direct models available to this routing model.
    #[schemars(length(min = 1))]
    pub targets: Vec<RoutingTarget>,
    /// Retry attempts on the current target before failing over, applied to every target that does not set its own `retries`. Absent falls back to the deployment-wide `upstream.retries` default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retries: Option<u32>,
    /// Max number of later targets to attempt after the initial target fails permanently. When omitted, all later targets may be attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fallbacks: Option<u32>,
    /// Whether upstream 429 participates in retries and failover.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_on_429: Option<bool>,
    /// Additional upstream HTTP status codes that participate in retries and failover. By default a non-429 4xx response is treated as a caller error and returned as-is; providers that use 4xx codes for transient conditions (model overload, queue full, quota exhaustion) can be listed here, for example `[408, 409]`. 5xx codes are already retryable, so listing them changes nothing. Authentication (`401`/`403`) and validation (`400`) codes should only be listed when the provider is known to use them for transient failures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(range(min = 400, max = 599)))]
    pub fallback_on_statuses: Option<Vec<u16>>,
    /// Policy to apply when every target is unavailable because of runtime health or cooldown state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when_all_unavailable: Option<WhenAllUnavailablePolicy>,
    /// Where the `consistent_hash` hash key comes from: an ordered chain of
    /// sources, the first non-empty value winning. Defaults to the
    /// `x-aisix-routing-key` request header, falling back to the caller's
    /// API key id. Only valid with `strategy: consistent_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub hash_on: Option<Vec<HashOnSource>>,
}

impl Routing {
    // No `retries_or_default()`: an unset group budget no longer means
    // zero, it means "defer to the target, then to the deployment default".
    // Resolving that needs the target Model and the DP config, so it lives
    // in `aisix_proxy::routing::effective_retries`.

    /// The effective hash-key source chain for `consistent_hash`.
    pub fn hash_on_or_default(&self) -> Vec<HashOnSource> {
        match &self.hash_on {
            Some(chain) if !chain.is_empty() => chain.clone(),
            _ => default_hash_on(),
        }
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

    /// Configured status codes that opt into retry/failover; empty when
    /// unset (the default behavior).
    pub fn fallback_on_statuses_or_default(&self) -> &[u16] {
        self.fallback_on_statuses.as_deref().unwrap_or(&[])
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
            "strategy": "round_robin",
            "targets": [
                {"model": "primary", "weight": 90},
                {"model": "backup",  "weight": 10, "priority": -1}
            ],
            "retries": 2,
            "max_fallbacks": 1,
            "retry_on_429": true
        }"#;
        let r: Routing = serde_json::from_str(json).unwrap();
        assert_eq!(r.strategy, RoutingStrategy::RoundRobin);
        assert_eq!(r.targets.len(), 2);
        assert_eq!(r.targets[0].model, "primary");
        assert_eq!(r.targets[0].weight_or_default(), 90);
        assert_eq!(r.targets[0].priority_or_default(), 0);
        assert_eq!(r.targets[1].priority_or_default(), -1);
        assert_eq!(r.retries, Some(2));
        assert_eq!(r.max_fallbacks_or_default(), 1);
        assert!(r.retry_on_429_or_default());
    }

    #[test]
    fn strategy_defaults_to_failover() {
        let r: Routing =
            serde_json::from_str(r#"{"targets":[{"model":"a"},{"model":"b"}]}"#).unwrap();
        assert_eq!(r.strategy, RoutingStrategy::Failover);
        // Absent means "defer" now, not zero — see `effective_retries`.
        assert_eq!(r.retries, None);
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
            fallback_on_statuses: None,
            when_all_unavailable: None,
            hash_on: None,
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
            fallback_on_statuses: None,
            when_all_unavailable: None,
            hash_on: None,
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
    fn parses_consistent_hash_with_hash_on_chain() {
        let r: Routing = serde_json::from_str(
            r#"{
                "strategy": "consistent_hash",
                "hash_on": [
                    {"type": "header", "name": "x-session-id"},
                    {"type": "cookie", "name": "sid"},
                    {"type": "api_key"},
                    {"type": "client_ip"}
                ],
                "targets": [{"model": "a"}, {"model": "b"}]
            }"#,
        )
        .unwrap();
        assert_eq!(r.strategy, RoutingStrategy::ConsistentHash);
        let chain = r.hash_on_or_default();
        assert_eq!(chain.len(), 4);
        assert_eq!(chain[0], HashOnSource::header("x-session-id"));
        assert_eq!(chain[1], HashOnSource::cookie("sid"));
        assert_eq!(chain[2], HashOnSource::api_key());
        assert_eq!(chain[3], HashOnSource::client_ip());
    }

    #[test]
    fn hash_on_defaults_to_routing_key_header_then_api_key() {
        let r: Routing =
            serde_json::from_str(r#"{"strategy":"consistent_hash","targets":[{"model":"a"}]}"#)
                .unwrap();
        assert_eq!(r.hash_on_or_default(), default_hash_on());
        assert_eq!(
            default_hash_on()[0],
            HashOnSource::header("x-aisix-routing-key")
        );
    }

    #[test]
    fn removed_weighted_strategy_and_sticky_flag_are_rejected() {
        // `weighted` merged into `round_robin` and `sticky` was replaced by
        // `strategy: consistent_hash` (AISIX-Cloud#1206). The enum value must
        // fail row-level so a stale kine row cannot silently change meaning.
        let weighted: Result<Routing, _> =
            serde_json::from_str(r#"{"strategy":"weighted","targets":[{"model":"a"}]}"#);
        assert!(weighted.is_err());
        // `sticky` is now just an unknown field: lenient serde tolerates it
        // (forward/backward compat), the strict write-path schema rejects it.
        let sticky: Routing = serde_json::from_str(
            r#"{"strategy":"round_robin","sticky":true,"targets":[{"model":"a"}]}"#,
        )
        .unwrap();
        assert_eq!(sticky.strategy, RoutingStrategy::RoundRobin);
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
        assert!(!RoutingStrategy::ConsistentHash.is_metric_based());
    }

    #[test]
    fn tolerates_unknown_routing_fields_for_forward_compat() {
        // A newer control plane may ship fields ahead of this DP; serde must
        // accept them. The write path still rejects them via the strict schema
        // validator of the enclosing resource (validate_model in models/schema.rs).
        let r: Routing =
            serde_json::from_str(r#"{"strategy":"failover","targets":[{"model":"a"}],"foo":1}"#)
                .unwrap();
        assert_eq!(r.strategy, RoutingStrategy::Failover);
    }

    #[test]
    fn tolerates_unknown_target_fields_for_forward_compat() {
        // Same forward-compat contract as above, for the nested target struct.
        let t: RoutingTarget =
            serde_json::from_str(r#"{"model":"a","weight":2,"extra":true}"#).unwrap();
        assert_eq!(t.model, "a");
        assert_eq!(t.weight, Some(2));
    }
}
