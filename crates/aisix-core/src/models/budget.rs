//! `Budget` entity — monthly USD ceiling on token spend per ApiKey.
//!
//! Operators set:
//! - `name`: a human-readable label,
//! - `api_key_id`: which ApiKey this budget governs (V1 scope is
//!   per-key; per-team comes when Teams land),
//! - `monthly_usd_cap`: maximum dollars spent per calendar month,
//! - `usd_per_1k_tokens`: linear pricing for the v1 cost model. Once
//!   the gateway grows per-provider price tables this becomes a
//!   per-provider override; for now it's the unit cost everywhere.
//!
//! etcd path: `{prefix}/budgets/{uuid}`. Secondary index on `name`.
//!
//! The accumulated spend lives in process memory (see
//! `aisix_proxy::budget::BudgetTracker`) — V1 doesn't persist counters
//! across restarts. A future "budget store" PR can swap the tracker
//! for a Redis-backed implementation behind the same trait.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub name: String,
    pub api_key_id: String,
    pub monthly_usd_cap: f64,
    pub usd_per_1k_tokens: f64,

    /// Filled in by the snapshot loader from the etcd key path.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Budget {
    /// Cost in dollars for `tokens` tokens at this budget's pricing.
    pub fn cost_for(&self, tokens: u64) -> f64 {
        (tokens as f64 / 1_000.0) * self.usd_per_1k_tokens
    }
}

impl Resource for Budget {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "budgets"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> &'static str {
        r#"{
            "name": "team-a-monthly",
            "api_key_id": "key-uuid-1",
            "monthly_usd_cap": 100.0,
            "usd_per_1k_tokens": 0.005
        }"#
    }

    #[test]
    fn deserialises_full_budget() {
        let b: Budget = serde_json::from_str(sample()).unwrap();
        assert_eq!(b.name, "team-a-monthly");
        assert_eq!(b.api_key_id, "key-uuid-1");
        assert_eq!(b.monthly_usd_cap, 100.0);
        assert_eq!(b.usd_per_1k_tokens, 0.005);
    }

    #[test]
    fn cost_for_scales_linearly_per_thousand_tokens() {
        let b: Budget = serde_json::from_str(sample()).unwrap();
        assert!((b.cost_for(0) - 0.0).abs() < 1e-9);
        assert!((b.cost_for(1_000) - 0.005).abs() < 1e-9);
        // 250k tokens × $0.005/1k = $1.25
        assert!((b.cost_for(250_000) - 1.25).abs() < 1e-9);
    }

    #[test]
    fn rejects_unknown_top_level_fields() {
        let r: Result<Budget, _> = serde_json::from_str(
            r#"{"name":"x","api_key_id":"k","monthly_usd_cap":1.0,"usd_per_1k_tokens":0.1,"extra":true}"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn resource_trait_uses_name_and_budgets_kind() {
        let mut b: Budget = serde_json::from_str(sample()).unwrap();
        b.runtime_id = "budget-uuid-1".into();
        assert_eq!(<Budget as Resource>::kind(), "budgets");
        assert_eq!(b.id(), "budget-uuid-1");
        assert_eq!(b.name(), "team-a-monthly");
    }
}
