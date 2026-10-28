//! `Budget` and `BudgetPolicy` — the spend caps the control plane enforces
//! through the gateway.
//!
//! The control plane owns the spend counters. It writes one `budgets`
//! document per hard-stop budget that applies in the environment, and
//! marks the document `exhausted` once spend reaches the limit; the gateway
//! decides every request locally from those documents and never asks the
//! control plane per request. `budget_policy` is the environment's
//! singleton that says what the gateway does while it cannot deliver usage
//! to the control plane — the only input those counters have.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// A hard-stop spend cap that applies in this environment, and whether it
/// is currently exhausted.
///
/// The document's id is the budget's id. Only hard-stop budgets are
/// present: a budget that only warns never blocks a request, so it has no
/// document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Budget {
    /// What the budget caps.
    ///
    /// - `org`, `environment`: every API key in this environment.
    /// - `api_key`: the API key whose id is `scope_ref`.
    /// - `member`: every API key whose `user_id` is `scope_ref`.
    /// - `team`: every API key whose `team_id` is `scope_ref`.
    /// - `team_member`: every API key whose `team_id` is `scope_ref` and that
    ///   also names a member; each member is capped separately.
    /// - `tag`: every API key whose effective value for `tag.key` is
    ///   `tag.value`.
    /// - `provider_key`: every upstream call made with the provider key whose
    ///   id is `scope_ref`.
    pub scope: BudgetScope,

    /// Id of the capped resource. Present for every scope except `tag`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_ref: Option<String>,

    /// The capped tag. Present only for scope `tag`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<BudgetTag>,

    /// Accounting period. Periods follow the UTC calendar: a day starts at
    /// 00:00 UTC, a week on Monday at 00:00 UTC, and a month on the 1st at
    /// 00:00 UTC.
    pub period: BudgetPeriod,

    /// Set while the budget is exhausted for everything it applies to.
    /// For scope `team_member` it applies to every member of the team.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exhausted: Option<BudgetExhausted>,

    /// Members of the team who have exhausted their own share of a
    /// `team_member` budget, keyed by member id. Read only for scope
    /// `team_member`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub exhausted_members: BTreeMap<String, BudgetMemberExhausted>,

    /// Set by the loader from the key's id segment. Not part of the wire
    /// shape.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BudgetScope {
    Org,
    Environment,
    ApiKey,
    Member,
    Team,
    TeamMember,
    Tag,
    ProviderKey,
}

impl BudgetScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Org => "org",
            Self::Environment => "environment",
            Self::ApiKey => "api_key",
            Self::Member => "member",
            Self::Team => "team",
            Self::TeamMember => "team_member",
            Self::Tag => "tag",
            Self::ProviderKey => "provider_key",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BudgetTag {
    /// Tag key.
    pub key: String,
    /// Tag value.
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BudgetPeriod {
    Day,
    Week,
    Month,
}

impl BudgetPeriod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Day => "day",
            Self::Week => "week",
            Self::Month => "month",
        }
    }
}

/// The budget is exhausted for everything it applies to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BudgetExhausted {
    /// Start of the period the budget was exhausted in, as an RFC 3339
    /// timestamp. The block holds only while this is the start of the
    /// current period, so it lifts at the next period boundary on its own.
    /// Omitted, the block holds in every period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period_start: Option<String>,
    /// What a rejected request is told.
    pub reason: BudgetReason,
}

/// One member's share of a `team_member` budget is exhausted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BudgetMemberExhausted {
    /// Start of the period the member exhausted their share in, as an
    /// RFC 3339 timestamp. The block holds only while this is the start of
    /// the current period.
    pub period_start: String,
    /// What a rejected request is told.
    pub reason: BudgetReason,
}

/// The detail a request rejected by an exhausted budget is answered with.
/// Amounts are decimal strings in the organization's currency.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BudgetReason {
    /// Error type. The response always reports `billing_error`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Error code. The response always reports `budget_exceeded`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Human-readable message returned to the caller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Scope of the exhausted budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Id of the capped resource, or the tag for scope `tag`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_ref: Option<String>,
    /// The budget's limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit_usd: Option<String>,
    /// Spend in the period when the budget was exhausted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spent_usd: Option<String>,
    /// Limit minus spend; zero or negative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remaining_usd: Option<String>,
    /// Accounting period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period: Option<String>,
    /// Ignored: the gateway computes when the period resets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period_resets_at: Option<serde_json::Value>,
    /// Ignored: the gateway computes how long until the period resets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<serde_json::Value>,
}

impl Resource for Budget {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// Budgets are referenced by id only.
    fn name(&self) -> &str {
        "budget"
    }

    fn kind() -> &'static str {
        "budgets"
    }
}

/// What the gateway does with budgeted traffic while it cannot deliver
/// usage to the control plane, whose spend counters are what mark a budget
/// exhausted.
///
/// At most one document exists per environment, keyed by the environment
/// id. Without it the gateway behaves as `sticky`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BudgetPolicy {
    /// `sticky` keeps enforcing the last exhausted state the gateway
    /// received and lets everything else through. `closed` additionally
    /// rejects budgeted traffic once usage delivery has failed for
    /// `closed_grace_seconds`: a request from an API key that at least one
    /// budget applies to is answered `503 budget_unverified`, and an
    /// upstream whose provider key has a budget is not used.
    pub fail_mode: BudgetFailMode,

    /// How long usage delivery may fail before `closed` starts rejecting
    /// budgeted traffic, in seconds.
    #[schemars(range(min = 1))]
    pub closed_grace_seconds: u64,

    /// Set by the loader from the key's id segment. Not part of the wire
    /// shape.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BudgetFailMode {
    Sticky,
    Closed,
}

impl Resource for BudgetPolicy {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// Fixed identity: the row is a per-environment singleton.
    fn name(&self) -> &str {
        "budget_policy"
    }

    fn kind() -> &'static str {
        "budget_policy"
    }
}

#[cfg(test)]
mod tests {
    use crate::models::{validate_budget, validate_budget_lenient, validate_budget_policy};
    use serde_json::json;

    /// The per-scope shape is a write-path rule only: the stored row still
    /// loads, and a row without its target matches nothing.
    #[test]
    fn the_per_scope_shape_binds_the_write_path_only() {
        for doc in [
            json!({"scope": "tag", "period": "day"}),
            json!({"scope": "tag", "scope_ref": "x", "tag": {"key": "k", "value": "v"}, "period": "day"}),
            json!({"scope": "team", "period": "day"}),
            json!({"scope": "team", "scope_ref": "t", "tag": {"key": "k", "value": "v"}, "period": "day"}),
            json!({"scope": "team", "scope_ref": "t", "period": "day", "exhausted_members": {}}),
        ] {
            assert!(validate_budget(&doc).is_err(), "strict accepted {doc}");
            assert!(
                validate_budget_lenient(&doc).is_ok(),
                "lenient rejected {doc}"
            );
        }
        for doc in [
            json!({"scope": "tag", "tag": {"key": "k", "value": "v"}, "period": "week"}),
            json!({"scope": "team_member", "scope_ref": "t", "period": "month",
                   "exhausted_members": {"u": {"period_start": "2026-10-01T00:00:00Z", "reason": {}}}}),
            json!({"scope": "provider_key", "scope_ref": "pk", "period": "day",
                   "exhausted": {"reason": {"message": "m", "limit_usd": "0.00"}}}),
        ] {
            assert!(validate_budget(&doc).is_ok(), "strict rejected {doc}");
        }
    }

    #[test]
    fn the_policy_grace_is_a_positive_integer() {
        assert!(
            validate_budget_policy(&json!({"fail_mode": "closed", "closed_grace_seconds": 1}))
                .is_ok()
        );
        assert!(
            validate_budget_policy(&json!({"fail_mode": "closed", "closed_grace_seconds": 0}))
                .is_err()
        );
        assert!(
            validate_budget_policy(&json!({"fail_mode": "open", "closed_grace_seconds": 60}))
                .is_err()
        );
    }
}
