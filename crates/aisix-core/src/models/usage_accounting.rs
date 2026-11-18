//! `UsageAccounting` — how the environment counts the usage the gateway
//! estimated itself.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// How the environment counts usage the gateway estimated locally because
/// the upstream reported none.
///
/// At most one document exists per environment, keyed by the environment
/// id. Without it estimated usage counts like reported usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct UsageAccounting {
    /// When `true`, a request whose usage the gateway estimated (any token
    /// counter it counted locally, so its usage event carries
    /// `usage_estimated: true`) adds none of its tokens to any token rate
    /// limit window (`tpm`, `tpd`, and a policy's `max_tokens`).
    /// Request-count limits and concurrency still count it, and its usage
    /// event still reports the estimated tokens.
    pub exclude_estimated_usage: bool,

    /// Set by the loader from the key's id segment. Not part of the wire
    /// shape.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Resource for UsageAccounting {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// Fixed identity: the row is a per-environment singleton.
    fn name(&self) -> &str {
        "usage_accounting"
    }

    fn kind() -> &'static str {
        "usage_accounting"
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::models::{validate_usage_accounting, validate_usage_accounting_lenient};

    #[test]
    fn exclude_estimated_usage_is_a_required_boolean() {
        assert!(validate_usage_accounting(&json!({"exclude_estimated_usage": true})).is_ok());
        assert!(validate_usage_accounting(&json!({"exclude_estimated_usage": false})).is_ok());
        assert!(validate_usage_accounting(&json!({})).is_err());
        assert!(validate_usage_accounting(&json!({"exclude_estimated_usage": "yes"})).is_err());
        assert!(validate_usage_accounting_lenient(&json!({})).is_err());
    }

    #[test]
    fn unknown_fields_are_refused_on_write_and_tolerated_on_load() {
        let doc = json!({"exclude_estimated_usage": true, "later": 1});
        assert!(validate_usage_accounting(&doc).is_err());
        assert!(validate_usage_accounting_lenient(&doc).is_ok());
    }
}
