//! `RateLimitPolicy` entity — standalone rate-limit rules stored in etcd
//! under `rate_limit_policies/<uuid>`.
//!
//! Each policy targets a single subject via `(scope, scope_ref)`:
//! - `api_key`     — matches by API key entry ID
//! - `model`       — matches by model entry ID
//! - `team`        — matches by team ID on the API key (one shared bucket)
//! - `member`      — matches by user ID on the API key
//! - `team_member` — matches by team ID, but buckets per member: every
//!   key in the team inherits this default with its own independent
//!   counter keyed on the API key's user ID
//!
//! The proxy iterates all policies on each request, converts the
//! `window`+`max_requests`/`max_tokens` into a `RateLimit`, and
//! reserves under `policy:<scope>:<scope_ref>:<policy_id>` — with the
//! member's `user_id` appended for the `team_member` scope.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::resource::Resource;

/// Subject a [`RateLimitPolicy`] targets, paired with `scope_ref`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PolicyScope {
    ApiKey,
    Model,
    Team,
    Member,
    TeamMember,
}

impl PolicyScope {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ApiKey => "api_key",
            Self::Model => "model",
            Self::Team => "team",
            Self::Member => "member",
            Self::TeamMember => "team_member",
        }
    }
}

impl std::fmt::Display for PolicyScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fixed-window length a [`RateLimitPolicy`] applies its limits over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PolicyWindow {
    Second,
    Minute,
    Hour,
}

impl PolicyWindow {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Second => "second",
            Self::Minute => "minute",
            Self::Hour => "hour",
        }
    }
}

impl std::fmt::Display for PolicyWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RateLimitPolicy {
    #[schemars(length(min = 1))]
    pub name: String,
    pub scope: PolicyScope,
    #[schemars(length(min = 1))]
    pub scope_ref: String,
    pub window: PolicyWindow,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_tokens: Option<u64>,

    #[serde(skip)]
    pub(crate) runtime_id: String,
}

/// The one cross-field invariant `schemars` can't derive: a policy must cap at
/// least one of `max_requests` / `max_tokens`. Injected as a top-level `anyOf`
/// by [`crate::models::schema::rate_limit_policy_root_schema`].
pub fn rate_limit_policy_any_of() -> Value {
    json!([
        { "required": ["max_requests"] },
        { "required": ["max_tokens"] }
    ])
}

impl Resource for RateLimitPolicy {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    #[allow(clippy::misnamed_getters)]
    fn name(&self) -> &str {
        &self.scope_ref
    }

    fn kind() -> &'static str {
        "rate_limit_policies"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_with_all_fields() {
        let p: RateLimitPolicy = serde_json::from_str(
            r#"{
              "name": "team-quota",
              "scope": "team",
              "scope_ref": "team-uuid-1",
              "window": "minute",
              "max_requests": 100,
              "max_tokens": 50000
            }"#,
        )
        .unwrap();
        assert_eq!(p.name, "team-quota");
        assert_eq!(p.scope, PolicyScope::Team);
        assert_eq!(p.scope_ref, "team-uuid-1");
        assert_eq!(p.window, PolicyWindow::Minute);
        assert_eq!(p.max_requests, Some(100));
        assert_eq!(p.max_tokens, Some(50000));
    }

    #[test]
    fn deserialises_with_only_max_requests() {
        let p: RateLimitPolicy = serde_json::from_str(
            r#"{
              "name": "key-rpm",
              "scope": "api_key",
              "scope_ref": "key-uuid-1",
              "window": "minute",
              "max_requests": 60
            }"#,
        )
        .unwrap();
        assert_eq!(p.max_requests, Some(60));
        assert!(p.max_tokens.is_none());
    }

    #[test]
    fn rejects_unknown_fields() {
        let r: Result<RateLimitPolicy, _> = serde_json::from_str(
            r#"{
              "name": "x",
              "scope": "team",
              "scope_ref": "t1",
              "window": "minute",
              "extra": true
            }"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn resource_trait_returns_correct_kind() {
        assert_eq!(RateLimitPolicy::kind(), "rate_limit_policies");
    }

    #[test]
    fn resource_name_returns_scope_ref() {
        let mut p: RateLimitPolicy = serde_json::from_str(
            r#"{
              "name": "test",
              "scope": "member",
              "scope_ref": "member-uuid-1",
              "window": "hour",
              "max_tokens": 1000000
            }"#,
        )
        .unwrap();
        p.runtime_id = "policy-1".into();
        assert_eq!(p.id(), "policy-1");
        assert_eq!(p.name(), "member-uuid-1");
    }
}
