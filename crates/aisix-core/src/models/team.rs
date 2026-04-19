//! `Team` entity — a named group of ApiKeys sharing a budget and rate limit.
//!
//! Teams allow operators to manage quotas at a group level rather than per
//! individual key. A `Team` is associated with zero or more ApiKey ids
//! (`members`) and optionally references a `Budget` id for spend enforcement
//! and a `RateLimit` shared across all member keys.
//!
//! etcd path: `{prefix}/teams/{uuid}`. Secondary index on `name`.

use serde::{Deserialize, Serialize};

use super::rate_limit::RateLimit;
use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Team {
    /// Human-readable label, must be unique within the gateway.
    pub name: String,

    /// ApiKey ids that belong to this team. Membership is tracked here
    /// rather than on the ApiKey so teams remain an opt-in grouping.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub members: Vec<String>,

    /// Optional reference to a `Budget` entry id that caps team spend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_id: Option<String>,

    /// Optional shared rate limit applied across all team members
    /// (additive with any per-key limits).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimit>,

    /// Filled in by the snapshot loader from the etcd key path.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Resource for Team {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "teams"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> &'static str {
        r#"{
            "name": "platform-team",
            "members": ["key-uuid-1", "key-uuid-2"],
            "budget_id": "budget-uuid-1"
        }"#
    }

    #[test]
    fn deserialises_full_team() {
        let t: Team = serde_json::from_str(sample()).unwrap();
        assert_eq!(t.name, "platform-team");
        assert_eq!(t.members.len(), 2);
        assert_eq!(t.budget_id.as_deref(), Some("budget-uuid-1"));
        assert!(t.rate_limit.is_none());
    }

    #[test]
    fn deserialises_minimal_team_name_only() {
        let t: Team = serde_json::from_str(r#"{"name":"minimal"}"#).unwrap();
        assert_eq!(t.name, "minimal");
        assert!(t.members.is_empty());
        assert!(t.budget_id.is_none());
    }

    #[test]
    fn round_trips_through_json() {
        let t: Team = serde_json::from_str(sample()).unwrap();
        let json = serde_json::to_string(&t).unwrap();
        let t2: Team = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }

    #[test]
    fn rejects_unknown_top_level_fields() {
        let r: Result<Team, _> = serde_json::from_str(r#"{"name":"x","members":[],"rogue": true}"#);
        assert!(r.is_err(), "should reject unknown fields");
    }

    #[test]
    fn resource_trait_returns_teams_kind() {
        let mut t: Team = serde_json::from_str(r#"{"name":"t"}"#).unwrap();
        t.runtime_id = "team-uuid-1".into();
        assert_eq!(<Team as Resource>::kind(), "teams");
        assert_eq!(t.id(), "team-uuid-1");
        assert_eq!(t.name(), "t");
    }
}
