//! `McpPolicy` entity — environment-default and team-level MCP tool access
//! policies stored in etcd under `mcp_policies/<uuid>`.
//!
//! Policies lift MCP tool access from per-key `allowed_tools` configuration
//! to a layered grant: an `env`-scoped policy sets the environment-wide
//! default, a `team`-scoped policy replaces that default for the keys that
//! belong to the team, and each key narrows (never widens) the inherited
//! grant through its `mcp_access` block. `deny` patterns from every
//! applicable level are always subtracted, so a tool denied at the
//! environment level stays unavailable regardless of team or key
//! configuration.
//!
//! Keys without an `mcp_access` block keep the pre-policy behavior: their
//! `allowed_tools` list is the entire allow side (no inheritance), with
//! policy `deny` patterns still subtracted. The effective-ACL computation
//! lives with the MCP gateway endpoint, which resolves it per request.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// Which API keys an MCP access policy applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpPolicyScope {
    /// The environment-wide default, applied to keys whose team has no
    /// enabled policy of its own.
    Env,
    /// A team-level policy, applied to the keys that belong to the team
    /// named by `scope_ref`. It replaces the environment default for those
    /// keys.
    Team,
}

/// What an MCP access policy grants before `deny` patterns are subtracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpPolicyMode {
    /// Grants no MCP tools.
    None,
    /// Grants exactly the tools matched by the `allow` patterns.
    Selected,
    /// Grants every tool on every registered MCP server, including servers
    /// and tools added after the policy is created.
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpPolicy {
    /// Which API keys the policy applies to: the whole environment or one
    /// team.
    pub scope: McpPolicyScope,

    /// Team identifier the policy targets. Required when `scope` is `team`;
    /// omitted for an environment-default policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub scope_ref: Option<String>,

    /// What the policy grants: `none`, `selected` (the `allow` patterns), or
    /// `all` current and future tools.
    pub mode: McpPolicyMode,

    /// Namespaced `<server>__<tool>` patterns granted when `mode` is
    /// `selected`. Entries are matched as single-`*` globs, the same form as
    /// an API key's `allowed_tools`: `"<server>__*"` grants every tool on one
    /// server and an entry without a `*` matches one tool exactly. Ignored
    /// for the other modes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,

    /// Namespaced `<server>__<tool>` patterns subtracted from the effective
    /// grant of every key the policy applies to, using the same single-`*`
    /// glob matching as `allow`. Deny always wins: a tool matched here stays
    /// unavailable even when a team policy or a key's own configuration
    /// would otherwise grant it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deny: Vec<String>,

    /// Whether the policy is applied. A disabled policy is kept but
    /// ignored. Treated as `true` when omitted.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// etcd-key uuid. Filled by the loader and never included in the JSON
    /// payload.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

fn default_enabled() -> bool {
    true
}

/// How an API key combines with the applicable MCP access policies:
/// `inherit`, `restrict`, or `deny`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpAccessMode {
    /// The key uses the inherited grant unchanged: its team's enabled policy
    /// when one exists, otherwise the environment-default policy.
    Inherit,
    /// The key uses the intersection of the inherited grant and its own
    /// `allow` patterns — a restriction can only narrow what the policies
    /// grant, never widen it.
    Restrict,
    /// The key has no MCP tool access at all.
    Deny,
}

/// Policy-driven MCP access configuration on an API key. When present, this
/// block supersedes the key's `allowed_tools` list: the allow side of the
/// key's effective grant is computed from the applicable MCP access policies
/// and `mode`, and `allowed_tools` is not consulted.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpAccess {
    /// How the key combines with the applicable policies: `inherit`,
    /// `restrict`, or `deny`.
    pub mode: McpAccessMode,

    /// Namespaced `<server>__<tool>` patterns intersected with the inherited
    /// grant when `mode` is `restrict`, using the same single-`*` glob
    /// matching as `allowed_tools`. Ignored for the other modes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,

    /// Namespaced `<server>__<tool>` patterns subtracted from the key's
    /// effective grant, using the same single-`*` glob matching as `allow`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deny: Vec<String>,
}

impl Resource for McpPolicy {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// The by-name index key: the targeted team id, or `"env"` for the
    /// environment-default policy. Lookups during effective-ACL resolution
    /// iterate and filter on `(scope, scope_ref)` rather than relying on
    /// this index, so a malformed row can never shadow the default.
    #[allow(clippy::misnamed_getters)]
    fn name(&self) -> &str {
        self.scope_ref.as_deref().unwrap_or("env")
    }

    fn kind() -> &'static str {
        "mcp_policies"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_env_default_policy() {
        let p: McpPolicy = serde_json::from_str(
            r#"{
              "scope": "env",
              "mode": "selected",
              "allow": ["github__*", "postgres__query"],
              "deny": ["github__delete_repository"]
            }"#,
        )
        .unwrap();
        assert_eq!(p.scope, McpPolicyScope::Env);
        assert!(p.scope_ref.is_none());
        assert_eq!(p.mode, McpPolicyMode::Selected);
        assert_eq!(p.allow, vec!["github__*", "postgres__query"]);
        assert_eq!(p.deny, vec!["github__delete_repository"]);
        assert!(p.enabled);
    }

    #[test]
    fn deserialises_team_policy() {
        let p: McpPolicy = serde_json::from_str(
            r#"{
              "scope": "team",
              "scope_ref": "team-uuid-1",
              "mode": "all"
            }"#,
        )
        .unwrap();
        assert_eq!(p.scope, McpPolicyScope::Team);
        assert_eq!(p.scope_ref.as_deref(), Some("team-uuid-1"));
        assert_eq!(p.mode, McpPolicyMode::All);
        assert!(p.allow.is_empty());
        assert!(p.deny.is_empty());
    }

    #[test]
    fn tolerates_unknown_fields_for_forward_compat() {
        // cp-api may ship new fields ahead of the DP rolling out; serde must
        // accept them. The write path still rejects them via the strict
        // schema validators (validate_mcp_policy in models/schema.rs).
        let p: McpPolicy =
            serde_json::from_str(r#"{"scope":"env","mode":"all","extra":1}"#).unwrap();
        assert_eq!(p.mode, McpPolicyMode::All);
    }

    #[test]
    fn rejects_unknown_mode_and_scope() {
        assert!(serde_json::from_str::<McpPolicy>(r#"{"scope":"org","mode":"all"}"#).is_err());
        assert!(serde_json::from_str::<McpPolicy>(r#"{"scope":"env","mode":"open"}"#).is_err());
    }

    #[test]
    fn enabled_defaults_true_and_roundtrips_false() {
        let active: McpPolicy = serde_json::from_str(r#"{"scope":"env","mode":"all"}"#).unwrap();
        assert!(active.enabled);

        let disabled: McpPolicy =
            serde_json::from_str(r#"{"scope":"env","mode":"all","enabled":false}"#).unwrap();
        assert!(!disabled.enabled);
    }

    #[test]
    fn resource_trait_points_at_scope_ref_and_kind() {
        assert_eq!(McpPolicy::kind(), "mcp_policies");

        let mut env: McpPolicy = serde_json::from_str(r#"{"scope":"env","mode":"all"}"#).unwrap();
        env.runtime_id = "p-env".into();
        assert_eq!(env.id(), "p-env");
        assert_eq!(env.name(), "env");

        let mut team: McpPolicy =
            serde_json::from_str(r#"{"scope":"team","scope_ref":"team-uuid-1","mode":"none"}"#)
                .unwrap();
        team.runtime_id = "p-team".into();
        assert_eq!(team.name(), "team-uuid-1");
    }

    #[test]
    fn mcp_access_deserialises_all_modes() {
        let inherit: McpAccess = serde_json::from_str(r#"{"mode":"inherit"}"#).unwrap();
        assert_eq!(inherit.mode, McpAccessMode::Inherit);
        assert!(inherit.allow.is_empty());
        assert!(inherit.deny.is_empty());

        let restrict: McpAccess = serde_json::from_str(
            r#"{"mode":"restrict","allow":["github__*"],"deny":["github__delete_repository"]}"#,
        )
        .unwrap();
        assert_eq!(restrict.mode, McpAccessMode::Restrict);
        assert_eq!(restrict.allow, vec!["github__*"]);
        assert_eq!(restrict.deny, vec!["github__delete_repository"]);

        let deny: McpAccess = serde_json::from_str(r#"{"mode":"deny"}"#).unwrap();
        assert_eq!(deny.mode, McpAccessMode::Deny);
    }

    #[test]
    fn mcp_access_tolerates_unknown_fields_for_forward_compat_but_rejects_unknown_modes() {
        // cp-api may ship new fields ahead of the DP rolling out; serde must
        // accept them (the write path still rejects them via the strict
        // schema validators in models/schema.rs). Unknown enum values stay
        // hard errors.
        let a: McpAccess = serde_json::from_str(r#"{"mode":"inherit","extra":1}"#).unwrap();
        assert_eq!(a.mode, McpAccessMode::Inherit);
        assert!(serde_json::from_str::<McpAccess>(r#"{"mode":"legacy"}"#).is_err());
    }

    #[test]
    fn empty_lists_stay_off_the_wire() {
        let p: McpPolicy = serde_json::from_str(r#"{"scope":"env","mode":"all"}"#).unwrap();
        let v = serde_json::to_value(&p).unwrap();
        assert!(v.get("allow").is_none());
        assert!(v.get("deny").is_none());
        assert!(v.get("scope_ref").is_none());

        let a: McpAccess = serde_json::from_str(r#"{"mode":"inherit"}"#).unwrap();
        let v = serde_json::to_value(&a).unwrap();
        assert!(v.get("allow").is_none());
        assert!(v.get("deny").is_none());
    }
}
