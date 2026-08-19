//! `McpPolicy` entity — environment-level and team-level MCP tool access
//! policies stored in etcd under `mcp_policies/<uuid>`.
//!
//! MCP tool access is resolved from up to three layers of the same shape,
//! each an `allow`/`deny` pattern pair: the `env`-scoped policy, the
//! `team`-scoped policy of the key's team, and the key's own `mcp_access`
//! block. A tool is permitted when **every present layer** allows it and
//! **no** layer denies it — allow intersects, deny unions. A layer that is
//! absent (no policy row, a disabled one, or no `mcp_access` block) imposes
//! no constraint; with no layer present at all the grant is empty, so MCP
//! access is always granted explicitly.
//!
//! Because `allow` is required on every layer, a layer that only wants to
//! subtract tools spells its allow side `["*"]`. The effective-ACL
//! computation lives with the MCP gateway endpoint, which resolves it per
//! request.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// Which API keys an MCP access policy applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpPolicyScope {
    /// Applied to every key in the environment.
    Env,
    /// Applied to the keys belonging to the team named by `scope_ref`, on
    /// top of the environment policy rather than in place of it.
    Team,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpPolicy {
    /// Which API keys the policy applies to: the whole environment or one
    /// team.
    pub scope: McpPolicyScope,

    /// Team identifier the policy targets. Required when `scope` is `team`;
    /// omitted for an environment policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub scope_ref: Option<String>,

    /// Namespaced `<server>__<tool>` patterns this layer allows. Entries are
    /// matched as single-`*` globs: `"*"` allows every tool, `"<server>__*"`
    /// every tool on one server, and an entry without a `*` matches one tool
    /// exactly. An empty list allows nothing, which is how a policy blocks
    /// all MCP access; a policy that only means to subtract tools writes
    /// `["*"]` here and lists them under `deny`.
    ///
    /// The write path requires the field (the strict schema adds it to
    /// `required`), so a layer never allows something by omission. The
    /// runtime loader defaults it to empty instead of rejecting the row:
    /// a document written before the layered shape would otherwise fail
    /// to deserialize, and a skipped `api_key` row stops authenticating
    /// altogether rather than merely losing MCP access.
    #[serde(default)]
    pub allow: Vec<String>,

    /// Namespaced `<server>__<tool>` patterns subtracted from the effective
    /// grant of every key the policy applies to, using the same single-`*`
    /// glob matching as `allow`. Deny always wins: a tool matched here stays
    /// unavailable however the other layers allow it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deny: Vec<String>,

    /// Whether the policy is applied. A disabled policy is kept but
    /// contributes neither its allow nor its deny side. Treated as `true`
    /// when omitted.
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

/// The API key's own layer of the MCP tool ACL, the same `allow`/`deny`
/// shape an MCP access policy carries. Present means the key constrains its
/// grant; omitted means the key adds no constraint of its own and takes
/// whatever the environment and team layers leave.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpAccess {
    /// Namespaced `<server>__<tool>` patterns this key allows, intersected
    /// with the environment and team layers. Same single-`*` glob matching a
    /// policy's `allow` uses; an empty list leaves the key no MCP access,
    /// and `["*"]` narrows nothing (useful with `deny` alone).
    ///
    /// Required on the write path and defaulted by the runtime loader,
    /// for the reason given on [`McpPolicy::allow`].
    #[serde(default)]
    pub allow: Vec<String>,

    /// Namespaced `<server>__<tool>` patterns subtracted from this key's
    /// effective grant, using the same single-`*` glob matching as `allow`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deny: Vec<String>,

    /// Compatibility tombstone for the pre-0.10.0 `mode` selector. The
    /// control plane projects `"mode": "deny"` alongside the layered
    /// shape so a 0.9.x data plane — where `mode` is required and `deny`
    /// means "no MCP tool access" — still loads the whole api_key row,
    /// fail-closed, instead of skipping it (a skipped row stops the key
    /// authenticating for EVERY kind of traffic). This generation
    /// consumes and ignores the value; the field exists only so the
    /// loader does not report the tombstone as partial compat on every
    /// row. Any JSON shape is accepted so a malformed tombstone can
    /// never kill the row. Hidden from the schemas — the strict write
    /// path closes unknown fields, so resource authors cannot set it —
    /// and never re-serialized. Retire together with the CP emission
    /// once 0.9.x is out of the supported upgrade window.
    #[serde(default, rename = "mode", skip_serializing)]
    #[schemars(skip)]
    pub legacy_mode: Option<serde_json::Value>,
}

impl Resource for McpPolicy {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// The by-name index key: the targeted team id, or `"env"` for the
    /// environment policy. Lookups during effective-ACL resolution iterate
    /// and filter on `(scope, scope_ref)` rather than relying on this index,
    /// so a malformed row can never shadow the environment layer.
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
    fn deserialises_env_policy() {
        let p: McpPolicy = serde_json::from_str(
            r#"{
              "scope": "env",
              "allow": ["github__*", "postgres__query"],
              "deny": ["github__delete_repository"]
            }"#,
        )
        .unwrap();
        assert_eq!(p.scope, McpPolicyScope::Env);
        assert!(p.scope_ref.is_none());
        assert_eq!(p.allow, vec!["github__*", "postgres__query"]);
        assert_eq!(p.deny, vec!["github__delete_repository"]);
        assert!(p.enabled);
    }

    #[test]
    fn deserialises_team_policy() {
        let p: McpPolicy = serde_json::from_str(
            r#"{"scope": "team", "scope_ref": "team-uuid-1", "allow": ["*"]}"#,
        )
        .unwrap();
        assert_eq!(p.scope, McpPolicyScope::Team);
        assert_eq!(p.scope_ref.as_deref(), Some("team-uuid-1"));
        assert_eq!(p.allow, vec!["*"]);
        assert!(p.deny.is_empty());
    }

    #[test]
    fn the_loader_defaults_a_missing_allow_to_empty() {
        // A row written before the layered shape — e.g. the old
        // `{"mode":"inherit"}` key block — must still deserialize. It
        // resolves to a layer allowing nothing (fail-closed) rather than
        // being skipped, which for an api_key would drop the whole key.
        // The write path still rejects it; see
        // `mcp_policy_requires_an_explicit_allow_side` in schema.rs.
        let p: McpPolicy = serde_json::from_str(r#"{"scope":"env"}"#).unwrap();
        assert!(p.allow.is_empty());

        let a: McpAccess = serde_json::from_str(r#"{"mode":"inherit"}"#).unwrap();
        assert!(a.allow.is_empty());
    }

    #[test]
    fn tolerates_unknown_fields_for_forward_compat() {
        // cp-api may ship new fields ahead of the DP rolling out; serde must
        // accept them. The write path still rejects them via the strict
        // schema validators (validate_mcp_policy in models/schema.rs).
        let p: McpPolicy =
            serde_json::from_str(r#"{"scope":"env","allow":["*"],"extra":1}"#).unwrap();
        assert_eq!(p.allow, vec!["*"]);
    }

    #[test]
    fn rejects_unknown_scope() {
        assert!(serde_json::from_str::<McpPolicy>(r#"{"scope":"org","allow":["*"]}"#).is_err());
    }

    #[test]
    fn enabled_defaults_true_and_roundtrips_false() {
        let active: McpPolicy = serde_json::from_str(r#"{"scope":"env","allow":["*"]}"#).unwrap();
        assert!(active.enabled);

        let disabled: McpPolicy =
            serde_json::from_str(r#"{"scope":"env","allow":["*"],"enabled":false}"#).unwrap();
        assert!(!disabled.enabled);
    }

    #[test]
    fn resource_trait_points_at_scope_ref_and_kind() {
        assert_eq!(McpPolicy::kind(), "mcp_policies");

        let mut env: McpPolicy = serde_json::from_str(r#"{"scope":"env","allow":["*"]}"#).unwrap();
        env.runtime_id = "p-env".into();
        assert_eq!(env.id(), "p-env");
        assert_eq!(env.name(), "env");

        let mut team: McpPolicy =
            serde_json::from_str(r#"{"scope":"team","scope_ref":"team-uuid-1","allow":[]}"#)
                .unwrap();
        team.runtime_id = "p-team".into();
        assert_eq!(team.name(), "team-uuid-1");
    }

    #[test]
    fn mcp_access_deserialises_allow_and_deny() {
        let narrow: McpAccess =
            serde_json::from_str(r#"{"allow":["github__*"],"deny":["github__delete_repository"]}"#)
                .unwrap();
        assert_eq!(narrow.allow, vec!["github__*"]);
        assert_eq!(narrow.deny, vec!["github__delete_repository"]);

        // The old `mode: deny` is now an empty allow list.
        let blocked: McpAccess = serde_json::from_str(r#"{"allow":[]}"#).unwrap();
        assert!(blocked.allow.is_empty());
        assert!(blocked.deny.is_empty());
    }

    #[test]
    fn mcp_access_consumes_the_legacy_mode_tombstone() {
        // The CP projects `"mode": "deny"` next to the layered shape so a
        // 0.9.x DP loads the row fail-closed. This generation reads its
        // own half, tolerates any tombstone shape, and never re-emits it.
        let a: McpAccess =
            serde_json::from_str(r#"{"mode":"deny","allow":["github__*"],"deny":["x__y"]}"#)
                .unwrap();
        assert_eq!(a.allow, vec!["github__*"]);
        assert_eq!(a.deny, vec!["x__y"]);
        assert_eq!(a.legacy_mode, Some(serde_json::json!("deny")));

        let malformed: McpAccess = serde_json::from_str(r#"{"allow":[],"mode":5}"#).unwrap();
        assert_eq!(malformed.legacy_mode, Some(serde_json::json!(5)));

        let v = serde_json::to_value(&a).unwrap();
        assert!(v.get("mode").is_none());
    }

    #[test]
    fn empty_deny_stays_off_the_wire() {
        let p: McpPolicy = serde_json::from_str(r#"{"scope":"env","allow":["*"]}"#).unwrap();
        let v = serde_json::to_value(&p).unwrap();
        assert!(v.get("deny").is_none());
        assert!(v.get("scope_ref").is_none());
        assert_eq!(v["allow"], serde_json::json!(["*"]));

        let a: McpAccess = serde_json::from_str(r#"{"allow":["*"]}"#).unwrap();
        let v = serde_json::to_value(&a).unwrap();
        assert!(v.get("deny").is_none());
    }
}
