//! `ClaimMapping` entity — a rule mapping verified JWT claims to an
//! existing API key, stored in etcd under `claim_mappings/<uuid>`
//! (AISIX-Cloud#564).
//!
//! The direct `(jwt_provider, jwt_subject)` binding on an API key admits
//! exactly one pre-registered identity per key. Claim mappings admit a
//! *class* of identities instead: after a token passes the full
//! [`OidcProvider`](super::oidc_provider::OidcProvider) verification and
//! no key binds the token's subject directly, the enabled mappings whose
//! `jwt_provider` names the matched trust provider are evaluated in
//! `priority` order (ties broken by `name`), and the first mapping whose
//! `match` conditions all hold selects the API key named by
//! `resolve.api_key_id`. The request then runs as that key — its model
//! and tool access, rate limits, and budget apply unchanged, and the
//! token's identity claim is recorded for usage attribution.
//!
//! Claims only ever *select* a key an operator already created; no claim
//! value becomes configuration. A token matching no mapping is rejected,
//! never admitted with defaults.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// How one [`ClaimMatch`] compares the claim's value against `values`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClaimMatchOp {
    /// The claim must be a string equal to one of `values`. An array
    /// claim never matches `exact`.
    Exact,
    /// The claim must be an array containing one of `values` among its
    /// string items; non-string items are ignored, matching the
    /// `bound_claims` array semantics. A string claim never matches
    /// `contains`.
    Contains,
}

/// One claim condition. A mapping matches only when every condition
/// holds (logical AND); within one condition, `values` are alternatives
/// (logical OR).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ClaimMatch {
    /// Claim to inspect. Dots traverse nested objects (for example
    /// `realm_access.roles`). A missing claim never matches.
    #[schemars(length(min = 1))]
    pub claim: String,

    /// Comparison operator. A claim whose JSON type does not fit the
    /// operator (an array for `exact`, a string for `contains`) never
    /// matches — mistyped claims deny rather than surprise.
    pub op: ClaimMatchOp,

    /// Accepted values; the condition holds when any one matches.
    #[schemars(length(min = 1))]
    pub values: Vec<String>,
}

/// What a matched mapping resolves to. Targets always reference
/// existing resources — a dangling reference rejects the request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ClaimResolve {
    /// Id of the API key the request runs as. The key's model and tool
    /// access, rate limits, and budget apply exactly as if the caller
    /// had presented the key itself.
    #[schemars(length(min = 1))]
    pub api_key_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ClaimMapping {
    /// Human-readable mapping name, unique within the environment.
    #[schemars(length(min = 1))]
    pub name: String,

    /// Name of the OIDC provider whose tokens this mapping applies to.
    /// A mapping never matches a token verified by a different
    /// provider, so two providers cannot select each other's keys.
    #[schemars(length(min = 1))]
    pub jwt_provider: String,

    /// Evaluation order among the provider's mappings: lower values are
    /// evaluated first, ties are broken by `name`. Defaults to 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub priority: u32,

    /// Claim conditions, all of which must hold for the mapping to
    /// match.
    #[serde(rename = "match")]
    #[schemars(length(min = 1))]
    pub match_: Vec<ClaimMatch>,

    /// The API key a matching token resolves to.
    pub resolve: ClaimResolve,

    /// Whether the mapping participates in evaluation. A disabled
    /// mapping is kept but skipped. Treated as `true` when omitted.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// etcd-key uuid. Filled by the loader and never included in the
    /// JSON payload.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

fn default_enabled() -> bool {
    true
}

fn is_zero(v: &u32) -> bool {
    *v == 0
}

impl Resource for ClaimMapping {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "claim_mappings"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_minimal_mapping_with_defaults() {
        let m: ClaimMapping = serde_json::from_str(
            r#"{
              "name": "finance-dept",
              "jwt_provider": "corp-keycloak",
              "match": [
                {"claim": "department", "op": "exact", "values": ["finance"]}
              ],
              "resolve": {"api_key_id": "11111111-1111-1111-1111-111111111111"}
            }"#,
        )
        .unwrap();
        assert_eq!(m.name, "finance-dept");
        assert_eq!(m.jwt_provider, "corp-keycloak");
        assert_eq!(m.priority, 0);
        assert_eq!(m.match_.len(), 1);
        assert_eq!(m.match_[0].claim, "department");
        assert_eq!(m.match_[0].op, ClaimMatchOp::Exact);
        assert_eq!(m.match_[0].values, vec!["finance"]);
        assert_eq!(m.resolve.api_key_id, "11111111-1111-1111-1111-111111111111");
        assert!(m.enabled);
    }

    #[test]
    fn deserialises_full_mapping() {
        let m: ClaimMapping = serde_json::from_str(
            r#"{
              "name": "mcp-admins",
              "jwt_provider": "corp-keycloak",
              "priority": 200,
              "match": [
                {"claim": "groups", "op": "contains", "values": ["mcp-admin", "platform"]},
                {"claim": "realm_access.department", "op": "exact", "values": ["ai-lab"]}
              ],
              "resolve": {"api_key_id": "k-admin"},
              "enabled": false
            }"#,
        )
        .unwrap();
        assert_eq!(m.priority, 200);
        assert_eq!(m.match_.len(), 2);
        assert_eq!(m.match_[0].op, ClaimMatchOp::Contains);
        assert_eq!(m.match_[1].claim, "realm_access.department");
        assert!(!m.enabled);
    }

    #[test]
    fn tolerates_unknown_fields_for_forward_compat() {
        // A newer control plane may ship fields ahead of this DP; serde
        // must accept them. The write path still rejects them via the
        // strict schema validator (validate_claim_mapping in models/schema.rs).
        let m: ClaimMapping = serde_json::from_str(
            r#"{
              "name": "x",
              "jwt_provider": "p",
              "match": [{"claim": "c", "op": "exact", "values": ["v"]}],
              "resolve": {"api_key_id": "k"},
              "extra": 1
            }"#,
        )
        .unwrap();
        assert_eq!(m.name, "x");
    }

    #[test]
    fn defaults_stay_off_the_wire() {
        let m: ClaimMapping = serde_json::from_str(
            r#"{
              "name": "x",
              "jwt_provider": "p",
              "match": [{"claim": "c", "op": "exact", "values": ["v"]}],
              "resolve": {"api_key_id": "k"}
            }"#,
        )
        .unwrap();
        let v = serde_json::to_value(&m).unwrap();
        assert!(v.get("priority").is_none());
        // enabled serialises with its default value — meaningful to echo
        // back through exports, matching OidcProvider.
        assert_eq!(v["enabled"], true);
        // The matcher list round-trips under the wire name `match`.
        assert_eq!(v["match"][0]["op"], "exact");
    }

    #[test]
    fn resource_trait_points_at_name_and_kind() {
        assert_eq!(ClaimMapping::kind(), "claim_mappings");
        let mut m: ClaimMapping = serde_json::from_str(
            r#"{
              "name": "finance-dept",
              "jwt_provider": "p",
              "match": [{"claim": "c", "op": "exact", "values": ["v"]}],
              "resolve": {"api_key_id": "k"}
            }"#,
        )
        .unwrap();
        m.runtime_id = "cm-1".into();
        assert_eq!(m.id(), "cm-1");
        assert_eq!(m.name(), "finance-dept");
    }
}
