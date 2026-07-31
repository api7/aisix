//! `OidcProvider` entity — an external identity provider the gateway
//! trusts for inbound JWT authentication, stored in etcd under
//! `oidc_providers/<uuid>`.
//!
//! When at least one enabled provider exists, a request whose bearer
//! token is a JWT (instead of a gateway API key) is authenticated
//! against the provider matching the token's `iss` claim: the token's
//! signature is verified against the provider's JWKS, its registered
//! claims (`exp`, `aud`) and the provider's scope/claim requirements
//! are enforced, and the value of `identity_claim` selects the API key
//! whose `jwt_subject` equals it. The request then proceeds with that
//! key's permissions, rate limits, and budget — external identities
//! never widen access beyond a key an operator explicitly created.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// Expected value(s) for one bound claim: a single string, or a list
/// matched as any-of.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum BoundClaimExpect {
    /// The claim must equal (or, for array claims, contain) this value.
    One(String),
    /// The claim must equal (or, for array claims, contain) at least one
    /// of these values.
    #[schemars(length(min = 1))]
    Any(Vec<String>),
}

impl BoundClaimExpect {
    /// Iterate the accepted values regardless of form.
    pub fn accepted(&self) -> impl Iterator<Item = &str> {
        match self {
            BoundClaimExpect::One(v) => std::slice::from_ref(v).iter().map(String::as_str),
            BoundClaimExpect::Any(vs) => vs.as_slice().iter().map(String::as_str),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OidcProvider {
    /// Human-readable provider name, unique within the environment
    /// (e.g. `"corp-keycloak"`).
    #[schemars(length(min = 1))]
    pub name: String,

    /// Expected `iss` claim, compared byte-for-byte against the token's
    /// issuer. A JWT whose issuer matches no enabled provider is
    /// rejected.
    #[schemars(length(min = 1))]
    pub issuer: String,

    /// Accepted `aud` values. The token's audience (a string or an
    /// array) must contain at least one of these. Every token must
    /// carry an audience claim.
    #[schemars(length(min = 1))]
    pub audiences: Vec<String>,

    /// JWKS endpoint URL the signing keys are fetched from. When
    /// omitted, the endpoint is resolved once from the issuer's OIDC
    /// discovery document (`<issuer>/.well-known/openid-configuration`)
    /// and cached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub jwks_uri: Option<String>,

    /// Claim whose value selects the API key to act as: the request is
    /// bound to the key whose `jwt_subject` equals this claim's value.
    /// Dots traverse nested objects (e.g. `"resource_access.account"`).
    /// Defaults to `sub`.
    #[serde(default = "default_identity_claim")]
    #[schemars(length(min = 1))]
    pub identity_claim: String,

    /// Scopes that must all be present in the token's `scope` claim
    /// (a space-delimited string or an array of strings). An empty list
    /// requires nothing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_scopes: Vec<String>,

    /// Additional claim requirements, all of which must hold. Keys name
    /// claims (dots traverse nested objects, e.g.
    /// `"realm_access.roles"`); each requirement is satisfied when the
    /// claim equals — or, for array claims, contains — one of the
    /// expected values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_claims: Option<BTreeMap<String, BoundClaimExpect>>,

    /// Clock-skew allowance in seconds applied to time-based claims
    /// (`exp`, `nbf`). Defaults to 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    #[schemars(range(max = 300))]
    pub leeway_secs: u64,

    /// Whether the provider participates in JWT authentication. A
    /// disabled provider is kept but ignored. Treated as `true` when
    /// omitted.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// etcd-key uuid. Filled by the loader and never included in the
    /// JSON payload.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

fn default_identity_claim() -> String {
    "sub".to_string()
}

fn default_enabled() -> bool {
    true
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

impl Resource for OidcProvider {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// The by-name index key is the display name. Issuer lookups during
    /// authentication iterate and filter on `issuer` rather than relying
    /// on this index, so a duplicate name can never shadow a provider.
    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "oidc_providers"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_minimal_provider_with_defaults() {
        let p: OidcProvider = serde_json::from_str(
            r#"{
              "name": "corp-keycloak",
              "issuer": "https://sso.example.com/realms/agents",
              "audiences": ["aisix-gateway"]
            }"#,
        )
        .unwrap();
        assert_eq!(p.name, "corp-keycloak");
        assert_eq!(p.issuer, "https://sso.example.com/realms/agents");
        assert_eq!(p.audiences, vec!["aisix-gateway"]);
        assert!(p.jwks_uri.is_none());
        assert_eq!(p.identity_claim, "sub");
        assert!(p.required_scopes.is_empty());
        assert!(p.bound_claims.is_none());
        assert_eq!(p.leeway_secs, 0);
        assert!(p.enabled);
    }

    #[test]
    fn deserialises_full_provider() {
        let p: OidcProvider = serde_json::from_str(
            r#"{
              "name": "corp-keycloak",
              "issuer": "https://sso.example.com/realms/agents",
              "audiences": ["aisix-gateway", "aisix-alt"],
              "jwks_uri": "https://sso.example.com/realms/agents/protocol/openid-connect/certs",
              "identity_claim": "azp",
              "required_scopes": ["ai.access"],
              "bound_claims": {
                "department": "ai-lab",
                "realm_access.roles": ["agent", "batch-agent"]
              },
              "leeway_secs": 30,
              "enabled": false
            }"#,
        )
        .unwrap();
        assert_eq!(p.audiences.len(), 2);
        assert_eq!(p.identity_claim, "azp");
        assert_eq!(p.required_scopes, vec!["ai.access"]);
        let bound = p.bound_claims.as_ref().unwrap();
        assert_eq!(
            bound.get("department"),
            Some(&BoundClaimExpect::One("ai-lab".into()))
        );
        assert_eq!(
            bound.get("realm_access.roles"),
            Some(&BoundClaimExpect::Any(vec![
                "agent".into(),
                "batch-agent".into()
            ]))
        );
        assert_eq!(p.leeway_secs, 30);
        assert!(!p.enabled);
    }

    #[test]
    fn rejects_unknown_fields() {
        let r: Result<OidcProvider, _> = serde_json::from_str(
            r#"{"name":"x","issuer":"https://x","audiences":["a"],"extra":1}"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn defaults_stay_off_the_wire() {
        let p: OidcProvider =
            serde_json::from_str(r#"{"name":"x","issuer":"https://x","audiences":["a"]}"#).unwrap();
        let v = serde_json::to_value(&p).unwrap();
        assert!(v.get("jwks_uri").is_none());
        assert!(v.get("required_scopes").is_none());
        assert!(v.get("bound_claims").is_none());
        assert!(v.get("leeway_secs").is_none());
        // identity_claim and enabled serialize with their default values —
        // both are meaningful to echo back through the Admin API.
        assert_eq!(v["identity_claim"], "sub");
        assert_eq!(v["enabled"], true);
    }

    #[test]
    fn bound_claim_expect_accepted_iterates_both_forms() {
        let one = BoundClaimExpect::One("a".into());
        assert_eq!(one.accepted().collect::<Vec<_>>(), vec!["a"]);
        let any = BoundClaimExpect::Any(vec!["a".into(), "b".into()]);
        assert_eq!(any.accepted().collect::<Vec<_>>(), vec!["a", "b"]);
    }

    #[test]
    fn resource_trait_points_at_name_and_kind() {
        assert_eq!(OidcProvider::kind(), "oidc_providers");
        let mut p: OidcProvider =
            serde_json::from_str(r#"{"name":"corp","issuer":"https://x","audiences":["a"]}"#)
                .unwrap();
        p.runtime_id = "op-1".into();
        assert_eq!(p.id(), "op-1");
        assert_eq!(p.name(), "corp");
    }
}
