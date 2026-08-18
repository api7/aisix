//! `McpAuthSettings` entity — how the environment authenticates callers
//! of `/mcp`, stored in etcd under `mcp_auth_settings/<uuid>` (the
//! control plane keys the singleton row by the environment id).
//!
//! Two independent settings live here, both optional and both off
//! without the row:
//!
//! - `resource_url` activates the `/mcp` OAuth 2.1 resource-server
//!   discovery surface (AISIX-Cloud#1143) when at least one enabled
//!   [`OidcProvider`](super::oidc_provider::OidcProvider) also exists:
//!   the RFC 9728 Protected Resource Metadata document is served under
//!   `/.well-known/oauth-protected-resource`, and `/mcp` auth failures
//!   carry a `WWW-Authenticate` challenge pointing at it.
//! - `anonymous` lets callers that present NO credential at all reach
//!   named MCP entries as a bound API-key principal (AISIX-Cloud#1313).
//!   An invalid, expired or disabled credential is still rejected —
//!   anonymous is the no-credential path, never a downgrade.
//!
//! Both are absent by default, so an environment without the row keeps
//! the pre-#1143 behavior byte for byte: every `/mcp` request needs a
//! valid gateway credential and no discovery surface is published.
//!
//! At most one row exists per environment. The declarative resources
//! file rejects a document carrying more than one entry at load, and the
//! runtime resolvers fail closed if a duplicate reaches a live snapshot
//! anyway (a stale etcd key), rather than picking one by id order.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::resource::Resource;

// No `deny_unknown_fields`: since issue #871 strictness lives in the
// schema layer, not the structs. The strict write schema closes the
// root while the lenient read schema does not, which is what lets the
// etcd loader report a row carrying a newer cp-api field as partially
// compatible instead of dropping it.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpAuthSettings {
    /// Canonical URI of this environment's `/mcp` endpoint, e.g.
    /// `https://gw.example.com/mcp`. Published verbatim as the PRM
    /// document's `resource` (never derived from the request Host
    /// header) and the value the trust providers' `audiences` must
    /// include for OAuth-for-MCP tokens to validate. The URL path must
    /// be exactly `/mcp` — the gateway's fixed MCP route. Unset leaves
    /// the OAuth discovery surface dormant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub resource_url: Option<String>,

    /// Anonymous access to named `/mcp` entries. Unset (the default)
    /// means every `/mcp` request needs a valid gateway credential.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anonymous: Option<McpAnonymousAccess>,

    /// etcd-key uuid. Filled by the loader and never included in the
    /// JSON payload.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

/// Anonymous access configuration for this environment's `/mcp`
/// entries.
///
/// A request that carries NO gateway credential and arrives from
/// `source_cidrs` runs as the `api_key_id` principal: its MCP tool
/// grant, rate limits, budget, guardrails and usage attribution all
/// apply, so anonymous traffic stays governable instead of bypassing
/// the pipeline. A request carrying a credential is authenticated
/// normally and a bad one is rejected — never downgraded to anonymous.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpAnonymousAccess {
    /// Whether anonymous access is served. `false` keeps the
    /// configuration but closes the door, so an operator can suspend it
    /// without losing the principal and allowlists.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// The API key anonymous traffic runs as. Everything keyed on a
    /// principal — MCP tool ACL, per-server and per-key rate limits,
    /// budget, guardrail scopes, usage events — resolves through this
    /// key, which is why anonymous callers stay attributable. The key
    /// must carry an explicit MCP grant: a key left on `inherit` would
    /// pick up the environment-default policy, so an `all` default
    /// would silently hand every registered tool to anonymous callers
    /// (the control plane rejects that at write time).
    #[schemars(length(min = 1))]
    pub api_key_id: String,

    /// Client source CIDRs allowed to enter anonymously. Required and
    /// non-empty: with no credential to check, network reachability is
    /// the only gate in front of the principal. Matched against the
    /// source IP the proxy's real-ip chain resolves, never against a
    /// caller-supplied header value.
    #[schemars(length(min = 1))]
    pub source_cidrs: Vec<String>,

    /// Registered MCP server names reachable anonymously at
    /// `/mcp/{server}`.
    ///
    /// This list is ALSO the anonymous principal's ceiling: the tools of
    /// the listed servers are intersected with the key's own grant on
    /// BOTH entries. Without that, a key whose grant is wider than the
    /// list would let an anonymous caller reach an unlisted server's
    /// tools through the aggregated endpoint by naming
    /// `<server>__<tool>` directly — `/mcp/{server}` closed, aggregated
    /// `/mcp` open. Empty means no scoped entry is served (and, with
    /// `aggregate_entry`, that the aggregated one exposes nothing).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub servers: Vec<String>,

    /// Whether the aggregated `/mcp` endpoint serves anonymous callers.
    /// Off by default: it is the entry a standard MCP client uses for
    /// OAuth discovery, and its tools carry the `<server>__<tool>`
    /// namespace that a client migrating from a single-server endpoint
    /// does not use. Turning it on suppresses the `WWW-Authenticate`
    /// discovery hint there, since a no-credential request succeeds
    /// instead of producing the 401 that carries it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub aggregate_entry: bool,
}

fn default_enabled() -> bool {
    true
}

/// Cross-field rules for [`McpAnonymousAccess`], injected as an `allOf`
/// on the produced schema so the strict write path and the lenient etcd
/// read path enforce the same coupling.
pub fn mcp_auth_settings_coupling() -> Value {
    json!([
        // An anonymous block that names no entry can never serve a
        // request. Rejecting it beats accepting configuration that
        // silently does nothing (the operator's next move is to wonder
        // why anonymous "does not work").
        {
            "if": { "required": ["anonymous"] },
            "then": { "properties": { "anonymous": { "anyOf": [
                {
                    "title": "Scoped entries",
                    "required": ["servers"],
                    "properties": { "servers": { "minItems": 1 } }
                },
                {
                    "title": "Aggregated entry",
                    "required": ["aggregate_entry"],
                    "properties": { "aggregate_entry": { "const": true } }
                }
            ] } } }
        }
    ])
}

impl Resource for McpAuthSettings {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// Fixed identity: the row is a per-environment singleton, so the
    /// by-name index key is a constant rather than a user-chosen label.
    fn name(&self) -> &str {
        "mcp_auth_settings"
    }

    fn kind() -> &'static str {
        "mcp_auth_settings"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::schema::{validate_mcp_auth_settings, validate_mcp_auth_settings_lenient};

    #[test]
    fn deserialises_minimal_settings() {
        let s: McpAuthSettings =
            serde_json::from_str(r#"{"resource_url": "https://gw.example.com/mcp"}"#).unwrap();
        assert_eq!(
            s.resource_url.as_deref(),
            Some("https://gw.example.com/mcp")
        );
        assert!(s.anonymous.is_none());
    }

    #[test]
    fn unknown_fields_close_on_write_and_stay_tolerated_on_read() {
        // #871: the write contract rejects an unknown field, the etcd
        // read path tolerates it (and reports it), and serde must not
        // pre-empt either decision.
        let doc =
            serde_json::json!({"resource_url": "https://gw.example.com/mcp", "future_field": 1});
        assert!(validate_mcp_auth_settings(&doc).is_err());
        assert!(validate_mcp_auth_settings_lenient(&doc).is_ok());
        let parsed: McpAuthSettings = serde_json::from_value(doc).expect("serde stays tolerant");
        assert_eq!(
            parsed.resource_url.as_deref(),
            Some("https://gw.example.com/mcp")
        );
    }

    #[test]
    fn both_settings_are_optional() {
        // The row carries two independent settings; a row with neither
        // is inert, not invalid. cp-api writes one row per environment
        // and clearing one setting must not require deleting the other.
        assert!(validate_mcp_auth_settings(&serde_json::json!({})).is_ok());
        let s: McpAuthSettings = serde_json::from_str("{}").unwrap();
        assert!(s.resource_url.is_none());
        assert!(s.anonymous.is_none());
    }

    #[test]
    fn anonymous_needs_a_principal_and_a_source_allowlist() {
        // Both are load-bearing: the principal is what the request runs
        // as, and with no credential to check the CIDR list is the only
        // gate in front of it.
        assert!(validate_mcp_auth_settings(&serde_json::json!({
            "anonymous": { "source_cidrs": ["10.0.0.0/8"], "servers": ["docs"] }
        }))
        .is_err());
        assert!(validate_mcp_auth_settings(&serde_json::json!({
            "anonymous": { "api_key_id": "ak-1", "servers": ["docs"] }
        }))
        .is_err());
        assert!(validate_mcp_auth_settings(&serde_json::json!({
            "anonymous": {
                "api_key_id": "ak-1", "source_cidrs": [], "servers": ["docs"]
            }
        }))
        .is_err());
    }

    #[test]
    fn anonymous_must_name_at_least_one_entry() {
        let base = |extra: serde_json::Value| {
            let mut anon = serde_json::json!({
                "api_key_id": "ak-1",
                "source_cidrs": ["10.0.0.0/8"]
            });
            let obj = anon.as_object_mut().unwrap();
            for (k, v) in extra.as_object().unwrap() {
                obj.insert(k.clone(), v.clone());
            }
            serde_json::json!({ "anonymous": anon })
        };
        // Neither entry named: the block could never serve a request.
        assert!(validate_mcp_auth_settings(&base(serde_json::json!({}))).is_err());
        assert!(validate_mcp_auth_settings(&base(serde_json::json!({ "servers": [] }))).is_err());
        assert!(
            validate_mcp_auth_settings(&base(serde_json::json!({ "aggregate_entry": false })))
                .is_err()
        );
        // Either one alone is enough.
        assert!(
            validate_mcp_auth_settings(&base(serde_json::json!({ "servers": ["docs"] }))).is_ok()
        );
        assert!(
            validate_mcp_auth_settings(&base(serde_json::json!({ "aggregate_entry": true })))
                .is_ok()
        );
    }

    #[test]
    fn anonymous_defaults_to_enabled() {
        let s: McpAuthSettings = serde_json::from_value(serde_json::json!({
            "anonymous": {
                "api_key_id": "ak-1",
                "source_cidrs": ["10.0.0.0/8"],
                "servers": ["docs"]
            }
        }))
        .unwrap();
        let anon = s.anonymous.expect("anonymous block");
        assert!(anon.enabled);
        assert!(!anon.aggregate_entry, "the aggregated entry stays opt-in");
        assert_eq!(anon.servers, ["docs"]);
    }

    #[test]
    fn resource_trait_uses_fixed_identity() {
        assert_eq!(McpAuthSettings::kind(), "mcp_auth_settings");
        let mut s: McpAuthSettings =
            serde_json::from_str(r#"{"resource_url": "https://gw.example.com/mcp"}"#).unwrap();
        s.runtime_id = "env-uuid-1".into();
        assert_eq!(s.id(), "env-uuid-1");
        assert_eq!(s.name(), "mcp_auth_settings");
    }
}
