//! `McpAuthSettings` entity — the environment's inbound MCP OAuth
//! discovery identity, stored in etcd under `mcp_auth_settings/<uuid>`
//! (the control plane keys the singleton row by the environment id).
//!
//! Carrying a valid row activates the `/mcp` OAuth 2.1 resource-server
//! discovery surface (AISIX-Cloud#1143) when at least one enabled
//! [`OidcProvider`](super::oidc_provider::OidcProvider) also exists:
//! the RFC 9728 Protected Resource Metadata document is served under
//! `/.well-known/oauth-protected-resource`, and `/mcp` auth failures
//! carry a `WWW-Authenticate` challenge pointing at it. Without the
//! row the surface stays dormant and behavior is unchanged.
//!
//! At most one row exists per environment; the declarative resources
//! file rejects a document carrying more than one entry at load.

use serde::{Deserialize, Serialize};

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
    /// be exactly `/mcp` — the gateway's fixed MCP route.
    #[schemars(length(min = 1))]
    pub resource_url: String,

    /// etcd-key uuid. Filled by the loader and never included in the
    /// JSON payload.
    #[serde(skip)]
    pub(crate) runtime_id: String,
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

    #[test]
    fn deserialises_minimal_settings() {
        let s: McpAuthSettings =
            serde_json::from_str(r#"{"resource_url": "https://gw.example.com/mcp"}"#).unwrap();
        assert_eq!(s.resource_url, "https://gw.example.com/mcp");
    }

    #[test]
    fn unknown_fields_close_on_write_and_stay_tolerated_on_read() {
        // #871: the write contract rejects an unknown field, the etcd
        // read path tolerates it (and reports it), and serde must not
        // pre-empt either decision.
        let doc =
            serde_json::json!({"resource_url": "https://gw.example.com/mcp", "future_field": 1});
        assert!(crate::models::schema::validate_mcp_auth_settings(&doc).is_err());
        assert!(crate::models::schema::validate_mcp_auth_settings_lenient(&doc).is_ok());
        let parsed: McpAuthSettings = serde_json::from_value(doc).expect("serde stays tolerant");
        assert_eq!(parsed.resource_url, "https://gw.example.com/mcp");
    }

    #[test]
    fn rejects_missing_resource_url() {
        let r: Result<McpAuthSettings, _> = serde_json::from_str(r#"{}"#);
        assert!(r.is_err());
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
