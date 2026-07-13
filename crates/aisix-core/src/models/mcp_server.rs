//! `McpServer` entity — a registered upstream MCP server.
//!
//! Registers an upstream Model Context Protocol (MCP) server so the gateway can
//! front it: its tools are aggregated into the gateway's own MCP endpoint under
//! the namespace `<name>__<tool>`, and tool calls are routed back to it.
//! The upstream credential is held by the gateway and is never exposed to the
//! calling client.
//!
//! etcd path: `{prefix}/mcp_servers/{uuid}`. Secondary index on `name`.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct McpServer {
    /// Operator-facing label, unique within the gateway. It is used as the
    /// namespace prefix for this server's tools, which are exposed to clients as
    /// `<name>__<tool>`, so it must not contain the reserved separator `__`.
    // `display_name` is the field's former name; stored documents and
    // callers that still use it keep deserializing (schema-side acceptance
    // lives in `schema::mcp_server_root_schema`). Re-serialization always
    // emits `name`.
    #[serde(alias = "display_name")]
    #[schemars(length(min = 1))]
    pub name: String,

    /// The upstream server's MCP endpoint URL, reached over the Streamable HTTP
    /// transport, such as `https://api.example.com/mcp`.
    #[schemars(length(min = 1))]
    pub url: String,

    /// Transport used to reach the upstream server. Streamable HTTP is the only
    /// supported transport.
    #[serde(default)]
    pub transport: McpTransport,

    /// How the gateway authenticates to the upstream server. The credential is
    /// held by the gateway and is never forwarded from or exposed to the calling
    /// client.
    #[serde(default)]
    pub auth_type: McpAuthType,

    /// Authentication credential for the upstream server. Its meaning follows
    /// `auth_type`: the bearer token when `auth_type` is `bearer` (sent as
    /// `Authorization: Bearer <secret>`), the API key when `auth_type` is
    /// `api_key` (sent as `x-api-key: <secret>`), or the OAuth client secret
    /// when `auth_type` is `oauth2`. Leave unset when `auth_type` is `none`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,

    // Cross-field coupling (`oauth2` requires `client_id` + `secret` +
    // `token_url`; `bearer`/`api_key` require `secret`) is deliberately NOT
    // expressed in this flat schema — that would force restructuring the
    // resource into a oneOf. The control plane enforces the coupling strictly
    // at write time, this gateway's own Admin API re-checks it on write, and
    // the runtime degrades gracefully when a snapshot-loaded server is
    // mis-configured: its credential exchange fails, its tools become
    // unavailable, and the failure is logged like any other upstream failure.
    /// OAuth client identifier used for the OAuth 2.0 client credentials
    /// grant. Required when `auth_type` is `oauth2`; ignored otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,

    /// OAuth token endpoint URL where the gateway exchanges the client
    /// credentials for an access token, such as
    /// `https://auth.example.com/oauth/token`. Required when `auth_type` is
    /// `oauth2`; ignored otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_url: Option<String>,

    /// OAuth scopes to request. Joined with spaces into the `scope` parameter
    /// of the token request. Only used when `auth_type` is `oauth2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scopes: Option<Vec<String>>,

    /// Maximum time, in milliseconds, to wait for a single upstream operation
    /// (establishing the session, listing tools, or calling a tool). Must be at
    /// least `1` when set. When omitted, the gateway applies a built-in default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub timeout_ms: Option<u64>,

    /// Whether this server is active. When `false`, its tools are not listed and
    /// cannot be called.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Filled in by the snapshot loader from the etcd key path.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

fn default_enabled() -> bool {
    true
}

/// Transport used to reach an upstream MCP server.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    /// Streamable HTTP transport: a single endpoint that serves both POST and
    /// GET.
    #[default]
    StreamableHttp,
}

/// How the gateway authenticates to an upstream MCP server.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum McpAuthType {
    /// No authentication; the server is reached as-is.
    #[default]
    None,
    /// Bearer token authentication. The token is supplied in `secret` and sent
    /// as `Authorization: Bearer <secret>`.
    Bearer,
    /// API key authentication. The key is supplied in `secret` and sent as an
    /// `x-api-key: <secret>` header on every upstream request.
    ApiKey,
    /// OAuth 2.0 client credentials grant. The gateway exchanges `client_id`,
    /// the client secret in `secret`, and the optional `scopes` at `token_url`
    /// for an access token, and sends it as `Authorization: Bearer
    /// <access_token>` on every upstream request. Access tokens are cached
    /// until shortly before their reported expiry.
    #[serde(rename = "oauth2")]
    OAuth2,
}

impl Resource for McpServer {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "mcp_servers"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_minimal_mcp_server() {
        let s: McpServer = serde_json::from_str(
            r#"{"display_name":"github","url":"https://api.example.com/mcp"}"#,
        )
        .unwrap();
        assert_eq!(s.name, "github");
        assert_eq!(s.url, "https://api.example.com/mcp");
        // Defaults.
        assert_eq!(s.transport, McpTransport::StreamableHttp);
        assert_eq!(s.auth_type, McpAuthType::None);
        assert!(s.secret.is_none());
        assert!(s.client_id.is_none());
        assert!(s.token_url.is_none());
        assert!(s.scopes.is_none());
        assert!(s.timeout_ms.is_none());
        assert!(s.enabled);
    }

    #[test]
    fn deserialises_with_bearer_auth() {
        let s: McpServer = serde_json::from_str(
            r#"{"display_name":"gh","url":"https://x/mcp","auth_type":"bearer","secret":"tok","timeout_ms":5000,"enabled":false}"#,
        )
        .unwrap();
        assert_eq!(s.auth_type, McpAuthType::Bearer);
        assert_eq!(s.secret.as_deref(), Some("tok"));
        assert_eq!(s.timeout_ms, Some(5000));
        assert!(!s.enabled);
    }

    #[test]
    fn deserialises_with_api_key_auth() {
        let s: McpServer = serde_json::from_str(
            r#"{"display_name":"gh","url":"https://x/mcp","auth_type":"api_key","secret":"k-1"}"#,
        )
        .unwrap();
        assert_eq!(s.auth_type, McpAuthType::ApiKey);
        assert_eq!(s.secret.as_deref(), Some("k-1"));
    }

    #[test]
    fn deserialises_with_oauth2_auth() {
        let s: McpServer = serde_json::from_str(
            r#"{"display_name":"gh","url":"https://x/mcp","auth_type":"oauth2","secret":"cs-1","client_id":"cid","token_url":"https://auth/x/token","scopes":["read","write"]}"#,
        )
        .unwrap();
        assert_eq!(s.auth_type, McpAuthType::OAuth2);
        assert_eq!(s.secret.as_deref(), Some("cs-1"));
        assert_eq!(s.client_id.as_deref(), Some("cid"));
        assert_eq!(s.token_url.as_deref(), Some("https://auth/x/token"));
        assert_eq!(
            s.scopes.as_deref(),
            Some(&["read".to_string(), "write".to_string()][..])
        );
    }

    #[test]
    fn oauth2_round_trips_and_omits_unset_optionals() {
        let original: McpServer = serde_json::from_str(
            r#"{"display_name":"gh","url":"https://x/mcp","auth_type":"oauth2","secret":"cs","client_id":"cid","token_url":"https://auth/token"}"#,
        )
        .unwrap();
        let s = serde_json::to_string(&original).unwrap();
        // The oauth2 tag serialises as `oauth2` (not a snake_cased `o_auth2`)
        // and unset optionals (`scopes` here) are omitted entirely.
        assert!(s.contains(r#""auth_type":"oauth2""#), "got: {s}");
        assert!(!s.contains("scopes"), "unset scopes must be omitted: {s}");
        let back: McpServer = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn rejects_unknown_fields() {
        let r: Result<McpServer, _> =
            serde_json::from_str(r#"{"display_name":"x","url":"u","extra":1}"#);
        assert!(r.is_err());
    }

    // ---- `display_name` → `name` rename ----

    #[test]
    fn accepts_canonical_name_spelling() {
        let s: McpServer =
            serde_json::from_str(r#"{"name":"github","url":"https://x/mcp"}"#).unwrap();
        assert_eq!(s.name, "github");
    }

    #[test]
    fn serialises_label_under_name_only() {
        // Emission contract: re-serialization uses the canonical `name`,
        // never the former `display_name` spelling (the fixtures above
        // keep exercising the deserialize-side alias).
        let s: McpServer =
            serde_json::from_str(r#"{"display_name":"github","url":"https://x/mcp"}"#).unwrap();
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains(r#""name":"github""#), "got: {json}");
        assert!(!json.contains("display_name"), "got: {json}");
    }

    #[test]
    fn rejects_document_carrying_both_spellings() {
        // serde maps the alias onto the same field, so a document that
        // carries both spellings is a duplicate-field error — the
        // ambiguity is rejected instead of one value silently winning.
        let r: Result<McpServer, _> = serde_json::from_str(
            r#"{"name":"github","display_name":"github-old","url":"https://x/mcp"}"#,
        );
        let err = r.expect_err("both spellings in one document must be rejected");
        assert!(
            err.to_string().contains("duplicate field"),
            "expected a duplicate-field error, got: {err}"
        );
    }

    #[test]
    fn rejects_unknown_transport_and_auth_type() {
        assert!(serde_json::from_str::<McpServer>(
            r#"{"display_name":"x","url":"u","transport":"stdio"}"#
        )
        .is_err());
        assert!(serde_json::from_str::<McpServer>(
            r#"{"display_name":"x","url":"u","auth_type":"oauth"}"#
        )
        .is_err());
    }

    #[test]
    fn resource_trait_routes_through_name() {
        let mut s: McpServer =
            serde_json::from_str(r#"{"display_name":"github","url":"https://x/mcp"}"#).unwrap();
        s.runtime_id = "uuid-mcp-1".into();
        assert_eq!(<McpServer as Resource>::kind(), "mcp_servers");
        assert_eq!(s.id(), "uuid-mcp-1");
        assert_eq!(s.name(), "github");
    }

    #[test]
    fn round_trip_omits_default_optionals() {
        let original = McpServer {
            name: "github".into(),
            url: "https://x/mcp".into(),
            transport: McpTransport::StreamableHttp,
            auth_type: McpAuthType::None,
            secret: None,
            client_id: None,
            token_url: None,
            scopes: None,
            timeout_ms: None,
            enabled: true,
            runtime_id: String::new(),
        };
        let s = serde_json::to_string(&original).unwrap();
        let back: McpServer = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }
}
