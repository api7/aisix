//! `McpServer` entity — a registered upstream MCP server.
//!
//! Registers an upstream Model Context Protocol (MCP) server so the gateway can
//! front it: its tools are aggregated into the gateway's own MCP endpoint under
//! the namespace `<display_name>__<tool>`, and tool calls are routed back to it.
//! The upstream credential is held by the gateway and is never exposed to the
//! calling client.
//!
//! etcd path: `{prefix}/mcp_servers/{uuid}`. Secondary index on `display_name`.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct McpServer {
    /// Operator-facing label, unique within the gateway. It is used as the
    /// namespace prefix for this server's tools, which are exposed to clients as
    /// `<display_name>__<tool>`, so it must not contain the reserved separator
    /// `__`.
    #[schemars(length(min = 1))]
    pub display_name: String,

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

    /// Authentication credential for the upstream server. Required when
    /// `auth_type` is `bearer`, where it is sent as `Authorization: Bearer
    /// <secret>` on every upstream request. Leave unset when `auth_type` is
    /// `none`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,

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
}

impl Resource for McpServer {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.display_name
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
        assert_eq!(s.display_name, "github");
        assert_eq!(s.url, "https://api.example.com/mcp");
        // Defaults.
        assert_eq!(s.transport, McpTransport::StreamableHttp);
        assert_eq!(s.auth_type, McpAuthType::None);
        assert!(s.secret.is_none());
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
    fn rejects_unknown_fields() {
        let r: Result<McpServer, _> =
            serde_json::from_str(r#"{"display_name":"x","url":"u","extra":1}"#);
        assert!(r.is_err());
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
    fn resource_trait_routes_through_display_name() {
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
            display_name: "github".into(),
            url: "https://x/mcp".into(),
            transport: McpTransport::StreamableHttp,
            auth_type: McpAuthType::None,
            secret: None,
            timeout_ms: None,
            enabled: true,
            runtime_id: String::new(),
        };
        let s = serde_json::to_string(&original).unwrap();
        let back: McpServer = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }
}
