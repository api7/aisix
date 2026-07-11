//! `A2aAgent` entity — a registered upstream A2A (Agent-to-Agent) agent.
//!
//! Registers an upstream agent that speaks the A2A protocol (HTTP + JSON-RPC
//! 2.0) so the gateway can front it: callers reach it through the gateway's own
//! `/a2a/<display_name>` endpoint, its agent card is served (with URLs rewritten
//! to the gateway) at `/a2a/<display_name>/.well-known/agent.json`, and
//! `message/send` / `message/stream` are routed through the same auth / ACL /
//! guardrail / quota pipeline as LLM and MCP traffic. The upstream credential is
//! held by the gateway and is never exposed to the calling client.
//!
//! This is the `a2a_http` backend: a self-hosted agent reached over HTTP.
//! Managed-platform backends (Bedrock AgentCore, Azure AI Foundry, Vertex Agent
//! Engine) and gateway-composed virtual agents are later additions and are not
//! part of this entity yet.
//!
//! etcd path: `{prefix}/a2a_agents/{uuid}`. Secondary index on `display_name`.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct A2aAgent {
    /// Operator-facing label, unique within the gateway. It is the path segment
    /// under which the agent is exposed to callers as `/a2a/<display_name>`, so
    /// it must be a single non-empty URL path segment.
    #[schemars(length(min = 1))]
    pub display_name: String,

    /// The upstream agent's base URL, such as `https://agents.example.com/a2a`.
    /// AISIX reaches this URL over HTTP with the A2A JSON-RPC 2.0 protocol and
    /// discovers the agent card relative to it.
    #[schemars(length(min = 1))]
    pub url: String,

    /// The A2A wire-format version AISIX uses for this agent. AISIX pins the
    /// version explicitly so the served agent card and accepted requests stay
    /// consistent.
    #[serde(default)]
    pub protocol_version: A2aProtocolVersion,

    /// How the gateway authenticates to the upstream agent. The credential is
    /// held by the gateway and is never forwarded from or exposed to the calling
    /// client.
    #[serde(default)]
    pub auth_type: A2aAuthType,

    /// Credential AISIX uses to authenticate to the upstream agent. For
    /// `bearer`, AISIX sends it as `Authorization: Bearer <secret>`; for
    /// `api_key`, AISIX sends it as `x-api-key: <secret>`. Leave unset for
    /// `none`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,

    // Cross-field coupling (`bearer`/`api_key` require `secret`) is
    // deliberately NOT expressed in this flat schema — that would force
    // restructuring the resource into a oneOf. The control plane enforces the
    // coupling strictly at write time, this gateway's own Admin API re-checks
    // it on write, and the runtime degrades gracefully when a snapshot-loaded
    // agent is misconfigured.
    /// Maximum time, in milliseconds, to wait for a single upstream operation,
    /// including fetching the agent card or invoking the agent. When omitted,
    /// AISIX applies a built-in default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub timeout_ms: Option<u64>,

    /// Whether this agent is active. When `false`, it is not served and cannot
    /// be reached.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Filled in by the snapshot loader from the etcd key path.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

fn default_enabled() -> bool {
    true
}

/// The A2A wire-format version pinned for an upstream agent.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
pub enum A2aProtocolVersion {
    /// A2A 1.0 wire format with protobuf-JSON envelopes and PascalCase methods.
    #[default]
    #[serde(rename = "1.0")]
    V1_0,
    /// A2A 0.3 wire format with `kind`-discriminated JSON-RPC objects.
    #[serde(rename = "0.3")]
    V0_3,
}

/// How the gateway authenticates to an upstream A2A agent.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum A2aAuthType {
    /// No authentication; the agent is reached as-is.
    #[default]
    None,
    /// Bearer token authentication. The token is supplied in `secret` and sent
    /// as `Authorization: Bearer <secret>`.
    Bearer,
    /// API key authentication. The key is supplied in `secret` and sent as an
    /// `x-api-key: <secret>` header on every upstream request.
    ApiKey,
}

impl Resource for A2aAgent {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.display_name
    }

    fn kind() -> &'static str {
        "a2a_agents"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_minimal_a2a_agent() {
        let a: A2aAgent = serde_json::from_str(
            r#"{"display_name":"invoice-processor","url":"https://agents.example.com/a2a"}"#,
        )
        .unwrap();
        assert_eq!(a.display_name, "invoice-processor");
        assert_eq!(a.url, "https://agents.example.com/a2a");
        // Defaults.
        assert_eq!(a.protocol_version, A2aProtocolVersion::V1_0);
        assert_eq!(a.auth_type, A2aAuthType::None);
        assert!(a.secret.is_none());
        assert!(a.timeout_ms.is_none());
        assert!(a.enabled);
    }

    #[test]
    fn deserialises_with_bearer_auth_and_pinned_v0_3() {
        let a: A2aAgent = serde_json::from_str(
            r#"{"display_name":"tr","url":"https://x/a2a","protocol_version":"0.3","auth_type":"bearer","secret":"tok","timeout_ms":5000,"enabled":false}"#,
        )
        .unwrap();
        assert_eq!(a.protocol_version, A2aProtocolVersion::V0_3);
        assert_eq!(a.auth_type, A2aAuthType::Bearer);
        assert_eq!(a.secret.as_deref(), Some("tok"));
        assert_eq!(a.timeout_ms, Some(5000));
        assert!(!a.enabled);
    }

    #[test]
    fn rejects_oauth2_auth_type_and_oauth_fields() {
        // `auth_type` accepts only `none` / `bearer` / `api_key` — the same
        // closed set as the control plane's a2a_agent resource.
        assert!(serde_json::from_str::<A2aAgent>(
            r#"{"display_name":"a","url":"https://x/a2a","auth_type":"oauth2","secret":"cs-1"}"#,
        )
        .is_err());
        // The OAuth-specific fields were removed with the `oauth2` arm, so a
        // document carrying one is rejected as an unknown field.
        for field in [
            r#""client_id":"cid""#,
            r#""token_url":"https://auth/x/token""#,
            r#""scopes":["read","write"]"#,
        ] {
            let doc = format!(r#"{{"display_name":"a","url":"https://x/a2a",{field}}}"#);
            assert!(
                serde_json::from_str::<A2aAgent>(&doc).is_err(),
                "field must be rejected as unknown: {doc}"
            );
        }
    }

    #[test]
    fn protocol_version_serialises_as_dotted_string() {
        let a: A2aAgent =
            serde_json::from_str(r#"{"display_name":"a","url":"https://x/a2a"}"#).unwrap();
        let s = serde_json::to_string(&a).unwrap();
        // Default V1_0 serialises as the wire string "1.0", not "v1_0".
        assert!(s.contains(r#""protocol_version":"1.0""#), "got: {s}");
    }

    #[test]
    fn api_key_round_trips_and_omits_unset_optionals() {
        let original: A2aAgent = serde_json::from_str(
            r#"{"display_name":"a","url":"https://x/a2a","auth_type":"api_key","secret":"k-1"}"#,
        )
        .unwrap();
        let s = serde_json::to_string(&original).unwrap();
        // The tag serialises as `api_key` (not a PascalCased `ApiKey`).
        assert!(s.contains(r#""auth_type":"api_key""#), "got: {s}");
        assert!(
            !s.contains("timeout_ms"),
            "unset timeout_ms must be omitted: {s}"
        );
        let back: A2aAgent = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn rejects_unknown_fields() {
        let r: Result<A2aAgent, _> =
            serde_json::from_str(r#"{"display_name":"x","url":"u","extra":1}"#);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_unknown_protocol_version_and_auth_type() {
        assert!(serde_json::from_str::<A2aAgent>(
            r#"{"display_name":"x","url":"u","protocol_version":"2.0"}"#
        )
        .is_err());
        assert!(serde_json::from_str::<A2aAgent>(
            r#"{"display_name":"x","url":"u","auth_type":"oauth"}"#
        )
        .is_err());
    }

    #[test]
    fn resource_trait_routes_through_display_name() {
        let mut a: A2aAgent =
            serde_json::from_str(r#"{"display_name":"invoice","url":"https://x/a2a"}"#).unwrap();
        a.runtime_id = "uuid-a2a-1".into();
        assert_eq!(<A2aAgent as Resource>::kind(), "a2a_agents");
        assert_eq!(a.id(), "uuid-a2a-1");
        assert_eq!(a.name(), "invoice");
    }

    #[test]
    fn round_trip_omits_default_optionals() {
        let original = A2aAgent {
            display_name: "invoice".into(),
            url: "https://x/a2a".into(),
            protocol_version: A2aProtocolVersion::V1_0,
            auth_type: A2aAuthType::None,
            secret: None,
            timeout_ms: None,
            enabled: true,
            runtime_id: String::new(),
        };
        let s = serde_json::to_string(&original).unwrap();
        let back: A2aAgent = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }
}
