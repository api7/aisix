//! `PassthroughRoute` entity — an explicit passthrough binding from a
//! gateway entry (path prefix and/or inbound `Host`) to one upstream target.
//!
//! Replaces the removed implicit `/passthrough/:provider/*rest` tunnel: the
//! route names its own upstream and credential handling instead of borrowing
//! them from "the first accessible Model of the provider", so there is no
//! implicit-selection ambiguity (AISIX-Cloud#1127) and no forced credential
//! replacement (AISIX-Cloud#1312).
//!
//! etcd path: `{prefix}/passthrough_routes/{uuid}`. Secondary index on `name`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct PassthroughRoute {
    /// Operator-facing label, unique within the gateway. Referenced by API
    /// keys' `allowed_routes` globs and used for usage attribution.
    #[serde(alias = "display_name")]
    #[schemars(length(min = 1))]
    pub name: String,

    /// Gateway path prefix this route serves, e.g. `/passthrough/openai`.
    /// The prefix is stripped before the remainder is joined to the target
    /// URL. Must start with `/` and must not claim a reserved gateway
    /// namespace (`/v1`, `/mcp`, `/a2a`, health probes). At least one of
    /// `path_prefix` / `hosts` is required; when both are set the request
    /// must satisfy both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = "^/"), length(min = 1))]
    pub path_prefix: Option<String>,

    /// Inbound `Host` values this route serves (the forward-proxy entry:
    /// a TLS-terminating device delivers plaintext traffic with the
    /// original host, e.g. `api.githubcopilot.com`). Matched
    /// case-insensitively, ignoring any `:port` suffix; a leading `*.`
    /// wildcard matches exactly one extra label
    /// (`*.githubcopilot.com` matches `proxy.githubcopilot.com`).
    /// Host-matched requests keep their full path (no prefix stripping
    /// unless `path_prefix` also matched).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hosts: Option<Vec<String>>,

    /// Explicit upstream base URL, e.g. `https://api.openai.com`. The
    /// matched request's remainder path and query are appended. Exactly one
    /// of `target_url` / `preserve_host` must be configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub target_url: Option<String>,

    /// Derive the target from the request's own `Host` header
    /// (`https://<host>`), for forward-proxy routes that fan one route out
    /// over several official hosts. Only legal when `hosts` is set — the
    /// matched allowlist is what makes the derived target non-attacker-
    /// controlled (SSRF guard).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub preserve_host: bool,

    /// How the gateway authenticates the caller on this route.
    #[serde(default)]
    pub auth_mode: PassthroughAuthMode,

    /// Header carrying the gateway credential (API key or JWT) when
    /// `auth_mode` is `header_key`, e.g. `x-aisix-api-key`. Lets
    /// `Authorization` carry the caller's own upstream credential. The
    /// header is stripped before forwarding. Required for `header_key`;
    /// ignored otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = "^[!#$%&'*+.^_`|~0-9A-Za-z-]+$"), length(min = 1))]
    pub auth_header_name: Option<String>,

    /// The API key this route's traffic runs as when `auth_mode` is
    /// `anonymous`: its `allowed_routes`, rate limits, budget and usage
    /// attribution all apply, so anonymous traffic keeps a stable,
    /// governable principal. Required for `anonymous`; ignored otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub anonymous_key_id: Option<String>,

    /// Client source CIDRs allowed to use this route. Required (non-empty)
    /// when `auth_mode` is `anonymous` — network reachability is the only
    /// gate left in front of the anonymous principal. Optional hardening
    /// for the other modes; unset means no route-level restriction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_cidrs: Option<Vec<String>>,

    /// How the upstream credential is produced.
    #[serde(default)]
    pub credential_mode: PassthroughCredentialMode,

    /// ProviderKey whose secret is injected upstream when `credential_mode`
    /// is `inject` (per-provider auth shape: `x-api-key` +
    /// `anthropic-version` for Anthropic, `Authorization: Bearer` otherwise;
    /// its `strip_headers` and TLS settings apply). Required for `inject`;
    /// forbidden for `forward_client`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub provider_key_id: Option<String>,

    /// Body-shape hint for auditing, guardrails and usage extraction.
    /// Parsing is best-effort: a body that does not match the declared
    /// shape degrades to `raw` handling, it is never rejected for shape.
    #[serde(default)]
    pub protocol: PassthroughProtocol,

    /// Relay `text/event-stream` upstream responses incrementally. When
    /// `false` streaming responses are fully buffered like any other body.
    #[serde(default = "default_true")]
    pub streaming: bool,

    /// Optional header carrying the end-user identity injected by the
    /// upstream network device (e.g. `x-aisix-user`). Its value is recorded
    /// on the usage event for per-employee audit attribution and stripped
    /// before forwarding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = "^[!#$%&'*+.^_`|~0-9A-Za-z-]+$"), length(min = 1))]
    pub identity_header: Option<String>,

    /// Maximum time, in milliseconds, for a non-streaming upstream exchange.
    /// Streaming responses are not bounded by this (the relay ends when the
    /// upstream stream ends or the client disconnects). When omitted, the
    /// gateway default request timeout applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub timeout_ms: Option<u64>,

    /// Whether this route is active. A disabled route matches nothing.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Filled in by the snapshot loader from the etcd key path.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

fn default_true() -> bool {
    true
}

/// How the gateway authenticates callers of a passthrough route.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PassthroughAuthMode {
    /// Standard gateway auth: an API key or JWT in `Authorization: Bearer`
    /// or `x-api-key`, exactly like the typed endpoints.
    #[default]
    GatewayKey,
    /// Gateway credential in the route's `auth_header_name` header;
    /// `Authorization` is left untouched for the upstream credential.
    HeaderKey,
    /// No inbound gateway credential. The request runs as the route's
    /// `anonymous_key_id` principal, restricted to `source_cidrs`.
    Anonymous,
}

/// How the upstream credential of a passthrough route is produced.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PassthroughCredentialMode {
    /// Strip inbound credential headers and inject the configured
    /// ProviderKey's secret (the legacy tunnel's behavior, now explicit).
    #[default]
    Inject,
    /// Forward the caller's own `Authorization` (and other credential
    /// headers) verbatim — bring-your-own-credential. Gateway side-channel
    /// headers are still stripped so the gateway credential never leaks
    /// upstream.
    ForwardClient,
}

/// Body-shape hint for a passthrough route.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PassthroughProtocol {
    /// No parsing: bodies are opaque byte blobs (guardrails scan them as
    /// one lossy-UTF-8 text).
    #[default]
    Raw,
    /// OpenAI-compatible chat envelope (`messages`, streamed
    /// `choices[].delta.content`, final-chunk / response `usage`).
    OpenaiChat,
    /// OpenAI-compatible legacy completions / FIM envelope (`prompt` [+
    /// `suffix`], streamed `choices[].text`, `usage`).
    OpenaiCompletions,
}

impl Resource for PassthroughRoute {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "passthrough_routes"
    }
}

impl PassthroughRoute {
    /// `true` when `host` (already lowercased, port stripped) matches one of
    /// the route's `hosts` patterns. A `*.` prefix matches exactly one extra
    /// leading label; everything else is an exact match.
    pub fn matches_host(&self, host: &str) -> bool {
        let Some(hosts) = &self.hosts else {
            return false;
        };
        hosts.iter().any(|pattern| {
            let p = pattern.to_ascii_lowercase();
            if let Some(suffix) = p.strip_prefix("*.") {
                match host.strip_suffix(suffix) {
                    // `label.` + suffix, with exactly one label consumed.
                    Some(head) => {
                        head.ends_with('.')
                            && !head[..head.len() - 1].is_empty()
                            && !head[..head.len() - 1].contains('.')
                    }
                    None => false,
                }
            } else {
                p == host
            }
        })
    }
}

/// Cross-field coupling for the flat `PassthroughRoute` schema, injected as
/// an `allOf` by [`crate::models::schema::passthrough_route_root_schema`]
/// (same pattern as [`super::mcp_server::mcp_server_credential_coupling`]).
/// Every rule here is enforced on both the strict declarative path and the
/// lenient etcd path so a route is never half-configured at dispatch time.
pub fn passthrough_route_coupling() -> Value {
    json!([
        // At least one match dimension.
        { "anyOf": [
            { "title": "Path-prefix match", "required": ["path_prefix"] },
            { "title": "Host match", "required": ["hosts"] }
        ] },
        // hosts, when present, is a non-empty list of non-empty strings.
        {
            "if": { "required": ["hosts"] },
            "then": { "properties": { "hosts": {
                "type": "array", "minItems": 1,
                "items": { "type": "string", "minLength": 1 }
            } } }
        },
        // path_prefix must not claim a reserved gateway namespace. The
        // proxy's typed routes always win over the fallback matcher anyway;
        // this rule keeps a shadowed-by-construction route from being
        // configured at all.
        {
            "if": { "required": ["path_prefix"] },
            "then": { "properties": { "path_prefix": {
                "not": { "pattern": "^/(v1|mcp|a2a|admin|livez|readyz|metrics)(/|$)" }
            } } }
        },
        // Exactly one target shape.
        {
            "oneOf": [
                {
                    "title": "Explicit target URL",
                    "required": ["target_url"],
                    "not": {
                        "properties": { "preserve_host": {
                            "const": true,
                            "description": "Set when the route derives its target from the inbound Host instead of target_url."
                        } },
                        "required": ["preserve_host"]
                    }
                },
                {
                    "title": "Preserve inbound host",
                    "properties": { "preserve_host": {
                        "const": true,
                        "description": "Set when the route derives its target from the inbound Host instead of target_url."
                    } },
                    "required": ["preserve_host"],
                    "not": { "required": ["target_url"] }
                }
            ]
        },
        // preserve_host derives the target from the inbound Host, so the
        // hosts allowlist is what bounds it (SSRF guard).
        {
            "if": { "properties": { "preserve_host": { "const": true } }, "required": ["preserve_host"] },
            "then": { "required": ["hosts"] }
        },
        // target_url must be http(s).
        {
            "if": { "required": ["target_url"] },
            "then": { "properties": { "target_url": { "pattern": "^https?://" } } }
        },
        // auth_mode couplings.
        {
            "if": { "properties": { "auth_mode": { "const": "header_key" } }, "required": ["auth_mode"] },
            "then": { "required": ["auth_header_name"] }
        },
        {
            "if": { "properties": { "auth_mode": { "const": "anonymous" } }, "required": ["auth_mode"] },
            "then": {
                "required": ["anonymous_key_id", "source_cidrs"],
                "properties": { "source_cidrs": {
                    "type": "array", "minItems": 1,
                    "items": { "type": "string", "minLength": 1 }
                } }
            }
        },
        // credential_mode couplings: inject needs a ProviderKey; a
        // forward_client route carrying one is a configuration error, not
        // an ignored field.
        {
            "if": {
                "anyOf": [
                    { "title": "credential_mode omitted (defaults to inject)", "not": { "required": ["credential_mode"] } },
                    { "title": "credential_mode: inject", "properties": { "credential_mode": { "const": "inject" } }, "required": ["credential_mode"] }
                ]
            },
            "then": { "required": ["provider_key_id"] }
        },
        {
            "if": { "properties": { "credential_mode": { "const": "forward_client" } }, "required": ["credential_mode"] },
            "then": { "not": { "required": ["provider_key_id"] } }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal() -> PassthroughRoute {
        serde_json::from_str(
            r#"{
              "name": "openai-tunnel",
              "path_prefix": "/passthrough/openai",
              "target_url": "https://api.openai.com",
              "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn minimal_route_defaults() {
        let r = minimal();
        assert_eq!(r.auth_mode, PassthroughAuthMode::GatewayKey);
        assert_eq!(r.credential_mode, PassthroughCredentialMode::Inject);
        assert_eq!(r.protocol, PassthroughProtocol::Raw);
        assert!(r.streaming);
        assert!(r.enabled);
        assert!(!r.preserve_host);
    }

    #[test]
    fn display_name_alias_accepted() {
        let r: PassthroughRoute = serde_json::from_str(
            r#"{"display_name":"x","path_prefix":"/x","target_url":"https://u","provider_key_id":"pk"}"#,
        )
        .unwrap();
        assert_eq!(r.name, "x");
    }

    #[test]
    fn host_matching_exact_and_wildcard() {
        let r: PassthroughRoute = serde_json::from_str(
            r#"{"name":"copilot","hosts":["api.githubcopilot.com","*.individual.githubcopilot.com"],
                "preserve_host":true,"credential_mode":"forward_client"}"#,
        )
        .unwrap();
        assert!(r.matches_host("api.githubcopilot.com"));
        // Wildcard: exactly one extra label.
        assert!(r.matches_host("proxy.individual.githubcopilot.com"));
        assert!(!r.matches_host("a.b.individual.githubcopilot.com"));
        // The bare suffix itself is not matched by `*.`.
        assert!(!r.matches_host("individual.githubcopilot.com"));
        assert!(!r.matches_host("evil.com"));
    }

    #[test]
    fn host_matching_is_case_insensitive_on_pattern() {
        let r: PassthroughRoute = serde_json::from_str(
            r#"{"name":"x","hosts":["API.Example.COM"],"preserve_host":true,
                "credential_mode":"forward_client"}"#,
        )
        .unwrap();
        // Callers pass the already-lowercased inbound host.
        assert!(r.matches_host("api.example.com"));
    }
}
