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
    /// Must start with `/`.
    ///
    /// How the prefix is treated depends on the target shape: a
    /// `target_url` route MOUNTS at the prefix, so it is stripped before
    /// the remainder is joined to the target base; a `preserve_host` route
    /// MIRRORS an upstream that owns its own path space, so the prefix only
    /// selects which requests the route claims and the complete path is
    /// forwarded unchanged.
    ///
    /// A route WITHOUT `hosts` must not claim a
    /// reserved gateway namespace (`/v1`, `/mcp`, `/a2a`, health probes) —
    /// the typed routes would shadow it. A route WITH `hosts` may use any
    /// prefix: host-matched requests dispatch ahead of the typed routes,
    /// which is what lets a forward proxy relay an upstream's own
    /// namespace (e.g. Copilot's `/mcp/...` on its chat host).
    /// At least one of `path_prefix` / `hosts` is required; when both are
    /// set the request must satisfy both.
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
    /// header is stripped before forwarding. Lowercase-only so the
    /// forbidden credential-slot list in the coupling is exhaustive
    /// (matching is case-insensitive on the wire regardless). Required
    /// for `header_key`; ignored otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = "^[!#$%&'*+.^_`|~0-9a-z-]+$"), length(min = 1))]
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

    /// Optional header carrying the end-user identity injected by the
    /// upstream network device (e.g. `x-aisix-user`). Its value is recorded
    /// on the usage event for per-employee audit attribution and stripped
    /// before forwarding. Lowercase-only so the forbidden credential-slot
    /// list in the coupling is exhaustive; credential-bearing names
    /// (`authorization`, `cookie`, …) are rejected outright — their value
    /// on the usage event would be credential retention, not identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = "^[!#$%&'*+.^_`|~0-9a-z-]+$"), length(min = 1))]
    pub identity_header: Option<String>,

    /// Inbound client headers forwarded to the upstream even when this
    /// route would otherwise strip them, as single-`*` glob patterns
    /// matched case-insensitively against the header name
    /// (`"authorization"`, `"x-trace-*"`). Empty — the default — overrides
    /// no stripping.
    ///
    /// A route forwards the caller's headers by default, so this field
    /// only matters for the ones it removes: the ProviderKey's
    /// `strip_headers` under `credential_mode: inject`, and the slot the
    /// gateway consumed to authenticate the caller. Naming `authorization`
    /// under `auth_mode: gateway_key` therefore puts the caller's own
    /// credential back on the upstream request in place of the one this
    /// route would inject, never both — which is what lets an internal
    /// service that already authorizes on the end user's `Authorization`
    /// keep doing so unchanged.
    ///
    /// A credential slot — `authorization`, `proxy-authorization`,
    /// `x-api-key`, `api-key`, `x-goog-api-key`, `cookie`, and the AWS
    /// SigV4 trio `x-amz-security-token` / `x-amz-date` /
    /// `x-amz-content-sha256` — and
    /// `traceparent` / `tracestate` are forwarded only when a pattern
    /// names them exactly. A glob such as `"*"` or `"x-*"` is a statement
    /// about the operator's own headers, not consent to hand a third party
    /// the caller's credential or to graft the caller's trace onto that
    /// party's telemetry, so a broad pattern overrides the rest of the
    /// strip set and leaves those alone.
    ///
    /// This route's own `auth_header_name` and `identity_header` are read
    /// the same way. Both are slots this route chose rather than ones the
    /// gateway owns — under `auth_mode: header_key` the first carries the
    /// gateway credential the caller authenticated with, and the second
    /// carries an end-user identity this route records and strips — so a
    /// glob does not sweep either, and a pattern that names one in full
    /// forwards it.
    ///
    /// Headers whose forwarding would break the exchange rather than
    /// change who it comes from are stripped whatever the patterns say:
    /// `host`, `content-length`, the hop-by-hop headers that describe the
    /// caller's own connection, and the gateway's `x-aisix-*` namespace.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forward_client_headers: Vec<String>,

    /// Maximum time, in milliseconds, for the upstream exchange. Bounds
    /// the response-header phase and any non-SSE body read, but never a
    /// healthy SSE relay (which ends with the upstream stream or the
    /// client hanging up). When omitted, the gateway default request
    /// timeout applies the same way.
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
/// Header names a route's `identity_header` / `auth_header_name` may never
/// take: the value of any of these IS a credential, and the identity slot
/// records its value onto the usage event (credential retention) while the
/// auth slot would repurpose a credential channel the modes already own.
/// The fields' schemars pattern forces lowercase, so this lowercase list is
/// exhaustive on every configuration path.
const FORBIDDEN_HEADER_SLOTS: [&str; 5] = [
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
];

pub fn passthrough_route_coupling() -> Value {
    json!([
        // At least one match dimension.
        { "anyOf": [
            { "title": "Path-prefix match", "required": ["path_prefix"] },
            { "title": "Host match", "required": ["hosts"] }
        ] },
        // path_prefix, when present, is a real non-null string (the
        // match-dimension `required` alone would accept an explicit null,
        // which deserializes as absent and voids the rule).
        {
            "if": { "required": ["path_prefix"] },
            "then": { "properties": { "path_prefix": {
                "type": "string", "minLength": 1
            } } }
        },
        // hosts, when present, is a non-empty list of non-empty,
        // non-wildcard-only host patterns: exact hosts, or a `*.` prefix
        // that keeps at least two literal labels (`*.example.com` — never
        // `*.com` or a bare `*`, which would make a `preserve_host` target
        // effectively unbounded).
        {
            "if": { "required": ["hosts"] },
            "then": { "properties": { "hosts": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "string", "minLength": 1,
                    "pattern": "^(\\*\\.)?([A-Za-z0-9-]+\\.)+[A-Za-z0-9-]+$|^[A-Za-z0-9-]+$"
                }
            } } }
        },
        // A path-only route must not claim a reserved gateway namespace:
        // the proxy's typed routes win over the fallback matcher, so such a
        // route is shadowed by construction and better rejected outright.
        //
        // A route that ALSO matches on `hosts` is exempt, and deliberately
        // so: host-matched requests are dispatched by the host middleware
        // wrapping the whole router, ahead of the typed routes, so the
        // prefix is reachable. Forward-proxy deployments need this — an
        // upstream owns its own path space, and GitHub Copilot's CLI
        // reaches its MCP server at `/mcp/readonly` on the same host it
        // uses for chat inference. Requiring `hosts` keeps the exemption
        // narrow: it cannot be used to shadow the gateway's own endpoints
        // for ordinary (host-less) traffic.
        {
            "if": {
                "required": ["path_prefix"],
                "not": { "required": ["hosts"] }
            },
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
        // target_url must be a real http(s) string — never an explicit
        // null, which `required` alone would accept (it deserializes as
        // absent and the dispatch would build an empty base).
        {
            "if": { "required": ["target_url"] },
            "then": { "properties": { "target_url": {
                "type": "string", "pattern": "^https?://"
            } } }
        },
        // The header-slot fields must be real lowercase header names and
        // never a credential channel: recording `Authorization` as the
        // identity would persist the caller's credential onto the usage
        // event, and consuming it as the gateway slot belongs to the
        // modes themselves.
        {
            "if": { "required": ["auth_header_name"] },
            "then": { "properties": { "auth_header_name": {
                "type": "string", "not": { "enum": FORBIDDEN_HEADER_SLOTS }
            } } }
        },
        {
            "if": { "required": ["identity_header"] },
            // Presence, not value, for the same reason as the caller-JWT
            // slot below: the property declares itself nullable and no
            // mode requires it, so an explicit null means "not
            // configured" rather than a document worth dropping the
            // route over. The coupled fields above are the opposite
            // case and keep their type pin.
            "then": { "properties": { "identity_header": {
                "not": { "enum": FORBIDDEN_HEADER_SLOTS }
            } } }
        },
        // auth_mode couplings. The mode-required companions are pinned to
        // non-null strings for the same explicit-null reason as target_url.
        {
            "if": { "properties": { "auth_mode": { "const": "header_key" } }, "required": ["auth_mode"] },
            "then": {
                "required": ["auth_header_name"],
                "properties": { "auth_header_name": { "type": "string", "minLength": 1 } }
            }
        },
        {
            "if": { "properties": { "auth_mode": { "const": "anonymous" } }, "required": ["auth_mode"] },
            "then": {
                "required": ["anonymous_key_id", "source_cidrs"],
                "properties": {
                    "anonymous_key_id": { "type": "string", "minLength": 1 },
                    "source_cidrs": {
                        "type": "array", "minItems": 1,
                        "items": { "type": "string", "minLength": 1 }
                    }
                }
            }
        },
        // Cross-mode leftovers are configuration errors, not ignored
        // fields: a companion outside its mode is never consulted, so a
        // row carrying one is rejected rather than half-honored. The CP
        // enforces the identical rule on create and patch.
        {
            "title": "auth_header_name only in header_key mode",
            "if": {
                "anyOf": [
                    { "title": "auth_mode omitted (defaults to gateway_key)", "not": { "required": ["auth_mode"] } },
                    { "title": "auth_mode: gateway_key or anonymous", "properties": { "auth_mode": { "enum": ["gateway_key", "anonymous"] } }, "required": ["auth_mode"] }
                ]
            },
            "then": { "not": { "required": ["auth_header_name"] } }
        },
        {
            "title": "anonymous_key_id only in anonymous mode",
            "if": {
                "anyOf": [
                    { "title": "auth_mode omitted (defaults to gateway_key)", "not": { "required": ["auth_mode"] } },
                    { "title": "auth_mode: gateway_key or header_key", "properties": { "auth_mode": { "enum": ["gateway_key", "header_key"] } }, "required": ["auth_mode"] }
                ]
            },
            "then": { "not": { "required": ["anonymous_key_id"] } }
        },
        // credential_mode couplings: inject needs a real ProviderKey id; a
        // forward_client route carrying one is a configuration error, not
        // an ignored field.
        {
            "if": {
                "anyOf": [
                    { "title": "credential_mode omitted (defaults to inject)", "not": { "required": ["credential_mode"] } },
                    { "title": "credential_mode: inject", "properties": { "credential_mode": { "const": "inject" } }, "required": ["credential_mode"] }
                ]
            },
            "then": {
                "required": ["provider_key_id"],
                "properties": { "provider_key_id": { "type": "string", "minLength": 1 } }
            }
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
    fn the_client_header_forward_is_a_free_list_of_patterns() {
        use crate::models::schema::{
            validate_passthrough_route, validate_passthrough_route_lenient,
        };

        let with = |patterns: serde_json::Value| {
            let mut v = json!({
                "name": "system-server",
                "path_prefix": "/passthrough/system",
                "target_url": "https://erp.internal",
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            });
            v["forward_client_headers"] = patterns;
            v
        };

        // Naming the credential slot the gateway consumed is the point of
        // the field, not an oversight, so the schema does not second-guess
        // it — the runtime decides what is deliverable.
        validate_passthrough_route(&with(json!(["authorization", "x-trace-*", "*"])))
            .expect("credential slots and globs are both accepted");
        validate_passthrough_route(&with(json!([])))
            .expect("an empty list is the default, spelled out");
        validate_passthrough_route_lenient(&with(json!(["authorization"])))
            .expect("the read path must not cost the route over a pattern");
    }

    #[test]
    fn minimal_route_defaults() {
        let r = minimal();
        assert_eq!(r.auth_mode, PassthroughAuthMode::GatewayKey);
        assert_eq!(r.credential_mode, PassthroughCredentialMode::Inject);
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

#[cfg(test)]
mod coupling_tests {
    use crate::models::schema::{validate_passthrough_route, validate_passthrough_route_lenient};
    use serde_json::json;

    fn base() -> serde_json::Value {
        json!({
            "name": "r",
            "path_prefix": "/p",
            "target_url": "https://u.example",
            "provider_key_id": "pk-1"
        })
    }

    #[test]
    fn explicit_null_on_coupled_fields_is_rejected() {
        // `required` alone would accept null (deserializes as absent) and
        // void the coupling — both validators must refuse it.
        for (field, value) in [
            ("target_url", json!(null)),
            ("provider_key_id", json!(null)),
            ("path_prefix", json!(null)),
        ] {
            let mut doc = base();
            doc[field] = value;
            assert!(
                validate_passthrough_route(&doc).is_err(),
                "strict must reject null {field}"
            );
            assert!(
                validate_passthrough_route_lenient(&doc).is_err(),
                "lenient must reject null {field}"
            );
        }
        // Mode-required companions: header_key with a null header name.
        let doc = json!({
            "name": "r", "path_prefix": "/p",
            "target_url": "https://u.example", "provider_key_id": "pk",
            "auth_mode": "header_key", "auth_header_name": null
        });
        assert!(validate_passthrough_route(&doc).is_err());
        // Anonymous with a null principal.
        let doc = json!({
            "name": "r", "path_prefix": "/p",
            "target_url": "https://u.example", "provider_key_id": "pk",
            "auth_mode": "anonymous", "anonymous_key_id": null,
            "source_cidrs": ["10.0.0.0/8"]
        });
        assert!(validate_passthrough_route(&doc).is_err());
    }

    #[test]
    fn explicit_null_on_an_optional_header_slot_clears_it() {
        // The opposite of the coupled fields above: no mode requires
        // these, they declare themselves nullable, and `null` is how a
        // resources.yaml author writes "not set" — a bare key parses to
        // it. Rejecting that costs the whole route on the lenient read
        // path, over a value the author meant to leave empty.
        let field = "identity_header";
        let mut doc = base();
        doc[field] = json!(null);
        validate_passthrough_route(&doc)
            .unwrap_or_else(|e| panic!("strict must accept null {field}: {e}"));
        validate_passthrough_route_lenient(&doc)
            .unwrap_or_else(|e| panic!("lenient must accept null {field}: {e}"));

        // Relaxing the type did not relax the rule the branch exists for.
        let mut doc = base();
        doc[field] = json!("authorization");
        assert!(
            validate_passthrough_route(&doc).is_err(),
            "a credential slot is still refused as an identity header"
        );
    }

    #[test]
    fn reserved_prefix_is_rejected_only_without_hosts() {
        // A host-less route on a gateway namespace is shadowed by the typed
        // routes, so it stays rejected…
        let mut doc = base();
        for p in [
            "/mcp",
            "/mcp/readonly",
            "/v1/chat/completions",
            "/a2a",
            "/metrics",
        ] {
            doc["path_prefix"] = json!(p);
            assert!(
                validate_passthrough_route(&doc).is_err(),
                "host-less route on {p} must stay rejected"
            );
        }
        // …but the same prefix WITH a host allowlist is exactly the
        // forward-proxy shape (Copilot CLI's MCP server sits at /mcp on the
        // same host as chat) and host dispatch runs before the typed routes.
        let doc = json!({
            "name": "copilot-cli-mcp",
            "hosts": ["api.business.githubcopilot.com"],
            "path_prefix": "/mcp",
            "preserve_host": true,
            "auth_mode": "header_key",
            "auth_header_name": "x-aisix-api-key",
            "credential_mode": "forward_client"
        });
        assert!(
            validate_passthrough_route(&doc).is_ok(),
            "host-matched route on /mcp must be accepted: {:?}",
            validate_passthrough_route(&doc)
        );
        assert!(validate_passthrough_route_lenient(&doc).is_ok());
    }

    #[test]
    fn removed_protocol_and_streaming_fields_are_unknown() {
        // The pre-0.10.0 dev cycle carried `protocol` / `streaming` route
        // fields; both were removed before the kind ever shipped (the
        // envelope is now detected per request, SSE always relays
        // incrementally). The strict write path rejects them like any
        // unknown field; the lenient etcd path tolerates-and-strips.
        for (field, value) in [
            ("protocol", json!("openai_responses")),
            ("streaming", json!(false)),
        ] {
            let mut doc = base();
            doc[field] = value;
            assert!(
                validate_passthrough_route(&doc).is_err(),
                "strict must reject unknown field {field}"
            );
            assert!(
                validate_passthrough_route_lenient(&doc).is_ok(),
                "lenient must tolerate unknown field {field}"
            );
        }
    }

    #[test]
    fn cross_mode_leftover_companions_are_rejected() {
        // A companion outside its mode is never consulted at runtime, so
        // both validators refuse the row instead of half-honoring it.
        // auth_header_name without header_key (auth_mode absent = gateway_key).
        let mut doc = base();
        doc["auth_header_name"] = json!("x-aisix-api-key");
        assert!(validate_passthrough_route(&doc).is_err());
        assert!(validate_passthrough_route_lenient(&doc).is_err());
        // anonymous_key_id on an explicit header_key route.
        let doc = json!({
            "name": "r", "path_prefix": "/p",
            "target_url": "https://u.example", "provider_key_id": "pk",
            "auth_mode": "header_key", "auth_header_name": "x-aisix-api-key",
            "anonymous_key_id": "ak-1"
        });
        assert!(validate_passthrough_route(&doc).is_err());
        // auth_header_name on an anonymous route.
        let doc = json!({
            "name": "r", "path_prefix": "/p",
            "target_url": "https://u.example", "provider_key_id": "pk",
            "auth_mode": "anonymous", "anonymous_key_id": "ak-1",
            "source_cidrs": ["10.0.0.0/8"],
            "auth_header_name": "x-aisix-api-key"
        });
        assert!(validate_passthrough_route(&doc).is_err());
        // source_cidrs is deliberately NOT mode-coupled: an extra IP
        // allowlist is honored on every mode.
        let mut doc = base();
        doc["source_cidrs"] = json!(["10.0.0.0/8"]);
        assert!(validate_passthrough_route(&doc).is_ok());
    }

    #[test]
    fn credential_bearing_header_slots_are_rejected() {
        for field in ["auth_header_name", "identity_header"] {
            for name in [
                "authorization",
                "proxy-authorization",
                "cookie",
                "set-cookie",
                "x-api-key",
                // The field pattern is lowercase-only, so a case variant
                // cannot sneak past the forbidden list either.
                "Authorization",
            ] {
                let mut doc = base();
                if field == "auth_header_name" {
                    doc["auth_mode"] = json!("header_key");
                }
                doc[field] = json!(name);
                assert!(
                    validate_passthrough_route(&doc).is_err(),
                    "{field}={name} must be rejected"
                );
            }
            // A benign custom header passes.
            let mut doc = base();
            if field == "auth_header_name" {
                doc["auth_mode"] = json!("header_key");
            }
            doc[field] = json!("x-aisix-user");
            assert!(
                validate_passthrough_route(&doc).is_ok(),
                "{field}=x-aisix-user must pass"
            );
        }
    }

    #[test]
    fn preserve_host_wildcards_need_two_literal_labels() {
        let mk = |host: &str| {
            json!({
                "name": "r",
                "hosts": [host],
                "preserve_host": true,
                "credential_mode": "forward_client"
            })
        };
        assert!(validate_passthrough_route(&mk("api.example.com")).is_ok());
        assert!(validate_passthrough_route(&mk("*.githubcopilot.com")).is_ok());
        // A single-label wildcard tail widens the derived target to any
        // registrable domain — rejected on every configuration path.
        assert!(validate_passthrough_route(&mk("*.com")).is_err());
        assert!(validate_passthrough_route(&mk("*")).is_err());
        assert!(validate_passthrough_route_lenient(&mk("*.com")).is_err());
    }
}
