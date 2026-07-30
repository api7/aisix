//! `McpServer` entity — a registered MCP tool source.
//!
//! Registers either an upstream Model Context Protocol (MCP) server the
//! gateway fronts (`type: mcp`), or a REST API described by an OpenAPI
//! document whose operations the gateway itself exposes as tools
//! (`type: openapi`). Either way the tools are aggregated into the gateway's
//! own MCP endpoint under the namespace `<name>__<tool>`, and tool calls are
//! routed back to the source. The upstream credential is held by the gateway
//! and is never exposed to the calling client.
//!
//! etcd path: `{prefix}/mcp_servers/{uuid}`. Secondary index on `name`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::resource::Resource;

// `Eq` is deliberately absent: `spec` holds a `serde_json::Value`, which is
// only `PartialEq` (JSON numbers are floats).
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
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
    // The name is the tool-namespace prefix: this server's tools are exposed to
    // MCP clients as `<name>__<tool>` and parsed back with `split_once("__")`
    // (see `aisix_mcp::gateway`). So the name must contain no `__` AND must not
    // end in `_`: `gh_` + `x` and `gh` + `_x` both serialize to `gh___x`, and the
    // split resolves the former to the non-existent server `gh`. The pattern
    // below rejects both shapes on every configuration path.
    #[schemars(regex(pattern = "^(?:[^_]|_[^_])*$"), length(min = 1))]
    pub name: String,

    /// What backs this server: a real upstream MCP server (`mcp`, the
    /// default), or a plain REST API described by an OpenAPI document
    /// (`openapi`) whose operations the gateway itself exposes as MCP tools.
    #[serde(rename = "type", default)]
    pub server_type: McpServerType,

    /// For `type: mcp`, the upstream server's MCP endpoint URL, reached over
    /// the Streamable HTTP transport, such as `https://api.example.com/mcp`.
    /// For `type: openapi`, the REST API's base URL that generated tool calls
    /// are issued against, such as `https://erp.internal/api/v1`.
    #[schemars(length(min = 1))]
    pub url: String,

    /// The OpenAPI 3.x document (as a JSON object) whose operations become
    /// this server's tools. Required when `type` is `openapi`; ignored
    /// otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<serde_json::Value>,

    /// Header name the API key is sent under when `type` is `openapi` and
    /// `auth_type` is `api_key`. Defaults to `x-api-key` when unset. Ignored
    /// for `type: mcp`, whose API-key header is fixed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub api_key_header: Option<String>,

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
    /// `api_key` (sent as `x-api-key: <secret>`, or under `api_key_header`
    /// for `type: openapi`), or the OAuth client secret when `auth_type` is
    /// `oauth2`. Leave unset when `auth_type` is `none`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,

    // Cross-field coupling (`auth_type` → credential set, and the openapi-only
    // `spec`/`api_key_header` fields) is expressed as an injected `allOf` of
    // `if`/`then` subschemas rather than in this flat struct — see
    // `mcp_server_credential_coupling`. That keeps the resource flat (no oneOf
    // restructuring) while giving the published schema and every runtime
    // validator one shared definition.
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

/// What backs a registered MCP server entry.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum McpServerType {
    /// A real upstream MCP server the gateway connects to.
    #[default]
    Mcp,
    /// A REST API described by an OpenAPI document; the gateway generates the
    /// tools itself and issues plain HTTP requests against `url`.
    Openapi,
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

/// The `auth_type` → credential coupling, as a JSON Schema `allOf` that
/// [`crate::models::schema::mcp_server_root_schema`] injects into the generated
/// schema. `schemars` cannot express a cross-field conditional, so this is the
/// single definition the published schema and every runtime validator share:
/// an incomplete credential set leaves the gateway authenticating upstream with
/// nothing, so it is rejected at load rather than at first tool call.
pub fn mcp_server_credential_coupling() -> Value {
    let secret_required = json!({
        "required": ["secret"],
        "properties": { "secret": { "type": "string", "minLength": 1 } }
    });
    json!([
        {
            "if": { "properties": { "auth_type": { "const": "bearer" } }, "required": ["auth_type"] },
            "then": secret_required
        },
        {
            "if": { "properties": { "auth_type": { "const": "api_key" } }, "required": ["auth_type"] },
            "then": secret_required
        },
        {
            "if": { "properties": { "auth_type": { "const": "oauth2" } }, "required": ["auth_type"] },
            "then": {
                "required": ["secret", "client_id", "token_url"],
                "properties": {
                    "secret": { "type": "string", "minLength": 1 },
                    "client_id": { "type": "string", "minLength": 1 },
                    "token_url": { "type": "string", "minLength": 1 }
                }
            }
        },
        // Note on diagnosability: expressing "not a Swagger 2.0 document" as a
        // schema constraint costs the targeted hint the write path used to emit
        // ("convert the spec to OpenAPI 3.x"). A validator reports the failing
        // pointer (`/spec`) and the constraint, not advice. The rule is what
        // matters most here — a Swagger 2.0 document is rejected on every path
        // — but a post-schema semantic hook in the loaders would let both the
        // rule and the advice live in one place.
        //
        // An OpenAPI-backed server carries the document; `api_key_header` names
        // the header the generated tools send the key in, so it only makes sense
        // alongside `auth_type: api_key`, and it has to be a legal header name
        // (RFC 7230 `tchar` — the same set `http::HeaderName` accepts).
        {
            "if": { "properties": { "type": { "const": "openapi" } }, "required": ["type"] },
            "then": {
                "required": ["spec"],
                "properties": {
                    "spec": {
                        "type": "object",
                        "not": { "required": ["swagger"] }
                    },
                    "api_key_header": { "pattern": "^[!#$%&'*+.^_`|~0-9A-Za-z-]+$" }
                },
                // Value-sensitive, not presence-based: `api_key_header` is an
                // `Option<String>`, so an explicit `null` means "absent" and must
                // not drag in the `auth_type` requirement. `dependencies` keys off
                // presence alone and would reject `api_key_header: null`.
                "allOf": [
                    {
                        "if": {
                            "properties": { "api_key_header": { "type": "string" } },
                            "required": ["api_key_header"]
                        },
                        "then": {
                            "properties": { "auth_type": { "const": "api_key" } },
                            "required": ["auth_type"]
                        }
                    }
                ]
            }
        },
        // A plain MCP server has neither. `spec`/`api_key_header` are
        // `Option`-typed, so an explicit `null` is equivalent to absent — hence
        // `type: null` rather than a `required` negation. `type` defaults to
        // `mcp`, so the missing-key case has to be covered too.
        {
            "if": {
                "anyOf": [
                    {
                        "title": "type is mcp",
                        "properties": { "type": { "const": "mcp" } },
                        "required": ["type"]
                    },
                    {
                        "title": "type is absent (defaults to mcp)",
                        "not": { "required": ["type"] }
                    }
                ]
            },
            "then": {
                "properties": {
                    "spec": { "type": "null" },
                    "api_key_header": { "type": "null" }
                }
            }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_pins_each_mcp_obligation_individually() {
        // Baseline: valid openapi-backed server with api-key auth.
        let base = json!({
            "name": "erp",
            "url": "https://erp.internal/api",
            "type": "openapi",
            "spec": {"openapi": "3.0.0", "paths": {"/a": {"get": {"operationId": "list_a"}}}},
            "auth_type": "api_key",
            "secret": "k",
            "api_key_header": "X-Key"
        });
        crate::models::schema::validate_mcp_server(&base).expect("baseline must be valid");

        // `spec` must be an object — pinned independently of the swagger check,
        // so a non-object that is NOT a swagger document still fails.
        let mut v = base.clone();
        v["spec"] = json!("just a string");
        crate::models::schema::validate_mcp_server(&v)
            .expect_err("a non-object spec must be rejected on its own");
        let mut v = base.clone();
        v["spec"] = json!([{"openapi": "3.0.0"}]);
        crate::models::schema::validate_mcp_server(&v)
            .expect_err("an array spec must be rejected on its own");

        // oauth2: each of the three obligations pinned separately.
        for omit in ["secret", "client_id", "token_url"] {
            let mut v = json!({
                "name": "erp", "url": "https://x/mcp", "auth_type": "oauth2",
                "secret": "cs", "client_id": "cid", "token_url": "https://auth/token"
            });
            v.as_object_mut().unwrap().remove(omit);
            crate::models::schema::validate_mcp_server(&v)
                .expect_err(&format!("oauth2 missing only {omit} must be rejected"));
        }

        // Credential fields: empty and null are as bad as missing.
        for bad in [json!(""), json!(null)] {
            let mut v = json!({"name": "s", "url": "https://x/mcp", "auth_type": "bearer"});
            v["secret"] = bad.clone();
            crate::models::schema::validate_mcp_server(&v)
                .expect_err("bearer with an empty or null secret must be rejected");
        }

        // openapi-only fields rejected on an EXPLICIT type: mcp, not just a
        // defaulted one.
        for field in ["spec", "api_key_header"] {
            let mut v = json!({"name": "gh", "url": "https://x/mcp", "type": "mcp"});
            v[field] = if field == "spec" {
                json!({"openapi": "3.0.0"})
            } else {
                json!("X-Key")
            };
            crate::models::schema::validate_mcp_server(&v).expect_err(&format!(
                "{field} on an explicit type: mcp must be rejected"
            ));
        }

        // api_key_header requires api_key auth specifically — `none` is not enough.
        let mut v = base.clone();
        v["auth_type"] = json!("none");
        v.as_object_mut().unwrap().remove("secret");
        crate::models::schema::validate_mcp_server(&v)
            .expect_err("a header with auth_type none must be rejected");

        // …but an explicit null header is "absent" and must stay valid.
        let mut v = base.clone();
        v["api_key_header"] = json!(null);
        v["auth_type"] = json!("none");
        v.as_object_mut().unwrap().remove("secret");
        crate::models::schema::validate_mcp_server(&v).expect("api_key_header: null means absent");
    }

    #[test]
    fn schema_rejects_tool_namespace_separator_in_name() {
        // Tools are exposed as `<name>__<tool>`, so `__` inside the name makes
        // the split ambiguous.
        let doc = json!({"name": "git__hub", "url": "https://x/mcp"});
        let err = crate::models::schema::validate_mcp_server(&doc)
            .expect_err("a name containing `__` must be rejected");
        assert!(err.path.contains("name"), "unexpected path: {}", err.path);

        // A single underscore inside or leading the name is fine.
        for good in ["git_hub", "_github", "github", "a_b_c"] {
            let doc = json!({"name": good, "url": "https://x/mcp"});
            crate::models::schema::validate_mcp_server(&doc)
                .unwrap_or_else(|e| panic!("{good} should be accepted: {e:?}"));
        }

        // A trailing `_` collides with the separator's first byte: `gh_` + `x`
        // and `gh` + `_x` both serialize to `gh___x`, and `split_once("__")`
        // resolves `gh_` to the non-existent server `gh`, so every tool call
        // against it fails.
        for bad in ["github_", "_", "a_b_"] {
            let doc = json!({"name": bad, "url": "https://x/mcp"});
            crate::models::schema::validate_mcp_server(&doc)
                .expect_err(&format!("{bad} ends in the separator's first byte"));
        }
    }

    #[test]
    fn schema_enforces_openapi_field_coupling() {
        let spec = json!({"openapi": "3.0.0", "paths": {"/a": {"get": {"operationId": "list_a"}}}});

        // `openapi` type requires a `spec`, and it must be an OpenAPI 3.x object.
        for bad in [
            json!({"name": "s", "url": "https://x/mcp", "type": "openapi"}),
            json!({"name": "s", "url": "https://x/mcp", "type": "openapi", "spec": "str"}),
            json!({"name": "s", "url": "https://x/mcp", "type": "openapi", "spec": {"swagger": "2.0"}}),
        ] {
            crate::models::schema::validate_mcp_server(&bad)
                .expect_err("an openapi server needs a 3.x spec object");
        }

        // `api_key_header` only makes sense with `auth_type: api_key`, and has to
        // be a legal header name. This one is guarded by draft-07 `dependencies`
        // — the draft 2019-09 spelling would be ignored, so keep this test.
        let wrong_auth = json!({
            "name": "s", "url": "https://x/mcp", "type": "openapi",
            "spec": spec, "api_key_header": "X-Key"
        });
        crate::models::schema::validate_mcp_server(&wrong_auth)
            .expect_err("api_key_header without auth_type api_key must be rejected");

        let bad_header = json!({
            "name": "s", "url": "https://x/mcp", "type": "openapi", "spec": spec,
            "auth_type": "api_key", "secret": "s", "api_key_header": "bad header"
        });
        crate::models::schema::validate_mcp_server(&bad_header)
            .expect_err("a header name with a space must be rejected");

        // A plain mcp server carries neither field — `type` defaults to `mcp`,
        // so the missing-key case is covered too.
        for bad in [
            json!({"name": "s", "url": "https://x/mcp", "spec": {"openapi": "3.0.0"}}),
            json!({"name": "s", "url": "https://x/mcp", "api_key_header": "X-Key"}),
        ] {
            crate::models::schema::validate_mcp_server(&bad)
                .expect_err("spec/api_key_header are openapi-only");
        }

        // The complete openapi shape validates.
        let ok = json!({
            "name": "s", "url": "https://x/mcp", "type": "openapi",
            "spec": {"openapi": "3.0.0"}, "auth_type": "api_key",
            "secret": "s", "api_key_header": "X-Key"
        });
        crate::models::schema::validate_mcp_server(&ok).expect("complete openapi server");
    }

    #[test]
    fn schema_requires_credentials_per_auth_type() {
        for auth in ["bearer", "api_key"] {
            let doc = json!({"name": "s", "url": "https://x/mcp", "auth_type": auth});
            crate::models::schema::validate_mcp_server(&doc)
                .expect_err(&format!("{auth} without a secret must be rejected"));
        }

        // oauth2 needs the client credentials and the token endpoint too.
        let partial =
            json!({"name": "s", "url": "https://x/mcp", "auth_type": "oauth2", "secret": "cs"});
        crate::models::schema::validate_mcp_server(&partial)
            .expect_err("oauth2 without client_id/token_url must be rejected");

        let complete = json!({
            "name": "s", "url": "https://x/mcp", "auth_type": "oauth2",
            "secret": "cs", "client_id": "cid", "token_url": "https://auth/token"
        });
        crate::models::schema::validate_mcp_server(&complete)
            .expect("a complete oauth2 credential set must be accepted");
    }

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
            server_type: McpServerType::Mcp,
            url: "https://x/mcp".into(),
            spec: None,
            api_key_header: None,
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
        // Unset openapi-mode fields are omitted from the wire shape entirely.
        assert!(!s.contains("spec"), "got: {s}");
        assert!(!s.contains("api_key_header"), "got: {s}");
        let back: McpServer = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }

    // ---- `type: openapi` ----

    #[test]
    fn defaults_to_mcp_type() {
        let s: McpServer =
            serde_json::from_str(r#"{"name":"github","url":"https://x/mcp"}"#).unwrap();
        assert_eq!(s.server_type, McpServerType::Mcp);
        assert!(s.spec.is_none());
        assert!(s.api_key_header.is_none());
    }

    #[test]
    fn deserialises_openapi_server_with_spec() {
        let s: McpServer = serde_json::from_str(
            r#"{"name":"erp","type":"openapi","url":"https://erp.internal/api",
                "spec":{"openapi":"3.0.0","paths":{}},
                "auth_type":"api_key","secret":"k","api_key_header":"X-ERP-Key"}"#,
        )
        .unwrap();
        assert_eq!(s.server_type, McpServerType::Openapi);
        assert_eq!(
            s.spec.as_ref().and_then(|v| v.get("openapi")),
            Some(&serde_json::Value::String("3.0.0".into()))
        );
        assert_eq!(s.api_key_header.as_deref(), Some("X-ERP-Key"));
    }

    #[test]
    fn openapi_type_round_trips() {
        let original: McpServer = serde_json::from_str(
            r#"{"name":"erp","type":"openapi","url":"https://erp.internal/api","spec":{"openapi":"3.1.0","paths":{"/a":{"get":{"operationId":"x"}}}}}"#,
        )
        .unwrap();
        let s = serde_json::to_string(&original).unwrap();
        assert!(s.contains(r#""type":"openapi""#), "got: {s}");
        let back: McpServer = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn rejects_unknown_server_type() {
        assert!(
            serde_json::from_str::<McpServer>(r#"{"name":"x","url":"u","type":"grpc"}"#).is_err()
        );
    }
}
