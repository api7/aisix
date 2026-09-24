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

use super::mcp_ref::McpServerIndex;
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

    /// Registered MCP server names anonymous callers may reach.
    ///
    /// This is the anonymous principal's CEILING, not merely the list of
    /// scoped entries to open: the listed servers' tools are
    /// intersected with the key's own grant on BOTH entries. Without
    /// that, a key whose grant is wider than the list would let an
    /// anonymous caller reach an unlisted server's tools through the
    /// aggregated endpoint by naming `<server>__<tool>` directly —
    /// `/mcp/{server}` closed, aggregated `/mcp` open.
    ///
    /// Required and non-empty, because an empty ceiling admits no tool
    /// on either entry: an anonymous block listing no server could
    /// never serve a useful request, including through
    /// `aggregate_entry`. It is also why a newly registered server is
    /// never anonymous by default — reaching anonymous callers is
    /// always a name added here.
    ///
    /// Read only when `server_ids` is absent; ignored entirely when it
    /// is present. It stays required either way — it is the only
    /// spelling a gateway one release behind the control plane can
    /// read.
    #[schemars(length(min = 1))]
    pub servers: Vec<String>,

    /// The same allowlist written by MCP server resource id
    /// (`mcp_servers/<id>`) instead of by server name, so renaming a
    /// server does not drop it out of the anonymous ceiling.
    ///
    /// Present — including as an empty array — it is authoritative and
    /// `servers` is ignored. An empty array therefore admits no server
    /// at all: every `/mcp/{server}` entry closes, and so does the
    /// aggregated `/mcp` entry even with `aggregate_entry` set, since
    /// an open door onto an empty room reads as enabled, serves
    /// nothing, and suppresses the `WWW-Authenticate` discovery hint a
    /// standard client would otherwise follow. An id naming no
    /// registered server admits nothing, and the other entries are
    /// unaffected. Absent or `null`, the allowlist falls back to
    /// `servers`.
    ///
    /// A gateway one release behind the control plane does not read
    /// this field and applies `servers` instead, so the two spellings
    /// must be written to mean the same thing. For a non-empty array
    /// that is simply the current names of the servers it lists. The
    /// empty array has no name-form spelling at all — `servers` is
    /// required and non-empty on every release — so a control plane that
    /// offers it MUST keep an older gateway from reading a permissive
    /// `servers` beside it; emptying the ceiling would otherwise close
    /// it on new gateways and leave it open on older ones, which is the
    /// wrong direction for an access control. Even then the older
    /// gateway keeps serving the aggregated entry when `aggregate_entry`
    /// is set: that entry closing on an empty ceiling is a rule only
    /// this release knows, and no `servers` value can carry it back.
    ///
    /// It cannot express "every server": each entry names one server
    /// exactly and an id is never a glob, so an enumeration of the
    /// servers registered today silently fails to cover one registered
    /// tomorrow. To admit every server, present and future, leave this
    /// absent and use the name form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_ids: Option<Vec<String>>,

    /// Whether the aggregated `/mcp` endpoint ALSO serves anonymous
    /// callers. It exposes the same allowlisted servers, under their
    /// `<server>__<tool>` namespaced names. It cannot stand in for the
    /// allowlist: with no server allowlisted the aggregated entry stays
    /// closed whatever this says, the same way every scoped entry does.
    ///
    /// Off by default: it is the entry a standard MCP client uses for
    /// OAuth discovery, and the namespaced names are not what a client
    /// migrating from a single-server endpoint uses. Turning it on
    /// suppresses the `WWW-Authenticate` discovery hint there, since a
    /// no-credential request succeeds instead of producing the 401 that
    /// carries it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub aggregate_entry: bool,
}

fn default_enabled() -> bool {
    true
}

/// The anonymous caller's server allowlist, in the spelling the settings
/// row wrote it.
///
/// The two spellings answer the same question — which registered servers
/// this environment offers anonymously — and both consumers (the
/// `/mcp/{server}` entry gate and the ceiling laid over the bound
/// principal's tool ACL) must read the same one, which is why the choice
/// is made once by [`McpAnonymousAccess::server_allowlist`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpServerAllowlist {
    /// Servers named as they are registered.
    Names(Vec<String>),
    /// Servers named by the resource id they are stored under.
    Ids(Vec<String>),
}

impl McpAnonymousAccess {
    /// **The single chokepoint for the anonymous server allowlist.** The
    /// row carries it one of two ways and this decides between them, so
    /// no consumer can read one spelling and miss the other:
    ///
    /// - `server_ids` present (an empty array included): it is
    ///   authoritative and `servers` is not read at all.
    /// - `server_ids` absent: `servers` is read by name, as before.
    pub fn server_allowlist(&self) -> McpServerAllowlist {
        match &self.server_ids {
            Some(ids) => McpServerAllowlist::Ids(ids.clone()),
            None => McpServerAllowlist::Names(self.servers.clone()),
        }
    }
}

impl McpServerAllowlist {
    /// Whether the allowlist names nothing at all — no server is offered
    /// anonymously, on either entry.
    ///
    /// Only the id spelling can reach this: `servers` is required and
    /// non-empty on both schemas, precisely because an empty ceiling
    /// "could never serve a useful request, including through
    /// `aggregate_entry`". `server_ids: []` says exactly that, and says
    /// it deliberately, so the aggregated entry closes with it rather
    /// than becoming an open door onto an empty room.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Names(names) => names.is_empty(),
            Self::Ids(ids) => ids.is_empty(),
        }
    }

    /// Whether the allowlist offers the registered server `name`
    /// anonymously — the `/mcp/{server}` entry gate.
    ///
    /// The id spelling resolves through the live server index, so a
    /// rename moves the entry to the server's new name with the settings
    /// document untouched; an id naming no registered server offers
    /// nothing.
    pub fn admits_server(&self, servers: &McpServerIndex, name: &str) -> bool {
        match self {
            Self::Names(names) => names.iter().any(|s| s == name),
            Self::Ids(ids) => servers
                .id_of(name)
                .is_some_and(|id| ids.iter().any(|listed| listed == id)),
        }
    }
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
        assert!(validate_mcp_auth_settings(&serde_json::json!({
            "anonymous": { "api_key_id": "ak-1", "source_cidrs": ["10.0.0.0/8"] }
        }))
        .is_err());
    }

    #[test]
    fn anonymous_must_name_at_least_one_server() {
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
        // The list is the principal's CEILING, so an absent or empty one
        // admits no tool on either entry. `aggregate_entry` cannot stand
        // in for it: on its own it would be an open door onto an empty
        // room, which reads as enabled and serves nothing.
        assert!(validate_mcp_auth_settings(&base(serde_json::json!({}))).is_err());
        assert!(validate_mcp_auth_settings(&base(serde_json::json!({ "servers": [] }))).is_err());
        assert!(
            validate_mcp_auth_settings(&base(serde_json::json!({ "aggregate_entry": true })))
                .is_err()
        );
        assert!(validate_mcp_auth_settings(&base(
            serde_json::json!({ "servers": [], "aggregate_entry": true })
        ))
        .is_err());

        assert!(
            validate_mcp_auth_settings(&base(serde_json::json!({ "servers": ["docs"] }))).is_ok()
        );
        assert!(validate_mcp_auth_settings(&base(
            serde_json::json!({ "servers": ["docs"], "aggregate_entry": true })
        ))
        .is_ok());
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

    /// An index over `(id, name)` pairs, so an id-spelled allowlist has
    /// registered servers to resolve against.
    fn server_index(servers: &[(&str, &str)]) -> McpServerIndex {
        let snap = crate::AisixSnapshot::default();
        for (id, name) in servers {
            let server: super::super::McpServer = serde_json::from_str(&format!(
                r#"{{"name":"{name}","url":"https://example.test/mcp"}}"#
            ))
            .unwrap();
            snap.mcp_servers
                .insert(crate::resource::ResourceEntry::new(*id, server, 1));
        }
        McpServerIndex::build(&snap.mcp_servers)
    }

    fn anonymous(extra: serde_json::Value) -> McpAnonymousAccess {
        let mut block = serde_json::json!({
            "api_key_id": "ak-1",
            "source_cidrs": ["10.0.0.0/8"],
            "servers": ["docs"]
        });
        let obj = block.as_object_mut().unwrap();
        for (k, v) in extra.as_object().expect("anonymous overrides") {
            obj.insert(k.clone(), v.clone());
        }
        serde_json::from_value(block).expect("valid anonymous block")
    }

    #[test]
    fn the_allowlist_reads_the_name_form_by_default() {
        let index = server_index(&[("s-docs", "docs"), ("s-kb", "kb")]);
        let anon = anonymous(serde_json::json!({}));
        assert_eq!(
            anon.server_allowlist(),
            McpServerAllowlist::Names(vec!["docs".into()])
        );
        assert!(anon.server_allowlist().admits_server(&index, "docs"));
        assert!(!anon.server_allowlist().admits_server(&index, "kb"));
    }

    #[test]
    fn the_id_form_decides_whenever_it_is_present() {
        let index = server_index(&[("s-docs", "docs"), ("s-kb", "kb")]);
        // `servers` says docs, `server_ids` says kb. The names are not
        // read at all.
        let anon = anonymous(serde_json::json!({ "server_ids": ["s-kb"] }));
        assert_eq!(
            anon.server_allowlist(),
            McpServerAllowlist::Ids(vec!["s-kb".into()])
        );
        assert!(anon.server_allowlist().admits_server(&index, "kb"));
        assert!(!anon.server_allowlist().admits_server(&index, "docs"));

        // An empty array is authoritative too: no server is offered
        // anonymously, even though `servers` still names one.
        let emptied = anonymous(serde_json::json!({ "server_ids": [] }));
        assert_eq!(emptied.server_allowlist(), McpServerAllowlist::Ids(vec![]));
        assert!(!emptied.server_allowlist().admits_server(&index, "docs"));
        assert!(!emptied.server_allowlist().admits_server(&index, "kb"));
    }

    #[test]
    fn an_allowlist_that_names_nothing_is_empty_in_either_spelling() {
        // The one consumer that reads this is the aggregated `/mcp` entry
        // gate: `aggregate_entry` cannot stand in for the allowlist, so an
        // allowlist naming nothing closes that entry the way it closes
        // every scoped one. Only the id spelling can reach the state —
        // `servers` is required non-empty on both schemas.
        assert!(anonymous(serde_json::json!({ "server_ids": [] }))
            .server_allowlist()
            .is_empty());
        assert!(!anonymous(serde_json::json!({ "server_ids": ["s-docs"] }))
            .server_allowlist()
            .is_empty());
        assert!(!anonymous(serde_json::json!({}))
            .server_allowlist()
            .is_empty());
        // An id that resolves to nothing is not the same as naming
        // nothing: the operator asked for a server, it is simply absent.
        assert!(!anonymous(serde_json::json!({ "server_ids": ["s-gone"] }))
            .server_allowlist()
            .is_empty());
    }

    #[test]
    fn an_explicit_null_means_the_same_as_omitted() {
        let index = server_index(&[("s-docs", "docs")]);
        let anon = anonymous(serde_json::json!({ "server_ids": null }));
        assert_eq!(
            anon.server_allowlist(),
            McpServerAllowlist::Names(vec!["docs".into()])
        );
        assert!(anon.server_allowlist().admits_server(&index, "docs"));

        // And the schema takes it on both paths — a row the read schema
        // refused would be dropped whole, taking the OAuth discovery
        // surface down with anonymous access.
        let doc = serde_json::json!({
            "anonymous": {
                "api_key_id": "ak-1", "source_cidrs": ["10.0.0.0/8"],
                "servers": ["docs"], "server_ids": null
            }
        });
        validate_mcp_auth_settings(&doc).unwrap();
        validate_mcp_auth_settings_lenient(&doc).unwrap();
    }

    #[test]
    fn the_id_form_follows_a_rename_and_ignores_an_unresolvable_id() {
        let anon = anonymous(serde_json::json!({ "server_ids": ["s-docs"] }));
        assert!(anon
            .server_allowlist()
            .admits_server(&server_index(&[("s-docs", "docs")]), "docs"));

        // Same id, new name; the settings document is untouched.
        let renamed = server_index(&[("s-docs", "handbook")]);
        assert!(anon.server_allowlist().admits_server(&renamed, "handbook"));
        assert!(!anon.server_allowlist().admits_server(&renamed, "docs"));

        // An id naming no registered server offers nothing, and leaves
        // the entry beside it alone.
        let partial = anonymous(serde_json::json!({ "server_ids": ["s-gone", "s-docs"] }));
        let index = server_index(&[("s-docs", "docs"), ("s-kb", "kb")]);
        assert!(partial.server_allowlist().admits_server(&index, "docs"));
        assert!(!partial.server_allowlist().admits_server(&index, "kb"));
    }

    #[test]
    fn the_name_form_stays_required_beside_the_id_form_on_both_paths() {
        // Stricter than the "write the name form beside the id form"
        // guard the other MCP id spellings carry, and deliberately so: a
        // gateway one release behind the control plane reads `servers`
        // only, and the lenient set requires it too, so the control plane
        // has no way to write an anonymous ceiling that release cannot
        // read at all.
        let without_names = serde_json::json!({
            "anonymous": {
                "api_key_id": "ak-1", "source_cidrs": ["10.0.0.0/8"],
                "server_ids": ["s-docs"]
            }
        });
        assert!(validate_mcp_auth_settings(&without_names).is_err());
        assert!(validate_mcp_auth_settings_lenient(&without_names).is_err());

        let with_names = serde_json::json!({
            "anonymous": {
                "api_key_id": "ak-1", "source_cidrs": ["10.0.0.0/8"],
                "servers": ["docs"], "server_ids": []
            }
        });
        validate_mcp_auth_settings(&with_names).unwrap();
        validate_mcp_auth_settings_lenient(&with_names).unwrap();
    }

    #[test]
    fn server_ids_stays_off_the_wire_when_absent() {
        let anon = anonymous(serde_json::json!({}));
        let v = serde_json::to_value(&anon).unwrap();
        assert!(v.get("server_ids").is_none());

        let with_ids = anonymous(serde_json::json!({ "server_ids": ["s-docs"] }));
        let v = serde_json::to_value(&with_ids).unwrap();
        assert_eq!(v["server_ids"], serde_json::json!(["s-docs"]));
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
