//! Inbound MCP OAuth 2.1 discovery surface (AISIX-Cloud#1143).
//!
//! Makes `/mcp` a spec-compliant OAuth 2.1 resource server per the MCP
//! authorization spec (2025-11-25): an RFC 9728 Protected Resource
//! Metadata document under `/.well-known/oauth-protected-resource`
//! (both the root and the path-insertion `/mcp` form), and RFC 6750
//! `WWW-Authenticate` challenges on `/mcp` auth failures so a standard
//! MCP client can discover the authorization server from a bare 401.
//!
//! The surface is ACTIVE only when the environment projects a valid
//! [`McpAuthSettings`] row (the canonical `/mcp` resource URL) AND at
//! least one enabled `oidc_providers` row exists. Dormant otherwise:
//! the well-known routes 404 and no challenge header is attached, so
//! every pre-#1143 environment is byte-identical to before.
//!
//! Token validation itself is unchanged — `crate::jwt` already enforces
//! signature/`iss`/`exp`/`aud` (inclusion semantics) and maps the
//! identity onto a bound API key. This module only adds the discovery
//! face in front of it.

use std::collections::BTreeSet;
use std::sync::Once;

use aisix_core::models::McpAuthSettings;
use aisix_core::AisixSnapshot;
use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::error::AuthChallenge;
use crate::state::ProxyState;

/// The resolved discovery identity of an active environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DiscoveryIdentity {
    /// The canonical `/mcp` resource URI, published verbatim as the PRM
    /// `resource` (never derived from the request Host header).
    pub resource_url: String,
    /// Absolute URL of the path-insertion PRM endpoint, carried in
    /// every challenge's `resource_metadata` attribute. Derived from
    /// `resource_url`'s origin, never from the request.
    pub challenge_url: String,
}

/// Resolve the discovery identity, or `None` while the surface is
/// dormant (no settings row, duplicate settings rows, no enabled
/// provider, or a malformed row).
///
/// A malformed row (unparseable URL, non-http(s) scheme, query or
/// fragment, path ≠ `/mcp`) and a duplicated singleton both keep the
/// surface dormant and log one process-wide warning — never per
/// request. The absent-row state is a legitimate steady state (every
/// pre-#1143 environment) and logs nothing.
pub(crate) fn discovery_identity(snapshot: &AisixSnapshot) -> Option<DiscoveryIdentity> {
    let entries = snapshot.mcp_auth_settings.entries();
    if entries.is_empty() {
        return None;
    }
    if entries.len() > 1 {
        // The CP keys the row by the environment id, so a second row can
        // only be a stale/migrated or hand-written key — and no ordering
        // over the ids says which one is current. Publishing either would
        // put a coin-flip URI in the PRM `resource` and in the audience
        // the tokens are checked against, so fail closed instead: the
        // surface stays dormant until the duplicate is removed.
        //
        // The check belongs here, not in the loader: the watch supervisor
        // applies puts incrementally and never re-runs the full-load path,
        // so a duplicate can appear in a live snapshot without the loader
        // ever seeing both rows together.
        static MULTI_WARN: Once = Once::new();
        MULTI_WARN.call_once(|| {
            tracing::warn!(
                target: "aisix::mcp_auth",
                count = entries.len(),
                "multiple mcp_auth_settings rows in snapshot; the OAuth discovery \
                 surface stays dormant until exactly one remains",
            );
        });
        return None;
    }
    let settings = &entries[0];

    let Some(identity) = validate_resource_url(&settings.value) else {
        static MALFORMED_WARN: Once = Once::new();
        MALFORMED_WARN.call_once(|| {
            tracing::warn!(
                target: "aisix::mcp_auth",
                "mcp_auth_settings.resource_url is not an absolute http(s) URL with path \
                 exactly /mcp and no query or fragment; the OAuth discovery surface \
                 stays dormant",
            );
        });
        return None;
    };

    if !crate::jwt::any_enabled_provider(snapshot) {
        return None;
    }
    Some(identity)
}

/// Parse and validate the configured resource URL: absolute `http`/
/// `https`, no query, no fragment, path exactly `/mcp` (the gateway's
/// fixed MCP route — the CP rejects anything else at write time; this
/// is defense in depth on the projected row).
fn validate_resource_url(settings: &McpAuthSettings) -> Option<DiscoveryIdentity> {
    let parsed = url::Url::parse(&settings.resource_url).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }
    // Userinfo would be published verbatim on the unauthenticated PRM
    // endpoint — a credential pasted into the URL must never activate
    // the surface (audit finding on #859).
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return None;
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return None;
    }
    if parsed.path() != "/mcp" {
        return None;
    }
    // `Origin::ascii_serialization` renders `scheme://host[:port]` with
    // default ports elided — exactly the base the well-known routes are
    // reachable under.
    let origin = parsed.origin().ascii_serialization();
    Some(DiscoveryIdentity {
        resource_url: settings.resource_url.clone(),
        challenge_url: format!("{origin}/.well-known/oauth-protected-resource/mcp"),
    })
}

/// The RFC 9728 Protected Resource Metadata document, derived entirely
/// from configuration: `authorization_servers` lists the enabled trust
/// providers' issuers, `scopes_supported` the union of their
/// `required_scopes` (omitted when empty).
fn prm_document(snapshot: &AisixSnapshot, identity: &DiscoveryIdentity) -> serde_json::Value {
    let mut issuers: Vec<String> = Vec::new();
    let mut scopes: BTreeSet<String> = BTreeSet::new();
    for entry in snapshot.oidc_providers.entries() {
        if !entry.value.enabled {
            continue;
        }
        issuers.push(entry.value.issuer.clone());
        scopes.extend(entry.value.required_scopes.iter().cloned());
    }
    issuers.sort();
    issuers.dedup();

    let mut doc = json!({
        "resource": identity.resource_url,
        "authorization_servers": issuers,
        "bearer_methods_supported": ["header"],
    });
    if !scopes.is_empty() {
        doc["scopes_supported"] = json!(scopes.into_iter().collect::<Vec<_>>());
    }
    doc
}

/// `GET /.well-known/oauth-protected-resource` and its RFC 9728
/// path-insertion sibling `…/mcp`. Unauthenticated by design (discovery
/// must precede auth). Registered with `any(...)` and the method check
/// done here, AFTER the dormancy check: a dormant environment answers a
/// bare empty 404 for EVERY method — byte-identical to the route not
/// existing (axum's pre-#1143 fallback) — while an active one answers
/// 405 + `Allow` for non-GET/HEAD.
pub(crate) async fn protected_resource_metadata(
    method: axum::http::Method,
    State(state): State<ProxyState>,
) -> Response {
    let snapshot = state.snapshot.load();
    let Some(identity) = discovery_identity(&snapshot) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if method != axum::http::Method::GET && method != axum::http::Method::HEAD {
        let mut response = StatusCode::METHOD_NOT_ALLOWED.into_response();
        response
            .headers_mut()
            .insert(header::ALLOW, HeaderValue::from_static("GET, HEAD"));
        return response;
    }
    Json(prm_document(&snapshot, &identity)).into_response()
}

/// The `WWW-Authenticate` value for one challenge classification.
/// Values interpolated into the header are filtered to RFC 6750 NQCHAR
/// (visible ASCII minus `"` and `\`) so a config value can neither
/// break out of the quoted attribute nor corrupt the space-separated
/// scope list, and `HeaderValue::from_str` cannot fail on them — a
/// stray control byte must lose one character, never the whole header
/// (the CP validates upstream; this is depth).
fn challenge_header_value(challenge: &AuthChallenge, challenge_url: &str) -> Option<HeaderValue> {
    let quote = |s: &str| -> String {
        s.chars()
            .filter(|c| matches!(*c, '\x21' | '\x23'..='\x5B' | '\x5D'..='\x7E'))
            .collect()
    };
    let url = quote(challenge_url);
    let value = match challenge {
        AuthChallenge::MissingCredentials => {
            format!("Bearer resource_metadata=\"{url}\"")
        }
        AuthChallenge::InvalidToken => {
            format!("Bearer error=\"invalid_token\", resource_metadata=\"{url}\"")
        }
        AuthChallenge::InsufficientScope { required_scopes } => {
            let scope = required_scopes
                .iter()
                .map(|s| quote(s))
                .collect::<Vec<_>>()
                .join(" ");
            format!(
                "Bearer error=\"insufficient_scope\", scope=\"{scope}\", \
                 resource_metadata=\"{url}\""
            )
        }
    };
    HeaderValue::from_str(&value).ok()
}

/// Route-scoped middleware on the `/mcp` routes only: when the response
/// carries an [`AuthChallenge`] marker (attached by `ProxyError`'s
/// renderer) and the surface is active, attach the `WWW-Authenticate`
/// header. Everything else — inactive environments, non-auth errors,
/// success responses — passes through untouched.
///
/// The snapshot is loaded here a second time (the auth decision loaded
/// its own copy), so a config swap mid-request can attach a challenge
/// reflecting newer config than the 401 was produced under. Accepted
/// eventual consistency: the client re-runs discovery and self-heals.
pub(crate) async fn challenge_middleware(
    State(state): State<ProxyState>,
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let Some(challenge) = response.extensions().get::<AuthChallenge>().cloned() else {
        return response;
    };
    let snapshot = state.snapshot.load();
    let Some(identity) = discovery_identity(&snapshot) else {
        return response;
    };
    if let Some(value) = challenge_header_value(&challenge, &identity.challenge_url) {
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, value);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::models::OidcProvider;
    use aisix_core::resource::ResourceEntry;

    fn settings_entry(id: &str, resource_url: &str) -> ResourceEntry<McpAuthSettings> {
        let s: McpAuthSettings =
            serde_json::from_str(&format!(r#"{{"resource_url": "{resource_url}"}}"#)).unwrap();
        ResourceEntry::new(id, s, 1)
    }

    fn provider_entry(id: &str, json: &str) -> ResourceEntry<OidcProvider> {
        let p: OidcProvider = serde_json::from_str(json).unwrap();
        ResourceEntry::new(id, p, 1)
    }

    fn active_snapshot() -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.mcp_auth_settings
            .insert(settings_entry("env-1", "https://gw.example.com/mcp"));
        snap.oidc_providers.insert(provider_entry(
            "op-1",
            r#"{"name":"corp","issuer":"https://sso.example.com/realms/agents",
                "audiences":["https://gw.example.com/mcp"],
                "required_scopes":["mcp:tools"]}"#,
        ));
        snap
    }

    #[test]
    fn dormant_without_settings_row() {
        let snap = AisixSnapshot::new();
        snap.oidc_providers.insert(provider_entry(
            "op-1",
            r#"{"name":"corp","issuer":"https://sso.example.com","audiences":["a"]}"#,
        ));
        assert_eq!(discovery_identity(&snap), None);
    }

    #[test]
    fn dormant_without_enabled_provider() {
        let snap = AisixSnapshot::new();
        snap.mcp_auth_settings
            .insert(settings_entry("env-1", "https://gw.example.com/mcp"));
        assert_eq!(discovery_identity(&snap), None);
        // A disabled provider does not activate the surface either.
        snap.oidc_providers.insert(provider_entry(
            "op-1",
            r#"{"name":"corp","issuer":"https://sso.example.com","audiences":["a"],
                "enabled": false}"#,
        ));
        assert_eq!(discovery_identity(&snap), None);
    }

    #[test]
    fn dormant_with_duplicate_settings_rows() {
        // The row is a per-environment singleton. A second one (a stale
        // or migrated key) must not be resolved by picking an id order:
        // that would publish a coin-flip `resource` URI and audience
        // target. Fail closed until exactly one row remains.
        let snap = active_snapshot();
        assert!(discovery_identity(&snap).is_some());
        // The stale row deliberately sorts BEFORE the live one, so a
        // reintroduced "smallest id wins" would resolve the stale URL.
        snap.mcp_auth_settings
            .insert(settings_entry("env-0", "https://stale.example.com/mcp"));
        assert_eq!(snap.mcp_auth_settings.entries().len(), 2);
        assert_eq!(discovery_identity(&snap), None);
    }

    #[test]
    fn dormant_on_malformed_resource_url() {
        for bad in [
            "not a url",
            "ftp://gw.example.com/mcp",
            "https://gw.example.com/other",
            "https://gw.example.com/mcp?x=1",
            "https://gw.example.com/mcp#frag",
            "https://gw.example.com/",
            // Userinfo would be served verbatim on the unauthenticated
            // PRM endpoint — never activate on it.
            "https://user:s3cret@gw.example.com/mcp",
            "https://user@gw.example.com/mcp",
        ] {
            let snap = AisixSnapshot::new();
            snap.mcp_auth_settings.insert(settings_entry("env-1", bad));
            snap.oidc_providers.insert(provider_entry(
                "op-1",
                r#"{"name":"corp","issuer":"https://sso.example.com","audiences":["a"]}"#,
            ));
            assert_eq!(discovery_identity(&snap), None, "must stay dormant: {bad}");
        }
    }

    #[test]
    fn active_identity_derives_challenge_url_from_origin() {
        let identity = discovery_identity(&active_snapshot()).expect("active");
        assert_eq!(identity.resource_url, "https://gw.example.com/mcp");
        assert_eq!(
            identity.challenge_url,
            "https://gw.example.com/.well-known/oauth-protected-resource/mcp"
        );
    }

    #[test]
    fn non_default_port_survives_in_challenge_url() {
        let snap = AisixSnapshot::new();
        snap.mcp_auth_settings
            .insert(settings_entry("env-1", "http://gw.internal:8443/mcp"));
        snap.oidc_providers.insert(provider_entry(
            "op-1",
            r#"{"name":"corp","issuer":"https://sso.example.com","audiences":["a"]}"#,
        ));
        let identity = discovery_identity(&snap).expect("active");
        assert_eq!(
            identity.challenge_url,
            "http://gw.internal:8443/.well-known/oauth-protected-resource/mcp"
        );
    }

    #[test]
    fn prm_document_lists_enabled_issuers_and_scope_union() {
        let snap = active_snapshot();
        // A second enabled provider with another scope, and a disabled
        // one that must not appear.
        snap.oidc_providers.insert(provider_entry(
            "op-2",
            r#"{"name":"partner","issuer":"https://partner.example.com",
                "audiences":["https://gw.example.com/mcp"],
                "required_scopes":["mcp:read","mcp:tools"]}"#,
        ));
        snap.oidc_providers.insert(provider_entry(
            "op-3",
            r#"{"name":"dormant","issuer":"https://off.example.com",
                "audiences":["a"],"enabled":false}"#,
        ));
        let identity = discovery_identity(&snap).unwrap();
        let doc = prm_document(&snap, &identity);
        assert_eq!(doc["resource"], "https://gw.example.com/mcp");
        assert_eq!(
            doc["authorization_servers"],
            json!([
                "https://partner.example.com",
                "https://sso.example.com/realms/agents"
            ])
        );
        assert_eq!(doc["bearer_methods_supported"], json!(["header"]));
        assert_eq!(doc["scopes_supported"], json!(["mcp:read", "mcp:tools"]));
    }

    #[test]
    fn prm_document_omits_empty_scopes_supported() {
        let snap = AisixSnapshot::new();
        snap.mcp_auth_settings
            .insert(settings_entry("env-1", "https://gw.example.com/mcp"));
        snap.oidc_providers.insert(provider_entry(
            "op-1",
            r#"{"name":"corp","issuer":"https://sso.example.com","audiences":["a"]}"#,
        ));
        let identity = discovery_identity(&snap).unwrap();
        let doc = prm_document(&snap, &identity);
        assert!(doc.get("scopes_supported").is_none());
    }

    #[test]
    fn challenge_values_per_classification() {
        let url = "https://gw.example.com/.well-known/oauth-protected-resource/mcp";
        let h = challenge_header_value(&AuthChallenge::MissingCredentials, url).unwrap();
        assert_eq!(
            h.to_str().unwrap(),
            format!("Bearer resource_metadata=\"{url}\"")
        );

        let h = challenge_header_value(&AuthChallenge::InvalidToken, url).unwrap();
        assert_eq!(
            h.to_str().unwrap(),
            format!("Bearer error=\"invalid_token\", resource_metadata=\"{url}\"")
        );

        let h = challenge_header_value(
            &AuthChallenge::InsufficientScope {
                required_scopes: vec!["mcp:tools".into(), "mcp:read".into()],
            },
            url,
        )
        .unwrap();
        assert_eq!(
            h.to_str().unwrap(),
            format!(
                "Bearer error=\"insufficient_scope\", scope=\"mcp:tools mcp:read\", \
                 resource_metadata=\"{url}\""
            )
        );
    }

    #[test]
    fn challenge_value_strips_quote_breakouts() {
        let h = challenge_header_value(
            &AuthChallenge::InsufficientScope {
                required_scopes: vec!["a\"b\\c".into()],
            },
            "https://gw.example.com/.well-known/oauth-protected-resource/mcp",
        )
        .unwrap();
        assert!(h.to_str().unwrap().contains("scope=\"abc\""));
    }

    #[test]
    fn challenge_value_survives_space_and_control_bytes_in_scopes() {
        // A space would corrupt the RFC 6750 space-separated scope list
        // and a control byte would previously make HeaderValue::from_str
        // fail — dropping the WHOLE header, resource_metadata included.
        // The NQCHAR filter must lose only the offending characters.
        let h = challenge_header_value(
            &AuthChallenge::InsufficientScope {
                required_scopes: vec!["mcp tools".into(), "a\u{7}b".into()],
            },
            "https://gw.example.com/.well-known/oauth-protected-resource/mcp",
        )
        .expect("header must survive hostile scope values");
        assert!(h.to_str().unwrap().contains("scope=\"mcptools ab\""));
    }
}
