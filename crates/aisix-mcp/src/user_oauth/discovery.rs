//! Where a server's OAuth endpoints are: what its configuration says, then
//! RFC 9728 protected-resource metadata and RFC 8414 authorization-server
//! metadata fetched from the upstream.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use reqwest::Url;
use serde::Deserialize;

use super::http::{send, Gate, HttpFailure};
use super::UserOAuthServer;
use crate::error::UserOAuthFailure;

/// How long a discovered document set is reused. Short, because nothing
/// tells a gateway when an upstream moves its endpoints.
const DISCOVERY_TTL: Duration = Duration::from_secs(300);

/// A server's OAuth endpoints, each from configuration when configured.
#[derive(Clone, Debug, Default)]
pub(super) struct Endpoints {
    pub authorization: Option<String>,
    pub token: Option<String>,
    pub registration: Option<String>,
    /// The RFC 8707 resource to name upstream, from the server's own
    /// protected-resource metadata; unknown when that was not fetched.
    pub resource: Option<String>,
    /// The issuer whose metadata named these endpoints.
    pub issuer: Option<String>,
    /// Scopes the server's protected-resource metadata asks for.
    pub scopes: Vec<String>,
}

#[derive(Default)]
pub(super) struct DiscoveryCache {
    entries: Mutex<HashMap<(String, String), (Instant, Endpoints)>>,
}

impl DiscoveryCache {
    /// The endpoints of `server`, contacting the upstream only for what its
    /// configuration leaves out.
    pub(super) async fn endpoints(
        &self,
        server: &UserOAuthServer,
    ) -> Result<Endpoints, UserOAuthFailure> {
        let configured = Endpoints {
            authorization: server.authorization_url.clone(),
            token: server.token_url.clone(),
            registration: server.registration_url.clone(),
            ..Default::default()
        };
        let complete = configured.authorization.is_some()
            && configured.token.is_some()
            && (server.client_id.is_some() || configured.registration.is_some());
        if complete {
            return Ok(configured);
        }
        let discovered = self.discovered(server).await?;
        Ok(Endpoints {
            authorization: configured.authorization.or(discovered.authorization),
            token: configured.token.or(discovered.token),
            registration: configured.registration.or(discovered.registration),
            resource: discovered.resource,
            issuer: discovered.issuer,
            scopes: discovered.scopes,
        })
    }

    /// The token endpoint alone, which is all a refresh needs.
    pub(super) async fn token_endpoint(
        &self,
        server: &UserOAuthServer,
    ) -> Result<String, UserOAuthFailure> {
        if let Some(url) = &server.token_url {
            return Ok(url.clone());
        }
        self.discovered(server)
            .await?
            .token
            .ok_or(UserOAuthFailure::Misconfigured)
    }

    async fn discovered(&self, server: &UserOAuthServer) -> Result<Endpoints, UserOAuthFailure> {
        let key = (server.id.clone(), server.url.clone());
        {
            let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
            if let Some((at, endpoints)) = entries.get(&key) {
                if at.elapsed() < DISCOVERY_TTL {
                    return Ok(endpoints.clone());
                }
            }
        }
        // One budget for however many documents it takes to find them.
        let endpoints = tokio::time::timeout(server.timeout, discover(server))
            .await
            .map_err(|_| UserOAuthFailure::Unavailable)??;
        self.entries
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(key, (Instant::now(), endpoints.clone()));
        Ok(endpoints)
    }
}

#[derive(Deserialize)]
struct ResourceMetadata {
    #[serde(default)]
    resource: Option<String>,
    #[serde(default)]
    authorization_servers: Vec<String>,
    #[serde(default)]
    scopes_supported: Vec<String>,
}

#[derive(Deserialize)]
struct ServerMetadata {
    #[serde(default)]
    issuer: Option<String>,
    #[serde(default)]
    authorization_endpoint: Option<String>,
    #[serde(default)]
    token_endpoint: Option<String>,
    #[serde(default)]
    registration_endpoint: Option<String>,
}

/// What one metadata fetch produced.
enum Fetched<T> {
    Document(T),
    /// The URL answered, with no such document.
    Absent,
    /// It did not answer in a way that says whether the document exists.
    Unknown,
}

async fn fetch_json<T: serde::de::DeserializeOwned>(
    server: &UserOAuthServer,
    url: &str,
) -> Result<Fetched<T>, UserOAuthFailure> {
    let gate = server.gate_for(url);
    let response = match send(&server.name, gate, url, server.timeout, |client, url| {
        client.get(url).header("accept", "application/json")
    })
    .await
    {
        Ok(response) => response,
        Err(HttpFailure::Transport) => return Ok(Fetched::Unknown),
        Err(HttpFailure::Blocked) => return Err(UserOAuthFailure::Misconfigured),
    };
    let status = response.status();
    if status.is_server_error() {
        return Ok(Fetched::Unknown);
    }
    if !status.is_success() {
        return Ok(Fetched::Absent);
    }
    Ok(response
        .json::<T>()
        .await
        .map_or(Fetched::Absent, Fetched::Document))
}

/// `resource_metadata` out of a `WWW-Authenticate` header (RFC 9728 §5.1).
fn challenge_metadata_url(header: &str) -> Option<String> {
    let (_, rest) = header.split_once("resource_metadata=")?;
    let rest = rest.trim_start();
    let value = match rest.strip_prefix('"') {
        Some(quoted) => quoted.split('"').next()?,
        None => rest.split([',', ' ']).next()?,
    };
    (!value.is_empty()).then(|| value.to_string())
}

/// `scope` out of a `WWW-Authenticate` header (RFC 6750 §3).
fn challenge_scopes(header: &str) -> Vec<String> {
    let found = header
        .match_indices("scope=\"")
        .find(|(at, _)| *at == 0 || !header.as_bytes()[at - 1].is_ascii_alphanumeric());
    found
        .and_then(|(at, _)| header[at + "scope=\"".len()..].split('"').next())
        .map(|scopes| scopes.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default()
}

/// `url` with `well_known` inserted between its origin and path, and with no
/// trailing slash (RFC 8414 §3.1, RFC 9728 §3.1).
fn path_insert(url: &Url, well_known: &str) -> String {
    let origin = url.origin().ascii_serialization();
    let path = url.path().trim_end_matches('/');
    format!("{origin}/.well-known/{well_known}{path}")
}

async fn discover(server: &UserOAuthServer) -> Result<Endpoints, UserOAuthFailure> {
    let url = Url::parse(&server.url).map_err(|_| UserOAuthFailure::Misconfigured)?;

    let probe = send(
        &server.name,
        Gate::Configured,
        &server.url,
        server.timeout,
        |client, url| {
            client
                .get(url)
                .header("accept", "application/json, text/event-stream")
        },
    )
    .await;
    let challenge = match &probe {
        Ok(response) => response
            .headers()
            .get(http::header::WWW_AUTHENTICATE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string),
        Err(HttpFailure::Blocked) => return Err(UserOAuthFailure::Misconfigured),
        Err(HttpFailure::Transport) => return Err(UserOAuthFailure::Unavailable),
    };
    let challenge_url = challenge.as_deref().and_then(challenge_metadata_url);

    // An answer that does not name the document keeps looking at the next
    // candidate; `uncertain` remembers that a failure, not an absence, may
    // be why nothing was found.
    let mut uncertain = false;
    let mut resource_metadata: Option<ResourceMetadata> = None;
    let candidates = match challenge_url {
        Some(url) => vec![url],
        None => {
            let mut urls = vec![path_insert(&url, "oauth-protected-resource")];
            let root = format!(
                "{}/.well-known/oauth-protected-resource",
                url.origin().ascii_serialization()
            );
            if urls[0] != root {
                urls.push(root);
            }
            urls
        }
    };
    for candidate in &candidates {
        match fetch_json::<ResourceMetadata>(server, candidate).await? {
            Fetched::Document(document) => {
                resource_metadata = Some(document);
                break;
            }
            Fetched::Absent => {}
            Fetched::Unknown => uncertain = true,
        }
    }

    let (resource, mut issuers, mut scopes) = match resource_metadata {
        Some(document) => (
            document.resource,
            document.authorization_servers,
            document.scopes_supported,
        ),
        None => (None, Vec::new(), Vec::new()),
    };
    if let Some(from_challenge) = challenge.as_deref().map(challenge_scopes) {
        if !from_challenge.is_empty() {
            scopes = from_challenge;
        }
    }
    // A server that names no authorization server is taken to be its own,
    // as the MCP authorization revisions before RFC 9728 had it.
    if issuers.is_empty() {
        issuers.push(url.origin().ascii_serialization());
    }

    for issuer in &issuers {
        let Ok(parsed) = Url::parse(issuer) else {
            continue;
        };
        let origin = parsed.origin().ascii_serialization();
        let mut urls = Vec::new();
        if !parsed.path().trim_matches('/').is_empty() {
            urls.push(path_insert(&parsed, "oauth-authorization-server"));
            urls.push(path_insert(&parsed, "openid-configuration"));
            urls.push(format!(
                "{}/.well-known/openid-configuration",
                issuer.trim_end_matches('/')
            ));
        }
        urls.push(format!("{origin}/.well-known/oauth-authorization-server"));
        urls.push(format!("{origin}/.well-known/openid-configuration"));
        for candidate in &urls {
            match fetch_json::<ServerMetadata>(server, candidate).await? {
                Fetched::Document(document)
                    if document.authorization_endpoint.is_some()
                        || document.token_endpoint.is_some() =>
                {
                    return Ok(Endpoints {
                        authorization: document.authorization_endpoint,
                        token: document.token_endpoint,
                        registration: document.registration_endpoint,
                        resource,
                        issuer: document.issuer.or_else(|| Some(issuer.clone())),
                        scopes,
                    });
                }
                Fetched::Document(_) | Fetched::Absent => {}
                Fetched::Unknown => uncertain = true,
            }
        }
    }
    tracing::warn!(
        target: "aisix::mcp_oauth",
        server = %server.name,
        uncertain,
        "found no OAuth authorization server metadata for an MCP server; configure \
         authorization_url and token_url if it publishes none",
    );
    Err(if uncertain {
        UserOAuthFailure::Unavailable
    } else {
        UserOAuthFailure::Misconfigured
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_resource_metadata_from_a_challenge() {
        assert_eq!(
            challenge_metadata_url(
                r#"Bearer error="invalid_token", resource_metadata="https://a.example/.well-known/x""#
            )
            .as_deref(),
            Some("https://a.example/.well-known/x")
        );
        assert_eq!(
            challenge_metadata_url("Bearer resource_metadata=https://a.example/m, scope=x")
                .as_deref(),
            Some("https://a.example/m")
        );
        assert_eq!(challenge_metadata_url("Bearer"), None);
    }

    #[test]
    fn reads_scopes_from_a_challenge_without_matching_a_longer_attribute() {
        assert_eq!(
            challenge_scopes(r#"Bearer scope="read write""#),
            vec!["read", "write"]
        );
        assert!(challenge_scopes(r#"Bearer xscope="read""#).is_empty());
    }

    #[test]
    fn inserts_well_known_between_origin_and_path() {
        let url = Url::parse("https://a.example:8443/t/mcp/").unwrap();
        assert_eq!(
            path_insert(&url, "oauth-protected-resource"),
            "https://a.example:8443/.well-known/oauth-protected-resource/t/mcp"
        );
    }
}
