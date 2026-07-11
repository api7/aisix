//! The upstream A2A client, behind the [`A2aBridge`] trait.
//!
//! A bridge targets one upstream agent and exposes just the two operations the
//! gateway needs in this first cut: fetch the agent's card, and forward a
//! JSON-RPC request to it. Aggregating bridges behind the downstream-facing
//! `/a2a/<agent>` endpoint, agent-card URL rewriting, and wiring into the
//! shared guardrail/quota pipeline come in later steps — this layer only proves
//! a governed tunnel to one real upstream.
//!
//! The upstream credential is held here on the gateway side and is never
//! exposed to the calling client, which presents only its AISIX key.
//!
//! Wire references (verified against the A2A specification):
//! - Agent card discovery: `https://{domain}/.well-known/agent-card.json`,
//!   an RFC 8615 well-known URI resolved at the domain origin.
//!   <https://a2a-protocol.org/latest/topics/agent-discovery/>
//! - `message/send` is a JSON-RPC 2.0 method whose envelope differs between the
//!   A2A 0.3 and 1.0 wire formats. This bridge forwards the caller's request
//!   verbatim and does not translate between versions, so the method name and
//!   body shape are the caller's concern, not this layer's.
//!   <https://a2a-protocol.org/latest/topics/life-of-a-task/>

use std::sync::OnceLock;
use std::time::Duration;

use aisix_core::{A2aAgent, A2aAuthType};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error::A2aError;

/// Default deadline for a single upstream operation (card fetch or send).
/// reqwest has no default request timeout, so without this a hung or slow
/// upstream pins the gateway request task indefinitely. Overridable per
/// upstream via the `A2aAgent.timeout_ms` field.
pub const DEFAULT_UPSTREAM_TIMEOUT: Duration = Duration::from_secs(30);

/// Header carrying the gateway-held key for `api_key` upstream auth.
const API_KEY_HEADER: &str = "x-api-key";

/// Standard RFC 8615 well-known path for an A2A agent card.
const AGENT_CARD_PATH: &str = "/.well-known/agent-card.json";

/// Hard cap on an upstream response body the gateway will buffer. A registered
/// agent is semi-trusted, but a compromised or misbehaving one must not be able
/// to OOM the gateway with a multi-gigabyte (or unbounded streaming) response.
/// Generous for a JSON-RPC task result; a per-agent override can be added later.
const MAX_UPSTREAM_BODY_BYTES: usize = 16 * 1024 * 1024;

/// The shared outbound HTTP client. Built once (a `reqwest::Client` is a
/// connection-pool handle — cloning is cheap and shares the pool) so every
/// upstream call reuses connections instead of standing up a fresh pool + TLS
/// handshake per request.
///
/// Redirects are refused: the data plane runs inside the customer's VPC, and a
/// compromised or MITM'd upstream returning `302 Location: http://169.254.169.254/…`
/// (or a loopback address) would otherwise turn the gateway into an SSRF pivot.
/// A legitimate A2A agent does not redirect its JSON-RPC endpoint or card. This
/// mirrors the MCP OAuth client, which refuses redirects for the same reason.
fn shared_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("reqwest client (redirect-disabled) builds")
        })
        .clone()
}

/// Read an upstream response body, refusing anything larger than
/// [`MAX_UPSTREAM_BODY_BYTES`]. An honestly-declared oversized `Content-Length`
/// is rejected up front; a lying or absent length (including a never-ending
/// stream) is caught as chunks accumulate, so the buffer can never exceed the
/// cap regardless of what the upstream claims.
async fn read_capped(resp: reqwest::Response) -> Result<Vec<u8>, A2aError> {
    if let Some(len) = resp.content_length() {
        if len > MAX_UPSTREAM_BODY_BYTES as u64 {
            return Err(A2aError::Request(format!(
                "upstream response too large: {len} bytes"
            )));
        }
    }
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| A2aError::Request(e.to_string()))?;
        if buf.len() + chunk.len() > MAX_UPSTREAM_BODY_BYTES {
            return Err(A2aError::Request(
                "upstream response exceeded size cap".to_string(),
            ));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// How the gateway authenticates to an upstream A2A agent. The credential is
/// held here on the gateway side and is never exposed to the calling client —
/// the client presents only its AISIX key.
#[derive(Clone)]
pub enum A2aAuth {
    /// No upstream auth — the agent is reachable as-is.
    None,
    /// Send `Authorization: Bearer <token>` on every upstream request. The
    /// token is the raw value, without the `Bearer ` prefix.
    Bearer(String),
    /// Send `x-api-key: <key>` on every upstream request.
    ApiKey(String),
}

// Hand-written so the gateway-held credential never lands in logs via `{:?}`.
// This crate is the credential holder; a derived `Debug` would print the token
// in plaintext the moment any caller logs an upstream.
impl std::fmt::Debug for A2aAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            A2aAuth::None => f.write_str("None"),
            A2aAuth::Bearer(_) => f.write_str("Bearer(***redacted***)"),
            A2aAuth::ApiKey(_) => f.write_str("ApiKey(***redacted***)"),
        }
    }
}

/// Connection parameters for a single upstream A2A agent.
#[derive(Clone)]
pub struct A2aUpstream {
    /// The agent's A2A service endpoint, where JSON-RPC requests are sent, e.g.
    /// `https://agents.example.com/a2a`. The agent card is discovered at the
    /// well-known path relative to this URL's origin.
    pub url: String,
    /// Upstream authentication, held gateway-side.
    pub auth: A2aAuth,
    /// Per-operation deadline. Defaults to [`DEFAULT_UPSTREAM_TIMEOUT`].
    pub timeout: Duration,
}

// Manual so a `Bearer` token cannot leak through `A2aUpstream`'s `Debug`
// (delegates to the redacting `A2aAuth` impl above).
impl std::fmt::Debug for A2aUpstream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2aUpstream")
            .field("url", &self.url)
            .field("auth", &self.auth)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Build an [`A2aUpstream`] from a registered [`A2aAgent`] resource.
pub fn upstream_from_a2a_agent(agent: &A2aAgent) -> A2aUpstream {
    let secret = agent.secret.clone().unwrap_or_default();
    let auth = match agent.auth_type {
        A2aAuthType::None => A2aAuth::None,
        A2aAuthType::Bearer => A2aAuth::Bearer(secret),
        A2aAuthType::ApiKey => A2aAuth::ApiKey(secret),
    };
    let timeout = agent
        .timeout_ms
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_UPSTREAM_TIMEOUT);
    A2aUpstream {
        url: agent.url.clone(),
        auth,
        timeout,
    }
}

/// An upstream agent's card, as fetched from its well-known URI.
///
/// Only the fields the gateway acts on are named; every other field (skills,
/// capabilities, version, security schemes, …) is preserved in [`Self::rest`]
/// so the card can be re-serialized losslessly when the `/a2a` endpoint rewrites
/// the `url` to point at the gateway.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentCard {
    /// The agent's advertised name.
    pub name: String,
    /// The A2A service endpoint the agent advertises for itself.
    pub url: String,
    /// Every other agent-card field, preserved verbatim for lossless round-trip.
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

/// A governed client tunnel to a single upstream A2A agent.
#[async_trait]
pub trait A2aBridge: Send + Sync {
    /// Fetch and parse the upstream agent's card from its well-known URI.
    async fn fetch_agent_card(&self) -> Result<AgentCard, A2aError>;

    /// Forward a JSON-RPC 2.0 request (such as `message/send`) to the upstream
    /// service endpoint and return its JSON-RPC response verbatim.
    async fn send(&self, request: &serde_json::Value) -> Result<serde_json::Value, A2aError>;
}

/// The default [`A2aBridge`], built on the workspace HTTP client.
#[derive(Debug)]
pub struct HttpBridge {
    upstream: A2aUpstream,
    client: reqwest::Client,
}

impl HttpBridge {
    /// Build a bridge for one upstream agent. Reuses the shared, redirect-free
    /// HTTP client (see [`shared_client`]); the per-agent timeout is applied
    /// per-request, so a shared client does not lose the per-agent deadline.
    pub fn new(upstream: A2aUpstream) -> Self {
        Self {
            upstream,
            client: shared_client(),
        }
    }

    /// Apply the gateway-held upstream credential to an outgoing request.
    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.upstream.auth {
            A2aAuth::None => req,
            A2aAuth::Bearer(token) => req.bearer_auth(token),
            A2aAuth::ApiKey(key) => req.header(API_KEY_HEADER, key),
        }
    }

    /// Resolve the agent-card well-known URI from the service endpoint's origin
    /// (RFC 8615): scheme + host + port, with the well-known path.
    fn agent_card_url(&self) -> Result<reqwest::Url, A2aError> {
        let mut url = reqwest::Url::parse(&self.upstream.url)
            .map_err(|e| A2aError::Connect(format!("invalid upstream url: {e}")))?;
        url.set_path(AGENT_CARD_PATH);
        url.set_query(None);
        Ok(url)
    }
}

#[async_trait]
impl A2aBridge for HttpBridge {
    async fn fetch_agent_card(&self) -> Result<AgentCard, A2aError> {
        let url = self.agent_card_url()?;
        let resp = self
            .apply_auth(self.client.get(url).timeout(self.upstream.timeout))
            .send()
            .await
            .map_err(|e| A2aError::Connect(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(A2aError::Connect(format!(
                "agent card fetch returned HTTP {}",
                resp.status().as_u16()
            )));
        }
        let bytes = read_capped(resp).await?;
        serde_json::from_slice::<AgentCard>(&bytes)
            .map_err(|e| A2aError::Request(format!("malformed agent card: {e}")))
    }

    async fn send(&self, request: &serde_json::Value) -> Result<serde_json::Value, A2aError> {
        let resp = self
            .apply_auth(
                self.client
                    .post(&self.upstream.url)
                    .timeout(self.upstream.timeout)
                    .json(request),
            )
            .send()
            .await
            .map_err(|e| A2aError::Connect(e.to_string()))?;
        if !resp.status().is_success() {
            // Surface the upstream STATUS only — never proxy the upstream error
            // body verbatim to the caller, which could leak upstream internals.
            return Err(A2aError::Request(format!(
                "upstream returned HTTP {}",
                resp.status().as_u16()
            )));
        }
        let bytes = read_capped(resp).await?;
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .map_err(|e| A2aError::Request(format!("malformed JSON-RPC response: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(auth_type: &str) -> A2aAgent {
        serde_json::from_str(&format!(
            r#"{{"display_name":"a","url":"https://x/a2a","auth_type":"{auth_type}","secret":"s"}}"#
        ))
        .unwrap()
    }

    #[test]
    fn upstream_maps_none_bearer_api_key() {
        let mut none = agent("none");
        none.secret = None;
        assert!(matches!(upstream_from_a2a_agent(&none).auth, A2aAuth::None));
        assert!(matches!(
            upstream_from_a2a_agent(&agent("bearer")).auth,
            A2aAuth::Bearer(_)
        ));
        assert!(matches!(
            upstream_from_a2a_agent(&agent("api_key")).auth,
            A2aAuth::ApiKey(_)
        ));
    }

    #[test]
    fn upstream_honours_timeout_ms() {
        let mut a = agent("none");
        a.timeout_ms = Some(1234);
        assert_eq!(
            upstream_from_a2a_agent(&a).timeout,
            Duration::from_millis(1234)
        );
        assert_eq!(
            upstream_from_a2a_agent(&agent("none")).timeout,
            DEFAULT_UPSTREAM_TIMEOUT
        );
    }

    #[test]
    fn auth_debug_redacts_credentials() {
        assert_eq!(
            format!("{:?}", A2aAuth::Bearer("tok".into())),
            "Bearer(***redacted***)"
        );
        assert_eq!(
            format!("{:?}", A2aAuth::ApiKey("k".into())),
            "ApiKey(***redacted***)"
        );
        // A bearer token must not leak through the upstream's Debug either.
        let up = A2aUpstream {
            url: "https://x/a2a".into(),
            auth: A2aAuth::Bearer("super-secret".into()),
            timeout: DEFAULT_UPSTREAM_TIMEOUT,
        };
        assert!(!format!("{up:?}").contains("super-secret"));
    }

    #[test]
    fn agent_card_url_is_origin_well_known() {
        let bridge = HttpBridge::new(A2aUpstream {
            url: "https://agents.example.com/a2a/v1".into(),
            auth: A2aAuth::None,
            timeout: DEFAULT_UPSTREAM_TIMEOUT,
        });
        assert_eq!(
            bridge.agent_card_url().unwrap().as_str(),
            "https://agents.example.com/.well-known/agent-card.json"
        );
    }

    #[test]
    fn agent_card_round_trips_unknown_fields() {
        let card: AgentCard = serde_json::from_str(
            r#"{"name":"Contract Reviewer","url":"https://x/a2a","version":"2.1.0","skills":[{"id":"s1"}]}"#,
        )
        .unwrap();
        assert_eq!(card.name, "Contract Reviewer");
        let back = serde_json::to_value(&card).unwrap();
        assert_eq!(back["version"], "2.1.0");
        assert_eq!(back["skills"][0]["id"], "s1");
    }
}
