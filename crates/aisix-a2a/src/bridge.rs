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
//! - Agent card discovery: the RFC 8615 well-known URI
//!   `https://{domain}/.well-known/agent-card.json`. The spec resolves it at the
//!   origin, but real deployments also publish it under the agent's own path
//!   prefix, so both bases are tried — see [`HttpBridge::agent_card_urls`].
//!   <https://a2a-protocol.org/latest/topics/agent-discovery/>
//! - Every request announces the wire version it speaks in the `A2A-Version`
//!   header; an agent that receives none must assume `0.3`. The version is
//!   pinned per agent on the `A2aAgent` resource.
//! - `message/send` is a JSON-RPC 2.0 method whose envelope differs between the
//!   A2A 0.3 and 1.0 wire formats. This bridge forwards the caller's request
//!   verbatim and does not translate between versions, so the method name and
//!   body shape are the caller's concern, not this layer's.
//!   <https://a2a-protocol.org/latest/topics/life-of-a-task/>

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use aisix_core::{A2aAgent, A2aAuthType, A2aProtocolVersion};
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

/// Header naming the A2A wire version the client speaks. The spec requires a
/// client to send it on every request and requires an agent to read an absent
/// value as `0.3`, so an unlabelled call to a 1.0 agent is not merely untidy —
/// the agent answers `VersionNotSupportedError` and the call never lands.
const VERSION_HEADER: &str = "A2A-Version";

/// Well-known agent-card paths, current spec first. RFC 8615 resolves a
/// well-known URI at the origin, but platforms that multiplex tenants under a
/// path prefix publish the card relative to the agent's own path instead, so
/// each of these is tried against both bases — see [`HttpBridge::agent_card_urls`].
const AGENT_CARD_PATHS: [&str; 2] = ["/.well-known/agent-card.json", "/.well-known/agent.json"];

/// Content type of an A2A streaming response, asked for on every streaming call.
const SSE_CONTENT_TYPE: &str = "text/event-stream";

/// Hard cap on a SINGLE buffered SSE event. A stream has no total size — a task
/// may push updates for hours — so the cap is per event: it bounds how much an
/// upstream can accumulate without ever emitting a newline, which is the only
/// way a streaming reader can be made to grow without limit.
const MAX_SSE_EVENT_BYTES: usize = 16 * 1024 * 1024;

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
            aisix_gateway::client_builder()
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
    /// well-known paths relative to this URL, then to its origin.
    pub url: String,
    /// Upstream authentication, held gateway-side.
    pub auth: A2aAuth,
    /// The wire version this agent speaks, announced to it on every request in
    /// the `A2A-Version` header.
    pub protocol_version: A2aProtocolVersion,
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
            .field("protocol_version", &self.protocol_version)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Warn — once per (agent, url) per process — when a credentialed agent is
/// reached over cleartext HTTP: the gateway-held secret travels unencrypted
/// on every call. Same posture as the MCP registry's warning (#879): a
/// warning, never a rejection, and no request-behavior change. Deduped
/// because the upstream is rebuilt per request. The scheme check mirrors
/// the URL parser (lowercased scheme, leading whitespace stripped).
fn warn_cleartext_credential(agent: &A2aAgent) {
    use std::sync::{Mutex, OnceLock};
    static WARNED: OnceLock<Mutex<std::collections::HashSet<(String, String)>>> = OnceLock::new();
    if agent.auth_type == A2aAuthType::None {
        return;
    }
    let cleartext = agent
        .url
        .trim_start()
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"));
    if !cleartext {
        return;
    }
    let mut warned = WARNED
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if warned.insert((agent.name.clone(), agent.url.clone())) {
        tracing::warn!(
            agent = %agent.name,
            url = %agent.url,
            "the gateway-held A2A agent credential is sent over cleartext http; anyone \
             on the network path can read it — serve this agent over https"
        );
    }
}

/// Build an [`A2aUpstream`] from a registered [`A2aAgent`] resource.
pub fn upstream_from_a2a_agent(agent: &A2aAgent) -> A2aUpstream {
    warn_cleartext_credential(agent);
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
        protocol_version: agent.protocol_version,
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

/// One event read off an upstream A2A stream: the JSON-RPC envelope carried in
/// a single SSE `data:` field, verbatim.
pub type A2aEvent = Result<serde_json::Value, A2aError>;

/// A stream of upstream A2A events.
pub type A2aEventStream = futures::stream::BoxStream<'static, A2aEvent>;

/// A governed client tunnel to a single upstream A2A agent.
#[async_trait]
pub trait A2aBridge: Send + Sync {
    /// Fetch and parse the upstream agent's card from its well-known URI.
    async fn fetch_agent_card(&self) -> Result<AgentCard, A2aError>;

    /// Forward a JSON-RPC 2.0 request (such as `message/send`) to the upstream
    /// service endpoint and return its JSON-RPC response verbatim.
    async fn send(&self, request: &serde_json::Value) -> Result<serde_json::Value, A2aError>;

    /// Open a streaming JSON-RPC call (`message/stream`, `tasks/resubscribe`)
    /// and yield each event the upstream pushes, as it arrives.
    ///
    /// The returned stream resolves only once the upstream has accepted the
    /// call, so a refusal surfaces as an error here rather than as a stream
    /// that opens and immediately dies.
    async fn send_stream(&self, request: &serde_json::Value) -> Result<A2aEventStream, A2aError>;
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

    /// Apply the gateway-held upstream credential and announce the wire version
    /// this agent is pinned to. Both belong on every outgoing request: the
    /// credential because the gateway is the one holding it, the version
    /// because an agent that receives no `A2A-Version` must assume `0.3`.
    fn prepare(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let req = req.header(VERSION_HEADER, self.upstream.protocol_version.as_wire_str());
        match &self.upstream.auth {
            A2aAuth::None => req,
            A2aAuth::Bearer(token) => req.bearer_auth(token),
            A2aAuth::ApiKey(key) => req.header(API_KEY_HEADER, key),
        }
    }

    /// Candidate agent-card URIs, most specific first: each well-known path
    /// resolved against the registered URL's own path, then against its origin.
    ///
    /// Both bases are real deployments. An agent that owns its domain publishes
    /// at the RFC 8615 origin URI — the only shape this bridge used to try, and
    /// what every agent registered before now still serves. A platform that
    /// multiplexes tenants under a path prefix (or a self-hosted agent behind an
    /// ingress path) publishes relative to its own path instead — the origin
    /// there belongs to the platform, not to any one agent. Resolving only one
    /// of the two locks out the other, so both are tried, specific first.
    ///
    /// A registered URL that is already at the origin yields the origin
    /// candidates alone — the two bases coincide and probing twice is waste.
    fn agent_card_urls(&self) -> Result<Vec<reqwest::Url>, A2aError> {
        let base = reqwest::Url::parse(&self.upstream.url)
            .map_err(|e| A2aError::Connect(format!("invalid upstream url: {e}")))?;
        let prefix = base.path().trim_end_matches('/').to_string();
        let mut urls = Vec::with_capacity(AGENT_CARD_PATHS.len() * 2);
        for path in AGENT_CARD_PATHS {
            if !prefix.is_empty() {
                let mut relative = base.clone();
                relative.set_path(&format!("{prefix}{path}"));
                relative.set_query(None);
                urls.push(relative);
            }
            let mut origin = base.clone();
            origin.set_path(path);
            origin.set_query(None);
            urls.push(origin);
        }
        Ok(urls)
    }

    /// Fetch and parse the card at exactly one candidate URI, within whatever
    /// is left of the fetch's overall budget.
    async fn fetch_card_at(
        &self,
        url: reqwest::Url,
        budget: Duration,
    ) -> Result<AgentCard, A2aError> {
        let resp = self
            .prepare(self.client.get(url).timeout(budget))
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
}

#[async_trait]
impl A2aBridge for HttpBridge {
    async fn fetch_agent_card(&self) -> Result<AgentCard, A2aError> {
        // Any failure moves to the next candidate, not just a 404: an upstream
        // that does not route the URI answers with whatever its catch-all
        // returns (the 405 in #913), and a prefix that happens to resolve can
        // still hand back something that is not a card.
        //
        // `timeout_ms` bounds the card fetch as ONE upstream operation, so the
        // whole walk shares a single deadline instead of handing each candidate
        // a fresh one. Otherwise a slow or hung upstream stretches the fetch to
        // `candidates × timeout_ms` — four times what the operator configured —
        // and pins a gateway request for the duration.
        let deadline = Instant::now() + self.upstream.timeout;
        let mut last_err = None;
        for url in self.agent_card_urls()? {
            let budget = deadline.saturating_duration_since(Instant::now());
            if budget.is_zero() {
                tracing::debug!(
                    agent_url = %self.upstream.url,
                    "A2A agent card fetch exhausted its deadline before trying every candidate"
                );
                break;
            }
            match self.fetch_card_at(url.clone(), budget).await {
                Ok(card) => return Ok(card),
                Err(err) => {
                    tracing::debug!(%url, error = %err, "A2A agent card candidate did not answer");
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| {
            A2aError::Connect("agent card fetch exceeded its timeout".to_string())
        }))
    }

    async fn send(&self, request: &serde_json::Value) -> Result<serde_json::Value, A2aError> {
        let resp = self
            .prepare(
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

    async fn send_stream(&self, request: &serde_json::Value) -> Result<A2aEventStream, A2aError> {
        // No per-request deadline here, deliberately. `timeout_ms` bounds one
        // upstream operation, and a stream is not one: an A2A task legitimately
        // runs for minutes or hours, pushing status updates the whole time.
        // Applying the unary deadline would cut every such task off at 30s.
        // Downstream liveness is the SSE keep-alive's job, not a hard cap's.
        let resp = self
            .prepare(
                self.client
                    .post(&self.upstream.url)
                    .header(reqwest::header::ACCEPT, SSE_CONTENT_TYPE)
                    .json(request),
            )
            .send()
            .await
            .map_err(|e| A2aError::Connect(e.to_string()))?;
        if !resp.status().is_success() {
            // Status only — never proxy the upstream's error body, same as
            // `send`.
            return Err(A2aError::Request(format!(
                "upstream returned HTTP {}",
                resp.status().as_u16()
            )));
        }
        Ok(Box::pin(sse_events(resp)))
    }
}

/// Parse an upstream SSE body into the JSON-RPC envelope of each `data:` field.
///
/// Deliberately minimal: A2A carries one JSON-RPC envelope per `data:` line, so
/// `event:` / `id:` / `retry:` fields and comments are metadata this gateway has
/// no use for and passes over. A `data:` line that is not JSON ends the stream
/// with an error rather than being skipped — a caller that silently dropped
/// events would report a truncated task as a complete one.
fn sse_events(resp: reqwest::Response) -> impl futures::Stream<Item = A2aEvent> + Send {
    async_stream::stream! {
        let mut bytes = resp.bytes_stream();
        let mut pending: Vec<u8> = Vec::new();
        loop {
            let chunk = match bytes.next().await {
                Some(Ok(chunk)) => chunk,
                Some(Err(e)) => {
                    yield Err(A2aError::Request(e.to_string()));
                    return;
                }
                None => break,
            };
            pending.extend_from_slice(&chunk);
            // A single event is bounded even though the stream is not: an
            // upstream that never emits a newline must not grow this buffer
            // without limit.
            if pending.len() > MAX_SSE_EVENT_BYTES {
                yield Err(A2aError::Request(
                    "upstream SSE event exceeded size cap".to_string(),
                ));
                return;
            }
            while let Some(newline) = pending.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = pending.drain(..=newline).collect();
                match parse_sse_data_line(&line) {
                    Ok(Some(event)) => yield Ok(event),
                    Ok(None) => {}
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
        }
        // A body that ends without its final newline still carries an event.
        if let Ok(Some(event)) = parse_sse_data_line(&pending) {
            yield Ok(event);
        }
    }
}

/// Extract the JSON-RPC envelope from one SSE line, or `None` when the line
/// carries no `data:` field.
fn parse_sse_data_line(line: &[u8]) -> Result<Option<serde_json::Value>, A2aError> {
    let text = std::str::from_utf8(line)
        .map_err(|_| A2aError::Request("upstream SSE event was not valid UTF-8".to_string()))?;
    let Some(payload) = text.trim_end_matches(['\r', '\n']).strip_prefix("data:") else {
        return Ok(None);
    };
    let payload = payload.trim();
    if payload.is_empty() {
        return Ok(None);
    }
    serde_json::from_str(payload)
        .map(Some)
        .map_err(|e| A2aError::Request(format!("malformed JSON-RPC event: {e}")))
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
            protocol_version: A2aProtocolVersion::V1_0,
            timeout: DEFAULT_UPSTREAM_TIMEOUT,
        };
        assert!(!format!("{up:?}").contains("super-secret"));
    }

    fn bridge_at(url: &str) -> HttpBridge {
        HttpBridge::new(A2aUpstream {
            url: url.into(),
            auth: A2aAuth::None,
            protocol_version: A2aProtocolVersion::V1_0,
            timeout: DEFAULT_UPSTREAM_TIMEOUT,
        })
    }

    fn card_candidates(url: &str) -> Vec<String> {
        bridge_at(url)
            .agent_card_urls()
            .unwrap()
            .iter()
            .map(|u| u.as_str().to_string())
            .collect()
    }

    #[test]
    fn card_candidates_try_the_registered_path_before_the_origin() {
        // #913: `set_path` used to replace the path outright, so a path-hosted
        // agent was asked for a card URI it never publishes. The prefix now
        // survives, and the origin URI stays as the fallback so agents
        // registered under the old behaviour keep resolving.
        assert_eq!(
            card_candidates("https://agents.example.com/v3/a2a/serve/abc"),
            vec![
                "https://agents.example.com/v3/a2a/serve/abc/.well-known/agent-card.json",
                "https://agents.example.com/.well-known/agent-card.json",
                "https://agents.example.com/v3/a2a/serve/abc/.well-known/agent.json",
                "https://agents.example.com/.well-known/agent.json",
            ]
        );
    }

    #[test]
    fn card_candidates_collapse_when_the_agent_owns_the_origin() {
        // The two bases coincide here, so the relative candidate would be a
        // byte-identical second request. A trailing slash is the same case.
        let expected = vec![
            "https://agents.example.com/.well-known/agent-card.json",
            "https://agents.example.com/.well-known/agent.json",
        ];
        assert_eq!(card_candidates("https://agents.example.com"), expected);
        assert_eq!(card_candidates("https://agents.example.com/"), expected);
    }

    #[test]
    fn card_candidates_drop_the_query_and_keep_the_port() {
        assert_eq!(
            card_candidates("http://127.0.0.1:8080/a2a?tenant=acme"),
            vec![
                "http://127.0.0.1:8080/a2a/.well-known/agent-card.json",
                "http://127.0.0.1:8080/.well-known/agent-card.json",
                "http://127.0.0.1:8080/a2a/.well-known/agent.json",
                "http://127.0.0.1:8080/.well-known/agent.json",
            ]
        );
    }

    #[test]
    fn sse_lines_yield_only_data_payloads() {
        let event = |line: &str| parse_sse_data_line(line.as_bytes()).unwrap();

        assert_eq!(
            event("data: {\"jsonrpc\":\"2.0\",\"id\":1}\n").unwrap()["id"],
            1
        );
        // No space after the colon is equally valid SSE.
        assert_eq!(event("data:{\"id\":2}\n").unwrap()["id"], 2);
        assert_eq!(event("data: {\"id\":3}\r\n").unwrap()["id"], 3);

        // Framing the gateway has no use for.
        assert!(event("event: task-update\n").is_none());
        assert!(event(": keep-alive comment\n").is_none());
        assert!(event("id: 42\n").is_none());
        assert!(event("retry: 1000\n").is_none());
        assert!(event("\n").is_none());
        assert!(event("data:\n").is_none());
    }

    #[test]
    fn a_data_line_that_is_not_json_is_an_error_not_a_skip() {
        // Silently dropping it would let a truncated task read as a complete
        // one, which is worse than failing the stream.
        let err = parse_sse_data_line(b"data: not-json\n").unwrap_err();
        assert!(
            matches!(err, A2aError::Request(ref m) if m.contains("malformed JSON-RPC event")),
            "got {err:?}"
        );
        assert!(parse_sse_data_line(b"data: \xff\xfe\n").is_err());
    }

    #[test]
    fn upstream_carries_the_pinned_protocol_version() {
        let mut pinned_03 = agent("none");
        pinned_03.protocol_version = A2aProtocolVersion::V0_3;
        assert_eq!(
            upstream_from_a2a_agent(&pinned_03).protocol_version,
            A2aProtocolVersion::V0_3
        );
        assert_eq!(A2aProtocolVersion::V1_0.as_wire_str(), "1.0");
        assert_eq!(A2aProtocolVersion::V0_3.as_wire_str(), "0.3");
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
