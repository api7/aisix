//! The HTTP client for everything the per-user OAuth flow fetches:
//! metadata documents, dynamic client registration and token requests.
//!
//! `outbound.ssrf_protection` covers only an address a third party chose: an
//! endpoint or document an upstream's metadata names on another authority than
//! the server's own `url`. What an operator configured, and what that server
//! publishes on its own authority, is reached like a JWKS: see [`Gate`]. Redirects
//! are never followed on either route: a token request carries a client secret or
//! a refresh token, and nothing about a redirect target has been decided.

use std::sync::OnceLock;
use std::time::Duration;

use aisix_gateway::ssrf::{self, Refusal, Route};
use reqwest::Url;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum HttpFailure {
    /// The request could not complete; trying again later may work.
    Transport,
    /// `outbound.ssrf_protection` refuses this URL, or it is not a URL. Only
    /// an operator can change the outcome.
    Blocked,
}

/// Whether `outbound.ssrf_protection` decides if a URL may be reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Gate {
    /// An operator chose it, directly or by naming the server that serves it.
    Configured,
    /// An upstream document chose it.
    Ssrf,
}

fn client(route: Route) -> Result<&'static reqwest::Client, HttpFailure> {
    static PINNED: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    static DIRECT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    static PROXIED: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    let cell = match route {
        Route::Pinned => &PINNED,
        Route::Direct => &DIRECT,
        Route::Proxied => &PROXIED,
    };
    cell.get_or_init(|| {
        let b = aisix_gateway::upstream_http::caller_chosen_peer_client_builder()
            .redirect(reqwest::redirect::Policy::none());
        let b = match route {
            Route::Pinned => b.dns_resolver(ssrf::direct_resolver(true)).no_proxy(),
            Route::Direct => b.dns_resolver(ssrf::direct_resolver(false)).no_proxy(),
            Route::Proxied => b,
        };
        b.build().map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|_| HttpFailure::Transport)
}

/// The client for what an operator chose, built as the JWKS one is.
fn configured_client() -> Result<&'static reqwest::Client, HttpFailure> {
    static CONFIGURED: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CONFIGURED
        .get_or_init(|| {
            aisix_gateway::client_builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|_| HttpFailure::Transport)
}

fn blocked(server: &str, url: &Url) -> HttpFailure {
    tracing::warn!(
        target: "aisix::mcp_oauth",
        server,
        url = %url,
        hint = %Refusal::blocked_hint(Some(url)),
        "outbound.ssrf_protection refused an OAuth endpoint of an MCP server",
    );
    HttpFailure::Blocked
}

/// Send `build(client, url)`, bounded by `timeout`, on the client `gate` and
/// the SSRF settings choose for `url`. The caller decides how to read the response;
/// this returns only whether one arrived.
pub(super) async fn send(
    server: &str,
    gate: Gate,
    url: &str,
    timeout: Duration,
    build: impl FnOnce(&reqwest::Client, Url) -> reqwest::RequestBuilder,
) -> Result<reqwest::Response, HttpFailure> {
    let Ok(parsed) = Url::parse(url) else {
        return Err(HttpFailure::Blocked);
    };
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(HttpFailure::Blocked);
    }
    let client = match gate {
        Gate::Configured => configured_client()?,
        Gate::Ssrf => match ssrf::settings().route(&parsed).await {
            Ok(route) => client(route)?,
            Err(Refusal::Unresolvable) => return Err(HttpFailure::Transport),
            Err(_) => return Err(blocked(server, &parsed)),
        },
    };
    match build(client, parsed.clone()).timeout(timeout).send().await {
        Ok(response) => Ok(response),
        Err(error) => match Refusal::from_error(&error) {
            Some(Refusal::Blocked(_)) => Err(blocked(server, &parsed)),
            _ => Err(HttpFailure::Transport),
        },
    }
}
