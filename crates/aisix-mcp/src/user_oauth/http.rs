//! The HTTP client for everything the per-user OAuth flow fetches:
//! metadata documents, dynamic client registration and token requests.
//!
//! Every URL goes through `outbound.ssrf_protection`, whether an operator
//! configured it or an upstream document named it. Redirects are never
//! followed: a token request carries a client secret or a refresh token, and
//! a redirect target would not have been checked.

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

/// Send `build(client, url)`, bounded by `timeout`, on the client the SSRF
/// settings choose for `url`. The caller decides how to read the response;
/// this returns only whether one arrived.
pub(super) async fn send(
    server: &str,
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
    let route = match ssrf::settings().route(&parsed).await {
        Ok(route) => route,
        Err(Refusal::Unresolvable) => return Err(HttpFailure::Transport),
        Err(_) => return Err(blocked(server, &parsed)),
    };
    let client = client(route)?;
    match build(client, parsed.clone()).timeout(timeout).send().await {
        Ok(response) => Ok(response),
        Err(error) => match Refusal::from_error(&error) {
            Some(Refusal::Blocked(_)) => Err(blocked(server, &parsed)),
            _ => Err(HttpFailure::Transport),
        },
    }
}
