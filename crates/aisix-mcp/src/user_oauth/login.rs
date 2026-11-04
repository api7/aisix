//! The gateway as an OIDC relying party of a trust provider: where the
//! provider's endpoints are, and the code exchange. What the ID token then
//! means is `aisix-proxy`'s to decide, alongside every other JWT. The provider's
//! `issuer` is the operator's, and so is what its own discovery document names:
//! the same trust a JWKS fetch runs on.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::flow::{token_request_named, TokenOutcome};
use super::http::{send, Gate, HttpFailure};
use super::UserOAuthManager;
use crate::error::UserOAuthFailure;

/// Short, because nothing tells a gateway when a provider moves its endpoints.
const DISCOVERY_TTL: Duration = Duration::from_secs(300);
const LOGIN_HTTP_TIMEOUT: Duration = Duration::from_secs(10);

/// A trust provider's OAuth endpoints, from its OIDC discovery document.
#[derive(Clone, Debug, Deserialize)]
pub struct LoginEndpoints {
    pub authorization_endpoint: String,
    pub token_endpoint: String,
}

#[derive(Default)]
pub(super) struct LoginDiscoveryCache {
    entries: Mutex<HashMap<String, (Instant, LoginEndpoints)>>,
}

/// What a provider's token endpoint said to a sign-in's code.
pub enum LoginExchange {
    Issued {
        id_token: String,
        /// The granted scope when it said. RFC 6749 §5.1: absent means what
        /// was requested.
        scope: Option<String>,
    },
    /// A refusal, with its RFC 6749 §5.2 code (empty when it gave none).
    Rejected(String),
    /// It answered with tokens but no ID token.
    NoIdToken,
    Failed(UserOAuthFailure),
}

/// A sign-in's code and what proves it, for [`UserOAuthManager::exchange_login_code`].
pub struct LoginCode<'a> {
    pub provider: &'a str,
    pub token_endpoint: &'a str,
    pub client_id: &'a str,
    pub client_secret: Option<&'a str>,
    pub code: &'a str,
    pub redirect_uri: &'a str,
    pub code_verifier: &'a str,
}

impl UserOAuthManager {
    /// `provider`'s endpoints, from `issuer`'s discovery document. A document
    /// that cannot be fetched falls back on the last one read.
    pub async fn login_endpoints(
        &self,
        provider: &str,
        issuer: &str,
    ) -> Result<LoginEndpoints, UserOAuthFailure> {
        let stale = {
            let entries = self
                .login_discovery
                .entries
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            match entries.get(issuer) {
                Some((at, endpoints)) if at.elapsed() < DISCOVERY_TTL => {
                    return Ok(endpoints.clone())
                }
                Some((_, endpoints)) => Some(endpoints.clone()),
                None => None,
            }
        };
        match fetch_endpoints(provider, issuer).await {
            Ok(endpoints) => {
                self.login_discovery
                    .entries
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .insert(issuer.to_owned(), (Instant::now(), endpoints.clone()));
                Ok(endpoints)
            }
            Err(failure) => stale.ok_or(failure),
        }
    }

    /// Exchange a sign-in's code at the provider, with `client_secret_post`
    /// when there is a secret.
    pub async fn exchange_login_code(&self, code: &LoginCode<'_>) -> LoginExchange {
        let mut form = vec![
            ("grant_type", "authorization_code".to_owned()),
            ("code", code.code.to_owned()),
            ("redirect_uri", code.redirect_uri.to_owned()),
            ("client_id", code.client_id.to_owned()),
            ("code_verifier", code.code_verifier.to_owned()),
        ];
        if let Some(secret) = code.client_secret {
            form.push(("client_secret", secret.to_owned()));
        }
        match token_request_named(
            code.provider,
            Gate::Configured,
            code.token_endpoint,
            &form,
            LOGIN_HTTP_TIMEOUT,
        )
        .await
        {
            TokenOutcome::Issued(issued) => match issued.id_token {
                Some(id_token) => LoginExchange::Issued {
                    id_token,
                    scope: issued.scope,
                },
                None => LoginExchange::NoIdToken,
            },
            TokenOutcome::Rejected(error) => LoginExchange::Rejected(error),
            TokenOutcome::Transient => LoginExchange::Failed(UserOAuthFailure::Unavailable),
            TokenOutcome::Blocked => LoginExchange::Failed(UserOAuthFailure::Misconfigured),
        }
    }
}

#[derive(Deserialize)]
struct DiscoveryDocument {
    #[serde(default)]
    issuer: Option<String>,
    #[serde(flatten)]
    endpoints: Option<LoginEndpoints>,
}

async fn fetch_endpoints(provider: &str, issuer: &str) -> Result<LoginEndpoints, UserOAuthFailure> {
    let url = format!(
        "{}/.well-known/openid-configuration",
        issuer.trim_end_matches('/')
    );
    let response = send(
        provider,
        Gate::Configured,
        &url,
        LOGIN_HTTP_TIMEOUT,
        |client, url| client.get(url).header("accept", "application/json"),
    )
    .await
    .map_err(|failure| match failure {
        HttpFailure::Transport => UserOAuthFailure::Unavailable,
        HttpFailure::Blocked => UserOAuthFailure::Misconfigured,
    })?;
    let status = response.status();
    if !status.is_success() {
        tracing::warn!(target: "aisix::mcp_oauth", provider, %status,
            "could not read the trust provider's OIDC discovery document");
        return Err(UserOAuthFailure::Unavailable);
    }
    let document: DiscoveryDocument = response.json().await.map_err(|e| {
        aisix_gateway::log_body_read_causes(&e);
        UserOAuthFailure::Unavailable
    })?;
    // OIDC Discovery 1.0 §4.3: the document must claim the issuer asked about.
    if document.issuer.as_deref() != Some(issuer) {
        tracing::warn!(target: "aisix::mcp_oauth", provider,
            "the trust provider's discovery document names another issuer");
        return Err(UserOAuthFailure::Misconfigured);
    }
    document.endpoints.ok_or_else(|| {
        tracing::warn!(target: "aisix::mcp_oauth", provider,
            "the trust provider's discovery document names no authorization or token endpoint");
        UserOAuthFailure::Misconfigured
    })
}
