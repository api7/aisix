//! The sign-in flow: from an MCP client's authorization request, through the
//! upstream's redirect back, to tokens stored under the caller's subject.
//!
//! Nothing about a flow in progress is stored: its state travels in an
//! encrypted cookie ([`FlowState`]), so a client backend that requests the
//! authorization URL with no cookie jar before the browser does leaves nothing
//! behind. What the client receives at the end is a code of the gateway's own.
//!
//! When the client's credential says who is calling at the token step (a
//! relayed sign-in), the client's PKCE challenge goes upstream and its verifier
//! follows, so the gateway never holds a secret of the client's. When the
//! identity is already known from a sign-in at a trust provider, the gateway
//! runs its own PKCE upstream and stores the tokens as soon as they arrive.

use std::collections::BTreeMap;
use std::sync::Arc;

use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::discovery::Endpoints;
use super::http::{send, Gate, HttpFailure};
use super::session::SessionIdentity;
use super::{
    digest, random_token, unix_now, RegisteredClient, TokenLabels, TokenRecord, UserOAuthManager,
    UserOAuthServer, UserSubject, CLIENT_TTL, CODE_TTL, NS_CLIENT, NS_CODE, NS_TOKEN, TOKEN_TTL,
};
use crate::error::UserOAuthFailure;
use crate::shared_store::{consume, PutCondition, StoreError};

/// How many times a sign-in re-reads a record that keeps moving under it.
const STORE_TOKEN_ATTEMPTS: usize = 3;

/// How long a flow's cookie lives at each step.
pub const FLOW_TTL_SECS: i64 = 600;

/// A flow in progress, as the browser carries it. Records written here are
/// read by gateways of other versions only for the few minutes a flow lives.
#[derive(Serialize, Deserialize)]
pub struct FlowState {
    pub handle: String,
    /// Unix seconds.
    pub expires_at: i64,
    pub server_id: String,
    pub server_name: String,
    pub client_id: String,
    pub redirect_uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_state: Option<String>,
    pub code_challenge: String,
    /// Proves a consent POST came from the page this flow rendered.
    pub csrf: String,
    pub stage: FlowStage,
}

/// Where a flow has got to. A step accepts only the stage before it, which is
/// what keeps a consent from being given before the sign-in it depends on.
#[derive(Serialize, Deserialize)]
#[serde(tag = "at", rename_all = "snake_case")]
pub enum FlowStage {
    /// The browser is at the trust provider.
    ProviderLogin {
        provider: String,
        verifier: String,
        nonce: String,
    },
    /// The browser is at the upstream authorization server.
    Upstream {
        /// Who the upstream's tokens are for, once a trust provider has said.
        /// Without it the tokens wait for the client's credential at the token
        /// step.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signed_in: Option<SessionIdentity>,
        /// The gateway's own PKCE verifier, when the client's is not used.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        verifier: Option<String>,
        leg: UpstreamLeg,
    },
    /// The user is deciding.
    Consent {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signed_in: Option<SessionIdentity>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        relayed: Option<RelayedCode>,
    },
}

/// What the callback needs to remember of an upstream authorization request.
#[derive(Serialize, Deserialize)]
pub struct UpstreamLeg {
    upstream_redirect_uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resource: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    issuer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    registered_client: Option<RegisteredClient>,
}

impl UpstreamLeg {
    /// Whether `iss` from the authorization response (RFC 9207) names the
    /// authorization server this flow went to.
    pub fn issuer_matches(&self, iss: &str) -> bool {
        self.issuer
            .as_deref()
            .is_none_or(|issuer| issuer.trim_end_matches('/') == iss.trim_end_matches('/'))
    }

    /// The upstream's `code`, kept for the client's token request.
    pub fn relay(self, upstream_code: &str) -> RelayedCode {
        RelayedCode {
            upstream_code: upstream_code.to_owned(),
            leg: self,
        }
    }
}

/// An upstream code waiting for the client's credential at the token step.
#[derive(Serialize, Deserialize)]
pub struct RelayedCode {
    upstream_code: String,
    leg: UpstreamLeg,
}

/// An unguessable PKCE verifier and its S256 challenge.
pub fn pkce_pair() -> (String, String) {
    let verifier = random_token();
    let challenge = pkce_challenge(&verifier);
    (verifier, challenge)
}

fn pkce_challenge(verifier: &str) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

#[derive(Serialize, Deserialize)]
struct CodeRecord {
    server_id: String,
    client_id: String,
    redirect_uri: String,
    code_challenge: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    relayed: Option<RelayedCode>,
    /// Who the sign-in established, when a trust provider did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    signed_in: Option<SessionIdentity>,
}

/// A client's token request.
pub struct RedeemRequest<'a> {
    pub code: &'a str,
    pub client_id: &'a str,
    pub redirect_uri: &'a str,
    pub code_verifier: &'a str,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RedeemError {
    /// The code, its binding or the verifier is wrong, or the upstream
    /// refused the exchange.
    InvalidGrant,
    Misconfigured,
    Unavailable,
}

/// What a token endpoint said.
pub(super) enum TokenOutcome {
    Issued(IssuedTokens),
    /// A refusal, with its RFC 6749 §5.2 code (empty when it gave none).
    Rejected(String),
    Transient,
    Blocked,
}

pub(super) struct IssuedTokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: Option<i64>,
    pub id_token: Option<String>,
    pub scope: Option<String>,
}

#[derive(Deserialize)]
struct TokenBody {
    #[serde(default)]
    access_token: Option<String>,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<serde_json::Value>,
    #[serde(default)]
    id_token: Option<String>,
    /// Loose like `expires_in`: only a trust provider's is read, and a shape
    /// some other server invents must not cost it the tokens beside it.
    #[serde(default)]
    scope: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<String>,
}

fn expires_at(expires_in: Option<&serde_json::Value>) -> Option<i64> {
    let seconds = match expires_in? {
        serde_json::Value::Number(n) => n.as_i64()?,
        serde_json::Value::String(s) => s.parse().ok()?,
        _ => return None,
    };
    Some(unix_now().saturating_add(seconds))
}

/// POST `form` to a token endpoint, bounded by the server's timeout. Reports
/// by status and error code alone: a token endpoint may reflect parameters
/// into its error text, and this form holds a code or a refresh token.
pub(super) async fn token_request(
    server: &UserOAuthServer,
    token_endpoint: &str,
    form: &[(&str, String)],
    timeout: std::time::Duration,
) -> TokenOutcome {
    let gate = server.gate_for(token_endpoint);
    token_request_named(&server.name, gate, token_endpoint, form, timeout).await
}

/// As [`token_request`], for a party that is not an MCP server.
pub(super) async fn token_request_named(
    name: &str,
    gate: Gate,
    token_endpoint: &str,
    form: &[(&str, String)],
    timeout: std::time::Duration,
) -> TokenOutcome {
    let response = match send(name, gate, token_endpoint, timeout, |client, url| {
        client
            .post(url)
            .header("accept", "application/json")
            .form(form)
    })
    .await
    {
        Ok(response) => response,
        Err(HttpFailure::Transport) => return TokenOutcome::Transient,
        Err(HttpFailure::Blocked) => return TokenOutcome::Blocked,
    };
    let status = response.status();
    if status.is_server_error() || status.as_u16() == 429 || status.is_redirection() {
        return TokenOutcome::Transient;
    }
    let body = response.json::<TokenBody>().await.ok();
    if !status.is_success() {
        let code = body.and_then(|b| b.error).unwrap_or_default();
        return TokenOutcome::Rejected(code);
    }
    let Some(body) = body else {
        return TokenOutcome::Transient;
    };
    if let Some(code) = body.error {
        return TokenOutcome::Rejected(code);
    }
    match body.access_token.filter(|t| !t.is_empty()) {
        Some(access_token) => TokenOutcome::Issued(IssuedTokens {
            access_token,
            refresh_token: body.refresh_token.filter(|t| !t.is_empty()),
            expires_at: expires_at(body.expires_in.as_ref()),
            id_token: body.id_token.filter(|t| !t.is_empty()),
            scope: body
                .scope
                .and_then(|scope| scope.as_str().map(str::to_owned)),
        }),
        None => TokenOutcome::Transient,
    }
}

/// Identify the gateway's OAuth client in a token request: the configured
/// one, else the registered one. A public client sends only its id.
pub(super) fn add_client(
    form: &mut Vec<(&str, String)>,
    server: &UserOAuthServer,
    registered: Option<&RegisteredClient>,
) {
    let (id, secret) = match (&server.client_id, registered) {
        (Some(id), _) => (Some(id.clone()), server.client_secret.clone()),
        (None, Some(client)) => (Some(client.client_id.clone()), client.client_secret.clone()),
        (None, None) => (None, None),
    };
    if let Some(id) = id {
        form.push(("client_id", id));
    }
    if let Some(secret) = secret {
        form.push(("client_secret", secret));
    }
}

fn store_failure(error: StoreError) -> UserOAuthFailure {
    tracing::warn!(target: "aisix::mcp_oauth", %error, "per-user OAuth storage failed");
    UserOAuthFailure::Unavailable
}

#[derive(Serialize)]
struct RegistrationRequest<'a> {
    client_name: &'a str,
    redirect_uris: [&'a str; 1],
    grant_types: [&'a str; 2],
    response_types: [&'a str; 1],
    token_endpoint_auth_method: &'a str,
}

impl UserOAuthManager {
    /// Where to send the user to sign in to `server`, and what the callback
    /// will need. `state` comes back on the redirect and `code_challenge`
    /// goes into the request. `scope` overrides what the server asks for.
    pub async fn begin_upstream(
        &self,
        server: &UserOAuthServer,
        upstream_redirect_uri: &str,
        state: &str,
        code_challenge: &str,
        scope: Option<&str>,
    ) -> Result<(String, UpstreamLeg), UserOAuthFailure> {
        let endpoints = self.discovery.endpoints(server).await?;
        let authorization_endpoint = endpoints
            .authorization
            .clone()
            .ok_or(UserOAuthFailure::Misconfigured)?;
        let mut url = reqwest::Url::parse(&authorization_endpoint)
            .ok()
            .filter(|u| matches!(u.scheme(), "http" | "https"))
            .ok_or(UserOAuthFailure::Misconfigured)?;
        let registered = self
            .registered_client(server, &endpoints, upstream_redirect_uri)
            .await?;
        let upstream_client_id = server
            .client_id
            .clone()
            .or_else(|| registered.as_ref().map(|c| c.client_id.clone()))
            .ok_or(UserOAuthFailure::Misconfigured)?;

        let scope = scope.map(str::to_owned).or_else(|| {
            let scopes = if server.scopes.is_empty() {
                &endpoints.scopes
            } else {
                &server.scopes
            };
            (!scopes.is_empty()).then(|| scopes.join(" "))
        });
        {
            let mut query = url.query_pairs_mut();
            query
                .append_pair("response_type", "code")
                .append_pair("client_id", &upstream_client_id)
                .append_pair("redirect_uri", upstream_redirect_uri)
                .append_pair("state", state)
                .append_pair("code_challenge", code_challenge)
                .append_pair("code_challenge_method", "S256");
            if let Some(scope) = &scope {
                query.append_pair("scope", scope);
            }
            if let Some(resource) = &endpoints.resource {
                query.append_pair("resource", resource);
            }
        }
        Ok((
            url.into(),
            UpstreamLeg {
                upstream_redirect_uri: upstream_redirect_uri.to_owned(),
                resource: endpoints.resource.clone(),
                issuer: endpoints.issuer.clone(),
                registered_client: registered,
            },
        ))
    }

    /// The single-use code the client gets for a flow the user has consented
    /// to.
    pub async fn issue_code(&self, flow: FlowState) -> Result<String, UserOAuthFailure> {
        let FlowStage::Consent { signed_in, relayed } = flow.stage else {
            return Err(UserOAuthFailure::Misconfigured);
        };
        let code = random_token();
        let labels = BTreeMap::from([("mcp_server_id", flow.server_id.as_str())]);
        let record = CodeRecord {
            server_id: flow.server_id.clone(),
            client_id: flow.client_id,
            redirect_uri: flow.redirect_uri,
            code_challenge: flow.code_challenge,
            relayed,
            signed_in,
        };
        self.store
            .put(
                NS_CODE,
                &digest(&[&code]),
                serde_json::to_vec(&record).expect("code record serializes"),
                CODE_TTL,
                &labels,
                PutCondition::CreateOnly,
            )
            .await
            .map_err(store_failure)?;
        Ok(code)
    }

    /// Spend `request`'s code, if it is one and everything bound to it matches.
    async fn take_code(
        &self,
        server: &UserOAuthServer,
        request: &RedeemRequest<'_>,
    ) -> Result<CodeRecord, RedeemError> {
        let value = consume(self.store.as_ref(), NS_CODE, &digest(&[request.code]))
            .await
            .map_err(|error| {
                store_failure(error);
                RedeemError::Unavailable
            })?
            .ok_or(RedeemError::InvalidGrant)?;
        let code: CodeRecord =
            serde_json::from_slice(&value).map_err(|_| RedeemError::InvalidGrant)?;
        if code.server_id != server.id
            || code.client_id != request.client_id
            || code.redirect_uri != request.redirect_uri
            || code.code_challenge != pkce_challenge(request.code_verifier)
        {
            return Err(RedeemError::InvalidGrant);
        }
        Ok(code)
    }

    /// Spend a code issued for a sign-in at a trust provider, returning who it
    /// established.
    pub async fn redeem_session_code(
        &self,
        server: &UserOAuthServer,
        request: &RedeemRequest<'_>,
    ) -> Result<SessionIdentity, RedeemError> {
        self.take_code(server, request)
            .await?
            .signed_in
            .ok_or(RedeemError::InvalidGrant)
    }

    /// Exchange a gateway-issued code upstream and store the tokens under
    /// `subject`.
    pub async fn redeem_code(
        self: &Arc<Self>,
        server: &UserOAuthServer,
        subject: &UserSubject,
        request: &RedeemRequest<'_>,
    ) -> Result<(), RedeemError> {
        let relayed = self
            .take_code(server, request)
            .await?
            .relayed
            .ok_or(RedeemError::InvalidGrant)?;
        self.exchange_upstream_code(
            server,
            subject,
            &relayed.upstream_code,
            relayed.leg,
            request.code_verifier,
        )
        .await
    }

    /// Exchange an upstream code and store the tokens under `subject`.
    pub async fn exchange_upstream_code(
        &self,
        server: &UserOAuthServer,
        subject: &UserSubject,
        upstream_code: &str,
        leg: UpstreamLeg,
        code_verifier: &str,
    ) -> Result<(), RedeemError> {
        let token_endpoint = self
            .discovery
            .token_endpoint(server)
            .await
            .map_err(|failure| match failure {
                UserOAuthFailure::Unavailable => RedeemError::Unavailable,
                _ => RedeemError::Misconfigured,
            })?;
        let mut form = vec![
            ("grant_type", "authorization_code".to_owned()),
            ("code", upstream_code.to_owned()),
            ("redirect_uri", leg.upstream_redirect_uri.clone()),
            ("code_verifier", code_verifier.to_owned()),
        ];
        if let Some(resource) = &leg.resource {
            form.push(("resource", resource.clone()));
        }
        add_client(&mut form, server, leg.registered_client.as_ref());
        let issued = match token_request(server, &token_endpoint, &form, server.timeout).await {
            TokenOutcome::Issued(issued) => issued,
            // The gateway's own registration, which no new sign-in can fix.
            TokenOutcome::Rejected(error) if error == "invalid_client" => {
                tracing::warn!(target: "aisix::mcp_oauth", server = %server.name,
                    "the upstream refused the gateway's OAuth client");
                return Err(RedeemError::Misconfigured);
            }
            TokenOutcome::Rejected(error) => {
                tracing::warn!(target: "aisix::mcp_oauth", server = %server.name, %error,
                    "the upstream refused a sign-in's authorization code");
                return Err(RedeemError::InvalidGrant);
            }
            TokenOutcome::Transient => return Err(RedeemError::Unavailable),
            TokenOutcome::Blocked => return Err(RedeemError::Misconfigured),
        };
        let record = TokenRecord {
            access_token: issued.access_token,
            refresh_token: issued.refresh_token,
            expires_at: issued.expires_at,
            resource: leg.resource,
            registered_client: leg.registered_client,
            redirect_uri: Some(leg.upstream_redirect_uri),
        };
        self.store_signed_in(server, subject, &record).await
    }

    /// A new sign-in replaces whatever the subject had. Every write names the
    /// version it replaces, so this reads first and retries if that moved.
    async fn store_signed_in(
        &self,
        server: &UserOAuthServer,
        subject: &UserSubject,
        record: &TokenRecord,
    ) -> Result<(), RedeemError> {
        let key = super::token_key(server, subject);
        let value = serde_json::to_vec(record).expect("token record serializes");
        let labels = TokenLabels::new(server, subject);
        for _ in 0..STORE_TOKEN_ATTEMPTS {
            let current = self.store.get(NS_TOKEN, &key).await.map_err(|error| {
                store_failure(error);
                RedeemError::Unavailable
            })?;
            let condition = match current {
                Some(current) => PutCondition::IfVersion(current.version),
                None => PutCondition::CreateOnly,
            };
            match self
                .store
                .put(
                    NS_TOKEN,
                    &key,
                    value.clone(),
                    TOKEN_TTL,
                    &labels.map(),
                    condition,
                )
                .await
            {
                Ok(_) => {
                    self.evict(&key);
                    self.remember(&key, record);
                    return Ok(());
                }
                Err(StoreError::PreconditionFailed) => {}
                Err(error) => {
                    store_failure(error);
                    return Err(RedeemError::Unavailable);
                }
            }
        }
        Err(RedeemError::Unavailable)
    }

    /// The client registered for this server with this authorization server
    /// and redirect URI, registering one first if nobody has; `None` when the
    /// server configures its own. The key carries the endpoint and the URI so
    /// that moving either starts over rather than using a client they do not
    /// know.
    async fn registered_client(
        &self,
        server: &UserOAuthServer,
        endpoints: &Endpoints,
        redirect_uri: &str,
    ) -> Result<Option<RegisteredClient>, UserOAuthFailure> {
        if server.client_id.is_some() {
            return Ok(None);
        }
        let registration_endpoint = endpoints
            .registration
            .as_deref()
            .ok_or(UserOAuthFailure::Misconfigured)?;
        let key = digest(&[&server.id, registration_endpoint, redirect_uri]);
        let lock = Arc::clone(
            self.registrations
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .entry(key.clone())
                .or_default(),
        );
        let _registering = lock.lock().await;
        let read = |stored: Option<crate::shared_store::StoredValue>| {
            stored.and_then(|s| serde_json::from_slice::<RegisteredClient>(&s.value).ok())
        };
        let stored = self
            .store
            .get(NS_CLIENT, &key)
            .await
            .map_err(store_failure)?;
        if let Some(client) = read(stored) {
            return Ok(Some(client));
        }
        let client = register(server, registration_endpoint, redirect_uri).await?;
        let labels = BTreeMap::from([("mcp_server_id", server.id.as_str())]);
        match self
            .store
            .put(
                NS_CLIENT,
                &key,
                serde_json::to_vec(&client).expect("client record serializes"),
                CLIENT_TTL,
                &labels,
                PutCondition::CreateOnly,
            )
            .await
        {
            Ok(_) => Ok(Some(client)),
            // Another gateway registered first; its client is the shared one.
            Err(StoreError::PreconditionFailed) => {
                let stored = self
                    .store
                    .get(NS_CLIENT, &key)
                    .await
                    .map_err(store_failure)?;
                read(stored).map(Some).ok_or(UserOAuthFailure::Unavailable)
            }
            Err(error) => Err(store_failure(error)),
        }
    }
}

/// RFC 7591 registration, as a public client.
async fn register(
    server: &UserOAuthServer,
    registration_endpoint: &str,
    redirect_uri: &str,
) -> Result<RegisteredClient, UserOAuthFailure> {
    let request = RegistrationRequest {
        client_name: "AISIX gateway",
        redirect_uris: [redirect_uri],
        grant_types: ["authorization_code", "refresh_token"],
        response_types: ["code"],
        token_endpoint_auth_method: "none",
    };
    let response = send(
        &server.name,
        server.gate_for(registration_endpoint),
        registration_endpoint,
        server.timeout,
        |client, url| client.post(url).json(&request),
    )
    .await
    .map_err(|failure| match failure {
        HttpFailure::Transport => UserOAuthFailure::Unavailable,
        HttpFailure::Blocked => UserOAuthFailure::Misconfigured,
    })?;
    let status = response.status();
    if status.is_server_error() {
        return Err(UserOAuthFailure::Unavailable);
    }
    if !status.is_success() {
        tracing::warn!(target: "aisix::mcp_oauth", server = %server.name, %status,
            "the upstream refused dynamic client registration; configure client_id instead");
        return Err(UserOAuthFailure::Misconfigured);
    }
    response
        .json::<RegisteredClient>()
        .await
        .map_err(|_| UserOAuthFailure::Unavailable)
}
