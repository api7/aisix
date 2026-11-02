//! The sign-in flow: from an MCP client's authorization request, through the
//! upstream's redirect back, to tokens stored under the caller's subject.
//!
//! The client's PKCE challenge goes upstream and its verifier follows at the
//! token step, so the gateway never holds a secret of the client's. What the
//! client receives instead of the upstream's code is a code of the gateway's
//! own.

use std::collections::BTreeMap;
use std::sync::Arc;

use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::discovery::Endpoints;
use super::http::{send, HttpFailure};
use super::{
    digest, random_token, unix_now, RegisteredClient, TokenLabels, TokenRecord, UserOAuthManager,
    UserOAuthServer, UserSubject, CLIENT_TTL, CODE_TTL, FLOW_TTL, NS_CLIENT, NS_CODE, NS_FLOW,
    NS_TOKEN, TOKEN_TTL,
};
use crate::error::UserOAuthFailure;
use crate::shared_store::{consume, PutCondition, StoreError};

/// How many times a sign-in re-reads a record that keeps moving under it.
const STORE_TOKEN_ATTEMPTS: usize = 3;

/// A validated authorization request from an MCP client.
pub struct AuthorizationRequest<'a> {
    pub client_id: &'a str,
    pub redirect_uri: &'a str,
    pub state: Option<&'a str>,
    pub code_challenge: &'a str,
    pub scope: Option<&'a str>,
    /// Where the upstream sends the user back to.
    pub upstream_redirect_uri: &'a str,
}

#[derive(Serialize, Deserialize)]
struct FlowRecord {
    server_id: String,
    server_name: String,
    client_id: String,
    redirect_uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    client_state: Option<String>,
    code_challenge: String,
    upstream_redirect_uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resource: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    issuer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    registered_client: Option<RegisteredClient>,
}

#[derive(Serialize, Deserialize)]
struct CodeRecord {
    server_id: String,
    client_id: String,
    redirect_uri: String,
    code_challenge: String,
    upstream_code: String,
    upstream_redirect_uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resource: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    registered_client: Option<RegisteredClient>,
}

/// A flow the upstream's redirect has just consumed.
pub struct CompletedFlow {
    record: FlowRecord,
}

impl CompletedFlow {
    pub fn redirect_uri(&self) -> &str {
        &self.record.redirect_uri
    }
    pub fn client_state(&self) -> Option<&str> {
        self.record.client_state.as_deref()
    }
    pub fn server_name(&self) -> &str {
        &self.record.server_name
    }
    pub fn server_id(&self) -> &str {
        &self.record.server_id
    }
    /// Whether `iss` from the authorization response (RFC 9207) names the
    /// authorization server this flow went to.
    pub fn issuer_matches(&self, iss: &str) -> bool {
        self.record
            .issuer
            .as_deref()
            .is_none_or(|issuer| issuer.trim_end_matches('/') == iss.trim_end_matches('/'))
    }
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
) -> TokenOutcome {
    let response = match send(
        &server.name,
        token_endpoint,
        server.timeout,
        |client, url| {
            client
                .post(url)
                .header("accept", "application/json")
                .form(form)
        },
    )
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
    /// Start a sign-in: record the flow and return the upstream authorization
    /// URL to send the user to.
    pub async fn begin_authorization(
        &self,
        server: &UserOAuthServer,
        request: &AuthorizationRequest<'_>,
    ) -> Result<String, UserOAuthFailure> {
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
            .registered_client(server, &endpoints, request.upstream_redirect_uri)
            .await?;
        let upstream_client_id = server
            .client_id
            .clone()
            .or_else(|| registered.as_ref().map(|c| c.client_id.clone()))
            .ok_or(UserOAuthFailure::Misconfigured)?;

        let state = random_token();
        let record = FlowRecord {
            server_id: server.id.clone(),
            server_name: server.name.clone(),
            client_id: request.client_id.to_owned(),
            redirect_uri: request.redirect_uri.to_owned(),
            client_state: request.state.map(str::to_owned),
            code_challenge: request.code_challenge.to_owned(),
            upstream_redirect_uri: request.upstream_redirect_uri.to_owned(),
            resource: endpoints.resource.clone(),
            issuer: endpoints.issuer.clone(),
            registered_client: registered,
        };
        let labels = BTreeMap::from([("mcp_server_id", server.id.as_str())]);
        self.store
            .put(
                NS_FLOW,
                &digest(&[&state]),
                serde_json::to_vec(&record).expect("flow record serializes"),
                FLOW_TTL,
                &labels,
                PutCondition::CreateOnly,
            )
            .await
            .map_err(store_failure)?;

        let scope = request.scope.map(str::to_owned).or_else(|| {
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
                .append_pair("redirect_uri", request.upstream_redirect_uri)
                .append_pair("state", &state)
                .append_pair("code_challenge", request.code_challenge)
                .append_pair("code_challenge_method", "S256");
            if let Some(scope) = &scope {
                query.append_pair("scope", scope);
            }
            if let Some(resource) = &endpoints.resource {
                query.append_pair("resource", resource);
            }
        }
        Ok(url.into())
    }

    /// Take the flow `state` names, once. `None` is every state this gateway
    /// did not issue, has already consumed, or has let expire.
    pub async fn take_flow(&self, state: &str) -> Result<Option<CompletedFlow>, UserOAuthFailure> {
        let value = consume(self.store.as_ref(), NS_FLOW, &digest(&[state]))
            .await
            .map_err(store_failure)?;
        Ok(value
            .and_then(|value| serde_json::from_slice(&value).ok())
            .map(|record| CompletedFlow { record }))
    }

    /// Keep the upstream's `code` here, and return the single-use code the
    /// client gets in its place.
    pub async fn issue_code(
        &self,
        flow: CompletedFlow,
        upstream_code: &str,
    ) -> Result<String, UserOAuthFailure> {
        let flow = flow.record;
        let code = random_token();
        let labels = BTreeMap::from([("mcp_server_id", flow.server_id.as_str())]);
        let record = CodeRecord {
            server_id: flow.server_id.clone(),
            client_id: flow.client_id,
            redirect_uri: flow.redirect_uri,
            code_challenge: flow.code_challenge,
            upstream_code: upstream_code.to_owned(),
            upstream_redirect_uri: flow.upstream_redirect_uri,
            resource: flow.resource,
            registered_client: flow.registered_client,
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

    /// Exchange a gateway-issued code upstream and store the tokens under
    /// `subject`.
    pub async fn redeem_code(
        self: &Arc<Self>,
        server: &UserOAuthServer,
        subject: &UserSubject,
        request: &RedeemRequest<'_>,
    ) -> Result<(), RedeemError> {
        let value = consume(self.store.as_ref(), NS_CODE, &digest(&[request.code]))
            .await
            .map_err(|error| {
                store_failure(error);
                RedeemError::Unavailable
            })?
            .ok_or(RedeemError::InvalidGrant)?;
        let code: CodeRecord =
            serde_json::from_slice(&value).map_err(|_| RedeemError::InvalidGrant)?;
        let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(Sha256::digest(request.code_verifier.as_bytes()));
        if code.server_id != server.id
            || code.client_id != request.client_id
            || code.redirect_uri != request.redirect_uri
            || code.code_challenge != challenge
        {
            return Err(RedeemError::InvalidGrant);
        }

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
            ("code", code.upstream_code),
            ("redirect_uri", code.upstream_redirect_uri),
            ("code_verifier", request.code_verifier.to_owned()),
        ];
        if let Some(resource) = &code.resource {
            form.push(("resource", resource.clone()));
        }
        add_client(&mut form, server, code.registered_client.as_ref());
        let issued = match token_request(server, &token_endpoint, &form).await {
            TokenOutcome::Issued(issued) => issued,
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
            resource: code.resource,
            registered_client: code.registered_client,
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
        let key = digest(&[&server.id, subject.id()]);
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
