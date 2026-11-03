//! Per-user OAuth for upstream MCP servers (`auth_type:
//! oauth2_authorization_code`).
//!
//! Each user signs in to the upstream authorization server once. The gateway
//! keeps that user's tokens in a [`SharedStore`] and injects the access token
//! on every upstream operation; an MCP client only ever holds a placeholder.
//!
//! Records written here are read by gateways of other versions, so they
//! tolerate unknown fields and every field but the one a record exists for
//! has a default.

mod discovery;
mod flow;
mod http;
mod login;
mod session;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use aisix_core::McpServer;
use futures::future::{BoxFuture, FutureExt, Shared};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::UserOAuthFailure;
use crate::shared_store::{PutCondition, SharedStore, StoreError};

pub use flow::{
    pkce_pair, FlowStage, FlowState, RedeemError, RedeemRequest, RelayedCode, UpstreamLeg,
    FLOW_TTL_SECS,
};
pub use login::{LoginCode, LoginEndpoints, LoginExchange};
pub use session::{
    SessionIdentity, SessionKeys, SessionToken, SessionTokenPair, ACCESS_TOKEN_PREFIX,
    ACCESS_TOKEN_TTL_SECS,
};

const NS_TOKEN: &str = "mcp_oauth_token";
const NS_CLIENT: &str = "mcp_oauth_client";
const NS_CODE: &str = "mcp_oauth_code";
const NS_LEASE: &str = "mcp_oauth_lease";
const NS_SESSION_KEY: &str = "mcp_oauth_session_key";
const NS_REFRESH_USED: &str = "mcp_oauth_refresh_used";

const TOKEN_TTL: Duration = Duration::from_secs(365 * 24 * 3600);
const CLIENT_TTL: Duration = Duration::from_secs(365 * 24 * 3600);
const CODE_TTL: Duration = Duration::from_secs(120);
const LEASE_TTL: Duration = Duration::from_secs(60);
/// How far ahead of its lease's expiry a refresh stops waiting, leaving time
/// to store what it got.
const LEASE_MARGIN: Duration = Duration::from_secs(5);

/// A token with less than this left is refreshed before it is used.
const REFRESH_SKEW_SECS: i64 = 60;
/// How long a stored token is reused here without reading the store, so a
/// revocation on another gateway takes effect.
const LOCAL_CACHE_MAX: Duration = Duration::from_secs(300);
const LOCAL_CACHE_CAPACITY: usize = 10_000;
/// How often a gateway waiting on another gateway's refresh re-reads the
/// record.
const LEASE_POLL: Duration = Duration::from_millis(200);
/// How long a gateway waits on another gateway's proactive refresh before it
/// goes back to the token still in the record. The holder's work is one token
/// request, so this is well past a healthy one, and a holder that is stuck or
/// gone would otherwise hold everyone else off for up to [`LEASE_TTL`].
const PROACTIVE_WAIT: Duration = Duration::from_secs(5);

/// Error codes a token endpoint answers a refresh with when the grant itself
/// is finished, so only a new sign-in can help. Anything else leaves the
/// stored record alone.
const DEFINITIVE_REFRESH_ERRORS: &[&str] =
    &["invalid_grant", "invalid_token", "unauthorized_client"];

/// Who a stored token belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UserSubject {
    id: String,
    /// What `id` says, hashed so that no two identities can collide: a provider
    /// name and a claim value may both contain the `:` that `id` joins them with.
    record_key: String,
    member_id: Option<String>,
}

impl UserSubject {
    /// `issuer` is the provider's as configured now, `None` on one that pins
    /// none, and `identity_claim` the claim it reads `subject` from now. Both are
    /// in the record key because the same value means someone else at another
    /// identity provider, or under another claim. `id` stays without them: the
    /// store's `subject` label comes from `id`, and the control plane purges a
    /// provider's tokens by that label's prefix.
    pub fn jwt(provider: &str, issuer: Option<&str>, identity_claim: &str, subject: &str) -> Self {
        let issuer = normalized_issuer(issuer.unwrap_or_default());
        Self {
            id: format!("jwt:{provider}:{subject}"),
            record_key: digest(&["jwt", provider, issuer, identity_claim, subject]),
            member_id: None,
        }
    }

    pub fn member(user_id: &str) -> Self {
        Self {
            id: format!("member:{user_id}"),
            record_key: digest(&["member", user_id]),
            member_id: Some(user_id.to_owned()),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }
}

/// The OAuth settings of one registered server.
#[derive(Clone)]
pub struct UserOAuthServer {
    pub id: String,
    pub name: String,
    pub url: String,
    client_id: Option<String>,
    client_secret: Option<String>,
    authorization_url: Option<String>,
    token_url: Option<String>,
    registration_url: Option<String>,
    scopes: Vec<String>,
    timeout: Duration,
}

fn non_empty(value: &Option<String>) -> Option<String> {
    value.clone().filter(|v| !v.is_empty())
}

impl UserOAuthServer {
    pub fn new(id: &str, server: &McpServer) -> Self {
        Self {
            id: id.to_owned(),
            name: server.name.clone(),
            url: server.url.clone(),
            client_id: non_empty(&server.client_id),
            client_secret: non_empty(&server.secret),
            authorization_url: non_empty(&server.authorization_url),
            token_url: non_empty(&server.token_url),
            registration_url: non_empty(&server.registration_url),
            scopes: server.scopes.clone().unwrap_or_default(),
            timeout: server
                .timeout_ms
                .map(Duration::from_millis)
                .unwrap_or(crate::bridge::DEFAULT_UPSTREAM_TIMEOUT),
        }
    }

    /// How `url`, about to be fetched for this server, is treated: as the
    /// operator's when this server's configuration names it or it shares the
    /// authority of the server's `url`, as an upstream's choice otherwise.
    fn gate_for(&self, url: &str) -> http::Gate {
        let origin = |url: &str| reqwest::Url::parse(url).ok().map(|url| url.origin());
        if self.token_url.as_deref() == Some(url)
            || self.registration_url.as_deref() == Some(url)
            || origin(url) == origin(&self.url)
        {
            http::Gate::Configured
        } else {
            http::Gate::Ssrf
        }
    }

    /// Scopes to advertise to MCP clients.
    pub fn scopes(&self) -> &[String] {
        &self.scopes
    }
}

/// The gateway's OAuth client at the upstream authorization server, when it
/// got one by registering rather than from configuration.
#[derive(Clone, Serialize, Deserialize)]
struct RegisteredClient {
    client_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    client_secret: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct TokenRecord {
    access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    refresh_token: Option<String>,
    /// Unix seconds. Absent when the token endpoint did not say, in which
    /// case only the upstream's own `401` ends the token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resource: Option<String>,
    /// The registered client the tokens were issued to. A configured
    /// `client_id` takes precedence over it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    registered_client: Option<RegisteredClient>,
    /// The `redirect_uri` of the sign-in, repeated on a refresh: Jira Data
    /// Center documents it there, and RFC 6749 §3.2 has every other token
    /// endpoint ignore it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    redirect_uri: Option<String>,
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

impl TokenRecord {
    fn needs_refresh(&self, now: i64) -> bool {
        self.expires_at
            .is_some_and(|at| at - now < REFRESH_SKEW_SECS)
    }

    fn expired(&self, now: i64) -> bool {
        self.expires_at.is_some_and(|at| at <= now)
    }
}

struct CachedToken {
    access_token: String,
    /// Unix seconds; when this token stops being usable at all.
    expires_at: Option<i64>,
    /// When to stop trusting this copy over the store.
    read_store_after: Instant,
}

/// `(server, subject)` for one in-flight refresh and one store key.
/// An issuer as compared, which a trailing slash does not change.
pub fn normalized_issuer(issuer: &str) -> &str {
    issuer.trim_end_matches('/')
}

fn digest(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part.as_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Where `subject`'s record for `server` is kept. The upstream's address is part
/// of it, so moving a server's `url` or `token_url` leaves the old record out of
/// reach: those users sign in again rather than having their tokens sent to the
/// new host. Purging the orphans is the store owner's cleanup.
fn token_key(server: &UserOAuthServer, subject: &UserSubject) -> String {
    digest(&[
        &server.id,
        &server.url,
        server.token_url.as_deref().unwrap_or_default(),
        &subject.record_key,
    ])
}

/// An unguessable URL-safe value for a flow handle or an authorization code.
pub fn random_token() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

type Flight = Shared<BoxFuture<'static, Result<String, UserOAuthFailure>>>;

/// What a refresh replaces.
enum Replacing {
    /// The token in this version of the record, close to expiring.
    Expiring { version: i64 },
    /// A token the upstream refused.
    Rejected(String),
}

impl Replacing {
    fn rejected(&self) -> Option<&str> {
        match self {
            Self::Expiring { .. } => None,
            Self::Rejected(token) => Some(token),
        }
    }
}

/// Everything per-user OAuth keeps per process: the store handle, the local
/// token cache, and the refreshes in flight.
pub struct UserOAuthManager {
    store: Arc<dyn SharedStore>,
    cache: Mutex<HashMap<String, CachedToken>>,
    inflight: Mutex<HashMap<String, Flight>>,
    discovery: discovery::DiscoveryCache,
    registrations: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    login_discovery: login::LoginDiscoveryCache,
    session_key: session::SessionKeyCache,
}

impl UserOAuthManager {
    pub fn new(store: Arc<dyn SharedStore>) -> Self {
        Self {
            store,
            cache: Mutex::default(),
            inflight: Mutex::default(),
            discovery: discovery::DiscoveryCache::default(),
            registrations: Mutex::default(),
            login_discovery: login::LoginDiscoveryCache::default(),
            session_key: session::SessionKeyCache::default(),
        }
    }

    /// `subject`'s access token for `server`, refreshed first when it is
    /// about to expire.
    pub async fn access_token(
        self: &Arc<Self>,
        server: &UserOAuthServer,
        subject: &UserSubject,
    ) -> Result<String, UserOAuthFailure> {
        let key = token_key(server, subject);
        let cached = self.cached(&key);
        if let Some((token, true)) = &cached {
            return Ok(token.clone());
        }
        self.token_from_store(server, subject, &key, cached).await
    }

    /// [`Self::access_token`] for a decision about whether `subject` is still
    /// connected, which a copy held here cannot answer: the store does, however
    /// recent the copy. Not for the request path, which relies on that copy to
    /// stay off the store.
    pub async fn stored_access_token(
        self: &Arc<Self>,
        server: &UserOAuthServer,
        subject: &UserSubject,
    ) -> Result<String, UserOAuthFailure> {
        let key = token_key(server, subject);
        let cached = self.cached(&key);
        self.token_from_store(server, subject, &key, cached).await
    }

    async fn token_from_store(
        self: &Arc<Self>,
        server: &UserOAuthServer,
        subject: &UserSubject,
        key: &str,
        cached: Option<(String, bool)>,
    ) -> Result<String, UserOAuthFailure> {
        let stored = match self.store.get(NS_TOKEN, key).await {
            Ok(stored) => stored,
            Err(error) => {
                // A token already in hand keeps working through a storage
                // outage, past the window that bounds how long it is trusted
                // over the store: there is no store to prefer.
                return match cached {
                    Some((token, _)) => Ok(token),
                    None => {
                        tracing::warn!(target: "aisix::mcp_oauth", server = %server.name, %error,
                            "could not read a user's stored MCP token");
                        Err(UserOAuthFailure::Unavailable)
                    }
                };
            }
        };
        let Some((record, version)) = stored.as_ref().and_then(decode_versioned) else {
            self.evict(key);
            return Err(UserOAuthFailure::AuthorizationRequired);
        };
        if !record.needs_refresh(unix_now()) {
            self.remember(key, &record);
            return Ok(record.access_token);
        }
        match self
            .refresh(server, subject, Replacing::Expiring { version })
            .await
        {
            // Whatever stopped the refresh, this token has not expired yet.
            Err(UserOAuthFailure::Unavailable) if !record.expired(unix_now()) => {
                Ok(record.access_token)
            }
            outcome => outcome,
        }
    }

    /// The upstream rejected `rejected`: return the token to try next, which
    /// a refresh produced unless another gateway already replaced it.
    pub async fn refresh_rejected(
        self: &Arc<Self>,
        server: &UserOAuthServer,
        subject: &UserSubject,
        rejected: &str,
    ) -> Result<String, UserOAuthFailure> {
        self.evict(&token_key(server, subject));
        self.refresh(server, subject, Replacing::Rejected(rejected.to_owned()))
            .await
    }

    /// The upstream rejected `rejected` even fresh from a refresh, so only a
    /// new sign-in can help. Leaves a record some other sign-in has replaced
    /// in the meantime alone.
    pub async fn discard_rejected(
        &self,
        server: &UserOAuthServer,
        subject: &UserSubject,
        rejected: &str,
    ) {
        let key = token_key(server, subject);
        self.evict(&key);
        if let Ok(Some(stored)) = self.store.get(NS_TOKEN, &key).await {
            if decode_token(&stored).is_some_and(|record| record.access_token == rejected) {
                self.delete_record(server, &key, Some(stored.version)).await;
            }
        }
    }

    /// Remove `subject`'s connection to `server`. This gateway forgets it at
    /// once; another keeps a cached copy for at most [`LOCAL_CACHE_MAX`].
    pub async fn disconnect(
        &self,
        server: &UserOAuthServer,
        subject: &UserSubject,
    ) -> Result<(), UserOAuthFailure> {
        let key = token_key(server, subject);
        self.evict(&key);
        self.store
            .delete(NS_TOKEN, &key, None)
            .await
            .map_err(|_| UserOAuthFailure::Unavailable)
    }

    fn cached(&self, key: &str) -> Option<(String, bool)> {
        let cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        let entry = cache.get(key)?;
        let now = unix_now();
        if entry.expires_at.is_some_and(|at| at <= now) {
            return None;
        }
        let fresh = entry.read_store_after > Instant::now()
            && entry
                .expires_at
                .is_none_or(|at| at - now >= REFRESH_SKEW_SECS);
        Some((entry.access_token.clone(), fresh))
    }

    fn remember(&self, key: &str, record: &TokenRecord) {
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        if cache.len() >= LOCAL_CACHE_CAPACITY {
            let now = unix_now();
            cache.retain(|_, entry| entry.expires_at.is_none_or(|at| at > now));
            if cache.len() >= LOCAL_CACHE_CAPACITY {
                let excess: Vec<String> = cache
                    .keys()
                    .take(LOCAL_CACHE_CAPACITY / 10)
                    .cloned()
                    .collect();
                for key in excess {
                    cache.remove(&key);
                }
            }
        }
        cache.insert(
            key.to_owned(),
            CachedToken {
                access_token: record.access_token.clone(),
                expires_at: record.expires_at,
                read_store_after: Instant::now() + LOCAL_CACHE_MAX,
            },
        );
    }

    fn evict(&self, key: &str) {
        self.cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(key);
    }

    async fn delete_record(&self, server: &UserOAuthServer, key: &str, version: Option<i64>) {
        // If this fails the record stays, and the next attempt ends the same
        // way: a refresh that is rejected again.
        if let Err(error) = self.store.delete(NS_TOKEN, key, version).await {
            if error != StoreError::PreconditionFailed {
                tracing::warn!(target: "aisix::mcp_oauth", server = %server.name, %error,
                    "could not delete a user's stored MCP token");
            }
        }
    }

    /// One refresh per `(server, subject)` in this process, spawned so that a
    /// caller going away cannot cut it off between the token endpoint
    /// rotating the refresh token and the new one being stored.
    async fn refresh(
        self: &Arc<Self>,
        server: &UserOAuthServer,
        subject: &UserSubject,
        replacing: Replacing,
    ) -> Result<String, UserOAuthFailure> {
        let key = token_key(server, subject);
        // Keyed on the token being replaced too: a caller holding a rejected
        // token cannot share the outcome of a proactive refresh, which may
        // hand that very token back. The lease still lets only one through.
        let flight_key = digest(&[&key, replacing.rejected().unwrap_or_default()]);
        let flight = self
            .inflight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(flight_key.clone())
            .or_insert_with(|| {
                let manager = Arc::clone(self);
                let server = server.clone();
                let subject = subject.clone();
                let task = tokio::spawn(async move {
                    let outcome = manager
                        .refresh_under_lease(&server, &subject, &key, replacing)
                        .await;
                    manager
                        .inflight
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .remove(&flight_key);
                    outcome
                });
                async move { task.await.unwrap_or(Err(UserOAuthFailure::Unavailable)) }
                    .boxed()
                    .shared()
            })
            .clone();
        // The caller stops waiting at the server's timeout. The refresh itself
        // runs on, so that what it obtains is stored for the next request.
        tokio::time::timeout(server.timeout, flight)
            .await
            .unwrap_or(Err(UserOAuthFailure::Unavailable))
    }

    /// One refresh per `(server, subject)` across gateways: the holder of the
    /// lease refreshes, everyone else waits for the record to change.
    async fn refresh_under_lease(
        &self,
        server: &UserOAuthServer,
        subject: &UserSubject,
        key: &str,
        replacing: Replacing,
    ) -> Result<String, UserOAuthFailure> {
        let labels = BTreeMap::from([("mcp_server_id", server.id.as_str())]);
        let lease = self
            .store
            .put(
                NS_LEASE,
                key,
                b"1".to_vec(),
                LEASE_TTL,
                &labels,
                PutCondition::CreateOnly,
            )
            .await;
        let lease_taken = Instant::now();
        let lease = match lease {
            Ok(version) => version,
            Err(StoreError::PreconditionFailed) => {
                return self.wait_for_refresh(server, key, &replacing).await;
            }
            Err(_) => return Err(UserOAuthFailure::Unavailable),
        };
        let outcome = self
            .refresh_holding_lease(server, subject, key, &replacing, lease_taken)
            .await;
        // Expires on its own if this fails.
        let _ = self.store.delete(NS_LEASE, key, Some(lease)).await;
        outcome
    }

    async fn wait_for_refresh(
        &self,
        server: &UserOAuthServer,
        key: &str,
        replacing: &Replacing,
    ) -> Result<String, UserOAuthFailure> {
        let started = Instant::now();
        let deadline = started + server.timeout;
        loop {
            // Read ahead of the record: a holder stores first and lets go of
            // the lease after, so with the lease gone the record is its last word.
            let holder_gone = matches!(replacing, Replacing::Expiring { .. })
                && self
                    .store
                    .get(NS_LEASE, key)
                    .await
                    .map_err(|_| UserOAuthFailure::Unavailable)?
                    .is_none();
            let stored = self
                .store
                .get(NS_TOKEN, key)
                .await
                .map_err(|_| UserOAuthFailure::Unavailable)?;
            let Some((record, version)) = stored.as_ref().and_then(decode_versioned) else {
                self.evict(key);
                return Err(UserOAuthFailure::AuthorizationRequired);
            };
            let settled = match replacing {
                Replacing::Rejected(rejected) => record.access_token != *rejected,
                // An upstream may end the old token the moment it issues the
                // new one, and answer the dead one with an empty `200` that no
                // retry can catch. So the old token is only a fallback, for a
                // holder that gave up or does not finish in time.
                Replacing::Expiring { version: held } => {
                    !record.expired(unix_now())
                        && (version != *held || holder_gone || started.elapsed() >= PROACTIVE_WAIT)
                }
            };
            if settled {
                self.remember(key, &record);
                return Ok(record.access_token);
            }
            if Instant::now() + LEASE_POLL >= deadline {
                return Err(UserOAuthFailure::Unavailable);
            }
            tokio::time::sleep(LEASE_POLL).await;
        }
    }

    async fn refresh_holding_lease(
        &self,
        server: &UserOAuthServer,
        subject: &UserSubject,
        key: &str,
        replacing: &Replacing,
        lease_taken: Instant,
    ) -> Result<String, UserOAuthFailure> {
        let stored = self
            .store
            .get(NS_TOKEN, key)
            .await
            .map_err(|_| UserOAuthFailure::Unavailable)?;
        let Some(stored) = stored else {
            self.evict(key);
            return Err(UserOAuthFailure::AuthorizationRequired);
        };
        let Some(record) = decode_token(&stored) else {
            self.evict(key);
            return Err(UserOAuthFailure::AuthorizationRequired);
        };
        let now = unix_now();
        // Someone else's refresh may have landed before this lease was taken.
        // A token that lives inside the skew always reads as needing one, so
        // a record that moved past the caller's read says it just happened.
        let needed = match replacing {
            Replacing::Rejected(rejected) => record.access_token == *rejected,
            Replacing::Expiring { version } => {
                record.expired(now) || (stored.version == *version && record.needs_refresh(now))
            }
        };
        if !needed {
            self.remember(key, &record);
            return Ok(record.access_token);
        }
        // Past this point a proactive refresh can fall back on the token it
        // was replacing, for what is left of its life.
        let fallback = (replacing.rejected().is_none() && !record.expired(now))
            .then(|| record.access_token.clone());

        let Some(refresh_token) = record.refresh_token.clone() else {
            if let Some(token) = fallback {
                return Ok(token);
            }
            self.evict(key);
            self.delete_record(server, key, Some(stored.version)).await;
            return Err(UserOAuthFailure::AuthorizationRequired);
        };
        let token_endpoint = match self.discovery.token_endpoint(server).await {
            Ok(url) => url,
            Err(failure) => return fallback.ok_or(failure),
        };
        let mut form = vec![
            ("grant_type", "refresh_token".to_owned()),
            ("refresh_token", refresh_token.clone()),
        ];
        if let Some(resource) = &record.resource {
            form.push(("resource", resource.clone()));
        }
        if let Some(redirect_uri) = &record.redirect_uri {
            form.push(("redirect_uri", redirect_uri.clone()));
        }
        flow::add_client(&mut form, server, record.registered_client.as_ref());
        // A refresh outliving its lease would let another gateway spend the
        // same rotating token, so the request ends before the lease does.
        let Some(budget) = (lease_taken + LEASE_TTL - LEASE_MARGIN)
            .checked_duration_since(Instant::now())
            .map(|left| left.min(server.timeout))
        else {
            return fallback.ok_or(UserOAuthFailure::Unavailable);
        };
        let issued = match flow::token_request(server, &token_endpoint, &form, budget).await {
            flow::TokenOutcome::Issued(issued) => issued,
            flow::TokenOutcome::Rejected(code)
                if DEFINITIVE_REFRESH_ERRORS.contains(&code.as_str()) =>
            {
                tracing::info!(target: "aisix::mcp_oauth", server = %server.name, error = %code,
                    "the upstream refused a user's refresh token; the user must sign in again");
                self.evict(key);
                self.delete_record(server, key, Some(stored.version)).await;
                return Err(UserOAuthFailure::AuthorizationRequired);
            }
            flow::TokenOutcome::Rejected(code) => {
                tracing::warn!(target: "aisix::mcp_oauth", server = %server.name, error = %code,
                    "the upstream refused a token refresh for a reason that says nothing about \
                     the grant; keeping the stored token");
                return fallback.ok_or(UserOAuthFailure::Unavailable);
            }
            flow::TokenOutcome::Transient => {
                return fallback.ok_or(UserOAuthFailure::Unavailable);
            }
            flow::TokenOutcome::Blocked => {
                return fallback.ok_or(UserOAuthFailure::Misconfigured);
            }
        };
        let refreshed = TokenRecord {
            access_token: issued.access_token,
            // A token endpoint that does not rotate omits it.
            refresh_token: issued.refresh_token.or(Some(refresh_token)),
            expires_at: issued.expires_at,
            resource: record.resource,
            registered_client: record.registered_client,
            redirect_uri: record.redirect_uri,
        };
        let value = serde_json::to_vec(&refreshed).expect("token record serializes");
        let label_values = TokenLabels::new(server, subject);
        match self
            .store
            .put(
                NS_TOKEN,
                key,
                value,
                TOKEN_TTL,
                &label_values.map(),
                PutCondition::IfVersion(stored.version),
            )
            .await
        {
            Ok(_) => {
                self.remember(key, &refreshed);
                Ok(refreshed.access_token)
            }
            // The record moved while this refresh ran: a disconnect, which
            // must not be undone, or a new sign-in, which is newer than this.
            Err(StoreError::PreconditionFailed) => {
                self.evict(key);
                let current = self
                    .store
                    .get(NS_TOKEN, key)
                    .await
                    .map_err(|_| UserOAuthFailure::Unavailable)?;
                match current.as_ref().and_then(decode_token) {
                    Some(current) => {
                        self.remember(key, &current);
                        Ok(current.access_token)
                    }
                    None => Err(UserOAuthFailure::AuthorizationRequired),
                }
            }
            // The upstream has already moved on, so this token is the only
            // working copy. If it rotated the refresh token, the stored one
            // is dead and the next refresh will send the user to sign in.
            Err(error) => {
                tracing::warn!(target: "aisix::mcp_oauth", server = %server.name, %error,
                    "could not store a user's refreshed MCP token");
                self.remember(key, &refreshed);
                Ok(refreshed.access_token)
            }
        }
    }
}

fn decode_token(stored: &crate::shared_store::StoredValue) -> Option<TokenRecord> {
    serde_json::from_slice(&stored.value).ok()
}

fn decode_versioned(stored: &crate::shared_store::StoredValue) -> Option<(TokenRecord, i64)> {
    Some((decode_token(stored)?, stored.version))
}

/// The labels of a token record, owned here so the borrowed map outlives the
/// call that takes it.
struct TokenLabels<'a> {
    server_id: &'a str,
    subject: &'a UserSubject,
}

impl<'a> TokenLabels<'a> {
    fn new(server: &'a UserOAuthServer, subject: &'a UserSubject) -> Self {
        Self {
            server_id: &server.id,
            subject,
        }
    }

    fn map(&self) -> BTreeMap<&str, &str> {
        let mut labels = BTreeMap::from([
            ("mcp_server_id", self.server_id),
            ("subject", self.subject.id()),
        ]);
        if let Some(member) = &self.subject.member_id {
            labels.insert("member_id", member);
        }
        labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_store::{MemoryStore, StoredValue};
    use async_trait::async_trait;

    /// A store that can be taken down and brought back.
    #[derive(Default)]
    pub(super) struct SwitchableStore {
        inner: MemoryStore,
        down: std::sync::atomic::AtomicBool,
    }

    impl SwitchableStore {
        pub(super) fn fail(&self, down: bool) {
            self.down.store(down, std::sync::atomic::Ordering::SeqCst);
        }

        fn check(&self) -> Result<(), StoreError> {
            if self.down.load(std::sync::atomic::Ordering::SeqCst) {
                Err(StoreError::Unavailable)
            } else {
                Ok(())
            }
        }
    }

    #[async_trait]
    impl SharedStore for SwitchableStore {
        async fn get(&self, ns: &str, key: &str) -> Result<Option<StoredValue>, StoreError> {
            self.check()?;
            self.inner.get(ns, key).await
        }
        async fn put(
            &self,
            ns: &str,
            key: &str,
            value: Vec<u8>,
            ttl: Duration,
            labels: &BTreeMap<&str, &str>,
            condition: PutCondition,
        ) -> Result<i64, StoreError> {
            self.check()?;
            self.inner.put(ns, key, value, ttl, labels, condition).await
        }
        async fn delete(&self, ns: &str, key: &str, v: Option<i64>) -> Result<(), StoreError> {
            self.check()?;
            self.inner.delete(ns, key, v).await
        }
    }

    /// A store whose every call fails, as one in an outage does.
    struct DownStore;

    #[async_trait]
    impl SharedStore for DownStore {
        async fn get(&self, _: &str, _: &str) -> Result<Option<StoredValue>, StoreError> {
            Err(StoreError::Unavailable)
        }
        async fn put(
            &self,
            _: &str,
            _: &str,
            _: Vec<u8>,
            _: Duration,
            _: &BTreeMap<&str, &str>,
            _: PutCondition,
        ) -> Result<i64, StoreError> {
            Err(StoreError::Unavailable)
        }
        async fn delete(&self, _: &str, _: &str, _: Option<i64>) -> Result<(), StoreError> {
            Err(StoreError::Unavailable)
        }
    }

    fn server() -> UserOAuthServer {
        let row: McpServer = serde_json::from_value(serde_json::json!({
            "name": "s",
            "url": "https://upstream.example/mcp",
            "auth_type": "oauth2_authorization_code",
        }))
        .unwrap();
        UserOAuthServer::new("server-id", &row)
    }

    /// A copy this gateway read so long ago that it should prefer the store.
    fn stale_copy(manager: &UserOAuthManager, expires_at: Option<i64>) {
        let key = token_key(&server(), &UserSubject::member("u"));
        manager.cache.lock().unwrap().insert(
            key,
            CachedToken {
                access_token: "cached".to_owned(),
                expires_at,
                read_store_after: Instant::now(),
            },
        );
    }

    #[test]
    fn identities_that_read_alike_keep_separate_records() {
        let (a, b) = (
            UserSubject::jwt("corp", None, "sub", "x:alice"),
            UserSubject::jwt("corp:x", None, "sub", "alice"),
        );
        assert_eq!(a.id(), b.id());
        assert_ne!(token_key(&server(), &a), token_key(&server(), &b));
    }

    #[test]
    fn a_record_follows_the_issuer_and_the_label_does_not() {
        let at = |issuer| UserSubject::jwt("corp", issuer, "sub", "alice");
        let key = |subject: &UserSubject| token_key(&server(), subject);
        let idp = at(Some("https://idp.example.com"));

        for other in [at(Some("https://other.example.com")), at(None)] {
            assert_eq!(idp.id(), other.id());
            assert_eq!(idp.id(), "jwt:corp:alice");
            assert_ne!(key(&idp), key(&other));
        }
        assert_eq!(key(&idp), key(&at(Some("https://idp.example.com/"))));
    }

    #[test]
    fn a_record_follows_the_identity_claim_and_the_label_does_not() {
        let under = |claim| UserSubject::jwt("corp", Some("https://idp"), claim, "alice");
        assert_eq!(under("sub").id(), under("given_name").id());
        assert_ne!(
            token_key(&server(), &under("sub")),
            token_key(&server(), &under("given_name"))
        );
    }

    /// A copy this gateway still trusts, as a recent read leaves it.
    fn fresh_copy(manager: &UserOAuthManager, expires_at: Option<i64>) {
        manager.remember(
            &token_key(&server(), &UserSubject::member("u")),
            &TokenRecord {
                access_token: "cached".to_owned(),
                refresh_token: None,
                expires_at,
                resource: None,
                registered_client: None,
                redirect_uri: None,
            },
        );
    }

    #[tokio::test]
    async fn the_store_decides_over_a_fresh_copy_and_a_gone_record_takes_the_copy_with_it() {
        let manager = Arc::new(UserOAuthManager::new(Arc::new(MemoryStore::default())));
        let subject = UserSubject::member("u");
        fresh_copy(&manager, None);
        assert_eq!(
            manager.stored_access_token(&server(), &subject).await,
            Err(UserOAuthFailure::AuthorizationRequired)
        );
        assert_eq!(
            manager.access_token(&server(), &subject).await,
            Err(UserOAuthFailure::AuthorizationRequired)
        );
    }

    #[tokio::test]
    async fn a_store_read_survives_an_outage_on_a_copy_as_a_request_does() {
        let manager = Arc::new(UserOAuthManager::new(Arc::new(DownStore)));
        let subject = UserSubject::member("u");
        assert_eq!(
            manager.stored_access_token(&server(), &subject).await,
            Err(UserOAuthFailure::Unavailable)
        );
        fresh_copy(&manager, None);
        assert_eq!(
            manager
                .stored_access_token(&server(), &subject)
                .await
                .as_deref(),
            Ok("cached")
        );
    }

    #[tokio::test]
    async fn an_outage_is_survived_by_a_copy_past_its_window_while_it_is_valid() {
        let manager = Arc::new(UserOAuthManager::new(Arc::new(DownStore)));
        let subject = UserSubject::member("u");
        stale_copy(&manager, Some(unix_now() + 30));
        assert_eq!(
            manager.access_token(&server(), &subject).await.as_deref(),
            Ok("cached")
        );
        stale_copy(&manager, Some(unix_now() - 1));
        assert_eq!(
            manager.access_token(&server(), &subject).await,
            Err(UserOAuthFailure::Unavailable)
        );
    }

    #[tokio::test]
    async fn past_its_window_a_copy_loses_to_a_store_that_no_longer_has_it() {
        let manager = Arc::new(UserOAuthManager::new(Arc::new(MemoryStore::default())));
        let subject = UserSubject::member("u");
        stale_copy(&manager, Some(unix_now() + 3600));
        assert_eq!(
            manager.access_token(&server(), &subject).await,
            Err(UserOAuthFailure::AuthorizationRequired)
        );
    }

    /// A store that refuses leases only: the read still works.
    struct NoLeases(MemoryStore);

    #[async_trait]
    impl SharedStore for NoLeases {
        async fn get(&self, ns: &str, key: &str) -> Result<Option<StoredValue>, StoreError> {
            self.0.get(ns, key).await
        }
        async fn put(
            &self,
            ns: &str,
            key: &str,
            value: Vec<u8>,
            ttl: Duration,
            labels: &BTreeMap<&str, &str>,
            condition: PutCondition,
        ) -> Result<i64, StoreError> {
            if ns == NS_LEASE {
                return Err(StoreError::OverQuota);
            }
            self.0.put(ns, key, value, ttl, labels, condition).await
        }
        async fn delete(&self, ns: &str, key: &str, v: Option<i64>) -> Result<(), StoreError> {
            self.0.delete(ns, key, v).await
        }
    }

    async fn store_token(
        store: &dyn SharedStore,
        server: &UserOAuthServer,
        access_token: &str,
        expires_in: i64,
    ) {
        let record = TokenRecord {
            access_token: access_token.to_owned(),
            refresh_token: Some("r".to_owned()),
            expires_at: Some(unix_now() + expires_in),
            resource: None,
            registered_client: None,
            redirect_uri: None,
        };
        store
            .put(
                NS_TOKEN,
                &token_key(server, &UserSubject::member("u")),
                serde_json::to_vec(&record).unwrap(),
                TOKEN_TTL,
                &BTreeMap::new(),
                PutCondition::CreateOnly,
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_valid_token_survives_a_refresh_that_could_not_start() {
        let store = Arc::new(NoLeases(MemoryStore::default()));
        store_token(store.as_ref(), &server(), "still-good", 45).await;
        let manager = Arc::new(UserOAuthManager::new(store.clone()));
        let subject = UserSubject::member("u");
        assert_eq!(
            manager.access_token(&server(), &subject).await.as_deref(),
            Ok("still-good")
        );
    }

    /// Joining a proactive refresh would hand back the token just rejected,
    /// and the caller would then delete a record whose refresh token works.
    #[tokio::test]
    async fn a_rejected_token_does_not_come_back_from_a_proactive_refresh() {
        let store = Arc::new(MemoryStore::default());
        let manager = Arc::new(UserOAuthManager::new(store.clone()));
        let subject = UserSubject::member("u");
        let row: McpServer = serde_json::from_value(serde_json::json!({
            "name": "s",
            "url": "https://upstream.example/mcp",
            "auth_type": "oauth2_authorization_code",
            "client_id": "c",
            "token_url": "https://idp.example/token",
            "timeout_ms": 300,
        }))
        .unwrap();
        let server = UserOAuthServer::new("server-id", &row);
        store_token(store.as_ref(), &server, "x", 45).await;

        // Hold the lease, as a slow proactive refresh would.
        store
            .put(
                NS_LEASE,
                &token_key(&server, &subject),
                b"1".to_vec(),
                LEASE_TTL,
                &BTreeMap::new(),
                PutCondition::CreateOnly,
            )
            .await
            .unwrap();
        let (proactive, rejected) = tokio::join!(
            manager.access_token(&server, &subject),
            manager.refresh_rejected(&server, &subject, "x"),
        );
        assert_eq!(proactive.as_deref(), Ok("x"));
        assert_eq!(rejected, Err(UserOAuthFailure::Unavailable));
    }

    /// An authorization server that takes a while to answer and honours only
    /// the access token it issued last, as one that ends the old token on a
    /// refresh does.
    struct RotatingUpstream {
        token_url: String,
        valid: Arc<Mutex<String>>,
        refreshes: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl RotatingUpstream {
        async fn start(initial: &str, delay: Duration, expires_in: i64) -> Self {
            let valid = Arc::new(Mutex::new(initial.to_owned()));
            let refreshes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let app = axum::Router::new().route(
                "/token",
                axum::routing::post({
                    let valid = valid.clone();
                    let refreshes = refreshes.clone();
                    move || async move {
                        let n = 1 + refreshes.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        tokio::time::sleep(delay).await;
                        let token = format!("new-{n}");
                        *valid.lock().unwrap() = token.clone();
                        axum::Json(serde_json::json!({
                            "access_token": token,
                            "refresh_token": format!("r-{n}"),
                            "token_type": "Bearer",
                            "expires_in": expires_in,
                        }))
                    }
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let token_url = format!("http://{}/token", listener.local_addr().unwrap());
            tokio::spawn(async move { axum::serve(listener, app).await });
            Self {
                token_url,
                valid,
                refreshes,
            }
        }

        fn valid(&self) -> String {
            self.valid.lock().unwrap().clone()
        }

        fn refreshes(&self) -> usize {
            self.refreshes.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    fn server_at(token_url: &str, timeout: Duration) -> UserOAuthServer {
        let row: McpServer = serde_json::from_value(serde_json::json!({
            "name": "s",
            "url": "https://upstream.example/mcp",
            "auth_type": "oauth2_authorization_code",
            "client_id": "c",
            "token_url": token_url,
            "timeout_ms": timeout.as_millis() as u64,
        }))
        .unwrap();
        UserOAuthServer::new("server-id", &row)
    }

    /// Eight concurrent calls across two gateways on one store, all before
    /// the upstream has answered the refresh.
    async fn calls_on_two_gateways(
        store: Arc<MemoryStore>,
        server: &UserOAuthServer,
    ) -> Vec<Result<String, UserOAuthFailure>> {
        let gateways = [
            Arc::new(UserOAuthManager::new(store.clone())),
            Arc::new(UserOAuthManager::new(store)),
        ];
        let subject = UserSubject::member("u");
        futures::future::join_all((0..8).map(|i| {
            let gateway = gateways[i % 2].clone();
            let (server, subject) = (server.clone(), subject.clone());
            async move { gateway.access_token(&server, &subject).await }
        }))
        .await
    }

    #[tokio::test]
    async fn no_gateway_keeps_the_token_a_refresh_is_ending() {
        let upstream = RotatingUpstream::start("old", Duration::from_millis(400), 3600).await;
        let server = server_at(&upstream.token_url, Duration::from_secs(30));
        let store = Arc::new(MemoryStore::default());
        store_token(store.as_ref(), &server, "old", 45).await;

        let tokens = calls_on_two_gateways(store, &server).await;

        assert_eq!(upstream.refreshes(), 1);
        assert_eq!(upstream.valid(), "new-1");
        assert_eq!(tokens, vec![Ok("new-1".to_owned()); 8]);
    }

    /// Being inside the skew says nothing here: every token this upstream
    /// issues is, so only the record changing can end the wait in time.
    #[tokio::test]
    async fn a_token_that_lives_inside_the_skew_still_settles_on_the_refresh() {
        let upstream = RotatingUpstream::start("old", Duration::from_millis(400), 30).await;
        let server = server_at(&upstream.token_url, Duration::from_secs(30));
        let store = Arc::new(MemoryStore::default());
        store_token(store.as_ref(), &server, "old", 30).await;

        let started = Instant::now();
        let tokens = calls_on_two_gateways(store, &server).await;

        assert!(started.elapsed() < PROACTIVE_WAIT);
        assert_eq!(upstream.refreshes(), 1);
        assert_eq!(tokens, vec![Ok("new-1".to_owned()); 8]);
    }

    /// Take the lease, as another gateway's refresh does.
    async fn hold_lease(store: &dyn SharedStore, server: &UserOAuthServer) {
        store
            .put(
                NS_LEASE,
                &token_key(server, &UserSubject::member("u")),
                b"1".to_vec(),
                LEASE_TTL,
                &BTreeMap::new(),
                PutCondition::CreateOnly,
            )
            .await
            .unwrap();
    }

    /// What the holder of the lease does when its refresh lands, `after` from now.
    fn land_refresh(
        store: Arc<dyn SharedStore>,
        server: &UserOAuthServer,
        access_token: &str,
        after: Duration,
    ) {
        let key = token_key(server, &UserSubject::member("u"));
        let access_token = access_token.to_owned();
        tokio::spawn(async move {
            tokio::time::sleep(after).await;
            let current = store.get(NS_TOKEN, &key).await.unwrap().unwrap();
            let record = TokenRecord {
                access_token,
                refresh_token: Some("r2".to_owned()),
                expires_at: Some(unix_now() + 3600),
                resource: None,
                registered_client: None,
                redirect_uri: None,
            };
            store
                .put(
                    NS_TOKEN,
                    &key,
                    serde_json::to_vec(&record).unwrap(),
                    TOKEN_TTL,
                    &BTreeMap::new(),
                    PutCondition::IfVersion(current.version),
                )
                .await
                .unwrap();
        });
    }

    #[tokio::test]
    async fn a_holder_that_never_finishes_costs_the_bound_and_then_the_old_token_works() {
        let server = server_at("http://127.0.0.1:9/token", Duration::from_secs(30));
        let store = Arc::new(MemoryStore::default());
        store_token(store.as_ref(), &server, "old", 45).await;
        hold_lease(store.as_ref(), &server).await;

        let started = Instant::now();
        let token = Arc::new(UserOAuthManager::new(store))
            .access_token(&server, &UserSubject::member("u"))
            .await;

        assert_eq!(token.as_deref(), Ok("old"));
        let waited = started.elapsed();
        assert!(waited >= PROACTIVE_WAIT, "returned after {waited:?}");
        assert!(
            waited < PROACTIVE_WAIT + Duration::from_secs(3),
            "{waited:?}"
        );
    }

    /// Its refresh failed, or there was nothing to refresh with: no new record
    /// is coming, and the old token works.
    #[tokio::test]
    async fn a_holder_that_gave_up_ends_the_wait_at_once() {
        let server = server_at("http://127.0.0.1:9/token", Duration::from_secs(30));
        let store = Arc::new(MemoryStore::default());
        store_token(store.as_ref(), &server, "old", 45).await;
        hold_lease(store.as_ref(), &server).await;
        let subject = UserSubject::member("u");
        let manager = Arc::new(UserOAuthManager::new(store.clone()));

        let started = Instant::now();
        let (token, _) = tokio::join!(manager.access_token(&server, &subject), async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            let key = token_key(&server, &subject);
            store.delete(NS_LEASE, &key, None).await.unwrap();
        });

        assert_eq!(token.as_deref(), Ok("old"));
        assert!(started.elapsed() < PROACTIVE_WAIT);
    }

    /// A caller that read the old record and takes the lease after the refresh
    /// landed must not end the token others just received.
    #[tokio::test]
    async fn a_refresh_that_landed_before_the_lease_was_taken_is_not_repeated() {
        let upstream = RotatingUpstream::start("old", Duration::ZERO, 30).await;
        let server = server_at(&upstream.token_url, Duration::from_secs(30));
        let subject = UserSubject::member("u");
        let key = token_key(&server, &subject);
        let store = Arc::new(MemoryStore::default());
        store_token(store.as_ref(), &server, "old", 30).await;
        let read = store.get(NS_TOKEN, &key).await.unwrap().unwrap().version;
        store.delete(NS_TOKEN, &key, None).await.unwrap();
        store_token(store.as_ref(), &server, "landed", 30).await;

        let token = Arc::new(UserOAuthManager::new(store))
            .refresh(&server, &subject, Replacing::Expiring { version: read })
            .await;

        assert_eq!(token.as_deref(), Ok("landed"));
        assert_eq!(upstream.refreshes(), 0);
    }

    /// The copy kept for an outage must not stay the token the refresh ended.
    #[tokio::test]
    async fn the_token_a_wait_settled_on_is_what_an_outage_falls_back_to() {
        let server = server_at("http://127.0.0.1:9/token", Duration::from_secs(30));
        let store = Arc::new(SwitchableStore::default());
        store_token(store.as_ref(), &server, "old", 45).await;
        hold_lease(store.as_ref(), &server).await;
        land_refresh(store.clone(), &server, "new", Duration::from_millis(300));
        let subject = UserSubject::member("u");
        let manager = Arc::new(UserOAuthManager::new(store.clone()));
        assert_eq!(
            manager
                .stored_access_token(&server, &subject)
                .await
                .as_deref(),
            Ok("new")
        );

        store.fail(true);

        assert_eq!(
            manager
                .stored_access_token(&server, &subject)
                .await
                .as_deref(),
            Ok("new")
        );
    }

    /// An upstream may hand the same access token back with a later expiry.
    #[tokio::test]
    async fn a_refresh_that_keeps_the_access_token_still_ends_the_wait() {
        let server = server_at("http://127.0.0.1:9/token", Duration::from_secs(30));
        let store = Arc::new(MemoryStore::default());
        store_token(store.as_ref(), &server, "same", 45).await;
        hold_lease(store.as_ref(), &server).await;
        land_refresh(store.clone(), &server, "same", Duration::from_millis(300));

        let started = Instant::now();
        let token = Arc::new(UserOAuthManager::new(store))
            .access_token(&server, &UserSubject::member("u"))
            .await;

        assert_eq!(token.as_deref(), Ok("same"));
        assert!(started.elapsed() < PROACTIVE_WAIT);
    }

    /// Once the old token has expired it is no fallback, and the wait is the
    /// expired case's: the server's timeout, not the short bound.
    #[tokio::test]
    async fn a_token_that_expires_during_the_wait_is_waited_on_past_the_bound() {
        let server = server_at("http://127.0.0.1:9/token", Duration::from_secs(30));
        let store = Arc::new(MemoryStore::default());
        store_token(store.as_ref(), &server, "old", 2).await;
        hold_lease(store.as_ref(), &server).await;
        land_refresh(
            store.clone(),
            &server,
            "new",
            PROACTIVE_WAIT + Duration::from_millis(500),
        );

        let token = Arc::new(UserOAuthManager::new(store))
            .access_token(&server, &UserSubject::member("u"))
            .await;

        assert_eq!(token.as_deref(), Ok("new"));
    }

    #[tokio::test]
    async fn a_record_deleted_during_the_wait_requires_a_sign_in() {
        let server = server_at("http://127.0.0.1:9/token", Duration::from_secs(30));
        let store = Arc::new(MemoryStore::default());
        store_token(store.as_ref(), &server, "old", 45).await;
        hold_lease(store.as_ref(), &server).await;
        let subject = UserSubject::member("u");
        let manager = Arc::new(UserOAuthManager::new(store.clone()));

        let (token, _) = tokio::join!(manager.access_token(&server, &subject), async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            let key = token_key(&server, &subject);
            store.delete(NS_TOKEN, &key, None).await.unwrap();
        });

        assert_eq!(token, Err(UserOAuthFailure::AuthorizationRequired));
    }

    #[tokio::test]
    async fn a_store_outage_during_the_wait_leaves_the_old_token_working() {
        let server = server_at("http://127.0.0.1:9/token", Duration::from_secs(30));
        let store = Arc::new(SwitchableStore::default());
        store_token(store.as_ref(), &server, "old", 45).await;
        hold_lease(store.as_ref(), &server).await;
        let subject = UserSubject::member("u");
        let manager = Arc::new(UserOAuthManager::new(store.clone()));

        let started = Instant::now();
        let (token, _) = tokio::join!(manager.access_token(&server, &subject), async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            store.fail(true);
        });

        assert_eq!(token.as_deref(), Ok("old"));
        assert!(started.elapsed() < PROACTIVE_WAIT);
    }

    #[test]
    fn only_a_document_on_another_authority_is_left_to_the_ssrf_settings() {
        let row: McpServer = serde_json::from_value(serde_json::json!({
            "name": "s",
            "url": "http://mcp.example/m",
            "auth_type": "oauth2_authorization_code",
            "token_url": "https://idp.example/token",
            "registration_url": "https://idp.example/register",
        }))
        .unwrap();
        let server = UserOAuthServer::new("server-id", &row);
        for own in [
            "http://mcp.example/.well-known/x",
            "http://mcp.example:80/t",
            "https://idp.example/token",
            "https://idp.example/register",
        ] {
            assert_eq!(server.gate_for(own), http::Gate::Configured, "{own}");
        }
        for foreign in [
            "https://mcp.example/t",
            "http://mcp.example:8080/t",
            "http://other.example/t",
            "https://idp.example/.well-known/x",
            "not a url",
        ] {
            assert_eq!(server.gate_for(foreign), http::Gate::Ssrf, "{foreign}");
        }
    }

    /// Moving a server must not send its users' tokens to the new address.
    #[tokio::test]
    async fn a_moved_upstream_cannot_reach_the_old_record() {
        let store = Arc::new(MemoryStore::default());
        let manager = Arc::new(UserOAuthManager::new(store.clone()));
        let subject = UserSubject::member("u");
        let with = |url: &str, token_url: Option<&str>| {
            let row: McpServer = serde_json::from_value(serde_json::json!({
                "name": "s",
                "url": url,
                "auth_type": "oauth2_authorization_code",
                "token_url": token_url,
            }))
            .unwrap();
            UserOAuthServer::new("server-id", &row)
        };
        let original = with(
            "https://upstream.example/mcp",
            Some("https://idp.example/token"),
        );
        store_token(store.as_ref(), &original, "t", 3600).await;
        assert_eq!(
            manager.access_token(&original, &subject).await.as_deref(),
            Ok("t")
        );
        for moved in [
            with(
                "https://other.example/mcp",
                Some("https://idp.example/token"),
            ),
            with(
                "https://upstream.example/mcp",
                Some("https://other.example/token"),
            ),
            with("https://upstream.example/mcp", None),
        ] {
            assert_eq!(
                manager.access_token(&moved, &subject).await,
                Err(UserOAuthFailure::AuthorizationRequired),
            );
        }
        // Nor does it disturb the record for the address it was issued for.
        assert_eq!(
            manager.access_token(&original, &subject).await.as_deref(),
            Ok("t")
        );
    }

    #[test]
    fn a_record_from_another_version_still_reads() {
        let stored = StoredValue {
            value: br#"{"access_token":"a","added_later":{"x":1}}"#.to_vec(),
            version: 1,
            expires_at: None,
        };
        assert_eq!(decode_token(&stored).unwrap().access_token, "a");
    }
}
