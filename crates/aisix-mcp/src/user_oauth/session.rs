//! The gateway's own credentials for a keyless sign-in: session tokens handed
//! to MCP clients, and the cookie that carries a sign-in through the browser.
//!
//! Neither is stored. Both are encrypted under keys derived from one secret
//! per environment, kept in the shared store so every gateway of the
//! environment reads the same one, and held in memory after that: a request
//! with a session token costs no store round trip once a gateway has the key.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::RngCore;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, CHACHA20_POLY1305, NONCE_LEN};
use ring::hkdf;
use serde::{Deserialize, Serialize};

use super::flow::FlowState;
use super::{random_token, unix_now, UserOAuthManager, NS_REFRESH_USED, NS_SESSION_KEY};
use crate::error::UserOAuthFailure;
use crate::shared_store::{PutCondition, StoreError, StoredValue};

/// What a Bearer token starts with when it is a session's access token. Only
/// `/mcp/{server}` recognises it.
pub const ACCESS_TOKEN_PREFIX: &str = "aisix_mcp_at_";
const REFRESH_TOKEN_PREFIX: &str = "aisix_mcp_rt_";
pub const ACCESS_TOKEN_TTL_SECS: i64 = 3600;
/// Counted afresh at each rotation, with no absolute cap.
const REFRESH_TOKEN_TTL_SECS: i64 = 14 * 24 * 3600;

const SESSION_KEY_NAME: &str = "v1";
const SESSION_KEY_BYTES: usize = 32;
/// The store's maximum.
const SESSION_KEY_TTL: Duration = Duration::from_secs(365 * 24 * 3600);
const SESSION_KEY_RENEW_WITHIN_SECS: i64 = 30 * 24 * 3600;

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TokenKind {
    Access,
    Refresh,
}

/// What a session token says. Opaque to the client, and by hand not `Debug`:
/// the claims are a person's.
#[derive(Clone, Serialize, Deserialize)]
pub struct SessionToken {
    kind: TokenKind,
    /// The id of the MCP server it was issued for.
    pub server: String,
    pub client_id: String,
    /// The name of the trust provider that vouched for the identity.
    pub provider: String,
    /// The provider's `identity_claim` value at sign-in.
    pub identity: String,
    pub jti: String,
    pub iat: i64,
    pub exp: i64,
    /// The ID token's claims at sign-in, less those about the token itself.
    pub claims: serde_json::Value,
}

/// Who a session is for, as the sign-in established it.
#[derive(Clone, Serialize, Deserialize)]
pub struct SessionIdentity {
    pub provider: String,
    pub identity: String,
    pub claims: serde_json::Value,
}

pub struct SessionTokenPair {
    pub access_token: String,
    pub refresh_token: String,
}

/// The keys derived from an environment's secret.
pub struct SessionKeys {
    secret: Vec<u8>,
    /// When the store will drop the secret, if it said.
    expires_at: Option<i64>,
    access: LessSafeKey,
    refresh: LessSafeKey,
    flow: LessSafeKey,
}

fn derive(secret: &[u8], purpose: &'static [u8]) -> LessSafeKey {
    let info = [purpose];
    let unbound: UnboundKey = hkdf::Salt::new(hkdf::HKDF_SHA256, b"aisix-mcp-oauth")
        .extract(secret)
        .expand(&info, &CHACHA20_POLY1305)
        .expect("an HKDF output of one AEAD key is within bounds")
        .into();
    LessSafeKey::new(unbound)
}

fn seal(key: &LessSafeKey, aad: &[u8], plaintext: &[u8]) -> String {
    let mut nonce = [0u8; NONCE_LEN];
    rand::thread_rng().fill_bytes(&mut nonce);
    let mut sealed = plaintext.to_vec();
    key.seal_in_place_append_tag(
        Nonce::assume_unique_for_key(nonce),
        Aad::from(aad),
        &mut sealed,
    )
    .expect("a token or a cookie is far below the AEAD's length limit");
    let mut out = nonce.to_vec();
    out.append(&mut sealed);
    URL_SAFE_NO_PAD.encode(out)
}

fn open(key: &LessSafeKey, aad: &[u8], encoded: &str) -> Option<Vec<u8>> {
    let bytes = URL_SAFE_NO_PAD.decode(encoded).ok()?;
    if bytes.len() < NONCE_LEN {
        return None;
    }
    let (nonce, sealed) = bytes.split_at(NONCE_LEN);
    let nonce = Nonce::try_assume_unique_for_key(nonce).ok()?;
    let mut sealed = sealed.to_vec();
    let plaintext = key
        .open_in_place(nonce, Aad::from(aad), &mut sealed)
        .ok()?
        .to_vec();
    Some(plaintext)
}

impl SessionKeys {
    fn new(secret: Vec<u8>, expires_at: Option<i64>) -> Self {
        Self {
            access: derive(&secret, b"session-access"),
            refresh: derive(&secret, b"session-refresh"),
            flow: derive(&secret, b"flow-cookie"),
            secret,
            expires_at,
        }
    }

    fn from_stored(stored: &StoredValue) -> Result<Self, UserOAuthFailure> {
        if stored.value.len() != SESSION_KEY_BYTES {
            tracing::error!(target: "aisix::mcp_oauth",
                "the stored session key is not a key; sign-in cannot work until it is removed");
            return Err(UserOAuthFailure::Unavailable);
        }
        Ok(Self::new(stored.value.clone(), stored.expires_at))
    }

    fn needs_renewal(&self) -> bool {
        self.expires_at
            .is_some_and(|at| at - unix_now() < SESSION_KEY_RENEW_WITHIN_SECS)
    }

    fn mint(&self, kind: TokenKind, payload: &SessionToken) -> String {
        let (key, prefix) = match kind {
            TokenKind::Access => (&self.access, ACCESS_TOKEN_PREFIX),
            TokenKind::Refresh => (&self.refresh, REFRESH_TOKEN_PREFIX),
        };
        let plaintext = serde_json::to_vec(payload).expect("a session token serializes");
        format!("{prefix}{}", seal(key, prefix.as_bytes(), &plaintext))
    }

    /// A new access and refresh token for a client of `server`.
    pub fn mint_pair(
        &self,
        server: &str,
        client_id: &str,
        identity: &SessionIdentity,
    ) -> SessionTokenPair {
        let iat = unix_now();
        let token = |kind, ttl| SessionToken {
            kind,
            server: server.to_owned(),
            client_id: client_id.to_owned(),
            provider: identity.provider.clone(),
            identity: identity.identity.clone(),
            jti: random_token(),
            iat,
            exp: iat + ttl,
            claims: identity.claims.clone(),
        };
        SessionTokenPair {
            access_token: self.mint(
                TokenKind::Access,
                &token(TokenKind::Access, ACCESS_TOKEN_TTL_SECS),
            ),
            refresh_token: self.mint(
                TokenKind::Refresh,
                &token(TokenKind::Refresh, REFRESH_TOKEN_TTL_SECS),
            ),
        }
    }

    fn open_token(&self, kind: TokenKind, token: &str) -> Option<SessionToken> {
        let (key, prefix) = match kind {
            TokenKind::Access => (&self.access, ACCESS_TOKEN_PREFIX),
            TokenKind::Refresh => (&self.refresh, REFRESH_TOKEN_PREFIX),
        };
        let encoded = token.strip_prefix(prefix)?;
        let plaintext = open(key, prefix.as_bytes(), encoded)?;
        let payload: SessionToken = serde_json::from_slice(&plaintext).ok()?;
        (payload.kind == kind && payload.exp > unix_now()).then_some(payload)
    }

    /// The access token's contents, if it is one of this environment's and
    /// has not expired.
    pub fn open_access_token(&self, token: &str) -> Option<SessionToken> {
        self.open_token(TokenKind::Access, token)
    }

    /// As [`Self::open_access_token`], for a refresh token.
    pub fn open_refresh_token(&self, token: &str) -> Option<SessionToken> {
        self.open_token(TokenKind::Refresh, token)
    }

    /// The value of a flow cookie named after `flow.handle`.
    pub fn seal_flow(&self, flow: &FlowState) -> String {
        let plaintext = serde_json::to_vec(flow).expect("a flow state serializes");
        seal(&self.flow, flow.handle.as_bytes(), &plaintext)
    }

    /// The flow in a cookie named after `handle`, unless it has expired. The
    /// handle binds the value to its name, so a value cannot be replayed under
    /// another flow's name.
    pub fn open_flow(&self, handle: &str, value: &str) -> Option<FlowState> {
        let plaintext = open(&self.flow, handle.as_bytes(), value)?;
        let flow: FlowState = serde_json::from_slice(&plaintext).ok()?;
        (flow.expires_at > unix_now()).then_some(flow)
    }
}

/// The process's copy of the session key.
#[derive(Default)]
pub(super) struct SessionKeyCache {
    current: RwLock<Option<Arc<SessionKeys>>>,
    loading: tokio::sync::Mutex<()>,
}

impl SessionKeyCache {
    fn get(&self) -> Option<Arc<SessionKeys>> {
        self.current
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn set(&self, keys: SessionKeys) -> Arc<SessionKeys> {
        let keys = Arc::new(keys);
        *self.current.write().unwrap_or_else(|p| p.into_inner()) = Some(Arc::clone(&keys));
        keys
    }
}

fn store_failure(error: StoreError) -> UserOAuthFailure {
    tracing::warn!(target: "aisix::mcp_oauth", %error, "could not read or write the session key");
    UserOAuthFailure::Unavailable
}

impl UserOAuthManager {
    /// The session keys to verify a token presented on a request: the cached
    /// ones, or what the store holds if this process has none yet. `None` when
    /// the environment has no key, which makes every token invalid. Never
    /// writes, because whoever calls this has proven nothing yet.
    pub async fn session_keys(&self) -> Result<Option<Arc<SessionKeys>>, UserOAuthFailure> {
        if let Some(keys) = self.session_key.get() {
            return Ok(Some(keys));
        }
        let _loading = self.session_key.loading.lock().await;
        if let Some(keys) = self.session_key.get() {
            return Ok(Some(keys));
        }
        match self
            .store
            .get(NS_SESSION_KEY, SESSION_KEY_NAME)
            .await
            .map_err(store_failure)?
        {
            Some(stored) => Ok(Some(
                self.session_key.set(SessionKeys::from_stored(&stored)?),
            )),
            None => Ok(None),
        }
    }

    /// The session keys for a sign-in, creating the environment's secret if
    /// nobody has and renewing it when it nears the store's TTL. A sign-in is
    /// where this happens because a key nobody signs in with for that long
    /// has outlived every token it protected.
    pub async fn session_keys_for_sign_in(&self) -> Result<Arc<SessionKeys>, UserOAuthFailure> {
        let keys = match self.session_keys().await? {
            Some(keys) => keys,
            None => self.create_session_key().await?,
        };
        if keys.needs_renewal() {
            return Ok(self.renew_session_key(keys).await);
        }
        Ok(keys)
    }

    async fn create_session_key(&self) -> Result<Arc<SessionKeys>, UserOAuthFailure> {
        let _loading = self.session_key.loading.lock().await;
        if let Some(keys) = self.session_key.get() {
            return Ok(keys);
        }
        let mut secret = vec![0u8; SESSION_KEY_BYTES];
        rand::thread_rng().fill_bytes(&mut secret);
        match self
            .store
            .put(
                NS_SESSION_KEY,
                SESSION_KEY_NAME,
                secret.clone(),
                SESSION_KEY_TTL,
                &BTreeMap::new(),
                PutCondition::CreateOnly,
            )
            .await
        {
            Ok(_) => {
                let expires_at = unix_now() + SESSION_KEY_TTL.as_secs() as i64;
                Ok(self
                    .session_key
                    .set(SessionKeys::new(secret, Some(expires_at))))
            }
            // Another gateway created it first; its secret is the shared one.
            Err(StoreError::PreconditionFailed) => {
                let stored = self
                    .store
                    .get(NS_SESSION_KEY, SESSION_KEY_NAME)
                    .await
                    .map_err(store_failure)?
                    .ok_or(UserOAuthFailure::Unavailable)?;
                Ok(self.session_key.set(SessionKeys::from_stored(&stored)?))
            }
            Err(error) => Err(store_failure(error)),
        }
    }

    /// Extend the secret's life without changing it. A failure leaves the
    /// current key in use, and the next sign-in tries again.
    async fn renew_session_key(&self, keys: Arc<SessionKeys>) -> Arc<SessionKeys> {
        let _loading = self.session_key.loading.lock().await;
        let stored = match self.store.get(NS_SESSION_KEY, SESSION_KEY_NAME).await {
            Ok(stored) => stored,
            Err(error) => {
                store_failure(error);
                return keys;
            }
        };
        let condition = match &stored {
            Some(stored) if stored.value != keys.secret => {
                // The store's copy is the one other gateways use.
                return match SessionKeys::from_stored(stored) {
                    Ok(current) => self.session_key.set(current),
                    Err(_) => keys,
                };
            }
            Some(stored) => {
                if SessionKeys::from_stored(stored).is_ok_and(|fresh| !fresh.needs_renewal()) {
                    return self
                        .session_key
                        .set(SessionKeys::new(stored.value.clone(), stored.expires_at));
                }
                PutCondition::IfVersion(stored.version)
            }
            None => PutCondition::CreateOnly,
        };
        match self
            .store
            .put(
                NS_SESSION_KEY,
                SESSION_KEY_NAME,
                keys.secret.clone(),
                SESSION_KEY_TTL,
                &BTreeMap::new(),
                condition,
            )
            .await
        {
            Ok(_) => {
                let expires_at = unix_now() + SESSION_KEY_TTL.as_secs() as i64;
                self.session_key
                    .set(SessionKeys::new(keys.secret.clone(), Some(expires_at)))
            }
            // Someone else moved it; the next sign-in reads what they left.
            Err(StoreError::PreconditionFailed) => keys,
            Err(error) => {
                store_failure(error);
                keys
            }
        }
    }

    /// Record that a refresh token was used. `Ok(false)` when it already was:
    /// the one thing a replay is refused for, leaving the rest of its chain
    /// alone.
    pub async fn spend_refresh_token(
        &self,
        token: &SessionToken,
    ) -> Result<bool, UserOAuthFailure> {
        let labels = BTreeMap::from([("mcp_server_id", token.server.as_str())]);
        let remaining = (token.exp - unix_now()).max(1) as u64;
        match self
            .store
            .put(
                NS_REFRESH_USED,
                &token.jti,
                b"1".to_vec(),
                Duration::from_secs(remaining),
                &labels,
                PutCondition::CreateOnly,
            )
            .await
        {
            Ok(_) => Ok(true),
            Err(StoreError::PreconditionFailed) => Ok(false),
            Err(error) => {
                tracing::warn!(target: "aisix::mcp_oauth", %error,
                    "could not record the use of a refresh token");
                Err(UserOAuthFailure::Unavailable)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::flow::{FlowStage, FlowState};
    use super::*;
    use crate::shared_store::{MemoryStore, SharedStore};

    fn manager_over(store: Arc<dyn SharedStore>) -> UserOAuthManager {
        UserOAuthManager::new(store)
    }

    fn identity() -> SessionIdentity {
        SessionIdentity {
            provider: "corp".to_owned(),
            identity: "alice".to_owned(),
            claims: serde_json::json!({"sub": "alice"}),
        }
    }

    #[tokio::test]
    async fn two_gateways_on_one_store_share_a_key_created_once() {
        let store: Arc<dyn SharedStore> = Arc::new(MemoryStore::default());
        let (a, b) = (manager_over(store.clone()), manager_over(store.clone()));
        assert!(a.session_keys().await.unwrap().is_none());
        assert!(store
            .get(NS_SESSION_KEY, SESSION_KEY_NAME)
            .await
            .unwrap()
            .is_none());

        let keys_a = a.session_keys_for_sign_in().await.unwrap();
        let version = store
            .get(NS_SESSION_KEY, SESSION_KEY_NAME)
            .await
            .unwrap()
            .unwrap()
            .version;
        let keys_b = b.session_keys_for_sign_in().await.unwrap();
        assert_eq!(
            store
                .get(NS_SESSION_KEY, SESSION_KEY_NAME)
                .await
                .unwrap()
                .unwrap()
                .version,
            version
        );

        let pair = keys_a.mint_pair("server", "client", &identity());
        let opened = keys_b.open_access_token(&pair.access_token).unwrap();
        assert_eq!(opened.identity, "alice");
    }

    /// Answers a create-only write as though another gateway just won it.
    struct LosesTheRace {
        inner: MemoryStore,
    }

    #[async_trait::async_trait]
    impl SharedStore for LosesTheRace {
        async fn get(&self, ns: &str, key: &str) -> Result<Option<StoredValue>, StoreError> {
            self.inner.get(ns, key).await
        }
        async fn put(
            &self,
            ns: &str,
            key: &str,
            _value: Vec<u8>,
            ttl: Duration,
            labels: &BTreeMap<&str, &str>,
            _condition: PutCondition,
        ) -> Result<i64, StoreError> {
            self.inner
                .put(
                    ns,
                    key,
                    vec![7u8; SESSION_KEY_BYTES],
                    ttl,
                    labels,
                    PutCondition::CreateOnly,
                )
                .await?;
            Err(StoreError::PreconditionFailed)
        }
        async fn delete(&self, ns: &str, key: &str, v: Option<i64>) -> Result<(), StoreError> {
            self.inner.delete(ns, key, v).await
        }
    }

    #[tokio::test]
    async fn losing_the_create_race_adopts_the_winners_key() {
        let store = Arc::new(LosesTheRace {
            inner: MemoryStore::default(),
        });
        let manager = manager_over(store.clone());
        let keys = manager.session_keys_for_sign_in().await.unwrap();
        assert_eq!(keys.secret, vec![7u8; SESSION_KEY_BYTES]);
    }

    #[tokio::test]
    async fn a_key_under_thirty_days_is_renewed_with_the_same_value() {
        let store: Arc<dyn SharedStore> = Arc::new(MemoryStore::default());
        let secret = vec![9u8; SESSION_KEY_BYTES];
        store
            .put(
                NS_SESSION_KEY,
                SESSION_KEY_NAME,
                secret.clone(),
                Duration::from_secs(24 * 3600),
                &BTreeMap::new(),
                PutCondition::CreateOnly,
            )
            .await
            .unwrap();
        let manager = manager_over(store.clone());
        let keys = manager.session_keys_for_sign_in().await.unwrap();
        assert_eq!(keys.secret, secret);
        let stored = store
            .get(NS_SESSION_KEY, SESSION_KEY_NAME)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.value, secret);
        assert!(stored.expires_at.unwrap() > unix_now() + 300 * 24 * 3600);
    }

    #[tokio::test]
    async fn a_held_key_keeps_verifying_through_an_outage_but_no_key_is_an_error() {
        let store = Arc::new(super::super::tests::SwitchableStore::default());
        let manager = manager_over(store.clone());
        let keys = manager.session_keys_for_sign_in().await.unwrap();
        let pair = keys.mint_pair("server", "client", &identity());
        store.fail(true);
        let held = manager.session_keys().await.unwrap().unwrap();
        assert!(held.open_access_token(&pair.access_token).is_some());

        let fresh = manager_over(store.clone());
        assert!(matches!(
            fresh.session_keys().await,
            Err(UserOAuthFailure::Unavailable)
        ));
        assert!(matches!(
            fresh.session_keys_for_sign_in().await,
            Err(UserOAuthFailure::Unavailable)
        ));
    }

    #[test]
    fn a_token_opens_only_as_its_own_kind_and_under_its_own_key() {
        let keys = SessionKeys::new(vec![1u8; SESSION_KEY_BYTES], None);
        let other = SessionKeys::new(vec![2u8; SESSION_KEY_BYTES], None);
        let pair = keys.mint_pair("server", "client", &identity());
        assert!(pair.access_token.starts_with(ACCESS_TOKEN_PREFIX));
        assert!(pair.refresh_token.starts_with(REFRESH_TOKEN_PREFIX));
        assert!(!pair.access_token.contains('.'));
        assert!(keys.open_access_token(&pair.access_token).is_some());
        assert!(keys.open_refresh_token(&pair.refresh_token).is_some());
        assert!(keys.open_access_token(&pair.refresh_token).is_none());
        assert!(keys.open_refresh_token(&pair.access_token).is_none());
        assert!(other.open_access_token(&pair.access_token).is_none());
        // The same ciphertext under the other kind's prefix uses another key.
        let relabelled = pair
            .refresh_token
            .replacen(REFRESH_TOKEN_PREFIX, ACCESS_TOKEN_PREFIX, 1);
        assert!(keys.open_access_token(&relabelled).is_none());
        assert_ne!(
            keys.open_access_token(&pair.access_token).unwrap().jti,
            keys.open_refresh_token(&pair.refresh_token).unwrap().jti
        );
    }

    #[test]
    fn an_expired_token_does_not_open() {
        let keys = SessionKeys::new(vec![1u8; SESSION_KEY_BYTES], None);
        let now = unix_now();
        let expired = keys.mint(
            TokenKind::Access,
            &SessionToken {
                kind: TokenKind::Access,
                server: "s".to_owned(),
                client_id: "c".to_owned(),
                provider: "p".to_owned(),
                identity: "i".to_owned(),
                jti: "j".to_owned(),
                iat: now - 10,
                exp: now - 1,
                claims: serde_json::json!({}),
            },
        );
        assert!(keys.open_access_token(&expired).is_none());
    }

    #[test]
    fn a_flow_cookie_is_bound_to_its_handle_and_its_expiry() {
        let keys = SessionKeys::new(vec![1u8; SESSION_KEY_BYTES], None);
        let flow = |handle: &str, expires_at| FlowState {
            handle: handle.to_owned(),
            expires_at,
            server_id: "s".to_owned(),
            server_name: "s".to_owned(),
            client_id: "c".to_owned(),
            redirect_uri: "http://127.0.0.1/cb".to_owned(),
            client_state: None,
            code_challenge: "x".to_owned(),
            csrf: "y".to_owned(),
            stage: FlowStage::Consent {
                signed_in: None,
                relayed: None,
            },
        };
        let live = keys.seal_flow(&flow("aa", unix_now() + 60));
        assert!(keys.open_flow("aa", &live).is_some());
        assert!(keys.open_flow("bb", &live).is_none());
        let expired = keys.seal_flow(&flow("aa", unix_now() - 1));
        assert!(keys.open_flow("aa", &expired).is_none());
    }

    #[tokio::test]
    async fn a_refresh_token_is_spent_once_for_its_remaining_life() {
        let store: Arc<dyn SharedStore> = Arc::new(MemoryStore::default());
        let manager = manager_over(store.clone());
        let keys = manager.session_keys_for_sign_in().await.unwrap();
        let pair = keys.mint_pair("server", "client", &identity());
        let token = keys.open_refresh_token(&pair.refresh_token).unwrap();
        assert!(manager.spend_refresh_token(&token).await.unwrap());
        assert!(!manager.spend_refresh_token(&token).await.unwrap());
        let stored = store
            .get(NS_REFRESH_USED, &token.jti)
            .await
            .unwrap()
            .unwrap();
        assert!(stored.expires_at.unwrap() <= token.exp);
    }
}
