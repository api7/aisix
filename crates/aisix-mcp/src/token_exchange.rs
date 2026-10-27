//! OAuth token exchange on behalf of the caller (`auth_type:
//! oauth2_token_exchange`).
//!
//! The gateway trades the caller's own bearer token (the *subject*, which it
//! verified itself through a JWT trust provider on the same request) for an
//! access token issued for the upstream server, and presents that token
//! upstream. The subject never leaves the gateway except towards the token
//! endpoint.
//!
//! Exchanged tokens are cached process-globally, keyed on a digest of the
//! subject token, the whole exchange configuration and the server, so two
//! callers can never share an entry, and reused until shortly before their
//! reported expiry. [`invalidate`] drops an entry when the upstream rejects
//! the token.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

use aisix_core::McpTokenExchangeProfile;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::{McpError, TokenExchangeFailure};
use crate::oauth::{DEFAULT_TOKEN_LIFETIME, EXPIRY_SKEW, MAX_TOKEN_LIFETIME};

/// RFC 7523 JWT bearer grant, which Microsoft Entra ID's On-Behalf-Of flow
/// is built on.
const JWT_BEARER_GRANT: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";

/// RFC 6749 §5.2 error codes that blame the gateway's own client
/// registration or configuration rather than the caller's token, so a
/// fresh caller sign-in cannot fix them.
const GATEWAY_FAULT_ERRORS: &[&str] = &[
    "invalid_client",
    "unauthorized_client",
    "unsupported_grant_type",
    "invalid_target",
    "invalid_scope",
];

/// RFC 8693 token types that are not an access token, which the gateway
/// must not present upstream as a bearer even if the endpoint labels it so.
const NON_ACCESS_TOKEN_TYPES: &[&str] = &[
    "urn:ietf:params:oauth:token-type:refresh_token",
    "urn:ietf:params:oauth:token-type:id_token",
    "urn:ietf:params:oauth:token-type:saml1",
    "urn:ietf:params:oauth:token-type:saml2",
];

/// Ceiling on cached exchanged tokens. Entries are per caller token, so
/// unlike the client-credentials cache this one grows with traffic; past
/// the ceiling, expired entries are purged first and then the ones closest
/// to expiry.
const MAX_CACHED_TOKENS: usize = 10_000;

/// Exchange parameters for one upstream server.
#[derive(Clone)]
pub struct TokenExchangeConfig {
    /// `None` only on a row the write path would have refused; it fails
    /// as misconfigured rather than guess a protocol.
    pub profile: Option<McpTokenExchangeProfile>,
    /// Token endpoint URL (non-secret).
    pub token_url: String,
    /// The gateway's OAuth client identifier (non-secret).
    pub client_id: String,
    /// The gateway's OAuth client secret. Redacted from `Debug`.
    pub client_secret: String,
    /// Scopes requested for the upstream token, joined with spaces.
    pub scopes: Vec<String>,
    /// Identifies the upstream server the exchanged token is for, folded
    /// into the cache key.
    pub server: String,
}

impl std::fmt::Debug for TokenExchangeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenExchangeConfig")
            .field("profile", &self.profile)
            .field("token_url", &self.token_url)
            .field("client_id", &self.client_id)
            .field("client_secret", &"***redacted***")
            .field("scopes", &self.scopes)
            .field("server", &self.server)
            .finish()
    }
}

struct CachedToken {
    access_token: String,
    expires_at: Instant,
}

fn cache() -> &'static RwLock<HashMap<String, CachedToken>> {
    static CACHE: OnceLock<RwLock<HashMap<String, CachedToken>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// SHA-256 over every input that shapes the exchanged token, each
/// length-prefixed so no field value can shift a boundary. Nothing secret
/// is held in the key itself.
fn cache_key(cfg: &TokenExchangeConfig, subject: &str) -> String {
    let profile = match cfg.profile {
        Some(McpTokenExchangeProfile::EntraObo) => "entra_obo",
        None => "",
    };
    let scopes = cfg.scopes.join(" ");
    let mut hasher = Sha256::new();
    for part in [
        subject,
        profile,
        cfg.token_url.as_str(),
        cfg.client_id.as_str(),
        cfg.client_secret.as_str(),
        scopes.as_str(),
        cfg.server.as_str(),
    ] {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part.as_bytes());
    }
    hex::encode(hasher.finalize())
}

fn failure(kind: TokenExchangeFailure) -> McpError {
    McpError::TokenExchange(kind)
}

/// The upstream access token for `subject` under `cfg`, from the cache or
/// a fresh exchange. `None` subject — the caller was not authenticated
/// through a JWT trust provider — fails without contacting anyone.
pub(crate) async fn get_or_exchange(
    cfg: &TokenExchangeConfig,
    subject: Option<&str>,
) -> Result<String, McpError> {
    let Some(subject) = subject else {
        return Err(failure(TokenExchangeFailure::NoSubject));
    };
    if cfg.profile.is_none()
        || cfg.token_url.is_empty()
        || cfg.client_id.is_empty()
        || cfg.client_secret.is_empty()
        || cfg.scopes.is_empty()
    {
        return Err(failure(TokenExchangeFailure::Misconfigured));
    }
    let key = cache_key(cfg, subject);
    if let Some(token) = cached_token(&key) {
        return Ok(token);
    }
    let (access_token, lifetime) = exchange(cfg, subject).await?;
    let mut cache = cache()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if cache.len() >= MAX_CACHED_TOKENS {
        shrink(&mut cache);
    }
    cache.insert(
        key,
        CachedToken {
            access_token: access_token.clone(),
            expires_at: Instant::now() + lifetime,
        },
    );
    Ok(access_token)
}

/// Drop the cached token for `(cfg, subject)` so the next call exchanges
/// again — called when the upstream rejects the token.
pub(crate) fn invalidate(cfg: &TokenExchangeConfig, subject: Option<&str>) {
    let Some(subject) = subject else { return };
    cache()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&cache_key(cfg, subject));
}

fn cached_token(key: &str) -> Option<String> {
    let cache = cache()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = cache.get(key)?;
    if entry.expires_at <= Instant::now() + EXPIRY_SKEW {
        return None;
    }
    Some(entry.access_token.clone())
}

fn shrink(cache: &mut HashMap<String, CachedToken>) {
    let now = Instant::now();
    cache.retain(|_, entry| entry.expires_at > now + EXPIRY_SKEW);
    if cache.len() < MAX_CACHED_TOKENS {
        return;
    }
    let mut by_expiry: Vec<(Instant, String)> = cache
        .iter()
        .map(|(key, entry)| (entry.expires_at, key.clone()))
        .collect();
    by_expiry.sort_unstable();
    for (_, key) in by_expiry.into_iter().take(MAX_CACHED_TOKENS / 10 + 1) {
        cache.remove(&key);
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    #[serde(default)]
    access_token: Option<String>,
    #[serde(default)]
    token_type: Option<String>,
    #[serde(default)]
    issued_token_type: Option<String>,
    #[serde(default)]
    expires_in: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ErrorResponse {
    #[serde(default)]
    error: Option<String>,
}

/// POST the exchange and classify the outcome.
///
/// The form carries the client secret and the caller's token, so neither
/// the form nor the endpoint's error body is ever logged or echoed; only
/// the status and the RFC 6749 `error` code are.
async fn exchange(
    cfg: &TokenExchangeConfig,
    subject: &str,
) -> Result<(String, Duration), McpError> {
    let scope = cfg.scopes.join(" ");
    let form: Vec<(&str, &str)> = match cfg.profile {
        None => return Err(failure(TokenExchangeFailure::Misconfigured)),
        Some(McpTokenExchangeProfile::EntraObo) => vec![
            ("grant_type", JWT_BEARER_GRANT),
            ("assertion", subject),
            ("scope", scope.as_str()),
            ("requested_token_use", "on_behalf_of"),
            ("client_id", cfg.client_id.as_str()),
            ("client_secret", cfg.client_secret.as_str()),
        ],
    };
    let response = match crate::oauth::http_client()
        .post(&cfg.token_url)
        .header(http::header::ACCEPT, "application/json")
        .form(&form)
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => {
            tracing::warn!(
                server = %cfg.server,
                error = %crate::bridge::sanitize_error_message(&e.to_string()),
                "upstream token exchange request failed"
            );
            return Err(failure(TokenExchangeFailure::Unavailable));
        }
    };
    let status = response.status();
    if status.is_client_error() {
        let code = response
            .json::<ErrorResponse>()
            .await
            .ok()
            .and_then(|body| body.error);
        let gateway_fault = code
            .as_deref()
            .is_some_and(|code| GATEWAY_FAULT_ERRORS.contains(&code));
        // Only a code from the fixed list above is logged verbatim; any
        // other value is caller-influenced IdP text.
        tracing::warn!(
            server = %cfg.server,
            status = status.as_u16(),
            error = code.as_deref().filter(|_| gateway_fault).unwrap_or("-"),
            "upstream token exchange rejected"
        );
        return Err(failure(if gateway_fault {
            TokenExchangeFailure::Misconfigured
        } else {
            TokenExchangeFailure::SubjectRejected
        }));
    }
    if !status.is_success() {
        tracing::warn!(
            server = %cfg.server,
            status = status.as_u16(),
            "upstream token exchange failed"
        );
        return Err(failure(TokenExchangeFailure::Unavailable));
    }
    let unusable = |reason: &str| {
        tracing::warn!(server = %cfg.server, "upstream token exchange returned {reason}");
        failure(TokenExchangeFailure::Unavailable)
    };
    let Ok(token) = response.json::<TokenResponse>().await else {
        return Err(unusable("a malformed token response"));
    };
    let Some(access_token) = token.access_token.filter(|t| !t.is_empty()) else {
        return Err(unusable("no access_token"));
    };
    // Presented upstream as a bearer token, so a declared non-bearer type
    // or a non-access issued type must not be forwarded as one.
    if token
        .token_type
        .as_deref()
        .is_some_and(|t| !t.trim().eq_ignore_ascii_case("bearer"))
    {
        return Err(unusable("a non-bearer token_type"));
    }
    if token
        .issued_token_type
        .as_deref()
        .is_some_and(|t| NON_ACCESS_TOKEN_TYPES.contains(&t))
    {
        return Err(unusable("a token that is not an access token"));
    }
    let lifetime = token
        .expires_in
        .as_ref()
        .and_then(expires_in_secs)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_TOKEN_LIFETIME)
        .min(MAX_TOKEN_LIFETIME);
    Ok((access_token, lifetime))
}

/// `expires_in` as seconds. Entra ID returns it as a number; some
/// endpoints send a numeric string.
fn expires_in_secs(value: &serde_json::Value) -> Option<u64> {
    match value {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Shared handle used by the bridges: the config plus the caller's subject.
#[derive(Clone, Debug)]
pub struct TokenExchangeAuth {
    pub config: TokenExchangeConfig,
    pub subject: Option<SubjectToken>,
}

/// The caller's verified bearer token. Never printed.
#[derive(Clone)]
pub struct SubjectToken(pub Arc<str>);

impl std::fmt::Debug for SubjectToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SubjectToken(***redacted***)")
    }
}

impl TokenExchangeAuth {
    fn subject(&self) -> Option<&str> {
        self.subject.as_ref().map(|s| &*s.0)
    }

    pub(crate) async fn token(&self) -> Result<String, McpError> {
        get_or_exchange(&self.config, self.subject()).await
    }

    pub(crate) fn invalidate(&self) {
        invalidate(&self.config, self.subject());
    }
}
