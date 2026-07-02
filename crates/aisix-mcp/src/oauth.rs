//! OAuth 2.0 client-credentials token minting for upstream MCP servers.
//!
//! When an upstream is registered with `auth_type: oauth2`, the gateway mints
//! its own access token at the server's token endpoint (RFC 6749 §4.4, the
//! machine-to-machine grant) and presents it as `Authorization: Bearer
//! <access_token>`. The token is a gateway-held credential: the calling
//! agent's AISIX key is never forwarded upstream, in line with the MCP
//! authorization spec's no-token-passthrough requirement.
//!
//! Tokens are cached process-globally per `(token_url, client_id,
//! client_secret)` triple and reused until shortly before their reported
//! expiry; [`invalidate`] drops an entry early when the upstream rejects the
//! token so the next attempt re-mints.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::bridge::{OAuthClientConfig, DEFAULT_UPSTREAM_TIMEOUT};
use crate::error::McpError;

/// Stop reusing a cached token this long before its reported expiry, so an
/// upstream operation never starts with a token about to lapse mid-flight.
const EXPIRY_SKEW: Duration = Duration::from_secs(60);

/// Assumed lifetime when the token endpoint omits `expires_in` (RFC 6749 §5.1
/// only recommends it).
const DEFAULT_TOKEN_LIFETIME: Duration = Duration::from_secs(3600);

/// Cap on a reported `expires_in`. A hostile or broken identity provider can
/// return any u64; an absurd value would overflow `Instant + Duration` and
/// panic the connect task. Thirty days is beyond any sane token lifetime and
/// far below the overflow horizon.
const MAX_TOKEN_LIFETIME: Duration = Duration::from_secs(30 * 24 * 3600);

/// One minted upstream access token. Never printed: the struct deliberately
/// has no `Debug` impl, and nothing in this module logs the token value.
struct CachedToken {
    access_token: String,
    expires_at: Instant,
}

/// Process-global token cache. Guards are held only for map lookups/inserts,
/// never across an await; concurrent misses for the same key may fetch in
/// parallel (each minted token is valid — last insert wins). Entries are
/// evicted only by 401-invalidation or key rotation, so tokens for deleted
/// servers linger until restart — the map is bounded by the set of distinct
/// configs ever seen (~100 bytes each), which is acceptable.
fn cache() -> &'static RwLock<HashMap<String, CachedToken>> {
    static CACHE: OnceLock<RwLock<HashMap<String, CachedToken>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Shared HTTP client for token fetches, bounded per request by the same
/// default deadline as any other upstream MCP operation. Redirects are
/// disabled: a token endpoint never legitimately redirects, and following one
/// would re-POST the secret-bearing form to wherever it points (307/308 keep
/// the body; a chain may even downgrade to plain HTTP).
fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(DEFAULT_UPSTREAM_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            // Building a client with only a timeout and redirect policy set
            // cannot fail on any supported platform; fall back to the default
            // client if it does.
            .unwrap_or_default()
    })
}

/// Cache key over every field that shapes the minted token: token endpoint,
/// client identity, client secret, and the requested scopes. Two servers
/// sharing one OAuth client but requesting different scopes must never share
/// a token (one would be presented a token minted for the other's scopes).
/// The secret is folded in as a digest — never in plaintext — so a rotated
/// secret can never reuse the previous secret's token. Each component is
/// length-prefixed so no crafted field value can shift a boundary and make
/// two distinct configs collide.
fn cache_key(cfg: &OAuthClientConfig) -> String {
    let secret_digest = hex::encode(Sha256::digest(cfg.client_secret.as_bytes()));
    let joined_scopes = cfg.scopes.join(" ");
    format!(
        "{}:{}\x1f{}:{}\x1f{}\x1f{}:{}",
        cfg.token_url.len(),
        cfg.token_url,
        cfg.client_id.len(),
        cfg.client_id,
        secret_digest,
        joined_scopes.len(),
        joined_scopes
    )
}

/// Return the cached access token for `cfg`, minting a fresh one at the token
/// endpoint on a miss or when the cached token is within [`EXPIRY_SKEW`] of
/// expiry.
///
/// A config missing `client_id`, the client secret, or `token_url` fails here
/// with a clean error (no panic): the mis-configured server simply becomes
/// unavailable, like any upstream that cannot be reached.
pub(crate) async fn get_or_fetch(cfg: &OAuthClientConfig) -> Result<String, McpError> {
    if cfg.token_url.is_empty() || cfg.client_id.is_empty() || cfg.client_secret.is_empty() {
        return Err(McpError::Connect(
            "oauth2 upstream auth requires client_id, secret (the OAuth client secret), \
             and token_url"
                .to_string(),
        ));
    }

    let key = cache_key(cfg);
    if let Some(token) = cached_token(&key) {
        return Ok(token);
    }

    let (access_token, lifetime) = fetch_token(cfg).await?;
    let entry = CachedToken {
        access_token: access_token.clone(),
        expires_at: Instant::now() + lifetime,
    };
    cache()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(key, entry);
    Ok(access_token)
}

/// Drop the cached token for `cfg`, forcing the next [`get_or_fetch`] to
/// re-mint. Called when the upstream rejects the presented token with `401`
/// (revoked, or expired earlier than `expires_in` promised).
pub(crate) fn invalidate(cfg: &OAuthClientConfig) {
    cache()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&cache_key(cfg));
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

/// Token endpoint success payload (RFC 6749 §5.1). Unknown fields (`scope`,
/// `refresh_token`, …) are ignored; `token_type` is not needed — the gateway
/// always presents the token as `Bearer`.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

/// POST the client-credentials grant to the token endpoint and return the
/// minted `(access_token, lifetime)`.
///
/// Error hygiene: the request carries the client secret, so the form body is
/// never logged or echoed into errors; error responses are reported by status
/// only (identity providers may reflect request parameters into their error
/// payloads).
async fn fetch_token(cfg: &OAuthClientConfig) -> Result<(String, Duration), McpError> {
    let mut form: Vec<(&str, &str)> = vec![
        ("grant_type", "client_credentials"),
        ("client_id", cfg.client_id.as_str()),
        ("client_secret", cfg.client_secret.as_str()),
    ];
    let joined_scopes;
    if !cfg.scopes.is_empty() {
        joined_scopes = cfg.scopes.join(" ");
        form.push(("scope", joined_scopes.as_str()));
    }

    let response = http_client()
        .post(&cfg.token_url)
        .form(&form)
        .send()
        .await
        .map_err(|e| {
            // reqwest transport errors describe the connection (and may name
            // the non-secret token_url), never the form body.
            McpError::Connect(format!("upstream OAuth token request failed: {e}"))
        })?;

    let status = response.status();
    if !status.is_success() {
        return Err(McpError::Connect(format!(
            "upstream OAuth token endpoint returned HTTP {status}"
        )));
    }

    let token: TokenResponse = response.json().await.map_err(|_| {
        McpError::Connect(
            "upstream OAuth token endpoint returned a malformed token response".to_string(),
        )
    })?;
    if token.access_token.is_empty() {
        return Err(McpError::Connect(
            "upstream OAuth token endpoint returned an empty access_token".to_string(),
        ));
    }
    let lifetime = token
        .expires_in
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_TOKEN_LIFETIME)
        .min(MAX_TOKEN_LIFETIME);
    Ok((token.access_token, lifetime))
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;

    /// How a test token endpoint answers.
    #[derive(Clone)]
    enum TokenEndpointBehavior {
        /// `200` with `access_token: "tok-<n>"` (`n` = 1-based request count)
        /// and this `expires_in` (`None` → field omitted).
        Mint { expires_in: Option<u64> },
        /// A fixed status + fixed body.
        Static {
            status: axum::http::StatusCode,
            body: &'static str,
        },
    }

    #[derive(Clone)]
    struct TokenEndpointState {
        behavior: TokenEndpointBehavior,
        hits: Arc<AtomicUsize>,
        /// Every decoded form body, for asserting the request shape.
        requests: Arc<Mutex<Vec<HashMap<String, String>>>>,
    }

    struct TokenEndpoint {
        addr: SocketAddr,
        hits: Arc<AtomicUsize>,
        requests: Arc<Mutex<Vec<HashMap<String, String>>>>,
    }

    impl TokenEndpoint {
        fn url(&self) -> String {
            format!("http://{}/oauth/token", self.addr)
        }

        fn hits(&self) -> usize {
            self.hits.load(Ordering::SeqCst)
        }

        fn request(&self, index: usize) -> HashMap<String, String> {
            self.requests.lock().expect("requests lock")[index].clone()
        }
    }

    async fn handle_token_request(
        axum::extract::State(state): axum::extract::State<TokenEndpointState>,
        axum::extract::Form(form): axum::extract::Form<HashMap<String, String>>,
    ) -> axum::response::Response {
        use axum::response::IntoResponse;
        let n = state.hits.fetch_add(1, Ordering::SeqCst) + 1;
        state.requests.lock().expect("requests lock").push(form);
        match &state.behavior {
            TokenEndpointBehavior::Mint { expires_in } => {
                let mut body = serde_json::json!({
                    "access_token": format!("tok-{n}"),
                    "token_type": "Bearer",
                });
                if let Some(expires_in) = expires_in {
                    body["expires_in"] = (*expires_in).into();
                }
                axum::Json(body).into_response()
            }
            TokenEndpointBehavior::Static { status, body } => (*status, *body).into_response(),
        }
    }

    /// Stand up a real token endpoint on an ephemeral port. Each test gets
    /// its own port, so `token_url`-keyed cache entries never collide across
    /// tests sharing the process-global cache.
    async fn spawn_token_endpoint(behavior: TokenEndpointBehavior) -> TokenEndpoint {
        let state = TokenEndpointState {
            behavior,
            hits: Arc::new(AtomicUsize::new(0)),
            requests: Arc::new(Mutex::new(Vec::new())),
        };
        let hits = state.hits.clone();
        let requests = state.requests.clone();
        let app = axum::Router::new()
            .route("/oauth/token", axum::routing::post(handle_token_request))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve");
        });
        TokenEndpoint {
            addr,
            hits,
            requests,
        }
    }

    fn config(token_url: String) -> OAuthClientConfig {
        OAuthClientConfig {
            client_id: "cid".to_string(),
            client_secret: "s3cret".to_string(),
            token_url,
            scopes: Vec::new(),
        }
    }

    #[tokio::test]
    async fn mints_token_with_client_credentials_form() {
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Mint {
            expires_in: Some(3600),
        })
        .await;
        let cfg = config(endpoint.url());

        let token = get_or_fetch(&cfg).await.expect("mint token");
        assert_eq!(token, "tok-1");

        // The grant is the urlencoded client-credentials shape (RFC 6749
        // §4.4.2), with no `scope` parameter when no scopes are configured.
        let form = endpoint.request(0);
        assert_eq!(
            form.get("grant_type").map(String::as_str),
            Some("client_credentials")
        );
        assert_eq!(form.get("client_id").map(String::as_str), Some("cid"));
        assert_eq!(
            form.get("client_secret").map(String::as_str),
            Some("s3cret")
        );
        assert!(
            !form.contains_key("scope"),
            "no scope param when scopes are empty"
        );
    }

    #[tokio::test]
    async fn second_call_within_expiry_hits_the_cache() {
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Mint {
            expires_in: Some(3600),
        })
        .await;
        let cfg = config(endpoint.url());

        assert_eq!(get_or_fetch(&cfg).await.expect("first"), "tok-1");
        assert_eq!(get_or_fetch(&cfg).await.expect("second"), "tok-1");
        assert_eq!(endpoint.hits(), 1, "second call must be served from cache");
    }

    #[tokio::test]
    async fn missing_expires_in_defaults_to_a_cacheable_lifetime() {
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Mint { expires_in: None }).await;
        let cfg = config(endpoint.url());

        assert_eq!(get_or_fetch(&cfg).await.expect("first"), "tok-1");
        assert_eq!(get_or_fetch(&cfg).await.expect("second"), "tok-1");
        assert_eq!(endpoint.hits(), 1, "default lifetime must allow caching");
    }

    #[tokio::test]
    async fn token_expiring_within_the_skew_is_refetched() {
        // `expires_in` below EXPIRY_SKEW: valid to hand out now, but stale
        // for reuse — the second call must mint a fresh token (no sleeping).
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Mint {
            expires_in: Some(10),
        })
        .await;
        let cfg = config(endpoint.url());

        assert_eq!(get_or_fetch(&cfg).await.expect("first"), "tok-1");
        assert_eq!(get_or_fetch(&cfg).await.expect("second"), "tok-2");
        assert_eq!(endpoint.hits(), 2, "near-expiry token must be refetched");
    }

    #[tokio::test]
    async fn rotated_client_secret_never_reuses_the_old_token() {
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Mint {
            expires_in: Some(3600),
        })
        .await;
        let cfg = config(endpoint.url());
        assert_eq!(get_or_fetch(&cfg).await.expect("first"), "tok-1");

        let rotated = OAuthClientConfig {
            client_secret: "rotated".to_string(),
            ..cfg.clone()
        };
        assert_eq!(
            get_or_fetch(&rotated).await.expect("rotated"),
            "tok-2",
            "a rotated secret is a different cache key and must re-mint"
        );
        assert_eq!(endpoint.hits(), 2);

        // The original secret's entry is untouched.
        assert_eq!(get_or_fetch(&cfg).await.expect("original"), "tok-1");
        assert_eq!(endpoint.hits(), 2);
    }

    #[tokio::test]
    async fn invalidate_forces_a_refetch() {
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Mint {
            expires_in: Some(3600),
        })
        .await;
        let cfg = config(endpoint.url());

        assert_eq!(get_or_fetch(&cfg).await.expect("first"), "tok-1");
        invalidate(&cfg);
        assert_eq!(get_or_fetch(&cfg).await.expect("after invalidate"), "tok-2");
        assert_eq!(endpoint.hits(), 2);
    }

    #[tokio::test]
    async fn scopes_are_joined_with_spaces() {
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Mint {
            expires_in: Some(3600),
        })
        .await;
        let cfg = OAuthClientConfig {
            scopes: vec!["read".to_string(), "write".to_string()],
            ..config(endpoint.url())
        };

        get_or_fetch(&cfg).await.expect("mint token");
        let form = endpoint.request(0);
        assert_eq!(form.get("scope").map(String::as_str), Some("read write"));
    }

    #[tokio::test]
    async fn error_statuses_and_malformed_bodies_fail_cleanly() {
        // 500 with a body that must never be echoed into the error.
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Static {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            body: "identity provider exploded: client_secret=echoed-back",
        })
        .await;
        let cfg = config(endpoint.url());
        let err = get_or_fetch(&cfg).await.expect_err("500 must fail");
        let msg = err.to_string();
        assert!(msg.contains("HTTP 500"), "status should be reported: {msg}");
        assert!(
            !msg.contains("echoed-back") && !msg.contains("s3cret"),
            "neither the response body nor the secret may leak: {msg}"
        );

        // 200 but not JSON.
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Static {
            status: axum::http::StatusCode::OK,
            body: "not json",
        })
        .await;
        let err = get_or_fetch(&config(endpoint.url()))
            .await
            .expect_err("malformed body must fail");
        assert!(err.to_string().contains("malformed"), "got: {err}");

        // 200 JSON but no access_token.
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Static {
            status: axum::http::StatusCode::OK,
            body: r#"{"token_type":"Bearer"}"#,
        })
        .await;
        let err = get_or_fetch(&config(endpoint.url()))
            .await
            .expect_err("missing access_token must fail");
        assert!(err.to_string().contains("malformed"), "got: {err}");
    }

    #[tokio::test]
    async fn same_client_different_scopes_mint_distinct_tokens() {
        // Two servers sharing one OAuth client (same IdP, client_id, secret)
        // but requesting different scopes must never share a cached token —
        // one would be presented a token minted for the other's scopes.
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Mint {
            expires_in: Some(3600),
        })
        .await;
        let read_cfg = OAuthClientConfig {
            scopes: vec!["read".to_string()],
            ..config(endpoint.url())
        };
        let write_cfg = OAuthClientConfig {
            scopes: vec!["write".to_string()],
            ..config(endpoint.url())
        };

        assert_eq!(get_or_fetch(&read_cfg).await.expect("read"), "tok-1");
        assert_eq!(get_or_fetch(&write_cfg).await.expect("write"), "tok-2");
        assert_eq!(endpoint.hits(), 2, "different scopes must mint separately");
        assert_eq!(
            endpoint.request(1).get("scope").map(String::as_str),
            Some("write")
        );
        // Each scope set keeps its own cached token.
        assert_eq!(get_or_fetch(&read_cfg).await.expect("read again"), "tok-1");
        assert_eq!(endpoint.hits(), 2);
    }

    #[tokio::test]
    async fn absurd_expires_in_is_clamped_not_panicking() {
        // A hostile IdP can return any u64; an unclamped
        // `Instant::now() + Duration::from_secs(u64::MAX)` overflows and
        // panics the connect task.
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Mint {
            expires_in: Some(u64::MAX),
        })
        .await;
        let cfg = config(endpoint.url());
        assert_eq!(get_or_fetch(&cfg).await.expect("clamped mint"), "tok-1");
        // Still cached (clamped lifetime is far above the skew).
        assert_eq!(get_or_fetch(&cfg).await.expect("cached"), "tok-1");
        assert_eq!(endpoint.hits(), 1);
    }

    #[tokio::test]
    async fn token_endpoint_redirects_are_refused() {
        // A token endpoint never legitimately redirects; following one would
        // re-POST the secret-bearing form to wherever it points. The client
        // must refuse and fail with the redirect status, not follow it.
        let endpoint = spawn_token_endpoint(TokenEndpointBehavior::Static {
            status: axum::http::StatusCode::TEMPORARY_REDIRECT,
            body: "",
        })
        .await;
        let err = get_or_fetch(&config(endpoint.url()))
            .await
            .expect_err("redirect must fail");
        assert!(
            err.to_string().contains("HTTP 307"),
            "redirect should surface as its status: {err}"
        );
        assert_eq!(endpoint.hits(), 1, "the redirect must not be followed");
    }

    #[tokio::test]
    async fn incomplete_config_fails_without_contacting_anything() {
        for broken in [
            OAuthClientConfig {
                token_url: String::new(),
                ..config("ignored".to_string())
            },
            OAuthClientConfig {
                client_id: String::new(),
                ..config("http://127.0.0.1:9/oauth/token".to_string())
            },
            OAuthClientConfig {
                client_secret: String::new(),
                ..config("http://127.0.0.1:9/oauth/token".to_string())
            },
        ] {
            let err = get_or_fetch(&broken)
                .await
                .expect_err("incomplete oauth2 config must fail cleanly");
            assert!(
                err.to_string().contains("oauth2"),
                "error should name the misconfiguration: {err}"
            );
        }
    }
}
