//! GCP service-account → OAuth2 access token minting + in-process cache.
//!
//! Maps a service-account JSON credential into a short-lived bearer
//! token via the standard JWT-bearer assertion grant flow.
//!
//! First, build a JWT with claims `{iss, scope, aud, iat, exp}` and
//! sign with RS256 using the SA's RSA `private_key`. Then POST
//! `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer` plus
//! `assertion=<jwt>` to the SA's `token_uri` (typically
//! `https://oauth2.googleapis.com/token`). Parse `{access_token,
//! expires_in}` from the response. Cache keyed by SA `client_email`
//! with TTL refresh ~60s before expiry so an in-flight request never
//! lands on an expired token.
//!
//! # References
//!
//! - GCP OAuth2 SA flow:
//!   <https://developers.google.com/identity/protocols/oauth2/service-account#authorizingrequests>
//! - JWT Bearer grant (RFC 7523):
//!   <https://www.rfc-editor.org/rfc/rfc7523>
//! - Standard SA JSON shape: emitted verbatim by
//!   `gcloud iam service-accounts keys create`.

use aisix_gateway::BridgeError;
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Default scope for Vertex AI access — same as gcloud + the official
/// python-aiplatform SDK default. Narrower scopes (e.g.
/// `aiplatform.googleapis.com/cloud-platform`) work too but `cloud-
/// platform` is the SA-key default GCP recommends.
const VERTEX_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// JWT validity window per Google's docs: 1 hour. The returned token
/// has its OWN TTL (also typically 1h); cache uses that, not this.
const JWT_EXPIRY_SECS: u64 = 3600;

/// Refresh cached tokens at least this many seconds before their
/// reported expiry. Prevents a request from picking up a token that
/// expires while the request is mid-flight.
const TOKEN_REFRESH_SAFETY_MARGIN: Duration = Duration::from_secs(60);

/// Standard GCP service-account JSON key shape — minimum fields
/// needed for minting. Field names match the on-disk JSON
/// `gcloud iam service-accounts keys create` emits; fields we
/// don't consume (`project_id`, `private_key_id`, `client_id`,
/// `auth_uri`, `auth_provider_x509_cert_url`,
/// `client_x509_cert_url`) are omitted from the struct — serde's
/// default deserializer silently ignores them, so the operator
/// can paste the whole SA JSON verbatim without trimming.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ServiceAccountKey {
    /// Discriminator; must equal `"service_account"` for normal SA
    /// keys (vs. `"external_account"` / `"authorized_user"`). We
    /// only support SA keys.
    #[serde(rename = "type")]
    pub typ: String,
    /// PEM-encoded RSA private key. Multi-line in source JSON
    /// (`\n` escapes); jsonwebtoken's `from_rsa_pem` decodes from
    /// the byte slice directly.
    pub private_key: String,
    pub client_email: String,
    pub token_uri: String,
}

impl ServiceAccountKey {
    /// Cheap shape checks at parse time so the operator gets a fast,
    /// actionable error rather than waiting for a JWT-sign failure on
    /// the first chat. We intentionally do NOT attempt PEM parsing
    /// here (that's a heavier operation with its own error class);
    /// the first mint will catch a malformed key with a clear message.
    pub fn validate(&self) -> Result<(), BridgeError> {
        if self.typ != "service_account" {
            return Err(BridgeError::InvalidUpstreamConfig(format!(
                "vertex service_account_json.type = {:?}, want \"service_account\"",
                self.typ
            )));
        }
        if !self.private_key.starts_with("-----BEGIN") {
            return Err(BridgeError::InvalidUpstreamConfig(
                "vertex service_account_json.private_key is not PEM-formatted \
                 (expected `-----BEGIN PRIVATE KEY-----` or `-----BEGIN RSA PRIVATE KEY-----`)"
                    .into(),
            ));
        }
        if self.client_email.is_empty() {
            return Err(BridgeError::InvalidUpstreamConfig(
                "vertex service_account_json.client_email is empty".into(),
            ));
        }
        if self.token_uri.is_empty() {
            return Err(BridgeError::InvalidUpstreamConfig(
                "vertex service_account_json.token_uri is empty".into(),
            ));
        }
        Ok(())
    }
}

/// JWT claims for the bearer assertion. Field names per RFC 7523 §3.
#[derive(Serialize)]
struct JwtClaims<'a> {
    iss: &'a str,
    scope: &'a str,
    aud: &'a str,
    iat: u64,
    exp: u64,
}

/// Token-endpoint response shape per OAuth2 spec. `token_type` field
/// is always `"Bearer"` for SA flows; discarded here.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
}

#[derive(Clone)]
struct CachedToken {
    access_token: String,
    expires_at: Instant,
}

/// In-process token cache + minter. One instance per VertexBridge.
/// Cache is keyed by SA's `client_email` so multiple ProviderKeys
/// backed by the same SA share a token slot.
pub(crate) struct TokenMinter {
    client: Client,
    cache: Arc<RwLock<HashMap<String, CachedToken>>>,
    /// Test-only override for the SA-supplied `token_uri`. In
    /// production we POST directly to the SA's own `token_uri`
    /// (typically `https://oauth2.googleapis.com/token`). Tests
    /// substitute a wiremock URI here so the assertion flow runs
    /// without leaving the test process.
    #[cfg(test)]
    token_endpoint_override: Option<String>,
}

impl TokenMinter {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(test)]
            token_endpoint_override: None,
        }
    }

    /// Test-only seam: replace the SA's `token_uri` host with this
    /// URL. JWT contents are unchanged so the assertion shape is
    /// still verifiable end-to-end.
    #[cfg(test)]
    pub(crate) fn with_token_endpoint_override(mut self, url: impl Into<String>) -> Self {
        self.token_endpoint_override = Some(url.into());
        self
    }

    /// Resolve an access token for `sa`. Returns the cached token
    /// when one exists and is unexpired; otherwise mints a fresh one
    /// and caches it.
    pub async fn get_token(&self, sa: &ServiceAccountKey) -> Result<String, BridgeError> {
        // Read-lock for the common path.
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&sa.client_email) {
                if cached.expires_at > Instant::now() {
                    return Ok(cached.access_token.clone());
                }
            }
        }
        // Cache miss or expired — mint fresh under write-lock.
        let (access_token, expires_in_secs) = self.mint(sa).await?;
        let cached = CachedToken {
            access_token: access_token.clone(),
            expires_at: Instant::now()
                + Duration::from_secs(expires_in_secs).saturating_sub(TOKEN_REFRESH_SAFETY_MARGIN),
        };
        self.cache
            .write()
            .await
            .insert(sa.client_email.clone(), cached);
        Ok(access_token)
    }

    /// Mint a fresh token by signing the JWT and POSTing to the
    /// token endpoint. Returns `(access_token, expires_in_seconds)`.
    async fn mint(&self, sa: &ServiceAccountKey) -> Result<(String, u64), BridgeError> {
        sa.validate()?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| {
                BridgeError::Config(format!(
                    "vertex token mint: system clock before UNIX epoch: {e}"
                ))
            })?
            .as_secs();
        let claims = JwtClaims {
            iss: &sa.client_email,
            scope: VERTEX_SCOPE,
            aud: &sa.token_uri,
            iat: now,
            exp: now + JWT_EXPIRY_SECS,
        };
        let header = Header::new(Algorithm::RS256);
        let key = EncodingKey::from_rsa_pem(sa.private_key.as_bytes()).map_err(|e| {
            BridgeError::Config(format!(
                "vertex service_account_json.private_key invalid PEM: {e}"
            ))
        })?;
        let jwt = encode(&header, &claims, &key)
            .map_err(|e| BridgeError::Config(format!("vertex JWT sign failed: {e}")))?;

        let endpoint = self.resolve_token_endpoint(sa);
        let resp = self
            .client
            .post(&endpoint)
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                ("assertion", &jwt),
            ])
            .send()
            .await
            .map_err(|e| {
                BridgeError::Transport(format!("vertex token mint POST {endpoint}: {e}"))
            })?;

        let status = resp.status();
        if !status.is_success() {
            // Read Retry-After BEFORE consuming the body via .text() so a
            // 429/503 from GCP flows the upstream backoff hint into the
            // cooldown layer.
            let retry_after = aisix_gateway::parse_retry_after(resp.headers());
            // Cap body to 500 chars to keep error messages bounded.
            // GCP returns OAuth-shape errors `{error, error_description}`
            // — text is operator-actionable (invalid_grant etc.) and
            // does not echo the SA's private key.
            let body = resp.text().await.unwrap_or_default();
            let truncated: String = body.chars().take(500).collect();
            let msg = format!("vertex token mint upstream returned HTTP {status}: {truncated}");
            // Audit MEDIUM (PR #387): classify 5xx as a transient
            // upstream failure (502 with cooldown semantics) rather
            // than `Config` (500, operator must fix). A flapping GCP
            // token endpoint should not look operator-actionable to
            // the customer. 4xx (invalid_grant / bad SA / clock skew)
            // IS operator-actionable, so it stays Config.
            return Err(if status.is_server_error() {
                BridgeError::upstream_status_with_retry_after(status.as_u16(), msg, retry_after)
            } else {
                BridgeError::Config(msg)
            });
        }
        let parsed: TokenResponse = resp
            .json()
            .await
            .map_err(|e| BridgeError::UpstreamDecode(format!("vertex token mint response: {e}")))?;
        Ok((parsed.access_token, parsed.expires_in))
    }

    fn resolve_token_endpoint(&self, sa: &ServiceAccountKey) -> String {
        #[cfg(test)]
        if let Some(base) = &self.token_endpoint_override {
            return base.clone();
        }
        sa.token_uri.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{decode, DecodingKey, Validation};
    use wiremock::matchers::{body_string_contains, method};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    /// Generate a fresh 2048-bit RSA key pair PEM-encoded for tests.
    /// Returns `(private_pem, public_pem)`. Uses jsonwebtoken's
    /// internal RustCrypto dep transitively — we keep it test-only
    /// to avoid pulling another crate into prod.
    fn test_key_pair() -> (String, String) {
        // Hand-baked deterministic 2048-bit RSA key pair for tests.
        // Generated once via `openssl genpkey -algorithm RSA -out test.pem -pkeyopt rsa_keygen_bits:2048`
        // + `openssl rsa -in test.pem -pubout -out test.pub.pem`.
        // Deterministic so tests reproduce the same JWT signature byte
        // sequence across runs. NOT a real GCP key — safe to commit.
        let private_pem = include_str!("../test-fixtures/test_sa_private.pem").to_string();
        let public_pem = include_str!("../test-fixtures/test_sa_public.pem").to_string();
        (private_pem, public_pem)
    }

    fn sample_sa(private_pem: &str, token_uri: &str) -> ServiceAccountKey {
        ServiceAccountKey {
            typ: "service_account".into(),
            private_key: private_pem.to_string(),
            client_email: "tester@my-proj.iam.gserviceaccount.com".into(),
            token_uri: token_uri.to_string(),
        }
    }

    #[tokio::test]
    async fn mint_signs_jwt_with_correct_claims_and_posts_to_token_uri() {
        let (private_pem, public_pem) = test_key_pair();
        let server = MockServer::start().await;

        // Capture the inbound JWT assertion so we can decode + verify
        // its claims independently using the public key.
        let captured: Arc<std::sync::Mutex<Option<String>>> = Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured.clone();
        Mock::given(method("POST"))
            .and(body_string_contains("grant_type=urn"))
            .respond_with(move |req: &Request| {
                let body = String::from_utf8(req.body.clone()).unwrap();
                // Body is form-encoded: grant_type=...&assertion=<jwt>
                let assertion = body
                    .split('&')
                    .find_map(|kv| kv.strip_prefix("assertion="))
                    .unwrap_or_default();
                *captured_for_responder.lock().unwrap() = Some(urlencoding_decode(assertion));
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "access_token": "ya29.minted-by-mock",
                    "expires_in": 3600,
                    "token_type": "Bearer"
                }))
            })
            .mount(&server)
            .await;

        let sa = sample_sa(&private_pem, &server.uri());
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let token = minter.get_token(&sa).await.unwrap();
        assert_eq!(token, "ya29.minted-by-mock");

        // Decode the JWT against the matching public key and check claims.
        let jwt = captured.lock().unwrap().clone().expect("JWT captured");
        let decoding_key = DecodingKey::from_rsa_pem(public_pem.as_bytes()).unwrap();
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_audience(&[&server.uri()]);
        validation.validate_exp = false; // we test exp directly below
        let token_data = decode::<serde_json::Value>(&jwt, &decoding_key, &validation).unwrap();
        assert_eq!(
            token_data.claims["iss"].as_str().unwrap(),
            "tester@my-proj.iam.gserviceaccount.com"
        );
        assert_eq!(token_data.claims["scope"].as_str().unwrap(), VERTEX_SCOPE);
        assert_eq!(token_data.claims["aud"].as_str().unwrap(), server.uri());
        let iat = token_data.claims["iat"].as_u64().unwrap();
        let exp = token_data.claims["exp"].as_u64().unwrap();
        assert_eq!(exp - iat, JWT_EXPIRY_SECS);
    }

    #[tokio::test]
    async fn get_token_caches_within_ttl_and_only_calls_endpoint_once() {
        let (private_pem, _) = test_key_pair();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "ya29.cache-hit",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .expect(1) // critical: must be called EXACTLY once across 3 get_token calls
            .mount(&server)
            .await;

        let sa = sample_sa(&private_pem, &server.uri());
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        for _ in 0..3 {
            assert_eq!(minter.get_token(&sa).await.unwrap(), "ya29.cache-hit");
        }
    }

    #[tokio::test]
    async fn token_endpoint_5xx_surfaces_as_upstream_status_with_retry_hint() {
        // Audit MEDIUM (PR #387): 5xx is transient upstream — must
        // be UpstreamStatus (502 with cooldown semantics), not
        // Config (500 operator-must-fix).
        let (private_pem, _) = test_key_pair();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(503)
                    .insert_header("Retry-After", "30")
                    .set_body_string("upstream OAuth backend transient failure"),
            )
            .mount(&server)
            .await;
        let sa = sample_sa(&private_pem, &server.uri());
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let err = minter.get_token(&sa).await.err().unwrap();
        match err {
            BridgeError::UpstreamStatus {
                status,
                message,
                retry_after,
                ..
            } => {
                assert_eq!(status, 503);
                assert!(message.contains("HTTP 503"));
                assert!(message.contains("upstream OAuth backend transient failure"));
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(30)));
            }
            other => panic!("expected UpstreamStatus error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn token_endpoint_4xx_surfaces_as_config_error() {
        // Audit MEDIUM (PR #387): 4xx remains Config — invalid_grant
        // / bad SA / clock skew are operator-actionable.
        let (private_pem, _) = test_key_pair();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_string(
                r#"{"error":"invalid_grant","error_description":"JWT iat/exp out of range"}"#,
            ))
            .mount(&server)
            .await;
        let sa = sample_sa(&private_pem, &server.uri());
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let err = minter.get_token(&sa).await.err().unwrap();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("HTTP 400"));
                assert!(msg.contains("invalid_grant"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_pem_surfaces_clear_error_before_endpoint_call() {
        let server = MockServer::start().await;
        // Mock with .expect(0) — endpoint must NOT be called for a
        // pre-flight validation failure.
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let sa = ServiceAccountKey {
            typ: "service_account".into(),
            private_key: "not a PEM at all".into(),
            client_email: "x@y.z".into(),
            token_uri: server.uri(),
        };
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let err = minter.get_token(&sa).await.err().unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(msg.contains("not PEM-formatted"));
            }
            other => panic!("expected InvalidUpstreamConfig error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn wrong_type_field_rejected_before_endpoint_call() {
        let (private_pem, _) = test_key_pair();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let sa = ServiceAccountKey {
            typ: "external_account".into(),
            private_key: private_pem,
            client_email: "x@y.z".into(),
            token_uri: server.uri(),
        };
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let err = minter.get_token(&sa).await.err().unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(msg.contains("type"));
                assert!(msg.contains("external_account"));
                assert!(msg.contains("service_account"));
            }
            other => panic!("expected InvalidUpstreamConfig error, got {other:?}"),
        }
    }

    /// Minimal URL-decode for the captured assertion field. Sufficient
    /// for the JWT character set (`A-Za-z0-9-_.`), which the standard
    /// form-encoder leaves untouched.
    fn urlencoding_decode(s: &str) -> String {
        s.replace('+', " ").replace("%2B", "+").replace("%2F", "/")
    }
}
