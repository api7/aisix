//! Azure AD (Entra ID) `client_credentials` OAuth2 flow + token cache
//! for the Azure OpenAI Service bridge.
//!
//! Unlike Vertex (JWT-bearer assertion grant with RS256 signing),
//! Azure AAD's `client_credentials` flow is a straight form-encoded
//! POST — no JWT signing on the gateway side:
//!
//! ```text
//! POST https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token
//!   Content-Type: application/x-www-form-urlencoded
//!
//!   grant_type=client_credentials
//!   &client_id=<app-registration-uuid>
//!   &client_secret=<rotation-managed-secret>
//!   &scope=https://cognitiveservices.azure.com/.default
//! ```
//!
//! Response: `{access_token, expires_in, token_type: "Bearer"}`. The
//! customer then sends `Authorization: Bearer <access_token>` on every
//! Azure OpenAI request — replacing the `api-key:` header used by the
//! resource-key scheme.
//!
//! Cache is keyed by `(tenant_id, client_id)` because two distinct
//! AAD app registrations under the same tenant must NOT share a
//! token slot. Refresh ~60s before upstream-reported expiry so an
//! in-flight request never lands on an expired token.
//!
//! 5xx from the token endpoint surfaces as `BridgeError::UpstreamStatus`
//! with `Retry-After` propagated (transient AAD outage should hit the
//! cooldown layer, not the operator-must-fix path). 4xx remains
//! `BridgeError::Config` — `invalid_client` / `unauthorized_client` /
//! revoked secret IS operator-actionable. Pattern mirrors
//! `aisix-provider-vertex::token_mint` after the audit on
//! ai-gateway#387.
//!
//! # References
//!
//! - Microsoft identity platform — client credentials grant flow:
//!   <https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-client-creds-grant-flow>
//! - Azure OpenAI authentication (Entra ID section):
//!   <https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity>
//! - OAuth2 `client_credentials` spec (RFC 6749 §4.4):
//!   <https://www.rfc-editor.org/rfc/rfc6749#section-4.4>

use aisix_gateway::BridgeError;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Default scope for Azure OpenAI Service. Per Microsoft's docs the
/// `.default` suffix tells AAD to issue a token covering every
/// permission the app registration is configured for — the standard
/// pattern for non-interactive service-to-service auth.
const AZURE_OPENAI_SCOPE: &str = "https://cognitiveservices.azure.com/.default";

/// Refresh cached tokens at least this many seconds before their
/// reported expiry. Prevents a request from picking up a token that
/// expires while the request is mid-flight.
const TOKEN_REFRESH_SAFETY_MARGIN: Duration = Duration::from_secs(60);

/// AAD app-registration credentials for the `client_credentials`
/// grant. Field names match the JSON shape an operator pastes
/// directly from the Azure portal's "Certificates & secrets" view.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AadCredentials {
    /// AAD tenant UUID (or vanity domain). Embedded in the token
    /// endpoint URL path.
    pub tenant_id: String,
    /// App-registration (application) UUID.
    pub client_id: String,
    /// Client secret value (NOT the secret id). Confidential.
    pub client_secret: String,
}

impl AadCredentials {
    /// Cheap shape validation at parse time. Tenant id / client id
    /// must not contain URL-control characters (so we can interpolate
    /// the tenant into the token endpoint path without injection).
    pub fn validate(&self) -> Result<(), BridgeError> {
        for (name, value) in [
            ("tenant_id", &self.tenant_id),
            ("client_id", &self.client_id),
            ("client_secret", &self.client_secret),
        ] {
            if value.is_empty() {
                return Err(BridgeError::Config(format!(
                    "azure aad credentials.{name} is empty"
                )));
            }
        }
        for (name, value) in [
            ("tenant_id", &self.tenant_id),
            ("client_id", &self.client_id),
        ] {
            if value.contains('/')
                || value.contains('?')
                || value.contains('#')
                || value.contains(' ')
                || value.contains('\t')
                || value.contains('\n')
                || value.contains("..")
            {
                return Err(BridgeError::Config(format!(
                    "azure aad credentials.{name} {value:?} contains URL-control \
                     characters — reject `/`, `?`, `#`, whitespace, `..`"
                )));
            }
        }
        Ok(())
    }
}

/// Token-endpoint response shape per OAuth2 spec.
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

/// In-process token cache + minter. One instance per
/// `AzureOpenAiBridge`. Cache is keyed by `(tenant_id, client_id)`
/// so multiple ProviderKeys backed by the same AAD app share a slot,
/// but distinct apps under the same tenant don't collide.
pub(crate) struct TokenMinter {
    client: Client,
    cache: Arc<RwLock<HashMap<(String, String), CachedToken>>>,
    /// Test-only override for the AAD token endpoint. In production
    /// the URL is derived from the tenant id; tests substitute a
    /// wiremock URI here.
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

    /// Test-only seam: replace the `login.microsoftonline.com` host
    /// with this URL. Tenant id is still interpolated into the path
    /// (so the request URL shape is verifiable end-to-end against
    /// wiremock matchers).
    #[cfg(test)]
    pub(crate) fn with_token_endpoint_override(mut self, url: impl Into<String>) -> Self {
        self.token_endpoint_override = Some(url.into());
        self
    }

    /// Resolve an access token for `creds`. Returns the cached token
    /// when one exists and is unexpired; otherwise mints a fresh one
    /// and caches it.
    pub async fn get_token(&self, creds: &AadCredentials) -> Result<String, BridgeError> {
        let key = (creds.tenant_id.clone(), creds.client_id.clone());
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&key) {
                if cached.expires_at > Instant::now() {
                    return Ok(cached.access_token.clone());
                }
            }
        }
        let (access_token, expires_in_secs) = self.mint(creds).await?;
        let cached = CachedToken {
            access_token: access_token.clone(),
            expires_at: Instant::now()
                + Duration::from_secs(expires_in_secs).saturating_sub(TOKEN_REFRESH_SAFETY_MARGIN),
        };
        self.cache.write().await.insert(key, cached);
        Ok(access_token)
    }

    /// Mint a fresh token by POSTing the client_credentials grant
    /// to the AAD token endpoint. Returns `(access_token, expires_in)`.
    async fn mint(&self, creds: &AadCredentials) -> Result<(String, u64), BridgeError> {
        creds.validate()?;
        let endpoint = self.resolve_token_endpoint(creds);
        let resp = self
            .client
            .post(&endpoint)
            .form(&[
                ("grant_type", "client_credentials"),
                ("client_id", &creds.client_id),
                ("client_secret", &creds.client_secret),
                ("scope", AZURE_OPENAI_SCOPE),
            ])
            .send()
            .await
            .map_err(|e| {
                BridgeError::Transport(format!("azure aad token mint POST {endpoint}: {e}"))
            })?;

        let status = resp.status();
        if !status.is_success() {
            // Read Retry-After BEFORE consuming the body.
            let retry_after = aisix_gateway::parse_retry_after(resp.headers());
            let body = resp.text().await.unwrap_or_default();
            // Cap body to 500 chars. AAD error envelopes
            // `{error: "invalid_client", error_description: "..."}` are
            // operator-actionable; cap is defense-in-depth against a
            // mis-deployed front-door that returns an HTML error page.
            let truncated: String = body.chars().take(500).collect();
            let msg = format!("azure aad token mint upstream returned HTTP {status}: {truncated}");
            // Mirror Vertex audit MEDIUM (ai-gateway#387): 5xx is
            // transient upstream — should surface UpstreamStatus
            // (cooldown semantics) with the Retry-After hint. 4xx
            // (invalid_client / unauthorized_client / revoked secret)
            // is operator-actionable, stays Config.
            return Err(if status.is_server_error() {
                BridgeError::upstream_status_with_retry_after(status.as_u16(), msg, retry_after)
            } else {
                BridgeError::Config(msg)
            });
        }
        let parsed: TokenResponse = resp.json().await.map_err(|e| {
            BridgeError::UpstreamDecode(format!("azure aad token mint response: {e}"))
        })?;
        Ok((parsed.access_token, parsed.expires_in))
    }

    fn resolve_token_endpoint(&self, creds: &AadCredentials) -> String {
        #[cfg(test)]
        if let Some(base) = &self.token_endpoint_override {
            // Tests get a fixed URL — tenant id is still validated
            // upstream of this call.
            return base.clone();
        }
        format!(
            "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
            creds.tenant_id
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_string_contains, method};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    fn sample_creds() -> AadCredentials {
        AadCredentials {
            tenant_id: "11111111-1111-1111-1111-111111111111".into(),
            client_id: "22222222-2222-2222-2222-222222222222".into(),
            client_secret: "fake-secret-not-a-real-one".into(),
        }
    }

    #[tokio::test]
    async fn mint_posts_client_credentials_grant_with_correct_scope() {
        let server = MockServer::start().await;
        let captured_body: Arc<std::sync::Mutex<Option<String>>> =
            Arc::new(std::sync::Mutex::new(None));
        let captured_for_responder = captured_body.clone();
        Mock::given(method("POST"))
            .and(body_string_contains("grant_type=client_credentials"))
            .respond_with(move |req: &Request| {
                *captured_for_responder.lock().unwrap() =
                    Some(String::from_utf8(req.body.clone()).unwrap());
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "access_token": "aad.minted-by-mock",
                    "expires_in": 3600,
                    "token_type": "Bearer"
                }))
            })
            .mount(&server)
            .await;

        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let token = minter.get_token(&sample_creds()).await.unwrap();
        assert_eq!(token, "aad.minted-by-mock");

        let body = captured_body
            .lock()
            .unwrap()
            .clone()
            .expect("body captured");
        // Form fields per RFC 6749 §4.4 + Microsoft client-credentials docs.
        assert!(body.contains("grant_type=client_credentials"));
        assert!(body.contains("client_id=22222222-2222-2222-2222-222222222222"));
        assert!(body.contains("client_secret=fake-secret-not-a-real-one"));
        assert!(body.contains("scope=https%3A%2F%2Fcognitiveservices.azure.com%2F.default"));
    }

    #[tokio::test]
    async fn get_token_caches_within_ttl_and_only_calls_endpoint_once() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "aad.cache-hit",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .expect(1)
            .mount(&server)
            .await;
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let creds = sample_creds();
        for _ in 0..3 {
            assert_eq!(minter.get_token(&creds).await.unwrap(), "aad.cache-hit");
        }
    }

    #[tokio::test]
    async fn cache_separates_distinct_app_registrations_under_same_tenant() {
        // Two apps under the same tenant get distinct cache slots —
        // a regression that keyed only by tenant_id would serve
        // app B's request app A's token (or vice versa). Pin the
        // separation explicitly.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("client_id=aaaa"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "token-for-app-A",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(body_string_contains("client_id=bbbb"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "token-for-app-B",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .mount(&server)
            .await;

        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let app_a = AadCredentials {
            tenant_id: "shared-tenant".into(),
            client_id: "aaaa-aaaa".into(),
            client_secret: "secret-a".into(),
        };
        let app_b = AadCredentials {
            tenant_id: "shared-tenant".into(),
            client_id: "bbbb-bbbb".into(),
            client_secret: "secret-b".into(),
        };
        assert_eq!(minter.get_token(&app_a).await.unwrap(), "token-for-app-A");
        assert_eq!(minter.get_token(&app_b).await.unwrap(), "token-for-app-B");
        // Repeat — both must come from cache, not re-mint.
        assert_eq!(minter.get_token(&app_a).await.unwrap(), "token-for-app-A");
        assert_eq!(minter.get_token(&app_b).await.unwrap(), "token-for-app-B");
    }

    #[tokio::test]
    async fn token_endpoint_5xx_surfaces_as_upstream_status_with_retry_hint() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(503)
                    .insert_header("Retry-After", "45")
                    .set_body_string("Service Unavailable: please retry"),
            )
            .mount(&server)
            .await;
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let err = minter.get_token(&sample_creds()).await.err().unwrap();
        match err {
            BridgeError::UpstreamStatus {
                status,
                retry_after,
                ..
            } => {
                assert_eq!(status, 503);
                assert_eq!(retry_after, Some(Duration::from_secs(45)));
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn token_endpoint_4xx_invalid_client_surfaces_as_config_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_string(
                r#"{"error":"invalid_client","error_description":"AADSTS7000215: Invalid client secret provided"}"#,
            ))
            .mount(&server)
            .await;
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let err = minter.get_token(&sample_creds()).await.err().unwrap();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("HTTP 401"));
                assert!(msg.contains("invalid_client"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_tenant_id_rejected_before_endpoint_call() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let creds = AadCredentials {
            tenant_id: "".into(),
            client_id: "abc".into(),
            client_secret: "xyz".into(),
        };
        let err = minter.get_token(&creds).await.err().unwrap();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("tenant_id is empty"));
            }
            other => panic!("expected Config, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tenant_id_with_url_injection_rejected_before_endpoint_call() {
        // A `/` in tenant_id could redirect the token POST to a path
        // the operator never authorized. validate() must catch this
        // before we interpolate into the URL.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let creds = AadCredentials {
            tenant_id: "../malicious".into(),
            client_id: "abc".into(),
            client_secret: "xyz".into(),
        };
        let err = minter.get_token(&creds).await.err().unwrap();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("URL-control"));
            }
            other => panic!("expected Config, got {other:?}"),
        }
    }
}
