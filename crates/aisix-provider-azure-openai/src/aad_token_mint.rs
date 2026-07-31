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
//! Cache is keyed by a fingerprint of the whole credential: two
//! distinct AAD app registrations under the same tenant must NOT share
//! a token slot, and neither must a rotated `client_secret` and the
//! secret it replaced. Refresh ~60s before upstream-reported expiry so
//! an in-flight request never lands on an expired token.
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

use aisix_gateway::{credential_fingerprint, BridgeError};
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
    /// Optional AAD authority host override. Absent for the common
    /// public-cloud case, which defaults to
    /// `https://login.microsoftonline.com`. Set when the tenant lives
    /// in an Azure national / sovereign cloud whose AAD authority
    /// differs from public Azure:
    ///   - US Government: `https://login.microsoftonline.us`
    ///   - China (21Vianet): `https://login.chinacloudapi.cn`
    ///
    /// per Microsoft's national-cloud authentication endpoints table
    /// <https://learn.microsoft.com/en-us/entra/identity-platform/authentication-national-cloud>
    /// (the same value the Azure SDK reads from the
    /// `AZURE_AUTHORITY_HOST` environment variable). Must be a bare
    /// http(s) origin — the bridge interpolates the tenant and the
    /// `/oauth2/v2.0/token` path itself.
    #[serde(default)]
    pub authority_host: Option<String>,
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
                return Err(BridgeError::InvalidUpstreamCredentials(format!(
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
                return Err(BridgeError::InvalidUpstreamCredentials(format!(
                    "azure aad credentials.{name} {value:?} contains URL-control \
                     characters — reject `/`, `?`, `#`, whitespace, `..`"
                )));
            }
        }
        // Optional authority_host: when present it becomes the origin
        // of the token-endpoint URL, so it must be a bare http(s)
        // origin. Mirrors the Vertex `resolve_api_base` validation
        // (PR #392) — reject userinfo / query / fragment first (fixed
        // message, no echo, so pasted `user:pass@host` credentials
        // never surface in logs), then enforce the scheme.
        if let Some(host) = self
            .authority_host
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            if host.contains('@') || host.contains('?') || host.contains('#') {
                return Err(BridgeError::InvalidUpstreamConfig(
                    "azure aad credentials.authority_host must be a bare origin — \
                     reject userinfo (@), query (?), fragment (#)"
                        .into(),
                ));
            }
            if !(host.starts_with("https://") || host.starts_with("http://")) {
                return Err(BridgeError::InvalidUpstreamConfig(format!(
                    "azure aad credentials.authority_host must use http:// or https:// \
                     scheme, got {host:?}"
                )));
            }
            // Reject an embedded path component — only `scheme://host[:port]`
            // is a valid origin. A real path segment would silently redirect
            // the token endpoint (e.g. `.../evil/{tenant}/oauth2/v2.0/token`).
            // A bare trailing slash is tolerated for symmetry with the Vertex
            // `resolve_api_base` check. Backslashes are rejected too: the
            // WHATWG URL parser the HTTP client uses normalizes `\` to `/` on
            // http(s) URLs, so `host\evil` injects a path exactly like
            // `host/evil`. `host` has no `@`/`?`/`#` here, so echoing it is
            // safe. Audit #434 LOW-1 / #435 (+ #464 audit MEDIUM).
            let after_scheme = host
                .split_once("://")
                .map(|(_, rest)| rest)
                .unwrap_or(host)
                .trim_end_matches('/');
            if after_scheme.contains('/') || after_scheme.contains('\\') {
                return Err(BridgeError::InvalidUpstreamConfig(format!(
                    "azure aad credentials.authority_host must be a bare origin \
                     (scheme://host[:port]) with no path, got {host:?}"
                )));
            }
        }
        Ok(())
    }

    /// Token-cache key. Every field of the credential feeds the
    /// fingerprint — add new fields here when the struct grows, or a
    /// rotation that only touches the new field will serve the stale
    /// token for its full TTL. `authority_host` is in scope because it
    /// selects which AAD cloud minted the token.
    fn cache_key(&self) -> String {
        credential_fingerprint(&[
            &self.tenant_id,
            &self.client_id,
            &self.client_secret,
            self.authority_host.as_deref().unwrap_or(""),
        ])
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
/// `AzureOpenAiBridge`. Cache is keyed by a fingerprint of the whole
/// credential, so ProviderKeys backed by the same app registration
/// share a slot while distinct apps — and a rotated `client_secret`
/// under one app — get their own. Keying by `(tenant_id, client_id)`
/// alone would pin the pre-rotation token for its full TTL, since
/// rotation issues a new secret under the same app registration.
pub(crate) struct TokenMinter {
    client: Client,
    cache: Arc<RwLock<HashMap<String, CachedToken>>>,
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
        let key = creds.cache_key();
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

    /// Resolve the AAD token endpoint URL. Precedence:
    ///   1. `#[cfg(test)]` `token_endpoint_override` — unit-test seam,
    ///      a full fixed URL (tenant not interpolated).
    ///   2. `creds.authority_host` — production override for Azure
    ///      national / sovereign clouds (and the live-e2e mock). The
    ///      tenant + `/oauth2/v2.0/token` path is interpolated onto it.
    ///   3. Default public-cloud authority `login.microsoftonline.com`.
    ///
    /// `authority_host` is validated by [`AadCredentials::validate`]
    /// (bare http(s) origin, no userinfo / query / fragment) before
    /// this runs, so the only normalisation needed here is trimming a
    /// trailing slash to avoid a `//{tenant}` double slash.
    fn resolve_token_endpoint(&self, creds: &AadCredentials) -> String {
        #[cfg(test)]
        if let Some(base) = &self.token_endpoint_override {
            // Tests get a fixed URL — tenant id is still validated
            // upstream of this call.
            return base.clone();
        }
        let authority = creds
            .authority_host
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("https://login.microsoftonline.com")
            .trim_end_matches('/');
        format!("{authority}/{}/oauth2/v2.0/token", creds.tenant_id)
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
            authority_host: None,
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
            authority_host: None,
        };
        let app_b = AadCredentials {
            tenant_id: "shared-tenant".into(),
            client_id: "bbbb-bbbb".into(),
            client_secret: "secret-b".into(),
            authority_host: None,
        };
        assert_eq!(minter.get_token(&app_a).await.unwrap(), "token-for-app-A");
        assert_eq!(minter.get_token(&app_b).await.unwrap(), "token-for-app-B");
        // Repeat — both must come from cache, not re-mint.
        assert_eq!(minter.get_token(&app_a).await.unwrap(), "token-for-app-A");
        assert_eq!(minter.get_token(&app_b).await.unwrap(), "token-for-app-B");
    }

    #[tokio::test]
    async fn cache_remints_after_the_client_secret_is_rotated() {
        // Rotating an AAD client secret keeps the app registration —
        // same tenant_id + client_id — so a cache keyed on those alone
        // serves the pre-rotation token for its full hour. The gateway
        // would keep authenticating with the secret the operator just
        // replaced, and report success while doing it.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("client_secret=secret-old"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "token-from-old-secret",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(body_string_contains("client_secret=secret-new"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "token-from-new-secret",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .mount(&server)
            .await;

        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let before = AadCredentials {
            client_secret: "secret-old".into(),
            ..sample_creds()
        };
        let after = AadCredentials {
            client_secret: "secret-new".into(),
            ..sample_creds()
        };
        assert_eq!(
            minter.get_token(&before).await.unwrap(),
            "token-from-old-secret"
        );
        assert_eq!(
            minter.get_token(&after).await.unwrap(),
            "token-from-new-secret",
            "rotated client_secret must mint a fresh token, not reuse the app's cached one"
        );
    }

    #[tokio::test]
    async fn cache_separates_the_same_app_across_authority_hosts() {
        // authority_host selects which AAD cloud minted the token, so a
        // public-cloud token must never be served to a sovereign-cloud
        // request that happens to share tenant_id + client_id.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "token-per-authority",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .expect(2)
            .mount(&server)
            .await;

        let minter = TokenMinter::new(Client::new()).with_token_endpoint_override(server.uri());
        let public = sample_creds();
        let sovereign = AadCredentials {
            authority_host: Some("https://login.microsoftonline.us".into()),
            ..sample_creds()
        };
        minter.get_token(&public).await.unwrap();
        minter.get_token(&sovereign).await.unwrap();
        // `.expect(2)` above is the assertion: two distinct authority
        // hosts must not collapse into one cache slot.
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
            authority_host: None,
        };
        let err = minter.get_token(&creds).await.err().unwrap();
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(msg.contains("tenant_id is empty"));
            }
            other => panic!("expected InvalidUpstreamCredentials, got {other:?}"),
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
            authority_host: None,
        };
        let err = minter.get_token(&creds).await.err().unwrap();
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(msg.contains("URL-control"));
            }
            other => panic!("expected InvalidUpstreamCredentials, got {other:?}"),
        }
    }

    // ─── authority_host production seam (ai-gateway#413 / #302 D6.6) ───

    /// The default public-cloud authority is used when `authority_host`
    /// is absent — backward compat for every existing operator who
    /// pasted only `{tenant_id, client_id, client_secret}`.
    #[test]
    fn resolve_token_endpoint_defaults_to_public_cloud_when_authority_absent() {
        let minter = TokenMinter::new(Client::new());
        let creds = sample_creds();
        assert_eq!(
            minter.resolve_token_endpoint(&creds),
            "https://login.microsoftonline.com/11111111-1111-1111-1111-111111111111/oauth2/v2.0/token",
        );
    }

    /// A production `authority_host` (national cloud / sovereign /
    /// live-e2e mock) becomes the origin; the tenant + standard path
    /// are interpolated. This is the production code path — NO
    /// `#[cfg(test)]` override is set, so a regression that ignored
    /// `authority_host` would fail here.
    #[test]
    fn resolve_token_endpoint_honors_authority_host_in_production() {
        let minter = TokenMinter::new(Client::new());
        let creds = AadCredentials {
            tenant_id: "tenant-gov".into(),
            client_id: "app".into(),
            client_secret: "s".into(),
            // US Government cloud authority per Microsoft national-cloud docs.
            authority_host: Some("https://login.microsoftonline.us".into()),
        };
        assert_eq!(
            minter.resolve_token_endpoint(&creds),
            "https://login.microsoftonline.us/tenant-gov/oauth2/v2.0/token",
        );
    }

    /// A trailing slash on `authority_host` must not produce a `//`
    /// double-slash before the tenant segment.
    #[test]
    fn resolve_token_endpoint_trims_authority_host_trailing_slash() {
        let minter = TokenMinter::new(Client::new());
        let creds = AadCredentials {
            tenant_id: "t".into(),
            client_id: "app".into(),
            client_secret: "s".into(),
            authority_host: Some("https://login.microsoftonline.us/".into()),
        };
        assert_eq!(
            minter.resolve_token_endpoint(&creds),
            "https://login.microsoftonline.us/t/oauth2/v2.0/token",
        );
    }

    /// End-to-end through the production path: build the minter WITHOUT
    /// the `#[cfg(test)]` override, point `authority_host` at a wiremock
    /// server, and assert the client-credentials POST lands on the
    /// interpolated `/{tenant}/oauth2/v2.0/token` path. A regression
    /// that POSTed to the wrong path (or to public-cloud regardless of
    /// authority_host) would not match and the mock would 404.
    #[tokio::test]
    async fn mint_via_production_authority_host_posts_to_interpolated_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wiremock::matchers::path("/tenant-xyz/oauth2/v2.0/token"))
            .and(body_string_contains("grant_type=client_credentials"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "aad.gov-cloud-token",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .mount(&server)
            .await;

        // NOTE: no with_token_endpoint_override — exercises the real
        // production resolve_token_endpoint via authority_host.
        let minter = TokenMinter::new(Client::new());
        let creds = AadCredentials {
            tenant_id: "tenant-xyz".into(),
            client_id: "app".into(),
            client_secret: "s".into(),
            authority_host: Some(server.uri()),
        };
        let token = minter.get_token(&creds).await.unwrap();
        assert_eq!(token, "aad.gov-cloud-token");
    }

    #[test]
    fn validate_rejects_authority_host_without_scheme() {
        let creds = AadCredentials {
            tenant_id: "t".into(),
            client_id: "app".into(),
            client_secret: "s".into(),
            authority_host: Some("login.microsoftonline.us".into()),
        };
        let err = creds.validate().err().unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => assert!(msg.contains("http:// or https://")),
            other => panic!("expected InvalidUpstreamConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_authority_host_with_embedded_path() {
        // #435: a path segment would silently redirect the token endpoint
        // (e.g. `.../evil/{tenant}/oauth2/v2.0/token`) — reject it.
        let creds = AadCredentials {
            tenant_id: "t".into(),
            client_id: "app".into(),
            client_secret: "s".into(),
            authority_host: Some("https://login.microsoftonline.us/evil".into()),
        };
        let err = creds.validate().err().unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => assert!(
                msg.contains("bare origin") && msg.contains("no path"),
                "expected a bare-origin/no-path rejection; got {msg}"
            ),
            other => panic!("expected InvalidUpstreamConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_allows_authority_host_bare_origin_with_port() {
        // A `host:port` origin (no path) must still validate — the path
        // rejection must not false-positive on the `:port` colon, and a
        // bare trailing slash is tolerated (trimmed at URL-build time).
        for host in [
            "https://login.microsoftonline.us:8443",
            "https://login.microsoftonline.us/",
        ] {
            let creds = AadCredentials {
                tenant_id: "t".into(),
                client_id: "app".into(),
                client_secret: "s".into(),
                authority_host: Some(host.into()),
            };
            assert!(creds.validate().is_ok(), "{host} should validate");
        }
    }

    #[test]
    fn validate_rejects_authority_host_with_backslash_path() {
        // #464 audit: the WHATWG URL parser the HTTP client uses normalizes
        // `\` to `/` on http(s) URLs, so `host\evil` injects a path just like
        // `host/evil` — it must be rejected the same way.
        let creds = AadCredentials {
            tenant_id: "t".into(),
            client_id: "app".into(),
            client_secret: "s".into(),
            authority_host: Some("https://login.microsoftonline.us\\evil".into()),
        };
        let err = creds.validate().err().unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => assert!(
                msg.contains("bare origin") && msg.contains("no path"),
                "expected a bare-origin/no-path rejection; got {msg}"
            ),
            other => panic!("expected InvalidUpstreamConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_authority_host_with_userinfo_without_echoing_it() {
        let creds = AadCredentials {
            tenant_id: "t".into(),
            client_id: "app".into(),
            client_secret: "s".into(),
            authority_host: Some("https://user:pass@evil.example.com".into()),
        };
        let err = creds.validate().err().unwrap();
        match err {
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(msg.contains("bare origin"));
                // The pasted userinfo must NOT surface in the error.
                assert!(!msg.contains("pass"), "error leaked userinfo: {msg}");
                assert!(
                    !msg.contains("evil.example.com"),
                    "error leaked host: {msg}"
                );
            }
            other => panic!("expected InvalidUpstreamConfig, got {other:?}"),
        }
    }
}
