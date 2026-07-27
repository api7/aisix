//! Inbound OIDC/JWT authentication (AISIX-Cloud#1080, #1081).
//!
//! When the environment has at least one enabled [`OidcProvider`], a bearer
//! token that is a JWT is authenticated here instead of the API-key hash
//! lookup: the token's unverified `iss` selects the trust provider, the
//! signature is verified against the provider's JWKS (fetched and cached,
//! with a rate-limited refresh when an unknown `kid` appears so key
//! rotation needs no restart), the registered claims (`exp` required,
//! `aud` against the provider's accepted audiences, `nbf` when present)
//! and the provider's `required_scopes` / `bound_claims` are enforced, and
//! the value of the provider's `identity_claim` selects the API key whose
//! `jwt_subject` equals it. The request then proceeds as that key — its
//! `allowed_models`, rate limits, budget, and usage attribution all apply
//! unchanged.
//!
//! Design invariants:
//!
//! - **No fallback**: once a token is JWT-shaped and JWT auth is enabled,
//!   a validation failure is final — it is never retried as an API key.
//! - **Issuer allow-list**: a JWT whose `iss` matches no enabled provider
//!   is rejected; there is no catch-all validation path.
//! - **Default deny**: `exp`, `iss`, and `aud` must be present and valid
//!   on every token; a missing identity claim or an unmapped identity is
//!   a rejection, never an anonymous pass.
//! - **Asymmetric algorithms only**: HMAC family excluded, so a JWKS can
//!   never be confused into acting as a shared secret.
//! - Every decision (allow and deny, API-key and JWT path alike) is
//!   recorded on the `aisix_auth_decisions_total` metric, and denials are
//!   logged under `target: "aisix::auth"` with the detailed reason class —
//!   the raw token is never logged.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

use aisix_core::models::{BoundClaimExpect, OidcProvider};
use aisix_core::resource::ResourceEntry;
use aisix_core::{AisixSnapshot, ApiKey};
use base64::Engine;
use jsonwebtoken::jwk::JwkSet;
use jsonwebtoken::{Algorithm, DecodingKey, Validation};

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

/// How long a fetched JWKS (and a discovery-resolved JWKS URL) stays fresh.
const JWKS_TTL: Duration = Duration::from_secs(600);

/// Minimum interval between fetches for one JWKS URL outside the TTL
/// schedule — bounds both the unknown-`kid` refresh (a token signed by a
/// just-rotated key triggers at most one refetch per interval, so rotation
/// is picked up within a second while a stream of garbage `kid`s cannot
/// flood the identity provider) and retries after a failed fetch.
const JWKS_REFRESH_MIN_INTERVAL: Duration = Duration::from_secs(1);

/// Per-request deadline for JWKS / discovery fetches.
const JWKS_FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Cap on a JWKS / discovery response body. A real JWKS is a few KB; the
/// cap keeps a misconfigured URL (pointing at some arbitrary endpoint)
/// from ballooning memory.
const JWKS_MAX_BYTES: usize = 512 * 1024;

/// Verification algorithms accepted on inbound JWTs: the asymmetric
/// families only. HMAC is deliberately excluded — accepting it would let
/// a public JWKS double as a shared signing secret (algorithm-confusion).
const ALLOWED_ALGS: [Algorithm; 9] = [
    Algorithm::RS256,
    Algorithm::RS384,
    Algorithm::RS512,
    Algorithm::PS256,
    Algorithm::PS384,
    Algorithm::PS512,
    Algorithm::ES256,
    Algorithm::ES384,
    Algorithm::EdDSA,
];

/// Cap on signature attempts for a token without a `kid` against a
/// multi-key JWKS.
const MAX_KEYS_TRIED: usize = 8;

/// True when the bearer has the structural shape of a JWT: three
/// non-empty dot-separated segments whose first segment base64url-decodes
/// to a JSON object carrying `alg` (a JOSE header). The header check
/// keeps custom-imported API keys that merely contain dots on the
/// API-key path.
pub(crate) fn looks_like_jwt(token: &str) -> bool {
    let parts: Vec<&str> = token.splitn(4, '.').collect();
    if parts.len() != 3 || parts.iter().any(|p| p.is_empty()) {
        return false;
    }
    matches!(b64url_json(parts[0]), Some(v) if v.get("alg").is_some())
}

/// True when the snapshot has at least one enabled trust provider — the
/// gate for entering the JWT path at all.
pub(crate) fn any_enabled_provider(snapshot: &AisixSnapshot) -> bool {
    snapshot
        .oidc_providers
        .entries()
        .iter()
        .any(|e| e.value.enabled)
}

fn b64url_json(segment: &str) -> Option<serde_json::Value> {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(segment)
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Unverified peek at the payload's `iss`, used only to select the trust
/// provider. The selected provider's issuer is then pinned in the real
/// validation, so a forged `iss` still has to survive signature and
/// issuer verification against that provider's keys.
fn unverified_issuer(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    b64url_json(payload)?
        .get("iss")?
        .as_str()
        .map(str::to_string)
}

/// The enabled provider matching `iss`, ties broken by lowest id so
/// duplicate rows resolve deterministically (same discipline as the MCP
/// policy resolver).
fn provider_for_issuer(
    snapshot: &AisixSnapshot,
    iss: &str,
) -> Option<Arc<ResourceEntry<OidcProvider>>> {
    let mut best: Option<Arc<ResourceEntry<OidcProvider>>> = None;
    for entry in snapshot.oidc_providers.entries() {
        if !entry.value.enabled || entry.value.issuer != iss {
            continue;
        }
        match &best {
            Some(b) if b.id <= entry.id => {}
            _ => best = Some(entry),
        }
    }
    best
}

/// The API key bound to `subject` via `jwt_subject`, ties broken by
/// lowest id. The control plane enforces per-environment uniqueness, so
/// the tie-break only matters for a transient duplicate mid-sync.
fn key_for_subject(snapshot: &AisixSnapshot, subject: &str) -> Option<Arc<ResourceEntry<ApiKey>>> {
    let mut best: Option<Arc<ResourceEntry<ApiKey>>> = None;
    for entry in snapshot.apikeys.entries() {
        if entry.value.jwt_subject.as_deref() != Some(subject) {
            continue;
        }
        match &best {
            Some(b) if b.id <= entry.id => {}
            _ => best = Some(entry),
        }
    }
    best
}

/// Authenticate a JWT-shaped bearer. Called from the auth choke point
/// once [`looks_like_jwt`] and [`any_enabled_provider`] both hold.
pub(crate) async fn authenticate_jwt(
    state: &ProxyState,
    token: &str,
) -> Result<AuthenticatedKey, ProxyError> {
    let snapshot = state.snapshot.load();

    let header = match jsonwebtoken::decode_header(token) {
        Ok(h) => h,
        Err(_) => {
            return Err(deny(
                state,
                "jwt_malformed",
                "",
                None,
                ProxyError::JwtInvalid,
            ))
        }
    };
    let kid = header.kid.clone();

    let Some(iss) = unverified_issuer(token) else {
        return Err(deny(
            state,
            "jwt_missing_issuer",
            "",
            kid.as_deref(),
            ProxyError::JwtInvalid,
        ));
    };

    let Some(provider) = provider_for_issuer(&snapshot, &iss) else {
        return Err(deny(
            state,
            "jwt_untrusted_issuer",
            &iss,
            kid.as_deref(),
            ProxyError::JwtInvalid,
        ));
    };
    let prov = &provider.value;

    if !ALLOWED_ALGS.contains(&header.alg) {
        return Err(deny(
            state,
            "jwt_alg_not_allowed",
            &iss,
            kid.as_deref(),
            ProxyError::JwtInvalid,
        ));
    }

    // ── Signing keys ─────────────────────────────────────────────────
    let jwks_url = match resolve_jwks_url(prov).await {
        Ok(u) => u,
        Err(e) => {
            tracing::warn!(
                target: "aisix::auth",
                provider = %prov.name,
                issuer = %prov.issuer,
                error = %e,
                "cannot resolve the trust provider's JWKS endpoint",
            );
            return Err(deny(
                state,
                "jwks_unavailable",
                &iss,
                kid.as_deref(),
                ProxyError::JwksUnavailable,
            ));
        }
    };
    let jwks = match get_jwks(&jwks_url).await {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!(
                target: "aisix::auth",
                provider = %prov.name,
                issuer = %prov.issuer,
                error = %e,
                "cannot fetch the trust provider's JWKS",
            );
            return Err(deny(
                state,
                "jwks_unavailable",
                &iss,
                kid.as_deref(),
                ProxyError::JwksUnavailable,
            ));
        }
    };

    let mut candidates = candidate_keys(&jwks, kid.as_deref());
    if candidates.is_empty() {
        // Unknown (or absent-yet-unmatched) kid: the identity provider may
        // have just rotated its keys — refetch once, rate-limited.
        if let Some(fresh) = refresh_jwks_rate_limited(&jwks_url).await {
            candidates = candidate_keys(&fresh, kid.as_deref());
        }
    }
    if candidates.is_empty() {
        return Err(deny(
            state,
            "jwt_unknown_kid",
            &iss,
            kid.as_deref(),
            ProxyError::JwtInvalid,
        ));
    }

    // ── Signature + registered claims ────────────────────────────────
    let claims = match validate_with_keys(token, header.alg, prov, &candidates) {
        Ok(c) => c,
        Err((reason, err)) => return Err(deny(state, reason, &iss, kid.as_deref(), err)),
    };

    // ── Provider claim requirements ──────────────────────────────────
    if let Err(reason) = check_provider_claims(&claims, prov) {
        return Err(deny(
            state,
            reason,
            &iss,
            kid.as_deref(),
            ProxyError::JwtClaimsRejected,
        ));
    }

    // ── Identity mapping ─────────────────────────────────────────────
    let Some(subject) = nested_claim(&claims, &prov.identity_claim).and_then(|v| v.as_str()) else {
        return Err(deny(
            state,
            "jwt_identity_claim_missing",
            &iss,
            kid.as_deref(),
            ProxyError::JwtIdentityUnmapped,
        ));
    };

    let Some(entry) = key_for_subject(&snapshot, subject) else {
        tracing::warn!(
            target: "aisix::auth",
            method = "jwt",
            reason = "jwt_identity_unmapped",
            provider = %prov.name,
            issuer = %iss,
            subject = %subject,
            "rejected inbound JWT: no API key carries this jwt_subject",
        );
        state
            .metrics
            .record_auth_decision("jwt", false, "jwt_identity_unmapped");
        return Err(ProxyError::JwtIdentityUnmapped);
    };

    // Same lifecycle enforcement as the API-key path (#933).
    if entry.value.disabled {
        return Err(deny(
            state,
            "key_disabled",
            &iss,
            kid.as_deref(),
            ProxyError::ApiKeyDisabled,
        ));
    }
    if entry.value.is_expired_at(chrono::Utc::now()) {
        return Err(deny(
            state,
            "key_expired",
            &iss,
            kid.as_deref(),
            ProxyError::ApiKeyExpired,
        ));
    }

    state.metrics.record_auth_decision("jwt", true, "");
    tracing::debug!(
        target: "aisix::auth",
        method = "jwt",
        provider = %prov.name,
        issuer = %iss,
        subject = %subject,
        api_key_id = %entry.id,
        "jwt authentication succeeded",
    );
    Ok(AuthenticatedKey { entry })
}

/// Record a denial on the metric + decision log and hand back the error.
/// The raw token never appears here — only the reason class and the
/// token's routing metadata (issuer / kid).
fn deny(
    state: &ProxyState,
    reason: &'static str,
    issuer: &str,
    kid: Option<&str>,
    err: ProxyError,
) -> ProxyError {
    state.metrics.record_auth_decision("jwt", false, reason);
    tracing::warn!(
        target: "aisix::auth",
        method = "jwt",
        reason = %reason,
        issuer = %issuer,
        kid = %kid.unwrap_or(""),
        "rejected inbound JWT",
    );
    err
}

/// Verify signature + registered claims against each candidate key.
/// Signature/algorithm mismatches try the next key (rotation overlap with
/// an absent `kid`); claim-level failures are final — they read the same
/// for every key.
fn validate_with_keys(
    token: &str,
    alg: Algorithm,
    prov: &OidcProvider,
    keys: &[DecodingKey],
) -> Result<serde_json::Value, (&'static str, ProxyError)> {
    let mut validation = Validation::new(alg);
    validation.set_issuer(&[&prov.issuer]);
    validation.set_audience(&prov.audiences);
    // `aud`/`iss` are only checked when present — requiring them makes
    // absence a rejection (default deny), alongside the always-required
    // `exp`.
    validation.set_required_spec_claims(&["exp", "iss", "aud"]);
    validation.leeway = prov.leeway_secs;
    validation.validate_nbf = true;

    let mut last: Option<jsonwebtoken::errors::Error> = None;
    for key in keys {
        match jsonwebtoken::decode::<serde_json::Value>(token, key, &validation) {
            Ok(data) => return Ok(data.claims),
            Err(e) => {
                use jsonwebtoken::errors::ErrorKind;
                let retryable = matches!(
                    e.kind(),
                    ErrorKind::InvalidSignature | ErrorKind::InvalidAlgorithm
                );
                last = Some(e);
                if !retryable {
                    break;
                }
            }
        }
    }

    use jsonwebtoken::errors::ErrorKind;
    let (reason, err) = match last.as_ref().map(jsonwebtoken::errors::Error::kind) {
        Some(ErrorKind::ExpiredSignature) => ("jwt_expired", ProxyError::JwtExpired),
        Some(ErrorKind::ImmatureSignature) => ("jwt_not_yet_valid", ProxyError::JwtInvalid),
        Some(ErrorKind::InvalidAudience) => ("jwt_audience_mismatch", ProxyError::JwtInvalid),
        Some(ErrorKind::InvalidIssuer) => ("jwt_issuer_mismatch", ProxyError::JwtInvalid),
        Some(ErrorKind::MissingRequiredClaim(_)) => ("jwt_missing_claim", ProxyError::JwtInvalid),
        Some(ErrorKind::InvalidSignature) => ("jwt_bad_signature", ProxyError::JwtInvalid),
        _ => ("jwt_invalid", ProxyError::JwtInvalid),
    };
    Err((reason, err))
}

/// Enforce the provider's `required_scopes` and `bound_claims`. Returns
/// the denial reason class on the first unmet requirement.
fn check_provider_claims(
    claims: &serde_json::Value,
    prov: &OidcProvider,
) -> Result<(), &'static str> {
    if !prov.required_scopes.is_empty() {
        let scopes = token_scopes(claims);
        if !prov
            .required_scopes
            .iter()
            .all(|req| scopes.iter().any(|s| s == req))
        {
            return Err("jwt_scope_missing");
        }
    }
    if let Some(bound) = &prov.bound_claims {
        for (path, expect) in bound {
            let matched = nested_claim(claims, path)
                .is_some_and(|actual| bound_claim_matches(actual, expect));
            if !matched {
                return Err("jwt_bound_claim_mismatch");
            }
        }
    }
    Ok(())
}

/// The token's granted scopes: a `scope` claim as the OAuth
/// space-delimited string, or as an array of strings.
fn token_scopes(claims: &serde_json::Value) -> Vec<String> {
    match claims.get("scope") {
        Some(serde_json::Value::String(s)) => s.split_whitespace().map(str::to_string).collect(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

/// Resolve a claim by path, dots traversing nested objects
/// (`realm_access.roles`).
fn nested_claim<'a>(claims: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut cur = claims;
    for seg in path.split('.') {
        cur = cur.get(seg)?;
    }
    Some(cur)
}

/// A bound-claim requirement holds when the claim equals — or, for array
/// claims, contains — one of the expected values. Non-string claim shapes
/// never match (deny by default).
fn bound_claim_matches(actual: &serde_json::Value, expect: &BoundClaimExpect) -> bool {
    match actual {
        serde_json::Value::String(s) => expect.accepted().any(|e| e == s),
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|v| v.as_str())
            .any(|s| expect.accepted().any(|e| e == s)),
        _ => false,
    }
}

/// Decoding keys to try: an exact `kid` match when the token names one,
/// otherwise every signature-use key in the set (bounded) — an identity
/// provider mid-rotation may publish two keys, and some omit `kid`
/// entirely.
fn candidate_keys(jwks: &JwkSet, kid: Option<&str>) -> Vec<DecodingKey> {
    match kid {
        Some(kid) => jwks
            .find(kid)
            .and_then(|jwk| DecodingKey::from_jwk(jwk).ok())
            .into_iter()
            .collect(),
        None => jwks
            .keys
            .iter()
            .filter(|jwk| {
                jwk.common
                    .public_key_use
                    .as_ref()
                    .is_none_or(|u| matches!(u, jsonwebtoken::jwk::PublicKeyUse::Signature))
            })
            .filter_map(|jwk| DecodingKey::from_jwk(jwk).ok())
            .take(MAX_KEYS_TRIED)
            .collect(),
    }
}

// ── JWKS fetch + cache ───────────────────────────────────────────────

struct JwksEntry {
    /// The last successfully fetched key set and when it landed.
    jwks: Option<(Arc<JwkSet>, Instant)>,
    /// Last fetch attempt, success or failure — the rate-limit clock.
    last_attempt: Option<Instant>,
}

/// Process-global JWKS cache keyed by URL. Guards are held only for map
/// lookups/inserts, never across an await; concurrent misses may fetch in
/// parallel (each result is valid — last insert wins).
fn jwks_cache() -> &'static RwLock<HashMap<String, JwksEntry>> {
    static CACHE: OnceLock<RwLock<HashMap<String, JwksEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Discovery results keyed by issuer: the resolved JWKS URL.
fn discovery_cache() -> &'static RwLock<HashMap<String, (String, Instant)>> {
    static CACHE: OnceLock<RwLock<HashMap<String, (String, Instant)>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Shared HTTP client for JWKS / discovery fetches. Redirects are
/// disabled — a key endpoint never legitimately redirects, and following
/// one would fetch trust material from wherever it points.
fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        aisix_gateway::client_builder()
            .timeout(JWKS_FETCH_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_default()
    })
}

/// The JWKS URL for a provider: its configured `jwks_uri`, or the
/// `jwks_uri` advertised by the issuer's OIDC discovery document
/// (cached; a stale value keeps serving when a re-fetch fails).
async fn resolve_jwks_url(prov: &OidcProvider) -> Result<String, String> {
    if let Some(u) = &prov.jwks_uri {
        return Ok(u.clone());
    }
    let now = Instant::now();
    if let Some((url, at)) = discovery_cache().read().unwrap().get(&prov.issuer) {
        if now.duration_since(*at) < JWKS_TTL {
            return Ok(url.clone());
        }
    }
    let discovery_url = format!(
        "{}/.well-known/openid-configuration",
        prov.issuer.trim_end_matches('/')
    );
    match fetch_json(&discovery_url).await {
        Ok(doc) => {
            let jwks_uri = doc
                .get("jwks_uri")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "discovery document carries no jwks_uri".to_string())?
                .to_string();
            discovery_cache()
                .write()
                .unwrap()
                .insert(prov.issuer.clone(), (jwks_uri.clone(), now));
            Ok(jwks_uri)
        }
        Err(e) => {
            // Serve the stale resolution rather than failing auth outright.
            if let Some((url, _)) = discovery_cache().read().unwrap().get(&prov.issuer) {
                tracing::warn!(
                    target: "aisix::auth",
                    issuer = %prov.issuer,
                    error = %e,
                    "OIDC discovery re-fetch failed; keeping the previously resolved JWKS URL",
                );
                return Ok(url.clone());
            }
            Err(format!("OIDC discovery failed: {e}"))
        }
    }
}

/// The cached key set for `url`, fetching when absent or past
/// [`JWKS_TTL`]. A failed re-fetch keeps serving the stale set
/// (network-partition tolerance); with nothing cached the error
/// propagates and the request fails closed as retryable.
async fn get_jwks(url: &str) -> Result<Arc<JwkSet>, String> {
    let now = Instant::now();
    if let Some(entry) = jwks_cache().read().unwrap().get(url) {
        if let Some((jwks, fetched_at)) = &entry.jwks {
            if now.duration_since(*fetched_at) < JWKS_TTL {
                return Ok(jwks.clone());
            }
        }
    }
    refresh_jwks(url).await
}

/// One fetch for an unknown `kid`, suppressed inside
/// [`JWKS_REFRESH_MIN_INTERVAL`] of the previous attempt.
async fn refresh_jwks_rate_limited(url: &str) -> Option<Arc<JwkSet>> {
    let now = Instant::now();
    if let Some(entry) = jwks_cache().read().unwrap().get(url) {
        if let Some(at) = entry.last_attempt {
            if now.duration_since(at) < JWKS_REFRESH_MIN_INTERVAL {
                return None;
            }
        }
    }
    refresh_jwks(url).await.ok()
}

async fn refresh_jwks(url: &str) -> Result<Arc<JwkSet>, String> {
    // Stamp the attempt before awaiting so a slow endpoint is not
    // hammered by concurrent unknown-kid refreshes.
    {
        let mut map = jwks_cache().write().unwrap();
        map.entry(url.to_string())
            .or_insert(JwksEntry {
                jwks: None,
                last_attempt: None,
            })
            .last_attempt = Some(Instant::now());
    }
    match fetch_json(url).await.and_then(|v| {
        serde_json::from_value::<JwkSet>(v).map_err(|e| format!("not a JWKS document: {e}"))
    }) {
        Ok(set) => {
            let arc = Arc::new(set);
            jwks_cache()
                .write()
                .unwrap()
                .entry(url.to_string())
                .and_modify(|e| e.jwks = Some((arc.clone(), Instant::now())))
                .or_insert(JwksEntry {
                    jwks: Some((arc.clone(), Instant::now())),
                    last_attempt: Some(Instant::now()),
                });
            Ok(arc)
        }
        Err(e) => {
            if let Some(entry) = jwks_cache().read().unwrap().get(url) {
                if let Some((stale, _)) = &entry.jwks {
                    tracing::warn!(
                        target: "aisix::auth",
                        error = %e,
                        "JWKS re-fetch failed; keeping the previously fetched key set",
                    );
                    return Ok(stale.clone());
                }
            }
            Err(e)
        }
    }
}

async fn fetch_json(url: &str) -> Result<serde_json::Value, String> {
    let resp = http_client()
        .get(url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("endpoint returned status {}", resp.status()));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("reading response failed: {e}"))?;
    if bytes.len() > JWKS_MAX_BYTES {
        return Err(format!("response exceeds {JWKS_MAX_BYTES} bytes"));
    }
    serde_json::from_slice(&bytes).map_err(|e| format!("response is not JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::resource::ResourceEntry;
    use jsonwebtoken::{encode, EncodingKey, Header};

    /// Test-only RSA keypair. The private PEM signs fixture tokens; the
    /// JWK below is its public half (kid `test-kid-1`).
    const TEST_RSA_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDfARbZauGK4bRk
UL0gWcsvGyFBMVW6eeNcAy7U0APH92H5DSImyf1WhnvfDareRkXFBhiHy6Bj0wfz
7yE7kgPNhXB0l4r8mFd3biTklxt5fDKqvJZd473fFOkiM//DjB62lodXfDLwhr0o
zQi0xCnPzMyzQx9EVR1v1JwW/9lS4QaEgiVGDES9mh0kfnszw7sH5IFwKz2BgtHS
gHJ+Wykr7hB7DY103OxE69BXKA2bJ+k/0ai8dQiSzgfIEkailvy/2wZoOfVbEfWp
wXPuP+ipqn/9c9mbbjMRtUHOjgBQvqiwjix21nh8ZoeCA8z/YuvdgXXTJgoG0h+I
WLyQXSOxAgMBAAECggEABPxNak3uk0Ae3Cab8ScLblcBGX0vXqG5TgYk3A13JYIn
1r1kQpFoewXlq2PEVVTP3CrvOHX6dNDeetB2oed5SJ/PlvkJBUL9+EW7ncACarxh
QO+XaFZI7pL/7/ZRT6oIc7+OG2FuSByoX6BPLgS8BJEeZcbojOAJmPBGub2S5RHn
x/g/a58W+AmudYZY+aqVg84SBu8FQF7J3ygvT2we6k0xu7nPp23lpF9zQdLcDRlM
d1Dqu3JyQApKO4xtfcQFGbJzq6fIyaFX08mkQeewkek3XXf2JUmcnfCx37gOv8hy
7k8nPT1vzzFIVJFx/f+W91KmixmrNU7mlvpuHRBlKwKBgQD+vtCwzkU6wvXr6OaL
R3iT+QSt49aMHIi6u0SSDJnjVoQDVXivyybVRRCwYWXzng5ajt1fs9dsW2ELxco3
mCrf5ayrsUhjytSEvCXXfpomA75518s+r3Nlu7qccHTvlRxLzLk1rQ9UilEYVT3s
DF4xbu/91rJ9gNWiocv4xGa3PwKBgQDgGkEvTMsoiJQW0Drs+rohBThw24Bt0wvP
wSwgz71PxwJvEIT8qeCJDBINiXeTDPe8pxpO+As+iaBdJ5YQ7ctyuGvLA6892zto
/AcszvCL8R6sxcPt9ak4/GhY0weKT4DsjjPNOPWFY9ebZ/xD/6R9lb6Ksi+G/pXM
CusKpfzZDwKBgAw8hjG39sNX0hA+47QU/sm80Gi55Phd9oNhs22AhXPSGA1A8ccf
7wGXi7GtPARztyTKb//E17gwu3yhR5FcEdMnaR/mKCADAipOD1NGlYj17RRVNUIR
k21zkwcor7VCaFWLw+m8IlxhOHv+vDa2cV/WgFilE3XL1nc1ZmLQrE5pAoGAIig+
STxWNs5ia/u/D4HDvuaxzJnYQGULhtX1qOag/zjhCRamfnBSFfFuCvwp6pLua6W4
n9K0vAp0E97Fw7zK5qhvXZkpK69vpbfMTCsahOnyd/kIvQtViKcILIm1u4IUr3mZ
Ma191p/6K+i0jZS4eJ/LVA6GqffB00DSxGO6X0cCgYAA+KRVMdHHBiuL3XO0srlR
0lY0cuVX8TTsJf1AkLH8rutn3Xa7maLVOrNoUnhE6j5UmzojlzMGUTmi1sryMipU
MFt+Fn9pwKAtrgAFlmGhAsOBmC4fnn0jNN4aV6B5gSbQFLSGXmF3qCJHTLT2gPR3
jyxumGxNpoIV8LlzsMsaWQ==
-----END PRIVATE KEY-----";

    const TEST_JWKS: &str = r#"{"keys":[{"kty":"RSA","kid":"test-kid-1","use":"sig","alg":"RS256","n":"3wEW2WrhiuG0ZFC9IFnLLxshQTFVunnjXAMu1NADx_dh-Q0iJsn9VoZ73w2q3kZFxQYYh8ugY9MH8-8hO5IDzYVwdJeK_JhXd24k5JcbeXwyqryWXeO93xTpIjP_w4wetpaHV3wy8Ia9KM0ItMQpz8zMs0MfRFUdb9ScFv_ZUuEGhIIlRgxEvZodJH57M8O7B-SBcCs9gYLR0oByflspK-4Qew2NdNzsROvQVygNmyfpP9GovHUIks4HyBJGopb8v9sGaDn1WxH1qcFz7j_oqap__XPZm24zEbVBzo4AUL6osI4sdtZ4fGaHggPM_2Lr3YF10yYKBtIfiFi8kF0jsQ","e":"AQAB"}]}"#;

    fn test_provider(json: &str) -> OidcProvider {
        serde_json::from_str(json).unwrap()
    }

    fn base_provider() -> OidcProvider {
        test_provider(
            r#"{
              "name": "test-idp",
              "issuer": "https://idp.test/realms/agents",
              "audiences": ["aisix"]
            }"#,
        )
    }

    fn encoding_key() -> EncodingKey {
        EncodingKey::from_rsa_pem(TEST_RSA_PEM.as_bytes()).unwrap()
    }

    fn decoding_keys() -> Vec<DecodingKey> {
        let jwks: JwkSet = serde_json::from_str(TEST_JWKS).unwrap();
        candidate_keys(&jwks, Some("test-kid-1"))
    }

    fn sign(claims: &serde_json::Value) -> String {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some("test-kid-1".to_string());
        encode(&header, claims, &encoding_key()).unwrap()
    }

    fn future() -> i64 {
        chrono::Utc::now().timestamp() + 3600
    }

    fn valid_claims() -> serde_json::Value {
        serde_json::json!({
            "iss": "https://idp.test/realms/agents",
            "aud": "aisix",
            "sub": "agent-1",
            "exp": future(),
        })
    }

    #[test]
    fn looks_like_jwt_accepts_real_tokens_only() {
        assert!(looks_like_jwt(&sign(&valid_claims())));
        // Generated gateway keys.
        assert!(!looks_like_jwt("sk-3f5a1b2c"));
        // Custom-imported keys that merely contain dots: segments do not
        // decode to a JOSE header.
        assert!(!looks_like_jwt("a.b.c"));
        assert!(!looks_like_jwt("my.custom.key"));
        // Wrong segment counts.
        assert!(!looks_like_jwt("a.b"));
        assert!(!looks_like_jwt("a.b.c.d"));
        assert!(!looks_like_jwt(""));
        // A base64url JSON first segment without `alg` is not a JWT.
        let not_jose = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"{\"a\":1}");
        assert!(!looks_like_jwt(&format!("{not_jose}.x.y")));
    }

    #[test]
    fn validate_accepts_a_well_formed_token() {
        let claims = validate_with_keys(
            &sign(&valid_claims()),
            Algorithm::RS256,
            &base_provider(),
            &decoding_keys(),
        )
        .unwrap();
        assert_eq!(claims["sub"], "agent-1");
    }

    #[test]
    fn validate_rejects_expired_token_as_jwt_expired() {
        let mut c = valid_claims();
        c["exp"] = serde_json::json!(chrono::Utc::now().timestamp() - 3600);
        let (reason, err) = validate_with_keys(
            &sign(&c),
            Algorithm::RS256,
            &base_provider(),
            &decoding_keys(),
        )
        .unwrap_err();
        assert_eq!(reason, "jwt_expired");
        assert!(matches!(err, ProxyError::JwtExpired));
    }

    #[test]
    fn validate_requires_exp() {
        let mut c = valid_claims();
        c.as_object_mut().unwrap().remove("exp");
        let (reason, _) = validate_with_keys(
            &sign(&c),
            Algorithm::RS256,
            &base_provider(),
            &decoding_keys(),
        )
        .unwrap_err();
        assert_eq!(reason, "jwt_missing_claim");
    }

    #[test]
    fn validate_requires_audience_presence_and_match() {
        let mut missing = valid_claims();
        missing.as_object_mut().unwrap().remove("aud");
        let (reason, _) = validate_with_keys(
            &sign(&missing),
            Algorithm::RS256,
            &base_provider(),
            &decoding_keys(),
        )
        .unwrap_err();
        assert_eq!(reason, "jwt_missing_claim");

        let mut wrong = valid_claims();
        wrong["aud"] = serde_json::json!("someone-else");
        let (reason, _) = validate_with_keys(
            &sign(&wrong),
            Algorithm::RS256,
            &base_provider(),
            &decoding_keys(),
        )
        .unwrap_err();
        assert_eq!(reason, "jwt_audience_mismatch");

        // Array audiences match when any element is accepted.
        let mut array = valid_claims();
        array["aud"] = serde_json::json!(["other", "aisix"]);
        assert!(validate_with_keys(
            &sign(&array),
            Algorithm::RS256,
            &base_provider(),
            &decoding_keys()
        )
        .is_ok());
    }

    #[test]
    fn validate_rejects_wrong_issuer() {
        let mut c = valid_claims();
        c["iss"] = serde_json::json!("https://evil.test");
        let (reason, _) = validate_with_keys(
            &sign(&c),
            Algorithm::RS256,
            &base_provider(),
            &decoding_keys(),
        )
        .unwrap_err();
        assert_eq!(reason, "jwt_issuer_mismatch");
    }

    #[test]
    fn validate_rejects_future_nbf_and_accepts_past_nbf() {
        let mut c = valid_claims();
        c["nbf"] = serde_json::json!(future());
        let (reason, _) = validate_with_keys(
            &sign(&c),
            Algorithm::RS256,
            &base_provider(),
            &decoding_keys(),
        )
        .unwrap_err();
        assert_eq!(reason, "jwt_not_yet_valid");

        let mut ok = valid_claims();
        ok["nbf"] = serde_json::json!(chrono::Utc::now().timestamp() - 60);
        assert!(validate_with_keys(
            &sign(&ok),
            Algorithm::RS256,
            &base_provider(),
            &decoding_keys()
        )
        .is_ok());
    }

    #[test]
    fn validate_rejects_tampered_signature() {
        let token = sign(&valid_claims());
        let mut parts: Vec<String> = token.split('.').map(str::to_string).collect();
        // Re-encode the payload with a widened scope; the signature no
        // longer covers it.
        let mut payload = valid_claims();
        payload["sub"] = serde_json::json!("agent-admin");
        parts[1] = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let tampered = parts.join(".");
        let (reason, _) = validate_with_keys(
            &tampered,
            Algorithm::RS256,
            &base_provider(),
            &decoding_keys(),
        )
        .unwrap_err();
        assert_eq!(reason, "jwt_bad_signature");
    }

    #[test]
    fn leeway_tolerates_recent_expiry() {
        let mut prov = base_provider();
        prov.leeway_secs = 120;
        let mut c = valid_claims();
        c["exp"] = serde_json::json!(chrono::Utc::now().timestamp() - 30);
        assert!(validate_with_keys(&sign(&c), Algorithm::RS256, &prov, &decoding_keys()).is_ok());
    }

    #[test]
    fn scope_and_bound_claim_checks() {
        let prov = test_provider(
            r#"{
              "name": "test-idp",
              "issuer": "https://idp.test/realms/agents",
              "audiences": ["aisix"],
              "required_scopes": ["ai.access"],
              "bound_claims": {
                "department": "ai-lab",
                "realm_access.roles": ["agent", "batch"]
              }
            }"#,
        );

        let mut good = valid_claims();
        good["scope"] = serde_json::json!("openid ai.access");
        good["department"] = serde_json::json!("ai-lab");
        good["realm_access"] = serde_json::json!({"roles": ["other", "agent"]});
        assert!(check_provider_claims(&good, &prov).is_ok());

        // Scope may also arrive as an array.
        let mut array_scope = good.clone();
        array_scope["scope"] = serde_json::json!(["ai.access"]);
        assert!(check_provider_claims(&array_scope, &prov).is_ok());

        let mut no_scope = good.clone();
        no_scope["scope"] = serde_json::json!("openid");
        assert_eq!(
            check_provider_claims(&no_scope, &prov),
            Err("jwt_scope_missing")
        );

        let mut wrong_dept = good.clone();
        wrong_dept["department"] = serde_json::json!("finance");
        assert_eq!(
            check_provider_claims(&wrong_dept, &prov),
            Err("jwt_bound_claim_mismatch")
        );

        // A missing bound claim denies — never a silent pass.
        let mut missing = good.clone();
        missing.as_object_mut().unwrap().remove("department");
        assert_eq!(
            check_provider_claims(&missing, &prov),
            Err("jwt_bound_claim_mismatch")
        );

        // Non-string claim shapes never match.
        let mut numeric = good.clone();
        numeric["department"] = serde_json::json!(7);
        assert_eq!(
            check_provider_claims(&numeric, &prov),
            Err("jwt_bound_claim_mismatch")
        );
    }

    #[test]
    fn candidate_keys_selects_by_kid_and_falls_back_to_all() {
        let jwks: JwkSet = serde_json::from_str(TEST_JWKS).unwrap();
        assert_eq!(candidate_keys(&jwks, Some("test-kid-1")).len(), 1);
        assert!(candidate_keys(&jwks, Some("rotated-away")).is_empty());
        // No kid on the token: every signature key is a candidate.
        assert_eq!(candidate_keys(&jwks, None).len(), 1);
    }

    #[test]
    fn provider_and_subject_selection_tie_break_on_lowest_id() {
        let snapshot = AisixSnapshot::new();
        let mk = |id: &str, enabled: bool| {
            let mut p = base_provider();
            p.enabled = enabled;
            snapshot.oidc_providers.insert(ResourceEntry::new(id, p, 1));
        };
        mk("b-provider", true);
        mk("a-provider", true);
        mk("0-disabled", false);
        let picked = provider_for_issuer(&snapshot, "https://idp.test/realms/agents").unwrap();
        assert_eq!(picked.id, "a-provider");
        assert!(provider_for_issuer(&snapshot, "https://other.test").is_none());

        let mk_key = |id: &str, subject: Option<&str>| {
            let mut k: ApiKey =
                serde_json::from_str(r#"{"key_hash":"h","allowed_models":["*"]}"#).unwrap();
            k.jwt_subject = subject.map(str::to_string);
            // Distinct key_hash per row so the by-name index stays unique.
            k.key_hash = format!("hash-{id}");
            snapshot.apikeys.insert(ResourceEntry::new(id, k, 1));
        };
        mk_key("k-2", Some("agent-1"));
        mk_key("k-1", Some("agent-1"));
        mk_key("k-3", Some("agent-2"));
        mk_key("k-4", None);
        assert_eq!(key_for_subject(&snapshot, "agent-1").unwrap().id, "k-1");
        assert_eq!(key_for_subject(&snapshot, "agent-2").unwrap().id, "k-3");
        assert!(key_for_subject(&snapshot, "agent-9").is_none());
    }

    #[test]
    fn nested_claim_traverses_dots() {
        let v = serde_json::json!({"a": {"b": {"c": "x"}}, "flat": "y"});
        assert_eq!(nested_claim(&v, "a.b.c").unwrap(), "x");
        assert_eq!(nested_claim(&v, "flat").unwrap(), "y");
        assert!(nested_claim(&v, "a.missing").is_none());
    }

    #[test]
    fn unverified_issuer_reads_the_payload() {
        assert_eq!(
            unverified_issuer(&sign(&valid_claims())).as_deref(),
            Some("https://idp.test/realms/agents")
        );
        assert!(unverified_issuer("sk-abc").is_none());
    }
}
