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

use aisix_core::models::{BoundClaimExpect, ClaimMapping, ClaimMatch, ClaimMatchOp, OidcProvider};
use aisix_core::resource::ResourceEntry;
use aisix_core::{AisixSnapshot, ApiKey};
use base64::Engine;
use jsonwebtoken::jwk::JwkSet;
use jsonwebtoken::{Algorithm, DecodingKey, Validation};

use crate::auth::{AuthenticatedKey, JwtIdentity};
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

/// Upper bound on a bearer we will treat as a JWT. A real IdP token is a
/// few KB; the cap stops a several-hundred-KB `Authorization` header from
/// driving the base64/JSON work (done up to three times per request)
/// before anything is verified.
const MAX_JWT_BYTES: usize = 16 * 1024;

/// True when the bearer has the structural shape of a JWT: within the
/// size cap, three non-empty dot-separated segments whose first segment
/// base64url-decodes to a JSON object carrying `alg` (a JOSE header). The
/// header check keeps custom-imported API keys that merely contain dots
/// on the API-key path.
pub(crate) fn looks_like_jwt(token: &str) -> bool {
    if token.len() > MAX_JWT_BYTES {
        return false;
    }
    // Exactly three non-empty segments — checked on the iterator so the
    // per-request path allocates nothing.
    let mut parts = token.splitn(4, '.');
    let (Some(header), Some(payload), Some(sig), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    if header.is_empty() || payload.is_empty() || sig.is_empty() {
        return false;
    }
    matches!(b64url_json(header), Some(v) if v.get("alg").is_some())
}

/// True when the snapshot has at least one enabled trust provider — the
/// gate for entering the JWT path at all. O(1) on deployments with no
/// providers configured (the common case).
pub(crate) fn any_enabled_provider(snapshot: &AisixSnapshot) -> bool {
    !snapshot.oidc_providers.is_empty() && snapshot.oidc_providers.any(|e| e.value.enabled)
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

/// The enabled provider matching `iss`. Fails closed on ambiguity: if
/// two enabled providers claim the same issuer their audience/scope/
/// claim policies differ, so silently picking one would apply the wrong
/// policy — the request is denied instead. The control plane enforces
/// per-environment issuer uniqueness and the file loader rejects
/// duplicates, so this only guards a transient etcd race or a CP bug.
fn provider_for_issuer(
    snapshot: &AisixSnapshot,
    iss: &str,
) -> Option<Arc<ResourceEntry<OidcProvider>>> {
    let (found, ambiguous) = snapshot
        .oidc_providers
        .find_unique_by(|e| e.value.enabled && e.value.issuer == iss);
    if ambiguous {
        tracing::warn!(
            target: "aisix::auth",
            issuer = %clip(iss),
            "two enabled OIDC providers claim this issuer; failing closed — \
             their policies differ and neither can be chosen unambiguously",
        );
    }
    found
}

/// The API key bound to `subject` **as asserted by `provider_name`**,
/// plus whether the binding was ambiguous. A key whose `jwt_provider`
/// names a different trust provider is never a candidate: subjects are
/// namespaced by the provider that vouched for them, so a second
/// trusted provider cannot mint a token impersonating the first
/// provider's identity of the same name. Ambiguity (two keys sharing
/// one binding) is surfaced to the caller so it can fail closed rather
/// than fall through to the claim mappings — the CP enforces
/// `(jwt_provider, jwt_subject)` uniqueness and the file loader rejects
/// duplicates, so this only guards a transient race.
fn key_for_subject(
    snapshot: &AisixSnapshot,
    provider_name: &str,
    subject: &str,
) -> (Option<Arc<ResourceEntry<ApiKey>>>, bool) {
    let (found, ambiguous) = snapshot.apikeys.find_unique_by(|e| {
        e.value.jwt_subject.as_deref() == Some(subject)
            && e.value.jwt_provider.as_deref() == Some(provider_name)
    });
    if ambiguous {
        tracing::warn!(
            target: "aisix::auth",
            provider = %clip(provider_name),
            "two API keys share one jwt binding; failing closed — the identity \
             is ambiguous and neither key's limits can be applied",
        );
    }
    (found, ambiguous)
}

/// The highest-priority enabled claim mapping for `provider_name` whose
/// conditions all hold against the verified claims. Candidates are
/// ordered by `(priority, name, id)` — a total order, so evaluation is
/// deterministic across replicas and across snapshot updates even if a
/// control-plane bug ever produced duplicate names — and the same token
/// always resolves the same mapping.
fn matching_claim_mapping(
    snapshot: &AisixSnapshot,
    provider_name: &str,
    claims: &serde_json::Value,
) -> Option<Arc<ResourceEntry<ClaimMapping>>> {
    let mut candidates: Vec<_> = snapshot
        .claim_mappings
        .entries()
        .into_iter()
        .filter(|e| e.value.enabled && e.value.jwt_provider == provider_name)
        .collect();
    candidates.sort_by(|a, b| {
        (a.value.priority, a.value.name.as_str(), a.id.as_str()).cmp(&(
            b.value.priority,
            b.value.name.as_str(),
            b.id.as_str(),
        ))
    });
    candidates
        .into_iter()
        .find(|e| e.value.match_.iter().all(|m| claim_match_holds(claims, m)))
}

/// Whether one claim condition holds. A missing claim, or a claim whose
/// JSON type does not fit the operator, never matches (default deny):
/// `exact` requires a string claim equal to one of the accepted values,
/// `contains` an array of strings containing one of them.
fn claim_match_holds(claims: &serde_json::Value, m: &ClaimMatch) -> bool {
    let Some(actual) = nested_claim(claims, &m.claim) else {
        return false;
    };
    match m.op {
        ClaimMatchOp::Exact => actual
            .as_str()
            .is_some_and(|s| m.values.iter().any(|v| v == s)),
        ClaimMatchOp::Contains => actual.as_array().is_some_and(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str())
                .any(|s| m.values.iter().any(|v| v == s))
        }),
    }
}

/// Authenticate a JWT-shaped bearer. Called from the auth choke point
/// once [`looks_like_jwt`] and [`any_enabled_provider`] both hold, with
/// the snapshot the gate already loaded (avoids a second atomic load;
/// any change between them fails closed to `jwt_untrusted_issuer`).
pub(crate) async fn authenticate_jwt(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    token: &str,
    ctx: crate::auth::DenialContext<'_>,
) -> Result<AuthenticatedKey, ProxyError> {
    let d = Denier { state, ctx };
    let header = match jsonwebtoken::decode_header(token) {
        Ok(h) => h,
        Err(_) => return Err(deny(d, "jwt_malformed", "", None, ProxyError::JwtInvalid)),
    };
    let kid = header.kid.clone();

    let Some(iss) = unverified_issuer(token) else {
        return Err(deny(
            d,
            "jwt_missing_issuer",
            "",
            kid.as_deref(),
            ProxyError::JwtInvalid,
        ));
    };

    let Some(provider) = provider_for_issuer(snapshot, &iss) else {
        return Err(deny(
            d,
            "jwt_untrusted_issuer",
            &iss,
            kid.as_deref(),
            ProxyError::JwtInvalid,
        ));
    };
    let prov = &provider.value;

    if !ALLOWED_ALGS.contains(&header.alg) {
        return Err(deny(
            d,
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
                d,
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
                d,
                "jwks_unavailable",
                &iss,
                kid.as_deref(),
                ProxyError::JwksUnavailable,
            ));
        }
    };

    let mut candidates = candidate_keys(&jwks, kid.as_deref(), header.alg);
    if candidates.is_empty() {
        // Unknown (or absent-yet-unmatched) kid: the identity provider may
        // have just rotated its keys — refetch once, rate-limited.
        if let Some(fresh) = refresh_jwks_rate_limited(&jwks_url).await {
            candidates = candidate_keys(&fresh, kid.as_deref(), header.alg);
        }
    }
    if candidates.is_empty() {
        return Err(deny(
            d,
            "jwt_unknown_kid",
            &iss,
            kid.as_deref(),
            ProxyError::JwtInvalid,
        ));
    }

    // ── Signature + registered claims ────────────────────────────────
    let claims = match validate_with_keys(token, header.alg, prov, &candidates) {
        Ok(c) => c,
        Err((reason, err)) => return Err(deny(d, reason, &iss, kid.as_deref(), err)),
    };

    // ── Provider claim requirements ──────────────────────────────────
    if let Err(reason) = check_provider_claims(&claims, prov) {
        return Err(deny(
            d,
            reason,
            &iss,
            kid.as_deref(),
            ProxyError::JwtClaimsRejected,
        ));
    }

    // ── Identity mapping ─────────────────────────────────────────────
    let Some(subject) = nested_claim(&claims, &prov.identity_claim).and_then(|v| v.as_str()) else {
        return Err(deny(
            d,
            "jwt_identity_claim_missing",
            &iss,
            kid.as_deref(),
            ProxyError::JwtIdentityUnmapped,
        ));
    };

    // The direct `(jwt_provider, jwt_subject)` key binding is
    // authoritative for its subject — including its disabled/expired
    // lifecycle. Claim mappings only admit identities no key binds
    // explicitly, so adding a mapping can never reroute (or re-enable)
    // an identity an operator pinned to a specific key. An AMBIGUOUS
    // binding fails closed here for the same reason: the subject *is*
    // bound, just not resolvably, and letting it fall through to the
    // mappings would hand a mis-provisioned identity whatever a rule
    // grants.
    let (bound, ambiguous) = key_for_subject(snapshot, &prov.name, subject);
    if ambiguous {
        tracing::warn!(
            target: "aisix::auth",
            method = "jwt",
            reason = "jwt_binding_ambiguous",
            provider = %clip(&prov.name),
            issuer = %clip(&iss),
            subject = ?clip(subject),
            "rejected inbound JWT: two API keys claim this identity's binding",
        );
        state
            .metrics
            .record_auth_decision("jwt", false, "jwt_binding_ambiguous");
        return Err(ProxyError::JwtIdentityUnmapped);
    }
    let (entry, claim_mapping) = match bound {
        Some(entry) => (entry, None),
        None => match matching_claim_mapping(snapshot, &prov.name, &claims) {
            Some(mapping) => {
                let Some(entry) = snapshot
                    .apikeys
                    .get_by_id(&mapping.value.resolve.api_key_id)
                else {
                    tracing::warn!(
                        target: "aisix::auth",
                        method = "jwt",
                        reason = "claim_mapping_target_missing",
                        provider = %clip(&prov.name),
                        issuer = %clip(&iss),
                        subject = ?clip(subject),
                        claim_mapping = %clip(&mapping.value.name),
                        "rejected inbound JWT: the matched claim mapping resolves to \
                         an api key that does not exist",
                    );
                    state.metrics.record_auth_decision(
                        "jwt",
                        false,
                        "claim_mapping_target_missing",
                    );
                    return Err(ProxyError::JwtIdentityUnmapped);
                };
                (entry, Some(mapping.value.name.clone()))
            }
            None => {
                tracing::warn!(
                    target: "aisix::auth",
                    method = "jwt",
                    reason = "jwt_identity_unmapped",
                    provider = %clip(&prov.name),
                    issuer = %clip(&iss),
                    subject = ?clip(subject),
                    "rejected inbound JWT: no API key binds this identity to this \
                     provider and no claim mapping matched",
                );
                state
                    .metrics
                    .record_auth_decision("jwt", false, "jwt_identity_unmapped");
                return Err(ProxyError::JwtIdentityUnmapped);
            }
        },
    };

    // Same lifecycle enforcement as the API-key path (#933).
    if entry.value.disabled {
        return Err(deny(
            d,
            "key_disabled",
            &iss,
            kid.as_deref(),
            ProxyError::ApiKeyDisabled,
        ));
    }
    if entry.value.is_expired_at(chrono::Utc::now()) {
        return Err(deny(
            d,
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
        claim_mapping = ?claim_mapping,
        "jwt authentication succeeded",
    );
    Ok(AuthenticatedKey {
        entry,
        jwt: Some(Arc::new(JwtIdentity {
            subject: subject.to_string(),
            provider: prov.name.clone(),
            claim_mapping,
        })),
    })
}

/// Cap on attacker-controlled token metadata reproduced in the decision
/// log. `kid`, and `iss` before the issuer allow-list matches, come
/// straight from an unauthenticated token, so they are logged with
/// `Debug` (which escapes newlines and control bytes) and truncated —
/// a probe cannot forge log lines or inflate log volume through them.
const LOGGED_METADATA_MAX: usize = 128;

fn clip(s: &str) -> &str {
    match s.char_indices().nth(LOGGED_METADATA_MAX) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Record a denial on the metric + decision log and hand back the error.
/// The raw token never appears here — only the reason class and the
/// token's routing metadata (issuer / kid), escaped and truncated.
///
/// Pre-allow-list reason classes (a malformed token, an untrusted or
/// missing issuer) are the scanner-probe shapes: they carry no operator
/// signal beyond the metric, so they log at `debug`. Once a trust
/// provider has matched, a denial names a real configured issuer and is
/// worth a `warn`.
fn deny(
    d: Denier<'_>,
    reason: &'static str,
    issuer: &str,
    kid: Option<&str>,
    err: ProxyError,
) -> ProxyError {
    let Denier { state, ctx } = d;
    state.metrics.record_auth_decision("jwt", false, reason);
    let pre_match = matches!(
        reason,
        "jwt_malformed" | "jwt_missing_issuer" | "jwt_untrusted_issuer" | "jwt_alg_not_allowed"
    );
    if pre_match {
        tracing::debug!(
            target: "aisix::auth",
            method = "jwt",
            reason = %reason,
            issuer = ?clip(issuer),
            kid = ?clip(kid.unwrap_or("")),
            http_method = %ctx.method,
            path = %ctx.path,
            request_id = %ctx.request_id,
            source_ip = %ctx.source_ip.resolve(),
            "rejected inbound JWT (pre-verification)",
        );
    } else {
        tracing::warn!(
            target: "aisix::auth",
            method = "jwt",
            reason = %reason,
            issuer = ?clip(issuer),
            kid = ?clip(kid.unwrap_or("")),
            http_method = %ctx.method,
            path = %ctx.path,
            request_id = %ctx.request_id,
            source_ip = %ctx.source_ip.resolve(),
            "rejected inbound JWT",
        );
    }
    err
}

/// `state` + the request context every JWT denial line carries, bundled so
/// the twelve `deny` sites stay one argument wide.
#[derive(Clone, Copy)]
struct Denier<'a> {
    state: &'a ProxyState,
    ctx: crate::auth::DenialContext<'a>,
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
fn candidate_keys(jwks: &JwkSet, kid: Option<&str>, alg: Algorithm) -> Vec<DecodingKey> {
    match kid {
        Some(kid) => jwks
            .find(kid)
            .filter(|jwk| usable_for_verification(jwk, alg))
            .and_then(|jwk| DecodingKey::from_jwk(jwk).ok())
            .into_iter()
            .collect(),
        None => jwks
            .keys
            .iter()
            .filter(|jwk| usable_for_verification(jwk, alg))
            .filter_map(|jwk| DecodingKey::from_jwk(jwk).ok())
            .take(MAX_KEYS_TRIED)
            .collect(),
    }
}

/// True when a JWK may verify a signature at `alg`: not an
/// encryption-only key (RFC 7517 §4.2 `use`), and — when the key names
/// an algorithm — that algorithm (RFC 7517 §4.4 `alg`). Applied on both
/// the `kid`-matched and the fall-through paths so a `use:enc` or
/// wrong-`alg` key is never tried, even when its `kid` is named.
fn usable_for_verification(jwk: &jsonwebtoken::jwk::Jwk, alg: Algorithm) -> bool {
    let use_ok = jwk
        .common
        .public_key_use
        .as_ref()
        .is_none_or(|u| matches!(u, jsonwebtoken::jwk::PublicKeyUse::Signature));
    // `KeyAlgorithm` (JWK `alg`) and `Algorithm` (token `alg`) are
    // distinct enums with no cross-conversion; their variant names are
    // identical (RS256 … EdDSA), so compare the Debug spellings.
    let alg_ok = jwk
        .common
        .key_algorithm
        .is_none_or(|k| format!("{k:?}") == format!("{alg:?}"));
    use_ok && alg_ok
}

// ── JWKS fetch + cache ───────────────────────────────────────────────

struct JwksEntry {
    /// The last successfully fetched key set and when it landed.
    jwks: Option<(Arc<JwkSet>, Instant)>,
    /// Last fetch attempt, success or failure — the rate-limit clock.
    last_attempt: Option<Instant>,
}

/// One issuer's resolved discovery result plus its rate-limit clock.
struct DiscoveryEntry {
    /// The resolved `jwks_uri` and when discovery last succeeded.
    resolved: Option<(String, Instant)>,
    /// Last discovery attempt, success or failure.
    last_attempt: Option<Instant>,
}

/// Read a poisoned-lock-tolerant guard. A panic while some other request
/// held the lock only ever happened during a map op (never across an
/// await), so the map is structurally intact; recovering the inner value
/// keeps JWT auth alive instead of poisoning it process-wide.
fn read_recover<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn write_recover<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

/// Process-global JWKS cache keyed by URL. Guards are held only for map
/// lookups/inserts, never across an await; concurrent misses may fetch in
/// parallel (each result is valid — last insert wins).
fn jwks_cache() -> &'static RwLock<HashMap<String, JwksEntry>> {
    static CACHE: OnceLock<RwLock<HashMap<String, JwksEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Discovery results keyed by issuer.
fn discovery_cache() -> &'static RwLock<HashMap<String, DiscoveryEntry>> {
    static CACHE: OnceLock<RwLock<HashMap<String, DiscoveryEntry>>> = OnceLock::new();
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
/// `jwks_uri` advertised by the issuer's OIDC discovery document.
///
/// Discovery is cached (TTL) and rate-limited on failure the same way
/// the JWKS fetch is, so an issuer whose discovery document is down does
/// not get re-probed once per request. The advertised `jwks_uri` is
/// verified against the issuer (OIDC Discovery 1.0 §4.3) and constrained
/// to the issuer's own origin, so a compromised or misconfigured
/// discovery document cannot relocate trust material into our network.
async fn resolve_jwks_url(prov: &OidcProvider) -> Result<String, String> {
    if let Some(u) = &prov.jwks_uri {
        if url_has_credentials(u) {
            return Err("jwks_uri must not embed credentials".to_string());
        }
        return Ok(u.clone());
    }
    let now = Instant::now();
    let (stale, attempted_recently) = {
        let map = read_recover(discovery_cache());
        match map.get(&prov.issuer) {
            Some(entry) => {
                if let Some((url, at)) = &entry.resolved {
                    if now.duration_since(*at) < JWKS_TTL {
                        return Ok(url.clone());
                    }
                }
                (
                    entry.resolved.as_ref().map(|(u, _)| u.clone()),
                    entry
                        .last_attempt
                        .is_some_and(|at| now.duration_since(at) < JWKS_REFRESH_MIN_INTERVAL),
                )
            }
            None => (None, false),
        }
    };
    if attempted_recently {
        // Suppressed by the refresh interval: serve the stale resolution
        // rather than probe a down issuer once per request.
        return stale
            .ok_or_else(|| "OIDC discovery suppressed by the refresh interval".to_string());
    }

    // Stamp the attempt before awaiting so concurrent misses don't stampede.
    write_recover(discovery_cache())
        .entry(prov.issuer.clone())
        .or_insert(DiscoveryEntry {
            resolved: None,
            last_attempt: None,
        })
        .last_attempt = Some(now);

    let discovery_url = format!(
        "{}/.well-known/openid-configuration",
        prov.issuer.trim_end_matches('/')
    );
    match fetch_json(&discovery_url).await {
        Ok(doc) => {
            // §4.3: the document must claim the issuer we asked about.
            if doc.get("issuer").and_then(|v| v.as_str()) != Some(prov.issuer.as_str()) {
                return Err(
                    "discovery document issuer does not match the configured issuer".to_string(),
                );
            }
            let jwks_uri = doc
                .get("jwks_uri")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "discovery document carries no jwks_uri".to_string())?
                .to_string();
            // The advertised endpoint decides where trust material comes
            // from, so it must stay on the issuer's own origin — otherwise
            // the discovery document is an open redirect into our network.
            if !same_origin(&prov.issuer, &jwks_uri) {
                return Err("discovery jwks_uri is not on the issuer's origin".to_string());
            }
            if url_has_credentials(&jwks_uri) {
                return Err("discovery jwks_uri must not embed credentials".to_string());
            }
            write_recover(discovery_cache())
                .entry(prov.issuer.clone())
                .or_insert(DiscoveryEntry {
                    resolved: None,
                    last_attempt: Some(now),
                })
                .resolved = Some((jwks_uri.clone(), now));
            Ok(jwks_uri)
        }
        Err(e) => {
            // Serve the stale resolution rather than failing auth outright.
            if let Some(url) = stale {
                tracing::warn!(
                    target: "aisix::auth",
                    issuer = %clip(&prov.issuer),
                    error = %e,
                    "OIDC discovery re-fetch failed; keeping the previously resolved JWKS URL",
                );
                return Ok(url);
            }
            Err(format!("OIDC discovery failed: {e}"))
        }
    }
}

/// True when a URL embeds credentials that must never sit in a public
/// JWKS endpoint: userinfo (`user:pass@host`) or a credential query
/// parameter. A defense-in-depth mirror of the control plane's ingestion
/// check, for file-mode and discovery-returned URLs.
fn url_has_credentials(url: &str) -> bool {
    match reqwest::Url::parse(url) {
        Ok(u) => {
            if !u.username().is_empty() || u.password().is_some() {
                return true;
            }
            u.query_pairs().any(|(k, _)| {
                matches!(
                    k.to_ascii_lowercase().as_str(),
                    "access_token" | "token" | "client_secret" | "password" | "api_key" | "apikey"
                )
            })
        }
        // Unparseable here is caught elsewhere (fetch fails); treat as
        // credential-free so this check doesn't double-report.
        Err(_) => false,
    }
}

/// True when `candidate` shares scheme + host + port with `base`.
fn same_origin(base: &str, candidate: &str) -> bool {
    match (reqwest::Url::parse(base), reqwest::Url::parse(candidate)) {
        (Ok(b), Ok(c)) => {
            b.scheme() == c.scheme()
                && b.host_str() == c.host_str()
                && b.port_or_known_default() == c.port_or_known_default()
        }
        _ => false,
    }
}

/// The cached key set for `url`, fetching when absent or past
/// [`JWKS_TTL`]. A fetch attempted within [`JWKS_REFRESH_MIN_INTERVAL`]
/// suppresses another one: one inbound JWT must never become one
/// outbound JWKS fetch, or a slow/down endpoint turns every request into
/// a [`JWKS_FETCH_TIMEOUT`] wait and floods the identity provider. A
/// failed re-fetch keeps serving the stale set; with nothing cached the
/// error propagates and the request fails closed as retryable.
async fn get_jwks(url: &str) -> Result<Arc<JwkSet>, String> {
    let now = Instant::now();
    let (stale, attempted_recently) = {
        let map = read_recover(jwks_cache());
        match map.get(url) {
            Some(entry) => {
                if let Some((jwks, fetched_at)) = &entry.jwks {
                    if now.duration_since(*fetched_at) < JWKS_TTL {
                        return Ok(jwks.clone());
                    }
                }
                (
                    entry.jwks.as_ref().map(|(j, _)| j.clone()),
                    entry
                        .last_attempt
                        .is_some_and(|at| now.duration_since(at) < JWKS_REFRESH_MIN_INTERVAL),
                )
            }
            None => (None, false),
        }
    };
    if attempted_recently {
        return stale.ok_or_else(|| "JWKS fetch suppressed by the refresh interval".to_string());
    }
    refresh_jwks(url).await
}

/// One fetch for an unknown `kid`, suppressed inside
/// [`JWKS_REFRESH_MIN_INTERVAL`] of the previous attempt.
async fn refresh_jwks_rate_limited(url: &str) -> Option<Arc<JwkSet>> {
    let now = Instant::now();
    if let Some(entry) = read_recover(jwks_cache()).get(url) {
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
    // hammered by concurrent refreshes.
    {
        let mut map = write_recover(jwks_cache());
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
            write_recover(jwks_cache())
                .entry(url.to_string())
                .and_modify(|e| e.jwks = Some((arc.clone(), Instant::now())))
                .or_insert(JwksEntry {
                    jwks: Some((arc.clone(), Instant::now())),
                    last_attempt: Some(Instant::now()),
                });
            Ok(arc)
        }
        Err(e) => {
            if let Some(entry) = read_recover(jwks_cache()).get(url) {
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
    let mut resp = http_client()
        .get(url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("endpoint returned status {}", resp.status()));
    }
    // Reject on the advertised length when present, then enforce the cap
    // while streaming — `bytes()` would buffer the whole body first, so a
    // hostile endpoint could OOM a small pod before any size check.
    if resp
        .content_length()
        .is_some_and(|n| n > JWKS_MAX_BYTES as u64)
    {
        return Err(format!("response exceeds {JWKS_MAX_BYTES} bytes"));
    }
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| format!("reading response failed: {e}"))?
    {
        if buf.len() + chunk.len() > JWKS_MAX_BYTES {
            return Err(format!("response exceeds {JWKS_MAX_BYTES} bytes"));
        }
        buf.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&buf).map_err(|e| format!("response is not JSON: {e}"))
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
        candidate_keys(&jwks, Some("test-kid-1"), Algorithm::RS256)
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
        assert_eq!(
            candidate_keys(&jwks, Some("test-kid-1"), Algorithm::RS256).len(),
            1
        );
        assert!(candidate_keys(&jwks, Some("rotated-away"), Algorithm::RS256).is_empty());
        // No kid on the token: every signature key is a candidate.
        assert_eq!(candidate_keys(&jwks, None, Algorithm::RS256).len(), 1);
        // The JWK declares alg RS256, so a token claiming a different alg
        // finds no usable key (RFC 7517 §4.4).
        assert!(candidate_keys(&jwks, Some("test-kid-1"), Algorithm::PS256).is_empty());
        assert!(candidate_keys(&jwks, None, Algorithm::ES256).is_empty());
    }

    #[test]
    fn provider_selection_is_unique_and_fails_closed_on_duplicate_issuer() {
        let snapshot = AisixSnapshot::new();
        let mk = |id: &str, issuer: &str, enabled: bool| {
            let mut p = base_provider();
            p.issuer = issuer.to_string();
            p.enabled = enabled;
            snapshot.oidc_providers.insert(ResourceEntry::new(id, p, 1));
        };
        mk("corp", "https://corp.test", true);
        mk("partner", "https://partner.test", true);
        mk("disabled", "https://off.test", false);
        assert_eq!(
            provider_for_issuer(&snapshot, "https://corp.test")
                .unwrap()
                .id,
            "corp"
        );
        // A disabled provider is not selected even on an exact issuer match.
        assert!(provider_for_issuer(&snapshot, "https://off.test").is_none());
        assert!(provider_for_issuer(&snapshot, "https://unknown.test").is_none());

        // Two ENABLED providers claiming one issuer -> fail closed, not a
        // silent pick (their audience/scope policies differ).
        mk("corp-dup", "https://corp.test", true);
        assert!(provider_for_issuer(&snapshot, "https://corp.test").is_none());
    }

    #[test]
    fn key_selection_namespaces_by_provider_and_fails_closed_on_duplicate() {
        let snapshot = AisixSnapshot::new();
        let mk_key = |id: &str, subject: Option<&str>, provider: Option<&str>| {
            let mut k: ApiKey =
                serde_json::from_str(r#"{"key_hash":"h","allowed_models":["*"]}"#).unwrap();
            k.jwt_subject = subject.map(str::to_string);
            k.jwt_provider = provider.map(str::to_string);
            // Distinct key_hash per row so the by-name index stays unique.
            k.key_hash = format!("hash-{id}");
            snapshot.apikeys.insert(ResourceEntry::new(id, k, 1));
        };
        mk_key("k-1", Some("agent-1"), Some("corp"));
        mk_key("k-3", Some("agent-2"), Some("corp"));
        mk_key("k-4", None, None);
        // Same subject under a DIFFERENT provider resolves separately —
        // the cross-provider impersonation guard (audit H1).
        mk_key("k-5", Some("agent-1"), Some("partner"));
        assert_eq!(
            key_for_subject(&snapshot, "corp", "agent-1").0.unwrap().id,
            "k-1"
        );
        assert_eq!(
            key_for_subject(&snapshot, "corp", "agent-2").0.unwrap().id,
            "k-3"
        );
        assert_eq!(
            key_for_subject(&snapshot, "partner", "agent-1")
                .0
                .unwrap()
                .id,
            "k-5"
        );
        // No provider match -> no key, even though the subject exists —
        // and no ambiguity signal either.
        assert!(matches!(
            key_for_subject(&snapshot, "unknown", "agent-1"),
            (None, false)
        ));
        assert!(matches!(
            key_for_subject(&snapshot, "corp", "agent-9"),
            (None, false)
        ));

        // A duplicate (provider, subject) pair -> fail closed, and the
        // ambiguity is REPORTED so the auth path can reject instead of
        // falling through to the claim mappings.
        mk_key("k-1-dup", Some("agent-1"), Some("corp"));
        assert!(matches!(
            key_for_subject(&snapshot, "corp", "agent-1"),
            (None, true)
        ));
    }

    #[test]
    fn same_origin_matches_scheme_host_port() {
        assert!(same_origin(
            "https://sso.example.com/realms/x",
            "https://sso.example.com/realms/x/certs"
        ));
        // default port equivalence
        assert!(same_origin(
            "https://sso.example.com",
            "https://sso.example.com:443/certs"
        ));
        // different host / scheme / port all rejected
        assert!(!same_origin(
            "https://sso.example.com",
            "https://evil.example.com/certs"
        ));
        assert!(!same_origin(
            "https://sso.example.com",
            "http://sso.example.com/certs"
        ));
        assert!(!same_origin(
            "https://sso.example.com",
            "https://sso.example.com:8443/certs"
        ));
    }

    #[test]
    fn oversized_token_is_not_a_jwt() {
        let big = format!("{}.{}.{}", "a".repeat(MAX_JWT_BYTES), "b", "c");
        assert!(big.len() > MAX_JWT_BYTES);
        assert!(!looks_like_jwt(&big));
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

    fn mapping(json: serde_json::Value) -> ClaimMapping {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn claim_match_ops_are_strictly_typed() {
        let claims = serde_json::json!({
            "department": "finance",
            "groups": ["dev", "mcp-admin", 42],
            "realm_access": {"roles": ["agent"]},
            "count": 7,
        });
        let m = |claim: &str, op: &str, values: serde_json::Value| -> ClaimMatch {
            serde_json::from_value(serde_json::json!({
                "claim": claim, "op": op, "values": values
            }))
            .unwrap()
        };

        // exact: string equality against any accepted value.
        assert!(claim_match_holds(
            &claims,
            &m("department", "exact", serde_json::json!(["hr", "finance"]))
        ));
        assert!(!claim_match_holds(
            &claims,
            &m("department", "exact", serde_json::json!(["hr"]))
        ));
        // exact never matches an array claim, even one containing the value.
        assert!(!claim_match_holds(
            &claims,
            &m("groups", "exact", serde_json::json!(["mcp-admin"]))
        ));

        // contains: array membership; non-string items are ignored.
        assert!(claim_match_holds(
            &claims,
            &m("groups", "contains", serde_json::json!(["mcp-admin"]))
        ));
        assert!(!claim_match_holds(
            &claims,
            &m("groups", "contains", serde_json::json!(["ops"]))
        ));
        // contains never matches a string claim.
        assert!(!claim_match_holds(
            &claims,
            &m("department", "contains", serde_json::json!(["finance"]))
        ));

        // Dots traverse nested objects, as everywhere else in JWT config.
        assert!(claim_match_holds(
            &claims,
            &m(
                "realm_access.roles",
                "contains",
                serde_json::json!(["agent"])
            )
        ));

        // Missing claims and non-string/array shapes never match.
        assert!(!claim_match_holds(
            &claims,
            &m("missing", "exact", serde_json::json!(["x"]))
        ));
        assert!(!claim_match_holds(
            &claims,
            &m("count", "exact", serde_json::json!(["7"]))
        ));
    }

    #[test]
    fn mapping_selection_is_priority_ordered_and_provider_scoped() {
        let snapshot = AisixSnapshot::new();
        let mk = |id: &str, m: serde_json::Value| {
            snapshot
                .claim_mappings
                .insert(ResourceEntry::new(id, mapping(m), 1));
        };
        // Both match `department=finance`; the lower priority value wins.
        mk(
            "cm-broad",
            serde_json::json!({
                "name": "broad", "jwt_provider": "corp", "priority": 200,
                "match": [{"claim": "department", "op": "exact", "values": ["finance"]}],
                "resolve": {"api_key_id": "k-broad"},
            }),
        );
        mk(
            "cm-narrow",
            serde_json::json!({
                "name": "narrow", "jwt_provider": "corp", "priority": 100,
                "match": [{"claim": "department", "op": "exact", "values": ["finance"]}],
                "resolve": {"api_key_id": "k-narrow"},
            }),
        );
        // Same priority as `narrow` but later in name order — the tie
        // break is deterministic, never insertion order.
        mk(
            "cm-tie",
            serde_json::json!({
                "name": "zz-tie", "jwt_provider": "corp", "priority": 100,
                "match": [{"claim": "department", "op": "exact", "values": ["finance"]}],
                "resolve": {"api_key_id": "k-tie"},
            }),
        );
        // Would win on priority, but is disabled.
        mk(
            "cm-off",
            serde_json::json!({
                "name": "off", "jwt_provider": "corp", "priority": 1, "enabled": false,
                "match": [{"claim": "department", "op": "exact", "values": ["finance"]}],
                "resolve": {"api_key_id": "k-off"},
            }),
        );
        // Would win on priority, but belongs to another provider.
        mk(
            "cm-partner",
            serde_json::json!({
                "name": "partner-rule", "jwt_provider": "partner", "priority": 1,
                "match": [{"claim": "department", "op": "exact", "values": ["finance"]}],
                "resolve": {"api_key_id": "k-partner"},
            }),
        );

        let claims = serde_json::json!({"department": "finance"});
        assert_eq!(
            matching_claim_mapping(&snapshot, "corp", &claims)
                .unwrap()
                .id,
            "cm-narrow"
        );
        assert_eq!(
            matching_claim_mapping(&snapshot, "partner", &claims)
                .unwrap()
                .id,
            "cm-partner"
        );
        // Every condition must hold: a rule with one unmet condition is
        // skipped even at the best priority.
        let missing = serde_json::json!({"department": "hr"});
        assert!(matching_claim_mapping(&snapshot, "corp", &missing).is_none());
    }

    #[test]
    fn mapping_conditions_are_conjunctive() {
        let snapshot = AisixSnapshot::new();
        snapshot.claim_mappings.insert(ResourceEntry::new(
            "cm-and",
            mapping(serde_json::json!({
                "name": "and-rule", "jwt_provider": "corp",
                "match": [
                    {"claim": "department", "op": "exact", "values": ["finance"]},
                    {"claim": "groups", "op": "contains", "values": ["mcp-admin"]},
                ],
                "resolve": {"api_key_id": "k-and"},
            })),
            1,
        ));
        let both = serde_json::json!({"department": "finance", "groups": ["mcp-admin"]});
        let one = serde_json::json!({"department": "finance", "groups": ["dev"]});
        assert!(matching_claim_mapping(&snapshot, "corp", &both).is_some());
        assert!(matching_claim_mapping(&snapshot, "corp", &one).is_none());
    }
}
