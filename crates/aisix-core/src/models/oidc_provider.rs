//! `OidcProvider` entity — an external identity provider the gateway
//! trusts for inbound JWT authentication, stored in etcd under
//! `oidc_providers/<uuid>`.
//!
//! When at least one enabled provider exists, a request whose bearer
//! token is a JWT (instead of a gateway API key) is authenticated
//! against the provider matching the token's `iss` claim: the token's
//! signature is verified against the provider's JWKS, its registered
//! claims (`exp`, `aud`) and the provider's scope/claim requirements
//! are enforced, and the value of `identity_claim` selects the API key
//! whose `jwt_subject` equals it. The request then proceeds with that
//! key's permissions, rate limits, and budget — external identities
//! never widen access beyond a key an operator explicitly created.
//!
//! A provider verifies in one of two **modes**, derived from the row
//! rather than declared by it:
//!
//! - **JWKS mode** (`hmac_secret` absent) — asymmetric signatures
//!   verified against keys fetched from `jwks_uri` or OIDC discovery.
//!   `issuer` and `audiences` are both mandatory.
//! - **HMAC mode** (`hmac_secret` present) — HS256/HS384/HS512
//!   signatures verified against the shared secret, with no key fetch
//!   and no discovery. `issuer` and `audiences` become optional, and
//!   `jwks_uri` must be absent.
//!
//! The requirements the JSON Schema cannot express — the mode's own
//! field coupling and the secret's length floor — are enforced by
//! [`OidcProvider::validate_semantics`], so a row that breaks them is
//! rejected at load exactly like a schema failure.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// Minimum length of an HMAC shared secret, in bytes of its UTF-8
/// encoding. RFC 7518 §3.2 requires a key at least as long as the
/// digest of the weakest accepted algorithm (HS256 → 256 bits), and a
/// shorter one weakens every token the provider ever verifies.
pub const HMAC_SECRET_MIN_BYTES: usize = 32;

/// An HMAC shared secret. A plain string on the wire — the stored
/// document, the resources file and the export all carry it as one —
/// with a [`Debug`] that reveals nothing, so a provider row printed in a
/// log, an error or a diagnostic cannot leak the key that authenticates
/// every one of its callers.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HmacSecret(String);

impl HmacSecret {
    /// The secret's raw bytes — the HMAC key exactly as configured, with
    /// no base64 decoding and no derivation.
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    /// Length of the key material in bytes.
    pub fn len_bytes(&self) -> usize {
        self.0.len()
    }
}

impl From<String> for HmacSecret {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl std::fmt::Debug for HmacSecret {
    /// Never renders the secret. `Debug` is what `#[derive(Debug)]` on
    /// the enclosing resource, `tracing`'s `?` sigil and most error
    /// formatting reach for, so redacting here covers every one of them
    /// at once.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HmacSecret(<redacted>)")
    }
}

impl schemars::JsonSchema for HmacSecret {
    fn schema_name() -> String {
        "HmacSecret".to_string()
    }

    /// Inlined rather than referenced: the field is a plain string on
    /// the contract, and a `$ref` would hide that behind a definition
    /// and swallow the field's own description.
    fn is_referenceable() -> bool {
        false
    }

    fn json_schema(generator: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        // A plain non-empty string on the contract. The byte floor is
        // NOT expressed here on purpose: JSON Schema's `minLength`
        // counts characters, and the floor is a count of UTF-8 bytes —
        // the two disagree for any non-ASCII secret, in both directions.
        // `validate_semantics` owns it instead.
        let mut schema = <String as schemars::JsonSchema>::json_schema(generator).into_object();
        schema.string().min_length = Some(1);
        schema.into()
    }
}

/// Expected value(s) for one bound claim: a single string, or a list
/// matched as any-of.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum BoundClaimExpect {
    /// The claim must equal (or, for array claims, contain) this value.
    One(String),
    /// The claim must equal (or, for array claims, contain) at least one
    /// of these values.
    #[schemars(length(min = 1))]
    Any(Vec<String>),
}

impl BoundClaimExpect {
    /// Iterate the accepted values regardless of form.
    pub fn accepted(&self) -> impl Iterator<Item = &str> {
        match self {
            BoundClaimExpect::One(v) => std::slice::from_ref(v).iter().map(String::as_str),
            BoundClaimExpect::Any(vs) => vs.as_slice().iter().map(String::as_str),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct OidcProvider {
    /// Human-readable provider name, unique within the environment
    /// (e.g. `"corp-keycloak"`).
    #[schemars(length(min = 1))]
    pub name: String,

    /// Expected `iss` claim, compared byte-for-byte against the token's
    /// issuer. A JWT carrying this issuer is verified against this
    /// provider and no other. Required unless `hmac_secret` is set;
    /// when a shared-secret provider omits it, the token's `iss` claim
    /// is not consulted at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub issuer: Option<String>,

    /// Accepted `aud` values. The token's audience (a string or an
    /// array) must contain at least one of these. Required unless
    /// `hmac_secret` is set; when a shared-secret provider omits it,
    /// the `aud` claim is ignored.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(min = 1))]
    pub audiences: Vec<String>,

    /// JWKS endpoint URL the signing keys are fetched from. When
    /// omitted, the endpoint is resolved once from the issuer's OIDC
    /// discovery document (`<issuer>/.well-known/openid-configuration`)
    /// and cached. Must be absent when `hmac_secret` is set — a
    /// shared-secret provider fetches no keys.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub jwks_uri: Option<String>,

    /// Shared secret for HMAC-signed tokens, whose presence selects
    /// HMAC verification for this provider. Its UTF-8 bytes are the
    /// HMAC key as-is — nothing is base64-decoded or derived — and it
    /// must be at least 32 bytes long. Tokens are then accepted only
    /// under `HS256`, `HS384` or `HS512`, no signing keys are fetched,
    /// and `issuer` / `audiences` become optional. Omit it for a
    /// standard OIDC provider whose tokens are verified against a JWKS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hmac_secret: Option<HmacSecret>,

    /// Claim whose value selects the API key to act as: the request is
    /// bound to the key whose `jwt_subject` equals this claim's value.
    /// Dots traverse nested objects (e.g. `"resource_access.account"`).
    /// Defaults to `sub`.
    #[serde(default = "default_identity_claim")]
    #[schemars(length(min = 1))]
    pub identity_claim: String,

    /// Scopes that must all be present in the token's `scope` claim
    /// (a space-delimited string or an array of strings). An empty list
    /// requires nothing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_scopes: Vec<String>,

    /// Additional claim requirements, all of which must hold. Keys name
    /// claims (dots traverse nested objects, e.g.
    /// `"realm_access.roles"`); each requirement is satisfied when the
    /// claim equals — or, for array claims, contains — one of the
    /// expected values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_claims: Option<BTreeMap<String, BoundClaimExpect>>,

    /// Clock-skew allowance in seconds applied to time-based claims
    /// (`exp`, `nbf`). Defaults to 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    #[schemars(range(max = 300))]
    pub leeway_secs: u64,

    /// Whether the provider participates in JWT authentication. A
    /// disabled provider is kept but ignored. Treated as `true` when
    /// omitted.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// etcd-key uuid. Filled by the loader and never included in the
    /// JSON payload.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl OidcProvider {
    /// The shared secret when this provider verifies in HMAC mode.
    /// Mode is derived from the row: a secret means HMAC, its absence
    /// means JWKS. There is no separate mode field to disagree with.
    pub fn hmac_secret(&self) -> Option<&HmacSecret> {
        self.hmac_secret.as_ref()
    }

    /// True when this provider verifies asymmetric signatures against a
    /// fetched key set — the only mode with an authorization server to
    /// advertise.
    pub fn is_jwks_mode(&self) -> bool {
        self.hmac_secret.is_none()
    }

    /// Requirements the JSON Schema cannot express, run by every load
    /// path after deserialization. A row that fails is rejected exactly
    /// like a schema failure, because each failure leaves the provider
    /// unable to verify anything: the returned message names the field
    /// and the rule, and never the secret.
    pub fn validate_semantics(&self) -> Result<(), String> {
        match &self.hmac_secret {
            Some(secret) => {
                if self.jwks_uri.is_some() {
                    return Err(
                        "`jwks_uri` must be absent when `hmac_secret` is set — a shared-secret \
                         provider verifies tokens against the secret and fetches no signing keys"
                            .to_string(),
                    );
                }
                if secret.len_bytes() < HMAC_SECRET_MIN_BYTES {
                    return Err(format!(
                        "`hmac_secret` must be at least {HMAC_SECRET_MIN_BYTES} bytes long \
                         (got {})",
                        secret.len_bytes()
                    ));
                }
            }
            None => {
                if self.issuer.is_none() {
                    return Err(
                        "`issuer` is required unless `hmac_secret` is set — it is what selects \
                         this provider for an inbound token"
                            .to_string(),
                    );
                }
                if self.audiences.is_empty() {
                    return Err(
                        "`audiences` is required unless `hmac_secret` is set — every token \
                         verified against a JWKS must name an audience this gateway accepts"
                            .to_string(),
                    );
                }
            }
        }
        Ok(())
    }
}

fn default_identity_claim() -> String {
    "sub".to_string()
}

fn default_enabled() -> bool {
    true
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

impl Resource for OidcProvider {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// The by-name index key is the display name. Issuer lookups during
    /// authentication iterate and filter on `issuer` rather than relying
    /// on this index, so a duplicate name can never shadow a provider.
    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "oidc_providers"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_minimal_provider_with_defaults() {
        let p: OidcProvider = serde_json::from_str(
            r#"{
              "name": "corp-keycloak",
              "issuer": "https://sso.example.com/realms/agents",
              "audiences": ["aisix-gateway"]
            }"#,
        )
        .unwrap();
        assert_eq!(p.name, "corp-keycloak");
        assert_eq!(
            p.issuer.as_deref(),
            Some("https://sso.example.com/realms/agents")
        );
        assert_eq!(p.audiences, vec!["aisix-gateway"]);
        assert!(p.jwks_uri.is_none());
        assert_eq!(p.identity_claim, "sub");
        assert!(p.required_scopes.is_empty());
        assert!(p.bound_claims.is_none());
        assert_eq!(p.leeway_secs, 0);
        assert!(p.enabled);
    }

    #[test]
    fn deserialises_full_provider() {
        let p: OidcProvider = serde_json::from_str(
            r#"{
              "name": "corp-keycloak",
              "issuer": "https://sso.example.com/realms/agents",
              "audiences": ["aisix-gateway", "aisix-alt"],
              "jwks_uri": "https://sso.example.com/realms/agents/protocol/openid-connect/certs",
              "identity_claim": "azp",
              "required_scopes": ["ai.access"],
              "bound_claims": {
                "department": "ai-lab",
                "realm_access.roles": ["agent", "batch-agent"]
              },
              "leeway_secs": 30,
              "enabled": false
            }"#,
        )
        .unwrap();
        assert_eq!(p.audiences.len(), 2);
        assert_eq!(p.identity_claim, "azp");
        assert_eq!(p.required_scopes, vec!["ai.access"]);
        let bound = p.bound_claims.as_ref().unwrap();
        assert_eq!(
            bound.get("department"),
            Some(&BoundClaimExpect::One("ai-lab".into()))
        );
        assert_eq!(
            bound.get("realm_access.roles"),
            Some(&BoundClaimExpect::Any(vec![
                "agent".into(),
                "batch-agent".into()
            ]))
        );
        assert_eq!(p.leeway_secs, 30);
        assert!(!p.enabled);
    }

    #[test]
    fn tolerates_unknown_fields_for_forward_compat() {
        // A newer control plane may ship fields ahead of this DP; serde
        // must accept them. The write path still rejects them via the
        // strict schema validator (validate_oidc_provider in models/schema.rs).
        let p: OidcProvider = serde_json::from_str(
            r#"{"name":"x","issuer":"https://x","audiences":["a"],"extra":1}"#,
        )
        .unwrap();
        assert_eq!(p.name, "x");
    }

    #[test]
    fn defaults_stay_off_the_wire() {
        let p: OidcProvider =
            serde_json::from_str(r#"{"name":"x","issuer":"https://x","audiences":["a"]}"#).unwrap();
        let v = serde_json::to_value(&p).unwrap();
        assert!(v.get("jwks_uri").is_none());
        assert!(v.get("required_scopes").is_none());
        assert!(v.get("bound_claims").is_none());
        assert!(v.get("leeway_secs").is_none());
        // identity_claim and enabled serialize with their default values —
        // both are meaningful to echo back through the Admin API.
        assert_eq!(v["identity_claim"], "sub");
        assert_eq!(v["enabled"], true);
    }

    #[test]
    fn bound_claim_expect_accepted_iterates_both_forms() {
        let one = BoundClaimExpect::One("a".into());
        assert_eq!(one.accepted().collect::<Vec<_>>(), vec!["a"]);
        let any = BoundClaimExpect::Any(vec!["a".into(), "b".into()]);
        assert_eq!(any.accepted().collect::<Vec<_>>(), vec!["a", "b"]);
    }

    #[test]
    fn resource_trait_points_at_name_and_kind() {
        assert_eq!(OidcProvider::kind(), "oidc_providers");
        let mut p: OidcProvider =
            serde_json::from_str(r#"{"name":"corp","issuer":"https://x","audiences":["a"]}"#)
                .unwrap();
        p.runtime_id = "op-1".into();
        assert_eq!(p.id(), "op-1");
        assert_eq!(p.name(), "corp");
    }

    // ── HMAC (shared-secret) mode ────────────────────────────────────

    const LONG_ENOUGH: &str = "shared-secret-that-is-long-enough-32";

    fn parse(doc: serde_json::Value) -> OidcProvider {
        serde_json::from_value(doc).unwrap()
    }

    #[test]
    fn mode_is_derived_from_the_secret_alone() {
        let jwks = parse(serde_json::json!({
            "name": "corp", "issuer": "https://x", "audiences": ["a"],
        }));
        assert!(jwks.is_jwks_mode());
        assert!(jwks.hmac_secret().is_none());

        let hmac = parse(serde_json::json!({"name": "shared", "hmac_secret": LONG_ENOUGH}));
        assert!(!hmac.is_jwks_mode());
        assert_eq!(
            hmac.hmac_secret().unwrap().as_bytes(),
            LONG_ENOUGH.as_bytes()
        );
    }

    #[test]
    fn an_hmac_provider_needs_neither_issuer_nor_audiences() {
        let p = parse(serde_json::json!({"name": "shared", "hmac_secret": LONG_ENOUGH}));
        assert!(p.issuer.is_none());
        assert!(p.audiences.is_empty());
        p.validate_semantics().unwrap();
    }

    #[test]
    fn a_jwks_provider_still_requires_issuer_and_audiences() {
        let no_issuer = parse(serde_json::json!({"name": "corp", "audiences": ["a"]}));
        assert!(no_issuer
            .validate_semantics()
            .unwrap_err()
            .contains("`issuer` is required"));

        let no_audiences = parse(serde_json::json!({"name": "corp", "issuer": "https://x"}));
        assert!(no_audiences
            .validate_semantics()
            .unwrap_err()
            .contains("`audiences` is required"));
    }

    #[test]
    fn an_hmac_provider_may_not_also_name_a_jwks_endpoint() {
        let p = parse(serde_json::json!({
            "name": "shared", "hmac_secret": LONG_ENOUGH, "jwks_uri": "https://x/jwks",
        }));
        assert!(p
            .validate_semantics()
            .unwrap_err()
            .contains("`jwks_uri` must be absent"));
    }

    #[test]
    fn the_secret_floor_counts_bytes_not_characters() {
        // One byte short of the floor.
        let short = "x".repeat(HMAC_SECRET_MIN_BYTES - 1);
        let p = parse(serde_json::json!({"name": "shared", "hmac_secret": short}));
        let err = p.validate_semantics().unwrap_err();
        assert!(err.contains("at least 32 bytes"), "{err}");
        assert!(
            !err.contains(&short),
            "the message must not echo the secret: {err}"
        );

        // Exactly at the floor passes …
        let exact = "x".repeat(HMAC_SECRET_MIN_BYTES);
        parse(serde_json::json!({"name": "shared", "hmac_secret": exact}))
            .validate_semantics()
            .unwrap();

        // … and 31 multi-byte characters clear it while a character
        // count would not, which is why the floor is not a `minLength`.
        let multibyte = "é".repeat(HMAC_SECRET_MIN_BYTES - 1);
        assert_eq!(multibyte.chars().count(), HMAC_SECRET_MIN_BYTES - 1);
        parse(serde_json::json!({"name": "shared", "hmac_secret": multibyte}))
            .validate_semantics()
            .unwrap();
    }

    #[test]
    fn the_secret_round_trips_as_a_plain_string_and_never_debug_prints() {
        let p = parse(serde_json::json!({"name": "shared", "hmac_secret": LONG_ENOUGH}));
        // On the wire it is the string the operator wrote — the stored
        // document, the resources file and the export all read it back.
        assert_eq!(
            serde_json::to_value(&p).unwrap()["hmac_secret"],
            LONG_ENOUGH
        );
        // In a log line, an error, or any `{:?}`, it is not.
        let rendered = format!("{p:?}");
        assert!(!rendered.contains(LONG_ENOUGH), "{rendered}");
        assert!(rendered.contains("<redacted>"), "{rendered}");
        assert!(!format!("{:?}", p.hmac_secret().unwrap()).contains(LONG_ENOUGH));
    }

    #[test]
    fn a_jwks_provider_serializes_no_hmac_secret_key() {
        let p = parse(serde_json::json!({
            "name": "corp", "issuer": "https://x", "audiences": ["a"],
        }));
        assert!(serde_json::to_value(&p)
            .unwrap()
            .get("hmac_secret")
            .is_none());
    }
}
