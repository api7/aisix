//! File-only sugar → canonical resource documents.
//!
//! The resources file accepts a small amount of sugar on top of the
//! canonical resource shapes (`schemas/resources/*.schema.json`):
//!
//! - `models[].provider_key` — a provider-key *name*, resolved to the
//!   derived `provider_key_id`. Mutually exclusive with an explicit
//!   `provider_key_id`.
//! - `api_keys[].display_name` — required file-side identity, stripped
//!   before validation (the canonical `api_key` document has no name
//!   field).
//! - `api_keys[].key_env` — the NAME of an environment variable whose
//!   value is the plaintext caller key; hashed to `key_hash` (SHA-256,
//!   lowercase hex — [`crate::models::ApiKey::hash_bearer`]) and then
//!   dropped. Exactly one of `key_env` / `key_hash` must be present.
//! - `rate_limit_policies[].scope_ref` — for `scope: api_key` /
//!   `scope: model`, a *name* resolved to the referenced entry's derived
//!   id; other scopes pass through verbatim.
//!
//! Everything this module emits must be exactly a canonical resource
//! document: the caller then runs the same JSON-Schema validators and
//! typed serde models the etcd path uses. Canonical schemas are not
//! relaxed for the file source.

use serde_json::Value;
use std::collections::BTreeMap;
use uuid::Uuid;

use super::yaml::EnvLookup;

/// Fixed UUIDv5 namespace for deterministic file-resource ids.
///
/// Derived once as `uuid5(NAMESPACE_URL,
/// "https://github.com/api7/aisix#resources-file")` and pinned here so
/// the value can never drift; a unit test re-derives it. Every resource
/// loaded from the file gets `uuid5(FILE_RESOURCE_NAMESPACE,
/// "<kind>/<identity>")` — stable across reloads and across processes,
/// so references and rate-limit counters keyed by id survive a SIGHUP.
pub const FILE_RESOURCE_NAMESPACE: Uuid = Uuid::from_bytes([
    0x63, 0xe5, 0x0a, 0xb2, 0x67, 0x7a, 0x54, 0xd3, 0x8d, 0x1e, 0xc0, 0xcb, 0x29, 0xce, 0xae, 0x94,
]);

/// Deterministic id for a file-defined resource: UUIDv5 over
/// `"<kind>/<identity>"` in [`FILE_RESOURCE_NAMESPACE`].
pub fn derive_id(kind: &str, identity: &str) -> String {
    Uuid::new_v5(
        &FILE_RESOURCE_NAMESPACE,
        format!("{kind}/{identity}").as_bytes(),
    )
    .to_string()
}

/// How a kind names its entries in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IdentityField {
    /// `display_name` (provider_keys, models, api_keys).
    DisplayName,
    /// `name` (guardrails, cache_policies, observability_exporters,
    /// rate_limit_policies).
    Name,
    /// `name`, with `display_name` accepted as the alternative spelling
    /// (mcp_servers, a2a_agents — mirroring their schemas).
    NameOrDisplayName,
}

impl IdentityField {
    /// Human description for "missing identity" errors.
    pub(crate) fn describe(self) -> &'static str {
        match self {
            Self::DisplayName => "display_name",
            Self::Name => "name",
            Self::NameOrDisplayName => "name (or display_name)",
        }
    }

    /// Extract the identity string from a JSON entry, if present and a
    /// non-empty string.
    pub(crate) fn extract(self, doc: &Value) -> Option<String> {
        let field = |key: &str| {
            doc.get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        match self {
            Self::DisplayName => field("display_name"),
            Self::Name => field("name"),
            Self::NameOrDisplayName => field("name").or_else(|| field("display_name")),
        }
    }
}

/// Per-kind identity → derived-id map, built in the first pass so later
/// sugar (name references) can resolve against every defined entry.
pub(crate) type IdentityMaps = BTreeMap<&'static str, BTreeMap<String, String>>;

fn known_names(maps: &IdentityMaps, kind: &str) -> String {
    let names: Vec<&str> = maps
        .get(kind)
        .map(|m| m.keys().map(String::as_str).collect())
        .unwrap_or_default();
    if names.is_empty() {
        format!("no {kind} are defined in this file")
    } else {
        format!("defined {kind}: {}", names.join(", "))
    }
}

/// `models[].provider_key` (name) → `provider_key_id` (derived id).
pub(crate) fn desugar_model(doc: &mut Value, maps: &IdentityMaps) -> Result<(), String> {
    let obj = match doc.as_object_mut() {
        Some(o) => o,
        None => return Ok(()), // non-object entries error upstream
    };
    let Some(name_value) = obj.get("provider_key") else {
        return Ok(());
    };
    let Some(name) = name_value.as_str() else {
        return Err("`provider_key` must be a string (a provider key display_name)".into());
    };
    if obj.contains_key("provider_key_id") {
        return Err(
            "`provider_key` (a name reference) and `provider_key_id` are mutually \
             exclusive — set exactly one"
                .into(),
        );
    }
    let resolved = maps
        .get("provider_keys")
        .and_then(|m| m.get(name))
        .cloned()
        .ok_or_else(|| {
            format!(
                "`provider_key` references unknown provider key {name:?} ({})",
                known_names(maps, "provider_keys")
            )
        })?;
    obj.remove("provider_key");
    obj.insert("provider_key_id".into(), Value::String(resolved));
    Ok(())
}

/// `api_keys[]`: strip the identity-only `display_name`, resolve
/// `key_env` XOR `key_hash`. The plaintext read from the environment is
/// hashed and dropped — it must never surface in errors, logs, or the
/// emitted document, so every error here names the *variable*, not its
/// value.
pub(crate) fn desugar_api_key(doc: &mut Value, env: EnvLookup<'_>) -> Result<(), String> {
    let obj = match doc.as_object_mut() {
        Some(o) => o,
        None => return Ok(()),
    };
    // Identity extraction already require display_name to be present;
    // here it only needs stripping (the canonical api_key document has
    // no name field and its schema would reject one).
    obj.remove("display_name");

    let has_key_env = obj.contains_key("key_env");
    let has_key_hash = obj.contains_key("key_hash");
    match (has_key_env, has_key_hash) {
        (true, true) => {
            return Err("`key_env` and `key_hash` are mutually exclusive — set exactly one".into())
        }
        (false, false) => {
            return Err(
                "one of `key_env` (environment variable holding the plaintext key) or \
                 `key_hash` (SHA-256 hex of the plaintext) is required"
                    .into(),
            )
        }
        (true, false) => {
            let var = match obj.get("key_env").and_then(Value::as_str) {
                Some(v) if !v.is_empty() => v.to_string(),
                _ => {
                    return Err(
                        "`key_env` must be a non-empty string (an environment variable name)"
                            .into(),
                    )
                }
            };
            let plaintext = match env(&var) {
                Some(v) if !v.is_empty() => v,
                _ => {
                    return Err(format!(
                        "`key_env` environment variable `{var}` is unset or empty"
                    ))
                }
            };
            if plaintext.contains("${") {
                // Double indirection: the variable holds an uninterpolated
                // `${...}` reference instead of the actual key.
                return Err(format!(
                    "`key_env` environment variable `{var}` contains an uninterpolated \
                     `${{...}}` reference — set it to the plaintext key itself"
                ));
            }
            let hash = crate::models::ApiKey::hash_bearer(&plaintext);
            obj.remove("key_env");
            obj.insert("key_hash".into(), Value::String(hash));
        }
        (false, true) => {}
    }
    Ok(())
}

/// `rate_limit_policies[].scope_ref`: for `scope: api_key` / `scope:
/// model`, resolve the name against the file-defined entries and replace
/// it with the derived id (the proxy matches policies by entry id).
/// Other scopes (`team`, `member`, `team_member`, or anything the schema
/// will reject later) pass through verbatim.
pub(crate) fn desugar_rate_limit_policy(
    doc: &mut Value,
    maps: &IdentityMaps,
) -> Result<(), String> {
    let obj = match doc.as_object_mut() {
        Some(o) => o,
        None => return Ok(()),
    };
    let scope_kind = match obj.get("scope").and_then(Value::as_str) {
        Some("api_key") => "api_keys",
        Some("model") => "models",
        _ => return Ok(()),
    };
    let Some(name) = obj.get("scope_ref").and_then(Value::as_str) else {
        // Missing / non-string scope_ref: canonical validation reports it.
        return Ok(());
    };
    let resolved = maps
        .get(scope_kind)
        .and_then(|m| m.get(name))
        .cloned()
        .ok_or_else(|| {
            format!(
                "`scope_ref` references unknown {} {name:?} ({})",
                if scope_kind == "api_keys" {
                    "api key"
                } else {
                    "model"
                },
                known_names(maps, scope_kind)
            )
        })?;
    obj.insert("scope_ref".into(), Value::String(resolved));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn maps_with(kind: &'static str, names: &[&str]) -> IdentityMaps {
        let mut maps = IdentityMaps::new();
        maps.insert(
            kind,
            names
                .iter()
                .map(|n| (n.to_string(), derive_id(kind, n)))
                .collect(),
        );
        maps
    }

    #[test]
    fn namespace_constant_matches_its_documented_derivation() {
        assert_eq!(
            FILE_RESOURCE_NAMESPACE,
            Uuid::new_v5(
                &Uuid::NAMESPACE_URL,
                b"https://github.com/api7/aisix#resources-file"
            ),
        );
    }

    #[test]
    fn derive_id_is_deterministic_and_kind_scoped() {
        let a1 = derive_id("models", "gpt-4o");
        let a2 = derive_id("models", "gpt-4o");
        assert_eq!(a1, a2, "same input must derive the same id");
        // Pin one concrete value so any namespace / input-shape change
        // shows up as a test diff, not a silent id migration.
        assert_eq!(a1, "6efd592d-a73b-573d-8e2c-e9e103692f92");
        // Kind participates in the input, so identical names in
        // different kinds never collide.
        assert_ne!(derive_id("models", "x"), derive_id("api_keys", "x"));
        // Valid UUID text form.
        assert!(Uuid::parse_str(&a1).is_ok());
    }

    #[test]
    fn model_provider_key_name_resolves_to_derived_id() {
        let maps = maps_with("provider_keys", &["openai-prod"]);
        let mut doc = json!({
            "display_name": "gpt-4o",
            "provider": "openai",
            "model_name": "gpt-4o-2024-11-20",
            "provider_key": "openai-prod"
        });
        desugar_model(&mut doc, &maps).unwrap();
        assert!(doc.get("provider_key").is_none());
        assert_eq!(
            doc["provider_key_id"],
            json!(derive_id("provider_keys", "openai-prod"))
        );
    }

    #[test]
    fn model_provider_key_unknown_name_lists_defined_keys() {
        let maps = maps_with("provider_keys", &["a-key", "b-key"]);
        let mut doc = json!({"display_name": "m", "provider_key": "nope"});
        let err = desugar_model(&mut doc, &maps).unwrap_err();
        assert!(err.contains("\"nope\""), "unexpected: {err}");
        assert!(err.contains("a-key"), "unexpected: {err}");
        assert!(err.contains("b-key"), "unexpected: {err}");
    }

    #[test]
    fn model_provider_key_conflicts_with_provider_key_id() {
        let maps = maps_with("provider_keys", &["openai-prod"]);
        let mut doc = json!({
            "display_name": "m",
            "provider_key": "openai-prod",
            "provider_key_id": "11111111-1111-1111-1111-111111111111"
        });
        let err = desugar_model(&mut doc, &maps).unwrap_err();
        assert!(err.contains("mutually exclusive"), "unexpected: {err}");
    }

    #[test]
    fn api_key_env_hashes_and_strips_sugar_fields() {
        let env = |name: &str| (name == "CALLER_KEY").then(|| "sk-plain".to_string());
        let mut doc = json!({
            "display_name": "ci-bot",
            "key_env": "CALLER_KEY",
            "allowed_models": ["*"]
        });
        desugar_api_key(&mut doc, &env).unwrap();
        assert!(
            doc.get("display_name").is_none(),
            "identity must be stripped"
        );
        assert!(doc.get("key_env").is_none(), "sugar must be stripped");
        // SHA-256 lowercase hex, identical to ApiKey::hash_bearer.
        assert_eq!(
            doc["key_hash"],
            json!(crate::models::ApiKey::hash_bearer("sk-plain"))
        );
        let hash = doc["key_hash"].as_str().unwrap();
        assert_eq!(hash.len(), 64);
        assert!(hash
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        // The plaintext itself must not appear anywhere in the document.
        assert!(!doc.to_string().contains("sk-plain"));
    }

    #[test]
    fn api_key_env_xor_key_hash_is_enforced() {
        let env = |_: &str| Some("sk-x".to_string());
        let mut both =
            json!({"display_name": "k", "key_env": "V", "key_hash": "ab", "allowed_models": []});
        assert!(desugar_api_key(&mut both, &env)
            .unwrap_err()
            .contains("mutually exclusive"));
        let mut neither = json!({"display_name": "k", "allowed_models": []});
        assert!(desugar_api_key(&mut neither, &env)
            .unwrap_err()
            .contains("required"));
        // key_hash alone passes through untouched.
        let mut hash_only = json!({"display_name": "k", "key_hash": "cafe", "allowed_models": []});
        desugar_api_key(&mut hash_only, &env).unwrap();
        assert_eq!(hash_only["key_hash"], json!("cafe"));
    }

    #[test]
    fn api_key_env_missing_empty_or_indirect_value_errors_without_leaking() {
        let mut doc = json!({"display_name": "k", "key_env": "MISSING", "allowed_models": []});
        let err = desugar_api_key(&mut doc, &|_| None).unwrap_err();
        assert!(err.contains("`MISSING`"), "unexpected: {err}");
        assert!(err.contains("unset or empty"), "unexpected: {err}");

        let mut doc = json!({"display_name": "k", "key_env": "EMPTY", "allowed_models": []});
        let err = desugar_api_key(&mut doc, &|_| Some(String::new())).unwrap_err();
        assert!(err.contains("unset or empty"), "unexpected: {err}");

        // The env value looks like an uninterpolated reference — double
        // indirection is rejected, and the value itself never appears in
        // the error message.
        let mut doc = json!({"display_name": "k", "key_env": "REF", "allowed_models": []});
        let err =
            desugar_api_key(&mut doc, &|_| Some("${REAL_SECRET_KEY}".to_string())).unwrap_err();
        assert!(err.contains("uninterpolated"), "unexpected: {err}");
        assert!(
            !err.contains("REAL_SECRET_KEY"),
            "the env value must never surface in errors: {err}"
        );
    }

    #[test]
    fn scope_ref_resolves_per_scope_and_passes_team_scopes_verbatim() {
        let mut maps = maps_with("models", &["gpt-4o"]);
        maps.insert(
            "api_keys",
            [("ci-bot".to_string(), derive_id("api_keys", "ci-bot"))]
                .into_iter()
                .collect(),
        );

        let mut model_scope = json!({"name": "p", "scope": "model", "scope_ref": "gpt-4o"});
        desugar_rate_limit_policy(&mut model_scope, &maps).unwrap();
        assert_eq!(
            model_scope["scope_ref"],
            json!(derive_id("models", "gpt-4o"))
        );

        let mut key_scope = json!({"name": "p", "scope": "api_key", "scope_ref": "ci-bot"});
        desugar_rate_limit_policy(&mut key_scope, &maps).unwrap();
        assert_eq!(
            key_scope["scope_ref"],
            json!(derive_id("api_keys", "ci-bot"))
        );

        // team / member / team_member pass through verbatim.
        for scope in ["team", "member", "team_member"] {
            let mut doc = json!({"name": "p", "scope": scope, "scope_ref": "team-uuid-9"});
            desugar_rate_limit_policy(&mut doc, &maps).unwrap();
            assert_eq!(doc["scope_ref"], json!("team-uuid-9"));
        }
    }

    #[test]
    fn scope_ref_unknown_name_is_an_error_listing_candidates() {
        let maps = maps_with("models", &["gpt-4o"]);
        let mut doc = json!({"name": "p", "scope": "model", "scope_ref": "missing-model"});
        let err = desugar_rate_limit_policy(&mut doc, &maps).unwrap_err();
        assert!(err.contains("\"missing-model\""), "unexpected: {err}");
        assert!(err.contains("gpt-4o"), "unexpected: {err}");
    }
}
