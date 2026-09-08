//! `ApiKey` entity — the caller-facing credential presented in
//! `Authorization: Bearer <plaintext>` (spec §3, §7).
//!
//! Self-hosted CP (prd-09a §9A.7B.4): the KV payload stores
//! **`key_hash`** (SHA-256 hex of the plaintext bearer) instead of
//! the plaintext. cp-api stores only the hash and shows the
//! plaintext to the user exactly once at create time. The DP proxy
//! hashes incoming bearer tokens (`aisix-proxy/src/auth.rs`) and
//! looks up by the hash. Net security win: no plaintext API key
//! ever sits in the DB or KV.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::mcp_policy::McpAccess;
use super::rate_limit::{McpRateLimit, RateLimit};
use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ApiKey {
    /// SHA-256 hexadecimal hash of the plaintext bearer. The proxy hashes
    /// incoming bearer tokens before lookup.
    #[schemars(length(min = 1))]
    pub key_hash: String,

    /// Operator-facing label for this key, as shown in the dashboard.
    /// Read only by the `${request.api_key.name}` header template
    /// (AISIX-Cloud#1112); never used for authentication or routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,

    /// Model names this key may use, matched as single-`*` globs. Read only
    /// when `allowed_model_ids` is absent; ignored entirely when it is
    /// present. When both are omitted the key may use no model — model
    /// access is granted explicitly.
    #[serde(default)]
    pub allowed_models: Vec<String>,

    /// Models this key may use, named by resource id rather than by name, so
    /// renaming a model does not change what this key may reach. Present —
    /// including as an empty array — it is authoritative and `allowed_models`
    /// is ignored; each id is resolved against the current models in the
    /// snapshot and the resolved name is matched with the same single-`*`
    /// glob rule. An id naming a wildcard model therefore grants every name
    /// that model's pattern covers — including a name an exact-match model
    /// of its own serves, which is how the name form behaves too. An id
    /// matching no model grants nothing.
    ///
    /// Set to `null` it means the same as omitted: the key falls back to
    /// `allowed_models`. A producer must therefore write `[]`, never
    /// `null`, for a key that is meant to grant no model — the two are
    /// opposite grants, and an empty list is the one that is authoritative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_model_ids: Option<Vec<String>>,

    /// Request, token, and concurrency limits for this key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimit>,

    /// Team this API key belongs to. Used for matching team-scope
    /// rate limit policies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub team_id: Option<String>,

    /// Org member who owns this key. Used for matching member-scope
    /// rate limit policies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub user_id: Option<String>,

    /// Readable display name of the owning member. Used only for telemetry
    /// labels alongside `user_id`; never used for authentication or routing.
    /// When omitted, telemetry reports the user name as `"unknown"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_name: Option<String>,

    /// External identity bound to this key for JWT authentication.
    /// When a request presents a valid JWT issued by the
    /// `oidc_providers` entry named in `jwt_provider`, the value of
    /// that provider's `identity_claim` selects the key whose
    /// `jwt_subject` equals it, and the request proceeds with this
    /// key's permissions, rate limits, and budget. The `(jwt_provider,
    /// jwt_subject)` pair is unique within the environment. When
    /// omitted, the key is never selected by JWT authentication.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub jwt_subject: Option<String>,

    /// Name of the `oidc_providers` entry permitted to assert this
    /// key's `jwt_subject`. A subject is only ever resolved for the
    /// trust provider named here, so a second trusted provider cannot
    /// mint a token impersonating this provider's identity of the same
    /// name. Required whenever `jwt_subject` is set; ignored otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub jwt_provider: Option<String>,

    /// This key's own layer of the MCP tool ACL, as namespaced
    /// `<server>__<tool>` glob patterns, or as `allow_ids` / `deny_ids`
    /// entries naming the server by resource id. It is intersected with the
    /// environment and team MCP access policies: every present layer must
    /// allow a tool and no layer may deny it. When omitted the key adds no
    /// constraint of its own — but with no layer present anywhere the grant
    /// is empty, so MCP access is always granted explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_access: Option<McpAccess>,

    /// Per-MCP-server limits for this key, keyed by the registered MCP server
    /// name — the `<server>` half of the `<server>__<tool>` names the gateway
    /// exposes. A `tools/call` is metered against the entry for the server it
    /// targets **and** the key's own `rate_limit`, each in its own counter, so
    /// a burst against one server never consumes another's budget. A server
    /// with no entry here is bounded by `rate_limit` alone. Only tool calls
    /// are metered; the `initialize` / `tools/list` handshake is not.
    ///
    /// Read only when `mcp_rate_limits_by_id` is absent; ignored entirely
    /// when it is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_rate_limits: Option<BTreeMap<String, McpRateLimit>>,

    /// The same per-server limits, keyed by the MCP server's resource id
    /// (`mcp_servers/<id>`) rather than by its name, so renaming a server
    /// does not detach the limit that was set for it.
    ///
    /// Present — including as an empty object — it is authoritative and
    /// `mcp_rate_limits` is ignored; an empty object therefore leaves every
    /// server bounded by `rate_limit` alone. A key naming no registered
    /// server imposes nothing, and the other entries are unaffected. Set to
    /// `null` it means the same as omitted: the key falls back to
    /// `mcp_rate_limits`.
    ///
    /// Each key names one server exactly; there is no "every server" key,
    /// which is the same as today — a server with no entry is bounded by
    /// `rate_limit` alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_rate_limits_by_id: Option<BTreeMap<String, McpRateLimit>>,

    /// A2A agents this key may reach, named by their registered names. Entries
    /// are matched as single-`*` globs, mirroring `allowed_models`: `"*"` grants
    /// every agent and an entry without a `*` matches one agent exactly. When
    /// omitted, set to `null`, or set to an empty list, the key has no A2A
    /// agent access — access is granted explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_agents: Option<Vec<String>>,

    /// Passthrough routes this key may use, named by their registered names.
    /// Entries are matched as single-`*` globs, mirroring `allowed_models`:
    /// `"*"` grants every route and an entry without a `*` matches one route
    /// exactly. When omitted, set to `null`, or set to an empty list, the key
    /// may use no passthrough route — access is granted explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_routes: Option<Vec<String>>,

    /// RFC 3339 timestamp after which the key stops authenticating.
    /// Requests presenting an expired key are rejected with `401`.
    /// When omitted or set to `null`, the key never expires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,

    /// Administratively disabled. A disabled key is rejected with `401`
    /// until it is enabled again; the key itself is preserved. Treated
    /// as `false` when omitted.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub disabled: bool,

    /// etcd-key uuid. Filled by the loader and never included in the JSON payload.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl ApiKey {
    /// Canonical hash function for converting an `Authorization:
    /// Bearer <plaintext>` value to the form persisted in the
    /// snapshot (and on the cp-api side as `api_keys.key_hash`).
    /// SHA-256, lowercase hex. Both sides MUST use this exact
    /// function — test fixtures and the `aisix-proxy::auth`
    /// extractor both call through here.
    pub fn hash_bearer(plaintext: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(plaintext.as_bytes());
        hex::encode(h.finalize())
    }

    /// True if the key's `expires_at` deadline has passed at `now`.
    /// Keys without a deadline never expire. The comparison is strict
    /// (`<`): the key is still valid at the deadline instant itself,
    /// matching the established gateway-ecosystem semantics.
    pub fn is_expired_at(&self, now: DateTime<Utc>) -> bool {
        self.expires_at.is_some_and(|deadline| deadline < now)
    }

    /// True if this key is allowed to call the model the caller addressed.
    ///
    /// **The single chokepoint for the key→model ACL.** Every request path
    /// that gates on model access calls this and nothing else, so the two
    /// grant shapes below can never diverge across the endpoint family. It
    /// keys on the caller-addressed entry — the name the request names, or
    /// the stored name of the entry it references — and never on whatever
    /// target dispatch later picks, for every model kind.
    ///
    /// The key grants models one of two ways:
    ///
    /// - `allowed_model_ids` present (an empty array included): each id is
    ///   resolved to the current name of the model carrying it and that name
    ///   is matched as a single-`*` glob, so an id naming a wildcard model
    ///   still grants every name its pattern covers. An id resolving to no
    ///   model grants nothing. `allowed_models` is not read at all.
    /// - `allowed_model_ids` absent: `allowed_models` names are matched as
    ///   single-`*` globs — `"*"` grants every model, `"openai/*"` every
    ///   `openai/*` name, an entry without a `*` matches exactly.
    ///
    /// With neither present the key may use no model. Resolution is done per
    /// request against the live table rather than cached, so a model rename
    /// takes effect on the next request with no rewrite of any key document.
    pub fn can_access(&self, snapshot: &super::AisixSnapshot, model_name: &str) -> bool {
        match &self.allowed_model_ids {
            Some(ids) => ids.iter().any(|id| {
                snapshot
                    .models
                    .get_by_id(id)
                    .is_some_and(|m| crate::wildcard::wildcard_matches(m.value.name(), model_name))
            }),
            None => self
                .allowed_models
                .iter()
                .any(|n| crate::wildcard::wildcard_matches(n, model_name)),
        }
    }

    /// The limits this key carries for one MCP server, named as it is
    /// registered (the `<server>` namespace of a `<server>__<tool>` call).
    /// `None` when the key sets no limit for that server.
    ///
    /// **The single chokepoint for the key→MCP-server limit.** The key
    /// carries the limits one of two ways and this decides between them, so
    /// no caller can read one shape and miss the other:
    ///
    /// - `mcp_rate_limits_by_id` present (an empty object included): the
    ///   server's registered name is resolved to the resource id it is
    ///   stored under and the limit is looked up by that id, so a rename
    ///   keeps the limit attached. `mcp_rate_limits` is not read at all.
    /// - `mcp_rate_limits_by_id` absent: `mcp_rate_limits` is looked up by
    ///   the server's name, as before.
    ///
    /// Resolution is done per request against the live table rather than
    /// cached, so a server rename takes effect on the next request with no
    /// rewrite of any key document.
    pub fn mcp_rate_limit<'a>(
        &'a self,
        servers: &'a super::McpServerIndex,
        server: &'a str,
    ) -> Option<McpServerLimit<'a>> {
        match &self.mcp_rate_limits_by_id {
            Some(by_id) => {
                // Bucketed on the id, not the name: a rename must not hand
                // the key a fresh window, which is the whole reason the
                // limit was attached by id.
                let id = servers.id_of(server)?;
                Some(McpServerLimit {
                    bucket: id,
                    limits: by_id.get(id)?,
                })
            }
            // Bucketed on the name, which is also what selected it: a
            // rename detaches a name-keyed limit outright, so there is no
            // window to carry over, and keying these on the id instead
            // would reset every counter in the fleet at upgrade.
            None => Some(McpServerLimit {
                bucket: server,
                limits: self.mcp_rate_limits.as_ref()?.get(server)?,
            }),
        }
    }

    /// True if this key may reach the given A2A agent, named by its registered
    /// name.
    ///
    /// Semantics mirror [`ApiKey::can_access`]: entries are single-`*`
    /// globs, so `"*"` grants every agent; entries without a `*` match exactly.
    /// A key with no `allowed_agents` (or an empty list) may reach no A2A agent
    /// — access is granted explicitly.
    pub fn can_access_agent(&self, agent: &str) -> bool {
        match &self.allowed_agents {
            None => false,
            Some(allowed) => allowed
                .iter()
                .any(|a| crate::wildcard::wildcard_matches(a, agent)),
        }
    }

    /// True if this key may use the given passthrough route, named by its
    /// registered name.
    ///
    /// Semantics mirror [`ApiKey::can_access_agent`]: entries are single-`*`
    /// globs, so `"*"` grants every route; entries without a `*` match
    /// exactly. A key with no `allowed_routes` (or an empty list) may use no
    /// passthrough route — access is granted explicitly.
    pub fn can_access_route(&self, route: &str) -> bool {
        match &self.allowed_routes {
            None => false,
            Some(allowed) => allowed
                .iter()
                .any(|r| crate::wildcard::wildcard_matches(r, route)),
        }
    }

    /// Iterate over the names of models this key may access, filtering them
    /// against a known universe of model names. Delegates to [`Self::can_access`]
    /// so the listing can never advertise a name the request path would
    /// reject, whichever grant shape the key carries.
    pub fn accessible_models<'a>(
        &'a self,
        snapshot: &super::AisixSnapshot,
        all_models: impl Iterator<Item = &'a str> + 'a,
    ) -> Vec<&'a str> {
        all_models
            .filter(|name| self.can_access(snapshot, name))
            .collect()
    }
}

/// One key's limits for one MCP server, with the identity its counter is
/// bucketed on — the server's resource id when the limit was attached by
/// id, its name when it was attached by name.
///
/// The two must not be confused: a counter that changes bucket resets the
/// window it was in the middle of.
pub struct McpServerLimit<'a> {
    pub bucket: &'a str,
    pub limits: &'a McpRateLimit,
}

impl Resource for ApiKey {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// For ApiKey the "secondary-indexed" field is `key_hash` — the
    /// proxy hashes the incoming bearer once and uses that as the
    /// lookup key. The name-index in the snapshot therefore points
    /// from key_hash → id.
    fn name(&self) -> &str {
        &self.key_hash
    }

    /// Path segment under `/aisix/<env>/`. v3 (prd-09a §9A.7B.2) uses
    /// the underscored form `api_keys` to align with cp-api migration
    /// 008's table name. v2 used `apikeys` with no underscore. The v3
    /// dp-manager only writes the underscored form.
    fn kind() -> &'static str {
        "api_keys"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SHA-256 hex of `"sk-my-api-key-123"`.
    const SAMPLE_PLAINTEXT: &str = "sk-my-api-key-123";
    const SAMPLE_HASH: &str = "91ed2dbc407561556f3e7be98ba0bd2a57986d6a868c482d867d19c6d40d201c";

    /// A snapshot holding one model per `(id, display_name)` pair, so an
    /// `allowed_model_ids` entry has something to resolve against.
    fn snapshot_with_models(models: &[(&str, &str)]) -> super::super::AisixSnapshot {
        let snap = super::super::AisixSnapshot::default();
        for (id, display_name) in models {
            let model: crate::models::Model = serde_json::from_str(&format!(
                r#"{{
                  "display_name": "{display_name}",
                  "provider": "openai",
                  "model_name": "gpt-4o",
                  "provider_key_id": "11111111-1111-1111-1111-111111111111"
                }}"#
            ))
            .unwrap();
            snap.models
                .insert(crate::resource::ResourceEntry::new(*id, model, 1));
        }
        snap
    }

    fn empty_snapshot() -> super::super::AisixSnapshot {
        super::super::AisixSnapshot::default()
    }

    fn sample() -> ApiKey {
        serde_json::from_str(&format!(
            r#"{{
              "key_hash": "{SAMPLE_HASH}",
              "allowed_models": ["my-gpt4", "my-claude"],
              "rate_limit": {{"rpm": 60, "concurrency": 5}}
            }}"#
        ))
        .unwrap()
    }

    #[test]
    fn deserialises_spec_sample() {
        let k = sample();
        assert_eq!(k.key_hash, SAMPLE_HASH);
        assert_eq!(k.allowed_models.len(), 2);
        assert_eq!(k.rate_limit.as_ref().unwrap().concurrency, Some(5));
    }

    #[test]
    fn key_hash_is_sha256_of_plaintext() {
        // Pin the SAMPLE_HASH constant to its plaintext so future
        // fixture rotations can't drift one without the other.
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(SAMPLE_PLAINTEXT.as_bytes());
        let got = hex::encode(h.finalize());
        assert_eq!(got, SAMPLE_HASH);
    }

    #[test]
    fn empty_allowed_models_denies_everything() {
        let k = ApiKey {
            key_hash: "abc".into(),
            display_name: None,
            allowed_models: vec![],
            allowed_model_ids: None,
            rate_limit: None,
            team_id: None,
            user_id: None,
            user_name: None,
            jwt_subject: None,
            jwt_provider: None,
            mcp_access: None,
            allowed_routes: None,
            mcp_rate_limits: None,
            mcp_rate_limits_by_id: None,
            allowed_agents: None,
            expires_at: None,
            disabled: false,
            runtime_id: String::new(),
        };
        let snap = empty_snapshot();
        assert!(!k.can_access(&snap, "my-gpt4"));
        assert!(!k.can_access(&snap, "anything"));
    }

    #[test]
    fn mcp_access_block_roundtrips_and_defaults_absent() {
        // A key with no block of its own adds no layer to the ACL.
        let unconstrained = sample();
        assert!(unconstrained.mcp_access.is_none());
        let v = serde_json::to_value(&unconstrained).unwrap();
        assert!(v.get("mcp_access").is_none());

        let k: ApiKey = serde_json::from_str(
            r#"{
              "key_hash": "h",
              "allowed_models": [],
              "mcp_access": {"allow": ["github__*"], "deny": ["github__delete_repo"]}
            }"#,
        )
        .unwrap();
        let access = k.mcp_access.as_ref().unwrap();
        assert_eq!(access.allow, vec!["github__*"]);
        assert_eq!(access.deny, vec!["github__delete_repo"]);
        // Round-trip preserves the block.
        let v = serde_json::to_value(&k).unwrap();
        assert_eq!(v["mcp_access"]["allow"], serde_json::json!(["github__*"]));
    }

    #[test]
    fn mcp_access_tolerates_unknown_inner_fields_for_forward_compat() {
        // cp-api may ship new `mcp_access` fields ahead of the DP rolling
        // out; serde must accept them (the write path still rejects them
        // via `validate_apikey` in models/schema.rs).
        let k: ApiKey = serde_json::from_str(
            r#"{"key_hash":"h","allowed_models":[],"mcp_access":{"allow":["*"],"widen":["*"]}}"#,
        )
        .unwrap();
        assert_eq!(k.mcp_access.unwrap().allow, vec!["*"]);
    }

    #[test]
    fn can_access_agent_enforces_allowlist() {
        // No `allowed_agents` (or null / empty) → no A2A agent access.
        let none: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":["*"]}"#).unwrap();
        assert!(!none.can_access_agent("invoice-processor"));
        let empty: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":[],"allowed_agents":[]}"#)
                .unwrap();
        assert!(!empty.can_access_agent("invoice-processor"));

        // Exact name grants only that agent.
        let specific: ApiKey = serde_json::from_str(
            r#"{"key_hash":"h","allowed_models":[],"allowed_agents":["invoice-processor"]}"#,
        )
        .unwrap();
        assert!(specific.can_access_agent("invoice-processor"));
        assert!(!specific.can_access_agent("translator"));

        // Wildcard grants every agent.
        let wildcard: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":[],"allowed_agents":["*"]}"#)
                .unwrap();
        assert!(wildcard.can_access_agent("anything"));
    }

    #[test]
    fn can_access_checks_whitelist() {
        let k = sample();
        let snap = empty_snapshot();
        assert!(k.can_access(&snap, "my-gpt4"));
        assert!(k.can_access(&snap, "my-claude"));
        assert!(!k.can_access(&snap, "other"));
    }

    #[test]
    fn wildcard_grants_access_to_any_model() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"abc","allowed_models":["*"]}"#).unwrap();
        let snap = empty_snapshot();
        assert!(k.can_access(&snap, "my-gpt4"));
        assert!(k.can_access(&snap, "literally-anything"));
    }

    #[test]
    fn glob_entry_grants_matching_names() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"abc","allowed_models":["openai/*"]}"#).unwrap();
        let snap = empty_snapshot();
        assert!(k.can_access(&snap, "openai/gpt-4o"));
        assert!(k.can_access(&snap, "openai/gpt-4o-mini"));
        assert!(!k.can_access(&snap, "anthropic/claude"));
        assert!(!k.can_access(&snap, "openai")); // prefix must be followed by the glob
    }

    #[test]
    fn accessible_models_honors_glob_entry() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"abc","allowed_models":["openai/*"]}"#).unwrap();
        let universe = ["openai/gpt-4o", "openai/o1", "anthropic/claude"];
        let mut accessible = k.accessible_models(&empty_snapshot(), universe.iter().copied());
        accessible.sort_unstable();
        assert_eq!(accessible, vec!["openai/gpt-4o", "openai/o1"]);
    }

    #[test]
    fn accessible_models_expands_wildcard_to_full_universe() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"abc","allowed_models":["*"]}"#).unwrap();
        let universe = ["a", "b", "c"];
        let accessible = k.accessible_models(&empty_snapshot(), universe.iter().copied());
        assert_eq!(accessible, vec!["a", "b", "c"]);
    }

    #[test]
    fn accessible_models_filters_explicit_list() {
        let k = sample(); // allowed: ["my-gpt4", "my-claude"]
        let universe = ["my-gpt4", "my-claude", "other"];
        let mut accessible = k.accessible_models(&empty_snapshot(), universe.iter().copied());
        accessible.sort_unstable();
        assert_eq!(accessible, vec!["my-claude", "my-gpt4"]);
    }

    #[test]
    fn accessible_models_empty_list_returns_nothing() {
        let k: ApiKey = serde_json::from_str(r#"{"key_hash":"abc","allowed_models":[]}"#).unwrap();
        let universe = ["a", "b"];
        assert!(k
            .accessible_models(&empty_snapshot(), universe.iter().copied())
            .is_empty());
    }

    #[test]
    fn tolerates_unknown_fields_for_forward_compat() {
        // cp-api may ship new fields ahead of the DP rolling out; serde
        // must accept them (the write path still rejects them via
        // `validate_apikey` in models/schema.rs).
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"x","allowed_models":[],"extra":1}"#).unwrap();
        assert_eq!(k.key_hash, "x");
    }

    #[test]
    fn resource_trait_points_at_key_and_kind() {
        let mut k = sample();
        k.runtime_id = "uuid-ak".into();
        assert_eq!(<ApiKey as Resource>::kind(), "api_keys");
        assert_eq!(k.id(), "uuid-ak");
        // Resource::name now returns key_hash, not plaintext.
        assert_eq!(k.name(), SAMPLE_HASH);
    }

    #[test]
    fn deserialises_with_team_and_user_fields() {
        let k: ApiKey = serde_json::from_str(&format!(
            r#"{{
              "key_hash": "{SAMPLE_HASH}",
              "allowed_models": ["gpt-4o"],
              "team_id": "team-uuid-1",
              "user_id": "member-uuid-1",
              "user_name": "Alice Example"
            }}"#
        ))
        .unwrap();
        assert_eq!(k.team_id.as_deref(), Some("team-uuid-1"));
        assert_eq!(k.user_id.as_deref(), Some("member-uuid-1"));
        assert_eq!(k.user_name.as_deref(), Some("Alice Example"));
    }

    #[test]
    fn absent_team_user_fields_default_to_none() {
        let k = sample();
        assert!(k.team_id.is_none());
        assert!(k.user_id.is_none());
        assert!(k.user_name.is_none());
    }

    #[test]
    fn jwt_subject_roundtrips_and_defaults_absent() {
        // Every pre-existing key payload lacks `jwt_subject`; it must load
        // as None and stay off the wire so mixed-fleet DPs keep accepting
        // the row.
        let legacy = sample();
        assert!(legacy.jwt_subject.is_none());
        let v = serde_json::to_value(&legacy).unwrap();
        assert!(v.get("jwt_subject").is_none());

        let k: ApiKey = serde_json::from_str(
            r#"{"key_hash":"h","allowed_models":[],"jwt_subject":"agent-billing-01","jwt_provider":"corp-idp"}"#,
        )
        .unwrap();
        assert_eq!(k.jwt_subject.as_deref(), Some("agent-billing-01"));
        assert_eq!(k.jwt_provider.as_deref(), Some("corp-idp"));
        let v = serde_json::to_value(&k).unwrap();
        assert_eq!(v["jwt_subject"], "agent-billing-01");
        assert_eq!(v["jwt_provider"], "corp-idp");
    }

    #[test]
    fn absent_lifecycle_fields_mean_active_forever() {
        // Every pre-existing key payload lacks `expires_at`/`disabled`;
        // they must keep authenticating unchanged.
        let k = sample();
        assert!(k.expires_at.is_none());
        assert!(!k.disabled);
        assert!(!k.is_expired_at(chrono::Utc::now()));
    }

    #[test]
    fn explicit_null_expires_at_means_never_expires() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":[],"expires_at":null}"#)
                .unwrap();
        assert!(k.expires_at.is_none());
        assert!(!k.is_expired_at(chrono::Utc::now()));
    }

    #[test]
    fn is_expired_at_honors_deadline() {
        let k: ApiKey = serde_json::from_str(
            r#"{"key_hash":"h","allowed_models":[],"expires_at":"2030-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let before = "2029-12-31T23:59:59Z".parse().unwrap();
        let at = "2030-01-01T00:00:00Z".parse().unwrap();
        let after = "2030-01-01T00:00:01Z".parse().unwrap();
        assert!(!k.is_expired_at(before));
        // Strict comparison: still valid at the deadline instant,
        // expired strictly after it (ecosystem-aligned boundary).
        assert!(!k.is_expired_at(at));
        assert!(k.is_expired_at(after));
    }

    #[test]
    fn rejects_malformed_expires_at() {
        // A non-RFC3339 string must fail deserialization so the loader
        // rejects the row instead of silently treating the key as
        // never-expiring. Note the rejection is fail-closed on full
        // loads/resyncs (the key is absent from the snapshot); on the
        // incremental watch path a rejected UPDATE keeps the previous
        // version serving until the next resync (pre-existing
        // supervisor behavior shared by every resource kind).
        let r: Result<ApiKey, _> =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":[],"expires_at":"tomorrow"}"#);
        assert!(r.is_err());
    }

    #[test]
    fn disabled_roundtrips_and_defaults_false() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":[],"disabled":true}"#)
                .unwrap();
        assert!(k.disabled);
        // `disabled: false` is the default and stays off the wire.
        let v = serde_json::to_value(sample()).unwrap();
        assert!(v.get("disabled").is_none());
        assert!(v.get("expires_at").is_none());
    }

    #[test]
    fn allowed_model_ids_grant_by_resource_id() {
        let snap = snapshot_with_models(&[("m-1", "my-gpt4"), ("m-2", "my-claude")]);
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_model_ids":["m-1"]}"#).unwrap();
        assert!(k.can_access(&snap, "my-gpt4"));
        assert!(!k.can_access(&snap, "my-claude"));
    }

    #[test]
    fn allowed_model_ids_follow_a_rename() {
        let snap = snapshot_with_models(&[("m-1", "my-gpt4")]);
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_model_ids":["m-1"]}"#).unwrap();
        assert!(k.can_access(&snap, "my-gpt4"));

        // Same id, new name: the key document is untouched.
        let renamed = snapshot_with_models(&[("m-1", "my-gpt4-v2")]);
        assert!(renamed.models.get_by_name("my-gpt4").is_none());
        assert!(k.can_access(&renamed, "my-gpt4-v2"));
        assert!(!k.can_access(&renamed, "my-gpt4"));
    }

    #[test]
    fn allowed_model_ids_naming_a_wildcard_model_keep_its_pattern() {
        let snap = snapshot_with_models(&[("m-1", "gpt-*")]);
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_model_ids":["m-1"]}"#).unwrap();
        assert!(k.can_access(&snap, "gpt-4o"));
        assert!(!k.can_access(&snap, "claude-sonnet"));
    }

    #[test]
    fn allowed_model_ids_win_over_allowed_models() {
        let snap = snapshot_with_models(&[("m-1", "my-gpt4"), ("m-2", "my-claude")]);
        let k: ApiKey = serde_json::from_str(
            r#"{"key_hash":"h","allowed_models":["my-claude"],"allowed_model_ids":["m-1"]}"#,
        )
        .unwrap();
        assert!(k.can_access(&snap, "my-gpt4"));
        assert!(!k.can_access(&snap, "my-claude"));

        // Even `allowed_models: ["*"]` is ignored once ids are present.
        let widened: ApiKey = serde_json::from_str(
            r#"{"key_hash":"h","allowed_models":["*"],"allowed_model_ids":[]}"#,
        )
        .unwrap();
        assert!(!widened.can_access(&snap, "my-gpt4"));
    }

    #[test]
    fn unresolvable_id_grants_nothing_but_leaves_the_rest() {
        let snap = snapshot_with_models(&[("m-1", "my-gpt4")]);
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_model_ids":["m-1","m-gone"]}"#)
                .unwrap();
        assert!(k.can_access(&snap, "my-gpt4"));
        assert!(!k.can_access(&snap, "m-gone"));
        assert!(!k.can_access(&snap, "anything-else"));
    }

    #[test]
    fn neither_grant_field_loads_and_denies_everything() {
        let snap = snapshot_with_models(&[("m-1", "my-gpt4")]);
        let k: ApiKey = serde_json::from_str(r#"{"key_hash":"h"}"#).unwrap();
        assert!(k.allowed_models.is_empty());
        assert!(k.allowed_model_ids.is_none());
        assert!(!k.can_access(&snap, "my-gpt4"));
        assert!(!k.can_access(&snap, "*"));
    }

    #[test]
    fn allowed_model_ids_stays_off_the_wire_when_absent() {
        let v = serde_json::to_value(sample()).unwrap();
        assert!(v.get("allowed_model_ids").is_none());

        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_model_ids":["m-1"]}"#).unwrap();
        let v = serde_json::to_value(&k).unwrap();
        assert_eq!(v["allowed_model_ids"], serde_json::json!(["m-1"]));
    }

    /// A registered-server index over one `(id, name)` pair per server.
    fn server_index(servers: &[(&str, &str)]) -> super::super::McpServerIndex {
        let snap = super::super::AisixSnapshot::default();
        for (id, name) in servers {
            let server: crate::models::McpServer = serde_json::from_str(&format!(
                r#"{{"name":"{name}","url":"https://example.test/mcp"}}"#
            ))
            .unwrap();
            snap.mcp_servers
                .insert(crate::resource::ResourceEntry::new(*id, server, 1));
        }
        super::super::McpServerIndex::build(&snap.mcp_servers)
    }

    const ONE_RPM: &str = r#"{"rpm":1}"#;

    #[test]
    fn mcp_rate_limits_are_looked_up_by_server_name_by_default() {
        let servers = server_index(&[("s-github", "github")]);
        let k: ApiKey = serde_json::from_str(&format!(
            r#"{{"key_hash":"h","mcp_rate_limits":{{"github":{ONE_RPM}}}}}"#
        ))
        .unwrap();
        assert!(k.mcp_rate_limit(&servers, "github").is_some());
        assert!(k.mcp_rate_limit(&servers, "slack").is_none());
    }

    #[test]
    fn mcp_rate_limits_by_id_are_looked_up_through_the_server_index() {
        let servers = server_index(&[("s-github", "github"), ("s-slack", "slack")]);
        let k: ApiKey = serde_json::from_str(&format!(
            r#"{{"key_hash":"h","mcp_rate_limits_by_id":{{"s-github":{ONE_RPM}}}}}"#
        ))
        .unwrap();
        assert!(k.mcp_rate_limit(&servers, "github").is_some());
        assert!(k.mcp_rate_limit(&servers, "slack").is_none());
    }

    #[test]
    fn mcp_rate_limits_by_id_follow_a_rename() {
        let k: ApiKey = serde_json::from_str(&format!(
            r#"{{"key_hash":"h","mcp_rate_limits_by_id":{{"s-github":{ONE_RPM}}}}}"#
        ))
        .unwrap();
        assert!(k
            .mcp_rate_limit(&server_index(&[("s-github", "github")]), "github")
            .is_some());

        // Same id, new name: the key document is untouched.
        let renamed = server_index(&[("s-github", "github-v2")]);
        assert!(k.mcp_rate_limit(&renamed, "github-v2").is_some());
        assert!(k.mcp_rate_limit(&renamed, "github").is_none());
    }

    #[test]
    fn mcp_rate_limits_by_id_win_over_the_name_form() {
        let servers = server_index(&[("s-github", "github"), ("s-slack", "slack")]);
        let k: ApiKey = serde_json::from_str(&format!(
            r#"{{"key_hash":"h",
                 "mcp_rate_limits":{{"slack":{ONE_RPM}}},
                 "mcp_rate_limits_by_id":{{"s-github":{ONE_RPM}}}}}"#
        ))
        .unwrap();
        assert!(k.mcp_rate_limit(&servers, "github").is_some());
        assert!(
            k.mcp_rate_limit(&servers, "slack").is_none(),
            "the name form is not read at all once the id form is present"
        );

        // An empty object is authoritative too: every server is bounded by
        // `rate_limit` alone.
        let emptied: ApiKey = serde_json::from_str(&format!(
            r#"{{"key_hash":"h",
                 "mcp_rate_limits":{{"github":{ONE_RPM}}},
                 "mcp_rate_limits_by_id":{{}}}}"#
        ))
        .unwrap();
        assert!(emptied.mcp_rate_limit(&servers, "github").is_none());
    }

    #[test]
    fn an_unresolvable_server_id_imposes_no_limit_and_spares_its_neighbours() {
        let servers = server_index(&[("s-slack", "slack")]);
        let k: ApiKey = serde_json::from_str(&format!(
            r#"{{"key_hash":"h","mcp_rate_limits_by_id":{{"s-gone":{ONE_RPM},"s-slack":{ONE_RPM}}}}}"#
        ))
        .unwrap();
        assert!(k.mcp_rate_limit(&servers, "slack").is_some());
        assert!(k.mcp_rate_limit(&servers, "s-gone").is_none());
    }

    #[test]
    fn the_counter_bucket_is_whichever_identity_selected_the_limit() {
        // A rename detaches a name-keyed limit outright, so its counter has
        // no window to carry over — but an id-keyed limit survives the
        // rename, and bucketing it on the name would hand the key a fresh
        // window at the exact moment the feature exists to be transparent.
        let servers = server_index(&[("s-github", "github")]);

        let by_name: ApiKey = serde_json::from_str(&format!(
            r#"{{"key_hash":"h","mcp_rate_limits":{{"github":{ONE_RPM}}}}}"#
        ))
        .unwrap();
        assert_eq!(
            by_name.mcp_rate_limit(&servers, "github").unwrap().bucket,
            "github"
        );

        let by_id: ApiKey = serde_json::from_str(&format!(
            r#"{{"key_hash":"h","mcp_rate_limits_by_id":{{"s-github":{ONE_RPM}}}}}"#
        ))
        .unwrap();
        assert_eq!(
            by_id.mcp_rate_limit(&servers, "github").unwrap().bucket,
            "s-github"
        );
        // And it stays that bucket across the rename.
        let renamed = server_index(&[("s-github", "github-v2")]);
        assert_eq!(
            by_id.mcp_rate_limit(&renamed, "github-v2").unwrap().bucket,
            "s-github"
        );
    }

    #[test]
    fn mcp_rate_limits_by_id_stays_off_the_wire_when_absent() {
        let v = serde_json::to_value(sample()).unwrap();
        assert!(v.get("mcp_rate_limits_by_id").is_none());

        let k: ApiKey = serde_json::from_str(&format!(
            r#"{{"key_hash":"h","mcp_rate_limits_by_id":{{"s-1":{ONE_RPM}}}}}"#
        ))
        .unwrap();
        let v = serde_json::to_value(&k).unwrap();
        assert!(v["mcp_rate_limits_by_id"]["s-1"].is_object());
    }

    #[test]
    fn accessible_models_follows_the_id_grant() {
        let snap = snapshot_with_models(&[("m-1", "my-gpt4"), ("m-2", "my-claude")]);
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_model_ids":["m-2"]}"#).unwrap();
        let names = ["my-gpt4", "my-claude"];
        assert_eq!(
            k.accessible_models(&snap, names.iter().copied()),
            vec!["my-claude"]
        );
    }
}
