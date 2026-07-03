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

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::rate_limit::RateLimit;
use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ApiKey {
    /// SHA-256 hexadecimal hash of the plaintext bearer. The proxy hashes
    /// incoming bearer tokens before lookup.
    #[schemars(length(min = 1))]
    pub key_hash: String,

    /// Model identifiers this key may use. An empty array denies access to every model.
    pub allowed_models: Vec<String>,

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

    /// MCP tools this key may call, as namespaced `<server>__<tool>` names
    /// (the form the gateway exposes). Entries are matched as single-`*`
    /// globs, mirroring `allowed_models`: `"*"` grants every tool and
    /// `"<server>__*"` grants every tool on one server (e.g. `"github__*"`);
    /// an entry without a `*` matches one tool exactly. When omitted or set
    /// to `null`, the key has no MCP tool access — access is granted
    /// explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,

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

    /// True if this key is allowed to call the given Model.
    ///
    /// Entries are matched as single-`*` globs, so `"*"` grants every model and
    /// `"openai/*"` grants every `openai/*` name (pairing with wildcard Models);
    /// entries without a `*` match exactly. An empty `allowed_models` list denies
    /// everything (spec §3 authz rule).
    pub fn can_access(&self, model_name: &str) -> bool {
        self.allowed_models
            .iter()
            .any(|n| crate::wildcard::wildcard_matches(n, model_name))
    }

    /// True if this key may call the given MCP tool, named in the gateway's
    /// namespaced `<server>__<tool>` form.
    ///
    /// Entries are matched as single-`*` globs, so `"*"` grants every tool and
    /// `"<server>__*"` grants every tool on that server (e.g. `"github__*"`
    /// permits `github__create_issue`); entries without a `*` match exactly.
    /// A key with no `allowed_tools` (or an empty list) may call no MCP tools —
    /// access is granted explicitly, matching [`ApiKey::can_access`].
    ///
    /// Currently exercised only by tests: the live MCP enforcement path builds
    /// an `aisix_mcp::ToolAcl` from `allowed_tools` and uses the identical
    /// matcher, so this method is kept in lockstep as the documented mirror.
    pub fn can_access_tool(&self, tool: &str) -> bool {
        match &self.allowed_tools {
            None => false,
            Some(allowed) => allowed
                .iter()
                .any(|t| crate::wildcard::wildcard_matches(t, tool)),
        }
    }

    /// Iterate over the names of models this key may access, filtering them
    /// against a known universe of model names. Delegates to [`Self::can_access`]
    /// so glob entries stay consistent with per-request authz: `*` expands to
    /// the full universe and `openai/*` to every matching name.
    pub fn accessible_models<'a>(
        &'a self,
        all_models: impl Iterator<Item = &'a str> + 'a,
    ) -> Vec<&'a str> {
        all_models.filter(|name| self.can_access(name)).collect()
    }
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
            allowed_models: vec![],
            rate_limit: None,
            team_id: None,
            user_id: None,
            user_name: None,
            allowed_tools: None,
            expires_at: None,
            disabled: false,
            runtime_id: String::new(),
        };
        assert!(!k.can_access("my-gpt4"));
        assert!(!k.can_access("anything"));
    }

    #[test]
    fn can_access_tool_enforces_namespaced_allowlist() {
        // No `allowed_tools` configured → no MCP tool access.
        let none: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":["*"]}"#).unwrap();
        assert!(!none.can_access_tool("github__create_issue"));

        // Explicit null has the same no-access behavior as omission.
        let null: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":["*"],"allowed_tools":null}"#)
                .unwrap();
        assert!(!null.can_access_tool("github__create_issue"));

        // Empty list also denies everything.
        let empty: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":[],"allowed_tools":[]}"#)
                .unwrap();
        assert!(!empty.can_access_tool("github__create_issue"));

        // Exact namespaced names.
        let specific: ApiKey = serde_json::from_str(
            r#"{"key_hash":"h","allowed_models":[],"allowed_tools":["github__create_issue"]}"#,
        )
        .unwrap();
        assert!(specific.can_access_tool("github__create_issue"));
        assert!(!specific.can_access_tool("github__delete_repo"));

        // Wildcard grants every tool.
        let wildcard: ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":[],"allowed_tools":["*"]}"#)
                .unwrap();
        assert!(wildcard.can_access_tool("anything__at_all"));

        // Per-server wildcard grants every tool on that server only.
        let per_server: ApiKey = serde_json::from_str(
            r#"{"key_hash":"h","allowed_models":[],"allowed_tools":["github__*"]}"#,
        )
        .unwrap();
        assert!(per_server.can_access_tool("github__create_issue"));
        assert!(per_server.can_access_tool("github__delete_repo"));
        // It must not leak across the server boundary: a different server,
        // and a server whose name merely shares the prefix, are both denied.
        assert!(!per_server.can_access_tool("slack__post_message"));
        assert!(!per_server.can_access_tool("githubenterprise__create_issue"));

        // The glob is a single `*` anywhere, not only trailing (same as
        // `allowed_models`): `"*__readonly"` is a genuine any-server grant of
        // a same-named tool. Pinned so the breadth stays intentional.
        let any_server: ApiKey = serde_json::from_str(
            r#"{"key_hash":"h","allowed_models":[],"allowed_tools":["*__readonly"]}"#,
        )
        .unwrap();
        assert!(any_server.can_access_tool("github__readonly"));
        assert!(any_server.can_access_tool("slack__readonly"));
        // The suffix still anchors — a longer tool name doesn't match.
        assert!(!any_server.can_access_tool("github__readonly_admin"));
    }

    #[test]
    fn can_access_checks_whitelist() {
        let k = sample();
        assert!(k.can_access("my-gpt4"));
        assert!(k.can_access("my-claude"));
        assert!(!k.can_access("other"));
    }

    #[test]
    fn wildcard_grants_access_to_any_model() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"abc","allowed_models":["*"]}"#).unwrap();
        assert!(k.can_access("my-gpt4"));
        assert!(k.can_access("literally-anything"));
    }

    #[test]
    fn glob_entry_grants_matching_names() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"abc","allowed_models":["openai/*"]}"#).unwrap();
        assert!(k.can_access("openai/gpt-4o"));
        assert!(k.can_access("openai/gpt-4o-mini"));
        assert!(!k.can_access("anthropic/claude"));
        assert!(!k.can_access("openai")); // prefix must be followed by the glob
    }

    #[test]
    fn accessible_models_honors_glob_entry() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"abc","allowed_models":["openai/*"]}"#).unwrap();
        let universe = ["openai/gpt-4o", "openai/o1", "anthropic/claude"];
        let mut accessible = k.accessible_models(universe.iter().copied());
        accessible.sort_unstable();
        assert_eq!(accessible, vec!["openai/gpt-4o", "openai/o1"]);
    }

    #[test]
    fn accessible_models_expands_wildcard_to_full_universe() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key_hash":"abc","allowed_models":["*"]}"#).unwrap();
        let universe = ["a", "b", "c"];
        let accessible = k.accessible_models(universe.iter().copied());
        assert_eq!(accessible, vec!["a", "b", "c"]);
    }

    #[test]
    fn accessible_models_filters_explicit_list() {
        let k = sample(); // allowed: ["my-gpt4", "my-claude"]
        let universe = ["my-gpt4", "my-claude", "other"];
        let mut accessible = k.accessible_models(universe.iter().copied());
        accessible.sort_unstable();
        assert_eq!(accessible, vec!["my-claude", "my-gpt4"]);
    }

    #[test]
    fn accessible_models_empty_list_returns_nothing() {
        let k: ApiKey = serde_json::from_str(r#"{"key_hash":"abc","allowed_models":[]}"#).unwrap();
        let universe = ["a", "b"];
        assert!(k.accessible_models(universe.iter().copied()).is_empty());
    }

    #[test]
    fn rejects_unknown_fields() {
        let r: Result<ApiKey, _> =
            serde_json::from_str(r#"{"key_hash":"x","allowed_models":[],"extra":1}"#);
        assert!(r.is_err());
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
}
