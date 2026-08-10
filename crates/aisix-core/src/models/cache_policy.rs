//! `CachePolicy` entity — per-env prompt-response cache rules. The
//! control plane (cp-api) writes these to etcd at
//! `/aisix/<env>/cache_policies/<uuid>`; the DP loads them on watch
//! and `aisix-proxy::cache_gate` consults them on every chat request.
//!
//! Backends supported: `memory` + `redis` — the *storage* dimension.
//! Matching is layered: every policy does exact-fingerprint matching,
//! and a policy that carries a [`SemanticCacheConfig`] additionally
//! matches by embedding similarity when the exact layer misses.
//!
//! See `crates/aisix-cache` for the cache backend itself; this module
//! is the wire shape only.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// Cache backend choice for requests matched by a cache policy. `redis` requires `cache.redis`. Otherwise matching requests are not cached.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CacheBackend {
    #[default]
    Memory,
    Redis,
}

/// Sharing boundary for cache entries created under a policy. Applies to
/// both matching layers: an entry written in one scope bucket is never
/// served to a request in another.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CacheScope {
    /// Entries are private to the API key that created them. The safe
    /// default: one caller's answers are never replayed to another.
    #[default]
    ApiKey,
    /// Entries are shared by every API key in the environment. Pick this
    /// for shared-knowledge traffic (FAQ, documentation Q&A) where
    /// cross-caller reuse is the point.
    Env,
}

/// Embedding-similarity matching for a cache policy. When present, a
/// request that misses the exact layer is embedded and compared against
/// stored entries; the nearest entry at or above `threshold` cosine
/// similarity is served. Only requests whose messages are entirely text
/// participate — requests containing images or audio never match by
/// similarity.
///
/// On `backend: redis`, similarity matching requires a Redis server
/// with vector search (Redis 8 or later, or the search module) in
/// `single` or `sentinel` mode; without it — or in `cluster` mode —
/// the policy keeps serving exact matches only and a warning is
/// logged.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct SemanticCacheConfig {
    /// Name of the `embedding` model used to embed requests. The model
    /// must exist in the same environment and carry an `embedding`
    /// block; its `dimensions` value fixes the vector size for this
    /// policy's entries.
    #[schemars(length(min = 1))]
    pub embedding_model: String,

    /// Minimum cosine similarity for a stored entry to be served, in
    /// `[0, 1]`. Higher is stricter. Values below `0.9` noticeably
    /// increase wrong-answer risk for most embedding models.
    #[schemars(range(min = 0.0, max = 1.0))]
    pub threshold: f32,

    /// Upper bound on stored entries for this policy on the `memory`
    /// backend; the oldest entry is evicted first. Shared backends
    /// bound growth by TTL instead and ignore this value. The ceiling
    /// keeps the per-request similarity scan and the per-policy vector
    /// memory bounded; workloads needing more entries belong on a
    /// shared backend.
    #[serde(default = "default_semantic_max_entries")]
    #[schemars(range(min = 1, max = 10000))]
    pub max_entries: u32,

    /// Per-call deadline for the embedding request in milliseconds.
    /// `0` or absent disables the embedding-specific deadline. On
    /// timeout the request proceeds to the upstream uncached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_timeout_ms: Option<u64>,
}

impl SemanticCacheConfig {
    /// Per-call embedding deadline. Folds the `0`/absent sentinel into
    /// `None` so callers can apply it unconditionally.
    pub fn embedding_timeout(&self) -> Option<std::time::Duration> {
        self.embedding_timeout_ms
            .filter(|&ms| ms > 0)
            .map(std::time::Duration::from_millis)
    }
}

fn default_semantic_max_entries() -> u32 {
    1000
}

/// A prompt-response cache rule. Requests covered by an enabled policy
/// are served from cache when an identical request was answered before
/// (exact matching), and — when `semantic` is configured — when a
/// sufficiently similar request was.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct CachePolicy {
    /// Operator-facing name that surfaces in metric labels and cache headers.
    #[schemars(length(min = 1, max = 120))]
    pub name: String,

    /// When false, the cache gate skips this policy. Allows operators
    /// to stage a rule before enabling it.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Cache backend used for matching requests.
    #[serde(default)]
    pub backend: CacheBackend,

    /// Cache entry TTL in seconds.
    #[serde(default = "default_ttl_seconds")]
    #[schemars(range(min = 1, max = 604800))]
    pub ttl_seconds: u32,

    /// Free-form scope. Supports `"all"`, `"model:<name>"`, and
    /// `"api_key:<id>"`. See `parsed_applies_to`.
    #[serde(default = "default_applies_to")]
    #[schemars(length(min = 1, max = 255))]
    pub applies_to: String,

    /// Sharing boundary for entries created under this policy:
    /// `api_key` (default) keeps entries private to the caller that
    /// created them; `env` shares them across the environment.
    #[serde(default)]
    pub scope: CacheScope,

    /// Invalidation counter. Entries are readable only while their
    /// stored generation matches; a purge bumps this value, making
    /// every earlier entry unreachable at once. Managed by the purge
    /// operation — not set directly. Full-document updates must carry
    /// the current value forward: writing a lower (or omitted, i.e.
    /// `0`) value re-exposes entries stored under that earlier
    /// generation until their TTL passes.
    #[serde(default)]
    pub purge_generation: u32,

    /// Embedding-similarity matching. Absent: the policy matches
    /// exactly-identical requests only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic: Option<SemanticCacheConfig>,

    /// Set by the loader from the kine path's UUID segment. The DP
    /// uses this for metric labels and log correlation. Not part of
    /// the wire shape.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

fn default_enabled() -> bool {
    true
}

fn default_ttl_seconds() -> u32 {
    3600
}

fn default_applies_to() -> String {
    "all".to_string()
}

impl Resource for CachePolicy {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "cache_policies"
    }
}

impl CachePolicy {
    /// Set the runtime id (the kine path UUID). Used by the loader.
    pub fn with_runtime_id(mut self, id: impl Into<String>) -> Self {
        self.runtime_id = id.into();
        self
    }

    /// Parse `applies_to` into a typed matcher. Stage 3 understands:
    ///
    ///   - `"all"`            → matches every request in the env
    ///   - `"model:<name>"`   → matches requests targeting that model alias
    ///   - `"api_key:<id>"`   → matches requests authenticated by that api_key UUID
    ///
    /// Anything else (including the empty string) parses as `All` —
    /// the conservative default keeps caching on for legacy / future
    /// policy values rather than silently disabling them on a typo.
    /// cp-api validation prevents the empty-string case at write time
    /// (see internal/cpapi/resources/cache_policies.go::validateCachePolicyShape),
    /// so the conservative branch is dead in practice.
    pub fn parsed_applies_to(&self) -> AppliesTo {
        let raw = self.applies_to.trim();
        if let Some(rest) = raw.strip_prefix("model:") {
            return AppliesTo::Model(rest.trim().to_string());
        }
        if let Some(rest) = raw.strip_prefix("api_key:") {
            return AppliesTo::ApiKey(rest.trim().to_string());
        }
        AppliesTo::All
    }
}

/// Typed view of `CachePolicy::applies_to`. The proxy uses this to
/// pick the first matching enabled policy on every request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppliesTo {
    /// Every request in the env matches this policy.
    All,
    /// Only requests whose `req.model` equals the inner string match.
    /// String-compare against the model alias the caller asked for —
    /// router fan-out happens AFTER cache lookup, so the alias is the
    /// stable identifier here.
    Model(String),
    /// Only requests authenticated by the api_key whose UUID equals
    /// the inner string. The UUID is the cp-api row id, the same
    /// value the dashboard exposes on the api keys page.
    ApiKey(String),
}

impl AppliesTo {
    /// True iff this matcher accepts a request with the given
    /// (model, api_key_id) pair. The caller is responsible for
    /// passing the values it has at cache-lookup time — both are
    /// stable strings, so no heap allocation is needed beyond the
    /// references the proxy already holds.
    pub fn matches(&self, model: &str, api_key_id: &str) -> bool {
        match self {
            AppliesTo::All => true,
            AppliesTo::Model(want) => model == want,
            AppliesTo::ApiKey(want) => api_key_id == want,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserialises_minimal_memory_policy() {
        let v = json!({
            "name": "prod-default",
            "backend": "memory"
        });
        let p: CachePolicy = serde_json::from_value(v).unwrap();
        assert_eq!(p.name, "prod-default");
        assert!(p.enabled, "enabled defaults to true");
        assert_eq!(p.backend, CacheBackend::Memory);
        assert_eq!(p.ttl_seconds, 3600);
        assert_eq!(p.applies_to, "all");
    }

    #[test]
    fn deserialises_redis_policy_with_overrides() {
        let v = json!({
            "name": "shared-cluster",
            "enabled": false,
            "backend": "redis",
            "ttl_seconds": 600,
            "applies_to": "model:gpt-4o"
        });
        let p: CachePolicy = serde_json::from_value(v).unwrap();
        assert!(!p.enabled);
        assert_eq!(p.backend, CacheBackend::Redis);
        assert_eq!(p.ttl_seconds, 600);
        assert_eq!(p.applies_to, "model:gpt-4o");
    }

    #[test]
    fn resource_kind_matches_kine_path_segment() {
        assert_eq!(<CachePolicy as Resource>::kind(), "cache_policies");
    }

    #[test]
    fn runtime_id_round_trips_through_with_runtime_id() {
        let p: CachePolicy =
            serde_json::from_value(json!({"name": "x", "backend": "memory"})).unwrap();
        let p = p.with_runtime_id("uuid-1");
        assert_eq!(<CachePolicy as Resource>::id(&p), "uuid-1");
    }

    #[test]
    fn applies_to_all_matches_anything() {
        let p: CachePolicy =
            serde_json::from_value(json!({"name": "x", "applies_to": "all"})).unwrap();
        assert_eq!(p.parsed_applies_to(), AppliesTo::All);
        assert!(p.parsed_applies_to().matches("any-model", "any-key"));
    }

    #[test]
    fn applies_to_model_matches_only_named_model() {
        let p: CachePolicy =
            serde_json::from_value(json!({"name": "x", "applies_to": "model:gpt-4o"})).unwrap();
        assert_eq!(p.parsed_applies_to(), AppliesTo::Model("gpt-4o".into()));
        assert!(p.parsed_applies_to().matches("gpt-4o", "any-key"));
        assert!(!p.parsed_applies_to().matches("claude-3-opus", "any-key"));
    }

    #[test]
    fn applies_to_api_key_matches_only_named_key() {
        let kid = "11111111-1111-1111-1111-111111111111";
        let p: CachePolicy = serde_json::from_value(json!({
            "name": "x",
            "applies_to": format!("api_key:{kid}")
        }))
        .unwrap();
        assert_eq!(p.parsed_applies_to(), AppliesTo::ApiKey(kid.into()));
        assert!(p.parsed_applies_to().matches("gpt-4o", kid));
        assert!(!p.parsed_applies_to().matches("gpt-4o", "different-key-id"));
    }

    #[test]
    fn applies_to_unknown_prefix_falls_back_to_all() {
        // cp-api validation rejects this on write, but a hand-edited
        // kine row could surface here — we deliberately fall back to
        // All rather than disabling caching on an unknown discriminator.
        let p: CachePolicy =
            serde_json::from_value(json!({"name": "x", "applies_to": "team:eng"})).unwrap();
        assert_eq!(p.parsed_applies_to(), AppliesTo::All);
    }

    #[test]
    fn unknown_fields_are_tolerated_for_forward_compat() {
        // cp-api may ship new fields ahead of the DP rolling out;
        // serde must accept them (no `deny_unknown_fields`).
        let v = json!({
            "name": "future",
            "backend": "memory",
            "future_knob": "ignored"
        });
        let p: CachePolicy = serde_json::from_value(v).unwrap();
        assert_eq!(p.name, "future");
    }

    #[test]
    fn scope_and_generation_default_for_legacy_rows() {
        // Rows written before the semantic-cache release carry neither
        // field; they must parse with the documented defaults.
        let p: CachePolicy =
            serde_json::from_value(json!({"name": "old", "backend": "memory"})).unwrap();
        assert_eq!(p.scope, CacheScope::ApiKey);
        assert_eq!(p.purge_generation, 0);
        assert!(p.semantic.is_none());
    }

    #[test]
    fn deserialises_full_semantic_policy() {
        let v = json!({
            "name": "faq",
            "backend": "memory",
            "scope": "env",
            "purge_generation": 3,
            "semantic": {
                "embedding_model": "text-embedding-3-small",
                "threshold": 0.92,
                "embedding_timeout_ms": 2000
            }
        });
        let p: CachePolicy = serde_json::from_value(v).unwrap();
        assert_eq!(p.scope, CacheScope::Env);
        assert_eq!(p.purge_generation, 3);
        let sem = p.semantic.as_ref().unwrap();
        assert_eq!(sem.embedding_model, "text-embedding-3-small");
        assert!((sem.threshold - 0.92).abs() < 1e-6);
        assert_eq!(sem.max_entries, 1000, "max_entries defaults to 1000");
        assert_eq!(
            sem.embedding_timeout(),
            Some(std::time::Duration::from_millis(2000))
        );
    }

    #[test]
    fn semantic_block_tolerates_unknown_fields_for_forward_compat() {
        // Same forward-compat contract as the policy root: a future
        // knob inside `semantic` must not make this DP drop the row.
        let v = json!({
            "name": "faq",
            "semantic": {
                "embedding_model": "e",
                "threshold": 0.9,
                "future_knob": true
            }
        });
        let p: CachePolicy = serde_json::from_value(v).unwrap();
        assert!(p.semantic.is_some());
    }

    #[test]
    fn semantic_zero_timeout_means_no_deadline() {
        let sem: SemanticCacheConfig = serde_json::from_value(json!({
            "embedding_model": "e",
            "threshold": 0.9,
            "embedding_timeout_ms": 0
        }))
        .unwrap();
        assert_eq!(sem.embedding_timeout(), None);
    }

    #[test]
    fn semantic_threshold_is_required() {
        let v = json!({
            "name": "faq",
            "semantic": {"embedding_model": "e"}
        });
        assert!(serde_json::from_value::<CachePolicy>(v).is_err());
    }
}
