//! `CachePolicy` entity — per-env prompt-response cache rules. The
//! control plane (cp-api) writes these to etcd at
//! `/aisix/<env>/cache_policies/<uuid>`; the DP loads them on watch
//! and `aisix-proxy::cache_gate` consults them on every chat request.
//!
//! Backends supported: `memory` + `redis`. Semantic backends were
//! removed pending DP-side wiring of the embedding client +
//! chat-dispatch integration — see ai-gateway issue #116 and the
//! TODO issue tracking re-introduction.
//!
//! See `crates/aisix-cache` for the cache backend itself; this module
//! is the wire shape only.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// Cache backend choice. The DP selects the cache instance per
/// matched policy: `Memory` uses the in-process cache (always
/// available); `Redis` uses the shared redis cache iff the deployment
/// configured `cache.redis`. A `Redis` policy on a DP without redis
/// gets NO caching (`cache_status = disabled`) — never a silent
/// fallback to node-local memory.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CacheBackend {
    #[default]
    Memory,
    Redis,
}

/// Top-level `CachePolicy` resource shape. Mirrors what cp-api writes
/// to kine. `name` is operator-facing; `enabled` flips the policy on
/// without delete + recreate. `applies_to` is parsed into a typed
/// matcher (see `parsed_applies_to`).
///
/// `deny_unknown_fields` is intentionally NOT set so cp-api can ship
/// new fields ahead of a DP rollout without a hard reject. New
/// optional fields land at `#[serde(default)]` here on the next DP
/// release.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct CachePolicy {
    /// Operator-facing name; surfaces in metric labels + cache headers.
    pub name: String,

    /// When false the cache gate skips this policy. Lets operators
    /// stage a rule (write it, sanity-check it, then flip it on).
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Which cache instance serves requests matched by this policy.
    /// `memory` always works; `redis` requires the DP to have
    /// `cache.redis` configured — otherwise matching requests get no
    /// caching at all (visible as `cache_status = disabled`).
    #[serde(default)]
    pub backend: CacheBackend,

    /// TTL hint in seconds. Per-policy TTL is honored by the cache
    /// backend on each entry. Default 3600 matches the cp-api
    /// validator.
    #[serde(default = "default_ttl_seconds")]
    pub ttl_seconds: u32,

    /// Free-form scope. v1 understands "all", "model:<name>",
    /// "api_key:<id>". See `parsed_applies_to`.
    #[serde(default = "default_applies_to")]
    pub applies_to: String,

    /// Set by the loader from the kine path's UUID segment. The DP
    /// uses this for metric labels + log correlation; not part of
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
}
