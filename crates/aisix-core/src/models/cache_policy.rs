//! `CachePolicy` entity — per-env prompt-response cache rules. The
//! control plane (cp-api) writes these to etcd at
//! `/aisix/<env>/cache_policies/<uuid>`; the DP loads them on watch
//! and `aisix-proxy::cache_gate` consults them on every chat request.
//!
//! Stage 2 (this PR) honors only:
//!   - `enabled` — flag the policy on/off
//!   - existence of any matching policy enables / disables the cache
//!     for the request
//!
//! Stage 3+ extensions:
//!   - `applies_to` parsed into a real matcher (currently treated as
//!     "all" if any policy is present)
//!   - `ttl_seconds` propagated into the cache backend per entry
//!   - `backend` switching between memory / redis / redis_semantic
//!   - semantic-mode (`similarity_threshold` + `embedding_model`) once
//!     the embedding client + pgvector backend land
//!
//! See `crates/aisix-cache` for the cache backend itself; this module
//! is the wire shape only.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// Cache backend choice. Stage 2 only enforces `Memory`. The other
/// variants persist in cp-api + ship through kine but the DP falls
/// back to memory until each backend wires up.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheBackend {
    #[default]
    Memory,
    Redis,
    RedisSemantic,
    Qdrant,
}

/// Top-level `CachePolicy` resource shape. Mirrors what cp-api writes
/// to kine. `name` is operator-facing; `enabled` flips the policy on
/// without delete + recreate. `applies_to` is parsed by the cache
/// gate (Stage 3); for now any enabled policy is treated as
/// "applies to all chat completions in this env".
///
/// `deny_unknown_fields` is intentionally NOT set so cp-api can ship
/// new fields ahead of a DP rollout without a hard reject. New
/// optional fields land at `#[serde(default)]` here on the next DP
/// release.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CachePolicy {
    /// Operator-facing name; surfaces in metric labels + cache headers.
    pub name: String,

    /// When false the cache gate skips this policy. Lets operators
    /// stage a rule (write it, sanity-check it, then flip it on).
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Backend hint. Stage 2 enforces `memory` only; other variants
    /// fall back to memory at the DP and surface "configured but
    /// not yet enforced" in the dashboard.
    #[serde(default)]
    pub backend: CacheBackend,

    /// TTL hint in seconds. Stage 2 honors the cache backend's
    /// configured TTL globally; per-policy TTL lands in Stage 3.
    /// Default 3600 matches the cp-api validator.
    #[serde(default = "default_ttl_seconds")]
    pub ttl_seconds: u32,

    /// Free-form scope. v1 understands "all", "model:<name>",
    /// "api_key:<id>". Stage 2 treats any non-empty value as "all"
    /// — applies_to parsing lands in Stage 3.
    #[serde(default = "default_applies_to")]
    pub applies_to: String,

    /// Semantic-mode similarity floor. Required by cp-api for
    /// `redis_semantic` / `qdrant`; ignored by the DP until those
    /// backends wire up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity_threshold: Option<f32>,

    /// Semantic-mode embedding model. Same Stage-3-or-later note.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,

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
        assert!(p.similarity_threshold.is_none());
    }

    #[test]
    fn deserialises_full_semantic_policy() {
        let v = json!({
            "name": "semantic-experiment",
            "enabled": false,
            "backend": "redis_semantic",
            "ttl_seconds": 600,
            "applies_to": "model:gpt-4o",
            "similarity_threshold": 0.92,
            "embedding_model": "text-embedding-3-small"
        });
        let p: CachePolicy = serde_json::from_value(v).unwrap();
        assert!(!p.enabled);
        assert_eq!(p.backend, CacheBackend::RedisSemantic);
        assert_eq!(p.ttl_seconds, 600);
        assert_eq!(p.applies_to, "model:gpt-4o");
        assert_eq!(p.similarity_threshold, Some(0.92));
        assert_eq!(p.embedding_model.as_deref(), Some("text-embedding-3-small"));
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
