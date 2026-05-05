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

    /// Returns true when this policy's `applies_to` string targets the
    /// given (model name, api_key uuid) pair.
    ///
    /// Grammar:
    /// - `""` or `"all"` — matches every request
    /// - `"model:<display_name>"` — matches when the request's
    ///   resolved model name is exactly `<display_name>`
    /// - `"api_key:<uuid>"` — matches when the authenticated
    ///   ApiKey row's id is `<uuid>`
    /// - anything else — **does not match anything**
    ///
    /// The "unknown prefix → no match" default is deliberately strict.
    /// cp-api accepts any non-empty string today, so an operator who
    /// writes `applies_to = "production"` (a free-form label) would
    /// have silently been treated as "all" pre-Stage-3. Strict feedback
    /// (cache stays disabled, dashboard's `cache_status="disabled"`
    /// surfaces it) is safer than silently caching every request in
    /// the env.
    pub fn applies_to_request(&self, model_name: &str, api_key_id: &str) -> bool {
        let s = self.applies_to.as_str().trim();
        if s.is_empty() || s == "all" {
            return true;
        }
        if let Some(rest) = s.strip_prefix("model:") {
            return rest == model_name;
        }
        if let Some(rest) = s.strip_prefix("api_key:") {
            return rest == api_key_id;
        }
        false
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

    fn policy_with_applies_to(s: &str) -> CachePolicy {
        let mut p: CachePolicy =
            serde_json::from_value(json!({"name": "x", "backend": "memory"})).unwrap();
        p.applies_to = s.to_string();
        p
    }

    #[test]
    fn applies_to_all_matches_everything() {
        assert!(policy_with_applies_to("all").applies_to_request("gpt-4o", "ak-1"));
        assert!(policy_with_applies_to("").applies_to_request("anything", ""));
        // Whitespace-only also collapses to All — operators paste freely.
        assert!(policy_with_applies_to("   ").applies_to_request("gpt-4o", "ak-1"));
    }

    #[test]
    fn applies_to_model_matches_only_exact_name() {
        let p = policy_with_applies_to("model:gpt-4o");
        assert!(p.applies_to_request("gpt-4o", "any-key"));
        assert!(!p.applies_to_request("gpt-4o-mini", "any-key"));
        assert!(!p.applies_to_request("", "any-key"));
        // Case sensitive — Bedrock/OpenAI model ids are case-sensitive
        // upstream so we don't normalise here.
        assert!(!p.applies_to_request("GPT-4O", "any-key"));
    }

    #[test]
    fn applies_to_api_key_matches_only_exact_uuid() {
        let p = policy_with_applies_to("api_key:11111111-2222-3333-4444-555555555555");
        assert!(p.applies_to_request("any-model", "11111111-2222-3333-4444-555555555555"));
        assert!(!p.applies_to_request("any-model", "other-uuid"));
        assert!(!p.applies_to_request("any-model", ""));
    }

    #[test]
    fn applies_to_unknown_prefix_matches_nothing() {
        // Strict: free-form labels like "production" or "team:foo" no
        // longer accidentally cache every request in the env.
        // Operators see cache_status=disabled and learn to fix it.
        let p = policy_with_applies_to("production");
        assert!(!p.applies_to_request("gpt-4o", "ak-1"));

        let p = policy_with_applies_to("team:platform");
        assert!(!p.applies_to_request("gpt-4o", "ak-1"));

        let p = policy_with_applies_to("model:");
        // Empty model name: only matches a request with literally
        // empty model name (which the proxy rejects upstream anyway).
        assert!(p.applies_to_request("", "ak-1"));
        assert!(!p.applies_to_request("gpt-4o", "ak-1"));
    }
}
