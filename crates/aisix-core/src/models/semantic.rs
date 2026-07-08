//! Semantic-routing config attached to a [`Model`](super::Model).
//!
//! When a Model carries a `semantic` block, the proxy treats it as a
//! virtual router that picks a target by the *meaning* of the request,
//! rather than by health/weight ([`Routing`](super::Routing)) or by
//! fanning out to a panel ([`EnsembleConfig`](super::EnsembleConfig)).
//!
//! Per request the proxy embeds the latest user message via the
//! `embedding_model`, scores it against each route's example embeddings
//! (cosine, aggregated per route), and dispatches to the highest route
//! whose score clears its threshold — or to `default` when none does.
//! Route example vectors are computed once at apply time and cached, so
//! the per-request cost is a single embedding call plus local arithmetic.
//!
//! Routes reference direct Models by `display_name`, the same way routing
//! targets and ensemble panel members do. Mutual exclusivity with the
//! direct-upstream fields, `routing`, and `ensemble` is enforced by the
//! runtime schema (`super::schema`), not by this type.

use serde::{Deserialize, Serialize};

/// Distance metric used to compare the request embedding against route
/// example embeddings. v1 supports cosine only.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    /// Cosine similarity — higher is more similar.
    #[default]
    Cosine,
}

/// How a request's per-example similarity scores collapse into one score
/// for the route. v1 uses `max` (the single best-matching example),
/// chosen for explainability — the UI surfaces "the top matching example"
/// — over semantic-router's default sum/mean-over-top_k.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Aggregation {
    /// Route score = the highest cosine across its examples.
    #[default]
    Max,
}

/// One semantic route: a labeled set of example utterances whose
/// embeddings define the route. A request that scores high enough against
/// them dispatches to `target`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SemanticRoute {
    /// Operator-facing route label. Surfaced in the `x-aisix-route`
    /// response header and access logs (e.g. `prod-chat -> route:legal`).
    #[schemars(length(min = 1))]
    pub name: String,
    /// Direct model alias that receives traffic matching this route.
    #[schemars(length(min = 1))]
    pub target: String,
    /// Human-facing description. Documentation only — v1 matches on
    /// `examples`, not on this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub description: Option<String>,
    /// Example utterances that define this route. AISIX embeds each example
    /// when applying the configuration and caches the vector. A request is
    /// matched against these examples. At least one example is required.
    #[schemars(length(min = 1), inner(length(min = 1)))]
    pub examples: Vec<String>,
    /// Per-route similarity threshold. A request matches this route only
    /// when its aggregated score is `>=` this value. When omitted, the
    /// router-level threshold applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub threshold: Option<f32>,
}

/// Matching parameters shared across every route in a semantic router.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SemanticMatch {
    /// Similarity metric. v1: cosine.
    #[serde(default)]
    pub distance_metric: DistanceMetric,
    /// Per-example score aggregation. v1: max.
    #[serde(default)]
    pub aggregation: Aggregation,
    /// Default similarity threshold for routes that do not set their own
    /// `threshold`. Higher is stricter.
    #[schemars(range(min = 0.0, max = 1.0))]
    pub threshold: f32,
}

/// Mode for an enum-only failure policy: route to the router's `default`,
/// or fail the request with `503`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingFailureMode {
    /// Route to the semantic router's `default` model.
    #[default]
    Default,
    /// Reject the request with `503`.
    Fail,
}

/// What the router does when the embedding call errors or times out. Use
/// `"default"` to route to the router's default model, `"fail"` to reject the
/// request with `503`, or `{ "target": "<direct alias>" }` to route to a
/// specific fallback model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum OnEmbeddingFailure {
    /// `"default"` or `"fail"`.
    Mode(EmbeddingFailureMode),
    /// `{ "target": "<direct alias>" }` — route to a specific safe model.
    Target {
        /// Direct-model alias to route to when embedding fails.
        #[schemars(length(min = 1))]
        target: String,
    },
}

impl Default for OnEmbeddingFailure {
    fn default() -> Self {
        OnEmbeddingFailure::Mode(EmbeddingFailureMode::Default)
    }
}

/// Semantic-routing config: pick a target by request meaning.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Semantic {
    /// Alias of an `embedding`-modality Model used to embed the request
    /// and (at apply time) the route examples.
    #[schemars(length(min = 1))]
    pub embedding_model: String,
    /// Routes evaluated for each request. At least one is required.
    #[schemars(length(min = 1))]
    pub routes: Vec<SemanticRoute>,
    /// Direct model alias used when no route clears its threshold.
    #[schemars(length(min = 1))]
    pub default: String,
    /// Shared matching parameters (metric, aggregation, default threshold).
    pub r#match: SemanticMatch,
    /// Per-call deadline for the embedding request in milliseconds. `0` or
    /// absent disables the embedding-specific deadline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_timeout_ms: Option<u64>,
    /// Behavior when the embedding call fails or times out. Defaults to
    /// routing to `default`.
    #[serde(default, skip_serializing_if = "is_default_failure")]
    pub on_embedding_failure: OnEmbeddingFailure,
}

fn is_default_failure(p: &OnEmbeddingFailure) -> bool {
    *p == OnEmbeddingFailure::default()
}

impl Semantic {
    /// Effective threshold for a route: its own `threshold` if set,
    /// otherwise the router-level `match.threshold`.
    pub fn route_threshold(&self, route: &SemanticRoute) -> f32 {
        route.threshold.unwrap_or(self.r#match.threshold)
    }

    /// Per-call embedding deadline. Folds the `0`/absent sentinel into
    /// `None` like [`Model::request_timeout`](super::Model::request_timeout)
    /// so callers can apply it unconditionally.
    pub fn embedding_timeout(&self) -> Option<std::time::Duration> {
        self.embedding_timeout_ms
            .filter(|&ms| ms > 0)
            .map(std::time::Duration::from_millis)
    }

    /// Every direct-model alias this router can dispatch to: each route's
    /// `target` plus `default`. Used by the loader for reference-integrity
    /// checks and the runtime for resolution.
    pub fn referenced_targets(&self) -> impl Iterator<Item = &str> {
        self.routes
            .iter()
            .map(|r| r.target.as_str())
            .chain(std::iter::once(self.default.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_full_semantic_block() {
        let json = r#"{
            "embedding_model": "bge-m3",
            "routes": [
                {
                    "name": "legal",
                    "target": "claude-opus",
                    "description": "Contract & legal risk analysis",
                    "examples": ["分析这份合同里的潜在风险", "Review this NDA"],
                    "threshold": 0.8
                },
                {
                    "name": "translate",
                    "target": "gpt-4o-mini",
                    "examples": ["帮我翻译这句话"]
                }
            ],
            "default": "gpt-4o",
            "match": {"distance_metric": "cosine", "aggregation": "max", "threshold": 0.75},
            "embedding_timeout_ms": 500,
            "on_embedding_failure": {"target": "gpt-4o-mini"}
        }"#;
        let s: Semantic = serde_json::from_str(json).unwrap();
        assert_eq!(s.embedding_model, "bge-m3");
        assert_eq!(s.routes.len(), 2);
        assert_eq!(s.routes[0].name, "legal");
        assert_eq!(s.routes[0].target, "claude-opus");
        assert_eq!(s.routes[0].examples.len(), 2);
        assert_eq!(s.default, "gpt-4o");
        assert_eq!(s.r#match.threshold, 0.75);
        // Per-route override beats the router default; absent override
        // falls back to it.
        assert_eq!(s.route_threshold(&s.routes[0]), 0.8);
        assert_eq!(s.route_threshold(&s.routes[1]), 0.75);
        assert_eq!(
            s.embedding_timeout(),
            Some(std::time::Duration::from_millis(500))
        );
        assert_eq!(
            s.on_embedding_failure,
            OnEmbeddingFailure::Target {
                target: "gpt-4o-mini".into()
            }
        );
    }

    #[test]
    fn minimal_semantic_block_uses_defaults() {
        let s: Semantic = serde_json::from_str(
            r#"{
                "embedding_model": "bge-m3",
                "routes": [{"name": "a", "target": "m", "examples": ["hi"]}],
                "default": "fallback",
                "match": {"threshold": 0.5}
            }"#,
        )
        .unwrap();
        assert_eq!(s.r#match.distance_metric, DistanceMetric::Cosine);
        assert_eq!(s.r#match.aggregation, Aggregation::Max);
        assert_eq!(s.embedding_timeout(), None);
        // Absent on_embedding_failure defaults to routing to `default`.
        assert_eq!(s.on_embedding_failure, OnEmbeddingFailure::default());
        assert_eq!(
            s.on_embedding_failure,
            OnEmbeddingFailure::Mode(EmbeddingFailureMode::Default)
        );
    }

    #[test]
    fn on_embedding_failure_accepts_bare_string_modes() {
        for (raw, expected) in [
            ("\"default\"", EmbeddingFailureMode::Default),
            ("\"fail\"", EmbeddingFailureMode::Fail),
        ] {
            let p: OnEmbeddingFailure = serde_json::from_str(raw).unwrap();
            assert_eq!(p, OnEmbeddingFailure::Mode(expected));
        }
    }

    #[test]
    fn referenced_targets_lists_routes_then_default() {
        let s: Semantic = serde_json::from_str(
            r#"{
                "embedding_model": "e",
                "routes": [
                    {"name": "a", "target": "m1", "examples": ["x"]},
                    {"name": "b", "target": "m2", "examples": ["y"]}
                ],
                "default": "d",
                "match": {"threshold": 0.5}
            }"#,
        )
        .unwrap();
        let refs: Vec<&str> = s.referenced_targets().collect();
        assert_eq!(refs, vec!["m1", "m2", "d"]);
    }

    #[test]
    fn rejects_unknown_semantic_field() {
        let r: Result<Semantic, _> = serde_json::from_str(
            r#"{"embedding_model":"e","routes":[{"name":"a","target":"m","examples":["x"]}],"default":"d","match":{"threshold":0.5},"foo":1}"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn rejects_unknown_route_field() {
        let r: Result<SemanticRoute, _> =
            serde_json::from_str(r#"{"name":"a","target":"m","examples":["x"],"bogus":true}"#);
        assert!(r.is_err());
    }

    #[test]
    fn on_embedding_failure_serializes_to_proposal_wire_shape() {
        // Bare-string mode round-trips as a string.
        let mode = OnEmbeddingFailure::Mode(EmbeddingFailureMode::Fail);
        assert_eq!(
            serde_json::to_value(&mode).unwrap(),
            serde_json::json!("fail")
        );
        // Target round-trips as { "target": "alias" }, not a nested object.
        let target = OnEmbeddingFailure::Target {
            target: "safe".into(),
        };
        assert_eq!(
            serde_json::to_value(&target).unwrap(),
            serde_json::json!({"target": "safe"})
        );
    }
}
