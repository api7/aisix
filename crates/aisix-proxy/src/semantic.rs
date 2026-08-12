//! Semantic-routing runtime: embed the request, score it against each
//! route's cached example embeddings, and resolve a single direct-model
//! target (or the router's `default`).
//!
//! [`resolve`] produces a one-element `attempt_models` list that the
//! existing chat dispatch loop then drives exactly like a routing target —
//! so semantic routing reuses all of the streaming / failover / telemetry
//! machinery and only adds the "which target" decision on top.
//!
//! The scoring core ([`cosine_similarity`], [`decide`],
//! [`embedding_failure_target`]) and the example-vector cache
//! ([`SemanticVectorCache`]) are pure and unit-tested in isolation; the
//! async embedding call lives in [`resolve`].

use std::sync::Arc;

use dashmap::DashMap;

use aisix_core::models::{EmbeddingFailureMode, OnEmbeddingFailure, Semantic};
use aisix_core::resource::ResourceEntry;
use aisix_core::{AisixSnapshot, Model};
use aisix_gateway::{EmbeddingRequest, EmbeddingVector};

use crate::error::ProxyError;
use crate::routing::AttemptModel;
use crate::state::ProxyState;

/// Cosine similarity of two equal-length vectors. Returns `0.0` for a
/// length mismatch or a zero-magnitude vector, so a degenerate embedding
/// can never inject `NaN` into the `max` aggregation.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Outcome of scoring a request against a semantic router.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RouteDecision {
    /// Index into `Semantic::routes` of the winning route, or `None` when
    /// no route cleared its threshold (the caller falls back to `default`).
    pub winner: Option<usize>,
    /// Per-route aggregated score, aligned with `Semantic::routes`.
    pub scores: Vec<f32>,
}

/// Score `request_vec` against each route's example vectors and pick the
/// highest-scoring route that clears its (per-route or router-level)
/// threshold. `max` aggregation: a route's score is its single
/// best-matching example. `route_example_vecs[i]` are the cached vectors
/// for `semantic.routes[i]`.
pub(crate) fn decide(
    semantic: &Semantic,
    request_vec: &[f32],
    route_example_vecs: &[Vec<Arc<Vec<f32>>>],
) -> RouteDecision {
    let mut scores = Vec::with_capacity(semantic.routes.len());
    for ex_vecs in route_example_vecs {
        let best = ex_vecs
            .iter()
            .map(|v| cosine_similarity(request_vec, v))
            .fold(f32::NEG_INFINITY, f32::max);
        scores.push(if best.is_finite() { best } else { 0.0 });
    }
    let mut winner: Option<usize> = None;
    let mut best = f32::NEG_INFINITY;
    for (i, route) in semantic.routes.iter().enumerate() {
        if scores[i] >= semantic.route_threshold(route) && scores[i] > best {
            best = scores[i];
            winner = Some(i);
        }
    }
    RouteDecision { winner, scores }
}

/// Direct-model alias to dispatch to when the embedding call fails, per
/// `on_embedding_failure`. `None` means the policy is `fail` — the caller
/// returns `503`.
pub(crate) fn embedding_failure_target(semantic: &Semantic) -> Option<&str> {
    match &semantic.on_embedding_failure {
        OnEmbeddingFailure::Mode(EmbeddingFailureMode::Default) => Some(&semantic.default),
        OnEmbeddingFailure::Mode(EmbeddingFailureMode::Fail) => None,
        OnEmbeddingFailure::Target { target } => Some(target),
    }
}

/// Per-instance cache of route example-utterance embeddings, populated
/// lazily on the first request that needs them and reused across requests
/// so the steady-state per-request cost is a single embedding call for the
/// prompt. Keyed by `(embedding_model_id, dimensions, example_text)` —
/// changing the embedding model or its dimensions auto-invalidates, since
/// a stale vector of the wrong dimension must never be served.
#[derive(Debug, Default)]
pub struct SemanticVectorCache {
    vectors: DashMap<(String, u32, String), Arc<Vec<f32>>>,
}

impl SemanticVectorCache {
    pub(crate) fn get(
        &self,
        embedding_model_id: &str,
        dimensions: u32,
        text: &str,
    ) -> Option<Arc<Vec<f32>>> {
        self.vectors
            .get(&(embedding_model_id.to_string(), dimensions, text.to_string()))
            .map(|e| e.clone())
    }

    pub(crate) fn insert(
        &self,
        embedding_model_id: &str,
        dimensions: u32,
        text: &str,
        vec: Arc<Vec<f32>>,
    ) {
        self.vectors.insert(
            (embedding_model_id.to_string(), dimensions, text.to_string()),
            vec,
        );
    }
}

/// Resolve a semantic router to a single direct-model attempt + the name
/// of the route that matched (`None` when the request fell through to
/// `default`). The returned `Vec<AttemptModel>` always has exactly one
/// element; the chat dispatch loop drives it like any routing target.
pub(crate) async fn resolve(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    router_entry: &ResourceEntry<Model>,
    prompt: &str,
    source_ip: &str,
    request_id: &str,
) -> Result<(Vec<AttemptModel>, Option<String>), ProxyError> {
    let semantic = router_entry
        .value
        .semantic
        .as_ref()
        .expect("resolve called on a non-semantic model");
    let router = &router_entry.value;

    // No user text to classify (e.g. a system-only or tool-only request):
    // route to `default` without an embedding call rather than embedding an
    // empty string, which could spuriously match a route.
    if prompt.trim().is_empty() {
        let (attempt, _) =
            select_eligible(state, snapshot, router, source_ip, &semantic.default, None)?;
        return Ok((vec![attempt], None));
    }

    // Resolve the embedding model + its modality metadata. A dangling or
    // wrong-kind reference is a config error; degrade via the failure
    // policy rather than 500.
    let embed_entry = match snapshot.models.get_by_name(&semantic.embedding_model) {
        Some(e) if e.value.is_embedding() => e,
        other => {
            tracing::warn!(
                router = %router_entry.value.display_name,
                embedding_model = %semantic.embedding_model,
                found = other.is_some(),
                "semantic router references a missing or non-embedding embedding_model; \
                 applying on_embedding_failure",
            );
            return fallback(state, snapshot, router, source_ip, semantic);
        }
    };
    let dims = embed_entry
        .value
        .embedding
        .as_ref()
        .map(|e| e.dimensions)
        .unwrap_or(0);

    // Batch the prompt (index 0) with every uncached example text in one
    // embedding call. Steady state: only the prompt is uncached.
    let mut pending: Vec<String> = Vec::new();
    for route in &semantic.routes {
        for ex in &route.examples {
            if state
                .semantic_cache
                .get(&embed_entry.id, dims, ex)
                .is_none()
                && !pending.iter().any(|p| p == ex)
            {
                pending.push(ex.clone());
            }
        }
    }
    let mut to_embed: Vec<String> = Vec::with_capacity(1 + pending.len());
    to_embed.push(prompt.to_string());
    to_embed.extend(pending.iter().cloned());

    // Deadline chain for the embed sub-call: the router-level knob wins,
    // then the embedding model's own `timeout`, then the deployment
    // default — previously only the router knob applied, so a hung
    // embedding upstream stalled every semantic request unbounded.
    let embed_deadline = semantic.embedding_timeout().or_else(|| {
        crate::routing::effective_timeouts(&embed_entry.value, None, state.default_timeouts).request
    });
    let vectors = match embed_texts(
        state,
        snapshot,
        &embed_entry,
        embed_deadline,
        request_id,
        &to_embed,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                router = %router_entry.value.display_name,
                error = %e,
                "semantic embedding call failed; applying on_embedding_failure",
            );
            return fallback(state, snapshot, router, source_ip, semantic);
        }
    };

    let mut iter = vectors.into_iter();
    let prompt_vec = iter
        .next()
        .expect("embed_texts returns at least the prompt vector");
    for text in &pending {
        if let Some(v) = iter.next() {
            state
                .semantic_cache
                .insert(&embed_entry.id, dims, text, Arc::new(v));
        }
    }

    let route_vecs: Vec<Vec<Arc<Vec<f32>>>> = semantic
        .routes
        .iter()
        .map(|route| {
            route
                .examples
                .iter()
                .filter_map(|ex| state.semantic_cache.get(&embed_entry.id, dims, ex))
                .collect()
        })
        .collect();

    let decision = decide(semantic, &prompt_vec, &route_vecs);
    let (attempt, route_name): (AttemptModel, Option<String>) = match decision.winner {
        Some(i) => {
            let (attempt, fell_back) = select_eligible(
                state,
                snapshot,
                router,
                source_ip,
                semantic.routes[i].target.as_str(),
                Some(semantic.default.as_str()),
            )?;
            // `x-aisix-route` reports the route that actually served the
            // request: a winner displaced by its target's gates is a
            // fall-through to `default`, not a served route.
            let name = (!fell_back).then(|| semantic.routes[i].name.clone());
            (attempt, name)
        }
        None => {
            let (attempt, _) = select_eligible(
                state,
                snapshot,
                router,
                source_ip,
                semantic.default.as_str(),
                None,
            )?;
            (attempt, None)
        }
    };
    tracing::debug!(
        router = %router_entry.value.display_name,
        resolved_route = ?route_name,
        target = %attempt.model.display_name,
        "semantic routing decision",
    );
    Ok((vec![attempt], route_name))
}

/// Apply the `on_embedding_failure` policy: route to the fallback target,
/// or surface `503` when the policy is `fail`.
fn fallback(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    router: &Model,
    source_ip: &str,
    semantic: &Semantic,
) -> Result<(Vec<AttemptModel>, Option<String>), ProxyError> {
    match embedding_failure_target(semantic) {
        Some(alias) => {
            let (attempt, _) = select_eligible(state, snapshot, router, source_ip, alias, None)?;
            Ok((vec![attempt], None))
        }
        None => Err(ProxyError::ProviderUnavailable),
    }
}

/// Resolve a direct-model alias to the single `AttemptModel` the dispatch
/// loop will drive.
fn attempt_for_target(snapshot: &AisixSnapshot, alias: &str) -> Result<AttemptModel, ProxyError> {
    let entry = snapshot.models.get_by_name(alias).ok_or_else(|| {
        ProxyError::InvalidRequest(format!(
            "semantic router target {alias:?} does not resolve to a known model",
        ))
    })?;
    Ok(AttemptModel {
        id: entry.id.clone(),
        model: entry.value.clone(),
    })
}

/// Pick the dispatchable target among `{primary, fallback}` — the
/// semantic single-winner analogue of the group filter
/// (`routing::targets_allowed_for_ip` + `filter_attempt_models`):
///
/// - the member's client-IP allowlist is a HARD gate: an excluded
///   candidate is dropped, and with every candidate excluded the caller
///   gets the same 403 as the entry gate, naming only the alias they
///   addressed (`routing.rs` does the same for all-excluded groups, so
///   a router never becomes a probe for which members exist);
/// - member health (request-path cooldown / background-unhealthy) is a
///   SOFT preference: prefer an available candidate, but with none
///   available dispatch the primary anyway — the router has no
///   `when_all_unavailable` knob, and the semantically-right target
///   beats failing the request on an advisory signal.
///
/// Returns the attempt plus whether the fallback displaced the primary
/// (so the caller can clear the served-route attribution).
fn select_eligible(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    router: &Model,
    source_ip: &str,
    primary: &str,
    fallback_alias: Option<&str>,
) -> Result<(AttemptModel, bool), ProxyError> {
    let mut candidates: Vec<(AttemptModel, bool)> = Vec::with_capacity(2);
    candidates.push((attempt_for_target(snapshot, primary)?, false));
    if let Some(alias) = fallback_alias {
        // A dangling `default` only matters when it must actually serve;
        // as a mere fallback candidate it is skipped, not fatal.
        if alias != primary {
            if let Ok(attempt) = attempt_for_target(snapshot, alias) {
                candidates.push((attempt, true));
            }
        }
    }
    candidates.retain(|(attempt, fell_back)| {
        let allowed = attempt.model.ip_allowed(source_ip);
        if !allowed {
            tracing::debug!(
                router = %router.display_name,
                target = %attempt.model.display_name,
                fallback = fell_back,
                "semantic target excluded by its allowed_cidrs",
            );
        }
        allowed
    });
    if candidates.is_empty() {
        return Err(ProxyError::ModelIpRestricted(router.display_name.clone()));
    }
    let picked = candidates
        .iter()
        .position(|(attempt, _)| {
            let stale_after = attempt
                .model
                .background_model_check
                .as_ref()
                .map(|cfg| std::time::Duration::from_secs(cfg.stale_after_seconds));
            matches!(
                state
                    .runtime_status
                    .status_with_stale(&attempt.id, stale_after)
                    .status,
                crate::RuntimeStatus::Healthy | crate::RuntimeStatus::NotApplicable
            )
        })
        .unwrap_or(0);
    let (attempt, fell_back) = candidates.swap_remove(picked);
    if fell_back {
        tracing::debug!(
            router = %router.display_name,
            served = %attempt.model.display_name,
            "semantic route target unavailable; serving default",
        );
    }
    Ok((attempt, fell_back))
}

/// Embed `texts` through the embedding model's bridge in one batched call,
/// returning one float vector per input in input order. Shared by the
/// semantic router (route matching) and the cache gate's semantic layer
/// (`aisix-proxy::chat`), which is why the deadline arrives as a plain
/// `Option<Duration>` rather than either feature's config type.
pub(crate) async fn embed_texts(
    state: &ProxyState,
    snapshot: &AisixSnapshot,
    embed_entry: &ResourceEntry<Model>,
    timeout: Option<std::time::Duration>,
    request_id: &str,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, ProxyError> {
    let model = &embed_entry.value;
    crate::dispatch::require_provider(model)?;
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
    let bridge = crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value)
        .ok_or(ProxyError::ProviderUnavailable)?;
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();
    let dimensions = model.embedding.as_ref().map(|e| e.dimensions);

    let req = EmbeddingRequest {
        model: upstream_model,
        input: texts.to_vec(),
        input_was_single: texts.len() == 1,
        encoding_format: Some("float".to_string()),
        dimensions,
    };
    let ctx = {
        let base = crate::dispatch::bridge_ctx(
            request_id,
            &embed_entry.id,
            Arc::new(model.clone()),
            &pk_entry.id,
            Arc::new(pk_entry.value.clone()),
            None,
        );
        match timeout {
            Some(d) => base.with_deadline(d),
            None => base,
        }
    };

    let resp = bridge.embed(&req, &ctx).await.map_err(ProxyError::Bridge)?;
    let mut data = resp.data;
    data.sort_by_key(|d| d.index);
    let expected_dims = dimensions.map(|d| d as usize);
    let mut out = Vec::with_capacity(data.len());
    for obj in data {
        match obj.embedding {
            EmbeddingVector::Float(v) => {
                // Surface a wrong-dimension response explicitly instead of
                // letting cosine_similarity fold the length mismatch into
                // 0.0 (which would silently route every request to default).
                if let Some(expected) = expected_dims {
                    if v.len() != expected {
                        return Err(ProxyError::InvalidRequest(format!(
                            "embedding endpoint returned a {}-dim vector; expected {expected}",
                            v.len(),
                        )));
                    }
                }
                out.push(v);
            }
            EmbeddingVector::Base64(_) => {
                return Err(ProxyError::InvalidRequest(
                    "embedding endpoint returned base64 vectors; semantic routing needs \
                     encoding_format=float support"
                        .into(),
                ));
            }
        }
    }
    if out.len() != texts.len() {
        return Err(ProxyError::InvalidRequest(format!(
            "embedding endpoint returned {} vectors for {} inputs",
            out.len(),
            texts.len(),
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semantic(json: &str) -> Semantic {
        serde_json::from_str(json).unwrap()
    }

    fn arc(v: Vec<f32>) -> Arc<Vec<f32>> {
        Arc::new(v)
    }

    #[test]
    fn cosine_identical_is_one_orthogonal_is_zero() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        // Opposite direction → -1.
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_degenerate_inputs_are_zero_not_nan() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0]), 0.0); // length mismatch
        assert!(!cosine_similarity(&[0.0], &[0.0]).is_nan());
    }

    #[test]
    fn cosine_is_scale_invariant() {
        // Same direction, different magnitude → still ~1.
        let s = cosine_similarity(&[1.0, 1.0], &[5.0, 5.0]);
        assert!((s - 1.0).abs() < 1e-6, "expected ~1, got {s}");
    }

    fn router() -> Semantic {
        semantic(
            r#"{
                "embedding_model": "bge-m3",
                "routes": [
                    {"name": "legal", "target": "opus", "examples": ["a"], "threshold": 0.8},
                    {"name": "code",  "target": "sonnet", "examples": ["b"]}
                ],
                "default": "gpt-4o",
                "match": {"threshold": 0.5}
            }"#,
        )
    }

    #[test]
    fn decide_picks_highest_route_clearing_its_threshold() {
        let s = router();
        let req = vec![1.0, 0.0];
        // legal example == request (cos 1.0 ≥ 0.8 ✓); code example orthogonal (0.0 < 0.5).
        let route_vecs = vec![vec![arc(vec![1.0, 0.0])], vec![arc(vec![0.0, 1.0])]];
        let d = decide(&s, &req, &route_vecs);
        assert_eq!(d.winner, Some(0));
        assert!((d.scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decide_falls_through_to_default_when_none_clear() {
        let s = router();
        let req = vec![1.0, 0.0];
        // Both examples weakly aligned: legal 0.6 (< 0.8 own threshold),
        // code 0.4 (< 0.5 router threshold) → no winner.
        let route_vecs = vec![
            vec![arc(vec![0.6, 0.8])],   // cos ≈ 0.6
            vec![arc(vec![0.4, 0.917])], // cos ≈ 0.4
        ];
        let d = decide(&s, &req, &route_vecs);
        assert_eq!(d.winner, None);
    }

    #[test]
    fn decide_uses_router_threshold_when_route_has_no_override() {
        let s = router();
        let req = vec![1.0, 0.0];
        // code route (no override → 0.5): give it cos ≈ 0.71 (clears 0.5).
        // legal route (0.8): give it cos ≈ 0.71 (does NOT clear 0.8).
        let route_vecs = vec![vec![arc(vec![1.0, 1.0])], vec![arc(vec![1.0, 1.0])]];
        let d = decide(&s, &req, &route_vecs);
        assert_eq!(
            d.winner,
            Some(1),
            "only the code route clears its threshold"
        );
    }

    #[test]
    fn decide_max_aggregation_takes_best_example() {
        let s = router();
        let req = vec![1.0, 0.0];
        // legal has two examples: one orthogonal (0.0), one identical (1.0).
        // max → 1.0 clears 0.8.
        let route_vecs = vec![
            vec![arc(vec![0.0, 1.0]), arc(vec![1.0, 0.0])],
            vec![arc(vec![0.0, 1.0])],
        ];
        let d = decide(&s, &req, &route_vecs);
        assert_eq!(d.winner, Some(0));
        assert!((d.scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn embedding_failure_target_maps_each_policy() {
        let default_policy = router();
        assert_eq!(embedding_failure_target(&default_policy), Some("gpt-4o"));

        let fail = semantic(
            r#"{"embedding_model":"e","routes":[{"name":"a","target":"m","examples":["x"]}],
                "default":"d","match":{"threshold":0.5},"on_embedding_failure":"fail"}"#,
        );
        assert_eq!(embedding_failure_target(&fail), None);

        let target = semantic(
            r#"{"embedding_model":"e","routes":[{"name":"a","target":"m","examples":["x"]}],
                "default":"d","match":{"threshold":0.5},"on_embedding_failure":{"target":"safe"}}"#,
        );
        assert_eq!(embedding_failure_target(&target), Some("safe"));
    }

    #[test]
    fn cache_round_trips_and_dimension_change_invalidates() {
        let cache = SemanticVectorCache::default();
        assert!(cache.get("bge", 1024, "hello").is_none());
        cache.insert("bge", 1024, "hello", arc(vec![1.0, 2.0]));
        assert_eq!(
            cache.get("bge", 1024, "hello").unwrap().as_slice(),
            &[1.0, 2.0]
        );
        // Different dimensions → different key → miss (auto-invalidation).
        assert!(cache.get("bge", 512, "hello").is_none());
        // Different embedding model → miss.
        assert!(cache.get("other", 1024, "hello").is_none());
    }
}
