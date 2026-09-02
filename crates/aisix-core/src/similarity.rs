//! Vector-similarity primitives shared by every embedding-scored
//! feature.
//!
//! Semantic routing (`aisix-proxy`), the semantic cache and the
//! `kind: "semantic"` guardrail (`aisix-guardrails`) all score a request
//! vector against a set of prototype vectors the same way. Keeping ONE
//! implementation here is what stops the three from drifting apart on
//! the degenerate cases below — each of which silently mis-routes,
//! mis-caches or mis-screens rather than failing loudly.

/// Cosine similarity of two equal-length vectors. Returns `0.0` for a
/// length mismatch or a zero-magnitude vector, so a degenerate embedding
/// can never inject `NaN` into a `max` aggregation.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// The best (highest) cosine similarity between `probe` and any
/// prototype, or `None` when there are no prototypes.
///
/// `max` aggregation: a prototype SET scores as its single closest
/// member, so adding an unrelated example widens the set and never
/// dilutes the members already in it. Takes an iterator rather than a
/// slice because callers hold their vectors behind different wrappers
/// (`Vec<f32>` here, `Arc<Vec<f32>>` in the routing cache) and neither
/// should have to allocate a temporary to be scored.
///
/// A non-finite score is SKIPPED, not folded in. [`cosine_similarity`]
/// returns `NaN` when a component overflows f32 to infinity (`inf / inf`),
/// and every comparison against `NaN` is false — so a naive `current >=
/// score` fold lets one poisoned prototype evict a perfectly good score
/// that came before it. Skipping keeps the surviving prototypes'
/// verdict. `None` still means "no prototype scored", which now also
/// covers "every prototype scored non-finite".
pub fn best_similarity<'a>(
    probe: &[f32],
    prototypes: impl IntoIterator<Item = &'a [f32]>,
) -> Option<f32> {
    best_similarity_by(probe, prototypes.into_iter().map(|p| ((), p))).map(|(_, score)| score)
}

/// The item whose vector has the highest cosine similarity to `probe`.
///
/// This is the keyed form of [`best_similarity`]: callers that need the
/// winning prototype's identity (rather than only its score) carry any key
/// alongside each vector. The key can be an index, a cache-entry reference,
/// or another cheap handle; no temporary vector is required.
pub fn best_similarity_by<'a, K>(
    probe: &[f32],
    prototypes: impl IntoIterator<Item = (K, &'a [f32])>,
) -> Option<(K, f32)> {
    let mut best: Option<(K, f32)> = None;
    for (key, p) in prototypes {
        let score = cosine_similarity(probe, p);
        if !score.is_finite() {
            continue;
        }
        if best.as_ref().is_none_or(|(_, current)| score > *current) {
            best = Some((key, score));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one_orthogonal_is_zero() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_degenerate_inputs_are_zero_not_nan() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0]), 0.0);
        assert!(!cosine_similarity(&[0.0], &[0.0]).is_nan());
    }

    #[test]
    fn cosine_is_scale_invariant() {
        let s = cosine_similarity(&[1.0, 1.0], &[5.0, 5.0]);
        assert!((s - 1.0).abs() < 1e-6);
    }

    fn slices(v: &[Vec<f32>]) -> impl Iterator<Item = &[f32]> {
        v.iter().map(Vec::as_slice)
    }

    #[test]
    fn best_similarity_takes_the_closest_prototype() {
        let protos = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let got = best_similarity(&[1.0, 0.0], slices(&protos)).unwrap();
        assert!((got - 1.0).abs() < 1e-6);
    }

    #[test]
    fn best_similarity_by_returns_the_closest_key() {
        let protos = [vec![0.0, 1.0], vec![1.0, 0.0]];
        let got = best_similarity_by(
            &[1.0, 0.0],
            protos
                .iter()
                .enumerate()
                .map(|(index, vector)| (index, vector.as_slice())),
        )
        .unwrap();
        assert_eq!(got.0, 1);
        assert!((got.1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn best_similarity_is_none_without_prototypes() {
        let protos: Vec<Vec<f32>> = Vec::new();
        assert_eq!(best_similarity(&[1.0, 0.0], slices(&protos)), None);
    }

    #[test]
    fn best_similarity_skips_a_poisoned_prototype() {
        // A prototype carrying a component that overflowed f32 to
        // infinity (a provider magnitude f32 cannot hold) scores
        // `inf / inf` = NaN. It must not evict the exact match that came
        // before it — the naive `current >= score` fold did exactly that,
        // because every comparison against NaN is false.
        let protos = vec![vec![1.0, 0.0], vec![f32::INFINITY, 0.0]];
        assert!(cosine_similarity(&[1.0, 0.0], &protos[1]).is_nan());
        let got = best_similarity(&[1.0, 0.0], slices(&protos)).unwrap();
        assert!(
            (got - 1.0).abs() < 1e-6,
            "poisoned prototype evicted the match: {got}"
        );
    }

    #[test]
    fn best_similarity_is_none_when_every_prototype_is_poisoned() {
        let protos = vec![vec![f32::INFINITY, 0.0]];
        assert_eq!(best_similarity(&[1.0, 0.0], slices(&protos)), None);
    }

    #[test]
    fn best_similarity_never_returns_neg_infinity() {
        // Every prototype is degenerate → each scores 0.0, and the fold
        // seed must not survive as a score.
        let protos = vec![vec![0.0, 0.0]];
        assert_eq!(best_similarity(&[1.0, 0.0], slices(&protos)), Some(0.0));
    }
}
