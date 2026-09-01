//! Per-request collection of ENFORCE-mode guardrail hits
//! (AISIX-Cloud#1330 audit chain).
//!
//! Enforcing outcomes were, until this module, visible only as Prometheus
//! aggregates (`GuardrailMetricsSink`, AISIX-Cloud#1076) and — for blocks —
//! as an operator-facing error string. Neither answers the per-request
//! audit question: *which configured policy rewrote or refused THIS
//! request*. Monitor mode already answered it via
//! `UsageEvent::guardrail_monitor_hits`; enforcing mode did not.
//!
//! [`LiveGuardrailIndex::resolve`](crate::LiveGuardrailIndex::resolve)
//! mints one log per resolved chain — and a chain is resolved once per
//! request — so the log's lifetime is the request's. The handler reads it
//! back with [`GuardrailChain::enforced_hits`](crate::GuardrailChain::enforced_hits)
//! when it builds the telemetry event.
//!
//! The same log carries the similarity summaries a `kind: semantic`
//! guardrail produces (AISIX-Cloud#1467). They ride here rather than in a
//! second per-request object because the handle is already threaded to
//! every point that builds a usage event, and a score is recorded on the
//! same executions this log already sees — including the ones that allowed.
//!
//! Names, counts and indices only: the matched value, the block reason, the
//! screened text and the example text never enter this log (#153 / #932
//! no-leak criterion).

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use aisix_core::models::{GuardrailEnforcedHit, GuardrailScore};

/// Coalescing key: one entry per guardrail row, per hook, per action —
/// and, for the one action that carries a cause, per cause. The cause is
/// part of the key rather than merged in because two different causes are
/// two different facts; in practice a member refuses at most once per
/// hook, so the extra dimension adds no entries.
type HitKey = (String, &'static str, &'static str, String);

/// Coalescing key for similarity scores: one entry per guardrail row, per
/// hook, per example list. Deliberately NOT keyed by screened text — the
/// point of the score entry is the closest call across a request, not a
/// row per message (see [`GuardrailScore`]).
type ScoreKey = (String, &'static str, &'static str);

/// Per-request accumulator of enforced guardrail hits and similarity
/// scores.
///
/// Coalescing matters for correctness, not just tidiness: the redacting
/// hooks run once per JSON string leaf on the `/mcp` path, so a tool
/// result with forty scannable leaves would otherwise produce forty
/// identical entries. Entries merge by
/// `(guardrail_name, hook, action, error_type)`, summing per-detector
/// counts and elapsed time.
#[derive(Debug, Default)]
pub struct GuardrailAuditLog {
    hits: Mutex<BTreeMap<HitKey, GuardrailEnforcedHit>>,
    /// Similarity screening summaries (AISIX-Cloud#1467), merged by
    /// `(guardrail_name, hook, direction)`. A separate map rather than a
    /// second action on `hits`: a score is recorded on every execution,
    /// including the ones that allowed, so folding it into the enforced-hit
    /// array would silently widen a field whose whole meaning is "a policy
    /// acted here".
    scores: Mutex<BTreeMap<ScoreKey, GuardrailScore>>,
}

impl GuardrailAuditLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one enforcing outcome. `counts` is empty for both refusal
    /// actions. `error_type` is the bounded cause carried by
    /// `blocked_unavailable` and `None` everywhere else
    /// (AISIX-Cloud#1365).
    ///
    /// A poisoned lock is swallowed rather than propagated: losing an
    /// audit entry is bad, but panicking a request that a guardrail just
    /// allowed through is worse.
    pub fn record(
        &self,
        guardrail_name: &str,
        hook: &'static str,
        action: &'static str,
        error_type: Option<&str>,
        elapsed: Duration,
        counts: &BTreeMap<String, u32>,
    ) {
        let Ok(mut hits) = self.hits.lock() else {
            return;
        };
        let error_type = error_type.unwrap_or_default();
        let entry = hits
            .entry((
                guardrail_name.to_owned(),
                hook,
                action,
                error_type.to_owned(),
            ))
            .or_insert_with(|| GuardrailEnforcedHit {
                guardrail_name: guardrail_name.to_owned(),
                hook: hook.to_owned(),
                action: action.to_owned(),
                error_type: error_type.to_owned(),
                ..Default::default()
            });
        for (detector, n) in counts {
            let slot = entry.counts.entry(detector.clone()).or_insert(0);
            *slot = slot.saturating_add(*n);
        }
        entry.duration_us = entry
            .duration_us
            .saturating_add(elapsed.as_micros().min(u32::MAX as u128) as u32);
    }

    /// Record one similarity-screening summary. Repeat calls for the same
    /// `(guardrail_name, hook, direction)` — a `/mcp` tool result scanned
    /// leaf by leaf, a response checked once from cache and once from the
    /// upstream — keep the value CLOSEST to changing the verdict, which is
    /// the maximum on `deny` and the minimum on `allow`. Keeping the last
    /// one instead would make the reported number depend on scan order.
    ///
    /// A poisoned lock is swallowed for the same reason as [`Self::record`].
    pub fn record_score(&self, score: GuardrailScore) {
        let Ok(mut scores) = self.scores.lock() else {
            return;
        };
        let key = (
            score.guardrail_name.clone(),
            hook_key(&score.hook),
            direction_key(&score.direction),
        );
        match scores.entry(key) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(score);
            }
            std::collections::btree_map::Entry::Occupied(mut slot) => {
                let keep = if score.direction == DIRECTION_ALLOW {
                    score.score < slot.get().score
                } else {
                    score.score > slot.get().score
                };
                if keep {
                    slot.insert(score);
                }
            }
        }
    }

    /// Snapshot the similarity scores recorded so far, in
    /// `(name, hook, direction)` order. Non-destructive, for the same
    /// reason as [`Self::snapshot`].
    pub fn score_snapshot(&self) -> Vec<GuardrailScore> {
        self.scores
            .lock()
            .map(|scores| scores.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Snapshot the hits recorded so far, in
    /// `(name, hook, action, error_type)` order.
    ///
    /// Non-destructive on purpose: a handler that emits more than one
    /// usage event per request (a per-attempt event plus the terminal
    /// one) must not have the first read silently empty the log for the
    /// second.
    pub fn snapshot(&self) -> Vec<GuardrailEnforcedHit> {
        self.hits
            .lock()
            .map(|hits| hits.values().cloned().collect())
            .unwrap_or_default()
    }
}

/// `allow` refuses BELOW its threshold, so its coalescing keeps the
/// minimum; every other direction keeps the maximum.
const DIRECTION_ALLOW: &str = "allow";

/// Intern the hook into the `&'static str` the key needs. An unrecognised
/// value keys as `"other"` rather than being dropped: losing an entry is a
/// worse failure than an odd key, and the entry still carries the string it
/// was recorded with.
fn hook_key(hook: &str) -> &'static str {
    match hook {
        "input" => "input",
        "output" => "output",
        _ => "other",
    }
}

fn direction_key(direction: &str) -> &'static str {
    match direction {
        "deny" => "deny",
        DIRECTION_ALLOW => "allow",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(pairs: &[(&str, u32)]) -> BTreeMap<String, u32> {
        pairs.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect()
    }

    #[test]
    fn repeated_masks_by_the_same_guardrail_coalesce_into_one_entry() {
        let log = GuardrailAuditLog::new();
        log.record(
            "pii-guard",
            "output",
            "masked",
            None,
            Duration::from_micros(40),
            &counts(&[("email", 2)]),
        );
        log.record(
            "pii-guard",
            "output",
            "masked",
            None,
            Duration::from_micros(60),
            &counts(&[("email", 1), ("phone", 3)]),
        );

        let hits = log.snapshot();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].guardrail_name, "pii-guard");
        assert_eq!(hits[0].hook, "output");
        assert_eq!(hits[0].action, "masked");
        assert_eq!(hits[0].counts, counts(&[("email", 3), ("phone", 3)]));
        assert_eq!(hits[0].duration_us, 100);
    }

    #[test]
    fn hook_and_action_split_entries_apart() {
        let log = GuardrailAuditLog::new();
        let none = BTreeMap::new();
        log.record(
            "g",
            "input",
            "masked",
            None,
            Duration::ZERO,
            &counts(&[("ip", 1)]),
        );
        log.record(
            "g",
            "output",
            "masked",
            None,
            Duration::ZERO,
            &counts(&[("ip", 1)]),
        );
        log.record("g", "output", "blocked", None, Duration::ZERO, &none);
        log.record(
            "other",
            "input",
            "masked",
            None,
            Duration::ZERO,
            &counts(&[("ip", 1)]),
        );

        let hits = log.snapshot();
        assert_eq!(hits.len(), 4);
        // BTreeMap order: ("g","input","masked"), ("g","output","blocked"),
        // ("g","output","masked"), ("other","input","masked").
        assert_eq!(
            hits.iter()
                .map(|h| (
                    h.guardrail_name.as_str(),
                    h.hook.as_str(),
                    h.action.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("g", "input", "masked"),
                ("g", "output", "blocked"),
                ("g", "output", "masked"),
                ("other", "input", "masked"),
            ]
        );
    }

    fn score(direction: &str, value: f32, index: u32) -> GuardrailScore {
        GuardrailScore {
            guardrail_name: "sem".into(),
            hook: "input".into(),
            direction: direction.into(),
            score: value,
            threshold: 0.75,
            matched: value >= 0.75,
            top_example_index: index,
            embedding_model: "embed-1".into(),
        }
    }

    #[test]
    fn repeated_deny_scores_keep_the_highest_and_its_example() {
        // The `/mcp` path scans a tool result leaf by leaf, so one hook can
        // score many times. Keeping the last would make the reported number
        // depend on scan order and routinely under-report the closest call.
        let log = GuardrailAuditLog::new();
        log.record_score(score("deny", 0.41, 0));
        log.record_score(score("deny", 0.68, 3));
        log.record_score(score("deny", 0.52, 1));

        let scores = log.score_snapshot();
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].score, 0.68);
        assert_eq!(
            scores[0].top_example_index, 3,
            "the index must travel with the score it belongs to"
        );
    }

    #[test]
    fn repeated_allow_scores_keep_the_lowest() {
        // `allow` refuses BELOW its threshold, so its closest call is the
        // minimum. Folding both directions with `max` would report the
        // safest text and hide the one that nearly failed.
        let log = GuardrailAuditLog::new();
        log.record_score(score("allow", 0.90, 0));
        log.record_score(score("allow", 0.33, 2));
        log.record_score(score("allow", 0.71, 1));

        let scores = log.score_snapshot();
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].score, 0.33);
        assert_eq!(scores[0].top_example_index, 2);
    }

    #[test]
    fn hook_and_direction_split_score_entries_apart() {
        let log = GuardrailAuditLog::new();
        log.record_score(score("deny", 0.4, 0));
        log.record_score(score("allow", 0.4, 0));
        let mut output = score("deny", 0.9, 0);
        output.hook = "output".into();
        log.record_score(output);
        let mut other_row = score("deny", 0.4, 0);
        other_row.guardrail_name = "sem-2".into();
        log.record_score(other_row);

        let scores = log.score_snapshot();
        assert_eq!(
            scores
                .iter()
                .map(|s| (
                    s.guardrail_name.as_str(),
                    s.hook.as_str(),
                    s.direction.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("sem", "input", "allow"),
                ("sem", "input", "deny"),
                ("sem", "output", "deny"),
                ("sem-2", "input", "deny"),
            ]
        );
    }

    #[test]
    fn score_snapshot_does_not_drain() {
        let log = GuardrailAuditLog::new();
        log.record_score(score("deny", 0.4, 0));
        assert_eq!(log.score_snapshot().len(), 1);
        assert_eq!(log.score_snapshot().len(), 1);
    }

    #[test]
    fn snapshot_does_not_drain() {
        let log = GuardrailAuditLog::new();
        log.record(
            "g",
            "input",
            "blocked",
            None,
            Duration::ZERO,
            &BTreeMap::new(),
        );
        assert_eq!(log.snapshot().len(), 1);
        assert_eq!(log.snapshot().len(), 1);
    }
}
