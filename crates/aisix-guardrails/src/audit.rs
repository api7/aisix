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
//! Names and counts only: the matched value and the block reason never
//! enter this log (#153 / #932 no-leak criterion).

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use aisix_core::models::GuardrailEnforcedHit;

/// Coalescing key: one entry per guardrail row, per hook, per action —
/// and, for the one action that carries a cause, per cause. The cause is
/// part of the key rather than merged in because two different causes are
/// two different facts; in practice a member refuses at most once per
/// hook, so the extra dimension adds no entries.
type HitKey = (String, &'static str, &'static str, String);

/// Per-request accumulator of enforced guardrail hits.
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
        log.record("g", "input", "masked", None, Duration::ZERO, &counts(&[("ip", 1)]));
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

    #[test]
    fn snapshot_does_not_drain() {
        let log = GuardrailAuditLog::new();
        log.record("g", "input", "blocked", None, Duration::ZERO, &BTreeMap::new());
        assert_eq!(log.snapshot().len(), 1);
        assert_eq!(log.snapshot().len(), 1);
    }
}
