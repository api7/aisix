//! Compose multiple guardrails into one. First [`GuardrailVerdict::Block`]
//! short-circuits the chain; subsequent guardrails are not consulted.
//! The chain attributes each `Block` to the member that fired: it carries
//! the member's configured name in `GuardrailVerdict::Block::guardrail_name`
//! and prefixes the operator-facing `reason` with it (#519 B.4b), so both
//! the wire envelope and the ops logs say WHICH rule blocked.
//! Useful for building a single `Arc<dyn Guardrail>` to hand to the
//! proxy from a config-driven list.

use std::sync::Arc;

use aisix_core::AppliedGuardrail;
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;

use aisix_core::models::GuardrailMonitorHit;

use crate::{Guardrail, GuardrailVerdict, Redaction, SegmentsOutcome, StreamOutputPolicy};

/// One chain member: the runtime guardrail plus the operator-facing name
/// of the row it was built from. The name is what `Block` verdicts are
/// attributed to; chains built without row context ([`GuardrailChain::new`])
/// fall back to the impl's static [`Guardrail::name`].
#[derive(Clone)]
struct ChainMember {
    name: String,
    guardrail: Arc<dyn Guardrail>,
}

#[derive(Clone)]
pub struct GuardrailChain {
    members: Vec<ChainMember>,
    /// The `{kind, hook}` of each guardrail that materialised into this
    /// chain, captured at build time. Carried onto the telemetry
    /// `UsageEvent` so the dashboard can show which guardrails governed a
    /// request (#379). Empty for chains built via [`GuardrailChain::new`]
    /// (the in-memory test path); populated by the snapshot build points
    /// (`build_chain_from_snapshot` and `GuardrailIndex::resolve`).
    applied: Vec<AppliedGuardrail>,
}

impl std::fmt::Debug for GuardrailChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardrailChain")
            .field("guardrails", &self.member_names())
            .finish()
    }
}

impl GuardrailChain {
    pub fn new(guardrails: Vec<Arc<dyn Guardrail>>) -> Self {
        Self {
            members: guardrails
                .into_iter()
                .map(|g| ChainMember {
                    name: g.name().to_owned(),
                    guardrail: g,
                })
                .collect(),
            applied: Vec::new(),
        }
    }

    /// Build a chain that also carries each member's configured (row) name
    /// — used for `Block` attribution (#519 B.4b) — and the `{kind, hook}`
    /// of each member for applied-guardrail telemetry (#379). Used by the
    /// snapshot build points; `applied` is expected to line up 1:1 with
    /// `members`, but the chain's runtime behaviour does not depend on
    /// that — `applied` is telemetry-only.
    pub fn new_with_applied(
        members: Vec<(String, Arc<dyn Guardrail>)>,
        applied: Vec<AppliedGuardrail>,
    ) -> Self {
        Self {
            members: members
                .into_iter()
                .map(|(name, guardrail)| ChainMember { name, guardrail })
                .collect(),
            applied,
        }
    }

    /// The `{kind, hook}` set of guardrails that governed this request,
    /// in chain order. Empty when the chain was built without applied
    /// metadata (e.g. [`GuardrailChain::new`]).
    pub fn applied(&self) -> &[AppliedGuardrail] {
        &self.applied
    }

    /// The members' configured names, in evaluation order. The snapshot
    /// build points sort rows `created_at`-ascending (id-tiebreak) before
    /// building, so this order is deterministic and matches the dashboard
    /// listing (#519 B.4a).
    pub fn member_names(&self) -> Vec<&str> {
        self.members.iter().map(|m| m.name.as_str()).collect()
    }

    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// Attribute a member's `Block` verdict to its configured name: fill
/// `guardrail_name` and prefix the ops-log `reason`. A verdict that is
/// already attributed (a nested chain) passes through untouched so the
/// innermost — most specific — name wins and the reason isn't
/// double-prefixed.
fn attribute_block(
    member_name: &str,
    reason: String,
    guardrail_name: Option<String>,
) -> GuardrailVerdict {
    match guardrail_name {
        Some(_) => GuardrailVerdict::Block {
            reason,
            guardrail_name,
        },
        None => GuardrailVerdict::Block {
            reason: format!("guardrail '{member_name}': {reason}"),
            guardrail_name: Some(member_name.to_owned()),
        },
    }
}

#[async_trait]
impl Guardrail for GuardrailChain {
    fn name(&self) -> &'static str {
        "chain"
    }

    fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The strictest streamed-output policy across the chain's
    /// **output-hook** members. Only guardrails that actually inspect the
    /// output influence hold-back; an input-only member must not force the
    /// response to buffer (#466). If any output member wants hold-back, the
    /// whole stream holds back and the full chain's `check_output` runs on
    /// the held content.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        self.members
            .iter()
            .filter(|m| m.guardrail.runs_on_output())
            .map(|m| m.guardrail.stream_output_policy())
            .fold(
                StreamOutputPolicy::EndOfStreamCheck,
                StreamOutputPolicy::stricter,
            )
    }

    fn runs_on_output(&self) -> bool {
        self.members.iter().any(|m| m.guardrail.runs_on_output())
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        let mut bypass: Option<String> = None;
        for m in &self.members {
            match m.guardrail.check_input(req).await {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } => return attribute_block(&m.name, reason, guardrail_name),
                GuardrailVerdict::Bypass { reason } => {
                    // First bypass sticks; downstream guardrails still
                    // get to inspect the request (they may Block).
                    if bypass.is_none() {
                        bypass = Some(reason);
                    }
                }
            }
        }
        match bypass {
            Some(reason) => GuardrailVerdict::Bypass { reason },
            None => GuardrailVerdict::Allow,
        }
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        let mut bypass: Option<String> = None;
        for m in &self.members {
            match m.guardrail.check_output(resp).await {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } => return attribute_block(&m.name, reason, guardrail_name),
                GuardrailVerdict::Bypass { reason } => {
                    if bypass.is_none() {
                        bypass = Some(reason);
                    }
                }
            }
        }
        match bypass {
            Some(reason) => GuardrailVerdict::Bypass { reason },
            None => GuardrailVerdict::Allow,
        }
    }

    // Observed folds (AISIX-Cloud#562): same short-circuit semantics as the
    // plain folds, but every member's monitor-mode observations are
    // collected — including the ones made before an enforcing member
    // blocks, so a monitored rule's hit isn't erased by a peer's Block.
    async fn check_input_observed(
        &self,
        req: &ChatFormat,
    ) -> (GuardrailVerdict, Vec<GuardrailMonitorHit>) {
        let mut bypass: Option<String> = None;
        let mut hits: Vec<GuardrailMonitorHit> = Vec::new();
        for m in &self.members {
            let (verdict, member_hits) = m.guardrail.check_input_observed(req).await;
            hits.extend(member_hits);
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } => return (attribute_block(&m.name, reason, guardrail_name), hits),
                GuardrailVerdict::Bypass { reason } => {
                    if bypass.is_none() {
                        bypass = Some(reason);
                    }
                }
            }
        }
        let verdict = match bypass {
            Some(reason) => GuardrailVerdict::Bypass { reason },
            None => GuardrailVerdict::Allow,
        };
        (verdict, hits)
    }

    async fn check_output_observed(
        &self,
        resp: &ChatResponse,
    ) -> (GuardrailVerdict, Vec<GuardrailMonitorHit>) {
        let mut bypass: Option<String> = None;
        let mut hits: Vec<GuardrailMonitorHit> = Vec::new();
        for m in &self.members {
            let (verdict, member_hits) = m.guardrail.check_output_observed(resp).await;
            hits.extend(member_hits);
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } => return (attribute_block(&m.name, reason, guardrail_name), hits),
                GuardrailVerdict::Bypass { reason } => {
                    if bypass.is_none() {
                        bypass = Some(reason);
                    }
                }
            }
        }
        let verdict = match bypass {
            Some(reason) => GuardrailVerdict::Bypass { reason },
            None => GuardrailVerdict::Allow,
        };
        (verdict, hits)
    }

    async fn check_input_non_segment_observed(
        &self,
        req: &ChatFormat,
    ) -> (GuardrailVerdict, Vec<GuardrailMonitorHit>) {
        let mut bypass: Option<String> = None;
        let mut hits: Vec<GuardrailMonitorHit> = Vec::new();
        for m in &self.members {
            let (verdict, member_hits) = m.guardrail.check_input_non_segment_observed(req).await;
            hits.extend(member_hits);
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } => return (attribute_block(&m.name, reason, guardrail_name), hits),
                GuardrailVerdict::Bypass { reason } => {
                    if bypass.is_none() {
                        bypass = Some(reason);
                    }
                }
            }
        }
        let verdict = match bypass {
            Some(reason) => GuardrailVerdict::Bypass { reason },
            None => GuardrailVerdict::Allow,
        };
        (verdict, hits)
    }

    async fn check_output_non_segment_observed(
        &self,
        resp: &ChatResponse,
    ) -> (GuardrailVerdict, Vec<GuardrailMonitorHit>) {
        let mut bypass: Option<String> = None;
        let mut hits: Vec<GuardrailMonitorHit> = Vec::new();
        for m in &self.members {
            let (verdict, member_hits) = m.guardrail.check_output_non_segment_observed(resp).await;
            hits.extend(member_hits);
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } => return (attribute_block(&m.name, reason, guardrail_name), hits),
                GuardrailVerdict::Bypass { reason } => {
                    if bypass.is_none() {
                        bypass = Some(reason);
                    }
                }
            }
        }
        let verdict = match bypass {
            Some(reason) => GuardrailVerdict::Bypass { reason },
            None => GuardrailVerdict::Allow,
        };
        (verdict, hits)
    }

    fn moderates_segments(&self) -> bool {
        self.members
            .iter()
            .any(|m| m.guardrail.moderates_segments())
    }

    /// Fold over segment-moderating members only. A Block short-circuits
    /// (attributed like the check folds); masked texts COMPOSE — each
    /// member moderates the previous member's masked output, mirroring
    /// `fold_redactions`; the first Bypass reason sticks. Counts merge.
    async fn moderate_input_segments(&self, texts: &[String]) -> SegmentsOutcome {
        fold_segments(&self.members, texts, true).await
    }

    async fn moderate_output_segments(&self, texts: &[String]) -> SegmentsOutcome {
        fold_segments(&self.members, texts, false).await
    }

    /// The check fold minus segment-moderating members — the pass those
    /// members are consulted through is `moderate_*_segments`, run by the
    /// same call sites. Recurses via the member's own
    /// `check_input_non_segment` so a nested chain filters its own members
    /// rather than being skipped wholesale.
    async fn check_input_non_segment(&self, req: &ChatFormat) -> GuardrailVerdict {
        let mut bypass: Option<String> = None;
        for m in &self.members {
            match m.guardrail.check_input_non_segment(req).await {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } => return attribute_block(&m.name, reason, guardrail_name),
                GuardrailVerdict::Bypass { reason } => {
                    if bypass.is_none() {
                        bypass = Some(reason);
                    }
                }
            }
        }
        match bypass {
            Some(reason) => GuardrailVerdict::Bypass { reason },
            None => GuardrailVerdict::Allow,
        }
    }

    async fn check_output_non_segment(&self, resp: &ChatResponse) -> GuardrailVerdict {
        let mut bypass: Option<String> = None;
        for m in &self.members {
            match m.guardrail.check_output_non_segment(resp).await {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                } => return attribute_block(&m.name, reason, guardrail_name),
                GuardrailVerdict::Bypass { reason } => {
                    if bypass.is_none() {
                        bypass = Some(reason);
                    }
                }
            }
        }
        match bypass {
            Some(reason) => GuardrailVerdict::Bypass { reason },
            None => GuardrailVerdict::Allow,
        }
    }

    fn redacts_input(&self) -> bool {
        self.members.iter().any(|m| m.guardrail.redacts_input())
    }

    fn redacts_output(&self) -> bool {
        self.members.iter().any(|m| m.guardrail.redacts_output())
    }

    /// Members apply in chain order, each rewriting the previous member's
    /// output, so stacked redacting guardrails compose. Counts merge across
    /// members.
    fn redact_input_text(&self, text: &str) -> Option<Redaction> {
        fold_redactions(
            text,
            self.members
                .iter()
                .filter_map(|m| m.guardrail.redacts_input().then_some(&m.guardrail)),
            true,
        )
    }

    fn redact_output_text(&self, text: &str) -> Option<Redaction> {
        fold_redactions(
            text,
            self.members
                .iter()
                .filter_map(|m| m.guardrail.redacts_output().then_some(&m.guardrail)),
            false,
        )
    }
}

/// Fold the texts through each segment-moderating member. Mirrors the
/// check folds (first Block short-circuits with attribution, first Bypass
/// reason sticks) plus mask composition: each member moderates the
/// previous member's masked output. Counts merge across members.
async fn fold_segments(members: &[ChainMember], texts: &[String], input: bool) -> SegmentsOutcome {
    let mut masked: Option<Vec<String>> = None;
    let mut counts = std::collections::BTreeMap::new();
    let mut bypass: Option<String> = None;
    let mut monitor_hits: Vec<GuardrailMonitorHit> = Vec::new();
    for m in members {
        if !m.guardrail.moderates_segments() {
            continue;
        }
        let src: &[String] = masked.as_deref().unwrap_or(texts);
        let mut outcome = if input {
            m.guardrail.moderate_input_segments(src).await
        } else {
            m.guardrail.moderate_output_segments(src).await
        };
        monitor_hits.append(&mut outcome.monitor_hits);
        match outcome.verdict {
            GuardrailVerdict::Allow => {}
            GuardrailVerdict::Block {
                reason,
                guardrail_name,
            } => {
                return SegmentsOutcome {
                    verdict: attribute_block(&m.name, reason, guardrail_name),
                    masked: None,
                    counts: std::collections::BTreeMap::new(),
                    monitor_hits,
                }
            }
            GuardrailVerdict::Bypass { reason } => {
                if bypass.is_none() {
                    bypass = Some(reason);
                }
            }
        }
        if let Some(new_masked) = outcome.masked {
            // Implementations uphold alignment with THEIR input; refuse a
            // drifted length here so a broken member can't desync slots.
            // Counts merge ONLY with an accepted mask — they describe
            // APPLIED anonymization (`redacted_entity_counts`), so a
            // refused mask must not inflate them.
            if new_masked.len() == src.len() {
                masked = Some(new_masked);
                Redaction::merge_counts(&mut counts, &outcome.counts);
            } else {
                tracing::warn!(
                    member = %m.name,
                    expected = src.len(),
                    got = new_masked.len(),
                    "segment moderation returned misaligned mask; keeping originals",
                );
            }
        }
    }
    SegmentsOutcome {
        verdict: match bypass {
            Some(reason) => GuardrailVerdict::Bypass { reason },
            None => GuardrailVerdict::Allow,
        },
        masked,
        counts,
        monitor_hits,
    }
}

/// Fold `text` through each member's redactor, merging counts. `None`
/// when no member changed anything.
fn fold_redactions<'a>(
    text: &str,
    members: impl Iterator<Item = &'a Arc<dyn Guardrail>>,
    input: bool,
) -> Option<Redaction> {
    let mut current: Option<Redaction> = None;
    for g in members {
        let src = current.as_ref().map_or(text, |r| r.text.as_str());
        let next = if input {
            g.redact_input_text(src)
        } else {
            g.redact_output_text(src)
        };
        if let Some(r) = next {
            current = Some(match current.take() {
                None => r,
                Some(mut acc) => {
                    acc.text = r.text;
                    Redaction::merge_counts(&mut acc.counts, &r.counts);
                    acc
                }
            });
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KeywordBlocklist, KeywordRule};
    use aisix_gateway::{ChatMessage, FinishReason, UsageStats};

    fn req(msg: &str) -> ChatFormat {
        ChatFormat::new("m", vec![ChatMessage::user(msg)])
    }

    fn resp(content: &str) -> ChatResponse {
        ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: ChatMessage::assistant(content),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(0, 0),
        }
    }

    #[tokio::test]
    async fn empty_chain_allows_everything() {
        let chain = GuardrailChain::empty();
        assert_eq!(chain.check_input(&req("hi")).await, GuardrailVerdict::Allow);
        assert_eq!(
            chain.check_output(&resp("hi")).await,
            GuardrailVerdict::Allow,
        );
    }

    #[tokio::test]
    async fn first_block_short_circuits_subsequent_guardrails() {
        // Both would block on the same input; the first wins so the
        // reason is deterministic.
        let chain = GuardrailChain::new(vec![
            Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("alpha")])),
            Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("beta")])),
        ]);
        let v = chain.check_input(&req("alpha and beta")).await;
        if let GuardrailVerdict::Block { reason, .. } = v {
            assert!(reason.contains("alpha"));
        } else {
            panic!("expected Block");
        }
    }

    #[tokio::test]
    async fn allow_falls_through_to_next_guardrail() {
        // First guardrail allows everything; second blocks on its literal.
        let chain = GuardrailChain::new(vec![
            Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal(
                "nope-not-here",
            )])),
            Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("long")])),
        ]);
        let v = chain.check_input(&req("this is way too long")).await;
        assert!(v.is_block());
    }

    /// #519 B.4b: a chain Block carries the firing member's configured
    /// name — both as the structured `guardrail_name` (for the wire
    /// envelope) and as a `guardrail '<name>': ` prefix on the ops-log
    /// reason.
    #[tokio::test]
    async fn block_is_attributed_to_the_firing_member_by_name() {
        let chain = GuardrailChain::new_with_applied(
            vec![
                (
                    "pass-through".to_owned(),
                    Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal(
                        "never-matches",
                    )])) as Arc<dyn Guardrail>,
                ),
                (
                    "block-secrets".to_owned(),
                    Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("AKIA")])),
                ),
            ],
            Vec::new(),
        );

        match chain.check_input(&req("here is AKIAEXAMPLE")).await {
            GuardrailVerdict::Block {
                reason,
                guardrail_name,
            } => {
                assert_eq!(guardrail_name.as_deref(), Some("block-secrets"));
                assert!(
                    reason.starts_with("guardrail 'block-secrets': "),
                    "reason must be prefixed with the firing member's name: {reason}",
                );
            }
            other => panic!("expected Block, got {other:?}"),
        }

        // Output side uses the same attribution path.
        match chain.check_output(&resp("the AKIA secret")).await {
            GuardrailVerdict::Block { guardrail_name, .. } => {
                assert_eq!(guardrail_name.as_deref(), Some("block-secrets"));
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    /// A nested chain's Block is already attributed; the outer chain must
    /// pass it through (innermost name wins, no double prefix).
    #[tokio::test]
    async fn nested_chain_block_keeps_innermost_attribution() {
        let inner = GuardrailChain::new_with_applied(
            vec![(
                "inner-rule".to_owned(),
                Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("AKIA")]))
                    as Arc<dyn Guardrail>,
            )],
            Vec::new(),
        );
        let outer = GuardrailChain::new_with_applied(
            vec![(
                "outer-chain".to_owned(),
                Arc::new(inner) as Arc<dyn Guardrail>,
            )],
            Vec::new(),
        );

        match outer.check_input(&req("AKIA")).await {
            GuardrailVerdict::Block {
                reason,
                guardrail_name,
            } => {
                assert_eq!(guardrail_name.as_deref(), Some("inner-rule"));
                assert!(
                    reason.starts_with("guardrail 'inner-rule': "),
                    "no double prefix expected: {reason}",
                );
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    /// Bypass doesn't short-circuit: a downstream Block must still
    /// fire. This is the failure mode that matters when an operator
    /// stacks a Bedrock guardrail (which can bypass on AWS 5xx) on
    /// top of a keyword guardrail (which is local + always available).
    #[tokio::test]
    async fn bypass_does_not_short_circuit_keyword_block() {
        struct AlwaysBypass;
        #[async_trait]
        impl Guardrail for AlwaysBypass {
            fn name(&self) -> &'static str {
                "always-bypass"
            }
            async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
                GuardrailVerdict::Bypass {
                    reason: "test".into(),
                }
            }
        }
        let chain = GuardrailChain::new(vec![
            Arc::new(AlwaysBypass),
            Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("AKIA")])),
        ]);
        // Bypass first, then a keyword Block — Block must win.
        let v = chain.check_input(&req("here is AKIAEXAMPLE")).await;
        assert!(v.is_block(), "expected Block, got {v:?}");
    }

    /// When no guardrail blocks but at least one bypassed, the chain's
    /// verdict is the first bypass reason — chat handler attaches
    /// it to the telemetry event.
    #[tokio::test]
    async fn bypass_propagates_when_no_block_fires() {
        struct AlwaysBypass(&'static str);
        #[async_trait]
        impl Guardrail for AlwaysBypass {
            fn name(&self) -> &'static str {
                "always-bypass"
            }
            async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
                GuardrailVerdict::Bypass {
                    reason: self.0.into(),
                }
            }
        }
        let chain = GuardrailChain::new(vec![
            Arc::new(AlwaysBypass("first")),
            Arc::new(AlwaysBypass("second")),
        ]);
        let v = chain.check_input(&req("hello")).await;
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "first"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn output_check_short_circuits_on_first_block() {
        let chain = GuardrailChain::new(vec![
            Arc::new(KeywordBlocklist::output_only(vec![KeywordRule::literal(
                "secret",
            )])),
            Arc::new(KeywordBlocklist::output_only(vec![KeywordRule::literal(
                "answer",
            )])),
        ]);
        // The first keyword guardrail fires before the second.
        let v = chain.check_output(&resp("the secret answer")).await;
        if let GuardrailVerdict::Block { reason, .. } = v {
            assert!(reason.contains("secret"));
        } else {
            panic!("expected Block");
        }
    }

    #[test]
    fn input_only_member_does_not_force_streamed_output_holdback() {
        // #466 regression: the trait default stream policy is now BufferFull
        // (secure-by-default), but a chain whose only member is input-only must
        // NOT buffer the response stream — it never inspects output.
        let input_only = GuardrailChain::new(vec![Arc::new(KeywordBlocklist::input_only(vec![
            KeywordRule::literal("x"),
        ]))]);
        assert!(!input_only.runs_on_output());
        assert!(
            !input_only.stream_output_policy().holds_back(),
            "input-only chain must fold to a non-holding policy"
        );

        // An output guardrail folds to the default hold-back policy.
        let output = GuardrailChain::new(vec![Arc::new(KeywordBlocklist::output_only(vec![
            KeywordRule::literal("x"),
        ]))]);
        assert!(output.runs_on_output());
        assert!(
            output.stream_output_policy().holds_back(),
            "output chain must fold to a holding policy"
        );

        // A mixed chain (input-only + output) still holds back because of the
        // output member; the input-only member is skipped, not the driver.
        let mixed = GuardrailChain::new(vec![
            Arc::new(KeywordBlocklist::input_only(vec![KeywordRule::literal(
                "x",
            )])),
            Arc::new(KeywordBlocklist::output_only(vec![KeywordRule::literal(
                "y",
            )])),
        ]);
        assert!(mixed.runs_on_output());
        assert!(mixed.stream_output_policy().holds_back());

        // Empty chain → nothing runs on output, no hold-back.
        let empty = GuardrailChain::new(vec![]);
        assert!(!empty.runs_on_output());
        assert!(!empty.stream_output_policy().holds_back());
    }

    // --- segment moderation folds (#932 bedrock follow-up) ---------------

    /// A stub segment moderator: uppercases every slot and reports a
    /// fixed count key, or blocks/bypasses on demand.
    struct StubSegments {
        verdict: GuardrailVerdict,
        mask: bool,
    }
    #[async_trait]
    impl Guardrail for StubSegments {
        fn name(&self) -> &'static str {
            "stub-segments"
        }
        fn moderates_segments(&self) -> bool {
            true
        }
        async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
            panic!("segment member must not be consulted via check_input_non_segment");
        }
        async fn moderate_input_segments(&self, texts: &[String]) -> crate::SegmentsOutcome {
            let mut counts = std::collections::BTreeMap::new();
            counts.insert("STUB".to_owned(), texts.len() as u32);
            crate::SegmentsOutcome {
                verdict: self.verdict.clone(),
                masked: self
                    .mask
                    .then(|| texts.iter().map(|t| t.to_uppercase()).collect()),
                counts,
                monitor_hits: Vec::new(),
            }
        }
    }

    /// The non-segment check fold skips segment members (they're consulted
    /// via the segment pass) while normal members still run — the panic in
    /// the stub's `check_input` proves the skip.
    #[tokio::test]
    async fn check_input_non_segment_skips_segment_members_but_not_others() {
        let chain = GuardrailChain::new(vec![
            Arc::new(StubSegments {
                verdict: GuardrailVerdict::Allow,
                mask: false,
            }),
            Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("AKIA")])),
        ]);
        // Keyword member still blocks...
        assert!(chain
            .check_input_non_segment(&req("here is AKIAEXAMPLE"))
            .await
            .is_block());
        // ...and a clean request is Allow (the stub's check_input would
        // have panicked if consulted).
        assert_eq!(
            chain.check_input_non_segment(&req("clean")).await,
            GuardrailVerdict::Allow,
        );
        // The FULL fold still consults every member (unconverted call
        // sites keep blob-mode coverage) — the stub panics to prove it
        // WOULD be consulted there; assert via catch_unwind-free route:
        // moderates_segments visibility.
        assert!(chain.moderates_segments());
    }

    /// Segment masks compose across members in chain order, counts merge,
    /// and a Block short-circuits with attribution.
    #[tokio::test]
    async fn segment_fold_composes_masks_and_attributes_blocks() {
        // Two maskers: uppercase then uppercase again (idempotent — the
        // composition is observable via counts merging to 2 members).
        let chain = GuardrailChain::new_with_applied(
            vec![
                (
                    "mask-a".to_owned(),
                    Arc::new(StubSegments {
                        verdict: GuardrailVerdict::Allow,
                        mask: true,
                    }) as Arc<dyn Guardrail>,
                ),
                (
                    "mask-b".to_owned(),
                    Arc::new(StubSegments {
                        verdict: GuardrailVerdict::Allow,
                        mask: true,
                    }),
                ),
            ],
            Vec::new(),
        );
        let texts = vec!["hello".to_owned(), "world".to_owned()];
        let out = chain.moderate_input_segments(&texts).await;
        assert_eq!(out.verdict, GuardrailVerdict::Allow);
        assert_eq!(
            out.masked,
            Some(vec!["HELLO".to_owned(), "WORLD".to_owned()]),
        );
        assert_eq!(out.counts.get("STUB"), Some(&4), "2 members × 2 slots");

        // Block short-circuits and is attributed to the firing member.
        let blocking = GuardrailChain::new_with_applied(
            vec![(
                "seg-blocker".to_owned(),
                Arc::new(StubSegments {
                    verdict: GuardrailVerdict::block("pii blocked"),
                    mask: false,
                }) as Arc<dyn Guardrail>,
            )],
            Vec::new(),
        );
        match blocking.moderate_input_segments(&texts).await.verdict {
            GuardrailVerdict::Block { guardrail_name, .. } => {
                assert_eq!(guardrail_name.as_deref(), Some("seg-blocker"))
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    /// A member returning a mask whose length drifted from ITS input is
    /// refused (originals kept) — the chain-level alignment guard.
    #[tokio::test]
    async fn segment_fold_refuses_misaligned_member_mask() {
        struct Drifting;
        #[async_trait]
        impl Guardrail for Drifting {
            fn name(&self) -> &'static str {
                "drifting"
            }
            fn moderates_segments(&self) -> bool {
                true
            }
            async fn moderate_input_segments(&self, _texts: &[String]) -> crate::SegmentsOutcome {
                let mut counts = std::collections::BTreeMap::new();
                counts.insert("EMAIL".to_owned(), 3);
                crate::SegmentsOutcome {
                    verdict: GuardrailVerdict::Allow,
                    masked: Some(vec!["only-one".to_owned()]),
                    counts,
                    monitor_hits: Vec::new(),
                }
            }
        }
        let chain = GuardrailChain::new(vec![Arc::new(Drifting)]);
        let texts = vec!["a".to_owned(), "b".to_owned()];
        let out = chain.moderate_input_segments(&texts).await;
        assert_eq!(out.masked, None, "drifted mask must be refused");
        assert!(
            out.counts.is_empty(),
            "a refused mask's counts describe anonymization that was NOT \
             applied — they must not reach redacted_entity_counts",
        );
        assert_eq!(out.verdict, GuardrailVerdict::Allow);
    }

    #[test]
    fn new_has_empty_applied_and_new_with_applied_reports_it() {
        // `new` (the in-memory/test constructor) carries no applied metadata;
        // `new_with_applied` (the snapshot build points) reports it verbatim.
        assert!(GuardrailChain::new(vec![]).applied().is_empty());

        let applied = vec![
            AppliedGuardrail {
                kind: "keyword".to_owned(),
                hook: "input".to_owned(),
            },
            AppliedGuardrail {
                kind: "aliyun_text_moderation".to_owned(),
                hook: "both".to_owned(),
            },
        ];
        let chain = GuardrailChain::new_with_applied(vec![], applied.clone());
        assert_eq!(chain.applied(), applied.as_slice());
    }
}
