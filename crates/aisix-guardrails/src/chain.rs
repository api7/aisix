//! Compose multiple guardrails into one. First [`GuardrailVerdict::Block`]
//! short-circuits the chain; subsequent guardrails are not consulted.
//! The chain attributes each `Block` to the member that fired: it carries
//! the member's configured name in `GuardrailVerdict::Block::guardrail_name`
//! and prefixes the operator-facing `reason` with it (#519 B.4b), so both
//! the wire envelope and the ops logs say WHICH rule blocked.
//! Useful for building a single `Arc<dyn Guardrail>` to hand to the
//! proxy from a config-driven list.

use std::sync::Arc;
use std::time::Instant;

use aisix_core::AppliedGuardrail;
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;

use aisix_core::models::{
    GuardrailEnforcedHit, GuardrailExecution, GuardrailMetricsSink, GuardrailMonitorHit,
};

use crate::audit::GuardrailAuditLog;
use crate::{Guardrail, GuardrailVerdict, Redaction, SegmentsOutcome, StreamOutputPolicy};

/// One chain member: the runtime guardrail plus the operator-facing name
/// and `kind` of the row it was built from. The name is what `Block`
/// verdicts are attributed to; chains built without row context
/// ([`GuardrailChain::new`]) fall back to the impl's static
/// [`Guardrail::name`] for both.
#[derive(Clone)]
struct ChainMember {
    name: String,
    kind: String,
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
    /// Per-execution telemetry receiver (AISIX-Cloud#1076). `None` (the
    /// default) records nothing; `LiveGuardrailIndex::resolve` attaches
    /// the metrics layer's sink so every fold below reports each member's
    /// phase/result/duration.
    sink: Option<Arc<dyn GuardrailMetricsSink>>,
    /// Per-REQUEST accumulator of enforcing hits (AISIX-Cloud#1330).
    /// `None` (the default) records nothing; `LiveGuardrailIndex::resolve`
    /// mints a fresh log on every resolve, and a chain is resolved once
    /// per request, so the log never spans two requests. Unlike `sink` —
    /// a process-global metrics receiver — this one is read back by the
    /// handler that emits the request's usage event.
    audit: Option<Arc<GuardrailAuditLog>>,
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
                    kind: g.name().to_owned(),
                    guardrail: g,
                })
                .collect(),
            applied: Vec::new(),
            sink: None,
            audit: None,
        }
    }

    /// Build a chain that also carries each member's configured (row) name
    /// — used for `Block` attribution (#519 B.4b) — and the `{kind, hook}`
    /// of each member for applied-guardrail telemetry (#379). Used by the
    /// snapshot build points; `applied` is expected to line up 1:1 with
    /// `members` (each member's `kind` label is taken from it), but the
    /// chain's runtime behaviour does not depend on that — `applied` is
    /// telemetry-only.
    pub fn new_with_applied(
        members: Vec<(String, Arc<dyn Guardrail>)>,
        applied: Vec<AppliedGuardrail>,
    ) -> Self {
        Self {
            members: members
                .into_iter()
                .enumerate()
                .map(|(i, (name, guardrail))| ChainMember {
                    kind: applied
                        .get(i)
                        .map(|a| a.kind.clone())
                        .unwrap_or_else(|| guardrail.name().to_owned()),
                    name,
                    guardrail,
                })
                .collect(),
            applied,
            sink: None,
            audit: None,
        }
    }

    /// Attach a per-execution telemetry sink (AISIX-Cloud#1076). Called by
    /// `LiveGuardrailIndex::resolve` on every resolved chain; `None`
    /// disables recording (the default for test-built chains).
    pub fn with_metrics_sink(mut self, sink: Option<Arc<dyn GuardrailMetricsSink>>) -> Self {
        self.sink = sink;
        self
    }

    /// Attach the request's enforced-hit and score log
    /// (AISIX-Cloud#1330, #1467). Called by `LiveGuardrailIndex::resolve`
    /// with a freshly minted log; `None` (the default for test-built
    /// chains) records nothing.
    ///
    /// Members that report similarity scores are rebound to the log here.
    /// This is the only point that has both — the index's members are
    /// shared by every request and the log is minted per request — so a
    /// chain that skips it scores nothing, which is why the two are one
    /// call rather than two.
    pub fn with_audit_log(mut self, audit: Option<Arc<GuardrailAuditLog>>) -> Self {
        if let Some(log) = audit.as_ref() {
            for m in &mut self.members {
                if let Some(bound) = m.guardrail.bind_score_log(log) {
                    m.guardrail = bound;
                }
            }
        }
        self.audit = audit;
        self
    }

    /// The similarity scores recorded on this request so far
    /// (AISIX-Cloud#1467). Empty when no scoring guardrail ran or when the
    /// chain carries no log. Non-destructive — see
    /// [`GuardrailAuditLog::score_snapshot`].
    pub fn scores(&self) -> Vec<aisix_core::GuardrailScore> {
        self.audit
            .as_ref()
            .map(|a| a.score_snapshot())
            .unwrap_or_default()
    }

    /// The ENFORCE-mode hits recorded on this request so far: which
    /// configured guardrail masked or blocked, on which hook, with what
    /// per-detector counts. Empty when no guardrail enforced anything (the
    /// dominant case) or when the chain carries no audit log.
    ///
    /// Non-destructive — see [`GuardrailAuditLog::snapshot`].
    pub fn enforced_hits(&self) -> Vec<GuardrailEnforcedHit> {
        self.audit
            .as_ref()
            .map(|a| a.snapshot())
            .unwrap_or_default()
    }

    /// The request's audit log handle, for a caller that outlives the
    /// chain value: a streaming emitter running inside a `move` closure
    /// after the handler frame is gone, or a handler whose chain is
    /// erased to `Arc<dyn Guardrail>` (the trait carries no `enforced_hits`).
    /// Clone it at the same point `applied()` is snapshotted and read it
    /// with [`GuardrailAuditLog::snapshot`] when the usage event is built.
    ///
    /// `None` for a chain resolved with nothing attached — that request
    /// can never produce an enforced hit, so the absent log and an empty
    /// snapshot mean the same thing.
    pub fn audit_log(&self) -> Option<Arc<GuardrailAuditLog>> {
        self.audit.clone()
    }

    /// Borrow both execution receivers for one fold.
    fn recorders(&self) -> Recorders<'_> {
        Recorders {
            sink: self.sink.as_deref(),
            audit: self.audit.as_deref(),
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

/// Classify one member execution for the metrics sink (AISIX-Cloud#1076).
///
/// The result is the ENFORCED outcome: a monitor-mode member's downgraded
/// Block surfaces as `would_block` (via its hits), not `blocked`. `masked`
/// only arises on the segment pass — the sync per-field redactors are not
/// timed here. The `error_type` is the bounded per-kind failure tag a
/// `Bypass` carries (e.g. `lakera_timeout`) — and, since AISIX-Cloud#1365,
/// the same tag on a fail-CLOSED block, which is an availability failure
/// wearing a block's clothes.
///
/// `result` deliberately stays `blocked` for that case. It is a shipped
/// Prometheus label (AISIX-Cloud#1076) with operator alerting attached, and
/// splitting its value domain would silently stop an existing
/// `result="blocked"` alert from counting outages. Populating `error_type`
/// only widens an existing label instead, so `sum by (result)` is unchanged
/// while `error_type != "none"` separates the two. The audit event, which
/// is unreleased, does split them — see [`record_execution`].
fn classify_execution<'v>(
    verdict: &'v GuardrailVerdict,
    masked: bool,
    hits: &[GuardrailMonitorHit],
) -> (&'static str, Option<&'v str>) {
    match verdict {
        GuardrailVerdict::Block { unavailable, .. } => ("blocked", unavailable.as_deref()),
        GuardrailVerdict::Bypass { reason } => ("bypassed", Some(reason.as_str())),
        GuardrailVerdict::Allow => {
            if masked {
                ("masked", None)
            } else if hits.iter().any(|h| h.action == "would_block") {
                ("would_block", None)
            } else if hits.iter().any(|h| h.action == "would_mask") {
                ("would_mask", None)
            } else {
                ("allowed", None)
            }
        }
    }
}

/// The two receivers a fold reports each member execution to: the
/// process-global metrics sink (AISIX-Cloud#1076) and the request's
/// enforced-hit log (AISIX-Cloud#1330). Either may be absent — a
/// test-built chain has neither — and they are gated independently, so a
/// deployment without the metrics layer still produces audit entries.
#[derive(Clone, Copy, Default)]
struct Recorders<'a> {
    sink: Option<&'a dyn GuardrailMetricsSink>,
    audit: Option<&'a GuardrailAuditLog>,
}

/// Report one member execution. `hits` are the MEMBER's own hits from
/// this call, not the fold's accumulator. `counts` are the member's
/// per-entity mask counts, and only the segment pass has any — the check
/// folds pass `None`.
///
/// The audit log takes only the two ENFORCED outcomes it exists to
/// record: `masked` and `blocked`. `allowed` is not an event, `bypassed`
/// is already carried by `guardrail_bypassed_reason`, and the two
/// `would_*` results belong to `guardrail_monitor_hits` — routing them
/// here would make an enforcing hit indistinguishable from a staged one.
///
/// The two are told apart on the audit event (AISIX-Cloud#1365): a member
/// with `fail_open: false` whose upstream is
/// unreachable returns `Block`, not `Bypass`, and records
/// `blocked_unavailable` plus the bounded failure tag rather than
/// `blocked`. That matters because the naive read of `action = "blocked"`
/// is "this content violated policy X" — which, for a provider outage on a
/// fail-closed row, is a wrong answer rather than a missing one, and one a
/// compliance review would act on.
// Five of the eight describe one member execution (verdict, masked, hits,
// counts) plus where to report it; splitting them into a struct would put a
// name on a grouping that exists only here.
#[allow(clippy::too_many_arguments)]
fn record_execution(
    to: Recorders<'_>,
    member: &ChainMember,
    phase: &'static str,
    started: Instant,
    verdict: &GuardrailVerdict,
    masked: bool,
    hits: &[GuardrailMonitorHit],
    counts: Option<&std::collections::BTreeMap<String, u32>>,
) {
    if to.sink.is_none() && to.audit.is_none() {
        return;
    }
    let (result, error_type) = classify_execution(verdict, masked, hits);
    let elapsed = started.elapsed();
    if let Some(sink) = to.sink {
        sink.record_guardrail_execution(&GuardrailExecution {
            guardrail_name: &member.name,
            kind: &member.kind,
            phase,
            result,
            error_type,
            elapsed,
        });
    }
    if let (Some(audit), "blocked" | "masked") = (to.audit, result) {
        // A `blocked` member reports no counts: the block short-circuits
        // before any span is rewritten, and the reason stays off the event
        // (#153 no-leak). Counts ride an APPLIED mask only — the same
        // invariant `redacted_entity_counts` upholds in `fold_segments`,
        // so the two attributions on one event cannot disagree.
        let empty = std::collections::BTreeMap::new();
        let counts = match (result, counts) {
            ("masked", Some(counts)) => counts,
            _ => &empty,
        };
        // A tagged block is an outage, not a policy decision. Splitting it
        // into its own action — rather than leaving it to a reader who
        // remembers to also check `error_type` — is what makes the naive
        // query `action = 'blocked'` correct by construction
        // (AISIX-Cloud#1365).
        let action = match (result, error_type) {
            ("blocked", Some(_)) => "blocked_unavailable",
            _ => result,
        };
        audit.record(&member.name, phase, action, error_type, elapsed, counts);
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
    // Carried through untouched: whether the block was a content decision
    // or a fail-closed availability failure is the member's fact, not the
    // chain's, and re-attributing the name must not lose it
    // (AISIX-Cloud#1365).
    unavailable: Option<String>,
) -> GuardrailVerdict {
    match guardrail_name {
        Some(_) => GuardrailVerdict::Block {
            reason,
            guardrail_name,
            unavailable,
        },
        None => GuardrailVerdict::Block {
            reason: format!("guardrail '{member_name}': {reason}"),
            guardrail_name: Some(member_name.to_owned()),
            unavailable,
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

    /// A nested chain binds its own members and answers `Some` only when
    /// one of them took the bind — so an outer chain replaces this member
    /// exactly when doing so changes anything.
    fn bind_score_log(&self, log: &Arc<GuardrailAuditLog>) -> Option<Arc<dyn Guardrail>> {
        let mut bound = self.clone();
        let mut any = false;
        for m in &mut bound.members {
            if let Some(g) = m.guardrail.bind_score_log(log) {
                m.guardrail = g;
                any = true;
            }
        }
        any.then(|| Arc::new(bound) as Arc<dyn Guardrail>)
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        let mut bypass: Option<String> = None;
        for m in &self.members {
            let started = Instant::now();
            let verdict = m.guardrail.check_input(req).await;
            record_execution(
                self.recorders(),
                m,
                "input",
                started,
                &verdict,
                false,
                &[],
                None,
            );
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                    unavailable,
                } => return attribute_block(&m.name, reason, guardrail_name, unavailable),
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
            let started = Instant::now();
            let verdict = m.guardrail.check_output(resp).await;
            record_execution(
                self.recorders(),
                m,
                "output",
                started,
                &verdict,
                false,
                &[],
                None,
            );
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                    unavailable,
                } => return attribute_block(&m.name, reason, guardrail_name, unavailable),
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
            let started = Instant::now();
            let (verdict, member_hits) = m.guardrail.check_input_observed(req).await;
            record_execution(
                self.recorders(),
                m,
                "input",
                started,
                &verdict,
                false,
                &member_hits,
                None,
            );
            hits.extend(member_hits);
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                    unavailable,
                } => {
                    return (
                        attribute_block(&m.name, reason, guardrail_name, unavailable),
                        hits,
                    )
                }
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
            let started = Instant::now();
            let (verdict, member_hits) = m.guardrail.check_output_observed(resp).await;
            record_execution(
                self.recorders(),
                m,
                "output",
                started,
                &verdict,
                false,
                &member_hits,
                None,
            );
            hits.extend(member_hits);
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                    unavailable,
                } => {
                    return (
                        attribute_block(&m.name, reason, guardrail_name, unavailable),
                        hits,
                    )
                }
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
            let started = Instant::now();
            let (verdict, member_hits) = m.guardrail.check_input_non_segment_observed(req).await;
            // A segment-moderating member answers via the segment pass —
            // this call is an instant Allow, not an execution; recording
            // it would pollute the member's series with zero-length
            // "allowed" samples.
            if !m.guardrail.moderates_segments() {
                record_execution(
                    self.recorders(),
                    m,
                    "input",
                    started,
                    &verdict,
                    false,
                    &member_hits,
                    None,
                );
            }
            hits.extend(member_hits);
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                    unavailable,
                } => {
                    return (
                        attribute_block(&m.name, reason, guardrail_name, unavailable),
                        hits,
                    )
                }
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
            let started = Instant::now();
            let (verdict, member_hits) = m.guardrail.check_output_non_segment_observed(resp).await;
            if !m.guardrail.moderates_segments() {
                record_execution(
                    self.recorders(),
                    m,
                    "output",
                    started,
                    &verdict,
                    false,
                    &member_hits,
                    None,
                );
            }
            hits.extend(member_hits);
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                    unavailable,
                } => {
                    return (
                        attribute_block(&m.name, reason, guardrail_name, unavailable),
                        hits,
                    )
                }
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
        fold_segments(&self.members, self.recorders(), texts, true).await
    }

    async fn moderate_output_segments(&self, texts: &[String]) -> SegmentsOutcome {
        fold_segments(&self.members, self.recorders(), texts, false).await
    }

    /// The check fold minus segment-moderating members — the pass those
    /// members are consulted through is `moderate_*_segments`, run by the
    /// same call sites. Recurses via the member's own
    /// `check_input_non_segment` so a nested chain filters its own members
    /// rather than being skipped wholesale.
    async fn check_input_non_segment(&self, req: &ChatFormat) -> GuardrailVerdict {
        let mut bypass: Option<String> = None;
        for m in &self.members {
            let started = Instant::now();
            let verdict = m.guardrail.check_input_non_segment(req).await;
            if !m.guardrail.moderates_segments() {
                record_execution(
                    self.recorders(),
                    m,
                    "input",
                    started,
                    &verdict,
                    false,
                    &[],
                    None,
                );
            }
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                    unavailable,
                } => return attribute_block(&m.name, reason, guardrail_name, unavailable),
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
            let started = Instant::now();
            let verdict = m.guardrail.check_output_non_segment(resp).await;
            if !m.guardrail.moderates_segments() {
                record_execution(
                    self.recorders(),
                    m,
                    "output",
                    started,
                    &verdict,
                    false,
                    &[],
                    None,
                );
            }
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block {
                    reason,
                    guardrail_name,
                    unavailable,
                } => return attribute_block(&m.name, reason, guardrail_name, unavailable),
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
            self.members.iter().filter(|m| m.guardrail.redacts_input()),
            true,
            self.audit.as_deref(),
        )
    }

    fn redact_output_text(&self, text: &str) -> Option<Redaction> {
        fold_redactions(
            text,
            self.members.iter().filter(|m| m.guardrail.redacts_output()),
            false,
            self.audit.as_deref(),
        )
    }
}

/// Fold the texts through each segment-moderating member. Mirrors the
/// check folds (first Block short-circuits with attribution, first Bypass
/// reason sticks) plus mask composition: each member moderates the
/// previous member's masked output. Counts merge across members.
async fn fold_segments(
    members: &[ChainMember],
    to: Recorders<'_>,
    texts: &[String],
    input: bool,
) -> SegmentsOutcome {
    let phase = if input { "input" } else { "output" };
    let mut masked: Option<Vec<String>> = None;
    let mut counts = std::collections::BTreeMap::new();
    let mut bypass: Option<String> = None;
    let mut monitor_hits: Vec<GuardrailMonitorHit> = Vec::new();
    for m in members {
        if !m.guardrail.moderates_segments() {
            continue;
        }
        let src: &[String] = masked.as_deref().unwrap_or(texts);
        let started = Instant::now();
        let mut outcome = if input {
            m.guardrail.moderate_input_segments(src).await
        } else {
            m.guardrail.moderate_output_segments(src).await
        };
        // Report the mask the fold will ACCEPT, not the one the member
        // returned: a drifted-length mask is refused below and the
        // originals are kept, so recording it as applied would put
        // `action: "masked"` on a request where nothing was rewritten.
        let accepted_mask = outcome
            .masked
            .as_ref()
            .is_some_and(|new| new.len() == src.len());
        record_execution(
            to,
            m,
            phase,
            started,
            &outcome.verdict,
            accepted_mask,
            &outcome.monitor_hits,
            Some(&outcome.counts),
        );
        monitor_hits.append(&mut outcome.monitor_hits);
        match outcome.verdict {
            GuardrailVerdict::Allow => {}
            GuardrailVerdict::Block {
                reason,
                guardrail_name,
                unavailable,
            } => {
                return SegmentsOutcome {
                    verdict: attribute_block(&m.name, reason, guardrail_name, unavailable),
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
    members: impl Iterator<Item = &'a ChainMember>,
    input: bool,
    audit: Option<&GuardrailAuditLog>,
) -> Option<Redaction> {
    let phase = if input { "input" } else { "output" };
    let mut current: Option<Redaction> = None;
    for m in members {
        let g = &m.guardrail;
        let src = current.as_ref().map_or(text, |r| r.text.as_str());
        let started = Instant::now();
        let next = if input {
            g.redact_input_text(src)
        } else {
            g.redact_output_text(src)
        };
        // The sync per-field redactors bypass `record_execution` entirely
        // (they are not verdicts), so this is the ONLY place an in-process
        // mask can be attributed to the row that applied it — the
        // `kind: "pii"` MCP path the audit chain exists for. Recorded only
        // when this member actually rewrote something.
        if let (Some(audit), Some(r)) = (audit, next.as_ref()) {
            audit.record(
                &m.name,
                phase,
                "masked",
                /* error_type */ None,
                started.elapsed(),
                &r.counts,
            );
        }
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

    /// AISIX-Cloud#1330: the audit log and the metrics sink are gated
    /// independently. `record_execution` used to bail the moment the sink
    /// was absent — a chain built without the metrics layer (every
    /// test-built chain, and any deployment that has not wired it) would
    /// then silently record no audit entry at all.
    #[tokio::test]
    async fn a_block_is_audited_even_with_no_metrics_sink_attached() {
        let audit = Arc::new(GuardrailAuditLog::new());
        let chain = GuardrailChain::new_with_applied(
            vec![(
                "deny-secrets".to_owned(),
                Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("nope")]))
                    as Arc<dyn Guardrail>,
            )],
            vec![AppliedGuardrail {
                kind: "keyword".to_owned(),
                hook: "both".to_owned(),
            }],
        )
        .with_audit_log(Some(Arc::clone(&audit)));
        assert!(chain.sink.is_none(), "no metrics sink for this chain");

        assert!(matches!(
            chain.check_input(&req("nope")).await,
            GuardrailVerdict::Block { .. }
        ));

        let hits = chain.enforced_hits();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].guardrail_name, "deny-secrets");
        assert_eq!(hits[0].hook, "input");
        assert_eq!(hits[0].action, "blocked");
    }

    /// AISIX-Cloud#1365: a fail-CLOSED refusal is an outage, not a policy
    /// decision, and the audit trail has to say which. Before this the two
    /// were the same `action: "blocked"` entry, so a provider outage on a
    /// `fail_open: false` row stamped every request in the window as a
    /// content violation — a wrong answer for a compliance review, not a
    /// missing one.
    #[tokio::test]
    async fn a_fail_closed_block_is_audited_apart_from_a_policy_block() {
        struct Unavailable;
        #[async_trait]
        impl Guardrail for Unavailable {
            fn name(&self) -> &'static str {
                "unavailable"
            }
            async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
                // What every remote guardrail's `handle_failure` returns
                // when it cannot reach its upstream and `fail_open` is off.
                GuardrailVerdict::block_unavailable(
                    "lakera guard unavailable (lakera_timeout)",
                    "lakera_timeout",
                )
            }
        }

        let audit = Arc::new(GuardrailAuditLog::new());
        let chain = GuardrailChain::new_with_applied(
            vec![(
                "lakera-prod".to_owned(),
                Arc::new(Unavailable) as Arc<dyn Guardrail>,
            )],
            vec![AppliedGuardrail {
                kind: "lakera".to_owned(),
                hook: "input".to_owned(),
            }],
        )
        .with_audit_log(Some(Arc::clone(&audit)));

        // The REQUEST still gets refused — separating the two must not
        // weaken the fail-closed guarantee itself.
        assert!(chain.check_input(&req("anything")).await.is_block());

        let hits = chain.enforced_hits();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].guardrail_name, "lakera-prod");
        assert_eq!(hits[0].action, "blocked_unavailable");
        assert_eq!(hits[0].error_type, "lakera_timeout");

        // ...and the ordinary content block is untouched: it keeps the
        // bare `blocked` action and gains no cause, so `action =
        // "blocked"` still means exactly "this content violated policy".
        let audit = Arc::new(GuardrailAuditLog::new());
        let policy = GuardrailChain::new(vec![Arc::new(KeywordBlocklist::new(vec![
            KeywordRule::literal("nope"),
        ])) as Arc<dyn Guardrail>])
        .with_audit_log(Some(Arc::clone(&audit)));
        assert!(policy.check_input(&req("nope")).await.is_block());
        let hits = policy.enforced_hits();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].action, "blocked");
        assert!(hits[0].error_type.is_empty());
    }

    /// The Prometheus `result` label deliberately does NOT split
    /// (AISIX-Cloud#1365): it is a shipped label with operator alerting
    /// attached, so an existing `result="blocked"` alert must keep
    /// counting outages. The cause rides `error_type`, which was already
    /// a label and merely gains values.
    #[tokio::test]
    async fn the_metrics_result_label_keeps_counting_fail_closed_as_blocked() {
        #[derive(Default)]
        struct Recorder(std::sync::Mutex<Vec<(String, String)>>);
        impl GuardrailMetricsSink for Recorder {
            fn record_guardrail_execution(&self, exec: &GuardrailExecution<'_>) {
                self.0.lock().unwrap().push((
                    exec.result.to_owned(),
                    exec.error_type.unwrap_or("none").to_owned(),
                ));
            }
        }
        struct Unavailable;
        #[async_trait]
        impl Guardrail for Unavailable {
            fn name(&self) -> &'static str {
                "unavailable"
            }
            async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
                GuardrailVerdict::block_unavailable("presidio unavailable", "presidio_5xx")
            }
        }

        let sink = Arc::new(Recorder::default());
        let chain = GuardrailChain::new(vec![Arc::new(Unavailable) as Arc<dyn Guardrail>])
            .with_metrics_sink(Some(Arc::clone(&sink) as Arc<dyn GuardrailMetricsSink>));
        assert!(chain.check_input(&req("hi")).await.is_block());

        assert_eq!(
            sink.0.lock().unwrap().as_slice(),
            [("blocked".to_owned(), "presidio_5xx".to_owned())],
        );
    }

    /// An allowed request records nothing: `allowed` and `bypassed` are
    /// not enforcement events, and the two monitor-mode results belong to
    /// `guardrail_monitor_hits` instead.
    #[tokio::test]
    async fn an_allowed_request_records_no_enforced_hit() {
        let audit = Arc::new(GuardrailAuditLog::new());
        let chain = GuardrailChain::new(vec![Arc::new(KeywordBlocklist::new(vec![
            KeywordRule::literal("nope"),
        ])) as Arc<dyn Guardrail>])
        .with_audit_log(Some(Arc::clone(&audit)));

        assert!(matches!(
            chain.check_input(&req("fine")).await,
            GuardrailVerdict::Allow
        ));
        assert!(chain.enforced_hits().is_empty());
    }

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
                ..
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
                ..
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

    // --- per-execution metrics sink (AISIX-Cloud#1076) --------------------

    /// Owned copy of one recorded execution, captured by [`RecordingSink`].
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Recorded {
        guardrail: String,
        kind: String,
        phase: &'static str,
        result: &'static str,
        error_type: Option<String>,
    }

    #[derive(Default)]
    struct RecordingSink(std::sync::Mutex<Vec<Recorded>>);

    impl GuardrailMetricsSink for RecordingSink {
        fn record_guardrail_execution(&self, exec: &GuardrailExecution<'_>) {
            self.0.lock().unwrap().push(Recorded {
                guardrail: exec.guardrail_name.to_owned(),
                kind: exec.kind.to_owned(),
                phase: exec.phase,
                result: exec.result,
                error_type: exec.error_type.map(str::to_owned),
            });
        }
    }

    impl RecordingSink {
        fn take(&self) -> Vec<Recorded> {
            std::mem::take(&mut self.0.lock().unwrap())
        }
    }

    fn sinked_chain(
        members: Vec<(String, Arc<dyn Guardrail>)>,
        applied: Vec<AppliedGuardrail>,
    ) -> (GuardrailChain, Arc<RecordingSink>) {
        let sink = Arc::new(RecordingSink::default());
        let chain = GuardrailChain::new_with_applied(members, applied)
            .with_metrics_sink(Some(sink.clone()));
        (chain, sink)
    }

    fn applied_kw() -> AppliedGuardrail {
        AppliedGuardrail {
            kind: "keyword".to_owned(),
            hook: "both".to_owned(),
        }
    }

    /// Every member consulted by a fold is recorded with its row name, the
    /// `kind` from the 1:1 applied metadata, the fold's phase, and the
    /// enforced result — including the member that short-circuits and the
    /// members before it.
    #[tokio::test]
    async fn sink_records_each_member_with_name_kind_phase_result() {
        let (chain, sink) = sinked_chain(
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
            vec![applied_kw(), applied_kw()],
        );

        assert!(chain
            .check_input_observed(&req("here is AKIAEXAMPLE"))
            .await
            .0
            .is_block());
        assert_eq!(
            sink.take(),
            vec![
                Recorded {
                    guardrail: "pass-through".to_owned(),
                    kind: "keyword".to_owned(),
                    phase: "input",
                    result: "allowed",
                    error_type: None,
                },
                Recorded {
                    guardrail: "block-secrets".to_owned(),
                    kind: "keyword".to_owned(),
                    phase: "input",
                    result: "blocked",
                    error_type: None,
                },
            ],
        );

        // Output fold records phase="output"; a member AFTER the block is
        // not consulted, so it must not be recorded.
        assert!(chain
            .check_output_observed(&resp("the AKIA"))
            .await
            .0
            .is_block());
        let records = sink.take();
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|r| r.phase == "output"));
    }

    /// A fail-open member's `Bypass` records `result=bypassed` with the
    /// bounded failure tag as `error_type`.
    #[tokio::test]
    async fn sink_records_bypass_with_error_type() {
        struct AlwaysBypass;
        #[async_trait]
        impl Guardrail for AlwaysBypass {
            fn name(&self) -> &'static str {
                "always-bypass"
            }
            async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
                GuardrailVerdict::Bypass {
                    reason: "lakera_timeout".into(),
                }
            }
        }
        let (chain, sink) = sinked_chain(
            vec![("remote".to_owned(), Arc::new(AlwaysBypass) as _)],
            vec![AppliedGuardrail {
                kind: "lakera".to_owned(),
                hook: "both".to_owned(),
            }],
        );
        assert!(chain.check_input_observed(&req("hi")).await.0.is_bypass());
        assert_eq!(
            sink.take(),
            vec![Recorded {
                guardrail: "remote".to_owned(),
                kind: "lakera".to_owned(),
                phase: "input",
                result: "bypassed",
                error_type: Some("lakera_timeout".to_owned()),
            }],
        );
    }

    /// The segment pass records its members too: a mask records
    /// `result=masked`; the non-segment fold must NOT also record a
    /// zero-length "allowed" execution for the same member.
    #[tokio::test]
    async fn sink_records_segment_mask_and_skips_segment_members_in_non_segment_fold() {
        let (chain, sink) = sinked_chain(
            vec![
                (
                    "seg-masker".to_owned(),
                    Arc::new(StubSegments {
                        verdict: GuardrailVerdict::Allow,
                        mask: true,
                    }) as Arc<dyn Guardrail>,
                ),
                (
                    "kw".to_owned(),
                    Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal(
                        "never-matches",
                    )])),
                ),
            ],
            vec![
                AppliedGuardrail {
                    kind: "bedrock".to_owned(),
                    hook: "both".to_owned(),
                },
                applied_kw(),
            ],
        );

        // Non-segment pass: only the keyword member records.
        let (v, _) = chain.check_input_non_segment_observed(&req("clean")).await;
        assert_eq!(v, GuardrailVerdict::Allow);
        assert_eq!(
            sink.take(),
            vec![Recorded {
                guardrail: "kw".to_owned(),
                kind: "keyword".to_owned(),
                phase: "input",
                result: "allowed",
                error_type: None,
            }],
        );

        // Segment pass: only the segment member records, as masked.
        let out = chain.moderate_input_segments(&["hello".to_owned()]).await;
        assert_eq!(out.verdict, GuardrailVerdict::Allow);
        assert_eq!(
            sink.take(),
            vec![Recorded {
                guardrail: "seg-masker".to_owned(),
                kind: "bedrock".to_owned(),
                phase: "input",
                result: "masked",
                error_type: None,
            }],
        );
    }

    /// A chain with no sink attached records nothing and behaves
    /// identically — the default for test-built chains.
    #[tokio::test]
    async fn no_sink_is_a_no_op() {
        let chain = GuardrailChain::new(vec![Arc::new(KeywordBlocklist::new(vec![
            KeywordRule::literal("AKIA"),
        ]))]);
        assert!(chain.check_input(&req("AKIA")).await.is_block());
    }
}
