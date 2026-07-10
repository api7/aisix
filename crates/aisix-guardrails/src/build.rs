//! Build a runtime [`GuardrailChain`] from a typed snapshot of
//! `aisix_core::Guardrail` rows.
//!
//! Called by the DP every time the etcd watch supervisor swaps in a
//! new snapshot. The chain composes one runtime guardrail per
//! enabled domain row, in deterministic order so the operator's
//! `reason` strings stay stable across rebuilds.
//!
//! Disabled rows and rows whose `hook_point` excludes both lifecycle
//! sites are dropped here — they don't even allocate. Invalid regex
//! patterns are logged and skipped (the DP refuses to apply a rule
//! it can't compile, so a typo doesn't silently disarm the policy).

use std::sync::{Arc, Mutex};

use aisix_core::models::{
    AisixSnapshot, AppliedGuardrail, Guardrail as DomainGuardrail, GuardrailAttachment,
    GuardrailHookPoint, GuardrailKind, GuardrailMonitorHit, GuardrailScopeType, KeywordPattern,
};
use aisix_core::snapshot::ResourceTable;
use aisix_core::SnapshotHandle;
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;

use crate::index::{GuardrailIndex, RequestContext, ScopeKind};
use crate::keyword::{KeywordBlocklist, KeywordRule};
use crate::pii::{builtin_rule, PiiAction, PiiGuardrail, PiiRule};
use crate::{
    Guardrail, GuardrailChain, GuardrailVerdict, Redaction, SegmentsOutcome, StreamOutputPolicy,
};

/// A snapshot table's guardrail entries in deterministic chain order:
/// `created_at` ascending (RFC3339 strings in a fixed offset compare
/// correctly lexicographically), rows without `created_at` after rows
/// that have it, ties broken by etcd id — so the order is always total.
///
/// `ResourceTable::entries()` is backed by a `DashMap`, whose iteration
/// order is arbitrary and varies run-to-run; building the chain straight
/// off it made "which Block fires first" random when multiple guardrails
/// match (#519 B.4a). The dashboard lists guardrails oldest-first, so the
/// chain evaluates oldest-first too. cp-api doesn't project `created_at`
/// yet — until it does, every row falls back to the id tiebreak, which is
/// still deterministic.
fn sorted_guardrail_entries(
    table: &ResourceTable<DomainGuardrail>,
) -> Vec<Arc<aisix_core::resource::ResourceEntry<DomainGuardrail>>> {
    let mut entries = table.entries();
    entries.sort_by(|a, b| {
        let ka = (
            a.value.created_at.is_none(),
            a.value.created_at.as_deref(),
            a.id.as_str(),
        );
        let kb = (
            b.value.created_at.is_none(),
            b.value.created_at.as_deref(),
            b.id.as_str(),
        );
        ka.cmp(&kb)
    });
    entries
}

/// Build a chain from a snapshot's `guardrails` table.
///
/// Rows are evaluated in deterministic `created_at`-ascending order (see
/// [`sorted_guardrail_entries`]). Each row produces at most one runtime
/// `dyn Guardrail`. Failures (invalid regex, etc.) are logged and the
/// row is skipped — same contract the loader uses for malformed etcd
/// rows.
///
/// `bedrock_endpoint_url` is the deployment-wide override for the
/// AWS Bedrock endpoint URL (sourced from
/// `aisix_core::Config::bedrock_endpoint_url`). `None` → SDK
/// default (real AWS Bedrock); `Some(url)` → every kind=bedrock
/// dispatcher built from this snapshot is pointed at `url`.
pub fn build_chain_from_snapshot(
    table: &ResourceTable<DomainGuardrail>,
    bedrock_endpoint_url: Option<&str>,
) -> GuardrailChain {
    let mut chain: Vec<(String, Arc<dyn Guardrail>)> = Vec::new();
    // `applied` mirrors `chain` 1:1 — the `{kind, hook}` of each member that
    // actually materialised, for applied-guardrail telemetry (#379). Pushed
    // only on the `Ok(Some)` path so inert/invalid rows (which never join the
    // chain) never show up as "governed this request".
    let mut applied: Vec<AppliedGuardrail> = Vec::new();

    let entries = sorted_guardrail_entries(table);
    for entry in entries.iter() {
        let row = &entry.value;
        if !row.enabled {
            continue;
        }
        match build_one(row, bedrock_endpoint_url) {
            Ok(Some(g)) => {
                chain.push((row.name.clone(), g));
                applied.push(applied_for(row));
            }
            Ok(None) => {
                // Rule was technically valid but inert (e.g. empty
                // keyword list). Skip silently — operators see this
                // shape when they're staging a rule.
            }
            Err(err) => {
                tracing::warn!(
                    name = %row.name,
                    id = %entry.id,
                    error = %err,
                    "skipping guardrail with invalid config",
                );
            }
        }
    }

    GuardrailChain::new_with_applied(chain, applied)
}

/// The `{kind, hook}` telemetry descriptor for a guardrail row that
/// materialised into a chain (#379). Captured here — the build points are the
/// only place the domain row's `kind` + `hook_point` are in scope alongside
/// the runtime guardrail. `hook` is the configured hook_point, not a
/// per-request verdict (v1 records the attached set, not which side fired).
fn applied_for(row: &DomainGuardrail) -> AppliedGuardrail {
    AppliedGuardrail {
        kind: row.config.kind_str().to_owned(),
        hook: row.hook_point.as_str().to_owned(),
    }
}

/// Build the runtime guardrail for a row, applying its `enforcement_mode`
/// and `mandatory` policy.
///
/// `enforcement_mode` `block` (the default) returns the guardrail as-is;
/// `monitor` wraps it in [`MonitorGuardrail`] so it observes violations
/// without blocking. `mandatory: true` wraps the result in
/// [`MandatoryGuardrail`] so a remote guardrail that can't evaluate blocks
/// the request instead of failing open. `mandatory` is applied outermost:
/// a monitored guardrail still never blocks on its *content* decisions, but
/// being unavailable is an infra failure that mandatory makes fatal.
fn build_one(
    row: &DomainGuardrail,
    bedrock_endpoint_url: Option<&str>,
) -> Result<Option<Arc<dyn Guardrail>>, BuildError> {
    Ok(build_one_inner(row, bedrock_endpoint_url)?
        .map(|g| apply_enforcement_mode(row, g))
        .map(|g| apply_mandatory(row, g)))
}

/// Wrap `inner` in [`MandatoryGuardrail`] when `row.mandatory` is set, so a
/// fail-open remote guardrail that couldn't reach its upstream blocks
/// instead of bypassing. A no-op for the default (`mandatory: false`) and
/// for guardrails that never emit `Bypass` (e.g. keyword) — so it's only
/// ever paid for by rows that opt in.
fn apply_mandatory(row: &DomainGuardrail, inner: Arc<dyn Guardrail>) -> Arc<dyn Guardrail> {
    if row.mandatory {
        Arc::new(MandatoryGuardrail {
            row_name: row.name.clone(),
            inner,
        })
    } else {
        inner
    }
}

/// Wrap `inner` per the row's `enforcement_mode`. See [`build_one`].
fn apply_enforcement_mode(row: &DomainGuardrail, inner: Arc<dyn Guardrail>) -> Arc<dyn Guardrail> {
    match row.enforcement_mode.as_str() {
        "block" => inner,
        "monitor" => Arc::new(MonitorGuardrail {
            row_name: row.name.clone(),
            inner,
        }),
        other => {
            tracing::warn!(
                guardrail_name = %row.name,
                enforcement_mode = %other,
                "unknown enforcement_mode; treating as 'block'",
            );
            inner
        }
    }
}

fn build_one_inner(
    row: &DomainGuardrail,
    bedrock_endpoint_url: Option<&str>,
) -> Result<Option<Arc<dyn Guardrail>>, BuildError> {
    match &row.config {
        GuardrailKind::Keyword(cfg) => {
            if cfg.patterns.is_empty() {
                return Ok(None);
            }
            let mut rules = Vec::with_capacity(cfg.patterns.len());
            for p in &cfg.patterns {
                let rule = match p {
                    KeywordPattern::Literal(s) => KeywordRule::literal(s.clone()),
                    KeywordPattern::Regex(s) => {
                        KeywordRule::regex(s).map_err(|e| BuildError::InvalidRegex {
                            pattern: s.clone(),
                            source: e,
                        })?
                    }
                };
                rules.push(rule);
            }
            // Map domain hook_point onto the runtime KeywordBlocklist
            // constructors. `Both` is the default; the input/output
            // narrowed forms exist for rules that are too expensive
            // to run on the other side.
            let blocklist = match row.hook_point {
                GuardrailHookPoint::Input => KeywordBlocklist::input_only(rules),
                GuardrailHookPoint::Output => KeywordBlocklist::output_only(rules),
                GuardrailHookPoint::Both => KeywordBlocklist::new(rules),
            };
            Ok(Some(Arc::new(blocklist)))
        }
        GuardrailKind::Pii(cfg) => {
            if cfg.detectors.is_empty() && cfg.custom_patterns.is_empty() {
                return Ok(None);
            }
            let default_action =
                PiiAction::parse(&cfg.default_action).ok_or_else(|| BuildError::InvalidValue {
                    field: "default_action",
                    value: cfg.default_action.clone(),
                })?;
            let mut rules: Vec<PiiRule> =
                Vec::with_capacity(cfg.detectors.len() + cfg.custom_patterns.len());
            for d in &cfg.detectors {
                let action = match d.action.as_deref() {
                    None => default_action,
                    Some(s) => PiiAction::parse(s).ok_or_else(|| BuildError::InvalidValue {
                        field: "detectors[].action",
                        value: s.to_owned(),
                    })?,
                };
                let rule = builtin_rule(&d.detector_type, action).ok_or_else(|| {
                    BuildError::InvalidValue {
                        field: "detectors[].type",
                        value: d.detector_type.clone(),
                    }
                })?;
                rules.push(rule);
            }
            for p in &cfg.custom_patterns {
                let action = match p.action.as_deref() {
                    None => default_action,
                    Some(s) => PiiAction::parse(s).ok_or_else(|| BuildError::InvalidValue {
                        field: "custom_patterns[].action",
                        value: s.to_owned(),
                    })?,
                };
                let rule = PiiRule::new(p.name.clone(), &p.regex, action, None).map_err(|e| {
                    BuildError::InvalidRegex {
                        pattern: p.regex.clone(),
                        source: e,
                    }
                })?;
                rules.push(rule);
            }
            let on_exceeded_fail_open = cfg.on_buffer_exceeded == "fail_open";
            let g = PiiGuardrail::new(
                rules,
                row.hook_point,
                usize::try_from(cfg.max_buffer_bytes).unwrap_or(usize::MAX),
                on_exceeded_fail_open,
            );
            Ok(Some(Arc::new(g)))
        }
        #[cfg(feature = "bedrock")]
        GuardrailKind::Bedrock(cfg) => {
            // Phase 2: build the AWS-SDK-backed dispatcher. cp-api
            // already decrypted the secret at projection time, so
            // the BedrockConfig in the snapshot carries plaintext
            // credentials. The endpoint URL is forwarded from
            // bootstrap config (Config.bedrock_endpoint_url).
            let g = crate::bedrock::BedrockGuardrail::new(
                row.name.clone(),
                cfg,
                row.hook_point,
                row.fail_open,
                bedrock_endpoint_url.map(str::to_owned),
            );
            Ok(Some(Arc::new(g)))
        }
        #[cfg(not(feature = "bedrock"))]
        GuardrailKind::Bedrock(_) => {
            // Built without --features bedrock. Skip + warn so an
            // operator who happens to deploy a Bedrock row to a
            // pruned-build DP sees the misconfig in logs.
            Err(BuildError::FeatureDisabled("bedrock"))
        }
        #[cfg(feature = "azure-content-safety")]
        GuardrailKind::AzureContentSafety(cfg) => {
            // P1: HTTP-based Prompt Shield dispatcher. cp-api already
            // decrypted the api_key at projection time; the config carries
            // plaintext. No deployment-wide endpoint override needed —
            // the endpoint is per-row (each customer has their own Azure CS
            // resource).
            let g = crate::prompt_shield::PromptShieldGuardrail::new(
                row.name.clone(),
                cfg,
                row.hook_point,
                row.fail_open,
            );
            Ok(Some(Arc::new(g)))
        }
        #[cfg(not(feature = "azure-content-safety"))]
        GuardrailKind::AzureContentSafety(_) => {
            // Built without --features azure-content-safety. Skip + warn.
            Err(BuildError::FeatureDisabled("azure-content-safety"))
        }
        #[cfg(feature = "azure-content-safety")]
        GuardrailKind::AzureContentSafetyTextModeration(cfg) => {
            // P2: HTTP-based text:analyze dispatcher. cp-api already
            // decrypted the api_key at projection time; the config carries
            // plaintext. Endpoint is per-row (each customer's own resource).
            let g = crate::text_moderation::TextModerationGuardrail::new(
                row.name.clone(),
                cfg,
                row.hook_point,
                row.fail_open,
            );
            Ok(Some(Arc::new(g)))
        }
        #[cfg(not(feature = "azure-content-safety"))]
        GuardrailKind::AzureContentSafetyTextModeration(_) => {
            Err(BuildError::FeatureDisabled("azure-content-safety"))
        }
        #[cfg(feature = "aliyun-text-moderation")]
        GuardrailKind::AliyunTextModeration(cfg) => {
            // #603: HTTP-based TextModerationPlus dispatcher. cp-api already
            // decrypted the access_key_secret at projection time; the config
            // carries plaintext. Endpoint is per-row (derived from the row's
            // region, or an explicit override for tests/dev).
            let g = crate::aliyun::AliyunTextModerationGuardrail::new(
                row.name.clone(),
                cfg,
                row.hook_point,
                row.fail_open,
            );
            Ok(Some(Arc::new(g)))
        }
        #[cfg(not(feature = "aliyun-text-moderation"))]
        GuardrailKind::AliyunTextModeration(_) => {
            Err(BuildError::FeatureDisabled("aliyun-text-moderation"))
        }
        #[cfg(feature = "lakera")]
        GuardrailKind::Lakera(cfg) => {
            // #52: HTTP-based /v2/guard dispatcher. cp-api already decrypted
            // the api_key at projection time; the config carries plaintext.
            // Endpoint is per-row (default api.lakera.ai, overridable for
            // regional/self-hosted deployments and tests).
            let g = crate::lakera::LakeraGuardrail::new(
                row.name.clone(),
                cfg,
                row.hook_point,
                row.fail_open,
            );
            Ok(Some(Arc::new(g)))
        }
        #[cfg(not(feature = "lakera"))]
        GuardrailKind::Lakera(_) => Err(BuildError::FeatureDisabled("lakera")),
        #[cfg(feature = "openai-moderation")]
        GuardrailKind::OpenaiModeration(cfg) => {
            // #52: HTTP-based /moderations dispatcher. cp-api already
            // decrypted the api_key at projection time; the config carries
            // plaintext. Endpoint is per-row (default api.openai.com/v1).
            // Moderation scores are 0..=1; a threshold outside that range
            // can never (or always) fire, so reject the row rather than
            // silently running a policy the operator didn't intend.
            for (category, threshold) in &cfg.category_thresholds {
                if !(0.0..=1.0).contains(threshold) {
                    return Err(BuildError::InvalidValue {
                        field: "category_thresholds",
                        value: format!("{category}={threshold}"),
                    });
                }
            }
            let g = crate::openai_moderation::OpenaiModerationGuardrail::new(
                row.name.clone(),
                cfg,
                row.hook_point,
                row.fail_open,
            );
            Ok(Some(Arc::new(g)))
        }
        #[cfg(not(feature = "openai-moderation"))]
        GuardrailKind::OpenaiModeration(_) => Err(BuildError::FeatureDisabled("openai-moderation")),
        #[cfg(feature = "presidio")]
        GuardrailKind::Presidio(cfg) => {
            // #52: analyze→anonymize dispatcher against customer-run
            // Presidio containers (no vendor secret). The enum-ish fields
            // (`default_action`, per-entity actions, `operator`) are
            // resolved here so a typo can't silently weaken the policy.
            let default_action =
                PiiAction::parse(&cfg.default_action).ok_or_else(|| BuildError::InvalidValue {
                    field: "default_action",
                    value: cfg.default_action.clone(),
                })?;
            let mut entity_actions = std::collections::BTreeMap::new();
            for e in &cfg.entities {
                if let Some(s) = e.action.as_deref() {
                    let action = PiiAction::parse(s).ok_or_else(|| BuildError::InvalidValue {
                        field: "entities[].action",
                        value: s.to_owned(),
                    })?;
                    entity_actions.insert(e.entity_type.to_uppercase(), action);
                }
            }
            let anonymizers = crate::presidio::operator_config(&cfg.operator).ok_or_else(|| {
                BuildError::InvalidValue {
                    field: "operator",
                    value: cfg.operator.clone(),
                }
            })?;
            let g = crate::presidio::PresidioGuardrail::new(
                row.name.clone(),
                cfg,
                row.hook_point,
                row.fail_open,
                default_action,
                entity_actions,
                anonymizers,
            );
            Ok(Some(Arc::new(g)))
        }
        #[cfg(not(feature = "presidio"))]
        GuardrailKind::Presidio(_) => Err(BuildError::FeatureDisabled("presidio")),
    }
}

#[derive(Debug, thiserror::Error)]
enum BuildError {
    #[error("invalid regex {pattern:?}: {source}")]
    InvalidRegex {
        pattern: String,
        source: regex::Error,
    },
    /// An enum-ish config field carrying a value the DP doesn't know
    /// (unknown built-in detector id, unrecognised action). The row is
    /// skipped + warned rather than silently running a weaker policy.
    #[error("invalid {field} value {value:?}")]
    InvalidValue { field: &'static str, value: String },
    /// A guardrail kind whose runtime dispatch was compiled out via
    /// feature flags (e.g. a pruned build that excluded `--features bedrock`
    /// or `--features azure-content-safety`). The chain treats the row as
    /// disabled and the warn log surfaces the kind name so the misconfig is visible.
    ///
    /// Always declared in the enum (not behind `#[cfg]`) so `build_one` can
    /// reference it from any `not(feature = "…")` arm. When all features are
    /// enabled (the default), the variant exists but is never constructed —
    /// the dead_code lint is suppressed below.
    #[allow(dead_code)]
    #[error("guardrail kind {0:?} not compiled into this build; treating row as disabled")]
    FeatureDisabled(&'static str),
}

/// `enforcement_mode: monitor` decorator. Runs the wrapped guardrail exactly
/// as configured but never blocks: a `Block` verdict is logged (the operator's
/// audit signal — "this rule WOULD have blocked") and downgraded to `Allow`.
/// `Allow` and `Bypass` pass through unchanged.
///
/// `runs_on_output` delegates to the inner guardrail so a monitor-mode output
/// rule still gets its `check_output` called and can record what it observed.
/// `stream_output_policy` is forced to `EndOfStreamCheck`, though: a guardrail
/// that can never block must not make the streamed response hold back —
/// monitor mode observes at end-of-stream without adding hold-back latency,
/// and it can never weaken a *blocking* peer's hold-back (the chain folds to
/// the strictest member).
struct MonitorGuardrail {
    row_name: String,
    inner: Arc<dyn Guardrail>,
}

impl MonitorGuardrail {
    /// Log the mask counts a monitor-mode redacting guardrail WOULD have
    /// applied. Counts carry detector names only, never matched values.
    fn observe_redaction(&self, hook: &'static str, r: Option<Redaction>) {
        if let Some(r) = r {
            tracing::info!(
                guardrail_name = %self.row_name,
                hook,
                counts = ?r.counts,
                "guardrail in monitor mode observed maskable spans; not redacting (enforcement_mode=monitor)",
            );
        }
    }

    fn observe(&self, hook: &'static str, verdict: GuardrailVerdict) -> GuardrailVerdict {
        match verdict {
            GuardrailVerdict::Block { reason, .. } => {
                tracing::info!(
                    guardrail_name = %self.row_name,
                    hook,
                    reason = %reason,
                    "guardrail in monitor mode observed a violation; not blocking (enforcement_mode=monitor)",
                );
                GuardrailVerdict::Allow
            }
            other => other,
        }
    }

    /// `would_block` telemetry hit for a downgraded Block (AISIX-Cloud#562).
    fn would_block_hit(&self, hook: &'static str, reason: &str) -> GuardrailMonitorHit {
        GuardrailMonitorHit {
            guardrail_name: self.row_name.clone(),
            hook: hook.to_owned(),
            action: "would_block".to_owned(),
            reason: reason.to_owned(),
            counts: std::collections::BTreeMap::new(),
        }
    }

    /// `would_mask` telemetry hit for suppressed mask counts.
    fn would_mask_hit(
        &self,
        hook: &'static str,
        counts: std::collections::BTreeMap<String, u32>,
    ) -> GuardrailMonitorHit {
        GuardrailMonitorHit {
            guardrail_name: self.row_name.clone(),
            hook: hook.to_owned(),
            action: "would_mask".to_owned(),
            reason: String::new(),
            counts,
        }
    }

    /// Downgrade a verdict, recording a `would_block` hit alongside the
    /// existing ops-log line.
    fn observe_hit(
        &self,
        hook: &'static str,
        verdict: GuardrailVerdict,
        hits: &mut Vec<GuardrailMonitorHit>,
    ) -> GuardrailVerdict {
        if let GuardrailVerdict::Block { ref reason, .. } = verdict {
            hits.push(self.would_block_hit(hook, reason));
        }
        self.observe(hook, verdict)
    }

    /// Observe a segment outcome (AISIX-Cloud#562): a Block downgrades to
    /// Allow with a `would_block` hit; an inner mask is suppressed (never
    /// written back) with a `would_mask` hit carrying the provider's
    /// entity counts. Bypass passes through — monitor mode doesn't change
    /// availability semantics.
    fn observe_segments(&self, hook: &'static str, outcome: SegmentsOutcome) -> SegmentsOutcome {
        let mut hits = outcome.monitor_hits;
        if outcome.masked.is_some() {
            tracing::info!(
                guardrail_name = %self.row_name,
                hook,
                counts = ?outcome.counts,
                "guardrail in monitor mode observed maskable spans; not redacting (enforcement_mode=monitor)",
            );
            hits.push(self.would_mask_hit(hook, outcome.counts));
        }
        let verdict = self.observe_hit(hook, outcome.verdict, &mut hits);
        SegmentsOutcome {
            verdict,
            masked: None,
            counts: std::collections::BTreeMap::new(),
            monitor_hits: hits,
        }
    }

    /// Probe the inner SYNC redactor (kind=pii) with the hook's scan text
    /// and record what it would have masked. Redaction stays suppressed —
    /// this only recovers the counts for telemetry. Segment moderators
    /// (bedrock/lakera/presidio) report through the segment pass instead.
    fn probe_redaction(
        &self,
        hook: &'static str,
        redacts: bool,
        redact: impl FnOnce() -> Option<Redaction>,
        hits: &mut Vec<GuardrailMonitorHit>,
    ) {
        if !redacts {
            return;
        }
        let r = redact();
        if let Some(ref red) = r {
            hits.push(self.would_mask_hit(hook, red.counts.clone()));
        }
        self.observe_redaction(hook, r);
    }
}

#[async_trait]
impl Guardrail for MonitorGuardrail {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        self.observe("input", self.inner.check_input(req).await)
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        self.observe("output", self.inner.check_output(resp).await)
    }

    async fn check_input_observed(
        &self,
        req: &ChatFormat,
    ) -> (GuardrailVerdict, Vec<GuardrailMonitorHit>) {
        let mut hits = Vec::new();
        let verdict = self.observe_hit("input", self.inner.check_input(req).await, &mut hits);
        // Recover the would-mask counts the suppressed sync redactor
        // (kind=pii) would have produced, from the same text its
        // check_input scans.
        let text: String = req
            .messages
            .iter()
            .map(crate::message_scan_text)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        self.probe_redaction(
            "input",
            self.inner.redacts_input() && !text.is_empty(),
            || self.inner.redact_input_text(&text),
            &mut hits,
        );
        (verdict, hits)
    }

    async fn check_output_observed(
        &self,
        resp: &ChatResponse,
    ) -> (GuardrailVerdict, Vec<GuardrailMonitorHit>) {
        let mut hits = Vec::new();
        let verdict = self.observe_hit("output", self.inner.check_output(resp).await, &mut hits);
        let text = resp.guardrail_output_text();
        self.probe_redaction(
            "output",
            self.inner.redacts_output() && !text.is_empty(),
            || self.inner.redact_output_text(&text),
            &mut hits,
        );
        (verdict, hits)
    }

    /// Delegate so a monitored segment moderator (bedrock/lakera/presidio)
    /// is consulted through the segment pass — ONE provider call whose
    /// verdict AND mask are observed with full fidelity — instead of the
    /// blob path, where a maskable outcome degrades to a would-block.
    fn moderates_segments(&self) -> bool {
        self.inner.moderates_segments()
    }

    async fn moderate_input_segments(&self, texts: &[String]) -> SegmentsOutcome {
        self.observe_segments("input", self.inner.moderate_input_segments(texts).await)
    }

    async fn moderate_output_segments(&self, texts: &[String]) -> SegmentsOutcome {
        self.observe_segments("output", self.inner.moderate_output_segments(texts).await)
    }

    fn stream_output_policy(&self) -> StreamOutputPolicy {
        StreamOutputPolicy::EndOfStreamCheck
    }

    fn runs_on_output(&self) -> bool {
        self.inner.runs_on_output()
    }

    // Monitor mode observes without modifying: redaction is suppressed the
    // same way Block verdicts are downgraded. The would-be mask counts are
    // logged so operators can stage a redaction rule and audit its impact
    // before enforcing it.
    fn redacts_input(&self) -> bool {
        false
    }

    fn redacts_output(&self) -> bool {
        false
    }

    fn redact_input_text(&self, text: &str) -> Option<Redaction> {
        self.observe_redaction("input", self.inner.redact_input_text(text));
        None
    }

    fn redact_output_text(&self, text: &str) -> Option<Redaction> {
        self.observe_redaction("output", self.inner.redact_output_text(text));
        None
    }
}

/// `mandatory: true` decorator. A remote guardrail that can't reach its
/// upstream returns `Bypass` when `fail_open` is set — the request proceeds
/// unscanned. For a guardrail an operator marked mandatory that fail-open is
/// the wrong call: the point of `mandatory` is that the rule MUST evaluate,
/// so an unreachable upstream is a hard failure. This decorator upgrades a
/// `Bypass` verdict to `Block`, overriding `fail_open` on the failure path.
/// `Allow` and `Block` pass through unchanged, and only remote guardrails
/// ever emit `Bypass`, so keyword rows wrapped here are behaviourally
/// untouched.
///
/// Stream policy + `runs_on_output` delegate to the inner guardrail so the
/// decorator doesn't change hold-back behaviour — it only rewrites the
/// verdict a failed evaluation produces.
struct MandatoryGuardrail {
    row_name: String,
    inner: Arc<dyn Guardrail>,
}

impl MandatoryGuardrail {
    fn enforce(&self, hook: &'static str, verdict: GuardrailVerdict) -> GuardrailVerdict {
        match verdict {
            GuardrailVerdict::Bypass { reason } => {
                tracing::warn!(
                    guardrail_name = %self.row_name,
                    hook,
                    reason = %reason,
                    "mandatory guardrail could not evaluate; blocking (mandatory=true overrides fail_open)",
                );
                // Carry the row name so downstream handlers can name the
                // guardrail in the 422 envelope (#519 B.4b) — `block()` would
                // drop it to `None` and surface an unnamed content-filter block.
                GuardrailVerdict::Block {
                    reason: format!("mandatory guardrail unavailable: {reason}"),
                    guardrail_name: Some(self.row_name.clone()),
                }
            }
            other => other,
        }
    }
}

#[async_trait]
impl Guardrail for MandatoryGuardrail {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        self.enforce("input", self.inner.check_input(req).await)
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        self.enforce("output", self.inner.check_output(resp).await)
    }

    fn stream_output_policy(&self) -> StreamOutputPolicy {
        self.inner.stream_output_policy()
    }

    fn runs_on_output(&self) -> bool {
        self.inner.runs_on_output()
    }

    // Mandatory only rewrites the failure (Bypass) verdict; redaction
    // passes straight through to the inner guardrail.
    fn redacts_input(&self) -> bool {
        self.inner.redacts_input()
    }

    fn redacts_output(&self) -> bool {
        self.inner.redacts_output()
    }

    fn redact_input_text(&self, text: &str) -> Option<Redaction> {
        self.inner.redact_input_text(text)
    }

    fn redact_output_text(&self, text: &str) -> Option<Redaction> {
        self.inner.redact_output_text(text)
    }
}

/// Adapter that wraps a snapshot handle and rebuilds the runtime
/// chain whenever the snapshot pointer changes. The chat handler
/// holds an `Arc<dyn Guardrail>` pointing at this; it never sees
/// the rebuild.
///
/// Cheap path (cache hit): one atomic load + one pointer compare,
/// then a clone of an `Arc<GuardrailChain>`. Rebuild path (cache
/// miss): runs through the entries table and recompiles regexes.
/// Compilation only happens on the first call after each snapshot
/// store from the etcd supervisor — typical run is one or zero
/// rebuilds per minute even on a chatty configuration.
///
/// `bedrock_endpoint_url` is captured at construct time and reused
/// on every rebuild; this is a deployment-wide setting (sourced
/// from `aisix_core::Config::bedrock_endpoint_url`) and doesn't
/// change while the DP is running.
pub struct LiveGuardrailChain {
    snapshot: SnapshotHandle<AisixSnapshot>,
    bedrock_endpoint_url: Option<String>,
    cache: Mutex<Cache>,
}

struct Cache {
    last_version: u64,
    chain: Arc<GuardrailChain>,
}

impl LiveGuardrailChain {
    pub fn new(
        snapshot: SnapshotHandle<AisixSnapshot>,
        bedrock_endpoint_url: Option<String>,
    ) -> Arc<Self> {
        // Read version before load so that a concurrent store() between
        // the two reads causes current() to see a version bump and rebuild,
        // rather than caching stale data under the new version.
        let last_version = snapshot.version();
        let snap = snapshot.load();
        let chain = Arc::new(build_chain_from_snapshot(
            &snap.guardrails,
            bedrock_endpoint_url.as_deref(),
        ));
        Arc::new(Self {
            snapshot,
            bedrock_endpoint_url,
            cache: Mutex::new(Cache {
                last_version,
                chain,
            }),
        })
    }

    fn current(&self) -> Arc<GuardrailChain> {
        let cur_version = self.snapshot.version();
        let mut cache = self
            .cache
            .lock()
            .expect("LiveGuardrailChain mutex poisoned");
        if cache.last_version != cur_version {
            let snap = self.snapshot.load();
            cache.chain = Arc::new(build_chain_from_snapshot(
                &snap.guardrails,
                self.bedrock_endpoint_url.as_deref(),
            ));
            cache.last_version = cur_version;
        }
        Arc::clone(&cache.chain)
    }
}

#[async_trait]
impl Guardrail for LiveGuardrailChain {
    fn name(&self) -> &'static str {
        "live_chain"
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        self.current().check_input(req).await
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        self.current().check_output(resp).await
    }

    /// Delegate streamed-output gating to the live inner chain so this exported
    /// wrapper can't silently diverge from `GuardrailChain`'s hold-back
    /// semantics if it is ever used directly as a streaming chain (#466).
    /// Without these it would inherit the trait defaults (`BufferFull` +
    /// `runs_on_output() == true`) and always hold back, ignoring its inner
    /// members' hooks.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        self.current().stream_output_policy()
    }

    fn runs_on_output(&self) -> bool {
        self.current().runs_on_output()
    }

    fn redacts_input(&self) -> bool {
        self.current().redacts_input()
    }

    fn redacts_output(&self) -> bool {
        self.current().redacts_output()
    }

    fn redact_input_text(&self, text: &str) -> Option<Redaction> {
        self.current().redact_input_text(text)
    }

    fn redact_output_text(&self, text: &str) -> Option<Redaction> {
        self.current().redact_output_text(text)
    }
}

// ---------------------------------------------------------------------------
// GuardrailIndex builder
// ---------------------------------------------------------------------------

/// Build a [`GuardrailIndex`] from a snapshot's `guardrails` and
/// `guardrail_attachments` tables.
///
/// For each enabled attachment, the function:
/// 1. Looks up the guardrail definition by `attachment.guardrail_id`.
/// 2. Skips the attachment if the guardrail is disabled or unknown.
/// 3. Builds the runtime guardrail via [`build_one`] (same path as
///    `build_chain_from_snapshot`).
/// 4. Adds an entry to the index carrying the attachment's scope +
///    priority.
///
/// The resulting index is pre-sorted by priority (descending) so
/// `GuardrailIndex::resolve` can walk it linearly.
pub fn build_index_from_snapshot(
    guardrails: &ResourceTable<DomainGuardrail>,
    attachments: &ResourceTable<GuardrailAttachment>,
    bedrock_endpoint_url: Option<&str>,
) -> GuardrailIndex {
    let mut entries = Vec::new();
    // Track guardrail IDs that have ANY attachment record (enabled or not).
    // The backward-compat fallback below only fires for guardrails that have
    // zero attachment rows — operators who explicitly disabled an attachment
    // are expressing intent; we must not override it with the env-scope fallback.
    let mut attached_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Deterministic attachment order: `GuardrailIndex::new` sorts by
    // (priority desc, scope-specificity desc) with a STABLE sort, so
    // insertion order decides the remaining ties. `entries()` is DashMap-
    // backed (arbitrary, run-to-run-varying order) — sort by id so equal-
    // priority/equal-specificity entries resolve in a stable order too
    // (#519 B.4a, same bug class as the chain-build path).
    let mut attachment_entries = attachments.entries();
    attachment_entries.sort_by(|a, b| a.id.cmp(&b.id));

    for attachment_arc in attachment_entries.iter() {
        let attachment = &attachment_arc.value;
        // Track ALL attachment references (enabled or not) so the backward-compat
        // fallback below treats "has an explicit attachment" as opt-in to P0c
        // attachment semantics — even if all attachments are currently disabled.
        attached_ids.insert(attachment.guardrail_id.clone());

        if !attachment.enabled {
            continue;
        }

        let gid = &attachment.guardrail_id;
        let guardrail_arc = match guardrails.get_by_id(gid) {
            Some(e) => e,
            None => {
                tracing::warn!(
                    attachment_id = %attachment_arc.id,
                    guardrail_id = %gid,
                    "attachment references unknown guardrail; skipping",
                );
                continue;
            }
        };

        let row = &guardrail_arc.value;
        if !row.enabled {
            continue;
        }

        let runtime_guardrail = match build_one(row, bedrock_endpoint_url) {
            Ok(Some(g)) => g,
            Ok(None) => continue, // inert (e.g. empty keyword list)
            Err(err) => {
                tracing::warn!(
                    guardrail_id = %gid,
                    error = %err,
                    "skipping guardrail with invalid config in index build",
                );
                continue;
            }
        };

        let scope_kind = match attachment.scope_type {
            GuardrailScopeType::Env => ScopeKind::Env,
            GuardrailScopeType::Model => ScopeKind::Model,
            GuardrailScopeType::ApiKey => ScopeKind::ApiKey,
            GuardrailScopeType::Team => ScopeKind::Team,
        };

        entries.push(GuardrailIndex::push_entry(
            gid.clone(),
            row.name.clone(),
            scope_kind,
            attachment.scope_id.clone(),
            attachment.priority,
            runtime_guardrail,
            applied_for(row),
        ));
    }

    // Backward compat: a guardrail definition that has no enabled attachment
    // fires on every request in the env at priority 0 (same as the pre-P0c
    // behavior where all guardrails in the snapshot were applied globally).
    // This covers the rolling-upgrade window where the DP has been updated to
    // P0c but the CP hasn't yet written attachment rows for existing guardrails.
    //
    // TODO(P0c-cleanup): Remove this block once the CP is fully rolled out and
    // guaranteed to write at least one attachment row for every guardrail
    // (tracked in https://github.com/api7/ai-gateway/issues/417).
    // After removal, a guardrail with zero attachment rows is a silent no-op —
    // operators must explicitly attach it to a scope.
    // Same deterministic created_at-ascending order as the chain-build
    // path: these implicit entries all share priority 0 + env scope, so
    // without a pre-sort their relative order — and which Block fires
    // first — would follow the DashMap's arbitrary iteration (#519 B.4a).
    for guardrail_arc in sorted_guardrail_entries(guardrails) {
        if attached_ids.contains(guardrail_arc.id.as_str()) {
            continue; // explicit attachment governs this guardrail
        }
        let row = &guardrail_arc.value;
        if !row.enabled {
            continue;
        }
        match build_one(row, bedrock_endpoint_url) {
            Ok(Some(g)) => {
                tracing::info!(
                    guardrail_id = %guardrail_arc.id,
                    guardrail_name = %row.name,
                    "guardrail has no attachment rows; applying as implicit env-scope at priority 0 (backward-compat rolling-upgrade window)",
                );
                entries.push(GuardrailIndex::push_entry(
                    guardrail_arc.id.clone(),
                    row.name.clone(),
                    ScopeKind::Env,
                    None,
                    0,
                    g,
                    applied_for(row),
                ));
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(
                    guardrail_id = %guardrail_arc.id,
                    error = %err,
                    "skipping guardrail with invalid config (no-attachment backward-compat path)",
                );
            }
        }
    }

    GuardrailIndex::from_entries(entries)
}

// ---------------------------------------------------------------------------
// LiveGuardrailIndex — lazy-rebuild adapter over a snapshot handle
// ---------------------------------------------------------------------------

/// Wraps a snapshot handle and rebuilds the runtime index whenever the
/// snapshot pointer changes. The proxy chat handler calls `resolve(ctx)`
/// on each request to get the applicable `GuardrailChain`.
///
/// Rebuild semantics are identical to `LiveGuardrailChain`: one atomic
/// load + one version compare on the hot path; a full index build (linear
/// in the number of attachment rows) only on the first call after each
/// snapshot swap.
pub struct LiveGuardrailIndex {
    snapshot: SnapshotHandle<AisixSnapshot>,
    bedrock_endpoint_url: Option<String>,
    cache: Mutex<IndexCache>,
}

struct IndexCache {
    last_version: u64,
    index: Arc<GuardrailIndex>,
}

impl LiveGuardrailIndex {
    pub fn new(
        snapshot: SnapshotHandle<AisixSnapshot>,
        bedrock_endpoint_url: Option<String>,
    ) -> Arc<Self> {
        // Read version before load — same ordering discipline as LiveGuardrailChain.
        let last_version = snapshot.version();
        let snap = snapshot.load();
        let index = Arc::new(build_index_from_snapshot(
            &snap.guardrails,
            &snap.guardrail_attachments,
            bedrock_endpoint_url.as_deref(),
        ));
        Arc::new(Self {
            snapshot,
            bedrock_endpoint_url,
            cache: Mutex::new(IndexCache {
                last_version,
                index,
            }),
        })
    }

    fn current(&self) -> Arc<GuardrailIndex> {
        let cur_version = self.snapshot.version();

        // Fast path: return cached index without building.
        {
            let cache = self
                .cache
                .lock()
                .expect("LiveGuardrailIndex mutex poisoned");
            if cache.last_version == cur_version {
                return Arc::clone(&cache.index);
            }
        }

        // Build the new index OUTSIDE the lock so a panic (e.g. from a
        // badly-behaved regex engine) does not poison the mutex.
        let snap = self.snapshot.load();
        let new_index = Arc::new(build_index_from_snapshot(
            &snap.guardrails,
            &snap.guardrail_attachments,
            self.bedrock_endpoint_url.as_deref(),
        ));

        // Re-acquire and store. A concurrent rebuild (rare) is harmless —
        // both produce equivalent indexes from the same snapshot version.
        let mut cache = self
            .cache
            .lock()
            .expect("LiveGuardrailIndex mutex poisoned");
        if cache.last_version != cur_version {
            cache.index = new_index;
            cache.last_version = cur_version;
        }
        Arc::clone(&cache.index)
    }

    /// Resolve the guardrail chain applicable to `ctx`.
    ///
    /// Cheap on the cache-hit path (one lock acquire + version compare +
    /// arc clone + `O(n)` linear walk over attachment rows). Rebuilds only
    /// on snapshot version change.
    pub fn resolve(&self, ctx: &RequestContext<'_>) -> GuardrailChain {
        self.current().resolve(ctx)
    }

    /// `true` when the resolved index has no guardrail entries — neither
    /// from explicit attachment rows nor from the backward-compat implicit
    /// env-scope fallback for no-attachment guardrails. Callers can use
    /// this to skip chain allocation on the hot path.
    pub fn is_empty(&self) -> bool {
        self.current().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::models::Guardrail as DomainGuardrail;
    use aisix_core::resource::ResourceEntry;
    use aisix_gateway::{ChatFormat, ChatMessage};

    fn entry(_name: &str, id: &str, row: DomainGuardrail) -> ResourceEntry<DomainGuardrail> {
        // `name` is documentary at the call site; the row's own
        // `name` field is what the chain logs as.
        ResourceEntry::new(id, row, 1)
    }

    fn parse(json: &str) -> DomainGuardrail {
        serde_json::from_str(json).unwrap()
    }

    fn req(msg: &str) -> ChatFormat {
        ChatFormat::new("m", vec![ChatMessage::user(msg)])
    }

    #[tokio::test]
    async fn enabled_keyword_row_blocks_matching_input() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "block-secrets",
            "g-1",
            parse(
                r#"{
                    "name": "block-secrets",
                    "kind": "keyword",
                    "patterns": [
                        { "kind": "literal", "value": "AKIA" }
                    ]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        assert_eq!(chain.len(), 1);
        let v = chain.check_input(&req("here is AKIAEXAMPLE")).await;
        assert!(v.is_block());
    }

    /// P1-3: `enforcement_mode: monitor` observes but never blocks. The same
    /// keyword rule that blocks under the default `block` mode must Allow the
    /// matching input when the row is in monitor mode — operators get the
    /// audit log without the request being rejected.
    #[tokio::test]
    async fn monitor_mode_observes_but_does_not_block() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "watch-secrets",
            "g-1",
            parse(
                r#"{
                    "name": "watch-secrets",
                    "enforcement_mode": "monitor",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        assert_eq!(chain.len(), 1, "monitor-mode row still materialises");
        // Would block under `block` mode; monitor downgrades to Allow.
        let v = chain.check_input(&req("here is AKIAEXAMPLE")).await;
        assert!(!v.is_block(), "monitor mode must not block, got {v:?}",);
        assert_eq!(v, GuardrailVerdict::Allow);
        // Output hook is monitored the same way.
        let resp = ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: ChatMessage::assistant("leaking AKIAEXAMPLE"),
            finish_reason: aisix_gateway::FinishReason::Stop,
            usage: aisix_gateway::UsageStats::new(0, 0),
        };
        assert!(!chain.check_output(&resp).await.is_block());
    }

    /// AISIX-Cloud#562: the observed check surfaces what monitor mode
    /// suppressed — a downgraded Block becomes a `would_block` hit and a
    /// suppressed pii mask becomes a `would_mask` hit with the detector
    /// counts. Verdicts stay downgraded; names only, never values.
    #[tokio::test]
    async fn monitor_mode_observed_check_reports_hits() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "watch-pii",
            "g-1",
            parse(
                r#"{
                    "name": "watch-pii",
                    "enforcement_mode": "monitor",
                    "kind": "pii",
                    "detectors": [
                        { "type": "email", "action": "mask" },
                        { "type": "us_ssn", "action": "block" }
                    ]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);

        // mask-action detector match → would_mask hit with counts.
        let (v, hits) = chain
            .check_input_observed(&req("mail alice@example.com ok"))
            .await;
        assert_eq!(v, GuardrailVerdict::Allow);
        assert_eq!(hits.len(), 1, "hits: {hits:?}");
        assert_eq!(hits[0].guardrail_name, "watch-pii");
        assert_eq!(hits[0].hook, "input");
        assert_eq!(hits[0].action, "would_mask");
        assert_eq!(hits[0].counts.get("email"), Some(&1));
        assert!(
            !format!("{hits:?}").contains("alice@example.com"),
            "matched value must never ride a hit",
        );

        // block-action detector match → would_block hit carrying the reason.
        let (v, hits) = chain.check_input_observed(&req("ssn 123-45-6789")).await;
        assert_eq!(v, GuardrailVerdict::Allow);
        assert_eq!(hits.len(), 1, "hits: {hits:?}");
        assert_eq!(hits[0].action, "would_block");
        assert!(
            hits[0].reason.contains("us_ssn"),
            "reason: {}",
            hits[0].reason
        );
        assert!(!hits[0].reason.contains("123-45-6789"));

        // clean input → no hits.
        let (v, hits) = chain.check_input_observed(&req("all fine")).await;
        assert_eq!(v, GuardrailVerdict::Allow);
        assert!(hits.is_empty(), "hits: {hits:?}");
    }

    /// An ENFORCING (block-mode) guardrail must not produce monitor hits —
    /// its Block is real and already carried by the verdict.
    #[tokio::test]
    async fn enforcing_guardrail_produces_no_monitor_hits() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "hard-block",
            "g-1",
            parse(
                r#"{
                    "name": "hard-block",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        let (v, hits) = chain
            .check_input_observed(&req("here is AKIAEXAMPLE"))
            .await;
        assert!(v.is_block());
        assert!(hits.is_empty(), "hits: {hits:?}");
    }

    /// A monitor-mode hit made BEFORE an enforcing peer blocks must survive
    /// the short-circuit — the chain collects hits as it folds.
    #[tokio::test]
    async fn monitor_hit_survives_enforcing_peer_block() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "watch-first",
            "g-1",
            parse(
                r#"{
                    "name": "watch-first",
                    "enforcement_mode": "monitor",
                    "kind": "keyword",
                    "created_at": "2024-01-01T00:00:00Z",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        table.insert(entry(
            "block-second",
            "g-2",
            parse(
                r#"{
                    "name": "block-second",
                    "kind": "keyword",
                    "created_at": "2024-01-02T00:00:00Z",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        let (v, hits) = chain
            .check_input_observed(&req("here is AKIAEXAMPLE"))
            .await;
        assert!(v.is_block(), "enforcing peer still blocks");
        assert_eq!(
            hits.len(),
            1,
            "monitor hit collected before the block: {hits:?}"
        );
        assert_eq!(hits[0].guardrail_name, "watch-first");
        assert_eq!(hits[0].action, "would_block");
    }

    /// A monitor-mode guardrail must not force streamed output to hold back —
    /// it can never block, so hold-back would be pure latency. It folds to the
    /// no-hold-back policy (and, in a mixed chain, can't weaken a blocking
    /// peer because the chain keeps the strictest member's policy).
    #[tokio::test]
    async fn monitor_mode_does_not_force_stream_holdback() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "watch-out",
            "g-1",
            parse(
                r#"{
                    "name": "watch-out",
                    "enforcement_mode": "monitor",
                    "kind": "keyword",
                    "hook_point": "output",
                    "patterns": [{ "kind": "literal", "value": "secret" }]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        assert!(
            !chain.stream_output_policy().holds_back(),
            "monitor-mode output rule must not hold the stream back",
        );
    }

    /// An unrecognised enforcement_mode is treated as `block` (fail-safe).
    #[tokio::test]
    async fn unknown_enforcement_mode_falls_back_to_block() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "g",
            "g-1",
            parse(
                r#"{
                    "name": "g",
                    "enforcement_mode": "audit-only-typo",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        assert!(
            chain
                .check_input(&req("here is AKIAEXAMPLE"))
                .await
                .is_block(),
            "unknown mode must default to block, not silently pass through",
        );
    }

    /// A stub remote guardrail that always fails open (returns `Bypass`),
    /// standing in for a Bedrock/Azure guardrail whose upstream is down.
    struct AlwaysBypass;
    #[async_trait]
    impl Guardrail for AlwaysBypass {
        fn name(&self) -> &'static str {
            "always-bypass"
        }
        async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
            GuardrailVerdict::Bypass {
                reason: "upstream_unreachable".into(),
            }
        }
        async fn check_output(&self, _resp: &ChatResponse) -> GuardrailVerdict {
            GuardrailVerdict::Bypass {
                reason: "upstream_unreachable".into(),
            }
        }
    }

    fn row_with_mandatory(mandatory: bool) -> DomainGuardrail {
        let mut v = serde_json::json!({
            "name": "remote",
            "kind": "keyword",
            "patterns": [{ "kind": "literal", "value": "x" }],
        });
        if mandatory {
            v["mandatory"] = serde_json::Value::Bool(true);
        }
        serde_json::from_value(v).unwrap()
    }

    fn resp(text: &str) -> ChatResponse {
        ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: ChatMessage::assistant(text),
            finish_reason: aisix_gateway::FinishReason::Stop,
            usage: aisix_gateway::UsageStats::new(0, 0),
        }
    }

    /// #911 finding [26]: `mandatory: true` turns a fail-open `Bypass` into a
    /// `Block`, so a remote guardrail marked mandatory can't be silently
    /// skipped when its upstream is unreachable. Before the fix the field was
    /// parsed but never enforced — a mandatory guardrail still failed open.
    #[tokio::test]
    async fn mandatory_upgrades_bypass_to_block() {
        let g = apply_mandatory(&row_with_mandatory(true), Arc::new(AlwaysBypass));
        let vin = g.check_input(&req("hi")).await;
        // The block must carry the row name so the 422 envelope can name the
        // guardrail (#519 B.4b) rather than surfacing an unnamed block.
        assert_eq!(
            vin,
            GuardrailVerdict::Block {
                reason: "mandatory guardrail unavailable: upstream_unreachable".to_string(),
                guardrail_name: Some("remote".to_string()),
            },
            "mandatory input Bypass must become a named Block, got {vin:?}",
        );
        let vout = g.check_output(&resp("hi")).await;
        assert_eq!(
            vout,
            GuardrailVerdict::Block {
                reason: "mandatory guardrail unavailable: upstream_unreachable".to_string(),
                guardrail_name: Some("remote".to_string()),
            },
            "mandatory output Bypass must become a named Block, got {vout:?}",
        );
    }

    /// The default (`mandatory: false`) keeps the fail-open behaviour: a
    /// `Bypass` stays a `Bypass`.
    #[tokio::test]
    async fn non_mandatory_leaves_bypass_untouched() {
        let g = apply_mandatory(&row_with_mandatory(false), Arc::new(AlwaysBypass));
        assert!(
            g.check_input(&req("hi")).await.is_bypass(),
            "non-mandatory guardrail must keep failing open",
        );
    }

    /// Mandatory only rewrites the failure verdict — `Allow` and `Block`
    /// pass through, so a healthy mandatory guardrail never becomes a false
    /// block and a real block is preserved.
    #[tokio::test]
    async fn mandatory_passes_allow_and_block_through() {
        struct AlwaysAllow;
        #[async_trait]
        impl Guardrail for AlwaysAllow {
            fn name(&self) -> &'static str {
                "always-allow"
            }
            async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
                GuardrailVerdict::Allow
            }
        }
        struct AlwaysBlock;
        #[async_trait]
        impl Guardrail for AlwaysBlock {
            fn name(&self) -> &'static str {
                "always-block"
            }
            async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
                GuardrailVerdict::block("nope")
            }
        }
        let allow = apply_mandatory(&row_with_mandatory(true), Arc::new(AlwaysAllow));
        assert_eq!(allow.check_input(&req("hi")).await, GuardrailVerdict::Allow);
        let block = apply_mandatory(&row_with_mandatory(true), Arc::new(AlwaysBlock));
        assert!(block.check_input(&req("hi")).await.is_block());
    }

    fn aliyun_row_against(endpoint: &str, extra: &str) -> DomainGuardrail {
        parse(&format!(
            r#"{{
                "name": "aliyun-monitor",
                "kind": "aliyun_text_moderation",
                "region": "cn-shanghai",
                "endpoint": "{endpoint}",
                "access_key_id": "ak",
                "access_key_secret": "sk"{extra}
            }}"#,
        ))
    }

    /// AISIX-Cloud#1010: `enforcement_mode: "monitor"` must never block —
    /// including when the remote provider call itself FAILS, not just when
    /// content is flagged. With `fail_open: false` a provider 5xx surfaces
    /// as a `Block` from the inner guardrail; the monitor wrapper must
    /// downgrade it. Composed through `build_one` so the decorator
    /// ordering itself is what's pinned.
    #[tokio::test]
    async fn monitor_downgrades_provider_failure_block() {
        use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let row = aliyun_row_against(
            &server.uri(),
            r#", "enforcement_mode": "monitor", "fail_open": false"#,
        );
        let g = build_one(&row, None).unwrap().unwrap();
        assert_eq!(
            g.check_input(&req("hello")).await,
            GuardrailVerdict::Allow,
            "monitor mode must downgrade a fail-closed provider-failure Block",
        );
    }

    /// The documented exception to the above: `mandatory: true` is applied
    /// OUTSIDE the monitor wrapper (`build_one`), so provider
    /// unavailability stays fatal even in monitor mode — the fail-open
    /// `Bypass` passes through the monitor wrapper untouched and is then
    /// upgraded to a named `Block`.
    #[tokio::test]
    async fn mandatory_keeps_unavailability_fatal_in_monitor_mode() {
        use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let row = aliyun_row_against(
            &server.uri(),
            r#", "enforcement_mode": "monitor", "mandatory": true"#,
        );
        let g = build_one(&row, None).unwrap().unwrap();
        assert!(
            g.check_input(&req("hello")).await.is_block(),
            "mandatory must keep provider unavailability fatal in monitor mode",
        );
    }

    #[tokio::test]
    async fn disabled_row_is_dropped() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "g",
            "g-1",
            parse(
                r#"{
                    "name": "g",
                    "enabled": false,
                    "kind": "keyword",
                    "patterns": [
                        { "kind": "literal", "value": "AKIA" }
                    ]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        assert_eq!(chain.len(), 0);
    }

    #[tokio::test]
    async fn empty_pattern_list_is_inert() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "g",
            "g-1",
            parse(
                r#"{
                    "name": "g",
                    "kind": "keyword",
                    "patterns": []
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        assert_eq!(chain.len(), 0, "empty list adds nothing to the chain");
    }

    #[tokio::test]
    async fn invalid_regex_is_skipped_with_warning() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "good",
            "g-1",
            parse(
                r#"{
                    "name": "good",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "ok" }]
                }"#,
            ),
        ));
        // Domain layer accepts arbitrary strings as Regex(...); the
        // regex compile only happens here. Inject a row with an
        // unclosed bracket — the schema layer doesn't compile
        // regexes either, so this slips through to us.
        table.insert(entry(
            "bad",
            "g-2",
            parse(
                r#"{
                    "name": "bad",
                    "kind": "keyword",
                    "patterns": [{ "kind": "regex", "value": "[unclosed" }]
                }"#,
            ),
        ));

        let chain = build_chain_from_snapshot(&table, None);
        // Only the good row makes it in.
        assert_eq!(chain.len(), 1);
        let v = chain.check_input(&req("ok")).await;
        assert!(v.is_block());
    }

    /// #52: an openai_moderation row with a category threshold outside
    /// 0..=1 is rejected at build time (moderation scores are 0..=1, so
    /// such a threshold can never — or always — fire).
    #[cfg(feature = "openai-moderation")]
    #[tokio::test]
    async fn openai_moderation_out_of_range_threshold_skips_row() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "bad-threshold",
            "g-1",
            parse(
                r#"{
                    "name": "bad-threshold",
                    "kind": "openai_moderation",
                    "api_key": "sk-x",
                    "category_thresholds": { "violence": 1.5 }
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        assert_eq!(chain.len(), 0, "out-of-range threshold row must be skipped");
    }

    /// Phase 2 contract: kind=bedrock rows materialise into the
    /// runtime chain alongside keyword rows. We don't hit AWS in
    /// this test (the request never makes it past chain
    /// composition) — we just pin that both kinds compose into the
    /// final chain length, and that the keyword Block still fires.
    #[cfg(feature = "bedrock")]
    #[tokio::test]
    async fn bedrock_kind_materialises_alongside_keyword_in_chain() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "bedrock-row",
            "g-1",
            parse(
                r#"{
                    "name": "bedrock-row",
                    "kind": "bedrock",
                    "guardrail_id": "abcdefgh1234",
                    "guardrail_version": "DRAFT",
                    "region": "us-east-1",
                    "aws_credentials": {
                        "kind": "static",
                        "access_key_id": "AKIA",
                        "secret_access_key": "test-secret-plaintext"
                    },
                    "latency_mode": { "kind": "serial" }
                }"#,
            ),
        ));
        table.insert(entry(
            "keyword-row",
            "g-2",
            parse(
                r#"{
                    "name": "keyword-row",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        // Both rows compose. We don't probe the bedrock arm — its
        // own tests cover the dispatch path; this one only pins the
        // chain composition contract.
        assert_eq!(chain.len(), 2);
    }

    #[tokio::test]
    async fn live_chain_rebuilds_on_snapshot_swap() {
        let initial = AisixSnapshot::new();
        let handle = SnapshotHandle::new(initial);
        let live = LiveGuardrailChain::new(handle.clone(), None);

        // Empty snapshot → no rules → input passes.
        assert!(!live.check_input(&req("AKIA-EXAMPLE")).await.is_block());

        // Build a new snapshot that adds a blocking keyword rule
        // and store it. The next check_input must rebuild and
        // reflect the new policy.
        let next = AisixSnapshot::new();
        next.guardrails.insert(entry(
            "block-secrets",
            "g-1",
            parse(
                r#"{
                    "name": "block-secrets",
                    "kind": "keyword",
                    "patterns": [
                        { "kind": "literal", "value": "AKIA" }
                    ]
                }"#,
            ),
        ));
        handle.store(next);

        assert!(live.check_input(&req("AKIA-EXAMPLE")).await.is_block());
    }

    // -----------------------------------------------------------------------
    // Deterministic chain order (#519 B.4a)
    // -----------------------------------------------------------------------

    fn keyword_row(name: &str, created_at: Option<&str>) -> DomainGuardrail {
        let mut v = serde_json::json!({
            "name": name,
            "kind": "keyword",
            "patterns": [{ "kind": "literal", "value": "AKIA" }],
        });
        if let Some(ts) = created_at {
            v["created_at"] = serde_json::Value::String(ts.to_owned());
        }
        serde_json::from_value(v).unwrap()
    }

    /// (id, name, created_at) rows in deliberately shuffled insertion
    /// order. Expected chain order: rows WITH created_at ascending (ties
    /// broken by id), then rows WITHOUT created_at by id.
    const SHUFFLED_ROWS: [(&str, &str, Option<&str>); 10] = [
        ("g-09", "i", Some("2026-01-05T00:00:00Z")),
        ("g-03", "c", Some("2026-01-01T00:00:00Z")),
        ("g-10", "j", None),
        ("g-05", "e", Some("2026-01-02T00:00:00Z")),
        ("g-01", "a", None),
        ("g-07", "g", Some("2026-01-04T00:00:00Z")),
        ("g-02", "b", Some("2026-01-03T00:00:00Z")),
        // same timestamp as g-02 → id tiebreak
        ("g-08", "h", Some("2026-01-03T00:00:00Z")),
        ("g-04", "d", None),
        // same timestamp as g-03 → id tiebreak
        ("g-06", "f", Some("2026-01-01T00:00:00Z")),
    ];

    const EXPECTED_ORDER: [&str; 10] = ["c", "f", "e", "b", "h", "g", "i", "a", "d", "j"];

    fn shuffled_table() -> ResourceTable<DomainGuardrail> {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        for (id, name, ts) in SHUFFLED_ROWS {
            table.insert(entry(name, id, keyword_row(name, ts)));
        }
        table
    }

    /// The chain evaluates rows created_at-ascending (id tiebreak; rows
    /// without created_at last) regardless of insertion order. The table
    /// is DashMap-backed — without the build-time sort the chain follows
    /// the map's arbitrary, run-to-run-varying iteration order and this
    /// assertion fails intermittently (#519 B.4a).
    #[test]
    fn chain_order_is_created_at_ascending_with_id_tiebreak() {
        let chain = build_chain_from_snapshot(&shuffled_table(), None);
        assert_eq!(chain.member_names(), EXPECTED_ORDER);
    }

    /// cp-api doesn't project `created_at` yet — a table where every row
    /// lacks it must still build in a deterministic (id-ascending) order.
    #[test]
    fn chain_order_falls_back_to_id_when_created_at_absent() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        for (id, name) in [("g-3", "z"), ("g-1", "y"), ("g-2", "x")] {
            table.insert(entry(name, id, keyword_row(name, None)));
        }
        let chain = build_chain_from_snapshot(&table, None);
        assert_eq!(chain.member_names(), ["y", "x", "z"]);
    }

    /// The production per-request path: guardrails with no attachment
    /// rows fall back to implicit env-scope entries that all share
    /// priority 0, so their relative order in the resolved chain is
    /// decided by the index build's iteration — which must be the same
    /// created_at-ascending order as the chain-build path (#519 B.4a).
    #[test]
    fn no_attachment_fallback_resolves_in_created_at_order() {
        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();
        let index = build_index_from_snapshot(&shuffled_table(), &attachments, None);
        let chain = index.resolve(&RequestContext {
            model_id: "m",
            api_key_id: "k",
            team_id: None,
        });
        assert_eq!(chain.member_names(), EXPECTED_ORDER);
    }

    // -----------------------------------------------------------------------
    // build_index_from_snapshot tests
    // -----------------------------------------------------------------------

    fn parse_attachment(json: &str) -> GuardrailAttachment {
        serde_json::from_str(json).unwrap()
    }

    fn attachment_entry(id: &str, row: GuardrailAttachment) -> ResourceEntry<GuardrailAttachment> {
        ResourceEntry::new(id, row, 1)
    }

    #[tokio::test]
    async fn enabled_attachment_builds_index_entry() {
        let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
        guardrails.insert(entry(
            "secrets",
            "g-1",
            parse(
                r#"{
                    "name": "block-secrets",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));

        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();
        attachments.insert(attachment_entry(
            "a-1",
            parse_attachment(
                r#"{
                    "guardrail_id": "g-1",
                    "scope_type": "env",
                    "priority": 50
                }"#,
            ),
        ));

        let index = build_index_from_snapshot(&guardrails, &attachments, None);
        assert_eq!(index.len(), 1);

        let ctx = RequestContext {
            model_id: "m1",
            api_key_id: "k1",
            team_id: None,
        };
        let chain = index.resolve(&ctx);
        assert!(chain.check_input(&req("here AKIA")).await.is_block());
    }

    #[tokio::test]
    async fn disabled_attachment_is_skipped_in_index() {
        let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
        guardrails.insert(entry(
            "g",
            "g-1",
            parse(
                r#"{
                    "name": "g",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));

        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();
        attachments.insert(attachment_entry(
            "a-1",
            parse_attachment(
                r#"{
                    "guardrail_id": "g-1",
                    "scope_type": "env",
                    "priority": 50,
                    "enabled": false
                }"#,
            ),
        ));

        let index = build_index_from_snapshot(&guardrails, &attachments, None);
        assert_eq!(index.len(), 0);
        // Verify the guardrail does not fire (not just that the index is empty).
        let ctx = RequestContext {
            model_id: "m",
            api_key_id: "k",
            team_id: None,
        };
        assert!(
            !index
                .resolve(&ctx)
                .check_input(&req("here AKIA"))
                .await
                .is_block(),
            "disabled-only-attachment guardrail must not block any request",
        );
    }

    #[tokio::test]
    async fn one_enabled_one_disabled_attachment_fires_exactly_once() {
        // Verifies the HashSet boundary: a guardrail with one enabled + one disabled
        // attachment must fire exactly once (via the enabled attachment) and must NOT
        // trigger the backward-compat env-scope fallback.
        let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
        guardrails.insert(entry(
            "g",
            "g-1",
            parse(
                r#"{"name":"g","kind":"keyword","patterns":[{"kind":"literal","value":"AKIA"}]}"#,
            ),
        ));
        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();
        attachments.insert(attachment_entry(
            "a-enabled",
            parse_attachment(r#"{"guardrail_id":"g-1","scope_type":"env","priority":50}"#),
        ));
        attachments.insert(attachment_entry(
            "a-disabled",
            parse_attachment(
                r#"{"guardrail_id":"g-1","scope_type":"model","scope_id":"m1","priority":10,"enabled":false}"#,
            ),
        ));

        let index = build_index_from_snapshot(&guardrails, &attachments, None);
        // Exactly one entry — from the enabled attachment only.
        // The disabled attachment must NOT produce a second entry or trigger the fallback.
        assert_eq!(
            index.len(),
            1,
            "enabled+disabled attachments: exactly 1 entry expected",
        );
        let ctx = RequestContext {
            model_id: "any",
            api_key_id: "any",
            team_id: None,
        };
        assert!(
            index
                .resolve(&ctx)
                .check_input(&req("here AKIA"))
                .await
                .is_block(),
            "env-scope enabled attachment must still fire",
        );
    }

    #[tokio::test]
    async fn no_attachment_guardrail_fires_globally_backward_compat() {
        // Core backward-compat contract: a guardrail with ZERO attachment rows
        // must fire on every request (env-scope at priority 0), preserving
        // the pre-P0c "apply globally" behavior during rolling upgrade.
        let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
        guardrails.insert(entry(
            "g",
            "g-1",
            parse(
                r#"{
                    "name": "g",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();

        let index = build_index_from_snapshot(&guardrails, &attachments, None);
        assert_eq!(
            index.len(),
            1,
            "no-attachment guardrail must appear as env-scope entry",
        );

        let ctx = RequestContext {
            model_id: "any-model",
            api_key_id: "any-key",
            team_id: None,
        };
        assert!(
            index
                .resolve(&ctx)
                .check_input(&req("here AKIA"))
                .await
                .is_block(),
            "no-attachment guardrail must block matching requests",
        );
    }

    #[tokio::test]
    async fn attachment_referencing_unknown_guardrail_is_skipped() {
        let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
        // "g-99" is not inserted — attachment points to a missing definition.

        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();
        attachments.insert(attachment_entry(
            "a-1",
            parse_attachment(
                r#"{
                    "guardrail_id": "g-99",
                    "scope_type": "env",
                    "priority": 50
                }"#,
            ),
        ));

        let index = build_index_from_snapshot(&guardrails, &attachments, None);
        assert_eq!(index.len(), 0);
    }

    #[tokio::test]
    async fn disabled_guardrail_with_enabled_attachment_is_skipped() {
        let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
        guardrails.insert(entry(
            "g",
            "g-1",
            parse(
                r#"{
                    "name": "g",
                    "enabled": false,
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));

        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();
        attachments.insert(attachment_entry(
            "a-1",
            parse_attachment(
                r#"{
                    "guardrail_id": "g-1",
                    "scope_type": "env",
                    "priority": 50
                }"#,
            ),
        ));

        let index = build_index_from_snapshot(&guardrails, &attachments, None);
        assert_eq!(index.len(), 0);
    }

    #[tokio::test]
    async fn live_index_rebuilds_on_snapshot_swap() {
        let initial = AisixSnapshot::new();
        let handle = SnapshotHandle::new(initial);
        let live = LiveGuardrailIndex::new(handle.clone(), None);

        let ctx = RequestContext {
            model_id: "m1",
            api_key_id: "k1",
            team_id: None,
        };

        // Empty snapshot → no rules → input passes.
        assert!(!live
            .resolve(&ctx)
            .check_input(&req("AKIA-EXAMPLE"))
            .await
            .is_block());
        assert!(live.is_empty());

        // Swap in a snapshot that attaches a blocking keyword guardrail env-wide.
        let next = AisixSnapshot::new();
        next.guardrails.insert(entry(
            "block-secrets",
            "g-1",
            parse(
                r#"{
                    "name": "block-secrets",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        next.guardrail_attachments.insert(attachment_entry(
            "a-1",
            parse_attachment(
                r#"{
                    "guardrail_id": "g-1",
                    "scope_type": "env",
                    "priority": 50
                }"#,
            ),
        ));
        handle.store(next);

        assert!(live
            .resolve(&ctx)
            .check_input(&req("AKIA-EXAMPLE"))
            .await
            .is_block());
        assert!(!live.is_empty());
    }

    #[tokio::test]
    async fn hook_point_input_only_skips_output() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "g",
            "g-1",
            parse(
                r#"{
                    "name": "g",
                    "kind": "keyword",
                    "hook_point": "input",
                    "patterns": [{ "kind": "literal", "value": "secret" }]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None);
        // input check fires...
        assert!(chain.check_input(&req("secret")).await.is_block());
        // ...but output check is a noop on this rule.
        use aisix_gateway::{ChatResponse, FinishReason, UsageStats};
        let resp = ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: ChatMessage::assistant("secret"),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(0, 0),
        };
        assert!(!chain.check_output(&resp).await.is_block());
    }

    // -----------------------------------------------------------------------
    // applied-guardrails capture (#379 A1)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn build_chain_reports_applied_kind_and_hook() {
        // build_chain_from_snapshot is one of the two capture points: the
        // resulting chain must report each materialised row's kind + hook,
        // in the table's id-sorted iteration order.
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "kw-input",
            "g-1",
            parse(
                r#"{
                    "name": "kw-input",
                    "kind": "keyword",
                    "hook_point": "input",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        table.insert(entry(
            "kw-output",
            "g-2",
            parse(
                r#"{
                    "name": "kw-output",
                    "kind": "keyword",
                    "hook_point": "output",
                    "patterns": [{ "kind": "literal", "value": "secret" }]
                }"#,
            ),
        ));

        let chain = build_chain_from_snapshot(&table, None);
        // `applied` mirrors the chain 1:1 (pushed in lockstep); the absolute
        // member order is a `ResourceTable::entries()` concern tested
        // elsewhere, so sort by hook before comparing to pin only that BOTH
        // rows are captured with the right kind + hook.
        let mut applied = chain.applied().to_vec();
        applied.sort_by(|a, b| a.hook.cmp(&b.hook));
        assert_eq!(
            applied,
            vec![
                AppliedGuardrail {
                    kind: "keyword".to_owned(),
                    hook: "input".to_owned(),
                },
                AppliedGuardrail {
                    kind: "keyword".to_owned(),
                    hook: "output".to_owned(),
                },
            ],
        );
    }

    #[tokio::test]
    async fn applied_excludes_inert_and_disabled_rows() {
        // `applied` is pushed only on Ok(Some) — it records what actually
        // governs the request. An empty keyword list (inert / Ok(None)) and a
        // disabled row (dropped) must not appear, so `applied` never claims a
        // guardrail ran that didn't.
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "inert",
            "g-1",
            parse(r#"{ "name": "inert", "kind": "keyword", "patterns": [] }"#),
        ));
        table.insert(entry(
            "off",
            "g-2",
            parse(
                r#"{
                    "name": "off",
                    "enabled": false,
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "x" }]
                }"#,
            ),
        ));
        table.insert(entry(
            "live",
            "g-3",
            parse(
                r#"{
                    "name": "live",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));

        let chain = build_chain_from_snapshot(&table, None);
        assert_eq!(chain.len(), 1, "only the live row materialises");
        assert_eq!(
            chain.applied(),
            &[AppliedGuardrail {
                kind: "keyword".to_owned(),
                hook: "both".to_owned(),
            }],
            "applied reports only the row that actually governs the request",
        );
    }

    #[tokio::test]
    async fn resolved_chain_reports_applied_and_mirrors_dedup() {
        // The per-request path (index.resolve, the capture point the proxy
        // actually uses): the resolved chain reports each member's kind + hook,
        // and `applied` mirrors the deduplicated chain 1:1 — a guardrail
        // attached via two scopes still appears exactly once.
        let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
        guardrails.insert(entry(
            "kw",
            "g-1",
            parse(
                r#"{
                    "name": "kw",
                    "kind": "keyword",
                    "hook_point": "input",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();
        attachments.insert(attachment_entry(
            "a-env",
            parse_attachment(r#"{ "guardrail_id": "g-1", "scope_type": "env", "priority": 50 }"#),
        ));
        attachments.insert(attachment_entry(
            "a-model",
            parse_attachment(
                r#"{ "guardrail_id": "g-1", "scope_type": "model", "scope_id": "m-A", "priority": 100 }"#,
            ),
        ));

        let index = build_index_from_snapshot(&guardrails, &attachments, None);
        let chain = index.resolve(&RequestContext {
            model_id: "m-A",
            api_key_id: "k",
            team_id: None,
        });
        assert_eq!(chain.len(), 1, "dedup keeps a single runtime guardrail");
        assert_eq!(
            chain.applied(),
            &[AppliedGuardrail {
                kind: "keyword".to_owned(),
                hook: "input".to_owned(),
            }],
            "applied mirrors the deduplicated chain, not the raw entry count",
        );
    }

    #[tokio::test]
    async fn resolved_chain_applied_empty_when_no_attachment_matches() {
        // A model-scoped attachment that doesn't match the request resolves to
        // an empty chain — and `applied` must be empty too, so the telemetry
        // event never claims a guardrail governed a request it didn't.
        let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
        guardrails.insert(entry(
            "kw",
            "g-1",
            parse(
                r#"{
                    "name": "kw",
                    "kind": "keyword",
                    "hook_point": "output",
                    "patterns": [{ "kind": "literal", "value": "x" }]
                }"#,
            ),
        ));
        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();
        attachments.insert(attachment_entry(
            "a-model",
            parse_attachment(
                r#"{ "guardrail_id": "g-1", "scope_type": "model", "scope_id": "m-A", "priority": 10 }"#,
            ),
        ));

        let index = build_index_from_snapshot(&guardrails, &attachments, None);
        let chain = index.resolve(&RequestContext {
            model_id: "m-OTHER",
            api_key_id: "k",
            team_id: None,
        });
        assert!(chain.is_empty());
        assert!(
            chain.applied().is_empty(),
            "no matching attachment → empty applied set",
        );
    }
}
