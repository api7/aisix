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

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use aisix_core::models::{
    AisixSnapshot, AppliedGuardrail, Guardrail as DomainGuardrail, GuardrailAttachment,
    GuardrailHookPoint, GuardrailKind, GuardrailMonitorHit, GuardrailScopeType, KeywordPattern,
};
use aisix_core::snapshot::ResourceTable;
use aisix_core::{ConfigStatus, IncomingRejection, SnapshotHandle};
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;

use crate::index::{GuardrailIndex, RequestContext, ScopeKind};
use crate::keyword::{KeywordBlocklist, KeywordRule};
use crate::pii::{builtin_rule, PiiAction, PiiGuardrail, PiiRule};
use crate::{
    Guardrail, GuardrailChain, GuardrailEmbedderSlot, GuardrailVerdict, Redaction, SegmentsOutcome,
    StreamOutputPolicy,
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
    embedder: &GuardrailEmbedderSlot,
) -> GuardrailChain {
    build_chain_from_snapshot_reported(table, bedrock_endpoint_url, embedder).0
}

fn build_chain_from_snapshot_reported(
    table: &ResourceTable<DomainGuardrail>,
    bedrock_endpoint_url: Option<&str>,
    embedder: &GuardrailEmbedderSlot,
) -> (GuardrailChain, Vec<GuardrailBuildRejection>) {
    let mut chain: Vec<(String, Arc<dyn Guardrail>)> = Vec::new();
    // `applied` mirrors `chain` 1:1 — the `{kind, hook}` of each member that
    // actually materialised, for applied-guardrail telemetry (#379). Pushed
    // only on the `Ok(Some)` path so inert/invalid rows (which never join the
    // chain) never show up as "governed this request".
    let mut applied: Vec<AppliedGuardrail> = Vec::new();
    let mut rejected = Vec::new();

    let entries = sorted_guardrail_entries(table);
    for entry in entries.iter() {
        let row = &entry.value;
        if !row.enabled {
            continue;
        }
        match build_one(row, bedrock_endpoint_url, embedder) {
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
                rejected.push(GuardrailBuildRejection {
                    id: entry.id.clone(),
                    reason: err.status_reason(),
                });
            }
        }
    }

    (GuardrailChain::new_with_applied(chain, applied), rejected)
}

/// One enabled guardrail row that would not join the chain.
///
/// Returned by [`unbuildable_guardrail_rows`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnbuildableGuardrailRow {
    /// Operator-facing row name, as written in the configuration.
    pub name: String,
    /// Why the row does not build, in the same wording the boot log uses.
    /// A `kind: custom` script error carries the engine's line and column.
    pub reason: String,
}

/// Report the enabled guardrail rows in `table` whose configuration does
/// not build — [`build_chain_from_snapshot`]'s error arm, made returnable.
///
/// The gateway logs those errors, reports them through config status, and
/// keeps serving. A pre-deployment check still needs a synchronous report:
/// the row silently not existing is the defect, even though the file itself
/// loaded successfully.
///
/// [`BuildError::EmbedderUnavailable`] is deliberately excluded. The
/// embedding dispatcher is a runtime capability, not part of the
/// configuration, so a `kind: semantic` row fails to build here for a
/// reason that says nothing about the file being checked.
pub fn unbuildable_guardrail_rows(
    table: &ResourceTable<DomainGuardrail>,
    bedrock_endpoint_url: Option<&str>,
) -> Vec<UnbuildableGuardrailRow> {
    let embedder = GuardrailEmbedderSlot::none();
    sorted_guardrail_entries(table)
        .iter()
        .filter(|entry| entry.value.enabled)
        .filter_map(
            |entry| match build_one(&entry.value, bedrock_endpoint_url, &embedder) {
                Ok(_) => None,
                Err(BuildError::EmbedderUnavailable) => None,
                Err(err) => Some(UnbuildableGuardrailRow {
                    name: entry.value.name.clone(),
                    reason: err.to_string(),
                }),
            },
        )
        .collect()
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

/// Build the runtime guardrail for a row, applying its `enforcement_mode`.
///
/// `block` (the default) returns the guardrail as-is; `monitor` wraps it
/// in [`MonitorGuardrail`] so it observes violations without blocking.
///
/// Monitor mode is unconditional: a monitored row never blocks, for any
/// reason. What to do about a failed evaluation is `fail_open`'s job, and
/// a monitored row does not act on that decision either — an operator who
/// wants an unreachable provider to refuse traffic is asking for
/// enforcement, which is what `block` mode is.
fn build_one(
    row: &DomainGuardrail,
    bedrock_endpoint_url: Option<&str>,
    embedder: &GuardrailEmbedderSlot,
) -> Result<Option<Arc<dyn Guardrail>>, BuildError> {
    Ok(build_one_inner(row, bedrock_endpoint_url, embedder)?
        .map(|g| apply_enforcement_mode(row, g)))
}

/// Wrap `inner` per the row's `enforcement_mode`. See [`build_one`].
fn apply_enforcement_mode(row: &DomainGuardrail, inner: Arc<dyn Guardrail>) -> Arc<dyn Guardrail> {
    match row.enforcement_mode.as_str() {
        "block" => inner,
        "monitor" => Arc::new(MonitorGuardrail {
            row_name: row.name.clone(),
            telemetry_reason_safe: row.config.kind_str() != "custom",
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
    embedder: &GuardrailEmbedderSlot,
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
                            field: "patterns[].value",
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
                // `replacement` only means something to a mask rewrite; on a
                // block-action pattern it would be accepted-but-unread config
                // (the #962 class), so the row is rejected instead. cp-api
                // validates the same combination at write time; this covers
                // the declarative-file source and defends the invariant.
                if p.replacement.is_some() && action == PiiAction::Block {
                    return Err(BuildError::ReplacementOnBlock {
                        name: p.name.clone(),
                    });
                }
                let rule = PiiRule::new(p.name.clone(), &p.regex, action, None)
                    .map_err(|e| BuildError::InvalidRegex {
                        field: "custom_patterns[].regex",
                        pattern: p.regex.clone(),
                        source: e,
                    })?
                    .with_replacement(p.replacement.clone());
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
        #[cfg(feature = "aliyun-text-moderation")]
        GuardrailKind::AliyunAiGuardrail(cfg) => {
            // #1070: MultiModalGuard dispatcher (Aliyun AI Guardrails — a
            // different product from TextModerationPlus above, same signing
            // scheme). cp-api already decrypted the access_key_secret at
            // projection time; the config carries plaintext.
            let g = crate::aliyun_ai_guardrail::AliyunAiGuardrail::new(
                row.name.clone(),
                cfg,
                row.hook_point,
                row.fail_open,
            );
            Ok(Some(Arc::new(g)))
        }
        #[cfg(not(feature = "aliyun-text-moderation"))]
        GuardrailKind::AliyunAiGuardrail(_) => {
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
        GuardrailKind::Semantic(cfg) => {
            // A row with neither list can never reach a verdict. Skipping
            // it here (rather than letting it screen every request to
            // discover that) keeps it exactly as inert as an empty
            // keyword blocklist.
            if cfg.deny_examples.is_empty() && cfg.allow_examples.is_empty() {
                return Ok(None);
            }
            // Unlike the compile-time kinds, this one needs a RUNTIME
            // capability the guardrails crate cannot supply itself: a
            // dispatcher for the `embedding`-kind Model the row names.
            // Without one the row is skipped and warned rather than
            // silently admitting everything it was meant to screen.
            let Some(embedder) = embedder.get() else {
                return Err(BuildError::EmbedderUnavailable);
            };
            let g = crate::semantic::SemanticGuardrail::new(
                &row.name,
                cfg,
                row.hook_point,
                row.fail_open,
                Arc::clone(embedder),
            );
            Ok(Some(Arc::new(g)))
        }
        GuardrailKind::Custom(cfg) => {
            // A whitespace-only script clears the schema's `minLength: 1`
            // and reaches here; an omitted or empty one is refused by BOTH
            // schemas, so it never gets this far. It must not build: an
            // empty module parses, exports no hook, and every hook would
            // return Allow — a row that looks configured and screens
            // nothing.
            if cfg.script.trim().is_empty() {
                return Err(BuildError::InvalidValue {
                    field: "script",
                    value: String::new(),
                });
            }
            // Compiling here (rather than on the first request that hits
            // the row) keeps the refusal off the request path: it parses
            // only, so nothing the operator wrote runs on the config-apply
            // path. The refusal is logged and published through config
            // status; `aisix validate` reports it synchronously, and cp-api's
            // own esbuild pass catches it at save time.
            let g = crate::custom::CustomGuardrail::new(
                row.name.clone(),
                cfg,
                row.hook_point,
                row.fail_open,
                embedder.clone(),
            )
            .map_err(BuildError::ScriptCompile)?;
            Ok(Some(Arc::new(g)))
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum BuildError {
    #[error("invalid regex {pattern:?}: {source}")]
    InvalidRegex {
        field: &'static str,
        pattern: String,
        source: regex::Error,
    },
    /// An enum-ish config field carrying a value the DP doesn't know
    /// (unknown built-in detector id, unrecognised action). The row is
    /// skipped + warned rather than silently running a weaker policy.
    #[error("invalid {field} value {value:?}")]
    InvalidValue { field: &'static str, value: String },
    /// A `custom_patterns[].replacement` on a pattern whose effective
    /// action is `block` — the replacement would never be read (never
    /// half-honor a knob, #963). Carries the pattern NAME only, never
    /// the replacement text or a matched value.
    #[error(
        "custom_patterns[].replacement requires action=mask (pattern {name:?} resolves to block)"
    )]
    ReplacementOnBlock { name: String },
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
    /// A `kind: "semantic"` row built by a caller that supplied no
    /// embedding dispatcher. The operator fix is not a node repair:
    /// nothing is missing from the node, the chain was built outside
    /// the proxy (a bare unit-test construction, a tool that only wants
    /// the compile-time kinds).
    #[error(
        "no embedding dispatcher available for the semantic guardrail; \
         treating row as disabled"
    )]
    EmbedderUnavailable,
    /// A `kind: "custom"` row whose script does not compile. Carries the
    /// engine's own message, which names the offending line and column, so
    /// the operator sees the same diagnostic their editor would show.
    #[error("custom guardrail script does not compile: {0}")]
    ScriptCompile(crate::custom::CompileError),
}

impl BuildError {
    /// Bounded, value-free diagnostic for unauthenticated status surfaces.
    /// Detailed values remain in the local warning log and validation output.
    fn status_reason(&self) -> String {
        let (category, field) = match self {
            Self::InvalidRegex { field, .. } => ("invalid_regex", *field),
            Self::InvalidValue { field, .. } => ("invalid_value", *field),
            Self::ReplacementOnBlock { .. } => {
                ("incompatible_fields", "custom_patterns[].replacement")
            }
            Self::FeatureDisabled(_) => ("feature_disabled", "kind"),
            Self::EmbedderUnavailable => ("runtime_unavailable", "embedding_model"),
            Self::ScriptCompile(_) => ("compile_failed", "script"),
        };
        format!("guardrail runtime build failed: {category} at config.{field}")
    }
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
    /// Built-in guardrails produce code-owned detector/category reasons.
    /// A custom script's reason is arbitrary operator code and may contain
    /// `ctx.text` or `ctx.secrets`, so it must never enter usage telemetry.
    telemetry_reason_safe: bool,
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
            reason: if self.telemetry_reason_safe {
                reason.to_owned()
            } else {
                String::new()
            },
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

    /// Forward the bind and re-wrap. A monitor-mode row is the one an
    /// operator is actively tuning, so it is the LAST place scores may go
    /// missing.
    fn bind_score_log(&self, log: &Arc<crate::GuardrailAuditLog>) -> Option<Arc<dyn Guardrail>> {
        self.inner.bind_score_log(log).map(|inner| {
            Arc::new(MonitorGuardrail {
                row_name: self.row_name.clone(),
                telemetry_reason_safe: self.telemetry_reason_safe,
                inner,
            }) as Arc<dyn Guardrail>
        })
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
    embedder: GuardrailEmbedderSlot,
    config_status: Option<ConfigStatus>,
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
        embedder: GuardrailEmbedderSlot,
    ) -> Arc<Self> {
        Self::new_with_status(snapshot, bedrock_endpoint_url, embedder, None)
    }

    /// Construct the legacy whole-environment chain with configuration-build
    /// reporting. Production uses [`LiveGuardrailIndex`], but this exported
    /// adapter must preserve the same status contract for direct consumers.
    pub fn new_with_status(
        snapshot: SnapshotHandle<AisixSnapshot>,
        bedrock_endpoint_url: Option<String>,
        embedder: GuardrailEmbedderSlot,
        config_status: Option<ConfigStatus>,
    ) -> Arc<Self> {
        // Read version before load so that a concurrent store() between
        // the two reads causes current() to see a version bump and rebuild,
        // rather than caching stale data under the new version.
        let last_version = snapshot.version();
        let snap = snapshot.load();
        let (chain, rejected) = build_chain_from_snapshot_reported(
            &snap.guardrails,
            bedrock_endpoint_url.as_deref(),
            &embedder,
        );
        publish_build_rejections(config_status.as_ref(), rejected);
        Arc::new(Self {
            snapshot,
            bedrock_endpoint_url,
            embedder,
            config_status,
            cache: Mutex::new(Cache {
                last_version,
                chain: Arc::new(chain),
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
            let (chain, rejected) = build_chain_from_snapshot_reported(
                &snap.guardrails,
                self.bedrock_endpoint_url.as_deref(),
                &self.embedder,
            );
            publish_build_rejections(self.config_status.as_ref(), rejected);
            cache.chain = Arc::new(chain);
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
///
/// **Attachments are the whole of a guardrail's scope.** A guardrail with no
/// enabled attachment governs nothing — it is not a no-op by oversight but by
/// definition, because an attachment is the only thing that says which traffic
/// the rule inspects.
///
/// This used to have a backward-compat arm that applied a guardrail carrying
/// ZERO attachment rows to the entire environment at priority 0, for the P0c
/// rolling-upgrade window where the data plane had attachments and the control
/// plane had not written any yet. That window closed, and the arm was actively
/// harmful: because it keyed on the ABSENCE of rows, removing a guardrail's
/// last attachment WIDENED it. Deleting the one model a guardrail was scoped
/// to promoted that guardrail to the whole environment (AISIX-Cloud#1450) —
/// the exact opposite of what the operator asked for, silently, on a security
/// control. Scope now comes from the attachments and from nothing else.
pub fn build_index_from_snapshot(
    guardrails: &ResourceTable<DomainGuardrail>,
    attachments: &ResourceTable<GuardrailAttachment>,
    bedrock_endpoint_url: Option<&str>,
    embedder: &GuardrailEmbedderSlot,
) -> GuardrailIndex {
    build_index_from_snapshot_reported(guardrails, attachments, bedrock_endpoint_url, embedder).0
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GuardrailBuildRejection {
    id: String,
    reason: String,
}

/// Build the production index and retain one failure per guardrail row.
/// Multiple attachments may reference the same row; reporting it once keeps
/// `/status/config` row-oriented like loader rejections.
fn build_index_from_snapshot_reported(
    guardrails: &ResourceTable<DomainGuardrail>,
    attachments: &ResourceTable<GuardrailAttachment>,
    bedrock_endpoint_url: Option<&str>,
    embedder: &GuardrailEmbedderSlot,
) -> (GuardrailIndex, Vec<GuardrailBuildRejection>) {
    let mut entries = Vec::new();
    let mut rejected = BTreeMap::<String, GuardrailBuildRejection>::new();

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

        let runtime_guardrail = match build_one(row, bedrock_endpoint_url, embedder) {
            Ok(Some(g)) => g,
            Ok(None) => continue, // inert (e.g. empty keyword list)
            Err(err) => {
                tracing::warn!(
                    guardrail_id = %gid,
                    error = %err,
                    "skipping guardrail with invalid config in index build",
                );
                rejected
                    .entry(gid.clone())
                    .or_insert_with(|| GuardrailBuildRejection {
                        id: gid.clone(),
                        reason: err.status_reason(),
                    });
                continue;
            }
        };

        let scope_kind = match attachment.scope_type {
            GuardrailScopeType::Env => ScopeKind::Env,
            GuardrailScopeType::Model => ScopeKind::Model,
            GuardrailScopeType::McpServer => ScopeKind::McpServer,
            GuardrailScopeType::ApiKey => ScopeKind::ApiKey,
            GuardrailScopeType::Team => ScopeKind::Team,
            GuardrailScopeType::PassthroughRoute => ScopeKind::PassthroughRoute,
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

    (
        GuardrailIndex::from_entries(entries),
        rejected.into_values().collect(),
    )
}

fn publish_build_rejections(
    config_status: Option<&ConfigStatus>,
    rejected: Vec<GuardrailBuildRejection>,
) {
    let Some(config_status) = config_status else {
        return;
    };
    let seen_at = chrono::Utc::now();
    config_status.record_build_rejections(
        rejected
            .into_iter()
            .map(|row| IncomingRejection {
                // Keep the synthetic identity in the heartbeat's kine-key
                // shape so cp-api can still derive resource_kind/id. The
                // non-UUID environment segment prevents collision with a
                // real stored key.
                identity: format!("/aisix/runtime/guardrails/{}", row.id),
                resource_kind: "guardrails".to_string(),
                resource_id: row.id,
                last_error_kind: "schema_failed".to_string(),
                last_error: row.reason,
                seen_at,
                serving_stale_since: None,
            })
            .collect(),
    );
}

/// Ids of the enabled guardrails no attachment names — the rows that are
/// loaded, counted, and inert.
///
/// A DISABLED attachment still counts as naming its guardrail: switching an
/// attachment off is the operator saying "not here, for now", and reporting
/// that as an oversight would train them to ignore the line.
fn unattached_ids(
    guardrails: &ResourceTable<DomainGuardrail>,
    attachments: &ResourceTable<GuardrailAttachment>,
) -> Vec<(String, String)> {
    let attached: std::collections::HashSet<String> = attachments
        .entries()
        .into_iter()
        .map(|a| a.value.guardrail_id.clone())
        .collect();
    let mut out: Vec<(String, String)> = guardrails
        .entries()
        .into_iter()
        .filter(|e| e.value.enabled && !attached.contains(e.id.as_str()))
        .map(|e| (e.id.clone(), e.value.name.clone()))
        .collect();
    out.sort();
    out
}

/// Names of the enabled-but-unattached guardrails, for `aisix validate` —
/// the one place an operator looks BEFORE deploying, where the runtime WARN
/// below has not had a chance to fire yet.
pub fn unattached_guardrail_names(
    guardrails: &ResourceTable<DomainGuardrail>,
    attachments: &ResourceTable<GuardrailAttachment>,
) -> Vec<String> {
    unattached_ids(guardrails, attachments)
        .into_iter()
        .map(|(_, name)| name)
        .collect()
}

/// How long a guardrail must have been present-and-unattached before the
/// notice is due.
///
/// A guardrail and its attachment are separate documents in separate outbox
/// rows, so every ordinary creation has a build where the data plane holds
/// the guardrail and not yet its attachment. Reporting there named
/// correctly-attached guardrails as inspecting nothing — permanently, since
/// the notice is deduplicated per id for the process lifetime.
///
/// The grace is measured in TIME, not in builds. A build is not a unit of
/// duration: the index is rebuilt on every snapshot version change AND
/// concurrently by every request that arrives during one rebuild, so two or
/// more builds fit between the two writes — sometimes within a single
/// version. Any "seen in the previous build" rule is therefore satisfied
/// while the attachment is still in flight. Thirty seconds is far above the
/// gap the control plane leaves — the two documents come from two separate
/// API calls a client round trip apart, each with an outbox tick behind it,
/// NOT from one batch — and short enough to name a genuinely inert rule
/// early in a deployment.
const UNATTACHED_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

/// How often `sweep_unattached_guardrails` should be called. Exported so the
/// interval and the grace stay legible together: a row is named between one
/// and two sweeps after the grace expires.
pub const UNATTACHED_SWEEP_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

/// Which enabled-but-unattached guardrails this build should report, given
/// when each guardrail id was first SEEN — attached or not — and what has
/// already been said.
///
/// Timing from first sight rather than from first sight *unattached* is what
/// separates the two states that look identical in one snapshot: a guardrail
/// whose attachment has not arrived yet, and one whose attachment is gone.
/// The first is new, so its clock has barely started; the second has been
/// around, so it is due immediately. Timing from "first seen unattached"
/// collapses them and makes a deleted attachment wait for a second build —
/// which, since builds only happen on config changes, means waiting for an
/// unrelated write that may never come.
///
/// Pure, and takes `now`, so the grace is testable without sleeping and
/// without a tracing subscriber — the shape the retired
/// `implicit_env_scope_first_seen` helper had, and for the same reason.
fn unattached_to_report(
    unattached: &[(String, String)],
    present_since: &std::collections::HashMap<String, std::time::Instant>,
    warned: &std::collections::HashSet<String>,
    now: std::time::Instant,
) -> Vec<(String, String)> {
    unattached
        .iter()
        .filter(|(id, _)| {
            !warned.contains(id)
                && present_since
                    .get(id)
                    .is_some_and(|t| now.saturating_duration_since(*t) >= UNATTACHED_GRACE)
        })
        .cloned()
        .collect()
}

/// (already reported, first build each enabled guardrail id was seen in)
type UnattachedState = (
    std::collections::HashSet<String>,
    std::collections::HashMap<String, std::time::Instant>,
);

/// Shared by the build-time notice and the startup report so a row named at
/// boot is not named again by the first rebuild.
static UNATTACHED_STATE: std::sync::OnceLock<std::sync::Mutex<UnattachedState>> =
    std::sync::OnceLock::new();

/// Cap on both halves of the state. Beyond it the notice degrades to
/// silence rather than to repetition — the right direction for a log line.
const MAX_REMEMBERED: usize = 4096;

fn unattached_state() -> std::sync::MutexGuard<'static, UnattachedState> {
    // Poison-tolerant: this state only shapes log lines, so a panic while the
    // lock was held must not wedge every later index build.
    UNATTACHED_STATE
        .get_or_init(|| {
            std::sync::Mutex::new((
                std::collections::HashSet::new(),
                std::collections::HashMap::new(),
            ))
        })
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Test-only view of the sweep's process-global state. The cap it exists to
/// check is on the CUMULATIVE set, so nothing short of driving real sweeps
/// across several generations of ids can see it.
#[cfg(test)]
fn unattached_state_for_test() -> (usize, usize) {
    let guard = unattached_state();
    (guard.0.len(), guard.1.len())
}

/// Test-only: backdate the presence stamps so a sweep sees rows that have
/// already outlived the grace, instead of the test sleeping through it.
#[cfg(test)]
fn backdate_unattached_stamps_for_test() {
    let mut guard = unattached_state();
    let aged = std::time::Instant::now()
        .checked_sub(UNATTACHED_GRACE * 2)
        .expect("test clock is past the grace");
    for stamp in guard.1.values_mut() {
        *stamp = aged;
    }
}

fn emit_unattached_notice(id: &str, name: &str) {
    tracing::warn!(
        guardrail_id = %id,
        guardrail_name = %name,
        "guardrail is enabled but has no attachment, so it inspects no traffic; \
         attach it to an environment, model, MCP server, API key, team or \
         passthrough route to put it in force",
    );
}

/// The presence timestamps for the next sweep: every enabled guardrail id,
/// keeping the timestamp an id already had.
///
/// Attached rows are timestamped too, and that is the whole point — the
/// stamp says "when this row appeared", which is what tells a guardrail
/// whose attachment has not arrived yet from one whose attachment is gone.
/// Stamping only the unattached ones collapses the two and makes a deleted
/// attachment serve out a fresh grace, as if the rule were new.
///
/// Sorted before the cap so truncation keeps the same ids each sweep,
/// rather than letting rows drift in and out and restart their clocks.
fn presence_timestamps(
    guardrails: &ResourceTable<DomainGuardrail>,
    previous: &std::collections::HashMap<String, std::time::Instant>,
    now: std::time::Instant,
) -> std::collections::HashMap<String, std::time::Instant> {
    let mut enabled: Vec<String> = guardrails
        .entries()
        .into_iter()
        .filter(|e| e.value.enabled)
        .map(|e| e.id.clone())
        .collect();
    enabled.sort();
    enabled
        .into_iter()
        .take(MAX_REMEMBERED)
        .map(|id| {
            let since = previous.get(&id).copied().unwrap_or(now);
            (id, since)
        })
        .collect()
}

/// Name the enabled guardrails that have been attached to nothing for longer
/// than the grace. Called on a timer; see `UNATTACHED_GRACE`.
///
/// A TIMER, and not the index build it used to ride on, because the thing
/// being reported does not coincide with a configuration change. A build
/// happens only when the snapshot version moves, so a notice emitted from
/// one can only ever be delivered by somebody writing something — and the
/// two cases that matter most are exactly the ones where nobody does:
///
///   - a scope target is deleted, its attachment goes with it, and the rule
///     that was screening traffic yesterday is now inert. The delete itself
///     is one version change, and if the row is younger than the grace at
///     that moment the notice is not yet due; the next build may be days
///     away, or never.
///   - a gateway restarts onto standing configuration. Nothing has changed,
///     so nothing rebuilds, so nothing is said.
///
/// Riding the build also put the emit on the request path, where
/// `LiveGuardrailIndex::current()` builds outside the lock and every request
/// arriving during one rebuild runs its own — which is how the original
/// "seen in two consecutive builds" rule managed to see two builds inside a
/// single snapshot version.
pub fn sweep_unattached_guardrails(
    guardrails: &ResourceTable<DomainGuardrail>,
    attachments: &ResourceTable<GuardrailAttachment>,
) {
    let mut guard = unattached_state();
    let (warned, present_since) = &mut *guard;

    let now = std::time::Instant::now();
    let unattached = unattached_ids(guardrails, attachments);
    let next = presence_timestamps(guardrails, present_since, now);

    for (id, name) in unattached_to_report(&unattached, &next, warned, now) {
        // Stop rather than emit-without-remembering. `next` caps what ONE
        // sweep can stamp, but `warned` is the union across every sweep, so
        // it grows with each generation of ids and nothing bounds it —
        // ~4096 rows reported, deleted, replaced, reported again. Skipping
        // only the insert would keep emitting and re-emit the same row on
        // every sweep from then on, which is a log flood rather than the
        // silence this cap promises.
        if warned.len() >= MAX_REMEMBERED {
            break;
        }
        emit_unattached_notice(&id, &name);
        warned.insert(id);
    }
    // Rebuilt rather than merged, so a guardrail that left the snapshot
    // drops out and one that comes back starts a fresh clock.
    *present_since = next;
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
    embedder: GuardrailEmbedderSlot,
    /// Per-execution telemetry receiver, attached to every resolved chain
    /// (AISIX-Cloud#1076). `None` (tests, standalone construction) records
    /// nothing; the server bootstrap wires the metrics layer's sink.
    metrics_sink: Option<Arc<dyn aisix_core::GuardrailMetricsSink>>,
    /// Load-observability handle. Production supplies it so a row that
    /// passes the lenient loader but fails runtime construction is visible on
    /// `/status/config` and the managed heartbeat.
    config_status: Option<ConfigStatus>,
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
        Self::new_with_sink(
            snapshot,
            bedrock_endpoint_url,
            None,
            GuardrailEmbedderSlot::none(),
        )
    }

    /// Like [`LiveGuardrailIndex::new`], also attaching a metrics sink to
    /// every chain [`LiveGuardrailIndex::resolve`] hands out and the
    /// process's embedding dispatcher for `kind: "semantic"` rows.
    pub fn new_with_sink(
        snapshot: SnapshotHandle<AisixSnapshot>,
        bedrock_endpoint_url: Option<String>,
        metrics_sink: Option<Arc<dyn aisix_core::GuardrailMetricsSink>>,
        embedder: GuardrailEmbedderSlot,
    ) -> Arc<Self> {
        Self::new_with_sink_and_status(snapshot, bedrock_endpoint_url, metrics_sink, embedder, None)
    }

    /// Production constructor: additionally reports lazy index-build
    /// failures through the shared configuration status.
    pub fn new_with_sink_and_status(
        snapshot: SnapshotHandle<AisixSnapshot>,
        bedrock_endpoint_url: Option<String>,
        metrics_sink: Option<Arc<dyn aisix_core::GuardrailMetricsSink>>,
        embedder: GuardrailEmbedderSlot,
        config_status: Option<ConfigStatus>,
    ) -> Arc<Self> {
        // Read version before load — same ordering discipline as LiveGuardrailChain.
        let last_version = snapshot.version();
        let snap = snapshot.load();
        let (index, rejected) = build_index_from_snapshot_reported(
            &snap.guardrails,
            &snap.guardrail_attachments,
            bedrock_endpoint_url.as_deref(),
            &embedder,
        );
        publish_build_rejections(config_status.as_ref(), rejected);
        Arc::new(Self {
            snapshot,
            bedrock_endpoint_url,
            embedder,
            metrics_sink,
            config_status,
            cache: Mutex::new(IndexCache {
                last_version,
                index: Arc::new(index),
            }),
        })
    }

    fn current(&self) -> Arc<GuardrailIndex> {
        loop {
            let build_version = self.snapshot.version();

            // Fast path: return cached index without building.
            {
                let cache = self
                    .cache
                    .lock()
                    .expect("LiveGuardrailIndex mutex poisoned");
                if cache.last_version >= build_version {
                    return Arc::clone(&cache.index);
                }
            }

            // Build outside the lock. A snapshot swap during this work makes
            // the result obsolete; try_install_index rejects it and the loop
            // retries from the newer version.
            let snap = self.snapshot.load();
            if self.snapshot.version() != build_version {
                continue;
            }
            let (new_index, rejected) = build_index_from_snapshot_reported(
                &snap.guardrails,
                &snap.guardrail_attachments,
                self.bedrock_endpoint_url.as_deref(),
                &self.embedder,
            );
            if let Some(index) = self.try_install_index(build_version, new_index, rejected) {
                return index;
            }
        }
    }

    /// Install a completed build only when it is still the newest snapshot.
    /// Publishing status while holding the same cache lock makes cache and
    /// rejection state advance in one monotonic version order.
    fn try_install_index(
        &self,
        build_version: u64,
        new_index: GuardrailIndex,
        rejected: Vec<GuardrailBuildRejection>,
    ) -> Option<Arc<GuardrailIndex>> {
        let mut cache = self
            .cache
            .lock()
            .expect("LiveGuardrailIndex mutex poisoned");
        if cache.last_version >= build_version || self.snapshot.version() != build_version {
            return None;
        }
        cache.index = Arc::new(new_index);
        cache.last_version = build_version;
        publish_build_rejections(self.config_status.as_ref(), rejected);
        Some(Arc::clone(&cache.index))
    }

    /// Resolve the guardrail chain applicable to `ctx`.
    ///
    /// Cheap on the cache-hit path (one lock acquire + version compare +
    /// arc clone + `O(n)` linear walk over attachment rows). Rebuilds only
    /// on snapshot version change.
    ///
    /// An empty index (no guardrails configured — the default) resolves
    /// to the same empty chain for every request, so that case returns
    /// early: no resolve walk, no applied-set copy, no sink attach (a
    /// chain with no members never reports to the sink). This is the one
    /// chokepoint every endpoint family resolves through, so the fast
    /// path covers them all.
    pub fn resolve(&self, ctx: &RequestContext<'_>) -> GuardrailChain {
        let index = self.current();
        if index.is_empty() {
            return GuardrailChain::empty();
        }
        let chain = index
            .resolve(ctx)
            .with_metrics_sink(self.metrics_sink.clone());
        // A fresh audit log per resolve — and a chain is resolved exactly
        // once per request — is what makes the log request-scoped
        // (AISIX-Cloud#1330). Skipped when nothing matched: a non-empty
        // index still resolves to an empty chain for every request outside
        // the scope of its one team-scoped row, and those must not pay an
        // allocation for a log no member can ever write to.
        if chain.is_empty() {
            return chain;
        }
        chain.with_audit_log(Some(Arc::new(crate::GuardrailAuditLog::new())))
    }

    /// `true` when the index has no entries — no enabled attachment names a
    /// guardrail this build can run. Callers use it to skip chain allocation
    /// on the hot path.
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

    #[test]
    fn an_empty_custom_script_is_rejected_rather_than_admitting_everything() {
        // `script` defaults at the type level so a row an older or broken
        // writer produced still deserializes (AGENTS.md), which makes the
        // empty case reachable here. An empty ES module parses and exports
        // nothing, so every hook would return Allow — a row that reads as
        // configured and screens nothing. It must not build.
        let row = parse(
            r#"{
                "id": "00000000-0000-0000-0000-0000000000ff",
                "name": "empty-script",
                "enabled": true,
                "hook_point": "both",
                "kind": "custom",
                "script": "   "
            }"#,
        );
        match build_one(&row, None, &GuardrailEmbedderSlot::none()) {
            Err(BuildError::InvalidValue {
                field: "script", ..
            }) => {}
            Err(other) => panic!("wrong error: {other:?}"),
            Ok(_) => panic!("an empty script must not build"),
        }
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());

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

    #[tokio::test]
    async fn custom_monitor_reason_cannot_enter_usage_telemetry() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "custom-monitor",
            "g-1",
            parse(
                r#"{
                    "name": "custom-monitor",
                    "enforcement_mode": "monitor",
                    "kind": "custom",
                    "secrets": {"TOKEN": "telemetry-secret"},
                    "script": "export function checkInput(ctx) { return { action: 'block', reason: ctx.secrets.TOKEN + ':' + ctx.text }; }"
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());

        let (verdict, hits) = chain
            .check_input_observed(&req("customer-private-text"))
            .await;
        assert_eq!(verdict, GuardrailVerdict::Allow);
        assert_eq!(hits.len(), 1, "hits: {hits:?}");
        assert_eq!(hits[0].action, "would_block");
        assert!(hits[0].reason.is_empty(), "hits: {hits:?}");
        let wire = serde_json::to_string(&hits).unwrap();
        assert!(!wire.contains("telemetry-secret"), "{wire}");
        assert!(!wire.contains("customer-private-text"), "{wire}");
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
        assert!(
            chain
                .check_input(&req("here is AKIAEXAMPLE"))
                .await
                .is_block(),
            "unknown mode must default to block, not silently pass through",
        );
    }

    /// AISIX-Cloud#1334: a capture-group custom pattern with a
    /// `replacement` builds and rewrites only group 1 through the chain.
    #[test]
    fn pii_custom_pattern_replacement_and_group_build_from_row() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "eda",
            "g-1",
            parse(
                r#"{
                    "name": "eda",
                    "kind": "pii",
                    "custom_patterns": [{
                        "name": "eda_version",
                        "regex": "version\\s*:\\s*(\\d+(?:\\.\\d+)+)",
                        "action": "mask",
                        "replacement": "***"
                    }]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
        assert_eq!(chain.len(), 1);
        let r = chain.redact_input_text("tool version: 12.1 done").unwrap();
        assert_eq!(r.text, "tool version: *** done");
    }

    /// A `replacement` on a pattern whose effective action is `block`
    /// (explicit or via `default_action`) rejects the row — the knob
    /// would otherwise be accepted but never read (#963).
    #[test]
    fn pii_replacement_on_block_action_rejects_row() {
        for row in [
            // Explicit per-pattern block.
            r#"{
                "name": "bad-explicit",
                "kind": "pii",
                "custom_patterns": [{
                    "name": "p", "regex": "x(y)", "action": "block", "replacement": "*"
                }]
            }"#,
            // Inherited block via default_action.
            r#"{
                "name": "bad-inherited",
                "kind": "pii",
                "default_action": "block",
                "custom_patterns": [{
                    "name": "p", "regex": "x(y)", "replacement": "*"
                }]
            }"#,
        ] {
            let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
            table.insert(entry("bad", "g-1", parse(row)));
            let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
            assert!(
                chain.is_empty(),
                "replacement+block row must be skipped, not half-honored",
            );
        }
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
        let g = build_one(&row, None, &GuardrailEmbedderSlot::none())
            .unwrap()
            .unwrap();
        assert_eq!(
            g.check_input(&req("hello")).await,
            GuardrailVerdict::Allow,
            "monitor mode must downgrade a fail-closed provider-failure Block",
        );
    }

    /// A monitor row that suppressed a Block still reports it, so the
    /// operator can see what would have happened.
    #[tokio::test]
    async fn downgraded_unavailability_block_still_emits_would_block() {
        use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let row = aliyun_row_against(&server.uri(), r#", "enforcement_mode": "monitor""#);
        let g = build_one(&row, None, &GuardrailEmbedderSlot::none())
            .unwrap()
            .unwrap();
        let (verdict, hits) = g.check_input_observed(&req("hello")).await;
        assert_eq!(verdict, GuardrailVerdict::Allow, "monitor downgrades it");
        assert_eq!(hits.len(), 1, "the suppression is still reported");
        assert_eq!(hits[0].action, "would_block");
    }

    /// Monitor mode is unconditional: a provider outage is still only
    /// observed, fail-closed default included. There is no configuration
    /// that makes a monitored row block — that is what `block` mode is
    /// for.
    #[tokio::test]
    async fn monitor_never_blocks_on_provider_failure() {
        use wiremock::{matchers::method, Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let row = aliyun_row_against(&server.uri(), r#", "enforcement_mode": "monitor""#);
        let g = build_one(&row, None, &GuardrailEmbedderSlot::none())
            .unwrap()
            .unwrap();
        assert_eq!(
            g.check_input(&req("hello")).await,
            GuardrailVerdict::Allow,
            "a monitored row must not block, whatever the failure",
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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

        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
        // Both rows compose. We don't probe the bedrock arm — its
        // own tests cover the dispatch path; this one only pins the
        // chain composition contract.
        assert_eq!(chain.len(), 2);
    }

    #[tokio::test]
    async fn live_chain_rebuilds_on_snapshot_swap() {
        let initial = AisixSnapshot::new();
        let handle = SnapshotHandle::new(initial);
        let live = LiveGuardrailChain::new(handle.clone(), None, GuardrailEmbedderSlot::none());

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

    #[tokio::test]
    async fn live_chain_reports_and_clears_build_rejections() {
        let broken = AisixSnapshot::new();
        broken.guardrails.insert(entry(
            "broken",
            "g-1",
            parse(
                r#"{
                    "name": "sensitive-name-do-not-expose",
                    "kind": "custom",
                    "script": "sensitive-script-do-not-expose export function checkInput( {"
                }"#,
            ),
        ));
        let handle = SnapshotHandle::new(broken);
        let status = ConfigStatus::new(aisix_core::SourceKind::Etcd);
        let live = LiveGuardrailChain::new_with_status(
            handle.clone(),
            None,
            GuardrailEmbedderSlot::none(),
            Some(status.clone()),
        );
        assert_eq!(status.view().rejected.len(), 1);
        assert_eq!(
            status.view().rejected[0].last_error,
            "guardrail runtime build failed: compile_failed at config.script"
        );
        let public_status = serde_json::to_string(&status.view()).unwrap();
        assert!(!public_status.contains("sensitive-name-do-not-expose"));
        assert!(!public_status.contains("sensitive-script-do-not-expose"));

        let fixed = AisixSnapshot::new();
        fixed.guardrails.insert(entry(
            "fixed",
            "g-1",
            parse(
                r#"{
                    "name": "fixed",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        handle.store(fixed);
        assert!(live.check_input(&req("AKIA")).await.is_block());
        assert!(status.view().rejected.is_empty());
    }

    #[test]
    fn runtime_status_redacts_invalid_regex_values() {
        let broken = AisixSnapshot::new();
        broken.guardrails.insert(entry(
            "private-name-do-not-expose",
            "g-private",
            parse(
                r#"{
                    "name": "private-name-do-not-expose",
                    "kind": "keyword",
                    "patterns": [{"kind":"regex","value":"private-pattern-do-not-expose[("}]
                }"#,
            ),
        ));
        let status = ConfigStatus::new(aisix_core::SourceKind::Etcd);
        let _live = LiveGuardrailChain::new_with_status(
            SnapshotHandle::new(broken),
            None,
            GuardrailEmbedderSlot::none(),
            Some(status.clone()),
        );

        let body = serde_json::to_string(&status.view()).unwrap();
        assert!(body.contains("invalid_regex at config.patterns[].value"));
        assert!(!body.contains("private-pattern-do-not-expose"));
        assert!(!body.contains("private-name-do-not-expose"));
        let heartbeat = status.rejection_snapshots();
        assert_eq!(heartbeat.len(), 1);
        assert_eq!(
            heartbeat[0].error,
            "guardrail runtime build failed: invalid_regex at config.patterns[].value"
        );
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
        let chain =
            build_chain_from_snapshot(&shuffled_table(), None, &GuardrailEmbedderSlot::none());
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
        assert_eq!(chain.member_names(), ["y", "x", "z"]);
    }

    /// The production per-request path. Every guardrail here carries an
    /// env attachment at the same priority, so nothing but the build's own
    /// iteration order separates them — which must be deterministic, or the
    /// DashMap's run-to-run order decides which Block fires first
    /// (#519 B.4a).
    #[test]
    fn index_ties_resolve_in_attachment_id_order() {
        // The index sorts by (priority desc, scope-specificity desc) with a
        // STABLE sort, so insertion order decides what is left — and insertion
        // walks attachments sorted by their own id. Every attachment here is
        // env-scope at the same priority, so attachment id is the ONLY
        // tiebreak; without the build-time sort the DashMap's arbitrary,
        // run-to-run-varying order would make this flap (#519 B.4a).
        //
        // Guardrail `created_at` deliberately does not decide this order: that
        // is the chain path's rule (`chain_order_is_created_at_ascending_with_id_tiebreak`),
        // and the two are different mechanisms. This test used to exercise the
        // index through the zero-attachment fallback, which borrowed the
        // chain's ordering; with that fallback retired (AISIX-Cloud#1450) the
        // index is reached only through explicit attachments.
        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();
        for (i, (guardrail_id, _, _)) in SHUFFLED_ROWS.iter().enumerate() {
            // Attachment ids run opposite to the row order they attach, so a
            // pass that forgot to sort would surface as the insertion order.
            let attachment_id = format!("a-{:02}", SHUFFLED_ROWS.len() - i);
            attachments.insert(attachment_entry(
                &attachment_id,
                parse_attachment(&format!(
                    r#"{{ "guardrail_id": "{guardrail_id}", "scope_type": "env", "priority": 100 }}"#
                )),
            ));
        }

        let index = build_index_from_snapshot(
            &shuffled_table(),
            &attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
        let chain = index.resolve(&RequestContext {
            passthrough_route_id: "",
            model_id: "m",
            mcp_server_id: "",
            api_key_id: "k",
            team_id: None,
        });
        // a-01 attaches the LAST row, a-10 the first — so ascending attachment
        // id walks SHUFFLED_ROWS backwards.
        let expected: Vec<&str> = SHUFFLED_ROWS
            .iter()
            .rev()
            .map(|(_, name, _)| *name)
            .collect();
        assert_eq!(chain.member_names(), expected);
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

        let index = build_index_from_snapshot(
            &guardrails,
            &attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
        assert_eq!(index.len(), 1);

        let ctx = RequestContext {
            passthrough_route_id: "",
            model_id: "m1",
            mcp_server_id: "",
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

        let index = build_index_from_snapshot(
            &guardrails,
            &attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
        assert_eq!(index.len(), 0);
        // Verify the guardrail does not fire (not just that the index is empty).
        let ctx = RequestContext {
            passthrough_route_id: "",
            model_id: "m",
            mcp_server_id: "",
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
        // A guardrail with one enabled + one disabled attachment must fire
        // exactly once — through the enabled one, with the disabled row
        // contributing no second entry that `resolve` would have to dedupe.
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

        let index = build_index_from_snapshot(
            &guardrails,
            &attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
        // Exactly one entry — from the enabled attachment only.
        // The disabled attachment must NOT produce a second entry or trigger the fallback.
        assert_eq!(
            index.len(),
            1,
            "enabled+disabled attachments: exactly 1 entry expected",
        );
        let ctx = RequestContext {
            passthrough_route_id: "",
            model_id: "any",
            mcp_server_id: "",
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

    /// A guardrail and its attachment arrive as separate documents, so a build
    /// falling between them holds the guardrail alone. Reporting there named a
    /// correctly-attached guardrail as inspecting nothing — permanently, since
    /// the notice is deduplicated per id for the process lifetime.
    #[test]
    fn the_unattached_notice_waits_out_the_grace() {
        use std::collections::{HashMap, HashSet};
        use std::time::{Duration, Instant};

        let rows = vec![("g-1".to_string(), "one".to_string())];
        let t0 = Instant::now();
        let mut first_seen: HashMap<String, Instant> = HashMap::new();
        let mut warned: HashSet<String> = HashSet::new();

        assert!(
            unattached_to_report(&rows, &first_seen, &warned, t0).is_empty(),
            "an id with no recorded first sighting is inside its attachment's window",
        );
        first_seen.insert("g-1".to_string(), t0);

        assert!(
            unattached_to_report(&rows, &first_seen, &warned, t0 + UNATTACHED_GRACE / 2).is_empty(),
            "half a grace in is still the window, however many builds have run",
        );

        let due = unattached_to_report(&rows, &first_seen, &warned, t0 + UNATTACHED_GRACE);
        assert_eq!(
            due.len(),
            1,
            "unattached for the whole grace is a standing property, not a race",
        );
        warned.insert("g-1".to_string());

        assert!(
            unattached_to_report(
                &rows,
                &first_seen,
                &warned,
                t0 + UNATTACHED_GRACE + Duration::from_secs(600),
            )
            .is_empty(),
            "and it is said once, not on every rebuild after",
        );
    }

    /// `warned` is the union across every sweep, not a per-sweep set, so
    /// the cap on what one sweep can stamp does not bound it. Rows get
    /// reported, deleted, and replaced by a new generation of ids, and the
    /// set grows with each one — this asserted otherwise in a comment
    /// before it was measured.
    ///
    /// The only test in this file that touches the process-global state;
    /// nothing else drives `sweep_unattached_guardrails`, so it owns it.
    #[test]
    fn the_reported_set_stops_growing_at_the_cap() {
        let attachments: ResourceTable<GuardrailAttachment> = ResourceTable::default();
        for generation in 0..3 {
            let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
            for i in 0..MAX_REMEMBERED {
                guardrails.insert(entry(
                    "secrets",
                    &format!("g-{generation}-{i}"),
                    parse(
                        r#"{
                            "name": "inert",
                            "kind": "keyword",
                            "patterns": [{ "kind": "literal", "value": "AKIA" }]
                        }"#,
                    ),
                ));
            }
            // First sweep stamps the generation, the backdate ages it past
            // the grace, the second reports it — the shape of a fleet that
            // churns guardrails faster than the cap.
            sweep_unattached_guardrails(&guardrails, &attachments);
            backdate_unattached_stamps_for_test();
            sweep_unattached_guardrails(&guardrails, &attachments);
        }

        let (warned, present_since) = unattached_state_for_test();
        assert!(
            warned <= MAX_REMEMBERED,
            "the reported set grew to {warned}, past its {MAX_REMEMBERED} cap",
        );
        assert!(
            present_since <= MAX_REMEMBERED,
            "stamps grew to {present_since}"
        );

        // …and past the cap it goes SILENT, rather than emitting rows it
        // cannot remember. Stopping and skipping-the-insert both keep the
        // set at the cap, so the size assertion above cannot tell them
        // apart — and the second one re-emits the same rows on every sweep
        // for the life of the process, which is the louder failure.
        crate::keep_callsites_enabled();
        let _capture_guard = crate::TRACING_CAPTURE_LOCK.blocking_lock();
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(CaptureWriter(buf.clone()))
            .finish();
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            let overflow: ResourceTable<DomainGuardrail> = ResourceTable::default();
            overflow.insert(entry(
                "secrets",
                "g-past-the-cap",
                parse(
                    r#"{
                        "name": "past-the-cap",
                        "kind": "keyword",
                        "patterns": [{ "kind": "literal", "value": "AKIA" }]
                    }"#,
                ),
            ));
            sweep_unattached_guardrails(&overflow, &attachments);
            backdate_unattached_stamps_for_test();
            sweep_unattached_guardrails(&overflow, &attachments);
        }
        let logged = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(
            !logged.contains("past-the-cap"),
            "reported a row it could not remember; it will say the same thing \
             again on every sweep from now on:\n{logged}",
        );
    }

    /// Minimal `MakeWriter` for the capture above; the crate's other
    /// capture helper is async and this test is not.
    #[derive(Clone)]
    struct CaptureWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CaptureWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureWriter {
        type Writer = CaptureWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// The grace's VALUE, not just its arithmetic. Every other test here
    /// expresses its instants as multiples of `UNATTACHED_GRACE`, so they
    /// hold for any positive value — including one small enough to put the
    /// original bug back, where a guardrail is named while the second of
    /// its two documents is still in flight.
    ///
    /// Two lower bounds, both load-bearing rather than arbitrary. The
    /// sweep is the only thing that can observe a row, so a grace shorter
    /// than two sweep periods can be satisfied by the first two sweeps
    /// after a guardrail appears — which is where its attachment still is.
    /// And the absolute floor is about the writer, not the reader. The
    /// guardrail and its attachment are not one write: the console creates
    /// the guardrail, then reconciles its attachments in a second API call,
    /// so the gap is a client round trip plus the outbox tick behind each —
    /// and the grace has to outlast that by a wide margin on a loaded
    /// control plane, not merely beat its median.
    #[test]
    fn the_grace_outlasts_two_sweeps_and_a_slow_two_document_write() {
        assert!(
            UNATTACHED_GRACE >= UNATTACHED_SWEEP_INTERVAL * 2,
            "grace {UNATTACHED_GRACE:?} must span at least two sweeps of \
             {UNATTACHED_SWEEP_INTERVAL:?}",
        );
        assert!(
            UNATTACHED_GRACE >= std::time::Duration::from_secs(20),
            "grace {UNATTACHED_GRACE:?} is too short to outlast a slow outbox drain",
        );
    }

    /// The clock is kept for ATTACHED guardrails too. That is the whole of
    /// what makes a deleted attachment reportable: without it the row is
    /// "first seen" on the build its attachment goes, serves out a fresh
    /// grace as if it were new, and — since a sweep only reports what has
    /// outlived the grace — the moment that mattered is gone.
    ///
    /// Pinned here rather than only through the sweep, because stamping the
    /// unattached rows alone leaves every other test in this file green.
    #[test]
    fn presence_is_timestamped_for_attached_rows_too() {
        use std::collections::HashMap;
        use std::time::Instant;

        let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
        guardrails.insert(entry(
            "secrets",
            "g-attached",
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
                r#"{ "guardrail_id": "g-attached", "scope_type": "env", "priority": 100 }"#,
            ),
        ));
        assert!(
            unattached_ids(&guardrails, &attachments).is_empty(),
            "fixture must be attached, or this test proves nothing",
        );

        let t0 = Instant::now();
        let stamped = presence_timestamps(&guardrails, &HashMap::new(), t0);
        assert_eq!(
            stamped.get("g-attached"),
            Some(&t0),
            "an attached guardrail must carry a timestamp; without it, losing its \
             attachment later reads as a brand-new row",
        );

        // …and the stamp survives, so age accumulates while it is attached.
        let later = t0 + UNATTACHED_GRACE * 3;
        let carried = presence_timestamps(&guardrails, &stamped, later);
        assert_eq!(
            carried.get("g-attached"),
            Some(&t0),
            "the clock must not restart on every sweep",
        );

        // Which is what makes the attachment's removal reportable at once.
        let orphaned = vec![("g-attached".to_string(), "block-secrets".to_string())];
        assert_eq!(
            unattached_to_report(
                &orphaned,
                &carried,
                &std::collections::HashSet::new(),
                later,
            )
            .len(),
            1,
        );
    }

    /// A guardrail that has been around and just lost its attachment is due
    /// at once. This is the one case the notice exists for — a scope target
    /// was deleted and the rule is now inert — and timing from "first seen
    /// UNATTACHED" instead of "first seen" would make it wait for a second
    /// build, i.e. for an unrelated config write that may never come.
    #[test]
    fn losing_an_attachment_is_reported_on_the_next_build() {
        use std::collections::{HashMap, HashSet};
        use std::time::Instant;

        let rows = vec![("g-1".to_string(), "one".to_string())];
        let created = Instant::now();
        let mut present_since: HashMap<String, Instant> = HashMap::new();
        // Present, and attached, since well before the grace.
        present_since.insert("g-1".to_string(), created);
        let warned: HashSet<String> = HashSet::new();

        let due = unattached_to_report(
            &rows,
            &present_since,
            &warned,
            created + UNATTACHED_GRACE * 2,
        );
        assert_eq!(
            due.len(),
            1,
            "a row that has outlived the grace is due the build its attachment goes",
        );
    }

    /// The clock has to be measured in time, not in builds. Two builds fit
    /// between a guardrail and its attachment whenever anything else is being
    /// written, so a consecutive-build rule reports a correctly-attached
    /// guardrail under ordinary config churn — the very bug it was meant to
    /// close, just harder to reproduce.
    #[test]
    fn many_builds_inside_the_grace_still_report_nothing() {
        use std::collections::{HashMap, HashSet};
        use std::time::Instant;

        let rows = vec![("g-1".to_string(), "one".to_string())];
        let t0 = Instant::now();
        let mut first_seen: HashMap<String, Instant> = HashMap::new();
        first_seen.insert("g-1".to_string(), t0);
        let warned: HashSet<String> = HashSet::new();

        for i in 1..=50 {
            let now = t0 + (UNATTACHED_GRACE / 100) * i;
            assert!(
                unattached_to_report(&rows, &first_seen, &warned, now).is_empty(),
                "build {i} is inside the grace and must stay quiet",
            );
        }
    }

    #[tokio::test]
    async fn guardrail_without_attachment_governs_nothing() {
        // AISIX-Cloud#1450. This used to assert the opposite: a guardrail with
        // ZERO attachment rows fired on every request, a rolling-upgrade
        // fallback from the window where the data plane read attachments and
        // the control plane had not written any yet.
        //
        // Keying on the ABSENCE of rows made removing a guardrail's last
        // attachment a scope WIDENING — delete the one model it was scoped to
        // and it started inspecting the entire environment. An attachment is
        // the only thing that says which traffic a rule sees, so no attachment
        // now means no traffic.
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

        let index = build_index_from_snapshot(
            &guardrails,
            &attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
        assert_eq!(
            index.len(),
            0,
            "an unattached guardrail must not enter the index",
        );

        let ctx = RequestContext {
            passthrough_route_id: "",
            model_id: "any-model",
            mcp_server_id: "",
            api_key_id: "any-key",
            team_id: None,
        };
        assert!(
            !index
                .resolve(&ctx)
                .check_input(&req("here AKIA"))
                .await
                .is_block(),
            "an unattached guardrail must not block anything, however well its pattern matches",
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

        let index = build_index_from_snapshot(
            &guardrails,
            &attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
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

        let index = build_index_from_snapshot(
            &guardrails,
            &attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
        assert_eq!(index.len(), 0);
    }

    // --- similarity scores reach the request's log (AISIX-Cloud#1467) -----

    /// Every text embeds to the same unit vector, so a candidate scores
    /// exactly 1.0 against any example — a deterministic "over threshold"
    /// with no bearing on which real model is configured.
    struct IdenticalEmbedder;

    #[async_trait]
    impl crate::GuardrailEmbedder for IdenticalEmbedder {
        async fn embed(
            &self,
            _model_alias: &str,
            texts: &[String],
            _cacheable: bool,
            _timeout: std::time::Duration,
        ) -> Result<Vec<Vec<f32>>, crate::EmbedFailure> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    fn semantic_row(enforcement_mode: &str) -> DomainGuardrail {
        parse(&format!(
            r#"{{
                "name": "semantic-row",
                "enabled": true,
                "hook_point": "input",
                "enforcement_mode": "{enforcement_mode}",
                "kind": "semantic",
                "embedding_model": "embed-1",
                "deny_examples": ["forbidden topic"],
                "deny_threshold": 0.75
            }}"#
        ))
    }

    fn semantic_index(enforcement_mode: &str) -> Arc<LiveGuardrailIndex> {
        let guardrails: ResourceTable<DomainGuardrail> = ResourceTable::default();
        guardrails.insert(entry("semantic", "g-1", semantic_row(enforcement_mode)));
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
        let mut snap = AisixSnapshot::new();
        snap.guardrails = guardrails;
        snap.guardrail_attachments = attachments;
        LiveGuardrailIndex::new_with_sink(
            SnapshotHandle::new(snap),
            None,
            None,
            GuardrailEmbedderSlot::new(Arc::new(IdenticalEmbedder)),
        )
    }

    fn any_ctx() -> RequestContext<'static> {
        RequestContext {
            passthrough_route_id: "",
            model_id: "m1",
            mcp_server_id: "",
            api_key_id: "k1",
            team_id: None,
        }
    }

    #[tokio::test]
    async fn a_resolved_chain_carries_the_semantic_score() {
        // The bind happens at resolve time, where the per-request log is
        // minted — the index's own member is shared and cannot hold one.
        let chain = semantic_index("block").resolve(&any_ctx());
        assert!(chain.check_input(&req("anything")).await.is_block());

        let scores = chain.scores();
        assert_eq!(scores.len(), 1, "{scores:?}");
        assert_eq!(scores[0].guardrail_name, "semantic-row");
        assert_eq!(scores[0].hook, "input");
        assert_eq!(scores[0].direction, "deny");
        assert_eq!(scores[0].embedding_model, "embed-1");
        assert!(scores[0].matched);
    }

    #[tokio::test]
    async fn a_monitor_mode_row_scores_through_the_decorator() {
        // Monitor mode is where an operator tunes a threshold, so it is the
        // last place the score may go missing. The decorator wraps the
        // semantic guardrail, so it has to forward the bind — without that
        // forward the chain resolves, the row runs, and the array is empty.
        let chain = semantic_index("monitor").resolve(&any_ctx());
        let (verdict, hits) = chain.check_input_observed(&req("anything")).await;
        assert!(
            matches!(verdict, GuardrailVerdict::Allow),
            "monitor never blocks: {verdict:?}"
        );
        assert_eq!(hits.len(), 1, "the suppressed block is still observed");

        let scores = chain.scores();
        assert_eq!(scores.len(), 1, "{scores:?}");
        assert_eq!(scores[0].guardrail_name, "semantic-row");
        assert!(scores[0].matched);
    }

    #[tokio::test]
    async fn a_chain_with_no_scoring_member_reports_nothing() {
        let table: ResourceTable<DomainGuardrail> = ResourceTable::default();
        table.insert(entry(
            "secrets",
            "g-1",
            parse(
                r#"{
                    "name": "block-secrets",
                    "enabled": true,
                    "hook_point": "input",
                    "kind": "keyword",
                    "patterns": [
                        { "kind": "literal", "value": "AKIA" }
                    ]
                }"#,
            ),
        ));
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none())
            .with_audit_log(Some(Arc::new(crate::GuardrailAuditLog::new())));
        assert!(chain.check_input(&req("AKIA-EXAMPLE")).await.is_block());
        assert!(chain.scores().is_empty());
    }

    #[tokio::test]
    async fn live_index_rebuilds_on_snapshot_swap() {
        let initial = AisixSnapshot::new();
        let handle = SnapshotHandle::new(initial);
        let live = LiveGuardrailIndex::new(handle.clone(), None);

        let ctx = RequestContext {
            passthrough_route_id: "",
            model_id: "m1",
            mcp_server_id: "",
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
    async fn live_index_reports_and_clears_lazy_build_rejections() {
        let broken = AisixSnapshot::new();
        broken.guardrails.insert(entry(
            "broken",
            "g-1",
            parse(
                r#"{
                    "name": "broken-script",
                    "kind": "custom",
                    "script": " "
                }"#,
            ),
        ));
        broken.guardrail_attachments.insert(attachment_entry(
            "a-1",
            parse_attachment(
                r#"{
                    "guardrail_id": "g-1",
                    "scope_type": "env",
                    "priority": 50
                }"#,
            ),
        ));
        // A blank semantic model is deliberately valid on the lenient read
        // path: the built guardrail fails closed at request time. It must not
        // be reclassified as a build rejection, which would drop the row and
        // admit traffic instead.
        broken.guardrails.insert(entry(
            "semantic-empty-model",
            "g-2",
            parse(
                r#"{
                    "name": "semantic-empty-model",
                    "kind": "semantic",
                    "embedding_model": "",
                    "deny_examples": ["forbidden"],
                    "deny_threshold": 0.75
                }"#,
            ),
        ));
        broken.guardrail_attachments.insert(attachment_entry(
            "a-2",
            parse_attachment(
                r#"{
                    "guardrail_id": "g-2",
                    "scope_type": "env",
                    "priority": 40
                }"#,
            ),
        ));
        let handle = SnapshotHandle::new(broken);
        let status = ConfigStatus::new(aisix_core::SourceKind::Etcd);
        status.record_load(aisix_core::LoadObservation {
            source_hash: "broken".into(),
            observed_revision: Some(1),
            applied: Some(aisix_core::AppliedSnapshot {
                config_hash: "broken".into(),
                revision: Some(1),
                resource_counts: [("guardrails".to_string(), 2)].into_iter().collect(),
            }),
            rejected: vec![],
            partially_compatible: vec![],
            partially_compatible_rows_by_kind: Default::default(),
            stale_served_rows_by_kind: Default::default(),
            is_reload: true,
            wholly_rejected: false,
        });
        let live = LiveGuardrailIndex::new_with_sink_and_status(
            handle.clone(),
            None,
            None,
            GuardrailEmbedderSlot::new(Arc::new(IdenticalEmbedder)),
            Some(status.clone()),
        );

        let rejected = status.view();
        assert_eq!(rejected.state, aisix_core::ConfigState::Degraded);
        assert_eq!(rejected.rejected.len(), 1, "{rejected:?}");
        let custom = rejected
            .rejected
            .iter()
            .find(|row| row.resource_id == "g-1")
            .expect("custom rejection");
        assert_eq!(custom.resource_kind, "guardrails");
        assert_eq!(
            custom.last_error,
            "guardrail runtime build failed: invalid_value at config.script"
        );
        let applied = rejected.applied.as_ref().unwrap();
        assert_eq!(applied.resource_counts["guardrails"], 1);
        assert_ne!(
            applied.config_hash.as_str(),
            rejected.source.source_hash.as_deref().unwrap()
        );
        assert_eq!(
            status
                .rejection_snapshots()
                .into_iter()
                .map(|row| row.key)
                .collect::<Vec<_>>(),
            vec!["/aisix/runtime/guardrails/g-1".to_string()],
            "heartbeat keys must retain the kine shape cp-api parses"
        );
        assert!(live
            .resolve(&any_ctx())
            .check_input(&req("anything"))
            .await
            .is_block());

        let fixed = AisixSnapshot::new();
        fixed.guardrails.insert(entry(
            "fixed",
            "g-1",
            parse(
                r#"{
                    "name": "fixed",
                    "kind": "keyword",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        fixed.guardrail_attachments.insert(attachment_entry(
            "a-1",
            parse_attachment(
                r#"{
                    "guardrail_id": "g-1",
                    "scope_type": "env",
                    "priority": 50
                }"#,
            ),
        ));
        handle.store(fixed);
        status.record_load(aisix_core::LoadObservation {
            source_hash: "fixed".into(),
            observed_revision: Some(2),
            applied: Some(aisix_core::AppliedSnapshot {
                config_hash: "fixed".into(),
                revision: Some(2),
                resource_counts: [("guardrails".to_string(), 1)].into_iter().collect(),
            }),
            rejected: vec![],
            partially_compatible: vec![],
            partially_compatible_rows_by_kind: Default::default(),
            stale_served_rows_by_kind: Default::default(),
            is_reload: false,
            wholly_rejected: false,
        });
        // Rebuilds are deliberately lazy, so the previous signal remains
        // until the first request observes the new snapshot.
        assert_eq!(status.view().rejected.len(), 1);
        let _ = live.resolve(&any_ctx());
        assert!(status.view().rejected.is_empty());
        let repaired = status.view();
        assert_eq!(repaired.state, aisix_core::ConfigState::Synced);
        assert_eq!(repaired.applied.unwrap().resource_counts["guardrails"], 1);
    }

    #[test]
    fn older_index_build_cannot_restore_a_superseded_rejection() {
        let broken = AisixSnapshot::new();
        broken.guardrails.insert(entry(
            "broken",
            "g-1",
            parse(r#"{"name":"broken","kind":"custom","script":" "}"#),
        ));
        broken.guardrail_attachments.insert(attachment_entry(
            "a-1",
            parse_attachment(r#"{"guardrail_id":"g-1","scope_type":"env","priority":0}"#),
        ));
        let handle = SnapshotHandle::new(broken);
        let old_version = handle.version();
        let status = ConfigStatus::new(aisix_core::SourceKind::Etcd);
        status.record_load(aisix_core::LoadObservation {
            source_hash: "old".into(),
            observed_revision: Some(1),
            applied: Some(aisix_core::AppliedSnapshot {
                config_hash: "old".into(),
                revision: Some(1),
                resource_counts: [("guardrails".to_string(), 1)].into_iter().collect(),
            }),
            rejected: vec![],
            partially_compatible: vec![],
            partially_compatible_rows_by_kind: Default::default(),
            stale_served_rows_by_kind: Default::default(),
            is_reload: true,
            wholly_rejected: false,
        });
        let live = LiveGuardrailIndex::new_with_sink_and_status(
            handle.clone(),
            None,
            None,
            GuardrailEmbedderSlot::none(),
            Some(status.clone()),
        );
        assert_eq!(status.view().rejected.len(), 1);

        let fixed = AisixSnapshot::new();
        fixed.guardrails.insert(entry(
            "fixed",
            "g-1",
            parse(
                r#"{"name":"fixed","kind":"keyword","patterns":[{"kind":"literal","value":"AKIA"}]}"#,
            ),
        ));
        fixed.guardrail_attachments.insert(attachment_entry(
            "a-1",
            parse_attachment(r#"{"guardrail_id":"g-1","scope_type":"env","priority":0}"#),
        ));
        handle.store(fixed);
        let new_version = handle.version();
        status.record_load(aisix_core::LoadObservation {
            source_hash: "new".into(),
            observed_revision: Some(2),
            applied: Some(aisix_core::AppliedSnapshot {
                config_hash: "new".into(),
                revision: Some(2),
                resource_counts: [("guardrails".to_string(), 1)].into_iter().collect(),
            }),
            rejected: vec![],
            partially_compatible: vec![],
            partially_compatible_rows_by_kind: Default::default(),
            stale_served_rows_by_kind: Default::default(),
            is_reload: false,
            wholly_rejected: false,
        });

        // Hold the old result until the newer build has installed, then let
        // it race the cache/status publication in the previously-buggy order.
        let (release_old, wait_old) = std::sync::mpsc::channel();
        let old_live = Arc::clone(&live);
        let old = std::thread::spawn(move || {
            wait_old.recv().unwrap();
            old_live.try_install_index(
                old_version,
                GuardrailIndex::from_entries(Vec::new()),
                vec![GuardrailBuildRejection {
                    id: "g-1".into(),
                    reason: "obsolete rejection".into(),
                }],
            )
        });

        let snap = handle.load();
        let (new_index, new_rejected) = build_index_from_snapshot_reported(
            &snap.guardrails,
            &snap.guardrail_attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
        assert!(live
            .try_install_index(new_version, new_index, new_rejected)
            .is_some());
        release_old.send(()).unwrap();
        assert!(old.join().unwrap().is_none());

        let view = status.view();
        assert!(
            view.rejected.is_empty(),
            "obsolete result restored: {view:?}"
        );
        assert_eq!(view.applied.unwrap().config_hash, "new");
    }

    #[tokio::test]
    async fn live_index_empty_fast_path_resolves_empty_chain() {
        // Zero-config fast path: an empty index resolves to an empty
        // chain with an empty applied set — the same observable shape
        // the full resolve walk produces on an empty index.
        let live = LiveGuardrailIndex::new(SnapshotHandle::new(AisixSnapshot::new()), None);
        let chain = live.resolve(&RequestContext {
            passthrough_route_id: "",
            model_id: "m",
            mcp_server_id: "",
            api_key_id: "k",
            team_id: None,
        });
        assert!(chain.is_empty());
        assert!(chain.applied().is_empty());
        assert!(!chain.check_input(&req("anything")).await.is_block());
    }

    // -----------------------------------------------------------------------
    // per-execution metrics sink (AISIX-Cloud#1076)
    // -----------------------------------------------------------------------

    #[derive(Default)]
    struct RecordingSink(std::sync::Mutex<Vec<(String, String, &'static str, &'static str)>>);

    impl aisix_core::GuardrailMetricsSink for RecordingSink {
        fn record_guardrail_execution(&self, exec: &aisix_core::GuardrailExecution<'_>) {
            self.0.lock().unwrap().push((
                exec.guardrail_name.to_owned(),
                exec.kind.to_owned(),
                exec.phase,
                exec.result,
            ));
        }
    }

    /// A sink attached via `LiveGuardrailIndex::new_with_sink` reaches every
    /// resolved chain: executions carry the row name, the row `kind`, and
    /// the enforced result — a monitor-mode member's suppressed outcome
    /// records as `would_block`/`would_mask`, not `blocked`/`masked`.
    #[tokio::test]
    async fn live_index_sink_records_resolved_chain_executions() {
        let snap = AisixSnapshot::new();
        snap.guardrails.insert(entry(
            "watch-pii",
            "g-1",
            parse(
                r#"{
                    "name": "watch-pii",
                    "enforcement_mode": "monitor",
                    "kind": "pii",
                    "created_at": "2024-01-01T00:00:00Z",
                    "detectors": [
                        { "type": "email", "action": "mask" },
                        { "type": "us_ssn", "action": "block" }
                    ]
                }"#,
            ),
        ));
        snap.guardrails.insert(entry(
            "block-secrets",
            "g-2",
            parse(
                r#"{
                    "name": "block-secrets",
                    "kind": "keyword",
                    "created_at": "2024-01-02T00:00:00Z",
                    "patterns": [{ "kind": "literal", "value": "AKIA" }]
                }"#,
            ),
        ));
        // Both rows need an explicit env attachment: scope comes only from
        // attachments now, so an unattached guardrail never reaches the index
        // (AISIX-Cloud#1450).
        for (attachment_id, guardrail_id) in [("a-1", "g-1"), ("a-2", "g-2")] {
            snap.guardrail_attachments.insert(attachment_entry(
                attachment_id,
                parse_attachment(&format!(
                    r#"{{ "guardrail_id": "{guardrail_id}", "scope_type": "env", "priority": 100 }}"#
                )),
            ));
        }
        let sink = Arc::new(RecordingSink::default());
        let live = LiveGuardrailIndex::new_with_sink(
            SnapshotHandle::new(snap),
            None,
            Some(sink.clone()),
            GuardrailEmbedderSlot::none(),
        );
        let ctx = RequestContext {
            passthrough_route_id: "",
            model_id: "m1",
            mcp_server_id: "",
            api_key_id: "k1",
            team_id: None,
        };

        // Monitor-mode pii mask hit + keyword block: the suppressed mask
        // records as would_mask; the enforcing keyword block as blocked.
        let (v, _) = live
            .resolve(&ctx)
            .check_input_observed(&req("mail alice@example.com and AKIA"))
            .await;
        assert!(v.is_block());
        assert_eq!(
            std::mem::take(&mut *sink.0.lock().unwrap()),
            vec![
                (
                    "watch-pii".to_owned(),
                    "pii".to_owned(),
                    "input",
                    "would_mask",
                ),
                (
                    "block-secrets".to_owned(),
                    "keyword".to_owned(),
                    "input",
                    "blocked",
                ),
            ],
        );

        // Monitor-mode would_block: the suppressed pii Block records as
        // would_block while the enforced verdict stays Allow.
        let (v, _) = live
            .resolve(&ctx)
            .check_input_observed(&req("ssn 123-45-6789"))
            .await;
        assert_eq!(v, GuardrailVerdict::Allow);
        let records = std::mem::take(&mut *sink.0.lock().unwrap());
        assert_eq!(
            records[0],
            (
                "watch-pii".to_owned(),
                "pii".to_owned(),
                "input",
                "would_block",
            ),
        );

        // The plain `new` constructor keeps recording off.
        let unsinked = LiveGuardrailIndex::new(SnapshotHandle::new(AisixSnapshot::new()), None);
        let _ = unsinked.resolve(&ctx).check_input(&req("hi")).await;
        assert!(sink.0.lock().unwrap().is_empty());
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
        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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

        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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

        let chain = build_chain_from_snapshot(&table, None, &GuardrailEmbedderSlot::none());
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

        let index = build_index_from_snapshot(
            &guardrails,
            &attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
        let chain = index.resolve(&RequestContext {
            passthrough_route_id: "",
            model_id: "m-A",
            mcp_server_id: "",
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
    async fn mcp_server_attachment_builds_into_the_index() {
        // The wire `scope_type: "mcp_server"` survives the snapshot build and
        // selects on the called server, leaving model traffic alone.
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
            "a-mcp",
            parse_attachment(
                r#"{ "guardrail_id": "g-1", "scope_type": "mcp_server", "scope_id": "mcp-A", "priority": 50 }"#,
            ),
        ));

        let index = build_index_from_snapshot(
            &guardrails,
            &attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
        assert_eq!(index.len(), 1, "the attachment must not be skipped");

        let matched = index.resolve(&RequestContext {
            passthrough_route_id: "",
            model_id: "",
            mcp_server_id: "mcp-A",
            api_key_id: "k",
            team_id: None,
        });
        assert_eq!(matched.len(), 1);

        let other_server = index.resolve(&RequestContext {
            passthrough_route_id: "",
            model_id: "",
            mcp_server_id: "mcp-B",
            api_key_id: "k",
            team_id: None,
        });
        assert!(other_server.is_empty());

        let llm = index.resolve(&RequestContext {
            passthrough_route_id: "",
            model_id: "m-A",
            mcp_server_id: "",
            api_key_id: "k",
            team_id: None,
        });
        assert!(llm.is_empty(), "model traffic carries no MCP server");
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

        let index = build_index_from_snapshot(
            &guardrails,
            &attachments,
            None,
            &GuardrailEmbedderSlot::none(),
        );
        let chain = index.resolve(&RequestContext {
            passthrough_route_id: "",
            model_id: "m-OTHER",
            mcp_server_id: "",
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
