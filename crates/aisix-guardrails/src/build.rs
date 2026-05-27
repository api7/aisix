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
    AisixSnapshot, Guardrail as DomainGuardrail, GuardrailAttachment, GuardrailHookPoint,
    GuardrailKind, GuardrailScopeType, KeywordPattern,
};
use aisix_core::snapshot::ResourceTable;
use aisix_core::SnapshotHandle;
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;

use crate::index::{GuardrailIndex, RequestContext, ScopeKind};
use crate::keyword::{KeywordBlocklist, KeywordRule};
use crate::{Guardrail, GuardrailChain, GuardrailVerdict};

/// Build a chain from a snapshot's `guardrails` table.
///
/// Iteration order matches the table's deterministic id-sort. Each
/// row produces at most one runtime `dyn Guardrail`. Failures
/// (invalid regex, etc.) are logged and the row is skipped — same
/// contract the loader uses for malformed etcd rows.
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
    let mut chain: Vec<Arc<dyn Guardrail>> = Vec::new();

    let entries = table.entries();
    for entry in entries.iter() {
        let row = &entry.value;
        if !row.enabled {
            continue;
        }
        match build_one(row, bedrock_endpoint_url) {
            Ok(Some(g)) => chain.push(g),
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

    GuardrailChain::new(chain)
}

fn build_one(
    row: &DomainGuardrail,
    bedrock_endpoint_url: Option<&str>,
) -> Result<Option<Arc<dyn Guardrail>>, BuildError> {
    // enforcement_mode is not yet implemented: warn so operators don't silently
    // assume "monitor" means pass-through. See doc comment on the field.
    if row.enforcement_mode != "block" {
        tracing::warn!(
            guardrail_name = %row.name,
            enforcement_mode = %row.enforcement_mode,
            "enforcement_mode is not yet implemented; DP will block regardless of this setting",
        );
    }
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
    }
}

#[derive(Debug, thiserror::Error)]
enum BuildError {
    #[error("invalid regex {pattern:?}: {source}")]
    InvalidRegex {
        pattern: String,
        source: regex::Error,
    },
    /// Reserved for guardrail kinds whose runtime dispatch is
    /// compiled out (e.g. a slim build that excluded the
    /// `--features bedrock` AWS SDK dependency). The chain treats
    /// these rows as disabled and the supervisor's warn log
    /// surfaces the kind name so a misconfigured environment is
    /// visible.
    #[cfg(not(feature = "bedrock"))]
    #[error("guardrail kind {0:?} not compiled into this build; treating row as disabled")]
    FeatureDisabled(&'static str),
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

    for attachment_arc in attachments.entries() {
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
            scope_kind,
            attachment.scope_id.clone(),
            attachment.priority,
            runtime_guardrail,
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
    for guardrail_arc in guardrails.entries() {
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
                    ScopeKind::Env,
                    None,
                    0,
                    g,
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
}
