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
    AisixSnapshot, Guardrail as DomainGuardrail, GuardrailHookPoint, GuardrailKind, KeywordPattern,
};
use aisix_core::snapshot::ResourceTable;
use aisix_core::SnapshotHandle;
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;

use crate::keyword::{KeywordBlocklist, KeywordRule};
use crate::{Guardrail, GuardrailChain, GuardrailVerdict};

/// Build a chain from a snapshot's `guardrails` table.
///
/// Iteration order matches the table's deterministic id-sort. Each
/// row produces at most one runtime `dyn Guardrail`. Failures
/// (invalid regex, etc.) are logged and the row is skipped — same
/// contract the loader uses for malformed etcd rows.
pub fn build_chain_from_snapshot(table: &ResourceTable<DomainGuardrail>) -> GuardrailChain {
    let mut chain: Vec<Arc<dyn Guardrail>> = Vec::new();

    let entries = table.entries();
    for entry in entries.iter() {
        let row = &entry.value;
        if !row.enabled {
            continue;
        }
        match build_one(row) {
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

fn build_one(row: &DomainGuardrail) -> Result<Option<Arc<dyn Guardrail>>, BuildError> {
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
            // credentials.
            let g = crate::bedrock::BedrockGuardrail::new(
                row.name.clone(),
                cfg,
                row.hook_point,
                row.fail_open,
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
pub struct LiveGuardrailChain {
    snapshot: SnapshotHandle<AisixSnapshot>,
    cache: Mutex<Cache>,
}

struct Cache {
    /// Pointer-identity of the snapshot the chain was built from,
    /// stored as `usize` for `Send + Sync` (this crate forbids
    /// `unsafe`). `Arc::as_ptr` is stable for the snapshot's
    /// lifetime; comparing the integer to a fresh load tells us
    /// cheaply whether the supervisor stored a new snapshot since
    /// we last rebuilt. We never deref the address.
    last_snapshot_addr: usize,
    chain: Arc<GuardrailChain>,
}

impl LiveGuardrailChain {
    pub fn new(snapshot: SnapshotHandle<AisixSnapshot>) -> Arc<Self> {
        // Eager-build at construct time so the very first chat
        // doesn't pay the rebuild cost. The pointer recorded here
        // is the snapshot at construct time — a subsequent store
        // from the supervisor flips the cache miss bit on next
        // check.
        let snap = snapshot.load();
        let chain = Arc::new(build_chain_from_snapshot(&snap.guardrails));
        let last_snapshot_addr = Arc::as_ptr(&snap) as usize;
        Arc::new(Self {
            snapshot,
            cache: Mutex::new(Cache {
                last_snapshot_addr,
                chain,
            }),
        })
    }

    fn current(&self) -> Arc<GuardrailChain> {
        let snap = self.snapshot.load();
        let cur_ptr = Arc::as_ptr(&snap) as usize;
        let mut cache = self
            .cache
            .lock()
            .expect("LiveGuardrailChain mutex poisoned");
        if cache.last_snapshot_addr != cur_ptr {
            cache.chain = Arc::new(build_chain_from_snapshot(&snap.guardrails));
            cache.last_snapshot_addr = cur_ptr;
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
        let chain = build_chain_from_snapshot(&table);
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
        let chain = build_chain_from_snapshot(&table);
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
        let chain = build_chain_from_snapshot(&table);
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

        let chain = build_chain_from_snapshot(&table);
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
        let chain = build_chain_from_snapshot(&table);
        // Both rows compose. We don't probe the bedrock arm — its
        // own tests cover the dispatch path; this one only pins the
        // chain composition contract.
        assert_eq!(chain.len(), 2);
    }

    #[tokio::test]
    async fn live_chain_rebuilds_on_snapshot_swap() {
        let initial = AisixSnapshot::new();
        let handle = SnapshotHandle::new(initial);
        let live = LiveGuardrailChain::new(handle.clone());

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
        let chain = build_chain_from_snapshot(&table);
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
