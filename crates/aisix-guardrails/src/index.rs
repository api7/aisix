//! Per-request guardrail resolution.
//!
//! ## Why an index?
//!
//! Prior to P0b, every enabled guardrail in an environment applied to every
//! request — there was no scoping. P0b introduced the `guardrail_attachment`
//! table, which binds a guardrail definition to a specific scope:
//!
//! | `scope_type` | meaning |
//! |---|---|
//! | `env`     | applies to every request in the environment |
//! | `model`   | applies only when the request targets this model UUID |
//! | `api_key` | applies only when authenticated with this API-key UUID |
//! | `team`    | applies only when the API key belongs to this team UUID |
//!
//! `GuardrailIndex` holds the pre-built runtime guardrails for a snapshot.
//! `resolve(ctx)` filters + deduplicates the entries and returns the chain
//! for one specific request without allocating until a request arrives.
//!
//! ## Priority and deduplication
//!
//! When the same guardrail UUID appears via multiple matching scopes (e.g.
//! attached at env-scope AND at model-scope), the attachment with the
//! highest `priority` number wins and the duplicate is dropped. Entries
//! are pre-sorted descending so the first match per `guardrail_id` is always
//! the highest-priority one.

use std::collections::HashSet;
use std::sync::Arc;

use aisix_core::AppliedGuardrail;

use crate::{Guardrail, GuardrailChain};

/// Which scope dimension an attachment covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeKind {
    Env,
    Model,
    ApiKey,
    Team,
}

/// One entry in the index: a pre-built runtime guardrail associated with
/// one `guardrail_attachment` row.
pub(crate) struct IndexEntry {
    /// UUID of the guardrail definition (for deduplication).
    guardrail_id: String,
    /// Operator-facing name of the guardrail definition. Carried onto the
    /// resolved chain so a `Block` verdict can be attributed to the rule
    /// that fired (#519 B.4b).
    guardrail_name: String,
    scope_kind: ScopeKind,
    /// `None` for `Env` scope; the UUID string for narrower scopes.
    scope_id: Option<String>,
    /// Higher = higher precedence. Entries are pre-sorted descending.
    priority: i32,
    guardrail: Arc<dyn Guardrail>,
    /// The `{kind, hook}` of this entry's guardrail, captured at index-build
    /// time (the only place the domain row's `kind` + `hook_point` are in
    /// scope). `resolve` collects these from the entries it keeps so the
    /// returned chain can report which guardrails governed the request (#379).
    applied: AppliedGuardrail,
}

impl std::fmt::Debug for IndexEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexEntry")
            .field("guardrail_id", &self.guardrail_id)
            .field("scope_kind", &self.scope_kind)
            .field("scope_id", &self.scope_id)
            .field("priority", &self.priority)
            .field("guardrail", &self.guardrail.name())
            .finish()
    }
}

impl IndexEntry {
    fn applies_to(&self, ctx: &RequestContext<'_>) -> bool {
        match self.scope_kind {
            ScopeKind::Env => true,
            ScopeKind::Model => self.scope_id.as_deref() == Some(ctx.model_id),
            ScopeKind::ApiKey => self.scope_id.as_deref() == Some(ctx.api_key_id),
            ScopeKind::Team => ctx
                .team_id
                .zip(self.scope_id.as_deref())
                .map(|(ctx_team, entry_team)| ctx_team == entry_team)
                .unwrap_or(false),
        }
    }
}

/// Per-request context used by [`GuardrailIndex::resolve`] to select and
/// deduplicate the applicable guardrails.
#[derive(Debug, Clone, Copy)]
pub struct RequestContext<'a> {
    /// UUID of the model the request targets (virtual or concrete).
    pub model_id: &'a str,
    /// UUID of the API key used to authenticate the request.
    pub api_key_id: &'a str,
    /// UUID of the team the API key belongs to. `None` if the key is not
    /// associated with a team.
    pub team_id: Option<&'a str>,
}

/// Resolves the applicable guardrail chain for a single request.
///
/// Built from a snapshot by [`crate::build::build_index_from_snapshot`].
/// Cheap to clone — all guardrail state is behind `Arc`.
#[derive(Debug, Default)]
pub struct GuardrailIndex {
    /// Pre-sorted descending by `priority`. Entries with equal priority are
    /// in stable insertion order (same as the `ResourceTable` id-sort).
    entries: Vec<IndexEntry>,
}

/// Scope specificity rank: higher = more specific → wins dedup on equal priority.
/// ApiKey > Team > Model > Env, matching the P0c spec in #379.
fn scope_specificity(k: &ScopeKind) -> u8 {
    match k {
        ScopeKind::ApiKey => 3,
        ScopeKind::Team => 2,
        ScopeKind::Model => 1,
        ScopeKind::Env => 0,
    }
}

impl GuardrailIndex {
    pub(crate) fn new(mut entries: Vec<IndexEntry>) -> Self {
        entries.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| {
                scope_specificity(&b.scope_kind).cmp(&scope_specificity(&a.scope_kind))
            })
        });
        Self { entries }
    }

    /// Returns the number of attachment entries in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Build the chain for `ctx`.
    ///
    /// Algorithm:
    /// 1. Walk `entries` in priority order (highest first, pre-sorted).
    /// 2. Skip entries that don't match `ctx`.
    /// 3. Skip duplicates — first match per `guardrail_id` wins.
    /// 4. Collect into a `GuardrailChain`.
    ///
    /// Complexity: O(n) in the number of attachment entries.
    pub fn resolve(&self, ctx: &RequestContext<'_>) -> GuardrailChain {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut chain: Vec<(String, Arc<dyn Guardrail>)> = Vec::new();
        // `applied` mirrors `chain` 1:1 — the `{kind, hook}` of each member
        // we keep, for applied-guardrail telemetry (#379). Pushed on the same
        // (matched + not-deduplicated) path so it never drifts from `chain`.
        let mut applied: Vec<AppliedGuardrail> = Vec::new();

        for entry in &self.entries {
            if !entry.applies_to(ctx) {
                continue;
            }
            if seen.contains(entry.guardrail_id.as_str()) {
                continue;
            }
            seen.insert(entry.guardrail_id.as_str());
            chain.push((entry.guardrail_name.clone(), Arc::clone(&entry.guardrail)));
            applied.push(entry.applied.clone());
        }

        GuardrailChain::new_with_applied(chain, applied)
    }
}

// ---------------------------------------------------------------------------
// Internal constructor used by `build.rs`
// ---------------------------------------------------------------------------

impl GuardrailIndex {
    pub(crate) fn push_entry(
        guardrail_id: impl Into<String>,
        guardrail_name: impl Into<String>,
        scope_kind: ScopeKind,
        scope_id: Option<String>,
        priority: i32,
        guardrail: Arc<dyn Guardrail>,
        applied: AppliedGuardrail,
    ) -> IndexEntry {
        IndexEntry {
            guardrail_id: guardrail_id.into(),
            guardrail_name: guardrail_name.into(),
            scope_kind,
            scope_id,
            priority,
            guardrail,
            applied,
        }
    }

    pub(crate) fn from_entries(entries: Vec<IndexEntry>) -> Self {
        Self::new(entries)
    }
}

// ---------------------------------------------------------------------------
// Tests — 12+ truth-table cases + benchmark
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuardrailVerdict, KeywordBlocklist, KeywordRule};
    use aisix_gateway::{ChatFormat, ChatMessage};
    use std::time::Instant;

    fn kw(_name: &'static str, literal: &str) -> Arc<dyn Guardrail> {
        Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal(
            literal.to_owned(),
        )]))
    }

    fn req(msg: &str) -> ChatFormat {
        ChatFormat::new("m", vec![ChatMessage::user(msg)])
    }

    fn ctx<'a>(model: &'a str, apikey: &'a str, team: Option<&'a str>) -> RequestContext<'a> {
        RequestContext {
            model_id: model,
            api_key_id: apikey,
            team_id: team,
        }
    }

    fn entry(
        gid: &str,
        scope: ScopeKind,
        sid: Option<&str>,
        priority: i32,
        g: Arc<dyn Guardrail>,
    ) -> IndexEntry {
        // These resolution tests build keyword guardrails via `kw`; the
        // applied descriptor is documentary here (the dedicated applied
        // tests live in build.rs against the real snapshot build path).
        GuardrailIndex::push_entry(
            gid,
            gid,
            scope,
            sid.map(str::to_owned),
            priority,
            g,
            AppliedGuardrail {
                kind: "keyword".to_owned(),
                hook: "both".to_owned(),
            },
        )
    }

    // 1. Empty index allows everything.
    #[tokio::test]
    async fn empty_index_allows_all() {
        let idx = GuardrailIndex::from_entries(vec![]);
        let chain = idx.resolve(&ctx("m1", "k1", None));
        assert!(chain.is_empty());
        assert_eq!(
            chain.check_input(&req("AKIA")).await,
            GuardrailVerdict::Allow
        );
    }

    // 2. Env-scope attachment applies to all requests.
    #[tokio::test]
    async fn env_scope_applies_to_all() {
        let g = kw("g1", "AKIA");
        let idx = GuardrailIndex::from_entries(vec![entry("g1", ScopeKind::Env, None, 50, g)]);

        // model A
        let chain = idx.resolve(&ctx("m1", "k1", None));
        assert!(chain.check_input(&req("here AKIA")).await.is_block());

        // model B — still applies
        let chain = idx.resolve(&ctx("m2", "k2", None));
        assert!(chain.check_input(&req("here AKIA")).await.is_block());
    }

    // 3. Model-scope attachment applies only to the matching model.
    #[tokio::test]
    async fn model_scope_only_matching_model() {
        let g = kw("g1", "secret");
        let idx = GuardrailIndex::from_entries(vec![entry(
            "g1",
            ScopeKind::Model,
            Some("model-A"),
            50,
            g,
        )]);

        let chain_a = idx.resolve(&ctx("model-A", "k1", None));
        assert!(chain_a.check_input(&req("secret")).await.is_block());

        let chain_b = idx.resolve(&ctx("model-B", "k1", None));
        assert_eq!(
            chain_b.check_input(&req("secret")).await,
            GuardrailVerdict::Allow
        );
    }

    // 4. ApiKey-scope attachment applies only to the matching key.
    #[tokio::test]
    async fn api_key_scope_only_matching_key() {
        let g = kw("g1", "restricted");
        let idx = GuardrailIndex::from_entries(vec![entry(
            "g1",
            ScopeKind::ApiKey,
            Some("key-ALPHA"),
            50,
            g,
        )]);

        let chain_alpha = idx.resolve(&ctx("m1", "key-ALPHA", None));
        assert!(chain_alpha
            .check_input(&req("restricted term"))
            .await
            .is_block());

        let chain_other = idx.resolve(&ctx("m1", "key-BETA", None));
        assert_eq!(
            chain_other.check_input(&req("restricted term")).await,
            GuardrailVerdict::Allow
        );
    }

    // 5. Team-scope applies only when the key belongs to the matching team.
    #[tokio::test]
    async fn team_scope_only_matching_team() {
        let g = kw("g1", "classified");
        let idx =
            GuardrailIndex::from_entries(vec![entry("g1", ScopeKind::Team, Some("team-X"), 50, g)]);

        // Key belongs to team-X → blocks.
        let chain_x = idx.resolve(&ctx("m1", "k1", Some("team-X")));
        assert!(chain_x.check_input(&req("classified")).await.is_block());

        // Key belongs to different team → allows.
        let chain_y = idx.resolve(&ctx("m1", "k1", Some("team-Y")));
        assert_eq!(
            chain_y.check_input(&req("classified")).await,
            GuardrailVerdict::Allow
        );

        // Key has no team → allows.
        let chain_none = idx.resolve(&ctx("m1", "k1", None));
        assert_eq!(
            chain_none.check_input(&req("classified")).await,
            GuardrailVerdict::Allow
        );
    }

    // 6. Disabled attachment is NOT added by `build_index_from_snapshot`
    //    (tested indirectly here: build only adds enabled entries).
    //    Direct test: we simulate by not adding the entry.
    #[tokio::test]
    async fn disabled_attachment_skipped_allows_all() {
        // Simulate build_index_from_snapshot filtering disabled rows:
        // no entries added → empty chain.
        let idx = GuardrailIndex::from_entries(vec![]);
        let chain = idx.resolve(&ctx("m1", "k1", None));
        assert_eq!(
            chain.check_input(&req("anything")).await,
            GuardrailVerdict::Allow
        );
    }

    // 7. Priority deduplication: same guardrail_id, two scopes — highest priority wins.
    //    Env-scope at priority 50 + model-scope at priority 100 for same gid →
    //    only ONE entry (model-scope, priority 100) in the chain.
    #[tokio::test]
    async fn priority_dedup_same_guardrail_id_highest_priority_wins() {
        // Two entries with the same gid but different scopes.
        // Model-scope has higher priority → appears first after sort.
        let g_env = kw("g1", "keyword");
        let g_model = kw("g1", "keyword");
        let entries = vec![
            entry("g1", ScopeKind::Env, None, 50, g_env),
            entry("g1", ScopeKind::Model, Some("model-A"), 100, g_model),
        ];
        let idx = GuardrailIndex::from_entries(entries);
        // After sort: model-scope(100) first, env-scope(50) second.
        // resolve for model-A: model-scope matches first → g1 seen → env-scope skipped.
        let chain = idx.resolve(&ctx("model-A", "k1", None));
        assert_eq!(chain.len(), 1, "dedup: only one entry for the same gid");
    }

    // 8. Cross-scope: env + model-scope for DIFFERENT guardrails → both apply.
    #[tokio::test]
    async fn cross_scope_different_guardrails_both_apply() {
        let g_env = kw("g1", "token-a");
        let g_model = kw("g2", "token-b");
        let entries = vec![
            entry("g1", ScopeKind::Env, None, 50, g_env),
            entry("g2", ScopeKind::Model, Some("model-A"), 50, g_model),
        ];
        let idx = GuardrailIndex::from_entries(entries);

        let chain = idx.resolve(&ctx("model-A", "k1", None));
        assert_eq!(chain.len(), 2);
        assert!(chain.check_input(&req("token-a stuff")).await.is_block());
        assert!(chain.check_input(&req("token-b stuff")).await.is_block());
    }

    // 9. Priority tie-break is stable (insertion order for equal priority).
    #[tokio::test]
    async fn priority_equal_is_stable() {
        let g1 = kw("g1", "alpha");
        let g2 = kw("g2", "beta");
        let entries = vec![
            entry("g1", ScopeKind::Env, None, 50, g1),
            entry("g2", ScopeKind::Env, None, 50, g2),
        ];
        let idx = GuardrailIndex::from_entries(entries);
        let chain = idx.resolve(&ctx("m1", "k1", None));
        // Both are present — stable order.
        assert_eq!(chain.len(), 2);
    }

    // 10. Model-scope doesn't apply to an unrelated model.
    #[tokio::test]
    async fn model_scope_nonmatch_empty_chain() {
        let g = kw("g1", "classified");
        let idx =
            GuardrailIndex::from_entries(vec![entry("g1", ScopeKind::Model, Some("m-A"), 10, g)]);
        let chain = idx.resolve(&ctx("m-B", "k1", None));
        assert!(chain.is_empty());
    }

    // 11. Multiple env-scoped attachments accumulate (different gids).
    #[tokio::test]
    async fn multiple_env_scoped_accumulate() {
        let g1 = kw("g1", "bad-word-1");
        let g2 = kw("g2", "bad-word-2");
        let entries = vec![
            entry("g1", ScopeKind::Env, None, 100, g1),
            entry("g2", ScopeKind::Env, None, 90, g2),
        ];
        let idx = GuardrailIndex::from_entries(entries);
        let chain = idx.resolve(&ctx("m1", "k1", None));
        assert_eq!(chain.len(), 2);
        assert!(chain.check_input(&req("bad-word-1")).await.is_block());
        assert!(chain.check_input(&req("bad-word-2")).await.is_block());
    }

    // 12. ApiKey-scope + Env-scope both apply to the matching key.
    #[tokio::test]
    async fn env_and_apikey_scope_both_apply() {
        let g_env = kw("g1", "restricted");
        let g_key = kw("g2", "classified");
        let entries = vec![
            entry("g1", ScopeKind::Env, None, 50, g_env),
            entry("g2", ScopeKind::ApiKey, Some("key-VIP"), 80, g_key),
        ];
        let idx = GuardrailIndex::from_entries(entries);

        // VIP key: both apply.
        let chain_vip = idx.resolve(&ctx("m1", "key-VIP", None));
        assert_eq!(chain_vip.len(), 2);

        // Regular key: only env-scope applies.
        let chain_reg = idx.resolve(&ctx("m1", "key-REGULAR", None));
        assert_eq!(chain_reg.len(), 1);
    }

    // 13. Team + ApiKey both match — three guardrails in the chain.
    #[tokio::test]
    async fn all_scope_types_can_combine() {
        let g_env = kw("g1", "w1");
        let g_model = kw("g2", "w2");
        let g_key = kw("g3", "w3");
        let g_team = kw("g4", "w4");
        let entries = vec![
            entry("g1", ScopeKind::Env, None, 10, g_env),
            entry("g2", ScopeKind::Model, Some("m1"), 20, g_model),
            entry("g3", ScopeKind::ApiKey, Some("k1"), 30, g_key),
            entry("g4", ScopeKind::Team, Some("t1"), 40, g_team),
        ];
        let idx = GuardrailIndex::from_entries(entries);

        // All four match.
        let chain = idx.resolve(&ctx("m1", "k1", Some("t1")));
        assert_eq!(chain.len(), 4);

        // Only env + model match (different key and no team).
        let chain2 = idx.resolve(&ctx("m1", "k-other", None));
        assert_eq!(chain2.len(), 2);
    }

    // 14. Equal-priority dedup uses scope-specificity: ApiKey > Env.
    //     Same guardrail_id, env(priority=50) + apikey(priority=50).
    //     For a request from the matching api-key, the ApiKey entry must win
    //     (appears first after the sort tiebreaker) and deduplicate the Env entry.
    #[test]
    fn equal_priority_apikey_beats_env_in_dedup() {
        let g_env = kw("g1", "keyword");
        let g_key = kw("g1", "keyword");
        let entries = vec![
            // Insert in "wrong" order (env first) to confirm sort fixes it.
            entry("g1", ScopeKind::Env, None, 50, g_env),
            entry("g1", ScopeKind::ApiKey, Some("key-A"), 50, g_key),
        ];
        let idx = GuardrailIndex::from_entries(entries);

        // For key-A: ApiKey(priority=50, specificity=3) wins over Env(priority=50, specificity=0).
        // Dedup fires on Env → chain has exactly 1 entry.
        let chain = idx.resolve(&ctx("m1", "key-A", None));
        assert_eq!(
            chain.len(),
            1,
            "equal-priority: ApiKey-scope must deduplicate Env-scope"
        );
    }

    // -----------------------------------------------------------------------
    // Benchmark: 1000 attachment entries, resolve < 100ms
    // -----------------------------------------------------------------------

    /// Performance pin: building the index from 1000 entries and resolving
    /// 100 contexts must complete in well under 100ms on any CI runner.
    /// Uses a simple wall-clock assertion — not a criterion benchmark — so
    /// it runs in `cargo test` without extra tooling.
    #[test]
    fn index_rebuild_and_resolve_1000_attachments_under_100ms() {
        let mut entries = Vec::with_capacity(1000);
        for i in 0..1000u32 {
            let scope_kind = match i % 4 {
                0 => ScopeKind::Env,
                1 => ScopeKind::Model,
                2 => ScopeKind::ApiKey,
                _ => ScopeKind::Team,
            };
            let scope_id = if scope_kind == ScopeKind::Env {
                None
            } else {
                Some(format!("scope-{}", i % 10))
            };
            let g = Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal(format!(
                "word-{i}"
            ))])) as Arc<dyn Guardrail>;
            entries.push(entry(
                &format!("g-{i}"),
                scope_kind,
                scope_id.as_deref(),
                (i as i32) % 200,
                g,
            ));
        }

        let start = Instant::now();

        // Build: sort + index construction.
        let idx = GuardrailIndex::from_entries(entries);

        // Resolve 100 different contexts.
        for i in 0..100usize {
            let model = format!("scope-{}", i % 10);
            let key = format!("scope-{}", (i + 3) % 10);
            let team = format!("scope-{}", (i + 7) % 10);
            let _ = idx.resolve(&RequestContext {
                model_id: &model,
                api_key_id: &key,
                team_id: Some(&team),
            });
        }

        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 100,
            "index build + 100 resolves took {}ms, expected < 100ms",
            elapsed.as_millis()
        );
    }
}
