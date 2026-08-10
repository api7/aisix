//! Semantic (embedding-similarity) storage — the L2 matching layer.
//!
//! Entries live *beside* the exact cache, not inside it: the [`Cache`]
//! trait is keyed by an opaque fingerprint string, while a semantic
//! lookup needs the candidate scope, the request embedding, and a
//! similarity threshold. The proxy consults the exact layer first and
//! only reaches for [`SemanticCacheStore`] on an exact miss.
//!
//! Scoping mirrors the exact key: candidates are pre-filtered by
//! `scope_fp` ([`crate::CacheKey::scope_fingerprint`] — model, sampling
//! params, extras, policy, generation, caller scope) so only the
//! *messages* are ever fuzzy. Purge works generationally: entries are
//! stored under the policy's `purge_generation`, and a bumped
//! generation makes every earlier entry unreachable.
//!
//! [`MemorySemanticCache`] is the in-process implementation (`backend:
//! memory`): per-policy entry lists scanned with brute-force cosine
//! under a map-shard read guard. The `max_entries` schema cap (10 000)
//! bounds the scan at ~15M f32 ops ≈ low-ms worst case and ~60 MB of
//! vectors per policy at 1536 dims; the 1000 default is microseconds.
//! Workloads beyond that belong on the shared (redis vector) backend.
//! Like the exact memory backend, it is per-instance — replicas do not
//! share entries.

use aisix_gateway::ChatResponse;
use async_trait::async_trait;
use dashmap::DashMap;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::cache::CacheError;

/// A semantic lookup match: the stored response plus how similar the
/// stored request was to the incoming one (cosine, `[0, 1]`-ish).
#[derive(Debug, Clone)]
pub struct SemanticHit {
    pub response: ChatResponse,
    pub similarity: f32,
    /// When the matched entry expires. Callers that copy the response
    /// into another layer (the exact-layer backfill) must cap that
    /// copy's TTL to this deadline, so a near-expiry entry cannot be
    /// granted a fresh full lifetime on every similar request.
    pub expires_at: Instant,
}

/// Storage seam for the semantic layer. Implementations must treat
/// `(policy_id, generation, scope_fp)` as the candidate partition:
/// lookups only ever compare against entries stored under the same
/// triple.
#[async_trait]
pub trait SemanticCacheStore: Send + Sync + 'static {
    /// Nearest stored entry in the partition with cosine similarity
    /// `>= threshold`, or `None`.
    async fn lookup(
        &self,
        policy_id: &str,
        generation: u32,
        scope_fp: &str,
        embedding: &[f32],
        threshold: f32,
    ) -> Result<Option<SemanticHit>, CacheError>;

    /// Store a response under the partition. `max_entries` caps the
    /// policy's entry count where the backend enforces one (the memory
    /// backend evicts oldest-first; shared backends may ignore it and
    /// bound growth by TTL).
    #[allow(clippy::too_many_arguments)]
    async fn store(
        &self,
        policy_id: &str,
        generation: u32,
        scope_fp: &str,
        embedding: Vec<f32>,
        response: ChatResponse,
        ttl: Duration,
        max_entries: u32,
    ) -> Result<(), CacheError>;
}

/// Cosine similarity of two equal-length vectors. Returns `0.0` for a
/// length mismatch or a zero-magnitude vector so a degenerate embedding
/// can never produce `NaN` (which would poison the max-tracking scan).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

struct SemanticEntry {
    scope_fp: String,
    embedding: Vec<f32>,
    response: ChatResponse,
    expires_at: Instant,
}

/// One policy's entries. `generation` is the `purge_generation` the
/// entries were stored under; a store or lookup with a different
/// generation treats the whole list as stale (lookups miss, the next
/// store resets it) — that lazy check is what makes purge O(1) without
/// any watch/callback plumbing.
struct PolicyEntries {
    generation: u32,
    entries: VecDeque<SemanticEntry>,
}

/// In-process semantic store for `backend: memory` policies.
#[derive(Default)]
pub struct MemorySemanticCache {
    policies: DashMap<String, PolicyEntries>,
}

impl MemorySemanticCache {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn entry_count(&self, policy_id: &str) -> usize {
        self.policies
            .get(policy_id)
            .map(|p| p.entries.len())
            .unwrap_or(0)
    }
}

#[async_trait]
impl SemanticCacheStore for MemorySemanticCache {
    async fn lookup(
        &self,
        policy_id: &str,
        generation: u32,
        scope_fp: &str,
        embedding: &[f32],
        threshold: f32,
    ) -> Result<Option<SemanticHit>, CacheError> {
        let Some(policy) = self.policies.get(policy_id) else {
            return Ok(None);
        };
        if policy.generation != generation {
            // Purged since these entries were written; read path stays
            // immutable, the next store resets the list.
            return Ok(None);
        }
        let now = Instant::now();
        let mut best: Option<SemanticHit> = None;
        for entry in &policy.entries {
            if entry.scope_fp != scope_fp || entry.expires_at <= now {
                continue;
            }
            let similarity = cosine_similarity(embedding, &entry.embedding);
            if similarity >= threshold
                && best.as_ref().map(|b| similarity > b.similarity) != Some(false)
            {
                best = Some(SemanticHit {
                    response: entry.response.clone(),
                    similarity,
                    expires_at: entry.expires_at,
                });
            }
        }
        Ok(best)
    }

    async fn store(
        &self,
        policy_id: &str,
        generation: u32,
        scope_fp: &str,
        embedding: Vec<f32>,
        response: ChatResponse,
        ttl: Duration,
        max_entries: u32,
    ) -> Result<(), CacheError> {
        let now = Instant::now();
        // Opportunistic reclamation of OTHER policies' storage: a policy
        // that was deleted (or just went idle) never runs its own store
        // path again, so front-drain its expired entries here and drop
        // buckets that emptied. Bounds orphaned memory at one TTL
        // (≤ 7 days) past the last write instead of process lifetime.
        // O(#policies) per store — policies are operator-configured and
        // few, and stores only happen on upstream misses.
        self.policies.retain(|id, p| {
            if id == policy_id {
                return true;
            }
            while p.entries.front().is_some_and(|e| e.expires_at <= now) {
                p.entries.pop_front();
            }
            !p.entries.is_empty()
        });
        let mut policy = self
            .policies
            .entry(policy_id.to_string())
            .or_insert_with(|| PolicyEntries {
                generation,
                entries: VecDeque::new(),
            });
        // A store from a request that read the policy BEFORE a purge
        // must not roll the bucket back and wipe post-purge entries:
        // drop the stale write instead. (Generations only ever move
        // forward — cp-api increments on purge.)
        if generation < policy.generation {
            return Ok(());
        }
        if generation > policy.generation {
            policy.generation = generation;
            policy.entries.clear();
        }
        // Drop expired entries from the front (insertion order ≈ expiry
        // order for a fixed TTL; mixed TTLs just expire lazily later).
        while policy.entries.front().is_some_and(|e| e.expires_at <= now) {
            policy.entries.pop_front();
        }
        policy.entries.push_back(SemanticEntry {
            scope_fp: scope_fp.to_string(),
            embedding,
            response,
            expires_at: now + ttl,
        });
        while policy.entries.len() > max_entries as usize {
            policy.entries.pop_front();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{ChatMessage, FinishReason, UsageStats};

    fn resp(text: &str) -> ChatResponse {
        ChatResponse {
            id: "cmpl-1".into(),
            model: "m".into(),
            message: ChatMessage::assistant(text),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(2, 3),
        }
    }

    const TTL: Duration = Duration::from_secs(60);

    #[tokio::test]
    async fn nearest_entry_above_threshold_wins() {
        let store = MemorySemanticCache::new();
        store
            .store("p", 0, "fp", vec![1.0, 0.0], resp("exact"), TTL, 100)
            .await
            .unwrap();
        store
            .store("p", 0, "fp", vec![0.9, 0.4359], resp("near"), TTL, 100)
            .await
            .unwrap();
        // Query aligned with the second entry: both clear 0.8, nearest wins.
        let hit = store
            .lookup("p", 0, "fp", &[0.9, 0.4359], 0.8)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(hit.response.message.content_str(), "near");
        assert!(hit.similarity > 0.99);
    }

    #[tokio::test]
    async fn below_threshold_misses() {
        let store = MemorySemanticCache::new();
        store
            .store("p", 0, "fp", vec![1.0, 0.0], resp("a"), TTL, 100)
            .await
            .unwrap();
        // cos = 0.7071 < 0.9.
        let got = store.lookup("p", 0, "fp", &[1.0, 1.0], 0.9).await.unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn scope_fp_partitions_candidates() {
        let store = MemorySemanticCache::new();
        store
            .store("p", 0, "fp-a", vec![1.0, 0.0], resp("a"), TTL, 100)
            .await
            .unwrap();
        let got = store
            .lookup("p", 0, "fp-b", &[1.0, 0.0], 0.5)
            .await
            .unwrap();
        assert!(got.is_none(), "entries must not cross scope fingerprints");
    }

    #[tokio::test]
    async fn policies_partition_candidates() {
        let store = MemorySemanticCache::new();
        store
            .store("p1", 0, "fp", vec![1.0, 0.0], resp("a"), TTL, 100)
            .await
            .unwrap();
        let got = store.lookup("p2", 0, "fp", &[1.0, 0.0], 0.5).await.unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn generation_bump_purges_lookups_and_resets_on_store() {
        let store = MemorySemanticCache::new();
        store
            .store("p", 0, "fp", vec![1.0, 0.0], resp("old"), TTL, 100)
            .await
            .unwrap();
        // Purge: lookups under the new generation miss.
        let got = store.lookup("p", 1, "fp", &[1.0, 0.0], 0.5).await.unwrap();
        assert!(got.is_none());
        // First store under the new generation resets the list.
        store
            .store("p", 1, "fp", vec![0.0, 1.0], resp("new"), TTL, 100)
            .await
            .unwrap();
        assert_eq!(store.entry_count("p"), 1);
        // Old-generation entry is gone even for old-generation lookups.
        let got = store.lookup("p", 0, "fp", &[1.0, 0.0], 0.5).await.unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn stale_generation_store_is_dropped_not_rolled_back() {
        let store = MemorySemanticCache::new();
        // Purged bucket already re-warmed at generation 1.
        store
            .store("p", 1, "fp", vec![0.0, 1.0], resp("new"), TTL, 100)
            .await
            .unwrap();
        // A request that read the pre-purge policy snapshot finishes its
        // upstream call late and stores under generation 0: dropped.
        store
            .store("p", 0, "fp", vec![1.0, 0.0], resp("stale"), TTL, 100)
            .await
            .unwrap();
        assert_eq!(store.entry_count("p"), 1);
        let hit = store
            .lookup("p", 1, "fp", &[0.0, 1.0], 0.9)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(hit.response.message.content_str(), "new");
        assert!(store
            .lookup("p", 0, "fp", &[1.0, 0.0], 0.5)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn store_reclaims_other_policies_expired_buckets() {
        let store = MemorySemanticCache::new();
        store
            .store(
                "idle",
                0,
                "fp",
                vec![1.0, 0.0],
                resp("orphan"),
                Duration::from_millis(20),
                100,
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        // A store on ANY other policy sweeps the fully-expired bucket.
        store
            .store("busy", 0, "fp", vec![0.0, 1.0], resp("live"), TTL, 100)
            .await
            .unwrap();
        assert_eq!(store.entry_count("idle"), 0);
        assert_eq!(store.entry_count("busy"), 1);
    }

    #[tokio::test]
    async fn lookup_reports_entry_expiry_for_backfill_capping() {
        let store = MemorySemanticCache::new();
        let ttl = Duration::from_secs(60);
        store
            .store("p", 0, "fp", vec![1.0, 0.0], resp("a"), ttl, 100)
            .await
            .unwrap();
        let hit = store
            .lookup("p", 0, "fp", &[1.0, 0.0], 0.9)
            .await
            .unwrap()
            .unwrap();
        // `expires_at` was set at store time, so measured from any later
        // instant the remaining lifetime is within (ttl - ε, ttl].
        let remaining = hit.expires_at.saturating_duration_since(Instant::now());
        assert!(remaining <= ttl);
        assert!(remaining > ttl - Duration::from_secs(5));
    }

    #[tokio::test]
    async fn expired_entries_never_match() {
        let store = MemorySemanticCache::new();
        store
            .store(
                "p",
                0,
                "fp",
                vec![1.0, 0.0],
                resp("a"),
                Duration::from_millis(20),
                100,
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let got = store.lookup("p", 0, "fp", &[1.0, 0.0], 0.5).await.unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn max_entries_evicts_oldest_first() {
        let store = MemorySemanticCache::new();
        store
            .store("p", 0, "fp", vec![1.0, 0.0], resp("first"), TTL, 2)
            .await
            .unwrap();
        store
            .store("p", 0, "fp", vec![0.0, 1.0], resp("second"), TTL, 2)
            .await
            .unwrap();
        store
            .store("p", 0, "fp", vec![-1.0, 0.0], resp("third"), TTL, 2)
            .await
            .unwrap();
        assert_eq!(store.entry_count("p"), 2);
        // "first" ([1,0]) was evicted; an aligned query now misses.
        let got = store.lookup("p", 0, "fp", &[1.0, 0.0], 0.9).await.unwrap();
        assert!(got.is_none());
        // "second" survives.
        let hit = store
            .lookup("p", 0, "fp", &[0.0, 1.0], 0.9)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(hit.response.message.content_str(), "second");
    }

    #[test]
    fn cosine_degenerate_inputs_are_zero_not_nan() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0]), 0.0);
        assert!((cosine_similarity(&[1.0, 1.0], &[5.0, 5.0]) - 1.0).abs() < 1e-6);
        assert!(!cosine_similarity(&[0.0], &[0.0]).is_nan());
    }
}
