//! End-to-end tests for the Redis semantic store against a live
//! vector-capable Redis (Redis 8+ / search module).
//!
//! Runs only when `CACHE_TEST_REDIS_VECTOR_URL` is set (CI points it at
//! the same `redis:8-alpine` service the exact-cache tests use). Each
//! test uses unique policy ids so runs never collide with leftovers of
//! earlier ones on a long-lived server.

#![cfg(feature = "redis")]

use std::time::Duration;

use aisix_cache::{RedisSemanticCache, SemanticCacheStore};
use aisix_core::{RedisConnConfig, RedisMode};
use aisix_gateway::{ChatMessage, ChatResponse, FinishReason, UsageStats};

fn vector_url() -> Option<String> {
    std::env::var("CACHE_TEST_REDIS_VECTOR_URL").ok()
}

fn single(url: &str) -> RedisConnConfig {
    RedisConnConfig {
        mode: RedisMode::Single,
        url: Some(url.to_string()),
        ..Default::default()
    }
}

/// Every test gets its own key/index namespace (via the env-namespace
/// hook) so concurrent tests — and concurrent CI runs against one
/// long-lived server — can never see each other's indexes. Without
/// this, the sweep test races sibling tests' just-created (still
/// empty) indexes away.
async fn connect_ns(url: &str, ns: &str) -> RedisSemanticCache {
    let store = RedisSemanticCache::connect(&single(url))
        .await
        .expect("connect")
        .with_env_namespace(ns);
    store.probe().await.expect("vector-capable redis required");
    store
}

fn resp(content: &str) -> ChatResponse {
    ChatResponse {
        id: "cmpl-sem-1".into(),
        model: "openai/gpt-4o".into(),
        message: ChatMessage::assistant(content),
        finish_reason: FinishReason::Stop,
        usage: UsageStats::new(3, 5),
    }
}

fn unique(name: &str) -> String {
    format!(
        "{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

const TTL: Duration = Duration::from_secs(60);

/// RediSearch indexes a write asynchronously relative to the command
/// reply: an `FT.SEARCH` issued immediately after the `HSET` can miss
/// the just-written document on a loaded runner (issue #922 — a
/// different test failed each CI run, always at a lookup directly
/// after a store). Bounded-retry a lookup the test REQUIRES to hit; a
/// real regression still fails, at the deadline instead of instantly.
/// Miss assertions stay direct — retrying a miss would prove nothing.
async fn lookup_hit(
    store: &RedisSemanticCache,
    policy: &str,
    generation: u32,
    scope_fp: &str,
    embedding: &[f32],
    threshold: f32,
    what: &str,
) -> aisix_cache::SemanticHit {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if let Some(hit) = store
            .lookup(policy, generation, scope_fp, embedding, threshold)
            .await
            .expect("lookup")
        {
            return hit;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{what}: no hit within the 2s index-visibility deadline",
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test]
async fn nearest_above_threshold_round_trips() {
    let Some(url) = vector_url() else { return };
    let policy = unique("p-roundtrip");
    let store = connect_ns(&url, &policy).await;

    store
        .store(
            &policy,
            0,
            "fp",
            "k1",
            vec![1.0, 0.0, 0.0, 0.0],
            resp("a"),
            TTL,
            100,
        )
        .await
        .unwrap();
    let hit = lookup_hit(
        &store,
        &policy,
        0,
        "fp",
        &[1.0, 0.0, 0.0, 0.0],
        0.9,
        "identical vector must hit",
    )
    .await;
    assert_eq!(hit.response.message.content_str(), "a");
    assert!(hit.similarity > 0.999, "got {}", hit.similarity);
    // Remaining lifetime is reported for the backfill cap.
    let remaining = hit
        .expires_at
        .saturating_duration_since(std::time::Instant::now());
    assert!(remaining <= TTL && remaining > TTL - Duration::from_secs(10));

    // Below threshold: orthogonal vector misses.
    let miss = store
        .lookup(&policy, 0, "fp", &[0.0, 1.0, 0.0, 0.0], 0.9)
        .await
        .unwrap();
    assert!(miss.is_none());
}

#[tokio::test]
async fn scope_fp_partitions_candidates() {
    let Some(url) = vector_url() else { return };
    let policy = unique("p-scope");
    let store = connect_ns(&url, &policy).await;

    store
        .store(
            &policy,
            0,
            "fp-a",
            "k1",
            vec![1.0, 0.0],
            resp("a"),
            TTL,
            100,
        )
        .await
        .unwrap();
    // Positive lookup FIRST: it doubles as the index-visibility
    // barrier, so the cross-partition miss below is a real verdict on
    // the scope filter — not a vacuous pass on a not-yet-indexed doc.
    lookup_hit(
        &store,
        &policy,
        0,
        "fp-a",
        &[1.0, 0.0],
        0.5,
        "same-partition entry must hit",
    )
    .await;
    let cross = store
        .lookup(&policy, 0, "fp-b", &[1.0, 0.0], 0.5)
        .await
        .unwrap();
    assert!(cross.is_none(), "entries must not cross scope fingerprints");
}

#[tokio::test]
async fn store_upserts_by_exact_key() {
    let Some(url) = vector_url() else { return };
    let policy = unique("p-upsert");
    let store = connect_ns(&url, &policy).await;

    store
        .store(
            &policy,
            0,
            "fp",
            "same-key",
            vec![1.0, 0.0],
            resp("v1"),
            TTL,
            100,
        )
        .await
        .unwrap();
    store
        .store(
            &policy,
            0,
            "fp",
            "same-key",
            vec![1.0, 0.0],
            resp("v2"),
            TTL,
            100,
        )
        .await
        .unwrap();
    let hit = lookup_hit(&store, &policy, 0, "fp", &[1.0, 0.0], 0.9, "upserted hit").await;
    assert_eq!(
        hit.response.message.content_str(),
        "v2",
        "refresh must replace the document in place",
    );
}

#[tokio::test]
async fn generation_rotation_orphans_old_entries() {
    let Some(url) = vector_url() else { return };
    let policy = unique("p-gen");
    let store = connect_ns(&url, &policy).await;

    store
        .store(
            &policy,
            0,
            "fp",
            "k1",
            vec![1.0, 0.0],
            resp("old"),
            TTL,
            100,
        )
        .await
        .unwrap();
    // Barrier: the generation-0 document is indexed and served before
    // the cross-generation miss below — otherwise a broken generation
    // partition could pass vacuously on a not-yet-indexed doc.
    lookup_hit(
        &store,
        &policy,
        0,
        "fp",
        &[1.0, 0.0],
        0.5,
        "generation-0 entry must hit before rotation",
    )
    .await;
    // Purged: lookups under the new generation must miss even though
    // the old document still physically exists.
    let miss = store
        .lookup(&policy, 1, "fp", &[1.0, 0.0], 0.5)
        .await
        .unwrap();
    assert!(miss.is_none());
    // Rewarm under the new generation.
    store
        .store(
            &policy,
            1,
            "fp",
            "k2",
            vec![0.0, 1.0],
            resp("new"),
            TTL,
            100,
        )
        .await
        .unwrap();
    let hit = lookup_hit(
        &store,
        &policy,
        1,
        "fp",
        &[0.0, 1.0],
        0.9,
        "new-generation hit",
    )
    .await;
    assert_eq!(hit.response.message.content_str(), "new");
    // The old generation's entries never resurface.
    let old = store
        .lookup(&policy, 1, "fp", &[1.0, 0.0], 0.5)
        .await
        .unwrap();
    assert!(old.is_none());
}

#[tokio::test]
async fn stale_generation_never_rotates_backwards() {
    let Some(url) = vector_url() else { return };
    let policy = unique("p-stale");
    let store = connect_ns(&url, &policy).await;

    // Live at generation 1 with a document.
    store
        .store(
            &policy,
            1,
            "fp",
            "k-new",
            vec![0.0, 1.0],
            resp("new"),
            TTL,
            100,
        )
        .await
        .unwrap();
    // A stale in-flight writer (pre-purge snapshot) must be dropped —
    // NOT recreate the generation-0 index and drop generation 1's
    // documents.
    store
        .store(
            &policy,
            0,
            "fp",
            "k-old",
            vec![1.0, 0.0],
            resp("stale"),
            TTL,
            100,
        )
        .await
        .unwrap();
    let hit = lookup_hit(
        &store,
        &policy,
        1,
        "fp",
        &[0.0, 1.0],
        0.9,
        "generation-1 document must survive the stale write",
    )
    .await;
    assert_eq!(hit.response.message.content_str(), "new");
    // Stale lookups miss rather than rotating.
    let stale = store
        .lookup(&policy, 0, "fp", &[1.0, 0.0], 0.5)
        .await
        .unwrap();
    assert!(stale.is_none());
}

#[tokio::test]
async fn sweep_drops_only_empty_indexes() {
    let Some(url) = vector_url() else { return };
    let live = unique("p-sweep-live");
    let dead = unique("p-sweep-dead");
    let store = connect_ns(&url, &live).await;

    store
        .store(&live, 0, "fp", "k1", vec![1.0, 0.0], resp("live"), TTL, 100)
        .await
        .unwrap();
    // An index whose documents all expired — the orphan shape a
    // deleted policy leaves behind.
    store
        .store(
            &dead,
            0,
            "fp",
            "k1",
            vec![1.0, 0.0],
            resp("dead"),
            Duration::from_millis(100),
            100,
        )
        .await
        .unwrap();
    // Redis expires documents lazily — `num_docs` catches up when the
    // active-expiry cycle reclaims the hash. Poll rather than assuming
    // one fixed sleep is enough.
    let mut dropped = 0;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        dropped = store.sweep_empty_indexes().await.unwrap();
        if dropped >= 1 {
            break;
        }
    }
    assert!(dropped >= 1, "the emptied index must be reclaimed");
    // The live policy still serves.
    let hit = lookup_hit(
        &store,
        &live,
        0,
        "fp",
        &[1.0, 0.0],
        0.9,
        "live index must survive the sweep",
    )
    .await;
    assert_eq!(hit.response.message.content_str(), "live");
    // Even if the sweep raced the live index away, the store self-heals
    // on the next touch (missing-index guard) — pin that too by
    // storing + looking up once more.
    store
        .store(
            &live,
            0,
            "fp",
            "k2",
            vec![0.0, 1.0],
            resp("live2"),
            TTL,
            100,
        )
        .await
        .unwrap();
    lookup_hit(
        &store,
        &live,
        0,
        "fp",
        &[0.0, 1.0],
        0.9,
        "self-healed index must serve the rewarmed entry",
    )
    .await;
}

#[tokio::test]
async fn dims_change_rotates_the_index() {
    let Some(url) = vector_url() else { return };
    let policy = unique("p-dims");
    let store = connect_ns(&url, &policy).await;

    store
        .store(
            &policy,
            0,
            "fp",
            "k1",
            vec![1.0, 0.0],
            resp("two-dim"),
            TTL,
            100,
        )
        .await
        .unwrap();
    // Same generation, different embedding dimensions (an in-place
    // embedding-model change): a separate index — no cross-space hits,
    // and storing works.
    let miss = store
        .lookup(&policy, 0, "fp", &[1.0, 0.0, 0.0], 0.0)
        .await
        .unwrap();
    assert!(miss.is_none());
    store
        .store(
            &policy,
            0,
            "fp",
            "k2",
            vec![1.0, 0.0, 0.0],
            resp("three-dim"),
            TTL,
            100,
        )
        .await
        .unwrap();
    let hit = lookup_hit(
        &store,
        &policy,
        0,
        "fp",
        &[1.0, 0.0, 0.0],
        0.9,
        "three-dim hit",
    )
    .await;
    assert_eq!(hit.response.message.content_str(), "three-dim");
}

#[tokio::test]
async fn entries_are_shared_across_store_instances() {
    let Some(url) = vector_url() else { return };
    let policy = unique("p-shared");
    let store_a = connect_ns(&url, &policy).await;
    let store_b = connect_ns(&url, &policy).await;

    store_a
        .store(
            &policy,
            0,
            "fp",
            "k1",
            vec![1.0, 0.0],
            resp("from-a"),
            TTL,
            100,
        )
        .await
        .unwrap();
    let hit = lookup_hit(
        &store_b,
        &policy,
        0,
        "fp",
        &[1.0, 0.0],
        0.9,
        "replica B must see replica A's entry",
    )
    .await;
    assert_eq!(hit.response.message.content_str(), "from-a");
}

#[tokio::test]
async fn ttl_expires_documents() {
    let Some(url) = vector_url() else { return };
    let policy = unique("p-ttl");
    let store = connect_ns(&url, &policy).await;

    store
        .store(
            &policy,
            0,
            "fp",
            "k1",
            vec![1.0, 0.0],
            resp("short"),
            Duration::from_millis(150),
            100,
        )
        .await
        .unwrap();
    // Redis expires documents lazily; poll until the reclaim lands
    // instead of trusting one fixed sleep.
    let mut miss = None;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        miss = store
            .lookup(&policy, 0, "fp", &[1.0, 0.0], 0.5)
            .await
            .unwrap();
        if miss.is_none() {
            break;
        }
    }
    assert!(miss.is_none(), "expired document must not match");
}

#[tokio::test]
async fn probe_fails_on_plain_redis() {
    // `CACHE_TEST_REDIS_PLAIN_URL` points at a server WITHOUT vector
    // search (CI: a redis:7 cluster node). Skip when absent — and skip
    // rather than fail when the target actually has vector support,
    // so local setups pointing it at a Redis 8 stay honest.
    let Some(url) = std::env::var("CACHE_TEST_REDIS_PLAIN_URL").ok() else {
        return;
    };
    let store = RedisSemanticCache::connect(&single(&url))
        .await
        .expect("connect");
    if store.probe().await.is_ok() {
        return; // vector-capable target: nothing to pin here
    }
    // Reaching here means the probe failed — which IS the assertion;
    // make it explicit for readers.
    assert!(store.probe().await.is_err());
}

#[tokio::test]
async fn lookup_recovers_after_external_index_loss() {
    // Redis restarting empty (or an operator dropping the index) must
    // not permanently disable the semantic layer: the memoized index
    // is evicted on the "index not found" error and the next touch
    // re-creates it.
    let Some(url) = vector_url() else { return };
    let policy = unique("p-recover");
    let store = connect_ns(&url, &policy).await;

    store
        .store(&policy, 0, "fp", "k1", vec![1.0, 0.0], resp("a"), TTL, 100)
        .await
        .unwrap();
    lookup_hit(
        &store,
        &policy,
        0,
        "fp",
        &[1.0, 0.0],
        0.9,
        "seeded entry must hit before the index loss",
    )
    .await;

    // Simulate the loss out-of-band through a raw connection.
    let raw = aisix_redis::connect(&single(&url)).await.expect("raw conn");
    let mut conn = raw.acquire().await.expect("acquire");
    redis::cmd("FT.DROPINDEX")
        .arg(format!("aisix:semcache:{policy}:idx:{policy}:0:2"))
        .arg("DD")
        .query_async::<()>(&mut conn)
        .await
        .expect("external drop");

    // First call fails visibly (fail-open at the gate) and evicts the
    // memo…
    assert!(store
        .lookup(&policy, 0, "fp", &[1.0, 0.0], 0.9)
        .await
        .is_err());
    // …the next one re-creates the (empty) index and misses cleanly…
    assert!(store
        .lookup(&policy, 0, "fp", &[1.0, 0.0], 0.9)
        .await
        .unwrap()
        .is_none());
    // …and the layer is fully functional again.
    store
        .store(&policy, 0, "fp", "k2", vec![0.0, 1.0], resp("b"), TTL, 100)
        .await
        .unwrap();
    lookup_hit(
        &store,
        &policy,
        0,
        "fp",
        &[0.0, 1.0],
        0.9,
        "re-created index must serve the rewarmed entry",
    )
    .await;
}
