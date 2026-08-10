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

async fn connect(url: &str) -> RedisSemanticCache {
    let store = RedisSemanticCache::connect(&single(url))
        .await
        .expect("connect");
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

#[tokio::test]
async fn nearest_above_threshold_round_trips() {
    let Some(url) = vector_url() else { return };
    let store = connect(&url).await;
    let policy = unique("p-roundtrip");

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
    let hit = store
        .lookup(&policy, 0, "fp", &[1.0, 0.0, 0.0, 0.0], 0.9)
        .await
        .unwrap()
        .expect("identical vector must hit");
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
    let store = connect(&url).await;
    let policy = unique("p-scope");

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
    let cross = store
        .lookup(&policy, 0, "fp-b", &[1.0, 0.0], 0.5)
        .await
        .unwrap();
    assert!(cross.is_none(), "entries must not cross scope fingerprints");
    let same = store
        .lookup(&policy, 0, "fp-a", &[1.0, 0.0], 0.5)
        .await
        .unwrap();
    assert!(same.is_some());
}

#[tokio::test]
async fn store_upserts_by_exact_key() {
    let Some(url) = vector_url() else { return };
    let store = connect(&url).await;
    let policy = unique("p-upsert");

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
    let hit = store
        .lookup(&policy, 0, "fp", &[1.0, 0.0], 0.9)
        .await
        .unwrap()
        .expect("hit");
    assert_eq!(
        hit.response.message.content_str(),
        "v2",
        "refresh must replace the document in place",
    );
}

#[tokio::test]
async fn generation_rotation_orphans_old_entries() {
    let Some(url) = vector_url() else { return };
    let store = connect(&url).await;
    let policy = unique("p-gen");

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
    let hit = store
        .lookup(&policy, 1, "fp", &[0.0, 1.0], 0.9)
        .await
        .unwrap()
        .expect("new-generation hit");
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
    let store = connect(&url).await;
    let policy = unique("p-stale");

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
    let hit = store
        .lookup(&policy, 1, "fp", &[0.0, 1.0], 0.9)
        .await
        .unwrap()
        .expect("generation-1 document must survive the stale write");
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
    let store = connect(&url).await;
    let live = unique("p-sweep-live");
    let dead = unique("p-sweep-dead");

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
    let hit = store
        .lookup(&live, 0, "fp", &[1.0, 0.0], 0.9)
        .await
        .unwrap()
        .expect("live index must survive the sweep");
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
    assert!(store
        .lookup(&live, 0, "fp", &[0.0, 1.0], 0.9)
        .await
        .unwrap()
        .is_some());
}

#[tokio::test]
async fn dims_change_rotates_the_index() {
    let Some(url) = vector_url() else { return };
    let store = connect(&url).await;
    let policy = unique("p-dims");

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
    let hit = store
        .lookup(&policy, 0, "fp", &[1.0, 0.0, 0.0], 0.9)
        .await
        .unwrap()
        .expect("three-dim hit");
    assert_eq!(hit.response.message.content_str(), "three-dim");
}

#[tokio::test]
async fn entries_are_shared_across_store_instances() {
    let Some(url) = vector_url() else { return };
    let store_a = connect(&url).await;
    let store_b = connect(&url).await;
    let policy = unique("p-shared");

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
    let hit = store_b
        .lookup(&policy, 0, "fp", &[1.0, 0.0], 0.9)
        .await
        .unwrap()
        .expect("replica B must see replica A's entry");
    assert_eq!(hit.response.message.content_str(), "from-a");
}

#[tokio::test]
async fn ttl_expires_documents() {
    let Some(url) = vector_url() else { return };
    let store = connect(&url).await;
    let policy = unique("p-ttl");

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
    tokio::time::sleep(Duration::from_millis(400)).await;
    let miss = store
        .lookup(&policy, 0, "fp", &[1.0, 0.0], 0.5)
        .await
        .unwrap();
    assert!(miss.is_none(), "expired document must not match");
}

#[tokio::test]
async fn probe_fails_on_plain_redis() {
    // Uses the PLAIN redis the exact-cache tests run against; skip when
    // absent or when it actually has vector support (e.g. local dev
    // pointing both vars at one Redis 8).
    let Some(url) = std::env::var("CACHE_TEST_REDIS_PLAIN_URL").ok() else {
        return;
    };
    let store = RedisSemanticCache::connect(&single(&url))
        .await
        .expect("connect");
    assert!(
        store.probe().await.is_err(),
        "probe must fail on a server without vector search",
    );
}
