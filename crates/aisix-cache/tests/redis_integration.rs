//! End-to-end Redis tests against a live Redis instance.
//!
//! Runs only when `CACHE_TEST_REDIS_URL` is set (e.g. on CI which spins
//! `redis:7-alpine` as a service). The unit test module in
//! `src/redis.rs` handles hermetic checks; this file proves the
//! request → upstream → cache round-trip actually round-trips.

#![cfg(feature = "redis")]

use std::time::Duration;

use aisix_cache::{Cache, RedisCache};
use aisix_core::{RedisConnConfig, RedisMode};
use aisix_gateway::{ChatMessage, ChatResponse, FinishReason, UsageStats};

fn redis_url() -> Option<String> {
    std::env::var("CACHE_TEST_REDIS_URL").ok()
}

fn single(url: &str) -> RedisConnConfig {
    RedisConnConfig {
        mode: RedisMode::Single,
        url: Some(url.to_string()),
        ..Default::default()
    }
}

fn cluster_cfg() -> Option<RedisConnConfig> {
    let nodes = std::env::var("CACHE_TEST_REDIS_CLUSTER_NODES").ok()?;
    Some(RedisConnConfig {
        mode: RedisMode::Cluster,
        nodes: nodes.split(',').map(|s| s.trim().to_string()).collect(),
        ..Default::default()
    })
}

fn sentinel_cfg() -> Option<RedisConnConfig> {
    let sentinels = std::env::var("CACHE_TEST_REDIS_SENTINELS").ok()?;
    let master = std::env::var("CACHE_TEST_REDIS_MASTER").ok()?;
    Some(RedisConnConfig {
        mode: RedisMode::Sentinel,
        sentinels: sentinels.split(',').map(|s| s.trim().to_string()).collect(),
        master_name: Some(master),
        ..Default::default()
    })
}

fn sample(content: &str) -> ChatResponse {
    ChatResponse {
        id: "cmpl-int-1".into(),
        model: "openai/gpt-4o".into(),
        message: ChatMessage::assistant(content),
        finish_reason: FinishReason::Stop,
        usage: UsageStats::new(3, 5),
    }
}

#[tokio::test]
async fn put_then_get_round_trips_against_real_redis() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: CACHE_TEST_REDIS_URL not set");
        return;
    };

    let cache = RedisCache::connect(&single(&url))
        .await
        .expect("redis connect")
        .with_prefix(format!("aisix:test:{}", uuid_like()));

    let key = "fp-roundtrip";
    cache.put(key, sample("hello back")).await.unwrap();
    let got = cache.get(key).await.unwrap().expect("hit");
    assert_eq!(got.message.content_str(), "hello back");
    assert_eq!(got.usage.total_tokens, 8);
}

#[tokio::test]
async fn put_then_get_preserves_null_content_through_cache() {
    // #395: a tool_calls response carries `message.content == None`. The
    // cache persists `ChatResponse` as JSON; this proves `None`
    // survives the store→load round-trip as `None` (not coerced to
    // `Some("")`), so a cache hit serves the same `content: null` the
    // upstream returned.
    let Some(url) = redis_url() else {
        eprintln!("skipping: CACHE_TEST_REDIS_URL not set");
        return;
    };

    let cache = RedisCache::connect(&single(&url))
        .await
        .expect("redis connect")
        .with_prefix(format!("aisix:test:{}", uuid_like()));

    let message: ChatMessage =
        serde_json::from_str(r#"{"role":"assistant","content":null}"#).unwrap();
    assert!(message.content.is_none());
    let resp = ChatResponse {
        id: "cmpl-null-1".into(),
        model: "openai/gpt-4o".into(),
        message,
        finish_reason: FinishReason::ToolCalls,
        usage: UsageStats::new(3, 5),
    };

    let key = "fp-null-content";
    cache.put(key, resp).await.unwrap();
    let got = cache.get(key).await.unwrap().expect("hit");
    assert!(
        got.message.content.is_none(),
        "null content must round-trip as None, not Some(\"\")"
    );
}

#[tokio::test]
async fn ttl_eviction_drops_entry_after_window() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: CACHE_TEST_REDIS_URL not set");
        return;
    };

    let cache = RedisCache::connect(&single(&url))
        .await
        .expect("redis connect")
        .with_prefix(format!("aisix:test:{}", uuid_like()))
        .with_ttl(Duration::from_secs(1));

    cache.put("ttl-key", sample("ephemeral")).await.unwrap();
    assert!(cache.get("ttl-key").await.unwrap().is_some());

    // Redis EX 1 means "expires sometime within the next second" —
    // sleep 1.5s to leave headroom.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    assert!(cache.get("ttl-key").await.unwrap().is_none());
}

#[tokio::test]
async fn put_with_ttl_honors_per_entry_window_over_global() {
    // Regression: a Redis-backed `CachePolicy` carries its own
    // `ttl_seconds`, which the proxy passes via `put_with_ttl`. The entry
    // must expire on that per-policy window, NOT the instance-global
    // default. With a 300s global and a 1s per-entry TTL, a backend that
    // drops the per-entry value keeps the entry alive well past 1.5s; the
    // contract requires it gone.
    let Some(url) = redis_url() else {
        eprintln!("skipping: CACHE_TEST_REDIS_URL not set");
        return;
    };

    let cache = RedisCache::connect(&single(&url))
        .await
        .expect("redis connect")
        .with_prefix(format!("aisix:test:{}", uuid_like()))
        .with_ttl(Duration::from_secs(300));

    cache
        .put_with_ttl(
            "per-entry-ttl",
            sample("short-lived"),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
    assert!(
        cache.get("per-entry-ttl").await.unwrap().is_some(),
        "entry must be present immediately after write"
    );

    // Per-entry TTL is 1s (EX 1 = expire within ≤1s); sleep past it with
    // headroom. The 300s instance global must not win.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    assert!(
        cache.get("per-entry-ttl").await.unwrap().is_none(),
        "per-policy ttl_seconds (1s) must be honored, not the 300s instance global"
    );
}

#[tokio::test]
async fn missing_key_returns_none() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: CACHE_TEST_REDIS_URL not set");
        return;
    };

    let cache = RedisCache::connect(&single(&url))
        .await
        .expect("redis connect")
        .with_prefix(format!("aisix:test:{}", uuid_like()));

    assert!(cache.get("does-not-exist").await.unwrap().is_none());
}

#[tokio::test]
async fn put_then_get_round_trips_against_cluster() {
    let Some(cfg) = cluster_cfg() else {
        eprintln!("skipping: CACHE_TEST_REDIS_CLUSTER_NODES not set");
        return;
    };
    let cache = RedisCache::connect(&cfg)
        .await
        .expect("cluster connect")
        .with_prefix(format!("aisix:test:{}", uuid_like()));

    cache.put("cluster-key", sample("clustered")).await.unwrap();
    let got = cache.get("cluster-key").await.unwrap().expect("hit");
    assert_eq!(got.message.content_str(), "clustered");
}

#[tokio::test]
async fn put_then_get_round_trips_against_sentinel() {
    let Some(cfg) = sentinel_cfg() else {
        eprintln!("skipping: CACHE_TEST_REDIS_SENTINELS / _MASTER not set");
        return;
    };
    let cache = RedisCache::connect(&cfg)
        .await
        .expect("sentinel connect")
        .with_prefix(format!("aisix:test:{}", uuid_like()));

    cache
        .put("sentinel-key", sample("via-master"))
        .await
        .unwrap();
    let got = cache.get("sentinel-key").await.unwrap().expect("hit");
    assert_eq!(got.message.content_str(), "via-master");
}

#[tokio::test]
async fn env_namespace_isolates_identical_fingerprints() {
    // Two environments pointed at the same (user-provided) Redis must not
    // share cache entries even for a byte-identical request — the key is a
    // content-only fingerprint, so `with_env_namespace` is the only thing
    // keeping them apart.
    let Some(url) = redis_url() else {
        eprintln!("skipping: CACHE_TEST_REDIS_URL not set");
        return;
    };
    // Same base prefix for both handles, so any isolation is due to the env
    // segment alone (not a stray unique prefix).
    let base = format!("aisix:test:{}", uuid_like());

    let env_a = RedisCache::connect(&single(&url))
        .await
        .expect("redis connect")
        .with_prefix(base.clone())
        .with_env_namespace("env-a");
    let env_b = RedisCache::connect(&single(&url))
        .await
        .expect("redis connect")
        .with_prefix(base.clone())
        .with_env_namespace("env-b");

    let key = "identical-fingerprint";
    env_a.put(key, sample("answer for A")).await.unwrap();

    // env-b must NOT see env-a's entry under the identical fingerprint.
    assert!(
        env_b.get(key).await.unwrap().is_none(),
        "distinct env namespaces must not share cache entries"
    );
    // env-a still reads its own.
    assert_eq!(
        env_a
            .get(key)
            .await
            .unwrap()
            .expect("hit")
            .message
            .content_str(),
        "answer for A"
    );

    // Control: empty env_id leaves the prefix unchanged, so a bare handle
    // shares the base namespace (standalone / v2 behaviour is preserved).
    let bare = RedisCache::connect(&single(&url))
        .await
        .expect("redis connect")
        .with_prefix(base.clone());
    let bare_ns = RedisCache::connect(&single(&url))
        .await
        .expect("redis connect")
        .with_prefix(base.clone())
        .with_env_namespace("");
    bare.put("empty-env", sample("shared")).await.unwrap();
    assert!(
        bare_ns.get("empty-env").await.unwrap().is_some(),
        "empty env_id must not change the namespace"
    );
}

/// Cheap unique-ish suffix to keep tests from clobbering each other.
/// We don't need cryptographic uniqueness — `cargo test` runs each test
/// file in a single process, so nanos + thread-id give plenty of spread.
fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{nanos:x}-{:?}", std::thread::current().id()).replace(['(', ')', ' '], "")
}
