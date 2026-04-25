//! End-to-end Redis tests against a live Redis instance.
//!
//! Runs only when `CACHE_TEST_REDIS_URL` is set (e.g. on CI which spins
//! `redis:7-alpine` as a service). The unit test module in
//! `src/redis.rs` handles hermetic checks; this file proves the
//! request → upstream → cache round-trip actually round-trips.

#![cfg(feature = "redis")]

use std::time::Duration;

use aisix_cache::{Cache, RedisCache};
use aisix_gateway::{ChatMessage, ChatResponse, FinishReason, UsageStats};

fn redis_url() -> Option<String> {
    std::env::var("CACHE_TEST_REDIS_URL").ok()
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

    let cache = RedisCache::connect(&url)
        .await
        .expect("redis connect")
        .with_prefix(format!("aisix:test:{}", uuid_like()));

    let key = "fp-roundtrip";
    cache.put(key, sample("hello back")).await.unwrap();
    let got = cache.get(key).await.unwrap().expect("hit");
    assert_eq!(got.message.content, "hello back");
    assert_eq!(got.usage.total_tokens, 8);
}

#[tokio::test]
async fn ttl_eviction_drops_entry_after_window() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: CACHE_TEST_REDIS_URL not set");
        return;
    };

    let cache = RedisCache::connect(&url)
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
async fn missing_key_returns_none() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: CACHE_TEST_REDIS_URL not set");
        return;
    };

    let cache = RedisCache::connect(&url)
        .await
        .expect("redis connect")
        .with_prefix(format!("aisix:test:{}", uuid_like()));

    assert!(cache.get("does-not-exist").await.unwrap().is_none());
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
