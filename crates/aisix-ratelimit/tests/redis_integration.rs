//! Shared-counter tests for `RedisStore` against a live Redis.
//!
//! Runs only when `RATELIMIT_TEST_REDIS_URL` is set (CI spins
//! `redis:7-alpine` as a service; absence is a no-op so local unit runs
//! stay hermetic). Two `RedisStore` instances stand in for two DP
//! replicas pointed at one Redis — the exact api7/AISIX-Cloud#798 shape:
//! a limit hit on one replica must already be hit on the other.

use std::time::Duration;

use aisix_core::{RateLimit, RateLimitScope};
use aisix_ratelimit::{RateStore, RedisStore};

fn redis_url() -> Option<String> {
    std::env::var("RATELIMIT_TEST_REDIS_URL").ok()
}

/// Unique bucket key per test so they don't clobber each other (the store
/// prefixes with a fixed `aisix:rl`; isolation comes from the key).
fn unique_key(tag: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("test:{tag}:{nanos:x}")
}

fn rl() -> RateLimit {
    RateLimit::default()
}

async fn store(url: &str) -> RedisStore {
    RedisStore::connect(url).await.expect("redis connect")
}

#[tokio::test]
async fn rpm_counter_is_shared_across_replicas() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    let a = store(&url).await;
    let b = store(&url).await;
    let key = unique_key("rpm");
    let limits = RateLimit {
        rpm: Some(1),
        ..rl()
    };

    // Replica A burns the only slot in the minute window.
    a.acquire(&key, &limits, "a-1")
        .await
        .expect("first allowed");

    // Replica B sees the SAME counter → rejected. Pre-#798 (per-replica
    // memory) this would have been allowed, doubling the limit.
    let err = b
        .acquire(&key, &limits, "b-1")
        .await
        .expect_err("second replica must be rejected by shared counter");
    assert!(
        matches!(
            err,
            aisix_ratelimit::RateLimitError::Requests {
                scope: RateLimitScope::Requests,
                ..
            }
        ),
        "got {err:?}"
    );
}

#[tokio::test]
async fn rps_window_rolls_over_on_the_shared_counter() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    let a = store(&url).await;
    let b = store(&url).await;
    let key = unique_key("rps");
    let limits = RateLimit {
        rps: Some(1),
        ..rl()
    };

    a.acquire(&key, &limits, "a-1")
        .await
        .expect("first allowed");
    assert!(
        b.acquire(&key, &limits, "b-1").await.is_err(),
        "same second is shared-rejected"
    );

    // Cross the 1s boundary — the next-second key is fresh.
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    b.acquire(&key, &limits, "b-2")
        .await
        .expect("next second has a fresh window");
}

#[tokio::test]
async fn token_usage_is_shared_across_replicas() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    let a = store(&url).await;
    let b = store(&url).await;
    let key = unique_key("tpm");
    let limits = RateLimit {
        tpm: Some(1_000),
        ..rl()
    };

    // A admits then over-commits the minute's token budget.
    a.acquire(&key, &limits, "a-1")
        .await
        .expect("first allowed");
    a.commit(&key, 1_500, "a-1").await;

    // B's pre-check sees tpm > 1000 on the shared counter → rejected.
    let err = b
        .acquire(&key, &limits, "b-1")
        .await
        .expect_err("token cap is shared");
    assert!(
        matches!(err, aisix_ratelimit::RateLimitError::Tokens { .. }),
        "got {err:?}"
    );
}

#[tokio::test]
async fn concurrency_slot_is_shared_and_released_across_replicas() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    let a = store(&url).await;
    let b = store(&url).await;
    let key = unique_key("conc");
    let limits = RateLimit {
        concurrency: Some(1),
        ..rl()
    };

    // A takes the only in-flight slot.
    a.acquire(&key, &limits, "a-1")
        .await
        .expect("first allowed");
    // B is blocked while A holds it.
    assert!(
        matches!(
            b.acquire(&key, &limits, "b-1").await,
            Err(aisix_ratelimit::RateLimitError::Concurrency)
        ),
        "concurrency slot must be shared across replicas"
    );

    // A finishes → releases the slot (sync + detached ZREM). The ZREM is
    // fire-and-forget, so poll until the slot frees (bounded) rather than
    // assuming a fixed propagation delay that could flake on slow CI.
    a.release(&key, "a-1");
    let mut acquired = false;
    for _ in 0..50 {
        if b.acquire(&key, &limits, "b-2").await.is_ok() {
            acquired = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(acquired, "slot must free up cluster-wide after release");
}

#[tokio::test]
async fn stale_concurrency_slot_is_reclaimed_after_ttl() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    // 1s slot lifetime: a never-released slot (crashed replica) is pruned.
    let a = store(&url).await.with_conc_ttl(1);
    let b = store(&url).await.with_conc_ttl(1);
    let key = unique_key("conc-ttl");
    let limits = RateLimit {
        concurrency: Some(1),
        ..rl()
    };

    a.acquire(&key, &limits, "a-leaked")
        .await
        .expect("first allowed");
    // Never release — simulate a crashed replica holding the slot.
    assert!(
        b.acquire(&key, &limits, "b-1").await.is_err(),
        "slot held while fresh"
    );

    tokio::time::sleep(Duration::from_millis(1_300)).await;
    b.acquire(&key, &limits, "b-2")
        .await
        .expect("stale slot reclaimed after conc_ttl");
}
