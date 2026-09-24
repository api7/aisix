//! End-to-end Redis tests against a live Redis instance.
//!
//! Runs only when `CACHE_TEST_REDIS_URL` is set (e.g. on CI which spins
//! `redis:7-alpine` as a service). The unit test module in
//! `src/redis.rs` handles hermetic checks; this file proves the
//! request → upstream → cache round-trip actually round-trips.

#![cfg(feature = "redis")]

use std::time::Duration;

use aisix_cache::{Cache, RedisCache, SemanticCacheStore};
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

/// A TCP relay in front of Redis that a test can black-hole: the socket
/// stays open and nothing is ever forwarded or answered, which is what a
/// paused container, a downed host or a partitioned network looks like
/// from the client end. An error reply would be the *cheap* failure —
/// the client learns immediately and no budget is spent.
struct RedisBlackhole {
    port: u16,
    hole: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl RedisBlackhole {
    async fn start(upstream: &str) -> Self {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let upstream = upstream
            .trim_start_matches("redis://")
            .trim_end_matches('/')
            .to_string();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let hole = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = hole.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut client, _)) = listener.accept().await else {
                    return;
                };
                let Ok(mut server) = tokio::net::TcpStream::connect(&upstream).await else {
                    continue;
                };
                let flag = flag.clone();
                tokio::spawn(async move {
                    let mut from_client = [0u8; 8192];
                    let mut from_server = [0u8; 8192];
                    loop {
                        let held = || flag.load(std::sync::atomic::Ordering::Relaxed);
                        tokio::select! {
                            n = client.read(&mut from_client) => {
                                let Ok(n) = n else { return };
                                if n == 0 {
                                    return;
                                }
                                if held() {
                                    continue;
                                }
                                if server.write_all(&from_client[..n]).await.is_err() {
                                    return;
                                }
                            }
                            n = server.read(&mut from_server) => {
                                let Ok(n) = n else { return };
                                if n == 0 {
                                    return;
                                }
                                if held() {
                                    continue;
                                }
                                if client.write_all(&from_server[..n]).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                });
            }
        });
        Self { port, hole }
    }

    fn url(&self) -> String {
        format!("redis://127.0.0.1:{}", self.port)
    }

    fn blackhole(&self) {
        self.hole.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// One request against a black-holed Redis must cost the cache ONE
/// command budget, not one per connection.
///
/// The cache subsystem holds two connections to the same `cache.redis` —
/// exact-KV and vector search — and a single chat request touches both
/// twice: exact lookup, semantic lookup, exact write, semantic write. A
/// breaker per connection meant each of those four found a breaker no
/// earlier operation had opened, so the request paid the budget four
/// times over (release QA measured 20.0s at the default 5s budget, 4.0s
/// at 2s, against 5.0s for an exact-only policy). The two connections
/// stay separate — they must not serialize on one pipeline — so what is
/// shared is the failure policy.
#[tokio::test]
async fn the_whole_cache_subsystem_pays_one_budget_per_request() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: CACHE_TEST_REDIS_URL not set");
        return;
    };
    // Below the 5s default so the bound below cannot pass on a build
    // that ignored the configured budget.
    const BUDGET_SECS: u64 = 2;
    let relay = RedisBlackhole::start(&url).await;
    let cfg = RedisConnConfig {
        timeout_secs: BUDGET_SECS,
        ..single(&relay.url())
    };

    // Exactly how the bootstrap wires them: one policy, two connections.
    let policy = aisix_cache::FailurePolicy::new(&cfg);
    let exact = aisix_cache::RedisCache::connect_with(&cfg, &policy)
        .await
        .expect("exact cache connects through the relay");
    let vector = aisix_cache::RedisSemanticCache::connect_with(&cfg, &policy)
        .await
        .expect("vector store connects through the relay");

    relay.blackhole();

    let started = std::time::Instant::now();
    // The four operations one chat request performs, in order.
    let _ = exact.get("subsystem-budget").await;
    let _ = vector
        .lookup("policy-1", 1, "scope", &[1.0, 0.0], 0.5)
        .await;
    let _ = exact.put("subsystem-budget", sample("hi")).await;
    let _ = vector
        .store(
            "policy-1",
            1,
            "scope",
            "subsystem-budget",
            vec![1.0, 0.0],
            sample("hi"),
            Duration::from_secs(60),
            100,
        )
        .await;
    let elapsed = started.elapsed();

    assert!(
        elapsed >= Duration::from_secs(BUDGET_SECS),
        "one operation must actually reach the blackhole and spend its \
         budget; {elapsed:?} means the relay was never in the path",
    );
    assert!(
        elapsed < Duration::from_millis(BUDGET_SECS * 1000 + 1_500),
        "the request must pay ONE budget for the subsystem, not one per \
         connection; took {elapsed:?}",
    );
}
