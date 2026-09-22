//! Shared-counter tests for `RedisStore` against a live Redis.
//!
//! Runs only when `RATELIMIT_TEST_REDIS_URL` is set (CI spins
//! `redis:7-alpine` as a service; absence is a no-op so local unit runs
//! stay hermetic). Two `RedisStore` instances stand in for two DP
//! replicas pointed at one Redis — the exact api7/AISIX-Cloud#798 shape:
//! a limit hit on one replica must already be hit on the other.

use std::time::Duration;

use aisix_core::{RateLimit, RateLimitScope, RedisConnConfig, RedisMode};
use aisix_obs::metrics::Metrics;
use aisix_ratelimit::{RateStore, RedisStore};

fn redis_url() -> Option<String> {
    std::env::var("RATELIMIT_TEST_REDIS_URL").ok()
}

fn single(url: &str) -> RedisConnConfig {
    RedisConnConfig {
        mode: RedisMode::Single,
        url: Some(url.to_string()),
        ..Default::default()
    }
}

/// Cluster seed nodes, e.g. `RATELIMIT_TEST_REDIS_CLUSTER_NODES=redis://127.0.0.1:7000,redis://127.0.0.1:7001`.
fn cluster_cfg() -> Option<RedisConnConfig> {
    let nodes = std::env::var("RATELIMIT_TEST_REDIS_CLUSTER_NODES").ok()?;
    Some(RedisConnConfig {
        mode: RedisMode::Cluster,
        nodes: nodes.split(',').map(|s| s.trim().to_string()).collect(),
        ..Default::default()
    })
}

/// Sentinel topology, e.g. `RATELIMIT_TEST_REDIS_SENTINELS=redis://127.0.0.1:26379`
/// plus `RATELIMIT_TEST_REDIS_MASTER=mymaster`.
fn sentinel_cfg() -> Option<RedisConnConfig> {
    let sentinels = std::env::var("RATELIMIT_TEST_REDIS_SENTINELS").ok()?;
    let master = std::env::var("RATELIMIT_TEST_REDIS_MASTER").ok()?;
    Some(RedisConnConfig {
        mode: RedisMode::Sentinel,
        sentinels: sentinels.split(',').map(|s| s.trim().to_string()).collect(),
        master_name: Some(master),
        ..Default::default()
    })
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
    RedisStore::connect(&single(url))
        .await
        .expect("redis connect")
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

/// The shared backend must describe a refusal exactly as the local one
/// does: the 429's `x-ratelimit-*` headers are the same contract on a
/// clustered deployment as on a single node.
///
/// The Lua script reports the refused dimension as an INDEX into the
/// same `request_dims` / `token_dims` list the caller pushed into ARGV,
/// so the name is reconstructed on the Rust side. A key carrying several
/// windows is what makes that mapping observable — with a single window
/// any index would decode to the same name.
#[tokio::test]
async fn refusal_detail_survives_the_shared_backend() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    let a = store(&url).await;
    let b = store(&url).await;
    let key = unique_key("detail");
    // rps is looser than rpm here, so the SECOND dimension in the
    // request list is the one that refuses — an off-by-one in the index
    // mapping would report `rps`.
    let limits = RateLimit {
        rps: Some(10),
        rpm: Some(1),
        ..rl()
    };

    a.acquire(&key, &limits, "a-1")
        .await
        .expect("first allowed");
    let err = b
        .acquire(&key, &limits, "b-1")
        .await
        .expect_err("shared counter must refuse the second replica");

    let detail = err.detail();
    assert_eq!(detail.dimension, "rpm", "got {err:?}");
    assert_eq!(detail.limit, 1);
    assert_eq!(detail.remaining, 0);
    assert!(
        (1..=60).contains(&detail.reset_secs),
        "a minute window resets within the minute, got {}",
        detail.reset_secs
    );
}

/// A concurrency refusal from the shared backend carries the gauge's
/// own state plus the fixed hint — the same shape the local store
/// produces, so a client cannot tell the backends apart.
#[tokio::test]
async fn concurrency_refusal_detail_survives_the_shared_backend() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    let a = store(&url).await;
    let b = store(&url).await;
    let key = unique_key("conc-detail");
    let limits = RateLimit {
        concurrency: Some(1),
        ..rl()
    };

    a.acquire(&key, &limits, "a-1")
        .await
        .expect("first allowed");
    let err = b
        .acquire(&key, &limits, "b-1")
        .await
        .expect_err("shared concurrency gauge must refuse");

    let detail = err.detail();
    assert_eq!(detail.dimension, "concurrency", "got {err:?}");
    assert_eq!(detail.limit, 1);
    assert_eq!(detail.remaining, 0);
    assert_eq!(
        detail.reset_secs,
        aisix_ratelimit::CONCURRENCY_RETRY_AFTER_SECS
    );
    assert_eq!(detail.reset_secs, 60);
}

/// Sleep until the next whole second starts.
///
/// The rps counter buckets server-side: the Lua script reads `now` from
/// `redis.call('TIME')` and derives `ws = now - (now % window)`. So the
/// boundary that matters is Redis's clock at script-execution time, and
/// a test that starts at an arbitrary offset into a second is racing
/// whatever is left of it. Aligning first puts a full second between the
/// pair and the next boundary, so only a Redis stalled for more than a
/// second could still split them — which is a real failure, not a race.
///
/// This uses the client clock to align against a server-side bucket,
/// which holds because CI runs Redis as a service container on the same
/// host. A remote Redis with clock skew would need the alignment read
/// from `TIME` instead.
async fn wait_for_second_boundary() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after the epoch");
    let remainder = Duration::from_nanos(u64::from(now.subsec_nanos()));
    tokio::time::sleep(Duration::from_secs(1) - remainder).await;
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

    // The rps counter buckets by whole wall-clock second, so both
    // acquires have to land inside the SAME bucket for the second one to
    // be rejected. Starting mid-second, two Redis round-trips can
    // straddle the boundary and the second call is legitimately allowed
    // — the assertion below then fails with nothing wrong in the code.
    // Wait out whatever is left of the current second first, so the pair
    // begins with a full one ahead of it.
    wait_for_second_boundary().await;

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

/// #950: the shared backend must draw the boundary where the local one
/// does. Committed usage landing EXACTLY on the token cap is a spent
/// budget, so the next request — on any replica — is refused. A `>` here
/// would make the observable limit depend on which backend is
/// configured, which is the one thing swapping them must never change.
#[tokio::test]
async fn a_token_window_exactly_on_the_limit_refuses_the_next_request() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    let a = store(&url).await;
    let b = store(&url).await;
    let key = unique_key("tpm-exact");
    let limits = RateLimit {
        tpm: Some(1_000),
        ..rl()
    };

    a.acquire(&key, &limits, "a-1")
        .await
        .expect("first allowed");
    a.commit(&key, 1_000, "a-1").await; // the whole budget, exactly

    let err = b
        .acquire(&key, &limits, "b-1")
        .await
        .expect_err("a fully consumed token budget must stop admitting");
    assert!(
        matches!(err, aisix_ratelimit::RateLimitError::Tokens { .. }),
        "got {err:?}"
    );

    // One token short still admits, so the assertion above is about the
    // budget being spent rather than about any usage at all.
    let under = unique_key("tpm-under");
    a.acquire(&under, &limits, "a-2")
        .await
        .expect("first allowed");
    a.commit(&under, 999, "a-2").await;
    b.acquire(&under, &limits, "b-2")
        .await
        .expect("one token of budget left still admits");
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
            Err(aisix_ratelimit::RateLimitError::Concurrency { .. })
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

/// Redis Cluster: the multi-key acquire/commit Lua must route to the slot
/// owning the `{bucket}` hash tag and enforce one shared window. A wrong
/// (or missing) routing key would surface as a CROSSSLOT/MOVED error here.
#[tokio::test]
async fn cluster_shared_counter_routes_multi_key_lua() {
    let Some(cfg) = cluster_cfg() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_CLUSTER_NODES not set");
        return;
    };
    let a = RedisStore::connect(&cfg).await.expect("cluster connect");
    let b = RedisStore::connect(&cfg).await.expect("cluster connect");
    let key = unique_key("cluster-rpm");
    let limits = RateLimit {
        rpm: Some(1),
        tpm: Some(1_000),
        concurrency: Some(5),
        ..rl()
    };

    // First acquire (touches conc ZSET + rpm + tpm keys, all hash-tagged
    // to one slot) must succeed — proves the EVAL routed correctly.
    a.acquire(&key, &limits, "a-1")
        .await
        .expect("first allowed on cluster");
    // commit also runs a multi-key script on the same slot.
    a.commit(&key, 10, "a-1").await;
    // Second replica is rejected by the shared rpm counter — assert the
    // specific rejection, not just any error (which would also pass on a
    // routing/connection failure and mask a real cluster regression).
    let err = b
        .acquire(&key, &limits, "b-1")
        .await
        .expect_err("second replica must be rejected by the shared rpm counter");
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

/// Redis Sentinel: connect resolves the master through the sentinels, and
/// the shared-counter semantics work end-to-end against the discovered
/// master.
#[tokio::test]
async fn sentinel_shared_counter_round_trips() {
    let Some(cfg) = sentinel_cfg() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_SENTINELS / _MASTER not set");
        return;
    };
    let a = RedisStore::connect(&cfg).await.expect("sentinel connect");
    let b = RedisStore::connect(&cfg).await.expect("sentinel connect");
    let key = unique_key("sentinel-rpm");
    let limits = RateLimit {
        rpm: Some(1),
        ..rl()
    };

    a.acquire(&key, &limits, "a-1")
        .await
        .expect("first allowed via sentinel master");
    let err = b
        .acquire(&key, &limits, "b-1")
        .await
        .expect_err("second replica must be rejected by the shared rpm counter");
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
async fn env_namespace_isolates_model_alias_bucket() {
    // The model inline rate limit buckets on the env-local alias
    // (`model:<name>`), which is NOT a globally-unique id. Two environments
    // sharing one Redis must keep independent counters for the same alias —
    // `with_env_namespace` is what separates them. (The api_key / policy
    // buckets are UUIDs and never collided; the prefix just covers them too.)
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    // A shared run tag keeps the test hermetic; the env-a / env-b suffixes are
    // what must isolate the two counters.
    let run = unique_key("modelns");
    let env_a_ns = format!("{run}:env-a");
    let env_b_ns = format!("{run}:env-b");

    let env_a = store(&url).await.with_env_namespace(&env_a_ns);
    let env_b = store(&url).await.with_env_namespace(&env_b_ns);

    // Identical alias bucket in both environments.
    let key = "model:gpt-4o";
    let limits = RateLimit {
        rpm: Some(1),
        ..rl()
    };

    // env-a burns its single rpm slot for the alias.
    env_a
        .acquire(key, &limits, "a-1")
        .await
        .expect("env-a first allowed");
    env_a
        .acquire(key, &limits, "a-2")
        .await
        .expect_err("env-a is now at its own limit");

    // env-b, identical alias bucket, is unaffected — a distinct counter.
    env_b
        .acquire(key, &limits, "b-1")
        .await
        .expect("different env must not share the model:<alias> counter");

    // Control: a second handle in env-a shares the exhausted counter, proving
    // the isolation comes from the env namespace, not the handle identity.
    let env_a2 = store(&url).await.with_env_namespace(&env_a_ns);
    env_a2
        .acquire(key, &limits, "a-3")
        .await
        .expect_err("same env shares the counter");
}

/// A TCP relay in front of Redis that a test can break, in either of the
/// two shapes a real outage takes.
///
/// - [`cut`](RedisCutoff::cut) — the relay answers each request with a
///   Redis error instead of forwarding it. Redis is *reachable* and
///   refusing, which is the cheap failure: the client learns immediately.
/// - [`blackhole`](RedisCutoff::blackhole) — the socket stays open and
///   nothing is ever forwarded or answered, which is what a stopped
///   container, a downed host or a partitioned network looks like from
///   the client end. Nothing arrives, nothing is refused, and without a
///   command budget the caller waits on TCP retransmission for minutes.
struct RedisCutoff {
    port: u16,
    cut: std::sync::Arc<std::sync::atomic::AtomicBool>,
    hole: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl RedisCutoff {
    async fn start(upstream: &str) -> Self {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let upstream = upstream
            .trim_start_matches("redis://")
            .trim_end_matches('/')
            .to_string();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let cut = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hole = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = cut.clone();
        let hole_flag = hole.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut client, _)) = listener.accept().await else {
                    return;
                };
                let Ok(mut server) = tokio::net::TcpStream::connect(&upstream).await else {
                    continue;
                };
                let flag = flag.clone();
                let hole_flag = hole_flag.clone();
                tokio::spawn(async move {
                    let mut from_client = [0u8; 8192];
                    let mut from_server = [0u8; 8192];
                    loop {
                        tokio::select! {
                            n = client.read(&mut from_client) => {
                                let Ok(n) = n else { return };
                                if n == 0 {
                                    return;
                                }
                                if hole_flag.load(std::sync::atomic::Ordering::Relaxed) {
                                    // Swallowed: no forward, no reply, no
                                    // close. The peer has simply gone quiet.
                                } else if flag.load(std::sync::atomic::Ordering::Relaxed) {
                                    if client
                                        .write_all(b"-ERR simulated redis outage\r\n")
                                        .await
                                        .is_err()
                                    {
                                        return;
                                    }
                                } else if server.write_all(&from_client[..n]).await.is_err() {
                                    return;
                                }
                            }
                            n = server.read(&mut from_server) => {
                                let Ok(n) = n else { return };
                                // EOF first: a closed upstream read stays
                                // ready forever, so `continue`-ing on it
                                // would spin the relay task.
                                if n == 0 {
                                    return;
                                }
                                if hole_flag.load(std::sync::atomic::Ordering::Relaxed) {
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
        Self { port, cut, hole }
    }

    fn url(&self) -> String {
        format!("redis://127.0.0.1:{}", self.port)
    }

    fn cut(&self) {
        self.cut.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn blackhole(&self) {
        self.hole.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Forward again. Connections opened from now on work; the ones the
    /// blackhole swallowed mid-handshake were abandoned by the client
    /// when its own budget expired, which is what a Redis coming back up
    /// looks like from the caller's end.
    fn heal(&self) {
        self.hole.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

fn counter_value(rendered: &str, operation: &str) -> f64 {
    let needle = format!("aisix_redis_failures_total{{operation=\"{operation}\"}} ");
    rendered
        .lines()
        .find_map(|l| l.trim().strip_prefix(&needle))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0.0)
}

/// #1060: the shared limiter fails OPEN — an unreachable Redis silently
/// degrades every replica to its own in-memory counters, so the cluster
/// stops enforcing one global window and nothing about the answer to a
/// caller changes. The warning logs once per outage, which cannot say how
/// long or how hard it is failing. `aisix_redis_failures_total` is the
/// only signal an operator can scrape, and before this it was never
/// emitted at all.
#[tokio::test]
async fn a_redis_outage_counts_every_failed_operation() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    let metrics = Metrics::new(false);
    let relay = RedisCutoff::start(&url).await;
    let store = RedisStore::connect(&single(&relay.url()))
        .await
        .expect("redis connect through the relay")
        .with_metrics(metrics.clone());
    let key = unique_key("outage");
    let limits = RateLimit {
        rpm: Some(10),
        ..rl()
    };

    // Healthy: the operation succeeds and nothing is counted.
    store
        .acquire(&key, &limits, "m-1")
        .await
        .expect("allowed while Redis is reachable");
    assert_eq!(counter_value(&metrics.render(), "ratelimit_acquire"), 0.0);

    relay.cut();

    // Still admitted — that is the fail-open contract, and exactly why
    // the outage is invisible without the counter.
    store
        .acquire(&key, &limits, "m-2")
        .await
        .expect("fail-open keeps admitting");
    let first = counter_value(&metrics.render(), "ratelimit_acquire");
    assert!(first >= 1.0, "the failed acquire must be counted: {first}");

    // Every subsequent failure counts, not just the first: the log line
    // is one-shot per outage and cannot report a rate.
    store
        .acquire(&key, &limits, "m-3")
        .await
        .expect("fail-open keeps admitting");
    assert!(
        counter_value(&metrics.render(), "ratelimit_acquire") > first,
        "each failed operation counts, not only the one that logged",
    );
}

/// A Redis that stops answering without closing the socket must degrade
/// the request, not hold it.
///
/// The store has always failed open on `Err`, but with no command budget
/// the `Err` never arrived: the command sat on TCP retransmission and the
/// caller — a live proxy request — hung with it. Reported against 1.2.0-rc.1
/// and reproduced identically on v1.1.0, where a `docker stop` of Redis
/// left every rate-limited request unanswered past three minutes.
///
/// Two properties, because each fails without the other: the first
/// command must return inside the budget, and the ones behind it must not
/// each pay that budget again for as long as the outage lasts.
#[tokio::test]
async fn a_silent_redis_fails_open_within_the_command_budget() {
    let Some(url) = redis_url() else {
        eprintln!("skipping: RATELIMIT_TEST_REDIS_URL not set");
        return;
    };
    let relay = RedisCutoff::start(&url).await;
    // Below the 5s default on purpose: a bound of 6s would pass whether or
    // not the per-block field reached the connection, so it would not be a
    // check at all.
    const BUDGET_SECS: u64 = 2;
    let cfg = RedisConnConfig {
        timeout_secs: BUDGET_SECS,
        ..single(&relay.url())
    };
    let store = RedisStore::connect(&cfg)
        .await
        .expect("redis connect through the relay");
    let key = unique_key("blackhole");
    let limits = RateLimit {
        rpm: Some(10),
        ..rl()
    };

    store
        .acquire(&key, &limits, "m-1")
        .await
        .expect("allowed while Redis answers");

    relay.blackhole();

    let started = std::time::Instant::now();
    store
        .acquire(&key, &limits, "m-2")
        .await
        .expect("fail-open still admits");
    let first = started.elapsed();
    assert!(
        first >= Duration::from_secs(BUDGET_SECS),
        "the budget is what makes it give up; a faster return means the \
         blackhole was not reached, took {first:?}",
    );
    assert!(
        first < Duration::from_millis(BUDGET_SECS * 1000 + 1_500),
        "the first silent command must give up on its own budget, not the \
         default, took {first:?}",
    );

    // Behind it the breaker is open, so this one costs nothing at all —
    // otherwise every request for the length of the outage carries the
    // full budget as added latency.
    let started = std::time::Instant::now();
    store
        .acquire(&key, &limits, "m-3")
        .await
        .expect("fail-open still admits");
    let second = started.elapsed();
    assert!(
        second < Duration::from_millis(500),
        "the command behind a failure must short-circuit, took {second:?}",
    );
}

/// A store whose connection never landed must BE the store an outage
/// produces — every operation failing open to the per-replica counters —
/// and it must take the shared backend over on its own once Redis
/// answers.
///
/// `connect_or_attach_later` is the only production constructor
/// (`RedisStore::connect` is now test-only), and the e2e that covers the
/// boot exercises `acquire` and `commit` alone. The other three
/// operations reach the same empty slot and are what a `Drop`, a
/// post-stream token add and the `x-ratelimit-*` headers run through.
#[tokio::test]
async fn a_store_that_never_connected_fails_open_and_attaches_later() {
    let Some(url) = redis_url() else { return };
    let relay = RedisCutoff::start(&url).await;
    // Silent from the first SYN: the store's own connect is what meets it.
    relay.blackhole();

    let mut cfg = single(&relay.url());
    cfg.timeout_secs = 1;
    let key = unique_key("boot-degraded");
    // rpm=1 is what the fallback must refuse on; the concurrency cap is
    // deliberately ABOVE the two requests below, or the concurrency gate
    // would refuse the second one first and the rpm window would never
    // be the thing under test.
    let limits = RateLimit {
        rpm: Some(1),
        concurrency: Some(2),
        ..Default::default()
    };

    let metrics = Metrics::new(false);
    let started = std::time::Instant::now();
    let (store, unreachable) = RedisStore::connect_or_attach_later(&cfg)
        .await
        .expect("an unreachable Redis is a diagnostic, not a fatal config error");
    let elapsed = started.elapsed();
    let store = store.with_metrics(metrics.clone());

    // The whole point: it returned, on the configured budget, with the
    // failure as a diagnostic rather than as an error to propagate.
    assert!(
        unreachable.is_some(),
        "an unreachable Redis must be reported"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "connect_or_attach_later spent {elapsed:?}, not the configured 1s budget"
    );

    // Fails OPEN, and the fallback still ENFORCES: the second request in
    // the same minute is refused by the local window, not admitted.
    store
        .acquire(&key, &limits, "boot-1")
        .await
        .expect("the local fallback admits the first request");
    let refused = store
        .acquire(&key, &limits, "boot-2")
        .await
        .expect_err("the local fallback still enforces rpm=1");
    assert!(matches!(
        refused,
        aisix_ratelimit::RateLimitError::Requests { .. }
    ));

    // The remaining three operations run against the same empty slot.
    // `peek` reports the local window rather than nothing at all.
    let status = store
        .peek(&key, &limits)
        .await
        .expect("peek falls back to the local window");
    assert_eq!(status.rpm_limit, Some(1));
    assert_eq!(status.rpm_used, 1);
    store.add_tokens(&key, 7);
    store.release(&key, "boot-1");

    // The degradation is counted, which is the only signal it produces
    // while no log level is raised.
    assert!(
        counter_value(&metrics.render(), "ratelimit_acquire") > 0.0,
        "a degraded acquire must be counted"
    );

    // Redis comes up. The background task attaches it with no restart.
    relay.heal();
    let peer = RedisStore::connect(&single(&relay.url()))
        .await
        .expect("a healed Redis connects");
    wait_for_attach(&store, &peer).await;

    // Discriminating: a SECOND store, connected normally to the same
    // Redis, is already over rpm=1 because of the request the first one
    // just made. While the first store was counting locally this could
    // not hold — the peer's window would be its own and empty.
    let shared_key = unique_key("boot-attached");
    store
        .acquire(&shared_key, &limits, "attached-1")
        .await
        .expect("the attached store admits the first request");
    peer.acquire(&shared_key, &limits, "peer-1")
        .await
        .expect_err("the attached store wrote to the SHARED counter, not to its local one");
}

/// Block until `store` has attached its shared backend, judged by an
/// effect only the shared backend can produce: a token add that `peer` —
/// a second store connected normally to the same Redis — can read back.
///
/// Not `peek`, and not a successful `acquire`: both of those answer from
/// the local fallback too, so a gate built on either falls through
/// immediately and leaves whatever follows it asserting against a store
/// that may still be degraded. (Both gates here were written that way
/// first; a mutation check caught it.)
async fn wait_for_attach(store: &RedisStore, peer: &RedisStore) {
    let probe = unique_key("attach-probe");
    let limits = RateLimit {
        tpm: Some(1_000_000),
        ..Default::default()
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        store.add_tokens(&probe, 1);
        tokio::time::sleep(Duration::from_millis(200)).await;
        if peer
            .peek(&probe, &limits)
            .await
            .is_some_and(|s| s.tpm_used > 0)
        {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the shared backend never attached"
        );
    }
}

/// A reservation taken on the local fallback and committed after the
/// shared backend attached must give its local concurrency slot back.
///
/// `LocalStore` has no ttl on `in_flight`, so a slot left behind is left
/// behind for the life of the process: the local baseline for that bucket
/// is permanently raised, and a LATER Redis outage then refuses traffic
/// on a path whose entire contract is to fail open. The boot-degraded
/// state makes this reachable on every cold start that precedes its
/// Redis, where before it needed the exact instant a breaker closed.
#[tokio::test]
async fn a_commit_after_the_backend_attaches_returns_the_local_slot() {
    let Some(url) = redis_url() else { return };
    let relay = RedisCutoff::start(&url).await;
    relay.blackhole();

    let mut cfg = single(&relay.url());
    cfg.timeout_secs = 1;
    let key = unique_key("boot-handover");
    // concurrency=1 and nothing else, so the only thing that can refuse
    // the probe at the end is a leaked slot.
    let limits = RateLimit {
        concurrency: Some(1),
        ..Default::default()
    };

    let (store, unreachable) = RedisStore::connect_or_attach_later(&cfg)
        .await
        .expect("an unreachable Redis is a diagnostic, not a fatal config error");
    assert!(unreachable.is_some());

    // Slot taken locally, while there is no connection.
    store
        .acquire(&key, &limits, "handover-1")
        .await
        .expect("the local fallback admits it");

    relay.heal();
    let peer = RedisStore::connect(&single(&relay.url()))
        .await
        .expect("a healed Redis connects");
    wait_for_attach(&store, &peer).await;

    // Committed against the shared backend, which never held the member.
    store.commit(&key, 0, "handover-1").await;

    // Now force the store back onto the local counters and check the slot
    // came back. Without the release in `commit` this refuses.
    relay.blackhole();
    store
        .acquire(&key, &limits, "handover-2")
        .await
        .expect("the local concurrency slot was returned on commit");
}

/// A config the process rejects BY ITSELF must still end the boot.
///
/// Everything that depends on a server's answer is degraded around and
/// retried in the background, because a server can come back and a
/// credential can be corrected on it. A `url` the driver cannot parse
/// can do neither: nothing outside this process will ever change it, so
/// a gateway that started would be quietly never going to enforce a
/// shared limit, which is strictly worse than the boot failure.
#[tokio::test]
async fn a_config_the_driver_cannot_use_is_still_fatal() {
    let cfg = RedisConnConfig {
        mode: RedisMode::Single,
        url: Some("not-a-redis-url".into()),
        ..Default::default()
    };
    let err = RedisStore::connect_or_attach_later(&cfg)
        .await
        .expect_err("a malformed url must not be degraded around");
    assert!(aisix_redis::is_boot_fatal(&err), "{err:?}");
}

/// A refused credential is NOT fatal — and it is not an outage either.
///
/// The server answered, so the gateway serves degraded and keeps
/// retrying (the credential may be fixed on the server side), but the
/// operator is told a refusal rather than sent to look at the network.
/// Those are two separate claims and this pins both.
///
/// Synthetic errors, so this pins the CLASSIFIER and nothing else. It
/// does not prove a real Redis refusal ever reaches it, and for a while
/// none did — every driver reported a refusal as a connectivity failure,
/// so the refused branch was unreachable in production while a test like
/// this stayed green. `crates/aisix-redis/tests/auth_connect.rs` is what
/// answers that question, against a live server.
#[test]
fn a_refused_credential_is_told_apart_from_an_outage() {
    let refused = redis::RedisError::from((
        redis::ErrorKind::AuthenticationFailed,
        "WRONGPASS invalid username-password pair",
    ));
    assert!(!aisix_redis::is_boot_fatal(&refused));
    assert_eq!(aisix_redis::failure_reason(&refused), "refused");

    let unreachable = redis::RedisError::from((
        redis::ErrorKind::IoError,
        "redis connect timed out",
        "no connection within 5s".to_string(),
    ));
    assert!(!aisix_redis::is_boot_fatal(&unreachable));
    assert_eq!(aisix_redis::failure_reason(&unreachable), "unreachable");
}

/// The whole point of degrading rather than exiting: the credential is
/// corrected ON THE SERVER, and the running gateway picks it up.
///
/// An ACL user rather than `requirepass`, because `requirepass` is
/// server-wide and every other test in this file shares the server. The
/// user is created with one password, the store is configured with
/// another — so the first connect is refused, exactly as a typo'd
/// `AISIX_RATELIMIT__REDIS__PASSWORD` would be — and then the user's
/// password is changed to the configured one with the gateway still
/// running.
#[tokio::test]
async fn a_refused_credential_corrected_on_the_server_is_adopted_without_a_restart() {
    let Some(url) = redis_url() else { return };
    let user = format!("aisix-heal-{}", unique_key("u").replace(':', "-"));
    let wanted = "the-password-the-gateway-holds";

    let admin = aisix_redis::connect(&single(&url))
        .await
        .expect("the admin connection must come up");
    let set_user = |password: &str| {
        let (user, password) = (user.clone(), password.to_string());
        let admin = &admin;
        async move {
            let mut handle = admin.acquire().await.expect("an admin handle");
            redis::cmd("ACL")
                .arg("SETUSER")
                .arg(&user)
                .arg("on")
                .arg(format!(">{password}"))
                .arg("~*")
                .arg("+@all")
                .query_async::<()>(&mut handle)
                .await
                .expect("the ACL user must be settable");
        }
    };
    set_user("not-the-password-the-gateway-holds").await;

    let cfg = RedisConnConfig {
        username: Some(user.clone()),
        password: Some(wanted.into()),
        ..single(&url)
    };
    let (store, degraded) = RedisStore::connect_or_attach_later(&cfg)
        .await
        .expect("a refused credential must NOT end the boot");
    let degraded = degraded.expect("a refused credential must degrade, not connect");
    assert_eq!(
        aisix_redis::failure_reason(&degraded),
        "refused",
        "the operator must be told the server refused, not that it is unreachable: {degraded}"
    );

    // Still enforcing, per replica: a degraded limiter that had stopped
    // counting would serve both of these.
    let key = unique_key("heal");
    let limits = RateLimit {
        rpm: Some(1),
        ..Default::default()
    };
    assert!(store.acquire(&key, &limits, "heal-1").await.is_ok());
    assert!(
        store.acquire(&key, &limits, "heal-2").await.is_err(),
        "per-replica enforcement must hold while the shared backend is refused"
    );

    // The operator fixes it on the server. Nothing restarts.
    set_user(wanted).await;
    let peer = RedisStore::connect(&cfg)
        .await
        .expect("the corrected credential must connect");
    wait_for_attach(&store, &peer).await;

    // The Redis is shared with every other test here and outlives the
    // run, so the user this test invented does not get to accumulate on
    // it one password at a time.
    let mut handle = admin.acquire().await.expect("an admin handle");
    let _ = redis::cmd("ACL")
        .arg("DELUSER")
        .arg(&user)
        .query_async::<()>(&mut handle)
        .await;
}
