//! Sentinel failover / reconnect coverage for [`aisix_redis::RedisConn`].
//!
//! Runs only when `REDIS_FAILOVER_SENTINELS` and `REDIS_FAILOVER_MASTER`
//! are set, pointing at a master + replica + sentinel topology (CI brings
//! one up). The test forces a Sentinel failover, then proves that after
//! [`RedisConn::note_error`] the next [`RedisConn::acquire`] re-resolves
//! the *new* master — a write (rejected by a read-only replica) succeeds
//! only if the connection followed the promotion.

use std::time::Duration;

use aisix_core::{RedisConnConfig, RedisMode};
use aisix_redis::{connect, RedisConn};
use redis::AsyncCommands;

fn sentinel_cfg() -> Option<(RedisConnConfig, String, String)> {
    let sentinels = std::env::var("REDIS_FAILOVER_SENTINELS").ok()?;
    let master = std::env::var("REDIS_FAILOVER_MASTER").ok()?;
    let first_sentinel = sentinels.split(',').next().unwrap().trim().to_string();
    let cfg = RedisConnConfig {
        mode: RedisMode::Sentinel,
        sentinels: sentinels.split(',').map(|s| s.trim().to_string()).collect(),
        master_name: Some(master.clone()),
        // Optional: when the master is password-protected this exercises
        // the ACL auth path to the Sentinel-discovered master too.
        username: std::env::var("REDIS_FAILOVER_USERNAME").ok(),
        password: std::env::var("REDIS_FAILOVER_PASSWORD").ok(),
        ..Default::default()
    };
    Some((cfg, first_sentinel, master))
}

async fn master_addr(sentinel_url: &str, master: &str) -> (String, String) {
    let client = redis::Client::open(sentinel_url).expect("sentinel client");
    let mut conn = client
        .get_multiplexed_async_connection()
        .await
        .expect("sentinel conn");
    let addr: Vec<String> = redis::cmd("SENTINEL")
        .arg("get-master-addr-by-name")
        .arg(master)
        .query_async(&mut conn)
        .await
        .expect("get-master-addr");
    (addr[0].clone(), addr[1].clone())
}

async fn sentinel_conn(sentinel_url: &str) -> redis::aio::MultiplexedConnection {
    redis::Client::open(sentinel_url)
        .expect("sentinel client")
        .get_multiplexed_async_connection()
        .await
        .expect("sentinel conn")
}

/// Whether the sentinel is in TILT mode. A sentinel that finds its own
/// timer ran more than two seconds late — a runner that descheduled it
/// under load — stops acting for thirty seconds, failovers included, so
/// a failover it accepted does not start until TILT ends.
async fn sentinel_tilted(sentinel_url: &str) -> bool {
    let info: String = redis::cmd("INFO")
        .arg("sentinel")
        .query_async(&mut sentinel_conn(sentinel_url).await)
        .await
        .expect("INFO sentinel");
    info.lines().any(|l| l.trim() == "sentinel_tilt:1")
}

/// Sentinel's own view of the failover, for the message of a test that
/// gave up on it.
async fn sentinel_state(sentinel_url: &str, master: &str) -> String {
    let mut conn = sentinel_conn(sentinel_url).await;
    let info: String = redis::cmd("INFO")
        .arg("sentinel")
        .query_async(&mut conn)
        .await
        .unwrap_or_else(|e| format!("INFO sentinel failed: {e}"));
    let fields: Vec<String> = redis::cmd("SENTINEL")
        .arg("MASTER")
        .arg(master)
        .query_async(&mut conn)
        .await
        .unwrap_or_else(|e| vec![format!("SENTINEL MASTER failed: {e}")]);
    let replicas: redis::Value = redis::cmd("SENTINEL")
        .arg("REPLICAS")
        .arg(master)
        .query_async(&mut conn)
        .await
        .unwrap_or_else(|e| redis::Value::SimpleString(format!("SENTINEL REPLICAS failed: {e}")));
    format!(
        "{}\nmaster: {fields:?}\nreplicas: {replicas:?}",
        info.trim()
    )
}

/// Whether Sentinel is running a failover of `master` right now.
async fn failover_in_progress(sentinel_url: &str, master: &str) -> bool {
    let fields: Vec<String> = redis::cmd("SENTINEL")
        .arg("MASTER")
        .arg(master)
        .query_async(&mut sentinel_conn(sentinel_url).await)
        .await
        .expect("SENTINEL MASTER");
    fields.chunks(2).any(|kv| {
        kv[0] == "flags"
            && kv
                .get(1)
                .is_some_and(|f| f.contains("failover_in_progress"))
    })
}

/// Ask Sentinel for a failover; what it answered, for the failure message.
/// A refusal is not fatal: the caller asks again until the deadline.
async fn request_failover(sentinel_url: &str, master: &str) -> String {
    redis::cmd("SENTINEL")
        .arg("FAILOVER")
        .arg(master)
        .query_async::<redis::Value>(&mut sentinel_conn(sentinel_url).await)
        .await
        .map_or_else(|e| e.to_string(), |v| format!("{v:?}"))
}

/// How long TILT lasts once entered (Sentinel's `SENTINEL_TILT_PERIOD`).
const TILT_PERIOD: Duration = Duration::from_secs(30);

#[tokio::test]
async fn sentinel_reresolves_master_after_failover() {
    let Some((cfg, sentinel_url, master)) = sentinel_cfg() else {
        eprintln!("skipping: REDIS_FAILOVER_SENTINELS / _MASTER not set");
        return;
    };

    let conn: RedisConn = connect(&cfg).await.expect("sentinel connect");

    // Write succeeds against the current master.
    let mut handle = conn.acquire().await.expect("acquire");
    let _: () = handle
        .set("aisix:failover:probe", "before")
        .await
        .expect("write to initial master");

    let (_, port_before) = master_addr(&sentinel_url, &master).await;

    // Force a failover; the old master is demoted to a (read-only) replica,
    // and wait until Sentinel reports a different master port.
    //
    // A failover takes a few seconds, but Sentinel only runs it outside
    // TILT, and a loaded runner can tip it into TILT at any moment —
    // before the failover is requested or halfway through. Sentinel then
    // accepts the request and does nothing for TILT_PERIOD; a failover it
    // had already started is past its own `failover-timeout` by then and
    // is aborted. That is Sentinel's safety rule, not something this test
    // or the client can shorten, so the wait is extended by a TILT period
    // whenever one is observed, and a failover Sentinel is no longer
    // running is requested again.
    let mut deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut tilt_seen_until = tokio::time::Instant::now();
    let mut requests = Vec::new();
    let mut promoted = false;
    while tokio::time::Instant::now() < deadline {
        let (_, port_now) = master_addr(&sentinel_url, &master).await;
        if port_now != port_before {
            promoted = true;
            break;
        }
        let now = tokio::time::Instant::now();
        if now >= tilt_seen_until && sentinel_tilted(&sentinel_url).await {
            tilt_seen_until = now + TILT_PERIOD;
            deadline = deadline.max(tilt_seen_until + Duration::from_secs(30));
        }
        // The port is read again after the flags: a failover that ended
        // between the first read and the flags has already switched it,
        // and asking again would fail straight back.
        if now >= tilt_seen_until
            && !failover_in_progress(&sentinel_url, &master).await
            && master_addr(&sentinel_url, &master).await.1 == port_before
        {
            requests.push(request_failover(&sentinel_url, &master).await);
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(
        promoted,
        "sentinel did not promote a new master in time; failover requests: {requests:?}\n\
         sentinel state:\n{}",
        sentinel_state(&sentinel_url, &master).await,
    );

    // The cached connection now points at the demoted (read-only) old
    // master. note_error() drops it so acquire re-resolves the new master.
    conn.note_error().await;

    // A write must succeed again — proving acquire followed the promotion.
    // Retry briefly: the freshly promoted master may take a moment to
    // accept writes after role change.
    //
    // The budget is deliberately much longer than the role change needs.
    // Nothing above has failed a command through THIS connection — the
    // failover and the promotion poll both use their own clients, and
    // `note_error` only drops the cached master — so the cool-off is
    // normally closed here and the first iteration re-resolves straight
    // away. What the budget is for is the case where it is not: if a write
    // below lands on a socket the failover dropped, that is an I/O error,
    // the cool-off opens, and nothing gets through for `BREAKER_WINDOW`
    // (30s). Sized to outlast one such window rather than to race it.
    let mut wrote = false;
    for _ in 0..200 {
        // acquire itself can fail transiently right after a failover (the
        // sentinel master lookup may briefly error); retry rather than
        // abort, since absorbing that instability is the loop's whole job.
        let Ok(mut handle) = conn.acquire().await else {
            conn.note_error().await;
            tokio::time::sleep(Duration::from_millis(250)).await;
            continue;
        };
        if handle
            .set::<_, _, ()>("aisix:failover:probe", "after")
            .await
            .is_ok()
        {
            wrote = true;
            break;
        }
        conn.note_error().await;
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    assert!(
        wrote,
        "write must succeed against the re-resolved master after failover"
    );
}
