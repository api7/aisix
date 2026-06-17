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

    // Force a failover; the old master is demoted to a (read-only) replica.
    {
        let client = redis::Client::open(sentinel_url.as_str()).expect("sentinel client");
        let mut s = client
            .get_multiplexed_async_connection()
            .await
            .expect("sentinel conn");
        let _: redis::Value = redis::cmd("SENTINEL")
            .arg("FAILOVER")
            .arg(&master)
            .query_async(&mut s)
            .await
            .expect("trigger failover");
    }

    // Wait until Sentinel reports a different master port (promotion done).
    let mut promoted = false;
    for _ in 0..60 {
        let (_, port_now) = master_addr(&sentinel_url, &master).await;
        if port_now != port_before {
            promoted = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    assert!(promoted, "sentinel did not promote a new master in time");

    // The cached connection now points at the demoted (read-only) old
    // master. note_error() drops it so acquire re-resolves the new master.
    conn.note_error().await;

    // A write must succeed again — proving acquire followed the promotion.
    // Retry briefly: the freshly promoted master may take a moment to
    // accept writes after role change.
    let mut wrote = false;
    for _ in 0..40 {
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
