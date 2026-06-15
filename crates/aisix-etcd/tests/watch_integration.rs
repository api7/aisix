//! Integration test: watch stream stays alive and delivers events.
//!
//! Requires a real etcd — gated by `ETCD_TEST_URL`.  CI sets this via
//! the etcd service container; local runs without the var are no-ops.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aisix_etcd::provider::ConfigProvider;
use aisix_etcd::EtcdConfigProvider;
use etcd_client::Client;
use futures::StreamExt;
use tokio::time::timeout;

fn etcd_url() -> Option<String> {
    std::env::var("ETCD_TEST_URL").ok()
}

fn unique_prefix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("/aisix-etcd-it/{nanos:x}")
}

/// The core regression test for issue #237: after `watch()` returns,
/// the stream must stay open and deliver events.  Before the fix the
/// `Watcher` was dropped immediately, closing the gRPC client→server
/// half and causing an instant EOF.
#[tokio::test]
async fn watch_stream_delivers_events_after_put() {
    let url = match etcd_url() {
        Some(u) => u,
        None => {
            eprintln!("ETCD_TEST_URL not set — skipping");
            return;
        }
    };

    let prefix = unique_prefix();
    let endpoints = vec![url.clone()];
    let provider = EtcdConfigProvider::connect(&endpoints, prefix.clone(), None)
        .await
        .expect("connect");

    let (_entries, revision) = provider.load_all().await.expect("load_all");

    let mut stream = provider.watch(revision + 1).await.expect("watch");

    // Give the runtime a moment — with the old bug the gRPC stream
    // would close asynchronously once the Watcher was dropped.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Write a key via a separate client so the watch should see it.
    let mut writer = Client::connect([url.as_str()], None)
        .await
        .expect("writer connect");
    let test_key = format!("{prefix}/models/watch-test");
    let test_value = br#"{"display_name":"t","provider":"openai","model_name":"gpt-4o","provider_key_id":"11111111-1111-1111-1111-111111111111"}"#;
    writer
        .put(test_key.as_bytes(), test_value.as_ref(), None)
        .await
        .expect("put");

    // The stream must deliver the event within a reasonable window.
    let event = timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timed out — watch stream likely dead (watcher dropped?)")
        .expect("stream ended unexpectedly");

    match event.expect("watch error") {
        aisix_etcd::WatchEvent::Put(entry) => {
            assert_eq!(entry.key, test_key);
            assert_eq!(entry.value, test_value);
        }
        other => panic!("expected Put, got {other:?}"),
    }

    // Cleanup
    writer
        .delete(test_key.as_bytes(), None)
        .await
        .expect("cleanup delete");
}

/// #519 B.3: the supervisor's applied revision (read by the heartbeat as
/// `applied_revision`) must catch up to the header revision returned to a
/// writer — for puts AND deletes. This is the exact comparison cp-api
/// performs: it writes through kine, records the response revision W, and
/// treats a DP with `applied_revision >= W` as caught up.
#[tokio::test]
async fn supervisor_applied_revision_catches_up_to_writer_revision() {
    let url = match etcd_url() {
        Some(u) => u,
        None => {
            eprintln!("ETCD_TEST_URL not set — skipping");
            return;
        }
    };

    let prefix = unique_prefix();
    let endpoints = vec![url.clone()];
    let provider = EtcdConfigProvider::connect(&endpoints, prefix.clone(), None)
        .await
        .expect("connect");
    let supervisor = std::sync::Arc::new(aisix_etcd::Supervisor::new(
        std::sync::Arc::new(provider),
        prefix.clone(),
    ));
    let status = supervisor.watch_status();
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
    let run = tokio::spawn(supervisor.clone().run(cancel_rx));

    let mut writer = Client::connect([url.as_str()], None)
        .await
        .expect("writer connect");

    async fn wait_for_revision(status: &aisix_etcd::WatchStatus, want: i64, what: &str) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let got = status.snapshot().revision;
            if got >= want {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "applied revision {got} did not reach {what} revision {want} within 5s",
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    // Put: the writer's response header revision is what cp-api records
    // as W; the supervisor must report >= W once the event is applied.
    let test_key = format!("{prefix}/models/rev-test");
    let test_value = br#"{"display_name":"t","provider":"openai","model_name":"gpt-4o","provider_key_id":"11111111-1111-1111-1111-111111111111"}"#;
    let put_resp = writer
        .put(test_key.as_bytes(), test_value.as_ref(), None)
        .await
        .expect("put");
    let put_rev = put_resp.header().expect("put header").revision();
    wait_for_revision(&status, put_rev, "put").await;

    // Delete: same contract. Before the #519 B.3 fix the Delete arm
    // dropped the event's mod_revision, so applied_revision stalled here.
    let del_resp = writer
        .delete(test_key.as_bytes(), None)
        .await
        .expect("delete");
    let del_rev = del_resp.header().expect("delete header").revision();
    wait_for_revision(&status, del_rev, "delete").await;

    let _ = cancel_tx.send(true);
    let _ = run.await;
}

/// Regression test: an etcd transaction writes multiple keys atomically in a
/// single revision, producing a single WatchResponse with multiple events.
/// Before the fix, only the first event was emitted and the rest were dropped.
#[tokio::test]
async fn watch_stream_delivers_all_events_from_batched_response() {
    let url = match etcd_url() {
        Some(u) => u,
        None => {
            eprintln!("ETCD_TEST_URL not set — skipping");
            return;
        }
    };

    let prefix = unique_prefix();
    let endpoints = vec![url.clone()];
    let provider = EtcdConfigProvider::connect(&endpoints, prefix.clone(), None)
        .await
        .expect("connect");

    let (_entries, revision) = provider.load_all().await.expect("load_all");
    let mut stream = provider.watch(revision + 1).await.expect("watch");
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut writer = Client::connect([url.as_str()], None)
        .await
        .expect("writer connect");

    // Write 4 keys in a single transaction so etcd emits one multi-event
    // WatchResponse. This is the exact scenario the buffer fix addresses.
    let keys: Vec<String> = (0..4).map(|i| format!("{prefix}/batch/{i}")).collect();
    let txn = etcd_client::Txn::new().and_then(
        keys.iter()
            .map(|k| etcd_client::TxnOp::put(k.as_bytes(), b"v", None))
            .collect::<Vec<_>>(),
    );
    writer.txn(txn).await.expect("txn");

    // All 4 events must arrive.
    let mut received = Vec::new();
    for _ in 0..4 {
        let ev = timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("timed out waiting for batched event")
            .expect("stream ended early");
        match ev.expect("watch error") {
            aisix_etcd::WatchEvent::Put(entry) => received.push(entry.key),
            other => panic!("expected Put, got {other:?}"),
        }
    }
    received.sort();
    let mut expected = keys.clone();
    expected.sort();
    assert_eq!(received, expected, "all batched events must be delivered");

    // Cleanup
    for key in &keys {
        writer
            .delete(key.as_bytes(), None)
            .await
            .expect("cleanup delete");
    }
}
