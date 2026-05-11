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
