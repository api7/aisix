//! A watch the server cancels while keeping the gRPC stream open must end
//! the gateway's watch, or the gateway keeps a dead watch and applies no
//! further configuration. kine sends exactly that (`Canceled`, compact
//! revision 0, no reason) when it drops a watcher whose consumer fell
//! behind.
//!
//! Real etcd never cancels an established watch without a compact
//! revision, so these tests speak gRPC over real HTTP/2 from a fake etcd
//! with hand-encoded `WatchResponse` frames.
//!
//! A test binary of its own: the WARN assertion installs a GLOBAL
//! subscriber, because a scoped one loses events to tracing's per-callsite
//! interest cache when another test in the same binary reaches the same
//! callsite with no subscriber first.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use aisix_etcd::{ConfigProvider, EtcdConfigProvider, ProviderError, Supervisor, WatchEvent};
use futures::StreamExt;

#[derive(Clone)]
struct SharedBuf(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl tracing_subscriber::fmt::MakeWriter<'_> for SharedBuf {
    type Writer = Self;
    fn make_writer(&self) -> Self {
        self.clone()
    }
}

fn captured_logs() -> Arc<Mutex<Vec<u8>>> {
    static LOGS: OnceLock<Arc<Mutex<Vec<u8>>>> = OnceLock::new();
    LOGS.get_or_init(|| {
        let logs = Arc::new(Mutex::new(Vec::new()));
        tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(SharedBuf(logs.clone()))
            .init();
        logs
    })
    .clone()
}

fn varint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

/// One gRPC-framed `WatchResponse`, encoded by hand: header
/// `{revision: 1}`, then fields 2-6 as etcd's `rpc.proto` numbers them.
fn watch_response_frame(
    watch_id: i64,
    created: bool,
    canceled: bool,
    compact_revision: i64,
    reason: &str,
) -> Vec<u8> {
    let mut msg = vec![0x0a, 2, 0x18, 1];
    msg.push(0x10);
    varint(&mut msg, watch_id as u64);
    if created {
        msg.extend([0x18, 1]);
    }
    if canceled {
        msg.extend([0x20, 1]);
    }
    if compact_revision != 0 {
        msg.push(0x28);
        varint(&mut msg, compact_revision as u64);
    }
    if !reason.is_empty() {
        msg.push(0x32);
        varint(&mut msg, reason.len() as u64);
        msg.extend(reason.as_bytes());
    }
    let mut frame = vec![0];
    frame.extend((msg.len() as u32).to_be_bytes());
    frame.extend(msg);
    frame
}

/// How the fake etcd answers each watch create.
#[derive(Clone, Copy)]
enum WatchAnswer {
    /// Confirms the watch, then cancels it on a stream it keeps open —
    /// what kine sends a watcher it dropped as a slow consumer.
    CreatedThenCancelled {
        compact_revision: i64,
        reason: &'static str,
    },
    /// Refuses the watch in the create response itself.
    RefusedAtCreate,
}

/// An etcd over real HTTP/2 whose Range answers revision 1 with no
/// keys and whose Watch answers per `answer`, holding every watch
/// stream open afterwards. Returns the endpoint and the counts of
/// Range and Watch calls.
async fn spawn_cancelling_etcd(
    answer: WatchAnswer,
) -> (
    String,
    Arc<std::sync::atomic::AtomicUsize>,
    Arc<std::sync::atomic::AtomicUsize>,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let ranges = Arc::new(AtomicUsize::new(0));
    let watches = Arc::new(AtomicUsize::new(0));
    let (r, w) = (ranges.clone(), watches.clone());
    tokio::spawn(async move {
        while let Ok((socket, _)) = listener.accept().await {
            let (r, w) = (r.clone(), w.clone());
            tokio::spawn(async move {
                let Ok(mut conn) = h2::server::handshake(socket).await else {
                    return;
                };
                let mut held = Vec::new();
                while let Some(Ok((request, mut respond))) = conn.accept().await {
                    let response = http::Response::builder()
                        .status(200)
                        .header("content-type", "application/grpc")
                        .body(())
                        .unwrap();
                    let mut body = respond.send_response(response, false).unwrap();
                    if request.uri().path() == "/etcdserverpb.KV/Range" {
                        r.fetch_add(1, Ordering::SeqCst);
                        body.send_data(vec![0, 0, 0, 0, 4, 0x0a, 2, 0x18, 1].into(), false)
                            .unwrap();
                        let mut trailers = http::HeaderMap::new();
                        trailers.insert("grpc-status", http::HeaderValue::from_static("0"));
                        body.send_trailers(trailers).unwrap();
                        continue;
                    }
                    assert_eq!(request.uri().path(), "/etcdserverpb.Watch/Watch");
                    w.fetch_add(1, Ordering::SeqCst);
                    match answer {
                        WatchAnswer::CreatedThenCancelled {
                            compact_revision,
                            reason,
                        } => {
                            body.send_data(
                                watch_response_frame(0, true, false, 0, "").into(),
                                false,
                            )
                            .unwrap();
                            body.send_data(
                                watch_response_frame(0, false, true, compact_revision, reason)
                                    .into(),
                                false,
                            )
                            .unwrap();
                        }
                        WatchAnswer::RefusedAtCreate => {
                            body.send_data(
                                watch_response_frame(-1, true, true, 0, "permission denied").into(),
                                false,
                            )
                            .unwrap();
                        }
                    }
                    // Open, like the real one: the cancel is the only
                    // sign the watch is over.
                    held.push((request, body));
                }
            });
        }
    });
    (format!("http://{addr}"), ranges, watches)
}

async fn next_watch_item(answer: WatchAnswer) -> Result<WatchEvent, ProviderError> {
    let (endpoint, _, _) = spawn_cancelling_etcd(answer).await;
    let provider = EtcdConfigProvider::connect(&[endpoint], "/aisix", None, None, None)
        .await
        .unwrap();
    let mut stream = provider.watch(2).await.expect("the watch is confirmed");
    tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("a cancelled watch must end its stream; before the fix it idled forever")
        .expect("the cancel is reported as an item, not a silent end")
}

#[tokio::test]
async fn a_watch_cancelled_without_a_reason_ends_as_a_watch_error() {
    let err = next_watch_item(WatchAnswer::CreatedThenCancelled {
        compact_revision: 0,
        reason: "",
    })
    .await
    .unwrap_err();
    let ProviderError::Watch(msg) = err else {
        panic!("a plain cancel is a watch failure, got {err:?}");
    };
    assert!(
        msg.contains("etcd cancelled the watch: no reason given"),
        "{msg}"
    );
}

#[tokio::test]
async fn a_watch_cancelled_with_a_reason_carries_it() {
    let err = next_watch_item(WatchAnswer::CreatedThenCancelled {
        compact_revision: 0,
        reason: "watch closed by server",
    })
    .await
    .unwrap_err();
    let ProviderError::Watch(msg) = err else {
        panic!("a plain cancel is a watch failure, got {err:?}");
    };
    assert!(msg.contains("watch closed by server"), "{msg}");
}

#[tokio::test]
async fn a_cancel_that_reports_compaction_is_compaction() {
    // With the compact revision set — what etcd and kine both send.
    let err = next_watch_item(WatchAnswer::CreatedThenCancelled {
        compact_revision: 5,
        reason: "mvcc: required revision has been compacted",
    })
    .await
    .unwrap_err();
    assert!(matches!(err, ProviderError::Compacted), "{err:?}");
    // And with only the reason to go on.
    let err = next_watch_item(WatchAnswer::CreatedThenCancelled {
        compact_revision: 0,
        reason: "etcdserver: mvcc: required revision has been compacted",
    })
    .await
    .unwrap_err();
    assert!(matches!(err, ProviderError::Compacted), "{err:?}");
}

#[tokio::test]
async fn a_watch_refused_at_create_is_a_watch_error() {
    let (endpoint, _, _) = spawn_cancelling_etcd(WatchAnswer::RefusedAtCreate).await;
    let provider = EtcdConfigProvider::connect(&[endpoint], "/aisix", None, None, None)
        .await
        .unwrap();
    let err = tokio::time::timeout(Duration::from_secs(5), provider.watch(2))
        .await
        .expect("the create response arrives at once")
        .err()
        .expect("a refused create cannot hand back a live watch");
    let ProviderError::Watch(msg) = err else {
        panic!("a refused create is a watch failure, got {err:?}");
    };
    assert!(
        msg.contains("cancelled the watch as it was created"),
        "{msg}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_supervisor_rereads_after_a_cancel_and_backs_off_while_it_repeats() {
    use std::sync::atomic::Ordering;

    let logs = captured_logs();
    // Every watch is dropped the way kine drops a slow consumer.
    let (endpoint, ranges, watches) = spawn_cancelling_etcd(WatchAnswer::CreatedThenCancelled {
        compact_revision: 0,
        reason: "",
    })
    .await;
    let provider = Arc::new(
        EtcdConfigProvider::connect(&[endpoint], "/aisix", None, None, None)
            .await
            .unwrap(),
    );
    let supervisor = Arc::new(Supervisor::new(provider, "/aisix"));
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
    let run = tokio::spawn(supervisor.clone().run(cancel_rx));

    // Recovery: the next cycle re-reads everything and watches again.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while ranges.load(Ordering::SeqCst) < 2 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "a cancelled watch must lead to a fresh read; before the fix the \
             gateway kept the dead watch and never read again",
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // A server that keeps cancelling is retried on the backoff (1s,
    // then 2s), not every 100ms: within 2.5s of the first watch there
    // are at most three.
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    let watched = watches.load(Ordering::SeqCst);
    assert!(
        watched <= 3,
        "cancel loop is not backed off: {watched} watches"
    );

    cancel_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .unwrap()
        .unwrap();
    let logs = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
    assert!(
        logs.lines()
            .any(|l| l.contains("WARN") && l.contains("etcd cancelled the watch: no reason given")),
        "the cancel and its reason are logged at WARN:\n{logs}",
    );
}
