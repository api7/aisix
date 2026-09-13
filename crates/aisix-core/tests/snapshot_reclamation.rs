use aisix_core::SnapshotHandle;
use std::sync::{mpsc, Arc};
use std::thread::{self, ThreadId};
use std::time::Duration;

#[derive(Clone)]
struct Observed {
    revision: usize,
    dropped: mpsc::Sender<(usize, ThreadId)>,
}

impl Drop for Observed {
    fn drop(&mut self) {
        let _ = self.dropped.send((self.revision, thread::current().id()));
    }
}

fn reclaim_after_last_reader(use_rcu: bool) {
    let (dropped, events) = mpsc::channel();
    let handle = SnapshotHandle::new(Observed {
        revision: 0,
        dropped,
    });
    let old = handle.load();
    let weak = Arc::downgrade(&old);
    if use_rcu {
        handle.rcu(|current| Observed {
            revision: current.revision + 1,
            dropped: current.dropped.clone(),
        });
    } else {
        handle.store(Observed {
            revision: 1,
            dropped: old.dropped.clone(),
        });
    }
    assert_eq!(handle.version(), 1);
    assert_eq!(handle.load().revision, 1);
    assert_eq!(old.revision, 0);
    assert_eq!(events.try_recv(), Err(mpsc::TryRecvError::Empty));
    drop(old);

    let (revision, destructor_thread) = events.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(revision, 0);
    assert_ne!(destructor_thread, thread::current().id());
    assert!(weak.upgrade().is_none());
    assert_eq!(events.try_recv(), Err(mpsc::TryRecvError::Empty));
}

#[test]
fn store_reclaims_after_last_reader_without_another_write() {
    reclaim_after_last_reader(false);
}

#[test]
fn rcu_reclaims_after_last_reader_without_another_write() {
    reclaim_after_last_reader(true);
}

#[test]
fn a_long_reader_does_not_hold_up_reclaiming_newer_snapshots() {
    let (dropped, events) = mpsc::channel();
    let handle = SnapshotHandle::new(Observed {
        revision: 0,
        dropped: dropped.clone(),
    });
    let long = handle.load();
    handle.store(Observed {
        revision: 1,
        dropped: dropped.clone(),
    });
    let short = handle.load();
    handle.rcu(|current| Observed {
        revision: current.revision + 1,
        dropped: current.dropped.clone(),
    });
    drop(short);
    let (revision, worker) = events.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(revision, 1);
    assert_ne!(worker, thread::current().id());
    assert_eq!(long.revision, 0);
    drop(long);
    assert_eq!(
        events.recv_timeout(Duration::from_secs(2)).unwrap(),
        (0, worker)
    );
    assert_eq!(handle.load().revision, 2);
}

#[tokio::test(flavor = "current_thread")]
async fn cancelling_a_reader_reclaims_its_snapshot_off_the_worker() {
    let (dropped, events) = mpsc::channel();
    let handle = SnapshotHandle::new(Observed {
        revision: 0,
        dropped: dropped.clone(),
    });
    let reader = handle.load();
    let weak = Arc::downgrade(&reader);
    let (started, wait_started) = tokio::sync::oneshot::channel();
    let request = tokio::spawn(async move {
        let _reader = reader;
        started.send(()).unwrap();
        std::future::pending::<()>().await;
    });
    wait_started.await.unwrap();
    handle.store(Observed {
        revision: 1,
        dropped,
    });
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    let (revision, destructor_thread) = events.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(revision, 0);
    assert_ne!(destructor_thread, thread::current().id());
    assert!(weak.upgrade().is_none());
}

struct SlowDrop {
    started: Option<mpsc::Sender<()>>,
    release: std::sync::Mutex<mpsc::Receiver<()>>,
}

impl Drop for SlowDrop {
    fn drop(&mut self) {
        if let Some(started) = self.started.take() {
            started.send(()).unwrap();
            self.release.get_mut().unwrap().recv().unwrap();
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn last_reader_drop_keeps_tcp_worker_available() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let (started, wait_started) = mpsc::channel();
    let (release, wait_release) = mpsc::channel();
    let handle = SnapshotHandle::new(SlowDrop {
        started: Some(started),
        release: std::sync::Mutex::new(wait_release),
    });
    let reader = handle.load();
    let (_, dummy) = mpsc::channel();
    handle.store(SlowDrop {
        started: None,
        release: std::sync::Mutex::new(dummy),
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut byte = [0];
        socket.read_exact(&mut byte).await.unwrap();
        socket.write_all(&byte).await.unwrap();
    });
    let client = thread::spawn(move || {
        use std::io::{Read, Write};
        wait_started.recv_timeout(Duration::from_secs(2)).unwrap();
        let mut socket = std::net::TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        socket.write_all(&[42]).unwrap();
        let mut byte = [0];
        let response = socket.read_exact(&mut byte);
        // Release the destructor even on failure, so the unfixed variant
        // reports a failed assertion instead of hanging its runtime.
        release.send(()).unwrap();
        response.unwrap();
        assert_eq!(byte, [42]);
    });
    drop(reader);
    let result = tokio::task::spawn_blocking(move || client.join())
        .await
        .unwrap();
    result.unwrap();
    server.await.unwrap();
}
