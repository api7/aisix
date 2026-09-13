use std::sync::{mpsc, Arc, OnceLock};
use std::time::Duration;

trait Retired: Send {
    fn reclaim(&mut self) -> bool;
}

impl<S: Send + Sync> Retired for Option<Arc<S>> {
    fn reclaim(&mut self) -> bool {
        // Unlike a strong-count check, try_unwrap also handles a racing Weak
        // upgrade without letting its reader become the final destructor.
        match Arc::try_unwrap(self.take().expect("pending snapshot")) {
            Ok(snapshot) => {
                drop(snapshot);
                true
            }
            Err(snapshot) => {
                *self = Some(snapshot);
                false
            }
        }
    }
}

type Snapshot = Box<dyn Retired>;

fn run(receiver: mpsc::Receiver<Snapshot>) {
    let mut pending: Vec<Snapshot> = Vec::new();
    loop {
        let next = if pending.is_empty() {
            receiver
                .recv()
                .map_err(|_| mpsc::RecvTimeoutError::Disconnected)
        } else {
            // A long request may retain an old snapshot after configuration
            // writes stop. Reclaim it without requiring another publication.
            receiver.recv_timeout(Duration::from_millis(10))
        };
        match next {
            Ok(snapshot) => pending.push(snapshot),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
        pending.retain_mut(|snapshot| !snapshot.reclaim());
    }
}

pub(super) fn retire<S: Send + Sync + 'static>(snapshot: Arc<S>) {
    static SENDER: OnceLock<mpsc::SyncSender<Snapshot>> = OnceLock::new();
    let sender = SENDER.get_or_init(|| {
        // Backpressure applies to publishers, never readers. Only snapshots
        // still held by readers stay in the reclaimer's pending list.
        let (sender, receiver) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("snapshot-reclaim".into())
            .spawn(move || run(receiver))
            .expect("start snapshot reclaimer");
        sender
    });
    sender
        .send(Box::new(Some(snapshot)))
        .unwrap_or_else(|_| panic!("snapshot reclaimer stopped"));
}
