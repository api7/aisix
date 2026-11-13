//! A dedicated thread pool for local token estimation.
//!
//! Tokenizing a response the upstream reported no usage for costs tens of
//! milliseconds of CPU per MiB of text. On a request worker that time is
//! taken from every other connection the worker serves — under
//! thread-per-core, one worker per core, a small request queued behind one
//! large estimate waits for all of it. Estimation therefore runs here, on
//! threads of its own, and the request that needs it awaits the result.
//!
//! The pool is plain OS threads fed by a channel, so it serves the
//! thread-per-core workers and the multi-thread runtime alike. It has one
//! thread per available CPU and admits at most [`SLOTS_PER_THREAD`] jobs
//! per thread, running or queued: a job holds the text it counts (up to
//! `OUTPUT_ACCUMULATION_CAP`) plus its request, so the queue stays bounded
//! and estimation demand beyond the pool waits in the requests that
//! need it ([`reserve`]) rather than piling up here.

use std::sync::{mpsc, Arc, Mutex, OnceLock};

use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};

/// Admitted jobs per pool thread: the one it runs and one waiting behind it.
const SLOTS_PER_THREAD: usize = 2;

type Job = Box<dyn FnOnce() + Send>;

struct Pool {
    slots: Arc<Semaphore>,
    jobs: Mutex<mpsc::Sender<Job>>,
}

/// `None` when not a single pool thread could be started (a pids cgroup
/// refusing threads): estimation then runs inline, as it did before the
/// pool existed, rather than queueing for threads that do not exist.
fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let started = (0..threads)
            .filter(|i| {
                let rx = Arc::clone(&rx);
                std::thread::Builder::new()
                    .name(format!("token-estimate-{i}"))
                    .spawn(move || work(&rx))
                    .is_ok()
            })
            .count();
        (started > 0).then(|| Pool {
            slots: Arc::new(Semaphore::new(started * SLOTS_PER_THREAD)),
            jobs: Mutex::new(tx),
        })
    })
    .as_ref()
}

fn work(rx: &Mutex<mpsc::Receiver<Job>>) {
    loop {
        let job = match rx.lock() {
            Ok(rx) => rx.recv(),
            Err(poisoned) => poisoned.into_inner().recv(),
        };
        let Ok(job) = job else { return };
        // A panicking job drops its result sender, which its caller sees;
        // the thread itself stays in the pool.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
    }
}

/// An admitted place in the pool. Holding one is what lets a job be
/// submitted; it is released when the job finishes.
pub(crate) struct Slot(Option<OwnedSemaphorePermit>);

/// Wait for a place in the pool. Cancel-safe: dropping the future before
/// it resolves takes nothing.
pub(crate) async fn reserve() -> Slot {
    match pool() {
        Some(pool) => Slot(Some(
            Arc::clone(&pool.slots)
                .acquire_owned()
                .await
                .expect("the estimation pool's semaphore is never closed"),
        )),
        None => Slot(None),
    }
}

/// A place in the pool if one is free right now — for a caller that
/// cannot wait (a Drop guard).
pub(crate) fn try_reserve() -> Option<Slot> {
    match pool() {
        Some(pool) => Arc::clone(&pool.slots)
            .try_acquire_owned()
            .ok()
            .map(|p| Slot(Some(p))),
        None => Some(Slot(None)),
    }
}

/// Start `job` in the slot. Without a pool it runs inline and the receiver
/// is ready on return.
pub(crate) fn submit<R: Send + 'static>(
    slot: Slot,
    job: impl FnOnce() -> R + Send + 'static,
) -> oneshot::Receiver<R> {
    let (tx, rx) = oneshot::channel();
    let Slot(permit) = slot;
    let task: Job = Box::new(move || {
        let _ = tx.send(job());
        drop(permit);
    });
    match pool() {
        Some(pool) => {
            let sent = match pool.jobs.lock() {
                Ok(jobs) => jobs.send(task),
                Err(poisoned) => poisoned.into_inner().send(task),
            };
            if let Err(mpsc::SendError(task)) = sent {
                task();
            }
        }
        None => task(),
    }
    rx
}

/// Run `job` on the pool, waiting for a place first, and return what it
/// produced. A panic inside `job` is re-raised here.
pub(crate) async fn run<R: Send + 'static>(job: impl FnOnce() -> R + Send + 'static) -> R {
    let slot = reserve().await;
    submit(slot, job)
        .await
        .unwrap_or_else(|_| panic!("token estimation job panicked"))
}

/// Run `job` on the pool for a caller that ends a request but cannot wait
/// (a Drop guard), handing its product to `then` exactly once.
///
/// `slot` is a place a caller that could wait reserved first (a stream's
/// normal end); with `None`, a free place is taken if there is one. With a
/// place, `job` runs on the pool and `then` on the caller's runtime once
/// it is done, in a task of its own so nothing dropped afterwards can lose
/// it; the returned handle lets a caller that can wait await it. Without
/// one — the pool is full, or there is no runtime to finish on — `job`
/// runs inline here, as before the pool existed: the work is never lost,
/// and what is pending never grows past the pool's bound.
pub(crate) fn run_then<R: Send + 'static>(
    slot: Option<Slot>,
    job: impl FnOnce() -> R + Send + 'static,
    then: impl FnOnce(R) + Send + 'static,
) -> Option<tokio::task::JoinHandle<()>> {
    let runtime = tokio::runtime::Handle::try_current().ok();
    let slot = runtime.as_ref().and_then(|_| slot.or_else(try_reserve));
    let (Some(runtime), Some(slot)) = (runtime, slot) else {
        then(job());
        return None;
    };
    let done = submit(slot, job);
    Some(runtime.spawn(tracing::Instrument::instrument(
        async move {
            if let Ok(product) = done.await {
                then(product);
            }
        },
        tracing::Span::current(),
    )))
}
