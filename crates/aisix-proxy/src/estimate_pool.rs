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
//! per thread, running or queued: a job owns the text it counts (up to
//! `OUTPUT_ACCUMULATION_CAP`) plus its request, so what the pool holds
//! stays bounded and estimation demand beyond it waits in the requests
//! that need it rather than piling up here.
//!
//! A stream guard's `Drop` cannot wait, and its usage emit has to happen
//! inside that `Drop` (the request's attribution cell decides what the
//! cancel guard and the access log write from it). So a stream counts its
//! estimate here before it releases its terminal frame
//! (`token_estimate::CountedAhead`), and a client that closes on that frame
//! leaves `Drop` a finished count; only a stream abandoned before its end
//! still estimates inline on its worker.
//!
//! Small estimates never come here: [`run`] counts them on the caller.

mod json_len;

use std::sync::{mpsc, Arc, Mutex, OnceLock};

use tokio::sync::{oneshot, Semaphore};

/// Text an estimate tokenizes, at most, to be counted inline on the calling
/// worker. Tokenizing costs about 35 ns per byte, so this blocks a worker
/// for at most ~0.6 ms; below it, handing the job to the pool and waking
/// the caller costs more CPU than the count itself.
pub(crate) const INLINE_MAX_BYTES: usize = 16 * 1024;

/// Admitted jobs per pool thread: the one it runs and one waiting behind it.
const SLOTS_PER_THREAD: usize = 2;

type Job = Box<dyn FnOnce() + Send>;

struct Pool {
    slots: Arc<Semaphore>,
    #[cfg(test)]
    capacity: u32,
    jobs: mpsc::Sender<Job>,
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
            #[cfg(test)]
            capacity: (started * SLOTS_PER_THREAD) as u32,
            jobs: tx,
        })
    })
    .as_ref()
}

thread_local! {
    static ON_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the caller is one of the pool's threads, which count with their
/// own tokenizer instances (`token_estimate::bpe_for`).
pub(crate) fn on_pool_thread() -> bool {
    ON_POOL.with(std::cell::Cell::get)
}

fn work(rx: &Mutex<mpsc::Receiver<Job>>) {
    ON_POOL.with(|on| on.set(true));
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

/// Run `job`, an estimate tokenizing at most `text_bytes` bytes, and return
/// what it produced: inline when that is at most [`INLINE_MAX_BYTES`],
/// otherwise on the pool, waiting for a place first. A panic inside `job`
/// is re-raised here.
///
/// Dropping the future while it waits for a place takes nothing; dropping
/// it once the job is admitted lets the job finish and discards its
/// product, so the job's place is held until then either way.
pub(crate) async fn run<R: Send + 'static>(
    text_bytes: usize,
    job: impl FnOnce() -> R + Send + 'static,
) -> R {
    if text_bytes <= INLINE_MAX_BYTES {
        return job();
    }
    let Some(pool) = pool() else {
        return job();
    };
    let permit = Arc::clone(&pool.slots)
        .acquire_owned()
        .await
        .expect("the estimation pool's semaphore is never closed");
    let (tx, rx) = oneshot::channel();
    let task: Job = Box::new(move || {
        let _ = tx.send(job());
        drop(permit);
    });
    if let Err(mpsc::SendError(task)) = pool.jobs.send(task) {
        task();
    }
    rx.await
        .unwrap_or_else(|_| panic!("token estimation job panicked"))
}

/// The size of the text an estimate tokenizes from `value`: its
/// compact-JSON length with strings counted unescaped. That bounds the
/// unescaped text the estimator reads; the subtrees it tokenizes as
/// escaped JSON (`count_json`) can exceed it by the escape expansion, at
/// worst about 6x for control characters. Measuring stops once the length
/// passes [`INLINE_MAX_BYTES`], and a string costs one addition whatever
/// its length, so probing a large request costs no more than a small one.
/// Any such length is reported as `INLINE_MAX_BYTES + 1`.
pub(crate) fn serialized_bytes_bound(value: &impl serde::Serialize) -> usize {
    json_len::json_len_within(value, INLINE_MAX_BYTES).unwrap_or(INLINE_MAX_BYTES + 1)
}

/// Take every place in the pool, so a test can hold a pooled estimate in
/// its wait for as long as it keeps the permit.
#[cfg(test)]
pub(crate) async fn hold_every_slot() -> tokio::sync::OwnedSemaphorePermit {
    let pool = pool().expect("the estimation pool starts its threads under test");
    Arc::clone(&pool.slots)
        .acquire_many_owned(pool.capacity)
        .await
        .expect("the estimation pool's semaphore is never closed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_run_off_the_calling_thread() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let caller = std::thread::current().id();
        let ran_on = rt.block_on(run(INLINE_MAX_BYTES + 1, || std::thread::current().id()));
        assert_ne!(ran_on, caller);
    }

    #[test]
    fn small_jobs_run_on_the_calling_thread() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let caller = std::thread::current().id();
        let ran_on = rt.block_on(run(INLINE_MAX_BYTES, || std::thread::current().id()));
        assert_eq!(ran_on, caller);
    }

    #[test]
    fn serialized_bound_stops_past_the_inline_limit() {
        let small = serde_json::json!({"input": "x".repeat(100)});
        assert!(serialized_bytes_bound(&small) >= serde_json::to_vec(&small).unwrap().len());
        let large = serde_json::json!({"input": "x".repeat(10 * INLINE_MAX_BYTES)});
        assert_eq!(serialized_bytes_bound(&large), INLINE_MAX_BYTES + 1);
    }
}
