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
//! cancel guard and the access log write from it), so a stream the client
//! abandons still estimates inline on its worker. Every path that can
//! wait — a buffered response, a bridge's terminal event, a stream's
//! normal end — comes here.

use std::sync::{mpsc, Arc, Mutex, OnceLock};

use tokio::sync::{oneshot, Semaphore};

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

/// Run `job` on the pool, waiting for a place first, and return what it
/// produced. A panic inside `job` is re-raised here.
///
/// Dropping the future while it waits for a place takes nothing; dropping
/// it once the job is admitted lets the job finish and discards its
/// product, so the job's place is held until then either way.
pub(crate) async fn run<R: Send + 'static>(job: impl FnOnce() -> R + Send + 'static) -> R {
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
    let sent = match pool.jobs.lock() {
        Ok(jobs) => jobs.send(task),
        Err(poisoned) => poisoned.into_inner().send(task),
    };
    if let Err(mpsc::SendError(task)) = sent {
        task();
    }
    rx.await
        .unwrap_or_else(|_| panic!("token estimation job panicked"))
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
        let ran_on = rt.block_on(run(|| std::thread::current().id()));
        assert_ne!(ran_on, caller);
    }
}
