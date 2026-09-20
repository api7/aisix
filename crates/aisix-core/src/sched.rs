//! Scheduling priority for the gateway's own background computation.
//!
//! A gateway process is one request worker per core plus a handful of
//! threads doing work that is proportional to the whole configuration
//! rather than to the change or the request in front of it: building a
//! snapshot out of a watch batch, hashing it, rendering the Prometheus
//! exposition, freeing a retired snapshot. At the deployment shape this
//! matters in — one worker, one core — that work and the request worker
//! are runnable at the same time on the same CPU and the kernel splits the
//! core between them, so a burst of configuration writes is paid for in
//! request latency.
//!
//! None of it is latency-sensitive — an apply that lands a little later
//! is invisible, while the core it took from a request worker is not —
//! so it runs at the lowest priority the scheduler offers and yields the
//! core whenever a request worker is runnable. The cost is real and not
//! small: CFS weights nice 19 at 15 against nice 0's 1024, so on a
//! saturated core this work gets what is left rather than a share. That
//! is the intended trade, and it is why the list below is only work
//! nothing waits on.
//!
//! What must NOT be demoted: the request workers themselves, and the
//! listener serving `/livez` and `/readyz` — a liveness probe that loses
//! the core to a config apply is exactly the failure this is meant to
//! prevent. The log writer thread is also left alone; it is I/O, not
//! computation, and anything queued behind it holds memory.

/// Drop the **calling thread** to the lowest scheduling priority.
///
/// Best-effort: lowering a thread's priority needs no capability, so this
/// only fails in sandboxes that block the syscall outright, and there is
/// nothing to do about it but keep running at the default priority.
///
/// Linux applies `setpriority(PRIO_PROCESS, tid, ..)` per thread, not per
/// process — a thread id is what the kernel calls a process here. That is
/// the whole reason this is per-thread at all, and it is why the value is
/// never restored: raising a priority back DOES need `CAP_SYS_NICE` or
/// `RLIMIT_NICE` headroom, which a container image does not have. Every
/// caller therefore demotes a thread it owns for the duration of that
/// thread's life ([`run_demoted`] gives one out per unit of work).
pub fn demote_current_thread() {
    #[cfg(target_os = "linux")]
    {
        // PRIO_PROCESS with a thread id is per-thread on Linux.
        let _ = rustix::process::setpriority_process(
            Some(
                rustix::process::Pid::from_raw(rustix::thread::gettid().as_raw_nonzero().get())
                    .expect("gettid returns a live thread id"),
            ),
            LOWEST_PRIORITY,
        );
    }
}

/// `nice` value for background computation: the lowest the scheduler takes.
#[cfg(target_os = "linux")]
const LOWEST_PRIORITY: i32 = 19;

// Test seam for the spawn-refused path, which no test can provoke for
// real without exhausting the whole test process' thread budget.
// Thread-local, so the test that sets it cannot divert a `run_demoted`
// running in parallel on another test thread.
#[cfg(test)]
thread_local! {
    static BLOCK_SPAWNING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn spawning_is_blocked() -> bool {
    #[cfg(test)]
    {
        BLOCK_SPAWNING.with(|blocked| blocked.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Run `work` on a dedicated demoted thread and return what it produced.
///
/// A scoped thread, so `work` may borrow — which is what lets an apply
/// keep taking `&self` and a borrowed batch. The caller blocks until it
/// finishes, so this changes which CPU share the work competes for, not
/// when it completes relative to its caller.
///
/// The caller must not be an async task on a runtime it would starve; the
/// one in-tree caller that is wraps this in `block_in_place`.
///
/// A panic inside `work` is re-raised on the calling thread, so a caller
/// that used to see one still does.
pub fn run_demoted<T: Send>(name: &'static str, work: impl FnOnce() -> T + Send) -> T {
    // A thread the kernel refuses (EAGAIN under a pids cgroup limit) must
    // not become a failed apply: the caller is the configuration watch
    // loop, its panic is swallowed by the task it runs in, and the
    // gateway would serve its last snapshot forever with `/readyz` still
    // green. Lower priority is an optimization and may not take the
    // work with it when it cannot be had.
    let pending = std::sync::Mutex::new(Some(work));
    let done = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        let spawned = if spawning_is_blocked() {
            None
        } else {
            std::thread::Builder::new()
                .name(name.to_owned())
                .spawn_scoped(scope, || {
                    demote_current_thread();
                    let work = pending
                        .lock()
                        .expect("background work")
                        .take()
                        .expect("background work runs once");
                    let value = work();
                    *done.lock().expect("background work product") = Some(value);
                })
                .ok()
        };
        // No handle means the closure never ran, so `pending` still holds
        // the work and the caller runs it below.
        if let Some(handle) = spawned {
            handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        }
    });
    if let Some(work) = pending.into_inner().expect("background work") {
        return work();
    }
    done.into_inner()
        .expect("background work product")
        .expect("a joined background thread produced its value")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_demoted_returns_the_work_product_from_the_new_thread() {
        let caller = std::thread::current().id();
        let (value, ran_on) = run_demoted("test-demoted", || (6 * 7, std::thread::current().id()));
        assert_eq!(value, 42);
        assert_ne!(
            ran_on, caller,
            "the work must not run on the caller's thread"
        );
    }

    #[test]
    fn run_demoted_may_borrow_from_the_caller() {
        let owned = [1_u32, 2, 3];
        let sum = run_demoted("test-demoted-borrow", || owned.iter().sum::<u32>());
        assert_eq!(sum, 6);
        assert_eq!(owned.len(), 3, "the borrow outlives the scope");
    }

    #[test]
    #[should_panic(expected = "work panicked")]
    fn run_demoted_re_raises_a_panic_on_the_caller() {
        run_demoted("test-demoted-panic", || panic!("work panicked"));
    }

    /// The caller is the configuration watch loop; a thread the kernel
    /// refuses must cost priority, not the apply.
    #[test]
    fn work_still_runs_when_no_thread_can_be_spawned() {
        let caller = std::thread::current().id();
        BLOCK_SPAWNING.with(|blocked| blocked.set(true));
        let (value, ran_on) =
            run_demoted("test-demoted-nospawn", || (7, std::thread::current().id()));
        BLOCK_SPAWNING.with(|blocked| blocked.set(false));
        assert_eq!(value, 7, "the work must still produce its value");
        assert_eq!(
            ran_on, caller,
            "and it must fall back to the caller's thread"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_demoted_thread_runs_at_the_lowest_priority_and_the_caller_does_not() {
        fn nice_of_current_thread() -> i32 {
            let tid = rustix::thread::gettid();
            rustix::process::getpriority_process(Some(
                rustix::process::Pid::from_raw(tid.as_raw_nonzero().get()).expect("live thread"),
            ))
            .expect("reading a thread's own priority always succeeds")
        }
        let before = nice_of_current_thread();
        let inside = run_demoted("test-demoted-nice", nice_of_current_thread);
        assert_eq!(inside, LOWEST_PRIORITY);
        assert_eq!(
            nice_of_current_thread(),
            before,
            "demotion must stay on the thread that was spawned for it",
        );
    }
}
