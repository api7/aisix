//! A bounded, lossy log sink: the request path never blocks on logging.
//!
//! The subscriber used to hand each formatted event straight to
//! `std::io::stderr`, from whatever thread produced it. In a container
//! that descriptor is a 64 KiB pipe to the runtime's log shim, and when
//! the shim stops draining it — kubelet rotating and gzipping the
//! container log is the routine cause, at this gateway's log volume every
//! ~40 seconds — the pipe fills and `write` blocks. With one request
//! worker per core that blocks the worker: measured on the benchmark
//! gateway, the only request thread sat in `pipe_write` for 0.65 s with
//! the process at 0.00 CPU, `/livez` took 663 ms, and every request in
//! flight finished in one burst when the pipe drained.
//!
//! So events go into a fixed-size queue and one dedicated thread drains
//! it. When the queue is full the NEW event is dropped — the alternative,
//! blocking the producer, is the bug. Drops are counted in
//! `aisix_log_lines_dropped_total` and, once the sink catches up, stated
//! once in a warning; a gap in the log that the log itself does not
//! account for would be worse than the gap.
//!
//! Two things deliberately stay synchronous. A panic still goes straight
//! to stderr, because the default panic hook writes there itself rather
//! than through the subscriber — nothing here may change that, though it
//! is worth knowing that a panic raised while the sink is stuck queues
//! behind the same descriptor lock the writer thread holds, exactly as it
//! did before any of this. And `shutdown` is called on the way out of
//! `main`, so a graceful exit empties the queue before the process goes.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crossbeam_queue::ArrayQueue;
use tracing_subscriber::fmt::MakeWriter;

/// Every event this process has ever dropped.
///
/// Read at scrape time rather than pushed: the writer thread starts in
/// `init_tracing`, long before `Metrics` exists, and this crate keeps no
/// global recorder for it to write to. `Metrics::sync_log_status` turns
/// it into `aisix_log_lines_dropped_total`.
static DROPPED_TOTAL: AtomicU64 = AtomicU64::new(0);

/// How many log events have been dropped since the process started.
pub(crate) fn dropped_total() -> u64 {
    DROPPED_TOTAL.load(Ordering::Relaxed)
}

/// Queue depth, in events.
///
/// At the log volume that provokes this (a few hundred lines a second,
/// ~500 bytes each) it absorbs roughly a minute of a stalled sink for
/// about 16 MiB of resident memory — an order of magnitude more than the
/// stalls that were measured, and still bounded.
pub(crate) const CAPACITY: usize = 32_768;

/// How long the writer thread waits for work before looking again at the
/// drop counter and the shutdown flag.
const IDLE_POLL: Duration = Duration::from_millis(100);

struct Shared {
    queue: ArrayQueue<Vec<u8>>,
    /// Events dropped and not yet named in a warning.
    unwarned: AtomicU64,
    /// Whether the writer is between taking work and finishing it. An
    /// empty queue is NOT an emptied one: the line the writer is parked
    /// inside `write` with has already been popped, so a drain that only
    /// looked at the queue would call a stuck sink drained and then join
    /// a thread that never returns.
    writing: AtomicBool,
    /// Set once, on the way out, to wake and retire the writer thread.
    stopping: AtomicBool,
    /// Guards nothing; paired with `wake` so the writer can sleep.
    idle: Mutex<()>,
    wake: Condvar,
}

impl Shared {
    fn push(&self, line: Vec<u8>) {
        if self.queue.push(line).is_err() {
            // Counted here rather than by the writer thread, because the
            // writer is parked inside the stuck sink for exactly as long
            // as the drops are happening — folding them in from there
            // would publish zero for the whole window the metric exists
            // to describe.
            DROPPED_TOTAL.fetch_add(1, Ordering::Relaxed);
            self.unwarned.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // Cheap when nobody is parked, which is the case whenever the
        // sink is keeping up. A notification that lands in the window
        // between the writer's emptiness check and its park is lost, and
        // costs that line up to `IDLE_POLL`; closing it would need the
        // producer to take a lock, which is the thing this must not do.
        self.wake.notify_one();
    }

    /// Nothing queued and nothing in flight.
    fn drained(&self) -> bool {
        self.queue.is_empty() && !self.writing.load(Ordering::Acquire)
    }
}

/// Handle for `MakeWriter`: cloned per event, holds no lock.
#[derive(Clone)]
pub(crate) struct QueueWriter {
    shared: Arc<Shared>,
    /// One event is one `write_all` from the fmt layer; this collects the
    /// bytes of the event being formatted right now.
    pending: Vec<u8>,
}

impl Write for QueueWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.pending.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if !self.pending.is_empty() {
            self.shared.push(std::mem::take(&mut self.pending));
        }
        Ok(())
    }
}

impl Drop for QueueWriter {
    fn drop(&mut self) {
        // The fmt layer drops the writer instead of flushing it.
        let _ = self.flush();
    }
}

/// `MakeWriter` handing out [`QueueWriter`]s onto one shared queue.
#[derive(Clone)]
pub(crate) struct LogQueue {
    shared: Arc<Shared>,
}

impl<'a> MakeWriter<'a> for LogQueue {
    type Writer = QueueWriter;

    fn make_writer(&'a self) -> Self::Writer {
        QueueWriter {
            shared: Arc::clone(&self.shared),
            pending: Vec::new(),
        }
    }
}

/// Everything the process keeps after installing the subscriber: the
/// queue to flush and the thread to retire.
pub(crate) struct LogWriter {
    shared: Arc<Shared>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl LogWriter {
    /// Start the writer thread draining into `sink`.
    pub(crate) fn start(
        mut sink: impl Write + Send + 'static,
        capacity: usize,
    ) -> (LogQueue, LogWriter) {
        let shared = Arc::new(Shared {
            queue: ArrayQueue::new(capacity),
            unwarned: AtomicU64::new(0),
            writing: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            idle: Mutex::new(()),
            wake: Condvar::new(),
        });
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("log-writer".into())
            .spawn(move || {
                // Deliberately NOT demoted: this thread is the only way a
                // queued line reaches the log, and anything waiting on it
                // is holding memory.
                let mut unreported = 0_u64;
                loop {
                    // Raised before the pop, so the flag is never false
                    // with a line already taken off the queue.
                    worker.writing.store(true, Ordering::Release);
                    let mut wrote = false;
                    while let Some(line) = worker.queue.pop() {
                        let _ = sink.write_all(&line);
                        wrote = true;
                    }
                    if wrote {
                        let _ = sink.flush();
                    }
                    worker.writing.store(false, Ordering::Release);
                    unreported += worker.unwarned.swap(0, Ordering::Relaxed);
                    // Only once the sink has caught up, so a sustained
                    // stall does not spend the queue on its own report.
                    if unreported > 0 && worker.queue.is_empty() {
                        tracing::warn!(
                            dropped = unreported,
                            "log sink fell behind; dropped log events",
                        );
                        unreported = 0;
                        continue;
                    }
                    if worker.stopping.load(Ordering::Acquire) && worker.queue.is_empty() {
                        return;
                    }
                    if worker.queue.is_empty() {
                        let guard = worker.idle.lock().expect("log writer idle lock");
                        let _ = worker.wake.wait_timeout(guard, IDLE_POLL);
                    }
                }
            })
            .expect("spawn the log writer thread");
        (
            LogQueue {
                shared: Arc::clone(&shared),
            },
            LogWriter {
                shared,
                thread: Mutex::new(Some(thread)),
            },
        )
    }

    /// Wait until the queue is empty, or `deadline` passes.
    ///
    /// Returns whether it drained. Does not stop the writer, so logging
    /// keeps working afterwards.
    pub(crate) fn flush(&self, deadline: Duration) -> bool {
        let until = Instant::now() + deadline;
        loop {
            if self.shared.drained() {
                return true;
            }
            if Instant::now() >= until {
                return false;
            }
            self.shared.wake.notify_one();
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Drain and retire the writer thread. Later events are discarded.
    ///
    /// The join is conditional on the drain having finished, and that is
    /// the whole point: the writer only checks the stop flag after
    /// emptying the queue, so a sink that is still refusing to accept
    /// bytes leaves it parked in `write` forever. Joining unconditionally
    /// would hang the process exactly in the scenario this module exists
    /// for, waiting for a consumer that has already stopped consuming.
    /// Abandoning the thread costs the queued lines, which a stuck sink
    /// was never going to take anyway.
    pub(crate) fn shutdown(&self, deadline: Duration) -> bool {
        let drained = self.flush(deadline);
        self.shared.stopping.store(true, Ordering::Release);
        self.shared.wake.notify_one();
        if !drained {
            return false;
        }
        if let Some(thread) = self.thread.lock().expect("log writer handle").take() {
            let _ = thread.join();
        }
        true
    }

    #[cfg(test)]
    fn dropped(&self) -> u64 {
        self.shared.unwarned.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn queued(&self) -> usize {
        self.shared.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sink that blocks in `write_all` until it is released, standing in
    /// for a container log pipe nobody is draining.
    #[derive(Clone)]
    struct BlockedSink {
        gate: Arc<(Mutex<bool>, Condvar)>,
        written: Arc<Mutex<Vec<u8>>>,
    }

    impl BlockedSink {
        fn new() -> Self {
            Self {
                gate: Arc::new((Mutex::new(false), Condvar::new())),
                written: Arc::new(Mutex::new(Vec::new())),
            }
        }
        fn release(&self) {
            *self.gate.0.lock().expect("gate") = true;
            self.gate.1.notify_all();
        }
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.written.lock().expect("written")).into_owned()
        }
    }

    impl Write for BlockedSink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let mut open = self.gate.0.lock().expect("gate");
            while !*open {
                open = self.gate.1.wait(open).expect("gate");
            }
            drop(open);
            self.written
                .lock()
                .expect("written")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn line(n: usize) -> Vec<u8> {
        format!("line-{n}\n").into_bytes()
    }

    #[test]
    fn a_blocked_sink_does_not_block_the_emitting_thread() {
        let sink = BlockedSink::new();
        let (queue, writer) = LogWriter::start(sink.clone(), 64);
        let started = Instant::now();
        for n in 0..32 {
            let mut w = queue.make_writer();
            w.write_all(&line(n)).expect("queued");
        }
        let emitted = started.elapsed();
        assert!(
            emitted < Duration::from_millis(200),
            "emitting must not wait on the sink, took {emitted:?}",
        );
        assert_eq!(writer.dropped(), 0, "nothing was dropped below the bound");
        sink.release();
        assert!(writer.shutdown(Duration::from_secs(5)), "queue drains");
        assert!(sink.text().contains("line-31"), "got: {}", sink.text());
    }

    #[test]
    fn events_past_the_bound_are_dropped_and_counted() {
        let sink = BlockedSink::new();
        let (queue, writer) = LogWriter::start(sink.clone(), 8);
        for n in 0..40 {
            let mut w = queue.make_writer();
            w.write_all(&line(n)).expect("accepted");
        }
        // The writer thread may have taken one line off the queue and be
        // parked inside the blocked sink, so the bound admits at most one
        // more than its capacity.
        let dropped = writer.dropped();
        assert!(
            (31..=32).contains(&dropped),
            "expected the 40 events minus the bound to be dropped, got {dropped}",
        );
        sink.release();
        writer.shutdown(Duration::from_secs(5));
        let text = sink.text();
        assert!(
            text.contains("line-0") && !text.contains("line-39"),
            "the NEW event is the one dropped, got: {text}",
        );
    }

    #[test]
    fn shutdown_flushes_what_is_queued() {
        let sink = BlockedSink::new();
        let (queue, writer) = LogWriter::start(sink.clone(), 1024);
        for n in 0..500 {
            let mut w = queue.make_writer();
            w.write_all(&line(n)).expect("accepted");
        }
        assert!(writer.queued() > 0, "the sink has not drained anything yet");
        sink.release();
        assert!(
            writer.shutdown(Duration::from_secs(5)),
            "shutdown reports a completed drain",
        );
        let text = sink.text();
        for n in [0, 250, 499] {
            assert!(text.contains(&format!("line-{n}\n")), "missing line-{n}");
        }
    }

    /// The scenario this module exists for must not become a process
    /// that will not exit.
    ///
    /// One line into a queue with room for 64 is the case that matters:
    /// the writer pops it and parks inside the sink, so the QUEUE is
    /// empty while the line is still unwritten. A drain that only asked
    /// the queue would call that drained and then join forever.
    #[test]
    fn shutdown_gives_up_on_a_sink_that_never_drains() {
        let sink = BlockedSink::new();
        let (queue, writer) = LogWriter::start(sink.clone(), 64);
        let mut w = queue.make_writer();
        w.write_all(&line(0)).expect("accepted");
        drop(w);
        // Let the writer take it off the queue and park in the sink.
        let until = Instant::now() + Duration::from_secs(5);
        while Instant::now() < until && writer.queued() > 0 {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(writer.queued(), 0, "the writer has taken the line");

        let started = Instant::now();
        assert!(
            !writer.shutdown(Duration::from_millis(200)),
            "a line still inside the sink is not a drained queue",
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "shutdown must not wait on a sink that is not consuming, took {:?}",
            started.elapsed(),
        );
        sink.release();
    }

    #[test]
    fn one_event_is_one_queue_entry_even_when_written_in_pieces() {
        let sink = BlockedSink::new();
        sink.release();
        let (queue, writer) = LogWriter::start(sink.clone(), 4);
        let mut w = queue.make_writer();
        w.write_all(b"half ").expect("accepted");
        w.write_all(b"an event\n").expect("accepted");
        drop(w);
        writer.shutdown(Duration::from_secs(5));
        assert_eq!(sink.text(), "half an event\n");
    }
}
