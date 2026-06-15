//! The shared, per-sink delivery pipeline.
//!
//! Generalises the proven `aisix-server` telemetry worker (bounded mpsc →
//! batch → flush) into a reusable component every sink runs behind, and adds
//! what telemetry deliberately skipped: retry with exponential backoff and
//! drop-with-metric backpressure. One pipeline per sink, so a slow or down
//! sink can never stall another or the request hot path.
//!
//! Division of labour: the pipeline owns *flow control* — a bounded queue,
//! count/time batching, retry/backoff and drop accounting. The sink owns
//! *wire encoding*, including chunking a batch down to its own per-request
//! byte limit inside `append_batch` (only the sink knows the encoded size).

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};

use super::{EventBatch, IdempotencyMarker, ObservabilitySink, SinkError, SinkRecord};

/// Tuning for a [`SinkPipeline`]. Defaults mirror the telemetry worker
/// (100-record batches, 5s flush, 1024-deep queue) plus a bounded retry.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Bound on the producer→worker queue. When full, `try_enqueue` drops
    /// the record (counted) rather than blocking the request hot path.
    pub queue_capacity: usize,
    /// Flush once this many records have buffered.
    pub max_batch: usize,
    /// Flush whatever is buffered at least this often.
    pub flush_interval: Duration,
    /// Max retry attempts for a [`SinkError::Transient`] batch before it is
    /// dropped (counted). `0` = no retry.
    pub max_retries: u32,
    /// First retry delay; doubles each attempt up to `max_backoff`.
    pub base_backoff: Duration,
    /// Ceiling on the exponential backoff delay.
    pub max_backoff: Duration,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1024,
            max_batch: 100,
            flush_interval: Duration::from_secs(5),
            max_retries: 4,
            base_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(5),
        }
    }
}

/// Live delivery counters for one sink, shared between the producer handle
/// and the worker. Cheap atomics; read via [`SinkStats::snapshot`].
///
/// Lifetime note: the stats live as long as the pipeline. A reconfigured
/// or removed-then-re-added exporter gets a fresh pipeline (see
/// `ExporterPipelines::get_or_create`) and therefore fresh zeroed
/// counters — consumers (heartbeat `exporter_health`, #519 D.2) must
/// treat the counters as resettable, not lifetime-of-process totals.
#[derive(Debug, Default)]
pub struct SinkStats {
    sent: AtomicU64,
    dropped: AtomicU64,
    retries: AtomicU64,
    delivered_batches: AtomicU64,
    failed_batches: AtomicU64,
    /// Unix seconds of the most recent batch outcome; 0 = never.
    last_success_unix: AtomicI64,
    last_failure_unix: AtomicI64,
    last_error: Mutex<Option<String>>,
}

/// A point-in-time read of [`SinkStats`] for health / dashboard surfaces.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SinkStatsSnapshot {
    /// Records the sink confirmed it accepted.
    pub sent: u64,
    /// Records dropped — queue-full backpressure plus retry-exhausted /
    /// permanent failures.
    pub dropped: u64,
    /// Retry attempts made across all batches.
    pub retries: u64,
    /// Batches the sink acknowledged.
    pub delivered_batches: u64,
    /// Batches given up on after retries (or a permanent error).
    pub failed_batches: u64,
    /// Masked excerpt of the most recent delivery error. Cleared on the
    /// next successful delivery, so `Some` means "currently failing".
    pub last_error: Option<String>,
    /// Unix seconds of the most recent successful batch delivery.
    pub last_success_unix: Option<i64>,
    /// Unix seconds of the most recent dropped batch. Unlike
    /// `last_error` this persists across later successes — it answers
    /// "when did this exporter last lose data".
    pub last_failure_unix: Option<i64>,
}

fn now_unix_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

impl SinkStats {
    fn add_sent(&self, n: u64) {
        self.sent.fetch_add(n, Ordering::Relaxed);
    }
    fn add_dropped(&self, n: u64) {
        self.dropped.fetch_add(n, Ordering::Relaxed);
    }
    fn add_retries(&self, n: u64) {
        self.retries.fetch_add(n, Ordering::Relaxed);
    }
    fn record_batch_delivered(&self) {
        self.delivered_batches.fetch_add(1, Ordering::Relaxed);
        self.last_success_unix
            .store(now_unix_secs(), Ordering::Relaxed);
    }
    fn record_batch_failed(&self) {
        self.failed_batches.fetch_add(1, Ordering::Relaxed);
        self.last_failure_unix
            .store(now_unix_secs(), Ordering::Relaxed);
    }
    fn set_error(&self, detail: String) {
        *self.last_error.lock() = Some(detail);
    }
    fn clear_error(&self) {
        *self.last_error.lock() = None;
    }

    /// Read the current counters.
    pub fn snapshot(&self) -> SinkStatsSnapshot {
        let to_opt = |v: i64| (v != 0).then_some(v);
        SinkStatsSnapshot {
            sent: self.sent.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
            delivered_batches: self.delivered_batches.load(Ordering::Relaxed),
            failed_batches: self.failed_batches.load(Ordering::Relaxed),
            last_error: self.last_error.lock().clone(),
            last_success_unix: to_opt(self.last_success_unix.load(Ordering::Relaxed)),
            last_failure_unix: to_opt(self.last_failure_unix.load(Ordering::Relaxed)),
        }
    }
}

/// Cheap, clonable producer handle the request hot path uses to enqueue
/// records. Cloning shares the same queue and stats.
#[derive(Clone)]
pub struct SinkHandle {
    name: Arc<str>,
    tx: mpsc::Sender<Arc<SinkRecord>>,
    stats: Arc<SinkStats>,
}

impl SinkHandle {
    /// Non-blocking enqueue. Returns `false` (and counts a drop) when the
    /// bounded queue is full or the worker has stopped — never blocks the
    /// request hot path.
    pub fn try_enqueue(&self, record: Arc<SinkRecord>) -> bool {
        match self.tx.try_send(record) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.stats.add_dropped(1);
                tracing::debug!(sink = %self.name, "sink queue full; record dropped");
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.stats.add_dropped(1);
                false
            }
        }
    }

    /// Snapshot of this sink's delivery counters.
    pub fn stats(&self) -> SinkStatsSnapshot {
        self.stats.snapshot()
    }

    /// Stable sink name (logs / metrics labels).
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// The per-sink worker: drains the queue, batches, and delivers with
/// retry/backoff. Build with [`SinkPipeline::new`]; the caller spawns
/// [`SinkPipeline::run`].
pub struct SinkPipeline {
    sink: Arc<dyn ObservabilitySink>,
    cfg: PipelineConfig,
    rx: mpsc::Receiver<Arc<SinkRecord>>,
    stats: Arc<SinkStats>,
}

impl SinkPipeline {
    /// Build a pipeline for one sink. Returns the producer handle and the
    /// worker; spawn `worker.run(cancel)` on the runtime.
    pub fn new(
        sink: Arc<dyn ObservabilitySink>,
        cfg: PipelineConfig,
    ) -> (SinkHandle, SinkPipeline) {
        let (tx, rx) = mpsc::channel(cfg.queue_capacity);
        let stats = Arc::new(SinkStats::default());
        let handle = SinkHandle {
            name: Arc::from(sink.name()),
            tx,
            stats: Arc::clone(&stats),
        };
        let worker = SinkPipeline {
            sink,
            cfg,
            rx,
            stats,
        };
        (handle, worker)
    }

    /// Drain → batch → deliver until the channel closes or `cancel` flips
    /// true. Performs one final flush on shutdown. Delivery failures are
    /// counted and logged, never propagated.
    pub async fn run(mut self, mut cancel: watch::Receiver<bool>) {
        let mut ticker = tokio::time::interval(self.cfg.flush_interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut buffer: Vec<Arc<SinkRecord>> = Vec::with_capacity(self.cfg.max_batch);

        tracing::info!(
            sink = %self.sink.name(),
            max_batch = self.cfg.max_batch,
            flush_interval_secs = self.cfg.flush_interval.as_secs(),
            max_retries = self.cfg.max_retries,
            "sink pipeline started",
        );

        loop {
            tokio::select! {
                maybe = self.rx.recv() => match maybe {
                    Some(record) => {
                        buffer.push(record);
                        if buffer.len() >= self.cfg.max_batch {
                            self.flush(&mut buffer).await;
                        }
                    }
                    None => {
                        self.flush(&mut buffer).await;
                        tracing::info!(sink = %self.sink.name(), "sink pipeline: channel closed, exiting");
                        return;
                    }
                },
                _ = ticker.tick() => {
                    self.flush(&mut buffer).await;
                }
                _ = cancel.changed() => {
                    if *cancel.borrow() {
                        while let Ok(record) = self.rx.try_recv() {
                            buffer.push(record);
                        }
                        self.flush(&mut buffer).await;
                        tracing::info!(sink = %self.sink.name(), "sink pipeline shutting down");
                        return;
                    }
                }
            }
        }
    }

    /// Take the buffer and deliver it as one batch (with retry). No-op when
    /// empty.
    async fn flush(&self, buffer: &mut Vec<Arc<SinkRecord>>) {
        if buffer.is_empty() {
            return;
        }
        let records = std::mem::take(buffer);
        buffer.reserve(self.cfg.max_batch);
        let count = records.len();
        let batch = EventBatch::new(records);
        self.deliver(&batch, count).await;
    }

    /// Deliver one batch, retrying transient failures with exponential
    /// backoff. At-least-once: a retried batch may re-send already-accepted
    /// records (the marker is `None` for the at-least-once sinks this phase
    /// serves; offset-token sinks set their own marker later).
    async fn deliver(&self, batch: &EventBatch, count: usize) {
        let marker = IdempotencyMarker::None;
        let mut attempt: u32 = 0;
        loop {
            match self.sink.append_batch(batch, &marker).await {
                Ok(ack) => {
                    self.stats.add_sent(ack.accepted as u64);
                    self.stats.record_batch_delivered();
                    self.stats.clear_error();
                    return;
                }
                Err(err) => {
                    let detail = masked(&err);
                    if err.is_transient() && attempt < self.cfg.max_retries {
                        attempt += 1;
                        self.stats.add_retries(1);
                        let delay = backoff(self.cfg.base_backoff, self.cfg.max_backoff, attempt);
                        tracing::warn!(
                            sink = %self.sink.name(),
                            attempt,
                            delay_ms = delay.as_millis() as u64,
                            error = %detail,
                            "sink delivery failed; retrying",
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    self.stats.record_batch_failed();
                    self.stats.add_dropped(count as u64);
                    self.stats.set_error(detail.clone());
                    tracing::warn!(
                        sink = %self.sink.name(),
                        dropped = count,
                        transient = err.is_transient(),
                        error = %detail,
                        "sink delivery dropped after retries",
                    );
                    return;
                }
            }
        }
    }
}

/// Exponential backoff: `base * 2^(attempt - 1)`, capped at `cap`.
fn backoff(base: Duration, cap: Duration, attempt: u32) -> Duration {
    let factor = 2u32.saturating_pow(attempt.saturating_sub(1));
    base.checked_mul(factor).unwrap_or(cap).min(cap)
}

/// Trim a sink error to a bounded, log-safe excerpt. The sink is responsible
/// for not embedding secrets in its error text; this only caps length so a
/// verbose upstream body can't flood the logs.
fn masked(err: &SinkError) -> String {
    err.to_string().chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::{backoff, PipelineConfig, SinkPipeline};
    use crate::sink::{
        BatchUnit, EventBatch, IdempotencyMarker, IdempotencyScheme, ObservabilitySink,
        OrderingScope, SinkAck, SinkCapabilities, SinkError, SinkHealth, SinkRecord, SinkResult,
    };
    use crate::usage::UsageEvent;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::watch;

    enum Mode {
        Ok,
        /// Fail with a transient error this many times, then succeed.
        TransientThenOk(AtomicU32),
        AlwaysTransient,
        Permanent,
    }

    /// A configurable sink that records the batch sizes it was handed.
    struct FakeSink {
        mode: Mode,
        batch_sizes: Mutex<Vec<usize>>,
    }

    impl FakeSink {
        fn new(mode: Mode) -> Arc<Self> {
            Arc::new(Self {
                mode,
                batch_sizes: Mutex::new(Vec::new()),
            })
        }
        fn delivered(&self) -> usize {
            self.batch_sizes.lock().iter().sum()
        }
    }

    #[async_trait::async_trait]
    impl ObservabilitySink for FakeSink {
        fn name(&self) -> &str {
            "fake"
        }

        fn capabilities(&self) -> SinkCapabilities {
            SinkCapabilities {
                idempotency: IdempotencyScheme::None,
                ordering: OrderingScope::None,
                batch_unit: BatchUnit::Records,
                max_batch_bytes: None,
                supports_partial_batch: false,
                supports_streaming_ingest: false,
            }
        }

        async fn append_batch(
            &self,
            batch: &EventBatch,
            _marker: &IdempotencyMarker,
        ) -> SinkResult {
            match &self.mode {
                Mode::Ok => {
                    self.batch_sizes.lock().push(batch.len());
                    Ok(SinkAck {
                        accepted: batch.len(),
                        ..SinkAck::default()
                    })
                }
                Mode::TransientThenOk(remaining) => {
                    if remaining.load(Ordering::Relaxed) > 0 {
                        remaining.fetch_sub(1, Ordering::Relaxed);
                        Err(SinkError::Transient("temporary".into()))
                    } else {
                        self.batch_sizes.lock().push(batch.len());
                        Ok(SinkAck {
                            accepted: batch.len(),
                            ..SinkAck::default()
                        })
                    }
                }
                Mode::AlwaysTransient => Err(SinkError::Transient("always failing".into())),
                Mode::Permanent => Err(SinkError::Permanent("bad credentials".into())),
            }
        }

        async fn healthcheck(&self) -> SinkHealth {
            SinkHealth::healthy()
        }
    }

    fn rec(i: u32) -> Arc<SinkRecord> {
        Arc::new(SinkRecord::metadata_only(UsageEvent {
            request_id: format!("req-{i}"),
            ..UsageEvent::default()
        }))
    }

    /// Fast config: no time-based flush (60s), tiny backoff so retry tests
    /// finish quickly.
    fn cfg() -> PipelineConfig {
        PipelineConfig {
            queue_capacity: 1024,
            max_batch: 100,
            flush_interval: Duration::from_secs(60),
            max_retries: 4,
            base_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(5),
        }
    }

    async fn wait_for(f: impl Fn() -> bool, within: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + within;
        while tokio::time::Instant::now() < deadline {
            if f() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        f()
    }

    #[tokio::test]
    async fn delivers_all_records_when_channel_closes() {
        let sink = FakeSink::new(Mode::Ok);
        let (handle, worker) = SinkPipeline::new(sink.clone(), cfg());
        for i in 0..5 {
            assert!(handle.try_enqueue(rec(i)));
        }
        // Closing the channel makes the worker drain + final-flush, then exit.
        drop(handle);
        let (_keep_alive, cancel) = watch::channel(false);
        worker.run(cancel).await;

        assert_eq!(sink.delivered(), 5);
    }

    #[tokio::test]
    async fn flushes_at_the_batch_ceiling() {
        let sink = FakeSink::new(Mode::Ok);
        let mut c = cfg();
        c.max_batch = 2;
        let (handle, worker) = SinkPipeline::new(sink.clone(), c);
        for i in 0..5 {
            assert!(handle.try_enqueue(rec(i)));
        }
        drop(handle);
        let (_keep_alive, cancel) = watch::channel(false);
        worker.run(cancel).await;

        // Two count-flushes of 2, then a final drain flush of 1.
        assert_eq!(*sink.batch_sizes.lock(), vec![2, 2, 1]);
    }

    #[tokio::test]
    async fn retries_transient_then_succeeds() {
        let sink = FakeSink::new(Mode::TransientThenOk(AtomicU32::new(2)));
        let (handle, worker) = SinkPipeline::new(sink.clone(), cfg());
        assert!(handle.try_enqueue(rec(0)));
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let jh = tokio::spawn(worker.run(cancel_rx));
        cancel_tx.send(true).unwrap();
        jh.await.unwrap();

        let s = handle.stats();
        assert_eq!(s.retries, 2, "two transient failures retried");
        assert_eq!(s.sent, 1, "record eventually delivered");
        assert_eq!(s.dropped, 0);
        assert_eq!(sink.delivered(), 1);
        // #519 D.2 delivery-health counters: an eventual success counts
        // one delivered batch, no failed batch, and stamps the success
        // timestamp (retried attempts are not failed batches).
        assert_eq!(s.delivered_batches, 1);
        assert_eq!(s.failed_batches, 0);
        assert!(s.last_success_unix.is_some(), "success timestamp recorded");
        assert!(s.last_failure_unix.is_none(), "no batch was dropped");
    }

    #[tokio::test]
    async fn drops_with_metric_after_exhausting_retries() {
        let sink = FakeSink::new(Mode::AlwaysTransient);
        let mut c = cfg();
        c.max_retries = 2;
        let (handle, worker) = SinkPipeline::new(sink.clone(), c);
        assert!(handle.try_enqueue(rec(0)));
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let jh = tokio::spawn(worker.run(cancel_rx));
        cancel_tx.send(true).unwrap();
        jh.await.unwrap();

        let s = handle.stats();
        assert_eq!(s.retries, 2, "retried up to the cap");
        assert_eq!(s.dropped, 1, "record dropped and counted");
        assert_eq!(s.failed_batches, 1);
        assert_eq!(s.sent, 0);
        assert!(
            s.last_error.is_some(),
            "last error recorded for the dashboard"
        );
        // #519 D.2: a dropped batch stamps the failure timestamp and
        // counts zero delivered batches.
        assert_eq!(s.delivered_batches, 0);
        assert!(s.last_failure_unix.is_some(), "failure timestamp recorded");
        assert!(s.last_success_unix.is_none(), "nothing was delivered");
    }

    #[tokio::test]
    async fn drops_permanent_error_without_retry() {
        let sink = FakeSink::new(Mode::Permanent);
        let (handle, worker) = SinkPipeline::new(sink.clone(), cfg());
        assert!(handle.try_enqueue(rec(0)));
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let jh = tokio::spawn(worker.run(cancel_rx));
        cancel_tx.send(true).unwrap();
        jh.await.unwrap();

        let s = handle.stats();
        assert_eq!(s.retries, 0, "permanent errors are not retried");
        assert_eq!(s.dropped, 1);
        assert_eq!(s.sent, 0);
    }

    #[tokio::test]
    async fn queue_full_drops_without_blocking() {
        // No worker draining — the bounded queue fills and over-capacity
        // enqueues are dropped, never blocking the caller.
        let sink = FakeSink::new(Mode::Ok);
        let mut c = cfg();
        c.queue_capacity = 2;
        let (handle, _worker) = SinkPipeline::new(sink, c);

        assert!(handle.try_enqueue(rec(0)));
        assert!(handle.try_enqueue(rec(1)));
        assert!(!handle.try_enqueue(rec(2)), "third enqueue is dropped");
        assert_eq!(handle.stats().dropped, 1);
    }

    #[tokio::test]
    async fn age_flush_delivers_a_partial_batch() {
        let sink = FakeSink::new(Mode::Ok);
        let mut c = cfg();
        c.flush_interval = Duration::from_millis(20);
        let (handle, worker) = SinkPipeline::new(sink.clone(), c);
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let jh = tokio::spawn(worker.run(cancel_rx));

        assert!(handle.try_enqueue(rec(0)));
        let flushed = wait_for(|| handle.stats().sent == 1, Duration::from_secs(2)).await;
        assert!(flushed, "a sub-ceiling record flushes on the age timer");

        cancel_tx.send(true).unwrap();
        jh.await.unwrap();
    }

    #[test]
    fn backoff_grows_exponentially_and_caps() {
        let base = Duration::from_millis(100);
        let cap = Duration::from_secs(1);
        assert_eq!(backoff(base, cap, 1), Duration::from_millis(100));
        assert_eq!(backoff(base, cap, 2), Duration::from_millis(200));
        assert_eq!(backoff(base, cap, 3), Duration::from_millis(400));
        assert_eq!(backoff(base, cap, 10), cap, "capped at max_backoff");
    }
}
