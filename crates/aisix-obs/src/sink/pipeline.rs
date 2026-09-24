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
// The retry budget is measured on the same clock the backoff sleeps on, so
// a paused-clock test drives the whole ladder rather than only its sleeps.
use tokio::time::Instant;

use super::{EventBatch, IdempotencyMarker, ObservabilitySink, SinkError, SinkRecord};
use crate::metrics::Metrics;

/// Tuning for a [`SinkPipeline`]. Defaults mirror the telemetry worker
/// (100-record batches, 5s flush) plus a time-budgeted retry.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Bound on the producer→worker queue. When full, `try_enqueue` drops
    /// the record (counted) rather than blocking the request hot path.
    pub queue_capacity: usize,
    /// Flush once this many records have buffered.
    pub max_batch: usize,
    /// Flush whatever is buffered at least this often.
    pub flush_interval: Duration,
    /// How long a [`SinkError::is_transient`] batch keeps being retried,
    /// measured from its FIRST attempt. When the budget is spent the batch
    /// is dropped (counted as `retries_exhausted`). `0` = no retry.
    ///
    /// A budget rather than an attempt count because what a receiver
    /// outage has is a duration, not a number of tries: at 200ms doubling
    /// to 5s, four attempts gave up 3.0s in, so a one-minute 503 window
    /// lost every batch that started inside it — measured, 19 batches and
    /// 824 records. Attempts are the wrong unit to express "survive an
    /// outage of length X" in, and tuning the count to reach X makes the
    /// early retries pointlessly dense.
    pub retry_budget: Duration,
    /// First retry delay; doubles each attempt up to `max_backoff`.
    pub base_backoff: Duration,
    /// Ceiling on the backoff delay, and on a `Retry-After` the sink asks
    /// for.
    pub max_backoff: Duration,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            // Deep enough to hold what arrives while one batch is being
            // retried: the retry budget is minutes now, and this queue is
            // what stands between that and losing the newest records.
            // It is also what a stalled exporter now holds in memory,
            // which is not free on the one path where records are not
            // shared between exporters: a full-capture exporter owns its
            // records, prompt and completion included.
            queue_capacity: 8192,
            max_batch: 100,
            flush_interval: Duration::from_secs(5),
            retry_budget: Duration::from_secs(300),
            base_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(30),
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
    /// Prometheus view of the drop counters below. `SinkStats` is the
    /// heartbeat's view and resets whenever an exporter is reconfigured
    /// (see its lifetime note), so it cannot answer "has this exporter
    /// ever lost data" across a config change; the counter family can.
    metrics: Option<Metrics>,
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
                self.record_drop("queue_full");
                tracing::debug!(sink = %self.name, "sink queue full; record dropped");
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.stats.add_dropped(1);
                self.record_drop("worker_stopped");
                false
            }
        }
    }

    fn record_drop(&self, reason: &str) {
        if let Some(m) = &self.metrics {
            m.record_otlp_fanout_drop(&self.name, reason, 1);
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
    metrics: Option<Metrics>,
}

impl SinkPipeline {
    /// Build a pipeline for one sink. Returns the producer handle and the
    /// worker; spawn `worker.run(cancel)` on the runtime.
    pub fn new(
        sink: Arc<dyn ObservabilitySink>,
        cfg: PipelineConfig,
    ) -> (SinkHandle, SinkPipeline) {
        Self::with_metrics(sink, cfg, None)
    }

    /// As [`Self::new`], with the Prometheus handle the fan-out counters
    /// are emitted on. `None` keeps the pipeline usable from tests and
    /// from any caller that has no recorder.
    pub fn with_metrics(
        sink: Arc<dyn ObservabilitySink>,
        cfg: PipelineConfig,
        metrics: Option<Metrics>,
    ) -> (SinkHandle, SinkPipeline) {
        let (tx, rx) = mpsc::channel(cfg.queue_capacity);
        let stats = Arc::new(SinkStats::default());
        let handle = SinkHandle {
            name: Arc::from(sink.name()),
            tx,
            stats: Arc::clone(&stats),
            metrics: metrics.clone(),
        };
        let worker = SinkPipeline {
            sink,
            cfg,
            rx,
            stats,
            metrics,
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
        // The retry loop watches the same signal, through its own handle:
        // `cancel` is borrowed by the select below for as long as each
        // iteration lasts, including the arm bodies.
        let mut shutdown = cancel.clone();

        tracing::info!(
            sink = %self.sink.name(),
            max_batch = self.cfg.max_batch,
            flush_interval_secs = self.cfg.flush_interval.as_secs(),
            retry_budget_secs = self.cfg.retry_budget.as_secs(),
            "sink pipeline started",
        );

        loop {
            tokio::select! {
                maybe = self.rx.recv() => match maybe {
                    Some(record) => {
                        buffer.push(record);
                        if buffer.len() >= self.cfg.max_batch {
                            self.flush(&mut buffer, None, &mut shutdown).await;
                        }
                    }
                    None => {
                        self.drain(&mut buffer, &mut shutdown).await;
                        tracing::info!(sink = %self.sink.name(), "sink pipeline: channel closed, exiting");
                        return;
                    }
                },
                _ = ticker.tick() => {
                    self.flush(&mut buffer, None, &mut shutdown).await;
                }
                _ = cancel.changed() => {
                    if *cancel.borrow() {
                        self.drain(&mut buffer, &mut shutdown).await;
                        tracing::info!(sink = %self.sink.name(), "sink pipeline shutting down");
                        return;
                    }
                }
            }
        }
    }

    /// Deliver everything queued, in batches of at most `max_batch`.
    ///
    /// The batch ceiling matters most here, because this is the one path
    /// that can meet a full queue: every other flush happens at or below
    /// the ceiling, while a shutdown can find thousands of records behind
    /// a receiver that has been failing for minutes. Handing all of them
    /// over as one batch is how a drain exceeds a receiver's payload
    /// limit, and a 413 is a PERMANENT error — the whole backlog would be
    /// dropped on its first attempt rather than delivered in pieces.
    async fn drain(
        &mut self,
        buffer: &mut Vec<Arc<SinkRecord>>,
        shutdown: &mut watch::Receiver<bool>,
    ) {
        // ONE deadline for the whole drain, not one per batch: batches go
        // out sequentially, so a per-batch budget multiplies by however
        // many the queue holds — 8192 records at 100 a batch is 82 of
        // them, and a receiver failing transiently would hold shutdown
        // for minutes.
        let deadline = Instant::now() + DRAIN_RETRY_BUDGET;
        loop {
            while buffer.len() < self.cfg.max_batch {
                match self.rx.try_recv() {
                    Ok(record) => buffer.push(record),
                    Err(_) => break,
                }
            }
            if buffer.is_empty() {
                return;
            }
            if Instant::now() >= deadline {
                // Attempting costs a request timeout apiece and the
                // deadline is already spent. Account what is left rather
                // than let it vanish unaccounted.
                let mut lost = buffer.len();
                buffer.clear();
                while self.rx.try_recv().is_ok() {
                    lost += 1;
                }
                self.record_drop(lost, "shutdown drain deadline reached", "worker_stopped");
                return;
            }
            self.flush(buffer, Some(deadline), shutdown).await;
        }
    }

    /// Take the buffer and deliver it as one batch (with retry). No-op when
    /// empty.
    async fn flush(
        &self,
        buffer: &mut Vec<Arc<SinkRecord>>,
        deadline: Option<Instant>,
        shutdown: &mut watch::Receiver<bool>,
    ) {
        if buffer.is_empty() {
            return;
        }
        let records = std::mem::take(buffer);
        buffer.reserve(self.cfg.max_batch);
        let count = records.len();
        let batch = EventBatch::new(records);
        self.deliver(&batch, count, deadline, shutdown).await;
    }

    /// Deliver one batch, retrying transient failures until the retry
    /// budget measured from the first attempt is spent. At-least-once: a
    /// retried batch may re-send already-accepted records (the marker is
    /// `None` for the at-least-once sinks this phase serves; offset-token
    /// sinks set their own marker later).
    ///
    /// The batches behind this one wait: one batch is in flight per sink,
    /// so a receiver that is down for minutes now holds the queue for
    /// minutes, and what overflows is dropped as `queue_full` — newest
    /// first, which is deliberate. Losing the newest records to a full
    /// queue is the same loss as losing the oldest to an exhausted retry,
    /// only bounded by a queue the operator can see filling.
    async fn deliver(
        &self,
        batch: &EventBatch,
        count: usize,
        deadline: Option<Instant>,
        shutdown: &mut watch::Receiver<bool>,
    ) {
        let marker = IdempotencyMarker::None;
        let started = Instant::now();
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
                    // One count per failed EXPORT ATTEMPT, so a sink that
                    // only ever succeeds on its third try is visible even
                    // though it never drops a record.
                    if let Some(m) = &self.metrics {
                        m.record_otlp_fanout_failure(self.sink.name());
                    }
                    // The drain's deadline is shared by every batch it
                    // still has to send, so a batch that is not the first
                    // may find little or none of it left.
                    let deadline = deadline.or_else(|| {
                        shutdown
                            .borrow()
                            .then(|| Instant::now() + DRAIN_RETRY_BUDGET)
                    });
                    if let Some(delay) = self.next_delay(&err, attempt, started, deadline) {
                        attempt += 1;
                        self.stats.add_retries(1);
                        tracing::warn!(
                            sink = %self.sink.name(),
                            attempt,
                            delay_ms = delay.as_millis() as u64,
                            retry_after = err.retry_after().is_some(),
                            draining = deadline.is_some(),
                            error = %detail,
                            "sink delivery failed; retrying",
                        );
                        // Waking on the shutdown signal rather than sleeping
                        // it out is what keeps the drain bounded: the budget
                        // shrinks the moment it arrives, and a backoff that
                        // is now tens of seconds long is not sat out.
                        sleep_or_shutdown(delay, deadline.is_some(), shutdown).await;
                        continue;
                    }
                    let reason = if err.is_transient() {
                        "retries_exhausted"
                    } else {
                        "permanent_error"
                    };
                    self.record_drop(count, &detail, reason);
                    return;
                }
            }
        }
    }

    /// How long to wait before re-attempting, or `None` when this batch is
    /// done: its budget is spent, or the failure is permanent.
    ///
    /// A `Retry-After` the sink asked for wins over the backoff ladder —
    /// retrying sooner than a throttling receiver asked for is what turns
    /// a throttle into a longer one — but it is capped by `max_backoff`
    /// and by whatever is left of the budget, so the last attempt lands on
    /// the budget boundary rather than past it.
    ///
    /// A drain passes its `deadline`, and that deadline belongs to the
    /// WHOLE drain rather than to this batch: a shutdown must not be held
    /// for the minutes the running budget is worth, nor for one drain
    /// budget per batch — batches go out sequentially, and a full queue
    /// is dozens of them.
    fn next_delay(
        &self,
        err: &SinkError,
        attempt: u32,
        started: Instant,
        deadline: Option<Instant>,
    ) -> Option<Duration> {
        if !err.is_transient() {
            return None;
        }
        let now = Instant::now();
        let mut remaining = self.cfg.retry_budget.checked_sub(now - started)?;
        if let Some(deadline) = deadline {
            remaining = remaining.min(deadline.saturating_duration_since(now));
        }
        if remaining.is_zero() {
            return None;
        }
        let delay = match err.retry_after() {
            // Floored as well as capped: `Retry-After: 0` is a legal
            // answer, and honouring it literally would re-attempt with no
            // wait at all — for the whole budget, at full rate, against a
            // receiver that has just said it is overloaded. The backoff
            // ladder cannot produce a zero delay, so only this path can.
            Some(asked) => asked.clamp(self.cfg.base_backoff, self.cfg.max_backoff),
            None => backoff(self.cfg.base_backoff, self.cfg.max_backoff, attempt + 1),
        };
        Some(delay.min(remaining))
    }

    fn record_drop(&self, count: usize, detail: &str, reason: &str) {
        self.stats.record_batch_failed();
        self.stats.add_dropped(count as u64);
        if let Some(m) = &self.metrics {
            m.record_otlp_fanout_drop(self.sink.name(), reason, count as u64);
        }
        self.stats.set_error(detail.to_string());
        tracing::warn!(
            sink = %self.sink.name(),
            dropped = count,
            reason,
            error = %detail,
            "sink delivery dropped",
        );
    }
}

/// What one batch may still spend on retries once shutdown is signalled.
/// It is what the drain could already spend before the budget became
/// minutes — four retries of a ladder that doubled 200ms to a 5s cap — so
/// shutdown is no slower than it was.
const DRAIN_RETRY_BUDGET: Duration = Duration::from_secs(3);

/// Wait out a backoff, returning early when shutdown is signalled so the
/// caller re-decides under the drain budget.
/// `draining` is the caller's own read of the signal, not a fresh one:
/// re-reading it here can see a flip that happened after `delay` was
/// computed, and then sleep out a delay sized for the running budget —
/// up to `max_backoff` — with no way to cut it short.
async fn sleep_or_shutdown(delay: Duration, draining: bool, shutdown: &mut watch::Receiver<bool>) {
    if draining {
        // The delay is already bounded by the drain budget, so there is
        // nothing left to cut short. Returning here instead would
        // re-attempt with no wait at all, and spin against the failing
        // sink for as long as the budget allowed.
        tokio::time::sleep(delay).await;
        return;
    }
    tokio::select! {
        _ = tokio::time::sleep(delay) => {}
        changed = shutdown.changed() => {
            // A closed channel will never signal again, so returning on it
            // would spin the retry loop through every backoff at once.
            if changed.is_err() {
                tokio::time::sleep(delay).await;
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
/// verbose upstream body can't flood the logs. Wide enough that a sink's
/// own 500-char detail (object URL + error source chain) survives with the
/// enum prefix — this cap is the last one before the log line / `last_error`,
/// so trimming tighter than the sinks re-hides the cause they now carry.
fn masked(err: &SinkError) -> String {
    err.to_string().chars().take(600).collect()
}

#[cfg(test)]
mod tests {
    use super::{backoff, PipelineConfig, SinkPipeline, DRAIN_RETRY_BUDGET};
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
        /// Always throttled, asking for this delay each time.
        AlwaysThrottled(Duration),
        /// Throttled once, asking for this delay, then succeed.
        ThrottledThenOk(Duration, AtomicU32),
        Permanent,
        /// Permanent failure carrying a caller-chosen detail string.
        PermanentDetail(String),
    }

    /// A configurable sink that records the batch sizes it was handed, and
    /// when each delivery attempt arrived.
    struct FakeSink {
        mode: Mode,
        batch_sizes: Mutex<Vec<usize>>,
        attempts: Mutex<Vec<tokio::time::Instant>>,
    }

    impl FakeSink {
        fn new(mode: Mode) -> Arc<Self> {
            Arc::new(Self {
                mode,
                batch_sizes: Mutex::new(Vec::new()),
                attempts: Mutex::new(Vec::new()),
            })
        }
        fn delivered(&self) -> usize {
            self.batch_sizes.lock().iter().sum()
        }
        fn attempts(&self) -> usize {
            self.attempts.lock().len()
        }
        /// Gaps between consecutive delivery attempts.
        fn gaps(&self) -> Vec<Duration> {
            self.attempts
                .lock()
                .windows(2)
                .map(|w| w[1] - w[0])
                .collect()
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
            self.attempts.lock().push(tokio::time::Instant::now());
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
                Mode::AlwaysThrottled(retry_after) => Err(SinkError::Throttled {
                    retry_after: *retry_after,
                    detail: "slow down".into(),
                }),
                Mode::ThrottledThenOk(retry_after, remaining) => {
                    if remaining.load(Ordering::Relaxed) > 0 {
                        remaining.fetch_sub(1, Ordering::Relaxed);
                        Err(SinkError::Throttled {
                            retry_after: *retry_after,
                            detail: "slow down".into(),
                        })
                    } else {
                        self.batch_sizes.lock().push(batch.len());
                        Ok(SinkAck {
                            accepted: batch.len(),
                            ..SinkAck::default()
                        })
                    }
                }
                Mode::Permanent => Err(SinkError::Permanent("bad credentials".into())),
                Mode::PermanentDetail(detail) => Err(SinkError::Permanent(detail.clone())),
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
            retry_budget: Duration::from_millis(50),
            base_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(5),
        }
    }

    /// Slack for assertions on the paused clock: a wakeup lands on the
    /// timer tick at or just after its deadline, never before it.
    const TICK: Duration = Duration::from_millis(50);

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
        // Two 1ms retries fit; the third attempt finds the budget spent.
        c.retry_budget = Duration::from_millis(2);
        let (handle, worker) = SinkPipeline::new(sink.clone(), c);
        assert!(handle.try_enqueue(rec(0)));
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let jh = tokio::spawn(worker.run(cancel_rx));
        cancel_tx.send(true).unwrap();
        jh.await.unwrap();

        let s = handle.stats();
        assert!(s.retries >= 1, "the batch was retried before being dropped");
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

    /// Drive a running pipeline holding one batch, with time paused so the
    /// retry ladder plays out instantly. The batch flushes on the count
    /// ceiling rather than on cancellation, because cancelling is what
    /// shortens the budget.
    ///
    /// Returns the virtual time the batch's whole delivery took. Stepping
    /// a second at a time is what advances the paused clock: auto-advance
    /// jumps to the earliest pending timer, and without a timer of our own
    /// the runtime is not idle while the worker sleeps.
    async fn one_batch_under_paused_time(sink: Arc<FakeSink>, cfg: PipelineConfig) -> Duration {
        tokio::time::pause();
        let mut cfg = cfg;
        cfg.max_batch = 1;
        let (handle, worker) = SinkPipeline::new(sink.clone(), cfg);
        assert!(handle.try_enqueue(rec(0)));
        // Closing the channel lets the worker exit once the batch is done.
        drop(handle);
        let started = tokio::time::Instant::now();
        let (_keep_alive, cancel_rx) = watch::channel(false);
        let worker = tokio::spawn(worker.run(cancel_rx));
        let mut stepped = Duration::ZERO;
        while !worker.is_finished() {
            assert!(
                stepped < Duration::from_secs(3_600),
                "the batch must reach an outcome"
            );
            tokio::time::sleep(Duration::from_secs(1)).await;
            stepped += Duration::from_secs(1);
        }
        worker.await.unwrap();
        started.elapsed()
    }

    /// The defect: four attempts over a 3.0s ladder meant a receiver
    /// outage longer than three seconds lost every batch that started
    /// inside it — a 60s window of 503s dropped 19 batches / 824 records,
    /// each 3.0s after its first attempt. Retrying is budgeted in time
    /// now, so a batch survives an outage of any length up to the budget.
    #[tokio::test]
    async fn a_transient_receiver_is_retried_for_the_whole_budget() {
        let sink = FakeSink::new(Mode::AlwaysTransient);
        let elapsed = one_batch_under_paused_time(sink.clone(), PipelineConfig::default()).await;

        let budget = PipelineConfig::default().retry_budget;
        assert!(
            elapsed >= budget && elapsed < budget + Duration::from_secs(5),
            "the batch is given up on at the budget, not before or after: {elapsed:?}"
        );
        // The ladder tops out at `max_backoff`, so the attempts are spread
        // across the budget rather than crowded into its first seconds.
        assert!(
            sink.attempts() > 10,
            "attempts across the budget: {}",
            sink.attempts()
        );
        assert!(
            sink.gaps()
                .iter()
                .all(|g| *g <= PipelineConfig::default().max_backoff + TICK),
            "no gap may exceed max_backoff: {:?}",
            sink.gaps()
        );
        assert_eq!(sink.delivered(), 0, "nothing was ever accepted");
    }

    /// A receiver that says how long to wait is telling us something our
    /// own ladder cannot know. Retrying sooner is what turns a throttle
    /// into a longer one.
    #[tokio::test]
    async fn a_retry_after_sets_the_next_attempt() {
        let sink = FakeSink::new(Mode::ThrottledThenOk(
            Duration::from_secs(10),
            AtomicU32::new(1),
        ));
        one_batch_under_paused_time(sink.clone(), PipelineConfig::default()).await;

        assert_eq!(sink.delivered(), 1, "delivered on the second attempt");
        let [gap] = sink.gaps()[..] else {
            panic!("one retry: {:?}", sink.gaps())
        };
        // Not the 200ms first rung of the backoff ladder. The upper bound
        // is the clock's step, not slack in the mechanism.
        assert!(
            gap >= Duration::from_secs(10) && gap < Duration::from_secs(12),
            "the wait is what the receiver asked for: {gap:?}",
        );
    }

    /// …but it does not get to park the queue behind it for as long as it
    /// likes: an hour of `Retry-After` is one batch holding a pipeline for
    /// an hour, and everything behind it dropping as `queue_full`.
    #[tokio::test]
    async fn a_retry_after_is_capped_at_the_backoff_ceiling() {
        let sink = FakeSink::new(Mode::AlwaysThrottled(Duration::from_secs(3_600)));
        let cfg = PipelineConfig {
            retry_budget: Duration::from_secs(120),
            ..PipelineConfig::default()
        };
        one_batch_under_paused_time(sink.clone(), cfg).await;

        assert!(
            sink.gaps()
                .iter()
                .all(|g| *g <= PipelineConfig::default().max_backoff + TICK),
            "capped at max_backoff: {:?}",
            sink.gaps()
        );
        assert!(
            sink.attempts() >= 4,
            "a capped wait still gets several attempts inside the budget: {}",
            sink.attempts()
        );
    }

    /// The shutdown drain is the one flush that can meet a FULL queue —
    /// every other one happens at or below the ceiling, while a drain can
    /// find thousands of records behind a receiver that has been failing
    /// for minutes. Handing all of them over as one batch is how a drain
    /// exceeds a receiver's payload limit, and a 413 is PERMANENT: the
    /// whole backlog would be dropped on its first attempt.
    #[tokio::test]
    async fn the_shutdown_drain_keeps_to_the_batch_ceiling() {
        const QUEUED: usize = 250;
        let sink = FakeSink::new(Mode::Ok);
        let mut c = cfg();
        c.max_batch = 10;
        let (handle, worker) = SinkPipeline::new(sink.clone(), c);
        for i in 0..QUEUED {
            assert!(handle.try_enqueue(rec(i as u32)));
        }
        let (cancel_tx, cancel_rx) = watch::channel(false);
        cancel_tx.send(true).unwrap();
        worker.run(cancel_rx).await;

        let sizes = sink.batch_sizes.lock().clone();
        assert!(
            sizes.iter().all(|n| *n <= 10),
            "no batch may exceed the ceiling: {sizes:?}"
        );
        assert_eq!(
            sizes.iter().sum::<usize>(),
            QUEUED,
            "and the whole backlog still goes: {sizes:?}"
        );
    }

    /// `Retry-After: 0` is a legal answer, and the one delay the backoff
    /// ladder can never produce. Taken literally it re-attempts with no
    /// wait for the whole budget — full rate against a receiver that just
    /// said it was overloaded.
    #[tokio::test]
    async fn a_zero_retry_after_does_not_become_a_spin() {
        let sink = FakeSink::new(Mode::AlwaysThrottled(Duration::ZERO));
        let cfg = PipelineConfig {
            retry_budget: Duration::from_secs(10),
            ..PipelineConfig::default()
        };
        one_batch_under_paused_time(sink.clone(), cfg).await;

        let base = PipelineConfig::default().base_backoff;
        let gaps = sink.gaps();
        // All but the last: the final wait is trimmed to what is left of
        // the budget, so that one attempt lands on the boundary.
        assert!(
            gaps[..gaps.len() - 1].iter().all(|g| *g >= base),
            "every wait must be at least the base backoff: {gaps:?}"
        );
        // 10s of budget at >=200ms a try; the ladder would be ~50 at the
        // floor, and unbounded without it.
        assert!(
            sink.attempts() <= 51,
            "attempts stay bounded by the floor: {}",
            sink.attempts()
        );
    }

    /// The drain budget belongs to the drain, not to each of its batches.
    /// They go out sequentially, so a per-batch budget multiplies by how
    /// many the queue holds — a full one is dozens — and a receiver
    /// failing transiently would hold shutdown for minutes.
    #[tokio::test]
    async fn the_whole_shutdown_drain_shares_one_deadline() {
        const QUEUED: usize = 250;
        tokio::time::pause();
        let sink = FakeSink::new(Mode::AlwaysTransient);
        // 25 batches.
        let c = PipelineConfig {
            max_batch: 10,
            ..PipelineConfig::default()
        };
        let (handle, worker) = SinkPipeline::new(sink.clone(), c);
        for i in 0..QUEUED {
            assert!(handle.try_enqueue(rec(i as u32)));
        }
        let (cancel_tx, cancel_rx) = watch::channel(false);
        cancel_tx.send(true).unwrap();
        let started = tokio::time::Instant::now();
        worker.run(cancel_rx).await;

        assert!(
            started.elapsed() <= DRAIN_RETRY_BUDGET + TICK,
            "the whole drain, not each batch: {:?}",
            started.elapsed()
        );
        // And nothing vanishes unaccounted: what the deadline cut short
        // is counted as lost, not silently forgotten.
        assert_eq!(handle.stats().dropped, QUEUED as u64);
    }

    /// The budget is minutes now, so a shutdown that waited on it would
    /// hold the process for minutes. It waits no longer than the old
    /// ladder could already take.
    #[tokio::test]
    async fn shutdown_does_not_wait_out_the_retry_budget() {
        tokio::time::pause();
        let sink = FakeSink::new(Mode::AlwaysTransient);
        let (handle, worker) = SinkPipeline::new(sink.clone(), PipelineConfig::default());
        assert!(handle.try_enqueue(rec(0)));
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let started = tokio::time::Instant::now();
        let jh = tokio::spawn(worker.run(cancel_rx));
        cancel_tx.send(true).unwrap();
        jh.await.unwrap();

        assert!(
            started.elapsed() <= DRAIN_RETRY_BUDGET + TICK,
            "the drain must not sit out the running retry budget: {:?}",
            started.elapsed()
        );
        assert_eq!(
            handle.stats().dropped,
            1,
            "the batch is accounted, not lost silently"
        );
    }

    #[tokio::test]
    async fn long_error_detail_keeps_its_cause_in_last_error() {
        // The sinks put the actionable cause (DNS/TLS failure) at the END of
        // a detail that can run ~500 chars — a partitioned object URL alone
        // is ~180. `masked` is the last cap before the warn line and
        // `last_error`; trimming tighter than the sinks' own caps re-hides
        // the cause they now carry (this fails with the old 200-char cap).
        let host = "h".repeat(400);
        let sink = FakeSink::new(Mode::PermanentDetail(format!(
            "PUT https://{host}/k: dns error: failed to lookup address information"
        )));
        let (handle, worker) = SinkPipeline::new(sink.clone(), cfg());
        assert!(handle.try_enqueue(rec(0)));
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let jh = tokio::spawn(worker.run(cancel_rx));
        cancel_tx.send(true).unwrap();
        jh.await.unwrap();

        let last = handle.stats().last_error.expect("error recorded");
        assert!(
            last.contains("failed to lookup address information"),
            "the cause at the end of the detail must survive masking, got: {last}"
        );
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
