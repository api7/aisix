//! Sender worker for the CP-side `/dp/telemetry` surface.
//!
//! Per prd-09a §9A.7B Phase 1:
//!
//! - Proxy handlers call [`UsageSink::try_emit`] (defined in aisix-obs)
//!   to push one event per chat completion onto an mpsc channel.
//! - This worker drains the channel, batches up to [`MAX_BATCH`]
//!   events or every [`FLUSH_INTERVAL`] (whichever fires first),
//!   POSTs the batch as `{ events: [...] }` to the CP's
//!   `/dp/telemetry` URL, and logs the outcome.
//! - On HTTP error the batch is re-sent — but only while the control
//!   plane's own answer proves re-sending cannot double-count. Every
//!   POST carries [`USAGE_BATCH_ID_HEADER`], the control plane records
//!   that id in the same transaction as the usage rows, and it answers
//!   every response from that handler with [`USAGE_BATCH_DEDUP_HEADER`].
//!   A failure whose response lacks that header — or carries it with an
//!   undefined value — is dropped, exactly as every failure was before.
//!   There is still no persistent disk queue: a batch is given up on once
//!   its OLDEST event is [`RETRY_BUDGET`] old (which can be before its
//!   first attempt, since the sender is single in-flight), or once the
//!   control plane has answered and refused it [`MAX_ANSWERED_FAILURES`]
//!   times in a row. The `received_at` column on the cp-api side records
//!   when CP saw the row, so dashboards distinguish "DP never sent" from
//!   "DP sent but CP rejected" via log correlation.
//!
//! mTLS: the sender presents the same on-disk bundle the heartbeat
//! worker uses. cp-api derives `env_id` and `dp_id` from the peer
//! cert SAN URI, so the request body doesn't carry them — same wire
//! shape as `/dp/heartbeat`.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tokio::time::Instant;
use uuid::Uuid;

use aisix_obs::{Metrics, UsageEvent, UsageEventLabels, UsageSink};

use crate::heartbeat::MtlsBundle;

/// Maximum number of events accumulated per outbound POST. Pinned in
/// code rather than config — at >100 events/batch the request body
/// approaches gin's default `MaxMultipartMemory` plumbing on the
/// receiving side, and we'd rather flush more often than tune that.
const MAX_BATCH: usize = 100;

/// Cadence at which the worker flushes whatever it has buffered, even
/// if the buffer hasn't filled. Keeps fresh requests visible on
/// /usage and /logs within ~5s end-to-end.
const FLUSH_INTERVAL: Duration = Duration::from_secs(5);

/// In-memory bound on the proxy → worker channel. Beyond it `try_emit`
/// warns and the event is dropped (`sink_full`) — telemetry must not
/// back-pressure the request hot path.
///
/// Sized for the control plane being SLOW rather than for the steady
/// rate: one POST is in flight at a time — a batch being re-sent holds
/// that slot for as long as it keeps failing — so the queue is the only
/// thing holding the traffic that arrives meanwhile, and a batch this
/// worker gives up on is gone for good. A stability round measured 1.1–1.8s per batch
/// for 8s at 173 req/s — ~1.4k events behind a queue that held 1024, so
/// 793 were dropped. At 16384 the same stall is absorbed whole, and the
/// queue only overflows once a sustained arrival rate outruns delivery
/// for minutes rather than seconds.
///
/// Cost is bounded by occupancy, not by the bound: the channel allocates
/// in small blocks as events are pushed, so a queue that never fills
/// never holds the memory for one that did.
const QUEUE_CAPACITY: usize = 16_384;

/// Request header carrying the id this sender mints once per batch. Every
/// re-send of that batch repeats the value verbatim, which is what lets the
/// control plane recognise a batch it has already committed and answer it
/// without writing anything twice.
const USAGE_BATCH_ID_HEADER: &str = "X-Aisix-Usage-Batch-Id";

/// Response header the control plane sets on EVERY response its telemetry
/// handler produces — success and failure alike. It is the only evidence
/// this sender has that a re-send cannot double-count.
const USAGE_BATCH_DEDUP_HEADER: &str = "X-Aisix-Usage-Batch-Dedup";

/// The one value [`USAGE_BATCH_DEDUP_HEADER`] is defined to carry.
///
/// Matched exactly rather than by presence: the header is a capability
/// assertion, and the protocol pins its value, so anything else is a
/// control plane this sender does not understand — including a future one
/// that uses the value to say something narrower. Reading such a header as
/// "de-duplication is on" is the one mistake that bills twice.
const USAGE_BATCH_DEDUP_ENABLED: &str = "1";

/// First wait before a batch is re-sent; doubles per attempt up to
/// [`RETRY_MAX_BACKOFF`].
const RETRY_INITIAL_BACKOFF: Duration = Duration::from_secs(1);

/// Ceiling on the wait between re-sends.
const RETRY_MAX_BACKOFF: Duration = Duration::from_secs(30);

/// How long a batch may be re-sent, measured from its OLDEST event's
/// `occurred_at` — NOT from its first attempt.
///
/// Bounded by billing, not by patience: cp-api's `usagepush` seals each
/// UTC hour's billing counters at `hour start + settleDelay + 1h`
/// (`settleDelay` is 1 hour today), and an event whose `occurred_at` falls
/// at the end of an hour is already near `hour start + 1h` when it is
/// emitted. A budget equal to `settleDelay` would therefore leave zero
/// margin: the rows land, and the hour they belong to has already been
/// billed without them. 30 minutes of event age keeps 30 minutes of margin.
///
/// Measuring from the first ATTEMPT would only hold that margin for the
/// batch at the head of the queue: this sender is single in-flight, so a
/// batch waiting behind one that burned the whole budget is already half an
/// hour old before its own first attempt, and its own budget would carry it
/// past the seal. A batch that is already past the budget when it reaches
/// the head is therefore dropped WITHOUT an attempt. Change either side
/// only together with the other's comment — the control-plane half is in
/// `internal/cpapi/usagepush` and prd-09 §9.6.6.
const RETRY_BUDGET: Duration = Duration::from_secs(30 * 60);

/// Consecutive failures that CARRIED A RESPONSE before a batch is given up
/// on, whatever the budget has left.
///
/// The control plane classifies a permanently unacceptable batch as a
/// non-retryable 4xx, which this sender already drops at once. This is the
/// safety net for the case that classification misses: a control plane
/// answering `5xx` for a batch it can never accept would otherwise hold the
/// single in-flight channel for the whole budget, and the queue behind it —
/// every event the gateway records meanwhile — overflows long before that,
/// which is a far larger loss than the batch itself.
///
/// A failure with NO response does not count against this cap: that is the
/// control-plane outage this feature exists for, and it is bounded by the
/// budget instead.
const MAX_ANSWERED_FAILURES: usize = 8;

/// `aisix_usage_event_drops_total{reason}` for a batch given up on without
/// ever being re-sent: the control plane answered without
/// [`USAGE_BATCH_DEDUP_HEADER`] (so re-sending could double-count), the
/// failure was not retryable, or the gateway is shutting down.
const DROP_SEND_FAILED: &str = "send_failed";

/// `aisix_usage_event_drops_total{reason}` for a batch that ran out of room
/// to keep trying: its events aged past [`RETRY_BUDGET`] (with or without
/// an attempt ever being made), or it was answered and refused
/// [`MAX_ANSWERED_FAILURES`] times in a row. Distinct from
/// [`DROP_SEND_FAILED`], which says the very first failure could not be
/// re-sent at all.
const DROP_RETRY_BUDGET_EXHAUSTED: &str = "retry_budget_exhausted";

/// Path the telemetry worker POSTs to, under `managed.cp_base_url`.
/// Derived from the heartbeat URL by swapping the suffix, so the two
/// stay in lock-step on a `cp_base_url` change.
pub const TELEMETRY_PATH: &str = "/dp/telemetry";

/// Configuration for the sender. Mirrors `HeartbeatConfig` — the URL
/// is the absolute `/dp/telemetry` endpoint on cp-api, the bundle is
/// the externally provisioned on-disk mTLS material, and `interval`
/// is the flush cadence (kept overridable so tests can speed up).
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub url: String,
    pub interval: Duration,
    pub mtls: MtlsBundle,
}

impl TelemetryConfig {
    /// Build with default flush interval (5s).
    pub fn new(url: String, mtls: MtlsBundle) -> Self {
        Self {
            url,
            interval: FLUSH_INTERVAL,
            mtls,
        }
    }
}

/// Spawn the worker. Returns:
///   - a [`UsageSink`] the proxy uses to enqueue events;
///   - a [`tokio::task::JoinHandle`] the caller awaits at shutdown
///     so the final in-flight batch drains cleanly.
///
/// The worker stops when `cancel` flips to `true` AND the channel is
/// drained (one final flush so we don't lose the tail). Errors during
/// individual flushes are logged, not propagated — same contract as
/// heartbeat::spawn.
pub fn spawn(
    cfg: TelemetryConfig,
    metrics: Metrics,
    mut cancel: watch::Receiver<bool>,
) -> (UsageSink, tokio::task::JoinHandle<()>) {
    let (tx, rx) = tokio::sync::mpsc::channel(QUEUE_CAPACITY);
    // Attached here as well as at the wiring point: this function holds the
    // handle, so a caller cannot end up with a sink whose queue drops are
    // invisible while the worker's are counted. (The wiring point still
    // attaches it, because the no-control-plane branch builds a
    // `UsageSink::disabled()` that never comes through here and still has
    // `sink_disabled` to report.)
    let sink = UsageSink::new(tx).with_metrics(metrics.clone());
    let handle = tokio::spawn(async move {
        run(cfg, metrics, rx, &mut cancel).await;
    });
    (sink, handle)
}

async fn run(
    cfg: TelemetryConfig,
    metrics: Metrics,
    mut rx: tokio::sync::mpsc::Receiver<UsageEvent>,
    cancel: &mut watch::Receiver<bool>,
) {
    let client = match build_client(&cfg.mtls) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            tracing::error!(error = %e, "telemetry: build mTLS client failed; worker disabled");
            return;
        }
    };
    let mut ticker = tokio::time::interval(cfg.interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut buffer: Vec<UsageEvent> = Vec::with_capacity(MAX_BATCH);
    let mut state = SenderState::default();

    tracing::info!(
        url = %cfg.url,
        flush_interval_secs = cfg.interval.as_secs(),
        max_batch = MAX_BATCH,
        retry_budget_secs = RETRY_BUDGET.as_secs(),
        "telemetry sender started (mTLS)",
    );

    loop {
        tokio::select! {
            // New event from the proxy. Buffer it; flush if we hit
            // the batch ceiling so a steady high-throughput stream
            // doesn't starve cp-api on a 5s cadence.
            maybe_event = rx.recv() => {
                match maybe_event {
                    Some(event) => {
                        buffer.push(event);
                        if buffer.len() >= MAX_BATCH {
                            flush(&client, &cfg, &metrics, &mut state, &mut buffer, Retry::Allowed, cancel).await;
                        }
                    }
                    None => {
                        // All senders dropped — proxy is shutting down.
                        // Drain anything left and exit.
                        flush(&client, &cfg, &metrics, &mut state, &mut buffer, Retry::Disabled, cancel).await;
                        tracing::info!("telemetry sender: channel closed, exiting");
                        return;
                    }
                }
            }
            _ = ticker.tick() => {
                if !buffer.is_empty() {
                    fill_ready_batch(&mut buffer, &mut rx);
                    flush(&client, &cfg, &metrics, &mut state, &mut buffer, Retry::Allowed, cancel).await;
                }
            }
            _ = cancel.changed() => {}
        }
        // Checked here rather than inside the `cancel` branch because a
        // flush above may have consumed the change notification while it
        // was waiting to re-send — `changed()` reports each version once,
        // so a branch that only fires on the notification would never see
        // the shutdown and the drain would never run.
        if *cancel.borrow() {
            // Final drain — post whatever is still queued, in
            // batches of at most MAX_BATCH. This is the one path
            // that can meet a FULL queue (every other flush
            // happens at or below the ceiling), and one POST of
            // everything queued is exactly what the ceiling
            // exists to prevent: cp-api rejects an oversized body
            // and the whole backlog is gone, unretried.
            //
            // `Retry::Disabled`: every batch here gets one attempt. Waiting
            // out a backoff would hold the whole shutdown open for a
            // control plane that is already failing.
            loop {
                fill_ready_batch(&mut buffer, &mut rx);
                if buffer.is_empty() {
                    break;
                }
                flush(
                    &client,
                    &cfg,
                    &metrics,
                    &mut state,
                    &mut buffer,
                    Retry::Disabled,
                    cancel,
                )
                .await;
            }
            tracing::info!("telemetry sender shutting down");
            return;
        }
    }
}

/// Whether a failed batch may be re-sent, or must be given up on after one
/// attempt. Shutdown takes [`Retry::Disabled`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Retry {
    Allowed,
    Disabled,
}

/// The one piece of state that outlives a single batch: what the most
/// recent exchange with the control plane said about batch de-duplication.
#[derive(Debug, Default)]
struct SenderState {
    /// Whether the most recent response this sender received carried
    /// [`USAGE_BATCH_DEDUP_HEADER`]. Starts `false` — until a response
    /// proves otherwise, this sender behaves exactly as it did before
    /// re-sending existed.
    dedup_seen: bool,
}

impl SenderState {
    /// Decide whether the batch that just produced `failure` may be re-sent,
    /// and record what the exchange said about the control plane.
    ///
    /// **Judged per response, never cached across exchanges.** dp-manager
    /// runs several replicas, a rolling upgrade puts old and new pods behind
    /// one Service, and a control-plane rollback to a version without batch
    /// de-duplication is supported. A "we saw the header once" flag would
    /// therefore keep re-sending to a pod that counts every re-send again.
    /// So: a response that carries the header and a retryable status may be
    /// re-sent; a response without the header may not, and clears the flag.
    ///
    /// A failure with NO response — connect error, timeout — carries no
    /// evidence at all, and that is the failure a control-plane outage
    /// actually produces. It falls back to the flag: re-send only if the
    /// most recent response this sender received carried the header and
    /// nothing since has come back without it.
    fn may_resend(&mut self, failure: &SendFailure) -> bool {
        match failure {
            SendFailure::Response {
                dedup, retryable, ..
            } => {
                self.dedup_seen = *dedup;
                *dedup && *retryable
            }
            SendFailure::NoResponse { .. } => self.dedup_seen,
        }
    }
}

fn fill_ready_batch(
    buffer: &mut Vec<UsageEvent>,
    rx: &mut tokio::sync::mpsc::Receiver<UsageEvent>,
) {
    while buffer.len() < MAX_BATCH {
        match rx.try_recv() {
            Ok(event) => buffer.push(event),
            Err(_) => break,
        }
    }
}

/// POST one batch and clear the buffer, re-sending it while that is safe.
/// Errors are logged, not propagated — telemetry losses must not stall the
/// worker.
async fn flush(
    client: &reqwest::Client,
    cfg: &TelemetryConfig,
    metrics: &Metrics,
    state: &mut SenderState,
    buffer: &mut Vec<UsageEvent>,
    retry: Retry,
    cancel: &mut watch::Receiver<bool>,
) {
    if buffer.is_empty() {
        return;
    }
    // Move events out into the request body; clear the buffer
    // unconditionally so a hung CP doesn't grow the buffer
    // indefinitely (worst case we drop the batch on error).
    let events: Vec<UsageEvent> = std::mem::take(buffer);
    buffer.reserve(MAX_BATCH);

    deliver(client, cfg, metrics, state, &events, retry, cancel).await;
}

/// Deliver one batch, re-sending the SAME batch — same id, same events, same
/// order — until it lands, until re-sending stops being safe, or until
/// [`RETRY_BUDGET`] runs out. Nothing else is sent meanwhile: the worker is
/// single in-flight by design, and arriving events queue up behind it.
async fn deliver(
    client: &reqwest::Client,
    cfg: &TelemetryConfig,
    metrics: &Metrics,
    state: &mut SenderState,
    events: &[UsageEvent],
    retry: Retry,
    cancel: &mut watch::Receiver<bool>,
) {
    let batch_id = Uuid::new_v4();
    let count = events.len();
    let budget = remaining_budget(events);
    if budget.is_zero() {
        // Already past the billing window the budget protects — see
        // `RETRY_BUDGET`. Sending it would land rows in an hour that has been
        // billed without them, so it is given up on unattempted.
        drop_batch(
            metrics,
            DROP_RETRY_BUDGET_EXHAUSTED,
            events,
            batch_id,
            0,
            None,
        );
        return;
    }
    let deadline = Instant::now() + budget;
    let mut backoff = RETRY_INITIAL_BACKOFF;
    let mut attempts = 0usize;
    let mut answered_failures = 0usize;
    let mut resending = false;
    // Flipped when shutdown interrupts a backoff: the batch in hand gets one
    // final attempt, and then this worker stops holding the drain open.
    let mut last_attempt = retry == Retry::Disabled;

    loop {
        attempts += 1;
        let failure = match send(client, cfg, batch_id, events).await {
            Attempt::Delivered { dedup, rejected } => {
                state.dedup_seen = dedup;
                // Delivered and final: the control plane judged these events
                // and would refuse them again, so there is nothing to re-send.
                // What there is, is a hole in the usage record that nothing
                // else reports.
                if rejected > 0 {
                    metrics.record_usage_events_rejected(rejected);
                    tracing::warn!(
                        %batch_id,
                        count,
                        rejected,
                        "control plane rejected usage events in an accepted telemetry batch (events dropped)",
                    );
                }
                if resending {
                    tracing::warn!(
                        %batch_id,
                        count,
                        attempts,
                        "telemetry batch delivered after re-sending",
                    );
                } else {
                    tracing::debug!(count, "telemetry batch flushed");
                }
                return;
            }
            Attempt::Failed(failure) => failure,
        };

        if matches!(failure, SendFailure::Response { .. }) {
            answered_failures += 1;
        }
        let may_resend = state.may_resend(&failure);
        if !may_resend || last_attempt {
            drop_batch(
                metrics,
                DROP_SEND_FAILED,
                events,
                batch_id,
                attempts,
                Some(&failure),
            );
            return;
        }

        let now = Instant::now();
        if answered_failures >= MAX_ANSWERED_FAILURES || now >= deadline {
            drop_batch(
                metrics,
                DROP_RETRY_BUDGET_EXHAUSTED,
                events,
                batch_id,
                attempts,
                Some(&failure),
            );
            return;
        }
        if !resending {
            resending = true;
            tracing::warn!(
                %batch_id,
                count,
                error = %failure,
                budget_secs = RETRY_BUDGET.as_secs(),
                "telemetry batch failed; re-sending (the control plane de-duplicates by batch id)",
            );
        }

        // Never sleep past the budget: the last attempt happens at the
        // deadline, not one backoff after it.
        let wait = backoff.min(deadline - now);
        tokio::select! {
            _ = tokio::time::sleep(wait) => {}
            // `Err` means the watch itself is gone, which only happens as
            // the process tears down. Treated as shutdown, because a
            // channel that resolves instantly forever would otherwise turn
            // the backoff into a hot loop of POSTs.
            changed = cancel.changed() => {
                if changed.is_err() {
                    last_attempt = true;
                }
            }
        }
        if *cancel.borrow() {
            last_attempt = true;
        }
        backoff = (backoff * 2).min(RETRY_MAX_BACKOFF);
    }
}

/// What is left of [`RETRY_BUDGET`] for this batch, measured from its OLDEST
/// event rather than from now: the sender is single in-flight, so a batch
/// that waited behind a long re-send is already old when it first goes out,
/// and the budget exists to keep its rows inside the control plane's billing
/// window rather than to cap how long this worker tries. `Duration::ZERO`
/// means the batch is already past it.
///
/// An event whose `occurred_at` does not parse cannot be aged, so it does not
/// shorten anything — what to do with a malformed timestamp is the control
/// plane's call, by its own rules.
fn remaining_budget(events: &[UsageEvent]) -> Duration {
    let now = chrono::Utc::now();
    let Some(oldest) = events
        .iter()
        .filter_map(|event| chrono::DateTime::parse_from_rfc3339(&event.occurred_at).ok())
        .map(|at| at.with_timezone(&chrono::Utc))
        .min()
    else {
        return RETRY_BUDGET;
    };
    // `to_std` fails on a negative span — an event stamped in the future,
    // which ages nothing.
    let age = (now - oldest).to_std().unwrap_or_default();
    RETRY_BUDGET.saturating_sub(age)
}

/// Give up on a batch: count every event it carried against the existing
/// drop counter under `reason`, and log the one line that closes the retry
/// the `re-sending` line opened.
///
/// Attribution is thinner here than at the queue: `UsageSink::try_emit`
/// takes the model and ProviderKey dimensions from the emitting handler's
/// label set, and the queue carries events, not label sets. The member pair
/// comes off the event exactly as it does there — same fields, same
/// `unknown` fallback — so "whose usage records were lost" stays answerable
/// for the one dimension the event itself supplies.
fn drop_batch(
    metrics: &Metrics,
    reason: &'static str,
    events: &[UsageEvent],
    batch_id: Uuid,
    attempts: usize,
    failure: Option<&SendFailure>,
) {
    for event in events {
        metrics.record_usage_event_drop(
            reason,
            UsageEventLabels {
                user_id: non_empty(&event.user_id),
                user_name: non_empty(&event.user_name),
                ..UsageEventLabels::default()
            },
        );
    }
    let error = match failure {
        Some(failure) => failure.to_string(),
        // `attempts = 0` beside it: there is no failure to name, because
        // nothing was sent.
        None => "older than the retry budget; not attempted".to_string(),
    };
    tracing::warn!(
        %batch_id,
        count = events.len(),
        attempts,
        reason,
        error,
        // Wording held verbatim from before re-sending existed: the
        // control-plane e2e log scans allowlist this line by its exact text
        // (`e2e/cases/dp_harness_test.go`, `dashboard/tests/e2e/dp-harness.ts`),
        // and an offline window is expected to produce it. The two lines that
        // bracket a re-send are new text and need allowlisting there.
        "telemetry batch failed (events dropped)",
    );
}

fn non_empty(value: &str) -> &str {
    if value.is_empty() {
        "unknown"
    } else {
        value
    }
}

#[derive(Debug, Serialize)]
struct TelemetryBody<'a> {
    events: &'a [UsageEvent],
}

/// The control plane's `2xx` answer to a batch. Decoded loosely — a newer
/// control plane may add fields or drop this one.
#[derive(Debug, Deserialize)]
struct TelemetryAnswer {
    #[serde(default)]
    rejected: u64,
}

/// What one POST of a batch came back as.
#[derive(Debug)]
enum Attempt {
    /// The control plane accepted the batch — including the case where it
    /// recognised a batch id it had already committed and wrote nothing.
    /// `dedup` reports whether the response carried
    /// [`USAGE_BATCH_DEDUP_HEADER`].
    Delivered {
        dedup: bool,
        /// Events the control plane refused inside the accepted batch —
        /// the `rejected` count of its `{accepted, rejected}` answer.
        rejected: u64,
    },
    Failed(SendFailure),
}

#[derive(Debug)]
enum SendFailure {
    /// The control plane answered. `dedup` says whether that response
    /// carried [`USAGE_BATCH_DEDUP_HEADER`]; `retryable` says whether the
    /// status is one a re-send could clear.
    Response {
        dedup: bool,
        retryable: bool,
        error: anyhow::Error,
    },
    /// No response at all — connect failure, timeout, TLS error. This
    /// exchange says nothing about what the control plane supports.
    NoResponse { error: anyhow::Error },
}

impl std::fmt::Display for SendFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendFailure::Response { error, .. } | SendFailure::NoResponse { error } => {
                write!(f, "{error:#}")
            }
        }
    }
}

/// A status a re-send can clear: the control plane was there but could not
/// take the batch right now. Everything else — a `400` for a malformed
/// batch id or body, and every other 4xx — is the control plane refusing
/// this batch, which no amount of re-sending changes.
fn is_retryable(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
}

async fn send(
    client: &reqwest::Client,
    cfg: &TelemetryConfig,
    batch_id: Uuid,
    events: &[UsageEvent],
) -> Attempt {
    let resp = client
        .post(&cfg.url)
        // Same as /dp/heartbeat — no Authorization header; cp-api
        // derives identity from the peer cert SAN URI.
        .header(USAGE_BATCH_ID_HEADER, batch_id.to_string())
        .json(&TelemetryBody { events })
        .send()
        .await
        .with_context(|| format!("POST {}", cfg.url));
    let resp = match resp {
        Ok(resp) => resp,
        Err(error) => return Attempt::Failed(SendFailure::NoResponse { error }),
    };

    // Read off THIS response, never remembered: see `SenderState::may_resend`.
    let dedup = resp
        .headers()
        .get(USAGE_BATCH_DEDUP_HEADER)
        .is_some_and(|value| value.as_bytes() == USAGE_BATCH_DEDUP_ENABLED.as_bytes());
    let status = resp.status();
    if status.is_success() {
        // A body that is not the expected answer reads as nothing rejected:
        // the batch was accepted, and there is no count to report.
        let rejected = resp
            .text()
            .await
            .ok()
            .and_then(|body| serde_json::from_str::<TelemetryAnswer>(&body).ok())
            .map_or(0, |answer| answer.rejected);
        return Attempt::Delivered { dedup, rejected };
    }
    let body = resp.text().await.unwrap_or_default();
    Attempt::Failed(SendFailure::Response {
        dedup,
        retryable: is_retryable(status),
        error: anyhow!(
            "telemetry {} returned {} — {}",
            cfg.url,
            status,
            body.trim().chars().take(200).collect::<String>()
        ),
    })
}

/// Build the mTLS reqwest client. Identical shape to heartbeat's
/// build_client — extracted out of heartbeat.rs would be nicer but
/// pulling that into a shared module is post-MVP polish and would
/// expand this PR's scope. The two clients are independently
/// constructed so a botched bundle on one path doesn't cascade.
fn build_client(mtls: &MtlsBundle) -> anyhow::Result<reqwest::Client> {
    let ca_pem = std::fs::read(&mtls.ca_cert_path)
        .with_context(|| format!("read {}", mtls.ca_cert_path.display()))?;
    let cert_pem = std::fs::read(&mtls.client_cert_path)
        .with_context(|| format!("read {}", mtls.client_cert_path.display()))?;
    let key_pem = std::fs::read(&mtls.client_key_path)
        .with_context(|| format!("read {}", mtls.client_key_path.display()))?;

    // Ensure a newline separates the two PEM blocks (see heartbeat.rs).
    let mut identity_pem = Vec::with_capacity(cert_pem.len() + key_pem.len() + 1);
    identity_pem.extend_from_slice(&key_pem);
    if !key_pem.ends_with(b"\n") {
        identity_pem.push(b'\n');
    }
    identity_pem.extend_from_slice(&cert_pem);
    let identity = reqwest::Identity::from_pem(&identity_pem)
        .context("build mTLS Identity from client cert + key")?;

    let ca = reqwest::Certificate::from_pem(&ca_pem).context("parse CA certificate")?;

    let mut builder = aisix_gateway::client_builder()
        .timeout(Duration::from_secs(10))
        .user_agent(format!("aisix-dp/{}", &*aisix_core::BUILD_VERSION))
        .identity(identity)
        .add_root_certificate(ca)
        // Pin HTTP/1.1 — see heartbeat::build_client. dp-manager cmux
        // routes one TLS port to gRPC (h2) vs REST (http1) by ALPN; once
        // the cloud-sink crates pulled reqwest's `http2` feature into the
        // workspace, this telemetry client advertised `h2` and cmux
        // misrouted the /dp/telemetry POSTs to the gRPC handler.
        .http1_only()
        .use_rustls_tls();
    // Mirror heartbeat::build_client — pick up the operator-supplied
    // extra trust root (managed.cp_ca_cert_file) when set.
    if let Some(extra) = mtls.extra_ca_pem.as_ref() {
        let extra_ca = reqwest::Certificate::from_pem(extra)
            .context("parse managed.cp_ca_cert_file as PEM certificate")?;
        builder = builder.add_root_certificate(extra_ca);
    }
    builder.build().context("build reqwest client with mTLS")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rcgen::{CertificateParams, DistinguishedName, KeyPair, PKCS_ECDSA_P256_SHA256};
    use std::path::Path;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn write_test_bundle(dir: &Path) -> MtlsBundle {
        let ca_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.distinguished_name = {
            let mut dn = DistinguishedName::new();
            dn.push(rcgen::DnType::CommonName, "aisix-test-ca");
            dn
        };
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_kp).unwrap();

        let leaf_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut leaf_params = CertificateParams::new(vec!["dp-test".to_string()]).unwrap();
        leaf_params.distinguished_name = {
            let mut dn = DistinguishedName::new();
            dn.push(rcgen::DnType::CommonName, "dp-test");
            dn
        };
        let leaf_cert = leaf_params.signed_by(&leaf_kp, &ca_cert, &ca_kp).unwrap();

        let ca_path = dir.join("ca.crt");
        let cert_path = dir.join("client.crt");
        let key_path = dir.join("client.key");
        std::fs::write(&ca_path, ca_cert.pem()).unwrap();
        std::fs::write(&cert_path, leaf_cert.pem()).unwrap();
        std::fs::write(&key_path, leaf_kp.serialize_pem()).unwrap();

        MtlsBundle {
            ca_cert_path: ca_path,
            client_cert_path: cert_path,
            client_key_path: key_path,
            extra_ca_pem: None,
        }
    }

    /// One timestamp for the whole test binary, so two calls building the
    /// "expected" and the "staged" copy of a batch produce byte-identical
    /// events. Fresh rather than fixed, because a batch is now aged by its
    /// oldest `occurred_at` and a hardcoded past date would put every test
    /// event past [`RETRY_BUDGET`].
    fn now_rfc3339() -> &'static str {
        static NOW: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        NOW.get_or_init(|| chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
    }

    /// An `occurred_at` `age` in the past — a batch that waited behind a
    /// long re-send.
    fn aged_rfc3339(age: Duration) -> String {
        (chrono::Utc::now() - chrono::Duration::from_std(age).unwrap())
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
    }

    fn sample_event(id: &str) -> UsageEvent {
        UsageEvent {
            request_id: id.into(),
            occurred_at: now_rfc3339().to_string(),
            model_id: "mod-uuid".into(),
            api_key_id: "ak-uuid".into(),
            prompt_tokens: 10,
            completion_tokens: 20,
            upstream_latency_ms: 30,
            status_code: 200,
            cost_usd: 0.001,
            guardrail_blocked: false,
            ..Default::default()
        }
    }

    fn plain_client() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn send_posts_events_array_with_no_authorization() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/telemetry"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "accepted": 2
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let cfg = TelemetryConfig::new(format!("{}/dp/telemetry", server.uri()), mtls);
        let events = vec![sample_event("req-1"), sample_event("req-2")];
        let batch_id = Uuid::new_v4();

        let attempt = send(&plain_client(), &cfg, batch_id, &events).await;
        // A control plane that does not set the header still delivers; it
        // is re-sending that its absence rules out.
        assert!(
            matches!(attempt, Attempt::Delivered { dedup: false, .. }),
            "{attempt:?}"
        );

        let received = server.received_requests().await.unwrap();
        let req = received.first().unwrap();
        // v3 telemetry MUST NOT carry Authorization — mTLS only.
        assert!(req.headers.get("authorization").is_none());
        assert_eq!(
            req.headers
                .get(USAGE_BATCH_ID_HEADER)
                .expect("every batch carries its id")
                .to_str()
                .unwrap(),
            batch_id.to_string(),
        );
        // Body wraps the array in `{ events: [...] }`.
        let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(body["events"].as_array().unwrap().len(), 2);
        assert_eq!(body["events"][0]["request_id"], "req-1");
    }

    #[tokio::test]
    async fn send_propagates_non_success_with_body_excerpt() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/telemetry"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": {"code": "INVALID_REQUEST", "message": "event 0: bad uuid"}
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let cfg = TelemetryConfig::new(format!("{}/dp/telemetry", server.uri()), mtls);
        let attempt = send(
            &plain_client(),
            &cfg,
            Uuid::new_v4(),
            &[sample_event("req-1")],
        )
        .await;
        let Attempt::Failed(failure) = attempt else {
            panic!("a 400 is a failure: {attempt:?}");
        };
        assert!(
            matches!(
                failure,
                SendFailure::Response {
                    retryable: false,
                    ..
                }
            ),
            "a 400 is the control plane refusing this batch: {failure:?}",
        );
        let s = format!("{failure}");
        assert!(s.contains("400"), "expected status: {s}");
        assert!(s.contains("INVALID_REQUEST"), "expected body excerpt: {s}");
    }

    /// The capability rule, stated at the one place that decides it: judge
    /// each response on its own, and fall back to the last response seen
    /// only when an exchange produced none at all.
    #[test]
    fn resend_permission_is_judged_per_response_and_never_cached() {
        let response = |dedup: bool, retryable: bool| SendFailure::Response {
            dedup,
            retryable,
            error: anyhow!("boom"),
        };
        let no_response = || SendFailure::NoResponse {
            error: anyhow!("connection refused"),
        };

        let mut state = SenderState::default();
        // Nothing seen yet: a connect failure cannot be re-sent.
        assert!(!state.may_resend(&no_response()));
        // A retryable failure whose response proves de-duplication.
        assert!(state.may_resend(&response(true, true)));
        // …and a connect failure after it inherits that evidence.
        assert!(state.may_resend(&no_response()));
        // A non-retryable status is never re-sent, header or not.
        assert!(!state.may_resend(&response(true, false)));
        assert!(state.may_resend(&no_response()));
        // A header-less response — an older replica behind the same
        // Service, or a rolled-back control plane — clears the evidence…
        assert!(!state.may_resend(&response(false, true)));
        // …so the connect failures after it are dropped again.
        assert!(!state.may_resend(&no_response()));
    }

    #[test]
    fn build_client_loads_real_mtls_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let mtls = write_test_bundle(dir.path());
        let _ = build_client(&mtls).expect("real bundle should build");
    }

    /// Mirror heartbeat regression: PEM without trailing newline.
    #[test]
    fn build_client_works_without_trailing_newlines() {
        let dir = tempfile::tempdir().unwrap();

        let ca_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.distinguished_name = {
            let mut dn = DistinguishedName::new();
            dn.push(rcgen::DnType::CommonName, "aisix-test-ca");
            dn
        };
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_kp).unwrap();

        let leaf_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut leaf_params = CertificateParams::new(vec!["dp-test".to_string()]).unwrap();
        leaf_params.distinguished_name = {
            let mut dn = DistinguishedName::new();
            dn.push(rcgen::DnType::CommonName, "dp-test");
            dn
        };
        let leaf_cert = leaf_params.signed_by(&leaf_kp, &ca_cert, &ca_kp).unwrap();

        let ca_path = dir.path().join("ca.crt");
        let cert_path = dir.path().join("client.crt");
        let key_path = dir.path().join("client.key");
        std::fs::write(&ca_path, ca_cert.pem().trim_end()).unwrap();
        std::fs::write(&cert_path, leaf_cert.pem().trim_end()).unwrap();
        std::fs::write(&key_path, leaf_kp.serialize_pem().trim_end()).unwrap();

        let mtls = MtlsBundle {
            ca_cert_path: ca_path,
            client_cert_path: cert_path,
            client_key_path: key_path,
            extra_ca_pem: None,
        };
        build_client(&mtls).expect("build_client must tolerate PEM without trailing newline");
    }
    type RecordedBatches = Arc<std::sync::Mutex<Vec<serde_json::Value>>>;

    async fn recording_server(first_status: u16) -> (MockServer, RecordedBatches) {
        let server = MockServer::start().await;
        let batches = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = Arc::clone(&batches);
        Mock::given(method("POST"))
            .and(path("/dp/telemetry"))
            .respond_with(move |request: &wiremock::Request| {
                assert!(request.headers.get("authorization").is_none());
                let mut batches = recorded.lock().unwrap();
                batches.push(serde_json::from_slice(&request.body).unwrap());
                ResponseTemplate::new(if batches.len() == 1 {
                    first_status
                } else {
                    200
                })
            })
            .mount(&server)
            .await;
        (server, batches)
    }

    /// [`recording_server`] that holds its FIRST response for `held`,
    /// leaving the worker inside one flush while a test stages what
    /// arrives behind it. The body is recorded before the wait, so the
    /// test can tell "in flight" from "not sent yet".
    async fn recording_server_holding_first(held: Duration) -> (MockServer, RecordedBatches) {
        let server = MockServer::start().await;
        let batches = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = Arc::clone(&batches);
        Mock::given(method("POST"))
            .and(path("/dp/telemetry"))
            .respond_with(move |request: &wiremock::Request| {
                let mut batches = recorded.lock().unwrap();
                batches.push(serde_json::from_slice(&request.body).unwrap());
                let response = ResponseTemplate::new(200);
                if batches.len() == 1 {
                    response.set_delay(held)
                } else {
                    response
                }
            })
            .mount(&server)
            .await;
        (server, batches)
    }

    /// One recorded POST: the batch id it carried and its body.
    type RecordedPosts = Arc<std::sync::Mutex<Vec<(String, serde_json::Value)>>>;

    /// One scripted answer: a status, whether the response carries
    /// [`USAGE_BATCH_DEDUP_HEADER`], and whether it is held past the
    /// client's own timeout.
    #[derive(Debug, Clone, Copy)]
    struct Answer {
        status: u16,
        /// `None` sends no header at all; `Some(v)` sends it with value `v`.
        dedup: Option<&'static str>,
        /// `true` answers so late that the sender gives up waiting — the
        /// failure a control-plane outage actually produces, with NO
        /// response to read a capability off. Deterministic where killing
        /// a mock server is not: a dropped listener keeps answering for an
        /// unspecified while.
        stalled: bool,
        /// JSON body of a `2xx` answer; `None` sends an empty body.
        body: Option<&'static str>,
    }

    /// Answers with the de-duplication header — a control plane whose
    /// ingestion is idempotent per batch.
    fn dedups(status: u16) -> Answer {
        Answer {
            status,
            dedup: Some(USAGE_BATCH_DEDUP_ENABLED),
            stalled: false,
            body: None,
        }
    }

    /// Answers with the header set to something the protocol does not
    /// define — a control plane this sender must not read as capable.
    fn dedup_value(status: u16, value: &'static str) -> Answer {
        Answer {
            status,
            dedup: Some(value),
            stalled: false,
            body: None,
        }
    }

    /// Answers WITHOUT it — an older replica behind the same Service, or a
    /// control plane rolled back below that capability.
    fn no_dedup(status: u16) -> Answer {
        Answer {
            status,
            dedup: None,
            stalled: false,
            body: None,
        }
    }

    /// Never answers in time.
    fn stalls() -> Answer {
        Answer {
            status: 200,
            dedup: Some(USAGE_BATCH_DEDUP_ENABLED),
            stalled: true,
            body: None,
        }
    }

    /// Longer than the sender's own request timeout, so an attempt that
    /// meets it fails with no response at all.
    const BEYOND_THE_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

    /// Mock that answers the i-th request with `script[i]`, repeating the
    /// last entry forever, and records what each request carried.
    ///
    /// The header is per RESPONSE rather than per server on purpose: behind
    /// one dp-manager Service a rolling upgrade answers from both kinds of
    /// replica.
    async fn scripted_server(script: Vec<Answer>) -> (MockServer, RecordedPosts) {
        let server = MockServer::start().await;
        let posts: RecordedPosts = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = Arc::clone(&posts);
        Mock::given(method("POST"))
            .and(path("/dp/telemetry"))
            .respond_with(move |request: &wiremock::Request| {
                // Read before locking: a failed expectation here must fail
                // the test, not poison the recording behind a confusing
                // second error.
                let batch_id = request
                    .headers
                    .get(USAGE_BATCH_ID_HEADER)
                    .expect("every batch carries its id")
                    .to_str()
                    .unwrap()
                    .to_string();
                let mut posts = recorded.lock().unwrap();
                posts.push((batch_id, serde_json::from_slice(&request.body).unwrap()));
                let answer = *script
                    .get(posts.len() - 1)
                    .unwrap_or_else(|| script.last().unwrap());
                let response = match answer.dedup {
                    Some(value) => ResponseTemplate::new(answer.status)
                        .insert_header(USAGE_BATCH_DEDUP_HEADER, value),
                    None => ResponseTemplate::new(answer.status),
                };
                let response = match answer.body {
                    Some(body) => response.set_body_raw(body, "application/json"),
                    None => response,
                };
                if answer.stalled {
                    response.set_delay(BEYOND_THE_REQUEST_TIMEOUT)
                } else {
                    response
                }
            })
            .mount(&server)
            .await;
        (server, posts)
    }

    fn test_config(url: String, dir: &tempfile::TempDir) -> TelemetryConfig {
        TelemetryConfig::new(url, write_test_bundle(dir.path()))
    }

    /// Sum of `aisix_usage_event_drops_total` samples carrying `reason`.
    fn drops_with_reason(metrics: &Metrics, reason: &str) -> u64 {
        metrics
            .render()
            .lines()
            .filter(|line| {
                line.starts_with("aisix_usage_event_drops_total{")
                    && line.contains(&format!("reason=\"{reason}\""))
            })
            .map(|line| line.rsplit(' ').next().unwrap().parse::<u64>().unwrap())
            .sum()
    }

    /// Virtual-time step for a phase whose control plane is UP: real HTTP is
    /// on the wire, and the client's own 10s timeout is measured on the same
    /// paused clock this advances. Small enough that an exchange in flight
    /// gets ~1000 polls before that timeout could fire, large enough that a
    /// one-second backoff is crossed in a hundred.
    const STEP_WHILE_SERVING: Duration = Duration::from_millis(10);

    /// Virtual-time step for a phase whose control plane never answers in
    /// time. Each attempt ends at the client's own request timeout, which is
    /// measured on this same paused clock, so advancing in seconds is what
    /// ENDS an attempt rather than something that could cut one short — and
    /// a 30-minute budget is 1800 cheap iterations.
    ///
    /// It does NOT give an attempt time to be DELIVERED. Ten steps cross
    /// the 10s timeout, so the request gets ten polls to reach a real
    /// socket and a real wiremock thread, against a thousand at
    /// [`STEP_WHILE_SERVING`] — and those ten pass in microseconds of
    /// wall clock. A phase that needs the server to have RECEIVED an
    /// attempt (anything that then reads `posts`) must reach that point
    /// on the small step and only then switch to this one.
    const STEP_WHILE_UNREACHABLE: Duration = Duration::from_secs(1);

    /// Poll the worker while virtual time moves forward in `step`s, until
    /// `done` holds.
    ///
    /// Time is paused, so a backoff only elapses when this advances it —
    /// which is what keeps a 30-minute retry budget a sub-second test. The
    /// step is the caller's because it is a trade: it has to cross a backoff
    /// in a reasonable number of iterations without racing the request
    /// timeout of an exchange that is on the wire right now.
    ///
    /// **The step also decides whether an attempt gets DELIVERED.**
    /// `posts` is pushed by the wiremock responder, on wiremock's own
    /// runtime, in real time; the client's 10s request timeout runs on
    /// the clock this advances, in instant steps. At one second a step an
    /// attempt gets ten polls to cross a real socket — microseconds of
    /// wall clock — before its own timeout cancels it, and a cancelled
    /// request is never recorded at all. At ten milliseconds it gets a
    /// thousand. So a phase that has to end with the server having
    /// RECEIVED an attempt belongs on the small step, whatever the
    /// condition is written in terms of; reaching it on the big step and
    /// then reading `posts` is the shape that flakes.
    ///
    /// An attempt the server ANSWERS carries no such race in either
    /// direction: the response cannot exist unless the responder ran, and
    /// the responder records the post before it replies.
    ///
    /// The small step is MARGIN, not a guarantee. One delivery measured
    /// 14 to 21 iterations of this loop against a budget of ten at the
    /// big step and a thousand at the small one — two orders of
    /// magnitude, which is why one is reliable and the other is a coin
    /// flip, but neither is an invariant. Making it one would mean
    /// synchronising on a server-side event rather than on a poll
    /// budget; nothing here needs that yet.
    async fn drive_until<F: std::future::Future<Output = ()> + ?Sized>(
        mut sender: std::pin::Pin<&mut F>,
        step: Duration,
        mut done: impl FnMut() -> bool,
        what: &str,
    ) {
        let wall_clock_deadline = std::time::Instant::now() + Duration::from_secs(60);
        while !done() {
            assert!(
                poll_sender(sender.as_mut()).await.is_pending(),
                "sender exited before {what}"
            );
            assert!(std::time::Instant::now() < wall_clock_deadline, "{what}");
            tokio::time::advance(step).await;
            tokio::task::yield_now().await;
        }
    }

    fn ordered_events(count: usize) -> Vec<UsageEvent> {
        (0..count)
            .map(|i| {
                // Distinct attempts may share a request ID; none may be deduplicated.
                let mut event = sample_event(&format!("request-{}", i / 3));
                event.model_id = format!("model-{}", i % 7);
                event.provider_kind = format!("provider-{}", i % 3);
                event.user_id = format!("user-{i}");
                event.prompt_tokens = i as u32;
                event.completion_tokens = (i * 2) as u32;
                event
            })
            .collect()
    }

    /// [`ordered_events`], stamped `age` in the past.
    fn aged_events(count: usize, age: Duration) -> Vec<UsageEvent> {
        let occurred_at = aged_rfc3339(age);
        ordered_events(count)
            .into_iter()
            .map(|mut event| {
                event.occurred_at = occurred_at.clone();
                event
            })
            .collect()
    }

    /// `?Sized` so the boxed `dyn Future` a shared staging helper returns
    /// drives through the same three helpers as an inline `Box::pin(run(..))`.
    async fn poll_sender<F: std::future::Future<Output = ()> + ?Sized>(
        mut sender: std::pin::Pin<&mut F>,
    ) -> std::task::Poll<()> {
        std::future::poll_fn(|cx| std::task::Poll::Ready(sender.as_mut().poll(cx))).await
    }

    async fn finish_sender<F: std::future::Future<Output = ()> + ?Sized>(
        mut sender: std::pin::Pin<&mut F>,
    ) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while poll_sender(sender.as_mut()).await.is_pending() {
            assert!(std::time::Instant::now() < deadline, "sender did not exit");
            // Keep the paused runtime runnable while real HTTP I/O completes.
            tokio::task::yield_now().await;
        }
    }

    fn assert_recorded_events(batches: &RecordedBatches, expected: &[UsageEvent]) -> Vec<usize> {
        let batches = batches.lock().unwrap();
        let actual: Vec<_> = batches
            .iter()
            .flat_map(|batch| batch["events"].as_array().unwrap().iter().cloned())
            .collect();
        assert_eq!(
            serde_json::json!(actual),
            serde_json::to_value(expected).unwrap()
        );
        batches
            .iter()
            .map(|batch| batch["events"].as_array().unwrap().len())
            .collect()
    }

    /// Channel bound for the backlog cases below. They are about what the
    /// worker does with a queue it has filled, not about how deep the real
    /// one is, so they size their own channel rather than staging
    /// [`QUEUE_CAPACITY`] events to fill it.
    const BACKLOG_QUEUE: usize = 1024;

    async fn run_interval_with_backlog(first_status: u16) {
        let (server, batches) = recording_server(first_status).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = TelemetryConfig::new(
            format!("{}/dp/telemetry", server.uri()),
            write_test_bundle(dir.path()),
        );
        let (tx, rx) = tokio::sync::mpsc::channel(BACKLOG_QUEUE);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        let expected = ordered_events(BACKLOG_QUEUE + 2);
        let mut events = ordered_events(BACKLOG_QUEUE + 2).into_iter();
        tokio::time::pause();
        let mut sender = Box::pin(run(cfg, Metrics::new(false), rx, &mut cancel_rx));
        // Consume the initial empty tick before staging a partial batch.
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        for event in events.by_ref().take(2) {
            tx.try_send(event).unwrap();
        }
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        assert_eq!(tx.capacity(), BACKLOG_QUEUE);
        assert!(batches.lock().unwrap().is_empty());
        for event in events {
            tx.try_send(event).unwrap();
        }
        assert_eq!(tx.capacity(), 0);
        tokio::time::advance(FLUSH_INTERVAL).await;

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while batches.lock().unwrap().is_empty() {
            assert!(poll_sender(sender.as_mut()).await.is_pending());
            assert!(std::time::Instant::now() < deadline, "no telemetry POST");
            tokio::task::yield_now().await;
        }
        // Both recv and the tick are ready: this must hold whichever wins.
        assert_eq!(
            batches.lock().unwrap()[0]["events"]
                .as_array()
                .unwrap()
                .len(),
            MAX_BATCH
        );
        drop(tx);
        finish_sender(sender.as_mut()).await;
        let sizes = assert_recorded_events(&batches, &expected);
        assert_eq!(sizes, [vec![MAX_BATCH; 10], vec![26]].concat());
    }

    #[tokio::test]
    async fn interval_fills_ready_backlog_without_reordering_events() {
        run_interval_with_backlog(200).await;
    }

    /// [`recording_server`] answers without [`USAGE_BATCH_DEDUP_HEADER`] —
    /// a control plane whose ingestion is not idempotent — so its failures
    /// are still dropped where they fall, and never carried into the next
    /// batch.
    #[tokio::test]
    async fn failed_interval_batch_without_dedup_support_is_dropped_not_retried() {
        run_interval_with_backlog(500).await;
    }

    #[tokio::test]
    async fn interval_flushes_partial_batch_without_waiting_for_more_events() {
        let (server, batches) = recording_server(200).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = TelemetryConfig::new(
            format!("{}/dp/telemetry", server.uri()),
            write_test_bundle(dir.path()),
        );
        let (tx, rx) = tokio::sync::mpsc::channel(QUEUE_CAPACITY);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let mut sender = Box::pin(run(cfg, Metrics::new(false), rx, &mut cancel_rx));
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        for event in ordered_events(2) {
            tx.try_send(event).unwrap();
        }
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        assert_eq!(tx.capacity(), QUEUE_CAPACITY);
        tokio::time::advance(FLUSH_INTERVAL - Duration::from_millis(1)).await;
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        assert!(batches.lock().unwrap().is_empty());
        tokio::time::advance(Duration::from_millis(1)).await;
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while batches.lock().unwrap().is_empty() {
            assert!(poll_sender(sender.as_mut()).await.is_pending());
            assert!(
                std::time::Instant::now() < deadline,
                "partial batch was not flushed"
            );
            tokio::task::yield_now().await;
        }
        drop(tx);
        finish_sender(sender.as_mut()).await;
        assert_eq!(
            assert_recorded_events(&batches, &ordered_events(2)),
            vec![2]
        );
    }

    #[tokio::test]
    async fn cancellation_drains_buffer_and_ready_queue_without_losing_attempts() {
        let (server, batches) = recording_server(200).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = TelemetryConfig::new(
            format!("{}/dp/telemetry", server.uri()),
            write_test_bundle(dir.path()),
        );
        let (tx, rx) = tokio::sync::mpsc::channel(QUEUE_CAPACITY);
        let (cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let mut sender = Box::pin(run(cfg, Metrics::new(false), rx, &mut cancel_rx));
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        let mut events = ordered_events(250).into_iter();
        for event in events.by_ref().take(2) {
            tx.try_send(event).unwrap();
        }
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        assert_eq!(tx.capacity(), QUEUE_CAPACITY);
        for event in events {
            tx.try_send(event).unwrap();
        }
        cancel_tx.send(true).unwrap();
        // Keep tx alive: completion must come from cancellation, not channel closure.
        finish_sender(sender.as_mut()).await;
        assert_recorded_events(&batches, &ordered_events(250));
        assert!(tx.is_closed());
    }

    /// The shutdown drain is the one flush that can meet a FULL queue —
    /// every other one happens at or below MAX_BATCH — and one POST of
    /// everything queued is what the ceiling exists to prevent: cp-api
    /// rejects an oversized body, and a rejected batch is not retried, so
    /// the whole backlog would go at once.
    #[tokio::test]
    async fn cancellation_posts_the_backlog_in_batches_not_in_one_body() {
        const QUEUED: usize = MAX_BATCH * 3 + 7;

        let (server, batches) = recording_server(200).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = TelemetryConfig::new(
            format!("{}/dp/telemetry", server.uri()),
            write_test_bundle(dir.path()),
        );
        let (tx, rx) = tokio::sync::mpsc::channel(QUEUED);
        let (cancel_tx, mut cancel_rx) = watch::channel(false);
        let mut sender = Box::pin(run(cfg, Metrics::new(false), rx, &mut cancel_rx));
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        for event in ordered_events(QUEUED) {
            tx.try_send(event).unwrap();
        }
        cancel_tx.send(true).unwrap();
        finish_sender(sender.as_mut()).await;

        let sizes = assert_recorded_events(&batches, &ordered_events(QUEUED));
        assert!(
            sizes.iter().all(|n| *n <= MAX_BATCH),
            "no POST may carry more than the batch ceiling: {sizes:?}"
        );
    }

    /// A batch this worker gives up on is gone — there is no retry — so the
    /// queue is the whole defence against a control plane that has gone
    /// slow. One POST is in flight at a time, and everything the proxy
    /// emits meanwhile has to fit.
    ///
    /// 1550 events is the shape a stability round measured behind an 8s
    /// control-plane stall at 173 req/s: the queue that held 1024 dropped
    /// 793 of them.
    #[tokio::test]
    async fn a_burst_arriving_while_one_post_is_in_flight_is_not_dropped() {
        // Staged first, to get the worker into a POST. The mock records a
        // request before it answers, so seeing the batch means the flush
        // is in flight rather than finished.
        const OPENING: usize = 150;
        const BURST: usize = 1_400;
        // Outlasts the staging below, which is a few microseconds of
        // non-blocking sends.
        const HELD: Duration = Duration::from_secs(5);

        let (server, batches) = recording_server_holding_first(HELD).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = TelemetryConfig::new(
            format!("{}/dp/telemetry", server.uri()),
            write_test_bundle(dir.path()),
        );
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let (sink, worker) = spawn(cfg, Metrics::new(false), cancel_rx);

        let expected = ordered_events(OPENING + BURST);
        let mut events = expected.clone().into_iter();
        for event in events.by_ref().take(OPENING) {
            sink.try_emit("test", event, aisix_obs::UsageEventLabels::default());
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while batches.lock().unwrap().is_empty() {
            assert!(std::time::Instant::now() < deadline, "no telemetry POST");
            tokio::task::yield_now().await;
        }

        // The control plane is now holding that POST, so nothing is being
        // drained while the rest of the burst arrives.
        for event in events {
            sink.try_emit("test", event, aisix_obs::UsageEventLabels::default());
        }
        drop(sink);
        worker.await.unwrap();

        assert_recorded_events(&batches, &expected);
    }

    /// Stage `events` behind a paused clock and run the worker until the
    /// first POST of them is on the wire. Returns the driving future, the
    /// queue handle (so a test can keep emitting) and the cancel sender.
    ///
    /// Every re-send case below starts here: a batch has to be in flight
    /// before its failure can mean anything.
    async fn staged_sender<'a>(
        cfg: TelemetryConfig,
        metrics: Metrics,
        cancel_rx: &'a mut watch::Receiver<bool>,
        events: usize,
    ) -> (
        std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>,
        tokio::sync::mpsc::Sender<UsageEvent>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::channel(BACKLOG_QUEUE);
        let mut sender: std::pin::Pin<Box<dyn std::future::Future<Output = ()>>> =
            Box::pin(run(cfg, metrics, rx, cancel_rx));
        // Consume the initial empty tick before staging the batch.
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        for event in ordered_events(events) {
            tx.try_send(event).unwrap();
        }
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        tokio::time::advance(FLUSH_INTERVAL).await;
        (sender, tx)
    }

    fn assert_identical_resends(posts: &RecordedPosts, attempts: usize) {
        let posts = posts.lock().unwrap();
        assert_eq!(posts.len(), attempts, "unexpected attempt count: {posts:?}");
        for post in posts.iter() {
            assert_eq!(
                (&post.0, &post.1),
                (&posts[0].0, &posts[0].1),
                "a re-send repeats the SAME batch id and the SAME events",
            );
        }
    }

    /// A `2xx` that names events the control plane refused is delivered and
    /// final — re-sending would be refused the same way — but it is a hole
    /// in the usage record, so it is counted and logged instead of passing
    /// silently.
    #[tokio::test]
    async fn rejected_events_in_an_accepted_batch_are_counted_and_not_resent() {
        let (server, posts) = scripted_server(vec![Answer {
            body: Some(r#"{"accepted":1,"rejected":1}"#),
            ..dedups(200)
        }])
        .await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let logs = CapturedLogs::default();
        let _subscriber = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_writer(logs.clone())
                .with_ansi(false)
                .finish(),
        );
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        let rejected_recorded = || {
            metrics
                .render()
                .lines()
                .any(|line| line == "aisix_usage_events_rejected_total 1")
        };
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            rejected_recorded,
            "the rejection was never recorded",
        )
        .await;
        // Long past every re-send backoff: a retry would have happened.
        for _ in 0..20 {
            assert!(poll_sender(sender.as_mut()).await.is_pending());
            tokio::time::advance(RETRY_MAX_BACKOFF).await;
            tokio::task::yield_now().await;
        }
        drop(tx);
        finish_sender(sender.as_mut()).await;

        assert_eq!(
            posts.lock().unwrap().len(),
            1,
            "a rejection is never re-sent"
        );
        assert!(rejected_recorded(), "{}", metrics.render());
        let logs = logs.text();
        let warn = logs
            .lines()
            .find(|line| line.contains("control plane rejected usage events"))
            .unwrap_or_else(|| panic!("no rejection WARN in: {logs}"));
        assert!(warn.contains("WARN"), "{warn}");
        assert!(
            warn.contains("rejected=1") && warn.contains("count=2"),
            "{warn}"
        );
        assert_eq!(drops_with_reason(&metrics, DROP_SEND_FAILED), 0);
    }

    /// An answer with nothing rejected — or no answer body at all, as an
    /// older control plane may send — records no rejection.
    #[tokio::test]
    async fn an_answer_without_rejections_records_none() {
        for body in [Some(r#"{"accepted":2,"rejected":0}"#), None] {
            let (server, _posts) = scripted_server(vec![Answer {
                body,
                ..dedups(200)
            }])
            .await;
            let dir = tempfile::tempdir().unwrap();
            let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
            let attempt = send(&plain_client(), &cfg, Uuid::new_v4(), &[sample_event("r")]).await;
            assert!(
                matches!(attempt, Attempt::Delivered { rejected: 0, .. }),
                "{attempt:?}"
            );
        }
    }

    /// `tracing` writer that keeps what was logged for assertions.
    #[derive(Clone, Default)]
    struct CapturedLogs(Arc<std::sync::Mutex<Vec<u8>>>);

    impl CapturedLogs {
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
        }
    }

    impl std::io::Write for CapturedLogs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// The outage this exists for: the control plane answers `503`, and its
    /// answer proves a re-send cannot double-count, so the batch goes again
    /// — same id, same events — and lands.
    ///
    /// The second response also covers the duplicate acknowledgement: a
    /// batch the control plane already committed is answered with an
    /// ordinary `200`, which this worker must read as delivered rather than
    /// as anything special.
    #[tokio::test]
    async fn a_retryable_failure_with_the_dedup_header_resends_the_same_batch() {
        let (server, posts) = scripted_server(vec![dedups(503), dedups(200)]).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || posts.lock().unwrap().len() >= 2,
            "the failed batch was never re-sent",
        )
        .await;
        drop(tx);
        finish_sender(sender.as_mut()).await;

        assert_identical_resends(&posts, 2);
        assert_eq!(
            posts.lock().unwrap()[0].1["events"]
                .as_array()
                .unwrap()
                .len(),
            2,
        );
        assert_eq!(drops_with_reason(&metrics, DROP_SEND_FAILED), 0);
        assert_eq!(drops_with_reason(&metrics, DROP_RETRY_BUDGET_EXHAUSTED), 0);
    }

    /// The header is a capability assertion with one defined value. A
    /// response that carries it with anything else is a control plane this
    /// sender does not understand, and re-sending to it could bill twice —
    /// so it is treated exactly like a response that carries no header.
    #[tokio::test]
    async fn a_dedup_header_with_an_unknown_value_does_not_allow_a_re_send() {
        let (server, posts) = scripted_server(vec![dedup_value(503, "0")]).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || drops_with_reason(&metrics, DROP_SEND_FAILED) == 2,
            "the batch was not dropped",
        )
        .await;
        drop(tx);
        finish_sender(sender.as_mut()).await;

        assert_eq!(posts.lock().unwrap().len(), 1, "it must not be re-sent");
        assert_eq!(drops_with_reason(&metrics, DROP_RETRY_BUDGET_EXHAUSTED), 0);
    }

    /// A `400` is the control plane refusing THIS batch — a malformed body,
    /// a batch id it will not accept. Re-sending it changes nothing, so it
    /// is dropped on the first answer even though de-duplication is on.
    #[tokio::test]
    async fn a_non_retryable_status_is_dropped_even_when_dedup_is_supported() {
        let (server, posts) = scripted_server(vec![dedups(400)]).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || drops_with_reason(&metrics, DROP_SEND_FAILED) == 2,
            "the refused batch was not dropped",
        )
        .await;
        drop(tx);
        finish_sender(sender.as_mut()).await;

        assert_eq!(posts.lock().unwrap().len(), 1, "a 400 is not re-sent");
        assert_eq!(drops_with_reason(&metrics, DROP_RETRY_BUDGET_EXHAUSTED), 0);
    }

    /// The failure a control-plane outage actually produces: no response at
    /// all. It proves nothing about the far side, so the worker falls back
    /// to the last response it DID see — here one that carried the header,
    /// so the batch is re-sent, and re-sent, until the budget that keeps it
    /// inside the billing window runs out.
    #[tokio::test]
    async fn no_response_after_a_dedup_bearing_response_resends_until_the_budget_ends() {
        // A retryable failure that proves de-duplication, and from then on
        // a control plane that never answers in time.
        let (server, posts) = scripted_server(vec![dedups(503), stalls()]).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let started = Instant::now();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        // The first exchange is real HTTP — small steps, so the paused clock
        // cannot outrun it — and the second attempt going out is the proof
        // that its header-bearing response was received.
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || posts.lock().unwrap().len() >= 2,
            "the failed batch was never re-sent",
        )
        .await;
        // Attempt 2 stalled, and the wait above has already delivered it.
        // This one takes the re-send AFTER it — attempt 3 — which is what
        // says a batch that met no response goes again. It stays on the
        // SMALL step because that delivery is what is being waited for:
        // at a second a step an attempt gets ten polls to cross a real
        // socket before its own 10s timeout cancels it, and a cancelled
        // request is never recorded. This wait replaces the `attempts > 2`
        // assertion that used to sit below it.
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || posts.lock().unwrap().len() > 2,
            "the stalling batch was never re-sent",
        )
        .await;
        // Only now may the clock move in seconds, to run the budget out.
        drive_until(
            sender.as_mut(),
            STEP_WHILE_UNREACHABLE,
            || drops_with_reason(&metrics, DROP_RETRY_BUDGET_EXHAUSTED) == 2,
            "the batch was not re-sent until the budget ran out",
        )
        .await;
        drop(tx);

        assert!(
            started.elapsed() >= RETRY_BUDGET,
            "the batch was given up on after {:?}, before the budget",
            started.elapsed(),
        );
        // One lock per statement: `assert_identical_resends` takes the lock
        // itself, and a guard still alive from an argument expression would
        // deadlock against it.
        // `attempts > 2` is not asserted here: the wait above establishes
        // it, and repeating it after the fact would be a check that
        // cannot fail. What is worth saying is that every attempt carried
        // the same batch — and that is asserted under ONE lock, because a
        // count read in one statement and re-checked in the next can
        // disagree with itself if a late recording lands in between.
        let posts = posts.lock().unwrap();
        let first = (&posts[0].0, &posts[0].1);
        assert!(
            posts.iter().all(|post| (&post.0, &post.1) == first),
            "a re-send repeats the SAME batch id and the SAME events: {posts:?}",
        );
        assert_eq!(
            drops_with_reason(&metrics, DROP_SEND_FAILED),
            0,
            "a batch re-sent to exhaustion is accounted apart from one \
             that was never re-sent",
        );
    }

    /// The other direction of the same rule: the last response carried no
    /// header — an older replica behind the same Service, or a rolled-back
    /// control plane — so the failures that follow it are dropped where they
    /// fall. Caching "we saw the header once" is what this forbids.
    #[tokio::test]
    async fn no_response_after_a_header_less_response_is_dropped_at_once() {
        // Fails with the header first, so the evidence IS set before the
        // header-less response clears it again.
        let (server, posts) = scripted_server(vec![dedups(503), no_dedup(200), stalls()]).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        // Attempt 1 fails with the header, attempt 2 delivers without it.
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || posts.lock().unwrap().len() >= 2,
            "the failed batch was never re-sent",
        )
        .await;

        for event in ordered_events(2) {
            tx.try_send(event).unwrap();
        }
        tokio::time::advance(FLUSH_INTERVAL).await;
        // Two phases, and the order is the point. The third attempt has to
        // REACH the server before the clock may outrun it — `posts` is
        // recorded by wiremock in real time — so it goes out on the small
        // step. Only once it has landed may the clock jump in seconds to
        // blow its request timeout.
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || posts.lock().unwrap().len() >= 3,
            "the batch that met no response was never re-sent",
        )
        .await;
        drive_until(
            sender.as_mut(),
            STEP_WHILE_UNREACHABLE,
            || drops_with_reason(&metrics, DROP_SEND_FAILED) == 2,
            "the batch was not dropped after a header-less response",
        )
        .await;

        assert_eq!(
            posts.lock().unwrap().len(),
            3,
            "the batch that met no response was not re-sent",
        );
        assert_eq!(drops_with_reason(&metrics, DROP_RETRY_BUDGET_EXHAUSTED), 0);
    }

    /// While a batch is being re-sent nothing else is sent — the worker is
    /// single in-flight — so the queue is what holds the traffic that
    /// arrives meanwhile, and the event that meets a FULL queue is the new
    /// one, dropped as `sink_full` exactly as before. (The queue here is
    /// [`BACKLOG_QUEUE`] rather than [`QUEUE_CAPACITY`] for the reason
    /// given there.)
    #[tokio::test]
    async fn events_keep_queueing_while_a_batch_is_being_resent() {
        let (server, posts) = scripted_server(vec![dedups(503)]).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || posts.lock().unwrap().len() >= 2,
            "the failed batch was never re-sent",
        )
        .await;

        let sink = UsageSink::new(tx.clone()).with_metrics(metrics.clone());
        for event in ordered_events(BACKLOG_QUEUE) {
            sink.try_emit("test", event, UsageEventLabels::default());
        }
        assert_eq!(tx.capacity(), 0, "the queue should now be full");
        assert_eq!(drops_with_reason(&metrics, "sink_full"), 0);
        sink.try_emit(
            "test",
            sample_event("overflow"),
            UsageEventLabels::default(),
        );
        assert_eq!(
            drops_with_reason(&metrics, "sink_full"),
            1,
            "a full queue drops the NEW event, not the batch in hand",
        );

        // …and the batch in hand is still the one being re-sent.
        let attempts = posts.lock().unwrap().len();
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || posts.lock().unwrap().len() > attempts,
            "re-sending stopped while the queue filled",
        )
        .await;
        let posts = posts.lock().unwrap();
        let first = (&posts[0].0, &posts[0].1);
        assert!(
            posts.iter().all(|post| (&post.0, &post.1) == first),
            "the batch in hand must keep going unchanged: {posts:?}",
        );
    }

    /// The budget and the backoff schedule are normative in the cross-plane
    /// protocol — the control plane's `settleDelay` is chosen against this
    /// number — so their values are pinned literally. Every other test here
    /// compares the sender against itself and stays green if these move.
    #[test]
    fn the_retry_schedule_is_the_one_the_control_plane_was_sized_against() {
        assert_eq!(RETRY_BUDGET, Duration::from_secs(30 * 60));
        assert_eq!(RETRY_INITIAL_BACKOFF, Duration::from_secs(1));
        assert_eq!(RETRY_MAX_BACKOFF, Duration::from_secs(30));
    }

    /// A status is re-sent only when the control plane could take the same
    /// batch later. Everything else is it refusing this batch, which no
    /// number of re-sends changes.
    #[test]
    fn only_a_temporary_refusal_is_re_sent() {
        use reqwest::StatusCode;
        for status in [
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
        ] {
            assert!(is_retryable(status), "{status} must be re-sent");
        }
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
            StatusCode::CONFLICT,
            StatusCode::PAYLOAD_TOO_LARGE,
            StatusCode::UNPROCESSABLE_ENTITY,
        ] {
            assert!(!is_retryable(status), "{status} is a refusal of THIS batch");
        }
    }

    /// What the sender-side drop samples can and cannot attribute. The
    /// member pair comes off the event, exactly as the queue's own drops
    /// take it; the model and provider-key dimensions are the emitting
    /// handler's label set, which the queue does not carry, so they read
    /// `unknown`. Documentation quotes this, so it is pinned rather than
    /// described.
    #[tokio::test]
    async fn a_sender_side_drop_names_the_member_and_nothing_else() {
        let (server, _posts) = scripted_server(vec![dedups(400)]).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || drops_with_reason(&metrics, DROP_SEND_FAILED) == 2,
            "the refused batch was not dropped",
        )
        .await;
        drop(tx);

        let rendered = metrics.render();
        let dropped: Vec<&str> = rendered
            .lines()
            .filter(|line| {
                line.starts_with("aisix_usage_event_drops_total{")
                    && line.contains("reason=\"send_failed\"")
            })
            .collect();
        // `ordered_events` gives each event its own member.
        assert_eq!(dropped.len(), 2, "one series per member: {dropped:?}");
        for line in &dropped {
            assert!(
                line.contains("model=\"unknown\"") && line.contains("provider_key_id=\"unknown\""),
                "the queue carries events, not the handler's labels: {line}",
            );
        }
        for member in ["user-0", "user-1"] {
            assert!(
                dropped
                    .iter()
                    .any(|line| line.contains(&format!("user_id=\"{member}\""))),
                "whose usage was lost must stay answerable, missing {member}: {dropped:?}",
            );
        }
    }

    /// The safety net for a control plane that answers `5xx` for a batch it
    /// can never accept: without a cap the single in-flight channel is held
    /// for the whole budget, and the queue behind it — every event the
    /// gateway records meanwhile — overflows long before that.
    #[tokio::test]
    async fn a_batch_answered_and_refused_eight_times_is_given_up_on() {
        let (server, posts) = scripted_server(vec![dedups(503)]).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || drops_with_reason(&metrics, DROP_RETRY_BUDGET_EXHAUSTED) == 2,
            "the refused batch was re-sent past the cap",
        )
        .await;
        assert_identical_resends(&posts, MAX_ANSWERED_FAILURES);

        // …and the channel is free for what queued up behind it.
        for event in ordered_events(2) {
            tx.try_send(event).unwrap();
        }
        tokio::time::advance(FLUSH_INTERVAL).await;
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || posts.lock().unwrap().len() > MAX_ANSWERED_FAILURES,
            "the next batch never got its turn",
        )
        .await;
        let posts = posts.lock().unwrap();
        assert_ne!(
            posts[MAX_ANSWERED_FAILURES].0, posts[0].0,
            "the batch after a give-up is a new batch, with its own id",
        );
    }

    /// The cap counts answered failures only. A control plane that has
    /// stopped answering is the outage this feature exists for, and it is
    /// bounded by the budget instead — so attempts that time out neither
    /// consume the cap nor are stopped by it.
    #[tokio::test]
    async fn failures_with_no_response_do_not_consume_the_cap() {
        // Alternating: every second attempt is answered, the rest stall past
        // the request timeout.
        let script: Vec<Answer> = (0..40)
            .map(|i| if i % 2 == 0 { dedups(503) } else { stalls() })
            .collect();
        let (server, posts) = scripted_server(script).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        // Small steps throughout: the answered attempts are real exchanges,
        // and a clock that outran one would turn it into a no-response
        // failure — the very distinction this test is about.
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || drops_with_reason(&metrics, DROP_RETRY_BUDGET_EXHAUSTED) == 2,
            "the batch was never given up on",
        )
        .await;
        // The stalled attempts in between are recorded on wiremock's
        // clock, and the exact count below includes them. Its own wait,
        // not a conjunct of the one above: folded together, a recording
        // that never arrives would spend the wall-clock deadline and then
        // report that the batch was never given up on — which it was.
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || posts.lock().unwrap().len() >= MAX_ANSWERED_FAILURES * 2 - 1,
            "the batch was given up on before its fifteenth attempt — a \
             stalled attempt consumed the cap",
        )
        .await;
        drop(tx);

        // Answered attempts are the odd ones, so the eighth of them is the
        // fifteenth attempt — the stalls in between cost nothing.
        assert_identical_resends(&posts, MAX_ANSWERED_FAILURES * 2 - 1);
    }

    /// The budget is measured from the batch's OLDEST event, not from its
    /// first attempt: the sender is single in-flight, so a batch that waited
    /// behind a long re-send is already old when its turn comes, and sending
    /// it would land rows in an hour the control plane has already billed.
    /// A younger batch behind it is unaffected.
    #[tokio::test]
    async fn a_batch_older_than_the_budget_is_dropped_without_an_attempt() {
        let (server, posts) = scripted_server(vec![dedups(200)]).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        let (tx, rx) = tokio::sync::mpsc::channel(BACKLOG_QUEUE);
        tokio::time::pause();
        let mut sender = Box::pin(run(cfg, metrics.clone(), rx, &mut cancel_rx));
        assert!(poll_sender(sender.as_mut()).await.is_pending());

        for event in aged_events(2, RETRY_BUDGET + Duration::from_secs(60)) {
            tx.try_send(event).unwrap();
        }
        assert!(poll_sender(sender.as_mut()).await.is_pending());
        tokio::time::advance(FLUSH_INTERVAL).await;
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || drops_with_reason(&metrics, DROP_RETRY_BUDGET_EXHAUSTED) == 2,
            "the stale batch was not dropped",
        )
        .await;
        assert!(
            posts.lock().unwrap().is_empty(),
            "a batch past the billing window must not be sent at all",
        );

        // The batch behind it is fresh, and goes out normally.
        for event in ordered_events(2) {
            tx.try_send(event).unwrap();
        }
        tokio::time::advance(FLUSH_INTERVAL).await;
        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || !posts.lock().unwrap().is_empty(),
            "the fresh batch behind it never went out",
        )
        .await;
        drop(tx);
        finish_sender(sender.as_mut()).await;
        assert_eq!(posts.lock().unwrap().len(), 1);
        assert_eq!(drops_with_reason(&metrics, DROP_SEND_FAILED), 0);
    }

    /// Shutdown gets one last attempt at the batch in hand and then stops —
    /// it never waits out a backoff. The proof is that the worker exits
    /// without this test advancing the clock: every remaining backoff is
    /// still pending on the paused timer.
    #[tokio::test]
    async fn shutdown_does_not_wait_on_a_retrying_batch() {
        let (server, posts) = scripted_server(vec![dedups(503)]).await;
        let dir = tempfile::tempdir().unwrap();
        let cfg = test_config(format!("{}/dp/telemetry", server.uri()), &dir);
        let metrics = Metrics::new(false);
        let (cancel_tx, mut cancel_rx) = watch::channel(false);
        tokio::time::pause();
        let (mut sender, tx) = staged_sender(cfg, metrics.clone(), &mut cancel_rx, 2).await;

        drive_until(
            sender.as_mut(),
            STEP_WHILE_SERVING,
            || posts.lock().unwrap().len() >= 2,
            "the failed batch was never re-sent",
        )
        .await;
        let attempts = posts.lock().unwrap().len();

        // Queued behind the batch in hand, so the drain has something of its
        // own to post — a backoff waited out there would hold shutdown open
        // just as surely.
        for event in ordered_events(2) {
            tx.try_send(event).unwrap();
        }
        cancel_tx.send(true).unwrap();
        // Keeps tx alive: the exit must come from cancellation, and from no
        // further virtual time passing.
        finish_sender(sender.as_mut()).await;

        assert_eq!(
            posts.lock().unwrap().len(),
            attempts + 2,
            "shutdown makes one final attempt at the batch in hand and one \
             at what it drains",
        );
        assert_eq!(drops_with_reason(&metrics, DROP_SEND_FAILED), 4);
        assert_eq!(drops_with_reason(&metrics, DROP_RETRY_BUDGET_EXHAUSTED), 0);
    }
}
