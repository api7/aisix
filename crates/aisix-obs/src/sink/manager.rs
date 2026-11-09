//! Per-exporter pipeline manager.
//!
//! Owns one [`SinkPipeline`] per configured exporter. The request hot path
//! resolves an exporter's pipeline with [`ExporterPipelines::get_or_create`]
//! — lazily starting it on first sighting, and rebuilding it when the
//! exporter's config changes — then enqueues into the returned handle.
//! [`ExporterPipelines::retain`], run on every configuration publish, stops
//! pipelines for exporters that left the snapshot.
//!
//! Lazy-on-first-sighting mirrors the previous `OtlpHttpFanOut` permit map:
//! it is immediately consistent with the snapshot (a just-added exporter
//! receives the very next request), avoiding the reconcile-loop race where a
//! newly-added exporter would miss requests until the next poll.
//!
//! Sink-agnostic — the caller supplies a `build` closure mapping an exporter
//! to an [`ObservabilitySink`], so the manager is shared by every
//! pipeline-backed sink family (`otlp`, `http_batch`, …).
//!
//! The first sighting happens on the request path, which under
//! `proxy.thread_per_core` is one worker's single-thread runtime, so a
//! pipeline must not inherit its runtime from the caller: it would stay on
//! that worker for life and take batching, encoding and delivery with it.
//! [`ExporterPipelines::set_runtime`] names where pipelines run instead.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use super::{ObservabilitySink, PipelineConfig, SinkHandle, SinkPipeline, SinkStatsSnapshot};
use crate::metrics::Metrics;

/// One running pipeline plus the bookkeeping needed to stop/rebuild it.
struct Running {
    /// Hash of the exporter's delivery-relevant config. A change rebuilds the
    /// pipeline (old stops, new starts).
    fingerprint: u64,
    handle: SinkHandle,
    cancel: watch::Sender<bool>,
    /// Detached worker; awaited only by [`ExporterPipelines::shutdown`].
    worker: JoinHandle<()>,
}

/// Manages the live set of per-exporter pipelines.
pub struct ExporterPipelines {
    running: Mutex<HashMap<String, Running>>,
    cfg: PipelineConfig,
    metrics: Option<Metrics>,
    /// Where pipelines run. Unset, they run on the runtime of the caller
    /// that starts them.
    runtime: OnceLock<tokio::runtime::Handle>,
}

impl ExporterPipelines {
    /// Build an empty manager. Pipelines start on first
    /// [`get_or_create`](Self::get_or_create).
    pub fn new(cfg: PipelineConfig) -> Self {
        Self {
            running: Mutex::new(HashMap::new()),
            cfg,
            metrics: None,
            runtime: OnceLock::new(),
        }
    }

    /// Emit the fan-out drop/failure counters on `metrics`. Every
    /// pipeline this manager starts inherits the handle.
    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Run every pipeline started from now on on `runtime`. Called once at
    /// boot, before any pipeline starts; a later call is ignored.
    pub fn set_runtime(&self, runtime: tokio::runtime::Handle) {
        let _ = self.runtime.set(runtime);
    }

    /// Number of running pipelines.
    pub fn len(&self) -> usize {
        self.running.lock().len()
    }

    /// True when no pipeline is running.
    pub fn is_empty(&self) -> bool {
        self.running.lock().is_empty()
    }

    /// Resolve the pipeline handle for exporter `key`, lazily starting it via
    /// `build` on first sighting — or rebuilding it (stop old, start new) when
    /// `fingerprint` changed (the exporter's config was updated). `build` runs
    /// only when a (re)start is needed; an unchanged exporter just returns its
    /// existing handle. The returned handle is cheap to clone and enqueue into.
    pub fn get_or_create(
        &self,
        key: &str,
        fingerprint: u64,
        build: impl FnOnce() -> Arc<dyn ObservabilitySink>,
    ) -> SinkHandle {
        self.get_or_create_if(key, fingerprint, || true, build)
            .expect("admitted unconditionally")
    }

    /// As [`Self::get_or_create`], but a pipeline is (re)started only when
    /// `admit` holds. `admit` runs under the same lock as [`Self::retain`],
    /// so a caller that judges liveness from state published before a
    /// `retain` cannot start a pipeline that `retain` has already stopped.
    pub fn get_or_create_if(
        &self,
        key: &str,
        fingerprint: u64,
        admit: impl FnOnce() -> bool,
        build: impl FnOnce() -> Arc<dyn ObservabilitySink>,
    ) -> Option<SinkHandle> {
        let mut running = self.running.lock();
        if let Some(existing) = running.get(key) {
            if existing.fingerprint == fingerprint {
                return Some(existing.handle.clone());
            }
        }
        if !admit() {
            return None;
        }
        // Config changed — stop the stale pipeline, then rebuild below.
        if let Some(old) = running.remove(key) {
            let _ = old.cancel.send(true);
            tracing::info!(exporter = %key, "rebuilding reconfigured exporter pipeline");
        }
        let sink = build();
        let (handle, pipeline) =
            SinkPipeline::with_metrics(sink, self.cfg.clone(), self.metrics.clone());
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let worker = match self.runtime.get() {
            Some(runtime) => runtime.spawn(pipeline.run(cancel_rx)),
            None => tokio::spawn(pipeline.run(cancel_rx)),
        };
        running.insert(
            key.to_string(),
            Running {
                fingerprint,
                handle: handle.clone(),
                cancel: cancel_tx,
                worker,
            },
        );
        Some(handle)
    }

    /// Stop pipelines whose exporter key is not in `live`. Stopped pipelines
    /// drain their queue and exit on their own (cancel signal); they are not
    /// aborted.
    pub fn retain(&self, live: &HashSet<String>) {
        let mut running = self.running.lock();
        running.retain(|key, pipeline| {
            if live.contains(key) {
                true
            } else {
                let _ = pipeline.cancel.send(true);
                tracing::info!(exporter = %key, "stopping removed exporter pipeline");
                false
            }
        });
    }

    /// Per-exporter delivery stats, keyed by exporter name (health/dashboard).
    pub fn stats(&self) -> HashMap<String, SinkStatsSnapshot> {
        self.running
            .lock()
            .iter()
            .map(|(key, pipeline)| (key.clone(), pipeline.handle.stats()))
            .collect()
    }

    /// Records waiting in each running pipeline's queue, keyed by exporter
    /// name. A pipeline already stopped by a config change drains outside
    /// this map and is not counted.
    pub fn queue_depths(&self) -> Vec<(String, u64)> {
        self.running
            .lock()
            .iter()
            .map(|(key, pipeline)| (key.clone(), pipeline.handle.queued()))
            .collect()
    }

    /// Stop every pipeline and await its worker, each performing a final
    /// drain first. The gateway does not call this on exit: what is still
    /// queued at SIGTERM is dropped.
    pub async fn shutdown(&self) {
        let pipelines: Vec<Running> = self.running.lock().drain().map(|(_, p)| p).collect();
        for pipeline in &pipelines {
            let _ = pipeline.cancel.send(true);
        }
        for pipeline in pipelines {
            let _ = pipeline.worker.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::{
        BatchUnit, EventBatch, IdempotencyMarker, IdempotencyScheme, ObservabilitySink,
        OrderingScope, SinkAck, SinkCapabilities, SinkHealth, SinkRecord, SinkResult,
    };
    use crate::usage::UsageEvent;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    /// A sink that adds each delivered batch's size to a shared counter.
    struct CountingSink {
        delivered: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ObservabilitySink for CountingSink {
        fn name(&self) -> &str {
            "counting"
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
            self.delivered.fetch_add(batch.len(), Ordering::Relaxed);
            Ok(SinkAck {
                accepted: batch.len(),
                ..SinkAck::default()
            })
        }
        async fn healthcheck(&self) -> SinkHealth {
            SinkHealth::healthy()
        }
    }

    fn rec() -> Arc<SinkRecord> {
        Arc::new(SinkRecord::metadata_only(UsageEvent::default()))
    }

    fn manager() -> ExporterPipelines {
        // Long flush interval so the only flush is the shutdown drain —
        // keeps delivery assertions deterministic.
        let cfg = PipelineConfig {
            flush_interval: std::time::Duration::from_secs(60),
            ..PipelineConfig::default()
        };
        ExporterPipelines::new(cfg)
    }

    /// A `build` closure that counts invocations and shares a delivery counter.
    fn counting(
        builds: &Arc<AtomicU32>,
        delivered: &Arc<AtomicUsize>,
    ) -> impl FnOnce() -> Arc<dyn ObservabilitySink> {
        let builds = Arc::clone(builds);
        let delivered = Arc::clone(delivered);
        move || {
            builds.fetch_add(1, Ordering::Relaxed);
            Arc::new(CountingSink { delivered }) as Arc<dyn ObservabilitySink>
        }
    }

    #[tokio::test]
    async fn get_or_create_starts_once_and_reuses() {
        let mgr = manager();
        let builds = Arc::new(AtomicU32::new(0));
        let delivered = Arc::new(AtomicUsize::new(0));
        let h1 = mgr.get_or_create("a", 1, counting(&builds, &delivered));
        let h2 = mgr.get_or_create("a", 1, counting(&builds, &delivered));
        assert_eq!(
            builds.load(Ordering::Relaxed),
            1,
            "second call reuses the pipeline"
        );
        assert_eq!(mgr.len(), 1);

        // Both handles point at the same pipeline.
        h1.try_enqueue(rec());
        h2.try_enqueue(rec());
        mgr.shutdown().await;
        assert_eq!(delivered.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn get_or_create_rebuilds_on_fingerprint_change() {
        let mgr = manager();
        let builds = Arc::new(AtomicU32::new(0));
        let delivered = Arc::new(AtomicUsize::new(0));
        mgr.get_or_create("a", 1, counting(&builds, &delivered));
        mgr.get_or_create("a", 2, counting(&builds, &delivered));
        assert_eq!(
            builds.load(Ordering::Relaxed),
            2,
            "config change rebuilds the pipeline"
        );
        assert_eq!(mgr.len(), 1, "still exactly one pipeline for exporter a");
    }

    /// A sink that records the name of the thread each batch is delivered on.
    struct ThreadRecordingSink {
        delivered_on: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl ObservabilitySink for ThreadRecordingSink {
        fn name(&self) -> &str {
            "thread-recording"
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
            let thread = std::thread::current();
            self.delivered_on
                .lock()
                .push(thread.name().unwrap_or("unnamed").to_string());
            Ok(SinkAck {
                accepted: batch.len(),
                ..SinkAck::default()
            })
        }
        async fn healthcheck(&self) -> SinkHealth {
            SinkHealth::healthy()
        }
    }

    /// The first sighting comes from a request worker's single-thread
    /// runtime; delivery has to happen on the runtime the manager was given,
    /// including for the pipeline a config change rebuilds from that worker.
    #[tokio::test]
    async fn pipelines_run_on_the_given_runtime_not_the_callers() {
        let control = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("control-rt")
            .enable_all()
            .build()
            .unwrap();
        let mgr = manager();
        mgr.set_runtime(control.handle().clone());

        let delivered_on = Arc::new(Mutex::new(Vec::new()));
        let build = || {
            Arc::new(ThreadRecordingSink {
                delivered_on: Arc::clone(&delivered_on),
            }) as Arc<dyn ObservabilitySink>
        };
        mgr.get_or_create("a", 1, build).try_enqueue(rec());
        mgr.get_or_create("a", 2, build).try_enqueue(rec());
        // `shutdown` awaits only the rebuilt pipeline; the stale one drains
        // on its own.
        mgr.shutdown().await;
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while delivered_on.lock().len() < 2 {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("both pipelines deliver");

        let delivered_on = delivered_on.lock().clone();
        assert!(
            delivered_on.iter().all(|thread| thread == "control-rt"),
            "delivered on {delivered_on:?}"
        );
        control.shutdown_background();
    }

    #[tokio::test]
    async fn retain_stops_absent_exporters() {
        let mgr = manager();
        let builds = Arc::new(AtomicU32::new(0));
        let delivered = Arc::new(AtomicUsize::new(0));
        mgr.get_or_create("a", 1, counting(&builds, &delivered));
        mgr.get_or_create("b", 1, counting(&builds, &delivered));
        assert_eq!(mgr.len(), 2);

        let live: HashSet<String> = ["a".to_string()].into_iter().collect();
        mgr.retain(&live);
        assert_eq!(mgr.len(), 1, "exporter b's pipeline was stopped");
    }
}
