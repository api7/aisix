//! Per-exporter pipeline manager.
//!
//! Owns one [`SinkPipeline`] per configured exporter and keeps the running
//! set reconciled against the desired set (the etcd snapshot's exporters).
//! The request hot path enqueues a record into every running pipeline; a
//! reconcile loop (driven by snapshot version changes) starts pipelines for
//! new exporters, stops removed ones, and rebuilds reconfigured ones.
//!
//! Shared by every pipeline-backed sink family (`otlp`, `http_batch`, …) —
//! the manager is sink-agnostic: the caller supplies a `build` closure that
//! turns one desired spec into an [`ObservabilitySink`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use super::{
    ObservabilitySink, PipelineConfig, SinkHandle, SinkPipeline, SinkRecord, SinkStatsSnapshot,
};

/// One running pipeline plus the bookkeeping needed to stop/rebuild it.
struct Running {
    /// Hash of the exporter's delivery-relevant config. A change rebuilds
    /// the pipeline (old stops, new starts).
    fingerprint: u64,
    handle: SinkHandle,
    cancel: watch::Sender<bool>,
    worker: JoinHandle<()>,
}

/// Manages the live set of per-exporter pipelines.
pub struct ExporterPipelines {
    running: Mutex<HashMap<String, Running>>,
    cfg: PipelineConfig,
}

impl ExporterPipelines {
    /// Build an empty manager. Pipelines appear on the first [`reconcile`].
    ///
    /// [`reconcile`]: ExporterPipelines::reconcile
    pub fn new(cfg: PipelineConfig) -> Self {
        Self {
            running: Mutex::new(HashMap::new()),
            cfg,
        }
    }

    /// Number of running pipelines.
    pub fn len(&self) -> usize {
        self.running.lock().len()
    }

    /// True when no pipeline is running (no exporters configured).
    pub fn is_empty(&self) -> bool {
        self.running.lock().is_empty()
    }

    /// Enqueue `record` into every running pipeline (fan-out). Non-blocking:
    /// a full queue drops the record on that pipeline (counted there). Cheap
    /// — the record is `Arc`-shared, so each pipeline gets a pointer clone.
    pub fn enqueue_to_all(&self, record: &Arc<SinkRecord>) {
        let running = self.running.lock();
        for pipeline in running.values() {
            pipeline.handle.try_enqueue(Arc::clone(record));
        }
    }

    /// Reconcile the running pipelines against `desired`:
    /// - start a pipeline for every desired key that isn't running;
    /// - stop pipelines whose key is no longer desired;
    /// - rebuild a pipeline whose `fingerprint` changed (stop old, start new).
    ///
    /// `build` is invoked only for new or changed exporters, so unchanged
    /// pipelines keep running untouched. Stopped pipelines drain their queue
    /// and exit on their own (cancel signal); they are not aborted.
    pub fn reconcile<T>(
        &self,
        desired: &[T],
        key: impl Fn(&T) -> String,
        fingerprint: impl Fn(&T) -> u64,
        build: impl Fn(&T) -> Arc<dyn ObservabilitySink>,
    ) {
        let mut running = self.running.lock();

        // Stop pipelines whose exporter disappeared.
        let desired_keys: HashSet<String> = desired.iter().map(&key).collect();
        running.retain(|name, pipeline| {
            if desired_keys.contains(name) {
                true
            } else {
                let _ = pipeline.cancel.send(true);
                tracing::info!(exporter = %name, "stopping removed exporter pipeline");
                false
            }
        });

        // Start new / rebuild changed.
        for spec in desired {
            let name = key(spec);
            let fp = fingerprint(spec);
            match running.get(&name) {
                Some(existing) if existing.fingerprint == fp => continue,
                Some(_) => {
                    if let Some(old) = running.remove(&name) {
                        let _ = old.cancel.send(true);
                        tracing::info!(exporter = %name, "rebuilding reconfigured exporter pipeline");
                    }
                }
                None => {}
            }

            let sink = build(spec);
            let (handle, pipeline) = SinkPipeline::new(sink, self.cfg.clone());
            let (cancel_tx, cancel_rx) = watch::channel(false);
            let worker = tokio::spawn(pipeline.run(cancel_rx));
            running.insert(
                name,
                Running {
                    fingerprint: fp,
                    handle,
                    cancel: cancel_tx,
                    worker,
                },
            );
        }
    }

    /// Per-exporter delivery stats, keyed by exporter name (health/dashboard).
    pub fn stats(&self) -> HashMap<String, SinkStatsSnapshot> {
        self.running
            .lock()
            .iter()
            .map(|(name, pipeline)| (name.clone(), pipeline.handle.stats()))
            .collect()
    }

    /// Stop every pipeline and await its worker (graceful shutdown). Each
    /// pipeline performs a final drain before exiting.
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

    /// A sink that adds each delivered batch's size to a shared counter, so
    /// tests can assert total fan-out delivery without tracking each sink.
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

    /// A desired exporter spec for the test reconcile calls.
    struct Spec {
        name: &'static str,
        fp: u64,
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

    #[tokio::test]
    async fn reconcile_starts_a_pipeline_per_exporter() {
        let mgr = manager();
        let delivered = Arc::new(AtomicUsize::new(0));
        let d = Arc::clone(&delivered);
        mgr.reconcile(
            &[Spec { name: "a", fp: 1 }, Spec { name: "b", fp: 1 }],
            |s| s.name.to_string(),
            |s| s.fp,
            move |_| {
                Arc::new(CountingSink {
                    delivered: Arc::clone(&d),
                }) as Arc<dyn ObservabilitySink>
            },
        );
        assert_eq!(mgr.len(), 2);
        assert!(!mgr.is_empty());
    }

    #[tokio::test]
    async fn enqueue_fans_out_to_every_pipeline() {
        let mgr = manager();
        let delivered = Arc::new(AtomicUsize::new(0));
        let d = Arc::clone(&delivered);
        mgr.reconcile(
            &[Spec { name: "a", fp: 1 }, Spec { name: "b", fp: 1 }],
            |s| s.name.to_string(),
            |s| s.fp,
            move |_| {
                Arc::new(CountingSink {
                    delivered: Arc::clone(&d),
                }) as Arc<dyn ObservabilitySink>
            },
        );
        mgr.enqueue_to_all(&rec());
        mgr.shutdown().await; // drains both pipelines

        assert_eq!(
            delivered.load(Ordering::Relaxed),
            2,
            "one record to each of two sinks"
        );
    }

    #[tokio::test]
    async fn reconcile_stops_removed_exporters() {
        let mgr = manager();
        let d = Arc::new(AtomicUsize::new(0));
        let build = |d: Arc<AtomicUsize>| {
            move |_: &Spec| {
                Arc::new(CountingSink {
                    delivered: Arc::clone(&d),
                }) as Arc<dyn ObservabilitySink>
            }
        };
        mgr.reconcile(
            &[Spec { name: "a", fp: 1 }, Spec { name: "b", fp: 1 }],
            |s| s.name.to_string(),
            |s| s.fp,
            build(Arc::clone(&d)),
        );
        assert_eq!(mgr.len(), 2);

        mgr.reconcile(
            &[Spec { name: "a", fp: 1 }],
            |s| s.name.to_string(),
            |s| s.fp,
            build(Arc::clone(&d)),
        );
        assert_eq!(mgr.len(), 1, "exporter b's pipeline was stopped");
    }

    #[tokio::test]
    async fn reconcile_is_idempotent_for_unchanged_exporters() {
        let mgr = manager();
        let builds = Arc::new(AtomicU32::new(0));
        let d = Arc::new(AtomicUsize::new(0));
        let make = |builds: Arc<AtomicU32>, d: Arc<AtomicUsize>| {
            move |_: &Spec| {
                builds.fetch_add(1, Ordering::Relaxed);
                Arc::new(CountingSink {
                    delivered: Arc::clone(&d),
                }) as Arc<dyn ObservabilitySink>
            }
        };
        let specs = [Spec { name: "a", fp: 1 }];
        mgr.reconcile(
            &specs,
            |s| s.name.to_string(),
            |s| s.fp,
            make(Arc::clone(&builds), Arc::clone(&d)),
        );
        mgr.reconcile(
            &specs,
            |s| s.name.to_string(),
            |s| s.fp,
            make(Arc::clone(&builds), Arc::clone(&d)),
        );
        assert_eq!(
            builds.load(Ordering::Relaxed),
            1,
            "unchanged exporter is not rebuilt"
        );
        assert_eq!(mgr.len(), 1);
    }

    #[tokio::test]
    async fn reconcile_rebuilds_on_fingerprint_change() {
        let mgr = manager();
        let builds = Arc::new(AtomicU32::new(0));
        let d = Arc::new(AtomicUsize::new(0));
        let make = |builds: Arc<AtomicU32>, d: Arc<AtomicUsize>| {
            move |_: &Spec| {
                builds.fetch_add(1, Ordering::Relaxed);
                Arc::new(CountingSink {
                    delivered: Arc::clone(&d),
                }) as Arc<dyn ObservabilitySink>
            }
        };
        mgr.reconcile(
            &[Spec { name: "a", fp: 1 }],
            |s| s.name.to_string(),
            |s| s.fp,
            make(Arc::clone(&builds), Arc::clone(&d)),
        );
        mgr.reconcile(
            &[Spec { name: "a", fp: 2 }],
            |s| s.name.to_string(),
            |s| s.fp,
            make(Arc::clone(&builds), Arc::clone(&d)),
        );
        assert_eq!(
            builds.load(Ordering::Relaxed),
            2,
            "config change rebuilds the pipeline"
        );
        assert_eq!(mgr.len(), 1, "still exactly one pipeline for exporter a");
    }
}
