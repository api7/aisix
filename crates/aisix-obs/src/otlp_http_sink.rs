//! Per-env OTLP/HTTP exporter — emits one OTLP-shaped span per chat request
//! to each configured `ObservabilityExporter` (kind=otlp_http).
//!
//! ## Design
//!
//! cp-api projects every configured exporter onto kine at
//! `/aisix/<env>/observability_exporters/<uuid>`. The DP loads them via the
//! existing etcd watch into `AisixSnapshot::observability_exporters`. After
//! every chat completion the proxy hot path hands the resulting `UsageEvent`
//! plus the live snapshot's exporter list to [`OtlpHttpFanOut::fan_out`],
//! which:
//!
//! 1. Filters to enabled exporters with `kind = OtlpHttp`.
//! 2. Resolves each exporter's [`crate::sink::SinkPipeline`] (lazily started
//!    on first sighting, immediately consistent with the snapshot) and
//!    enqueues one OTLP span record into it. The pipeline batches, retries
//!    transient failures with backoff, and drops-with-metric under
//!    backpressure — all off the request hot path. Spans are encoded per
//!    OpenTelemetry's GenAI semantic conventions as released in
//!    semantic-conventions v1.41.1
//!    (<https://github.com/open-telemetry/semantic-conventions/blob/v1.41.1/docs/gen-ai/gen-ai-spans.md>),
//!    the SERVER span per the HTTP server span conventions of the same
//!    release.
//!
//! ## Per-exporter knobs (#519 B.2)
//!
//! - **`sample_rate`** — fraction of requests exported, decided per REQUEST
//!   from a deterministic FNV-1a hash of `request_id` (absent = 1.0). The
//!   decision happens at fan-out time, before any pipeline work, so a
//!   dropped request enqueues nothing for that exporter; and because every
//!   attempt span of one request shares the `request_id`, a trace is
//!   exported whole or not at all — consistently across retries/fallbacks.
//! - **`content_mode` / `content_max_bytes`** — same opt-in content capture
//!   the SLS / Datadog sinks use ([`content_record`]); under `full` a
//!   complete message exchange is carried as the conventions' structured
//!   `gen_ai.input.messages` / `gen_ai.output.messages` /
//!   `gen_ai.system_instructions` / `gen_ai.tool.definitions`, and anything
//!   else as `aisix.content.prompt` / `aisix.content.response` (plus
//!   `aisix.content_truncated` when cut). Defaults to `metadata_only`, where
//!   the record never carries content fields, so it cannot leak.
//!
//! ## What's intentionally NOT here yet
//!
//! - **No gRPC** — `otlp_grpc` is a separate kind we'll add when a user
//!   actually asks for it; the JSON-over-HTTP form works against every
//!   receiver in the wild and avoids pulling in tonic on the hot path.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{
    AisixSnapshot, AliyunSlsConfig, DatadogConfig, ExporterKind, ObjectStoreConfig,
    ObservabilityExporter, OtlpHttpConfig, SlsContentMode,
};
use aisix_core::snapshot::SnapshotHandle;
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::metrics::Metrics;
use crate::sink::is_mcp;
use crate::sink::{
    build_object_store_sink, resolve_datadog_credential, resolve_sls_credential, AliyunSlsSink,
    BatchUnit, CapturedContent, DatadogSink, ErrorRedactor, EventBatch, ExporterPipelines,
    IdempotencyMarker, IdempotencyScheme, ObservabilitySink, OrderingScope, PipelineConfig,
    SinkAck, SinkCapabilities, SinkContent, SinkError, SinkHealth, SinkRecord, SinkResult,
    SinkStatsSnapshot,
};
use crate::usage::{GatewayEmbeddingCall, UsageEvent};

/// Wall-clock duration of an OTLP/HTTP POST before we abandon it.
/// Tight on purpose — we never want a slow exporter to backlog tokio
/// tasks for a wedged user receiver.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// `service.name` until the gateway's configured
/// `observability.service_name` is handed over — the same default that key
/// has.
const DEFAULT_SERVICE_NAME: &str = "aisix";

/// The gateway's status for a caller that closed the request (nginx's 499).
const CLIENT_CLOSED_REQUEST: u16 = 499;

/// Fans usage events out to every configured observability exporter — any
/// [`ExporterKind`], dispatched per kind to the matching
/// [`crate::sink::ObservabilitySink`] — each via its own
/// [`crate::sink::SinkPipeline`] (batched, retried, backpressured). Cheap
/// clonable handle; the per-exporter pipelines and the shared `reqwest::Client`
/// live behind an `Arc`. Pipelines start lazily on first sighting of an
/// exporter (immediately consistent with the snapshot) and are stopped by
/// [`OtlpHttpFanOut::reap_on_config_change`] when an exporter leaves it or
/// is disabled.
///
/// NOTE: the type name is historical — it drove only `otlp_http` originally
/// and now fans out all kinds. A rename to `ExporterFanOut` (plus the
/// `ProxyState::otlp_fan_out` field) is a mechanical, behaviour-preserving
/// follow-up kept out of this change to avoid churning every call site.
#[derive(Clone)]
pub struct OtlpHttpFanOut {
    inner: Arc<FanOutInner>,
}

struct FanOutInner {
    /// Per-exporter delivery pipelines (one batched worker each).
    exporters: ExporterPipelines,
    /// Shared HTTP client handed to every sink (connection-pool reuse).
    client: reqwest::Client,
    /// The published snapshot, once [`OtlpHttpFanOut::reap_on_config_change`]
    /// runs: a pipeline starts only for an exporter it still holds enabled,
    /// so a request carrying an older snapshot cannot restart one the reaper
    /// has stopped.
    published: std::sync::OnceLock<SnapshotHandle<AisixSnapshot>>,
    /// The resource `service.name` every OTLP export carries, once
    /// [`OtlpHttpFanOut::set_service_name`] hands it over.
    service_name: std::sync::OnceLock<String>,
}

/// Delivery tuning for otlp exporter pipelines. A short flush keeps a
/// single-request span visible quickly (the old fan-out posted each event
/// immediately); batching + retry + drop accounting come from the pipeline.
fn exporter_pipeline_config() -> PipelineConfig {
    PipelineConfig {
        flush_interval: Duration::from_secs(1),
        ..PipelineConfig::default()
    }
}

impl OtlpHttpFanOut {
    pub fn new() -> Self {
        Self::build(None)
    }

    /// As [`Self::new`], emitting the per-exporter fan-out drop and
    /// failure counters on `metrics`. Without it the fan-out still runs
    /// and still accounts drops in [`SinkStatsSnapshot`] — but nothing
    /// reaches `GET /metrics`, which is the gap #1060 names.
    pub fn with_metrics(metrics: Metrics) -> Self {
        Self::build(Some(metrics))
    }

    fn build(metrics: Option<Metrics>) -> Self {
        let client = aisix_gateway::client_builder()
            .timeout(REQUEST_TIMEOUT)
            .user_agent(format!("aisix-dp/{}", &*aisix_core::BUILD_VERSION))
            .build()
            // The client builder only fails on illegal TLS roots; the
            // default config is always valid.
            .expect("reqwest::Client default config is valid");
        Self {
            inner: Arc::new(FanOutInner {
                exporters: {
                    let p = ExporterPipelines::new(exporter_pipeline_config());
                    match metrics {
                        Some(m) => p.with_metrics(m),
                        None => p,
                    }
                },
                client,
                published: std::sync::OnceLock::new(),
                service_name: std::sync::OnceLock::new(),
            }),
        }
    }

    /// Name the service OTLP exports report as `service.name` — the
    /// gateway's `observability.service_name`. Called once at boot, before
    /// any exporter pipeline starts; a later call is ignored.
    pub fn set_service_name(&self, name: &str) {
        let _ = self.inner.service_name.set(name.to_string());
    }

    /// Whether a pipeline may start for exporter `name`: the published
    /// snapshot still has it enabled (always, before the reaper runs).
    fn admits(&self, name: &str) -> bool {
        self.inner.published.get().is_none_or(|snapshot| {
            snapshot
                .load()
                .observability_exporters
                .entries()
                .iter()
                .any(|e| e.value.enabled && e.value.name == name)
        })
    }

    /// Fan one event out to every enabled exporter, dispatched per kind and
    /// enqueued into that exporter's pipeline (lazily started on first
    /// sighting). The pipeline owns batching / retry / backpressure; enqueue is
    /// non-blocking, so this never blocks the request hot path. Empty list =
    /// cheap no-op.
    ///
    /// `content` is the request's captured prompt/response, or `None` when the
    /// handler captured none (the default). It is attached ONLY to an
    /// `aliyun_sls`, `datadog`, or `otlp_http` exporter whose
    /// `content_mode = full`; every other exporter — and the CP telemetry
    /// path, which is not in this loop — receives the shared metadata-only
    /// record, so prompt/response can never leak there.
    ///
    /// `trace` is the event's span snapshot (AISIX-Cloud#1279) — ids and
    /// boundaries frozen at emission time by the request's trace bundle.
    /// It rides every record unchanged (`SinkRecord::trace` is
    /// serde-skipped, so only the OTLP encoder reads it), which is what
    /// makes span ids byte-identical across delivery retries and across
    /// exporters. `None` (a caller predating the bundle, or a synthetic
    /// event) falls back to the legacy single-span encoding.
    ///
    /// `facts` are the event's export-only span facts — request parameters,
    /// the dispatched upstream, the inbound HTTP request — which never join
    /// the usage-event wire.
    pub fn fan_out<'a, I>(
        &self,
        event: &UsageEvent,
        content: Option<&CapturedContent>,
        trace: Option<&crate::trace::TraceEmission>,
        facts: Option<&Arc<crate::genai::SpanFacts>>,
        exporters: I,
    ) where
        I: IntoIterator<Item = &'a ObservabilityExporter>,
    {
        // The shared metadata-only record, built once on first sighting and
        // reused by every exporter that does not capture content.
        let mut metadata_record: Option<Arc<SinkRecord>> = None;
        for exp in exporters {
            if !exp.enabled {
                continue;
            }

            // Dispatch on kind: each arm fingerprints its delivery-relevant
            // config (so a dashboard edit rebuilds the pipeline) and lazily
            // builds the matching sink. All kinds share one `ExporterPipelines`
            // manager and the pooled client. A new `ExporterKind` variant
            // makes this `match` non-exhaustive — a compile-time prompt to wire
            // its sink here.
            let client = self.inner.client.clone();
            let handle = match &exp.kind {
                ExporterKind::OtlpHttp(cfg) => {
                    // Per-request sampling (#519 B.2), decided here — before
                    // any pipeline work — so an unsampled request enqueues
                    // nothing for THIS exporter (other exporters in the loop
                    // still see the event). Deterministic on `request_id`, so
                    // every attempt span of one request drops or ships
                    // together.
                    if !otlp_should_sample(cfg.sample_rate, &event.request_id) {
                        continue;
                    }
                    let fingerprint = fingerprint_otlp(cfg);
                    let name = exp.name.clone();
                    let endpoint = cfg.endpoint.clone();
                    let headers = cfg.headers.clone();
                    let service_name = self
                        .inner
                        .service_name
                        .get()
                        .cloned()
                        .unwrap_or_else(|| DEFAULT_SERVICE_NAME.to_string());
                    self.inner.exporters.get_or_create_if(
                        &exp.name,
                        fingerprint,
                        || self.admits(&exp.name),
                        move || {
                            Arc::new(
                                OtlpSink::new(name, endpoint, headers, client)
                                    .with_service_name(service_name),
                            ) as Arc<dyn ObservabilitySink>
                        },
                    )
                }
                ExporterKind::AliyunSls(cfg) => {
                    let fingerprint = fingerprint_sls(cfg);
                    let name = exp.name.clone();
                    let cfg = cfg.clone();
                    self.inner.exporters.get_or_create_if(
                        &exp.name,
                        fingerprint,
                        || self.admits(&exp.name),
                        move || {
                            // Resolve the AccessKey from the DP's local env at
                            // build time (the key never rode the kine path).
                            // Missing creds → empty key → SLS 401 surfaces as a
                            // delivery-health auth error, not a silent drop.
                            let (ak_id, ak_secret) =
                                resolve_sls_credential(&cfg.credential_ref).unwrap_or_default();
                            Arc::new(AliyunSlsSink::new(
                                name,
                                &cfg.endpoint,
                                &cfg.project,
                                &cfg.logstore,
                                ak_id,
                                ak_secret,
                                client,
                            )) as Arc<dyn ObservabilitySink>
                        },
                    )
                }
                ExporterKind::ObjectStore(cfg) => {
                    let fingerprint = fingerprint_object_store(cfg);
                    let name = exp.name.clone();
                    let cfg = cfg.clone();
                    self.inner.exporters.get_or_create_if(
                        &exp.name,
                        fingerprint,
                        || self.admits(&exp.name),
                        move || {
                            // Resolve cloud creds from the DP's local env and
                            // build the backend at build time. Missing creds or
                            // an un-buildable backend yield a sink that reports
                            // the reason via delivery health (never a silent
                            // drop) — mirroring the SLS path.
                            build_object_store_sink(name, &cfg)
                        },
                    )
                }
                ExporterKind::Datadog(cfg) => {
                    let fingerprint = fingerprint_datadog(cfg);
                    let name = exp.name.clone();
                    let cfg = cfg.clone();
                    self.inner.exporters.get_or_create_if(
                        &exp.name,
                        fingerprint,
                        || self.admits(&exp.name),
                        move || {
                            // Resolve the Datadog API key from the DP's local
                            // env at build time (the key never rode the kine
                            // path). Missing key → empty → Datadog 403 surfaces
                            // as a delivery-health auth error, not a silent drop
                            // — mirroring the SLS path.
                            let api_key =
                                resolve_datadog_credential(&cfg.credential_ref).unwrap_or_default();
                            Arc::new(DatadogSink::new(
                                name,
                                &cfg.site,
                                api_key,
                                &cfg.ddsource,
                                &cfg.tags,
                                &cfg.service,
                                client,
                            )) as Arc<dyn ObservabilitySink>
                        },
                    )
                }
            };

            // The exporter left the published snapshot after this request
            // loaded its own: its pipeline stays stopped.
            let Some(handle) = handle else { continue };

            // A content-bearing record for an SLS exporter that opted into
            // full capture (and only when the handler captured content); the
            // shared metadata-only record for everyone else.
            // Each exporter sees only the headers in its own lists, so one
            // that selects any gets a record of its own.
            let selects_headers =
                !exp.request_headers.is_empty() || !exp.upstream_response_headers.is_empty();
            let record = content_record(&exp.kind, event, content, trace, facts)
                .map(|record| {
                    if selects_headers {
                        Arc::new(Arc::unwrap_or_clone(record).with_exporter_headers(
                            &exp.request_headers,
                            &exp.upstream_response_headers,
                        ))
                    } else {
                        record
                    }
                })
                .unwrap_or_else(|| {
                    if selects_headers {
                        return Arc::new(
                            SinkRecord::metadata_only(event.clone())
                                .with_trace(trace.cloned())
                                .with_facts(facts.cloned())
                                .with_exporter_headers(
                                    &exp.request_headers,
                                    &exp.upstream_response_headers,
                                ),
                        );
                    }
                    Arc::clone(metadata_record.get_or_insert_with(|| {
                        Arc::new(
                            SinkRecord::metadata_only(event.clone())
                                .with_trace(trace.cloned())
                                .with_facts(facts.cloned()),
                        )
                    }))
                });
            handle.try_enqueue(record);
        }
    }

    /// Stop pipelines for exporters no longer present in `live` (the current
    /// snapshot's enabled exporter names, across all kinds).
    pub fn gc(&self, live: &std::collections::HashSet<String>) {
        self.inner.exporters.retain(live);
    }

    /// Stop the pipeline of every exporter that was deleted or disabled,
    /// each time a snapshot is published, until `cancel` fires. Without it
    /// the pipeline — task, client, queue — outlives its exporter for the
    /// life of the process (#954).
    ///
    /// Driven by the publish rather than by traffic or a timer: a deleted
    /// exporter is by definition one no request will touch again. It keys
    /// on the exporter table's generation, so a write to any other
    /// collection costs one comparison. Stopping only signals the worker,
    /// wherever it runs — under `proxy.thread_per_core` a pipeline lives on
    /// the worker runtime whose request first saw the exporter, not on the
    /// runtime running this loop.
    pub async fn reap_on_config_change(
        self,
        snapshot: SnapshotHandle<AisixSnapshot>,
        mut cancel: tokio::sync::watch::Receiver<bool>,
    ) {
        let _ = self.inner.published.set(snapshot.clone());
        let mut published = snapshot.subscribe();
        let mut reaped = None;
        loop {
            if *cancel.borrow() {
                return;
            }
            let snap = snapshot.load();
            let generation = snap.observability_exporters.generation();
            if reaped != Some(generation) {
                let live = snap
                    .observability_exporters
                    .entries()
                    .iter()
                    .filter(|e| e.value.enabled)
                    .map(|e| e.value.name.clone())
                    .collect();
                self.gc(&live);
                reaped = Some(generation);
            }
            tokio::select! {
                changed = published.changed() => if changed.is_err() { return },
                changed = cancel.changed() => if changed.is_err() { return },
            }
        }
    }

    /// Per-exporter delivery counters, keyed by exporter name. Read by the
    /// managed-mode heartbeat to report `exporter_health` to cp-api
    /// (#519 D.2). Counters are per-pipeline: a reconfigured exporter gets
    /// a rebuilt pipeline, so its counters reset — consumers must treat
    /// them as resettable.
    pub fn exporter_stats(&self) -> std::collections::HashMap<String, SinkStatsSnapshot> {
        self.inner.exporters.stats()
    }

    /// Records queued per exporter, keyed by exporter name.
    pub fn exporter_queue_depths(&self) -> Vec<(String, u64)> {
        self.inner.exporters.queue_depths()
    }

    /// Drain every exporter pipeline at graceful shutdown.
    pub async fn shutdown(&self) {
        self.inner.exporters.shutdown().await;
    }
}

impl Default for OtlpHttpFanOut {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash an `otlp_http` exporter's delivery-relevant config. A change to any
/// field — endpoint, headers, or the #519 B.2 knobs (sample_rate / content
/// config) — yields a new fingerprint, so the manager rebuilds the exporter's
/// pipeline against the edited config. `sample_rate` hashes by bit pattern
/// (`f64` has no `Hash`); the schema bounds it to finite 0.0–1.0, where bit
/// equality is value equality.
fn fingerprint_otlp(cfg: &OtlpHttpConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cfg.endpoint.hash(&mut hasher);
    cfg.headers.hash(&mut hasher);
    cfg.sample_rate.map(f64::to_bits).hash(&mut hasher);
    cfg.content_mode.hash(&mut hasher);
    cfg.content_max_bytes.hash(&mut hasher);
    hasher.finish()
}

/// Granularity of the sampling decision: rates are compared at 1/10_000
/// (0.01%) resolution, plenty for an operator-facing percentage knob.
const SAMPLE_PRECISION: u64 = 10_000;

/// Salt mixed into the sampling hash, drawn once per process.
///
/// Without it the bucket is a pure function of `request_id` — and since
/// AISIX-Cloud#1288 the caller may choose that id. A caller could then grind
/// FNV-1a (which is trivially cheap) for ids that land outside the configured
/// bucket and make its own traffic invisible to a sampled exporter, or inside
/// it to force itself into every trace. The salt is not caller-observable, so
/// the bucket is no longer predictable off the wire.
static SAMPLE_SALT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

fn sample_salt() -> u64 {
    *SAMPLE_SALT.get_or_init(|| uuid::Uuid::new_v4().as_u128() as u64)
}

/// Per-request sampling decision for an `otlp_http` exporter. `rate` is the
/// exporter's `sample_rate` (`None` = 1.0, the pre-knob default).
///
/// No clock: the request's `request_id` is hashed (FNV-1a 64) together with a
/// per-process salt into a bucket in `0..SAMPLE_PRECISION`, and the request is
/// sampled when the bucket falls below `rate × SAMPLE_PRECISION`. Within a
/// process the same id always gets the same decision, so the per-attempt
/// events of one request (#655, which share `request_id`) are exported
/// all-or-nothing — they are emitted by the process that served the request,
/// which is the only place that guarantee is needed.
///
/// Replicas therefore sample different subsets, and a restart reshuffles.
/// That is the deliberate cost of the salt (see [`SAMPLE_SALT`]); the property
/// it replaces — every replica choosing the identical set — only had meaning
/// when ids were gateway-minted UUIDs that could not repeat across replicas.
///
/// Note this samples per REQUEST, not per caller: a caller that reuses one id
/// across many requests still gets one decision for all of them, so a heavily
/// id-reusing client can skew what fraction of total traffic an exporter sees.
fn otlp_should_sample(rate: Option<f64>, request_id: &str) -> bool {
    let rate = rate.unwrap_or(1.0);
    if rate >= 1.0 {
        return true;
    }
    if rate <= 0.0 {
        return false;
    }
    let threshold = (rate * SAMPLE_PRECISION as f64).round() as u64;
    let mut salted = [0u8; 8].to_vec();
    salted.copy_from_slice(&sample_salt().to_le_bytes());
    salted.extend_from_slice(request_id.as_bytes());
    fnv1a_64(&salted) % SAMPLE_PRECISION < threshold
}

/// FNV-1a 64-bit — implemented inline (a fold over two constants) so the
/// sampling hash is under our control and documented, rather than tied to
/// `DefaultHasher`'s unspecified algorithm or a `rand` dependency.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |hash, b| {
        (hash ^ u64::from(*b)).wrapping_mul(PRIME)
    })
}

/// Hash an `aliyun_sls` exporter's delivery-relevant config. The fingerprint
/// covers only kine-visible fields (endpoint / project / logstore /
/// credential_ref), never the resolved AccessKey — rotating the secret under
/// the *same* reference therefore takes effect on the next DP restart, not
/// live. A ref change does rebuild the pipeline.
fn fingerprint_sls(cfg: &AliyunSlsConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cfg.endpoint.hash(&mut hasher);
    cfg.project.hash(&mut hasher);
    cfg.logstore.hash(&mut hasher);
    cfg.credential_ref.hash(&mut hasher);
    hasher.finish()
}

/// The content-bearing [`SinkRecord`] for one exporter, or `None` to fall back
/// to the shared metadata-only record.
///
/// Content is attached ONLY to an `aliyun_sls`, `datadog`, OR `otlp_http`
/// exporter whose `content_mode = full`, and ONLY when the handler captured
/// content — every other exporter (and the CP telemetry path, which never
/// enters the fan-out) gets metadata only. The captured prompt/response are
/// truncated to the exporter's `content_max_bytes`, ORing in any truncation
/// the handler already applied at capture time.
fn content_record(
    kind: &ExporterKind,
    event: &UsageEvent,
    content: Option<&CapturedContent>,
    trace: Option<&crate::trace::TraceEmission>,
    facts: Option<&Arc<crate::genai::SpanFacts>>,
) -> Option<Arc<SinkRecord>> {
    // The exporters that opt into full content capture share the
    // `SlsContentMode` model; pull each one's (mode, cap). Any other kind
    // never carries content, so prompt/response can't leak into it.
    let (mode, max_bytes) = match kind {
        ExporterKind::AliyunSls(cfg) => (cfg.content_mode, cfg.content_max_bytes),
        ExporterKind::Datadog(cfg) => (cfg.content_mode, cfg.content_max_bytes),
        ExporterKind::OtlpHttp(cfg) => (cfg.content_mode, cfg.content_max_bytes),
        _ => return None,
    };
    if mode != SlsContentMode::Full {
        return None;
    }
    let captured = content?;
    let mut sc = SinkContent::capture(&captured.prompt, &captured.response, max_bytes as usize);
    sc.truncated = sc.truncated || captured.truncated;
    sc.prompt_truncated = sc.prompt_truncated || captured.prompt_truncated;
    sc.response_truncated = sc.response_truncated || captured.response_truncated;
    sc.output = captured.output.clone();
    Some(Arc::new(
        SinkRecord::metadata_only(event.clone())
            .with_content(sc)
            .with_trace(trace.cloned())
            .with_facts(facts.cloned()),
    ))
}

/// The largest `content_max_bytes` among the env's enabled exporters that
/// capture full content, or `None` if none do.
///
/// A request handler calls this BEFORE doing any capture work: `None` means no
/// exporter wants prompt/response, so the handler skips capture entirely (no
/// body clone, no stream accumulation — zero hot-path cost). `Some(cap)` is the
/// bound the handler caps its capture at; each exporter then re-truncates to
/// its own (≤ cap) limit at delivery.
pub fn content_capture_cap<'a>(
    exporters: impl IntoIterator<Item = &'a ObservabilityExporter>,
) -> Option<u32> {
    exporters
        .into_iter()
        .filter(|e| e.enabled)
        .filter_map(|e| match &e.kind {
            ExporterKind::AliyunSls(cfg) if cfg.content_mode == SlsContentMode::Full => {
                Some(cfg.content_max_bytes)
            }
            ExporterKind::Datadog(cfg) if cfg.content_mode == SlsContentMode::Full => {
                Some(cfg.content_max_bytes)
            }
            ExporterKind::OtlpHttp(cfg) if cfg.content_mode == SlsContentMode::Full => {
                Some(cfg.content_max_bytes)
            }
            _ => None,
        })
        .max()
}

/// Hash an `object_store` exporter's delivery-relevant config. Covers only
/// kine-visible fields (provider / bucket / prefix / region / endpoint /
/// compression / credential_ref), never the resolved cloud key — rotating the
/// secret under the *same* reference therefore takes effect on the next DP
/// restart, not live. Any field change rebuilds the pipeline.
fn fingerprint_object_store(cfg: &ObjectStoreConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cfg.provider.hash(&mut hasher);
    cfg.bucket.hash(&mut hasher);
    cfg.prefix.hash(&mut hasher);
    cfg.region.hash(&mut hasher);
    cfg.endpoint.hash(&mut hasher);
    cfg.compression.hash(&mut hasher);
    cfg.credential_ref.hash(&mut hasher);
    hasher.finish()
}

/// Hash a `datadog` exporter's delivery-relevant config. Covers only
/// kine-visible fields (site / credential_ref / service / ddsource / tags /
/// content config), never the resolved API key — rotating the secret under the
/// *same* reference therefore takes effect on the next DP restart, not live. A
/// ref change (or any other field change) rebuilds the pipeline.
fn fingerprint_datadog(cfg: &DatadogConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cfg.site.hash(&mut hasher);
    cfg.credential_ref.hash(&mut hasher);
    cfg.service.hash(&mut hasher);
    cfg.ddsource.hash(&mut hasher);
    cfg.tags.hash(&mut hasher);
    cfg.content_mode.hash(&mut hasher);
    cfg.content_max_bytes.hash(&mut hasher);
    hasher.finish()
}

/// An [`ObservabilitySink`] over the OTLP/HTTP-JSON traces protocol — the
/// same wire shape as [`OtlpHttpFanOut`], but driven by the shared
/// [`crate::sink::SinkPipeline`] (batched, retried, backpressured) rather
/// than a per-event fire-and-forget spawn. One instance per configured
/// `otlp_http` exporter.
pub struct OtlpSink {
    name: String,
    endpoint: String,
    headers: BTreeMap<String, String>,
    client: reqwest::Client,
    service_name: String,
}

impl OtlpSink {
    /// Build a sink for one exporter. The `client` is shared across sinks so
    /// connection pools and TLS sessions are reused.
    pub fn new(
        name: impl Into<String>,
        endpoint: impl Into<String>,
        headers: BTreeMap<String, String>,
        client: reqwest::Client,
    ) -> Self {
        Self {
            name: name.into(),
            endpoint: endpoint.into(),
            headers,
            client,
            service_name: DEFAULT_SERVICE_NAME.to_string(),
        }
    }

    /// The resource `service.name` this sink's exports report.
    pub fn with_service_name(mut self, service_name: String) -> Self {
        self.service_name = service_name;
        self
    }
}

#[async_trait]
impl ObservabilitySink for OtlpSink {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities {
            idempotency: IdempotencyScheme::None,
            ordering: OrderingScope::None,
            batch_unit: BatchUnit::Records,
            // OTLP spans are small and receivers accept large payloads; the
            // sink does not split by bytes, so no pipeline-enforced ceiling.
            max_batch_bytes: None,
            supports_partial_batch: false,
            supports_streaming_ingest: false,
        }
    }

    async fn append_batch(&self, batch: &EventBatch, _marker: &IdempotencyMarker) -> SinkResult {
        if batch.is_empty() {
            return Ok(SinkAck::default());
        }
        // One export request carrying every record's spans — one POST, one
        // atomic retry unit (vs. the per-event fan-out's N spawns).
        let spans: Vec<Value> = batch
            .records
            .iter()
            .flat_map(|record| build_otlp_spans(record, &self.name))
            .collect();
        let body = otlp_export_request(spans, &self.service_name);
        let bytes = serde_json::to_vec(&body)
            .map_err(|e| SinkError::Permanent(format!("otlp encode: {e}")))?;

        let mut req = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .body(bytes);
        for (key, value) in &self.headers {
            req = req.header(key, value);
        }

        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(SinkAck {
                        accepted: batch.len(),
                        ..SinkAck::default()
                    });
                }
                let retry_after = crate::sink::retry_after_of(status, resp.headers());
                let text = resp.text().await.unwrap_or_default();
                let detail = format!(
                    "HTTP {}: {}",
                    status,
                    text.chars().take(200).collect::<String>()
                );
                // 5xx / 408 / 429 are worth retrying; other 4xx are
                // config/auth/payload errors that will fail identically.
                Err(crate::sink::http_status_error(status, retry_after, detail))
            }
            // Connect / DNS / timeout — transient by nature. reqwest's
            // Display hides the cause in `source()` — chain it.
            Err(e) => Err(SinkError::Transient(format!(
                "POST {}: {}",
                self.endpoint,
                crate::sink::error_chain(&e)
            ))),
        }
    }

    async fn healthcheck(&self) -> SinkHealth {
        // A real connectivity probe (and the control-plane "test connection"
        // affordance) lands with the health/metrics surface; until then a
        // sink reports healthy and its delivery errors surface via
        // `SinkStats::last_error`.
        SinkHealth::healthy()
    }

    fn error_redactor(&self) -> ErrorRedactor {
        self.headers
            .values()
            .fold(ErrorRedactor::default(), |r, value| {
                // A receiver may quote only the credential of `<scheme> <credential>`.
                let credential = value.trim().rsplit_once(' ').map_or("", |(_, c)| c).trim();
                // Only an auth-shaped value splits; `team a` must not make
                // every `a` in the error a secret.
                let credential = if credential.len() >= 8 {
                    credential
                } else {
                    ""
                };
                r.secret(value).secret(credential)
            })
    }
}

/// Encode a record without a trace snapshot as one CLIENT span: a caller that
/// predates the request trace bundle, or a usage row with no live request
/// behind it (the background batch-completion accounting).
///
/// Attribute names follow the OpenTelemetry GenAI semantic conventions
/// (semantic-conventions v1.41.1, `docs/gen-ai/gen-ai-spans.md`). Being the
/// only span, it also carries the inbound-request attributes a SERVER span
/// would otherwise hold.
fn build_otlp_span(record: &SinkRecord, exporter_name: &str) -> Value {
    let event = &record.usage;
    let trace_id = random_trace_id();
    let span_id = random_span_id();

    // The DP records `occurred_at` as RFC 3339; convert to nanos.
    // On parse failure (shouldn't happen in practice) fall back to
    // "now" so the span isn't silently dropped.
    let end_unix_nano =
        parse_rfc3339_to_unix_nano(&event.occurred_at).unwrap_or_else(now_unix_nano);
    // The span represents this ATTEMPT, so its duration is the
    // attempt-scoped upstream latency (the request-scoped
    // `downstream_latency_ms` rides along as an attribute instead).
    let latency_nanos = (event.upstream_latency_ms as u128).saturating_mul(1_000_000);
    let start_unix_nano = end_unix_nano.saturating_sub(latency_nanos);

    let mut attributes = event_attributes(record, exporter_name);
    merge_attributes(&mut attributes, inbound_request_attributes(record));
    let error = client_error_type(event);
    if let Some(error_type) = &error {
        attributes.push(attr_string("error.type", error_type));
    }
    let mut span = json!({
        "traceId": trace_id,
        "spanId":  span_id,
        "name":    operation_span_name(record),
        // SPAN_KIND_CLIENT: the gateway is the client of a remote model,
        // agent or MCP server.
        "kind":    3,
        "startTimeUnixNano": start_unix_nano.to_string(),
        "endTimeUnixNano":   end_unix_nano.to_string(),
        "attributes": attributes,
    });
    if error.is_some() {
        span["status"] = json!({ "code": STATUS_ERROR });
    }
    span
}

/// OTLP `Status.code` for a failed operation. A successful one leaves the
/// status unset, which the recording-errors convention requires.
const STATUS_ERROR: i64 = 2;

/// Encode one sink record into its OTLP span set.
///
/// A record carrying a [`crate::trace::TraceEmission`] (AISIX-Cloud#1279)
/// yields the request's span hierarchy — SERVER → logical CLIENT → attempt
/// CLIENT — with ids and nanosecond boundaries frozen at emission time, so
/// a delivery retry re-encodes byte-identical spans and every exporter
/// agrees on the trace. A record without one falls back to the single span
/// of [`build_otlp_span`].
///
/// Which span says what:
///
/// - The **carrier** (the attempt span when present, else the logical span,
///   else SERVER) is the GenAI operation span: it alone carries the
///   `gen_ai.*` attributes, the usage and the captured content, so a backend
///   that turns GenAI spans into model calls counts each attempt once.
/// - The **SERVER** span follows the HTTP server conventions: its name is
///   `{method} {route}`, it carries the inbound request's attributes, and a
///   4xx leaves its status unset.
/// - A **logical** span that is not the carrier groups the attempts and
///   carries only the joining `aisix.*` keys — no `gen_ai.*`, which would
///   make it read as a second model call.
fn build_otlp_spans(record: &SinkRecord, exporter_name: &str) -> Vec<Value> {
    use crate::trace::SpanRole;

    let Some(trace) = &record.trace else {
        return vec![build_otlp_span(record, exporter_name)];
    };
    let event = &record.usage;
    let trace_id = trace.trace_id.to_hex();
    let carrier = trace.carrier_role();
    let operation_name = operation_span_name(record);
    trace
        .spans
        .iter()
        .map(|span| {
            let is_carrier = Some(span.role) == carrier;
            let (name, kind, mut attributes, error) = match span.role {
                SpanRole::Server => {
                    let mut attributes = server_attributes(record, exporter_name);
                    if is_carrier {
                        // A cache hit, a refusal or a pre-dispatch error:
                        // the request's only span, so it carries the event.
                        let mut merged = event_attributes(record, exporter_name);
                        merge_attributes(&mut merged, attributes);
                        attributes = merged;
                    }
                    (
                        server_span_name(record),
                        2,
                        attributes,
                        server_error_type(event),
                    )
                }
                SpanRole::Logical | SpanRole::Attempt => {
                    let attributes = if is_carrier {
                        event_attributes(record, exporter_name)
                    } else {
                        joining_attributes(event, exporter_name)
                    };
                    (
                        operation_name.clone(),
                        3,
                        attributes,
                        client_error_type(event),
                    )
                }
            };
            if let Some(error_type) = &error {
                attributes.push(attr_string("error.type", error_type));
            }
            let mut json = json!({
                "traceId": trace_id,
                "spanId": span.span_id.to_hex(),
                "name": name,
                "kind": kind,
                "startTimeUnixNano": span.start_unix_nano.to_string(),
                "endTimeUnixNano": span.end_unix_nano.to_string(),
                "attributes": attributes,
                "flags": span_flags(span, trace),
            });
            if error.is_some() {
                json["status"] = json!({ "code": STATUS_ERROR });
            }
            if let Some(parent) = span.parent_span_id {
                json["parentSpanId"] = Value::String(parent.to_hex());
            }
            // The caller's tracestate belongs to the propagated context;
            // it rides the SERVER span, the one that continues it.
            if span.role == SpanRole::Server {
                if let Some(state) = &trace.tracestate {
                    json["traceState"] = Value::String(state.clone());
                }
            }
            json
        })
        .collect()
}

/// OTLP span `flags` (uint32): the low byte is the span's W3C trace-flags
/// — always `sampled`, since encoding IS the export decision, plus the
/// inbound octet's `random` bit (0x02) on a SERVER span continuing a
/// remote trace — and `HAS_IS_REMOTE` (bit 8) on every span, with
/// `IS_REMOTE` (bit 9) on a SERVER span whose parent arrived over the
/// wire in `traceparent`.
fn span_flags(span: &crate::trace::SpanEmit, trace: &crate::trace::TraceEmission) -> u32 {
    const SAMPLED: u32 = 0x01;
    const W3C_RANDOM: u32 = 0x02;
    const HAS_IS_REMOTE: u32 = 0x100;
    const IS_REMOTE: u32 = 0x200;
    let remote = span.role == crate::trace::SpanRole::Server
        && span.parent_span_id.is_some()
        && trace.remote_flags.is_some();
    let mut flags = SAMPLED | HAS_IS_REMOTE;
    if remote {
        flags |= IS_REMOTE;
        // The random bit describes the trace id, which the remote parent
        // minted — preserve its claim; never invent it for local roots.
        flags |= u32::from(trace.remote_flags.unwrap_or(0)) & W3C_RANDOM;
    }
    flags
}

/// `error.type` of a CLIENT span, or `None` when the operation did not
/// fail. Any status outside 1xx–3xx is a failure, except 499: a caller
/// that hung up cancelled the call on purpose, which the conventions say
/// not to record as an error.
fn client_error_type(event: &UsageEvent) -> Option<String> {
    let status = event.status_code;
    if status < 400 || status == CLIENT_CLOSED_REQUEST {
        return None;
    }
    Some(error_type_value(event))
}

/// `error.type` of the SERVER span: only a 5xx is the server's own failure;
/// a 4xx leaves a SERVER span's status unset (HTTP span status rules).
fn server_error_type(event: &UsageEvent) -> Option<String> {
    (event.status_code >= 500).then(|| error_type_value(event))
}

/// The gateway's own bounded error class when it named one, else the
/// status code as a string — both values the conventions allow.
fn error_type_value(event: &UsageEvent) -> String {
    if event.error_class.is_empty() {
        event.status_code.to_string()
    } else {
        event.error_class.clone()
    }
}

/// `gen_ai.operation.name` for this event: the conventions' value where one
/// applies, else the gateway's own operation, verbatim. `None` only for an
/// event that carries no operation at all.
fn genai_operation_name(event: &UsageEvent) -> Option<&str> {
    if is_a2a(event) {
        return Some("invoke_agent");
    }
    if is_mcp(event) {
        return Some("execute_tool");
    }
    match event.operation.as_str() {
        "chat" | "messages" | "responses" => Some("chat"),
        "completions" => Some("text_completion"),
        "embeddings" => Some("embeddings"),
        "" => None,
        other => Some(other),
    }
}

fn is_a2a(event: &UsageEvent) -> bool {
    event.operation == "a2a" || event.inbound_protocol == "a2a"
}

/// The JSON-RPC method of every MCP call the gateway records: usage is
/// emitted for tool calls only.
const MCP_TOOL_CALL_METHOD: &str = "tools/call";

/// Longest target this will append to a span name.
///
/// An A2A agent name is a registered resource, but an MCP tool name is the
/// caller's own `tools/call` `params.name` and is recorded before the tool is
/// known to exist — including on the quota-rejection path, which emits without
/// ever contacting an upstream. A trace backend indexes span names and most
/// derive RED metrics from them, so the one field a caller picks freely is
/// bounded before it gets there.
const MAX_SPAN_NAME_TARGET: usize = 64;

/// The name of the GenAI operation span (and of a logical span grouping its
/// attempts): `{operation} {target}`, where the target is the model the
/// request was made to, the agent invoked, or the passthrough route, and
/// MCP's own `{mcp.method.name} {gen_ai.tool.name}`. A target that is absent
/// or outside [`MAX_SPAN_NAME_TARGET`] leaves the bare operation, which the
/// conventions name as the fallback.
fn operation_span_name(record: &SinkRecord) -> String {
    let event = &record.usage;
    let (operation, target) = if is_mcp(event) {
        (MCP_TOOL_CALL_METHOD, event.mcp_tool_name.as_str())
    } else if is_a2a(event) {
        ("invoke_agent", event.a2a_agent_name.as_str())
    } else {
        let operation = genai_operation_name(event).unwrap_or("unknown");
        let target = if event.inbound_protocol == "passthrough" {
            event.passthrough_route_name.as_str()
        } else {
            upstream(record)
                .filter(|u| !u.request_model_caller_named)
                .map_or("", |u| u.request_model.as_str())
        };
        (operation, target)
    };
    if target.is_empty() || target.len() > MAX_SPAN_NAME_TARGET {
        operation.to_string()
    } else {
        format!("{operation} {target}")
    }
}

fn server_span_name(record: &SinkRecord) -> String {
    match record.facts.as_ref().and_then(|f| f.http.as_ref()) {
        Some(http) => http.span_name(),
        None => "HTTP".to_string(),
    }
}

fn upstream(record: &SinkRecord) -> Option<&crate::genai::UpstreamFacts> {
    record.facts.as_ref().and_then(|f| f.upstream.as_ref())
}

/// Append `extra`, skipping any key `attributes` already carries.
fn merge_attributes(attributes: &mut Vec<Value>, extra: Vec<Value>) {
    for attr in extra {
        let key = attr["key"].as_str().unwrap_or_default();
        if !attributes.iter().any(|a| a["key"] == key) {
            attributes.push(attr);
        }
    }
}

/// The keys every span of a request shares, so a backend can group them.
fn joining_attributes(event: &UsageEvent, exporter_name: &str) -> Vec<Value> {
    let mut attributes = vec![
        attr_string("aisix.exporter_name", exporter_name),
        attr_string("aisix.request_id", &event.request_id),
    ];
    if !event.operation.is_empty() {
        attributes.push(attr_string("aisix.operation", &event.operation));
    }
    attributes
}

/// What the caller's HTTP request looked like, under the HTTP and client
/// conventions' names: the resolved caller address, its `User-Agent`, and
/// the operator-selected request headers.
fn inbound_request_attributes(record: &SinkRecord) -> Vec<Value> {
    let event = &record.usage;
    let mut attributes = Vec::new();
    if !event.client_source_ip.is_empty() {
        attributes.push(attr_string("client.address", &event.client_source_ip));
    }
    if !event.client_user_agent.is_empty() {
        attributes.push(attr_string("user_agent.original", &event.client_user_agent));
    }
    // `http.request.header.<lowercase name>` is a string array; one element
    // here, because repeated occurrences were already joined into one value.
    for (name, value) in &record.request_headers {
        attributes.push(attr_string_array(
            &format!("http.request.header.{name}"),
            std::slice::from_ref(value),
        ));
    }
    attributes
}

/// The SERVER span's attributes: the HTTP server convention's request
/// shape, the response status, the inbound-request attributes, and the
/// joining keys.
fn server_attributes(record: &SinkRecord, exporter_name: &str) -> Vec<Value> {
    let event = &record.usage;
    let mut attributes = Vec::new();
    if let Some(http) = record.facts.as_ref().and_then(|f| f.http.as_ref()) {
        attributes.push(attr_string("http.request.method", http.method));
        if let Some(original) = &http.method_original {
            attributes.push(attr_string("http.request.method_original", original));
        }
        attributes.push(attr_string("url.path", &http.path));
        if let Some(scheme) = http.scheme {
            attributes.push(attr_string("url.scheme", scheme));
        }
        if let Some(route) = http.route {
            attributes.push(attr_string("http.route", route));
        }
        if let Some(query) = &http.query {
            attributes.push(attr_string("url.query", query));
        }
    }
    attributes.push(attr_int(
        "http.response.status_code",
        event.status_code as i64,
    ));
    attributes.extend(inbound_request_attributes(record));
    attributes.extend(joining_attributes(event, exporter_name));
    attributes
}

/// The GenAI operation span's attributes: the event's full attribute set,
/// plus captured content when the record carries it.
fn event_attributes(record: &SinkRecord, exporter_name: &str) -> Vec<Value> {
    let event = &record.usage;
    let facts = record.facts.as_deref();
    let upstream = facts.and_then(|f| f.upstream.as_ref());
    let mcp = is_mcp(event);
    let a2a = is_a2a(event);
    let operation = genai_operation_name(event);

    let mut attributes = Vec::new();
    if let Some(operation) = operation {
        attributes.push(attr_string("gen_ai.operation.name", operation));
    }
    // The gateway's own, finer reading of the same thing. The convention's
    // value collapses chat, messages and responses onto `chat`, which
    // cannot separate them (AISIX-Cloud#1461).
    if !event.operation.is_empty() {
        attributes.push(attr_string("aisix.operation", &event.operation));
    }
    if mcp {
        attributes.push(attr_string("mcp.method.name", MCP_TOOL_CALL_METHOD));
    }
    if let Some(upstream) = upstream {
        if !upstream.provider_name.is_empty() {
            attributes.push(attr_string("gen_ai.provider.name", &upstream.provider_name));
        }
        // The model name sent to the upstream — not the alias the caller
        // addressed, which is `aisix.requested_model`.
        if !upstream.request_model.is_empty() {
            attributes.push(attr_string("gen_ai.request.model", &upstream.request_model));
        }
        if !upstream.server_address.is_empty() {
            attributes.push(attr_string("server.address", &upstream.server_address));
            if let Some(port) = upstream.server_port {
                attributes.push(attr_int("server.port", i64::from(port)));
            }
        }
    }
    // The model alias the client sent (`model` field) — a Model-Group name
    // for routed requests (AISIX-Cloud#790).
    if !event.requested_model.is_empty() {
        attributes.push(attr_string("aisix.requested_model", &event.requested_model));
    }
    if let Some(params) = facts.and_then(|f| f.request.as_deref()) {
        push_request_params(&mut attributes, params, upstream);
    }
    if !event.provider_model_version.is_empty() {
        attributes.push(attr_string(
            "gen_ai.response.model",
            &event.provider_model_version,
        ));
    }
    if !event.provider_request_id.is_empty() {
        attributes.push(attr_string(
            "gen_ai.response.id",
            &event.provider_request_id,
        ));
    }
    if !event.finish_reason.is_empty() {
        attributes.push(attr_string_array(
            "gen_ai.response.finish_reasons",
            std::slice::from_ref(&event.finish_reason),
        ));
    }
    // Measured from this attempt's dispatch to the upstream's first
    // streamed frame — the convention's definition, in seconds.
    if event.upstream_ttft_ms > 0 {
        attributes.push(attr_double(
            "gen_ai.response.time_to_first_chunk",
            f64::from(event.upstream_ttft_ms) / 1000.0,
        ));
    }
    if !mcp {
        push_usage(&mut attributes, event, operation);
    }
    attributes.push(attr_int(
        "http.response.status_code",
        event.status_code as i64,
    ));
    if !event.api_key_id.is_empty() {
        // Custom attribute (no semconv yet) so reviewers can join
        // spans back to the AISIX api_key dashboard.
        attributes.push(attr_string("aisix.api_key_id", &event.api_key_id));
    }
    if !event.api_key_name.is_empty() {
        attributes.push(attr_string("aisix.api_key_name", &event.api_key_name));
    }
    // The org member behind the credential (AISIX-Cloud#1389). Exported
    // beside the key rather than derived from it downstream: a member can
    // hold several keys, and a key can be rebound to someone else.
    if !event.user_id.is_empty() {
        attributes.push(attr_string("aisix.user_id", &event.user_id));
    }
    if !event.team_id.is_empty() {
        attributes.push(attr_string("aisix.team_id", &event.team_id));
    }
    // The request's effective cost-attribution tags, one attribute per tag
    // key — the same `aisix.tags.<key>` path the Datadog log's nested
    // `aisix.tags` object facets on.
    for (key, value) in &event.tags {
        attributes.push(attr_string(&format!("aisix.tags.{key}"), value));
    }
    if !event.model_id.is_empty() {
        attributes.push(attr_string("aisix.model_id", &event.model_id));
    }
    if !event.requested_model_id.is_empty() {
        attributes.push(attr_string(
            "aisix.requested_model_id",
            &event.requested_model_id,
        ));
    }
    // The tier the upstream served on. `openai.request.service_tier`
    // beside it is what the request asked for, which can differ.
    if let Some(tier) = &event.service_tier {
        attributes.push(attr_string("aisix.service_tier", tier));
    }
    // Duration cost basis (#457): only the duration-billed audio models
    // carry it, so it stays off every other span.
    if event.audio_duration_seconds > 0.0 {
        attributes.push(attr_double(
            "aisix.audio.duration_seconds",
            event.audio_duration_seconds,
        ));
    }
    attributes.push(attr_string("aisix.exporter_name", exporter_name));
    attributes.push(attr_string("aisix.request_id", &event.request_id));
    if event.upstream_ttft_ms > 0 {
        attributes.push(attr_int(
            "aisix.upstream_ttft_ms",
            event.upstream_ttft_ms as i64,
        ));
    }
    // Request-scoped: present only on the attempt that delivered the
    // terminal response, so consumers read a request's caller-facing
    // latency off that one span rather than summing the group.
    if event.downstream_latency_ms > 0 {
        attributes.push(attr_int(
            "aisix.downstream_latency_ms",
            event.downstream_latency_ms as i64,
        ));
    }
    if let Some(overhead) = event.gateway_overhead_ms {
        attributes.push(attr_int("aisix.gateway_overhead_ms", i64::from(overhead)));
    }
    if let Some(tpot) = event.upstream_tpot_ms {
        attributes.push(attr_double("aisix.upstream_tpot_ms", tpot));
    }
    // `http.response.header.<lowercase name>`, a string array like its
    // request twin: this attempt's upstream response, as the exporter
    // selected it.
    for (name, value) in &record.upstream_response_headers {
        attributes.push(attr_string_array(
            &format!("http.response.header.{name}"),
            std::slice::from_ref(value),
        ));
    }
    // Per-attempt telemetry (#655). `request_id` is the trace/group key; a
    // failover request emits one span per attempt sharing it, ordered by
    // `aisix.attempt_index`. The OTLP encoder is an explicit allowlist, so
    // these are added here alongside the wire fields.
    attributes.push(attr_int("aisix.attempt_index", event.attempt_index as i64));
    if !event.attempt_kind.is_empty() {
        attributes.push(attr_string("aisix.attempt_kind", &event.attempt_kind));
    }
    if !event.attempt_model.is_empty() {
        attributes.push(attr_string("aisix.attempt_model", &event.attempt_model));
    }
    if !event.error_class.is_empty() {
        attributes.push(attr_string("aisix.error_class", &event.error_class));
    }
    if !event.error_message.is_empty() {
        attributes.push(attr_string("aisix.error_message", &event.error_message));
    }
    // Gateway-owned semantic embeddings are child work of this request, not
    // attempts for its client-selected model. OTLP attributes cannot carry a
    // nested object, so retain the bounded audit trail as a JSON string and
    // expose the number of calls omitted from that string separately. The
    // encoder is an allowlist: without this explicit projection OTLP loses
    // the audit fields that SLS and other sink records retain.
    let (gateway_calls, export_dropped) =
        gateway_embedding_calls_attribute(&event.gateway_embedding_calls);
    if let Some(calls) = gateway_calls {
        attributes.push(calls);
    }
    let gateway_calls_dropped = event
        .gateway_embedding_calls_dropped
        .saturating_add(export_dropped);
    if gateway_calls_dropped > 0 {
        attributes.push(attr_int(
            "aisix.gateway_embedding_calls_dropped",
            i64::from(gateway_calls_dropped),
        ));
    }
    // End-user identity a passthrough route's `identity_header` extracted —
    // the per-employee attribution of the forward-proxy scenario. Not gated
    // on the protocol so a future handler that learns to populate it exports
    // it without touching this sink.
    if !event.client_identity.is_empty() {
        attributes.push(attr_string_capped(
            "aisix.client_identity",
            &event.client_identity,
        ));
    }
    // JWT identity attribution (AISIX-Cloud#564): who the request ran as
    // when it authenticated with a JWT — the identity behind the (possibly
    // shared) api_key_id.
    if !event.jwt_subject.is_empty() {
        attributes.push(attr_string("aisix.jwt_subject", &event.jwt_subject));
    }
    if !event.jwt_provider.is_empty() {
        attributes.push(attr_string("aisix.jwt_provider", &event.jwt_provider));
    }
    if !event.jwt_claim_mapping.is_empty() {
        attributes.push(attr_string(
            "aisix.jwt_claim_mapping",
            &event.jwt_claim_mapping,
        ));
    }
    // Gateway-protocol attribution. An A2A or MCP call is not a model
    // inference, and encoding one as a bare `chat` span left every agent and
    // tool call in a trace backend indistinguishable from an LLM request —
    // with the agent, the method and the task it touched recorded nowhere at
    // all (AISIX-Cloud#1215).
    if a2a {
        if !event.a2a_agent_name.is_empty() {
            attributes.push(attr_string("gen_ai.agent.name", &event.a2a_agent_name));
        }
        // An A2A context ties a multi-turn exchange's tasks together —
        // exactly what semconv means by a conversation id.
        if !event.a2a_context_id.is_empty() {
            attributes.push(attr_string_capped(
                "gen_ai.conversation.id",
                &event.a2a_context_id,
            ));
        }
        for (key, value) in [
            ("aisix.a2a.method", &event.a2a_method),
            ("aisix.a2a.operation", &event.a2a_operation),
            ("aisix.a2a.protocol_version", &event.a2a_protocol_version),
            ("aisix.a2a.task_id", &event.a2a_task_id),
            ("aisix.a2a.task_state", &event.a2a_task_state),
        ] {
            if !value.is_empty() {
                attributes.push(attr_string_capped(key, value));
            }
        }
        if event.a2a_stream_event_count > 0 {
            attributes.push(attr_int(
                "aisix.a2a.stream_event_count",
                i64::from(event.a2a_stream_event_count),
            ));
        }
    } else if mcp {
        if !event.mcp_tool_name.is_empty() {
            attributes.push(attr_string_capped("gen_ai.tool.name", &event.mcp_tool_name));
        }
        if !event.mcp_server_name.is_empty() {
            attributes.push(attr_string("aisix.mcp.server_name", &event.mcp_server_name));
        }
    } else if event.inbound_protocol == "passthrough" && !event.passthrough_route_name.is_empty() {
        attributes.push(attr_string(
            "aisix.passthrough.route_name",
            &event.passthrough_route_name,
        ));
    }
    if let Some(content) = &record.content {
        push_content(&mut attributes, event, content);
    }
    attributes
}

/// The request parameters the inference conventions list, as the caller
/// sent them.
fn push_request_params(
    attributes: &mut Vec<Value>,
    params: &crate::genai::GenAiRequestParams,
    upstream: Option<&crate::genai::UpstreamFacts>,
) {
    if let Some(v) = params.max_tokens {
        attributes.push(attr_int("gen_ai.request.max_tokens", v));
    }
    for (key, value) in [
        ("gen_ai.request.temperature", params.temperature),
        ("gen_ai.request.top_p", params.top_p),
        ("gen_ai.request.top_k", params.top_k),
        ("gen_ai.request.frequency_penalty", params.frequency_penalty),
        ("gen_ai.request.presence_penalty", params.presence_penalty),
    ] {
        if let Some(v) = value {
            attributes.push(attr_double(key, v));
        }
    }
    if !params.stop_sequences.is_empty() {
        attributes.push(attr_string_array(
            "gen_ai.request.stop_sequences",
            &params.stop_sequences,
        ));
    }
    if let Some(v) = params.seed {
        attributes.push(attr_int("gen_ai.request.seed", v));
    }
    if let Some(v) = params.choice_count {
        attributes.push(attr_int("gen_ai.request.choice.count", v));
    }
    // Set if and only if the request streamed.
    if params.stream {
        attributes.push(attr_bool("gen_ai.request.stream", true));
    }
    if let Some(v) = params.output_type {
        attributes.push(attr_string("gen_ai.output.type", v));
    }
    if !params.encoding_formats.is_empty() {
        attributes.push(attr_string_array(
            "gen_ai.request.encoding_formats",
            &params.encoding_formats,
        ));
    }
    if let Some(v) = params.embedding_dimensions {
        attributes.push(attr_int("gen_ai.embeddings.dimension.count", v));
    }
    // Provider-specific attributes belong only on that provider's spans.
    if let Some(tier) = &params.openai_service_tier {
        if upstream.is_some_and(|u| u.provider_name == "openai") {
            attributes.push(attr_string("openai.request.service_tier", tier));
        }
    }
}

/// The usage counters, under the conventions' accounting: input includes
/// every cached token and output includes reasoning, whatever shape the
/// upstream reported them in. The cache and reasoning breakdowns appear only
/// when the upstream reported them. An embeddings span has no output.
fn push_usage(attributes: &mut Vec<Value>, event: &UsageEvent, operation: Option<&str>) {
    attributes.push(attr_int(
        "gen_ai.usage.input_tokens",
        i64::from(event.genai_input_tokens()),
    ));
    let cache_read = event.genai_cache_read_input_tokens();
    if cache_read > 0 {
        attributes.push(attr_int(
            "gen_ai.usage.cache_read.input_tokens",
            i64::from(cache_read),
        ));
    }
    if let Some(creation) = event.genai_cache_creation_input_tokens() {
        attributes.push(attr_int(
            "gen_ai.usage.cache_creation.input_tokens",
            i64::from(creation),
        ));
    }
    // Modality breakdowns, as the upstream reported them (each a subset of
    // its side's total). Named after the GenAI conventions' development
    // registry, which defines them beyond the v1.41 release.
    for (key, value) in [
        (
            "gen_ai.usage.audio.input_tokens",
            event.modality_tokens.prompt_audio_tokens,
        ),
        (
            "gen_ai.usage.image.input_tokens",
            event.modality_tokens.prompt_image_tokens,
        ),
    ] {
        if let Some(v) = value {
            attributes.push(attr_int(key, i64::from(v)));
        }
    }
    if operation == Some("embeddings") {
        return;
    }
    attributes.push(attr_int(
        "gen_ai.usage.output_tokens",
        i64::from(event.genai_output_tokens()),
    ));
    if event.reasoning_tokens > 0 {
        attributes.push(attr_int(
            "gen_ai.usage.reasoning.output_tokens",
            i64::from(event.reasoning_tokens),
        ));
    }
    for (key, value) in [
        (
            "gen_ai.usage.audio.output_tokens",
            event.modality_tokens.completion_audio_tokens,
        ),
        (
            "gen_ai.usage.image.output_tokens",
            event.modality_tokens.completion_image_tokens,
        ),
    ] {
        if let Some(v) = value {
            attributes.push(attr_int(key, i64::from(v)));
        }
    }
}

/// Captured content (`content_mode = full`).
///
/// A message-shaped payload whose capture is complete becomes the
/// conventions' structured attributes, each held to the exporter's
/// per-field cap. Everything else — a non-message payload (embedding input,
/// images, audio, jobs, agent text), content the capture cap cut, or content
/// that does not convert — stays under `aisix.content.prompt` /
/// `aisix.content.response`, with `aisix.content_truncated` when cut, so a
/// `gen_ai.*` content attribute is never malformed or partial.
fn push_content(attributes: &mut Vec<Value>, event: &UsageEvent, content: &SinkContent) {
    let enc = content.encode(event);
    for (key, value) in [
        ("gen_ai.tool.call.arguments", enc.tool_call_arguments),
        ("gen_ai.tool.call.result", enc.tool_call_result),
        (
            "gen_ai.input.messages",
            enc.structured.input_messages.as_deref(),
        ),
        (
            "gen_ai.system_instructions",
            enc.structured.system_instructions.as_deref(),
        ),
        (
            "gen_ai.tool.definitions",
            enc.structured.tool_definitions.as_deref(),
        ),
        (
            "gen_ai.output.messages",
            enc.structured.output_messages.as_deref(),
        ),
        ("aisix.content.prompt", enc.raw_prompt),
        ("aisix.content.response", enc.raw_response),
    ] {
        if let Some(value) = value {
            attributes.push(attr_string(key, value));
        }
    }
    if content.truncated {
        attributes.push(attr_bool("aisix.content_truncated", true));
    }
}

/// Wrap one or more spans into an OTLP/HTTP-JSON `ExportTraceServiceRequest`.
/// `service.name` is the gateway's configured `observability.service_name`;
/// the service and instrumentation-scope versions are the build's.
fn otlp_export_request(spans: Vec<Value>, service_name: &str) -> Value {
    let version = &*aisix_core::BUILD_VERSION;
    json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    attr_string("service.name", service_name),
                    attr_string("service.version", version),
                ],
            },
            "scopeSpans": [{
                "scope": { "name": "aisix-obs.otlp_http_sink", "version": version },
                "spans": spans,
            }],
        }],
    })
}

/// One event -> one-span export request over a metadata-only record (the
/// default shape). Test-only helper for the payload assertions; production
/// paths build spans via [`build_otlp_span`] and batch them through
/// [`otlp_export_request`] inside [`OtlpSink`].
#[cfg(test)]
fn build_otlp_traces_payload(event: &UsageEvent, exporter_name: &str) -> Value {
    let record = SinkRecord::metadata_only(event.clone());
    otlp_export_request(
        vec![build_otlp_span(&record, exporter_name)],
        DEFAULT_SERVICE_NAME,
    )
}

fn attr_string(key: &str, value: &str) -> Value {
    json!({
        "key": key,
        "value": { "stringValue": value },
    })
}

/// Longest caller-supplied attribute value carried on a span.
///
/// The gateway-protocol attributes below are copied from the caller's own
/// JSON-RPC envelope (the method it invoked, the ids it named), and the
/// request body limit defaults to unlimited, so their length is the caller's
/// choice. Every span rides in a batched export, so an unbounded value is a
/// delivery problem as much as a storage one.
const MAX_PROTOCOL_ATTR_BYTES: usize = 256;

/// Bound the serialized child-work audit on an OTLP span. The source ledger
/// retains at most 64 calls, but exporter input is public enough that the
/// encoder also protects itself from a synthetic or malformed `UsageEvent`.
///
/// The value is JSON rather than a semconv array because these child calls
/// have several scalar fields and OTLP attributes have one homogeneous value.
const MAX_GATEWAY_EMBEDDING_CALLS_ATTR_BYTES: usize = 32 * 1024;
const MAX_GATEWAY_EMBEDDING_CALLS_ATTR_ITEMS: usize = 64;
const MAX_GATEWAY_EMBEDDING_MODEL_ID_ATTR_BYTES: usize = 128;

#[derive(serde::Serialize)]
struct OtlpGatewayEmbeddingCall<'a> {
    count: u32,
    purpose: crate::usage::GatewayEmbeddingPurpose,
    embedding_model_id: &'a str,
    prompt_tokens: u32,
    total_tokens: u32,
    usage_source: crate::usage::GatewayEmbeddingUsageSource,
    latency_ms: u32,
    outcome: crate::usage::GatewayEmbeddingOutcome,
}

/// Encode the retained ordered prefix as valid JSON inside one bounded OTLP
/// string attribute. `dropped` includes both calls the request ledger omitted
/// and calls this exporter cannot fit, so a trace never silently looks
/// complete after either cap applies.
fn gateway_embedding_calls_attribute(calls: &[GatewayEmbeddingCall]) -> (Option<Value>, u32) {
    let mut encoded = String::from("[");
    let mut retained = 0usize;
    let mut dropped = 0u32;

    for (index, call) in calls.iter().enumerate() {
        if index == MAX_GATEWAY_EMBEDDING_CALLS_ATTR_ITEMS {
            dropped = dropped.saturating_add(usize_to_u32(calls.len().saturating_sub(index)));
            break;
        }
        let item = serde_json::to_string(&OtlpGatewayEmbeddingCall {
            count: call.count,
            purpose: call.purpose,
            embedding_model_id: gateway_embedding_model_id(&call.embedding_model_id),
            prompt_tokens: call.prompt_tokens,
            total_tokens: call.total_tokens,
            usage_source: call.usage_source,
            latency_ms: call.latency_ms,
            outcome: call.outcome,
        })
        // Serializing these scalar fields into an owned String cannot fail.
        .expect("gateway embedding audit is JSON-serializable");
        let separator = usize::from(retained > 0);
        if encoded
            .len()
            .saturating_add(separator)
            .saturating_add(item.len())
            .saturating_add(1)
            > MAX_GATEWAY_EMBEDDING_CALLS_ATTR_BYTES
        {
            // Preserve an ordered prefix, like the request ledger does, and
            // make every omitted suffix visible through the count attribute.
            dropped = dropped.saturating_add(usize_to_u32(calls.len().saturating_sub(index)));
            break;
        }
        if retained > 0 {
            encoded.push(',');
        }
        encoded.push_str(&item);
        retained += 1;
    }
    if retained == 0 {
        return (None, dropped);
    }
    encoded.push(']');
    debug_assert!(encoded.len() <= MAX_GATEWAY_EMBEDDING_CALLS_ATTR_BYTES);
    (
        Some(attr_string("aisix.gateway_embedding_calls", &encoded)),
        dropped,
    )
}

fn usize_to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Return an OTLP-safe embedding resource ID, or `unknown`.
///
/// Gateway embedding audit records can originate outside the proxy, so the
/// exporter treats this as its own final boundary. Reject rather than truncate
/// an unsafe value: keeping a prefix could still export its contents.
fn gateway_embedding_model_id(value: &str) -> &str {
    if !value.is_empty()
        && value.len() <= MAX_GATEWAY_EMBEDDING_MODEL_ID_ATTR_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        value
    } else {
        "unknown"
    }
}

fn capped_utf8(value: &str, max_bytes: usize) -> &str {
    let mut end = max_bytes.min(value.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

/// [`attr_string`] for a value a caller controls, cut to
/// [`MAX_PROTOCOL_ATTR_BYTES`] on a UTF-8 boundary.
fn attr_string_capped(key: &str, value: &str) -> Value {
    attr_string(key, capped_utf8(value, MAX_PROTOCOL_ATTR_BYTES))
}

fn attr_int(key: &str, value: i64) -> Value {
    json!({
        "key": key,
        // OTLP/JSON encodes int as a string to avoid JS Number precision loss.
        "value": { "intValue": value.to_string() },
    })
}

fn attr_double(key: &str, value: f64) -> Value {
    json!({
        "key": key,
        "value": { "doubleValue": value },
    })
}

fn attr_bool(key: &str, value: bool) -> Value {
    json!({
        "key": key,
        "value": { "boolValue": value },
    })
}

fn attr_string_array(key: &str, values: &[String]) -> Value {
    let arr: Vec<Value> = values.iter().map(|v| json!({"stringValue": v})).collect();
    json!({
        "key": key,
        "value": { "arrayValue": { "values": arr } },
    })
}

/// 16 random bytes as 32 lowercase-hex chars per OTLP/JSON spec.
fn random_trace_id() -> String {
    let bytes: [u8; 16] = rand_16();
    hex32(&bytes)
}

/// 8 random bytes as 16 lowercase-hex chars per OTLP/JSON spec.
fn random_span_id() -> String {
    let bytes: [u8; 8] = rand_8();
    hex16(&bytes)
}

fn rand_16() -> [u8; 16] {
    let u = uuid::Uuid::new_v4();
    *u.as_bytes()
}

fn rand_8() -> [u8; 8] {
    let u = uuid::Uuid::new_v4();
    let b = u.as_bytes();
    [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]
}

fn hex32(bytes: &[u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex16(bytes: &[u8; 8]) -> String {
    let mut s = String::with_capacity(16);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn parse_rfc3339_to_unix_nano(s: &str) -> Option<u128> {
    // Use chrono if available, fall back to naive epoch parsing.
    // We avoid pulling chrono into this crate by hand-parsing the
    // common DP-emitted RFC3339 form: `2006-01-02T15:04:05Z` or with
    // fractional seconds `.<digits>`.
    let dt = chrono_like_parse(s)?;
    let secs = dt.0 as u128;
    let nanos = dt.1 as u128;
    secs.checked_mul(1_000_000_000)
        .and_then(|n| n.checked_add(nanos))
}

/// Returns (unix_secs, sub_seconds_in_nanos) on success.
fn chrono_like_parse(s: &str) -> Option<(i64, u32)> {
    // Cheap-and-cheerful: split on the 'T', the seconds field, and 'Z'.
    // Wrong handling of timezone offsets — but the DP serialises UTC
    // with a 'Z' suffix everywhere, so this is sufficient for our
    // own emit shape.
    let s = s.strip_suffix('Z')?;
    let (date, time) = s.split_once('T')?;
    let mut date_parts = date.split('-');
    let y: i32 = date_parts.next()?.parse().ok()?;
    let mo: u32 = date_parts.next()?.parse().ok()?;
    let d: u32 = date_parts.next()?.parse().ok()?;

    let (h_m_s, frac_str) = match time.split_once('.') {
        Some((a, b)) => (a, b),
        None => (time, "0"),
    };
    let mut t_parts = h_m_s.split(':');
    let h: u32 = t_parts.next()?.parse().ok()?;
    let mi: u32 = t_parts.next()?.parse().ok()?;
    let se: u32 = t_parts.next()?.parse().ok()?;

    let secs = days_from_civil(y, mo, d).checked_mul(86_400)?
        + (h as i64) * 3600
        + (mi as i64) * 60
        + se as i64;

    // Truncate to 9 fractional digits.
    let frac_padded: String = frac_str
        .chars()
        .chain(std::iter::repeat('0'))
        .take(9)
        .collect();
    let nanos: u32 = frac_padded.parse().ok()?;

    Some((secs, nanos))
}

/// Howard Hinnant's `days_from_civil` (https://howardhinnant.github.io/date_algorithms.html).
/// Avoids depending on chrono just for the e2e build.
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let doy = (153 * if m > 2 { m - 3 } else { m + 9 } + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era as i64) * 146_097 + doe as i64 - 719_468
}

fn now_unix_nano() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

#[allow(dead_code)]
fn _ensure_arc_clone(_: Arc<()>) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event() -> UsageEvent {
        UsageEvent {
            request_id: "req-test-123".into(),
            occurred_at: "2026-05-01T12:00:00Z".into(),
            model_id: "mod-uuid".into(),
            api_key_id: "ak-uuid".into(),
            prompt_tokens: 10,
            completion_tokens: 5,
            upstream_latency_ms: 250,
            status_code: 200,
            provider_request_id: "chatcmpl-abc".into(),
            provider_model_version: "gpt-4o-2024-08-06".into(),
            finish_reason: "stop".into(),
            cost_usd: 0.001,
            operation: "image_generation".into(),
            ..Default::default()
        }
    }

    fn sample_exporter() -> ObservabilityExporter {
        // Round-trip through serde so the runtime_id (private) gets
        // populated by the loader path, just like in production. Kept
        // off the public API on purpose — callers must go through
        // the loader, not poke the field directly.
        serde_json::from_value(serde_json::json!({
            "name": "test-exp",
            "enabled": true,
            "kind": "otlp_http",
            "endpoint": "http://mock-otlp:4318/v1/traces",
            "headers": {"authorization": "Bearer xyz"}
        }))
        .unwrap()
    }

    fn sls_kind(content_mode: SlsContentMode, max_bytes: u32) -> ExporterKind {
        ExporterKind::AliyunSls(AliyunSlsConfig {
            endpoint: "ap-southeast-3.log.aliyuncs.com".into(),
            project: "p".into(),
            logstore: "l".into(),
            credential_ref: "r".into(),
            content_mode,
            content_max_bytes: max_bytes,
        })
    }

    /// A three-attempt failover bundle with a remote parent, for the
    /// hierarchy-encoding assertions (AISIX-Cloud#1279).
    fn hierarchy_record() -> SinkRecord {
        let (trace_id, parent_span_id, flags) = crate::trace::parse_traceparent(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        )
        .unwrap();
        let bundle =
            crate::trace::RequestTraceBundle::new(Some(crate::trace::RemoteTraceContext {
                trace_id,
                parent_span_id,
                flags,
                tracestate: Some("vendor=x".into()),
            }));
        bundle.start_attempt(0);
        bundle.end_attempt(0);
        bundle.start_attempt(1);
        bundle.end_attempt(1);
        let mut event = sample_event();
        event.attempt_index = 1;
        SinkRecord::metadata_only(event).with_trace(Some(bundle.emission(true, 1, 250, true)))
    }

    /// The hierarchy contract (AISIX-Cloud#1279): a terminal record with a
    /// remote parent encodes SERVER (kind 2, parented to the inbound
    /// traceparent, carrying traceState) → logical CLIENT → attempt
    /// CLIENT, all under the caller's trace id, with the event's full
    /// attribute set on the attempt span only.
    #[test]
    fn hierarchy_record_encodes_server_logical_attempt_spans() {
        let record = hierarchy_record();
        let spans = build_otlp_spans(&record, "exp-1");
        assert_eq!(spans.len(), 3);

        for span in &spans {
            assert_eq!(span["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
        }
        let server = &spans[0];
        let logical = &spans[1];
        let attempt = &spans[2];
        assert_eq!(server["kind"], 2);
        assert_eq!(server["parentSpanId"], "00f067aa0ba902b7");
        assert_eq!(server["traceState"], "vendor=x");
        // SAMPLED | HAS_IS_REMOTE | IS_REMOTE.
        assert_eq!(server["flags"], 0x301);

        assert_eq!(logical["kind"], 3);
        assert_eq!(logical["parentSpanId"], server["spanId"]);
        assert_eq!(logical["flags"], 0x101);
        assert!(logical.get("traceState").is_none());

        assert_eq!(attempt["kind"], 3);
        assert_eq!(attempt["parentSpanId"], logical["spanId"]);

        // The event's full attribute set rides the attempt span; the
        // structural spans carry only the joining keys.
        let keys = |span: &Value| -> Vec<String> {
            span["attributes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a["key"].as_str().unwrap().to_string())
                .collect()
        };
        assert!(keys(attempt).contains(&"gen_ai.response.id".to_string()));
        assert!(!keys(server).contains(&"gen_ai.response.id".to_string()));
        assert!(keys(server).contains(&"aisix.request_id".to_string()));
        // The request's kind is one of the joining keys, on every level:
        // a trace backend filters a whole trace by it at the SERVER root,
        // which reading it off the attempt span alone would not allow
        // (AISIX-Cloud#1461).
        let operation_of = |span: &Value| -> String {
            span["attributes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["key"] == "aisix.operation")
                .expect("aisix.operation must be present")["value"]["stringValue"]
                .as_str()
                .unwrap()
                .to_string()
        };
        for span in [server, logical, attempt] {
            assert_eq!(operation_of(span), "image_generation");
        }

        // Timestamps bracket: server ⊇ logical ⊇ attempt.
        let nanos =
            |span: &Value, key: &str| -> u128 { span[key].as_str().unwrap().parse().unwrap() };
        assert!(nanos(server, "startTimeUnixNano") <= nanos(logical, "startTimeUnixNano"));
        assert!(nanos(logical, "endTimeUnixNano") <= nanos(server, "endTimeUnixNano"));
        assert!(nanos(logical, "startTimeUnixNano") <= nanos(attempt, "startTimeUnixNano"));
        assert!(nanos(attempt, "endTimeUnixNano") <= nanos(logical, "endTimeUnixNano"));
    }

    fn attr<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
        span["attributes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["key"] == key)
            .map(|a| &a["value"])
    }

    fn hierarchy_with(event: UsageEvent, facts: crate::genai::SpanFacts) -> Vec<Value> {
        let bundle = crate::trace::RequestTraceBundle::new(None);
        bundle.start_attempt(0);
        bundle.end_attempt(0);
        let record = SinkRecord::metadata_only(event)
            .with_trace(Some(bundle.emission(true, 0, 10, true)))
            .with_facts(Some(Arc::new(facts)));
        build_otlp_spans(&record, "exp-1")
    }

    fn chat_facts() -> crate::genai::SpanFacts {
        crate::genai::SpanFacts {
            http: Some(Arc::new(crate::genai::HttpServerFacts::new(
                "POST",
                "/v1/chat/completions",
                Some("/v1/chat/completions"),
                Some("https"),
            ))),
            request: None,
            upstream: Some(crate::genai::UpstreamFacts {
                provider_name: "anthropic".into(),
                request_model: "claude-sonnet-4-5".into(),
                ..Default::default()
            }),
        }
    }

    /// SERVER follows the HTTP server conventions, the GenAI attributes ride
    /// the attempt alone, and a logical span grouping attempts carries none
    /// of them — a backend that maps `gen_ai.operation.name` / a model
    /// attribute to a model call would otherwise count it twice.
    #[test]
    fn only_the_carrier_span_reads_as_a_genai_operation() {
        let mut event = sample_event();
        event.operation = "chat".into();
        event.client_source_ip = "203.0.113.7".into();
        let spans = hierarchy_with(event, chat_facts());
        let (server, logical, attempt) = (&spans[0], &spans[1], &spans[2]);

        assert_eq!(server["name"], "POST /v1/chat/completions");
        assert_eq!(
            attr(server, "http.request.method").unwrap()["stringValue"],
            "POST"
        );
        assert_eq!(
            attr(server, "url.path").unwrap()["stringValue"],
            "/v1/chat/completions"
        );
        assert_eq!(attr(server, "url.scheme").unwrap()["stringValue"], "https");
        assert_eq!(
            attr(server, "http.route").unwrap()["stringValue"],
            "/v1/chat/completions"
        );
        assert_eq!(
            attr(server, "client.address").unwrap()["stringValue"],
            "203.0.113.7"
        );
        assert!(attr(attempt, "client.address").is_none());

        assert_eq!(attempt["name"], "chat claude-sonnet-4-5");
        assert_eq!(logical["name"], "chat claude-sonnet-4-5");
        for key in [
            "gen_ai.operation.name",
            "gen_ai.request.model",
            "gen_ai.provider.name",
        ] {
            assert!(attr(attempt, key).is_some(), "{key} on the attempt");
            assert!(attr(logical, key).is_none(), "{key} on the logical span");
            assert!(attr(server, key).is_none(), "{key} on the SERVER span");
        }
        for span in &spans {
            assert!(span.get("status").is_none(), "success leaves status unset");
        }
    }

    /// A 4xx is the CLIENT spans' error but not the SERVER span's; a 5xx is
    /// both; a caller's own hang-up (499) is neither.
    #[test]
    fn error_status_follows_the_span_kind() {
        let status_of = |status: u16, class: &str| {
            let mut event = sample_event();
            event.operation = "chat".into();
            event.status_code = status;
            event.error_class = class.into();
            hierarchy_with(event, chat_facts())
        };
        let spans = status_of(429, "rate_limit_exceeded");
        assert!(spans[0].get("status").is_none());
        assert!(attr(&spans[0], "error.type").is_none());
        for client in &spans[1..] {
            assert_eq!(client["status"]["code"], STATUS_ERROR);
            assert_eq!(
                attr(client, "error.type").unwrap()["stringValue"],
                "rate_limit_exceeded"
            );
        }
        let spans = status_of(502, "");
        for span in &spans {
            assert_eq!(span["status"]["code"], STATUS_ERROR);
            assert_eq!(attr(span, "error.type").unwrap()["stringValue"], "502");
        }
        let spans = status_of(499, "client_disconnected");
        for span in &spans {
            assert!(span.get("status").is_none());
            assert!(attr(span, "error.type").is_none());
        }
    }

    /// `gen_ai.usage.input_tokens` includes every cached token whichever
    /// shape the upstream reported it in, with the cache breakdown beside it.
    #[test]
    fn usage_follows_the_conventions_accounting() {
        let usage = |event: UsageEvent| build_otlp_span(&SinkRecord::metadata_only(event), "x");
        let anthropic = usage(UsageEvent {
            prompt_tokens: 10,
            cache_read_tokens: 300,
            cache_creation_tokens: 40,
            completion_tokens: 5,
            ..sample_event()
        });
        assert_eq!(
            attr(&anthropic, "gen_ai.usage.input_tokens").unwrap()["intValue"],
            "350"
        );
        assert_eq!(
            attr(&anthropic, "gen_ai.usage.cache_read.input_tokens").unwrap()["intValue"],
            "300"
        );
        assert_eq!(
            attr(&anthropic, "gen_ai.usage.cache_creation.input_tokens").unwrap()["intValue"],
            "40"
        );

        let openai = usage(UsageEvent {
            prompt_tokens: 120,
            cached_prompt_tokens: 100,
            cache_write_tokens: Some(20),
            completion_tokens: 30,
            reasoning_tokens: 12,
            ..sample_event()
        });
        assert_eq!(
            attr(&openai, "gen_ai.usage.input_tokens").unwrap()["intValue"],
            "120"
        );
        assert_eq!(
            attr(&openai, "gen_ai.usage.cache_read.input_tokens").unwrap()["intValue"],
            "100"
        );
        assert_eq!(
            attr(&openai, "gen_ai.usage.cache_creation.input_tokens").unwrap()["intValue"],
            "20"
        );
        assert_eq!(
            attr(&openai, "gen_ai.usage.reasoning.output_tokens").unwrap()["intValue"],
            "12"
        );

        let plain = usage(sample_event());
        assert!(attr(&plain, "gen_ai.usage.cache_read.input_tokens").is_none());
        assert!(attr(&plain, "gen_ai.usage.cache_creation.input_tokens").is_none());
    }

    /// A complete chat capture becomes the conventions' structured content;
    /// a cut one stays raw under `aisix.content.*`.
    #[test]
    fn chat_content_is_structured_unless_the_capture_was_cut() {
        let mut event = sample_event();
        event.operation = "chat".into();
        let prompt =
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).to_string();
        let record = |cap: u32| {
            content_record(
                &otlp_kind(SlsContentMode::Full, cap),
                &event,
                Some(&CapturedContent {
                    prompt: prompt.clone(),
                    response: "hello there".into(),
                    truncated: false,
                    prompt_truncated: false,
                    response_truncated: false,
                    output: None,
                }),
                None,
                None,
            )
            .unwrap()
        };
        let span = build_otlp_span(&record(4096), "x");
        assert_eq!(
            attr(&span, "gen_ai.input.messages").unwrap()["stringValue"],
            r#"[{"parts":[{"content":"hi","type":"text"}],"role":"user"}]"#
        );
        assert_eq!(
            attr(&span, "gen_ai.output.messages").unwrap()["stringValue"],
            r#"[{"finish_reason":"stop","parts":[{"content":"hello there","type":"text"}],"role":"assistant"}]"#
        );
        assert!(attr(&span, "aisix.content.prompt").is_none());

        // Only the answer cut: the prompt is still reported structured.
        let long_answer = content_record(
            &otlp_kind(SlsContentMode::Full, 2 * prompt.len() as u32),
            &event,
            Some(&CapturedContent::new(
                &prompt,
                &"x".repeat(4 * prompt.len()),
                8192,
            )),
            None,
            None,
        )
        .unwrap();
        let half = build_otlp_span(&long_answer, "x");
        assert!(attr(&half, "gen_ai.input.messages").is_some());
        assert!(attr(&half, "gen_ai.output.messages").is_none());
        assert!(attr(&half, "aisix.content.response").is_some());
        assert!(attr(&half, "aisix.content.prompt").is_none());

        let cut = build_otlp_span(&record(8), "x");
        assert!(attr(&cut, "gen_ai.input.messages").is_none());
        assert!(attr(&cut, "gen_ai.output.messages").is_none());
        assert!(attr(&cut, "aisix.content.prompt").is_some());
        assert_eq!(
            attr(&cut, "aisix.content_truncated").unwrap()["boolValue"],
            true
        );
    }

    /// Captured parts become the output message in order — reasoning, text,
    /// tool call with parsed arguments.
    #[test]
    fn structured_output_parts_become_the_output_message() {
        let mut event = sample_event();
        event.operation = "chat".into();
        let prompt =
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).to_string();
        let body = json!({"choices": [{"index": 0, "message": {
            "role": "assistant",
            "reasoning_content": "think",
            "content": "hello",
            "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{\"a\":1}"}}],
        }}]});
        let content = |cap: usize| {
            CapturedContent::new(&prompt, "hello", 4096).with_output(Some(
                crate::output::OutputCollector::from_body(
                    crate::output::OutputWire::OpenAiChat,
                    cap,
                    &body,
                ),
            ))
        };
        let span = |captured: CapturedContent| {
            let record = content_record(
                &otlp_kind(SlsContentMode::Full, 4096),
                &event,
                Some(&captured),
                None,
                None,
            )
            .unwrap();
            build_otlp_span(&record, "x")
        };
        let whole = span(content(4096));
        let output: Value = serde_json::from_str(
            attr(&whole, "gen_ai.output.messages").unwrap()["stringValue"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            output[0]["parts"],
            json!([
                {"type": "reasoning", "content": "think"},
                {"type": "text", "content": "hello"},
                {"type": "tool_call", "id": "c1", "name": "f", "arguments": {"a": 1}},
            ])
        );
        assert_eq!(output[0]["finish_reason"], "stop");
        assert!(attr(&whole, "aisix.content.response").is_none());

        // Parts the cap cut are not reported partially: the flat text
        // stands in, exactly as before parts were captured.
        let cut = span(content(6));
        assert_eq!(
            attr(&cut, "gen_ai.output.messages").unwrap()["stringValue"],
            r#"[{"finish_reason":"stop","parts":[{"content":"hello","type":"text"}],"role":"assistant"}]"#
        );
    }

    /// The per-request latency breakdown, the key's name, team and tags,
    /// TPOT and the modality counts each ride the operation span; absent
    /// ones stay off.
    #[test]
    fn span_carries_key_identity_tags_overhead_tpot_and_modality_counts() {
        let mut ev = sample_event();
        ev.api_key_name = "team-a-key".into();
        ev.team_id = "team-1".into();
        ev.tags = [
            ("department".to_string(), "rnd".to_string()),
            ("bg".to_string(), "cloud".to_string()),
        ]
        .into();
        ev.gateway_overhead_ms = Some(7);
        ev.upstream_tpot_ms = Some(12.5);
        ev.modality_tokens = aisix_gateway::ModalityTokens {
            prompt_audio_tokens: Some(30),
            prompt_image_tokens: Some(0),
            completion_audio_tokens: Some(40),
            completion_image_tokens: None,
        };
        let body = build_otlp_traces_payload(&ev, "x");
        let span = &body["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(
            attr(span, "aisix.api_key_name").unwrap()["stringValue"],
            "team-a-key"
        );
        assert_eq!(
            attr(span, "aisix.team_id").unwrap()["stringValue"],
            "team-1"
        );
        assert_eq!(
            attr(span, "aisix.tags.department").unwrap()["stringValue"],
            "rnd"
        );
        assert_eq!(attr(span, "aisix.tags.bg").unwrap()["stringValue"], "cloud");
        assert_eq!(
            attr(span, "aisix.gateway_overhead_ms").unwrap()["intValue"],
            "7"
        );
        assert_eq!(
            attr(span, "aisix.upstream_tpot_ms").unwrap()["doubleValue"],
            12.5
        );
        assert_eq!(
            attr(span, "gen_ai.usage.audio.input_tokens").unwrap()["intValue"],
            "30"
        );
        assert_eq!(
            attr(span, "gen_ai.usage.image.input_tokens").unwrap()["intValue"],
            "0"
        );
        assert_eq!(
            attr(span, "gen_ai.usage.audio.output_tokens").unwrap()["intValue"],
            "40"
        );
        assert!(attr(span, "gen_ai.usage.image.output_tokens").is_none());

        let bare = build_otlp_traces_payload(&sample_event(), "x").to_string();
        for key in [
            "aisix.api_key_name",
            "aisix.team_id",
            "aisix.tags.",
            "aisix.gateway_overhead_ms",
            "aisix.upstream_tpot_ms",
            "gen_ai.usage.audio.",
            "gen_ai.usage.image.",
        ] {
            assert!(!bare.contains(key), "{key}");
        }
    }

    #[test]
    fn export_resource_names_the_service_and_build() {
        let body = otlp_export_request(vec![], "edge-gw");
        let resource = &body["resourceSpans"][0]["resource"]["attributes"];
        assert_eq!(resource[0]["key"], "service.name");
        assert_eq!(resource[0]["value"]["stringValue"], "edge-gw");
        assert_eq!(resource[1]["key"], "service.version");
        assert_eq!(
            body["resourceSpans"][0]["scopeSpans"][0]["scope"]["version"],
            aisix_core::BUILD_VERSION.as_str()
        );
    }

    /// The regenerate-per-retry defect this PR removes: encoding the same
    /// record twice — what the delivery retry loop does — must produce
    /// byte-identical ids and timestamps.
    #[test]
    fn re_encoding_a_trace_record_is_byte_identical() {
        let record = hierarchy_record();
        let a = build_otlp_spans(&record, "exp-1");
        let b = build_otlp_spans(&record, "exp-1");
        assert_eq!(a, b);
    }

    /// A record without a trace snapshot (a synthetic event, or a caller
    /// predating the bundle) keeps the legacy single-span shape.
    #[test]
    fn traceless_record_falls_back_to_the_legacy_single_span() {
        let record = SinkRecord::metadata_only(sample_event());
        let spans = build_otlp_spans(&record, "exp-1");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0]["kind"], 3);
        assert!(spans[0].get("parentSpanId").is_none());
    }

    fn otlp_kind(content_mode: SlsContentMode, max_bytes: u32) -> ExporterKind {
        ExporterKind::OtlpHttp(OtlpHttpConfig {
            endpoint: "https://x/v1/traces".into(),
            headers: Default::default(),
            sample_rate: None,
            content_mode,
            content_max_bytes: max_bytes,
        })
    }

    fn datadog_kind(content_mode: SlsContentMode, max_bytes: u32) -> ExporterKind {
        ExporterKind::Datadog(DatadogConfig {
            site: "datadoghq.com".into(),
            credential_ref: "r".into(),
            service: "ai-gateway".into(),
            ddsource: "aisix-ai-gateway".into(),
            tags: vec![],
            content_mode,
            content_max_bytes: max_bytes,
        })
    }

    #[test]
    fn content_record_targets_only_full_capture_sls() {
        let event = sample_event();
        let captured = CapturedContent {
            prompt: "the prompt".into(),
            response: "the response".into(),
            truncated: false,
            prompt_truncated: false,
            response_truncated: false,
            output: None,
        };

        // otlp on its default (metadata_only) never carries content, even
        // when content was captured.
        let otlp = otlp_kind(SlsContentMode::MetadataOnly, 1024);
        assert!(content_record(&otlp, &event, Some(&captured), None, None).is_none());

        // object_store never carries content either — content_record gates on
        // the AliyunSls variant, so any other kind gets metadata-only even when
        // content was captured (no prompt/response leak into S3 / GCS / Azure).
        let objstore = serde_json::from_value::<ObservabilityExporter>(serde_json::json!({
            "name": "o",
            "enabled": true,
            "kind": "object_store",
            "provider": "s3",
            "bucket": "b",
            "prefix": "p",
            "credential_ref": "r"
        }))
        .unwrap()
        .kind;
        assert!(content_record(&objstore, &event, Some(&captured), None, None).is_none());

        // sls metadata_only → no content.
        let meta = sls_kind(SlsContentMode::MetadataOnly, 1024);
        assert!(content_record(&meta, &event, Some(&captured), None, None).is_none());

        // sls full but nothing captured → falls back to metadata.
        let full = sls_kind(SlsContentMode::Full, 1024);
        assert!(content_record(&full, &event, None, None, None).is_none());

        // sls full + captured content → a content-bearing record that still
        // carries the metadata.
        let rec = content_record(&full, &event, Some(&captured), None, None)
            .expect("full-capture sls with content yields a content record");
        let c = rec.content.as_ref().expect("content attached");
        assert_eq!(c.prompt, "the prompt");
        assert_eq!(c.response, "the response");
        assert!(!c.truncated);
        assert_eq!(rec.usage.request_id, "req-test-123");

        // Per-exporter cap truncates + flags.
        let big = CapturedContent {
            prompt: "a".repeat(500),
            response: "ok".into(),
            truncated: false,
            prompt_truncated: false,
            response_truncated: false,
            output: None,
        };
        let rec = content_record(
            &sls_kind(SlsContentMode::Full, 16),
            &event,
            Some(&big),
            None,
            None,
        )
        .unwrap();
        let c = rec.content.as_ref().unwrap();
        assert_eq!(c.prompt.len(), 16);
        assert!(c.truncated, "oversize content must flag truncated");

        // Handler-side truncation propagates even when the per-exporter cap
        // did not cut.
        let pre = CapturedContent {
            prompt: "short".into(),
            response: "short".into(),
            truncated: true,
            prompt_truncated: true,
            response_truncated: false,
            output: None,
        };
        let rec = content_record(&full, &event, Some(&pre), None, None).unwrap();
        assert!(
            rec.content.as_ref().unwrap().truncated,
            "source truncation must propagate"
        );

        // datadog behaves identically to sls: metadata_only → no content,
        // full + captured content → a content-bearing record (same shared
        // `SlsContentMode` plumbing).
        let dd_meta = datadog_kind(SlsContentMode::MetadataOnly, 1024);
        assert!(content_record(&dd_meta, &event, Some(&captured), None, None).is_none());
        let dd_full = datadog_kind(SlsContentMode::Full, 1024);
        assert!(content_record(&dd_full, &event, None, None, None).is_none());
        let rec = content_record(&dd_full, &event, Some(&captured), None, None)
            .expect("full-capture datadog with content yields a content record");
        let c = rec.content.as_ref().expect("content attached");
        assert_eq!(c.prompt, "the prompt");
        assert_eq!(c.response, "the response");
        // Per-exporter cap truncates a datadog record too.
        let rec = content_record(
            &datadog_kind(SlsContentMode::Full, 16),
            &event,
            Some(&big),
            None,
            None,
        )
        .unwrap();
        assert_eq!(rec.content.as_ref().unwrap().prompt.len(), 16);
        assert!(rec.content.as_ref().unwrap().truncated);

        // otlp_http behaves identically (#519 B.2): full + captured content →
        // a content-bearing record; full with nothing captured → metadata.
        let otlp_full = otlp_kind(SlsContentMode::Full, 1024);
        assert!(content_record(&otlp_full, &event, None, None, None).is_none());
        let rec = content_record(&otlp_full, &event, Some(&captured), None, None)
            .expect("full-capture otlp with content yields a content record");
        let c = rec.content.as_ref().expect("content attached");
        assert_eq!(c.prompt, "the prompt");
        assert_eq!(c.response, "the response");
        // Per-exporter cap truncates an otlp record too.
        let rec = content_record(
            &otlp_kind(SlsContentMode::Full, 16),
            &event,
            Some(&big),
            None,
            None,
        )
        .unwrap();
        assert_eq!(rec.content.as_ref().unwrap().prompt.len(), 16);
        assert!(rec.content.as_ref().unwrap().truncated);
    }

    #[test]
    fn span_carries_content_attrs_only_for_content_bearing_records() {
        let event = sample_event();

        // Metadata-only record (the default): no content keys on the span.
        let meta = SinkRecord::metadata_only(event.clone());
        let span = build_otlp_span(&meta, "x");
        let keys: Vec<&str> = span["attributes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["key"].as_str().unwrap())
            .collect();
        assert!(!keys.contains(&"gen_ai.prompt"));
        assert!(!keys.contains(&"gen_ai.completion"));
        assert!(!keys.contains(&"aisix.content_truncated"));

        // Content-bearing record (content_mode = full) of a payload that is
        // not a message list (the sample is an image generation): it rides
        // the bounded `aisix.content.*` keys, never a `gen_ai.*` message.
        let captured = CapturedContent {
            prompt: "what is 2+2?".into(),
            response: "4".into(),
            truncated: false,
            prompt_truncated: false,
            response_truncated: false,
            output: None,
        };
        let rec = content_record(
            &otlp_kind(SlsContentMode::Full, 1024),
            &event,
            Some(&captured),
            None,
            None,
        )
        .unwrap();
        let span = build_otlp_span(&rec, "x");
        let attrs = span["attributes"].as_array().unwrap();
        let find = |k: &str| attrs.iter().find(|a| a["key"] == k);
        assert_eq!(
            find("aisix.content.prompt").expect("prompt attr")["value"]["stringValue"],
            "what is 2+2?"
        );
        assert_eq!(
            find("aisix.content.response").expect("response attr")["value"]["stringValue"],
            "4"
        );
        for retired in [
            "gen_ai.prompt",
            "gen_ai.completion",
            "gen_ai.input.messages",
        ] {
            assert!(find(retired).is_none(), "{retired} must not be emitted");
        }
        // Nothing was truncated → no flag.
        assert!(find("aisix.content_truncated").is_none());

        // Truncation (here: per-exporter cap cut the prompt) flags the span.
        let big = CapturedContent {
            prompt: "a".repeat(500),
            response: "ok".into(),
            truncated: false,
            prompt_truncated: false,
            response_truncated: false,
            output: None,
        };
        let rec = content_record(
            &otlp_kind(SlsContentMode::Full, 16),
            &event,
            Some(&big),
            None,
            None,
        )
        .unwrap();
        let span = build_otlp_span(&rec, "x");
        let attrs = span["attributes"].as_array().unwrap();
        let flag = attrs
            .iter()
            .find(|a| a["key"] == "aisix.content_truncated")
            .expect("truncated flag");
        assert_eq!(flag["value"]["boolValue"], true);
    }

    #[test]
    fn sampling_extremes_drop_all_or_keep_all() {
        let ids: Vec<String> = (0..1000).map(|i| format!("req-{i}")).collect();
        for id in &ids {
            // Absent (None) = 1.0 = the pre-knob behaviour: keep everything.
            assert!(otlp_should_sample(None, id));
            assert!(otlp_should_sample(Some(1.0), id));
            // 0.0 drops everything.
            assert!(!otlp_should_sample(Some(0.0), id));
        }
    }

    #[test]
    fn sampling_is_deterministic_and_roughly_proportional() {
        // Same id → same decision, every time (the all-or-nothing guarantee
        // for a request's per-attempt spans, which share request_id).
        for id in ["req-a", "req-b", "req-c"] {
            let first = otlp_should_sample(Some(0.5), id);
            for _ in 0..10 {
                assert_eq!(otlp_should_sample(Some(0.5), id), first);
            }
        }

        // Across many distinct ids the kept fraction tracks the rate. The
        // per-process salt makes the exact count vary between runs, so the
        // band is wide: at n=1000, p=0.5 the standard deviation is ~16, and
        // ±150 is over nine of them.
        let kept = (0..1000)
            .filter(|i| otlp_should_sample(Some(0.5), &format!("req-{i}")))
            .count();
        assert!(
            (350..=650).contains(&kept),
            "rate 0.5 kept {kept}/1000 — hash badly skewed"
        );
    }

    // AISIX-Cloud#1288: the request id is caller-supplied, so the sampling
    // bucket must not be a pure function of it. Without the salt a caller can
    // grind ids offline until one falls outside a sampled exporter's bucket
    // and never appear in a trace again.
    #[test]
    fn sampling_bucket_is_not_predictable_from_the_request_id_alone() {
        // The unsalted hash is what an attacker can compute off the wire.
        // Over many ids it must disagree with the real decision often enough
        // that precomputing an evasive id is not possible.
        let disagreements = (0..1000)
            .filter(|i| {
                let id = format!("req-{i}");
                let unsalted = fnv1a_64(id.as_bytes()) % SAMPLE_PRECISION
                    < (0.5 * SAMPLE_PRECISION as f64) as u64;
                unsalted != otlp_should_sample(Some(0.5), &id)
            })
            .count();
        assert!(
            disagreements > 100,
            "only {disagreements}/1000 ids differ from the unsalted bucket — \
             the salt is not reaching the hash"
        );
    }

    #[test]
    fn fingerprint_otlp_covers_the_new_knobs() {
        let base = OtlpHttpConfig {
            endpoint: "https://x/v1/traces".into(),
            headers: Default::default(),
            sample_rate: None,
            content_mode: SlsContentMode::MetadataOnly,
            content_max_bytes: 128 * 1024,
        };
        assert_eq!(
            fingerprint_otlp(&base),
            fingerprint_otlp(&base.clone()),
            "same config must fingerprint identically"
        );

        // Each knob edit must rebuild the pipeline (#519 B.2).
        let mut sampled = base.clone();
        sampled.sample_rate = Some(0.5);
        assert_ne!(fingerprint_otlp(&base), fingerprint_otlp(&sampled));

        let mut full = base.clone();
        full.content_mode = SlsContentMode::Full;
        assert_ne!(fingerprint_otlp(&base), fingerprint_otlp(&full));

        let mut capped = base.clone();
        capped.content_max_bytes = 4096;
        assert_ne!(fingerprint_otlp(&base), fingerprint_otlp(&capped));

        let mut moved = base.clone();
        moved.endpoint = "https://y/v1/traces".into();
        assert_ne!(fingerprint_otlp(&base), fingerprint_otlp(&moved));
    }

    #[test]
    fn content_capture_cap_picks_max_enabled_full_sls() {
        fn sls(name: &str, enabled: bool, mode: &str, max: u32) -> ObservabilityExporter {
            serde_json::from_value(serde_json::json!({
                "name": name,
                "enabled": enabled,
                "kind": "aliyun_sls",
                "endpoint": "ap-southeast-3.log.aliyuncs.com",
                "project": "p",
                "logstore": "l",
                "credential_ref": "r",
                "content_mode": mode,
                "content_max_bytes": max,
            }))
            .unwrap()
        }

        // No exporter wants content → None (handler skips capture).
        let otlp = sample_exporter();
        let meta = sls("a", true, "metadata_only", 1024);
        assert_eq!(content_capture_cap([&otlp, &meta]), None);

        // One full-capture sls → its cap.
        let full = sls("a", true, "full", 4096);
        assert_eq!(content_capture_cap([&full]), Some(4096));

        // Max across several full-capture exporters.
        let full_b = sls("b", true, "full", 8192);
        assert_eq!(content_capture_cap([&full, &full_b]), Some(8192));

        // A disabled full-capture exporter is ignored.
        let disabled = sls("a", false, "full", 4096);
        assert_eq!(content_capture_cap([&disabled]), None);

        // A full-capture datadog exporter counts toward the cap too, and the
        // max is taken across both kinds.
        fn datadog(name: &str, enabled: bool, mode: &str, max: u32) -> ObservabilityExporter {
            serde_json::from_value(serde_json::json!({
                "name": name,
                "enabled": enabled,
                "kind": "datadog",
                "site": "datadoghq.com",
                "credential_ref": "r",
                "service": "ai-gateway",
                "content_mode": mode,
                "content_max_bytes": max,
            }))
            .unwrap()
        }
        let dd_full = datadog("dd", true, "full", 16384);
        assert_eq!(content_capture_cap([&dd_full]), Some(16384));
        assert_eq!(content_capture_cap([&full, &dd_full]), Some(16384));
        let dd_meta = datadog("dd", true, "metadata_only", 4096);
        assert_eq!(content_capture_cap([&dd_meta]), None);

        // A full-capture otlp_http exporter counts toward the cap too
        // (#519 B.2); a metadata-only one (the default) does not — the
        // `sample_exporter()` fixture above already proved the default shape
        // yields None.
        fn otlp_exp(name: &str, mode: &str, max: u32) -> ObservabilityExporter {
            serde_json::from_value(serde_json::json!({
                "name": name,
                "enabled": true,
                "kind": "otlp_http",
                "endpoint": "https://x/v1/traces",
                "content_mode": mode,
                "content_max_bytes": max,
            }))
            .unwrap()
        }
        let otlp_full = otlp_exp("ot", "full", 32768);
        assert_eq!(content_capture_cap([&otlp_full]), Some(32768));
        assert_eq!(content_capture_cap([&dd_full, &otlp_full]), Some(32768));
        assert_eq!(
            content_capture_cap([&otlp_exp("ot", "metadata_only", 4096)]),
            None
        );
    }

    /// GenAI `output_tokens` includes reasoning. A record that counts its
    /// reasoning beside the completion (Gemini thoughts) exports the sum;
    /// one that counts it inside (OpenAI) exports the completion as is.
    #[test]
    fn output_tokens_include_reasoning_counted_beside_the_completion() {
        let output = |event: UsageEvent| {
            let body = build_otlp_traces_payload(&event, "test-exp");
            let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
                .as_array()
                .unwrap()
                .clone();
            attrs
                .iter()
                .find(|a| a["key"] == "gen_ai.usage.output_tokens")
                .expect("output tokens exported")["value"]["intValue"]
                .clone()
        };
        let gemini = UsageEvent {
            prompt_tokens: 100,
            completion_tokens: 20,
            reasoning_tokens: 30,
            total_tokens: 150,
            ..sample_event()
        };
        assert_eq!(output(gemini), serde_json::json!("50"));
        let openai = UsageEvent {
            prompt_tokens: 40,
            completion_tokens: 25,
            reasoning_tokens: 10,
            total_tokens: 65,
            ..sample_event()
        };
        assert_eq!(output(openai), serde_json::json!("25"));
    }

    #[test]
    fn payload_carries_genai_semconv_attributes() {
        let body = build_otlp_traces_payload(&sample_event(), "test-exp");
        let span = &body["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        // An operation the conventions have no value for is named by the
        // gateway's own; no upstream was resolved, so there is no target.
        assert_eq!(span["name"], "image_generation");
        // Success leaves the status unset.
        assert!(span.get("status").is_none());
        let attrs = span["attributes"].as_array().unwrap();
        let keys: Vec<&str> = attrs.iter().map(|a| a["key"].as_str().unwrap()).collect();
        assert!(
            !keys.contains(&"gen_ai.system"),
            "replaced by gen_ai.provider.name"
        );
        assert!(!keys.contains(&"error.type"));
        assert_eq!(
            attrs
                .iter()
                .find(|a| a["key"] == "gen_ai.operation.name")
                .unwrap()["value"]["stringValue"],
            "image_generation"
        );
        // The semconv value collapses every OpenAI-shaped route onto `chat`;
        // this is the one that says which endpoint (AISIX-Cloud#1461), so the
        // VALUE is what has to be checked — the key alone would be satisfied
        // by an empty attribute.
        let operation = attrs
            .iter()
            .find(|a| a["key"] == "aisix.operation")
            .expect("aisix.operation must be present");
        assert_eq!(operation["value"]["stringValue"], "image_generation");
        assert!(keys.contains(&"gen_ai.response.model"));
        assert!(keys.contains(&"gen_ai.response.id"));
        assert!(keys.contains(&"gen_ai.usage.input_tokens"));
        assert!(keys.contains(&"gen_ai.usage.output_tokens"));
        assert!(keys.contains(&"gen_ai.response.finish_reasons"));
        assert!(keys.contains(&"http.response.status_code"));
        assert!(keys.contains(&"aisix.api_key_id"));
        assert!(keys.contains(&"aisix.model_id"));
        assert!(keys.contains(&"aisix.exporter_name"));
        assert!(keys.contains(&"aisix.request_id"));
    }

    #[test]
    fn an_a2a_call_exports_as_an_agent_span_not_a_chat_one() {
        // Every gateway-protocol event used to be encoded as `chat` /
        // `chat.completions`, so an agent call was indistinguishable from a
        // model inference in a trace backend and the agent, method and task it
        // touched appeared nowhere (AISIX-Cloud#1215).
        let mut ev = sample_event();
        ev.inbound_protocol = "a2a".into();
        ev.a2a_agent_name = "invoice-processor".into();
        ev.a2a_method = "SendStreamingMessage".into();
        ev.a2a_operation = "message/stream".into();
        ev.a2a_protocol_version = "1.0".into();
        ev.a2a_task_id = "task-9".into();
        ev.a2a_context_id = "ctx-4".into();
        ev.a2a_task_state = "working".into();

        let body = build_otlp_traces_payload(&ev, "test-exp");
        let span = &body["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(span["name"], "invoke_agent invoice-processor");
        let attrs = span["attributes"].as_array().unwrap();
        let find = |k: &str| attrs.iter().find(|a| a["key"] == k);
        let string_at = |k: &str| find(k).unwrap()["value"]["stringValue"].clone();
        assert_eq!(string_at("gen_ai.operation.name"), "invoke_agent");
        assert_eq!(string_at("gen_ai.agent.name"), "invoice-processor");
        // A2A's context id is semconv's conversation id — the thread a
        // multi-turn exchange's tasks hang off.
        assert_eq!(string_at("gen_ai.conversation.id"), "ctx-4");
        assert_eq!(string_at("aisix.a2a.method"), "SendStreamingMessage");
        assert_eq!(string_at("aisix.a2a.operation"), "message/stream");
        assert_eq!(string_at("aisix.a2a.protocol_version"), "1.0");
        assert_eq!(string_at("aisix.a2a.task_id"), "task-9");
        assert_eq!(string_at("aisix.a2a.task_state"), "working");
    }

    #[test]
    fn a_passthrough_relay_exports_route_and_identity_not_a_chat_span() {
        // The passthrough handler emits usage events with
        // `inbound_protocol = "passthrough"`; without the explicit branch a
        // Copilot relay exported as a bare `chat.completions` span with the
        // route and the device-injected end-user identity recorded nowhere.
        let mut ev = sample_event();
        ev.inbound_protocol = "passthrough".into();
        ev.operation = "passthrough".into();
        ev.passthrough_route_name = "copilot-chat".into();
        ev.client_identity = "alice@example.com".into();

        let body = build_otlp_traces_payload(&ev, "test-exp");
        let span = &body["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(span["name"], "passthrough copilot-chat");
        let attrs = span["attributes"].as_array().unwrap();
        let find = |k: &str| attrs.iter().find(|a| a["key"] == k);
        let string_at = |k: &str| find(k).unwrap()["value"]["stringValue"].clone();
        assert_eq!(string_at("gen_ai.operation.name"), "passthrough");
        assert_eq!(string_at("aisix.passthrough.route_name"), "copilot-chat");
        assert_eq!(string_at("aisix.client_identity"), "alice@example.com");
    }

    #[test]
    fn client_identity_exports_without_a_protocol_gate() {
        // The identity attribute must not depend on the passthrough branch:
        // any handler that populates the field gets it exported.
        let mut ev = sample_event();
        ev.client_identity = "bob@example.com".into();
        let body = build_otlp_traces_payload(&ev, "test-exp");
        let span = &body["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        let attrs = span["attributes"].as_array().unwrap();
        let id = attrs
            .iter()
            .find(|a| a["key"] == "aisix.client_identity")
            .unwrap();
        assert_eq!(id["value"]["stringValue"], "bob@example.com");
    }

    #[test]
    fn a_streamed_a2a_call_carries_its_event_count() {
        // A unary call has no count, and exporting a zero would read as "the
        // stream carried nothing" rather than "there was no stream".
        let mut streamed = sample_event();
        streamed.inbound_protocol = "a2a".into();
        streamed.a2a_agent_name = "invoice-processor".into();
        streamed.a2a_stream_event_count = 17;
        let body = build_otlp_traces_payload(&streamed, "test-exp");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(
            attrs
                .iter()
                .find(|a| a["key"] == "aisix.a2a.stream_event_count")
                .unwrap()["value"]["intValue"],
            "17"
        );

        let mut unary = sample_event();
        unary.inbound_protocol = "a2a".into();
        let body = build_otlp_traces_payload(&unary, "test-exp");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap()
            .clone();
        assert!(attrs
            .iter()
            .all(|a| a["key"] != "aisix.a2a.stream_event_count"));
    }

    #[test]
    fn an_mcp_call_exports_as_a_tool_span() {
        let mut ev = sample_event();
        ev.inbound_protocol = "mcp".into();
        ev.mcp_server_name = "github".into();
        ev.mcp_tool_name = "create_issue".into();

        let body = build_otlp_traces_payload(&ev, "test-exp");
        let span = &body["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        // MCP's own naming: `{mcp.method.name} {gen_ai.tool.name}`.
        assert_eq!(span["name"], "tools/call create_issue");
        let attrs = span["attributes"].as_array().unwrap();
        let find = |k: &str| attrs.iter().find(|a| a["key"] == k);
        assert_eq!(
            find("gen_ai.operation.name").unwrap()["value"]["stringValue"],
            "execute_tool"
        );
        assert_eq!(
            find("gen_ai.tool.name").unwrap()["value"]["stringValue"],
            "create_issue"
        );
        assert_eq!(
            find("aisix.mcp.server_name").unwrap()["value"]["stringValue"],
            "github"
        );
        assert_eq!(
            find("mcp.method.name").unwrap()["value"]["stringValue"],
            "tools/call"
        );
        // A tool call has no token usage to report.
        assert!(find("gen_ai.usage.input_tokens").is_none());
    }

    #[test]
    fn a_caller_chosen_target_cannot_grow_the_span_name() {
        // An MCP tool name is the caller's own `tools/call` `params.name`,
        // recorded before the tool is known to exist, and the request body
        // limit defaults to unlimited. A trace backend indexes span names, so
        // an oversized one falls back to the bare operation and travels as an
        // attribute instead — itself cut to a fixed cap.
        let mut ev = sample_event();
        ev.inbound_protocol = "mcp".into();
        ev.mcp_tool_name = "t".repeat(4096);

        let body = build_otlp_traces_payload(&ev, "test-exp");
        let span = &body["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(span["name"], "tools/call");
        let attrs = span["attributes"].as_array().unwrap();
        let tool = attrs
            .iter()
            .find(|a| a["key"] == "gen_ai.tool.name")
            .unwrap()["value"]["stringValue"]
            .as_str()
            .unwrap();
        assert_eq!(tool.len(), MAX_PROTOCOL_ATTR_BYTES);
    }

    #[test]
    fn a_capped_attribute_is_cut_on_a_char_boundary() {
        // A multi-byte character straddling the cap must not produce invalid
        // UTF-8 in the export body.
        let mut ev = sample_event();
        ev.inbound_protocol = "a2a".into();
        ev.a2a_method = "情".repeat(200);

        let body = build_otlp_traces_payload(&ev, "test-exp");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap()
            .clone();
        let method = attrs
            .iter()
            .find(|a| a["key"] == "aisix.a2a.method")
            .unwrap()["value"]["stringValue"]
            .as_str()
            .unwrap();
        assert!(method.len() <= MAX_PROTOCOL_ATTR_BYTES);
        assert!(method.chars().all(|c| c == '情'));
    }

    #[test]
    fn a_gateway_protocol_span_without_a_target_keeps_a_bare_name() {
        // A pre-dispatch rejection resolves no agent, so there is no
        // low-cardinality target to append — semconv says name the span after
        // the operation alone rather than inventing one.
        let mut ev = sample_event();
        ev.inbound_protocol = "a2a".into();
        let body = build_otlp_traces_payload(&ev, "test-exp");
        assert_eq!(
            body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"],
            "invoke_agent"
        );
    }

    /// A wildcard row's upstream model is the caller's own substitution, so
    /// it stays out of the span name (unbounded cardinality) but is still
    /// the request-model attribute.
    #[test]
    fn a_caller_named_upstream_model_stays_out_of_the_span_name() {
        let mut ev = sample_event();
        ev.operation = "chat".into();
        let record =
            SinkRecord::metadata_only(ev).with_facts(Some(Arc::new(crate::genai::SpanFacts {
                upstream: Some(crate::genai::UpstreamFacts {
                    request_model: "anything-the-caller-sent".into(),
                    request_model_caller_named: true,
                    ..Default::default()
                }),
                ..Default::default()
            })));
        let span = build_otlp_span(&record, "x");
        assert_eq!(span["name"], "chat");
        assert_eq!(
            attr(&span, "gen_ai.request.model").unwrap()["stringValue"],
            "anything-the-caller-sent"
        );
    }

    #[test]
    fn requested_model_id_rides_beside_model_id() {
        // A group request: the addressed group and the dispatched target are
        // two ids, and a span must carry both so it stays joinable to the
        // group across a rename.
        let mut ev = sample_event();
        ev.requested_model = "smart-group".into();
        ev.requested_model_id = "group-uuid".into();
        let span = build_otlp_span(&SinkRecord::metadata_only(ev), "x");
        assert_eq!(
            attr(&span, "aisix.requested_model_id").unwrap()["stringValue"],
            "group-uuid"
        );
        assert_eq!(
            attr(&span, "aisix.model_id").unwrap()["stringValue"],
            "mod-uuid"
        );

        // Nothing resolved: no blank attribute, the same as `aisix.model_id`.
        let span = build_otlp_span(&SinkRecord::metadata_only(UsageEvent::default()), "x");
        assert!(attr(&span, "aisix.requested_model_id").is_none());
        assert!(attr(&span, "aisix.model_id").is_none());
    }

    #[test]
    fn served_service_tier_is_its_own_attribute() {
        let mut ev = sample_event();
        ev.service_tier = Some("priority".into());
        let span = build_otlp_span(&SinkRecord::metadata_only(ev), "x");
        assert_eq!(
            attr(&span, "aisix.service_tier").unwrap()["stringValue"],
            "priority"
        );

        let span = build_otlp_span(&SinkRecord::metadata_only(sample_event()), "x");
        assert!(attr(&span, "aisix.service_tier").is_none());
    }

    #[test]
    fn inference_spans_are_named_by_operation_and_requested_model() {
        // Every chat-shaped route maps onto the conventions' `chat`, the
        // legacy completions route onto `text_completion` and embeddings onto
        // `embeddings`; the span is named after the model the request was
        // made to, not the caller's alias.
        for (operation, genai) in [
            ("chat", "chat"),
            ("messages", "chat"),
            ("responses", "chat"),
            ("completions", "text_completion"),
            ("embeddings", "embeddings"),
            ("rerank", "rerank"),
        ] {
            let mut ev = sample_event();
            ev.operation = operation.into();
            ev.requested_model = "my-alias".into();
            let record =
                SinkRecord::metadata_only(ev).with_facts(Some(Arc::new(crate::genai::SpanFacts {
                    upstream: Some(crate::genai::UpstreamFacts {
                        provider_name: "openai".into(),
                        request_model: "gpt-4o".into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                })));
            let span = build_otlp_span(&record, "test-exp");
            assert_eq!(span["name"], format!("{genai} gpt-4o"), "{operation}");
            let attrs = span["attributes"].as_array().unwrap();
            let string_at = |k: &str| {
                attrs.iter().find(|a| a["key"] == k).unwrap()["value"]["stringValue"].clone()
            };
            assert_eq!(string_at("gen_ai.operation.name"), genai);
            assert_eq!(string_at("gen_ai.request.model"), "gpt-4o");
            assert_eq!(string_at("aisix.requested_model"), "my-alias");
            assert_eq!(string_at("gen_ai.provider.name"), "openai");
        }
    }

    #[test]
    fn payload_carries_per_attempt_attributes() {
        // A failed fallback attempt (#655): zero tokens, error info, target.
        let mut ev = sample_event();
        ev.attempt_index = 1;
        ev.attempt_kind = "fallback".into();
        ev.attempt_model = "secondary".into();
        ev.error_class = "upstream_status".into();
        ev.error_message = "upstream returned 502".into();
        let body = build_otlp_traces_payload(&ev, "test-exp");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let find = |k: &str| attrs.iter().find(|a| a["key"] == k);
        assert_eq!(
            find("aisix.attempt_index").unwrap()["value"]["intValue"],
            "1"
        );
        assert_eq!(
            find("aisix.attempt_kind").unwrap()["value"]["stringValue"],
            "fallback"
        );
        assert_eq!(
            find("aisix.attempt_model").unwrap()["value"]["stringValue"],
            "secondary"
        );
        assert_eq!(
            find("aisix.error_class").unwrap()["value"]["stringValue"],
            "upstream_status"
        );
        assert_eq!(
            find("aisix.error_message").unwrap()["value"]["stringValue"],
            "upstream returned 502"
        );

        // A direct (non-routing) success omits the routing-only attrs but
        // still carries attempt_index=0.
        let plain = build_otlp_traces_payload(&sample_event(), "test-exp");
        let plain_attrs = plain["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let keys: Vec<&str> = plain_attrs
            .iter()
            .map(|a| a["key"].as_str().unwrap())
            .collect();
        assert!(keys.contains(&"aisix.attempt_index"));
        assert!(!keys.contains(&"aisix.attempt_kind"));
        assert!(!keys.contains(&"aisix.attempt_model"));
        assert!(!keys.contains(&"aisix.error_class"));
    }

    #[test]
    fn payload_carries_a_bounded_gateway_embedding_child_audit() {
        let mut event = sample_event();
        // These belong only to the parent event. The child-audit schema has
        // no text, provider-error, or credential fields, so they must never
        // cross into its serialized OTLP attribute.
        event.error_message = "provider-error-sentinel".into();
        event.api_key_id = "credential-sentinel".into();
        event.requested_model = "caller-text-sentinel".into();
        // One more than the OTLP resource-ID cap, with an unsafe multibyte
        // value: both caps must remain visible without exporting the input.
        event.gateway_embedding_calls = (0..=MAX_GATEWAY_EMBEDDING_CALLS_ATTR_ITEMS)
            .map(|index| GatewayEmbeddingCall {
                count: 1,
                purpose: crate::usage::GatewayEmbeddingPurpose::SemanticRoute,
                embedding_model_id: format!("{}-{index}", "情".repeat(100)),
                prompt_tokens: index as u32,
                total_tokens: index as u32 + 1,
                usage_source: crate::usage::GatewayEmbeddingUsageSource::Reported,
                latency_ms: index as u32 + 2,
                outcome: crate::usage::GatewayEmbeddingOutcome::Succeeded,
            })
            .collect();
        event.gateway_embedding_calls_dropped = 7;

        let body = build_otlp_traces_payload(&event, "test-exp");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let find = |key: &str| attrs.iter().find(|attr| attr["key"] == key);
        let encoded = find("aisix.gateway_embedding_calls")
            .expect("child audit is in the OTLP allowlist")["value"]["stringValue"]
            .as_str()
            .expect("child audit is serialized JSON");
        assert!(encoded.len() <= MAX_GATEWAY_EMBEDDING_CALLS_ATTR_BYTES);
        let calls: Vec<Value> =
            serde_json::from_str(encoded).expect("the bounded child audit remains valid JSON");
        assert_eq!(calls.len(), MAX_GATEWAY_EMBEDDING_CALLS_ATTR_ITEMS);
        for sentinel in [
            "provider-error-sentinel",
            "credential-sentinel",
            "caller-text-sentinel",
        ] {
            assert!(
                !encoded.contains(sentinel),
                "parent-only sentinel leaked into child audit: {sentinel}"
            );
        }
        assert_eq!(calls[0]["purpose"], "semantic_route");
        let model_id = calls[0]["embedding_model_id"].as_str().unwrap();
        assert_eq!(model_id, "unknown");
        assert!(!encoded.contains('情'));
        assert_eq!(
            find("aisix.gateway_embedding_calls_dropped")
                .expect("ledger and exporter truncation are both visible")["value"]["intValue"],
            "8",
        );
    }

    #[test]
    fn gateway_embedding_audit_ids_are_sanitized_at_the_otlp_boundary() {
        let unsafe_ids = [
            ("chinese-leak-", "chinese-leak-情".to_owned()),
            ("control-leak-", "control-leak-\u{1b}[31m".to_owned()),
            ("at-leak-", "at-leak@example".to_owned()),
            (
                "long-leak-",
                format!(
                    "long-leak-{}",
                    "a".repeat(MAX_GATEWAY_EMBEDDING_MODEL_ID_ATTR_BYTES)
                ),
            ),
        ];
        let mut audit_calls = vec![GatewayEmbeddingCall {
            count: 1,
            purpose: crate::usage::GatewayEmbeddingPurpose::SemanticRoute,
            embedding_model_id: "embed_model-01".into(),
            prompt_tokens: 0,
            total_tokens: 0,
            usage_source: crate::usage::GatewayEmbeddingUsageSource::Unavailable,
            latency_ms: 0,
            outcome: crate::usage::GatewayEmbeddingOutcome::Failed,
        }];
        audit_calls.extend(unsafe_ids.iter().map(|(_, id)| GatewayEmbeddingCall {
            count: 1,
            purpose: crate::usage::GatewayEmbeddingPurpose::SemanticRoute,
            embedding_model_id: id.clone(),
            prompt_tokens: 0,
            total_tokens: 0,
            usage_source: crate::usage::GatewayEmbeddingUsageSource::Unavailable,
            latency_ms: 0,
            outcome: crate::usage::GatewayEmbeddingOutcome::Failed,
        }));

        let (attribute, dropped) = gateway_embedding_calls_attribute(&audit_calls);
        assert_eq!(dropped, 0);
        let attribute = attribute.expect("audit attribute is retained");
        let encoded = attribute["value"]["stringValue"]
            .as_str()
            .expect("audit attribute is JSON text");
        let calls: Vec<Value> =
            serde_json::from_str(encoded).expect("audit attribute remains valid JSON");
        assert_eq!(calls[0]["embedding_model_id"], "embed_model-01");
        for (call, (prefix, _)) in calls.iter().skip(1).zip(unsafe_ids.iter()) {
            assert_eq!(call["embedding_model_id"], "unknown");
            assert!(
                !encoded.contains(prefix),
                "unsafe embedding resource ID prefix leaked into OTLP JSON: {prefix}"
            );
        }
    }

    #[test]
    fn payload_carries_client_attribution_when_present() {
        let mut ev = sample_event();
        ev.client_source_ip = "203.0.113.7".into();
        ev.client_user_agent = "codex-cli/1.2".into();
        let body = build_otlp_traces_payload(&ev, "test-exp");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        // Under the client / user-agent conventions' names, replacing the
        // former `aisix.client_*` keys.
        let ip = attrs.iter().find(|a| a["key"] == "client.address");
        let ua = attrs.iter().find(|a| a["key"] == "user_agent.original");
        assert_eq!(
            ip.expect("client.address attr")["value"]["stringValue"],
            "203.0.113.7"
        );
        assert_eq!(
            ua.expect("user_agent.original attr")["value"]["stringValue"],
            "codex-cli/1.2"
        );
        assert!(!body.to_string().contains("aisix.client_source_ip"));
        assert!(!body.to_string().contains("aisix.client_user_agent"));
    }

    /// Each exporter reports only the names in its own lists: the record
    /// built for an exporter selecting `x-sub-user` drops `x-department`
    /// although the request carried it, and response headers ride the
    /// span under `http.response.header.<name>`.
    #[test]
    fn payload_carries_the_exporters_own_headers_as_semconv_string_arrays() {
        let mut ev = sample_event();
        ev.request_headers = [
            ("x-sub-user".to_string(), "alice".to_string()),
            ("x-department".to_string(), "eng, ops".to_string()),
        ]
        .into();
        ev.upstream_response_headers = [
            ("x-upstream-region".to_string(), "us-east".to_string()),
            ("x-other".to_string(), "unselected".to_string()),
        ]
        .into();
        let record = SinkRecord::metadata_only(ev).with_exporter_headers(
            &["x-sub-user".to_string(), "x-department".to_string()],
            &["x-upstream-region".to_string()],
        );
        let body = otlp_export_request(
            vec![build_otlp_span(&record, "test-exp")],
            DEFAULT_SERVICE_NAME,
        );
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        assert!(!body.to_string().contains("x-other"));
        for (key, value) in [
            ("http.request.header.x-sub-user", "alice"),
            ("http.request.header.x-department", "eng, ops"),
            ("http.response.header.x-upstream-region", "us-east"),
        ] {
            let attr = attrs
                .iter()
                .find(|a| a["key"] == key)
                .unwrap_or_else(|| panic!("{key} attr missing"));
            assert_eq!(
                attr["value"],
                json!({"arrayValue": {"values": [{"stringValue": value}]}}),
            );
        }
        let mut unselected = sample_event();
        unselected.request_headers = [("x-sub-user".to_string(), "alice".to_string())].into();
        let none = build_otlp_traces_payload(&unselected, "x");
        assert!(!none.to_string().contains("http.request.header."));
    }

    #[test]
    fn payload_omits_client_attribution_when_empty() {
        let body = build_otlp_traces_payload(&sample_event(), "x");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let keys: Vec<&str> = attrs.iter().map(|a| a["key"].as_str().unwrap()).collect();
        assert!(!keys.contains(&"client.address"));
        assert!(!keys.contains(&"user_agent.original"));
    }

    #[test]
    fn payload_marks_5xx_as_error_status() {
        let mut ev = sample_event();
        ev.status_code = 503;
        let body = build_otlp_traces_payload(&ev, "x");
        assert_eq!(
            body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["status"]["code"],
            2
        );
    }

    fn otlp_test_client() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    }

    fn batch_of(n: usize) -> EventBatch {
        let records = (0..n)
            .map(|_| Arc::new(crate::sink::SinkRecord::metadata_only(sample_event())))
            .collect();
        EventBatch::new(records)
    }

    #[tokio::test]
    async fn otlp_sink_posts_one_request_with_all_spans() {
        let server = wiremock::MockServer::builder().start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/traces"))
            .respond_with(wiremock::ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let sink = OtlpSink::new(
            "test-exp",
            format!("{}/v1/traces", server.uri()),
            BTreeMap::new(),
            OtlpHttpFanOut::new().inner.client.clone(),
        );

        let ack = sink
            .append_batch(&batch_of(3), &IdempotencyMarker::None)
            .await
            .expect("2xx delivers the batch");
        assert_eq!(ack.accepted, 3);

        let reqs = server.received_requests().await.unwrap();
        assert_eq!(reqs.len(), 1, "one batched request, not three spawns");
        assert_eq!(
            reqs[0].headers["user-agent"],
            format!("aisix-dp/{}", &*aisix_core::BUILD_VERSION)
        );
        let body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
        let spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
            .as_array()
            .unwrap();
        assert_eq!(spans.len(), 3, "all three spans in one export request");
    }

    #[tokio::test]
    async fn otlp_sink_classifies_5xx_as_transient() {
        let server = wiremock::MockServer::builder().start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let sink = OtlpSink::new("e", server.uri(), BTreeMap::new(), otlp_test_client());
        let err = sink
            .append_batch(&batch_of(1), &IdempotencyMarker::None)
            .await
            .unwrap_err();
        assert!(err.is_transient(), "5xx must be retryable: {err}");
    }

    #[tokio::test]
    async fn otlp_sink_classifies_4xx_as_permanent() {
        let server = wiremock::MockServer::builder().start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(400))
            .mount(&server)
            .await;
        let sink = OtlpSink::new("e", server.uri(), BTreeMap::new(), otlp_test_client());
        let err = sink
            .append_batch(&batch_of(1), &IdempotencyMarker::None)
            .await
            .unwrap_err();
        assert!(!err.is_transient(), "4xx must be permanent: {err}");
    }

    /// The `last_error` a pipeline records for `sink`'s first failed
    /// delivery — the text the warn log and the heartbeat carry too.
    async fn first_last_error(sink: OtlpSink) -> String {
        let (handle, worker) = crate::sink::SinkPipeline::new(
            Arc::new(sink),
            PipelineConfig {
                flush_interval: Duration::from_millis(20),
                ..PipelineConfig::default()
            },
        );
        let (cancel_tx, cancel) = tokio::sync::watch::channel(false);
        let run = tokio::spawn(worker.run(cancel));
        assert!(handle.try_enqueue(Arc::new(SinkRecord::metadata_only(sample_event()))));
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let last = loop {
            if let Some(e) = handle.stats().last_error {
                break e;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no delivery error recorded"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        cancel_tx.send(true).unwrap();
        run.abort();
        last
    }

    /// The headers carry the receiver's credential, which must not reach
    /// `last_error` through a receiver that echoes it back in its error
    /// body. A configured endpoint is reported as configured, userinfo
    /// included — it gets no special handling.
    #[tokio::test]
    async fn otlp_delivery_errors_carry_no_configured_credential() {
        let headers = BTreeMap::from([(
            "authorization".to_string(),
            "Bearer otlp-header-token".to_string(),
        )]);
        let closed = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let unreachable = first_last_error(OtlpSink::new(
            "e",
            format!("http://otlp-user:otlp-pass@127.0.0.1:{closed}/v1/traces"),
            headers.clone(),
            otlp_test_client(),
        ))
        .await;
        assert!(
            unreachable.contains(&format!(
                "POST http://otlp-user:otlp-pass@127.0.0.1:{closed}/v1/traces"
            )),
            "the endpoint is reported as configured: {unreachable}"
        );

        let server = wiremock::MockServer::builder().start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(401)
                    .set_body_string("rejected token otlp-header-token"),
            )
            .mount(&server)
            .await;
        let echoed = first_last_error(OtlpSink::new(
            "e",
            server
                .uri()
                .replace("http://", "http://otlp-user:otlp-pass@"),
            headers,
            otlp_test_client(),
        ))
        .await;
        assert!(echoed.contains("HTTP 401"), "{echoed}");

        for last in [unreachable, echoed] {
            assert!(
                !last.contains("otlp-header-token"),
                "the header credential leaked into: {last}"
            );
        }
    }

    #[test]
    fn payload_omits_empty_optional_fields() {
        let mut ev = sample_event();
        ev.provider_request_id = String::new();
        ev.provider_model_version = String::new();
        ev.finish_reason = String::new();
        let body = build_otlp_traces_payload(&ev, "x");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let keys: Vec<&str> = attrs.iter().map(|a| a["key"].as_str().unwrap()).collect();
        assert!(!keys.contains(&"gen_ai.response.id"));
        assert!(!keys.contains(&"gen_ai.response.model"));
        assert!(!keys.contains(&"gen_ai.response.finish_reasons"));
        // ttft_ms = 0 (default) → omitted
        assert!(!keys.contains(&"aisix.upstream_ttft_ms"));
    }

    #[test]
    fn payload_includes_ttft_when_set() {
        let mut ev = sample_event();
        ev.upstream_ttft_ms = 42;
        let body = build_otlp_traces_payload(&ev, "test-exp");
        let attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
            .as_array()
            .unwrap();
        let ttft_attr = attrs.iter().find(|a| a["key"] == "aisix.upstream_ttft_ms");
        assert!(
            ttft_attr.is_some(),
            "aisix.upstream_ttft_ms should be present"
        );
        assert_eq!(ttft_attr.unwrap()["value"]["intValue"], "42");
    }

    #[test]
    fn rfc3339_round_trip() {
        // 2026-05-01T12:00:00Z = 1_777_636_800 unix seconds.
        // (epoch + 56 years + 14 leap days + 120 days into 2026 + 12h)
        let nanos = parse_rfc3339_to_unix_nano("2026-05-01T12:00:00Z").unwrap();
        assert_eq!(nanos, 1_777_636_800 * 1_000_000_000);
    }

    #[test]
    fn rfc3339_with_fractional_seconds() {
        let nanos = parse_rfc3339_to_unix_nano("2026-05-01T12:00:00.5Z").unwrap();
        assert_eq!(nanos, 1_777_636_800 * 1_000_000_000 + 500_000_000);
    }

    #[test]
    fn fan_out_is_a_no_op_on_empty_exporter_list() {
        // Smoke: building the fan-out struct + calling on an empty
        // iterator shouldn't panic and shouldn't spawn tasks. We
        // can't easily count spawned tasks, but if the call returned
        // and the test process didn't hang, we're good.
        let f = OtlpHttpFanOut::new();
        f.fan_out(&sample_event(), None, None, None, std::iter::empty());
    }

    #[test]
    fn disabled_exporter_is_skipped() {
        // Build a disabled exporter with a deliberately bogus
        // endpoint; if the fan-out tried to POST to it the spawned
        // task would log a warning, but never panic. We can't easily
        // assert "no task was spawned" without instrumentation;
        // contenting ourselves with "doesn't crash" + the
        // production code path's `if !exp.enabled { continue }`.
        let mut exp = sample_exporter();
        exp.enabled = false;
        let f = OtlpHttpFanOut::new();
        f.fan_out(&sample_event(), None, None, None, std::iter::once(&exp));
    }

    #[tokio::test]
    async fn fan_out_sampling_zero_skips_only_that_exporter() {
        // Two otlp exporters on one fan-out: `sampled-out` at rate 0.0 and
        // `control` with the rate absent (= 1.0). One event must reach the
        // control's receiver while the rate-0 exporter enqueues nothing —
        // its pipeline is never even created.
        let zero_srv = wiremock::MockServer::builder().start().await;
        let ctrl_srv = wiremock::MockServer::builder().start().await;
        for srv in [&zero_srv, &ctrl_srv] {
            wiremock::Mock::given(wiremock::matchers::method("POST"))
                .respond_with(wiremock::ResponseTemplate::new(200))
                .mount(srv)
                .await;
        }
        let mk = |name: &str, uri: &str, rate: Option<f64>| -> ObservabilityExporter {
            let mut v = serde_json::json!({
                "name": name,
                "enabled": true,
                "kind": "otlp_http",
                "endpoint": format!("{uri}/v1/traces"),
            });
            if let Some(r) = rate {
                v["sample_rate"] = serde_json::json!(r);
            }
            serde_json::from_value(v).unwrap()
        };
        let zero = mk("sampled-out", &zero_srv.uri(), Some(0.0));
        let control = mk("control", &ctrl_srv.uri(), None);

        let f = OtlpHttpFanOut::new();
        f.fan_out(&sample_event(), None, None, None, [&zero, &control]);

        // The control's span lands after the 1s flush…
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if !ctrl_srv.received_requests().await.unwrap().is_empty() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "control exporter saw no OTLP POST within 5s"
            );
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        // …while the rate-0 exporter never even created a pipeline (stats
        // read before shutdown, which tears pipelines down)…
        assert!(!f.exporter_stats().contains_key("sampled-out"));
        assert!(f.exporter_stats().contains_key("control"));
        f.shutdown().await;

        // …and delivered nothing: shutdown drained every pipeline, so
        // anything enqueued would have been flushed by now.
        assert!(zero_srv.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn fan_out_delivers_a_span_to_a_real_receiver() {
        // The new fan-out enqueues into a per-exporter pipeline (1s flush);
        // a single request's span lands at the receiver after the flush.
        let server = wiremock::MockServer::builder().start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/traces"))
            .respond_with(wiremock::ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let exp: ObservabilityExporter = serde_json::from_value(serde_json::json!({
            "name": "real-otlp",
            "enabled": true,
            "kind": "otlp_http",
            "endpoint": format!("{}/v1/traces", server.uri()),
            "headers": {}
        }))
        .unwrap();

        let f = OtlpHttpFanOut::new();
        f.fan_out(&sample_event(), None, None, None, std::iter::once(&exp));

        // Poll for the batched POST (flush_interval is 1s).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if !server.received_requests().await.unwrap().is_empty() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no OTLP POST within 5s"
            );
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        f.shutdown().await;
    }

    /// #954: publishing a snapshot that deletes one exporter and disables
    /// another drops both from the delivery-health report — the view the
    /// managed-mode heartbeat sends the control plane — with no traffic in
    /// between, while the exporter that is still configured keeps its
    /// pipeline.
    #[tokio::test]
    async fn config_publish_reaps_deleted_and_disabled_exporters() {
        let receiver = wiremock::MockServer::builder().start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(200))
            .mount(&receiver)
            .await;
        let endpoint = format!("{}/v1/traces", receiver.uri());
        let exporter = |name: &str, enabled: bool| -> ObservabilityExporter {
            serde_json::from_value(serde_json::json!({
                "name": name,
                "enabled": enabled,
                "kind": "otlp_http",
                "endpoint": endpoint,
            }))
            .unwrap()
        };
        let snapshot_with = |rows: &[(&str, bool)]| {
            let snap = AisixSnapshot::new();
            for (name, enabled) in rows {
                snap.observability_exporters
                    .insert(aisix_core::ResourceEntry::new(
                        format!("id-{name}"),
                        exporter(name, *enabled),
                        1,
                    ));
            }
            snap
        };
        let handle = SnapshotHandle::new(snapshot_with(&[
            ("keep", true),
            ("gone", true),
            ("off", true),
        ]));
        let f = OtlpHttpFanOut::new();
        let live = handle.load().observability_exporters.entries();
        f.fan_out(
            &sample_event(),
            None,
            None,
            None,
            live.iter().map(|e| &e.value),
        );
        let mut names: Vec<_> = f.exporter_stats().into_keys().collect();
        names.sort();
        assert_eq!(names, ["gone", "keep", "off"]);

        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let reaper = tokio::spawn(f.clone().reap_on_config_change(handle.clone(), cancel_rx));
        handle.store(snapshot_with(&[("keep", true), ("off", false)]));

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while f.exporter_stats().len() != 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "deleted/disabled exporters still reported: {:?}",
                f.exporter_stats().keys().collect::<Vec<_>>()
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(f.exporter_stats().contains_key("keep"));

        // A request that loaded the snapshot before the publish emits after
        // the reap: the removed exporters' pipelines stay stopped.
        f.fan_out(
            &sample_event(),
            None,
            None,
            None,
            live.iter().map(|e| &e.value),
        );
        assert_eq!(
            f.exporter_stats().into_keys().collect::<Vec<_>>(),
            ["keep"],
            "a stale snapshot restarted a reaped pipeline"
        );

        cancel_tx.send(true).unwrap();
        reaper.await.unwrap();
        f.shutdown().await;
    }
}
