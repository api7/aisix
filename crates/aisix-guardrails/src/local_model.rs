//! Local CPU embedding-model semantic guardrail — the runtime behind
//! `kind: "semantic"` (AISIX-Cloud#1331 → #1363).
//!
//! Implements the three-layer pipeline per user-configured category
//! (`aisix_core::models::SemanticCategory`). Pipeline per text segment,
//! per category:
//!   1. the category's candidate regexes find spans with exact byte
//!      offsets — the layer is deliberately broad, precision lives in
//!      ② and ③;
//!   2. rule scoring ([`rules`]): hotword-group proximity co-occurrence
//!      raises a candidate's score, negative patterns lower it, and a
//!      double threshold resolves decisive candidates right here — high
//!      scores rewrite and low scores release WITHOUT a model call; only
//!      the uncertain band continues;
//!   3. a context window around each remaining candidate
//!      (±[`WINDOW_CONTEXT_CHARS`] chars — the keyword-proximity window
//!      magnitude mainstream DLP engines use, typically 50–300 chars) is
//!      embedded by the local model and its cosine similarity against
//!      the category DESCRIPTION's embedding is gated by the category
//!      threshold; above-threshold candidates are rewritten in place to
//!      the category's replacement.
//!
//! Everything not rewritten is returned byte-identical. Categories
//! compose in config order: each category moderates the text the
//! previous one produced (the same composition rule as chain members).
//!
//! The verdict is always `Allow` — this guardrail rewrites, never blocks
//! (the design issue's "只改写不阻断" hard constraint), and every
//! resource cap degrades to doing less, never to blocking or stalling.
//!
//! Wire-in: [`SemanticGuardrail`] rows implement the async
//! segment-moderation hooks ([`Guardrail::moderate_input_segments`] /
//! [`Guardrail::moderate_output_segments`]) — the same mask write-back
//! channel the Bedrock ANONYMIZE pass uses — so the proxy's existing
//! collect→moderate→apply walkers do the per-field rewrite. The sync
//! `redact_*_text` hooks stay unimplemented: inference must not run
//! inline on a tokio worker.
//!
//! Ownership: ONE process-wide [`SemanticRuntime`] (model bundle,
//! session lanes, embedding cache) is verified at boot and shared by
//! every `kind: "semantic"` row the chain builder compiles. The heavy
//! state is lazy: boot only verifies the bundle's `manifest.json`
//! (per-file sha256); ONNX sessions load on the first request that needs
//! an embedding, and each category's description prototype is embedded
//! on its first model-band judgement and cached by content, so a
//! config-only change (say, a threshold tune) rebuilds the chain without
//! re-embedding anything. A node with no valid bundle simply lacks the
//! capability — rows are skipped at build with a warning and the
//! heartbeat stops advertising `semantic`.
//!
//! Threading (inherited from the #1271 assessment): inference runs in
//! `spawn_blocking`, bounded by a semaphore sized to the session pool —
//! the configured LANES (`GUARDRAIL_LOCAL_MODEL_LANES`, default 1;
//! api7/aisix#1001). Each session runs with ONE intra-op thread and
//! intra/inter-op **spinning disabled** — ONNX Runtime's default
//! spin-wait burns ~9.4% of a core while completely idle, taxing
//! deployments that never send guardrail traffic. The request's async
//! worker never blocks: it awaits the JoinHandle and keeps serving its
//! other connections; only blocking-pool threads compute. Those threads
//! are NOT core-pinned — on a saturated host they compete with the
//! serving workers for scheduling; hard business/model core
//! partitioning needs a dedicated pinned inference pool.
//!
//! Scaling notes (measured on a 12-core avx2+vnni host):
//! - One lane sustains ~50 inferences/s (p50 ≈ 19 ms per ~35-token
//!   window; roughly linear in tokens).
//! - Throughput scales by adding lanes (api7/aisix#1001):
//!   `GUARDRAIL_LOCAL_MODEL_LANES` = N sessions behind
//!   `Semaphore::new(N)`. Each lane pays its own weight copy —
//!   measured: the first lane costs ~192 MiB resident, each additional
//!   lane ~102 MiB. Lane dispatch is a centralized free-list —
//!   deliberately NO worker↔session binding.
//! - A window embedding is category-agnostic in principle, but v1 keeps
//!   the per-category loop simple: overlapping windows across categories
//!   may embed twice, bounded by the per-pass call cap.
//! - The candidate-regex layer is the all-traffic cost. At many
//!   categories the per-category patterns should merge into one
//!   multi-pattern automaton compiled at build time — tracked on the
//!   design issue. The rust regex engine is non-backtracking, so
//!   operator-supplied patterns cannot ReDoS the data plane.
//!
//! Model contract: the model directory holds `model.onnx` +
//! `tokenizer.json` + `manifest.json`. The manifest pins each file's
//! sha256, the embedding dimension, and the calibrated default
//! threshold; a bundle that fails verification is refused rather than
//! silently mis-scoring (a wrong-model prototype space invalidates every
//! calibrated threshold). The reference bundle is
//! `ibm-granite/granite-embedding-97m-multilingual-r2`'s official int8
//! ONNX export (`onnx/model_quint8_avx2.onnx`, standard `ai.onnx` opset
//! only). Per the repo's `1_Pooling/config.json` + `modules.json`,
//! sentence embedding = CLS pooling over `last_hidden_state` followed by
//! L2 normalization; the graph takes `input_ids` + `attention_mask`.
//! <https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2>
//!
//! Offline builds: `ort-sys` honors `ORT_OFFLINE=1` (skip the prebuilt
//! ONNX Runtime download) with `ORT_LIB_PATH` pointing at a pre-fetched
//! library (see `ort-sys` `build/vars.rs`).

#[cfg(test)]
mod adversarial_corpus;
mod rules;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use ort::session::Session;
use ort::value::Tensor;
use regex::Regex;
use sha2::Digest;

use aisix_core::models::{GuardrailHookPoint, GuardrailMetricsSink, SemanticCategory};

use crate::{
    Guardrail, GuardrailVerdict, SegmentsOutcome, StreamOutputPolicy,
    DEFAULT_STREAM_OUTPUT_BUFFER_BYTES,
};

use rules::{RuleDecision, RuleScorer};

/// Environment variable overriding the model-bundle directory
/// (`model.onnx` + `tokenizer.json` + `manifest.json`). Unset → the
/// image's bundled default path ([`DEFAULT_MODEL_DIR`]). Set, the
/// bundle MUST verify — a failure is boot-fatal (an operator override
/// that silently isn't there would leak the very content it exists to
/// rewrite), whereas a missing/corrupt default bundle only marks the
/// capability failed.
pub const MODEL_DIR_ENV: &str = "GUARDRAIL_LOCAL_MODEL_DIR";
/// Optional inference-lane count (default 1, clamped to
/// [`MAX_LANES`]). Each lane is one ONNX session — one more core the
/// guardrail may use and roughly one more ~100 MiB weight copy
/// resident (measured); see the module scaling notes.
pub const LANES_ENV: &str = "GUARDRAIL_LOCAL_MODEL_LANES";
/// Retired MVP env var: the score gate is per-category config now
/// (`categories[].threshold`). Set → warned about and ignored.
pub const THRESHOLD_ENV: &str = "GUARDRAIL_LOCAL_MODEL_THRESHOLD";
/// Retired MVP env var: the prototype strategy is derived from the
/// category data (v1: description only). Set → warned about and ignored.
pub const PROTOTYPES_ENV: &str = "GUARDRAIL_LOCAL_MODEL_PROTOTYPES";
/// Retired MVP env var: the rule proximity window is a fixed default
/// pending per-category demand. Set → warned about and ignored.
pub const RULE_WINDOW_ENV: &str = "GUARDRAIL_LOCAL_MODEL_RULE_WINDOW";

/// The image's bundled model path — the Dockerfile bakes the bundle
/// here, so a standard-image node has the capability out of the box.
pub const DEFAULT_MODEL_DIR: &str = "/usr/local/aisix/guardrail-model";

/// Upper clamp for [`LANES_ENV`]: lanes are cores, and no sane host
/// grants the guardrail more than this.
const MAX_LANES: usize = 32;

/// Context chars kept on each side of a candidate when cutting the
/// window the model judges.
const WINDOW_CONTEXT_CHARS: usize = 50;

/// Hard cap on model invocations per moderation pass, SHARED across all
/// categories of a row. Candidates past the cap are left untouched and
/// a warning is logged — degrade to doing less, never to blocking or
/// stalling.
const MAX_MODEL_CALLS_PER_PASS: usize = 8;

/// Hard cap on rule-scored candidates per SEGMENT per category. Rule
/// scoring is µs-cheap per candidate but re-scans a proximity window
/// each time, so a crafted body that is nothing but candidates turns
/// the scoring loop into a linear CPU amplifier on the async worker
/// (measured ~91 ms/MiB at the default window). Candidates past the cap
/// are RELEASED unscored with a warning.
const MAX_RULE_SCORED_SPANS_PER_SEGMENT: usize = 4096;

/// Hard byte cap on a single candidate span. A real sensitive value is
/// short by definition; without this cap a crafted run matches as ONE
/// giant span and each model call becomes a max-truncation inference
/// stalling the lane (audit finding on PR #999). Over-cap spans are
/// dropped BEFORE budget accounting.
const MAX_CANDIDATE_SPAN_BYTES: usize = 64;

/// How long a pass waits for an inference lane before degrading that
/// span to the rule layers. Bounds the guardrail's latency contribution
/// under lane saturation — the queue never stalls a request.
const LANE_WAIT_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Debug, thiserror::Error)]
pub enum LocalModelError {
    #[error("local-model guardrail: tokenizer: {0}")]
    Tokenizer(String),
    #[error("local-model guardrail: onnx runtime: {0}")]
    Ort(#[from] ort::Error),
    #[error("local-model guardrail: {0}")]
    Model(String),
    #[error("local-model guardrail: manifest: {0}")]
    Manifest(String),
}

/// Reasons a span degrades to the rule layers instead of getting its
/// embedding judgement. Bounded vocabulary — these land on a metric
/// label (`aisix_guardrail_semantic_degraded_total{reason}`), never
/// free text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticDegradeReason {
    /// The ONNX engine failed to initialize (bad bundle discovered at
    /// first use, or the dimension probe disagreed with the manifest).
    EngineFailed,
    /// The category's description prototype could not be embedded.
    PrototypeUnavailable,
    /// The per-pass model-call budget was exhausted.
    BudgetExhausted,
    /// No inference lane freed up within [`LANE_WAIT_TIMEOUT`].
    QueueTimeout,
    /// A single inference failed.
    InferenceFailed,
}

impl SemanticDegradeReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EngineFailed => "engine_failed",
            Self::PrototypeUnavailable => "prototype_unavailable",
            Self::BudgetExhausted => "budget_exhausted",
            Self::QueueTimeout => "queue_timeout",
            Self::InferenceFailed => "inference_failed",
        }
    }
}

// ─── Manifest ────────────────────────────────────────────────────────────────

/// `manifest.json` in the model directory: the version-consistency unit
/// binding the model files, the embedding dimension, and the calibrated
/// default threshold together (spec §5.5 of AISIX-Cloud#1363). The
/// bundle is refused when any part disagrees — masking with a wrong
/// model/threshold pairing silently mis-scores, which is worse than not
/// masking at all.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelManifest {
    /// Manifest format version; this build understands version 1.
    pub manifest_version: u32,
    /// Upstream model identity, e.g.
    /// `ibm-granite/granite-embedding-97m-multilingual-r2`.
    pub model_id: String,
    /// Upstream revision (commit hash) the files were fetched from.
    #[serde(default)]
    pub revision: Option<String>,
    /// The embedding dimension the engine must produce; probed on the
    /// first inference.
    pub embedding_dim: usize,
    /// File name → `sha256:<hex>` for every file in the bundle.
    pub files: BTreeMap<String, String>,
    /// Calibrated default thresholds for this model.
    pub calibration: ManifestCalibration,
}

/// The calibrated default score gates that ship with a model bundle.
/// Unknown fields are tolerated so a future strategy's calibration can
/// ride the same manifest.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ManifestCalibration {
    /// Default absolute-cosine gate for the description strategy (the
    /// v1 judgement form). Categories without their own `threshold` use
    /// this.
    pub description: f32,
}

impl ModelManifest {
    /// Parse `manifest.json` under `dir` and verify every listed file's
    /// sha256. Blocking (hashes ~120 MB) — call off the async runtime.
    pub fn load_and_verify(dir: &Path) -> Result<Self, LocalModelError> {
        let path = dir.join("manifest.json");
        let raw = std::fs::read(&path)
            .map_err(|e| LocalModelError::Manifest(format!("{}: {e}", path.display())))?;
        let manifest: ModelManifest = serde_json::from_slice(&raw)
            .map_err(|e| LocalModelError::Manifest(format!("{}: {e}", path.display())))?;
        if manifest.manifest_version != 1 {
            return Err(LocalModelError::Manifest(format!(
                "unsupported manifest_version {} (this build understands 1)",
                manifest.manifest_version
            )));
        }
        for required in ["model.onnx", "tokenizer.json"] {
            if !manifest.files.contains_key(required) {
                return Err(LocalModelError::Manifest(format!(
                    "manifest lists no {required}"
                )));
            }
        }
        if manifest.embedding_dim == 0 {
            return Err(LocalModelError::Manifest("embedding_dim is 0".into()));
        }
        if !manifest.calibration.description.is_finite() {
            return Err(LocalModelError::Manifest(
                "calibration.description is not finite".into(),
            ));
        }
        for (name, want) in &manifest.files {
            // File names come from the manifest we just read; refuse
            // anything that could escape the bundle directory.
            if name.contains('/') || name.contains('\\') || name.starts_with('.') {
                return Err(LocalModelError::Manifest(format!(
                    "invalid file name {name:?}"
                )));
            }
            let want_hex = want.strip_prefix("sha256:").ok_or_else(|| {
                LocalModelError::Manifest(format!("{name}: digest must be sha256:<hex>"))
            })?;
            let path = dir.join(name);
            let bytes = std::fs::read(&path)
                .map_err(|e| LocalModelError::Manifest(format!("{}: {e}", path.display())))?;
            let got_hex = hex_digest(&bytes);
            if !got_hex.eq_ignore_ascii_case(want_hex) {
                return Err(LocalModelError::Manifest(format!(
                    "{name}: sha256 mismatch (manifest {want_hex}, file {got_hex})"
                )));
            }
        }
        Ok(manifest)
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let digest = sha2::Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// [`LANES_ENV`] parse rule: default 1 when unset or malformed (zero
/// included — a zero-lane guardrail is a misconfiguration, not a
/// disable switch), clamped to [`MAX_LANES`].
pub fn parse_lanes(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1)
        .min(MAX_LANES)
}

// ─── Blocking inference core ─────────────────────────────────────────────────

/// One shared tokenizer (`encode` takes `&self` and is thread-safe) + a
/// pool of ONNX sessions — the inference LANES (api7/aisix#1001). Each
/// session sits behind its own mutex (`Session::run` takes `&mut`); the
/// engine's semaphore admits at most `sessions.len()` concurrent
/// inferences, so an admitted task always finds a free session.
struct Embedder {
    tokenizer: tokenizers::Tokenizer,
    sessions: Vec<Mutex<Session>>,
}

impl Embedder {
    fn load(dir: &Path, lanes: usize) -> Result<Self, LocalModelError> {
        let lanes = lanes.max(1);
        let tokenizer_path = dir.join("tokenizer.json");
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            LocalModelError::Tokenizer(format!("{}: {e}", tokenizer_path.display()))
        })?;
        // Enforce truncation here instead of trusting the operator's
        // tokenizer.json (the reference export allows 32K tokens): a
        // window is ~100 chars by construction, so 512 tokens
        // (`TruncationParams` default) is pure headroom — this is the
        // second bound, after the candidate-span byte cap, that keeps a
        // single inference's cost fixed no matter what the caller sends.
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams::default()))
            .map_err(|e| LocalModelError::Tokenizer(format!("truncation config: {e}")))?;
        let model_path = dir.join("model.onnx");
        // One intra-op thread per session and NO spin-wait: throughput
        // scales by LANES (each lane single-threaded is the best
        // per-core efficiency for short windows), and the guardrail
        // must not tax a gateway that isn't sending it traffic (see
        // module doc).
        let build = |b: ort::session::builder::SessionBuilder| {
            b.with_intra_threads(1)?
                .with_inter_threads(1)?
                .with_intra_op_spinning(false)?
                .with_inter_op_spinning(false)
        };
        let mut sessions = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            let session = build(Session::builder()?)
                .map_err(|e| LocalModelError::Model(format!("session options: {e}")))?
                .commit_from_file(&model_path)?;
            sessions.push(Mutex::new(session));
        }
        Ok(Self {
            tokenizer,
            sessions,
        })
    }

    /// Take an idle session from the pool. The caller holds a semaphore
    /// permit and permits == sessions, so a free lane exists whenever
    /// this runs; the final blocking `lock()` is a defensive fallback
    /// that cannot deadlock (some lane always releases). A panic in a
    /// previous run poisons that lane's mutex; recover instead of
    /// panicking forever after — `Session::run` is stateless, so the
    /// session itself is unharmed.
    fn acquire_free_session(&self) -> std::sync::MutexGuard<'_, Session> {
        for lane in &self.sessions {
            match lane.try_lock() {
                Ok(guard) => return guard,
                Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                    return poisoned.into_inner();
                }
                Err(std::sync::TryLockError::WouldBlock) => continue,
            }
        }
        self.sessions[0]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Embed one text: encode (with the tokenizer's special-token
    /// template), run the graph, CLS-pool `last_hidden_state`, L2
    /// normalize. Blocking — call from `spawn_blocking` only.
    fn embed(&self, text: &str) -> Result<Vec<f32>, LocalModelError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| LocalModelError::Tokenizer(e.to_string()))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&t| i64::from(t)).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&t| i64::from(t))
            .collect();
        let len = ids.len() as i64;

        let started = Instant::now();
        let mut session = self.acquire_free_session();
        let outputs = session.run(ort::inputs! {
            "input_ids" => Tensor::from_array((vec![1, len], ids))?,
            "attention_mask" => Tensor::from_array((vec![1, len], mask))?,
        })?;
        let (shape, data) = outputs
            .get("last_hidden_state")
            .ok_or_else(|| LocalModelError::Model("model has no last_hidden_state output".into()))?
            .try_extract_tensor::<f32>()?;
        // Expected [1, seq, hidden]; CLS pooling = the first hidden-size
        // slice (the template's leading special token).
        if shape.len() != 3 {
            return Err(LocalModelError::Model(format!(
                "last_hidden_state has rank {} (expected 3)",
                shape.len()
            )));
        }
        let hidden = shape[2] as usize;
        let mut cls = data
            .get(..hidden)
            .ok_or_else(|| LocalModelError::Model("empty last_hidden_state".into()))?
            .to_vec();
        let norm = cls.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut cls {
                *v /= norm;
            }
        }
        tracing::debug!(
            infer_us = started.elapsed().as_micros() as u64,
            tokens = encoding.get_ids().len(),
            "local-model guardrail inference"
        );
        Ok(cls)
    }
}

/// Dot product of two L2-normalized vectors = cosine similarity.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Max cosine over a prototype set (nearest prototype decides). Kept for
/// the calibration probes; the v1 runtime judges against a single
/// description prototype.
#[cfg(test)]
fn max_cosine(prototypes: &[Vec<f32>], v: &[f32]) -> f32 {
    prototypes
        .iter()
        .map(|p| cosine(p, v))
        .fold(f32::NEG_INFINITY, f32::max)
}

// ─── The process-wide runtime ────────────────────────────────────────────────

/// Boot-time capability state, shared with the heartbeat so the
/// advertised `supported_guardrail_kinds` tracks reality (spec §5.4):
/// `semantic` is advertised while the bundle stays verified and the
/// engine has not failed.
#[derive(Debug)]
pub struct SemanticCapability {
    /// 0 = verified (advertise), 1 = failed (stop advertising).
    state: AtomicU8,
}

impl SemanticCapability {
    fn verified() -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU8::new(0),
        })
    }

    fn set_failed(&self) {
        self.state.store(1, Ordering::Relaxed);
    }

    /// `true` while the node can serve `kind: "semantic"`.
    pub fn is_ready(&self) -> bool {
        self.state.load(Ordering::Relaxed) == 0
    }
}

/// The loaded engine: sessions + the lane semaphore.
struct Engine {
    embedder: Arc<Embedder>,
    /// Bounds in-flight `spawn_blocking` inference tasks. Sized to the
    /// session-pool size: more permits would only queue on the session
    /// mutexes from inside blocking threads.
    permits: Arc<tokio::sync::Semaphore>,
}

/// The process-wide semantic-guardrail runtime: one verified model
/// bundle, lazily loaded sessions, and a content-addressed prototype
/// cache shared by every `kind: "semantic"` row.
pub struct SemanticRuntime {
    model_dir: PathBuf,
    lanes: usize,
    manifest: ModelManifest,
    /// Lazily initialized engine. `Some(None)` = initialization failed
    /// permanently (until restart) — retrying a multi-second session
    /// load per request would amplify an outage.
    engine: tokio::sync::OnceCell<Option<Arc<Engine>>>,
    /// description text → embedded prototype, so chain rebuilds (config
    /// churn) and repeated rows never re-embed unchanged text. Keyed by
    /// the exact text — the manifest (and thus the model) is fixed for
    /// the process lifetime.
    prototype_cache: Mutex<std::collections::HashMap<String, Arc<Vec<f32>>>>,
    capability: Arc<SemanticCapability>,
    /// Metrics receiver for the semantic-specific series (model calls,
    /// degrades). `None` records nothing (tests).
    sink: Option<Arc<dyn GuardrailMetricsSink>>,
}

impl std::fmt::Debug for SemanticRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticRuntime")
            .field("model_dir", &self.model_dir)
            .field("lanes", &self.lanes)
            .field("model_id", &self.manifest.model_id)
            .finish()
    }
}

impl SemanticRuntime {
    /// Verify the bundle under `model_dir` and construct the runtime.
    /// Blocking (hashes the bundle) — the server bootstrap wraps it in
    /// `spawn_blocking`. Does NOT load ONNX sessions; those load on
    /// first use.
    pub fn load(
        model_dir: PathBuf,
        lanes: usize,
        sink: Option<Arc<dyn GuardrailMetricsSink>>,
    ) -> Result<Self, LocalModelError> {
        let manifest = ModelManifest::load_and_verify(&model_dir)?;
        tracing::info!(
            model_dir = %model_dir.display(),
            model_id = %manifest.model_id,
            revision = manifest.revision.as_deref().unwrap_or("unpinned"),
            embedding_dim = manifest.embedding_dim,
            lanes,
            "semantic guardrail model bundle verified"
        );
        Ok(Self {
            model_dir,
            lanes,
            manifest,
            engine: tokio::sync::OnceCell::new(),
            prototype_cache: Mutex::new(std::collections::HashMap::new()),
            capability: SemanticCapability::verified(),
            sink,
        })
    }

    /// Test-only runtime with no verified bundle: the engine never
    /// initializes, so every model-band judgement degrades to the rule
    /// layers. Lets proxy/unit tests exercise `kind: "semantic"` rows
    /// without the 120 MB model files.
    pub fn for_tests_without_model() -> Self {
        Self {
            model_dir: PathBuf::from("/nonexistent"),
            lanes: 1,
            manifest: ModelManifest {
                manifest_version: 1,
                model_id: "test/none".into(),
                revision: None,
                embedding_dim: 1,
                files: BTreeMap::new(),
                calibration: ManifestCalibration { description: 0.80 },
            },
            engine: tokio::sync::OnceCell::new(),
            prototype_cache: Mutex::new(std::collections::HashMap::new()),
            capability: SemanticCapability::verified(),
            sink: None,
        }
    }

    /// The capability handle the heartbeat reads.
    pub fn capability(&self) -> Arc<SemanticCapability> {
        Arc::clone(&self.capability)
    }

    /// The verified manifest (calibration defaults, model identity).
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    fn record_degrade(&self, reason: SemanticDegradeReason) {
        if let Some(sink) = &self.sink {
            sink.record_semantic_degrade(reason.as_str());
        }
    }

    fn record_model_call(&self) {
        if let Some(sink) = &self.sink {
            sink.record_semantic_model_call();
        }
    }

    /// Drive the engine load to completion. Spawned detached at chain
    /// build time (see `build.rs`): the load happens when a semantic
    /// row appears, and this task is the persistent awaiter that keeps
    /// the `OnceCell` init alive when request-side awaiters give up.
    pub async fn warm_engine(&self) {
        let _ = self.engine().await;
    }

    /// The lazily loaded engine; `None` after a failed initialization
    /// (capability flips to failed, heartbeat stops advertising).
    async fn engine(&self) -> Option<Arc<Engine>> {
        self.engine
            .get_or_init(|| async {
                let dir = self.model_dir.clone();
                let lanes = self.lanes;
                let dim = self.manifest.embedding_dim;
                let loaded = tokio::task::spawn_blocking(move || {
                    let embedder = Embedder::load(&dir, lanes)?;
                    // Dimension probe: one inference proves the graph
                    // produces what the manifest claims — a mismatched
                    // model would otherwise silently mis-score every
                    // judgement against the calibrated thresholds.
                    let probe = embedder.embed("probe")?;
                    if probe.len() != dim {
                        return Err(LocalModelError::Manifest(format!(
                            "model produces {}-dim embeddings, manifest says {dim}",
                            probe.len()
                        )));
                    }
                    Ok::<_, LocalModelError>(embedder)
                })
                .await;
                match loaded {
                    Ok(Ok(embedder)) => {
                        tracing::info!(lanes = self.lanes, "semantic guardrail engine loaded");
                        Some(Arc::new(Engine {
                            embedder: Arc::new(embedder),
                            permits: Arc::new(tokio::sync::Semaphore::new(self.lanes)),
                        }))
                    }
                    Ok(Err(err)) => {
                        tracing::warn!(
                            error = %err,
                            model_dir = %self.model_dir.display(),
                            "semantic guardrail engine failed to load; \
                             semantic rows degrade to rule layers until restart"
                        );
                        self.capability.set_failed();
                        None
                    }
                    Err(join) => {
                        tracing::warn!(
                            error = %join,
                            "semantic guardrail engine load task failed"
                        );
                        self.capability.set_failed();
                        None
                    }
                }
            })
            .await
            .clone()
    }

    /// Embed one text off the async runtime: bounded by the lane
    /// semaphore (with [`LANE_WAIT_TIMEOUT`]), executed in
    /// `spawn_blocking`. The permit MOVES INTO the blocking closure: a
    /// `spawn_blocking` task keeps running when its awaiter is dropped
    /// (client disconnect), so a permit held in the async scope would
    /// release while the session is still busy and repeated
    /// cancellations would pile threads up toward the pool cap (audit
    /// finding on PR #999).
    async fn embed_text(&self, text: String) -> Result<Vec<f32>, SemanticDegradeReason> {
        // The engine wait is bounded like the lane wait: while the
        // detached warm task is still loading the session, a request
        // releases its span after 500ms instead of stalling on the
        // multi-second disk-bound load.
        let engine = match tokio::time::timeout(LANE_WAIT_TIMEOUT, self.engine()).await {
            Ok(Some(engine)) => engine,
            Ok(None) => return Err(SemanticDegradeReason::EngineFailed),
            Err(_) => return Err(SemanticDegradeReason::QueueTimeout),
        };
        let permit = match tokio::time::timeout(
            LANE_WAIT_TIMEOUT,
            Arc::clone(&engine.permits).acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => return Err(SemanticDegradeReason::EngineFailed),
            Err(_) => return Err(SemanticDegradeReason::QueueTimeout),
        };
        let embedder = Arc::clone(&engine.embedder);
        self.record_model_call();
        let joined = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            embedder.embed(&text)
        })
        .await;
        match joined {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(err)) => {
                tracing::warn!(error = %err, "semantic guardrail inference failed");
                Err(SemanticDegradeReason::InferenceFailed)
            }
            Err(join) => {
                tracing::warn!(error = %join, "semantic guardrail inference task join failed");
                Err(SemanticDegradeReason::InferenceFailed)
            }
        }
    }

    /// The cached prototype embedding for `text` (a category
    /// description), embedding it on first use.
    async fn prototype_for(&self, text: &str) -> Result<Arc<Vec<f32>>, SemanticDegradeReason> {
        if let Some(hit) = self
            .prototype_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(text)
            .cloned()
        {
            return Ok(hit);
        }
        let vector = Arc::new(self.embed_text(text.to_owned()).await?);
        self.prototype_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(text.to_owned(), Arc::clone(&vector));
        Ok(vector)
    }
}

// ─── Candidate generation ────────────────────────────────────────────────────

/// Layer ①: a category's compiled candidate generator — the union of
/// its candidate patterns, de-overlapped.
struct CandidateFinder {
    patterns: Vec<Regex>,
}

impl CandidateFinder {
    fn new(patterns: Vec<Regex>) -> Self {
        Self { patterns }
    }

    /// Candidate spans (byte ranges) in `text`, ascending and
    /// non-overlapping. When spans overlap, the EARLIEST start wins and
    /// the LONGER one wins at equal starts (a broader token pattern
    /// starts at-or-before the dotted run inside it, so it beats the
    /// inner span — masking only the digits of `ICADV12.3` would leak
    /// the `ICADV` identity); ties break toward the earlier pattern.
    /// Spans longer than [`MAX_CANDIDATE_SPAN_BYTES`] are not
    /// candidates.
    fn spans(&self, text: &str) -> Vec<Range<usize>> {
        let mut all: Vec<Range<usize>> = Vec::new();
        for re in &self.patterns {
            for m in re.find_iter(text) {
                let r = m.range();
                if r.len() <= MAX_CANDIDATE_SPAN_BYTES {
                    all.push(r);
                }
            }
        }
        // Sort by start ascending, longer first at the same start (the
        // per-pattern lists arrive in order, so equal (start, len) keeps
        // the earlier pattern's span; duplicates collapse in the sweep).
        all.sort_by(|a, b| a.start.cmp(&b.start).then(b.len().cmp(&a.len())));
        let mut spans: Vec<Range<usize>> = Vec::with_capacity(all.len());
        for r in all {
            match spans.last() {
                Some(last) if r.start < last.end => {
                    // Overlap: the earlier span won either by position
                    // (started first) or by length (same start, sorted
                    // longer-first). Drop the newcomer.
                }
                _ => spans.push(r),
            }
        }
        spans
    }
}

/// The context window around `span`: `ctx` chars on each side, snapped
/// to char boundaries, clamped to the text.
fn window_bounds(text: &str, span: &Range<usize>, ctx: usize) -> Range<usize> {
    if ctx == 0 {
        return span.clone();
    }
    let before = &text[..span.start];
    let start = before
        .char_indices()
        .rev()
        .nth(ctx - 1)
        .map_or(0, |(i, _)| i);
    let after = &text[span.end..];
    let end = span.end
        + after
            .char_indices()
            .nth(ctx)
            .map_or(after.len(), |(i, _)| i);
    start..end
}

/// Rewrite `spans` (ascending, non-overlapping) in `text` to
/// `replacement`, right-to-left so earlier offsets stay valid.
fn apply_masks(text: &str, spans: &[Range<usize>], replacement: &str) -> String {
    let mut out = text.to_owned();
    for span in spans.iter().rev() {
        out.replace_range(span.clone(), replacement);
    }
    out
}

/// The default mask token for a category without a configured
/// `replacement`: `[<NAME>_REDACTED]`, name uppercased with
/// non-alphanumerics folded to `_` — the same shape the pii kind uses.
fn default_mask_token(name: &str) -> String {
    let folded: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("[{folded}_REDACTED]")
}

// ─── Compiled categories and the row guardrail ───────────────────────────────

/// Why a category failed to compile. The chain builder maps these onto
/// its own error type; the row is skipped with a warning either way.
#[derive(Debug, thiserror::Error)]
pub enum CategoryCompileError {
    #[error("category {category:?}: invalid {field} regex {pattern:?}: {source}")]
    InvalidRegex {
        category: String,
        field: &'static str,
        pattern: String,
        source: regex::Error,
    },
    #[error("category {category:?}: unsupported action {action:?} (only \"mask\")")]
    UnsupportedAction { category: String, action: String },
    #[error("duplicate category name {name:?}")]
    DuplicateName { name: String },
    #[error("category {category:?}: no candidate patterns")]
    NoCandidatePatterns { category: String },
}

/// One compiled category: patterns and scorer ready to run, plus the
/// lazily embedded description prototype.
struct CompiledCategory {
    name: String,
    description: String,
    replacement: String,
    threshold: f32,
    finder: CandidateFinder,
    rules: RuleScorer,
    /// Embedded description, resolved through the runtime's cache on the
    /// first model-band judgement. `Some(None)` = unavailable (embedding
    /// failed) — model-band spans release until restart resolves it.
    prototype: tokio::sync::OnceCell<Option<Arc<Vec<f32>>>>,
}

/// Compile the categories of a `kind: "semantic"` row. `default_threshold`
/// comes from the runtime's manifest calibration.
fn compile_categories(
    categories: &[SemanticCategory],
    default_threshold: f32,
) -> Result<Vec<CompiledCategory>, CategoryCompileError> {
    let mut seen = std::collections::HashSet::new();
    let mut compiled = Vec::with_capacity(categories.len());
    for cat in categories {
        if !seen.insert(cat.name.clone()) {
            return Err(CategoryCompileError::DuplicateName {
                name: cat.name.clone(),
            });
        }
        if let Some(action) = cat.action.as_deref() {
            if action != "mask" {
                return Err(CategoryCompileError::UnsupportedAction {
                    category: cat.name.clone(),
                    action: action.to_owned(),
                });
            }
        }
        if cat.candidate_patterns.is_empty() {
            return Err(CategoryCompileError::NoCandidatePatterns {
                category: cat.name.clone(),
            });
        }
        let compile = |field: &'static str, pattern: &str| {
            Regex::new(pattern).map_err(|source| CategoryCompileError::InvalidRegex {
                category: cat.name.clone(),
                field,
                pattern: pattern.to_owned(),
                source,
            })
        };
        let mut candidates = Vec::with_capacity(cat.candidate_patterns.len());
        for p in &cat.candidate_patterns {
            candidates.push(compile("candidate_patterns", p)?);
        }
        let mut negatives = Vec::with_capacity(cat.negative_patterns.len());
        for p in &cat.negative_patterns {
            negatives.push((compile("negative_patterns", p)?, p.clone()));
        }
        let groups: Vec<Vec<String>> = cat.hotword_groups.iter().map(|g| g.terms.clone()).collect();
        let rules = RuleScorer::compile(rules::DEFAULT_PROXIMITY_CHARS, &groups, negatives)
            .map_err(|(pattern, source)| CategoryCompileError::InvalidRegex {
                category: cat.name.clone(),
                field: "hotword_groups",
                pattern,
                source,
            })?;
        compiled.push(CompiledCategory {
            name: cat.name.clone(),
            description: cat.description.clone(),
            replacement: cat
                .replacement
                .clone()
                .unwrap_or_else(|| default_mask_token(&cat.name)),
            threshold: cat
                .threshold
                .filter(|t| t.is_finite())
                .unwrap_or(default_threshold),
            finder: CandidateFinder::new(candidates),
            rules,
            prototype: tokio::sync::OnceCell::new(),
        });
    }
    Ok(compiled)
}

/// The runtime guardrail for one `kind: "semantic"` row. Always-`Allow`;
/// masks via the segment hooks; honors the row's `hook_point`.
pub struct SemanticGuardrail {
    runtime: Arc<SemanticRuntime>,
    categories: Vec<CompiledCategory>,
    hook_point: GuardrailHookPoint,
}

impl SemanticGuardrail {
    /// Compile a row's categories against the process runtime.
    pub fn from_config(
        runtime: Arc<SemanticRuntime>,
        categories: &[SemanticCategory],
        hook_point: GuardrailHookPoint,
    ) -> Result<Self, CategoryCompileError> {
        let compiled = compile_categories(categories, runtime.manifest.calibration.description)?;
        Ok(Self {
            runtime,
            categories: compiled,
            hook_point,
        })
    }

    /// Mask one segment with one category. Returns the rewritten text
    /// and how many spans were masked; `budget` is the row's shared
    /// per-pass model-call cap — layer-② decisions are budget-free, so
    /// the cap only meters candidates that reach layer ③. Every failure
    /// arm leaves the span untouched (rewrite less, never block).
    async fn mask_segment_category(
        &self,
        cat: &CompiledCategory,
        text: &str,
        budget: &mut usize,
    ) -> (String, u32) {
        let spans = cat.finder.spans(text);
        if spans.len() > MAX_RULE_SCORED_SPANS_PER_SEGMENT {
            tracing::warn!(
                category = %cat.name,
                candidates = spans.len(),
                cap = MAX_RULE_SCORED_SPANS_PER_SEGMENT,
                "semantic guardrail: candidate cap reached; the tail is released unscored"
            );
        }
        let mut hits: Vec<Range<usize>> = Vec::new();
        let mut over_budget = false;
        for span in spans.into_iter().take(MAX_RULE_SCORED_SPANS_PER_SEGMENT) {
            match cat.rules.decide(text, &span) {
                RuleDecision::Mask => hits.push(span),
                RuleDecision::Pass => {}
                RuleDecision::Model => {
                    if *budget == 0 {
                        if !over_budget {
                            over_budget = true;
                            self.runtime
                                .record_degrade(SemanticDegradeReason::BudgetExhausted);
                            tracing::warn!(
                                category = %cat.name,
                                cap = MAX_MODEL_CALLS_PER_PASS,
                                "semantic guardrail: model-call cap reached; \
                                 uncertain candidates left unmasked"
                            );
                        }
                        continue;
                    }
                    // Only a PERMANENT failure (engine unavailable) may
                    // cache `None`; a transient one (queue timeout under
                    // burst, a one-off inference error) leaves the cell
                    // empty so the next model-band span retries the
                    // ~ms-scale prototype embed instead of degrading the
                    // category until restart.
                    let prototype = cat
                        .prototype
                        .get_or_try_init(|| async {
                            match self.runtime.prototype_for(&cat.description).await {
                                Ok(v) => Ok(Some(v)),
                                Err(reason @ SemanticDegradeReason::EngineFailed) => {
                                    // Record the UNDERLYING cause — that is
                                    // what the operator has to fix.
                                    self.runtime.record_degrade(reason);
                                    tracing::warn!(
                                        category = %cat.name,
                                        reason = reason.as_str(),
                                        "semantic guardrail: description prototype \
                                         unavailable; category degrades to rule layers"
                                    );
                                    Ok(None)
                                }
                                Err(reason) => Err(reason),
                            }
                        })
                        .await;
                    let prototype = match prototype {
                        Ok(Some(prototype)) => Arc::clone(prototype),
                        Ok(None) => {
                            // Permanently-failed category: keep the ongoing
                            // signal alive — every span that would have been
                            // judged is one more degrade sample, not a
                            // one-off at failure time.
                            self.runtime
                                .record_degrade(SemanticDegradeReason::PrototypeUnavailable);
                            continue;
                        }
                        Err(reason) => {
                            self.runtime.record_degrade(reason);
                            continue;
                        }
                    };
                    *budget -= 1;
                    let window = text[window_bounds(text, &span, WINDOW_CONTEXT_CHARS)].to_owned();
                    match self.runtime.embed_text(window).await {
                        Ok(vector) => {
                            let score = cosine(&prototype, &vector);
                            tracing::debug!(
                                category = %cat.name,
                                score,
                                threshold = cat.threshold,
                                "semantic guardrail window judged"
                            );
                            if score >= cat.threshold {
                                hits.push(span);
                            }
                        }
                        Err(reason) => {
                            self.runtime.record_degrade(reason);
                        }
                    }
                }
            }
        }
        if hits.is_empty() {
            (text.to_owned(), 0)
        } else {
            let count = hits.len() as u32;
            (apply_masks(text, &hits, &cat.replacement), count)
        }
    }

    /// One moderation pass over a request's (or response's) text
    /// segments — the shared body of both segment hooks. Categories
    /// compose in config order over each segment.
    async fn moderate(&self, texts: &[String]) -> SegmentsOutcome {
        let mut budget = MAX_MODEL_CALLS_PER_PASS;
        let mut masked: Vec<String> = Vec::with_capacity(texts.len());
        let mut counts: BTreeMap<String, u32> = BTreeMap::new();
        let mut total = 0u32;
        for text in texts {
            let mut current = text.clone();
            for cat in &self.categories {
                let (rewritten, count) =
                    self.mask_segment_category(cat, &current, &mut budget).await;
                if count > 0 {
                    *counts.entry(cat.name.clone()).or_insert(0) += count;
                    total += count;
                    current = rewritten;
                }
            }
            masked.push(current);
        }
        SegmentsOutcome {
            verdict: GuardrailVerdict::Allow,
            masked: (total > 0).then_some(masked),
            counts,
            monitor_hits: Vec::new(),
        }
    }

    fn runs_on(&self, output: bool) -> bool {
        match self.hook_point {
            GuardrailHookPoint::Both => true,
            GuardrailHookPoint::Input => !output,
            GuardrailHookPoint::Output => output,
        }
    }
}

#[async_trait]
impl Guardrail for SemanticGuardrail {
    fn name(&self) -> &'static str {
        "semantic"
    }

    /// Consulted through the segment hooks only (the mask write-back
    /// channel); the plain `check_*` hooks stay default-`Allow`.
    fn moderates_segments(&self) -> bool {
        true
    }

    fn runs_on_output(&self) -> bool {
        self.hook_point != GuardrailHookPoint::Input
    }

    /// Masking a streamed response needs the whole response held back (a
    /// span can cross any chunk boundary) — but past the buffer cap this
    /// guardrail must release UNMASKED, not block: its contract is
    /// "rewrite, never block", so past-cap content degrades to fewer
    /// masks, the same fail-open arm as every other cap here. A chain
    /// member with a stricter policy (e.g. fail-closed pii) still wins
    /// the fold.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        StreamOutputPolicy::BufferFull {
            max_buffer_bytes: DEFAULT_STREAM_OUTPUT_BUFFER_BYTES,
            on_exceeded_fail_open: true,
        }
    }

    async fn moderate_input_segments(&self, texts: &[String]) -> SegmentsOutcome {
        if !self.runs_on(false) {
            return SegmentsOutcome::allow();
        }
        self.moderate(texts).await
    }

    async fn moderate_output_segments(&self, texts: &[String]) -> SegmentsOutcome {
        if !self.runs_on(true) {
            return SegmentsOutcome::allow();
        }
        self.moderate(texts).await
    }
}
