//! Local CPU embedding-model guardrail — the second-tier guardrail
//! (AISIX-Cloud#1331), grown from the MVP vertical slice.
//!
//! Implements the design issue's three-layer pipeline for one hardcoded
//! category (no prototype library resource, no standard risk categories
//! yet). Pipeline per text segment:
//!   1. regex finds candidate spans with exact byte offsets
//!      (dotted number runs, the EDA-version candidate shape);
//!   2. rule scoring ([`rules`]): hotword proximity co-occurrence raises
//!      a candidate's score, negative patterns lower it, and a double
//!      threshold resolves decisive candidates right here — high scores
//!      rewrite and low scores release WITHOUT a model call; only the
//!      uncertain band continues;
//!   3. a context window around each remaining candidate
//!      (±[`WINDOW_CONTEXT_CHARS`] chars — the keyword-proximity window
//!      magnitude mainstream DLP engines use, typically 50–300 chars) is
//!      embedded by the local model and compared, by cosine similarity,
//!      against the category's prototype vector set (encoded at load
//!      time; see [`PrototypeStrategy`]); above-threshold candidates are
//!      rewritten in place to [`MASK_REPLACEMENT`].
//!
//! Everything not rewritten is returned byte-identical.
//!
//! The verdict is always `Allow` — this guardrail rewrites, never blocks
//! (the design issue's "只改写不阻断" hard constraint).
//!
//! Wire-in: implements the async segment-moderation hooks
//! ([`Guardrail::moderate_input_segments`] /
//! [`Guardrail::moderate_output_segments`]) — the same mask write-back
//! channel the Bedrock ANONYMIZE pass uses — so the proxy's existing
//! collect→moderate→apply walkers do the per-field rewrite and no proxy
//! plumbing changes. The sync `redact_*_text` hooks stay unimplemented:
//! inference must not run inline on a tokio worker.
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
//! Scaling notes (MVP review, measured on a 12-core avx2+vnni host):
//! - One lane sustains ~50 inferences/s (p50 ≈ 19 ms per ~35-token
//!   window; roughly linear in tokens). The acceptance shape spends TWO
//!   inferences per request (input + output pass).
//! - Throughput scales by adding lanes (api7/aisix#1001, implemented):
//!   `GUARDRAIL_LOCAL_MODEL_LANES` = N sessions behind
//!   `Semaphore::new(N)`. `run(&mut self)` forbids concurrent runs on
//!   one session even though the ONNX Runtime C API documents `Run` as
//!   thread-safe with shared read-only weights; this crate forbids
//!   `unsafe`, so the shared-weights form waits on an upstream `&self`
//!   run, a thin unsafe shim crate, or the sidecar deployment form.
//!   Until then each lane pays its own weight copy — measured: the
//!   first lane costs ~192 MiB resident (weights + tokenizer + arena),
//!   each additional lane ~102 MiB (its weight copy + arena).
//!   Lane dispatch is a centralized free-list — deliberately NO
//!   worker↔session binding (sessions are interchangeable; the central
//!   queue load-balances the uneven per-worker accept distribution).
//!   ONE ORT `Session` is a loaded model instance, not a conversation:
//!   every `run` is stateless, so lanes are freely interchangeable.
//! - Per-audit cost ≈ (#candidate windows × inference) + (#prototypes ×
//!   384-dim dot ≈ 1 µs). A window embedding is category-agnostic: a
//!   span matched by several categories embeds ONCE and compares
//!   against the whole prototype library, so prototype count stays
//!   latency-noise up to ~1e5 vectors (then: ANN index).
//! - The candidate-regex layer is the all-traffic cost. At many
//!   categories the per-category patterns must merge into one
//!   multi-pattern automaton (`RegexSet` / aho-corasick), compiled at
//!   prototype-library build time and atomically swapped — never per
//!   request. The rust regex engine is non-backtracking, so
//!   operator-supplied patterns cannot ReDoS the data plane.
//!
//! Model contract (upstream docs): the model directory holds the two
//! deliverables `model.onnx` + `tokenizer.json`. The MVP target is
//! `ibm-granite/granite-embedding-97m-multilingual-r2`'s official int8
//! ONNX export (`onnx/model_quint8_avx2.onnx`, 93.7 MiB, standard
//! `ai.onnx` opset only — the q4/q4f16/bnb4 variants carry
//! `com.microsoft` ops and are ruled out). Per the repo's
//! `1_Pooling/config.json` + `modules.json`, sentence embedding = CLS
//! pooling over `last_hidden_state` followed by L2 normalization; the
//! ModernBERT graph takes `input_ids` + `attention_mask` (no
//! `token_type_ids`).
//! <https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2>
//!
//! Offline builds: `ort-sys` honors `ORT_OFFLINE=1` (skip the prebuilt
//! ONNX Runtime download) with `ORT_LIB_PATH` pointing at a pre-fetched
//! library (see `ort-sys` `build/vars.rs`).

mod rules;

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use ort::session::Session;
use ort::value::Tensor;
use regex::Regex;

use crate::{
    Guardrail, GuardrailVerdict, SegmentsOutcome, StreamOutputPolicy,
    DEFAULT_STREAM_OUTPUT_BUFFER_BYTES,
};

use rules::{RuleDecision, RuleScorer};

/// Environment variable holding the model directory (`model.onnx` +
/// `tokenizer.json`). Set → the server bootstrap loads and injects the
/// guardrail; unset → the feature is completely inert.
pub const MODEL_DIR_ENV: &str = "GUARDRAIL_LOCAL_MODEL_DIR";
/// Optional cosine-similarity gate override (default: the configured
/// strategy's calibrated `default_threshold`).
pub const THRESHOLD_ENV: &str = "GUARDRAIL_LOCAL_MODEL_THRESHOLD";
/// Optional inference-lane count (default 1, clamped to
/// [`MAX_LANES`]). Each lane is one ONNX session — one more core the
/// guardrail may use and roughly one more ~100 MiB weight copy resident (measured); see
/// the module scaling notes and api7/aisix#1001.
pub const LANES_ENV: &str = "GUARDRAIL_LOCAL_MODEL_LANES";
/// Optional layer-② hotword proximity window override, in chars each
/// side of a candidate (default [`rules::DEFAULT_PROXIMITY_CHARS`],
/// clamped to [`rules::MAX_PROXIMITY_CHARS`]; malformed or zero →
/// default). Zero is a misconfiguration, not a mode — the same rule as
/// [`LANES_ENV`]: a zero window finds no hotword, so layer ② silently
/// stops masking while looking configured.
pub const RULE_WINDOW_ENV: &str = "GUARDRAIL_LOCAL_MODEL_RULE_WINDOW";
/// Optional layer-③ prototype strategy: `description`, `max`, or
/// `centroid` (default [`PrototypeStrategy::default`]; malformed →
/// default with a warning — silently landing on the wrong vector space
/// would invalidate the operator's calibrated threshold).
pub const PROTOTYPES_ENV: &str = "GUARDRAIL_LOCAL_MODEL_PROTOTYPES";

/// Upper clamp for [`LANES_ENV`]: lanes are cores, and no sane host
/// grants the guardrail more than this.
const MAX_LANES: usize = 32;

/// The ONE category this module ships: EDA-software version numbers.
/// Under [`PrototypeStrategy::Description`] the prototype vector is the
/// load-time embedding of this sentence — the v1 "customer types one
/// Chinese description" path from the design issue, collapsed to a
/// compile-time constant.
const PROTOTYPE_DESCRIPTION_ZH: &str = "EDA 软件的版本号";

/// Sample sentences for the sample-based prototype strategies — the v2
/// "customer supplies example sentences" path from the design issue,
/// collapsed to a compile-time constant set (synthesized; real customer
/// corpus not yet available). Coverage is by SHAPE, not by string: the
/// upgrade/rollback phrasing and the tool-name+version phrasing that the
/// single description prototype measurably missed, in Chinese and
/// English. Tool names and numbers are deliberately DIFFERENT from the
/// probe corpus (Spectre/Xcelium here, Virtuoso in the probes) so the
/// calibration probes measure shape generalization, not string overlap.
const PROTOTYPE_SAMPLES: &[&str] = &[
    "布局布线工具升级到 21.15 之后跑得快多了",
    "仿真器回退到 19.03 才恢复正常",
    "这个后端工具的版本是 33.0",
    "综合工具的版本号是 2020.09,不要外传",
    "Spectre 23.1.0 在这个工艺角下会崩溃",
    "签核工具装的是 22.4 这个版本",
    "We upgraded the place-and-route tool to 21.15",
    "The simulator crashed on release 6.2.1",
    "Xcelium 23.09 fails on this testbench",
    "The sign-off tool version is 2020.09",
];

/// How the category's prototype vector set is built at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrototypeStrategy {
    /// One vector: the embedded category description (the MVP form).
    Description,
    /// One vector per [`PROTOTYPE_SAMPLES`] entry; a window scores by its
    /// MAX cosine over the set (nearest sample decides).
    SampleMax,
    /// One vector: the L2-renormalized mean of the sample embeddings; a
    /// window scores against the class centroid.
    SampleCentroid,
}

impl PrototypeStrategy {
    fn parse(raw: Option<&str>) -> Self {
        let Some(raw) = raw else {
            return Self::default();
        };
        match raw.to_ascii_lowercase().as_str() {
            "description" => Self::Description,
            "max" => Self::SampleMax,
            "centroid" => Self::SampleCentroid,
            other => {
                tracing::warn!(
                    value = other,
                    default = ?Self::default(),
                    "unrecognized {PROTOTYPES_ENV}; using the default strategy"
                );
                Self::default()
            }
        }
    }

    /// Cosine gate calibrated per strategy with this module's
    /// `#[ignore]` probe matrix (cosine absolute scale shifts with the
    /// prototype construction, so one shared default would be wrong for
    /// two of the three). Measured bands (granite-97m int8):
    /// - `Description` keeps the MVP calibration: acceptance positive
    ///   ~0.90, every probed negative ≤0.76, hard positives 0.75–0.79
    ///   below the gate — the measured single-prototype recall gap
    ///   (negative hard margin in all five phrasings swept) that
    ///   layer ② now covers.
    /// - `SampleMax`: hard positives ≥0.8316, negatives ≤0.7867
    ///   (hard margin +0.0449); 0.82 sits precision-leaning in that
    ///   band — 0.033 above the negative ceiling.
    /// - `SampleCentroid`: hard positives ≥0.8616, negatives ≤0.8370
    ///   (hard margin +0.0246); 0.85 likewise.
    ///
    /// All three lean precision — a mask false-positive corrupts user
    /// content; layer ② carries recall for anchored shapes.
    fn default_threshold(self) -> f32 {
        match self {
            Self::Description => 0.80,
            Self::SampleMax => 0.82,
            Self::SampleCentroid => 0.85,
        }
    }
}

impl Default for PrototypeStrategy {
    /// `SampleMax`: the probe matrix (module tests) measures the widest
    /// positive hard margin here (+0.0449 vs +0.0246 for the centroid —
    /// averaging ten shape-diverse samples into one vector costs
    /// nearest-shape resolution).
    fn default() -> Self {
        Self::SampleMax
    }
}

/// Candidate shape: a dotted number run (`12.1`, `2022.4`, `6.1.8`).
/// Plain integers are out of MVP scope.
const CANDIDATE_PATTERN: &str = r"\d+(?:\.\d+)+";

/// Context chars kept on each side of a candidate when cutting the
/// window the model judges.
const WINDOW_CONTEXT_CHARS: usize = 50;

/// Hard cap on model invocations per moderation pass (the design
/// issue's "每请求模型调用次数上限" recommendation). Candidates past
/// the cap are left untouched and a warning is logged — degrade to
/// doing less, never to blocking or stalling.
const MAX_MODEL_CALLS_PER_PASS: usize = 8;

/// Hard cap on rule-scored candidates per SEGMENT. Rule scoring is
/// µs-cheap per candidate but re-scans a proximity window each time, so
/// a crafted body that is nothing but candidates (`1.1 1.1 …`) turns
/// the per-segment scoring loop into a linear CPU amplifier on the
/// async worker — measured ~91 ms of synchronous work per MiB at the
/// default window and ~6× that at the window clamp (audit finding on
/// this PR; the request-body limit defaults to unlimited). Candidates
/// past the cap are RELEASED unscored with a warning — the same
/// fail-open arm as every other cap here — and a segment with thousands
/// of dotted-number runs is a log dump, not prose a version leaks
/// through.
const MAX_RULE_SCORED_SPANS_PER_SEGMENT: usize = 4096;

/// Hardcoded rewrite for an above-threshold candidate span.
const MASK_REPLACEMENT: &str = "***";

/// Detector label used in redaction counts (never the matched value —
/// #153 / #932 no-leak rule).
const DETECTOR_NAME: &str = "eda_version";

#[derive(Debug, thiserror::Error)]
pub enum LocalModelError {
    #[error("local-model guardrail: tokenizer: {0}")]
    Tokenizer(String),
    #[error("local-model guardrail: onnx runtime: {0}")]
    Ort(#[from] ort::Error),
    #[error("local-model guardrail: {0}")]
    Model(String),
}

/// Load-time configuration. MVP surface: env vars only — deliberately
/// NO control-plane resource; see the design issue for what the real
/// config object must eventually carry (model id + dimension + prefix
/// convention + threshold table + prototype-library version).
///
/// Target form (not built): the category data becomes a prototype
/// LIBRARY resource — etcd stores category TEXT only (name, action,
/// replacement, candidate patterns, description/sample sentences); the
/// DP encodes vectors with its own loaded model off the hot path
/// (cached by model id + text hash) and atomically swaps the compiled
/// library on a watch tick, the same propagation the guardrail index
/// already uses — so category updates land without a restart. Vectors
/// never travel through config, which removes the "encoded by which
/// model" mismatch for text-sourced prototypes; swapping the MODEL
/// itself stays a cold operation (all vectors rebuilt).
pub struct LocalModelConfig {
    pub model_dir: PathBuf,
    pub threshold: f32,
    /// Inference lanes = ONNX sessions = max concurrent inferences.
    pub lanes: usize,
    /// Layer-② hotword proximity window (chars each side of a span).
    pub rule_window: usize,
    /// Layer-③ prototype-set construction.
    pub prototypes: PrototypeStrategy,
}

impl LocalModelConfig {
    /// `None` when [`MODEL_DIR_ENV`] is unset (guardrail disabled). A
    /// malformed or out-of-range threshold falls back to the strategy's
    /// default rather than failing boot — the gate is a tuning knob, not
    /// a correctness one. The range check matters: `"NaN"` parses as a
    /// valid f32 and would make `score >= threshold` always false — a
    /// configured-looking guardrail that silently never masks. Lanes and
    /// the rule window follow the same lenient rule (malformed →
    /// default).
    pub fn from_env() -> Option<Self> {
        let model_dir = PathBuf::from(std::env::var_os(MODEL_DIR_ENV)?);
        let prototypes = PrototypeStrategy::parse(std::env::var(PROTOTYPES_ENV).ok().as_deref());
        let threshold = std::env::var(THRESHOLD_ENV)
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|t| t.is_finite() && (0.0..=1.0).contains(t))
            .unwrap_or_else(|| prototypes.default_threshold());
        let lanes = parse_lanes(std::env::var(LANES_ENV).ok().as_deref());
        let rule_window = std::env::var(RULE_WINDOW_ENV)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&w| w >= 1)
            .unwrap_or(rules::DEFAULT_PROXIMITY_CHARS)
            .min(rules::MAX_PROXIMITY_CHARS);
        Some(Self {
            model_dir,
            threshold,
            lanes,
            rule_window,
            prototypes,
        })
    }
}

/// [`LANES_ENV`] parse rule: default 1 when unset or malformed (zero
/// included — a zero-lane guardrail is a misconfiguration, not a
/// disable switch; disabling is unsetting [`MODEL_DIR_ENV`]), clamped
/// to [`MAX_LANES`].
fn parse_lanes(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1)
        .min(MAX_LANES)
}

/// The blocking inference core: one shared tokenizer (`encode` takes
/// `&self` and is thread-safe) + a pool of ONNX sessions — the
/// inference LANES (api7/aisix#1001). Each session sits behind its own
/// mutex (`Session::run` takes `&mut`); the guardrail's semaphore
/// admits at most `sessions.len()` concurrent inferences, so an
/// admitted task always finds a free session. Dispatch is a
/// centralized free-list: any request takes whichever session is idle —
/// deliberately NO worker↔session binding (sessions are stateless and
/// interchangeable; a central queue load-balances the uneven per-worker
/// accept distribution).
struct Embedder {
    tokenizer: tokenizers::Tokenizer,
    sessions: Vec<Mutex<Session>>,
}

impl Embedder {
    fn load(dir: &std::path::Path, lanes: usize) -> Result<Self, LocalModelError> {
        // `parse_lanes` never yields 0, but `LocalModelConfig`'s fields
        // are `pub` and a future programmatic constructor (the
        // control-plane integration) could hand-build one; an empty
        // pool would panic at the prototype embed with a confusing
        // boot error instead of just working.
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
        // module doc). Builder-stage errors carry the builder back for
        // recovery (`ort::Error<SessionBuilder>`); this path never
        // recovers, so they fold to their message.
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
    /// session itself is unharmed, and turning every later inference
    /// into a panic would silently disable the operator's masking until
    /// restart (the exact state load treats as boot-fatal).
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

/// A window's score against the prototype set: max cosine over the set
/// (with one vector this IS plain cosine, so all three strategies score
/// through here).
fn prototype_score(prototypes: &[Vec<f32>], v: &[f32]) -> f32 {
    prototypes
        .iter()
        .map(|p| cosine(p, v))
        .fold(f32::NEG_INFINITY, f32::max)
}

/// L2-renormalized mean of a set of L2-normalized vectors — the
/// [`PrototypeStrategy::SampleCentroid`] construction.
fn centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    let dim = vectors.first().map_or(0, Vec::len);
    let mut mean = vec![0.0f32; dim];
    for v in vectors {
        for (m, x) in mean.iter_mut().zip(v) {
            *m += x;
        }
    }
    let norm = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for m in &mut mean {
            *m /= norm;
        }
    }
    mean
}

/// Hard byte cap on a single candidate span. A real version number is
/// short by definition; without this cap `\d+(?:\.\d+)+` matches an
/// arbitrarily long `1.1.1...` run as ONE span, and since the window is
/// span + context, a crafted request would turn each model call into a
/// max-truncation inference and stall the single lane for everyone
/// (audit finding on PR #999). Over-cap spans are dropped BEFORE the
/// per-pass budget so they cannot starve legitimate candidates either.
const MAX_CANDIDATE_SPAN_BYTES: usize = 64;

/// Candidate spans (byte ranges) in `text`, in order. Spans longer than
/// [`MAX_CANDIDATE_SPAN_BYTES`] are not candidates (see the constant).
fn candidate_spans(re: &Regex, text: &str) -> Vec<Range<usize>> {
    re.find_iter(text)
        .map(|m| m.range())
        .filter(|s| s.len() <= MAX_CANDIDATE_SPAN_BYTES)
        .collect()
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

/// Rewrite `spans` (ascending, non-overlapping — regex `find_iter`
/// order) in `text` to [`MASK_REPLACEMENT`], right-to-left so earlier
/// offsets stay valid.
fn apply_masks(text: &str, spans: &[Range<usize>]) -> String {
    let mut out = text.to_owned();
    for span in spans.iter().rev() {
        out.replace_range(span.clone(), MASK_REPLACEMENT);
    }
    out
}

/// The runtime guardrail. Always-`Allow`; masks via the segment hooks.
pub struct LocalModelGuardrail {
    embedder: Arc<Embedder>,
    /// L2-normalized prototype vector set (see [`PrototypeStrategy`]).
    prototypes: Vec<Vec<f32>>,
    threshold: f32,
    candidate_re: Regex,
    /// Layer-② scorer (hotword proximity + negative patterns).
    rules: RuleScorer,
    /// Bounds in-flight `spawn_blocking` inference tasks. Sized to the
    /// session-pool size (the configured lanes): more permits would
    /// only queue on the session mutexes from inside blocking threads.
    permits: Arc<tokio::sync::Semaphore>,
}

impl LocalModelGuardrail {
    /// Load tokenizer + the session pool and encode the category's
    /// prototype set. Blocking (N model loads + up to
    /// `PROTOTYPE_SAMPLES`-many inferences) — the server bootstrap
    /// wraps it in `spawn_blocking`.
    pub fn load(config: &LocalModelConfig) -> Result<Self, LocalModelError> {
        let started = Instant::now();
        let embedder = Embedder::load(&config.model_dir, config.lanes)?;
        let embed_samples = || {
            PROTOTYPE_SAMPLES
                .iter()
                .map(|s| embedder.embed(s))
                .collect::<Result<Vec<_>, _>>()
        };
        let prototypes = match config.prototypes {
            PrototypeStrategy::Description => vec![embedder.embed(PROTOTYPE_DESCRIPTION_ZH)?],
            PrototypeStrategy::SampleMax => embed_samples()?,
            PrototypeStrategy::SampleCentroid => vec![centroid(&embed_samples()?)],
        };
        tracing::info!(
            model_dir = %config.model_dir.display(),
            threshold = config.threshold,
            lanes = config.lanes,
            rule_window = config.rule_window,
            strategy = ?config.prototypes,
            prototypes = prototypes.len(),
            load_ms = started.elapsed().as_millis() as u64,
            "local-model guardrail loaded (category: EDA software version)"
        );
        Ok(Self {
            embedder: Arc::new(embedder),
            prototypes,
            threshold: config.threshold,
            candidate_re: Regex::new(CANDIDATE_PATTERN).expect("candidate pattern must compile"),
            rules: RuleScorer::new(config.rule_window),
            permits: Arc::new(tokio::sync::Semaphore::new(config.lanes)),
        })
    }

    /// Embed one window off the async runtime: bounded by the
    /// semaphore, executed in `spawn_blocking`. The permit MOVES INTO
    /// the blocking closure: a `spawn_blocking` task keeps running when
    /// its awaiter is dropped (client disconnect), so a permit held in
    /// this async scope would release while the session is still busy —
    /// letting the next request park a second blocking-pool thread on
    /// the session mutex, and repeated cancellations pile threads up
    /// toward the pool cap (audit finding on PR #999). Held by the
    /// closure, the permit releases only when the inference actually
    /// finishes.
    async fn embed_window(&self, window: String) -> Result<Vec<f32>, LocalModelError> {
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .expect("inference semaphore never closed");
        let embedder = Arc::clone(&self.embedder);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            embedder.embed(&window)
        })
        .await
        .map_err(|e| LocalModelError::Model(format!("inference task join: {e}")))?
    }

    /// Mask one segment. Returns the rewritten text and how many spans
    /// were masked; `budget` is the shared per-pass model-call cap —
    /// layer-② decisions are budget-free (µs of regex work), so the cap
    /// only meters candidates that reach layer ③, and an exhausted
    /// budget skips THOSE while later rule-decided candidates still
    /// resolve. Model failure on a candidate leaves that span untouched
    /// (rewrite less, never block — the fail-open arm of "只改写不阻断"
    /// that also answers "model unavailable": the pipeline degrades to
    /// ①+②).
    async fn mask_segment(&self, text: &str, budget: &mut usize) -> (String, u32) {
        let mut hits: Vec<Range<usize>> = Vec::new();
        let (mut rule_masked, mut rule_passed, mut model_judged) = (0u32, 0u32, 0u32);
        let mut over_budget = false;
        let spans = candidate_spans(&self.candidate_re, text);
        if spans.len() > MAX_RULE_SCORED_SPANS_PER_SEGMENT {
            tracing::warn!(
                candidates = spans.len(),
                cap = MAX_RULE_SCORED_SPANS_PER_SEGMENT,
                "local-model guardrail: candidate cap reached; the tail is released unscored"
            );
        }
        for span in spans.into_iter().take(MAX_RULE_SCORED_SPANS_PER_SEGMENT) {
            match self.rules.decide(text, &span) {
                RuleDecision::Mask => {
                    rule_masked += 1;
                    hits.push(span);
                }
                RuleDecision::Pass => rule_passed += 1,
                RuleDecision::Model => {
                    if *budget == 0 {
                        if !over_budget {
                            over_budget = true;
                            tracing::warn!(
                                cap = MAX_MODEL_CALLS_PER_PASS,
                                "local-model guardrail: model-call cap reached; uncertain candidates left unmasked"
                            );
                        }
                        continue;
                    }
                    *budget -= 1;
                    model_judged += 1;
                    let window = text[window_bounds(text, &span, WINDOW_CONTEXT_CHARS)].to_owned();
                    match self.embed_window(window).await {
                        Ok(vector) => {
                            let score = prototype_score(&self.prototypes, &vector);
                            tracing::debug!(
                                score,
                                threshold = self.threshold,
                                "local-model window judged"
                            );
                            if score >= self.threshold {
                                hits.push(span);
                            }
                        }
                        Err(err) => {
                            tracing::warn!(error = %err, "local-model guardrail inference failed; span left unmasked");
                        }
                    }
                }
            }
        }
        if rule_masked + rule_passed + model_judged > 0 {
            tracing::debug!(
                rule_masked,
                rule_passed,
                model_judged,
                "local-model segment candidates resolved"
            );
        }
        if hits.is_empty() {
            (text.to_owned(), 0)
        } else {
            let count = hits.len() as u32;
            (apply_masks(text, &hits), count)
        }
    }

    /// One moderation pass over a request's (or response's) text
    /// segments — the shared body of both segment hooks.
    async fn moderate(&self, texts: &[String]) -> SegmentsOutcome {
        let mut budget = MAX_MODEL_CALLS_PER_PASS;
        let mut masked: Vec<String> = Vec::with_capacity(texts.len());
        let mut total = 0u32;
        for text in texts {
            let (rewritten, count) = self.mask_segment(text, &mut budget).await;
            masked.push(rewritten);
            total += count;
        }
        let mut counts = BTreeMap::new();
        if total > 0 {
            counts.insert(DETECTOR_NAME.to_owned(), total);
        }
        SegmentsOutcome {
            verdict: GuardrailVerdict::Allow,
            masked: (total > 0).then_some(masked),
            counts,
            monitor_hits: Vec::new(),
        }
    }
}

#[async_trait]
impl Guardrail for LocalModelGuardrail {
    fn name(&self) -> &'static str {
        "local_model"
    }

    /// Consulted through the segment hooks only (the mask write-back
    /// channel); the plain `check_*` hooks stay default-`Allow`.
    fn moderates_segments(&self) -> bool {
        true
    }

    /// Masking a streamed response needs the whole response held back (a
    /// span can cross any chunk boundary) — but past the buffer cap this
    /// guardrail must release UNMASKED, not block: the trait default's
    /// fail-closed overflow would turn a >cap streamed response into a
    /// `content_filter` error from a guardrail whose contract is
    /// "rewrite, never block" (audit finding on PR #999). Past-cap
    /// content degrades to fewer masks, the same fail-open arm as
    /// inference failure and the per-pass call cap. A chain member with
    /// a stricter policy (e.g. fail-closed pii) still wins the fold.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        StreamOutputPolicy::BufferFull {
            max_buffer_bytes: DEFAULT_STREAM_OUTPUT_BUFFER_BYTES,
            on_exceeded_fail_open: true,
        }
    }

    async fn moderate_input_segments(&self, texts: &[String]) -> SegmentsOutcome {
        self.moderate(texts).await
    }

    async fn moderate_output_segments(&self, texts: &[String]) -> SegmentsOutcome {
        self.moderate(texts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn re() -> Regex {
        Regex::new(CANDIDATE_PATTERN).unwrap()
    }

    #[test]
    fn candidate_spans_find_dotted_runs_only() {
        let text = "版本是 12.1,构建号 2022.4.1,端口 8080";
        let spans = candidate_spans(&re(), text);
        let values: Vec<&str> = spans.iter().map(|s| &text[s.clone()]).collect();
        assert_eq!(values, vec!["12.1", "2022.4.1"]);
    }

    #[test]
    fn candidate_spans_drop_oversized_runs() {
        // A crafted `1.1.1...` run over the byte cap is one regex match
        // but NOT a candidate — it must vanish before budget accounting
        // so it can neither stall the lane nor starve real candidates.
        let bomb = "1.1".repeat(60); // 180 bytes, single match
        let text = format!("前缀 {bomb} 中缀 12.1 后缀");
        let spans = candidate_spans(&re(), &text);
        let values: Vec<&str> = spans.iter().map(|s| &text[s.clone()]).collect();
        assert_eq!(values, vec!["12.1"]);
    }

    #[test]
    fn window_bounds_snap_to_char_boundaries() {
        let text = "这个 EDA 软件的版本是 12.1,请勿外传";
        let span = candidate_spans(&re(), text).remove(0);
        // A tiny context still lands on char boundaries around CJK.
        let w = window_bounds(text, &span, 3);
        let window = &text[w];
        assert!(window.contains("12.1"), "window: {window}");
        assert_eq!(window, "本是 12.1,请勿");
    }

    #[test]
    fn window_bounds_clamp_to_text_edges() {
        let text = "12.1 只有后文";
        let span = candidate_spans(&re(), text).remove(0);
        let w = window_bounds(text, &span, 50);
        assert_eq!(&text[w], text);
    }

    #[test]
    fn apply_masks_rewrites_right_to_left() {
        let text = "从 12.1 升到 13.0 了";
        let spans = candidate_spans(&re(), text);
        assert_eq!(apply_masks(text, &spans), "从 *** 升到 *** 了");
    }

    #[test]
    fn config_from_env_requires_model_dir() {
        // Isolated var names are process-global; this test only checks the
        // parse fallback path via the public constructor contract.
        let cfg = LocalModelConfig {
            model_dir: PathBuf::from("/nonexistent"),
            threshold: PrototypeStrategy::default().default_threshold(),
            lanes: 1,
            rule_window: rules::DEFAULT_PROXIMITY_CHARS,
            prototypes: PrototypeStrategy::default(),
        };
        assert!(LocalModelGuardrail::load(&cfg).is_err());
    }

    #[test]
    fn prototype_strategy_parses_and_defaults() {
        assert_eq!(PrototypeStrategy::parse(None), PrototypeStrategy::default());
        assert_eq!(
            PrototypeStrategy::parse(Some("description")),
            PrototypeStrategy::Description
        );
        assert_eq!(
            PrototypeStrategy::parse(Some("max")),
            PrototypeStrategy::SampleMax
        );
        assert_eq!(
            PrototypeStrategy::parse(Some("centroid")),
            PrototypeStrategy::SampleCentroid
        );
        // Case-insensitive: an operator typing `Description` means it.
        assert_eq!(
            PrototypeStrategy::parse(Some("Description")),
            PrototypeStrategy::Description
        );
        assert_eq!(
            PrototypeStrategy::parse(Some("MAX")),
            PrototypeStrategy::SampleMax
        );
        assert_eq!(
            PrototypeStrategy::parse(Some("bogus")),
            PrototypeStrategy::default()
        );
    }

    #[test]
    fn centroid_is_the_renormalized_mean() {
        let c = centroid(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        assert!((c[0] - inv_sqrt2).abs() < 1e-6 && (c[1] - inv_sqrt2).abs() < 1e-6);
        // Max-over-set picks the nearest prototype.
        let set = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!((prototype_score(&set, &[0.0, 1.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_lanes_defaults_and_clamps() {
        // Unset / malformed / zero → 1 (zero is a misconfiguration, not
        // a disable switch); valid values pass; huge values clamp.
        assert_eq!(parse_lanes(None), 1);
        assert_eq!(parse_lanes(Some("")), 1);
        assert_eq!(parse_lanes(Some("abc")), 1);
        assert_eq!(parse_lanes(Some("-2")), 1);
        assert_eq!(parse_lanes(Some("0")), 1);
        assert_eq!(parse_lanes(Some("4")), 4);
        assert_eq!(parse_lanes(Some("9999")), MAX_LANES);
    }

    // ── model-backed tests (need the real model files) ──────────────────
    //
    // Run explicitly with the model directory present:
    //   GUARDRAIL_LOCAL_MODEL_DIR=~/.cache/aisix-local-guardrail-mvp \
    //     cargo test -p aisix-guardrails --features local-model -- --ignored

    fn load_from_env() -> Option<LocalModelGuardrail> {
        let cfg = LocalModelConfig::from_env()?;
        Some(LocalModelGuardrail::load(&cfg).expect("model files present but load failed"))
    }

    /// The MVP probe matrix, re-run for the prototype-set experiment:
    /// the same 7 probe windows (1 acceptance-style positive, 2 hard
    /// positives, 4 hard negatives) scored against 5 single-description
    /// prototype phrasings (the MVP sweep that measured NEGATIVE margin
    /// in every column) plus the two sample-based strategies. Prints the
    /// full matrix and each column's hard margin
    /// (min over hard positives − max over negatives).
    ///
    /// The MVP's original 5-phrasing sweep was scratch work; these
    /// phrasings reconstruct it (the shipped description first) and are
    /// committed so the experiment stays repeatable.
    #[tokio::test]
    #[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with model.onnx + tokenizer.json"]
    async fn probe_similarity_matrix() {
        let Some(g) = load_from_env() else { return };
        let phrasings = [
            PROTOTYPE_DESCRIPTION_ZH,
            "软件版本号",
            "芯片设计软件的版本号",
            "提到了 EDA 工具的具体版本号",
            "EDA 软件的版本信息,比如某个工具的版本是 12.1",
        ];
        let windows = [
            ("ACC", "这个 EDA 软件的版本是 12.1,请确认兼容性"),
            ("POS", "我们把仿真工具升级到 2022.4 之后速度快了很多"),
            ("POS", "Virtuoso IC6.1.8 出现了崩溃"),
            ("NEG", "Elapsed: 12.345s, Memory: 4.2 GB"),
            ("NEG", "服务器的 IP 地址是 10.2.255.1"),
            ("NEG", "圆周率约等于 3.14159"),
            ("NEG", "工艺节点是 0.13um,良率还行"),
        ];

        // Columns: each phrasing as a single-vector prototype set, then
        // the sample set (max) and its centroid.
        let mut columns: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
        for p in phrasings {
            let v = g.embed_window(p.to_owned()).await.unwrap();
            columns.push((format!("desc:{p}"), vec![v]));
        }
        let mut samples = Vec::new();
        for s in PROTOTYPE_SAMPLES {
            samples.push(g.embed_window((*s).to_owned()).await.unwrap());
        }
        columns.push(("samples-max".to_owned(), samples.clone()));
        columns.push(("samples-centroid".to_owned(), vec![centroid(&samples)]));

        for (name, prototypes) in &columns {
            let mut hard_pos_min = f32::INFINITY;
            let mut neg_max = f32::NEG_INFINITY;
            println!("── column: {name}");
            for (kind, text) in windows {
                let v = g.embed_window(text.to_owned()).await.unwrap();
                let s = prototype_score(prototypes, &v);
                println!("  {kind} {s:.4}  {text}");
                match kind {
                    "POS" => hard_pos_min = hard_pos_min.min(s),
                    "NEG" => neg_max = neg_max.max(s),
                    _ => {}
                }
            }
            println!(
                "  hard margin (min POS − max NEG): {:+.4}",
                hard_pos_min - neg_max
            );

            // Pin the calibration contract for the sample strategies:
            // the margin the experiment claims stays open, and the
            // shipped default gate sits strictly inside it. The
            // description columns stay unasserted — their negative
            // margin is the documented MVP finding, not a contract.
            let gate = match name.as_str() {
                "samples-max" => Some(PrototypeStrategy::SampleMax.default_threshold()),
                "samples-centroid" => Some(PrototypeStrategy::SampleCentroid.default_threshold()),
                _ => None,
            };
            if let Some(gate) = gate {
                assert!(
                    neg_max < gate && gate <= hard_pos_min,
                    "{name}: default gate {gate} outside the measured band ({neg_max:.4}, {hard_pos_min:.4}]"
                );
            }
        }
    }

    /// A candidate flood past [`MAX_RULE_SCORED_SPANS_PER_SEGMENT`]
    /// releases the tail unscored: a rule-maskable sentence hidden
    /// beyond the cap stays untouched (fail-open — rewrite less, never
    /// stall), and the flood burns at most the model-call budget.
    #[tokio::test]
    #[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with model.onnx + tokenizer.json"]
    async fn candidate_flood_releases_the_tail() {
        let Some(g) = load_from_env() else { return };
        let flood = "1.1 ".repeat(MAX_RULE_SCORED_SPANS_PER_SEGMENT);
        let text = format!("{flood}这个 EDA 软件的版本是 12.1");
        let outcome = g.moderate_input_segments(&[text]).await;
        assert_eq!(outcome.verdict, GuardrailVerdict::Allow);
        assert!(
            outcome.masked.is_none(),
            "the over-cap tail must be released unscored"
        );
    }

    /// The acceptance matrix end to end through the segment hook, on the
    /// default configuration: both hard positives and the MVP acceptance
    /// positive are rewritten; every hard negative comes back untouched
    /// (`masked == None` ⇒ byte-identical passthrough upstream).
    #[tokio::test]
    #[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with model.onnx + tokenizer.json"]
    async fn acceptance_matrix_end_to_end() {
        let Some(g) = load_from_env() else { return };
        let masked_cases = [
            ("这个 EDA 软件的版本是 12.1", "这个 EDA 软件的版本是 ***"),
            (
                "我们把仿真工具升级到 2022.4 之后速度快了很多",
                "我们把仿真工具升级到 *** 之后速度快了很多",
            ),
            ("Virtuoso IC6.1.8 出现了崩溃", "Virtuoso IC*** 出现了崩溃"),
        ];
        for (input, want) in masked_cases {
            let outcome = g.moderate_input_segments(&[input.to_owned()]).await;
            assert_eq!(outcome.verdict, GuardrailVerdict::Allow);
            let masked = outcome
                .masked
                .unwrap_or_else(|| panic!("must mask: {input}"));
            assert_eq!(masked[0], want);
        }
        let passthrough_cases = [
            "Elapsed: 12.345s, Memory: 4.2 GB",
            "服务器的 IP 地址是 10.2.255.1",
            "圆周率约等于 3.14159",
            "工艺节点是 0.13um,良率还行",
        ];
        for input in passthrough_cases {
            let outcome = g.moderate_input_segments(&[input.to_owned()]).await;
            assert_eq!(outcome.verdict, GuardrailVerdict::Allow);
            assert!(
                outcome.masked.is_none(),
                "negative must pass untouched: {input} → {:?}",
                outcome.masked
            );
        }
    }

    /// The acceptance path in miniature: the segment hook masks the
    /// version number and only the version number.
    #[tokio::test]
    #[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with model.onnx + tokenizer.json"]
    async fn masks_acceptance_sentence() {
        let Some(g) = load_from_env() else { return };
        let texts = vec!["这个 EDA 软件的版本是 12.1".to_owned()];
        let outcome = g.moderate_input_segments(&texts).await;
        assert_eq!(outcome.verdict, GuardrailVerdict::Allow);
        let masked = outcome.masked.expect("must mask the version number");
        assert_eq!(masked[0], "这个 EDA 软件的版本是 ***");
        assert_eq!(outcome.counts.get(DETECTOR_NAME), Some(&1));
    }

    /// Lanes are interchangeable and safe under concurrency: with a
    /// 2-lane pool, 8 concurrent embeds of the same window all succeed
    /// and agree (sessions share nothing but identical weights, and a
    /// single-threaded run is deterministic).
    #[tokio::test]
    #[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with model.onnx + tokenizer.json"]
    async fn lanes_run_concurrently_and_agree() {
        let Some(mut cfg) = LocalModelConfig::from_env() else {
            return;
        };
        cfg.lanes = 2;
        let g = std::sync::Arc::new(
            LocalModelGuardrail::load(&cfg).expect("model files present but load failed"),
        );
        let window = "这个 EDA 软件的版本是 12.1,请确认兼容性";
        let tasks: Vec<_> = (0..8)
            .map(|_| {
                let g = std::sync::Arc::clone(&g);
                tokio::spawn(async move { g.embed_window(window.to_owned()).await })
            })
            .collect();
        let mut vectors = Vec::new();
        for t in tasks {
            vectors.push(t.await.unwrap().expect("concurrent embed failed"));
        }
        let reference = &vectors[0];
        for v in &vectors[1..] {
            let agreement = cosine(v, reference);
            assert!(agreement > 0.9999, "lanes disagree: cosine {agreement}");
        }
    }

    /// Throughput calibration for api7/aisix#1001: lanes come from
    /// GUARDRAIL_LOCAL_MODEL_LANES, so one binary sweeps 1/2/4 lanes.
    /// Prints inferences/s over a saturating concurrent batch.
    #[tokio::test]
    #[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with model.onnx + tokenizer.json"]
    async fn probe_lane_throughput() {
        let Some(g) = load_from_env().map(std::sync::Arc::new) else {
            return;
        };
        let window = "这个 EDA 软件的版本是 12.1,请确认与工艺库的兼容性之后再安排回归测试";
        // Warm-up.
        for _ in 0..3 {
            g.embed_window(window.to_owned()).await.unwrap();
        }
        let total = 64usize;
        let started = Instant::now();
        let tasks: Vec<_> = (0..total)
            .map(|_| {
                let g = std::sync::Arc::clone(&g);
                tokio::spawn(async move { g.embed_window(window.to_owned()).await })
            })
            .collect();
        for t in tasks {
            t.await.unwrap().expect("embed failed");
        }
        let secs = started.elapsed().as_secs_f64();
        println!(
            "lanes={} total={} wall={:.2}s throughput={:.1} inferences/s",
            g.embedder.sessions.len(),
            total,
            secs,
            total as f64 / secs
        );
    }

    /// Rough single-inference latency figure for the MVP report.
    #[tokio::test]
    #[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with model.onnx + tokenizer.json"]
    async fn probe_inference_latency() {
        let Some(g) = load_from_env() else { return };
        let window =
            "这个 EDA 软件的版本是 12.1,请确认与工艺库的兼容性之后再安排回归测试".to_owned();
        // Warm-up, then timed runs.
        for _ in 0..3 {
            g.embed_window(window.clone()).await.unwrap();
        }
        let mut samples = Vec::new();
        for _ in 0..20 {
            let t = Instant::now();
            g.embed_window(window.clone()).await.unwrap();
            samples.push(t.elapsed().as_micros() as u64);
        }
        samples.sort_unstable();
        println!(
            "inference us over 20 runs: p50={} min={} max={}",
            samples[samples.len() / 2],
            samples[0],
            samples[samples.len() - 1]
        );
    }
}
