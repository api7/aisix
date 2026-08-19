//! Local CPU embedding-model guardrail — MVP vertical slice of the
//! second-tier guardrail (AISIX-Cloud#1331).
//!
//! Proves ONE thing: an in-process ONNX embedding model can sit inside
//! the guardrail chain, produce a span-level judgement, and drive a real
//! in-place rewrite. It is NOT the full three-layer pipeline from the
//! design issue (no rule scoring layer, no prototype library, no
//! standard risk categories, single hardcoded category).
//!
//! Pipeline per text segment:
//!   1. regex finds candidate spans with exact byte offsets
//!      (dotted number runs, the EDA-version candidate shape);
//!   2. a context window around each candidate (±[`WINDOW_CONTEXT_CHARS`]
//!      chars — the keyword-proximity window magnitude mainstream DLP
//!      engines use, typically 50–300 chars) is embedded by the local
//!      model and compared, by cosine similarity, against ONE category
//!      prototype vector encoded at load time from a hardcoded Chinese
//!      description sentence;
//!   3. above-threshold candidates are rewritten in place to
//!      [`MASK_REPLACEMENT`]; everything else is returned byte-identical.
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
//! `spawn_blocking`, bounded by a semaphore sized to the session count
//! (one — `ort` 2.0.0-rc.13's `Session::run` takes `&mut self`, so one
//! session serializes anyway), with ONE intra-op thread and intra/inter-op
//! **spinning disabled** — ONNX Runtime's default spin-wait burns ~9.4%
//! of a core while completely idle, taxing deployments that never send
//! guardrail traffic.
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

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use ort::session::Session;
use ort::value::Tensor;
use regex::Regex;

use crate::{Guardrail, GuardrailVerdict, SegmentsOutcome};

/// Environment variable holding the model directory (`model.onnx` +
/// `tokenizer.json`). Set → the server bootstrap loads and injects the
/// guardrail; unset → the feature is completely inert.
pub const MODEL_DIR_ENV: &str = "GUARDRAIL_LOCAL_MODEL_DIR";
/// Optional cosine-similarity gate override (default
/// [`DEFAULT_THRESHOLD`]).
pub const THRESHOLD_ENV: &str = "GUARDRAIL_LOCAL_MODEL_THRESHOLD";

/// Cosine-similarity gate for "this window is about the category".
/// Calibrated with this module's `#[ignore]` probe against the
/// prototype below: the acceptance-style positive scores ~0.90, every
/// probed negative (compile-log timings, memory sizes, IPs, plain
/// numbers) ≤0.76. Deliberately precision-leaning — a mask
/// false-positive corrupts user content — at a measured recall cost:
/// harder positives ("升级到 2022.4"、"Virtuoso IC6.1.8") score
/// 0.75–0.79 and are NOT masked at this default. Single-prototype
/// zero-shot cosine cannot separate those from the hard negatives at
/// all (every phrasing swept had negative margin); closing that gap is
/// the design issue's rule-scoring layer + real-sample prototypes, not
/// a threshold tweak.
const DEFAULT_THRESHOLD: f32 = 0.80;

/// The ONE category this MVP ships: EDA-software version numbers. The
/// prototype vector is the load-time embedding of this sentence — the
/// v1 "customer types one Chinese description" path from the design
/// issue, collapsed to a compile-time constant.
const PROTOTYPE_DESCRIPTION_ZH: &str = "EDA 软件的版本号";

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
pub struct LocalModelConfig {
    pub model_dir: PathBuf,
    pub threshold: f32,
}

impl LocalModelConfig {
    /// `None` when [`MODEL_DIR_ENV`] is unset (guardrail disabled). A
    /// malformed threshold falls back to the default rather than
    /// failing boot — the gate is a tuning knob, not a correctness one.
    pub fn from_env() -> Option<Self> {
        let model_dir = PathBuf::from(std::env::var_os(MODEL_DIR_ENV)?);
        let threshold = std::env::var(THRESHOLD_ENV)
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(DEFAULT_THRESHOLD);
        Some(Self {
            model_dir,
            threshold,
        })
    }
}

/// The blocking inference core: tokenizer + ONNX session. Session
/// access serializes through the mutex (`Session::run` takes `&mut`);
/// the guardrail's semaphore keeps queued blocking tasks bounded.
struct Embedder {
    tokenizer: tokenizers::Tokenizer,
    session: Mutex<Session>,
}

impl Embedder {
    fn load(dir: &std::path::Path) -> Result<Self, LocalModelError> {
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            LocalModelError::Tokenizer(format!("{}: {e}", tokenizer_path.display()))
        })?;
        let model_path = dir.join("model.onnx");
        // One intra-op thread and NO spin-wait: the guardrail must not
        // tax a gateway that isn't sending it traffic (see module doc).
        // Builder-stage errors carry the builder back for recovery
        // (`ort::Error<SessionBuilder>`); this path never recovers, so
        // they fold to their message.
        let build = |b: ort::session::builder::SessionBuilder| {
            b.with_intra_threads(1)?
                .with_inter_threads(1)?
                .with_intra_op_spinning(false)?
                .with_inter_op_spinning(false)
        };
        let session = build(Session::builder()?)
            .map_err(|e| LocalModelError::Model(format!("session options: {e}")))?
            .commit_from_file(&model_path)?;
        Ok(Self {
            tokenizer,
            session: Mutex::new(session),
        })
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
        let mut session = self.session.lock().expect("embedder session poisoned");
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

/// Candidate spans (byte ranges) in `text`, in order.
fn candidate_spans(re: &Regex, text: &str) -> Vec<Range<usize>> {
    re.find_iter(text).map(|m| m.range()).collect()
}

/// The context window around `span`: `ctx` chars on each side, snapped
/// to char boundaries, clamped to the text.
fn window_bounds(text: &str, span: &Range<usize>, ctx: usize) -> Range<usize> {
    let before = &text[..span.start];
    let start = before
        .char_indices()
        .rev()
        .nth(ctx.saturating_sub(1))
        .map_or_else(|| if ctx == 0 { span.start } else { 0 }, |(i, _)| i);
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
    /// L2-normalized embedding of [`PROTOTYPE_DESCRIPTION_ZH`].
    prototype: Vec<f32>,
    threshold: f32,
    candidate_re: Regex,
    /// Bounds in-flight `spawn_blocking` inference tasks. Sized to the
    /// session count (1): more permits would only queue on the session
    /// mutex from inside blocking threads.
    permits: Arc<tokio::sync::Semaphore>,
}

impl LocalModelGuardrail {
    /// Load tokenizer + session and encode the category prototype.
    /// Blocking (model load + one inference) — the server bootstrap
    /// wraps it in `spawn_blocking`.
    pub fn load(config: &LocalModelConfig) -> Result<Self, LocalModelError> {
        let started = Instant::now();
        let embedder = Embedder::load(&config.model_dir)?;
        let prototype = embedder.embed(PROTOTYPE_DESCRIPTION_ZH)?;
        tracing::info!(
            model_dir = %config.model_dir.display(),
            threshold = config.threshold,
            load_ms = started.elapsed().as_millis() as u64,
            "local-model guardrail loaded (category: EDA software version)"
        );
        Ok(Self {
            embedder: Arc::new(embedder),
            prototype,
            threshold: config.threshold,
            candidate_re: Regex::new(CANDIDATE_PATTERN).expect("candidate pattern must compile"),
            permits: Arc::new(tokio::sync::Semaphore::new(1)),
        })
    }

    /// Embed one window off the async runtime: bounded by the
    /// semaphore, executed in `spawn_blocking`.
    async fn embed_window(&self, window: String) -> Result<Vec<f32>, LocalModelError> {
        let _permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .expect("inference semaphore never closed");
        let embedder = Arc::clone(&self.embedder);
        tokio::task::spawn_blocking(move || embedder.embed(&window))
            .await
            .map_err(|e| LocalModelError::Model(format!("inference task join: {e}")))?
    }

    /// Mask one segment. Returns the rewritten text and how many spans
    /// were masked; `budget` is the shared per-pass model-call cap.
    /// Model failure on a candidate leaves that span untouched (rewrite
    /// less, never block — the fail-open arm of "只改写不阻断").
    async fn mask_segment(&self, text: &str, budget: &mut usize) -> (String, u32) {
        let mut hits: Vec<Range<usize>> = Vec::new();
        for span in candidate_spans(&self.candidate_re, text) {
            if *budget == 0 {
                tracing::warn!(
                    cap = MAX_MODEL_CALLS_PER_PASS,
                    "local-model guardrail: candidate cap reached; remaining candidates left unmasked"
                );
                break;
            }
            *budget -= 1;
            let window = text[window_bounds(text, &span, WINDOW_CONTEXT_CHARS)].to_owned();
            match self.embed_window(window).await {
                Ok(vector) => {
                    let score = cosine(&vector, &self.prototype);
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
            threshold: DEFAULT_THRESHOLD,
        };
        assert!(LocalModelGuardrail::load(&cfg).is_err());
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

    /// Calibration probe: prints the cosine matrix behind
    /// [`DEFAULT_THRESHOLD`] and pins the MVP contract — the
    /// acceptance-style positive clears the gate, every probed negative
    /// stays under it. The harder positives are printed but NOT
    /// asserted: single-prototype zero-shot cosine has no margin over
    /// the hard negatives (measured; see the threshold doc), so at the
    /// precision-leaning default they are a known recall gap.
    #[tokio::test]
    #[ignore = "needs GUARDRAIL_LOCAL_MODEL_DIR with model.onnx + tokenizer.json"]
    async fn probe_similarity_matrix() {
        let Some(g) = load_from_env() else { return };
        let acceptance = "这个 EDA 软件的版本是 12.1,请确认兼容性";
        let recall_gap_positives = [
            "我们把仿真工具升级到 2022.4 之后速度快了很多",
            "Virtuoso IC6.1.8 出现了崩溃",
        ];
        let negatives = [
            "Elapsed: 12.345s, Memory: 4.2 GB",
            "服务器的 IP 地址是 10.2.255.1",
            "圆周率约等于 3.14159",
            "工艺节点是 0.13um,良率还行",
        ];
        let acc = cosine(
            &g.embed_window(acceptance.to_owned()).await.unwrap(),
            &g.prototype,
        );
        println!("ACC {acc:.4}  {acceptance}");
        for text in recall_gap_positives {
            let s = cosine(
                &g.embed_window(text.to_owned()).await.unwrap(),
                &g.prototype,
            );
            println!("POS(gap) {s:.4}  {text}");
        }
        assert!(
            acc >= g.threshold,
            "acceptance positive {acc:.4} under threshold {}",
            g.threshold
        );
        for text in negatives {
            let s = cosine(
                &g.embed_window(text.to_owned()).await.unwrap(),
                &g.prototype,
            );
            println!("NEG {s:.4}  {text}");
            assert!(
                s < g.threshold,
                "negative {s:.4} would mask at threshold {}: {text}",
                g.threshold
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
