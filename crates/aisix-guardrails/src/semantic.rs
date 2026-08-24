//! kind=semantic guardrail (AISIX-Cloud#1375) — embedding-similarity
//! screening against operator-supplied EXAMPLE texts.
//!
//! Detection-only: it blocks, never rewrites. Monitor-before-enforce
//! comes from the row's `enforcement_mode`, as with every other
//! detecting kind.
//!
//! Decision rule, per screened text:
//!
//! | condition                                              | verdict |
//! |--------------------------------------------------------|---------|
//! | best deny similarity ≥ `deny_threshold`                 | Block   |
//! | `allow_examples` non-empty, best allow < `allow_threshold` | Block |
//! | otherwise                                               | Allow   |
//!
//! **Deny wins over allow** — the industry-standard precedence, and the
//! only safe one: an allow example that happens to sit near a deny
//! example must not launder traffic past the deny list.
//!
//! The kind is direction-neutral. `hook_point` decides whether requests,
//! responses or both are screened, and the SAME example lists apply to
//! whichever hooks run; two directions needing different examples are
//! two guardrail rows. (Kong ships this as two plugins,
//! `ai-semantic-prompt-guard` and `ai-semantic-response-guard`, because
//! its plugin config has no hook selector; ours does.)
//!
//! Text selection differs deliberately from every other kind here.
//! The remote-API kinds CONCATENATE a request's messages into one blob,
//! because they bill and rate-limit per call. Concatenation is wrong for
//! a similarity judgement: cosine against a prototype falls as unrelated
//! text is appended, so a short attack buried in a long conversation
//! scores as noise. This kind therefore screens each message
//! SEPARATELY, newest first, up to `max_screened_texts`.
//!
//! Screening newest-first is what makes the cap safe rather than a hole:
//! chat clients resend the whole history every turn, so a message that
//! falls off the tail today was screened as the newest message of an
//! earlier request. The cap drops re-reads, not unread content. (The
//! upstream baseline screens ONLY the single newest user message and is
//! walked past by leading with a benign opener; screening the newest N
//! costs one batched call and closes that.)
//!
//! Embedding dispatch goes through [`GuardrailEmbedder`], injected by
//! the proxy — see its doc for why the call cannot be made from this
//! crate directly. Two calls per screened hook at most: one for the
//! example prototypes (cacheable, so a warm process pays it once per
//! example set) and one batching every candidate text.
//!
//! Behavior matrix (failure modes). The effective `fail_open` is the
//! outer `Guardrail::fail_open` on the INPUT hook and the independent
//! `SemanticConfig::output_fail_open` (default fail-closed) on the
//! OUTPUT hook:
//!
//! | embedding dispatch      | `fail_open` | Verdict                              |
//! |-------------------------|-------------|--------------------------------------|
//! | scored, under threshold | n/a         | Allow                                |
//! | scored, over threshold  | n/a         | Block { reason }                     |
//! | model alias unresolved  | true        | Bypass { "semantic_embed_unresolved" } |
//! | timeout                 | true        | Bypass { "semantic_embed_timeout" }  |
//! | provider error          | true        | Bypass { "semantic_embed_upstream" } |
//! | any failure             | false       | Block { unavailable: <same tag> }    |
//!
//! Block reasons carry the matched EXAMPLE'S INDEX and the score, never
//! the screened text and never the example text (#153): a reason that
//! echoed either would let a caller enumerate the list by probing.

use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{GuardrailHookPoint, SemanticConfig};
use aisix_core::{best_similarity, cosine_similarity};
use aisix_gateway::{ChatFormat, ChatResponse, Role};
use async_trait::async_trait;

use crate::{
    EmbedFailure, Guardrail, GuardrailEmbedder, GuardrailVerdict, StreamOutputPolicy,
};

/// Input-hook text selection: screen every message rather than only the
/// user's. Mirrors the `concatenate_all_content` option the remote kinds
/// expose, minus the concatenation this kind cannot use.
const TEXT_SOURCE_ALL: &str = "all_messages";

pub struct SemanticGuardrail {
    embedder: Arc<dyn GuardrailEmbedder>,
    embedding_model: String,
    deny_examples: Vec<String>,
    allow_examples: Vec<String>,
    deny_threshold: f32,
    allow_threshold: f32,
    timeout: Duration,
    max_screened_texts: usize,
    scan_all_messages: bool,
    hook_point: GuardrailHookPoint,
    /// Fail-open policy for the INPUT hook (the outer `Guardrail::fail_open`).
    fail_open: bool,
    /// Fail-open policy for the OUTPUT hook (default fail-closed).
    output_fail_open: bool,
    max_buffer_bytes: u64,
    on_buffer_exceeded_fail_open: bool,
}

impl SemanticGuardrail {
    /// Caller owns `hook_point` and `fail_open` (they live on the
    /// `Guardrail` row, not in the kind's config block).
    pub fn new(
        cfg: &SemanticConfig,
        hook_point: GuardrailHookPoint,
        fail_open: bool,
        embedder: Arc<dyn GuardrailEmbedder>,
    ) -> Self {
        Self {
            embedder,
            embedding_model: cfg.embedding_model.clone(),
            deny_examples: cfg.deny_examples.clone(),
            allow_examples: cfg.allow_examples.clone(),
            deny_threshold: cfg.deny_threshold,
            allow_threshold: cfg.allow_threshold,
            timeout: Duration::from_millis(cfg.timeout_ms),
            max_screened_texts: cfg.max_screened_texts as usize,
            scan_all_messages: cfg.text_source == TEXT_SOURCE_ALL,
            hook_point,
            fail_open,
            output_fail_open: cfg.output_fail_open,
            max_buffer_bytes: cfg.max_buffer_bytes,
            on_buffer_exceeded_fail_open: cfg.on_buffer_exceeded != "fail_closed",
        }
    }

    fn hook_enabled(&self, hook: GuardrailHookPoint) -> bool {
        self.hook_point == GuardrailHookPoint::Both || self.hook_point == hook
    }

    /// `true` when no example is configured at all. Such a row cannot
    /// reach a verdict, so it must not spend an embedding call finding
    /// that out on every request.
    fn screens_nothing(&self) -> bool {
        self.deny_examples.is_empty() && self.allow_examples.is_empty()
    }

    async fn screen(&self, texts: Vec<String>, fail_open: bool) -> GuardrailVerdict {
        if texts.is_empty() || self.screens_nothing() {
            return GuardrailVerdict::Allow;
        }

        // One prototype call (cacheable: the example set is config, fixed
        // for the row) and one candidate call. Deny examples come first
        // so the split below needs no second lookup.
        let mut prototypes = self.deny_examples.clone();
        prototypes.extend(self.allow_examples.iter().cloned());
        let prototype_vecs = match self
            .embedder
            .embed(&self.embedding_model, &prototypes, true, self.timeout)
            .await
        {
            Ok(v) if v.len() == prototypes.len() => v,
            Ok(_) => return self.on_failure(EmbedFailure::Upstream, fail_open),
            Err(failure) => return self.on_failure(failure, fail_open),
        };

        let candidate_vecs = match self
            .embedder
            .embed(&self.embedding_model, &texts, false, self.timeout)
            .await
        {
            Ok(v) if v.len() == texts.len() => v,
            Ok(_) => return self.on_failure(EmbedFailure::Upstream, fail_open),
            Err(failure) => return self.on_failure(failure, fail_open),
        };

        let (deny_vecs, allow_vecs) = prototype_vecs.split_at(self.deny_examples.len());

        for candidate in &candidate_vecs {
            if let Some(verdict) = self.judge(candidate, deny_vecs, allow_vecs) {
                return verdict;
            }
        }
        GuardrailVerdict::Allow
    }

    /// Score ONE candidate. `Some(Block)` when it is refused, `None`
    /// when it passes both gates.
    fn judge(
        &self,
        candidate: &[f32],
        deny_vecs: &[Vec<f32>],
        allow_vecs: &[Vec<f32>],
    ) -> Option<GuardrailVerdict> {
        // Deny first, unconditionally: a text matching both lists is
        // refused, never laundered by its allow score.
        if let Some((index, score)) = self.closest(candidate, deny_vecs) {
            if score >= self.deny_threshold {
                return Some(GuardrailVerdict::block(format!(
                    "semantic deny example #{index} matched (similarity {score:.3} \
                     >= threshold {:.3})",
                    self.deny_threshold
                )));
            }
        }

        if !allow_vecs.is_empty() {
            // An empty prototype set is impossible here, so the
            // `unwrap_or` is only a total-function guard.
            let best = best_similarity(candidate, allow_vecs.iter().map(Vec::as_slice))
                .unwrap_or(f32::NEG_INFINITY);
            if best < self.allow_threshold {
                return Some(GuardrailVerdict::block(format!(
                    "no semantic allow example matched (best similarity {best:.3} \
                     < threshold {:.3})",
                    self.allow_threshold
                )));
            }
        }
        None
    }

    /// The closest prototype and its score, for the deny side — the
    /// INDEX is what makes a block actionable in an ops log without
    /// echoing either the example or the screened text (#153).
    fn closest(&self, candidate: &[f32], prototypes: &[Vec<f32>]) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, prototype) in prototypes.iter().enumerate() {
            let score = cosine_similarity(candidate, prototype);
            if !score.is_finite() {
                continue;
            }
            if best.is_none_or(|(_, current)| score > current) {
                best = Some((i, score));
            }
        }
        best
    }

    fn on_failure(&self, failure: EmbedFailure, fail_open: bool) -> GuardrailVerdict {
        let tag = failure.as_str();
        tracing::warn!(
            guardrail = "semantic",
            embedding_model = %self.embedding_model,
            failure = tag,
            fail_open,
            "semantic guardrail could not embed"
        );
        if fail_open {
            GuardrailVerdict::Bypass { reason: tag.into() }
        } else {
            GuardrailVerdict::block_unavailable(format!("semantic guardrail unavailable ({tag})"), tag)
        }
    }
}

/// The texts the INPUT hook screens: one per message, NEWEST FIRST,
/// capped at `max_screened_texts`. Empty messages are skipped before the
/// cap so they can't consume budget.
fn collect_input_texts(req: &ChatFormat, all_messages: bool, cap: usize) -> Vec<String> {
    req.messages
        .iter()
        .rev()
        .filter(|m| all_messages || m.role == Role::User)
        .map(crate::message_scan_text)
        .filter(|s| !s.is_empty())
        .take(cap)
        .collect()
}

#[async_trait]
impl Guardrail for SemanticGuardrail {
    fn name(&self) -> &'static str {
        "semantic"
    }

    /// Its streamed-output hold-back policy applies only when it
    /// inspects output (#466): an input-only row must not buffer the
    /// response for a check it never runs.
    fn runs_on_output(&self) -> bool {
        matches!(
            self.hook_point,
            GuardrailHookPoint::Output | GuardrailHookPoint::Both
        )
    }

    /// Always whole-response hold-back, like kind=pii and kind=presidio.
    /// A similarity judgement is only meaningful on a complete text: a
    /// sliding window would both score partial prose against whole-text
    /// prototypes and release the opening of a response before the
    /// window carrying the violation is judged.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        StreamOutputPolicy::BufferFull {
            max_buffer_bytes: self.max_buffer_bytes as usize,
            on_exceeded_fail_open: self.on_buffer_exceeded_fail_open,
        }
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        if !self.hook_enabled(GuardrailHookPoint::Input) {
            return GuardrailVerdict::Allow;
        }
        let texts = collect_input_texts(req, self.scan_all_messages, self.max_screened_texts);
        self.screen(texts, self.fail_open).await
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        if !self.hook_enabled(GuardrailHookPoint::Output) {
            return GuardrailVerdict::Allow;
        }
        let text = resp.guardrail_output_text();
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        self.screen(vec![text], self.output_fail_open).await
    }
}
