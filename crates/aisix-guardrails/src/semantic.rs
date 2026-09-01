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
//! `SemanticConfig::output_fail_open` on the OUTPUT hook.
//!
//! Those two defaults point in OPPOSITE directions, and the asymmetry is
//! inherited rather than chosen here: the row-level `fail_open` defaults
//! to `true` for every kind that can be unavailable, while
//! `output_fail_open` defaults to `false`. So an unscreenable REQUEST is
//! admitted by default and an unscreenable RESPONSE is refused. An
//! operator who wants unscreenable requests refused sets
//! `fail_open: false` on the row:
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
//!
//! Every execution also reports its numbers as [`GuardrailScore`] summaries
//! on the request's telemetry — pass or block, enforce or monitor
//! (AISIX-Cloud#1467). A similarity policy is untunable without them: the
//! verdict says whether the threshold was crossed, and an operator whose
//! threshold is slightly too high sees only a guardrail that never fires.
//! The score sink is the request's [`GuardrailAuditLog`], bound per request
//! by the chain — see [`Guardrail::bind_score_log`].

use std::sync::Arc;
use std::time::Duration;

use aisix_core::cosine_similarity;
use aisix_core::models::{GuardrailHookPoint, GuardrailScore, SemanticConfig};
use aisix_gateway::{ChatFormat, ChatResponse, Role};
use async_trait::async_trait;

use crate::{
    EmbedFailure, Guardrail, GuardrailAuditLog, GuardrailEmbedder, GuardrailVerdict,
    StreamOutputPolicy,
};

/// Input-hook text selection: screen every message rather than only the
/// user's. Mirrors the `concatenate_all_content` option the remote kinds
/// expose, minus the concatenation this kind cannot use.
const TEXT_SOURCE_ALL: &str = "all_messages";

/// Everything the row configures, shared by every request. Split out so
/// binding the per-request score log costs two atomic bumps instead of
/// copying the example lists onto each request.
struct SemanticParams {
    embedder: Arc<dyn GuardrailEmbedder>,
    /// The configured (row) name — the identity the score entries carry.
    /// The kind's static `name()` is `"semantic"` for every row, which
    /// would make two rows' scores indistinguishable.
    row_name: String,
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

pub struct SemanticGuardrail {
    cfg: Arc<SemanticParams>,
    /// Where this execution's [`GuardrailScore`] summaries go. `None` on the
    /// instance the index holds — that one is shared by every request and so
    /// can own no per-request state; the chain binds a clone carrying the
    /// request's log (see [`Guardrail::bind_score_log`]). A chain built
    /// without a log scores nothing, exactly as it audits nothing.
    scores: Option<Arc<GuardrailAuditLog>>,
}

impl SemanticGuardrail {
    /// Caller owns `row_name`, `hook_point` and `fail_open` (they live on
    /// the `Guardrail` row, not in the kind's config block).
    pub fn new(
        row_name: impl Into<String>,
        cfg: &SemanticConfig,
        hook_point: GuardrailHookPoint,
        fail_open: bool,
        embedder: Arc<dyn GuardrailEmbedder>,
    ) -> Self {
        Self {
            cfg: Arc::new(SemanticParams {
                embedder,
                row_name: row_name.into(),
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
                // Compare against the OPEN value, not the closed one: an
                // unrecognised string must land fail-closed.
                on_buffer_exceeded_fail_open: cfg.on_buffer_exceeded == "fail_open",
            }),
            scores: None,
        }
    }

    fn hook_enabled(&self, hook: GuardrailHookPoint) -> bool {
        self.cfg.hook_point == GuardrailHookPoint::Both || self.cfg.hook_point == hook
    }

    /// `true` when no example is configured at all. Such a row cannot
    /// reach a verdict, so it must not spend an embedding call finding
    /// that out on every request.
    fn screens_nothing(&self) -> bool {
        self.cfg.deny_examples.is_empty() && self.cfg.allow_examples.is_empty()
    }

    async fn screen(
        &self,
        texts: Vec<String>,
        fail_open: bool,
        hook: &'static str,
    ) -> GuardrailVerdict {
        if texts.is_empty() || self.screens_nothing() {
            return GuardrailVerdict::Allow;
        }

        // One prototype call (cacheable: the example set is config, fixed
        // for the row) and one candidate call. Deny examples come first
        // so the split below needs no second lookup.
        let mut prototypes = self.cfg.deny_examples.clone();
        prototypes.extend(self.cfg.allow_examples.iter().cloned());
        let prototype_vecs = match self
            .cfg
            .embedder
            .embed(
                &self.cfg.embedding_model,
                &prototypes,
                true,
                self.cfg.timeout,
            )
            .await
        {
            Ok(v) if v.len() == prototypes.len() => v,
            Ok(_) => return self.on_failure(EmbedFailure::Upstream, fail_open),
            Err(failure) => return self.on_failure(failure, fail_open),
        };

        let candidate_vecs = match self
            .cfg
            .embedder
            .embed(&self.cfg.embedding_model, &texts, false, self.cfg.timeout)
            .await
        {
            Ok(v) if v.len() == texts.len() => v,
            Ok(_) => return self.on_failure(EmbedFailure::Upstream, fail_open),
            Err(failure) => return self.on_failure(failure, fail_open),
        };

        let (deny_vecs, allow_vecs) = prototype_vecs.split_at(self.cfg.deny_examples.len());

        // The closest call in each direction, across the texts actually
        // judged. The verdict loop still stops at the first refusal — the
        // candidate batch is embedded up front, so scoring the rest would
        // cost no upstream call, but reporting a text the guardrail never
        // consulted would misstate what it did.
        let mut deny_peak: Option<(usize, f32)> = None;
        let mut allow_trough: Option<(usize, f32)> = None;
        let mut verdict = GuardrailVerdict::Allow;

        for candidate in &candidate_vecs {
            let scored = self.judge(candidate, deny_vecs, allow_vecs);
            if let Some((i, score)) = scored.deny {
                if deny_peak.is_none_or(|(_, peak)| score > peak) {
                    deny_peak = Some((i, score));
                }
            }
            if let Some((i, score)) = scored.allow {
                if allow_trough.is_none_or(|(_, trough)| score < trough) {
                    allow_trough = Some((i, score));
                }
            }
            if let Some(refusal) = scored.verdict {
                verdict = refusal;
                break;
            }
        }

        self.report(hook, "deny", self.cfg.deny_threshold, deny_peak);
        self.report(hook, "allow", self.cfg.allow_threshold, allow_trough);
        verdict
    }

    /// Hand one direction's summary to the request's score log. A `None`
    /// extreme means the direction was never scored — an empty example
    /// list, or a deny refusal that short-circuited before the allow gate —
    /// and reports nothing rather than a fabricated zero.
    fn report(
        &self,
        hook: &'static str,
        direction: &'static str,
        threshold: f32,
        extreme: Option<(usize, f32)>,
    ) {
        let (Some(log), Some((index, score))) = (self.scores.as_ref(), extreme) else {
            return;
        };
        log.record_score(GuardrailScore {
            guardrail_name: self.cfg.row_name.clone(),
            hook: hook.to_owned(),
            direction: direction.to_owned(),
            score,
            threshold,
            matched: score >= threshold,
            top_example_index: index as u32,
            embedding_model: self.cfg.embedding_model.clone(),
        });
    }

    /// Score ONE candidate against both lists.
    fn judge(
        &self,
        candidate: &[f32],
        deny_vecs: &[Vec<f32>],
        allow_vecs: &[Vec<f32>],
    ) -> JudgedCandidate {
        let mut judged = JudgedCandidate::default();

        // Deny first, unconditionally: a text matching both lists is
        // refused, never laundered by its allow score.
        judged.deny = closest(candidate, deny_vecs);
        if let Some((index, score)) = judged.deny {
            if score >= self.cfg.deny_threshold {
                judged.verdict = Some(GuardrailVerdict::block(format!(
                    "semantic deny example #{index} matched (similarity {score:.3} \
                     >= threshold {:.3})",
                    self.cfg.deny_threshold
                )));
                return judged;
            }
        }

        if !allow_vecs.is_empty() {
            judged.allow = closest(candidate, allow_vecs);
            // An empty prototype set is impossible here, so the
            // `unwrap_or` is only a total-function guard.
            let best = judged.allow.map_or(f32::NEG_INFINITY, |(_, score)| score);
            if best < self.cfg.allow_threshold {
                judged.verdict = Some(GuardrailVerdict::block(format!(
                    "no semantic allow example matched (best similarity {best:.3} \
                     < threshold {:.3})",
                    self.cfg.allow_threshold
                )));
            }
        }
        judged
    }

    fn on_failure(&self, failure: EmbedFailure, fail_open: bool) -> GuardrailVerdict {
        let tag = failure.as_str();
        tracing::warn!(
            guardrail = "semantic",
            embedding_model = %self.cfg.embedding_model,
            failure = tag,
            fail_open,
            "semantic guardrail could not embed"
        );
        if fail_open {
            GuardrailVerdict::Bypass { reason: tag.into() }
        } else {
            GuardrailVerdict::block_unavailable(
                format!("semantic guardrail unavailable ({tag})"),
                tag,
            )
        }
    }
}

/// One candidate's scores plus the refusal they produced, if any. The two
/// scores are what the telemetry summary folds; the verdict is what the
/// caller enforces. `allow` stays `None` when the deny gate refused first —
/// the allow list was never consulted for that text.
#[derive(Default)]
struct JudgedCandidate {
    deny: Option<(usize, f32)>,
    allow: Option<(usize, f32)>,
    verdict: Option<GuardrailVerdict>,
}

/// The closest prototype and its score. The INDEX is what makes both a
/// block reason and a score entry actionable without echoing either the
/// example or the screened text (#153).
fn closest(candidate: &[f32], prototypes: &[Vec<f32>]) -> Option<(usize, f32)> {
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
            self.cfg.hook_point,
            GuardrailHookPoint::Output | GuardrailHookPoint::Both
        )
    }

    /// This kind reports similarity scores, so it takes a per-request bind.
    fn bind_score_log(&self, log: &Arc<GuardrailAuditLog>) -> Option<Arc<dyn Guardrail>> {
        Some(Arc::new(Self {
            cfg: Arc::clone(&self.cfg),
            scores: Some(Arc::clone(log)),
        }))
    }

    /// Always whole-response hold-back, like kind=pii and kind=presidio.
    /// A similarity judgement is only meaningful on a complete text: a
    /// sliding window would both score partial prose against whole-text
    /// prototypes and release the opening of a response before the
    /// window carrying the violation is judged.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        StreamOutputPolicy::BufferFull {
            max_buffer_bytes: self.cfg.max_buffer_bytes as usize,
            on_exceeded_fail_open: self.cfg.on_buffer_exceeded_fail_open,
        }
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        if !self.hook_enabled(GuardrailHookPoint::Input) {
            return GuardrailVerdict::Allow;
        }
        let texts =
            collect_input_texts(req, self.cfg.scan_all_messages, self.cfg.max_screened_texts);
        self.screen(texts, self.cfg.fail_open, "input").await
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        if !self.hook_enabled(GuardrailHookPoint::Output) {
            return GuardrailVerdict::Allow;
        }
        let text = resp.guardrail_output_text();
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        self.screen(vec![text], self.cfg.output_fail_open, "output")
            .await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use aisix_gateway::{ChatMessage, FinishReason, UsageStats};

    use super::*;

    /// Deterministic stand-in for a real embedding model: a text is
    /// mapped to a one-hot vector by the first TOPIC word it contains, so
    /// cosine is exactly 1.0 within a topic and 0.0 across topics. That
    /// makes every threshold assertion below exact rather than
    /// approximate, and keeps the tests about the DECISION rule rather
    /// than about any model's scoring.
    #[derive(Default)]
    struct StubEmbedder {
        /// Every batch it was asked for, in order — the cache/one-call
        /// assertions read this.
        calls: Mutex<Vec<Vec<String>>>,
        fail: Option<EmbedFailure>,
        wrong_length: bool,
        call_count: AtomicUsize,
    }

    const TOPICS: [&str; 3] = ["jailbreak", "refund", "weather"];

    /// The configured row name every score entry is attributed to.
    const ROW: &str = "semantic-row";

    impl StubEmbedder {
        fn failing(failure: EmbedFailure) -> Self {
            Self {
                fail: Some(failure),
                ..Default::default()
            }
        }

        fn batches(&self) -> Vec<Vec<String>> {
            self.calls.lock().unwrap().clone()
        }
    }

    /// A text carrying this marker lands halfway between the `jailbreak`
    /// axis and the unclassified one, so it scores exactly
    /// `1/sqrt(2) = 0.707` against a jailbreak example: a graded value the
    /// one-hot topics cannot produce, and the only way to check that a
    /// score fold takes the extreme rather than the last value.
    const HALF_JAILBREAK: &str = "borderline";

    fn vector_for(text: &str) -> Vec<f32> {
        let mut v = vec![0.0; TOPICS.len() + 1];
        if text.to_lowercase().contains(HALF_JAILBREAK) {
            v[0] = 1.0;
            v[TOPICS.len()] = 1.0;
            return v;
        }
        for (i, topic) in TOPICS.iter().enumerate() {
            if text.to_lowercase().contains(topic) {
                v[i] = 1.0;
                return v;
            }
        }
        // Unclassified text is its own orthogonal topic, so it matches
        // nothing rather than accidentally matching everything.
        v[TOPICS.len()] = 1.0;
        v
    }

    #[async_trait]
    impl GuardrailEmbedder for StubEmbedder {
        async fn embed(
            &self,
            _model_alias: &str,
            texts: &[String],
            _cacheable: bool,
            _timeout: Duration,
        ) -> Result<Vec<Vec<f32>>, EmbedFailure> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            self.calls.lock().unwrap().push(texts.to_vec());
            if let Some(failure) = self.fail {
                return Err(failure);
            }
            let mut out: Vec<Vec<f32>> = texts.iter().map(|t| vector_for(t)).collect();
            if self.wrong_length {
                out.pop();
            }
            Ok(out)
        }
    }

    fn cfg(deny: &[&str], allow: &[&str]) -> SemanticConfig {
        SemanticConfig {
            embedding_model: "embed-1".into(),
            deny_examples: deny.iter().map(|s| (*s).to_string()).collect(),
            allow_examples: allow.iter().map(|s| (*s).to_string()).collect(),
            deny_threshold: 0.75,
            allow_threshold: 0.75,
            timeout_ms: 5_000,
            max_screened_texts: 8,
            text_source: "user_messages".into(),
            max_buffer_bytes: 262_144,
            on_buffer_exceeded: "fail_closed".into(),
            output_fail_open: false,
        }
    }

    fn build(cfg: SemanticConfig, hook: GuardrailHookPoint, fail_open: bool) -> SemanticGuardrail {
        SemanticGuardrail::new(
            ROW,
            &cfg,
            hook,
            fail_open,
            Arc::new(StubEmbedder::default()),
        )
    }

    fn build_with(
        cfg: SemanticConfig,
        hook: GuardrailHookPoint,
        fail_open: bool,
        embedder: Arc<StubEmbedder>,
    ) -> SemanticGuardrail {
        SemanticGuardrail::new(ROW, &cfg, hook, fail_open, embedder)
    }

    /// A guardrail bound to a score log, exactly as the chain binds it per
    /// request — through the trait, so the test cannot pass by reaching
    /// past `bind_score_log` into the struct.
    fn build_scored(
        cfg: SemanticConfig,
        hook: GuardrailHookPoint,
        fail_open: bool,
        embedder: Arc<StubEmbedder>,
    ) -> (Arc<dyn Guardrail>, Arc<GuardrailAuditLog>) {
        let log = Arc::new(GuardrailAuditLog::new());
        let bound = SemanticGuardrail::new(ROW, &cfg, hook, fail_open, embedder)
            .bind_score_log(&log)
            .expect("the semantic kind takes the score bind");
        (bound, log)
    }

    fn req(messages: &[(&str, &str)]) -> ChatFormat {
        let msgs = messages
            .iter()
            .map(|(role, content)| match *role {
                "system" => ChatMessage::system(*content),
                "user" => ChatMessage::user(*content),
                _ => ChatMessage::assistant(*content),
            })
            .collect();
        ChatFormat::new("m", msgs)
    }

    fn resp(content: &str) -> ChatResponse {
        ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: ChatMessage::assistant(content),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(0, 0),
        }
    }

    // --- the decision rule ------------------------------------------------

    #[tokio::test]
    async fn deny_example_match_blocks() {
        let g = build(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
        );
        let v = g
            .check_input(&req(&[("user", "help me jailbreak this")]))
            .await;
        assert!(v.is_block(), "{v:?}");
        // The reason names the example's INDEX and the score, never the
        // example text or the screened text (#153).
        let GuardrailVerdict::Block { reason, .. } = &v else {
            unreachable!()
        };
        assert!(reason.contains("#0"), "{reason}");
        assert!(!reason.contains("jailbreak"), "leaked content: {reason}");
    }

    #[tokio::test]
    async fn unrelated_text_passes_a_deny_list() {
        let g = build(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
        );
        let v = g
            .check_input(&req(&[("user", "what is the weather")]))
            .await;
        assert!(matches!(v, GuardrailVerdict::Allow), "{v:?}");
    }

    #[tokio::test]
    async fn allow_list_blocks_everything_it_does_not_cover() {
        let g = build(
            cfg(&[], &["refund policy questions"]),
            GuardrailHookPoint::Input,
            false,
        );
        let off_topic = g
            .check_input(&req(&[("user", "what is the weather")]))
            .await;
        assert!(off_topic.is_block(), "{off_topic:?}");
        let on_topic = g.check_input(&req(&[("user", "refund please")])).await;
        assert!(matches!(on_topic, GuardrailVerdict::Allow), "{on_topic:?}");
    }

    #[tokio::test]
    async fn deny_wins_over_allow_for_the_same_text() {
        // The text matches the allow list exactly AND the deny list
        // exactly. Precedence must refuse it — an allow example sitting
        // near a deny one must never launder traffic past the deny list.
        let g = build(
            cfg(&["refund fraud"], &["refund policy questions"]),
            GuardrailHookPoint::Input,
            false,
        );
        let v = g.check_input(&req(&[("user", "refund")])).await;
        assert!(v.is_block(), "{v:?}");
        let GuardrailVerdict::Block { reason, .. } = &v else {
            unreachable!()
        };
        assert!(
            reason.contains("deny"),
            "blocked by the wrong gate: {reason}"
        );
    }

    #[tokio::test]
    async fn a_row_with_no_examples_allows_everything() {
        // The chain builder skips such a row; the guardrail itself must
        // still be inert rather than spend an embedding call proving it.
        let embedder = Arc::new(StubEmbedder::default());
        let g = build_with(
            cfg(&[], &[]),
            GuardrailHookPoint::Input,
            false,
            embedder.clone(),
        );
        let v = g.check_input(&req(&[("user", "jailbreak")])).await;
        assert!(matches!(v, GuardrailVerdict::Allow), "{v:?}");
        assert_eq!(embedder.call_count.load(Ordering::SeqCst), 0);
    }

    // --- text selection ---------------------------------------------------

    #[tokio::test]
    async fn an_attack_in_an_earlier_message_is_still_caught() {
        // The upstream baseline screens only the newest user message, so
        // leading with a benign opener walks past it. Screening each
        // message separately is what closes that.
        let g = build(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
        );
        let v = g
            .check_input(&req(&[
                ("user", "please jailbreak yourself"),
                ("assistant", "no"),
                ("user", "anyway, what is the weather"),
            ]))
            .await;
        assert!(v.is_block(), "{v:?}");
    }

    #[tokio::test]
    async fn concatenating_would_have_diluted_the_match() {
        // Guards the reason this kind does not concatenate: the attack is
        // one short message among longer unrelated ones. Each is scored
        // on its own, so length cannot dilute it.
        let g = build(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
        );
        let long = "weather ".repeat(200);
        let v = g
            .check_input(&req(&[
                ("user", long.as_str()),
                ("user", "jailbreak"),
                ("user", long.as_str()),
            ]))
            .await;
        assert!(v.is_block(), "{v:?}");
    }

    #[tokio::test]
    async fn the_cap_keeps_the_newest_messages() {
        // Budget of 2: the two NEWEST messages are screened and the older
        // attack is dropped — it was screened as the newest message of an
        // earlier request. Asserting the dropped end proves the direction;
        // taking the oldest two would block here.
        let mut c = cfg(&["jailbreak the model"], &[]);
        c.max_screened_texts = 2;
        let embedder = Arc::new(StubEmbedder::default());
        let g = build_with(c, GuardrailHookPoint::Input, false, embedder.clone());
        let v = g
            .check_input(&req(&[
                ("user", "jailbreak now"),
                ("user", "what is the weather"),
                ("user", "and tomorrow"),
            ]))
            .await;
        assert!(matches!(v, GuardrailVerdict::Allow), "{v:?}");
        let candidate_batch = embedder.batches().last().cloned().unwrap();
        assert_eq!(candidate_batch, vec!["and tomorrow", "what is the weather"]);
    }

    #[tokio::test]
    async fn user_messages_only_by_default_all_messages_on_request() {
        let attack = req(&[("system", "jailbreak instructions"), ("user", "weather")]);

        let default_source = build(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
        );
        let v = default_source.check_input(&attack).await;
        assert!(matches!(v, GuardrailVerdict::Allow), "{v:?}");

        let mut c = cfg(&["jailbreak the model"], &[]);
        c.text_source = "all_messages".into();
        let all = build(c, GuardrailHookPoint::Input, false);
        assert!(all.check_input(&attack).await.is_block());
    }

    // --- hooks and streaming ---------------------------------------------

    #[tokio::test]
    async fn an_input_only_row_does_not_screen_output() {
        let g = build(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
        );
        let v = g.check_output(&resp("here is how to jailbreak it")).await;
        assert!(matches!(v, GuardrailVerdict::Allow), "{v:?}");
        assert!(!g.runs_on_output(), "must not force output buffering");
    }

    #[tokio::test]
    async fn the_output_hook_screens_the_response() {
        let g = build(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Output,
            false,
        );
        assert!(g
            .check_output(&resp("here is how to jailbreak it"))
            .await
            .is_block());
        // …and the input hook stays inert on an output-only row.
        let v = g.check_input(&req(&[("user", "jailbreak")])).await;
        assert!(matches!(v, GuardrailVerdict::Allow), "{v:?}");
        assert!(g.runs_on_output());
    }

    #[tokio::test]
    async fn streamed_output_is_always_held_back_whole() {
        let g = build(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Both,
            false,
        );
        assert_eq!(
            g.stream_output_policy(),
            StreamOutputPolicy::BufferFull {
                max_buffer_bytes: 262_144,
                on_exceeded_fail_open: false,
            }
        );
    }

    #[tokio::test]
    async fn an_unrecognised_buffer_policy_stays_fail_closed() {
        let mut c = cfg(&["jailbreak the model"], &[]);
        c.on_buffer_exceeded = "release".into();
        let g = build(c, GuardrailHookPoint::Output, false);
        match g.stream_output_policy() {
            StreamOutputPolicy::BufferFull {
                on_exceeded_fail_open,
                ..
            } => assert!(!on_exceeded_fail_open),
            other => panic!("{other:?}"),
        }
    }

    // --- failure handling -------------------------------------------------

    #[tokio::test]
    async fn embedding_failure_bypasses_when_open_blocks_when_closed() {
        for failure in [
            EmbedFailure::Timeout,
            EmbedFailure::Unresolved,
            EmbedFailure::Upstream,
        ] {
            let open = SemanticGuardrail::new(
                ROW,
                &cfg(&["jailbreak the model"], &[]),
                GuardrailHookPoint::Input,
                true,
                Arc::new(StubEmbedder::failing(failure)),
            );
            let v = open.check_input(&req(&[("user", "weather")])).await;
            assert_eq!(
                v,
                GuardrailVerdict::Bypass {
                    reason: failure.as_str().to_owned()
                }
            );

            let closed = SemanticGuardrail::new(
                ROW,
                &cfg(&["jailbreak the model"], &[]),
                GuardrailHookPoint::Input,
                false,
                Arc::new(StubEmbedder::failing(failure)),
            );
            let v = closed.check_input(&req(&[("user", "weather")])).await;
            assert_eq!(v.unavailable_tag(), Some(failure.as_str()), "{v:?}");
        }
    }

    #[tokio::test]
    async fn the_output_hook_has_its_own_fail_open_switch() {
        // fail_open=true on input, output_fail_open=false (the default):
        // an outage bypasses the request check and still blocks the
        // response one.
        let mut c = cfg(&["jailbreak the model"], &[]);
        c.output_fail_open = false;
        let g = SemanticGuardrail::new(
            ROW,
            &c,
            GuardrailHookPoint::Both,
            true,
            Arc::new(StubEmbedder::failing(EmbedFailure::Timeout)),
        );
        assert!(matches!(
            g.check_input(&req(&[("user", "weather")])).await,
            GuardrailVerdict::Bypass { .. }
        ));
        assert!(g.check_output(&resp("anything")).await.is_block());
    }

    #[tokio::test]
    async fn a_short_vector_batch_is_a_failure_not_a_silent_pass() {
        // A provider that answers with fewer vectors than inputs would
        // otherwise leave some candidate unscored — a screening hole that
        // looks exactly like a clean pass.
        let embedder = Arc::new(StubEmbedder {
            wrong_length: true,
            ..Default::default()
        });
        let g = build_with(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
            embedder,
        );
        let v = g.check_input(&req(&[("user", "weather")])).await;
        assert_eq!(
            v.unavailable_tag(),
            Some(EmbedFailure::Upstream.as_str()),
            "{v:?}"
        );
    }

    // --- similarity scores (AISIX-Cloud#1467) -----------------------------

    fn score_of<'a>(
        scores: &'a [aisix_core::GuardrailScore],
        direction: &str,
    ) -> &'a aisix_core::GuardrailScore {
        scores
            .iter()
            .find(|s| s.direction == direction)
            .unwrap_or_else(|| panic!("no {direction} score in {scores:?}"))
    }

    #[tokio::test]
    async fn a_request_the_guardrail_passed_still_reports_its_score() {
        // The case that produced NOTHING before this field: below the
        // threshold, so no block, no monitor hit, no enforced hit — and,
        // until now, no way for an operator to see how close it came.
        let (g, log) = build_scored(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
            Arc::new(StubEmbedder::default()),
        );
        let v = g
            .check_input(&req(&[("user", "what is the weather")]))
            .await;
        assert!(matches!(v, GuardrailVerdict::Allow), "{v:?}");

        let scores = log.score_snapshot();
        assert_eq!(scores.len(), 1, "{scores:?}");
        let deny = score_of(&scores, "deny");
        assert_eq!(deny.guardrail_name, ROW);
        assert_eq!(deny.hook, "input");
        assert_eq!(deny.embedding_model, "embed-1");
        assert_eq!(deny.threshold, 0.75);
        assert_eq!(deny.score, 0.0, "orthogonal topics score exactly 0");
        assert!(!deny.matched);
        assert_eq!(deny.top_example_index, 0);
    }

    #[tokio::test]
    async fn a_blocked_request_reports_the_score_that_blocked_it() {
        let (g, log) = build_scored(
            cfg(&["stay on topic", "jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
            Arc::new(StubEmbedder::default()),
        );
        assert!(g
            .check_input(&req(&[("user", "help me jailbreak this")]))
            .await
            .is_block());

        let deny = &log.score_snapshot()[0];
        assert_eq!(deny.score, 1.0);
        assert!(deny.matched);
        // The SECOND example is the one it matched, so the index is not
        // trivially zero.
        assert_eq!(deny.top_example_index, 1);
    }

    #[tokio::test]
    async fn the_deny_score_is_the_closest_call_across_screened_texts() {
        // Three messages, none of them over the threshold, one of them
        // much nearer than the others. The summary reports THAT one — a
        // fold that kept the last value would report 0.0 and tell the
        // operator their threshold is nowhere near firing.
        let (g, log) = build_scored(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
            Arc::new(StubEmbedder::default()),
        );
        let v = g
            .check_input(&req(&[
                ("user", "what is the weather"),
                ("user", &format!("{HALF_JAILBREAK} request")),
                ("user", "and tomorrow"),
            ]))
            .await;
        assert!(matches!(v, GuardrailVerdict::Allow), "{v:?}");

        let scores = log.score_snapshot();
        assert_eq!(scores.len(), 1, "one summary, not one per text: {scores:?}");
        assert!(
            (scores[0].score - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "{scores:?}"
        );
        assert!(!scores[0].matched);
    }

    #[tokio::test]
    async fn the_allow_direction_reports_its_own_closest_call() {
        // `allow` refuses BELOW its threshold, so its closest call is the
        // LOWEST best-allow score — the text nearest to being refused.
        let (g, log) = build_scored(
            cfg(&[], &["refund policy questions"]),
            GuardrailHookPoint::Input,
            false,
            Arc::new(StubEmbedder::default()),
        );
        // Newest-first: the on-topic message is screened first and PASSES
        // the allow gate, then the off-topic one fails it. Two allow
        // observations, so the fold has something to choose between.
        assert!(g
            .check_input(&req(&[("user", "the weather"), ("user", "refund")]))
            .await
            .is_block());

        let scores = log.score_snapshot();
        assert_eq!(scores.len(), 1, "{scores:?}");
        let allow = score_of(&scores, "allow");
        assert_eq!(allow.score, 0.0, "the off-topic text is the closest call");
        assert!(
            !allow.matched,
            "matched is `score >= threshold` — false here, and false is what refused",
        );
        assert_eq!(allow.threshold, 0.75);
    }

    #[tokio::test]
    async fn a_deny_refusal_reports_no_allow_score_it_never_computed() {
        // The deny gate short-circuits before the allow list is consulted,
        // so reporting an allow number would describe a comparison that
        // never happened.
        let (g, log) = build_scored(
            cfg(&["refund fraud"], &["refund policy questions"]),
            GuardrailHookPoint::Input,
            false,
            Arc::new(StubEmbedder::default()),
        );
        assert!(g.check_input(&req(&[("user", "refund")])).await.is_block());

        let scores = log.score_snapshot();
        assert_eq!(
            scores
                .iter()
                .map(|s| s.direction.as_str())
                .collect::<Vec<_>>(),
            vec!["deny"],
            "{scores:?}"
        );
    }

    #[tokio::test]
    async fn the_output_hook_scores_under_its_own_hook_name() {
        let (g, log) = build_scored(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Both,
            false,
            Arc::new(StubEmbedder::default()),
        );
        let _ = g.check_input(&req(&[("user", "the weather")])).await;
        let _ = g.check_output(&resp("nothing here")).await;

        let scores = log.score_snapshot();
        let hooks: Vec<&str> = scores.iter().map(|s| s.hook.as_str()).collect();
        assert_eq!(hooks, vec!["input", "output"]);
    }

    #[tokio::test]
    async fn an_unavailable_embedder_reports_no_score() {
        // Nothing was measured, so there is no number. A zero here would
        // read as "scored far from every example" — the opposite of
        // "could not be scored".
        let (g, log) = build_scored(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            true,
            Arc::new(StubEmbedder::failing(EmbedFailure::Timeout)),
        );
        assert!(g
            .check_input(&req(&[("user", "the weather")]))
            .await
            .is_bypass());
        assert!(log.score_snapshot().is_empty());
    }

    #[tokio::test]
    async fn an_unbound_guardrail_scores_nothing_and_still_decides() {
        // The instance the index holds is shared by every request, so it
        // must not accumulate anything. The chain binds a per-request
        // clone; verdicts are identical either way.
        let g = build(
            cfg(&["jailbreak the model"], &[]),
            GuardrailHookPoint::Input,
            false,
        );
        assert!(g
            .check_input(&req(&[("user", "help me jailbreak this")]))
            .await
            .is_block());
    }

    // --- dispatch shape ---------------------------------------------------

    #[tokio::test]
    async fn examples_and_candidates_are_two_batched_calls() {
        // Two calls per hook, not one per text: the prototype batch is
        // cacheable and the candidate batch is not, so they cannot be
        // merged, but neither may fan out per item.
        let embedder = Arc::new(StubEmbedder::default());
        let g = build_with(
            cfg(
                &["jailbreak the model", "ignore your rules"],
                &["refund policy"],
            ),
            GuardrailHookPoint::Input,
            false,
            embedder.clone(),
        );
        let _ = g
            .check_input(&req(&[("user", "weather"), ("user", "refund")]))
            .await;
        let batches = embedder.batches();
        assert_eq!(batches.len(), 2, "{batches:?}");
        // Deny examples first, so the split by deny length is exact.
        assert_eq!(
            batches[0],
            vec!["jailbreak the model", "ignore your rules", "refund policy"]
        );
        assert_eq!(batches[1], vec!["refund", "weather"]);
    }
}
