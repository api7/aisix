//! Local token estimation for usage telemetry (AISIX-Cloud#1074).
//!
//! When an upstream response — streaming or not — carries no `usage`
//! block, the emit paths fall back to counting tokens locally so usage
//! events, post-stream TPM/TPD accounting, and metrics record real
//! numbers instead of silent zeros. Upstream values always win when
//! non-zero (per-field `or` semantics, matching the reference
//! stream-rebuild implementations), and any event that carries an
//! estimated component sets `UsageEvent::usage_estimated`.
//!
//! On the bridged `/v1/responses` and `/v1/messages` paths the estimate
//! is ALSO what the client is told, so the caller cannot read a zero
//! against a bill they cannot reconcile. Those paths fill per counter,
//! exactly as the record does: a counter the upstream reported stands,
//! and one it left at zero is filled even when the other arrived in the
//! same usage block. A total reported beside such a zero no longer adds
//! up once it is filled, so it is recomputed rather than echoed.
//!
//! Counting follows the OpenAI cookbook message scheme: 3 tokens per
//! message, +1 per `name`, +3 reply priming, tool definitions rendered
//! to the de-facto TypeScript-ish form (+9, −4 when a system message is
//! present). The completion side counts the accumulated output text as
//! plain text with no overhead.
//!
//! Images follow OpenAI's vision token formula wherever they appear as a
//! content part, top-level or nested in a tool output, with
//! dimensions read from the header of inline data only. Media and opaque
//! payloads — image, file and audio data, `encrypted_content`, thinking
//! signatures, generated-image bytes — are never tokenized. A string in a
//! text slot counts as text whatever it contains, `data:` URI included,
//! since that is how the upstream bills it.
//!
//! Encoding selection mirrors tiktoken's model map (`gpt-4o`/`o1`/... →
//! `o200k_base`, `gpt-4`/`gpt-3.5` → `cl100k_base`) and falls back to
//! `cl100k_base` for unknown/non-OpenAI models. Counts for models with
//! proprietary tokenizers (Claude, Gemini, ...) are therefore
//! approximations — the `usage_estimated` flag exists so consumers can
//! tell these numbers from upstream-reported ones.

use serde_json::Value;
use tiktoken_rs::tokenizer::{get_tokenizer, Tokenizer};
use tiktoken_rs::CoreBPE;

/// Bound on the output text accumulated for completion-side estimation.
/// ~1 MiB of UTF-8 comfortably exceeds the largest real model outputs
/// (128K tokens ≈ 0.5 MiB of English text); past the cap the estimate
/// becomes a lower bound instead of the buffer growing without limit.
pub const OUTPUT_ACCUMULATION_CAP: usize = 1 << 20;

/// Append `s` to an estimation accumulator, hard-capped at
/// [`OUTPUT_ACCUMULATION_CAP`]: the final fragment is truncated at a
/// char boundary and anything past the cap is dropped (the estimate
/// becomes a lower bound). Checking the cap per push — rather than
/// once per chunk — keeps a single oversized chunk from overshooting
/// the buffer by its full size.
pub fn push_capped(buf: &mut String, s: &str) {
    let remaining = OUTPUT_ACCUMULATION_CAP.saturating_sub(buf.len());
    if remaining == 0 || s.is_empty() {
        return;
    }
    if s.len() <= remaining {
        buf.push_str(s);
        return;
    }
    let mut cut = remaining;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    buf.push_str(&s[..cut]);
}

const TOKENS_PER_MESSAGE: u32 = 3;
const TOKENS_PER_NAME: u32 = 1;
/// "Every reply is primed with `<|start|>assistant<|message|>`."
const REPLY_PRIMING: u32 = 3;
const TOOLS_OVERHEAD: u32 = 9;
/// Rendered tool definitions absorb part of the system message frame.
const TOOLS_WITH_SYSTEM_DISCOUNT: u32 = 4;
/// OpenAI's vision token formula: `low`/`auto` detail is a flat 85;
/// any other detail costs 85 + 170 per 512-px tile of the image after it
/// is fitted within 2000 px on the long side and 768 px on the short one.
const IMAGE_TOKENS_BASE: u32 = 85;
const IMAGE_TOKENS_PER_TILE: u32 = 170;
const IMAGE_MAX_LONG_SIDE: u64 = 2000;
const IMAGE_MAX_SHORT_SIDE: u64 = 768;
const IMAGE_TILE_SIDE: u64 = 512;
/// High detail on a remote URL: the gateway never fetches images, so
/// the dimensions are unknown.
const IMAGE_TOKENS_REMOTE_HIGH: u32 = 250;
/// Dimensions assumed when inline image data has no readable header.
const IMAGE_FALLBACK_DIMENSIONS: (u32, u32) = (300, 300);
/// Decoded bytes inspected for an inline image's dimensions. PNG, GIF and
/// WebP carry them in the first 30 bytes; a JPEG's SOF segment follows any
/// APP segments (EXIF with an embedded thumbnail, multi-segment ICC
/// profiles), each up to 64 KiB, so the bound leaves room for several.
/// Anything past it is never decoded.
const IMAGE_HEADER_DECODE_BOUND: usize = 256 * 1024;

/// The request-side input for prompt-token estimation, captured (moved,
/// not copied where the call site allows) before the response starts and
/// tokenized only if estimation actually runs.
pub enum PromptInput {
    /// OpenAI-shaped chat request (`/v1/chat/completions`).
    Chat(Box<aisix_gateway::chat::ChatFormat>),
    /// Raw Anthropic `/v1/messages` request body.
    Anthropic(Value),
    /// Raw `/v1/responses` request body.
    Responses(Value),
    /// Outbound `/v1/responses` wire bytes, parsed only if estimation
    /// runs: a stream that keeps its estimator to the end would otherwise
    /// hold a second parsed copy of the request for its whole lifetime.
    ResponsesWire(bytes::Bytes),
    /// Outbound `/v1/messages` bytes with the same deferred parsing.
    AnthropicWire(bytes::Bytes),
}

/// Prompt shape + the model name that selects the tokenizer encoding.
/// Cloning shares the captured request: a stream's terminal fill hands a
/// clone to the estimation pool while its guard keeps the original, so a
/// disconnect during the count still leaves the guard able to estimate.
#[derive(Clone)]
pub struct Estimator {
    model: String,
    input: std::sync::Arc<PromptInput>,
}

/// Outcome of [`fill_missing`]: the (possibly filled) token counts and
/// whether estimation supplied any non-zero component.
#[derive(Clone, Copy)]
pub struct Filled {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub estimated: bool,
}

impl Estimator {
    pub fn new(model: impl Into<String>, input: PromptInput) -> Self {
        Self {
            model: model.into(),
            input: std::sync::Arc::new(input),
        }
    }

    /// Count the prompt side of the captured request.
    pub fn count_prompt(&self) -> u32 {
        let bpe = bpe_for(&self.model);
        match &*self.input {
            PromptInput::Chat(req) => count_chat_prompt(bpe, req),
            PromptInput::Anthropic(body) => count_anthropic_prompt(bpe, body),
            PromptInput::Responses(body) => count_responses_prompt(bpe, body),
            PromptInput::ResponsesWire(wire) => {
                count_responses_prompt(bpe, &crate::util::outbound_value(wire))
            }
            PromptInput::AnthropicWire(wire) => {
                count_anthropic_prompt(bpe, &crate::util::outbound_value(wire))
            }
        }
    }

    pub fn count_output(&self, text: &str) -> u32 {
        count_text(&self.model, text)
    }

    /// The size of the prompt text [`Self::count_prompt`] tokenizes, for
    /// [`crate::estimate_pool::run`]: the stored wire bytes, or the captured
    /// request's serialized size up to the inline limit (see
    /// [`crate::estimate_pool::serialized_bytes_bound`] for how far that
    /// bounds it).
    fn prompt_bytes_bound(&self) -> usize {
        use crate::estimate_pool::serialized_bytes_bound;
        match &*self.input {
            PromptInput::Chat(req) => serialized_bytes_bound(&ChatPromptProbe(req)),
            PromptInput::Anthropic(body) | PromptInput::Responses(body) => {
                serialized_bytes_bound(body)
            }
            PromptInput::ResponsesWire(wire) | PromptInput::AnthropicWire(wire) => wire.len(),
        }
    }
}

/// Count `text` as plain text with the model's encoding — the
/// completion-side rule (no message overhead).
pub fn count_text(model: &str, text: &str) -> u32 {
    enc(bpe_for(model), text)
}

/// [`count_text`] summed over `texts`, with the model's encoding resolved
/// once for the batch.
pub fn count_texts<'a>(model: &str, texts: impl IntoIterator<Item = &'a str>) -> u32 {
    let bpe = bpe_for(model);
    texts
        .into_iter()
        .fold(0u32, |n, text| n.saturating_add(enc(bpe, text)))
}

/// What [`count_chat_prompt`] reads of a chat request, for the size probe:
/// the messages as stored and the request's extra fields (`tools`,
/// `tool_choice`). `ChatMessage`'s own `Serialize` writes a block-array
/// message's joined text out as `content`, which would build that text
/// before the probe could stop.
struct ChatPromptProbe<'a>(&'a aisix_gateway::chat::ChatFormat);

struct ChatMessageProbe<'a>(&'a aisix_gateway::chat::ChatMessage);

impl serde::Serialize for ChatPromptProbe<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("ChatPromptProbe", 2)?;
        s.serialize_field(
            "messages",
            &self
                .0
                .messages
                .iter()
                .map(ChatMessageProbe)
                .collect::<Vec<_>>(),
        )?;
        s.serialize_field("extra", &self.0.extra)?;
        s.end()
    }
}

impl serde::Serialize for ChatMessageProbe<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let m = self.0;
        let mut s = serializer.serialize_struct("ChatMessageProbe", 6)?;
        s.serialize_field("role", role_label(&m.role))?;
        s.serialize_field("content", &m.content)?;
        s.serialize_field("content_blocks", &m.content_blocks)?;
        s.serialize_field("name", &m.name)?;
        s.serialize_field("tool_call_id", &m.tool_call_id)?;
        s.serialize_field("extra", &m.extra)?;
        s.end()
    }
}

/// Per-field `or` fill: upstream values win when non-zero; zeros are
/// replaced with local counts. `output_text: None` disables the
/// completion side (nothing was delivered, or nothing was accumulated).
/// Sub-counters (cache/reasoning) are never estimated.
pub fn fill_missing(
    est: &Estimator,
    upstream_prompt: u32,
    upstream_completion: u32,
    output_text: Option<&str>,
) -> Filled {
    let mut filled = Filled {
        prompt_tokens: upstream_prompt,
        completion_tokens: upstream_completion,
        estimated: false,
    };
    if upstream_prompt == 0 {
        let n = est.count_prompt();
        if n > 0 {
            filled.prompt_tokens = n;
            filled.estimated = true;
        }
    }
    if upstream_completion == 0 {
        if let Some(text) = output_text {
            let n = est.count_output(text);
            if n > 0 {
                filled.completion_tokens = n;
                filled.estimated = true;
            }
        }
    }
    filled
}

/// Whether [`fill_missing`] has anything to tokenize for these counters.
pub fn needs_counting(upstream_prompt: u32, upstream_completion: u32, has_output: bool) -> bool {
    upstream_prompt == 0 || (upstream_completion == 0 && has_output)
}

/// An upper bound on the text [`fill_missing`] tokenizes for these inputs.
fn estimate_bytes(
    est: &Estimator,
    upstream_prompt: u32,
    upstream_completion: u32,
    output_text: Option<&str>,
) -> usize {
    let prompt_bytes = if upstream_prompt == 0 {
        est.prompt_bytes_bound()
    } else {
        0
    };
    let output_bytes = match output_text {
        Some(text) if upstream_completion == 0 => text.len(),
        _ => 0,
    };
    prompt_bytes.saturating_add(output_bytes)
}

/// [`fill_missing`] through [`crate::estimate_pool::run`], so a large count
/// never runs on a request worker. The caller waits; the worker's other
/// connections do not. Counters that need no counting return without
/// leaving the worker.
pub async fn fill_missing_pooled(
    est: Estimator,
    upstream_prompt: u32,
    upstream_completion: u32,
    output_text: Option<String>,
) -> Filled {
    if !needs_counting(upstream_prompt, upstream_completion, output_text.is_some()) {
        return Filled {
            prompt_tokens: upstream_prompt,
            completion_tokens: upstream_completion,
            estimated: false,
        };
    }
    let text_bytes = estimate_bytes(
        &est,
        upstream_prompt,
        upstream_completion,
        output_text.as_deref(),
    );
    crate::estimate_pool::run(text_bytes, move || {
        fill_missing(
            &est,
            upstream_prompt,
            upstream_completion,
            output_text.as_deref(),
        )
    })
    .await
}

/// What the cancel guard records for a call whose caller left while its
/// estimate was pending: the usage as the handler's own record would carry
/// it, with the estimate filled in.
#[derive(Default)]
pub(crate) struct CancelledUsage {
    pub usage: aisix_gateway::chat::UsageStats,
    pub usage_estimated: bool,
    /// How long the attempt took to answer, for a family whose attempt is
    /// settled only after the estimate: the row then says the call reached
    /// its upstream and how long it took, as the settled attempt would.
    pub answered_after_ms: Option<u32>,
}

pub(crate) type CancelledUsageFn = Box<dyn FnOnce() -> CancelledUsage + Send>;

impl CancelledUsage {
    /// The upstream's counters with [`fill_missing`]'s result in place.
    pub(crate) fn filled(
        upstream: aisix_gateway::chat::UsageStats,
        filled: Filled,
        answered_after_ms: Option<u32>,
    ) -> Self {
        Self {
            usage: aisix_gateway::chat::UsageStats {
                prompt_tokens: filled.prompt_tokens,
                completion_tokens: filled.completion_tokens,
                ..upstream
            },
            usage_estimated: filled.estimated,
            answered_after_ms,
        }
    }

    /// Write the counters onto the cancel guard's terminal event the way a
    /// handler's own record writes them.
    pub(crate) fn apply(self, event: &mut aisix_obs::UsageEvent) {
        let u = self.usage;
        event.prompt_tokens = u.prompt_tokens;
        event.completion_tokens = aisix_gateway::chat::recorded_completion_tokens(
            u.prompt_tokens,
            u.completion_tokens,
            u.reasoning_tokens,
            u.reasoning_folded_into_completion,
            u.upstream_total_tokens,
        );
        event.total_tokens = u.upstream_total_tokens;
        event.cached_prompt_tokens = u.cached_prompt_tokens;
        event.cache_write_tokens = u.cache_write_tokens;
        event.modality_tokens = u.modality;
        event.cache_creation_ttl = u.cache_creation_ttl;
        event.reasoning_tokens = u.reasoning_tokens;
        event.cache_creation_tokens = u.cache_creation_tokens;
        event.cache_read_tokens = u.cache_read_tokens;
        event.usage_estimated = self.usage_estimated;
        if let Some(ms) = self.answered_after_ms {
            event.upstream_latency_ms = ms;
        }
    }
}

/// The rate-limit layers a non-streamed call commits its tokens to once
/// its usage is known, and the token cost its handler commits for a given
/// usage. Carried into [`run_recorded_on_cancel`] so a caller that leaves
/// during the count still has the call's tokens counted against TPM/TPD.
pub(crate) struct RateCommit {
    limiter: std::sync::Arc<aisix_ratelimit::Limiter>,
    keys: Vec<aisix_ratelimit::BucketKey>,
    tokens: fn(&aisix_gateway::chat::UsageStats) -> u64,
}

impl RateCommit {
    /// The layers of `reservations` that a commit would charge; `None`
    /// when there are none.
    pub(crate) fn new<'a>(
        limiter: &std::sync::Arc<aisix_ratelimit::Limiter>,
        reservations: impl IntoIterator<Item = Option<&'a aisix_ratelimit::MultiReservation>>,
        tokens: fn(&aisix_gateway::chat::UsageStats) -> u64,
    ) -> Option<Self> {
        let keys: Vec<_> = reservations
            .into_iter()
            .flatten()
            .flat_map(aisix_ratelimit::MultiReservation::keys)
            .collect();
        (!keys.is_empty()).then(|| Self {
            limiter: std::sync::Arc::clone(limiter),
            keys,
            tokens,
        })
    }

    /// Charge `usage`'s token cost to every layer, as the handler's own
    /// `commit_tokens` would; the dropped reservations release their
    /// concurrency slots themselves.
    fn commit(&self, usage: &aisix_gateway::chat::UsageStats) {
        let tokens = (self.tokens)(usage);
        for key in &self.keys {
            self.limiter.add_tokens_post_stream(key, tokens);
        }
    }
}

/// Run a non-streamed call's estimate through [`crate::estimate_pool::run`]
/// after its upstream answered. While a pooled count is awaited, a caller
/// that hangs up drops the handler before it records the call or commits
/// its tokens; the cancel guard then runs `job` inline from its `Drop`,
/// records `on_cancel` of the result, and charges `rate` with it, so the
/// row bills the call and TPM/TPD count it instead of reading zero.
pub(crate) async fn run_recorded_on_cancel<R: Send + 'static>(
    text_bytes: usize,
    job: impl Fn() -> R + Send + Sync + 'static,
    on_cancel: impl FnOnce(R) -> CancelledUsage + Send + 'static,
    rate: Option<RateCommit>,
) -> R {
    if text_bytes <= crate::estimate_pool::INLINE_MAX_BYTES {
        return job();
    }
    let job = std::sync::Arc::new(job);
    let inline = std::sync::Arc::clone(&job);
    crate::attribution::publish_cancelled_usage(Box::new(move || {
        let usage = on_cancel(inline());
        if let Some(rate) = rate {
            rate.commit(&usage.usage);
        }
        usage
    }));
    let result = crate::estimate_pool::run(text_bytes, move || job()).await;
    crate::attribution::settle_cancelled_usage();
    result
}

/// [`fill_missing`] for a non-streamed response, recorded by the cancel
/// guard if the caller leaves while it is counted (see
/// [`run_recorded_on_cancel`]). `upstream` is the response's usage as the
/// upstream reported it, whose counters the record carries beside these.
pub(crate) async fn fill_missing_recorded(
    est: Estimator,
    upstream: aisix_gateway::chat::UsageStats,
    output_text: Option<String>,
    rate: Option<RateCommit>,
    // When the caller's attempt began, for a family that settles its
    // attempt only after this count (see `CancelledUsage::answered_after_ms`).
    attempt_started: Option<std::time::Instant>,
) -> Filled {
    let (upstream_prompt, upstream_completion) =
        (upstream.prompt_tokens, upstream.completion_tokens);
    if !needs_counting(upstream_prompt, upstream_completion, output_text.is_some()) {
        return Filled {
            prompt_tokens: upstream_prompt,
            completion_tokens: upstream_completion,
            estimated: false,
        };
    }
    let text_bytes = estimate_bytes(
        &est,
        upstream_prompt,
        upstream_completion,
        output_text.as_deref(),
    );
    let answered_after_ms = attempt_started.map(|t| crate::attempt::measured_ms(t.elapsed()));
    run_recorded_on_cancel(
        text_bytes,
        move || {
            fill_missing(
                &est,
                upstream_prompt,
                upstream_completion,
                output_text.as_deref(),
            )
        },
        move |filled| CancelledUsage::filled(upstream, filled, answered_after_ms),
        rate,
    )
    .await
}

/// A count made ahead of the moment it is needed, keyed by the inputs it
/// was made from. A stream counts before it releases its terminal frame: a
/// client that closes as soon as it reads that frame drops the stream there,
/// and the guard's Drop — which must emit in place and cannot wait — then
/// takes this count instead of counting on the request worker. It answers
/// only for the key it was counted at; for any other the caller counts
/// inline as before.
pub struct CountedAhead<K, V>(Option<(K, V)>);

impl<K, V> Default for CountedAhead<K, V> {
    fn default() -> Self {
        Self(None)
    }
}

impl<K: PartialEq, V> CountedAhead<K, V> {
    /// Count at `key` unless the count already held is for it.
    pub async fn count<Fut: std::future::Future<Output = V>>(
        &mut self,
        key: K,
        count: impl FnOnce() -> Fut,
    ) {
        if self.0.as_ref().is_some_and(|(counted, _)| *counted == key) {
            return;
        }
        let value = count().await;
        self.0 = Some((key, value));
    }

    /// The count held for `key`, else `inline()`.
    pub fn take_or(&mut self, key: &K, inline: impl FnOnce() -> V) -> V {
        match self.0.take() {
            Some((counted, value)) if counted == *key => value,
            _ => inline(),
        }
    }
}

/// What one stream estimate counts: the upstream counters (zero where it
/// reported none) and the delivered output text. The output is identified
/// by its length and a revision: a buffer that only grows is identified by
/// its length alone (revision 0), and one that can be replaced by other
/// text bumps the revision on each replacement.
#[derive(Clone, Copy)]
pub struct EstimateInputs<'a> {
    pub upstream_prompt: u32,
    pub upstream_completion: u32,
    pub output: Option<&'a str>,
    pub output_revision: u32,
}

type CountKey = (u32, u32, Option<(usize, u32)>);

impl EstimateInputs<'_> {
    fn key(&self) -> CountKey {
        (
            self.upstream_prompt,
            self.upstream_completion,
            self.output.map(|t| (t.len(), self.output_revision)),
        )
    }
}

/// A stream's estimator and its count ahead. Each streaming guard builds
/// its [`EstimateInputs`] in one place and passes them to both
/// [`Self::count_ahead`] and [`Self::fill`], so the count ahead cannot
/// drift from what the guard's completion counts.
#[derive(Default)]
pub struct StreamEstimate {
    estimator: Option<Estimator>,
    counted: CountedAhead<CountKey, Filled>,
}

impl StreamEstimate {
    pub fn new(estimator: Option<Estimator>) -> Self {
        Self {
            estimator,
            counted: CountedAhead::default(),
        }
    }

    /// Count the estimate ahead, through [`crate::estimate_pool::run`].
    pub async fn count_ahead(&mut self, inputs: EstimateInputs<'_>) {
        let Self { estimator, counted } = self;
        let Some(est) = estimator.as_ref() else {
            return;
        };
        if !needs_counting(
            inputs.upstream_prompt,
            inputs.upstream_completion,
            inputs.output.is_some(),
        ) {
            return;
        }
        counted
            .count(inputs.key(), || {
                fill_missing_pooled(
                    est.clone(),
                    inputs.upstream_prompt,
                    inputs.upstream_completion,
                    inputs.output.map(str::to_owned),
                )
            })
            .await;
    }

    /// [`fill_missing`], once: from the count ahead when it was counted from
    /// these same inputs, otherwise inline (Drop cannot wait). `None` when
    /// there is no estimator, or it was already used.
    pub fn fill(&mut self, inputs: EstimateInputs<'_>) -> Option<Filled> {
        let est = self.estimator.take()?;
        Some(self.counted.take_or(&inputs.key(), || {
            fill_missing(
                &est,
                inputs.upstream_prompt,
                inputs.upstream_completion,
                inputs.output,
            )
        }))
    }
}

fn bpe_for(model: &str) -> &'static CoreBPE {
    match get_tokenizer(model) {
        Some(Tokenizer::O200kHarmony) => tiktoken_rs::o200k_harmony_singleton(),
        Some(Tokenizer::O200kBase) => tiktoken_rs::o200k_base_singleton(),
        Some(Tokenizer::P50kBase) => tiktoken_rs::p50k_base_singleton(),
        Some(Tokenizer::P50kEdit) => tiktoken_rs::p50k_edit_singleton(),
        Some(Tokenizer::R50kBase | Tokenizer::Gpt2) => tiktoken_rs::r50k_base_singleton(),
        Some(Tokenizer::Cl100kBase) | None => tiktoken_rs::cl100k_base_singleton(),
    }
}

fn clamp(n: usize) -> u32 {
    n.min(u32::MAX as usize) as u32
}

/// Encode in slices of this size. The tokenizer's regex engine is
/// superlinear on long unbroken runs (a multi-MiB single piece burns
/// seconds of CPU) and its vendored wrapper unwraps regex failures —
/// a ~1 MiB whitespace run panics with a fancy-regex StackOverflow.
/// Slicing linearizes the cost and keeps any failure small; each
/// boundary can split at most one token (+1 per slice, negligible).
const ENCODE_SLICE_BYTES: usize = 64 * 1024;

/// Total bytes exactly tokenized per [`enc`] call; the tail beyond it
/// extrapolates at ~4 bytes/token. Real prompts sit far below this
/// (≈256K tokens of English), so accuracy is unaffected — the cap
/// exists so an adversarial multi-MiB piece can't pin a runtime
/// worker inside a Drop guard.
const EXACT_COUNT_BUDGET: usize = 1 << 20;

/// Estimated-token fallback rate for text the exact pass skipped or
/// the tokenizer failed on: the ~4-bytes-per-token rule of thumb.
fn approx_tokens(bytes: usize) -> u32 {
    clamp(bytes / 4)
}

/// Panic-proof, cost-bounded token count. Estimation runs on the
/// request hot path and inside Drop guards (where a second panic
/// during an unwind aborts the process), so the tokenizer must never
/// panic out of here and must never burn unbounded CPU: encode in
/// bounded slices, catch any tokenizer panic, and extrapolate the
/// remainder past the exact-count budget.
fn enc(bpe: &CoreBPE, text: &str) -> u32 {
    let mut n: u32 = 0;
    let mut rest = text;
    let mut budget = EXACT_COUNT_BUDGET;
    while !rest.is_empty() {
        if budget == 0 {
            return n.saturating_add(approx_tokens(rest.len()));
        }
        let mut cut = ENCODE_SLICE_BYTES.min(rest.len());
        while !rest.is_char_boundary(cut) {
            cut += 1;
        }
        let (head, tail) = rest.split_at(cut);
        let counted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bpe.encode_ordinary(head).len()
        }))
        .map(clamp)
        .unwrap_or_else(|_| approx_tokens(head.len()));
        n = n.saturating_add(counted);
        budget = budget.saturating_sub(cut);
        rest = tail;
    }
    n
}

// ---------------------------------------------------------------------
// Chat (`/v1/chat/completions`)
// ---------------------------------------------------------------------

fn count_chat_prompt(bpe: &CoreBPE, req: &aisix_gateway::chat::ChatFormat) -> u32 {
    let mut n: u32 = 0;
    let mut has_system = false;
    for m in &req.messages {
        let role = role_label(&m.role);
        if m.role.is_instruction() {
            has_system = true;
        }
        n = n.saturating_add(TOKENS_PER_MESSAGE);
        n = n.saturating_add(enc(bpe, role));
        if let Some(blocks) = m.content_blocks.as_deref() {
            n = n.saturating_add(count_content_blocks(bpe, blocks));
        } else if let Some(text) = m.content.as_deref() {
            n = n.saturating_add(enc(bpe, text));
        }
        if let Some(name) = m.name.as_deref() {
            n = n
                .saturating_add(TOKENS_PER_NAME)
                .saturating_add(enc(bpe, name));
        }
        if let Some(id) = m.tool_call_id.as_deref() {
            n = n.saturating_add(enc(bpe, id));
        }
        for (key, value) in &m.extra {
            match key.as_str() {
                "tool_calls" | "function_call" => {
                    n = n.saturating_add(count_tool_call_arguments(bpe, value));
                }
                _ => {
                    if let Some(s) = value.as_str() {
                        n = n.saturating_add(enc(bpe, s));
                    }
                }
            }
        }
    }
    n.saturating_add(count_request_extra(
        bpe,
        req.extra.get("tools"),
        req.extra.get("tool_choice"),
        has_system,
    ))
}

fn role_label(role: &aisix_gateway::chat::Role) -> &'static str {
    match role {
        aisix_gateway::chat::Role::System => "system",
        aisix_gateway::chat::Role::Developer => "developer",
        aisix_gateway::chat::Role::User => "user",
        aisix_gateway::chat::Role::Assistant => "assistant",
        aisix_gateway::chat::Role::Tool => "tool",
        aisix_gateway::chat::Role::Function => "function",
    }
}

/// History `tool_calls` / legacy `function_call`: only the call
/// `arguments` strings are counted; the function *names* are covered by
/// the rendered tool definitions.
fn count_tool_call_arguments(bpe: &CoreBPE, value: &Value) -> u32 {
    let mut n: u32 = 0;
    let calls: &[Value] = match value {
        Value::Array(a) => a.as_slice(),
        v @ Value::Object(_) => std::slice::from_ref(v),
        _ => return 0,
    };
    for call in calls {
        let args = call
            .get("function")
            .and_then(|f| f.get("arguments"))
            .or_else(|| call.get("arguments"));
        if let Some(Value::String(s)) = args {
            n = n.saturating_add(enc(bpe, s));
        }
    }
    n
}

// ---------------------------------------------------------------------
// Anthropic (`/v1/messages`)
// ---------------------------------------------------------------------

fn count_anthropic_prompt(bpe: &CoreBPE, body: &Value) -> u32 {
    let mut n: u32 = 0;
    let mut has_system = false;
    if let Some(system) = body.get("system") {
        if !system.is_null() {
            has_system = true;
            n = n.saturating_add(TOKENS_PER_MESSAGE);
            n = n.saturating_add(enc(bpe, "system"));
            n = n.saturating_add(count_content_value(bpe, system));
        }
    }
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for m in messages {
            n = n.saturating_add(TOKENS_PER_MESSAGE);
            if let Some(role) = m.get("role").and_then(|r| r.as_str()) {
                n = n.saturating_add(enc(bpe, role));
            }
            if let Some(content) = m.get("content") {
                n = n.saturating_add(count_content_value(bpe, content));
            }
        }
    }
    n.saturating_add(count_request_extra(
        bpe,
        body.get("tools"),
        body.get("tool_choice"),
        has_system,
    ))
}

// ---------------------------------------------------------------------
// Responses (`/v1/responses`)
// ---------------------------------------------------------------------

/// The Responses API has no cookbook counting scheme; items are close
/// enough to chat messages that each item takes the per-message overhead
/// and its text-bearing fields are counted like message content.
fn count_responses_prompt(bpe: &CoreBPE, body: &Value) -> u32 {
    let mut n: u32 = 0;
    let mut has_system = false;
    if let Some(instructions) = body.get("instructions").and_then(|i| i.as_str()) {
        has_system = true;
        n = n.saturating_add(TOKENS_PER_MESSAGE);
        n = n.saturating_add(enc(bpe, "system"));
        n = n.saturating_add(enc(bpe, instructions));
    }
    match body.get("input") {
        Some(Value::String(text)) => {
            n = n.saturating_add(TOKENS_PER_MESSAGE);
            n = n.saturating_add(enc(bpe, "user"));
            n = n.saturating_add(enc(bpe, text));
        }
        Some(Value::Array(items)) => {
            for item in items {
                // Tool declarations, counted with the request's tools below.
                if aisix_gateway::tool_calls::is_responses_additional_tools_item(item) {
                    continue;
                }
                n = n.saturating_add(TOKENS_PER_MESSAGE);
                match item.get("role").and_then(|r| r.as_str()) {
                    Some(role) => {
                        // Message-shaped item.
                        if role == "system" || role == "developer" {
                            has_system = true;
                        }
                        n = n.saturating_add(enc(bpe, role));
                        if let Some(content) = item.get("content") {
                            n = n.saturating_add(count_content_value(bpe, content));
                        }
                    }
                    // The tools a client tool search loaded are counted with
                    // the request's tools below; the model reads the item
                    // itself as the result naming them.
                    None if aisix_gateway::tool_calls::client_tool_search_item_call_id(
                        item,
                        "tool_search_output",
                    )
                    .is_some() =>
                    {
                        n = n.saturating_add(enc(
                            bpe,
                            &aisix_gateway::tool_calls::client_tool_search_output_text(item),
                        ));
                    }
                    None => {
                        // Non-message item (function_call, function_call_output,
                        // reasoning, ...).
                        n = n.saturating_add(count_part(bpe, item));
                    }
                }
            }
        }
        _ => {}
    }
    let tools = responses_tool_definitions(body);
    n.saturating_add(count_request_extra(
        bpe,
        Some(&tools),
        body.get("tool_choice"),
        has_system,
    ))
}

/// Every tool definition a Responses request declares, top-level and in
/// `additional_tools` items alike, plus the tools a client tool search
/// loaded, with each `namespace` tool's sub-tools standing in for the
/// namespace: each sub-tool reaches the model as a tool of its own, and a
/// client `tool_search` as the function it is offered as.
fn responses_tool_definitions(body: &Value) -> Value {
    let declared =
        aisix_gateway::tool_calls::responses_declared_tools(body.get("tools"), body.get("input"));
    Value::Array(
        declared
            .iter()
            .flat_map(|t| match t.get("type").and_then(Value::as_str) {
                Some("namespace") => t
                    .get("tools")
                    .and_then(Value::as_array)
                    .map(|members| members.iter().collect())
                    .unwrap_or_default(),
                _ => vec![t.as_ref()],
            })
            .map(|t| {
                if aisix_gateway::tool_calls::is_client_tool_search(t) {
                    aisix_gateway::tool_calls::client_tool_search_function(t)
                } else {
                    t.clone()
                }
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------
// Shared content walkers
// ---------------------------------------------------------------------

/// `content` that is either a bare string or an array of typed blocks.
fn count_content_value(bpe: &CoreBPE, content: &Value) -> u32 {
    match content {
        Value::String(s) => enc(bpe, s),
        Value::Array(blocks) => count_content_blocks(bpe, blocks),
        _ => 0,
    }
}

fn count_content_blocks(bpe: &CoreBPE, blocks: &[Value]) -> u32 {
    blocks
        .iter()
        .fold(0u32, |n, b| n.saturating_add(count_value(bpe, b)))
}

/// A value inside a content container: strings count as text, arrays are
/// walked element by element, typed objects go through [`count_part`], and
/// any other object is data counted as its compact JSON.
fn count_value(bpe: &CoreBPE, v: &Value) -> u32 {
    match v {
        Value::String(s) => enc(bpe, s),
        Value::Array(a) => count_content_blocks(bpe, a),
        Value::Object(_) if v.get("type").is_some_and(Value::is_string) => count_part(bpe, v),
        Value::Object(_) => count_json(bpe, v),
        _ => 0,
    }
}

fn count_json(bpe: &CoreBPE, v: &Value) -> u32 {
    serde_json::to_string(v).map_or(0, |s| enc(bpe, &s))
}

/// One typed object — a content part at any depth (top-level message
/// content, a tool output, `tool_result.content`, a screenshot) or a
/// role-less Responses input item. Known part types take their own rule,
/// so an image or file nested in a tool output is priced as media, never
/// tokenized as text; any other object goes through [`count_fields`].
fn count_part(bpe: &CoreBPE, part: &Value) -> u32 {
    let ty = part.get("type").and_then(|t| t.as_str());
    let text_field = |key: &str| {
        part.get(key)
            .and_then(|t| t.as_str())
            .map_or(0, |s| enc(bpe, s))
    };
    match ty {
        Some("text" | "input_text" | "output_text" | "summary_text" | "reasoning_text") => {
            text_field("text")
        }
        Some("refusal") => text_field("refusal"),
        Some("thinking") => aisix_gateway::chat::reasoning_part_slots(part)
            .into_iter()
            .fold(0u32, |n, slot| n.saturating_add(enc(bpe, slot.text))),
        Some("image_url") => {
            let image = part.get("image_url");
            let detail = image.and_then(|i| i.get("detail")).and_then(|d| d.as_str());
            let url = image
                .and_then(|i| i.get("url").or(Some(i)))
                .and_then(|u| u.as_str());
            image_tokens(detail, url.map(ImageSource::Url))
        }
        Some("input_image" | "computer_screenshot" | "image") => {
            let detail = part.get("detail").and_then(|d| d.as_str());
            let source = match part.get("image_url").and_then(|u| u.as_str()) {
                Some(url) => Some(ImageSource::Url(url)),
                None => part.get("source").and_then(anthropic_image_source),
            };
            image_tokens(detail, source)
        }
        Some("document") => count_document(bpe, part),
        // Files and audio are not estimated; their payloads (`file_data`,
        // `data`) are never text.
        Some("redacted_thinking" | "input_file" | "file" | "input_audio") => 0,
        _ => count_fields(bpe, part, ty),
    }
}

/// A typed block's or item's own identifiers and flags.
const TYPED_OBJECT_METADATA_KEYS: &[&str] = &[
    "type",
    "id",
    "status",
    "tool_use_id",
    "cache_control",
    "is_error",
];

/// Opaque payloads at any depth: `encrypted_content` on reasoning,
/// compaction and search-result items, and thinking `signature`s.
const OPAQUE_KEYS: &[&str] = &["encrypted_content", "signature"];

/// Fields holding content parts (tool outputs, `tool_result.content`,
/// reasoning summaries, code interpreter `outputs`), walked part by part.
/// Any other object or array field is an argument object
/// (`tool_use.input`, `computer_call.action`) and counts as its compact
/// JSON.
const CONTENT_CONTAINER_KEYS: &[&str] = &["content", "output", "outputs", "summary"];

fn count_fields(bpe: &CoreBPE, obj: &Value, ty: Option<&str>) -> u32 {
    let Some(obj) = obj.as_object() else {
        return 0;
    };
    let mut n: u32 = 0;
    for (key, v) in obj {
        if OPAQUE_KEYS.contains(&key.as_str())
            || (ty.is_some() && TYPED_OBJECT_METADATA_KEYS.contains(&key.as_str()))
            // The generated image's base64 bytes.
            || (ty == Some("image_generation_call") && key == "result")
        {
            continue;
        }
        n = n.saturating_add(match v {
            Value::String(s) => enc(bpe, s),
            Value::Array(_) | Value::Object(_)
                if CONTENT_CONTAINER_KEYS.contains(&key.as_str()) =>
            {
                count_value(bpe, v)
            }
            Value::Array(_) | Value::Object(_) => count_json(bpe, v),
            _ => 0,
        });
    }
    n
}

/// Anthropic `document`: `title` and `context` are text, and so is a
/// `text` or `content` source; a base64, URL or file source (a PDF) is not
/// estimated.
fn count_document(bpe: &CoreBPE, doc: &Value) -> u32 {
    let text = |key: &str| {
        doc.get(key)
            .and_then(|t| t.as_str())
            .map_or(0, |s| enc(bpe, s))
    };
    let source = doc.get("source");
    let body = match source.and_then(|s| s.get("type")).and_then(|t| t.as_str()) {
        Some("text") => source
            .and_then(|s| s.get("data"))
            .and_then(|d| d.as_str())
            .map_or(0, |s| enc(bpe, s)),
        Some("content") => source
            .and_then(|s| s.get("content"))
            .map_or(0, |c| count_content_value(bpe, c)),
        _ => 0,
    };
    text("title")
        .saturating_add(text("context"))
        .saturating_add(body)
}

// ---------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------

enum ImageSource<'a> {
    /// A URL string: remote (`http(s)://`), a `data:` URI, or anything else.
    Url(&'a str),
    /// Raw base64 image bytes (Anthropic `source.type: base64`).
    Base64(&'a str),
}

fn anthropic_image_source(source: &Value) -> Option<ImageSource<'_>> {
    match source.get("type").and_then(|t| t.as_str()) {
        Some("base64") => source
            .get("data")
            .and_then(|d| d.as_str())
            .map(ImageSource::Base64),
        Some("url") => source
            .get("url")
            .and_then(|u| u.as_str())
            .map(ImageSource::Url),
        _ => None,
    }
}

fn has_prefix_ignore_case(s: &str, prefix: &str) -> bool {
    s.get(..prefix.len())
        .is_some_and(|h| h.eq_ignore_ascii_case(prefix))
}

/// The base64 payload of a `data:<mime>;base64,<payload>` URI (scheme and
/// parameter names are case-insensitive).
fn data_uri_base64(s: &str) -> Option<&str> {
    if !has_prefix_ignore_case(s, "data:") {
        return None;
    }
    let (header, payload) = s[5..].split_once(',')?;
    let marker = header.len().checked_sub(";base64".len())?;
    header
        .get(marker..)?
        .eq_ignore_ascii_case(";base64")
        .then_some(payload)
}

fn is_remote_url(s: &str) -> bool {
    has_prefix_ignore_case(s, "http://") || has_prefix_ignore_case(s, "https://")
}

/// OpenAI's vision token formula. Dimensions are read only from inline
/// data; a remote URL is never fetched.
fn image_tokens(detail: Option<&str>, source: Option<ImageSource<'_>>) -> u32 {
    if matches!(detail.unwrap_or("auto"), "low" | "auto") {
        return IMAGE_TOKENS_BASE;
    }
    let base64 = match source {
        Some(ImageSource::Url(url)) if is_remote_url(url) => return IMAGE_TOKENS_REMOTE_HIGH,
        Some(ImageSource::Url(url)) => data_uri_base64(url),
        Some(ImageSource::Base64(b64)) => Some(b64),
        None => None,
    };
    let (w, h) = base64
        .and_then(inline_image_dimensions)
        .unwrap_or(IMAGE_FALLBACK_DIMENSIONS);
    high_detail_tokens(w, h)
}

fn high_detail_tokens(width: u32, height: u32) -> u32 {
    let (long, short) = (u64::from(width.max(height)), u64::from(width.min(height)));
    // The two downscales compose to a single factor, so each final side
    // is one exact ratio of the originals.
    let (long, short) = if long > IMAGE_MAX_LONG_SIDE {
        if short * IMAGE_MAX_LONG_SIDE > IMAGE_MAX_SHORT_SIDE * long {
            (
                long as f64 * IMAGE_MAX_SHORT_SIDE as f64 / short as f64,
                IMAGE_MAX_SHORT_SIDE as f64,
            )
        } else {
            (
                IMAGE_MAX_LONG_SIDE as f64,
                short as f64 * IMAGE_MAX_LONG_SIDE as f64 / long as f64,
            )
        }
    } else if short > IMAGE_MAX_SHORT_SIDE {
        (
            long as f64 * IMAGE_MAX_SHORT_SIDE as f64 / short as f64,
            IMAGE_MAX_SHORT_SIDE as f64,
        )
    } else {
        (long as f64, short as f64)
    };
    // Resized images have whole pixels.
    let tiles = |side: f64| (side.floor() / IMAGE_TILE_SIDE as f64).ceil() as u32;
    IMAGE_TOKENS_BASE.saturating_add(
        IMAGE_TOKENS_PER_TILE.saturating_mul(tiles(long).saturating_mul(tiles(short))),
    )
}

/// Width and height from the header of a base64-encoded PNG, GIF, JPEG or
/// WebP. PNG, GIF and WebP are read from the first few bytes; only a JPEG
/// decodes further, up to [`IMAGE_HEADER_DECODE_BOUND`] bytes.
fn inline_image_dimensions(base64: &str) -> Option<(u32, u32)> {
    const SHORT_HEADER_BYTES: usize = 48;
    let head = decode_base64_prefix(base64, SHORT_HEADER_BYTES)?;
    let (w, h) = if head.starts_with(&[0xFF, 0xD8]) {
        jpeg_dimensions(&decode_base64_prefix(base64, IMAGE_HEADER_DECODE_BOUND)?)?
    } else {
        png_dimensions(&head)
            .or_else(|| gif_dimensions(&head))
            .or_else(|| webp_dimensions(&head))?
    };
    (w > 0 && h > 0).then_some((w, h))
}

/// Decode at most `max_bytes` from the start of a base64 string.
fn decode_base64_prefix(base64: &str, max_bytes: usize) -> Option<Vec<u8>> {
    use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
    use base64::Engine;
    const ENGINE: GeneralPurpose = GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
    );
    let take = base64.len().min(max_bytes / 3 * 4) / 4 * 4;
    ENGINE.decode(base64.get(..take)?).ok()
}

fn be_u16(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from(u16::from_be_bytes(
        b.get(at..at + 2)?.try_into().ok()?,
    )))
}

fn le_u16(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from(u16::from_le_bytes(
        b.get(at..at + 2)?.try_into().ok()?,
    )))
}

fn le_u24(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at + 3)?;
    Some(u32::from(s[0]) | u32::from(s[1]) << 8 | u32::from(s[2]) << 16)
}

fn png_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    if !b.starts_with(b"\x89PNG\r\n\x1a\n") || b.get(12..16)? != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(b.get(16..20)?.try_into().ok()?);
    let h = u32::from_be_bytes(b.get(20..24)?.try_into().ok()?);
    Some((w, h))
}

fn gif_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    if !(b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a")) {
        return None;
    }
    Some((le_u16(b, 6)?, le_u16(b, 8)?))
}

/// Walk the marker segments to the first start-of-frame (SOF0–SOF15,
/// excluding DHT `C4`, JPG `C8` and DAC `CC`).
fn jpeg_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    if !b.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut i = 2;
    loop {
        if *b.get(i)? != 0xFF {
            return None;
        }
        while *b.get(i)? == 0xFF {
            i += 1;
        }
        let marker = *b.get(i)?;
        i += 1;
        if marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
            continue;
        }
        let len = be_u16(b, i)? as usize;
        if (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            return Some((be_u16(b, i + 5)?, be_u16(b, i + 3)?));
        }
        if len < 2 {
            return None;
        }
        i += len;
    }
}

fn webp_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    if b.get(0..4)? != b"RIFF" || b.get(8..12)? != b"WEBP" {
        return None;
    }
    match b.get(12..16)? {
        b"VP8X" => Some((le_u24(b, 24)? + 1, le_u24(b, 27)? + 1)),
        b"VP8 " => {
            if b.get(23..26)? != [0x9D, 0x01, 0x2A] {
                return None;
            }
            Some((le_u16(b, 26)? & 0x3FFF, le_u16(b, 28)? & 0x3FFF))
        }
        b"VP8L" => {
            if *b.get(20)? != 0x2F {
                return None;
            }
            let bits = u32::from_le_bytes(b.get(21..25)?.try_into().ok()?);
            Some(((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1))
        }
        _ => None,
    }
}

/// Reply priming + tool definitions + `tool_choice`.
fn count_request_extra(
    bpe: &CoreBPE,
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    has_system: bool,
) -> u32 {
    let mut n = REPLY_PRIMING;
    let tools = tools.and_then(|t| t.as_array()).filter(|t| !t.is_empty());
    if let Some(tools) = tools {
        n = n.saturating_add(enc(bpe, &format_tool_definitions(tools)));
        n = n.saturating_add(TOOLS_OVERHEAD);
        if has_system {
            n = n.saturating_sub(TOOLS_WITH_SYSTEM_DISCOUNT);
        }
    }
    match tool_choice {
        Some(Value::String(s)) if s == "none" => n = n.saturating_add(1),
        Some(tc @ Value::Object(_)) => {
            n = n.saturating_add(7);
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .or_else(|| tc.get("name"))
                .and_then(|v| v.as_str());
            if let Some(name) = name {
                n = n.saturating_add(enc(bpe, name));
            }
        }
        _ => {}
    }
    n
}

/// Render tool definitions in the TypeScript-namespace form the OpenAI
/// tokenizer overhead was measured against. Accepts both the OpenAI
/// shape (`{type, function: {name, description, parameters}}`) and flat
/// shapes (Anthropic `{name, description, input_schema}`, Responses
/// `{type, name, description, parameters}`).
fn format_tool_definitions(tools: &[Value]) -> String {
    let mut lines: Vec<String> = vec!["namespace functions {".into(), String::new()];
    for tool in tools {
        if !tool.is_object() {
            continue;
        }
        let function = match tool.get("function") {
            Some(f @ Value::Object(_)) => f.clone(),
            _ => {
                let params = tool
                    .get("input_schema")
                    .or_else(|| tool.get("parameters"))
                    .cloned()
                    .filter(|p| p.is_object())
                    .unwrap_or_else(|| Value::Object(Default::default()));
                serde_json::json!({
                    "name": tool.get("name").cloned().unwrap_or(Value::Null),
                    "description": tool.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": params,
                })
            }
        };
        let Some(name) = function.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        if let Some(desc) = function.get("description").and_then(|d| d.as_str()) {
            if !desc.is_empty() {
                lines.push(format!("// {desc}"));
            }
        }
        let parameters = function.get("parameters").cloned().unwrap_or(Value::Null);
        let has_properties = parameters
            .get("properties")
            .and_then(|p| p.as_object())
            .is_some_and(|p| !p.is_empty());
        if has_properties {
            lines.push(format!("type {name} = (_: {{"));
            lines.push(format_object_parameters(&parameters, 0));
            lines.push("}) => any;".into());
        } else {
            lines.push(format!("type {name} = () => any;"));
        }
        lines.push(String::new());
    }
    lines.push("} // namespace functions".into());
    lines.join("\n")
}

fn format_object_parameters(parameters: &Value, indent: usize) -> String {
    let Some(properties) = parameters.get("properties").and_then(|p| p.as_object()) else {
        return String::new();
    };
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(|r| r.as_array())
        .map(|r| r.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let mut lines: Vec<String> = Vec::new();
    for (key, props) in properties {
        if let Some(desc) = props.get("description").and_then(|d| d.as_str()) {
            lines.push(format!("// {desc}"));
        }
        let question = if required.contains(&key.as_str()) {
            ""
        } else {
            "?"
        };
        lines.push(format!("{key}{question}: {},", format_type(props, indent)));
    }
    let pad = " ".repeat(indent);
    lines
        .iter()
        .map(|l| format!("{pad}{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_type(props: &Value, indent: usize) -> String {
    let enum_values = |props: &Value| -> Option<String> {
        props.get("enum").and_then(|e| e.as_array()).map(|items| {
            items
                .iter()
                .map(|i| match i {
                    Value::String(s) => format!("\"{s}\""),
                    other => format!("\"{other}\""),
                })
                .collect::<Vec<_>>()
                .join(" | ")
        })
    };
    match props.get("type").and_then(|t| t.as_str()) {
        Some("string") => enum_values(props).unwrap_or_else(|| "string".into()),
        Some("array") => match props.get("items") {
            Some(items) => format!("{}[]", format_type(items, indent)),
            None => "any[]".into(),
        },
        Some("object") => format!("{{\n{}\n}}", format_object_parameters(props, indent + 2)),
        Some("integer" | "number") => enum_values(props).unwrap_or_else(|| "number".into()),
        Some("boolean") => "boolean".into(),
        Some("null") => "null".into(),
        _ => "any".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::chat::{ChatFormat, ChatMessage, Role};
    use serde_json::json;

    fn chat_est(model: &str, req: ChatFormat) -> Estimator {
        Estimator::new(model, PromptInput::Chat(Box::new(req)))
    }

    fn msg(role: Role, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: Some(content.to_string()),
            content_blocks: None,
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn the_chat_size_probe_bounds_block_text_without_building_it() {
        let mut big = msg(Role::User, "");
        big.content = None;
        big.content_blocks = Some(vec![
            serde_json::json!({"type": "text", "text": "x".repeat(1 << 20)}),
        ]);
        let est = chat_est("relay-model", ChatFormat::new("relay-model", vec![big]));
        assert_eq!(
            est.prompt_bytes_bound(),
            crate::estimate_pool::INLINE_MAX_BYTES + 1
        );

        let small = chat_est(
            "relay-model",
            ChatFormat::new("relay-model", vec![msg(Role::User, "hello there")]),
        );
        let bound = small.prompt_bytes_bound();
        assert!(bound >= "user".len() + "hello there".len(), "{bound}");
        assert!(bound <= crate::estimate_pool::INLINE_MAX_BYTES, "{bound}");
    }

    #[tokio::test]
    async fn a_count_ahead_answers_only_for_the_inputs_it_was_counted_from() {
        let est = chat_est(
            "relay-model",
            ChatFormat::new("relay-model", vec![msg(Role::User, "hi")]),
        );
        let inputs = |completion: u32, output: &'static str, revision: u32| EstimateInputs {
            upstream_prompt: 0,
            upstream_completion: completion,
            output: Some(output),
            output_revision: revision,
        };
        let mut once = StreamEstimate::new(Some(est.clone()));
        once.count_ahead(inputs(0, "Hello world", 0)).await;

        // Fill each case from a fresh count made at the same inputs.
        async fn filled(est: &Estimator, later: EstimateInputs<'static>) -> Filled {
            let mut ahead = StreamEstimate::new(Some(est.clone()));
            ahead
                .count_ahead(EstimateInputs {
                    upstream_prompt: 0,
                    upstream_completion: 0,
                    output: Some("Hello world"),
                    output_revision: 0,
                })
                .await;
            ahead.fill(later).unwrap()
        }

        let same = filled(&est, inputs(0, "Hello world", 0)).await;
        let inline = fill_missing(&est, 0, 0, Some("Hello world"));
        assert_eq!(
            (same.prompt_tokens, same.completion_tokens, same.estimated),
            (
                inline.prompt_tokens,
                inline.completion_tokens,
                inline.estimated
            ),
        );

        // More output arrived after the count ahead: it must not stand in.
        let grown = filled(&est, inputs(0, "Hello world, and then some more", 0)).await;
        let fresh = fill_missing(&est, 0, 0, Some("Hello world, and then some more"));
        assert_eq!(grown.completion_tokens, fresh.completion_tokens);
        assert_ne!(grown.completion_tokens, same.completion_tokens);

        // A counter the upstream reported after the count ahead stands.
        let reported = filled(&est, inputs(7, "Hello world", 0)).await;
        assert_eq!(reported.completion_tokens, 7);

        // Output replaced by other text of the same length.
        let replaced = filled(&est, inputs(0, "1 2 3 4 5 6", 1)).await;
        let fresh = fill_missing(&est, 0, 0, Some("1 2 3 4 5 6"));
        assert_eq!(replaced.completion_tokens, fresh.completion_tokens);
        assert_ne!(replaced.completion_tokens, same.completion_tokens);

        // An estimate fills once.
        assert!(once.fill(inputs(0, "Hello world", 0)).is_some());
        assert!(once.fill(inputs(0, "Hello world", 0)).is_none());
    }

    #[test]
    fn encoding_selection_mirrors_tiktoken_model_map() {
        assert!(std::ptr::eq(
            bpe_for("gpt-4o-2024-08-06"),
            tiktoken_rs::o200k_base_singleton(),
        ));
        assert!(std::ptr::eq(
            bpe_for("gpt-4"),
            tiktoken_rs::cl100k_base_singleton(),
        ));
        // Unknown / non-OpenAI models fall back to cl100k_base.
        assert!(std::ptr::eq(
            bpe_for("claude-sonnet-4-5"),
            tiktoken_rs::cl100k_base_singleton(),
        ));
        assert!(std::ptr::eq(
            bpe_for(""),
            tiktoken_rs::cl100k_base_singleton(),
        ));
    }

    #[test]
    fn count_text_plain() {
        assert_eq!(count_text("gpt-4", ""), 0);
        // cl100k_base: "Hello" + " world" = 2 tokens.
        assert_eq!(count_text("gpt-4", "Hello world"), 2);
        // Unknown model uses the same fallback encoding.
        assert_eq!(count_text("some-proxy-model", "Hello world"), 2);
    }

    /// Audit HIGH-1 regression: a ~1 MiB unbroken whitespace run makes
    /// the tokenizer's regex engine fail (StackOverflow) and its
    /// vendored wrapper panic. `enc` slices the input and catches the
    /// panic, so estimation must return a sane count instead of
    /// panicking (which inside a Drop guard during an unwind would
    /// abort the whole process).
    #[test]
    fn adversarial_whitespace_run_does_not_panic() {
        let evil = " ".repeat(1 << 20) + "x";
        let n = count_text("gpt-4", &evil);
        assert!(n > 0, "a 1 MiB run must still produce a count, got {n}");
    }

    /// The estimation accumulator is a hard cap: a single oversized
    /// push cannot overshoot it, and the truncation lands on a char
    /// boundary.
    #[test]
    fn push_capped_is_a_hard_cap() {
        let mut buf = String::new();
        push_capped(&mut buf, &"a".repeat(OUTPUT_ACCUMULATION_CAP - 1));
        // Multi-byte char straddling the cap: truncated cleanly, never
        // past the cap, never a panic.
        push_capped(&mut buf, "汉汉汉");
        assert!(buf.len() <= OUTPUT_ACCUMULATION_CAP);
        assert!(buf.is_char_boundary(buf.len()));
        // Full buffer: further pushes are no-ops.
        let len = buf.len();
        push_capped(&mut buf, &"b".repeat(64));
        assert!(buf.len() <= OUTPUT_ACCUMULATION_CAP);
        assert!(buf.len() >= len);
    }

    /// Text past the exact-count budget extrapolates at ~4 bytes/token
    /// instead of burning unbounded CPU: the count keeps growing with
    /// input size beyond the budget.
    #[test]
    fn exact_count_budget_extrapolates_tail() {
        let base = "hello world ".repeat((1 << 20) / 12 + 1); // ≈ budget-sized
        let doubled = base.repeat(2);
        let n1 = count_text("gpt-4", &base);
        let n2 = count_text("gpt-4", &doubled);
        assert!(n2 > n1, "tail beyond the budget must still be counted");
        // The extrapolated tail (~4 B/token) stays within 2× of the
        // exact rate for this corpus (~3 B/token), so the total is the
        // right order of magnitude.
        assert!(n2 < n1 * 3);
    }

    #[test]
    fn chat_prompt_counts_cookbook_overhead() {
        // One user message: 3 (per-message) + 1 ("user") + 1 ("Hello")
        // + 3 (reply priming) = 8.
        let req = ChatFormat::new("gpt-4", vec![msg(Role::User, "Hello")]);
        assert_eq!(chat_est("gpt-4", req).count_prompt(), 8);
    }

    /// Replayed reasoning parts count like `reasoning_content` does:
    /// Mistral's `thinking` array and Databricks' `reasoning` summary are
    /// text, their signatures are not.
    #[test]
    fn chat_prompt_counts_reasoning_parts() {
        let text = "the user wants the weather in Paris";
        let as_parts = |content: serde_json::Value| -> u32 {
            let m: ChatMessage =
                serde_json::from_value(json!({"role": "assistant", "content": content})).unwrap();
            chat_est("gpt-4", ChatFormat::new("gpt-4", vec![m])).count_prompt()
        };
        let mut slot = msg(Role::Assistant, "");
        slot.content = None;
        slot.extra.insert("reasoning_content".into(), json!(text));
        let want = chat_est("gpt-4", ChatFormat::new("gpt-4", vec![slot])).count_prompt();
        assert_eq!(
            as_parts(json!([{"type": "thinking", "signature": "sig",
                "thinking": [{"type": "text", "text": text}]}])),
            want,
        );
        assert_eq!(
            as_parts(json!([{"type": "thinking", "thinking": text, "signature": "sig"}])),
            want,
        );
        assert_eq!(
            as_parts(json!([{"type": "reasoning",
                "summary": [{"type": "summary_text", "text": text, "signature": "sig"}]}])),
            want,
        );
    }

    #[test]
    fn chat_prompt_counts_name_and_tool_calls() {
        let mut named = msg(Role::User, "Hello");
        named.name = Some("alice".into());
        let base = ChatFormat::new("gpt-4", vec![msg(Role::User, "Hello")]);
        let with_name = ChatFormat::new("gpt-4", vec![named]);
        let base_n = chat_est("gpt-4", base).count_prompt();
        let name_n = chat_est("gpt-4", with_name).count_prompt();
        // +1 name overhead +count("alice").
        assert_eq!(
            name_n,
            base_n + TOKENS_PER_NAME + count_text("gpt-4", "alice")
        );

        let mut assistant = msg(Role::Assistant, "");
        assistant.content = None;
        assistant.extra.insert(
            "tool_calls".into(),
            json!([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }]),
        );
        let req = ChatFormat::new("gpt-4", vec![assistant]);
        let n = chat_est("gpt-4", req).count_prompt();
        // 3 + count("assistant") + count(arguments) + 3 priming; the id and
        // function name are intentionally not counted.
        let expected = TOKENS_PER_MESSAGE
            + count_text("gpt-4", "assistant")
            + count_text("gpt-4", "{\"city\":\"Paris\"}")
            + REPLY_PRIMING;
        assert_eq!(n, expected);
    }

    #[test]
    fn tool_definitions_render_and_add_overhead() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["c", "f"]}
                    },
                    "required": ["city"]
                }
            }
        }]);
        let rendered = format_tool_definitions(tools.as_array().unwrap());
        assert_eq!(
            rendered,
            "namespace functions {\n\n// Get the weather\ntype get_weather = (_: {\n\
             // City name\ncity: string,\nunit?: \"c\" | \"f\",\n}) => any;\n\n\
             } // namespace functions"
        );

        let mut req = ChatFormat::new("gpt-4", vec![msg(Role::User, "Hi")]);
        req.extra.insert("tools".into(), tools.clone());
        let with_tools = chat_est("gpt-4", req).count_prompt();
        let plain = chat_est(
            "gpt-4",
            ChatFormat::new("gpt-4", vec![msg(Role::User, "Hi")]),
        )
        .count_prompt();
        assert_eq!(
            with_tools,
            plain + count_text("gpt-4", &rendered) + TOOLS_OVERHEAD
        );

        // Anthropic flat tool shape renders identically.
        let anthropic_tools = json!([{
            "name": "get_weather",
            "description": "Get the weather",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["city"]
            }
        }]);
        assert_eq!(
            format_tool_definitions(anthropic_tools.as_array().unwrap()),
            rendered
        );
    }

    #[test]
    fn tools_with_system_message_discount() {
        let tools = json!([{"type": "function", "function": {"name": "f"}}]);
        let mut with_system = ChatFormat::new(
            "gpt-4",
            vec![msg(Role::System, "Be brief"), msg(Role::User, "Hi")],
        );
        with_system.extra.insert("tools".into(), tools.clone());
        let mut without_system = ChatFormat::new("gpt-4", vec![msg(Role::User, "Hi")]);
        without_system.extra.insert("tools".into(), tools);
        let sys_msg_cost =
            TOKENS_PER_MESSAGE + count_text("gpt-4", "system") + count_text("gpt-4", "Be brief");
        assert_eq!(
            chat_est("gpt-4", with_system).count_prompt(),
            chat_est("gpt-4", without_system).count_prompt() + sys_msg_cost
                - TOOLS_WITH_SYSTEM_DISCOUNT
        );
    }

    #[test]
    fn tools_with_developer_message_discount() {
        let tools = json!([{"type": "function", "function": {"name": "f"}}]);
        let mut with_developer = ChatFormat::new(
            "gpt-4",
            vec![msg(Role::Developer, "Be brief"), msg(Role::User, "Hi")],
        );
        with_developer.extra.insert("tools".into(), tools.clone());
        let mut without_developer = ChatFormat::new("gpt-4", vec![msg(Role::User, "Hi")]);
        without_developer.extra.insert("tools".into(), tools);
        let developer_msg_cost =
            TOKENS_PER_MESSAGE + count_text("gpt-4", "developer") + count_text("gpt-4", "Be brief");
        assert_eq!(
            chat_est("gpt-4", with_developer).count_prompt(),
            chat_est("gpt-4", without_developer).count_prompt() + developer_msg_cost
                - TOOLS_WITH_SYSTEM_DISCOUNT
        );
    }

    #[test]
    fn anthropic_prompt_counts_system_blocks_and_tools() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "system": "Be helpful",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "tool_use", "id": "tu_1", "name": "get_weather",
                     "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "sunny"}
                ]}
            ]
        });
        let wire_bytes = serde_json::to_vec(&body).unwrap();
        let est = Estimator::new("claude-sonnet-4-5", PromptInput::Anthropic(body));
        let wire = Estimator::new(
            "claude-sonnet-4-5",
            PromptInput::AnthropicWire(wire_bytes.into()),
        );
        assert_eq!(wire.count_prompt(), est.count_prompt());
        let m = |s: &str| count_text("claude-sonnet-4-5", s);
        let expected = (TOKENS_PER_MESSAGE + m("system") + m("Be helpful"))
            + (TOKENS_PER_MESSAGE + m("user") + m("Hello"))
            + (TOKENS_PER_MESSAGE
                + m("assistant")
                + m("Hi")
                + m("get_weather")
                + m("{\"city\":\"Paris\"}"))
            + (TOKENS_PER_MESSAGE + m("user") + m("sunny"))
            + REPLY_PRIMING;
        assert_eq!(est.count_prompt(), expected);
    }

    #[test]
    fn responses_prompt_counts_string_and_item_inputs() {
        let simple = json!({"model": "gpt-4o", "input": "Hello"});
        let est = Estimator::new("gpt-4o", PromptInput::Responses(simple));
        let m = |s: &str| count_text("gpt-4o", s);
        assert_eq!(
            est.count_prompt(),
            TOKENS_PER_MESSAGE + m("user") + m("Hello") + REPLY_PRIMING
        );

        let items = json!({
            "model": "gpt-4o",
            "instructions": "Be brief",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "get_weather", "arguments": "{}", "status": "completed"}
            ]
        });
        let wire_bytes = serde_json::to_vec(&items).unwrap();
        let est = Estimator::new("gpt-4o", PromptInput::Responses(items));
        let expected = (TOKENS_PER_MESSAGE + m("system") + m("Be brief"))
            + (TOKENS_PER_MESSAGE + m("user") + m("Hello"))
            + (TOKENS_PER_MESSAGE + m("call_1") + m("get_weather") + m("{}"))
            + REPLY_PRIMING;
        assert_eq!(est.count_prompt(), expected);

        // The wire bytes of the same body count the same, parsed on demand.
        let wire = Estimator::new("gpt-4o", PromptInput::ResponsesWire(wire_bytes.into()));
        assert_eq!(wire.count_prompt(), expected);
    }

    /// Tools declared in an `additional_tools` input item count as tools,
    /// exactly as if the request had declared them at the top level, and a
    /// namespace counts each of its sub-tools, `custom` ones included.
    #[test]
    fn responses_prompt_counts_additional_tools_and_namespace_members() {
        let members = json!([
            {"type": "custom", "name": "exec", "description": "Run code."},
            {"type": "function", "name": "wait", "description": "Wait.",
             "parameters": {"type": "object", "properties": {"ms": {"type": "number"}}}},
        ]);
        let user = json!({"role": "user", "content": "hi"});
        let count =
            |body: Value| Estimator::new("gpt-4o", PromptInput::Responses(body)).count_prompt();
        let flat = count(json!({"input": [user.clone()], "tools": members.clone()}));
        let namespaced = json!([{"type": "namespace", "name": "functions", "tools": members}]);
        let top_level = count(json!({"input": [user.clone()], "tools": namespaced.clone()}));
        let lite = count(json!({"input": [
            {"type": "additional_tools", "role": "developer", "tools": namespaced.clone()},
            user.clone(),
        ]}));
        // Codex re-declaring a tool in a later item replaces it: counted once.
        let redeclared = count(json!({"input": [
            {"type": "additional_tools", "role": "developer", "tools": namespaced.clone()},
            user.clone(),
            {"type": "additional_tools", "role": "developer", "tools": namespaced},
        ]}));
        let bare = count(json!({"input": [user]}));
        assert!(flat > bare);
        assert_eq!(top_level, flat);
        assert_eq!(lite, flat);
        assert_eq!(redeclared, flat);
    }

    /// A client `tool_search` counts as the function it is offered as, and
    /// the tools a search loaded count as tools.
    #[test]
    fn responses_prompt_counts_the_client_search_and_the_tools_it_loaded() {
        let count =
            |body: Value| Estimator::new("gpt-4o", PromptInput::Responses(body)).count_prompt();
        let user = json!({"role": "user", "content": "hi"});
        let search = json!({"type": "tool_search", "execution": "client",
            "description": "Search the deferred tools.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}});
        let member = json!({"type": "function", "name": "read", "description": "Read a page of the docs.",
            "parameters": {"type": "object", "properties": {"page": {"type": "string"}}}});
        let output = |tools: Value| {
            json!({"type": "tool_search_output", "call_id": "ts_1",
            "execution": "client", "status": "completed", "tools": tools})
        };
        let bare = count(json!({"input": [user.clone()]}));
        let declared = count(json!({"input": [user.clone()], "tools": [search.clone()]}));
        let as_function = count(
            json!({"input": [user.clone()], "tools": [{"type": "function",
            "name": "tool_search", "description": search["description"],
            "parameters": search["parameters"]}]}),
        );
        assert!(declared > bare);
        assert_eq!(declared, as_function);
        let loaded_none =
            count(json!({"input": [user.clone(), output(json!([]))], "tools": [search.clone()]}));
        let loaded = count(json!({"input": [user, output(json!([
            {"type": "namespace", "name": "mcp__docs", "tools": [member]},
        ]))], "tools": [search]}));
        // Loading one tool adds at least its definition's description.
        assert!(loaded >= loaded_none + 5, "{loaded} vs {loaded_none}");
    }

    #[test]
    fn fill_missing_or_semantics() {
        let est = chat_est(
            "gpt-4",
            ChatFormat::new("gpt-4", vec![msg(Role::User, "Hello")]),
        );

        // Upstream reported both: untouched, not estimated.
        let f = fill_missing(&est, 12, 34, Some("Hello world"));
        assert_eq!(
            (f.prompt_tokens, f.completion_tokens, f.estimated),
            (12, 34, false)
        );

        // Both missing: both filled.
        let f = fill_missing(&est, 0, 0, Some("Hello world"));
        assert_eq!(f.prompt_tokens, 8);
        assert_eq!(f.completion_tokens, 2);
        assert!(f.estimated);

        // Partial: only the missing side fills.
        let f = fill_missing(&est, 12, 0, Some("Hello world"));
        assert_eq!(
            (f.prompt_tokens, f.completion_tokens, f.estimated),
            (12, 2, true)
        );

        // No output text (nothing delivered): completion stays 0, prompt fills.
        let f = fill_missing(&est, 0, 0, None);
        assert_eq!(
            (f.prompt_tokens, f.completion_tokens, f.estimated),
            (8, 0, true)
        );

        // Empty output text estimates to 0 — not marked estimated for it.
        let f = fill_missing(&est, 5, 0, Some(""));
        assert_eq!(
            (f.prompt_tokens, f.completion_tokens, f.estimated),
            (5, 0, false)
        );
    }

    fn b64(bytes: &[u8]) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut b = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
        b.extend_from_slice(&w.to_be_bytes());
        b.extend_from_slice(&h.to_be_bytes());
        b.extend_from_slice(&[8, 6, 0, 0, 0]);
        b
    }

    fn gif(w: u16, h: u16) -> Vec<u8> {
        let mut b = b"GIF89a".to_vec();
        b.extend_from_slice(&w.to_le_bytes());
        b.extend_from_slice(&h.to_le_bytes());
        b.extend_from_slice(&[0; 8]);
        b
    }

    /// SOI, an APP1 segment of `app1_len` bytes, a DHT segment (C4 must not
    /// be read as a frame), then SOF2.
    fn jpeg(w: u16, h: u16, app1_len: usize) -> Vec<u8> {
        let mut b = vec![0xFF, 0xD8, 0xFF, 0xE1];
        b.extend_from_slice(&((app1_len + 2) as u16).to_be_bytes());
        b.extend(std::iter::repeat_n(0xAB, app1_len));
        b.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x04, 0x00, 0x00]);
        b.extend_from_slice(&[0xFF, 0xC2, 0x00, 0x11, 0x08]);
        b.extend_from_slice(&h.to_be_bytes());
        b.extend_from_slice(&w.to_be_bytes());
        b.extend_from_slice(&[0; 12]);
        b
    }

    fn webp(chunk: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut b = b"RIFF\0\0\0\0WEBP".to_vec();
        b.extend_from_slice(chunk);
        b.extend_from_slice(&[0; 4]);
        b.extend_from_slice(body);
        b
    }

    fn data_uri(mime: &str, bytes: &[u8]) -> String {
        format!("data:{mime};base64,{}", b64(bytes))
    }

    #[test]
    fn high_detail_tile_formula() {
        // Fits already: 1 tile.
        assert_eq!(high_detail_tokens(300, 300), 255);
        assert_eq!(high_detail_tokens(512, 512), 255);
        // Short side 600 ≤ 768, no scaling: 2×2 tiles.
        assert_eq!(high_detail_tokens(800, 600), 765);
        // Short side scaled to 768: 768×768 → 2×2.
        assert_eq!(high_detail_tokens(1024, 1024), 765);
        // Both downscales: 4096×8192 → 1000×2000 → 768×1536 → 2×3.
        assert_eq!(high_detail_tokens(4096, 8192), 1105);
        assert_eq!(high_detail_tokens(8192, 4096), 1105);
        assert_eq!(high_detail_tokens(5000, 3000), 1105);
        // Long side only: 3000×1000 → 2000×667 → 4×2.
        assert_eq!(high_detail_tokens(3000, 1000), 1445);
        // 3000×769 → 2000×512.67, which is 512 whole pixels: 4×1.
        assert_eq!(high_detail_tokens(3000, 769), 765);
    }

    #[test]
    fn image_headers_are_read_from_each_format() {
        let dims = |bytes: &[u8]| inline_image_dimensions(&b64(bytes));
        assert_eq!(dims(&png(1024, 2048)), Some((1024, 2048)));
        assert_eq!(dims(&gif(640, 480)), Some((640, 480)));
        assert_eq!(dims(&jpeg(1920, 1080, 60_000)), Some((1920, 1080)));
        // VP8X: 24-bit (w-1, h-1).
        let mut vp8x = vec![0u8; 4];
        vp8x.extend_from_slice(&[0xFF, 0x0F, 0x00, 0x37, 0x04, 0x00]);
        assert_eq!(dims(&webp(b"VP8X", &vp8x)), Some((4096, 1080)));
        // VP8: frame tag, start code, 14-bit w/h.
        let mut vp8 = vec![0u8, 0, 0, 0x9D, 0x01, 0x2A];
        vp8.extend_from_slice(&800u16.to_le_bytes());
        vp8.extend_from_slice(&600u16.to_le_bytes());
        assert_eq!(dims(&webp(b"VP8 ", &vp8)), Some((800, 600)));
        // VP8L: signature, then 14-bit (w-1) and (h-1).
        let bits: u32 = (1000 - 1) | ((700 - 1) << 14);
        let mut vp8l = vec![0x2F];
        vp8l.extend_from_slice(&bits.to_le_bytes());
        assert_eq!(dims(&webp(b"VP8L", &vp8l)), Some((1000, 700)));
        // Unrecognized, or a JPEG whose frame header lies past the decode
        // bound (five maximal ICC segments ≈ 320 KiB).
        assert_eq!(dims(b"not an image at all"), None);
        let mut far = vec![0xFF, 0xD8];
        for _ in 0..5 {
            far.extend_from_slice(&[0xFF, 0xE2, 0xFF, 0xFF]);
            far.extend(std::iter::repeat_n(0, 0xFFFF - 2));
        }
        far.extend_from_slice(&jpeg(1920, 1080, 0)[2..]);
        assert_eq!(dims(&far), None);
    }

    #[test]
    fn image_token_rules() {
        let png_uri = data_uri("image/png", &png(1024, 2048));
        fn url(u: &str) -> Option<ImageSource<'_>> {
            Some(ImageSource::Url(u))
        }
        // Absent detail is auto; low and auto are flat.
        for detail in [None, Some("low"), Some("auto")] {
            assert_eq!(image_tokens(detail, url(&png_uri)), 85);
        }
        // High (and any unknown value) tiles inline data:
        // 1024×2048 → 768×1536 → 2×3.
        assert_eq!(image_tokens(Some("high"), url(&png_uri)), 1105);
        assert_eq!(image_tokens(Some("original"), url(&png_uri)), 1105);
        assert_eq!(
            image_tokens(
                Some("high"),
                Some(ImageSource::Base64(&b64(&gif(640, 480))))
            ),
            425
        );
        // Remote URLs are never fetched.
        assert_eq!(
            image_tokens(Some("high"), url("https://example.com/a.png")),
            250
        );
        assert_eq!(
            image_tokens(Some("high"), url("HTTP://example.com/a.png")),
            250
        );
        let upper = format!("DATA:image/png;BASE64,{}", b64(&png(1024, 2048)));
        assert_eq!(image_tokens(Some("high"), url(&upper)), 1105);
        // Unreadable inline data falls back to 300×300.
        let junk = data_uri("image/png", b"definitely not a png header");
        assert_eq!(image_tokens(Some("high"), url(&junk)), 255);
        assert_eq!(
            image_tokens(Some("high"), url("data:image/png;base64,!!!")),
            255
        );
        assert_eq!(image_tokens(Some("high"), None), 255);
    }

    #[test]
    fn image_parts_at_every_site_use_the_formula() {
        let png_uri = data_uri("image/png", &png(1024, 2048));
        let raw = b64(&png(1024, 2048));
        let parts = [
            (
                json!({"type": "image_url", "image_url": {"url": png_uri, "detail": "high"}}),
                1105,
            ),
            (
                json!({"type": "image_url", "image_url": {"url": png_uri}}),
                85,
            ),
            (
                json!({"type": "input_image", "image_url": png_uri, "detail": "high"}),
                1105,
            ),
            (
                json!({"type": "input_image", "image_url": png_uri, "detail": "low"}),
                85,
            ),
            (
                json!({"type": "input_image", "file_id": "file_1", "detail": "high"}),
                255,
            ),
            (
                json!({"type": "computer_screenshot", "image_url": png_uri}),
                85,
            ),
            (
                json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": raw}}),
                85,
            ),
        ];
        let bpe = bpe_for("gpt-4o");
        for (part, expected) in parts {
            assert_eq!(count_part(bpe, &part), expected, "{part}");
            // Nested inside a tool output, the same rule applies.
            let wrapped = json!({"type": "function_call_output", "call_id": "c", "output": [part]});
            assert_eq!(
                count_part(bpe, &wrapped),
                expected + count_text("gpt-4o", "c"),
                "{wrapped}"
            );
        }
    }

    #[test]
    fn opaque_payloads_count_zero() {
        let bpe = bpe_for("gpt-4o");
        let blob = "QUJD".repeat(50_000);
        let uri = format!("data:application/octet-stream;base64,{blob}");
        let zero = [
            json!({"type": "reasoning", "summary": [], "encrypted_content": blob}),
            json!({"type": "compaction", "encrypted_content": blob}),
            json!({"type": "thinking", "thinking": "", "signature": blob}),
            json!({"type": "redacted_thinking", "data": blob}),
            json!({"type": "image_generation_call", "result": blob}),
            json!({"type": "input_file", "filename": "a.pdf", "file_data": uri}),
            json!({"type": "file", "file": {"file_data": uri}}),
            json!({"type": "input_audio", "input_audio": {"data": blob, "format": "wav"}}),
            json!({"type": "document", "source": {"type": "base64", "data": blob}}),
        ];
        for v in zero {
            assert_eq!(count_part(bpe, &v), 0, "{v}");
        }
        // A structured (untyped) tool output is data: its compact JSON
        // counts, keys and numbers included.
        let output = json!({"type": "function_call_output", "id": "x",
            "output": {"status": "Hello", "temp": 21}});
        assert_eq!(
            count_part(bpe, &output),
            count_text("gpt-4o", r#"{"status":"Hello","temp":21}"#)
        );
        // Text next to the opaque payloads still counts.
        let reasoning = json!({"type": "reasoning", "encrypted_content": blob,
            "summary": [{"type": "summary_text", "text": "Hello"}]});
        assert_eq!(count_part(bpe, &reasoning), count_text("gpt-4o", "Hello"));
    }

    /// AISIX-Cloud#1816: a base64 image inside a tool output, next to an
    /// encrypted reasoning item, used to be tokenized as text (millions of
    /// tokens for a few MiB of base64).
    #[test]
    fn responses_tool_output_image_is_not_text() {
        let mut big = png(1024, 2048);
        big.extend(std::iter::repeat_n(0x5A, 3 << 20));
        let body = json!({
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "Hello"},
                {"type": "reasoning", "id": "rs_1", "summary": [],
                 "encrypted_content": "gAAAA".repeat(200_000)},
                {"type": "custom_tool_call_output", "call_id": "call_1", "output": [
                    {"type": "input_text", "text": "screenshot"},
                    {"type": "input_image", "image_url": data_uri("image/png", &big), "detail": "high"}
                ]}
            ]
        });
        let m = |s: &str| count_text("gpt-4o", s);
        let expected = (TOKENS_PER_MESSAGE + m("user") + m("Hello"))
            + TOKENS_PER_MESSAGE
            + (TOKENS_PER_MESSAGE + m("call_1") + m("screenshot") + 1105)
            + REPLY_PRIMING;
        let est = Estimator::new("gpt-4o", PromptInput::Responses(body));
        assert_eq!(est.count_prompt(), expected);
    }

    #[test]
    fn anthropic_tool_result_image_is_not_text() {
        let mut big = png(800, 600);
        big.extend(std::iter::repeat_n(0x5A, 3 << 20));
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": [
                    {"type": "text", "text": "done"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                     "data": b64(&big)}}
                ]}
            ]}]
        });
        let m = |s: &str| count_text("claude-sonnet-4-5", s);
        let est = Estimator::new("claude-sonnet-4-5", PromptInput::Anthropic(body));
        assert_eq!(
            est.count_prompt(),
            TOKENS_PER_MESSAGE + m("user") + m("done") + 85 + REPLY_PRIMING
        );
    }

    /// A `data:` URI outside a media slot is text the upstream bills, so it
    /// counts as text: wrapping a prompt in one and disconnecting before the
    /// usage event must not zero the estimate.
    #[test]
    fn data_uri_in_text_slots_counts_as_text() {
        let bpe = bpe_for("gpt-4o");
        let uri = format!(
            "data:text/plain;base64,{}",
            b64(&b"hello world ".repeat(2_000))
        );
        let m = count_text("gpt-4o", &uri);
        assert!(m > 1_000, "{m}");
        let est = Estimator::new(
            "gpt-4o",
            PromptInput::Responses(json!({"model": "gpt-4o", "instructions": uri, "input": uri})),
        );
        assert_eq!(
            est.count_prompt(),
            (TOKENS_PER_MESSAGE + count_text("gpt-4o", "system") + m)
                + (TOKENS_PER_MESSAGE + count_text("gpt-4o", "user") + m)
                + REPLY_PRIMING
        );
        for part in [
            json!({"type": "function_call_output", "output": uri}),
            json!({"type": "tool_result", "content": uri}),
            json!({"type": "text", "text": uri}),
        ] {
            assert_eq!(count_part(bpe, &part), m, "{part}");
        }
        let anthropic = Estimator::new(
            "gpt-4o",
            PromptInput::Anthropic(json!({"messages": [{"role": "user", "content": uri}]})),
        );
        assert_eq!(
            anthropic.count_prompt(),
            TOKENS_PER_MESSAGE + count_text("gpt-4o", "user") + m + REPLY_PRIMING
        );
    }

    /// Argument objects count as their compact JSON, keys and numbers
    /// included; only content containers are walked part by part.
    #[test]
    fn argument_objects_count_as_json() {
        let bpe = bpe_for("gpt-4o");
        let m = |s: &str| count_text("gpt-4o", s);
        let tool_use = json!({"type": "tool_use", "id": "tu_1", "name": "f",
            "input": {"city": "Paris", "days": 3}});
        assert_eq!(
            count_part(bpe, &tool_use),
            m("f") + m(r#"{"city":"Paris","days":3}"#)
        );
        let server_tool_use = json!({"type": "server_tool_use", "id": "srv_1", "name": "web_search",
            "input": {"query": "rust"}});
        assert_eq!(
            count_part(bpe, &server_tool_use),
            m("web_search") + m(r#"{"query":"rust"}"#)
        );
        let computer_call = json!({"type": "computer_call", "call_id": "c1",
            "action": {"type": "click", "x": 512, "y": 384, "button": "left"}});
        assert_eq!(
            count_part(bpe, &computer_call),
            m("c1") + m(r#"{"button":"left","type":"click","x":512,"y":384}"#)
        );
        // A data URI inside an argument object is part of that JSON text.
        let uri = format!("data:image/png;base64,{}", "QUJD".repeat(1_000));
        let with_uri = json!({"type": "tool_use", "name": "f", "input": {"img": uri}});
        assert_eq!(
            count_part(bpe, &with_uri),
            m("f") + m(&format!(r#"{{"img":"{uri}"}}"#))
        );
        // Code interpreter `outputs` is a content container: its image is
        // priced as media, not tokenized.
        let code = json!({"type": "code_interpreter_call", "id": "ci_1", "code": "plot()",
            "outputs": [{"type": "image", "url": uri}]});
        assert_eq!(count_part(bpe, &code), m("plot()") + 85);
    }

    #[test]
    fn anthropic_document_counts_its_text() {
        let bpe = bpe_for("gpt-4o");
        let m = |s: &str| count_text("gpt-4o", s);
        let pdf = json!({"type": "document", "title": "Report", "context": "Q3 figures",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0x".repeat(10_000)}});
        assert_eq!(count_part(bpe, &pdf), m("Report") + m("Q3 figures"));
        let url = json!({"type": "document", "source": {"type": "url", "url": "https://example.com/a.pdf"}});
        assert_eq!(count_part(bpe, &url), 0);
        let text = json!({"type": "document", "title": "Notes",
            "source": {"type": "text", "media_type": "text/plain", "data": "The cat sat."}});
        assert_eq!(count_part(bpe, &text), m("Notes") + m("The cat sat."));
        let content_str =
            json!({"type": "document", "source": {"type": "content", "content": "Hello"}});
        assert_eq!(count_part(bpe, &content_str), m("Hello"));
        let content_parts = json!({"type": "document", "source": {"type": "content", "content": [
            {"type": "text", "text": "Hello"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD".repeat(50_000)}}
        ]}});
        assert_eq!(count_part(bpe, &content_parts), m("Hello") + 85);
        // Nested in a tool_result, the same rule applies.
        let nested = json!({"type": "tool_result", "tool_use_id": "t", "content": [text]});
        assert_eq!(count_part(bpe, &nested), m("Notes") + m("The cat sat."));
    }
}
