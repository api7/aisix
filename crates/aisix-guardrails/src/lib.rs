//! aisix-guardrails — pluggable content-policy hooks.
//!
//! Two phases per request (spec §6):
//! - **input**: runs after auth + rate-limit but before bridge dispatch
//!   so a blocked prompt never reaches the upstream. A block here also
//!   short-circuits the cache write — no point storing a refusal.
//! - **output**: runs after the upstream response lands, before the
//!   cache write and the JSON render. Lets policies inspect the
//!   model's text and refuse if it crosses a line.
//!
//! Implementations:
//! - [`KeywordBlocklist`] — case-insensitive literal or regex patterns.
//! - [`GuardrailChain`] — composes multiple guardrails; first
//!   [`GuardrailVerdict::Block`] short-circuits.
//! - [`GuardrailIndex`] — P0c: resolves the per-request chain from a
//!   snapshot of guardrail definitions + attachment rows.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

#[cfg(feature = "aliyun-text-moderation")]
mod aliyun;
#[cfg(feature = "aliyun-text-moderation")]
mod aliyun_ai_guardrail;
mod audit;
#[cfg(feature = "bedrock")]
mod bedrock;
mod build;
mod chain;
#[cfg(any(feature = "azure-content-safety", feature = "aliyun-text-moderation"))]
mod chunk;
mod index;
mod keyword;
#[cfg(feature = "lakera")]
mod lakera;
#[cfg(feature = "local-model")]
mod local_model;
#[cfg(feature = "openai-moderation")]
mod openai_moderation;
mod pii;
#[cfg(feature = "presidio")]
mod presidio;
#[cfg(feature = "azure-content-safety")]
mod prompt_shield;
mod semantic;
#[cfg(feature = "azure-content-safety")]
mod text_moderation;
mod too_large;

use aisix_core::models::GuardrailMonitorHit;
use aisix_gateway::{ChatFormat, ChatMessage, ChatResponse};
use async_trait::async_trait;

/// Max bytes of an upstream guardrail-provider error body to echo into a log
/// line. Mirrors nginx's single-error-line cap (`NGX_MAX_ERROR_STR` = 2048) so
/// a verbose HTML error page or stack trace can't blow up the log.
pub(crate) const MAX_ERROR_BODY_LOG_BYTES: usize = 2048;

/// Truncate a guardrail-provider error body for logging: at most
/// [`MAX_ERROR_BODY_LOG_BYTES`] bytes, cut on a UTF-8 char boundary so a
/// multi-byte character is never split. Returned verbatim otherwise — the
/// whole point is to surface the provider's actual reason (e.g. Aliyun's
/// `InvalidAccessKeyId.NotFound`) that a bare status code hides.
pub(crate) fn truncate_error_body_for_log(body: &str) -> &str {
    if body.len() <= MAX_ERROR_BODY_LOG_BYTES {
        return body;
    }
    let mut end = MAX_ERROR_BODY_LOG_BYTES;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    &body[..end]
}

/// Read a guardrail provider's error-response body for logging, stopping once
/// [`MAX_ERROR_BODY_LOG_BYTES`] have arrived. We only ever log a snippet, so a
/// broken provider returning a huge 4xx body can't make us buffer the whole
/// thing. Reads chunk-by-chunk and gives up on the first read error — this is
/// best-effort diagnostics on a path that's already returning an error.
#[cfg(any(
    feature = "azure-content-safety",
    feature = "aliyun-text-moderation",
    feature = "lakera",
    feature = "openai-moderation",
    feature = "presidio",
))]
pub(crate) async fn read_error_body_capped(mut resp: reqwest::Response) -> String {
    truncate_error_body_for_log(&read_body_capped(&mut resp, MAX_ERROR_BODY_LOG_BYTES).await)
        .to_owned()
}

/// Read at most `cap` bytes of a response body, chunk by chunk, giving up on
/// the first read error.
///
/// Split out from [`read_error_body_capped`] because a caller that PARSES the
/// body needs a different budget from one that logs a snippet of it: a snippet
/// can stop anywhere, whereas a truncated body may simply not contain the field
/// being looked for. See `aliyun::MAX_ERROR_BODY_PARSE_BYTES`.
#[cfg(any(
    feature = "azure-content-safety",
    feature = "aliyun-text-moderation",
    feature = "lakera",
    feature = "openai-moderation",
    feature = "presidio",
))]
pub(crate) async fn read_body_capped(resp: &mut reqwest::Response, cap: usize) -> String {
    let mut buf: Vec<u8> = Vec::new();
    while buf.len() < cap {
        match resp.chunk().await {
            Ok(Some(chunk)) => buf.extend_from_slice(&chunk),
            Ok(None) | Err(_) => break,
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// The text a guardrail should scan for one message.
///
/// Scans every text surface the provider bridges can forward upstream, so
/// a caller can't hide a payload in one field while a benign value sits in
/// another. These are independent wire fields and the bridges forward
/// whichever is present:
///   * flat `content`;
///   * the `text`-type entries of `content_blocks` (empty `content` with
///     the text only in blocks is the round-trip shape, #465; a benign
///     `content` plus a payload in blocks is the split-field bypass);
///   * `extra["tool_calls"]` — history-replay tool calls travel upstream
///     verbatim through `extra` (the OpenAI bridge flattens them, the
///     Anthropic bridge translates them into `tool_use` blocks). The whole
///     payload is serialized so neither a function name nor an argument
///     can hide a banned token, matching `ChatResponse::guardrail_output_text`
///     and `redact_chat_format`, which already cover this surface.
///
/// Non-text content blocks (image/audio) are out of scope — multimodal
/// moderation is a separate feature. Every guardrail's input/output
/// collector goes through this so the families can't drift.
pub(crate) fn message_scan_text(m: &ChatMessage) -> String {
    let mut parts: Vec<String> = Vec::new();
    let content = m.content_str();
    if !content.is_empty() {
        parts.push(content.to_string());
    }
    if let Some(blocks) = m.content_blocks.as_ref() {
        parts.extend(
            blocks
                .iter()
                .filter(|b| b.get("type").and_then(serde_json::Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(serde_json::Value::as_str))
                .map(str::to_string),
        );
    }
    if let Some(tool_calls) = m.extra.get("tool_calls") {
        if !tool_calls.is_null() {
            parts.push(tool_calls.to_string());
        }
    }
    parts.join("\n")
}

/// The guardrail `kind` discriminators compiled into this binary whose
/// availability is decided at COMPILE time.
///
/// Every non-keyword kind sits behind a cargo feature (see `build.rs`'s
/// `BuildError::FeatureDisabled` arms); a DP built without one silently
/// rejects rows of that kind while the dashboard still offers it
/// (#519 B.6). The heartbeat reports this list so cp-api can hide /
/// flag kinds the connected DP can't serve. Strings MUST stay equal to
/// the serde `kind` tags in `aisix_core::models::GuardrailKind`
/// (`GuardrailKind::kind_str`).
///
/// `smart_redaction` is deliberately NOT here even when the `local-model`
/// feature is compiled in: serving it also needs the model bundle on
/// disk, a RUNTIME fact — heartbeat callers report
/// [`supported_kinds_with`] instead.
pub fn supported_kinds() -> &'static [&'static str] {
    &[
        "keyword",
        "pii",
        #[cfg(feature = "azure-content-safety")]
        "azure_content_safety",
        #[cfg(feature = "azure-content-safety")]
        "azure_content_safety_text_moderation",
        #[cfg(feature = "aliyun-text-moderation")]
        "aliyun_text_moderation",
        #[cfg(feature = "aliyun-text-moderation")]
        "aliyun_ai_guardrail",
        #[cfg(feature = "bedrock")]
        "bedrock",
        #[cfg(feature = "lakera")]
        "lakera",
        #[cfg(feature = "openai-moderation")]
        "openai_moderation",
        #[cfg(feature = "presidio")]
        "presidio",
        // No cargo feature and no on-disk asset: the embedding call goes
        // out over the provider bridges every build already has.
        "semantic",
    ]
}

/// [`supported_kinds`] plus the runtime-conditional `smart_redaction` kind:
/// advertised only while the node's model bundle stays verified (the
/// `LocalModelCapability` the runtime hands the heartbeat). Readiness
/// means "could serve" — the control plane creating a smart_redaction row is
/// what triggers the lazy engine load, so gating the advert on "already
/// active" would deadlock the create flow behind its own greying.
pub fn supported_kinds_with(smart_redaction_ready: bool) -> Vec<&'static str> {
    let mut kinds = supported_kinds().to_vec();
    #[cfg(feature = "local-model")]
    if smart_redaction_ready {
        kinds.push("smart_redaction");
    }
    #[cfg(not(feature = "local-model"))]
    let _ = smart_redaction_ready;
    kinds
}

/// Why a guardrail embedding dispatch produced no vector.
///
/// A closed vocabulary on purpose: the tag rides a `Bypass` reason and
/// the `unavailable` field of a `Block`, both of which reach metric
/// labels and the error envelope, so it must never carry free text or
/// screened content (#153).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedFailure {
    /// The alias names no `embedding`-kind Model in this environment, or
    /// the model has no usable provider credential.
    Unresolved,
    /// The embedding call did not answer inside the configured deadline.
    Timeout,
    /// The embedding call reached the provider and failed there, or
    /// answered with an unusable vector.
    Upstream,
}

impl EmbedFailure {
    /// The bounded tag used in verdict reasons and metric labels.
    pub fn as_str(self) -> &'static str {
        match self {
            EmbedFailure::Unresolved => "semantic_embed_unresolved",
            EmbedFailure::Timeout => "semantic_embed_timeout",
            EmbedFailure::Upstream => "semantic_embed_upstream",
        }
    }
}

/// Embeds text through the gateway's own provider bridges, for
/// `kind: "semantic"`.
///
/// The dispatch this needs is the one semantic ROUTING already performs
/// (`aisix_proxy::semantic::embed_texts`), but that lives a layer up: it
/// needs the provider hub and the model snapshot, and holding a
/// `ProxyState` from here would close a reference cycle through the
/// chain cache `ProxyState` itself owns. So the proxy injects an
/// implementation the same way it injects [`LocalModelRuntimeSlot`], and
/// the implementation keeps only the hub plus a snapshot handle.
#[async_trait]
pub trait GuardrailEmbedder: Send + Sync + 'static {
    /// Embed `texts` with the `embedding`-kind Model aliased
    /// `model_alias`, returning one vector per input, in input order.
    ///
    /// `cacheable` marks CONFIG-derived text — the example prototypes,
    /// which are fixed per row and worth memoising process-wide so a
    /// chain rebuild does not re-embed them. Request-derived text passes
    /// `false`: its cardinality is unbounded and caching it would grow
    /// without limit.
    async fn embed(
        &self,
        model_alias: &str,
        texts: &[String],
        cacheable: bool,
        timeout: std::time::Duration,
    ) -> Result<Vec<Vec<f32>>, EmbedFailure>;
}

/// The process-wide guardrail embedder, passed to the chain builders.
/// Always constructible — a caller that has no dispatch to offer passes
/// [`GuardrailEmbedderSlot::none`] and every `kind: "semantic"` row is
/// skipped with a warning (`BuildError::RuntimeUnavailable`).
#[derive(Clone, Default)]
pub struct GuardrailEmbedderSlot {
    embedder: Option<std::sync::Arc<dyn GuardrailEmbedder>>,
}

impl GuardrailEmbedderSlot {
    /// No embedder: `semantic` rows cannot be served.
    pub fn none() -> Self {
        Self::default()
    }

    /// An embedder: `semantic` rows compile against it.
    pub fn new(embedder: std::sync::Arc<dyn GuardrailEmbedder>) -> Self {
        Self {
            embedder: Some(embedder),
        }
    }

    pub(crate) fn get(&self) -> Option<&std::sync::Arc<dyn GuardrailEmbedder>> {
        self.embedder.as_ref()
    }
}

impl std::fmt::Debug for GuardrailEmbedderSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardrailEmbedderSlot")
            .field("present", &self.embedder.is_some())
            .finish()
    }
}

/// The process-wide smart-redaction runtime, passed to the chain
/// builders. Always constructible — a build without the `local-model`
/// feature, or a node without a verified model bundle, passes
/// [`LocalModelRuntimeSlot::none`] and every `kind: "smart_redaction"` row is
/// skipped with a warning (`BuildError::RuntimeUnavailable` /
/// `FeatureDisabled`).
#[derive(Clone, Default)]
pub struct LocalModelRuntimeSlot {
    #[cfg(feature = "local-model")]
    runtime: Option<std::sync::Arc<local_model::LocalModelRuntime>>,
}

impl LocalModelRuntimeSlot {
    /// No runtime: smart_redaction rows cannot be served.
    pub fn none() -> Self {
        Self::default()
    }

    /// A verified runtime: smart_redaction rows compile against it.
    #[cfg(feature = "local-model")]
    pub fn new(runtime: std::sync::Arc<local_model::LocalModelRuntime>) -> Self {
        Self {
            runtime: Some(runtime),
        }
    }

    #[cfg(feature = "local-model")]
    pub(crate) fn get(&self) -> Option<&std::sync::Arc<local_model::LocalModelRuntime>> {
        self.runtime.as_ref()
    }
}

impl std::fmt::Debug for LocalModelRuntimeSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(feature = "local-model")]
        return f
            .debug_struct("LocalModelRuntimeSlot")
            .field("present", &self.runtime.is_some())
            .finish();
        #[cfg(not(feature = "local-model"))]
        f.debug_struct("LocalModelRuntimeSlot").finish()
    }
}

#[cfg(feature = "aliyun-text-moderation")]
pub use aliyun::AliyunTextModerationGuardrail;
#[cfg(feature = "aliyun-text-moderation")]
pub use aliyun_ai_guardrail::AliyunAiGuardrail;
pub use audit::GuardrailAuditLog;
#[cfg(feature = "bedrock")]
pub use bedrock::BedrockGuardrail;
pub use build::{
    build_chain_from_snapshot, build_index_from_snapshot, LiveGuardrailChain, LiveGuardrailIndex,
};
pub use chain::GuardrailChain;
pub use index::{GuardrailIndex, RequestContext};
pub use keyword::{KeywordBlocklist, KeywordRule};
#[cfg(feature = "lakera")]
pub use lakera::LakeraGuardrail;
#[cfg(feature = "local-model")]
pub use local_model::{
    parse_lanes, CategoryCompileError, LocalModelCapability, LocalModelError, LocalModelRuntime,
    ModelManifest, SmartRedactionGuardrail, DEFAULT_MODEL_DIR, LANES_ENV, MODEL_DIR_ENV,
    PROTOTYPES_ENV, RULE_WINDOW_ENV, THRESHOLD_ENV,
};
#[cfg(feature = "openai-moderation")]
pub use openai_moderation::OpenaiModerationGuardrail;
pub use pii::{builtin_rule, PiiAction, PiiGuardrail, PiiRule, BUILTIN_DETECTORS};
#[cfg(feature = "presidio")]
pub use presidio::PresidioGuardrail;
#[cfg(feature = "azure-content-safety")]
pub use prompt_shield::PromptShieldGuardrail;
pub use semantic::SemanticGuardrail;
#[cfg(feature = "azure-content-safety")]
pub use text_moderation::TextModerationGuardrail;

/// What a guardrail decided about a request or response.
///
/// `Bypass` exists for remote-API guardrails (kind=bedrock) whose
/// upstream is unreachable but the operator configured `fail_open=true`:
/// the request goes through, but the bypass is recorded on the
/// telemetry event so a compliance audit can see what slipped past.
/// `Bypass` is **not** a block — the chain doesn't short-circuit on
/// it, and other guardrails downstream still get to inspect the
/// request. See PRD-09c §6.4.
#[derive(Debug, Clone, PartialEq)]
pub enum GuardrailVerdict {
    Allow,
    Block {
        /// Operator-facing detail (matched pattern, provider assessment).
        /// Goes to ops logs only — per #153 it must never reach the wire
        /// envelope (echoing matched content lets callers enumerate the
        /// blocklist / extract the blocked output).
        reason: String,
        /// The configured (row) name of the guardrail that fired, attached
        /// by [`GuardrailChain`] (#519 B.4b). Safe to surface in the error
        /// envelope — it's operator-assigned metadata, not matched content.
        /// `None` when the verdict came from a bare guardrail outside a
        /// chain.
        guardrail_name: Option<String>,
        /// Set when this block is an AVAILABILITY failure rather than a
        /// content decision: a remote guardrail with `fail_open: false`
        /// (or a `mandatory` row) that could not reach its upstream blocks
        /// instead of bypassing, and the two are otherwise
        /// indistinguishable to every consumer downstream
        /// (AISIX-Cloud#1365).
        ///
        /// Carries the same bounded per-kind failure tag a `Bypass` puts
        /// in its reason (e.g. `lakera_timeout`) — a closed vocabulary,
        /// never free text and never matched content, so it is safe on a
        /// metric label and on the wire (#153).
        unavailable: Option<String>,
    },
    Bypass {
        reason: String,
    },
}

/// Clamp a failure tag to the shape a metric label and an audit field can
/// safely carry: lowercase alphanumerics and underscores, at most 64 bytes.
///
/// Every producer already passes a `bypass_tag()` constant, so this is a
/// no-op today. It is here because the TYPE cannot say so: the tag is a
/// `String` (`MandatoryGuardrail` forwards whatever reason the inner
/// guardrail's `Bypass` carried), it lands on an unsanitized Prometheus
/// label, and it reaches the usage event that #153 forbids putting content
/// on. A future guardrail that returns a free-text bypass reason would
/// otherwise mint one metric series per distinct string.
pub(crate) fn bounded_failure_tag(tag: &str) -> String {
    let cleaned: String = tag
        .chars()
        .map(|c| c.to_ascii_lowercase())
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_')
        .take(64)
        .collect();
    if cleaned.is_empty() {
        "unknown".to_owned()
    } else {
        cleaned
    }
}

impl GuardrailVerdict {
    /// `Block` verdict with no guardrail-name attribution (the chain fills
    /// the name in). Implementations use this so they don't repeat
    /// `guardrail_name: None` at every block site.
    pub fn block(reason: impl Into<String>) -> Self {
        GuardrailVerdict::Block {
            reason: reason.into(),
            guardrail_name: None,
            unavailable: None,
        }
    }

    /// `Block` for a fail-CLOSED AVAILABILITY failure (AISIX-Cloud#1365):
    /// the guardrail could not evaluate, and its configuration says an
    /// un-evaluated request is refused rather than let through.
    ///
    /// `tag` is the guardrail kind's bounded failure tag — the same value
    /// the fail-OPEN branch puts in `Bypass::reason`, so one outage reads
    /// the same whichever way the row is configured.
    pub fn block_unavailable(reason: impl Into<String>, tag: impl Into<String>) -> Self {
        GuardrailVerdict::Block {
            reason: reason.into(),
            guardrail_name: None,
            unavailable: Some(bounded_failure_tag(&tag.into())),
        }
    }

    /// The bounded failure tag when this verdict is a fail-closed
    /// availability block; `None` for a content decision and for every
    /// non-block verdict.
    pub fn unavailable_tag(&self) -> Option<&str> {
        match self {
            GuardrailVerdict::Block {
                unavailable: Some(tag),
                ..
            } => Some(tag.as_str()),
            _ => None,
        }
    }

    pub fn is_block(&self) -> bool {
        matches!(self, GuardrailVerdict::Block { .. })
    }

    pub fn is_bypass(&self) -> bool {
        matches!(self, GuardrailVerdict::Bypass { .. })
    }

    /// Extract the bypass reason if this is a `Bypass` verdict, else
    /// `None`. Used by the chat handler to attach
    /// `guardrail_bypassed_reason` to the telemetry event.
    pub fn bypass_reason(&self) -> Option<&str> {
        match self {
            GuardrailVerdict::Bypass { reason } => Some(reason.as_str()),
            _ => None,
        }
    }

    /// Fold the verdicts of two split moderation passes over the same
    /// content (the non-segment check + the segment pass) into one:
    /// Block wins (`self` first), then Bypass (`self`'s reason first),
    /// else Allow.
    pub fn merged_with(self, other: GuardrailVerdict) -> GuardrailVerdict {
        match (self, other) {
            (b @ GuardrailVerdict::Block { .. }, _) => b,
            (_, b @ GuardrailVerdict::Block { .. }) => b,
            (by @ GuardrailVerdict::Bypass { .. }, _) => by,
            (_, by @ GuardrailVerdict::Bypass { .. }) => by,
            _ => GuardrailVerdict::Allow,
        }
    }
}

/// How a guardrail wants STREAMED output moderated. The proxy's SSE
/// builder queries [`Guardrail::stream_output_policy`] on the resolved
/// chain and applies the strictest member policy to decide whether to
/// hold streamed content back until it scans clean.
///
/// `EndOfStreamCheck` is the pre-P2 behavior — chunks are forwarded
/// live and `check_output` runs once at end-of-stream (so a block frame
/// arrives *after* the content already reached the client). The
/// hold-back variants buffer content until it passes.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum StreamOutputPolicy {
    /// Forward live; check once at end-of-stream. No hold-back. Default.
    #[default]
    EndOfStreamCheck,
    /// Sliding window: release a window of content only after it scans
    /// clean; `overlap_chars` is carried between windows so a span split
    /// across a boundary is still caught.
    Window {
        size_chars: usize,
        overlap_chars: usize,
    },
    /// Hold the whole response; scan once; release all or block.
    /// `max_buffer_bytes` caps the hold; `on_exceeded_fail_open` decides
    /// release-vs-block when the cap is exceeded.
    BufferFull {
        max_buffer_bytes: usize,
        on_exceeded_fail_open: bool,
    },
}

impl StreamOutputPolicy {
    /// `true` when this policy holds streamed content back until it
    /// scans clean (i.e. anything other than `EndOfStreamCheck`).
    pub fn holds_back(&self) -> bool {
        !matches!(self, StreamOutputPolicy::EndOfStreamCheck)
    }

    /// Coarse strictness rank: more hold-back = higher.
    fn rank(&self) -> u8 {
        match self {
            StreamOutputPolicy::EndOfStreamCheck => 0,
            StreamOutputPolicy::Window { .. } => 1,
            StreamOutputPolicy::BufferFull { .. } => 2,
        }
    }

    /// Pick the stricter of two policies (used to fold a chain into one).
    /// Higher rank wins; ties break toward the tighter parameters
    /// (smaller window, smaller buffer cap).
    pub fn stricter(self, other: Self) -> Self {
        use StreamOutputPolicy::*;
        match self.rank().cmp(&other.rank()) {
            std::cmp::Ordering::Less => other,
            std::cmp::Ordering::Greater => self,
            std::cmp::Ordering::Equal => match (self, other) {
                (
                    Window {
                        size_chars: a,
                        overlap_chars: oa,
                    },
                    Window {
                        size_chars: b,
                        overlap_chars: ob,
                    },
                ) => Window {
                    size_chars: a.min(b),
                    overlap_chars: oa.max(ob),
                },
                (
                    BufferFull {
                        max_buffer_bytes: a,
                        on_exceeded_fail_open: fa,
                    },
                    BufferFull {
                        max_buffer_bytes: b,
                        on_exceeded_fail_open: fb,
                    },
                ) => BufferFull {
                    max_buffer_bytes: a.min(b),
                    // fail-closed is stricter than fail-open.
                    on_exceeded_fail_open: fa && fb,
                },
                (s, _) => s,
            },
        }
    }
}

/// Default whole-response hold-back cap for output guardrails that don't
/// configure their own streaming policy (keyword, prompt shield, bedrock).
/// Matches the Azure text-moderation buffer-mode default.
pub const DEFAULT_STREAM_OUTPUT_BUFFER_BYTES: usize = 262_144;

/// One text-channel redaction outcome from
/// [`Guardrail::redact_input_text`] / [`Guardrail::redact_output_text`]:
/// the rewritten text plus per-detector match counts. Counts carry detector
/// NAMES only — the matched values are gone by construction, so this type
/// is safe to log and to attach to telemetry (#932 no-leak criterion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Redaction {
    pub text: String,
    /// detector name → number of masked spans.
    pub counts: std::collections::BTreeMap<String, u32>,
}

impl Redaction {
    /// Fold `other`'s counts into `self` (used by chains and by callers
    /// merging per-field redactions into one per-request summary).
    pub fn merge_counts(
        into: &mut std::collections::BTreeMap<String, u32>,
        other: &std::collections::BTreeMap<String, u32>,
    ) {
        for (k, v) in other {
            *into.entry(k.clone()).or_insert(0) += v;
        }
    }
}

/// Outcome of [`Guardrail::moderate_input_segments`] /
/// [`Guardrail::moderate_output_segments`] — remote moderation of a
/// request's text segments in ONE provider call (kind=bedrock).
///
/// `masked`, when present, is positionally aligned with the input
/// `texts` slice: `masked[i]` replaces `texts[i]`. Implementations MUST
/// uphold that alignment or return `masked: None` (the caller then keeps
/// the originals — the LiteLLM `_merge_masked_texts` defensive fallback:
/// never misapply masked content to the wrong slot).
///
/// `counts` mirrors [`Redaction::counts`]: entity NAMES only (e.g. a
/// Bedrock PII entity type like `EMAIL`), never matched values, so it is
/// safe for logs and telemetry (#153 / #932 no-leak criterion).
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentsOutcome {
    pub verdict: GuardrailVerdict,
    pub masked: Option<Vec<String>>,
    pub counts: std::collections::BTreeMap<String, u32>,
    /// Monitor-mode observations made during this pass (what an
    /// `enforcement_mode: monitor` member WOULD have done). Callers merge
    /// them into the request's `guardrail_monitor_hits` telemetry.
    pub monitor_hits: Vec<GuardrailMonitorHit>,
}

impl SegmentsOutcome {
    /// Plain Allow: nothing detected, nothing rewritten.
    pub fn allow() -> Self {
        Self {
            verdict: GuardrailVerdict::Allow,
            masked: None,
            counts: std::collections::BTreeMap::new(),
            monitor_hits: Vec::new(),
        }
    }

    /// Wrap a bare verdict (Block/Bypass paths carry no mask or counts).
    pub fn from_verdict(verdict: GuardrailVerdict) -> Self {
        Self {
            verdict,
            masked: None,
            counts: std::collections::BTreeMap::new(),
            monitor_hits: Vec::new(),
        }
    }
}

/// Pluggable content-policy hook. Production wires `Arc<dyn Guardrail>`
/// in `ProxyState`; tests construct in-memory chains directly.
#[async_trait]
pub trait Guardrail: Send + Sync + 'static {
    /// Stable name for log/metric labels.
    fn name(&self) -> &'static str;

    /// Inspect the incoming request. Default: allow everything.
    async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
        GuardrailVerdict::Allow
    }

    /// Inspect the upstream response. Default: allow everything.
    async fn check_output(&self, _resp: &ChatResponse) -> GuardrailVerdict {
        GuardrailVerdict::Allow
    }

    /// `true` when the guardrail will trivially `Allow` everything —
    /// callers can skip set-up work (buffer allocations, fixture
    /// synthesis) on the hot path. Default: `false` (assume work is
    /// needed). Concrete impls that know they're a no-op (e.g. an
    /// empty `GuardrailChain`) override to return `true`.
    fn is_empty(&self) -> bool {
        false
    }

    /// How this guardrail wants streamed OUTPUT moderated. Default:
    /// hold the whole response back ([`StreamOutputPolicy::BufferFull`],
    /// fail-closed) so an output-blocking guardrail can't leak content
    /// onto the wire before its check runs (#466 — secure-by-default).
    /// Guardrails that want partial streaming (Azure text moderation)
    /// override with `Window`.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        StreamOutputPolicy::BufferFull {
            max_buffer_bytes: DEFAULT_STREAM_OUTPUT_BUFFER_BYTES,
            on_exceeded_fail_open: false,
        }
    }

    /// Whether this guardrail actually inspects the OUTPUT hook. Drives
    /// whether its `stream_output_policy` participates in the streamed-output
    /// hold-back fold (#466): an input-only guardrail must NOT force output
    /// buffering — it never looks at the response, so holding the stream back
    /// for it is pure latency with no security benefit. Default: `true`
    /// (assume output-relevant, secure-leaning); input-only impls override
    /// to gate on their hook.
    fn runs_on_output(&self) -> bool {
        true
    }

    // --- redaction (#932) -------------------------------------------------
    //
    // Redaction is a separate, synchronous, text→text capability rather
    // than a mutation inside `check_input`/`check_output`: the check hooks
    // scan ONE concatenated blob per request, while redaction must be
    // applied per text FIELD (each message, each tool-call argument, each
    // streamed channel) so the caller controls which wire fields are
    // rewritten and structure is preserved. Callers run the check first
    // (Block wins over Mask), then apply the redactor to each field.

    /// `true` when this guardrail can rewrite REQUEST text. Cheap probe so
    /// call sites skip walking the body when nothing would change.
    fn redacts_input(&self) -> bool {
        false
    }

    /// `true` when this guardrail can rewrite RESPONSE text.
    fn redacts_output(&self) -> bool {
        false
    }

    /// Rewrite one request-side text field, masking sensitive spans.
    /// `None` = no capability or no matches (caller keeps the original).
    fn redact_input_text(&self, _text: &str) -> Option<Redaction> {
        None
    }

    /// Rewrite one response-side text field, masking sensitive spans.
    fn redact_output_text(&self, _text: &str) -> Option<Redaction> {
        None
    }

    // --- remote segment moderation (#932 bedrock follow-up) ---------------
    //
    // A remote-API guardrail that can MASK (Bedrock PII anonymize) can't
    // implement the sync per-field redact contract above — the mask comes
    // back from the provider call itself. Instead the proxy hands such a
    // guardrail ALL of a request's text segments at once (in wire-walker
    // order), gets verdict + positionally-aligned masked replacements from
    // ONE provider call, and writes them back per wire shape. Call sites
    // that run this pass pair it with `check_*_non_segment` so the
    // guardrail is consulted exactly once per hook.

    /// `true` when this guardrail moderates via the segment hooks below.
    /// Such a member is skipped by `check_input_non_segment` /
    /// `check_output_non_segment` (the segment pass covers it).
    fn moderates_segments(&self) -> bool {
        false
    }

    /// Moderate the request's text segments in one remote call. Only
    /// meaningful when [`Self::moderates_segments`] is `true`; the default
    /// allows so a caller that runs the pass unconditionally is safe.
    async fn moderate_input_segments(&self, _texts: &[String]) -> SegmentsOutcome {
        SegmentsOutcome::allow()
    }

    /// Moderate the response's text segments in one remote call.
    async fn moderate_output_segments(&self, _texts: &[String]) -> SegmentsOutcome {
        SegmentsOutcome::allow()
    }

    /// `check_input` minus segment-moderating members — used by call
    /// sites that ALSO run [`Self::moderate_input_segments`], so a
    /// segment member isn't consulted twice (and billed twice). For a
    /// leaf guardrail this is all-or-nothing: a segment moderator
    /// answers via the segment pass (Allow here), anything else answers
    /// via its normal check. [`GuardrailChain`] overrides with a
    /// member-filtered fold.
    async fn check_input_non_segment(&self, req: &ChatFormat) -> GuardrailVerdict {
        if self.moderates_segments() {
            GuardrailVerdict::Allow
        } else {
            self.check_input(req).await
        }
    }

    /// `check_output` minus segment-moderating members (see
    /// [`Self::check_input_non_segment`]).
    async fn check_output_non_segment(&self, resp: &ChatResponse) -> GuardrailVerdict {
        if self.moderates_segments() {
            GuardrailVerdict::Allow
        } else {
            self.check_output(resp).await
        }
    }

    // --- monitor-hit observation (AISIX-Cloud#562) -------------------------
    //
    // `enforcement_mode: monitor` downgrades Blocks and suppresses masks,
    // which erases the observation from the plain check return value. The
    // `*_observed` variants return the verdict PLUS the monitor hits made
    // during the check, so call sites can attach them to the request's
    // telemetry. Defaults delegate to the plain checks with no hits — only
    // the monitor decorator and the chain override these, so concrete
    // guardrail kinds never need to.

    /// [`Self::check_input`] plus any monitor-mode observations.
    async fn check_input_observed(
        &self,
        req: &ChatFormat,
    ) -> (GuardrailVerdict, Vec<GuardrailMonitorHit>) {
        (self.check_input(req).await, Vec::new())
    }

    /// [`Self::check_output`] plus any monitor-mode observations.
    async fn check_output_observed(
        &self,
        resp: &ChatResponse,
    ) -> (GuardrailVerdict, Vec<GuardrailMonitorHit>) {
        (self.check_output(resp).await, Vec::new())
    }

    /// [`Self::check_input_non_segment`] plus any monitor-mode observations.
    async fn check_input_non_segment_observed(
        &self,
        req: &ChatFormat,
    ) -> (GuardrailVerdict, Vec<GuardrailMonitorHit>) {
        if self.moderates_segments() {
            (GuardrailVerdict::Allow, Vec::new())
        } else {
            self.check_input_observed(req).await
        }
    }

    /// [`Self::check_output_non_segment`] plus any monitor-mode observations.
    async fn check_output_non_segment_observed(
        &self,
        resp: &ChatResponse,
    ) -> (GuardrailVerdict, Vec<GuardrailMonitorHit>) {
        if self.moderates_segments() {
            (GuardrailVerdict::Allow, Vec::new())
        } else {
            self.check_output_observed(resp).await
        }
    }
}

/// Serializes the tests (across modules) that install a log-capturing
/// tracing subscriber. `set_default` is thread-local, but tracing's
/// GLOBAL max-level hint is recomputed when any dispatcher is dropped —
/// a concurrently finishing capture test can lower it to OFF and make
/// this thread's `tracing::info!` fast-path away before reaching the
/// thread-local subscriber. One capture test at a time, process-wide.
/// A tokio mutex because the guard spans the captured async body.
#[cfg(test)]
pub(crate) static TRACING_CAPTURE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Keep every callsite emittable for the whole test binary; call it before
/// installing a capturing subscriber.
///
/// The lock above only orders the capture tests against each other. It does
/// nothing about the *other* piece of tracing global state: a callsite's
/// `Interest` is cached process-wide the first time that callsite is hit,
/// from whichever dispatcher the hitting thread has. With no global default
/// that is `NoSubscriber`, so any unrelated test reaching a guardrail's log
/// line first caches `Interest::never()` and the event is then skipped
/// everywhere — the capture sees only the events of crates whose callsites
/// happened to be registered under a subscriber.
///
/// A permissive global default removes both failure modes at once: no thread
/// ever falls back to `NoSubscriber`, and a permanently registered TRACE
/// dispatcher pins the global max-level hint. Registering it also
/// re-evaluates the callsites seen so far, so a lazy install still repairs a
/// cache poisoned earlier in the run.
#[cfg(test)]
pub(crate) fn keep_callsites_enabled() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // A bare registry writes nothing and formats nothing; it is here only
        // so that callsites register as enabled. The captured events are
        // rendered by each capture helper's own scoped subscriber.
        let _ = tracing::subscriber::set_global_default(tracing_subscriber::registry());
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_error_body_short_passes_through() {
        assert_eq!(truncate_error_body_for_log("boom"), "boom");
        // Exactly at the cap is not truncated.
        let at_cap = "a".repeat(MAX_ERROR_BODY_LOG_BYTES);
        assert_eq!(truncate_error_body_for_log(&at_cap), at_cap);
    }

    #[test]
    fn truncate_error_body_caps_length() {
        let big = "a".repeat(MAX_ERROR_BODY_LOG_BYTES + 500);
        let out = truncate_error_body_for_log(&big);
        assert_eq!(out.len(), MAX_ERROR_BODY_LOG_BYTES);
    }

    #[test]
    fn truncate_error_body_never_splits_a_char() {
        // '€' is 3 bytes; place a run of them so the byte cap lands mid-char.
        // The result must stay ≤ cap AND be valid UTF-8 (no split), i.e. end
        // on a char boundary just below the cap.
        let s = "€".repeat(MAX_ERROR_BODY_LOG_BYTES); // 3 * cap bytes
        let out = truncate_error_body_for_log(&s);
        assert!(out.len() <= MAX_ERROR_BODY_LOG_BYTES);
        assert!(
            out.len() > MAX_ERROR_BODY_LOG_BYTES - 3,
            "should fill the budget to within one char"
        );
        assert!(
            out.chars().all(|c| c == '€'),
            "must not emit a partial char"
        );
    }

    #[test]
    fn message_scan_text_falls_back_to_content_blocks() {
        // Flat content present → used verbatim.
        let flat: ChatMessage =
            serde_json::from_value(serde_json::json!({"role": "user", "content": "hello"}))
                .unwrap();
        assert_eq!(message_scan_text(&flat), "hello");

        // The #465 bypass shape: empty top-level content with the text
        // in an explicit content_blocks array (round-trip form). Must
        // be scanned, not skipped.
        let blocks_only: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": "",
            "content_blocks": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "http://x"}},
                {"type": "text", "text": "second"}
            ]
        }))
        .unwrap();
        assert_eq!(message_scan_text(&blocks_only), "first\nsecond");

        // Empty content, only a non-text block → nothing to scan.
        let image_only: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": "",
            "content_blocks": [{"type": "image_url", "image_url": {"url": "http://x"}}]
        }))
        .unwrap();
        assert_eq!(message_scan_text(&image_only), "");

        // Empty content, no blocks → empty.
        let empty: ChatMessage =
            serde_json::from_value(serde_json::json!({"role": "user", "content": ""})).unwrap();
        assert_eq!(message_scan_text(&empty), "");
    }

    #[test]
    fn message_scan_text_scans_content_blocks_even_when_flat_content_is_nonempty() {
        // Guardrail bypass: `content` and `content_blocks` are independent
        // wire fields, and the provider bridges forward `content_blocks`
        // when present. A caller that puts benign text in `content` and a
        // payload in `content_blocks` would slip the payload past a scan
        // that only reads `content`. The scanned text must be the UNION so
        // it is a superset of everything a bridge can forward upstream.
        let split: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": "benign cover text",
            "content_blocks": [{"type": "text", "text": "hidden payload"}]
        }))
        .unwrap();
        let scanned = message_scan_text(&split);
        assert!(
            scanned.contains("benign cover text") && scanned.contains("hidden payload"),
            "scan must cover both content and content_blocks, got {scanned:?}"
        );
    }

    #[test]
    fn message_scan_text_scans_tool_call_payload() {
        // Same bypass class via `extra["tool_calls"]`: history-replay tool
        // calls are forwarded upstream verbatim, so a payload in a
        // function name or arguments must be scanned. The whole payload is
        // serialized (matching guardrail_output_text), so both surfaces are
        // covered.
        let msg: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "lookup_evilname",
                    "arguments": "{\"q\":\"hidden arg payload\"}"
                }
            }]
        }))
        .unwrap();
        let scanned = message_scan_text(&msg);
        assert!(
            scanned.contains("hidden arg payload") && scanned.contains("lookup_evilname"),
            "scan must cover tool_call name and arguments, got {scanned:?}"
        );
    }

    struct DefaultPolicyGuardrail;
    impl Guardrail for DefaultPolicyGuardrail {
        fn name(&self) -> &'static str {
            "default-policy"
        }
    }

    #[test]
    fn default_stream_output_policy_holds_back() {
        // #466: a guardrail that doesn't override stream_output_policy
        // inherits a hold-back default, so an output-blocking guardrail can't
        // live-forward streamed content before its check (secure-by-default).
        let p = DefaultPolicyGuardrail.stream_output_policy();
        assert!(
            p.holds_back(),
            "default streamed-output policy must hold back"
        );
        assert!(matches!(
            p,
            StreamOutputPolicy::BufferFull {
                on_exceeded_fail_open: false,
                ..
            }
        ));
    }

    /// Pins `supported_kinds()` under the default feature set (all
    /// features on): exact contents, and every string round-trips
    /// through the config parser to the matching
    /// `GuardrailKind::kind_str` — so the heartbeat-reported list can
    /// never drift from the wire `kind` discriminators (#519 B.6).
    #[cfg(all(
        feature = "bedrock",
        feature = "azure-content-safety",
        feature = "aliyun-text-moderation",
        feature = "lakera",
        feature = "openai-moderation",
        feature = "presidio"
    ))]
    #[test]
    fn supported_kinds_matches_kind_str_under_default_features() {
        assert_eq!(
            supported_kinds(),
            &[
                "keyword",
                "pii",
                "azure_content_safety",
                "azure_content_safety_text_moderation",
                "aliyun_text_moderation",
                "aliyun_ai_guardrail",
                "bedrock",
                "lakera",
                "openai_moderation",
                "presidio",
                "semantic",
            ],
        );
        for kind in supported_kinds() {
            // Minimal valid config per kind; parse failure or a
            // kind_str mismatch means the heartbeat list drifted from
            // the schema's serde tags.
            let config = match *kind {
                "keyword" => serde_json::json!({
                    "kind": "keyword",
                    "patterns": [{"kind": "literal", "value": "x"}],
                }),
                "pii" => serde_json::json!({
                    "kind": "pii",
                    "detectors": [{"type": "email"}],
                }),
                "azure_content_safety" => serde_json::json!({
                    "kind": "azure_content_safety",
                    "endpoint": "https://x.cognitiveservices.azure.com",
                    "api_key": "k",
                }),
                "azure_content_safety_text_moderation" => serde_json::json!({
                    "kind": "azure_content_safety_text_moderation",
                    "endpoint": "https://x.cognitiveservices.azure.com",
                    "api_key": "k",
                }),
                "aliyun_text_moderation" => serde_json::json!({
                    "kind": "aliyun_text_moderation",
                    "region": "ap-southeast-1",
                    "access_key_id": "ak",
                    "access_key_secret": "sk",
                }),
                "aliyun_ai_guardrail" => serde_json::json!({
                    "kind": "aliyun_ai_guardrail",
                    "region": "cn-shanghai",
                    "access_key_id": "ak",
                    "access_key_secret": "sk",
                }),
                "bedrock" => serde_json::json!({
                    "kind": "bedrock",
                    "guardrail_id": "gr-1",
                    "guardrail_version": "1",
                    "region": "us-east-1",
                    "aws_credentials": {"kind": "static", "access_key_id": "ak", "secret_access_key": "sk"},
                    "latency_mode": {"kind": "serial"},
                }),
                "lakera" => serde_json::json!({
                    "kind": "lakera",
                    "api_key": "lk",
                }),
                "openai_moderation" => serde_json::json!({
                    "kind": "openai_moderation",
                    "api_key": "sk",
                }),
                "semantic" => serde_json::json!({
                    "kind": "semantic",
                    "embedding_model": "embed-1",
                    "deny_examples": ["x"],
                }),
                "presidio" => serde_json::json!({
                    "kind": "presidio",
                    "analyzer_url": "http://analyzer:3000",
                    "anonymizer_url": "http://anonymizer:3000",
                }),
                other => panic!("no parse fixture for kind {other:?}"),
            };
            let parsed: aisix_core::models::GuardrailKind = serde_json::from_value(config)
                .unwrap_or_else(|e| panic!("kind {kind:?} failed to parse: {e}"));
            assert_eq!(parsed.kind_str(), *kind);
        }
    }

    #[test]
    fn verdict_helpers() {
        assert!(!GuardrailVerdict::Allow.is_block());
        assert!(GuardrailVerdict::block("x").is_block());
        assert_eq!(
            GuardrailVerdict::block("x"),
            GuardrailVerdict::Block {
                reason: "x".into(),
                guardrail_name: None,
                unavailable: None,
            },
        );
        // A plain content block carries no cause; a fail-closed one does,
        // and that is the only difference a consumer can see
        // (AISIX-Cloud#1365).
        assert_eq!(GuardrailVerdict::block("x").unavailable_tag(), None);
        assert!(GuardrailVerdict::block_unavailable("x", "lakera_timeout").is_block());
        assert_eq!(
            GuardrailVerdict::block_unavailable("x", "lakera_timeout").unavailable_tag(),
            Some("lakera_timeout"),
        );
        assert_eq!(GuardrailVerdict::Allow.unavailable_tag(), None);
        // The tag reaches an unsanitized Prometheus label and the usage
        // event, so it is clamped at construction rather than trusted:
        // every producer passes a `bypass_tag()` constant today, but the
        // field's TYPE is `String` and cannot say so.
        assert_eq!(
            GuardrailVerdict::block_unavailable("x", "presidio_5xx").unavailable_tag(),
            Some("presidio_5xx"),
            "a real tag must survive the clamp unchanged",
        );
        assert_eq!(
            GuardrailVerdict::block_unavailable("x", "Lakera Timeout: 500ms!").unavailable_tag(),
            Some("lakeratimeout500ms"),
        );
        assert_eq!(
            GuardrailVerdict::block_unavailable("x", "!!!").unavailable_tag(),
            Some("unknown"),
        );
        assert_eq!(
            GuardrailVerdict::block_unavailable("x", "a".repeat(200))
                .unavailable_tag()
                .map(str::len),
            Some(64),
            "an unbounded tag must not mint an unbounded metric series",
        );
        assert_eq!(
            GuardrailVerdict::Bypass { reason: "y".into() }.unavailable_tag(),
            None,
        );
        assert!(!GuardrailVerdict::Allow.is_bypass());
        assert!(GuardrailVerdict::Bypass { reason: "y".into() }.is_bypass());
        assert!(!GuardrailVerdict::Bypass { reason: "y".into() }.is_block());
        assert_eq!(
            GuardrailVerdict::Bypass { reason: "y".into() }.bypass_reason(),
            Some("y"),
        );
        assert_eq!(GuardrailVerdict::Allow.bypass_reason(), None);
    }
}
