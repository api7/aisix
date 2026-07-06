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
#[cfg(feature = "bedrock")]
mod bedrock;
mod build;
mod chain;
mod index;
mod keyword;
#[cfg(feature = "lakera")]
mod lakera;
#[cfg(feature = "openai-moderation")]
mod openai_moderation;
mod pii;
#[cfg(feature = "presidio")]
mod presidio;
#[cfg(feature = "azure-content-safety")]
mod prompt_shield;
#[cfg(feature = "azure-content-safety")]
mod text_moderation;

use aisix_gateway::{ChatFormat, ChatMessage, ChatResponse};
use async_trait::async_trait;

/// The text a guardrail should scan for one message.
///
/// Prefers the flat `content` string; when it's empty, falls back to
/// concatenating the `text`-type entries of `content_blocks`. A caller
/// that sends the OpenAI content-block shape
/// (`content: [{ "type": "text", "text": "…" }]`) with an empty
/// top-level string would otherwise bypass moderation entirely (#465).
/// Non-text blocks (image/audio) are out of scope — multimodal
/// moderation is a separate feature. Every guardrail's input/output
/// collector goes through this so the families can't drift.
pub(crate) fn message_scan_text(m: &ChatMessage) -> String {
    let content = m.content_str();
    if !content.is_empty() {
        return content.to_string();
    }
    match m.content_blocks.as_ref() {
        Some(blocks) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(serde_json::Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(serde_json::Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    }
}

/// The guardrail `kind` discriminators compiled into this binary.
///
/// Every non-keyword kind sits behind a cargo feature (see `build.rs`'s
/// `BuildError::FeatureDisabled` arms); a DP built without one silently
/// rejects rows of that kind while the dashboard still offers it
/// (#519 B.6). The heartbeat reports this list so cp-api can hide /
/// flag kinds the connected DP can't serve. Strings MUST stay equal to
/// the serde `kind` tags in `aisix_core::models::GuardrailKind`
/// (`GuardrailKind::kind_str`).
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
        #[cfg(feature = "bedrock")]
        "bedrock",
        #[cfg(feature = "lakera")]
        "lakera",
        #[cfg(feature = "openai-moderation")]
        "openai_moderation",
        #[cfg(feature = "presidio")]
        "presidio",
    ]
}

#[cfg(feature = "aliyun-text-moderation")]
pub use aliyun::AliyunTextModerationGuardrail;
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
#[cfg(feature = "openai-moderation")]
pub use openai_moderation::OpenaiModerationGuardrail;
pub use pii::{builtin_rule, PiiAction, PiiGuardrail, PiiRule, BUILTIN_DETECTORS};
#[cfg(feature = "presidio")]
pub use presidio::PresidioGuardrail;
#[cfg(feature = "azure-content-safety")]
pub use prompt_shield::PromptShieldGuardrail;
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
    },
    Bypass {
        reason: String,
    },
}

impl GuardrailVerdict {
    /// `Block` verdict with no guardrail-name attribution (the chain fills
    /// the name in). Implementations use this so they don't repeat
    /// `guardrail_name: None` at every block site.
    pub fn block(reason: impl Into<String>) -> Self {
        GuardrailVerdict::Block {
            reason: reason.into(),
            guardrail_name: None,
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
}

impl SegmentsOutcome {
    /// Plain Allow: nothing detected, nothing rewritten.
    pub fn allow() -> Self {
        Self {
            verdict: GuardrailVerdict::Allow,
            masked: None,
            counts: std::collections::BTreeMap::new(),
        }
    }

    /// Wrap a bare verdict (Block/Bypass paths carry no mask or counts).
    pub fn from_verdict(verdict: GuardrailVerdict) -> Self {
        Self {
            verdict,
            masked: None,
            counts: std::collections::BTreeMap::new(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
                "bedrock",
                "lakera",
                "openai_moderation",
                "presidio",
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
            },
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
