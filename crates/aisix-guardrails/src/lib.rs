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
