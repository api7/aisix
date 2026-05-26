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
//! - [`MaxContentLength`] — caps total characters across input messages
//!   or output content.
//! - [`GuardrailChain`] — composes multiple guardrails; first
//!   [`GuardrailVerdict::Block`] short-circuits.
//! - [`GuardrailIndex`] — P0c: resolves the per-request chain from a
//!   snapshot of guardrail definitions + attachment rows.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

#[cfg(feature = "bedrock")]
mod bedrock;
mod build;
mod chain;
mod index;
mod keyword;
mod length;

use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;

#[cfg(feature = "bedrock")]
pub use bedrock::BedrockGuardrail;
pub use build::{
    build_chain_from_snapshot, build_index_from_snapshot, LiveGuardrailChain, LiveGuardrailIndex,
};
pub use chain::GuardrailChain;
pub use index::{GuardrailIndex, RequestContext};
pub use keyword::{KeywordBlocklist, KeywordRule};
pub use length::MaxContentLength;

/// What a guardrail decided about a request or response.
///
/// `Bypass` exists for remote-API guardrails (kind=bedrock) whose
/// upstream is unreachable but the operator configured `fail_open=true`:
/// the request goes through, but the bypass is recorded on the
/// telemetry event so a compliance audit can see what slipped past.
/// `Bypass` is **not** a block — the chain doesn't short-circuit on
/// it, and other guardrails downstream still get to inspect the
/// request. See PRD-09c §6.4.
///
/// `Rewrite` signals that the guardrail modified the request payload
/// (e.g. a PII-scrubbing guardrail that replaces tokens before the
/// prompt reaches the upstream). The modified payload is propagated to
/// all subsequent guardrails in the chain via [`GuardrailChain`] and
/// eventually substituted for the original request before bridge
/// dispatch. See `chain.rs` + PRD-09c §6.5.
#[derive(Debug, Clone)]
pub enum GuardrailVerdict {
    Allow,
    Block { reason: String },
    Bypass { reason: String },
    Rewrite { payload: Box<ChatFormat> },
}

impl PartialEq for GuardrailVerdict {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GuardrailVerdict::Allow, GuardrailVerdict::Allow) => true,
            (GuardrailVerdict::Block { reason: a }, GuardrailVerdict::Block { reason: b }) => {
                a == b
            }
            (GuardrailVerdict::Bypass { reason: a }, GuardrailVerdict::Bypass { reason: b }) => {
                a == b
            }
            // `ChatFormat` contains `f32` fields which don't implement `Eq`.
            // Tests use `is_rewrite()` rather than `==` for this variant.
            (GuardrailVerdict::Rewrite { .. }, GuardrailVerdict::Rewrite { .. }) => false,
            _ => false,
        }
    }
}

impl GuardrailVerdict {
    pub fn is_block(&self) -> bool {
        matches!(self, GuardrailVerdict::Block { .. })
    }

    pub fn is_bypass(&self) -> bool {
        matches!(self, GuardrailVerdict::Bypass { .. })
    }

    pub fn is_rewrite(&self) -> bool {
        matches!(self, GuardrailVerdict::Rewrite { .. })
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_helpers() {
        assert!(!GuardrailVerdict::Allow.is_block());
        assert!(GuardrailVerdict::Block { reason: "x".into() }.is_block());
        assert!(!GuardrailVerdict::Allow.is_bypass());
        assert!(GuardrailVerdict::Bypass { reason: "y".into() }.is_bypass());
        assert!(!GuardrailVerdict::Bypass { reason: "y".into() }.is_block());
        assert_eq!(
            GuardrailVerdict::Bypass { reason: "y".into() }.bypass_reason(),
            Some("y"),
        );
        assert_eq!(GuardrailVerdict::Allow.bypass_reason(), None);
        assert!(GuardrailVerdict::Rewrite {
            payload: Box::new(ChatFormat::new("m", vec![]))
        }
        .is_rewrite());
        assert!(!GuardrailVerdict::Rewrite {
            payload: Box::new(ChatFormat::new("m", vec![]))
        }
        .is_block());
    }
}
