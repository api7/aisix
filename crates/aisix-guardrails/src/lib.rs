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

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod build;
mod chain;
mod keyword;
mod length;

use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;

pub use build::{build_chain_from_snapshot, LiveGuardrailChain};
pub use chain::GuardrailChain;
pub use keyword::{KeywordBlocklist, KeywordRule};
pub use length::MaxContentLength;

/// What a guardrail decided about a request or response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailVerdict {
    Allow,
    Block { reason: String },
}

impl GuardrailVerdict {
    pub fn is_block(&self) -> bool {
        matches!(self, GuardrailVerdict::Block { .. })
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_helpers() {
        assert!(!GuardrailVerdict::Allow.is_block());
        assert!(GuardrailVerdict::Block { reason: "x".into() }.is_block());
    }
}
