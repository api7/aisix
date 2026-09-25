//! End-of-stream output-guardrail observation for the live-forward
//! streaming paths (AISIX-Cloud#1010).
//!
//! A live-forwarded stream has already put its bytes on the wire, so an
//! output-hook chain can no longer block it — which is exactly why only
//! the chains that *can't* block reach here: a block-capable chain
//! resolves to a hold-back [`aisix_guardrails::StreamOutputPolicy`] and
//! takes its endpoint's buffered branch instead. What remains is the
//! monitor-only chain, whose whole purpose is to observe without changing
//! delivery — so it must still get its scan, and must never hold the
//! stream back to get it.
//!
//! Shared by `/v1/responses` (#808) and the streamed audio transcription
//! relay (#998); both assemble the response text as it streams and hand
//! it here once the upstream ends.

use std::sync::Arc;

use aisix_gateway::{ChatMessage, ChatResponse, FinishReason, UsageStats};

/// Build the minimal internal `ChatResponse` an output guardrail needs to
/// scan: the assistant text in `message.content`. Only the text is read by
/// `check_output` (via `guardrail_output_text`); the other fields are
/// placeholders and never reach the client.
pub(crate) fn synth_chat_response(model: &str, text: String) -> ChatResponse {
    ChatResponse {
        id: String::new(),
        model: model.to_string(),
        message: ChatMessage::assistant(text),
        finish_reason: FinishReason::Stop,
        usage: UsageStats::default(),
    }
}

/// End-of-stream output observation for a live-forward path. Reachable
/// only when the output-hook chain's resolved streaming policy is
/// `EndOfStreamCheck` — today that is exactly the monitor-only chains,
/// which can never block. Runs the same two-phase scan as the buffered
/// branch (blob check + segment pass) so would-block / would-mask hits
/// reach telemetry; the bytes are already on the wire, so a `Block`
/// verdict (unreachable for monitor members) is logged, not enforced.
pub(crate) struct EosOutputScan {
    chain: Arc<aisix_guardrails::GuardrailChain>,
    upstream_model: String,
}

impl EosOutputScan {
    pub(crate) fn new(
        chain: Arc<aisix_guardrails::GuardrailChain>,
        upstream_model: String,
    ) -> Self {
        Self {
            chain,
            upstream_model,
        }
    }

    /// `local_segments` are the slots the buffered branch's mask walker
    /// would rewrite, when the caller can rebuild them; the local kinds
    /// judge those instead of the flattened text (#1027).
    pub(crate) async fn observe(
        self,
        text: &str,
        local_segments: Option<Vec<aisix_guardrails::ScanSegment>>,
    ) -> Vec<aisix_core::GuardrailMonitorHit> {
        // The whole generated text, as chat and `/v1/messages` scan it: a
        // monitor-only chain holds nothing back, so no hold cap applies.
        if text.is_empty() {
            return Vec::new();
        }
        let synth = synth_chat_response(&self.upstream_model, text.to_string());
        let (verdict, mut hits) = aisix_guardrails::Guardrail::check_output_non_segment_observed(
            self.chain.as_ref(),
            &synth,
        )
        .await;
        // Segment pass (bedrock/lakera/presidio members): offer the flattened
        // text as one segment so monitor-mode segment moderators record their
        // observations too. Masks are suppressed in monitor mode, and nothing
        // could be rewritten anyway — the counts are discarded.
        let mut seg_counts = crate::redact::RedactionCounts::new();
        let verdict = crate::redact::moderate_body_local_given(
            self.chain.as_ref(),
            crate::redact::Direction::Output,
            None,
            verdict,
            &mut seg_counts,
            &mut hits,
            local_segments,
            |g| {
                let _ = g.redact_output_text(text);
                crate::redact::RedactionCounts::new()
            },
        )
        .await;
        if let aisix_guardrails::GuardrailVerdict::Block { reason, .. } = verdict {
            tracing::warn!(
                guardrail_hook = "output",
                model = %self.upstream_model,
                reason = %reason,
                "output guardrail returned a block after live forward; \
                 response already sent (EndOfStreamCheck policy)",
            );
        }
        hits
    }
}
