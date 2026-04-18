//! Cap on total content length.
//!
//! Two thresholds:
//! - `max_input_chars`: total characters across all input messages.
//!   Catches "send me 1MB of text" abuse before it hits the upstream
//!   timeout / per-request token cap.
//! - `max_output_chars`: total characters of the assistant message.
//!   Catches misbehaving models or guardrail-jailbreak completions.
//!
//! Either bound can be `None` to disable that side. Counts are
//! `chars()`-based (Unicode scalar values) — same metric a user would
//! see counting graphemes by hand, and stable across re-encodings.

use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;

use crate::{Guardrail, GuardrailVerdict};

#[derive(Debug, Clone, Copy)]
pub struct MaxContentLength {
    pub max_input_chars: Option<usize>,
    pub max_output_chars: Option<usize>,
}

impl MaxContentLength {
    pub fn new(max_input_chars: Option<usize>, max_output_chars: Option<usize>) -> Self {
        Self {
            max_input_chars,
            max_output_chars,
        }
    }

    pub fn input_only(max: usize) -> Self {
        Self {
            max_input_chars: Some(max),
            max_output_chars: None,
        }
    }

    pub fn output_only(max: usize) -> Self {
        Self {
            max_input_chars: None,
            max_output_chars: Some(max),
        }
    }
}

#[async_trait]
impl Guardrail for MaxContentLength {
    fn name(&self) -> &'static str {
        "max_content_length"
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        let Some(cap) = self.max_input_chars else {
            return GuardrailVerdict::Allow;
        };
        let total: usize = req.messages.iter().map(|m| m.content.chars().count()).sum();
        if total > cap {
            return GuardrailVerdict::Block {
                reason: format!("input exceeds {cap} chars (was {total})"),
            };
        }
        GuardrailVerdict::Allow
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        let Some(cap) = self.max_output_chars else {
            return GuardrailVerdict::Allow;
        };
        let len = resp.message.content.chars().count();
        if len > cap {
            return GuardrailVerdict::Block {
                reason: format!("output exceeds {cap} chars (was {len})"),
            };
        }
        GuardrailVerdict::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{ChatMessage, FinishReason, UsageStats};

    fn req(parts: &[&str]) -> ChatFormat {
        ChatFormat::new("m", parts.iter().map(|p| ChatMessage::user(*p)).collect())
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

    #[tokio::test]
    async fn input_within_cap_passes() {
        let g = MaxContentLength::input_only(100);
        assert_eq!(
            g.check_input(&req(&["short prompt"])).await,
            GuardrailVerdict::Allow,
        );
    }

    #[tokio::test]
    async fn input_exceeds_cap_blocks_with_useful_reason() {
        let g = MaxContentLength::input_only(10);
        let v = g
            .check_input(&req(&["this is way too long for a tiny cap"]))
            .await;
        if let GuardrailVerdict::Block { reason } = v {
            assert!(reason.contains("exceeds 10"));
        } else {
            panic!("expected Block");
        }
    }

    #[tokio::test]
    async fn input_total_sums_across_messages() {
        let g = MaxContentLength::input_only(15);
        // Two 10-char messages → 20 total > 15 cap.
        let v = g.check_input(&req(&["1234567890", "1234567890"])).await;
        assert!(v.is_block());
    }

    #[tokio::test]
    async fn unbounded_input_never_blocks_input_check() {
        let g = MaxContentLength::output_only(5);
        let v = g.check_input(&req(&["any length is fine here"])).await;
        assert_eq!(v, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn output_exceeds_cap_blocks() {
        let g = MaxContentLength::output_only(5);
        let v = g.check_output(&resp("123456")).await;
        assert!(v.is_block());
    }

    #[tokio::test]
    async fn output_within_cap_passes() {
        let g = MaxContentLength::output_only(100);
        let v = g.check_output(&resp("ok")).await;
        assert_eq!(v, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn unicode_counted_by_chars_not_bytes() {
        // Each emoji is 4 bytes UTF-8 but 1 char (BMP scalar). 3 emojis
        // == 3 chars.
        let g = MaxContentLength::input_only(3);
        let v = g.check_input(&req(&["🚀🚀🚀"])).await; // 3 chars, fits
        assert_eq!(v, GuardrailVerdict::Allow);

        let v = g.check_input(&req(&["🚀🚀🚀🚀"])).await; // 4 chars, blocks
        assert!(v.is_block());
    }
}
