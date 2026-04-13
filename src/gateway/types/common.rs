//! Common types shared across all API formats.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    anthropic::CacheControl,
    openai::responses::{
        ContextManagement, ConversationReference, PromptCacheRetention, ReasoningConfig,
        ResponsePrompt, ResponseTextConfig, ResponsesStreamOptions, Truncation,
    },
};

/// Unified usage metrics across all modalities.
///
/// Fields are `Option` because different providers and modalities report
/// different subsets. The `merge()` method combines partial updates
/// (e.g., from streaming chunks) into a complete picture.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    // ── Text tokens ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,

    // ── Multimodal tokens ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<u32>,

    // ── Cache tokens (Anthropic / OpenAI) ──
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

impl Usage {
    fn derive_total_tokens(input_tokens: Option<u32>, output_tokens: Option<u32>) -> Option<u32> {
        input_tokens
            .zip(output_tokens)
            .map(|(input_tokens, output_tokens)| input_tokens + output_tokens)
    }

    /// Returns the reported total tokens, or derives it from input/output tokens.
    pub fn resolved_total_tokens(&self) -> Option<u32> {
        self.total_tokens
            .or_else(|| Self::derive_total_tokens(self.input_tokens, self.output_tokens))
    }

    /// Fills `total_tokens` from input/output token counts when it is missing.
    pub fn with_derived_total(mut self) -> Self {
        if self.total_tokens.is_none() {
            self.total_tokens = Self::derive_total_tokens(self.input_tokens, self.output_tokens);
        }
        self
    }

    /// Merge partial usage from another source (e.g., a streaming update).
    ///
    /// Non-None fields in `other` overwrite the corresponding fields in `self`.
    /// If `total_tokens` was previously auto-derived, it is recomputed after
    /// input or output token updates to avoid stale totals.
    pub fn merge(&mut self, other: &Usage) {
        let previous_auto_total = Self::derive_total_tokens(self.input_tokens, self.output_tokens);
        let had_explicit_total =
            self.total_tokens.is_some() && self.total_tokens != previous_auto_total;
        let token_counts_updated = other.input_tokens.is_some() || other.output_tokens.is_some();

        if other.input_tokens.is_some() {
            self.input_tokens = other.input_tokens;
        }
        if other.output_tokens.is_some() {
            self.output_tokens = other.output_tokens;
        }
        if other.total_tokens.is_some() {
            self.total_tokens = other.total_tokens;
        }
        if other.input_audio_tokens.is_some() {
            self.input_audio_tokens = other.input_audio_tokens;
        }
        if other.output_audio_tokens.is_some() {
            self.output_audio_tokens = other.output_audio_tokens;
        }
        if other.image_tokens.is_some() {
            self.image_tokens = other.image_tokens;
        }
        if other.cache_creation_input_tokens.is_some() {
            self.cache_creation_input_tokens = other.cache_creation_input_tokens;
        }
        if other.cache_read_input_tokens.is_some() {
            self.cache_read_input_tokens = other.cache_read_input_tokens;
        }

        if other.total_tokens.is_some() {
            return;
        }

        if self.total_tokens.is_none() || (token_counts_updated && !had_explicit_total) {
            self.total_tokens = Self::derive_total_tokens(self.input_tokens, self.output_tokens);
        }
    }
}

/// Bridge context: preserves fields from the source format that the hub
/// format (OpenAI Chat) cannot represent.
///
/// Populated during `ChatFormat::to_hub()`, consumed during `ChatFormat::from_hub()`.
#[derive(Debug, Clone, Default)]
pub struct BridgeContext {
    pub anthropic_messages_extras: Option<AnthropicMessagesExtras>,
    pub openai_responses_extras: Option<OpenAIResponsesExtras>,
    /// Catch-all for fields that don't map to any known extras.
    pub passthrough: HashMap<String, Value>,
}

/// Anthropic Messages-specific fields preserved across hub bridging.
#[derive(Debug, Clone)]
pub struct AnthropicMessagesExtras {
    pub metadata: Option<Value>,
    pub system_cache_control: Option<CacheControl>,
}

/// OpenAI Responses-specific fields preserved across hub bridging.
#[derive(Debug, Clone)]
pub struct OpenAIResponsesExtras {
    pub previous_response_id: Option<String>,
    pub instructions: Option<String>,
    pub store: Option<bool>,
    pub metadata: Option<Value>,
    pub background: Option<bool>,
    pub context_management: Option<Vec<ContextManagement>>,
    pub conversation: Option<ConversationReference>,
    pub include: Option<Vec<String>>,
    pub max_tool_calls: Option<u32>,
    pub prompt: Option<ResponsePrompt>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    pub reasoning: Option<ReasoningConfig>,
    pub safety_identifier: Option<String>,
    pub service_tier: Option<String>,
    pub stream_options: Option<ResponsesStreamOptions>,
    pub text: Option<ResponseTextConfig>,
    pub top_logprobs: Option<u8>,
    pub truncation: Option<Truncation>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_merge_overwrites_present_fields() {
        let mut base = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            ..Default::default()
        };
        let update = Usage {
            output_tokens: Some(50),
            cache_read_input_tokens: Some(5),
            ..Default::default()
        };
        base.merge(&update);
        assert_eq!(base.input_tokens, Some(10)); // unchanged
        assert_eq!(base.output_tokens, Some(50)); // overwritten
        assert_eq!(base.cache_read_input_tokens, Some(5)); // added
        assert_eq!(base.total_tokens, Some(60)); // auto-computed
    }

    #[test]
    fn usage_merge_preserves_explicit_total() {
        let mut base = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            total_tokens: Some(100), // explicit, not auto-computed
            ..Default::default()
        };
        let update = Usage {
            output_tokens: Some(50),
            ..Default::default()
        };
        base.merge(&update);
        // total_tokens already set, should NOT be overwritten by auto-compute
        assert_eq!(base.total_tokens, Some(100));
    }

    #[test]
    fn usage_default_is_all_none() {
        let u = Usage::default();
        assert!(u.input_tokens.is_none());
        assert!(u.output_tokens.is_none());
        assert!(u.total_tokens.is_none());
    }

    #[test]
    fn usage_serde_round_trip() {
        let usage = Usage {
            input_tokens: Some(100),
            output_tokens: Some(200),
            total_tokens: Some(300),
            cache_creation_input_tokens: Some(10),
            ..Default::default()
        };
        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.input_tokens, Some(100));
        assert_eq!(deserialized.output_tokens, Some(200));
        assert_eq!(deserialized.total_tokens, Some(300));
        assert_eq!(deserialized.cache_creation_input_tokens, Some(10));
        assert!(deserialized.image_tokens.is_none());
    }

    #[test]
    fn usage_merge_recomputes_previous_derived_total() {
        let mut base = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            total_tokens: Some(30),
            ..Default::default()
        };

        let update = Usage {
            output_tokens: Some(50),
            ..Default::default()
        };

        base.merge(&update);
        assert_eq!(base.total_tokens, Some(60));
    }

    #[test]
    fn usage_resolved_total_tokens_falls_back_to_derived_total() {
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            ..Default::default()
        };

        assert_eq!(usage.resolved_total_tokens(), Some(30));
    }

    #[test]
    fn usage_with_derived_total_fills_missing_total() {
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            ..Default::default()
        }
        .with_derived_total();

        assert_eq!(usage.total_tokens, Some(30));
    }

    #[test]
    fn usage_serde_skips_null_fields() {
        let json = serde_json::to_value(Usage::default()).unwrap();
        assert_eq!(json, serde_json::json!({}));
    }
}
