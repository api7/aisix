//! Common types shared across all API formats.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::anthropic::CacheControl;

/// Unified usage metrics across all modalities.
///
/// Fields are `Option` because different providers and modalities report
/// different subsets. The `merge()` method combines partial updates
/// (e.g., from streaming chunks) into a complete picture.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    // ── Text tokens ──
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
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
    /// Merge partial usage from another source (e.g., a streaming update).
    ///
    /// Non-None fields in `other` overwrite the corresponding fields in `self`.
    /// After merging, `total_tokens` is auto-computed if not explicitly set.
    pub fn merge(&mut self, other: &Usage) {
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
        // Auto-compute total if not explicitly set
        if self.total_tokens.is_none()
            && let (Some(i), Some(o)) = (self.input_tokens, self.output_tokens)
        {
            self.total_tokens = Some(i + o);
        }
    }
}

/// Bridge context: preserves fields from the source format that the hub
/// format (OpenAI Chat) cannot represent.
///
/// Populated during `ChatFormat::to_hub()`, consumed during `ChatFormat::from_hub()`.
#[derive(Debug, Clone, Default)]
pub struct BridgeContext {
    pub anthropic_extras: Option<AnthropicExtras>,
    pub responses_extras: Option<ResponsesExtras>,
    /// Catch-all for fields that don't map to any known extras.
    pub passthrough: HashMap<String, Value>,
}

/// Anthropic-specific fields preserved across hub bridging.
#[derive(Debug, Clone)]
pub struct AnthropicExtras {
    pub metadata: Option<Value>,
    pub system_cache_control: Option<CacheControl>,
}

/// Responses API-specific fields preserved across hub bridging.
#[derive(Debug, Clone)]
pub struct ResponsesExtras {
    pub previous_response_id: Option<String>,
    pub instructions: Option<String>,
    pub store: Option<bool>,
    pub metadata: Option<Value>,
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
}
