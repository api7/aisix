//! Provider-agnostic chat request / response types.
//!
//! The gateway normalises every client request into a [`ChatFormat`] and
//! hands it to whichever [`crate::bridge::Bridge`] implementation matches
//! the target provider. The response shape (either a full [`ChatResponse`]
//! or a stream of [`ChatChunk`]s) is symmetric: providers emit the normalised
//! form and the proxy layer re-encodes into whatever the client expects
//! (defaulting to OpenAI-compatible JSON).
//!
//! These types are deliberately a superset of OpenAI's shape because that
//! is the most permissive of the four providers we're targeting; fields
//! that don't map cleanly to a specific upstream become the provider's
//! responsibility to drop or translate.

use serde::{Deserialize, Deserializer, Serialize};

/// Role of a chat message. Matches OpenAI's taxonomy; providers that only
/// support system/user/assistant are expected to reject `Tool` at their
/// own boundary rather than silently collapsing roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One element of the OpenAI-shape `messages` array.
///
/// `deny_unknown_fields` is intentionally NOT applied here — OpenAI ships
/// new message-level fields regularly (`tool_calls` on assistant messages,
/// `refusal` since 2024-08, `audio` for the realtime/4o audio models) and
/// the standard OpenAI SDKs include them whenever they replay
/// conversation history. Rejecting them at the gateway breaks every user
/// that has had a tool round-trip in the conversation. Unknown fields
/// land in [`Self::extra`] via `flatten` so providers that care
/// (currently the OpenAI bridge) can forward them verbatim.
///
/// `content` accepts JSON `null` in addition to a string. OpenAI's
/// assistant-with-tool_calls shape uses `"content": null`; we collapse
/// that to an empty string for the gateway's internal representation,
/// which preserves serialisability for downstreams that don't accept
/// `null` (Anthropic, Gemini). Information loss is bounded — if the
/// upstream behaves differently for `""` vs `null`, the OpenAI bridge
/// can synthesise `null` from an empty string + a `tool_calls` entry in
/// `extra` at request-build time. See issue #110.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default, deserialize_with = "deserialize_content_string")]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Forward-compatible bag for OpenAI message fields the gateway
    /// doesn't model directly: `tool_calls`, `refusal`, `audio`, plus
    /// any future additions. Round-tripped verbatim so OpenAI
    /// conversation history replay works through the proxy without a
    /// schema bump every time OpenAI ships a new field.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty", flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Deserialize a `content` field that may be a string OR JSON `null`.
/// `null` collapses to `""`. The type stays `String` (not `Option<String>`)
/// so every existing caller — and every cross-provider bridge — keeps
/// working without an Option dance. The `null` case only matters for
/// OpenAI assistant-with-tool_calls history replay, where empty content
/// is also accepted by the upstream API; see
/// <https://platform.openai.com/docs/api-reference/chat/create>.
fn deserialize_content_string<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }
}

/// Normalised chat completion request.
///
/// `model` is the **public-facing** name from the Admin API (e.g.
/// `"my-gpt4"`), not the upstream model id. The gateway resolves this to
/// an `aisix_core::Model` before calling a Bridge; the Bridge receives
/// only the resolved [`crate::bridge::BridgeContext`] and translates the
/// `ChatFormat` to the provider's own request shape.
///
/// Unknown top-level fields are **not** rejected — OpenAI's API adds
/// params regularly (e.g. `top_k`, `seed`, `presence_penalty`), and each
/// Bridge is responsible for forwarding or ignoring them. Extras land in
/// the `extra` map via `#[serde(flatten)]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFormat {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Free-form extra fields the client sent. We don't strip unknown
    /// params at the gateway — each Bridge decides what to forward.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty", flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ChatFormat {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            extra: serde_json::Map::new(),
        }
    }

    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

/// Why a completion finished. Unknown upstream reasons collapse to
/// [`FinishReason::Other`] carrying the original string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Other(String),
}

/// Token usage stats from one upstream chat completion. The four
/// fine-grained counters that follow `total_tokens` carry the
/// provider-specific cache / reasoning detail used by cp-api's cost
/// formula (see `aisix-cloud:internal/dpmgr/dpstore/pricing.go`).
///
/// Provider-protocol mapping (the canonical comment lives in cp-api's
/// schema; mirrored here for grep-ability):
///
///   OpenAI Chat Completions response.usage:
///     prompt_tokens                              → prompt_tokens (TOTAL,
///                                                  includes cached_prompt)
///     completion_tokens                          → completion_tokens (TOTAL,
///                                                  includes reasoning)
///     prompt_tokens_details.cached_tokens        → cached_prompt_tokens
///     completion_tokens_details.reasoning_tokens → reasoning_tokens
///
///   Anthropic Messages API response.usage:
///     input_tokens                  → prompt_tokens (NON-cached input)
///     output_tokens                 → completion_tokens
///     cache_creation_input_tokens   → cache_creation_tokens
///     cache_read_input_tokens       → cache_read_tokens
///
/// Provider bridges that don't surface these (gemini, deepseek,
/// mistral, …) leave the four new counters at 0; cp-api treats 0 as
/// "no distinct rate" and falls back to the standard prompt /
/// completion price for that token class.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsageStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// OpenAI prompt-cache hit count. Subset of `prompt_tokens`.
    #[serde(default)]
    pub cached_prompt_tokens: u32,
    /// OpenAI o1/o3 reasoning tokens. Subset of `completion_tokens`.
    #[serde(default)]
    pub reasoning_tokens: u32,
    /// Anthropic cache_creation_input_tokens (cache write). Separate
    /// counter on top of input_tokens.
    #[serde(default)]
    pub cache_creation_tokens: u32,
    /// Anthropic cache_read_input_tokens (cache read). Separate
    /// counter on top of input_tokens.
    #[serde(default)]
    pub cache_read_tokens: u32,
}

impl UsageStats {
    pub fn new(prompt: u32, completion: u32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt.saturating_add(completion),
            ..Self::default()
        }
    }
}

/// Full (non-streaming) chat response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub message: ChatMessage,
    pub finish_reason: FinishReason,
    pub usage: UsageStats,
}

/// One streamed delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatChunk {
    pub id: String,
    pub model: String,
    pub delta: ChatDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageStats>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

/// Single embedding object as returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub index: u32,
    pub object: String,
    pub embedding: Vec<f32>,
}

/// Normalised embedding request.
///
/// The `input` is either a single string or a list of strings. We
/// represent both as `Vec<String>` — single-string inputs are wrapped in
/// a one-element vec by the proxy handler before passing to a Bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// The public-facing model name (resolved to an upstream model by the
    /// proxy before the Bridge sees it).
    pub model: String,
    /// Texts to embed. A single-string input is normalised to
    /// `vec![text]` by the proxy handler.
    pub input: Vec<String>,
    /// Optional encoding hint forwarded verbatim (`float` / `base64`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// Optional dimensions hint forwarded verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

/// Normalised embedding response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<EmbeddingObject>,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_format_round_trips_through_json() {
        let f = ChatFormat {
            model: "my-gpt4".into(),
            messages: vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::user("hi"),
            ],
            temperature: Some(0.2),
            top_p: None,
            max_tokens: Some(100),
            stream: Some(true),
            extra: serde_json::Map::new(),
        };

        let json = serde_json::to_string(&f).unwrap();
        let back: ChatFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "my-gpt4");
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.temperature, Some(0.2));
        assert!(back.is_streaming());
    }

    #[test]
    fn extras_capture_unknown_top_level_fields() {
        // `top_k` isn't a known field — it lands in `extra` so the Bridge
        // can decide whether to forward it to the upstream provider.
        let json = r#"{
            "model": "my-gpt4",
            "messages": [],
            "top_k": 40
        }"#;
        let f: ChatFormat = serde_json::from_str(json).unwrap();
        assert_eq!(f.extra.get("top_k").and_then(|v| v.as_u64()), Some(40));
    }

    #[test]
    fn is_streaming_defaults_to_false_when_unset() {
        let f = ChatFormat::new("m", vec![]);
        assert!(!f.is_streaming());
    }

    #[test]
    fn finish_reason_known_variants_are_snake_case() {
        let stop: FinishReason = serde_json::from_str(r#""stop""#).unwrap();
        let content_filter: FinishReason = serde_json::from_str(r#""content_filter""#).unwrap();
        assert_eq!(stop, FinishReason::Stop);
        assert_eq!(content_filter, FinishReason::ContentFilter);
    }

    #[test]
    fn usage_stats_saturates_total() {
        let u = UsageStats::new(u32::MAX, 10);
        assert_eq!(u.total_tokens, u32::MAX);
    }

    #[test]
    fn message_constructors_set_role() {
        assert_eq!(ChatMessage::system("x").role, Role::System);
        assert_eq!(ChatMessage::user("x").role, Role::User);
        assert_eq!(ChatMessage::assistant("x").role, Role::Assistant);
    }

    // ---- regression coverage for issue #110 -------------------------
    // Standard OpenAI / LangChain SDKs replay full conversation history
    // including assistant tool_calls / refusal / audio fields. Until
    // this fix the gateway answered such requests with HTTP 422 because
    // ChatMessage was deny_unknown_fields. The tests below pin the new
    // contract: deserialise, round-trip on serialise, and accept null
    // content.

    #[test]
    fn chat_message_accepts_assistant_with_tool_calls() {
        let json = r#"{
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{}"}}
            ]
        }"#;
        let m: ChatMessage = serde_json::from_str(json).expect("must accept tool_calls");
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.content, ""); // null collapses to empty string
        assert!(m.extra.contains_key("tool_calls"));
    }

    #[test]
    fn chat_message_accepts_refusal_field() {
        // OpenAI added `refusal` 2024-08 for safety-refused completions.
        let json = r#"{
            "role": "assistant",
            "content": "",
            "refusal": "I can't help with that."
        }"#;
        let m: ChatMessage = serde_json::from_str(json).expect("must accept refusal");
        assert_eq!(
            m.extra.get("refusal").and_then(|v| v.as_str()),
            Some("I can't help with that.")
        );
    }

    #[test]
    fn chat_message_accepts_audio_field() {
        // 4o-audio outputs include an `audio` block on assistant messages.
        let json = r#"{
            "role": "assistant",
            "content": "",
            "audio": {"id": "audio_1", "data": "...", "transcript": "hi"}
        }"#;
        let m: ChatMessage = serde_json::from_str(json).expect("must accept audio");
        assert!(m.extra.get("audio").and_then(|v| v.as_object()).is_some());
    }

    #[test]
    fn chat_message_accepts_null_content() {
        // The OpenAI assistant-with-tool_calls shape uses content: null;
        // we collapse to "" so downstream Bridges that don't accept null
        // (Anthropic, Gemini) still get a string.
        let json = r#"{"role": "assistant", "content": null}"#;
        let m: ChatMessage = serde_json::from_str(json).expect("must accept null content");
        assert_eq!(m.content, "");
    }

    #[test]
    fn chat_message_round_trips_full_openai_history_with_tool_calls() {
        // Full history shape the OpenAI SDK replays after a tool round.
        let json = r#"[
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "w", "arguments": "{}"}}]},
            {"role": "tool", "content": "75F", "tool_call_id": "c1"},
            {"role": "user", "content": "tomorrow?"}
        ]"#;
        let msgs: Vec<ChatMessage> =
            serde_json::from_str(json).expect("OpenAI replay history must parse");
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert!(msgs[1].extra.contains_key("tool_calls"));
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("c1"));

        // Re-serialise; tool_calls survives via the flatten extra map.
        let back = serde_json::to_string(&msgs).unwrap();
        assert!(
            back.contains("\"tool_calls\""),
            "tool_calls must round-trip through Serialize: {back}"
        );
    }

    #[test]
    fn chat_chunk_omits_optional_fields_on_wire() {
        let chunk = ChatChunk {
            id: "cmpl-1".into(),
            model: "m".into(),
            delta: ChatDelta {
                role: None,
                content: Some("hello".into()),
            },
            finish_reason: None,
            usage: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(!json.contains("\"finish_reason\""));
        assert!(!json.contains("\"usage\""));
        assert!(json.contains("\"content\":\"hello\""));
    }
}
