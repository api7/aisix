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

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
/// `content` accepts three wire shapes per
/// <https://platform.openai.com/docs/api-reference/chat/create>:
///   * a string (the common case);
///   * JSON `null` (OpenAI's assistant-with-tool_calls history shape);
///   * an array of typed content blocks
///     (`[{type: "text", text}, {type: "image_url", image_url: {url}}]` —
///     used by vision/multimodal callers).
///
/// We split the array form across two fields so existing call sites
/// keep their `&str` access path:
///   * [`Self::content`] holds the concatenated **text** of any text
///     blocks. For non-array shapes this is the original string (or
///     `""` for `null`). Bridges that don't speak content blocks
///     (Anthropic / Gemini cross-provider translation today) read this
///     and silently skip non-text blocks per docs §4.5.
///   * [`Self::content_blocks`] holds the **raw array** verbatim when
///     the caller sent the typed-block form. Bridges that DO support
///     content blocks (the OpenAI-compat bridge) forward this verbatim
///     to the upstream so vision input reaches OpenAI / Gemini /
///     DeepSeek upstreams unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "ChatMessageRaw")]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// Raw content-block array when the caller sent
    /// `content: [{type, ...}, ...]`. `None` for the bare-string and
    /// `null` content shapes. Bridges that support content blocks
    /// (the OpenAI-compat bridge) forward this verbatim to upstream;
    /// bridges that don't (Anthropic / Gemini cross-provider
    /// translation today) consult only `content` (concatenated text)
    /// per docs §4.5 ("skip non-text blocks silently on the inbound
    /// parse").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_blocks: Option<Vec<Value>>,
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

/// Wire-shape mirror for [`ChatMessage`]. The `content` field accepts
/// string OR null OR array per OpenAI's documented shape; we
/// deserialize through this struct and split into the `(text, blocks)`
/// pair on the way to [`ChatMessage`].
///
/// `content_blocks` is also accepted on the wire for round-trip
/// safety: the derived `Serialize` on [`ChatMessage`] emits both
/// `content` and `content_blocks` as separate top-level fields, so
/// re-deserialising must capture them both. (Without this, a cache
/// store-then-load round-trip would silently drop the typed blocks
/// into `extra` and the OpenAI bridge would forward only the
/// concatenated text, defeating vision.)
#[derive(Debug, Deserialize)]
struct ChatMessageRaw {
    role: Role,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    content_blocks: Option<Vec<Value>>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default, flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

impl From<ChatMessageRaw> for ChatMessage {
    fn from(raw: ChatMessageRaw) -> Self {
        let (content, derived_blocks) = split_content(raw.content);
        // If the wire form supplied `content_blocks` explicitly
        // (round-trip from a previous serialization), prefer it over
        // anything we'd derive from `content`. Otherwise use the
        // blocks extracted from the array-form of `content`.
        let content_blocks = raw.content_blocks.or(derived_blocks);
        Self {
            role: raw.role,
            content,
            content_blocks,
            name: raw.name,
            tool_call_id: raw.tool_call_id,
            extra: raw.extra,
        }
    }
}

/// Split a wire-form `content` value into the gateway's
/// `(extracted_text, raw_blocks)` representation:
///   * String → (string, None)
///   * null → ("", None) — OpenAI's assistant-with-tool_calls history
///     shape per <https://platform.openai.com/docs/api-reference/chat/create>
///   * Array → (concatenated text from `{type:"text", text}` blocks,
///     Some(raw array)) — vision / multimodal input. Non-text blocks
///     (e.g. `image_url`) are skipped on the text-extraction path but
///     preserved verbatim in the raw array for forwarding.
///   * Anything else → ("", None) — defensive default; unexpected
///     shapes don't fail the request, they degrade to an empty text
///     so the bridge can still dispatch.
fn split_content(v: Value) -> (String, Option<Vec<Value>>) {
    match v {
        Value::String(s) => (s, None),
        Value::Null => (String::new(), None),
        Value::Array(blocks) => {
            let text = blocks
                .iter()
                .filter_map(|b| {
                    let ty = b.get("type").and_then(Value::as_str)?;
                    if ty == "text" {
                        b.get("text").and_then(Value::as_str).map(str::to_owned)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("");
            (text, Some(blocks))
        }
        _ => (String::new(), None),
    }
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            content_blocks: None,
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            content_blocks: None,
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            content_blocks: None,
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
    /// DeepSeek-native `prompt_cache_hit_tokens`, preserved verbatim
    /// for client passthrough (#542). The canonical `cached_prompt_tokens`
    /// above carries the *normalized* (OpenAI-shape) cache-hit count for
    /// all providers; this Option carries the raw provider-native field
    /// so a DeepSeek-aware client reading the native name still works.
    /// `None` for providers that don't emit it — distinct from a real
    /// `Some(0)` so the renderer only emits the field when the upstream
    /// actually sent it. Bounded allowlist, not arbitrary passthrough.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_hit_tokens: Option<u32>,
    /// DeepSeek-native `prompt_cache_miss_tokens`, preserved verbatim
    /// (#542). See `prompt_cache_hit_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_miss_tokens: Option<u32>,
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

impl ChatResponse {
    /// The client-visible output text that content/DLP output guardrails
    /// must inspect: the assistant `content` plus any `tool_calls`
    /// material (function names + arguments, and Anthropic `tool_use`
    /// normalized into the same `extra["tool_calls"]` slot). Tool-call
    /// output is rendered to clients but would otherwise bypass output
    /// guardrails that only read `message.content` (#448).
    ///
    /// Reasoning/thinking content is intentionally NOT included — it is
    /// left out of output-guardrail scope by design.
    pub fn guardrail_output_text(&self) -> String {
        let mut out = self.message.content.clone();
        if let Some(tool_calls) = self.message.extra.get("tool_calls") {
            if !tool_calls.is_null() {
                if !out.is_empty() {
                    out.push('\n');
                }
                // Serialize the whole tool-call payload so no function
                // name or argument can escape inspection regardless of the
                // provider-specific shape.
                out.push_str(&tool_calls.to_string());
            }
        }
        out
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Reasoning-content slot the DP renders into `delta
    /// .reasoning_content` on the customer-visible SSE chunk. Populated
    /// by the Bridge after applying the
    /// [`response.reasoning_field`](aisix_core::ResponseOverrides::reasoning_field)
    /// path — issue #302 §5. `None` for upstreams that don't carry a
    /// reasoning field or where cp-api didn't configure a path. Matches
    /// DeepSeek's canonical `delta.reasoning_content` shape so the
    /// emitter is a passthrough.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

/// The vector returned on one embedding object. Untagged enum so JSON
/// round-trips OpenAI's documented `string | array` shape on the
/// `embedding` field (issue #393):
///
///   - `Float(vec![0.1, 0.2, ...])` → JSON array of numbers; what
///     OpenAI returns when the request carries
///     `encoding_format: "float"`.
///   - `Base64("BASE64STRING")` → JSON string; what OpenAI returns
///     when the request carries `encoding_format: "base64"` (the
///     SDK default). Stored verbatim — the gateway is a pure
///     pass-through for this field so callers who chose `base64`
///     for payload-size reasons see the same bytes the upstream
///     returned.
///
/// The gateway does NOT translate between the two formats. If a
/// future caller needs cross-format translation, that belongs at
/// the dispatcher, not the wire layer.
///
/// Reference: <https://platform.openai.com/docs/api-reference/embeddings/object>.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

/// Single embedding object as returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub index: u32,
    pub object: String,
    pub embedding: EmbeddingVector,
}

/// Normalised embedding request.
///
/// The `input` is either a single string or a list of strings. We
/// represent both as `Vec<String>` — single-string inputs are wrapped in
/// a one-element vec by the proxy handler before passing to a Bridge.
/// Per #162 / `docs/api-proxy.md` §4.4 ("both pass through"), the
/// original wire shape is preserved through `input_was_single` so the
/// bridge can serialise back to a single string when that's what the
/// caller sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// The public-facing model name (resolved to an upstream model by the
    /// proxy before the Bridge sees it).
    pub model: String,
    /// Texts to embed. A single-string input is normalised to
    /// `vec![text]` by the proxy handler; bridges consult
    /// `input_was_single` to decide the upstream wire shape.
    pub input: Vec<String>,
    /// `true` iff the caller originally sent `input` as a single
    /// string (not an array). Bridges that forward to upstreams
    /// supporting both shapes (OpenAI does) MUST preserve this on
    /// the wire, per docs §4.4 "both pass through". Defaults to
    /// `false` when missing on round-trip deserialisation so older
    /// callers / round-tripped requests that always wrote arrays
    /// don't change behaviour silently.
    #[serde(default)]
    pub input_was_single: bool,
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
    fn content_accepts_string() {
        let m: ChatMessage = serde_json::from_str(r#"{"role": "user", "content": "hi"}"#).unwrap();
        assert_eq!(m.content, "hi");
        assert!(m.content_blocks.is_none());
    }

    #[test]
    fn content_accepts_null_collapsing_to_empty_string() {
        let m: ChatMessage =
            serde_json::from_str(r#"{"role": "assistant", "content": null}"#).unwrap();
        assert_eq!(m.content, "");
        assert!(m.content_blocks.is_none());
    }

    #[test]
    fn content_accepts_typed_block_array_for_vision() {
        // OpenAI vision request shape per
        // <https://platform.openai.com/docs/guides/vision>.
        let m: ChatMessage = serde_json::from_str(
            r#"{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
                ]
            }"#,
        )
        .unwrap();
        // Concatenated text from text blocks (non-text blocks skipped).
        assert_eq!(m.content, "What's in this image?");
        // Raw blocks preserved verbatim for forwarding.
        let blocks = m.content_blocks.expect("blocks should be Some");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image_url");
        assert_eq!(blocks[1]["image_url"]["url"], "https://example.com/cat.jpg");
    }

    #[test]
    fn content_array_with_only_image_blocks_yields_empty_text_but_keeps_blocks() {
        let m: ChatMessage = serde_json::from_str(
            r#"{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(m.content, "");
        assert!(m.content_blocks.is_some());
    }

    #[test]
    fn content_array_concatenates_multiple_text_blocks() {
        let m: ChatMessage = serde_json::from_str(
            r#"{
                "role": "user",
                "content": [
                    {"type": "text", "text": "line one\n"},
                    {"type": "text", "text": "line two"}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(m.content, "line one\nline two");
    }

    #[test]
    fn content_blocks_round_trip_through_serialization() {
        // Regression test for PR #184 audit (C2): without this, a
        // cache store-then-load (or any debug serialise→deserialise)
        // would silently drop `content_blocks` into `extra` and the
        // OpenAI bridge would forward only the concatenated text,
        // defeating vision. ChatMessageRaw must accept
        // `content_blocks` on the wire.
        let original: ChatMessage = serde_json::from_str(
            r#"{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
                ]
            }"#,
        )
        .unwrap();
        assert!(original.content_blocks.is_some());

        // Serialise → string → deserialise. Blocks must survive.
        let json = serde_json::to_string(&original).unwrap();
        let round_tripped: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped.content, original.content);
        assert_eq!(round_tripped.content_blocks, original.content_blocks);
        // `content_blocks` MUST NOT have leaked into `extra` (which
        // would happen if ChatMessageRaw didn't capture the field).
        assert!(!round_tripped.extra.contains_key("content_blocks"));
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

    /// PR #442 audit MEDIUM-4 (forward-compat): an *old-shape*
    /// `UsageStats` JSON — written before the #542 native cache fields
    /// existed — must still deserialize cleanly into the new struct,
    /// defaulting the new Option fields to `None`. This pins the
    /// new-DP-reads-old-cache direction of a mixed-version rolling
    /// deploy (the `ChatResponse` type is persisted to the Redis cache).
    /// The reverse direction (old DP reading new-shape) is bounded +
    /// documented in the PR body.
    #[test]
    fn usage_stats_deserializes_old_shape_without_native_cache_fields() {
        let old = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cached_prompt_tokens": 40,
            "reasoning_tokens": 5,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0
        }"#;
        let u: UsageStats =
            serde_json::from_str(old).expect("old-shape UsageStats must deserialize");
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.cached_prompt_tokens, 40);
        assert_eq!(u.prompt_cache_hit_tokens, None);
        assert_eq!(u.prompt_cache_miss_tokens, None);
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
                tool_calls: None,
                reasoning_content: None,
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
