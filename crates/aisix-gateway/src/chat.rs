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

/// Role of a chat message. Provider bridges are responsible for preserving
/// roles that their upstream accepts and translating the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
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
///     and silently skip non-text blocks (images/audio): a cross-provider
///     request keeps only the text. Documented for users under
///     "Cross-Provider Content Limitations" in the provider-compatibility
///     reference.
///   * [`Self::content_blocks`] holds the **raw array** verbatim when
///     the caller sent the typed-block form. Bridges that DO support
///     content blocks (the OpenAI-compat bridge) forward this verbatim
///     to the upstream so vision input reaches OpenAI / Gemini /
///     DeepSeek upstreams unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "ChatMessageRaw")]
pub struct ChatMessage {
    pub role: Role,
    /// Assistant/text content. `Option<String>` (not `String`) so the
    /// OpenAI `string | null` shape round-trips faithfully: a `tool_calls`
    /// response upstream returns `content: null` and we must surface
    /// exactly `null` to the SDK caller, not `""` (#395). On the request
    /// path callers always send a string; `None` only arises on the
    /// response-projection path (or an inbound `content: null` from a
    /// history-replay assistant message).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Raw content-block array when the caller sent
    /// `content: [{type, ...}, ...]`. `None` for the bare-string and
    /// `null` content shapes. Bridges that support content blocks
    /// (the OpenAI-compat bridge) forward this verbatim to upstream.
    /// The Anthropic /v1/messages inbound parse translates its own
    /// block types INTO this OpenAI-shaped array (#722); bridges that
    /// don't understand blocks consult only `content` (concatenated
    /// text).
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
///   * String → (Some(string), None)
///   * null → (None, None) — OpenAI's assistant-with-tool_calls history
///     shape per <https://platform.openai.com/docs/api-reference/chat/create>.
///     Preserved as `None` so it round-trips back to `null` (#395).
///   * Array → (Some(concatenated text from `{type:"text", text}`
///     blocks), Some(raw array)) — vision / multimodal input. Non-text
///     blocks (e.g. `image_url`) are skipped on the text-extraction path
///     but preserved verbatim in the raw array for forwarding.
///   * Anything else → (None, None) — defensive default; unexpected
///     shapes don't fail the request, they degrade to absent text
///     so the bridge can still dispatch.
fn split_content(v: Value) -> (Option<String>, Option<Vec<Value>>) {
    match v {
        Value::String(s) => (Some(s), None),
        Value::Null => (None, None),
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
            (Some(text), Some(blocks))
        }
        _ => (None, None),
    }
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            content_blocks: None,
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }

    pub fn developer(content: impl Into<String>) -> Self {
        Self {
            role: Role::Developer,
            content: Some(content.into()),
            content_blocks: None,
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            content_blocks: None,
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            content_blocks: None,
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }

    /// The text content as a `&str`, treating absent (`null`) content as
    /// `""`. Use this for bridges/guardrails that need a plain string and
    /// for which the string-vs-null distinction is irrelevant.
    pub fn content_str(&self) -> &str {
        self.content.as_deref().unwrap_or("")
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

    /// Build usage for a provider that reports cache tokens as counters
    /// *separate* from `prompt_tokens` — i.e. Anthropic, where the true
    /// input is `input_tokens + cache_creation + cache_read` rather than
    /// a single `prompt_tokens` that already includes the cached part
    /// (the OpenAI shape). `total_tokens` therefore folds the cache
    /// counters in, so it stays the honest total instead of
    /// `prompt + completion` alone (#906 / AISIX-Cloud#906). OpenAI
    /// upstreams keep using `new()` — their cache hit is a subset of
    /// `prompt_tokens`, so it must NOT be added again here.
    pub fn with_cache(prompt: u32, completion: u32, cache_creation: u32, cache_read: u32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt
                .saturating_add(completion)
                .saturating_add(cache_creation)
                .saturating_add(cache_read),
            cache_creation_tokens: cache_creation,
            cache_read_tokens: cache_read,
            ..Self::default()
        }
    }

    /// Field-wise saturating sum of two usage records. Used to build an
    /// ensemble's client-facing aggregate usage — the sum of every panel
    /// member plus the judge (api7/AISIX-Cloud#804) — so a fan-out request
    /// reports its full multiplicative cost to the caller rather than a
    /// single sub-call's. The optional provider-native passthrough counters
    /// (DeepSeek hit/miss, #542) add with `None` treated as 0, staying
    /// `None` only when neither side carried one.
    pub fn saturating_add(&self, other: &UsageStats) -> UsageStats {
        fn add_opt(a: Option<u32>, b: Option<u32>) -> Option<u32> {
            match (a, b) {
                (None, None) => None,
                _ => Some(a.unwrap_or(0).saturating_add(b.unwrap_or(0))),
            }
        }
        UsageStats {
            prompt_tokens: self.prompt_tokens.saturating_add(other.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .saturating_add(other.completion_tokens),
            total_tokens: self.total_tokens.saturating_add(other.total_tokens),
            cached_prompt_tokens: self
                .cached_prompt_tokens
                .saturating_add(other.cached_prompt_tokens),
            reasoning_tokens: self.reasoning_tokens.saturating_add(other.reasoning_tokens),
            cache_creation_tokens: self
                .cache_creation_tokens
                .saturating_add(other.cache_creation_tokens),
            cache_read_tokens: self
                .cache_read_tokens
                .saturating_add(other.cache_read_tokens),
            prompt_cache_hit_tokens: add_opt(
                self.prompt_cache_hit_tokens,
                other.prompt_cache_hit_tokens,
            ),
            prompt_cache_miss_tokens: add_opt(
                self.prompt_cache_miss_tokens,
                other.prompt_cache_miss_tokens,
            ),
        }
    }

    // ── Client-facing protocol projections ─────────────────────────
    //
    // `UsageStats` stores whichever accounting shape the UPSTREAM used,
    // because that is what UsageEvent / Prometheus / billing must keep
    // reporting (AISIX-Cloud#1447). The client, though, must be answered
    // in ITS OWN protocol's accounting — so every client-facing renderer
    // projects through the methods below rather than copying fields.
    //
    // The two shapes disagree on one thing only: whether the cache
    // counters live INSIDE the input count or BESIDE it.
    //
    //   OpenAI shape     prompt_tokens INCLUDES cached_prompt_tokens
    //   Anthropic shape  prompt_tokens EXCLUDES cache_creation/read
    //
    // A single upstream fills exactly one family and zeroes the other,
    // so each projection is a plain fold that reduces to the identity
    // on its own shape. Summing rather than picking also keeps an
    // ENSEMBLE aggregate right: `saturating_add` can merge an
    // OpenAI-shape member with an Anthropic-shape one, and only the
    // additive form reports both members' cache hits.
    //
    // Both directions are defined here, together, on purpose: they are
    // two halves of one contract, and splitting them across the two
    // renderer crates is how one gets fixed while the other silently
    // keeps the old semantics.

    /// Total input the model processed, in either accounting shape —
    /// the quantity both protocols agree on, since it is what the
    /// upstream actually billed for.
    pub fn total_input_tokens(&self) -> u32 {
        self.prompt_tokens
            .saturating_add(self.cache_creation_tokens)
            .saturating_add(self.cache_read_tokens)
    }

    /// OpenAI `usage.prompt_tokens` / Responses `usage.input_tokens`:
    /// the FULL input, cache hits and cache writes included.
    pub fn openai_prompt_tokens(&self) -> u32 {
        self.total_input_tokens()
    }

    /// OpenAI `prompt_tokens_details.cached_tokens` /
    /// Responses `input_tokens_details.cached_tokens`: the cache-read
    /// portion, a SUBSET of [`Self::openai_prompt_tokens`]. A cache
    /// WRITE is billed input but is not a hit, so it is deliberately
    /// not counted here.
    pub fn openai_cached_tokens(&self) -> u32 {
        self.cached_prompt_tokens
            .saturating_add(self.cache_read_tokens)
    }

    /// OpenAI `usage.total_tokens`.
    ///
    /// Recomputed only when the projection actually MOVED tokens between
    /// fields — that is, when the upstream reported its cache counters
    /// beside the input. There the stored total was built under the
    /// other accounting and relaying it is what produced #1447's
    /// `40 + 10 = 150`; and a Bedrock Converse total, which excludes the
    /// cache entirely, would under-report.
    ///
    /// Otherwise the upstream's own total stands, because recomputing it
    /// can only lose information: a provider counting overhead we cannot
    /// see would be silently corrected downward, and one that reports a
    /// bare `total_tokens` with no breakdown would render as `0`.
    pub fn openai_total_tokens(&self) -> u32 {
        let converted = self.cache_creation_tokens > 0 || self.cache_read_tokens > 0;
        if !converted && self.total_tokens > 0 {
            return self.total_tokens;
        }
        self.openai_prompt_tokens()
            .saturating_add(self.completion_tokens)
    }

    /// Anthropic `usage.input_tokens`: NON-cached input only.
    pub fn anthropic_input_tokens(&self) -> u32 {
        self.prompt_tokens.saturating_sub(self.cached_prompt_tokens)
    }

    /// Anthropic `usage.cache_read_input_tokens`, a counter BESIDE
    /// `input_tokens`.
    pub fn anthropic_cache_read_input_tokens(&self) -> u32 {
        self.cache_read_tokens
            .saturating_add(self.cached_prompt_tokens)
    }

    /// Anthropic `usage.cache_creation_input_tokens`. No OpenAI-shape
    /// upstream reports a cache write, so there is nothing to fold in.
    pub fn anthropic_cache_creation_input_tokens(&self) -> u32 {
        self.cache_creation_tokens
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
        let mut out = self.message.content.clone().unwrap_or_default();
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
        assert_eq!(m.content.as_deref(), Some("hi"));
        assert!(m.content_blocks.is_none());
    }

    #[test]
    fn chat_message_accepts_and_round_trips_developer_role() {
        let m: ChatMessage = serde_json::from_str(
            r#"{"role": "developer", "content": "Follow application instructions"}"#,
        )
        .expect("developer is a documented Chat Completions message role");

        let value = serde_json::to_value(m).unwrap();
        assert_eq!(value["role"], "developer");
        assert_eq!(value["content"], "Follow application instructions");
    }

    #[test]
    fn chat_message_rejects_unknown_and_mis_cased_roles() {
        for role in ["Developer", "tool_result"] {
            let raw = format!(r#"{{"role":"{role}","content":"x"}}"#);
            assert!(
                serde_json::from_str::<ChatMessage>(&raw).is_err(),
                "role {role:?} must remain invalid"
            );
        }
    }

    #[test]
    fn content_null_preserved_as_none() {
        // #395: `content: null` (OpenAI's assistant-with-tool_calls shape)
        // is preserved as `None` so it round-trips back to JSON `null`,
        // not `""`.
        let m: ChatMessage =
            serde_json::from_str(r#"{"role": "assistant", "content": null}"#).unwrap();
        assert_eq!(m.content, None);
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
        assert_eq!(m.content.as_deref(), Some("What's in this image?"));
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
        // An array of only non-text blocks yields empty-but-present text
        // (the array form is never `null`); blocks are preserved.
        assert_eq!(m.content.as_deref(), Some(""));
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
        assert_eq!(m.content.as_deref(), Some("line one\nline two"));
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

    #[test]
    fn usage_stats_with_cache_folds_cache_into_total() {
        // #906: cache_creation / cache_read are input classes separate
        // from prompt_tokens, so the total is prompt + completion +
        // cache_creation + cache_read — not prompt + completion alone.
        let u = UsageStats::with_cache(10, 4, 200, 800);
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 4);
        assert_eq!(u.cache_creation_tokens, 200);
        assert_eq!(u.cache_read_tokens, 800);
        assert_eq!(u.total_tokens, 1014);
        // No cache present degrades to the plain prompt + completion total.
        assert_eq!(UsageStats::with_cache(10, 4, 0, 0).total_tokens, 14);
    }

    // ── Client-facing protocol projections ─────────────────────────
    //
    // The reporter's numbers from AISIX-Cloud#1447, used throughout so
    // the same call is followed across every source shape: 40 uncached
    // input + 30 cache write + 70 cache read + 10 output. Whoever
    // reports it, the model processed 140 input tokens and 150 total.
    const N: u32 = 40;
    const W: u32 = 30;
    const R: u32 = 70;
    const O: u32 = 10;

    /// What an Anthropic-shape upstream (anthropic, bedrock) stores.
    fn anthropic_shape() -> UsageStats {
        UsageStats::with_cache(N, O, W, R)
    }

    /// What an OpenAI-shape upstream (openai, azure, deepseek, gemini)
    /// stores for the SAME call: no cache-write concept, and the read is
    /// already inside `prompt_tokens`.
    fn openai_shape() -> UsageStats {
        UsageStats {
            prompt_tokens: N + W + R,
            completion_tokens: O,
            total_tokens: N + W + R + O,
            cached_prompt_tokens: R,
            ..UsageStats::default()
        }
    }

    #[test]
    fn projections_agree_on_total_input_across_source_shapes() {
        // The whole point of the projection layer: an OpenAI client and
        // an Anthropic client, in front of EITHER upstream shape, must
        // be told the model read the same 140 tokens.
        for u in [anthropic_shape(), openai_shape()] {
            assert_eq!(u.total_input_tokens(), N + W + R);
            assert_eq!(u.openai_prompt_tokens(), N + W + R);
            assert_eq!(
                u.anthropic_input_tokens()
                    + u.anthropic_cache_creation_input_tokens()
                    + u.anthropic_cache_read_input_tokens(),
                N + W + R,
                "Anthropic's three input counters must sum to the same input"
            );
        }
    }

    #[test]
    fn openai_projection_keeps_cached_a_subset_of_prompt() {
        for u in [anthropic_shape(), openai_shape()] {
            assert!(
                u.openai_cached_tokens() <= u.openai_prompt_tokens(),
                "cached_tokens > prompt_tokens is the contradiction #1447 reported"
            );
            assert_eq!(u.openai_cached_tokens(), R, "cache READ is the hit");
            assert_eq!(
                u.openai_total_tokens(),
                u.openai_prompt_tokens() + u.completion_tokens,
                "OpenAI clients decompose total into prompt + completion"
            );
            assert_eq!(u.openai_total_tokens(), N + W + R + O);
        }
    }

    #[test]
    fn anthropic_projection_keeps_input_free_of_cache() {
        // The cache READ is reportable by both shapes, so it converts
        // exactly either way.
        for u in [anthropic_shape(), openai_shape()] {
            assert_eq!(u.anthropic_cache_read_input_tokens(), R);
        }
        // The cache WRITE does not: OpenAI accounting has no such
        // bucket, so an OpenAI upstream's non-hit input is all plain
        // input. Fabricating a write to make the two shapes look alike
        // would invent a number the upstream never sent.
        let ant = anthropic_shape();
        assert_eq!(ant.anthropic_input_tokens(), N);
        assert_eq!(ant.anthropic_cache_creation_input_tokens(), W);

        let oai = openai_shape();
        assert_eq!(oai.anthropic_input_tokens(), N + W);
        assert_eq!(oai.anthropic_cache_creation_input_tokens(), 0);
    }

    /// `total_tokens` is only recomputed when the projection moved
    /// tokens. Recomputing unconditionally erases whatever an upstream
    /// counted outside `prompt + completion` — including the extreme
    /// case of one that reports a bare total and no breakdown, which
    /// would have rendered as `0`.
    #[test]
    fn openai_total_defers_to_the_upstream_unless_the_shape_converted() {
        // Converted: the stored total was built under Anthropic
        // accounting, so it must not be relayed as-is.
        assert_eq!(anthropic_shape().openai_total_tokens(), N + W + R + O);
        // …and a Bedrock-style total that EXCLUDES the cache is likewise
        // recomputed rather than under-reported.
        let bedrock_like = UsageStats {
            prompt_tokens: N,
            completion_tokens: O,
            total_tokens: N + O,
            cache_creation_tokens: W,
            cache_read_tokens: R,
            ..UsageStats::default()
        };
        assert_eq!(bedrock_like.openai_total_tokens(), N + W + R + O);

        // Not converted: the upstream's own total stands, even when it
        // exceeds prompt + completion because the provider is counting
        // overhead we cannot see.
        let with_overhead = UsageStats {
            prompt_tokens: 1000,
            completion_tokens: 400,
            total_tokens: 1500,
            ..UsageStats::default()
        };
        assert_eq!(with_overhead.openai_total_tokens(), 1500);

        // The degenerate upstream: a total and nothing else.
        let total_only = UsageStats {
            total_tokens: 42,
            ..UsageStats::default()
        };
        assert_eq!(total_only.openai_total_tokens(), 42);

        // A missing total still falls back to the parts.
        let no_total = UsageStats {
            prompt_tokens: 7,
            completion_tokens: 3,
            ..UsageStats::default()
        };
        assert_eq!(no_total.openai_total_tokens(), 10);
    }

    /// The first turn of a cached conversation WRITES without reading.
    /// Those tokens are billed input, so they are inside
    /// `openai_prompt_tokens` — and if nothing names them the caller
    /// sees an unexplained increase it cannot price, since a write costs
    /// more than plain input.
    #[test]
    fn a_cache_write_without_a_read_is_still_reportable() {
        let write_only = UsageStats::with_cache(N, O, W, 0);
        assert_eq!(write_only.openai_prompt_tokens(), N + W);
        assert_eq!(write_only.openai_cached_tokens(), 0, "a write is not a hit");
        assert_eq!(write_only.anthropic_cache_creation_input_tokens(), W);
        assert_eq!(write_only.openai_total_tokens(), N + W + O);
    }

    #[test]
    fn projections_are_the_identity_without_any_cache() {
        let u = UsageStats::new(N, O);
        assert_eq!(u.openai_prompt_tokens(), N);
        assert_eq!(u.openai_cached_tokens(), 0);
        assert_eq!(u.openai_total_tokens(), N + O);
        assert_eq!(u.anthropic_input_tokens(), N);
        assert_eq!(u.anthropic_cache_read_input_tokens(), 0);
        assert_eq!(u.anthropic_cache_creation_input_tokens(), 0);
    }

    #[test]
    fn projections_sum_both_cache_families_on_a_mixed_aggregate() {
        // An ensemble fans out to members on different providers, and
        // `saturating_add` merges their usage into ONE client-facing
        // record — the only place both cache representations co-occur.
        // Picking whichever is non-zero (an easy shortcut when each
        // upstream fills exactly one family) would silently drop a whole
        // member's cache hit here.
        let mixed = anthropic_shape().saturating_add(&openai_shape());
        assert_eq!(mixed.openai_prompt_tokens(), 2 * (N + W + R));
        assert_eq!(mixed.openai_cached_tokens(), 2 * R);
        assert_eq!(mixed.openai_total_tokens(), 2 * (N + W + R + O));

        // The Anthropic side sums to the same 2 × 140 of input, but
        // splits it differently, and correctly so: OpenAI accounting has
        // no cache-WRITE bucket, so that member's non-hit input is all
        // plain input. Only the Anthropic member contributes a write.
        assert_eq!(mixed.anthropic_input_tokens(), N + (W + N));
        assert_eq!(mixed.anthropic_cache_read_input_tokens(), 2 * R);
        assert_eq!(mixed.anthropic_cache_creation_input_tokens(), W);
        assert_eq!(
            mixed.anthropic_input_tokens()
                + mixed.anthropic_cache_creation_input_tokens()
                + mixed.anthropic_cache_read_input_tokens(),
            2 * (N + W + R)
        );
    }

    #[test]
    fn usage_stats_saturating_add_sums_every_field() {
        let a = UsageStats {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cached_prompt_tokens: 2,
            reasoning_tokens: 3,
            cache_creation_tokens: 1,
            cache_read_tokens: 4,
            prompt_cache_hit_tokens: Some(2),
            prompt_cache_miss_tokens: None,
        };
        let b = UsageStats {
            prompt_tokens: 20,
            completion_tokens: 7,
            total_tokens: 27,
            cached_prompt_tokens: 1,
            reasoning_tokens: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 6,
            prompt_cache_hit_tokens: None,
            prompt_cache_miss_tokens: Some(8),
        };
        let sum = a.saturating_add(&b);
        assert_eq!(sum.prompt_tokens, 30);
        assert_eq!(sum.completion_tokens, 12);
        assert_eq!(sum.total_tokens, 42);
        assert_eq!(sum.cached_prompt_tokens, 3);
        assert_eq!(sum.reasoning_tokens, 3);
        assert_eq!(sum.cache_creation_tokens, 1);
        assert_eq!(sum.cache_read_tokens, 10);
        // None + Some(8) = Some(8); Some(2) + None = Some(2). None + None stays None.
        assert_eq!(sum.prompt_cache_hit_tokens, Some(2));
        assert_eq!(sum.prompt_cache_miss_tokens, Some(8));
        assert_eq!(
            UsageStats::default()
                .saturating_add(&UsageStats::default())
                .prompt_cache_hit_tokens,
            None,
        );
        // Saturates instead of overflowing.
        assert_eq!(
            UsageStats::new(u32::MAX, 0)
                .saturating_add(&UsageStats::new(10, 0))
                .prompt_tokens,
            u32::MAX,
        );
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
        assert_eq!(ChatMessage::developer("x").role, Role::Developer);
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
        assert_eq!(m.content, None); // null preserved as None (#395)
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
        // we preserve it as `None` (#395) so it round-trips to JSON null.
        // Bridges that can't accept null (Anthropic, Gemini) read
        // `content_str()` which maps `None → ""`.
        let json = r#"{"role": "assistant", "content": null}"#;
        let m: ChatMessage = serde_json::from_str(json).expect("must accept null content");
        assert_eq!(m.content, None);
        assert_eq!(m.content_str(), "");
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
