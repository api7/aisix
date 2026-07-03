//! Anthropic `/v1/messages` wire shapes.
//!
//! Reference: <https://docs.anthropic.com/en/api/messages>
//!
//! Key differences from OpenAI that this module handles:
//!
//! - System prompt is a top-level `system` field, not a message with
//!   `role: "system"` — we collapse all leading system messages into one
//!   string and forward it there.
//! - Only `user` and `assistant` roles on the wire. `tool` messages from
//!   ChatFormat are rejected at the bridge boundary rather than being
//!   silently re-classified.
//! - Content is an array of blocks — we emit a single `{"type":"text",…}`
//!   block per message and read the concatenation of text blocks on the
//!   way back.
//! - `max_tokens` is required by Anthropic. We default to a safe ceiling
//!   when the client didn't set one, but log the fallback so operators
//!   can tune the default if desired.
//! - Streaming events are typed (`message_start`, `content_block_delta`,
//!   …). We only emit a `ChatChunk` when a delta carries content or a
//!   stop reason — other events just advance internal state.

use aisix_gateway::{
    ChatChunk, ChatDelta, ChatFormat, ChatMessage, ChatResponse, FinishReason, Role, UsageStats,
};
use serde::{Deserialize, Serialize};

/// Anthropic requires a non-zero `max_tokens`. Clients that omit it get
/// this ceiling — generous enough to cover normal completions, conservative
/// enough that a runaway prompt doesn't burn tokens silently.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<AnthropicMessage<'a>>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    pub stream: bool,
    /// Tools spec translated from the caller's OpenAI-shape `tools`
    /// (when present in `extra`). The gateway emits Anthropic's
    /// shape per <https://docs.anthropic.com/en/api/messages>:
    /// `{name, description, input_schema}`. `None` when the caller
    /// didn't request tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    /// `tool_choice` translated from OpenAI's shape per
    /// <https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice>
    /// to Anthropic's per
    /// <https://docs.anthropic.com/en/api/messages#parameter-tool_choice>:
    ///   "auto"|"none"|"required"           → `{type:<same>}` ("required" → "any")
    ///   {type:"function",function:{name}}  → `{type:"tool", name}`
    /// Forwarding the OpenAI shape verbatim would 400 the upstream.
    /// `None` when the caller didn't set tool_choice (and we strip
    /// it from `extra` to avoid double-emit / shape mismatch).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Caller's other extra fields (excluding `tools`, which is
    /// translated above). Anthropic-incompatible OpenAI-only fields
    /// here would cause a 400 upstream — operators are expected to
    /// configure their gateway client to send shape-appropriate
    /// extras. Trade-off: forward-compatibility with new Anthropic
    /// fields > strict filtering.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessage<'a> {
    pub role: &'a str,
    /// Polymorphic content blocks — text and `tool_result` blocks
    /// emit different shapes per
    /// <https://docs.anthropic.com/en/api/messages>. Stored as
    /// owned `Value` so OpenAI `Role::Tool` messages can be
    /// translated into Anthropic `{type:"tool_result", tool_use_id,
    /// content}` without lifetime gymnastics.
    pub content: Vec<serde_json::Value>,
    #[serde(skip)]
    _lifetime: std::marker::PhantomData<&'a ()>,
}

impl<'a> AnthropicMessage<'a> {
    /// Single-text-block message (the common case for
    /// system/user/assistant turns without tool use).
    pub(crate) fn text(role: &'a str, text: &'a str) -> Self {
        Self {
            role,
            content: vec![serde_json::json!({"type": "text", "text": text})],
            _lifetime: std::marker::PhantomData,
        }
    }

    /// Anthropic tool_result block per
    /// <https://docs.anthropic.com/en/api/messages#example-of-tool-use>.
    /// Translates the OpenAI `{role:"tool", tool_call_id, content}`
    /// turn so agent-loop round-trips work — without this, the
    /// caller's tool-result reply 400s at the Anthropic upstream.
    pub(crate) fn tool_result(tool_use_id: &str, content: &str) -> Self {
        Self {
            role: "user",
            content: vec![serde_json::json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            })],
            _lifetime: std::marker::PhantomData,
        }
    }

    /// Assistant turn replayed from conversation history, carrying its
    /// text (when any) plus any OpenAI-shape `tool_calls` translated into
    /// Anthropic `tool_use` blocks. An agent loop replays the assistant's
    /// prior tool calls before sending the matching tool results; without
    /// translating `tool_calls` here the following `tool_result` would
    /// reference a `tool_use` the upstream never saw and 400. Empty text
    /// with no tool calls degrades to an empty text block so the message
    /// isn't dropped (Anthropic rejects an empty `content` array).
    pub(crate) fn assistant(text: &str, tool_calls: Option<&[serde_json::Value]>) -> Self {
        let mut content: Vec<serde_json::Value> = Vec::new();
        if !text.is_empty() {
            content.push(serde_json::json!({"type": "text", "text": text}));
        }
        if let Some(tcs) = tool_calls {
            content.extend(tool_use_blocks_from_openai(tcs));
        }
        if content.is_empty() {
            content.push(serde_json::json!({"type": "text", "text": ""}));
        }
        Self {
            role: "assistant",
            content,
            _lifetime: std::marker::PhantomData,
        }
    }
}

/// Translate an array of OpenAI-shape `tool_calls`
/// (`{id, type:"function", function:{name, arguments}}`, `arguments` a
/// JSON string) into Anthropic `tool_use` content blocks
/// (`{type:"tool_use", id, name, input}`, `input` the parsed arguments
/// object). Entries missing an id or name are skipped; arguments that
/// don't parse to an object degrade to `{}`. Shared by the request-history
/// path ([`AnthropicMessage::assistant`]) and the response path
/// ([`chat_response_into_anthropic_json`]).
fn tool_use_blocks_from_openai(tool_calls: &[serde_json::Value]) -> Vec<serde_json::Value> {
    tool_calls
        .iter()
        .filter_map(|tc| {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())?;
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .filter(|s| !s.is_empty())?;
            let input = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                .filter(|v| v.is_object())
                .unwrap_or(serde_json::json!({}));
            Some(serde_json::json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }))
        })
        .collect()
}

/// Merge adjacent messages that share a role by concatenating their
/// content blocks. Anthropic requires strictly alternating user/assistant
/// turns; a multi-turn tool loop (or parallel tool calls) produces
/// consecutive same-role turns — e.g. several `tool_result` replies, each
/// a `user` turn — that the upstream rejects with "messages: roles must
/// alternate" unless folded into one message.
fn merge_consecutive_roles(messages: Vec<AnthropicMessage<'_>>) -> Vec<AnthropicMessage<'_>> {
    let mut merged: Vec<AnthropicMessage<'_>> = Vec::with_capacity(messages.len());
    for msg in messages {
        match merged.last_mut() {
            Some(last) if last.role == msg.role => last.content.extend(msg.content),
            _ => merged.push(msg),
        }
    }
    merged
}

#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    #[error("tool message missing tool_call_id field")]
    MissingToolCallId,
}

/// Split the gateway's flat ChatFormat into Anthropic's (system, messages)
/// shape. Consecutive system messages at the head are concatenated with
/// a blank line, matching how users typically compose multi-paragraph
/// system prompts in the OpenAI format.
///
/// Role::Tool turns translate to Anthropic's `{role:"user", content:
/// [{type:"tool_result", tool_use_id, content}]}` shape per
/// <https://docs.anthropic.com/en/api/messages> so agent-loop turn 2
/// (caller sends the tool's output back to the model) round-trips.
pub fn split_system<'a>(
    req: &'a ChatFormat,
) -> Result<(Option<String>, Vec<AnthropicMessage<'a>>), TranslateError> {
    let mut system_parts: Vec<&'a str> = Vec::new();
    let mut messages: Vec<AnthropicMessage<'a>> = Vec::new();
    let mut seen_non_system = false;

    for m in &req.messages {
        match m.role {
            Role::System => {
                if seen_non_system {
                    // System messages interleaved with user/assistant
                    // turns don't map cleanly; append as a user turn to
                    // preserve semantics without silently dropping them.
                    messages.push(AnthropicMessage::text("user", m.content_str()));
                } else {
                    system_parts.push(m.content_str());
                }
            }
            Role::User => {
                seen_non_system = true;
                messages.push(AnthropicMessage::text("user", m.content_str()));
            }
            Role::Assistant => {
                seen_non_system = true;
                let tool_calls = m
                    .extra
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .map(Vec::as_slice);
                messages.push(AnthropicMessage::assistant(m.content_str(), tool_calls));
            }
            Role::Tool => {
                seen_non_system = true;
                let tool_use_id = m
                    .tool_call_id
                    .as_deref()
                    .ok_or(TranslateError::MissingToolCallId)?;
                messages.push(AnthropicMessage::tool_result(tool_use_id, m.content_str()));
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    // Fold consecutive same-role turns so the alternating-role invariant
    // Anthropic enforces holds for multi-turn tool loops and parallel
    // tool calls.
    Ok((system, merge_consecutive_roles(messages)))
}

pub fn build_request<'a>(
    req: &'a ChatFormat,
    upstream_model: &'a str,
    system: Option<String>,
    messages: Vec<AnthropicMessage<'a>>,
    stream: bool,
) -> AnthropicRequest<'a> {
    // Pull `tools` and `tool_choice` out of the caller's extras and
    // translate to Anthropic shape; everything else passes through
    // verbatim. Forwarding the OpenAI tool_choice shape would 400
    // upstream — the field is removed from `extra` even when the
    // translation returns None (e.g. unrecognised value), to avoid
    // a shape-mismatch double-emit.
    let mut extras = req.extra.clone();
    let tools = extras
        .remove("tools")
        .and_then(translate_openai_tools_to_anthropic);
    let tool_choice = extras
        .remove("tool_choice")
        .and_then(translate_openai_tool_choice_to_anthropic);
    AnthropicRequest {
        model: upstream_model,
        messages,
        max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        system,
        temperature: req.temperature,
        top_p: req.top_p,
        stream,
        tools,
        tool_choice,
        extra: extras,
    }
}

/// Translate the caller's OpenAI-shape `tools` array into
/// Anthropic's tools-spec shape on the outbound axis. Field mapping
/// per <https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools>
/// and <https://docs.anthropic.com/en/api/messages#parameter-tools>:
///
///   OpenAI                                    Anthropic
///   {type: "function",                        {name,
///    function: {name, description,             description,
///               parameters}}                   input_schema}
///
/// Only `type: "function"` tools translate today; OpenAI's other
/// tool kinds (`code_interpreter`, `file_search`, …) have no
/// Anthropic equivalent and are dropped silently. Returns `None`
/// when the input isn't an array or when no entries translated —
/// keeping the field absent from the upstream wire shape so
/// Anthropic doesn't reject for empty-tools.
pub fn translate_openai_tools_to_anthropic(
    tools: serde_json::Value,
) -> Option<Vec<serde_json::Value>> {
    let arr = tools.as_array()?;
    let translated: Vec<serde_json::Value> = arr
        .iter()
        .filter_map(|t| {
            // OpenAI: `{type: "function", function: {name, description,
            // parameters}}`. Skip entries that don't fit this shape
            // (defensive — non-function tools have no Anthropic mapping).
            if t.get("type").and_then(|v| v.as_str()) != Some("function") {
                return None;
            }
            let function = t.get("function")?.as_object()?;
            let name = function.get("name")?.as_str()?;
            let mut anthropic_tool = serde_json::Map::new();
            anthropic_tool.insert("name".into(), name.into());
            if let Some(desc) = function.get("description") {
                anthropic_tool.insert("description".into(), desc.clone());
            }
            // OpenAI's `parameters` (JSON Schema) maps to Anthropic's
            // `input_schema` verbatim — both are JSON Schema.
            if let Some(params) = function.get("parameters") {
                anthropic_tool.insert("input_schema".into(), params.clone());
            }
            Some(serde_json::Value::Object(anthropic_tool))
        })
        .collect();
    if translated.is_empty() {
        None
    } else {
        Some(translated)
    }
}

/// Translate the caller's OpenAI-shape `tool_choice` to Anthropic's.
///
///   OpenAI                              Anthropic
///   "auto"                          →   {"type":"auto"}
///   "none"                          →   {"type":"none"}
///   "required"                      →   {"type":"any"}    (Anthropic's name for "must call something")
///   {type:"function",                   {"type":"tool",
///    function:{name:"X"}}           →    "name":"X"}
///
/// Returns None for unrecognised shapes — caller's value is discarded
/// rather than forwarded verbatim, since the OpenAI shape would 400
/// the Anthropic upstream.
pub fn translate_openai_tool_choice_to_anthropic(
    v: serde_json::Value,
) -> Option<serde_json::Value> {
    match v {
        serde_json::Value::String(s) => match s.as_str() {
            "auto" | "none" => Some(serde_json::json!({"type": s})),
            "required" => Some(serde_json::json!({"type": "any"})),
            _ => None,
        },
        serde_json::Value::Object(o) => {
            if o.get("type").and_then(|t| t.as_str()) != Some("function") {
                return None;
            }
            let name = o
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())?;
            Some(serde_json::json!({"type": "tool", "name": name}))
        }
        _ => None,
    }
}

/// Translate Anthropic-shape `tools` array into OpenAI's tools-spec shape.
///
///   Anthropic                                 OpenAI
///   {name,                                    {type: "function",
///    description,                              function: {name, description,
///    input_schema}                                        parameters}}
///
/// Returns `None` when the input isn't an array or when no entries
/// translated — keeping the field absent from the outbound request.
pub fn translate_anthropic_tools_to_openai(tools: serde_json::Value) -> Option<serde_json::Value> {
    let arr = tools.as_array()?;
    let translated: Vec<serde_json::Value> = arr
        .iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?;
            let mut function = serde_json::Map::new();
            function.insert("name".into(), name.into());
            if let Some(desc) = t.get("description") {
                function.insert("description".into(), desc.clone());
            }
            if let Some(schema) = t.get("input_schema") {
                function.insert("parameters".into(), schema.clone());
            }
            Some(serde_json::json!({
                "type": "function",
                "function": serde_json::Value::Object(function),
            }))
        })
        .collect();
    if translated.is_empty() {
        None
    } else {
        Some(serde_json::Value::Array(translated))
    }
}

/// Translate Anthropic-shape `tool_choice` to OpenAI's.
///
///   Anthropic                              OpenAI
///   {"type":"auto"}                    →   "auto"
///   {"type":"none"}                    →   "none"  (Anthropic doesn't officially
///                                          document this but clients may send it)
///   {"type":"any"}                     →   "required"
///   {"type":"tool", "name":"X"}        →   {type:"function", function:{name:"X"}}
///
/// Returns `None` for unrecognised shapes.
pub fn translate_anthropic_tool_choice_to_openai(
    v: serde_json::Value,
) -> Option<serde_json::Value> {
    let obj = v.as_object()?;
    let typ = obj.get("type").and_then(|t| t.as_str())?;
    match typ {
        "auto" | "none" => Some(serde_json::Value::String(typ.to_string())),
        "any" => Some(serde_json::Value::String("required".to_string())),
        "tool" => {
            let name = obj.get("name").and_then(|n| n.as_str())?;
            Some(serde_json::json!({
                "type": "function",
                "function": {"name": name}
            }))
        }
        _ => None,
    }
}

/// Rewrite `extra` (as filled by [`parse_inbound_request`], i.e. raw
/// Anthropic `/v1/messages` top-level fields) into the OpenAI chat shape
/// that non-Anthropic bridges expect. Whitelist-translate what maps
/// cleanly; drop everything else — Anthropic-only fields flattened onto
/// an OpenAI-compatible upstream request are rejected as unknown
/// parameters, e.g. 400 "Unknown parameter: 'context_management'"
/// (AISIX-Cloud#953). Mirrors the `/v1/responses` bridge's
/// whitelist-and-drop policy (#825) in the opposite direction.
///
/// Translations (matching LiteLLM's Anthropic→OpenAI adapter):
///   tools / tool_choice   → OpenAI shapes (existing helpers)
///   stop_sequences        → stop
///   metadata.user_id      → user
///   thinking              → reasoning_effort
pub fn translate_extras_to_openai_shape(extra: &mut serde_json::Map<String, serde_json::Value>) {
    let anthropic = std::mem::take(extra);
    for (key, value) in anthropic {
        match key.as_str() {
            "tools" => {
                if let Some(translated) = translate_anthropic_tools_to_openai(value) {
                    extra.insert("tools".to_string(), translated);
                }
            }
            "tool_choice" => {
                if let Some(translated) = translate_anthropic_tool_choice_to_openai(value) {
                    extra.insert("tool_choice".to_string(), translated);
                }
            }
            "stop_sequences" => {
                extra.insert("stop".to_string(), value);
            }
            "metadata" => {
                if let Some(user_id) = value.get("user_id").and_then(|v| v.as_str()) {
                    extra.insert("user".to_string(), user_id.into());
                }
            }
            "thinking" => {
                if let Some(effort) = reasoning_effort_from_thinking(&value) {
                    extra.insert("reasoning_effort".to_string(), effort.into());
                }
            }
            _ => {
                tracing::debug!(
                    field = %key,
                    "dropping Anthropic-only request field on cross-provider dispatch"
                );
            }
        }
    }
}

/// Bucket Anthropic `thinking` into an OpenAI `reasoning_effort` label.
/// Thresholds match LiteLLM's `reasoning_effort_from_thinking_budget`
/// (budget_tokens ≥ 4096 → high, ≥ 2048 → medium, ≥ 1024 → low, below →
/// minimal; `adaptive` → medium; `disabled`/unrecognised → None).
fn reasoning_effort_from_thinking(thinking: &serde_json::Value) -> Option<&'static str> {
    match thinking.get("type").and_then(|t| t.as_str())? {
        "enabled" => {
            let budget = thinking
                .get("budget_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            Some(match budget {
                b if b >= 4096 => "high",
                b if b >= 2048 => "medium",
                b if b >= 1024 => "low",
                _ => "minimal",
            })
        }
        "adaptive" => Some("medium"),
        _ => None,
    }
}

/// Non-streaming response shape from `/v1/messages`.
#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub model: String,
    #[serde(default)]
    pub content: Vec<AnthropicResponseBlock>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    /// Anthropic's `tool_use` content block. The model is asking to
    /// invoke a tool: `id` is the call id, `name` is the tool name,
    /// and `input` is a JSON object with the tool's arguments. Per
    /// docs §6 outbound-axis table ("tool_use ↔ tool_calls"), the
    /// gateway translates this into OpenAI's `tool_calls` shape on
    /// the response so OpenAI-SDK callers (and every agent framework
    /// built on that shape) work transparently against Anthropic
    /// upstreams.
    /// <https://docs.anthropic.com/en/api/messages#example-of-tool-use>
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    /// Future content-block types (e.g. `image` on output, `thinking`
    /// for reasoning models). Not surfaced today; accepted so unknown
    /// block types don't fail the whole response parse.
    #[serde(other)]
    Other,
}

// `#[serde(default)]` at the container level so a response missing any
// token counter deserializes to 0 rather than failing the whole body
// decode (same bug class as OpenAI usage, #474).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens written to the prompt cache (1.25× input rate). Optional
    /// — present only on requests with cache_control segments.
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Tokens served from the prompt cache (0.10× input rate).
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

pub fn response_into_chat_response(raw: AnthropicResponse) -> ChatResponse {
    let mut saw_text_block = false;
    let text = raw
        .content
        .iter()
        .filter_map(|b| match b {
            AnthropicResponseBlock::Text { text } => {
                saw_text_block = true;
                Some(text.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    // #395: when an Anthropic upstream returns only `tool_use` blocks
    // (no text block at all), surface `content: null` on the OpenAI
    // shape rather than `""` — same wire-shape parity fix as the OpenAI
    // passthrough. An explicit empty text block (`text: ""`) is distinct
    // and preserved as `Some("")`.
    let content = saw_text_block.then_some(text);

    // Translate Anthropic `tool_use` content blocks into OpenAI's
    // `message.tool_calls` shape so OpenAI-SDK callers see a
    // standard tool-call response. Field mapping per
    // <https://docs.anthropic.com/en/api/messages> and
    // <https://platform.openai.com/docs/api-reference/chat/object#chat-create-tool_calls>:
    //
    //   Anthropic                  OpenAI
    //   id          (string)   →   tool_calls[].id
    //   name        (string)   →   tool_calls[].function.name
    //   input       (object)   →   tool_calls[].function.arguments  (JSON-encoded string)
    //   (implicit)             →   tool_calls[].type: "function"
    //
    // `arguments` MUST be a JSON-encoded STRING in OpenAI's shape
    // (not the parsed object) so SDK consumers round-trip via
    // `JSON.parse(toolCall.function.arguments)`.
    let tool_calls: Vec<serde_json::Value> = raw
        .content
        .iter()
        .filter_map(|b| match b {
            AnthropicResponseBlock::ToolUse { id, name, input } => Some(serde_json::json!({
                "id": id,
                "type": "function",
                "function": {
                    "name": name,
                    // OpenAI emits `"{}"` (empty object) for no-args
                    // tool calls, not `"null"`. Normalise here so SDK
                    // consumers doing `JSON.parse(args)` get an
                    // object back even when Anthropic's `input`
                    // field is absent / null.
                    "arguments": match input {
                        serde_json::Value::Null => "{}".to_string(),
                        other => serde_json::to_string(other)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                },
            })),
            _ => None,
        })
        .collect();
    let mut extra = serde_json::Map::new();
    if !tool_calls.is_empty() {
        extra.insert(
            "tool_calls".to_string(),
            serde_json::Value::Array(tool_calls),
        );
    }

    let usage = raw
        .usage
        .map(|u| {
            // Anthropic bills cache_creation / cache_read as input classes
            // *on top of* input_tokens, so `total_tokens` must fold them
            // in — `input + output` alone under-counts (#906). Cache
            // counters stay separate; Anthropic doesn't use OpenAI's
            // cached-prompt / reasoning taxonomy (those default to 0).
            UsageStats::with_cache(
                u.input_tokens,
                u.output_tokens,
                u.cache_creation_input_tokens,
                u.cache_read_input_tokens,
            )
        })
        .unwrap_or_default();

    ChatResponse {
        id: raw.id,
        model: raw.model,
        message: ChatMessage {
            role: Role::Assistant,
            content,
            content_blocks: None,
            name: None,
            tool_call_id: None,
            extra,
        },
        finish_reason: map_stop_reason(raw.stop_reason.as_deref()),
        usage,
    }
}

fn map_stop_reason(raw: Option<&str>) -> FinishReason {
    match raw {
        Some("end_turn") | Some("stop_sequence") | None => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        Some(other) => FinishReason::Other(other.to_string()),
    }
}

/// Streaming events from Anthropic. Only variants that can yield user-
/// visible output or terminate the stream are modeled here; the rest are
/// quietly dropped by the Bridge.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart {
        message: AnthropicStreamStartMessage,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: AnthropicStreamDelta },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicStreamMessageDelta,
        #[serde(default)]
        usage: Option<AnthropicStreamUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    /// Catch-all for content_block_start / content_block_stop / ping /
    /// unknown event types — we don't need their state for chunk emission.
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamStartMessage {
    pub id: String,
    pub model: String,
    /// `message_start` carries the prompt token count in `usage.input_tokens`.
    /// Anthropic only sends it on this first event, so we must capture it here
    /// or prompt tokens are lost for the whole stream (TPM/budget/telemetry).
    #[serde(default)]
    pub usage: Option<AnthropicStreamStartUsage>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamStartUsage {
    #[serde(default)]
    pub input_tokens: Option<u32>,
    /// Cache write / read counters ride on `message_start` alongside
    /// `input_tokens` and are sent only there — capture them here or
    /// they're lost for the whole stream on the cross-protocol bridge
    /// path (#906).
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamMessageDelta {
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamUsage {
    #[serde(default)]
    pub output_tokens: Option<u32>,
}

/// Rolling state the Bridge carries across a stream so chunks can be
/// tagged with the message id/model even though only the first event
/// carries them.
#[derive(Debug, Default)]
pub struct StreamState {
    pub id: String,
    pub model: String,
    /// Prompt tokens captured from `message_start`; folded into the usage
    /// emitted on the terminal `message_delta` so the final `UsageStats`
    /// carries both prompt and completion (and a correct total).
    pub input_tokens: u32,
    /// Cache write / read counters captured from `message_start`, carried
    /// onto the terminal usage so the bridge doesn't drop them (#906).
    pub cache_creation_tokens: u32,
    pub cache_read_tokens: u32,
}

impl StreamState {
    pub fn update(&mut self, event: &AnthropicStreamEvent) {
        if let AnthropicStreamEvent::MessageStart { message } = event {
            self.id = message.id.clone();
            self.model = message.model.clone();
            // Reset on every message_start so a later message_start without
            // usage can't leave a stale prompt-token count from a prior one.
            self.input_tokens = message
                .usage
                .as_ref()
                .and_then(|u| u.input_tokens)
                .unwrap_or(0);
            self.cache_creation_tokens = message
                .usage
                .as_ref()
                .and_then(|u| u.cache_creation_input_tokens)
                .unwrap_or(0);
            self.cache_read_tokens = message
                .usage
                .as_ref()
                .and_then(|u| u.cache_read_input_tokens)
                .unwrap_or(0);
        }
    }

    /// Translate one event into an optional chunk to yield upstream.
    pub fn to_chunk(&self, event: &AnthropicStreamEvent) -> Option<ChatChunk> {
        match event {
            AnthropicStreamEvent::ContentBlockDelta {
                delta: AnthropicStreamDelta::TextDelta { text },
            } => Some(ChatChunk {
                id: self.id.clone(),
                model: self.model.clone(),
                delta: ChatDelta {
                    role: None,
                    content: Some(text.clone()),
                    tool_calls: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                usage: None,
            }),
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                let finish = delta
                    .stop_reason
                    .as_deref()
                    .map(|r| map_stop_reason(Some(r)));
                let usage = usage.as_ref().and_then(|u| {
                    u.output_tokens.map(|n| {
                        UsageStats::with_cache(
                            self.input_tokens,
                            n,
                            self.cache_creation_tokens,
                            self.cache_read_tokens,
                        )
                    })
                });
                if finish.is_none() && usage.is_none() {
                    return None;
                }
                Some(ChatChunk {
                    id: self.id.clone(),
                    model: self.model.clone(),
                    delta: ChatDelta::default(),
                    finish_reason: finish,
                    usage,
                })
            }
            _ => None,
        }
    }

    pub fn is_terminal(event: &AnthropicStreamEvent) -> bool {
        matches!(event, AnthropicStreamEvent::MessageStop)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Inbound translation — Anthropic protocol  →  internal ChatFormat.
//
// Used by the proxy's /v1/messages handler when the Model targeted by
// the request points at a non-Anthropic upstream: we accept the
// Anthropic-shaped body, translate to ChatFormat, and dispatch through
// the Hub. The reverse direction (ChatFormat → Anthropic wire request)
// is handled by `split_system` + `build_request` for the
// Anthropic-upstream case above.
//
// Trimmed to the MVP fields aisix supports today (text content blocks;
// tool_use, image, and thinking blocks land in a follow-up PR).

#[derive(Debug, thiserror::Error)]
pub enum AnthropicInboundError {
    #[error("body is not a JSON object")]
    NotAnObject,
    #[error("missing or non-string `model` field")]
    MissingModel,
    #[error("missing or non-array `messages` field")]
    MissingMessages,
    #[error("messages[{idx}] missing `role`")]
    MessageMissingRole { idx: usize },
    #[error("messages[{idx}] role {role:?} is not 'user', 'assistant' or 'system'")]
    UnsupportedRole { idx: usize, role: String },
    #[error("messages[{idx}].content must be a string or an array of text blocks")]
    UnsupportedContent { idx: usize },
    #[error("`system` field must be a string or an array of text blocks")]
    UnsupportedSystem,
}

/// Parse an Anthropic `POST /v1/messages` JSON body into the gateway's
/// internal [`ChatFormat`]. The `system` field is folded into a leading
/// system message; content blocks are concatenated text-only (non-text
/// blocks are skipped). Unrecognized top-level keys (`metadata`,
/// `tools`, `tool_choice`, etc.) flow into `ChatFormat::extra` so a
/// future tools-aware bridge can read them.
pub fn parse_inbound_request(
    body: &serde_json::Value,
) -> Result<ChatFormat, AnthropicInboundError> {
    use serde_json::Value;
    let obj = body.as_object().ok_or(AnthropicInboundError::NotAnObject)?;

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or(AnthropicInboundError::MissingModel)?
        .to_string();

    let raw_messages = obj
        .get("messages")
        .and_then(Value::as_array)
        .ok_or(AnthropicInboundError::MissingMessages)?;

    let mut messages: Vec<ChatMessage> = Vec::with_capacity(raw_messages.len() + 1);

    // `system`: prepend as leading system message. Anthropic accepts
    // string OR array of text blocks; we accept both shapes.
    if let Some(system) = obj.get("system") {
        let system_text = match system {
            Value::String(s) => s.clone(),
            Value::Array(blocks) => {
                let mut parts = Vec::new();
                for block in blocks {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        parts.push(text);
                    }
                }
                parts.join("\n")
            }
            Value::Null => String::new(),
            _ => return Err(AnthropicInboundError::UnsupportedSystem),
        };
        if !system_text.is_empty() {
            messages.push(ChatMessage::system(system_text));
        }
    }

    for (idx, m) in raw_messages.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or(AnthropicInboundError::MessageMissingRole { idx })?;

        let content = match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(blocks)) => {
                let mut parts = Vec::new();
                for block in blocks {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        parts.push(text);
                    }
                }
                parts.join("")
            }
            _ => return Err(AnthropicInboundError::UnsupportedContent { idx }),
        };

        match role {
            "user" => messages.push(ChatMessage::user(content)),
            "assistant" => messages.push(ChatMessage::assistant(content)),
            // Not in the Anthropic spec, but Claude Code/cc-switch send it
            // (#597). Keep it as a system message so OpenAI-compatible
            // upstreams receive it natively instead of a 400 here.
            "system" => messages.push(ChatMessage::system(content)),
            other => {
                return Err(AnthropicInboundError::UnsupportedRole {
                    idx,
                    role: other.to_string(),
                })
            }
        }
    }

    let mut chat = ChatFormat::new(model, messages);

    if let Some(t) = obj.get("temperature").and_then(Value::as_f64) {
        chat.temperature = Some(t as f32);
    }
    if let Some(t) = obj.get("top_p").and_then(Value::as_f64) {
        chat.top_p = Some(t as f32);
    }
    if let Some(t) = obj.get("max_tokens").and_then(Value::as_u64) {
        chat.max_tokens = Some(t as u32);
    }
    if let Some(s) = obj.get("stream").and_then(Value::as_bool) {
        chat.stream = Some(s);
    }

    // Pass remaining keys through `extra` so future bridges can use
    // them. We deliberately don't whitelist — bridges that don't
    // understand a key just ignore it.
    for (key, value) in obj {
        if !matches!(
            key.as_str(),
            "model" | "messages" | "system" | "temperature" | "top_p" | "max_tokens" | "stream"
        ) {
            chat.extra.insert(key.clone(), value.clone());
        }
    }

    Ok(chat)
}

// ─────────────────────────────────────────────────────────────────────
// Outbound translation — internal ChatResponse  →  Anthropic JSON.

/// Render an internal [`ChatResponse`] as the JSON an Anthropic
/// `/v1/messages` client expects. The reverse of
/// `response_into_chat_response`. `model_display_name` is the
/// operator-facing model name the client requested — we echo it back
/// rather than leaking the actual upstream id (e.g. `gpt-4o`) when
/// the underlying provider isn't Anthropic.
pub fn chat_response_into_anthropic_json(
    resp: &ChatResponse,
    model_display_name: &str,
) -> serde_json::Value {
    let stop_reason = match &resp.finish_reason {
        FinishReason::Stop => "end_turn",
        FinishReason::Length => "max_tokens",
        FinishReason::ContentFilter => "stop_sequence",
        FinishReason::ToolCalls => "tool_use",
        FinishReason::Other(_) => "end_turn",
    };

    let mut content: Vec<serde_json::Value> = Vec::new();

    if let Some(text) = resp.message.content.as_deref().filter(|s| !s.is_empty()) {
        content.push(serde_json::json!({"type": "text", "text": text}));
    }

    // Translate OpenAI-shape tool_calls from message.extra into
    // Anthropic tool_use content blocks so Anthropic clients see
    // the tool invocations the model requested.
    if let Some(tool_calls) = resp
        .message
        .extra
        .get("tool_calls")
        .and_then(|v| v.as_array())
    {
        content.extend(tool_use_blocks_from_openai(tool_calls));
    }

    if content.is_empty() {
        content.push(serde_json::json!({"type": "text", "text": ""}));
    }

    serde_json::json!({
        "id": resp.id,
        "type": "message",
        "role": "assistant",
        "model": model_display_name,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": serde_json::Value::Null,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
    })
}

// ─────────────────────────────────────────────────────────────────────
// Streaming SSE encoder — internal ChatChunk stream  →  Anthropic
// SSE events.
//
// State machine:
//   1. First chunk that carries content or a finish_reason → emit
//      `message_start`. If it carries content, also emit
//      `content_block_start` + `content_block_delta`.
//   2. Mid-stream chunks with content → `content_block_delta`.
//   3. Chunk carrying `finish_reason` → emit `content_block_stop`
//      (only if a content block was opened), `message_delta` (with
//      stop_reason + final usage), then `message_stop`. After
//      `finished` flips true the encoder is silent.
//
// Reference: https://docs.anthropic.com/en/api/streaming

/// One Anthropic SSE event, ready to be written to the wire as
/// `event: {event}\ndata: {data}\n\n`.
#[derive(Debug, Clone)]
pub struct AnthropicSseEvent {
    pub event: &'static str,
    pub data: serde_json::Value,
}

impl AnthropicSseEvent {
    pub fn to_sse_string(&self) -> String {
        format!(
            "event: {}\ndata: {}\n\n",
            self.event,
            serde_json::to_string(&self.data).expect("serde_json::Value always serializes"),
        )
    }
}

/// Per-tool-call accumulator used by the SSE encoder to track which
/// tool_use blocks have been started and at which content-block index.
#[derive(Debug)]
struct ToolCallState {
    id: String,
    name: String,
    content_block_index: usize,
    started: bool,
}

/// State machine for re-encoding a stream of internal `ChatChunk`s as
/// Anthropic SSE events.
#[derive(Debug)]
pub struct AnthropicSseEncoder {
    message_id: String,
    model_display_name: String,
    initial_input_tokens: u32,
    sent_message_start: bool,
    /// Index assigned to the text content block (if any).
    text_block_index: Option<usize>,
    finished: bool,
    /// Next content-block index to assign (shared across text + tool_use blocks).
    next_block_index: usize,
    /// Per-OpenAI-delta-index tool call state.
    tool_calls: std::collections::BTreeMap<u64, ToolCallState>,
    /// Stop reason captured at the `finish_reason` chunk while the
    /// closing `message_delta`/`message_stop` pair is withheld. With
    /// `stream_options.include_usage` (AISIX-Cloud#790) an OpenAI
    /// upstream sends its only `usage` frame AFTER the stop chunk;
    /// emitting the pair at the stop chunk would hand the client
    /// `output_tokens: 0` and drop the usage frame unread.
    pending_stop_reason: Option<&'static str>,
    /// Best-known cumulative usage across all chunks. Max semantics —
    /// robust to providers that double-emit usage.
    seen_input_tokens: u32,
    seen_output_tokens: u32,
    usage_seen: bool,
}

impl AnthropicSseEncoder {
    /// `message_id` is echoed in `message_start.message.id`.
    /// `model_display_name` is the operator-facing model name the
    /// client originally sent in `req.model`.
    /// `initial_input_tokens` is the best-known-at-stream-open input
    /// token count; pass 0 if unknown.
    pub fn new(
        message_id: impl Into<String>,
        model_display_name: impl Into<String>,
        initial_input_tokens: u32,
    ) -> Self {
        Self {
            message_id: message_id.into(),
            model_display_name: model_display_name.into(),
            initial_input_tokens,
            sent_message_start: false,
            text_block_index: None,
            finished: false,
            next_block_index: 0,
            tool_calls: std::collections::BTreeMap::new(),
            pending_stop_reason: None,
            seen_input_tokens: 0,
            seen_output_tokens: 0,
            usage_seen: false,
        }
    }

    /// Translate one chunk into the Anthropic SSE events to emit.
    /// Returns an empty Vec on no-op chunks.
    pub fn next_events(&mut self, chunk: &ChatChunk) -> Vec<AnthropicSseEvent> {
        if self.finished {
            return Vec::new();
        }

        if let Some(u) = chunk.usage.as_ref() {
            self.usage_seen = true;
            self.seen_input_tokens = self.seen_input_tokens.max(u.prompt_tokens);
            self.seen_output_tokens = self.seen_output_tokens.max(u.completion_tokens);
        }

        // Closing pair withheld at the stop chunk: only the trailing
        // usage frame releases it (stream end does too, via
        // `force_finish`). Post-stop chunks carry no renderable
        // content, so nothing else is emitted from here.
        if let Some(reason) = self.pending_stop_reason {
            if self.usage_seen {
                self.pending_stop_reason = None;
                return self.closing_pair(reason);
            }
            return Vec::new();
        }

        let mut events = Vec::new();

        let has_content = chunk
            .delta
            .content
            .as_deref()
            .is_some_and(|s| !s.is_empty());
        let has_tool_calls = chunk
            .delta
            .tool_calls
            .as_ref()
            .is_some_and(|v| !v.is_empty());
        let has_finish = chunk.finish_reason.is_some();

        if !self.sent_message_start && (has_content || has_tool_calls || has_finish) {
            events.push(self.message_start_event());
            self.sent_message_start = true;
        }

        // ── Text content block ──
        if self.text_block_index.is_none() && has_content {
            let idx = self.next_block_index;
            self.next_block_index += 1;
            self.text_block_index = Some(idx);
            events.push(content_block_start_event(idx));
        }

        if has_content {
            let idx = self.text_block_index.unwrap_or(0);
            events.push(AnthropicSseEvent {
                event: "content_block_delta",
                data: serde_json::json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "text_delta",
                        "text": chunk.delta.content.clone().unwrap_or_default(),
                    },
                }),
            });
        }

        // ── Tool-use content blocks ──
        if let Some(tool_calls) = &chunk.delta.tool_calls {
            for tc in tool_calls {
                let oai_index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);

                let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("");
                let arguments = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or("");

                let state = self.tool_calls.entry(oai_index).or_insert_with(|| {
                    let block_idx = self.next_block_index;
                    self.next_block_index += 1;
                    ToolCallState {
                        id: String::new(),
                        name: String::new(),
                        content_block_index: block_idx,
                        started: false,
                    }
                });

                if !id.is_empty() {
                    state.id = id.to_string();
                }
                if !name.is_empty() {
                    state.name = name.to_string();
                }

                // Emit content_block_start once id and name are known.
                if !state.started && !state.id.is_empty() && !state.name.is_empty() {
                    state.started = true;
                    events.push(AnthropicSseEvent {
                        event: "content_block_start",
                        data: serde_json::json!({
                            "type": "content_block_start",
                            "index": state.content_block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": state.id,
                                "name": state.name,
                                "input": {},
                            },
                        }),
                    });
                }

                if state.started && !arguments.is_empty() {
                    events.push(AnthropicSseEvent {
                        event: "content_block_delta",
                        data: serde_json::json!({
                            "type": "content_block_delta",
                            "index": state.content_block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            },
                        }),
                    });
                }
            }
        }

        // ── Finish ──
        if let Some(fr) = &chunk.finish_reason {
            // Close text block if open.
            if let Some(text_idx) = self.text_block_index {
                events.push(content_block_stop_event(text_idx));
            }
            // Close all open tool_use blocks.
            for state in self.tool_calls.values() {
                if state.started {
                    events.push(content_block_stop_event(state.content_block_index));
                }
            }

            let stop_reason = match fr {
                FinishReason::Stop => "end_turn",
                FinishReason::Length => "max_tokens",
                FinishReason::ContentFilter => "stop_sequence",
                FinishReason::ToolCalls => "tool_use",
                FinishReason::Other(_) => "end_turn",
            };
            if self.usage_seen {
                // Usage already known (provider attached it to the stop
                // chunk or earlier) — close out immediately.
                events.extend(self.closing_pair(stop_reason));
            } else {
                // OpenAI's `stream_options.include_usage` frame arrives
                // AFTER the stop chunk — withhold the closing pair so
                // it can carry real token counts.
                self.pending_stop_reason = Some(stop_reason);
            }
        }

        events
    }

    /// The closing `message_delta` + `message_stop` pair, carrying the
    /// best-known cumulative usage. `input_tokens` is included when
    /// known — on a translated stream `message_start` fires before any
    /// usage frame exists and always reports 0, so this is the only
    /// place the client can learn the prompt token count.
    fn closing_pair(&mut self, stop_reason: &'static str) -> Vec<AnthropicSseEvent> {
        let mut usage = serde_json::Map::new();
        if self.seen_input_tokens > 0 {
            usage.insert("input_tokens".into(), self.seen_input_tokens.into());
        }
        usage.insert("output_tokens".into(), self.seen_output_tokens.into());
        self.finished = true;
        vec![
            AnthropicSseEvent {
                event: "message_delta",
                data: serde_json::json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": serde_json::Value::Null,
                    },
                    "usage": serde_json::Value::Object(usage),
                }),
            },
            AnthropicSseEvent {
                event: "message_stop",
                data: serde_json::json!({"type": "message_stop"}),
            },
        ]
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Force-close the stream when the upstream ended without
    /// releasing the closing pair — either no `finish_reason` chunk at
    /// all (closes with `end_turn`), or a stop chunk arrived but the
    /// trailing usage frame never did (flushes the withheld pair with
    /// the real stop reason). Usage carries the best-known counts.
    /// Idempotent.
    pub fn force_finish(&mut self) -> Vec<AnthropicSseEvent> {
        if self.finished {
            return Vec::new();
        }
        // A withheld closing pair (stop seen, but the upstream ignored
        // `stream_options` and never sent a usage frame): flush it with
        // the real stop reason. Content blocks were already closed at
        // the stop chunk.
        if let Some(reason) = self.pending_stop_reason.take() {
            return self.closing_pair(reason);
        }
        let mut events = Vec::new();
        if !self.sent_message_start {
            events.push(self.message_start_event());
            self.sent_message_start = true;
        }
        if let Some(text_idx) = self.text_block_index {
            events.push(content_block_stop_event(text_idx));
        }
        for state in self.tool_calls.values() {
            if state.started {
                events.push(content_block_stop_event(state.content_block_index));
            }
        }
        events.extend(self.closing_pair("end_turn"));
        events
    }

    fn message_start_event(&self) -> AnthropicSseEvent {
        AnthropicSseEvent {
            event: "message_start",
            data: serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model_display_name,
                    "stop_reason": serde_json::Value::Null,
                    "stop_sequence": serde_json::Value::Null,
                    "usage": {
                        "input_tokens": self.initial_input_tokens,
                        "output_tokens": 0,
                    },
                },
            }),
        }
    }
}

fn content_block_start_event(index: usize) -> AnthropicSseEvent {
    AnthropicSseEvent {
        event: "content_block_start",
        data: serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        }),
    }
}

fn content_block_stop_event(index: usize) -> AnthropicSseEvent {
    AnthropicSseEvent {
        event: "content_block_stop",
        data: serde_json::json!({"type": "content_block_stop", "index": index}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_system_merges_leading_system_messages() {
        let req = ChatFormat::new(
            "claude",
            vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::system("respond concisely"),
                ChatMessage::user("hi"),
            ],
        );
        let (system, msgs) = split_system(&req).unwrap();
        assert_eq!(
            system.as_deref(),
            Some("you are helpful\n\nrespond concisely")
        );
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
    }

    #[test]
    fn split_system_mid_conversation_becomes_user_turn() {
        let req = ChatFormat::new(
            "claude",
            vec![
                ChatMessage::user("hi"),
                ChatMessage::system("forget everything"),
                ChatMessage::assistant("ok"),
            ],
        );
        let (system, msgs) = split_system(&req).unwrap();
        assert!(system.is_none());
        // The interleaved system message becomes a user turn and folds into
        // the adjacent user turn (alternating-role invariant): one user
        // message carrying both text blocks, then the assistant turn.
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content.len(), 2);
        assert_eq!(msgs[0].content[0]["text"], "hi");
        assert_eq!(msgs[0].content[1]["text"], "forget everything");
        assert_eq!(msgs[1].role, "assistant");
    }

    #[test]
    fn split_system_rejects_tool_role_without_tool_call_id() {
        // Tool turn must carry a tool_call_id (the OpenAI shape
        // pairs tool_calls[i].id with the next turn's tool_call_id).
        // Without one, we can't construct Anthropic's tool_result
        // block — error rather than silently dropping the turn.
        let req = ChatFormat::new(
            "claude",
            vec![ChatMessage {
                role: Role::Tool,
                content: Some("x".into()),
                content_blocks: None,
                name: None,
                tool_call_id: None,
                extra: serde_json::Map::new(),
            }],
        );
        assert!(matches!(
            split_system(&req),
            Err(TranslateError::MissingToolCallId)
        ));
    }

    #[test]
    fn split_system_translates_tool_role_to_anthropic_tool_result() {
        // Agent-loop turn 2: caller sends back the tool's output via
        // {role:"tool", tool_call_id, content}; gateway must
        // translate to Anthropic's
        // {role:"user", content:[{type:"tool_result", tool_use_id, content}]}.
        let req = ChatFormat::new(
            "claude",
            vec![
                ChatMessage::user("What's the weather in SF?"),
                tool_call_assistant("toolu_abc", "get_weather", "{\"city\":\"SF\"}"),
                ChatMessage {
                    role: Role::Tool,
                    content: Some("72F, sunny".into()),
                    content_blocks: None,
                    name: None,
                    tool_call_id: Some("toolu_abc".into()),
                    extra: serde_json::Map::new(),
                },
            ],
        );
        let (_system, msgs) = split_system(&req).unwrap();
        assert_eq!(msgs.len(), 3);
        // Tool turn became a user turn with a tool_result block.
        assert_eq!(msgs[2].role, "user");
        assert_eq!(msgs[2].content.len(), 1);
        assert_eq!(msgs[2].content[0]["type"], "tool_result");
        assert_eq!(msgs[2].content[0]["tool_use_id"], "toolu_abc");
        assert_eq!(msgs[2].content[0]["content"], "72F, sunny");
    }

    /// Build an assistant ChatMessage replaying a single tool call, the
    /// OpenAI history shape an agent loop sends back.
    fn tool_call_assistant(id: &str, name: &str, arguments: &str) -> ChatMessage {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "tool_calls".into(),
            serde_json::json!([{
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }]),
        );
        ChatMessage {
            role: Role::Assistant,
            content: None,
            content_blocks: None,
            name: None,
            tool_call_id: None,
            extra,
        }
    }

    #[test]
    fn split_system_translates_assistant_tool_calls_to_tool_use() {
        // Agent-loop turn 2: the caller replays the assistant's prior
        // tool call as OpenAI-shape `tool_calls` in message.extra. Without
        // translation the tool_use is dropped and the following
        // tool_result orphans → Anthropic 400.
        let req = ChatFormat::new(
            "claude",
            vec![
                ChatMessage::user("weather in SF?"),
                tool_call_assistant("toolu_1", "get_weather", "{\"city\":\"SF\"}"),
                ChatMessage {
                    role: Role::Tool,
                    content: Some("72F".into()),
                    content_blocks: None,
                    name: None,
                    tool_call_id: Some("toolu_1".into()),
                    extra: serde_json::Map::new(),
                },
            ],
        );
        let (_system, msgs) = split_system(&req).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1].role, "assistant");
        let block = &msgs[1].content[0];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["id"], "toolu_1");
        assert_eq!(block["name"], "get_weather");
        assert_eq!(block["input"]["city"], "SF");
        // The tool result alternates back as a user turn.
        assert_eq!(msgs[2].role, "user");
        assert_eq!(msgs[2].content[0]["type"], "tool_result");
    }

    #[test]
    fn split_system_merges_parallel_tool_results_into_one_user_turn() {
        // Parallel tool calls produce two consecutive tool_result turns;
        // they must fold into a single user message so roles still
        // alternate (assistant → user) for the upstream.
        let mut assistant_extra = serde_json::Map::new();
        assistant_extra.insert(
            "tool_calls".into(),
            serde_json::json!([
                {"id": "t1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "t2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ]),
        );
        let req = ChatFormat::new(
            "claude",
            vec![
                ChatMessage::user("go"),
                ChatMessage {
                    role: Role::Assistant,
                    content: None,
                    content_blocks: None,
                    name: None,
                    tool_call_id: None,
                    extra: assistant_extra,
                },
                ChatMessage {
                    role: Role::Tool,
                    content: Some("r1".into()),
                    content_blocks: None,
                    name: None,
                    tool_call_id: Some("t1".into()),
                    extra: serde_json::Map::new(),
                },
                ChatMessage {
                    role: Role::Tool,
                    content: Some("r2".into()),
                    content_blocks: None,
                    name: None,
                    tool_call_id: Some("t2".into()),
                    extra: serde_json::Map::new(),
                },
            ],
        );
        let (_system, msgs) = split_system(&req).unwrap();
        // user, assistant(2 tool_use), user(2 tool_result)
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].content.len(), 2);
        assert_eq!(msgs[2].role, "user");
        assert_eq!(msgs[2].content.len(), 2);
        assert_eq!(msgs[2].content[0]["tool_use_id"], "t1");
        assert_eq!(msgs[2].content[1]["tool_use_id"], "t2");
    }

    #[test]
    fn build_request_applies_default_max_tokens_when_unset() {
        let req = ChatFormat::new("claude", vec![ChatMessage::user("hi")]);
        let (_system, messages) = split_system(&req).unwrap();
        let built = build_request(&req, "claude-sonnet-4-5", None, messages, false);
        assert_eq!(built.max_tokens, DEFAULT_MAX_TOKENS);

        let req = ChatFormat {
            max_tokens: Some(256),
            ..ChatFormat::new("claude", vec![ChatMessage::user("hi")])
        };
        let (_system, messages) = split_system(&req).unwrap();
        let built = build_request(&req, "claude-sonnet-4-5", None, messages, false);
        assert_eq!(built.max_tokens, 256);
    }

    #[test]
    fn tool_use_block_translates_to_openai_tool_calls_in_extra() {
        // Anthropic Messages response with a tool_use content block
        // (the model decided to call a tool) — verbatim shape from
        // <https://docs.anthropic.com/en/api/messages#example-of-tool-use>.
        let body = r#"{
            "id": "msg_tool_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "get_weather",
                    "input": {"location": "San Francisco, CA", "unit": "celsius"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 8}
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);

        // stop_reason "tool_use" → finish_reason ToolCalls.
        assert_eq!(out.finish_reason, FinishReason::ToolCalls);
        // #395: when only tool_use blocks are emitted (no text), the
        // OpenAI-shape content is `null`, not `""`.
        assert_eq!(out.message.content, None);

        // tool_calls translation lives in `message.extra` so the
        // proxy renderer flattens it onto the wire as a top-level
        // OpenAI-shape field.
        let tool_calls = out
            .message
            .extra
            .get("tool_calls")
            .expect("tool_calls populated in extra")
            .as_array()
            .expect("tool_calls is an array");
        assert_eq!(tool_calls.len(), 1);
        let tc = &tool_calls[0];
        assert_eq!(tc["id"], "toolu_abc");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "get_weather");
        // OpenAI's `arguments` is a JSON-encoded STRING, not the
        // parsed object — SDK consumers `JSON.parse` it.
        let args_str = tc["function"]["arguments"]
            .as_str()
            .expect("arguments is a string");
        let args: serde_json::Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "celsius");
    }

    #[test]
    fn mixed_text_and_tool_use_blocks_both_surface() {
        // The model can emit text BEFORE invoking a tool. Both must
        // reach the OpenAI-SDK caller: text → message.content,
        // tool_use → message.extra["tool_calls"].
        let body = r#"{
            "id": "msg_mixed_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {"type": "tool_use", "id": "toolu_x", "name": "get_weather",
                 "input": {"location": "NYC"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 10}
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.message.content_str(), "Let me check the weather.");
        assert!(out.message.extra.get("tool_calls").is_some());
    }

    #[test]
    fn explicit_empty_text_block_stays_empty_string_not_null() {
        // #395 refinement: distinguish "no text block at all" (→ null)
        // from an explicit empty text block (→ ""). A response carrying
        // a `{"type":"text","text":""}` block must surface `Some("")`,
        // not `None`.
        let body = r#"{
            "id": "msg_empty_text_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": ""}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 0}
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.message.content, Some(String::new()));
    }

    #[test]
    fn parallel_tool_use_blocks_emit_array_in_order() {
        // Anthropic supports parallel tool calls — multiple tool_use
        // blocks in one response. Each must produce a tool_calls
        // entry, in the same order as the upstream emitted them.
        let body = r#"{
            "id": "msg_parallel_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"location": "SF"}},
                {"type": "tool_use", "id": "toolu_2", "name": "get_time",
                 "input": {"timezone": "PST"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        let tool_calls = out
            .message
            .extra
            .get("tool_calls")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0]["id"], "toolu_1");
        assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
        assert_eq!(tool_calls[1]["id"], "toolu_2");
        assert_eq!(tool_calls[1]["function"]["name"], "get_time");
    }

    #[test]
    fn tool_use_with_no_input_emits_empty_object_arguments() {
        // OpenAI emits `arguments: "{}"` for no-args tool calls, not
        // `"null"`. SDK consumers do `JSON.parse(arguments)` — `null`
        // yields a non-object, breaking idiomatic agent code.
        let body = r#"{
            "id": "msg_no_args",
            "type": "message",
            "role": "assistant",
            "model": "c",
            "content": [
                {"type": "tool_use", "id": "tu", "name": "noop"}
            ],
            "stop_reason": "tool_use"
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        let tc = &out.message.extra["tool_calls"][0];
        assert_eq!(tc["function"]["arguments"], "{}");
    }

    #[test]
    fn tool_choice_string_forms_translate_to_anthropic_object_shape() {
        // OpenAI: "auto" | "none" | "required"
        // Anthropic: {"type":"auto"} | {"type":"none"} | {"type":"any"}
        assert_eq!(
            translate_openai_tool_choice_to_anthropic(serde_json::json!("auto")),
            Some(serde_json::json!({"type": "auto"})),
        );
        assert_eq!(
            translate_openai_tool_choice_to_anthropic(serde_json::json!("none")),
            Some(serde_json::json!({"type": "none"})),
        );
        // "required" → "any" (Anthropic's name for "must call something")
        assert_eq!(
            translate_openai_tool_choice_to_anthropic(serde_json::json!("required")),
            Some(serde_json::json!({"type": "any"})),
        );
    }

    #[test]
    fn tool_choice_specific_function_translates_to_anthropic_tool() {
        // OpenAI: {type:"function", function:{name:"X"}}
        // Anthropic: {type:"tool", name:"X"}
        let openai = serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather"}
        });
        assert_eq!(
            translate_openai_tool_choice_to_anthropic(openai),
            Some(serde_json::json!({"type": "tool", "name": "get_weather"})),
        );
    }

    #[test]
    fn tool_choice_unrecognised_shape_drops_to_none() {
        // Strip the field rather than forwarding an OpenAI shape
        // Anthropic doesn't recognise.
        assert_eq!(
            translate_openai_tool_choice_to_anthropic(serde_json::json!("invalid_form")),
            None,
        );
        assert_eq!(
            translate_openai_tool_choice_to_anthropic(serde_json::json!(42)),
            None,
        );
    }

    // ─── Anthropic → OpenAI tool translation (#236) ──────────────

    #[test]
    fn anthropic_tools_translate_to_openai_function_shape() {
        let anthropic = serde_json::json!([
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        ]);
        let result = translate_anthropic_tools_to_openai(anthropic).unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["type"], "function");
        assert_eq!(arr[0]["function"]["name"], "get_weather");
        assert_eq!(arr[0]["function"]["description"], "Get current weather");
        assert_eq!(arr[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn anthropic_tool_without_description_or_schema_still_translates() {
        let anthropic = serde_json::json!([{"name": "noop"}]);
        let result = translate_anthropic_tools_to_openai(anthropic).unwrap();
        let tool = &result.as_array().unwrap()[0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["function"]["name"], "noop");
        assert!(tool["function"].get("description").is_none());
        assert!(tool["function"].get("parameters").is_none());
    }

    #[test]
    fn anthropic_tools_non_array_returns_none() {
        assert!(translate_anthropic_tools_to_openai(serde_json::json!("not_array")).is_none());
    }

    #[test]
    fn anthropic_tools_empty_array_returns_none() {
        assert!(translate_anthropic_tools_to_openai(serde_json::json!([])).is_none());
    }

    #[test]
    fn anthropic_tools_entries_without_name_are_skipped() {
        let anthropic = serde_json::json!([
            {"description": "no name field"},
            {"name": "valid", "description": "ok"}
        ]);
        let result = translate_anthropic_tools_to_openai(anthropic).unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["function"]["name"], "valid");
    }

    #[test]
    fn anthropic_tool_choice_auto_translates() {
        assert_eq!(
            translate_anthropic_tool_choice_to_openai(serde_json::json!({"type": "auto"})),
            Some(serde_json::json!("auto")),
        );
    }

    #[test]
    fn anthropic_tool_choice_any_translates_to_required() {
        assert_eq!(
            translate_anthropic_tool_choice_to_openai(serde_json::json!({"type": "any"})),
            Some(serde_json::json!("required")),
        );
    }

    #[test]
    fn anthropic_tool_choice_none_translates() {
        assert_eq!(
            translate_anthropic_tool_choice_to_openai(serde_json::json!({"type": "none"})),
            Some(serde_json::json!("none")),
        );
    }

    #[test]
    fn anthropic_tool_choice_specific_tool_translates() {
        let anthropic = serde_json::json!({"type": "tool", "name": "get_weather"});
        assert_eq!(
            translate_anthropic_tool_choice_to_openai(anthropic),
            Some(serde_json::json!({"type": "function", "function": {"name": "get_weather"}})),
        );
    }

    #[test]
    fn anthropic_tool_choice_unrecognised_returns_none() {
        assert!(
            translate_anthropic_tool_choice_to_openai(serde_json::json!({"type": "unknown"}))
                .is_none()
        );
        assert!(translate_anthropic_tool_choice_to_openai(serde_json::json!("auto")).is_none());
        assert!(translate_anthropic_tool_choice_to_openai(serde_json::json!(42)).is_none());
    }

    #[test]
    fn build_request_strips_tool_choice_from_extra() {
        // Even when the value is unrecognised, tool_choice MUST NOT
        // leak into `extra` — forwarding the OpenAI shape would 400
        // the upstream.
        let req = ChatFormat {
            extra: {
                let mut m = serde_json::Map::new();
                m.insert("tool_choice".to_string(), serde_json::json!("auto"));
                m.insert("custom_field".to_string(), serde_json::json!("kept"));
                m
            },
            ..ChatFormat::new("c", vec![ChatMessage::user("hi")])
        };
        let (_system, messages) = split_system(&req).unwrap();
        let built = build_request(&req, "c-name", None, messages, false);
        // tool_choice translated and on the typed field.
        assert_eq!(built.tool_choice, Some(serde_json::json!({"type": "auto"})));
        // tool_choice removed from `extra`; other fields preserved.
        assert!(!built.extra.contains_key("tool_choice"));
        assert_eq!(
            built.extra.get("custom_field"),
            Some(&serde_json::json!("kept"))
        );
    }

    #[test]
    fn non_streaming_response_concatenates_text_blocks() {
        let body = r#"{
            "id": "msg_01A",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "text", "text": "hel"},
                {"type": "text", "text": "lo"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2}
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.id, "msg_01A");
        assert_eq!(out.message.content_str(), "hello");
        assert_eq!(out.finish_reason, FinishReason::Stop);
        assert_eq!(out.usage.total_tokens, 5);
    }

    #[test]
    fn cache_creation_and_read_counters_populate_when_present() {
        // Verified shape from
        // https://docs.anthropic.com/en/api/messages (usage object
        // with cache_creation_input_tokens + cache_read_input_tokens).
        let body = r#"{
            "id": "msg_cache_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 800
            }
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.usage.prompt_tokens, 10);
        assert_eq!(out.usage.completion_tokens, 4);
        assert_eq!(out.usage.cache_creation_tokens, 200);
        assert_eq!(out.usage.cache_read_tokens, 800);
        // #906: cache_creation / cache_read are input classes on top of
        // input_tokens, so the honest total folds them in — not just
        // input + output (which would be 14 and under-count by 1000).
        assert_eq!(out.usage.total_tokens, 10 + 4 + 200 + 800);
        // Anthropic doesn't use OpenAI's cached_prompt / reasoning
        // taxonomy — these stay 0.
        assert_eq!(out.usage.cached_prompt_tokens, 0);
        assert_eq!(out.usage.reasoning_tokens, 0);
    }

    #[test]
    fn stop_reason_mappings_match_spec() {
        assert_eq!(map_stop_reason(Some("end_turn")), FinishReason::Stop);
        assert_eq!(map_stop_reason(Some("max_tokens")), FinishReason::Length);
        assert_eq!(map_stop_reason(Some("tool_use")), FinishReason::ToolCalls);
        assert_eq!(
            map_stop_reason(Some("exotic_reason")),
            FinishReason::Other("exotic_reason".into())
        );
        assert_eq!(map_stop_reason(None), FinishReason::Stop);
    }

    #[test]
    fn content_blocks_other_than_text_are_skipped() {
        // Tool-use blocks on a completion we're treating as plain text
        // should not break parsing; they're simply not surfaced yet.
        let body = r#"{
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {}},
                {"type": "text", "text": "done"}
            ]
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.message.content_str(), "done");
    }

    #[test]
    fn stream_events_deserialise_into_typed_variants() {
        let start: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_start","message":{"id":"msg_1","model":"claude","type":"message","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":1}}}"#,
        )
        .unwrap();
        assert!(matches!(start, AnthropicStreamEvent::MessageStart { .. }));

        let delta: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        )
        .unwrap();
        assert!(matches!(
            delta,
            AnthropicStreamEvent::ContentBlockDelta { .. }
        ));

        let msg_delta: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#,
        )
        .unwrap();
        assert!(matches!(
            msg_delta,
            AnthropicStreamEvent::MessageDelta { .. }
        ));

        let ping: AnthropicStreamEvent = serde_json::from_str(r#"{"type":"ping"}"#).unwrap();
        assert!(matches!(ping, AnthropicStreamEvent::Other));
    }

    #[test]
    fn stream_state_tracks_id_and_emits_text_delta() {
        let mut state = StreamState::default();
        let start: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_start","message":{"id":"msg_9","model":"claude-sonnet-4-5","type":"message","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":1}}}"#,
        )
        .unwrap();
        state.update(&start);
        assert_eq!(state.id, "msg_9");
        assert!(state.to_chunk(&start).is_none());

        let delta: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        )
        .unwrap();
        let chunk = state.to_chunk(&delta).unwrap();
        assert_eq!(chunk.id, "msg_9");
        assert_eq!(chunk.delta.content.as_deref(), Some("hi"));
    }

    #[test]
    fn stream_state_emits_finish_on_message_delta() {
        let state = StreamState {
            id: "msg".into(),
            model: "claude".into(),
            ..Default::default()
        };
        let end: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#,
        )
        .unwrap();
        let chunk = state.to_chunk(&end).unwrap();
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
        assert_eq!(chunk.usage.unwrap().completion_tokens, 3);
    }

    #[test]
    fn stream_state_carries_message_start_input_tokens_into_final_usage() {
        // message_start input_tokens must survive into the usage emitted on
        // the terminal message_delta — otherwise prompt tokens are dropped
        // for the whole stream (TPM/budget/telemetry undercount). See #450.
        let mut state = StreamState::default();
        let start: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_start","message":{"id":"m","model":"claude","type":"message","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":37,"output_tokens":1}}}"#,
        )
        .unwrap();
        state.update(&start);
        assert_eq!(state.input_tokens, 37);

        let end: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":52}}"#,
        )
        .unwrap();
        let usage = state.to_chunk(&end).unwrap().usage.unwrap();
        assert_eq!(usage.prompt_tokens, 37);
        assert_eq!(usage.completion_tokens, 52);
        assert_eq!(usage.total_tokens, 89);
    }

    #[test]
    fn stream_state_carries_message_start_cache_tokens_into_final_usage() {
        // #906: Anthropic sends cache_creation / cache_read only on
        // message_start. The cross-protocol bridge must carry them onto
        // the terminal usage (and fold them into the total), else an
        // OpenAI-shape client streaming against an Anthropic upstream
        // loses cache tokens entirely — not just from the total.
        let mut state = StreamState::default();
        let start: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_start","message":{"id":"m","model":"claude","type":"message","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1,"cache_creation_input_tokens":200,"cache_read_input_tokens":800}}}"#,
        )
        .unwrap();
        state.update(&start);
        assert_eq!(state.cache_creation_tokens, 200);
        assert_eq!(state.cache_read_tokens, 800);

        let end: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}"#,
        )
        .unwrap();
        let usage = state.to_chunk(&end).unwrap().usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 4);
        assert_eq!(usage.cache_creation_tokens, 200);
        assert_eq!(usage.cache_read_tokens, 800);
        assert_eq!(usage.total_tokens, 10 + 4 + 200 + 800);
        // Option A: cache is neither folded into prompt_tokens nor
        // mapped onto cached_prompt_tokens — the latter would
        // double-count cost in cp-api's pricing formula, which bills
        // cache_read as its own term.
        assert_eq!(usage.cached_prompt_tokens, 0);
    }

    // ─── parse_inbound_request ────────────────────────────────────

    #[test]
    fn parse_inbound_minimal_user_only() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        });
        let chat = parse_inbound_request(&body).unwrap();
        assert_eq!(chat.model, "claude-sonnet-4-5");
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, Role::User);
        assert_eq!(chat.messages[0].content_str(), "hi");
        assert_eq!(chat.max_tokens, Some(100));
    }

    #[test]
    fn parse_inbound_system_string_folds_to_leading_message() {
        let body = serde_json::json!({
            "model": "claude",
            "system": "you are helpful",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let chat = parse_inbound_request(&body).unwrap();
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, Role::System);
        assert_eq!(chat.messages[0].content_str(), "you are helpful");
        assert_eq!(chat.messages[1].role, Role::User);
    }

    #[test]
    fn parse_inbound_system_block_array_concatenates_with_newline() {
        let body = serde_json::json!({
            "model": "claude",
            "system": [
                {"type": "text", "text": "line1"},
                {"type": "text", "text": "line2"},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        });
        let chat = parse_inbound_request(&body).unwrap();
        assert_eq!(chat.messages[0].role, Role::System);
        assert_eq!(chat.messages[0].content_str(), "line1\nline2");
    }

    #[test]
    fn parse_inbound_content_block_array_concatenates_text_only() {
        let body = serde_json::json!({
            "model": "claude",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "image", "source": {"type": "base64", "data": "xx"}},
                    {"type": "text", "text": "world"},
                ],
            }],
        });
        let chat = parse_inbound_request(&body).unwrap();
        // Image block silently skipped; text concatenates.
        assert_eq!(chat.messages[0].content_str(), "hello world");
    }

    #[test]
    fn parse_inbound_unknown_top_level_keys_flow_to_extra() {
        let body = serde_json::json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_id": "abc"},
            "tools": [{"name": "get_weather"}],
        });
        let chat = parse_inbound_request(&body).unwrap();
        assert!(chat.extra.contains_key("metadata"));
        assert!(chat.extra.contains_key("tools"));
        assert!(!chat.extra.contains_key("model"));
        assert!(!chat.extra.contains_key("messages"));
    }

    /// #597: Claude Code/cc-switch send `role: "system"` inside `messages[]`
    /// even though the Anthropic spec only allows user/assistant. Parse it
    /// as Role::System instead of rejecting the whole request with a 400.
    #[test]
    fn parse_inbound_accepts_system_role_in_messages() {
        let body = serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "respond in French"},
                {"role": "user", "content": "hello again"},
            ],
        });
        let chat = parse_inbound_request(&body).unwrap();
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(chat.messages[0].role, Role::User);
        assert_eq!(chat.messages[1].role, Role::System);
        assert_eq!(chat.messages[1].content_str(), "respond in French");
        assert_eq!(chat.messages[2].role, Role::User);
    }

    #[test]
    fn parse_inbound_rejects_unknown_role() {
        let body = serde_json::json!({
            "model": "claude",
            "messages": [{"role": "tool", "content": "x"}],
        });
        let err = parse_inbound_request(&body).unwrap_err();
        assert!(matches!(err, AnthropicInboundError::UnsupportedRole { .. }));
    }

    #[test]
    fn parse_inbound_rejects_missing_model() {
        let body = serde_json::json!({"messages": []});
        assert!(matches!(
            parse_inbound_request(&body).unwrap_err(),
            AnthropicInboundError::MissingModel,
        ));
    }

    // ─── translate_extras_to_openai_shape (AISIX-Cloud#953) ───────

    #[test]
    fn extras_shape_drops_anthropic_only_fields() {
        let mut extra = serde_json::json!({
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
            "top_k": 40,
            "mcp_servers": [{"type": "url", "url": "https://example.com/mcp"}],
            "container": "container_abc",
            "service_tier": "standard_only",
            "betas": ["context-management-2025-06-27"],
            "anthropic_version": "2023-06-01",
        })
        .as_object()
        .unwrap()
        .clone();
        translate_extras_to_openai_shape(&mut extra);
        assert!(extra.is_empty(), "expected all dropped, got: {extra:?}");
    }

    #[test]
    fn extras_shape_translates_mappable_fields() {
        let mut extra = serde_json::json!({
            "stop_sequences": ["\n\nHuman:"],
            "metadata": {"user_id": "user-123"},
            "tools": [{"name": "get_time", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
        })
        .as_object()
        .unwrap()
        .clone();
        translate_extras_to_openai_shape(&mut extra);

        assert_eq!(extra.get("stop"), Some(&serde_json::json!(["\n\nHuman:"])));
        assert!(!extra.contains_key("stop_sequences"));
        assert_eq!(extra.get("user"), Some(&serde_json::json!("user-123")));
        assert!(!extra.contains_key("metadata"));
        assert_eq!(
            extra.get("tools").unwrap()[0]["function"]["name"],
            serde_json::json!("get_time")
        );
        assert_eq!(
            extra.get("tool_choice"),
            Some(&serde_json::json!("required"))
        );
    }

    #[test]
    fn extras_shape_metadata_without_user_id_is_dropped() {
        let mut extra = serde_json::json!({"metadata": {"foo": "bar"}})
            .as_object()
            .unwrap()
            .clone();
        translate_extras_to_openai_shape(&mut extra);
        assert!(extra.is_empty());
    }

    #[test]
    fn extras_shape_thinking_buckets_to_reasoning_effort() {
        for (thinking, expected) in [
            (
                serde_json::json!({"type": "enabled", "budget_tokens": 8000}),
                Some("high"),
            ),
            (
                serde_json::json!({"type": "enabled", "budget_tokens": 2048}),
                Some("medium"),
            ),
            (
                serde_json::json!({"type": "enabled", "budget_tokens": 1024}),
                Some("low"),
            ),
            (
                serde_json::json!({"type": "enabled", "budget_tokens": 100}),
                Some("minimal"),
            ),
            (serde_json::json!({"type": "adaptive"}), Some("medium")),
            (serde_json::json!({"type": "disabled"}), None),
        ] {
            let mut extra = serde_json::Map::new();
            extra.insert("thinking".to_string(), thinking.clone());
            translate_extras_to_openai_shape(&mut extra);
            assert_eq!(
                extra.get("reasoning_effort").and_then(|v| v.as_str()),
                expected,
                "thinking = {thinking}"
            );
            assert!(!extra.contains_key("thinking"));
        }
    }

    // ─── chat_response_into_anthropic_json ────────────────────────

    #[test]
    fn render_anthropic_response_basic_shape() {
        let resp = ChatResponse {
            id: "cmpl-1".into(),
            model: "gpt-4o".into(), // upstream — should NOT leak into output
            message: ChatMessage::assistant("hello"),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(7, 3),
        };
        let json = chat_response_into_anthropic_json(&resp, "my-claude-alias");
        assert_eq!(json["id"], "cmpl-1");
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["model"], "my-claude-alias");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "hello");
        assert_eq!(json["stop_reason"], "end_turn");
        assert!(json["stop_sequence"].is_null());
        assert_eq!(json["usage"]["input_tokens"], 7);
        assert_eq!(json["usage"]["output_tokens"], 3);
    }

    #[test]
    fn render_anthropic_response_finish_reason_mappings() {
        let mk = |fr: FinishReason| {
            let resp = ChatResponse {
                id: "x".into(),
                model: "u".into(),
                message: ChatMessage::assistant(""),
                finish_reason: fr,
                usage: UsageStats::new(0, 0),
            };
            chat_response_into_anthropic_json(&resp, "m")["stop_reason"]
                .as_str()
                .unwrap()
                .to_string()
        };
        assert_eq!(mk(FinishReason::Stop), "end_turn");
        assert_eq!(mk(FinishReason::Length), "max_tokens");
        assert_eq!(mk(FinishReason::ContentFilter), "stop_sequence");
        assert_eq!(mk(FinishReason::ToolCalls), "tool_use");
        assert_eq!(mk(FinishReason::Other("vendor".into())), "end_turn");
    }

    // ─── AnthropicSseEncoder ──────────────────────────────────────

    #[test]
    fn render_anthropic_response_translates_openai_tool_calls_to_tool_use() {
        let mut msg = ChatMessage::assistant("");
        msg.extra.insert(
            "tool_calls".to_string(),
            serde_json::json!([{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": "{\"timezone\":\"UTC\"}"
                }
            }]),
        );
        let resp = ChatResponse {
            id: "cmpl-tc".into(),
            model: "gpt-4o".into(),
            message: msg,
            finish_reason: FinishReason::ToolCalls,
            usage: UsageStats::new(10, 5),
        };
        let json = chat_response_into_anthropic_json(&resp, "my-model");
        assert_eq!(json["stop_reason"], "tool_use");
        let content = json["content"].as_array().unwrap();
        let tool_block = content.iter().find(|b| b["type"] == "tool_use");
        assert!(tool_block.is_some(), "tool_use block must be present");
        let tb = tool_block.unwrap();
        assert_eq!(tb["id"], "call_abc");
        assert_eq!(tb["name"], "get_time");
        assert_eq!(tb["input"]["timezone"], "UTC");
    }

    fn delta_chunk(text: &str) -> ChatChunk {
        ChatChunk {
            id: "cmpl-1".into(),
            model: "u".into(),
            delta: ChatDelta {
                role: None,
                content: Some(text.into()),
                tool_calls: None,
                reasoning_content: None,
            },
            finish_reason: None,
            usage: None,
        }
    }

    fn finish_chunk(out_tokens: u32) -> ChatChunk {
        ChatChunk {
            id: "cmpl-1".into(),
            model: "u".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::Stop),
            usage: Some(UsageStats::new(0, out_tokens)),
        }
    }

    #[test]
    fn sse_encoder_first_content_chunk_emits_message_start_then_block_start_then_delta() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "claude-alias", 5);
        let events = enc.next_events(&delta_chunk("hello"));
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(
            kinds,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );
        assert_eq!(
            events[0].data["message"]["usage"]["input_tokens"], 5,
            "initial input_tokens echoed in message_start"
        );
        assert_eq!(events[0].data["message"]["model"], "claude-alias");
        assert_eq!(events[2].data["delta"]["text"], "hello");
    }

    #[test]
    fn sse_encoder_subsequent_chunks_only_emit_deltas() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "alias", 0);
        let _ = enc.next_events(&delta_chunk("hel"));
        let events = enc.next_events(&delta_chunk("lo"));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event, "content_block_delta");
        assert_eq!(events[0].data["delta"]["text"], "lo");
    }

    #[test]
    fn sse_encoder_finish_chunk_after_content_emits_stop_trio() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "alias", 0);
        let _ = enc.next_events(&delta_chunk("hi"));
        let events = enc.next_events(&finish_chunk(2));
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(
            kinds,
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
        assert_eq!(events[1].data["delta"]["stop_reason"], "end_turn");
        assert_eq!(events[1].data["usage"]["output_tokens"], 2);
        assert!(enc.is_finished());
        // Subsequent chunks are silent.
        assert!(enc.next_events(&delta_chunk("ignored")).is_empty());
    }

    /// #790: OpenAI's `stream_options.include_usage` frame arrives
    /// AFTER the stop chunk. The closing pair must wait for it so the
    /// client sees real token counts instead of `output_tokens: 0`.
    #[test]
    fn sse_encoder_holds_close_until_post_stop_usage_frame() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "alias", 0);
        let _ = enc.next_events(&delta_chunk("hi"));

        // Stop chunk with NO usage — only the content block closes.
        let stop_no_usage = ChatChunk {
            id: "cmpl-1".into(),
            model: "u".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        };
        let events = enc.next_events(&stop_no_usage);
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(kinds, vec!["content_block_stop"]);
        assert!(!enc.is_finished());

        // The trailing usage-only frame releases the closing pair,
        // carrying both input and output tokens.
        let usage_only = ChatChunk {
            id: "cmpl-1".into(),
            model: "u".into(),
            delta: ChatDelta::default(),
            finish_reason: None,
            usage: Some(UsageStats::new(17, 23)),
        };
        let events = enc.next_events(&usage_only);
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(kinds, vec!["message_delta", "message_stop"]);
        assert_eq!(events[0].data["delta"]["stop_reason"], "end_turn");
        assert_eq!(events[0].data["usage"]["input_tokens"], 17);
        assert_eq!(events[0].data["usage"]["output_tokens"], 23);
        assert!(enc.is_finished());
    }

    /// An upstream that ignores `stream_options` never sends the usage
    /// frame — stream end (force_finish) must flush the withheld pair
    /// with the REAL stop reason, not `end_turn`.
    #[test]
    fn sse_encoder_force_finish_flushes_held_close_with_real_stop_reason() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "alias", 0);
        let _ = enc.next_events(&delta_chunk("hi"));
        let stop_no_usage = ChatChunk {
            id: "cmpl-1".into(),
            model: "u".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
        };
        let _ = enc.next_events(&stop_no_usage);
        assert!(!enc.is_finished());

        let events = enc.force_finish();
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(kinds, vec!["message_delta", "message_stop"]);
        assert_eq!(events[0].data["delta"]["stop_reason"], "tool_use");
        assert_eq!(events[0].data["usage"]["output_tokens"], 0);
        assert!(enc.is_finished());
    }

    #[test]
    fn sse_encoder_finish_only_chunk_skips_content_block_stop() {
        // Finish without prior content (e.g. blocked by guardrail) —
        // we still emit message_start + message_delta + message_stop
        // but NOT content_block_start/stop.
        let mut enc = AnthropicSseEncoder::new("msg_01", "alias", 0);
        let events = enc.next_events(&finish_chunk(0));
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(
            kinds,
            vec!["message_start", "message_delta", "message_stop"]
        );
    }

    #[test]
    fn sse_encoder_force_finish_after_content_emits_full_close() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "alias", 3);
        let _ = enc.next_events(&delta_chunk("hi"));
        let events = enc.force_finish();
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(
            kinds,
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
        assert!(enc.is_finished());
    }

    #[test]
    fn sse_encoder_force_finish_on_empty_stream_emits_message_start_then_close() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "alias", 0);
        let events = enc.force_finish();
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(
            kinds,
            vec!["message_start", "message_delta", "message_stop"]
        );
    }

    #[test]
    fn sse_event_renders_as_event_data_pair() {
        let ev = AnthropicSseEvent {
            event: "content_block_delta",
            data: serde_json::json!({"x": 1}),
        };
        let s = ev.to_sse_string();
        assert_eq!(s, "event: content_block_delta\ndata: {\"x\":1}\n\n");
    }

    // ─── Streaming tool_calls ──────────────────────────────────────

    fn tool_call_chunk(index: u64, id: &str, name: &str, arguments: &str) -> ChatChunk {
        let mut tc = serde_json::json!({"index": index});
        if !id.is_empty() {
            tc["id"] = serde_json::json!(id);
            tc["type"] = serde_json::json!("function");
        }
        let mut func = serde_json::Map::new();
        if !name.is_empty() {
            func.insert("name".into(), serde_json::json!(name));
        }
        if !arguments.is_empty() {
            func.insert("arguments".into(), serde_json::json!(arguments));
        }
        if !func.is_empty() {
            tc["function"] = serde_json::Value::Object(func);
        }
        ChatChunk {
            id: "cmpl-1".into(),
            model: "u".into(),
            delta: ChatDelta {
                role: None,
                content: None,
                tool_calls: Some(vec![tc]),
                reasoning_content: None,
            },
            finish_reason: None,
            usage: None,
        }
    }

    fn tool_finish_chunk() -> ChatChunk {
        ChatChunk {
            id: "cmpl-1".into(),
            model: "u".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::ToolCalls),
            usage: Some(UsageStats::new(10, 5)),
        }
    }

    #[test]
    fn sse_encoder_tool_call_emits_block_start_and_argument_deltas() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "m", 0);
        // First chunk: tool header with id+name and initial args.
        let events = enc.next_events(&tool_call_chunk(0, "call_1", "get_weather", "{\"loc"));
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(
            kinds,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );
        // content_block_start should be tool_use
        assert_eq!(events[1].data["content_block"]["type"], "tool_use");
        assert_eq!(events[1].data["content_block"]["id"], "call_1");
        assert_eq!(events[1].data["content_block"]["name"], "get_weather");
        // content_block_delta should be input_json_delta
        assert_eq!(events[2].data["delta"]["type"], "input_json_delta");
        assert_eq!(events[2].data["delta"]["partial_json"], "{\"loc");
    }

    #[test]
    fn sse_encoder_tool_call_subsequent_args_emit_delta_only() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "m", 0);
        enc.next_events(&tool_call_chunk(0, "call_1", "get_weather", ""));
        let events = enc.next_events(&tool_call_chunk(0, "", "", "ation\"}"));
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(kinds, vec!["content_block_delta"]);
        assert_eq!(events[0].data["delta"]["partial_json"], "ation\"}");
    }

    #[test]
    fn sse_encoder_tool_finish_closes_all_blocks() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "m", 0);
        enc.next_events(&tool_call_chunk(0, "call_1", "fn_a", "{}"));
        enc.next_events(&tool_call_chunk(1, "call_2", "fn_b", "{}"));
        let events = enc.next_events(&tool_finish_chunk());
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        // Should close both tool blocks, then message_delta + message_stop
        assert_eq!(
            kinds,
            vec![
                "content_block_stop",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert_eq!(events[2].data["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn sse_encoder_mixed_text_and_tool_call() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "m", 0);
        // Text first
        enc.next_events(&delta_chunk("thinking..."));
        // Then a tool call
        let events = enc.next_events(&tool_call_chunk(0, "call_1", "search", "{\"q\":\"x\"}"));
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(kinds, vec!["content_block_start", "content_block_delta"]);
        // Tool block should be at index 1 (text was 0)
        assert_eq!(events[0].data["index"], 1);
        // Finish
        let events = enc.next_events(&tool_finish_chunk());
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        // Close text block (0), tool block (1), then message_delta + stop
        assert_eq!(
            kinds,
            vec![
                "content_block_stop",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
    }

    #[test]
    fn sse_encoder_force_finish_closes_tool_blocks() {
        let mut enc = AnthropicSseEncoder::new("msg_01", "m", 0);
        enc.next_events(&tool_call_chunk(0, "call_1", "fn_a", "{}"));
        let events = enc.force_finish();
        let kinds: Vec<_> = events.iter().map(|e| e.event).collect();
        assert_eq!(
            kinds,
            vec!["content_block_stop", "message_delta", "message_stop"]
        );
    }
}
