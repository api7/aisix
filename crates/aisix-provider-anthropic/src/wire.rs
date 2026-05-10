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
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AnthropicRequest<'a> {
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
pub(crate) struct AnthropicMessage<'a> {
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
pub(crate) fn split_system<'a>(
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
                    messages.push(AnthropicMessage::text("user", &m.content));
                } else {
                    system_parts.push(&m.content);
                }
            }
            Role::User => {
                seen_non_system = true;
                messages.push(AnthropicMessage::text("user", &m.content));
            }
            Role::Assistant => {
                seen_non_system = true;
                messages.push(AnthropicMessage::text("assistant", &m.content));
            }
            Role::Tool => {
                seen_non_system = true;
                let tool_use_id = m
                    .tool_call_id
                    .as_deref()
                    .ok_or(TranslateError::MissingToolCallId)?;
                messages.push(AnthropicMessage::tool_result(tool_use_id, &m.content));
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    Ok((system, messages))
}

pub(crate) fn build_request<'a>(
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
pub(crate) fn translate_openai_tools_to_anthropic(
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
pub(crate) fn translate_openai_tool_choice_to_anthropic(
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

/// Non-streaming response shape from `/v1/messages`.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicResponse {
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
pub(crate) enum AnthropicResponseBlock {
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

#[derive(Debug, Default, Deserialize)]
pub(crate) struct AnthropicUsage {
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

pub(crate) fn response_into_chat_response(raw: AnthropicResponse) -> ChatResponse {
    let text = raw
        .content
        .iter()
        .filter_map(|b| match b {
            AnthropicResponseBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");

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
        .map(|u| UsageStats {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens.saturating_add(u.output_tokens),
            cache_creation_tokens: u.cache_creation_input_tokens,
            cache_read_tokens: u.cache_read_input_tokens,
            // Anthropic doesn't use OpenAI's cached-prompt-tokens or
            // reasoning-tokens taxonomy; leave at 0.
            cached_prompt_tokens: 0,
            reasoning_tokens: 0,
        })
        .unwrap_or_default();

    ChatResponse {
        id: raw.id,
        model: raw.model,
        message: ChatMessage {
            role: Role::Assistant,
            content: text,
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
pub(crate) enum AnthropicStreamEvent {
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
pub(crate) struct AnthropicStreamStartMessage {
    pub id: String,
    pub model: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicStreamDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicStreamMessageDelta {
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicStreamUsage {
    #[serde(default)]
    pub output_tokens: Option<u32>,
}

/// Rolling state the Bridge carries across a stream so chunks can be
/// tagged with the message id/model even though only the first event
/// carries them.
#[derive(Debug, Default)]
pub(crate) struct StreamState {
    pub id: String,
    pub model: String,
}

impl StreamState {
    pub fn update(&mut self, event: &AnthropicStreamEvent) {
        if let AnthropicStreamEvent::MessageStart { message } = event {
            self.id = message.id.clone();
            self.model = message.model.clone();
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
                },
                finish_reason: None,
                usage: None,
            }),
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                let finish = delta
                    .stop_reason
                    .as_deref()
                    .map(|r| map_stop_reason(Some(r)));
                let usage = usage
                    .as_ref()
                    .and_then(|u| u.output_tokens.map(|n| UsageStats::new(0, n)));
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
    #[error("messages[{idx}] role {role:?} is not 'user' or 'assistant'")]
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
    serde_json::json!({
        "id": resp.id,
        "type": "message",
        "role": "assistant",
        "model": model_display_name,
        "content": [{"type": "text", "text": resp.message.content}],
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

/// State machine for re-encoding a stream of internal `ChatChunk`s as
/// Anthropic SSE events.
#[derive(Debug)]
pub struct AnthropicSseEncoder {
    message_id: String,
    model_display_name: String,
    initial_input_tokens: u32,
    sent_message_start: bool,
    sent_content_block_start: bool,
    finished: bool,
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
            sent_content_block_start: false,
            finished: false,
        }
    }

    /// Translate one chunk into the Anthropic SSE events to emit.
    /// Returns an empty Vec on no-op chunks (e.g. usage-only).
    pub fn next_events(&mut self, chunk: &ChatChunk) -> Vec<AnthropicSseEvent> {
        if self.finished {
            return Vec::new();
        }

        let mut events = Vec::new();

        let has_content = chunk
            .delta
            .content
            .as_deref()
            .is_some_and(|s| !s.is_empty());
        let has_finish = chunk.finish_reason.is_some();

        if !self.sent_message_start && (has_content || has_finish) {
            events.push(self.message_start_event());
            self.sent_message_start = true;
        }

        if !self.sent_content_block_start && has_content {
            events.push(content_block_start_event());
            self.sent_content_block_start = true;
        }

        if has_content {
            events.push(AnthropicSseEvent {
                event: "content_block_delta",
                data: serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": chunk.delta.content.clone().unwrap_or_default(),
                    },
                }),
            });
        }

        if let Some(fr) = &chunk.finish_reason {
            if self.sent_content_block_start {
                events.push(content_block_stop_event());
            }
            let stop_reason = match fr {
                FinishReason::Stop => "end_turn",
                FinishReason::Length => "max_tokens",
                FinishReason::ContentFilter => "stop_sequence",
                FinishReason::ToolCalls => "tool_use",
                FinishReason::Other(_) => "end_turn",
            };
            let output_tokens = chunk
                .usage
                .as_ref()
                .map(|u| u.completion_tokens)
                .unwrap_or(0);
            events.push(AnthropicSseEvent {
                event: "message_delta",
                data: serde_json::json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": serde_json::Value::Null,
                    },
                    "usage": {"output_tokens": output_tokens},
                }),
            });
            events.push(AnthropicSseEvent {
                event: "message_stop",
                data: serde_json::json!({"type": "message_stop"}),
            });
            self.finished = true;
        }

        events
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Force-close the stream when the upstream ended without
    /// emitting a chunk with `finish_reason`. Emits the closing trio
    /// with `stop_reason: "end_turn"` and `output_tokens: 0`.
    /// Idempotent.
    pub fn force_finish(&mut self) -> Vec<AnthropicSseEvent> {
        if self.finished {
            return Vec::new();
        }
        let mut events = Vec::new();
        if !self.sent_message_start {
            events.push(self.message_start_event());
            self.sent_message_start = true;
        }
        if self.sent_content_block_start {
            events.push(content_block_stop_event());
        }
        events.push(AnthropicSseEvent {
            event: "message_delta",
            data: serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "stop_sequence": serde_json::Value::Null,
                },
                "usage": {"output_tokens": 0},
            }),
        });
        events.push(AnthropicSseEvent {
            event: "message_stop",
            data: serde_json::json!({"type": "message_stop"}),
        });
        self.finished = true;
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

fn content_block_start_event() -> AnthropicSseEvent {
    AnthropicSseEvent {
        event: "content_block_start",
        data: serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
    }
}

fn content_block_stop_event() -> AnthropicSseEvent {
    AnthropicSseEvent {
        event: "content_block_stop",
        data: serde_json::json!({"type": "content_block_stop", "index": 0}),
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
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1].role, "user"); // former system message
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
                content: "x".into(),
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
                // (skipping the assistant turn for brevity in test setup)
                ChatMessage {
                    role: Role::Tool,
                    content: "72F, sunny".into(),
                    content_blocks: None,
                    name: None,
                    tool_call_id: Some("toolu_abc".into()),
                    extra: serde_json::Map::new(),
                },
            ],
        );
        let (_system, msgs) = split_system(&req).unwrap();
        assert_eq!(msgs.len(), 2);
        // Tool turn became a user turn with a tool_result block.
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content.len(), 1);
        assert_eq!(msgs[1].content[0]["type"], "tool_result");
        assert_eq!(msgs[1].content[0]["tool_use_id"], "toolu_abc");
        assert_eq!(msgs[1].content[0]["content"], "72F, sunny");
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
        // Text content is empty when only tool_use blocks were
        // emitted (no text blocks present).
        assert_eq!(out.message.content, "");

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
        assert_eq!(out.message.content, "Let me check the weather.");
        assert!(out.message.extra.get("tool_calls").is_some());
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
        assert_eq!(out.message.content, "hello");
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
        assert_eq!(out.message.content, "done");
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
        };
        let end: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#,
        )
        .unwrap();
        let chunk = state.to_chunk(&end).unwrap();
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
        assert_eq!(chunk.usage.unwrap().completion_tokens, 3);
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
        assert_eq!(chat.messages[0].content, "hi");
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
        assert_eq!(chat.messages[0].content, "you are helpful");
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
        assert_eq!(chat.messages[0].content, "line1\nline2");
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
        assert_eq!(chat.messages[0].content, "hello world");
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

    fn delta_chunk(text: &str) -> ChatChunk {
        ChatChunk {
            id: "cmpl-1".into(),
            model: "u".into(),
            delta: ChatDelta {
                role: None,
                content: Some(text.into()),
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
}
