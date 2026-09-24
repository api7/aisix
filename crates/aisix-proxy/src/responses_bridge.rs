//! Cross-provider translation for `POST /v1/responses` (#825).
//!
//! The Responses API is OpenAI-specific, but clients such as the `codex`
//! CLI point it at non-OpenAI models. For an OpenAI upstream the handler
//! forwards the body verbatim (see [`crate::responses`]); for any other
//! provider this module translates the request into the gateway's
//! canonical [`ChatFormat`], so it can be dispatched through the same
//! provider [`Bridge`](aisix_gateway::Bridge) `/v1/chat/completions` uses,
//! and re-encodes the bridge's response back into the Responses API shape
//! — non-streaming JSON and streaming SSE. This mirrors the cross-provider
//! path of `/v1/messages` (`messages::cross_provider_dispatch`).
//!
//! Only the Responses fields that map cleanly onto chat completions are
//! carried (`instructions`, `input` — including its image / file / audio
//! content parts —, `tools`, `tool_choice`, `temperature`, `top_p`,
//! `max_output_tokens`, `stream`, `reasoning.effort`, and `text.format` as
//! `response_format`). Other OpenAI-only knobs (`store`,
//! `previous_response_id`, `text.verbosity`, …) are dropped rather than
//! forwarded — the downstream provider bridges flatten unknown `extra`
//! fields onto the upstream wire, where an OpenAI-only key would 400 (e.g.
//! Anthropic).

use std::sync::Arc;
use std::time::Instant;

use aisix_gateway::{
    ChatChunk, ChatChunkStream, ChatFormat, ChatMessage, ChatResponse, FinishReason, Role,
    UsageStats,
};
use serde_json::{json, Map, Value};
use uuid::Uuid;

/// Translate a `/v1/responses` request body into the gateway's canonical
/// [`ChatFormat`]. Unlike `responses::responses_input_to_chat` (which is a
/// lossy, text-only projection used solely for input-guardrail scanning),
/// this is the faithful transform actually dispatched upstream: it carries
/// roles, tool calls, tool results, tools, and sampling params.
pub fn responses_request_to_chat(model: &str, body: &Value) -> ChatFormat {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // Top-level `instructions` is the Responses-API system prompt.
    if let Some(instructions) = body.get("instructions").and_then(|v| v.as_str()) {
        if !instructions.is_empty() {
            messages.push(ChatMessage::system(instructions.to_string()));
        }
    }

    match body.get("input") {
        Some(Value::String(text)) => {
            if !text.is_empty() {
                messages.push(ChatMessage::user(text.clone()));
            }
        }
        Some(Value::Array(items)) => {
            for item in items {
                append_input_item(&mut messages, item);
            }
            messages = attach_replayed_reasoning(messages);
        }
        _ => {}
    }

    let mut chat = ChatFormat::new(model, messages);
    chat.temperature = body
        .get("temperature")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    chat.top_p = body.get("top_p").and_then(|v| v.as_f64()).map(|f| f as f32);
    // Responses calls the cap `max_output_tokens`; tolerate `max_tokens`
    // too for clients that send the chat-style name. A value that doesn't
    // fit u32 is dropped (left unset) rather than silently wrapped to a
    // small/zero cap.
    chat.max_tokens = body
        .get("max_output_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok());
    chat.stream = body.get("stream").and_then(|v| v.as_bool());

    // Tools/tool_choice ride `extra` in OpenAI chat shape; every provider
    // bridge translates that shape to its own (Anthropic, Gemini, …), so
    // emitting it here is all that's needed. `tool_choice` only travels
    // with a surviving `tools` list: a chat-completions upstream rejects
    // it on its own ("'tool_choice' is only allowed when 'tools' are
    // specified"), and the Responses API accepts requests that carry an
    // empty or hosted-tools-only list alongside one — a shape the Codex
    // CLI sends on every context compaction (AISIX-Cloud#1614).
    match body.get("tools").and_then(responses_tools_to_chat) {
        Some(tools) => {
            chat.extra.insert("tools".to_string(), tools);
            if let Some(tc) = body
                .get("tool_choice")
                .and_then(responses_tool_choice_to_chat)
            {
                chat.extra.insert("tool_choice".to_string(), tc);
            }
            // `parallel_tool_calls` is the same boolean in both APIs, and
            // travels under the same condition as `tool_choice`.
            if let Some(p) = body.get("parallel_tool_calls").and_then(Value::as_bool) {
                chat.extra
                    .insert("parallel_tool_calls".to_string(), Value::Bool(p));
            }
        }
        // A caller that asked for a tool call and lost it to this filter
        // gets prose back instead of an upstream 400; say so, or the
        // downgrade is invisible from the logs.
        None if body.get("tool_choice").is_some() || body.get("parallel_tool_calls").is_some() => {
            tracing::debug!(
                "dropping tool_choice/parallel_tool_calls on the chat bridge: no tool survived translation"
            )
        }
        None => {}
    }
    if let Some(effort) = body.pointer("/reasoning/effort").and_then(Value::as_str) {
        chat.extra
            .insert("reasoning_effort".to_string(), effort.into());
    }
    // Structured outputs: Responses spells them `text.format`, chat spells
    // them `response_format`. Dropping the field made a caller that asked for
    // a schema get prose back.
    if let Some(rf) = body
        .get("text")
        .and_then(responses_text_format_to_response_format)
    {
        chat.extra.insert("response_format".to_string(), rf);
    }
    chat
}

/// Append one Responses-API `input` array element as chat message(s).
fn append_input_item(messages: &mut Vec<ChatMessage>, item: &Value) {
    // A bare-string element is user text.
    if let Some(text) = item.as_str() {
        if !text.is_empty() {
            messages.push(ChatMessage::user(text.to_string()));
        }
        return;
    }

    match item.get("type").and_then(|t| t.as_str()) {
        // A prior assistant tool call replayed for the agent loop.
        Some("function_call") => {
            let call_id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            // A call to a namespace sub-tool goes back under the flattened
            // name the model was offered it as (see
            // [`namespace_chat_tool_name`]).
            let name = match item
                .get("namespace")
                .and_then(Value::as_str)
                .filter(|ns| !ns.is_empty())
            {
                Some(namespace) => namespace_chat_tool_name(namespace, name),
                None => name.to_string(),
            };
            let arguments = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
            push_tool_call(
                messages,
                json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }),
            );
        }
        // A prior custom-tool call replayed for the agent loop. The request
        // side gave the model a function tool taking one string
        // (`custom_tool_parameters`), so the replayed call has to go back in
        // that same shape or the history stops matching the tools the model
        // was given.
        Some("custom_tool_call") => {
            let call_id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let input = item.get("input").and_then(|v| v.as_str()).unwrap_or("");
            push_tool_call(
                messages,
                json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": wrap_custom_tool_input(input),
                    },
                }),
            );
        }
        // The tool result fed back by the caller → a `tool` role message.
        // A custom tool's result item is the same shape under a different
        // type name, and its `output` takes the same string-or-parts union.
        Some("function_call_output" | "custom_tool_call_output") => {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let output = item
                .get("output")
                .map(function_call_output_to_chat)
                .unwrap_or_else(|| ChatContent {
                    text: String::new(),
                    blocks: None,
                });
            messages.push(ChatMessage {
                role: Role::Tool,
                content: Some(output.text),
                content_blocks: output.blocks,
                name: None,
                tool_call_id: Some(call_id.to_string()),
                extra: Map::new(),
            });
        }
        // The model's own earlier chain-of-thought. It becomes an assistant
        // message carrying nothing but `reasoning_content`, which
        // [`attach_replayed_reasoning`] then folds onto the assistant turn it
        // belongs to. An item with no readable text (only
        // `encrypted_content`, which is another provider's ciphertext)
        // replays nothing.
        Some("reasoning") => {
            if let Some(text) = replayed_reasoning_text(item) {
                let mut extra = Map::new();
                extra.insert(REASONING_CONTENT.to_string(), Value::String(text));
                messages.push(ChatMessage {
                    role: Role::Assistant,
                    content: None,
                    content_blocks: None,
                    name: None,
                    tool_call_id: None,
                    extra,
                });
            }
        }
        // A `message` item (or an untyped `{role, content}` element).
        _ => {
            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = item
                .get("content")
                .map(responses_content_to_chat)
                .unwrap_or_else(|| ChatContent {
                    text: String::new(),
                    blocks: None,
                });
            // A turn made only of images / files / audio still has to reach
            // the upstream: it carries an array `content`, not the empty
            // string that used to erase it here.
            if content.is_empty() {
                return;
            }
            messages.push(content.into_message(match role {
                "assistant" => Role::Assistant,
                "system" | "developer" => Role::System,
                _ => Role::User,
            }));
        }
    }
}

/// Append an OpenAI-shape tool call, folding it into the immediately
/// preceding assistant message: parallel `function_call` items land in one
/// `tool_calls` array, and the calls a model made right after its text go
/// back on that same turn (content + `tool_calls`) — the shape the model
/// produced them in, which the Responses history splits into a `message`
/// item followed by `function_call` items. A replayed reasoning item's own
/// assistant message takes the calls that follow it the same way.
fn push_tool_call(messages: &mut Vec<ChatMessage>, tc: Value) {
    if let Some(last) = messages.last_mut() {
        if matches!(last.role, Role::Assistant) {
            match last.extra.get_mut("tool_calls") {
                Some(Value::Array(arr)) => arr.push(tc),
                _ => {
                    last.extra
                        .insert("tool_calls".to_string(), Value::Array(vec![tc]));
                }
            }
            return;
        }
    }
    let mut extra = Map::new();
    extra.insert("tool_calls".to_string(), Value::Array(vec![tc]));
    messages.push(ChatMessage {
        role: Role::Assistant,
        content: None,
        content_blocks: None,
        name: None,
        tool_call_id: None,
        extra,
    });
}

/// The chat message field a replayed reasoning item travels in — the one
/// the OpenAI-compatible reasoning models (DeepSeek, GLM, Qwen, Kimi, …)
/// read their own earlier chain-of-thought from on an assistant turn.
const REASONING_CONTENT: &str = "reasoning_content";

/// The plaintext of a replayed `reasoning` input item: its `content` parts
/// when they carry any text, otherwise its `summary` parts; each part
/// trimmed and the non-blank ones joined by newlines. `encrypted_content`
/// is never read — it is another provider's ciphertext and means nothing to
/// the upstream this request is going to.
fn replayed_reasoning_text(item: &Value) -> Option<String> {
    let parts_text = |parts: &Value, skip_opaque: bool| -> Option<String> {
        let parts = parts.as_array()?;
        let texts: Vec<&str> = parts
            .iter()
            .filter(|p| {
                !skip_opaque
                    || !matches!(
                        p.get("type").and_then(Value::as_str),
                        Some("encrypted_content" | "redacted_thinking")
                    )
            })
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect();
        (!texts.is_empty()).then(|| texts.join("\n"))
    };
    // Parts arrays only, the shape the API defines: they are also the only
    // shape the input mask rewrites, so a bare-string `content` would reach
    // the upstream past a Mask rule that reported a hit on it.
    item.get("content")
        .and_then(|c| parts_text(c, true))
        .or_else(|| item.get("summary").and_then(|s| parts_text(s, false)))
}

/// Fold each reasoning-only assistant message onto the assistant message
/// that follows it, so the chain-of-thought rides the turn that carries the
/// answer or the tool calls it led to — the placement the OpenAI-compatible
/// reasoning models expect, some of which reject a multi-turn request whose
/// assistant turns lost their reasoning. Consecutive reasoning items join,
/// in order, ahead of any reasoning the target already carries.
///
/// Reasoning that no assistant message follows (the next turn is the user's,
/// a tool result, or the end of the input) stays a message of its own so it
/// is still passed back rather than dropped.
fn attach_replayed_reasoning(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    fn flush_standalone(pending: &mut Vec<String>, out: &mut Vec<ChatMessage>) {
        for text in pending.drain(..) {
            let mut extra = Map::new();
            extra.insert(REASONING_CONTENT.to_string(), Value::String(text));
            out.push(ChatMessage {
                role: Role::Assistant,
                content: None,
                content_blocks: None,
                name: None,
                tool_call_id: None,
                extra,
            });
        }
    }

    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    let mut pending: Vec<String> = Vec::new();
    for mut m in messages {
        if m.is_reasoning_only() {
            if let Some(Value::String(text)) = m.extra.remove(REASONING_CONTENT) {
                pending.push(text);
            }
            continue;
        }
        if !pending.is_empty() {
            if matches!(m.role, Role::Assistant) {
                let existing = m
                    .extra
                    .get(REASONING_CONTENT)
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
                pending.extend(existing);
                m.extra.insert(
                    REASONING_CONTENT.to_string(),
                    Value::String(pending.join("\n")),
                );
                pending.clear();
            } else {
                flush_standalone(&mut pending, &mut out);
            }
        }
        out.push(m);
    }
    flush_standalone(&mut pending, &mut out);
    out
}

/// A Responses-API content slot rendered for a chat message: the
/// concatenated text of its text parts, plus the OpenAI chat content-block
/// array when the slot carried anything a chat message can only express as
/// blocks (an image, a file, audio).
///
/// `blocks` stays `None` for a text-only slot so the common case keeps the
/// bare-string wire shape it has always had; when it is `Some`, the
/// OpenAI-compatible bridge forwards the array verbatim and the bridges that
/// don't speak blocks (Anthropic / Gemini / Bedrock) fall back to `text` —
/// the documented cross-provider content limitation.
struct ChatContent {
    text: String,
    blocks: Option<Vec<Value>>,
}

impl ChatContent {
    fn into_message(self, role: Role) -> ChatMessage {
        ChatMessage {
            role,
            content: Some(self.text),
            content_blocks: self.blocks,
            name: None,
            tool_call_id: None,
            extra: Map::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.text.is_empty() && self.blocks.is_none()
    }
}

/// Translate a Responses-API content slot (a bare string, or an array of
/// typed input parts) into chat-completions content.
///
/// Part mapping, the OpenAI chat shape for each:
///   * `input_text` / `output_text` / `text` → `{type:"text", text}`
///   * `input_image` → `{type:"image_url", image_url:{url, detail?}}`, the
///     `image_url` passed through as given (an https URL or a `data:` URL)
///   * `input_file` → `{type:"file", file:{file_data?, filename?, file_id?}}`
///   * `input_audio` → `{type:"input_audio", input_audio:{data, format}}`
///
/// Parts that carry none of the above are skipped.
fn responses_content_to_chat(v: &Value) -> ChatContent {
    match v {
        Value::String(s) => ChatContent {
            text: s.clone(),
            blocks: None,
        },
        Value::Array(parts) => {
            let mut text = String::new();
            let mut blocks: Vec<Value> = Vec::new();
            let mut has_non_text = false;
            for part in parts {
                // A bare string element is text, as it is at the top level.
                if let Some(s) = part.as_str() {
                    text.push_str(s);
                    blocks.push(json!({"type": "text", "text": s}));
                    continue;
                }
                match part.get("type").and_then(Value::as_str) {
                    Some("input_image") => {
                        if let Some(block) = input_image_block(part) {
                            blocks.push(block);
                            has_non_text = true;
                        }
                    }
                    Some("input_file") => {
                        if let Some(block) = input_file_block(part) {
                            blocks.push(block);
                            has_non_text = true;
                        }
                    }
                    Some("input_audio") => {
                        if let Some(block) = input_audio_block(part) {
                            blocks.push(block);
                            has_non_text = true;
                        }
                    }
                    // `input_text` / `output_text` / `text`, and any other
                    // part that carries a `text` member.
                    _ => {
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                            blocks.push(json!({"type": "text", "text": t}));
                        }
                    }
                }
            }
            ChatContent {
                text,
                // Text-only slots keep the bare-string shape.
                blocks: has_non_text.then_some(blocks),
            }
        }
        _ => ChatContent {
            text: String::new(),
            blocks: None,
        },
    }
}

/// `input_image` → the chat `image_url` part. `detail` rides along only
/// when the caller set it, so an upstream applies its own default.
///
/// An `input_image` that carries only a `file_id` (an image uploaded to
/// OpenAI's Files API) has no chat-completions equivalent — the chat part
/// addresses an image by URL or `data:` URL and nothing else — so it maps
/// to no block at all rather than to an `image_url` with an empty `url`,
/// which every chat upstream rejects.
fn input_image_block(part: &Value) -> Option<Value> {
    let url = part
        .get("image_url")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?;
    let mut image_url = Map::new();
    image_url.insert("url".to_string(), json!(url));
    if let Some(detail) = part.get("detail").and_then(Value::as_str) {
        image_url.insert("detail".to_string(), json!(detail));
    }
    Some(json!({"type": "image_url", "image_url": Value::Object(image_url)}))
}

/// `input_file` → the chat `file` part, carrying whichever of
/// `file_data` / `filename` / `file_id` the caller sent.
fn input_file_block(part: &Value) -> Option<Value> {
    let mut file = Map::new();
    for key in ["file_data", "filename", "file_id"] {
        if let Some(v) = part.get(key).filter(|v| !v.is_null()) {
            file.insert(key.to_string(), v.clone());
        }
    }
    (!file.is_empty()).then(|| json!({"type": "file", "file": Value::Object(file)}))
}

/// `input_audio` → the chat `input_audio` part (`data` + `format`).
fn input_audio_block(part: &Value) -> Option<Value> {
    // The Responses part nests the pair under `input_audio`; tolerate the
    // flattened spelling some clients send.
    let src = part.get("input_audio").unwrap_or(part);
    let mut audio = Map::new();
    for key in ["data", "format"] {
        if let Some(v) = src.get(key).filter(|v| !v.is_null()) {
            audio.insert(key.to_string(), v.clone());
        }
    }
    (!audio.is_empty()).then(|| json!({"type": "input_audio", "input_audio": Value::Object(audio)}))
}

/// A `function_call_output.output` rendered as chat `tool` content.
///
/// The chat `tool` role is text-only, so the output is always a plain
/// string: OpenAI rejects a `tool` message carrying an `image_url` part
/// outright ("Image URLs are only allowed for messages with role 'user'"),
/// and the bridges that do not speak content blocks (Anthropic, Gemini,
/// Bedrock) read the concatenated text anyway — so forwarding the image
/// would turn a tool result that used to answer into a 400 without any
/// upstream gaining the image. Non-text parts are dropped and their text
/// siblings still reach the model.
fn function_call_output_to_chat(output: &Value) -> ChatContent {
    // A tool that returned JSON reaches the upstream as that JSON
    // serialised — a chat `tool` message carries a string, and rendering
    // the value as an empty one erased the result. An array is ambiguous:
    // it is the Responses content-part shape when its elements are parts,
    // and a plain JSON array (a list of records, say) otherwise, which
    // would parse as parts and come out empty. An array that is both
    // keeps its parts as text and serialises the rest in place, so no
    // element the tool returned is silently dropped. `null` and an
    // absent output stay the empty string.
    match output {
        Value::Object(_) | Value::Number(_) | Value::Bool(_) => {
            return ChatContent {
                text: serde_json::to_string(output).unwrap_or_default(),
                blocks: None,
            }
        }
        // An array holding no content part at all is one JSON value —
        // a list of records, say — and is serialised whole.
        Value::Array(items) if !items.is_empty() && !items.iter().any(is_content_part) => {
            return ChatContent {
                text: serde_json::to_string(output).unwrap_or_default(),
                blocks: None,
            }
        }
        // A mixed array is rendered element by element: every element
        // the model would otherwise never see arrives as its own JSON,
        // in the position the tool put it in.
        Value::Array(items) => {
            let mut text = String::new();
            for item in items {
                if is_content_part(item) {
                    if let Some(s) = item.as_str() {
                        text.push_str(s);
                    } else if let Some(t) = item.get("text").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                    // A typed non-text part (an image, a file, audio) has
                    // no text and no `tool`-role counterpart; it is the
                    // one thing this role cannot carry.
                } else {
                    text.push_str(&serde_json::to_string(item).unwrap_or_default());
                }
            }
            return ChatContent { text, blocks: None };
        }
        _ => {}
    }
    let mut content = responses_content_to_chat(output);
    content.blocks = None;
    content
}

/// Whether one array element is a Responses content part rather than a
/// member of a plain JSON array: a bare string, a typed part this bridge
/// maps, or anything carrying a `text` member.
fn is_content_part(item: &Value) -> bool {
    if item.is_string() {
        return true;
    }
    if item.get("text").is_some_and(Value::is_string) {
        return true;
    }
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("input_text" | "output_text" | "text" | "input_image" | "input_file" | "input_audio")
    )
}

/// Translate the Responses `text.format` object into the chat
/// `response_format` object:
///   * `{type:"json_schema", name, schema, strict, description}` →
///     `{type:"json_schema", json_schema:{name, schema, strict, description}}`
///     (members the caller omitted stay omitted)
///   * `{type:"json_object"}` → `{type:"json_object"}`
///   * `{type:"text"}`, anything else → `None`, so the field stays off the wire
///
/// `text.verbosity` has no chat-completions counterpart on this path and
/// keeps being dropped.
fn responses_text_format_to_response_format(text: &Value) -> Option<Value> {
    let format = text.get("format")?;
    match format.get("type").and_then(Value::as_str)? {
        "json_schema" => {
            let mut schema = Map::new();
            for key in ["name", "schema", "strict", "description"] {
                if let Some(v) = format.get(key).filter(|v| !v.is_null()) {
                    schema.insert(key.to_string(), v.clone());
                }
            }
            Some(json!({"type": "json_schema", "json_schema": Value::Object(schema)}))
        }
        "json_object" => Some(json!({"type": "json_object"})),
        _ => None,
    }
}

/// Translate Responses-API `tools` into OpenAI chat tools.
///
///   * `{type:"function", name, description, parameters}` →
///     `{type:"function", function:{name, description, parameters}}`
///   * `{type:"custom", name, description, format}` → a function tool with
///     the single-string schema in [`custom_tool_parameters`]; a freeform
///     tool has no chat counterpart, and a function tool taking one string
///     is the shape that keeps the model able to call it. A grammar under
///     `format.definition` rides along in the description, the only place a
///     chat upstream will read it.
///   * `{type:"namespace", name, description, tools:[…]}` → one function
///     tool per `function` sub-tool, named `<namespace>__<tool>` (see
///     [`namespace_chat_tool_name`]) and described by the namespace's
///     description followed by the sub-tool's own. A sub-tool whose
///     flattened name is already a top-level function's is left out, so
///     the model is never offered two tools under one name.
///   * hosted tools (`web_search*`, `file_search`, `code_interpreter`,
///     `mcp`, `computer_use*`, `image_generation`, …) have no chat
///     equivalent and are dropped.
///
/// Returns `None` when nothing translates so the field stays absent from
/// the wire.
fn responses_tools_to_chat(tools: &Value) -> Option<Value> {
    let arr = tools.as_array()?;
    let top_level_functions = top_level_function_names(arr);
    let out: Vec<Value> = arr
        .iter()
        .flat_map(|t| -> Vec<Value> {
            match t.get("type").and_then(|v| v.as_str()) {
                Some("function") => function_chat_tool(t, None).into_iter().collect(),
                Some("namespace") => namespace_members(t)
                    .filter(|(chat_name, _, _)| !top_level_functions.contains(chat_name.as_str()))
                    .filter_map(|(chat_name, member, description)| {
                        function_chat_tool(member, Some((chat_name, description)))
                    })
                    .collect(),
                Some("custom") => custom_chat_tool(t).into_iter().collect(),
                _ => Vec::new(),
            }
        })
        .collect();
    (!out.is_empty()).then_some(Value::Array(out))
}

/// A Responses `function` tool as a chat function tool. `rename` carries
/// the flattened name and combined description of a namespace sub-tool.
fn function_chat_tool(t: &Value, rename: Option<(String, Option<String>)>) -> Option<Value> {
    let name = t.get("name").and_then(|v| v.as_str())?;
    let mut func = Map::new();
    match rename {
        Some((chat_name, description)) => {
            func.insert("name".to_string(), json!(chat_name));
            if let Some(d) = description {
                func.insert("description".to_string(), json!(d));
            }
        }
        None => {
            func.insert("name".to_string(), json!(name));
            if let Some(d) = t.get("description") {
                func.insert("description".to_string(), d.clone());
            }
        }
    }
    if let Some(p) = t.get("parameters") {
        func.insert("parameters".to_string(), p.clone());
    }
    Some(json!({"type": "function", "function": Value::Object(func)}))
}

/// The chat function name a namespace sub-tool travels under: chat tools
/// are one flat list, so the namespace is folded into the name. Both
/// directions of the translation derive it here — the request side names
/// the tool and re-names replayed calls with it, the reply side maps a call
/// back through [`ResponsesReplyContext`].
fn namespace_chat_tool_name(namespace: &str, tool: &str) -> String {
    format!("{namespace}__{tool}")
}

/// Names of the top-level `function` tools a request declared.
fn top_level_function_names(tools: &[Value]) -> std::collections::BTreeSet<&str> {
    tools
        .iter()
        .filter(|t| t.get("type").and_then(Value::as_str) == Some("function"))
        .filter_map(|t| t.get("name").and_then(Value::as_str))
        .collect()
}

/// The `function` sub-tools of one `namespace` tool, each as
/// `(flattened chat name, sub-tool, description)`. The description is the
/// namespace's followed by the sub-tool's, so the model still reads what
/// the group is for; either alone when the other is missing.
fn namespace_members(t: &Value) -> impl Iterator<Item = (String, &Value, Option<String>)> {
    let namespace = t.get("name").and_then(Value::as_str).unwrap_or_default();
    let namespace_description = t
        .get("description")
        .and_then(Value::as_str)
        .filter(|d| !d.is_empty());
    t.get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(move |_| !namespace.is_empty())
        .filter(|m| m.get("type").and_then(Value::as_str) == Some("function"))
        .filter_map(move |m| {
            let name = m.get("name").and_then(Value::as_str)?;
            let own = m
                .get("description")
                .and_then(Value::as_str)
                .filter(|d| !d.is_empty());
            let description = match (namespace_description, own) {
                (Some(ns), Some(own)) => Some(format!("{ns}\n\n{own}")),
                (Some(ns), None) => Some(ns.to_string()),
                (None, own) => own.map(str::to_string),
            };
            Some((namespace_chat_tool_name(namespace, name), m, description))
        })
}

/// A Responses `custom` tool as a chat function tool taking one string.
fn custom_chat_tool(t: &Value) -> Option<Value> {
    let name = t.get("name").and_then(|v| v.as_str())?;
    let mut description = t
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    description.push_str(&custom_tool_grammar_suffix(t.get("format")));
    let mut func = Map::new();
    func.insert("name".to_string(), json!(name));
    if !description.is_empty() {
        func.insert("description".to_string(), json!(description));
    }
    func.insert("parameters".to_string(), custom_tool_parameters(name));
    Some(json!({"type": "function", "function": Value::Object(func)}))
}

/// The single string parameter a `custom` tool takes once it has been
/// translated into a function tool. Both directions of the translation
/// read this one constant — the request side wraps the freeform input in
/// it, the reply side unwraps it back out — so the pair cannot drift.
const CUSTOM_TOOL_INPUT_PARAM: &str = "content";

/// The JSON-schema a `custom` tool takes once it is a function tool: one
/// required string holding whatever the freeform tool would have received.
fn custom_tool_parameters(name: &str) -> Value {
    json!({
        "type": "object",
        "properties": {
            CUSTOM_TOOL_INPUT_PARAM: {
                "type": "string",
                "description": format!("The {name} content following the specified format"),
            }
        },
        "required": [CUSTOM_TOOL_INPUT_PARAM],
    })
}

/// A custom tool's freeform input, wrapped as the function `arguments`
/// string the single-parameter schema above describes.
fn wrap_custom_tool_input(input: &str) -> String {
    json!({ CUSTOM_TOOL_INPUT_PARAM: input }).to_string()
}

/// The freeform `input` of a custom tool call, unwrapped from the function
/// `arguments` the model produced against that schema.
///
/// A model that did not follow the schema — `arguments` that are not JSON,
/// or JSON without the parameter as a string — has its raw argument string
/// forwarded instead. That is the text the caller's freeform tool was going
/// to receive either way, and dropping it would lose the call's whole
/// payload.
fn unwrap_custom_tool_input(arguments: &str) -> String {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .as_ref()
        .and_then(Value::as_object)
        // Exactly the one member the wrapper has. A model that answered
        // with its freeform payload verbatim may itself have produced a
        // JSON object carrying a `content` field beside others — reading
        // that as the wrapper would deliver the inner string and silently
        // drop the rest, which is corruption rather than a fallback. An
        // object that IS exactly `{"content": "…"}` stays ambiguous and is
        // unwrapped; nothing on the wire can separate the two.
        .filter(|o| o.len() == 1)
        .and_then(|o| o.get(CUSTOM_TOOL_INPUT_PARAM))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| arguments.to_string())
}

/// What the reply translators need to know about the Responses request
/// they are answering.
///
/// The request side turns `custom` tools and namespace sub-tools into
/// ordinary chat function tools ([`responses_tools_to_chat`]), so a reply
/// arrives as a chat tool call carrying nothing that says which Responses
/// tool it came from; the caller is waiting for a `custom_tool_call` item,
/// or a `function_call` naming the sub-tool and its `namespace`, so only the
/// request's own tool list can tell them apart. Every Response object also
/// echoes the request's own settings ([`response_echo_fields`]).
#[derive(Debug, Clone)]
pub struct ResponsesReplyContext {
    custom_tools: std::collections::BTreeSet<String>,
    /// Chat function name → `(namespace, sub-tool name)`.
    namespace_tools: std::collections::BTreeMap<String, (String, String)>,
    echo: Map<String, Value>,
}

/// How one chat tool call is returned to a Responses caller.
enum ReplyToolCall<'a> {
    /// A `custom_tool_call` item.
    Custom,
    /// A `function_call` item, naming a namespace sub-tool when `namespace`
    /// is set.
    Function {
        name: &'a str,
        namespace: Option<&'a str>,
    },
}

/// A request that declared no tools and set nothing: every Response field
/// at its default.
impl Default for ResponsesReplyContext {
    fn default() -> Self {
        Self::from_request(&Value::Null)
    }
}

impl ResponsesReplyContext {
    pub fn from_request(body: &Value) -> Self {
        let tools: &[Value] = body
            .get("tools")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let custom_tools = tools
            .iter()
            .filter(|t| t.get("type").and_then(Value::as_str) == Some("custom"))
            .filter_map(|t| t.get("name").and_then(Value::as_str))
            .map(str::to_string)
            .collect();
        Self {
            custom_tools,
            namespace_tools: namespace_tool_map(tools),
            echo: response_echo_fields(body),
        }
    }

    fn tool_call<'a>(&'a self, chat_name: &'a str) -> ReplyToolCall<'a> {
        if self.custom_tools.contains(chat_name) {
            return ReplyToolCall::Custom;
        }
        match self.namespace_tools.get(chat_name) {
            Some((namespace, name)) => ReplyToolCall::Function {
                name,
                namespace: Some(namespace),
            },
            None => ReplyToolCall::Function {
                name: chat_name,
                namespace: None,
            },
        }
    }
}

/// The chat function names a model may call a namespace sub-tool by, each
/// mapped to `(namespace, sub-tool name)`: the flattened name it was
/// offered under, and — when no top-level function and no other namespace
/// uses it — the sub-tool's bare name, which a model sometimes answers with
/// instead. A flattened name that collides with a top-level function was
/// never offered (see [`responses_tools_to_chat`]) and maps to nothing, so
/// a call to that function stays a plain `function_call`.
fn namespace_tool_map(tools: &[Value]) -> std::collections::BTreeMap<String, (String, String)> {
    let top_level = top_level_function_names(tools);
    let members: Vec<(String, String)> = tools
        .iter()
        .filter(|t| t.get("type").and_then(Value::as_str) == Some("namespace"))
        .flat_map(|t| {
            let namespace = t.get("name").and_then(Value::as_str).unwrap_or_default();
            namespace_members(t).filter_map(move |(_, member, _)| {
                let name = member.get("name").and_then(Value::as_str)?;
                Some((namespace.to_string(), name.to_string()))
            })
        })
        .collect();
    let mut map = std::collections::BTreeMap::new();
    for (namespace, name) in &members {
        let chat_name = namespace_chat_tool_name(namespace, name);
        if !top_level.contains(chat_name.as_str()) {
            map.insert(chat_name, (namespace.clone(), name.clone()));
        }
    }
    for (namespace, name) in &members {
        let unique = members.iter().filter(|(_, n)| n == name).count() == 1;
        if unique && !top_level.contains(name.as_str()) {
            map.insert(name.clone(), (namespace.clone(), name.clone()));
        }
    }
    map
}

/// A custom tool's grammar, rendered for the tail of its description. Empty
/// when the tool carries no `format.definition`.
fn custom_tool_grammar_suffix(format: Option<&Value>) -> String {
    let Some(definition) = format
        .and_then(|f| f.get("definition"))
        .and_then(Value::as_str)
        .filter(|d| !d.is_empty())
    else {
        return String::new();
    };
    let syntax = format
        .and_then(|f| f.get("syntax"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!("\n\nFormat:\n```{syntax}\n{definition}\n```")
}

/// Translate Responses-API `tool_choice` to the provider-neutral OpenAI
/// chat shape every provider bridge translates onwards:
///
///   * `"auto"` / `"none"` / `"required"` pass through
///   * `{type:"function", name}`, `{type:"custom", name}`,
///     `{type:"tool", name}` → `{type:"function", function:{name}}`
///   * `{type:"allowed_tools", mode:"required"|"auto"}` → the bare mode;
///     chat has no way to restrict the model to a subset of the tools it
///     was given, so the subset itself is dropped
///   * `{type:"any"}` → `"required"`
///   * a choice naming a hosted tool type, or anything else → `None`, so
///     the field stays off the wire
fn responses_tool_choice_to_chat(tc: &Value) -> Option<Value> {
    match tc {
        Value::String(s) => Some(Value::String(s.clone())),
        Value::Object(o) => match o.get("type").and_then(|v| v.as_str())? {
            "function" | "custom" | "tool" => {
                let name = o.get("name").and_then(|v| v.as_str())?;
                // A namespace sub-tool was offered under its flattened name.
                let name = match o
                    .get("namespace")
                    .and_then(Value::as_str)
                    .filter(|ns| !ns.is_empty())
                {
                    Some(namespace) => namespace_chat_tool_name(namespace, name),
                    None => name.to_string(),
                };
                Some(json!({"type": "function", "function": {"name": name}}))
            }
            "any" => Some(Value::String("required".to_string())),
            "allowed_tools" => match o.get("mode").and_then(|v| v.as_str())? {
                mode @ ("auto" | "required") => {
                    tracing::debug!(
                        %mode,
                        "narrowing allowed_tools to its mode: the chat bridge cannot restrict the model to a subset of the tools"
                    );
                    Some(Value::String(mode.to_string()))
                }
                _ => None,
            },
            other => {
                tracing::debug!(
                    tool_choice = %other,
                    "dropping tool_choice on the chat bridge: no chat equivalent"
                );
                None
            }
        },
        _ => None,
    }
}

/// Build the non-streaming Responses-API response object from a bridge
/// [`ChatResponse`]. `requested_model` is echoed back (not the upstream
/// id). `created_at` is a unix timestamp stamped by the caller. `reply`
/// carries what the request declared (see [`ResponsesReplyContext`]).
pub fn chat_response_to_responses_json(
    resp: &ChatResponse,
    requested_model: &str,
    created_at: i64,
    reply: &ResponsesReplyContext,
) -> Value {
    let (status, incomplete_reason) = responses_status(&resp.finish_reason);
    let output = build_output_items(
        message_reasoning_text(&resp.message),
        resp.message.content.as_deref(),
        resp.message
            .extra
            .get("tool_calls")
            .and_then(|v| v.as_array()),
        reply,
    );
    response_resource(
        reply,
        ResponseState {
            id: &format!("resp_{}", Uuid::new_v4().simple()),
            created_at,
            model: requested_model,
            status,
            output: Value::Array(output),
            usage: responses_usage_json(&resp.usage),
            incomplete_reason,
            error: None,
        },
    )
}

/// The per-response half of a Response object; the request half comes
/// from [`ResponsesReplyContext`].
struct ResponseState<'a> {
    id: &'a str,
    created_at: i64,
    model: &'a str,
    status: &'a str,
    output: Value,
    /// `null` until the response has finished, and for a failed one.
    usage: Value,
    incomplete_reason: Option<&'static str>,
    /// `(code, message)` of a failed response.
    error: Option<(&'a str, &'a str)>,
}

/// A complete Response object — the full top-level field set the Responses
/// API defines, on every object the bridge emits (the non-streaming body
/// and the `response` of every lifecycle event), so a client that
/// validates the object against the API's schema accepts it.
fn response_resource(reply: &ResponsesReplyContext, state: ResponseState<'_>) -> Value {
    let mut obj = reply.echo.clone();
    obj.insert("id".to_string(), json!(state.id));
    obj.insert("object".to_string(), json!("response"));
    obj.insert("created_at".to_string(), json!(state.created_at));
    obj.insert(
        "completed_at".to_string(),
        if state.status == "completed" {
            json!(chrono::Utc::now().timestamp())
        } else {
            Value::Null
        },
    );
    obj.insert("status".to_string(), json!(state.status));
    obj.insert("model".to_string(), json!(state.model));
    obj.insert("output".to_string(), state.output);
    obj.insert("usage".to_string(), state.usage);
    obj.insert(
        "incomplete_details".to_string(),
        state
            .incomplete_reason
            .map_or(Value::Null, |reason| json!({"reason": reason})),
    );
    obj.insert(
        "error".to_string(),
        state.error.map_or(
            Value::Null,
            |(code, message)| json!({"code": code, "message": message}),
        ),
    );
    Value::Object(obj)
}

/// The top-level Response fields that describe the request rather than the
/// generation: the caller's own value where it sent one, the API's default
/// (or `null`) where it did not.
fn response_echo_fields(body: &Value) -> Map<String, Value> {
    let get = |key: &str| body.get(key).filter(|v| !v.is_null());
    let string_or_null = |key: &str| {
        get(key)
            .filter(|v| v.is_string())
            .cloned()
            .unwrap_or(Value::Null)
    };
    let int_or_null = |key: &str| {
        get(key)
            .filter(|v| v.is_u64())
            .cloned()
            .unwrap_or(Value::Null)
    };
    let number_or = |key: &str, default: Value| {
        get(key)
            .filter(|v| v.is_number())
            .cloned()
            .unwrap_or(default)
    };
    let bool_or = |key: &str, default: bool| {
        get(key)
            .filter(|v| v.is_boolean())
            .cloned()
            .unwrap_or(json!(default))
    };
    let string_or = |key: &str, default: &str| {
        get(key)
            .filter(|v| v.is_string())
            .cloned()
            .unwrap_or(json!(default))
    };

    let mut m = Map::new();
    m.insert("instructions".into(), string_or_null("instructions"));
    m.insert(
        "previous_response_id".into(),
        string_or_null("previous_response_id"),
    );
    m.insert(
        "tools".into(),
        Value::Array(
            get("tools")
                .and_then(Value::as_array)
                .map(|tools| tools.iter().map(echo_tool).collect())
                .unwrap_or_default(),
        ),
    );
    m.insert(
        "tool_choice".into(),
        get("tool_choice").cloned().unwrap_or(json!("auto")),
    );
    m.insert("truncation".into(), string_or("truncation", "disabled"));
    m.insert(
        "parallel_tool_calls".into(),
        bool_or("parallel_tool_calls", true),
    );
    m.insert("text".into(), echo_text(get("text")));
    m.insert("temperature".into(), number_or("temperature", json!(1.0)));
    m.insert("top_p".into(), number_or("top_p", json!(1.0)));
    m.insert(
        "presence_penalty".into(),
        number_or("presence_penalty", json!(0.0)),
    );
    m.insert(
        "frequency_penalty".into(),
        number_or("frequency_penalty", json!(0.0)),
    );
    m.insert(
        "top_logprobs".into(),
        get("top_logprobs")
            .filter(|v| v.is_u64())
            .cloned()
            .unwrap_or(json!(0)),
    );
    m.insert(
        "reasoning".into(),
        get("reasoning")
            .filter(|v| v.is_object())
            .map_or(Value::Null, |r| {
                let field = |key: &str| r.get(key).cloned().unwrap_or(Value::Null);
                json!({"effort": field("effort"), "summary": field("summary")})
            }),
    );
    m.insert("max_output_tokens".into(), int_or_null("max_output_tokens"));
    m.insert("max_tool_calls".into(), int_or_null("max_tool_calls"));
    m.insert("store".into(), bool_or("store", true));
    m.insert("background".into(), bool_or("background", false));
    m.insert("service_tier".into(), string_or("service_tier", "default"));
    m.insert(
        "metadata".into(),
        get("metadata")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or(json!({})),
    );
    m.insert(
        "safety_identifier".into(),
        string_or_null("safety_identifier"),
    );
    m.insert(
        "prompt_cache_key".into(),
        string_or_null("prompt_cache_key"),
    );
    if let Some(user) = get("user").filter(|v| v.is_string()) {
        m.insert("user".into(), user.clone());
    }
    m
}

/// One request tool as the Response object reports it. A `function` tool
/// carries every member the API defines for it, `null` where the caller
/// left one out; any other tool is echoed as sent.
fn echo_tool(tool: &Value) -> Value {
    let mut tool = tool.clone();
    if let Some(obj) = tool.as_object_mut() {
        if obj.get("type").and_then(Value::as_str) == Some("function") {
            for key in ["description", "parameters", "strict"] {
                obj.entry(key).or_insert(Value::Null);
            }
        }
    }
    tool
}

/// The request's `text` setting, with the plain-text format filled in when
/// the caller named none.
fn echo_text(text: Option<&Value>) -> Value {
    let mut text = text
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    if let Some(obj) = text.as_object_mut() {
        if obj.get("format").is_none_or(Value::is_null) {
            obj.insert("format".to_string(), json!({"type": "text"}));
        }
    }
    text
}

/// Map an internal finish reason to a Responses-API `status` plus optional
/// `incomplete_details.reason`.
fn responses_status(fr: &FinishReason) -> (&'static str, Option<&'static str>) {
    match fr {
        FinishReason::Length => ("incomplete", Some("max_output_tokens")),
        FinishReason::ContentFilter => ("incomplete", Some("content_filter")),
        _ => ("completed", None),
    }
}

/// A completed `reasoning` output item. The chain-of-thought rides a
/// `summary_text` part — the slot the Responses API defines for the
/// human-readable reasoning a client is allowed to render (its `content`
/// parts are the provider's own opaque/`reasoning_text` material, which a
/// chat upstream does not give us).
fn reasoning_item_json(item_id: &str, text: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": item_id,
        "status": "completed",
        "summary": [{"type": "summary_text", "text": text}],
    })
}

/// The upstream's chain-of-thought on a bridged non-streaming response.
///
/// The OpenAI-compatible response parser already normalises both spellings
/// the ecosystem uses — `message.reasoning_content` (DeepSeek / GLM / Qwen /
/// vLLM / SGLang) and `message.reasoning` (aggregators) — into this one
/// canonical slot on the way into [`ChatMessage`], so the bridge reads the
/// slot rather than re-deriving the spellings here.
fn message_reasoning_text(message: &ChatMessage) -> Option<&str> {
    message
        .extra
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// Assemble the `output` array: a `reasoning` item carrying the upstream's
/// chain-of-thought (when any), then a `message` item carrying the assistant
/// text (when any), followed by one tool-call item per tool call —
/// `custom_tool_call` for a call naming one of the request's `custom`
/// tools, `function_call` for everything else (see
/// [`ResponsesReplyContext`]).
fn build_output_items(
    reasoning: Option<&str>,
    text: Option<&str>,
    tool_calls: Option<&Vec<Value>>,
    reply: &ResponsesReplyContext,
) -> Vec<Value> {
    let mut output: Vec<Value> = Vec::new();
    // Reasoning leads the output array, as it does on a native Responses
    // upstream: a client renders the items in order, and the thinking that
    // produced an answer belongs before it.
    if let Some(reasoning) = reasoning {
        output.push(reasoning_item_json(
            &format!("rs_{}", Uuid::new_v4().simple()),
            reasoning,
        ));
    }
    if let Some(text) = text.filter(|s| !s.is_empty()) {
        output.push(json!({
            "type": "message",
            "id": format!("msg_{}", Uuid::new_v4().simple()),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }));
    }
    if let Some(tool_calls) = tool_calls {
        for tc in tool_calls {
            let call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or_default();
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or_default();
            let arguments = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("");
            output.push(match reply.tool_call(name) {
                ReplyToolCall::Custom => json!({
                    "type": "custom_tool_call",
                    "id": format!("ctc_{}", Uuid::new_v4().simple()),
                    "call_id": call_id,
                    "name": name,
                    "input": unwrap_custom_tool_input(arguments),
                    "status": "completed",
                }),
                ReplyToolCall::Function { name, namespace } => function_call_item(
                    &format!("fc_{}", Uuid::new_v4().simple()),
                    call_id,
                    name,
                    namespace,
                    arguments,
                    "completed",
                ),
            });
        }
    }
    output
}

/// A `function_call` output item. `namespace` rides along only for a
/// namespace sub-tool: the caller dispatches on the `{name, namespace}`
/// pair.
fn function_call_item(
    id: &str,
    call_id: &str,
    name: &str,
    namespace: Option<&str>,
    arguments: &str,
    status: &str,
) -> Value {
    let mut item = json!({
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    });
    if let Some(namespace) = namespace {
        item["namespace"] = json!(namespace);
    }
    item
}

/// Render usage in the Responses-API shape, which is OpenAI accounting:
/// `input_tokens` is the FULL input and `input_tokens_details.cached_tokens`
/// is a subset of it. Both are projected from [`UsageStats`] rather than
/// copied — an Anthropic-shape upstream keeps its cache counters beside
/// `prompt_tokens`, so copying produced an `input_tokens` that excluded the
/// cache while `cached_tokens` reported it, i.e. the self-contradictory
/// `cached_tokens > input_tokens` (AISIX-Cloud#1447).
fn responses_usage_json(u: &UsageStats) -> Value {
    let mut input_details = json!({"cached_tokens": u.openai_cached_tokens()});
    if let Some(cache_write) = u.cache_write_tokens {
        input_details["cache_write_tokens"] = cache_write.into();
    }
    // Preserve the additive Anthropic counter separately from OpenAI's
    // raw cache_write_tokens value. Their accounting is different.
    let cache_creation = u.anthropic_cache_creation_input_tokens();
    if cache_creation > 0 {
        input_details["cache_creation_tokens"] = cache_creation.into();
    }
    json!({
        "input_tokens": u.openai_prompt_tokens(),
        "input_tokens_details": input_details,
        "output_tokens": u.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": u.reasoning_tokens},
        "total_tokens": u.openai_total_tokens(),
    })
}

// ─────────────────────────────────────────────────────────────────────
// Streaming SSE encoder — internal ChatChunk stream → Responses-API
// SSE events.
//
// Event order for a text response:
//   response.created → response.in_progress
//   → response.output_item.added (message)
//   → response.content_part.added (output_text)
//   → response.output_text.delta ×N
//   → response.output_text.done → response.content_part.done
//   → response.output_item.done (message)
//   → response.completed
//
// Tool calls add, per call:
//   response.output_item.added (function_call)
//   → response.function_call_arguments.delta ×N
//   → response.function_call_arguments.done
//   → response.output_item.done (function_call)
//
// A chat upstream that streams its chain-of-thought (`delta
// .reasoning_content`) adds a `reasoning` item ahead of whatever it was
// reasoning towards:
//   response.output_item.added (reasoning)
//   → response.reasoning_summary_part.added
//   → response.reasoning_summary_text.delta ×N
//   → response.reasoning_summary_text.done
//   → response.reasoning_summary_part.done
//   → response.output_item.done (reasoning)
// It is closed by the first content/tool-call delta that follows (or by the
// finish), so the message / function_call item that follows opens at the
// NEXT output_index. Reasoning that arrives after a message item is already
// open opens a further reasoning item rather than reopening the closed one.
//
// `response.completed` carries the final output + usage. When an
// OpenAI-compatible upstream sends its usage frame AFTER the finish chunk
// (`stream_options.include_usage`), the completed event is withheld until
// the usage arrives (or `force_finish`), so token counts aren't zeroed.
//
// Reference: https://platform.openai.com/docs/api-reference/responses-streaming
// ─────────────────────────────────────────────────────────────────────

/// One Responses-API SSE event, written as `event: {type}\ndata: {json}\n\n`.
#[derive(Debug, Clone)]
pub struct ResponsesSseEvent {
    pub event_type: &'static str,
    pub data: Value,
}

impl ResponsesSseEvent {
    pub fn to_sse_string(&self) -> String {
        format!(
            "event: {}\ndata: {}\n\n",
            self.event_type,
            serde_json::to_string(&self.data).expect("serde_json::Value always serializes"),
        )
    }
}

/// Per-reasoning-item streaming state. One `reasoning` output item and the
/// single `summary_text` part it streams into.
#[derive(Debug)]
struct ReasoningState {
    item_id: String,
    output_index: u32,
    text: String,
}

/// Per-tool-call streaming state.
#[derive(Debug)]
struct ToolCallState {
    /// Minted when the call opens, before its name is known — so the item
    /// id is assembled from it once the kind is (see [`Self::item_id`]).
    item_uuid: String,
    call_id: String,
    name: String,
    /// The call names one of the request's `custom` tools, so it streams as
    /// a `custom_tool_call` item rather than a `function_call` one.
    custom: bool,
    /// `(namespace, sub-tool name)` when the call names a flattened
    /// namespace sub-tool. Fixed when the item is announced, like `custom`.
    namespace: Option<(String, String)>,
    output_index: u32,
    arguments: String,
    item_added: bool,
}

impl ToolCallState {
    /// `fc_…` for a function call, `ctc_…` for a custom tool call — the two
    /// item-id prefixes the Responses API uses for the two item types.
    fn item_id(&self) -> String {
        let prefix = if self.custom { "ctc" } else { "fc" };
        format!("{prefix}_{}", self.item_uuid)
    }

    /// The completed output item for this call.
    fn done_item(&self) -> Value {
        if self.custom {
            json!({
                "type": "custom_tool_call",
                "id": self.item_id(),
                "call_id": self.call_id,
                "name": self.name,
                "input": unwrap_custom_tool_input(&self.arguments),
                "status": "completed",
            })
        } else {
            self.function_call_item(&self.arguments, "completed")
        }
    }

    /// This call as a `function_call` item, under the sub-tool name and
    /// its namespace when it names a namespace sub-tool.
    fn function_call_item(&self, arguments: &str, status: &str) -> Value {
        let (name, namespace) = match &self.namespace {
            Some((namespace, name)) => (name.as_str(), Some(namespace.as_str())),
            None => (self.name.as_str(), None),
        };
        function_call_item(
            &self.item_id(),
            &self.call_id,
            name,
            namespace,
            arguments,
            status,
        )
    }
}

/// State machine re-encoding a `ChatChunk` stream as Responses-API SSE.
#[derive(Debug)]
pub struct ResponsesSseEncoder {
    response_id: String,
    model_display_name: String,
    created_at: i64,
    sequence_number: u64,
    sent_created: bool,
    finished: bool,
    /// Next output-item index to assign (shared by the message + tool items).
    next_output_index: u32,
    // Text message item.
    text_item_id: Option<String>,
    text_output_index: u32,
    text_accum: String,
    /// Set once the per-item `*.done` events have been emitted, so
    /// `close_items` is idempotent across the finish chunk + `force_finish`.
    items_closed: bool,
    /// The reasoning item currently streaming, if any.
    reasoning_open: Option<ReasoningState>,
    /// Reasoning items already closed, kept so `response.completed` can
    /// rebuild them into the final `output` array.
    reasoning_done: Vec<ReasoningState>,
    // Tool-call items keyed by the OpenAI delta index.
    tool_calls: std::collections::BTreeMap<u64, ToolCallState>,
    /// What the request declared (see [`ResponsesReplyContext`]).
    reply: ResponsesReplyContext,
    /// Serialized size of the request echo every Response object carries.
    echo_len: usize,
    /// Withheld terminal status + incomplete reason while waiting on a
    /// trailing usage frame.
    pending_status: Option<&'static str>,
    pending_reason: Option<&'static str>,
    // Accumulated usage (max semantics, robust to double-emit).
    usage_seen: bool,
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
    cache_write_tokens: Option<u32>,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
}

impl ResponsesSseEncoder {
    /// `reply` carries what the request declared (see
    /// [`ResponsesReplyContext`]).
    pub fn new(
        response_id: impl Into<String>,
        model_display_name: impl Into<String>,
        created_at: i64,
        reply: ResponsesReplyContext,
    ) -> Self {
        Self {
            echo_len: serde_json::to_string(&reply.echo).map_or(0, |s| s.len()),
            reply,
            response_id: response_id.into(),
            model_display_name: model_display_name.into(),
            created_at,
            sequence_number: 0,
            sent_created: false,
            finished: false,
            next_output_index: 0,
            text_item_id: None,
            text_output_index: 0,
            text_accum: String::new(),
            items_closed: false,
            reasoning_open: None,
            reasoning_done: Vec::new(),
            tool_calls: std::collections::BTreeMap::new(),
            pending_status: None,
            pending_reason: None,
            usage_seen: false,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            cache_write_tokens: None,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
        }
    }

    /// Take the next `sequence_number` for an event this relay emits
    /// outside the state machine — the terminal `error` frames. They are
    /// part of the same numbered event stream a client is reading, so they
    /// must continue its numbering rather than restart or repeat it.
    pub fn take_sequence_number(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    /// `response.created` then `response.in_progress`, each carrying the
    /// Response object with `status: "in_progress"`: the two events every
    /// stream opens with, a failed one included.
    fn opening_events(&mut self) -> [ResponsesSseEvent; 2] {
        let response = self.response_object("in_progress", false, None, None);
        [
            self.event("response.created", json!({"response": response.clone()})),
            self.event("response.in_progress", json!({"response": response})),
        ]
    }

    /// Build one event, stamping `type` + `sequence_number`.
    fn event(&mut self, event_type: &'static str, mut data: Value) -> ResponsesSseEvent {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        if let Value::Object(map) = &mut data {
            map.insert("type".to_string(), json!(event_type));
            map.insert("sequence_number".to_string(), json!(seq));
        }
        ResponsesSseEvent { event_type, data }
    }

    fn accumulate_usage(&mut self, chunk: &ChatChunk) {
        if let Some(u) = chunk.usage.as_ref() {
            self.usage_seen = true;
            self.prompt_tokens = self.prompt_tokens.max(u.prompt_tokens);
            self.completion_tokens = self.completion_tokens.max(u.completion_tokens);
            self.total_tokens = self.total_tokens.max(u.total_tokens);
            self.reasoning_tokens = self.reasoning_tokens.max(u.reasoning_tokens);
            self.cached_prompt_tokens = self.cached_prompt_tokens.max(u.cached_prompt_tokens);
            self.cache_write_tokens = self.cache_write_tokens.max(u.cache_write_tokens);
            self.cache_creation_tokens = self.cache_creation_tokens.max(u.cache_creation_tokens);
            self.cache_read_tokens = self.cache_read_tokens.max(u.cache_read_tokens);
        }
    }

    /// Adopt locally-estimated token counts as the client-visible usage,
    /// for a bridged stream whose upstream left them unreported. The
    /// internal usage record is filled from the same estimate, and a client
    /// reading `response.completed.usage` must not be told zero while the
    /// record says otherwise (AISIX-Cloud#1074). Per counter, and only
    /// into a zero: a number the upstream actually reported is never
    /// overridden, and a frame that reported one counter and left the
    /// other at zero still gets that zero filled — the record fills it
    /// the same way, and the two must not disagree. A no-op once the
    /// terminal event has gone out: the client must never be handed
    /// numbers contradicting what it was already sent.
    pub fn set_estimated_usage(&mut self, prompt_tokens: u32, completion_tokens: u32) {
        if self.finished {
            return;
        }
        let mut filled = false;
        if self.prompt_tokens == 0 && prompt_tokens > 0 {
            self.prompt_tokens = prompt_tokens;
            filled = true;
        }
        if self.completion_tokens == 0 && completion_tokens > 0 {
            self.completion_tokens = completion_tokens;
            filled = true;
        }
        if filled {
            // A total the upstream reported beside a zero sub-counter no
            // longer describes what the client is about to be told.
            // Zeroing it makes the projection derive prompt + completion,
            // the same arithmetic it uses when no total was reported.
            self.total_tokens = 0;
        }
    }

    fn usage_value(&self) -> Value {
        // Same projection as the non-streaming exit, so a stream and a
        // buffered call over the same upstream report identical usage.
        responses_usage_json(&UsageStats {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            cached_prompt_tokens: self.cached_prompt_tokens,
            cache_write_tokens: self.cache_write_tokens,
            reasoning_tokens: self.reasoning_tokens,
            cache_creation_tokens: self.cache_creation_tokens,
            cache_read_tokens: self.cache_read_tokens,
            ..Default::default()
        })
    }

    /// The assembled assistant output for an end-of-stream output guardrail
    /// scan: the full accumulated text plus the fully-reassembled tool calls
    /// in canonical OpenAI `{id, type, function:{name, arguments}}` shape (so
    /// an argument literal split across chunks is scanned as one string, not
    /// as disjoint fragments).
    pub fn assembled_assistant_message(&self) -> (String, Vec<Value>) {
        let mut tool_calls: Vec<(u32, Value)> = self
            .tool_calls
            .values()
            .map(|tc| {
                (
                    tc.output_index,
                    json!({
                        "id": tc.call_id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }),
                )
            })
            .collect();
        tool_calls.sort_by_key(|(idx, _)| *idx);
        (
            self.text_accum.clone(),
            tool_calls.into_iter().map(|(_, v)| v).collect(),
        )
    }

    /// The Response object embedded in a lifecycle event. `output` and
    /// `usage` are filled only on a terminal event that reports the
    /// generation (`completed` / `incomplete`); a failed response carries
    /// neither.
    fn response_object(
        &self,
        status: &str,
        with_output_and_usage: bool,
        incomplete_reason: Option<&'static str>,
        error: Option<(&str, &str)>,
    ) -> Value {
        response_resource(
            &self.reply,
            ResponseState {
                id: &self.response_id,
                created_at: self.created_at,
                model: &self.model_display_name,
                status,
                output: if with_output_and_usage {
                    Value::Array(self.final_output_items())
                } else {
                    json!([])
                },
                usage: if with_output_and_usage {
                    self.usage_value()
                } else {
                    Value::Null
                },
                incomplete_reason,
                error,
            },
        )
    }

    /// Rebuild the completed `output` array from accumulated state.
    fn final_output_items(&self) -> Vec<Value> {
        let mut items: Vec<(u32, Value)> = Vec::new();
        for r in self.reasoning_done.iter().chain(self.reasoning_open.iter()) {
            items.push((r.output_index, reasoning_item_json(&r.item_id, &r.text)));
        }
        if let Some(id) = self.text_item_id.as_ref() {
            items.push((
                self.text_output_index,
                json!({
                    "type": "message",
                    "id": id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": self.text_accum, "annotations": []}],
                }),
            ));
        }
        for tc in self.tool_calls.values() {
            items.push((tc.output_index, tc.done_item()));
        }
        items.sort_by_key(|(idx, _)| *idx);
        items.into_iter().map(|(_, v)| v).collect()
    }

    /// Translate one chunk into the SSE events to emit (possibly empty).
    pub fn next_events(&mut self, chunk: &ChatChunk) -> Vec<ResponsesSseEvent> {
        if self.finished {
            return Vec::new();
        }
        self.accumulate_usage(chunk);

        // Terminal status withheld for a trailing usage frame: release it
        // once usage lands. Post-finish chunks carry no renderable content.
        if let Some(status) = self.pending_status {
            if self.usage_seen {
                self.pending_status = None;
                let reason = self.pending_reason.take();
                return vec![self.completed_event(status, reason)];
            }
            return Vec::new();
        }

        let has_content = chunk
            .delta
            .content
            .as_deref()
            .is_some_and(|s| !s.is_empty());
        let has_tools = chunk
            .delta
            .tool_calls
            .as_ref()
            .is_some_and(|v| !v.is_empty());
        let has_reasoning = chunk
            .delta
            .reasoning_content
            .as_deref()
            .is_some_and(|s| !s.is_empty());
        let has_finish = chunk.finish_reason.is_some();

        let mut events = Vec::new();

        if !self.sent_created && (has_content || has_tools || has_reasoning || has_finish) {
            self.sent_created = true;
            events.extend(self.opening_events());
        }

        // ── Reasoning ──
        //
        // Emitted before the text/tool blocks below so a chunk carrying both
        // reasoning and content renders the thinking first, then closes the
        // reasoning item and opens the message item after it.
        if has_reasoning {
            let delta = chunk.delta.reasoning_content.clone().unwrap_or_default();
            if self.reasoning_open.is_none() {
                let item_id = format!("rs_{}", Uuid::new_v4().simple());
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                self.reasoning_open = Some(ReasoningState {
                    item_id: item_id.clone(),
                    output_index,
                    text: String::new(),
                });
                events.push(self.event(
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": {"type": "reasoning", "id": item_id, "status": "in_progress", "summary": []},
                    }),
                ));
                events.push(self.event(
                    "response.reasoning_summary_part.added",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""},
                    }),
                ));
            }
            let (item_id, output_index) = {
                let r = self.reasoning_open.as_mut().expect("just opened");
                r.text.push_str(&delta);
                (r.item_id.clone(), r.output_index)
            };
            events.push(self.event(
                "response.reasoning_summary_text.delta",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "delta": delta,
                }),
            ));
        }

        // The first content or tool-call delta after a reasoning run ends it,
        // so the item that follows opens at the next `output_index`.
        if has_content || has_tools {
            events.extend(self.close_reasoning());
        }

        // ── Text content ──
        if has_content {
            let delta = chunk.delta.content.clone().unwrap_or_default();
            if self.text_item_id.is_none() {
                let item_id = format!("msg_{}", Uuid::new_v4().simple());
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                self.text_item_id = Some(item_id.clone());
                self.text_output_index = output_index;
                events.push(self.event(
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": {"type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": []},
                    }),
                ));
                events.push(self.event(
                    "response.content_part.added",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }),
                ));
            }
            let item_id = self.text_item_id.clone().unwrap_or_default();
            let output_index = self.text_output_index;
            self.text_accum.push_str(&delta);
            events.push(self.event(
                "response.output_text.delta",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": delta,
                }),
            ));
        }

        // ── Tool calls ──
        if let Some(tool_calls) = chunk.delta.tool_calls.as_ref() {
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

                if !self.tool_calls.contains_key(&oai_index) {
                    let output_index = self.next_output_index;
                    self.next_output_index += 1;
                    self.tool_calls.insert(
                        oai_index,
                        ToolCallState {
                            item_uuid: Uuid::new_v4().simple().to_string(),
                            call_id: String::new(),
                            name: String::new(),
                            custom: false,
                            namespace: None,
                            output_index,
                            arguments: String::new(),
                            item_added: false,
                        },
                    );
                }
                let (custom, namespace) = match self.reply.tool_call(name) {
                    ReplyToolCall::Custom => (true, None),
                    ReplyToolCall::Function {
                        name: sub_name,
                        namespace: Some(namespace),
                    } => (false, Some((namespace.to_string(), sub_name.to_string()))),
                    ReplyToolCall::Function { .. } => (false, None),
                };
                let state = self.tool_calls.get_mut(&oai_index).expect("just inserted");
                if !id.is_empty() {
                    state.call_id = id.to_string();
                }
                if !name.is_empty() {
                    state.name = name.to_string();
                    // Only until the item has been announced: `.added`
                    // carries both the item id and the item type, and both
                    // are derived from this flag. An upstream that splits
                    // `function.name` across chunks overwrites the name on
                    // each one, and letting a later fragment flip the flag
                    // would leave `.done` disagreeing with the `.added`
                    // the client already read.
                    if !state.item_added {
                        state.custom = custom;
                        state.namespace = namespace;
                    }
                }

                // Emit output_item.added once the call id + name are known.
                if !state.item_added && !state.call_id.is_empty() && !state.name.is_empty() {
                    state.item_added = true;
                    let output_index = state.output_index;
                    let item = if state.custom {
                        json!({"type": "custom_tool_call", "id": state.item_id(), "call_id": state.call_id, "name": state.name, "input": "", "status": "in_progress"})
                    } else {
                        state.function_call_item("", "in_progress")
                    };
                    events.push(self.event(
                        "response.output_item.added",
                        json!({"output_index": output_index, "item": item}),
                    ));
                }

                if !arguments.is_empty() {
                    let state = self.tool_calls.get_mut(&oai_index).expect("present");
                    state.arguments.push_str(arguments);
                    // A custom tool's fragments are the function-call
                    // wrapper's JSON, not the freeform input the caller
                    // asked for; they are buffered and emitted as one
                    // unwrapped `custom_tool_call_input.delta` at the close.
                    // Streaming the wrapper through would hand the client
                    // pieces of `{"content":"…"}` under an event type whose
                    // payload is supposed to be the input itself.
                    if state.item_added && !state.custom {
                        let (item_id, output_index) = (state.item_id(), state.output_index);
                        events.push(self.event(
                            "response.function_call_arguments.delta",
                            json!({
                                "item_id": item_id,
                                "output_index": output_index,
                                "delta": arguments,
                            }),
                        ));
                    }
                }
            }
        }

        // ── Finish ──
        if let Some(fr) = chunk.finish_reason.as_ref() {
            events.extend(self.close_items());
            let (status, reason) = responses_status(fr);
            if self.usage_seen {
                events.push(self.completed_event(status, reason));
            } else {
                // Hold response.completed until the trailing usage frame.
                self.pending_status = Some(status);
                self.pending_reason = reason;
            }
        }

        events
    }

    /// Close the open `reasoning` item, if any: the summary text, then its
    /// part, then the item. Empty when no reasoning item is open, so every
    /// call site can invoke it unconditionally.
    fn close_reasoning(&mut self) -> Vec<ResponsesSseEvent> {
        let Some(r) = self.reasoning_open.take() else {
            return Vec::new();
        };
        let (item_id, output_index, text) = (r.item_id.clone(), r.output_index, r.text.clone());
        self.reasoning_done.push(r);
        vec![
            self.event(
                "response.reasoning_summary_text.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "text": text,
                }),
            ),
            self.event(
                "response.reasoning_summary_part.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": text},
                }),
            ),
            self.event(
                "response.output_item.done",
                json!({
                    "output_index": output_index,
                    "item": reasoning_item_json(&item_id, &text),
                }),
            ),
        ]
    }

    /// Emit the per-item `*.done` closing events for the open text + tool
    /// items. Idempotent: a no-op after the first call, so the finish chunk
    /// and a later `force_finish` (when the completed event was withheld for
    /// usage) don't double-emit the done events.
    fn close_items(&mut self) -> Vec<ResponsesSseEvent> {
        if self.items_closed {
            return Vec::new();
        }
        self.items_closed = true;
        // A stream that ended inside its reasoning run (nothing but thinking,
        // or a truncation) still owes the item's closing events.
        let mut events = self.close_reasoning();
        if self.text_item_id.is_some() {
            let item_id = self.text_item_id.clone().unwrap_or_default();
            let output_index = self.text_output_index;
            let text = self.text_accum.clone();
            events.push(self.event(
                "response.output_text.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                }),
            ));
            events.push(self.event(
                "response.content_part.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []},
                }),
            ));
            events.push(self.event(
                "response.output_item.done",
                json!({
                    "output_index": output_index,
                    "item": {"type": "message", "id": item_id, "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]},
                }),
            ));
        }
        let pending: Vec<u64> = self
            .tool_calls
            .iter()
            .filter(|(_, s)| s.item_added)
            .map(|(k, _)| *k)
            .collect();
        for k in pending {
            let (item_id, arguments, output_index, custom, done_item) = {
                let s = self.tool_calls.get(&k).expect("present");
                (
                    s.item_id(),
                    s.arguments.clone(),
                    s.output_index,
                    s.custom,
                    s.done_item(),
                )
            };
            if custom {
                // The buffered fragments become exactly one delta carrying
                // the unwrapped input, then the done event — a custom tool
                // never emits `response.function_call_arguments.*`.
                let input = unwrap_custom_tool_input(&arguments);
                events.push(self.event(
                    "response.custom_tool_call_input.delta",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": input,
                    }),
                ));
                events.push(self.event(
                    "response.custom_tool_call_input.done",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "input": input,
                    }),
                ));
            } else {
                events.push(self.event(
                    "response.function_call_arguments.done",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "arguments": arguments,
                    }),
                ));
            }
            events.push(self.event(
                "response.output_item.done",
                json!({"output_index": output_index, "item": done_item}),
            ));
        }
        events
    }

    fn completed_event(&mut self, status: &str, reason: Option<&'static str>) -> ResponsesSseEvent {
        self.finished = true;
        let event_type = if status == "completed" {
            "response.completed"
        } else {
            "response.incomplete"
        };
        let response = self.response_object(status, true, reason, None);
        self.event(event_type, json!({"response": response}))
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// What one encoded event costs against an output guardrail's
    /// hold-back budget. A lifecycle event repeats the request's own
    /// settings (`instructions`, `tools`, …); those are bounded by the
    /// request, not by what the model generated, and an agent client's run
    /// to tens of kilobytes — so they are not charged, or every guarded
    /// stream would fail closed on a fraction of the output it used to.
    pub fn buffer_cost(&self, event: &ResponsesSseEvent, encoded_len: usize) -> usize {
        if event.data.get("response").is_some() {
            encoded_len.saturating_sub(self.echo_len)
        } else {
            encoded_len
        }
    }

    /// Whether any chunk so far carried something a response is made of —
    /// content, reasoning, a tool call, or a finish reason. A stream that
    /// ends before one did produced no response at all: a bare `[DONE]`, or
    /// a usage-only frame, is not an answer.
    pub fn has_output(&self) -> bool {
        self.sent_created
    }

    /// The upstream sent its finish reason and the terminal event is being
    /// held for the trailing usage frame.
    pub fn awaiting_usage(&self) -> bool {
        !self.finished && self.pending_status.is_some()
    }

    /// Flush a clean close when the upstream stream ended without a finish
    /// chunk, or while the completed event was withheld for usage. Only for
    /// a stream that produced output (see [`Self::has_output`]); one that
    /// did not has failed rather than finished.
    pub fn force_finish(&mut self) -> Vec<ResponsesSseEvent> {
        if self.finished {
            return Vec::new();
        }
        let status = self.pending_status.take().unwrap_or("completed");
        let reason = self.pending_reason.take();
        let mut events = self.close_items();
        events.push(self.completed_event(status, reason));
        events
    }

    /// The terminal `response.failed` event: the Response object with
    /// `status: "failed"`, the `error` that ended it, no output and no
    /// usage. Whether a terminal event already reached the client is the
    /// relay's call, not this encoder's: under a held-back output
    /// guardrail the encoder has produced `response.completed` long before
    /// anything is released.
    pub fn failed_event(&mut self, code: &str, message: &str) -> ResponsesSseEvent {
        self.finished = true;
        let response = self.response_object("failed", false, None, Some((code, message)));
        self.event("response.failed", json!({"response": response}))
    }
}

/// End-of-stream telemetry captured by [`build_responses_bridge_stream`].
#[derive(Default, Debug)]
pub struct ResponsesStreamCompletion {
    /// `true` once the upstream stream reached EOF, i.e. the response was
    /// received in full. Stays `false` when the consumer went away first —
    /// the generator is then dropped at a suspension point and the tail
    /// never runs — which the telemetry closure reports as `499`.
    pub reached_end: bool,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub reasoning_tokens: u32,
    pub cached_prompt_tokens: u32,
    pub cache_write_tokens: Option<u32>,
    pub cache_creation_tokens: u32,
    pub cache_read_tokens: u32,
    pub finish_reason: String,
    /// Response object `id` reported by the **bridged upstream** — i.e. the
    /// chat-completion id the provider sent, not the `resp_…` this encoder
    /// mints for the client. The minted one is a gateway value and would be
    /// useless in the provider's console (AISIX-Cloud#1289).
    pub provider_request_id: String,
    /// Attempt-scoped time to the upstream's first generated chunk.
    pub upstream_ttft_ms: u32,
    /// Request-scoped time until the caller got its first response bytes.
    /// Trails `upstream_ttft_ms` by any hold-back guardrail scan.
    pub downstream_latency_ms: u32,
    /// Set when an output guardrail blocked the streamed response (a content
    /// block or a fail-closed buffer overflow). The upstream still billed, so
    /// the usage event carries the tokens but is marked blocked — matching
    /// the non-streaming path so the dashboard's Blocked tab + budget ledger
    /// see it.
    pub guardrail_blocked: bool,
    /// The upstream failure that ended the stream — a mid-stream error, or
    /// a stream that carried no response. The usage event reports it
    /// instead of a `200`.
    pub failure: Option<crate::attempt::StreamFailure>,
    /// Per-detector PII mask counts applied to the held stream at release
    /// (#932). Merged with the input-side counts by the on_complete emit.
    pub redacted_entity_counts: crate::redact::RedactionCounts,
    /// Monitor-mode guardrail observations made by the end-of-stream output
    /// check (AISIX-Cloud#562). Merged with the input-side hits by the
    /// on_complete emit.
    pub monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    /// Assembled assistant text for content-capturing exporters
    /// (AISIX-Cloud#947), accumulated across chunks ONLY when an exporter
    /// wants full content (bounded to the capture cap). Empty otherwise.
    /// Read by the on_complete telemetry closure; never reaches the CP sink.
    pub response_text: String,
    /// True when the Drop guard filled any token counter from the local
    /// estimator (AISIX-Cloud#1074).
    pub usage_estimated: bool,
    /// Generated output (content + reasoning + tool-call text) accumulated
    /// for the token-estimation fallback (AISIX-Cloud#1074). Always on,
    /// bounded to `token_estimate::OUTPUT_ACCUMULATION_CAP`; never leaves
    /// the process.
    est_output_text: String,
}

struct CompleteOnDrop<F: FnOnce(ResponsesStreamCompletion)> {
    slot: Option<(F, ResponsesStreamCompletion)>,
    /// Token-estimation fallback (AISIX-Cloud#1074); fills counters the
    /// upstream never reported before `on_complete` runs.
    estimator: Option<crate::token_estimate::Estimator>,
}

impl<F: FnOnce(ResponsesStreamCompletion)> CompleteOnDrop<F> {
    fn comp(&mut self) -> &mut ResponsesStreamCompletion {
        &mut self
            .slot
            .as_mut()
            .expect("stream completion guard accessed after drop")
            .1
    }
}

impl<F: FnOnce(ResponsesStreamCompletion)> Drop for CompleteOnDrop<F> {
    fn drop(&mut self) {
        if let Some((f, mut comp)) = self.slot.take() {
            // Token-estimation fallback (AISIX-Cloud#1074): fill the
            // counters the upstream never reported from the request +
            // the accumulated output text.
            if let Some(est) = self.estimator.take() {
                let filled = crate::token_estimate::fill_missing(
                    &est,
                    comp.prompt_tokens,
                    comp.completion_tokens,
                    Some(comp.est_output_text.as_str()),
                );
                if filled.estimated {
                    comp.prompt_tokens = filled.prompt_tokens;
                    comp.completion_tokens = filled.completion_tokens;
                    comp.usage_estimated = true;
                }
            }
            f(comp);
        }
    }
}

/// Wrap a bridge [`ChatChunkStream`] as a Responses-API SSE body, encoding
/// each chunk via [`ResponsesSseEncoder`]. An end-of-stream telemetry
/// callback fires from a Drop guard (so it runs on normal end and on client
/// disconnect).
///
/// When `output_guardrail` is `Some` and `hold_back` is true (the chain's
/// resolved streaming policy holds back — any block-capable output chain),
/// the encoded SSE is **held back** and released only after the assembled
/// assistant output passes the scan — mirroring the verbatim `/v1/responses`
/// path's secure BufferFull default (#719), so a configured output block
/// can't be bypassed by streaming a non-OpenAI model. The scan reads the
/// fully-reassembled text + tool calls (not raw deltas), and the buffer is
/// capped — an output guardrail must never release content it couldn't fully
/// buffer to scan, so an overflow fails closed. When `hold_back` is false
/// (EndOfStreamCheck — a monitor-only chain, which can never block), the
/// bytes forward live and the same end-of-stream scan runs for observation
/// only (AISIX-Cloud#1010). With no output guardrail the bytes forward live
/// unscanned.
#[allow(clippy::too_many_arguments)]
pub fn build_responses_bridge_stream(
    upstream: ChatChunkStream,
    encoder: ResponsesSseEncoder,
    // Request clock — what the CALLER waited for.
    started: Instant,
    // Attempt clock — how the UPSTREAM behaved on this call.
    attempt_started: Instant,
    output_guardrail: Option<Arc<aisix_guardrails::GuardrailChain>>,
    hold_back: bool,
    max_buffer_bytes: usize,
    model_label: String,
    // Largest content cap any content-capturing exporter wants
    // (AISIX-Cloud#947); `None` skips response-text accumulation entirely.
    content_cap: Option<u32>,
    // Token-estimation fallback context (AISIX-Cloud#1074); see
    // `CompleteOnDrop::estimator`.
    estimator: Option<crate::token_estimate::Estimator>,
    on_complete: impl FnOnce(ResponsesStreamCompletion) + Send + 'static,
) -> axum::body::Body {
    use futures::StreamExt;

    let mut encoder = encoder;
    let stream = async_stream::stream! {
        let mut guard = CompleteOnDrop {
            slot: Some((on_complete, ResponsesStreamCompletion::default())),
            estimator,
        };
        // Whether any event has left for the client yet; a failure that
        // would be the first one opens the stream itself.
        let mut sent_downstream = false;
        // Stamped on the first bytes that actually leave for the client —
        // under hold-back that is the release, not the upstream chunk.
        macro_rules! downstream_mark {
            () => {
                // Dead after the final release, which no failure follows.
                #[allow(unused_assignments)]
                {
                    sent_downstream = true;
                }
                if guard.comp().downstream_latency_ms == 0 {
                    guard.comp().downstream_latency_ms =
                        started.elapsed().as_millis().min(u32::MAX as u128) as u32;
                }
            };
        }
        let mut upstream = upstream;
        let mut first_chunk_seen = false;
        let buffering = output_guardrail.is_some() && hold_back;
        // Held SSE events when an output guardrail is attached; empty (and
        // unused) on the live-forward path.
        let mut held: Vec<bytes::Bytes> = Vec::new();
        let mut held_bytes = 0usize;
        let mut overflowed = false;
        while let Some(item) = upstream.next().await {
            match item {
                Ok(chunk) => {
                    // First upstream chunk of ANY type stops the TTFT clock —
                    // the industry convention (LiteLLM, caller-side gateways),
                    // so the figure matches external observers
                    // (AISIX-Cloud#1225).
                    if !first_chunk_seen {
                        first_chunk_seen = true;
                        guard.comp().upstream_ttft_ms =
                            attempt_started.elapsed().as_millis().min(u32::MAX as u128) as u32;
                    }
                    {
                        let comp = guard.comp();
                        if !chunk.id.is_empty() {
                            comp.provider_request_id =
                                crate::usage_attr::sanitize_provider_response_id(&chunk.id);
                        }
                        if let Some(fr) = chunk.finish_reason.as_ref() {
                            comp.finish_reason = finish_reason_label(fr);
                        }
                        // Content capture (AISIX-Cloud#947): assemble the
                        // assistant text for the observability fan-out,
                        // bounded to the cap so a long stream can't grow the
                        // buffer without limit. Only when an exporter wants
                        // full content — mirrors chat.rs's stream capture.
                        if let (Some(cap), Some(text)) =
                            (content_cap, chunk.delta.content.as_deref())
                        {
                            if comp.response_text.len() < cap as usize {
                                comp.response_text.push_str(text);
                            }
                        }
                        // Token-estimation accumulator (AISIX-Cloud#1074):
                        // all generated output, always on (whether the
                        // fallback is needed is only known at end-of-stream),
                        // bounded.
                        {
                            use crate::token_estimate::push_capped;
                            if let Some(text) = chunk.delta.content.as_deref() {
                                push_capped(&mut comp.est_output_text, text);
                            }
                            if let Some(text) = chunk.delta.reasoning_content.as_deref() {
                                push_capped(&mut comp.est_output_text, text);
                            }
                            if let Some(tcs) = chunk.delta.tool_calls.as_ref() {
                                for tc in tcs {
                                    if let Some(f) = tc.get("function") {
                                        if let Some(n) =
                                            f.get("name").and_then(|v| v.as_str())
                                        {
                                            push_capped(&mut comp.est_output_text, n);
                                        }
                                        if let Some(a) =
                                            f.get("arguments").and_then(|v| v.as_str())
                                        {
                                            push_capped(&mut comp.est_output_text, a);
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(u) = chunk.usage.as_ref() {
                            comp.prompt_tokens = comp.prompt_tokens.max(u.prompt_tokens);
                            comp.completion_tokens = comp.completion_tokens.max(u.completion_tokens);
                            comp.reasoning_tokens = comp.reasoning_tokens.max(u.reasoning_tokens);
                            comp.cached_prompt_tokens = comp.cached_prompt_tokens.max(u.cached_prompt_tokens);
                            comp.cache_write_tokens = comp.cache_write_tokens.max(u.cache_write_tokens);
                            comp.cache_creation_tokens = comp.cache_creation_tokens.max(u.cache_creation_tokens);
                            comp.cache_read_tokens = comp.cache_read_tokens.max(u.cache_read_tokens);
                        }
                    }
                    for ev in encoder.next_events(&chunk) {
                        let b = bytes::Bytes::from(ev.to_sse_string());
                        if buffering {
                            held_bytes += encoder.buffer_cost(&ev, b.len());
                            if held_bytes > max_buffer_bytes {
                                overflowed = true;
                                break;
                            }
                            held.push(b);
                        } else {
                            downstream_mark!();
                            yield Ok::<_, std::io::Error>(b);
                        }
                    }
                    if overflowed || encoder.is_finished() {
                        break;
                    }
                }
                Err(e) => {
                    // The upstream had already finished its answer and the
                    // connection dropped before the usage frame or `[DONE]`:
                    // nothing the caller asked for is missing, so the
                    // response completes, on estimated usage if need be.
                    if matches!(e, aisix_gateway::BridgeError::Transport(_))
                        && encoder.awaiting_usage()
                    {
                        break;
                    }
                    let message = e.to_string();
                    crate::attempt::StreamFailure::record(&mut guard.comp().failure, &e);
                    // The loop stops at the terminal event, so none went out.
                    yield Ok(failure_frames(
                        &mut encoder,
                        sent_downstream,
                        false,
                        e.error_type(),
                        upstream_failed_code(&e),
                        &message,
                    ));
                    return;
                }
            }
        }

        // The upstream closed a stream that never carried a single piece of
        // a response. Completing it would hand the caller an empty answer
        // as a success; failing it with a retryable code lets the client
        // retry the turn. Accounted exactly like a mid-stream upstream
        // failure: the usage record comes from the Drop guard.
        if !encoder.is_finished() && !encoder.has_output() {
            tracing::warn!(
                model = %model_label,
                "streaming /v1/responses (cross-provider) upstream returned an empty stream",
            );
            // Recorded as what the same empty stream is before the headers
            // go out — an aborted stream — with its own message.
            guard.comp().failure = Some(crate::attempt::StreamFailure {
                error_message: EMPTY_STREAM_MESSAGE.to_string(),
                ..crate::attempt::StreamFailure::from_bridge(
                    &aisix_gateway::BridgeError::StreamAborted,
                )
            });
            yield Ok(failure_frames(
                &mut encoder,
                sent_downstream,
                false,
                EMPTY_STREAM_CODE,
                EMPTY_STREAM_CODE,
                EMPTY_STREAM_MESSAGE,
            ));
            return;
        }
        // Token-estimation fallback (AISIX-Cloud#1074), run HERE rather than
        // from the Drop guard below: the terminal `response.completed` this
        // relay is about to synthesize carries the client-visible usage, and
        // it must be the same number the usage record gets — a client told
        // `output_tokens: 0` for a response it can see the text of has no way
        // to reconcile that with the dashboard. The guard keeps its own copy
        // of this fill for the stream a consumer abandoned before EOF, where
        // no terminal event is emitted at all.
        if let Some(est) = guard.estimator.take() {
            let filled = {
                let comp = guard.comp();
                crate::token_estimate::fill_missing(
                    &est,
                    comp.prompt_tokens,
                    comp.completion_tokens,
                    Some(comp.est_output_text.as_str()),
                )
            };
            if filled.estimated {
                let comp = guard.comp();
                comp.prompt_tokens = filled.prompt_tokens;
                comp.completion_tokens = filled.completion_tokens;
                comp.usage_estimated = true;
                encoder.set_estimated_usage(filled.prompt_tokens, filled.completion_tokens);
            }
        }

        if !encoder.is_finished() {
            for ev in encoder.force_finish() {
                let b = bytes::Bytes::from(ev.to_sse_string());
                if buffering {
                    held_bytes += encoder.buffer_cost(&ev, b.len());
                    if held_bytes > max_buffer_bytes {
                        overflowed = true;
                        break;
                    }
                    held.push(b);
                } else {
                    downstream_mark!();
                    yield Ok(b);
                }
            }
        }

        // Upstream EOF — the response was received in full. Record it before
        // the guardrail work below, which awaits a remote provider and is a
        // routine drop point for clients that close on the terminal frame.
        guard.comp().reached_end = true;

        // No output-hook guardrail: nothing to scan.
        let Some(chain) = output_guardrail.as_ref() else { return; };

        // Buffer overflow (hold-back mode only): an output guardrail must
        // not release content it couldn't fully buffer to scan — fail
        // closed (#719).
        if overflowed {
            tracing::warn!(
                guardrail_hook = "output",
                model = %model_label,
                max_buffer_bytes,
                "streaming /v1/responses (cross-provider) output exceeded buffer cap; failing closed",
            );
            guard.comp().guardrail_blocked = true;
            yield Ok(guardrail_failure_frames(
                &mut encoder,
                sent_downstream,
                // Only a held-back stream can overflow: nothing was sent.
                false,
                None,
                Some(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED),
            ));
            return;
        }

        // End-of-stream output guardrail (#719): scan the fully-reassembled
        // assistant output (canonical tool calls, so a literal split across
        // argument deltas can't slip through), then release or block. On the
        // live-forward path (EndOfStreamCheck — monitor-only chain,
        // AISIX-Cloud#1010) the same scan runs for observation: the bytes are
        // already on the wire, so a Block is signalled with a trailing error
        // frame rather than withheld bytes, mirroring the chat surface's
        // EndOfStreamCheck behavior.
        let (text, tool_calls) = encoder.assembled_assistant_message();
        if !text.is_empty() || !tool_calls.is_empty() {
            // Live mode releases oversized streams (that's the point of
            // AISIX-Cloud#1010), so the assembled text is unbounded here —
            // cap the scan input like the verbatim path's EosOutputScan
            // does, keeping the observation provider calls bounded. Held
            // (buffering) text is already capped by the hold-back budget.
            let text = if buffering {
                text
            } else {
                let mut text = text;
                let mut end = text
                    .len()
                    .min(aisix_guardrails::DEFAULT_STREAM_OUTPUT_BUFFER_BYTES);
                while end > 0 && !text.is_char_boundary(end) {
                    end -= 1;
                }
                text.truncate(end);
                text
            };
            // Live mode has no held frames for the segment pass to walk —
            // offer the flattened text as one segment so monitor-mode
            // segment moderators still record their observations.
            let live_seg_text = (!buffering).then(|| text.clone());
            let mut message = aisix_gateway::ChatMessage::assistant(text);
            if !tool_calls.is_empty() {
                message.extra.insert("tool_calls".to_string(), Value::Array(tool_calls));
            }
            let synth = ChatResponse {
                id: String::new(),
                model: model_label.clone(),
                message,
                finish_reason: FinishReason::Stop,
                usage: UsageStats::new(0, 0),
            };
            let (verdict, hits) =
                aisix_guardrails::Guardrail::check_output_non_segment_observed(
                    chain.as_ref(),
                    &synth,
                )
                .await;
            guard.comp().monitor_hits.extend(hits);
            // Segment pass over the held SSE frames: one Bedrock call; an
            // ANONYMIZE disposition rewrites the held bytes (#932 bedrock
            // follow-up).
            let mut seg_counts = crate::redact::RedactionCounts::new();
            let mut seg_hits = Vec::new();
            let mut joined: Vec<u8> = Vec::with_capacity(held_bytes);
            for b in &held {
                joined.extend_from_slice(b);
            }
            let mut seg_rewrote = false;
            let verdict = crate::redact::moderate_body(
                chain.as_ref(),
                crate::redact::Direction::Output,
                None,
                verdict,
                &mut seg_counts,
                &mut seg_hits,
                |g| match live_seg_text.as_deref() {
                    // Live-forward: observation only — nothing to rewrite.
                    Some(t) => {
                        let _ = g.redact_output_text(t);
                        crate::redact::RedactionCounts::new()
                    }
                    None => match crate::redact::redact_responses_sse(g, &joined) {
                        Some((rewritten, counts)) => {
                            joined = rewritten;
                            seg_rewrote = true;
                            counts
                        }
                        None => crate::redact::RedactionCounts::new(),
                    },
                },
            )
            .await;
            guard.comp().monitor_hits.extend(seg_hits);
            // `buffering` gate: only the hold-back walk can actually rewrite
            // wire bytes — the live walk is read-only, so a masked outcome
            // there (unreachable today) must not clobber the capture with a
            // rebuild from the empty `joined`.
            if buffering && !seg_counts.is_empty() {
                // Bedrock masked the held bytes — rebuild the content-
                // capture accumulator from the masked text channels,
                // keeping the original soft cap (#932 × AISIX-Cloud#947).
                if let Some(cap) = content_cap {
                    let mut rebuilt = crate::redact::responses_sse_text(&joined);
                    let mut cut = (cap as usize).min(rebuilt.len());
                    while cut < rebuilt.len() && !rebuilt.is_char_boundary(cut) {
                        cut += 1;
                    }
                    rebuilt.truncate(cut);
                    guard.comp().response_text = rebuilt;
                }
                crate::redact::merge_counts(
                    &mut guard.comp().redacted_entity_counts,
                    seg_counts,
                );
            }
            if let aisix_guardrails::GuardrailVerdict::Block {
                reason,
                guardrail_name,
                unavailable,
            } = verdict {
                tracing::warn!(
                    guardrail_hook = "output",
                    model = %model_label,
                    reason = %reason,
                    "guardrail blocked streaming /v1/responses (cross-provider) response",
                );
                guard.comp().guardrail_blocked = true;
                // On the live-forward path the terminal event is already on
                // the wire; the error frame is all that can follow it.
                let terminal_sent = !buffering && encoder.is_finished();
                yield Ok(guardrail_failure_frames(
                    &mut encoder,
                    sent_downstream,
                    terminal_sent,
                    guardrail_name.as_deref(),
                    unavailable.as_deref(),
                ));
                return;
            }
            if seg_rewrote {
                held = vec![bytes::Bytes::from(joined)];
            }
        }
        // Passed (#932): mask the held SSE frames (channel reassembly)
        // before release, then hand them to the client.
        if !held.is_empty() && aisix_guardrails::Guardrail::redacts_output(chain.as_ref()) {
            let mut joined: Vec<u8> = Vec::with_capacity(held_bytes);
            for b in &held {
                joined.extend_from_slice(b);
            }
            if let Some((rewritten, counts)) =
                crate::redact::redact_responses_sse(chain.as_ref(), &joined)
            {
                // The wire bytes were masked — mask the content-capture
                // accumulator too, or the exported content would carry
                // PII the client never saw (#932 × AISIX-Cloud#947).
                crate::redact::redact_captured_output(
                    chain.as_ref(),
                    &mut guard.comp().response_text,
                );
                crate::redact::merge_counts(
                    &mut guard.comp().redacted_entity_counts,
                    counts,
                );
                downstream_mark!();
                yield Ok(bytes::Bytes::from(rewritten));
                return;
            }
        }
        // Release the held events verbatim.
        for b in held {
            downstream_mark!();
            yield Ok(b);
        }
    };
    // Re-attach the request span: the body is polled after the request-id
    // middleware returns, so the end-of-stream output-guardrail check
    // would otherwise log without a `request_id` (AISIX-Cloud#1060).
    axum::body::Body::from_stream(crate::sse_keepalive::with_heartbeat(
        crate::request_id::in_request_span(stream),
        crate::sse_keepalive::interval(),
    ))
}

/// The Responses-API `error` event, as the API itself defines it — FLAT,
/// with the discriminant on the top-level `type`.
///
/// Every OTHER SSE error this crate emits nests under an `error` object,
/// and this one deliberately does not, because each surface matches its own
/// protocol rather than the crate's internal habit. The Responses event
/// stream is a union discriminated on `type`, and the official SDKs parse
/// it that way — `openai-python`'s `ResponseErrorEvent` is
/// `{type: Literal["error"], code, message, param, sequence_number}`,
/// generated from OpenAI's own OpenAPI spec. Nesting the payload would hand
/// a `responses.stream()` client an event it cannot classify at all. (The
/// mirror-image argument is why the Anthropic surface uses Anthropic's
/// closed `error.type` enum instead of ours.)
///
/// `sequence_number` continues the encoder's own numbering, so the error is
/// an ordinary member of the stream a client has been counting.
fn responses_error_frame(seq: u64, code: &str, message: &str) -> String {
    format!(
        "event: error\ndata: {}\n\n",
        json!({
            "type": "error",
            "code": code,
            "message": message,
            "param": Value::Null,
            "sequence_number": seq,
        })
    )
}

/// The frames that end a bridged stream on a failure: the flat `error`
/// event, then — unless a terminal event already reached the client —
/// `response.failed`, the event a client reads the failure from. A client
/// that does not recognise the flat `error` event (the Codex CLI is one)
/// otherwise sees only a connection that closed before the response ended,
/// and retries the turn without ever showing why.
///
/// The two carry the same message; `frame_code` is the flat event's code,
/// `failed_code` the Response object's `error.code`, which clients map to a
/// retry decision.
///
/// A failure that is the first thing the client receives — the upstream
/// failed or ended before producing anything, or a held-back stream was
/// withheld — is preceded by `response.created` and
/// `response.in_progress`: a stream that does not open with
/// `response.created` is one the OpenAI SDKs' `responses.stream()` helper
/// rejects with its own error, and the caller never sees this one. The
/// numbering then starts at 0: events a held-back stream withheld were
/// numbered, but the client never saw them.
fn failure_frames(
    encoder: &mut ResponsesSseEncoder,
    sent_downstream: bool,
    terminal_sent: bool,
    frame_code: &str,
    failed_code: &str,
    message: &str,
) -> bytes::Bytes {
    let mut frames = String::new();
    if !sent_downstream {
        encoder.sequence_number = 0;
        for ev in encoder.opening_events() {
            frames.push_str(&ev.to_sse_string());
        }
    }
    frames.push_str(&responses_error_frame(
        encoder.take_sequence_number(),
        frame_code,
        message,
    ));
    if !terminal_sent {
        frames.push_str(&encoder.failed_event(failed_code, message).to_sse_string());
    }
    bytes::Bytes::from(frames)
}

/// [`failure_frames`] for an output-guardrail stop — a block verdict, or a
/// held-back response that outgrew the buffer. Carries the firing
/// guardrail's name (#519 B.4b) but never the matched-pattern detail.
///
/// `response.failed` reports `invalid_prompt`: re-running the same turn
/// meets the same guardrail, so the code is one clients treat as a request
/// that will not succeed on retry, and show its message.
fn guardrail_failure_frames(
    encoder: &mut ResponsesSseEncoder,
    sent_downstream: bool,
    terminal_sent: bool,
    guardrail_name: Option<&str>,
    unavailable: Option<&str>,
) -> bytes::Bytes {
    failure_frames(
        encoder,
        sent_downstream,
        terminal_sent,
        "content_filter",
        "invalid_prompt",
        &crate::error::guardrail_block_message("response", guardrail_name, unavailable),
    )
}

/// The `error.code` of the `response.failed` that ends a stream on an
/// upstream failure: the flat frame's code, unless the upstream's own
/// in-band error named one of the codes clients act on specifically, which
/// is passed through so the client can (a context overflow, an exhausted
/// quota, a rate limit, an overloaded server).
fn upstream_failed_code(e: &aisix_gateway::BridgeError) -> &str {
    const PASSED_THROUGH: [&str; 5] = [
        "context_length_exceeded",
        "insufficient_quota",
        "rate_limit_exceeded",
        "server_is_overloaded",
        "slow_down",
    ];
    if let aisix_gateway::BridgeError::UpstreamInBand {
        parsed: Some(parsed),
        ..
    } = e
    {
        if let Some(code) = [parsed.code.as_deref(), parsed.kind.as_deref()]
            .into_iter()
            .flatten()
            .find(|c| PASSED_THROUGH.contains(c))
        {
            return code;
        }
    }
    e.error_type()
}

/// Code of the failure a stream that carried no response ends with. The
/// same one an upstream failure carries, which clients retry.
const EMPTY_STREAM_CODE: &str = "upstream_error";
const EMPTY_STREAM_MESSAGE: &str =
    "upstream returned an empty stream: no content, reasoning, tool call or finish reason";

fn finish_reason_label(reason: &FinishReason) -> String {
    match reason {
        FinishReason::Stop => "stop".into(),
        FinishReason::Length => "length".into(),
        FinishReason::ContentFilter => "content_filter".into(),
        FinishReason::ToolCalls => "tool_calls".into(),
        FinishReason::Other(s) => s.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{ChatDelta, Role};
    use std::collections::BTreeSet;

    /// A request that declared no tools and no settings.
    fn no_custom_tools() -> ResponsesReplyContext {
        ResponsesReplyContext::default()
    }

    /// A request that declared these `custom` tools.
    fn custom_tools(names: &[&str]) -> ResponsesReplyContext {
        let tools: Vec<Value> = names
            .iter()
            .map(|n| json!({"type": "custom", "name": n}))
            .collect();
        ResponsesReplyContext::from_request(&json!({"tools": tools}))
    }

    // ── Request translation ──────────────────────────────────────

    #[test]
    fn instructions_become_system_and_input_string_becomes_user() {
        let body = json!({
            "model": "opus-4.7",
            "instructions": "be terse",
            "input": "hi",
        });
        let chat = responses_request_to_chat("opus-4.7", &body);
        assert_eq!(chat.messages.len(), 2);
        assert!(matches!(chat.messages[0].role, Role::System));
        assert_eq!(chat.messages[0].content_str(), "be terse");
        assert!(matches!(chat.messages[1].role, Role::User));
        assert_eq!(chat.messages[1].content_str(), "hi");
    }

    #[test]
    fn input_array_messages_preserve_roles_and_text_parts() {
        let body = json!({
            "model": "m",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "part1"}, {"type": "input_text", "text": "part2"}]},
                {"role": "assistant", "content": "ok"},
            ],
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(chat.messages.len(), 2);
        assert!(matches!(chat.messages[0].role, Role::User));
        assert_eq!(chat.messages[0].content_str(), "part1part2");
        assert!(matches!(chat.messages[1].role, Role::Assistant));
    }

    /// The content parts of one user message, as the OpenAI-compatible
    /// bridge would put them on the wire.
    fn user_blocks(body: &Value) -> Vec<Value> {
        let chat = responses_request_to_chat("m", body);
        chat.messages[0]
            .content_blocks
            .clone()
            .expect("message carries typed content blocks")
    }

    fn image_body(image: Value) -> Value {
        json!({
            "model": "m",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "what is in this image?"},
                image,
            ]}],
        })
    }

    #[test]
    fn input_image_url_and_detail_become_a_chat_image_url_part() {
        let blocks = user_blocks(&image_body(json!({
            "type": "input_image",
            "image_url": "https://example.com/cat.png",
            "detail": "high",
        })));
        assert_eq!(
            blocks,
            vec![
                json!({"type": "text", "text": "what is in this image?"}),
                json!({"type": "image_url", "image_url": {
                    "url": "https://example.com/cat.png",
                    "detail": "high",
                }}),
            ]
        );
    }

    #[test]
    fn input_image_without_detail_leaves_detail_off_the_wire() {
        let blocks = user_blocks(&image_body(json!({
            "type": "input_image",
            "image_url": "https://example.com/cat.png",
        })));
        assert_eq!(
            blocks[1],
            json!({"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}})
        );
    }

    #[test]
    fn data_url_image_passes_through_verbatim() {
        let data_url = "data:image/png;base64,iVBORw0KGgo=";
        let blocks = user_blocks(&image_body(json!({
            "type": "input_image",
            "image_url": data_url,
        })));
        assert_eq!(blocks[1]["image_url"]["url"], data_url);
    }

    /// An `input_image` addressed only by uploaded-file id has no
    /// chat-completions counterpart; it must not become an `image_url` with
    /// an empty `url`, which a chat upstream rejects outright.
    #[test]
    fn file_id_only_input_image_yields_no_image_part() {
        let chat = responses_request_to_chat(
            "m",
            &image_body(json!({"type": "input_image", "file_id": "file-abc"})),
        );
        assert_eq!(chat.messages[0].content_str(), "what is in this image?");
        assert!(chat.messages[0].content_blocks.is_none());
    }

    #[test]
    fn input_file_becomes_a_chat_file_part_with_the_members_sent() {
        let blocks = user_blocks(&json!({
            "model": "m",
            "input": [{"role": "user", "content": [{
                "type": "input_file",
                "filename": "draft.pdf",
                "file_data": "data:application/pdf;base64,JVBERi0=",
            }]}],
        }));
        assert_eq!(
            blocks,
            vec![json!({"type": "file", "file": {
                "file_data": "data:application/pdf;base64,JVBERi0=",
                "filename": "draft.pdf",
            }})]
        );
    }

    #[test]
    fn input_audio_becomes_a_chat_input_audio_part() {
        let blocks = user_blocks(&json!({
            "model": "m",
            "input": [{"role": "user", "content": [{
                "type": "input_audio",
                "input_audio": {"data": "UklGRg==", "format": "wav"},
            }]}],
        }));
        assert_eq!(
            blocks,
            vec![json!({"type": "input_audio", "input_audio": {
                "data": "UklGRg==",
                "format": "wav",
            }})]
        );
    }

    /// A turn made only of non-text parts used to be erased: the empty
    /// concatenated text dropped the whole message and the upstream never
    /// saw the image.
    #[test]
    fn all_non_text_message_still_reaches_the_upstream() {
        let chat = responses_request_to_chat(
            "m",
            &json!({
                "model": "m",
                "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": "https://example.com/a.png"},
                ]}],
            }),
        );
        assert_eq!(chat.messages.len(), 1);
        assert!(matches!(chat.messages[0].role, Role::User));
        assert_eq!(
            chat.messages[0].content_blocks.as_deref(),
            Some(
                [json!({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}})]
                    .as_slice()
            )
        );
    }

    #[test]
    fn text_only_message_keeps_the_bare_string_shape() {
        let chat = responses_request_to_chat(
            "m",
            &json!({
                "model": "m",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            }),
        );
        assert_eq!(chat.messages[0].content_str(), "hi");
        assert!(chat.messages[0].content_blocks.is_none());
    }

    /// A tool result carrying an image forwards it as an `image_url` part;
    /// a text-only tool result stays a plain string.
    #[test]
    fn tool_output_array_keeps_its_text_and_drops_the_image() {
        // OpenAI answers 400 "Image URLs are only allowed for messages
        // with role 'user'" to a `tool` message carrying an image part,
        // and no bridge reads blocks off a tool message, so the image is
        // dropped and its text siblings still reach the model.
        let chat = responses_request_to_chat(
            "m",
            &json!({
                "model": "m",
                "input": [{"type": "function_call_output", "call_id": "call_1", "output": [
                    {"type": "input_text", "text": "screenshot:"},
                    {"type": "input_image", "image_url": "https://example.com/s.png"},
                ]}],
            }),
        );
        assert!(matches!(chat.messages[0].role, Role::Tool));
        assert_eq!(chat.messages[0].content_str(), "screenshot:");
        assert!(chat.messages[0].content_blocks.is_none());
    }

    #[test]
    fn text_only_tool_output_array_stays_a_string() {
        let chat = responses_request_to_chat(
            "m",
            &json!({
                "model": "m",
                "input": [{"type": "function_call_output", "call_id": "c", "output": [
                    {"type": "output_text", "text": "done"},
                ]}],
            }),
        );
        assert_eq!(chat.messages[0].content_str(), "done");
        assert!(chat.messages[0].content_blocks.is_none());
    }

    fn response_format(text: Value) -> Option<Value> {
        let body = json!({"model": "m", "input": "hi", "text": text});
        responses_request_to_chat("m", &body)
            .extra
            .get("response_format")
            .cloned()
    }

    #[test]
    fn text_format_json_schema_becomes_response_format() {
        assert_eq!(
            response_format(json!({"format": {
                "type": "json_schema",
                "name": "weather",
                "schema": {"type": "object", "properties": {"c": {"type": "number"}}},
                "strict": true,
                "description": "a forecast",
            }})),
            Some(json!({"type": "json_schema", "json_schema": {
                "name": "weather",
                "schema": {"type": "object", "properties": {"c": {"type": "number"}}},
                "strict": true,
                "description": "a forecast",
            }}))
        );
    }

    #[test]
    fn text_format_json_schema_omits_the_members_the_caller_omitted() {
        assert_eq!(
            response_format(json!({"format": {"type": "json_schema", "name": "n"}})),
            Some(json!({"type": "json_schema", "json_schema": {"name": "n"}}))
        );
    }

    #[test]
    fn text_format_json_object_becomes_response_format() {
        assert_eq!(
            response_format(json!({"format": {"type": "json_object"}})),
            Some(json!({"type": "json_object"}))
        );
    }

    #[test]
    fn text_format_text_and_absent_text_emit_no_response_format() {
        assert_eq!(response_format(json!({"format": {"type": "text"}})), None);
        assert_eq!(response_format(json!({})), None);
        let chat = responses_request_to_chat("m", &json!({"model": "m", "input": "hi"}));
        assert!(!chat.extra.contains_key("response_format"));
    }

    /// `text.verbosity` has no chat-completions counterpart on this path —
    /// it must not leak onto the upstream wire, where the bridges flatten
    /// `extra` and an unknown key 400s.
    #[test]
    fn text_verbosity_is_not_forwarded() {
        let chat = responses_request_to_chat(
            "m",
            &json!({"model": "m", "input": "hi", "text": {"verbosity": "low"}}),
        );
        assert!(!chat.extra.contains_key("verbosity"));
        assert!(!chat.extra.contains_key("response_format"));
    }

    #[test]
    fn function_call_and_output_become_assistant_tool_calls_and_tool_turn() {
        // The codex agent-loop history shape.
        let body = json!({
            "model": "m",
            "input": [
                {"role": "user", "content": "run ls"},
                {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
            ],
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(chat.messages.len(), 3);
        assert!(matches!(chat.messages[1].role, Role::Assistant));
        let tcs = chat.messages[1]
            .extra
            .get("tool_calls")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["id"], "call_1");
        assert_eq!(tcs[0]["function"]["name"], "shell");
        assert!(matches!(chat.messages[2].role, Role::Tool));
        assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(chat.messages[2].content_str(), "a.txt");
    }

    #[test]
    fn parallel_function_calls_fold_into_one_assistant_message() {
        let body = json!({
            "model": "m",
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}"},
            ],
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(chat.messages.len(), 1);
        let tcs = chat.messages[0]
            .extra
            .get("tool_calls")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(tcs.len(), 2);
    }

    #[test]
    fn tools_and_params_translate_to_chat_shape() {
        let body = json!({
            "model": "m",
            "input": "hi",
            "max_output_tokens": 256,
            "temperature": 0.5,
            "stream": true,
            "tools": [{"type": "function", "name": "get_weather", "description": "d", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "get_weather"},
            // Only the portable effort value is translated; the Responses
            // carrier object itself must not leak to another protocol.
            "reasoning": {"effort": "high", "summary": "auto"},
            "store": false,
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(chat.max_tokens, Some(256));
        assert_eq!(chat.temperature, Some(0.5));
        assert_eq!(chat.stream, Some(true));
        let tools = chat.extra.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(
            chat.extra.get("tool_choice").unwrap()["function"]["name"],
            "get_weather"
        );
        assert_eq!(chat.extra.get("reasoning_effort"), Some(&json!("high")));
        assert!(!chat.extra.contains_key("reasoning"));
        assert!(!chat.extra.contains_key("store"));
    }

    #[test]
    fn tool_choice_is_dropped_when_no_tool_survives_translation() {
        // The Codex CLI serialises its context-compaction call with an
        // empty tool list and `tool_choice: "auto"`. The Responses API
        // accepts that pair; a chat-completions upstream rejects the
        // choice without a `tools` key (AISIX-Cloud#1614).
        let empty = json!({
            "model": "m",
            "input": "Summarise",
            "tools": [],
            "tool_choice": "auto",
        });
        let chat = responses_request_to_chat("m", &empty);
        assert!(!chat.extra.contains_key("tools"));
        assert!(!chat.extra.contains_key("tool_choice"));

        // Same when the list holds only tools with no chat equivalent,
        // so translation filters every entry out.
        let hosted_only = json!({
            "model": "m",
            "input": "Summarise",
            "tools": [{"type": "web_search_preview"}],
            "tool_choice": {"type": "function", "name": "get_weather"},
        });
        let chat = responses_request_to_chat("m", &hosted_only);
        assert!(!chat.extra.contains_key("tools"));
        assert!(!chat.extra.contains_key("tool_choice"));
    }

    #[test]
    fn custom_tool_becomes_a_function_tool_taking_one_string() {
        let body = json!({
            "model": "m",
            "input": "patch it",
            "tools": [
                {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Edit a file",
                    "format": {"type": "grammar", "syntax": "lark", "definition": "start: TEXT"},
                },
                {"type": "web_search_preview"},
            ],
        });
        let chat = responses_request_to_chat("m", &body);
        let tools = chat.extra.get("tools").unwrap().as_array().unwrap();
        // The hosted tool is still filtered out; the other two survive.
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[1]["type"], "function");
        assert_eq!(tools[1]["function"]["name"], "apply_patch");
        assert_eq!(
            tools[1]["function"]["parameters"],
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The apply_patch content following the specified format",
                    }
                },
                "required": ["content"],
            })
        );
        // The grammar the freeform tool carried is the only instruction
        // the model gets about the expected shape.
        let description = tools[1]["function"]["description"].as_str().unwrap();
        assert_eq!(
            description,
            "Edit a file\n\nFormat:\n```lark\nstart: TEXT\n```"
        );
    }

    #[test]
    fn custom_tool_without_a_grammar_keeps_its_bare_description() {
        let body = json!({
            "model": "m",
            "input": "go",
            "tools": [{"type": "custom", "name": "freeform", "description": "d"}],
        });
        let chat = responses_request_to_chat("m", &body);
        let tools = chat.extra.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools[0]["function"]["description"], "d");
    }

    /// A custom tool call the caller replays has to go back upstream in the
    /// same single-string function shape the tool was offered in, or the
    /// history stops matching the tools list and the model re-asks.
    #[test]
    fn a_replayed_custom_tool_call_rewraps_its_input_as_function_arguments() {
        let body = json!({
            "model": "m",
            "tools": [{"type": "custom", "name": "apply_patch"}],
            "input": [
                {
                    "type": "custom_tool_call",
                    "id": "ctc_1",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "input": "*** Begin Patch",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "applied",
                },
            ],
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(chat.messages.len(), 2);

        assert!(matches!(chat.messages[0].role, Role::Assistant));
        let tool_calls = chat.messages[0].extra["tool_calls"].as_array().unwrap();
        assert_eq!(
            tool_calls[0],
            json!({
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": "{\"content\":\"*** Begin Patch\"}",
                },
            })
        );

        assert!(matches!(chat.messages[1].role, Role::Tool));
        assert_eq!(chat.messages[1].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(chat.messages[1].content.as_deref(), Some("applied"));
    }

    /// A custom tool's result takes the same string-or-content-parts union
    /// as `function_call_output`, so it reads through the same converter.
    #[test]
    fn a_custom_tool_result_carrying_content_parts_reads_like_a_function_one() {
        let body = json!({
            "model": "m",
            "input": [{
                "type": "custom_tool_call_output",
                "call_id": "c1",
                "output": [{"type": "output_text", "text": "done"}],
            }],
        });
        let chat = responses_request_to_chat("m", &body);
        assert!(matches!(chat.messages[0].role, Role::Tool));
        assert_eq!(chat.messages[0].content.as_deref(), Some("done"));
    }

    #[test]
    fn reply_context_reads_only_the_custom_entries_as_custom_tools() {
        let reply = ResponsesReplyContext::from_request(&json!({
            "tools": [
                {"type": "function", "name": "get_weather"},
                {"type": "custom", "name": "apply_patch"},
                {"type": "custom"},
                {"type": "web_search_preview"},
            ],
        }));
        assert_eq!(
            reply.custom_tools,
            BTreeSet::from(["apply_patch".to_string()])
        );
        assert!(ResponsesReplyContext::from_request(&json!({"input": "hi"}))
            .custom_tools
            .is_empty());
    }

    #[test]
    fn tool_choice_forms_normalise_to_the_provider_neutral_chat_shape() {
        let with_choice = |tc: Value| {
            let body = json!({
                "model": "m",
                "input": "hi",
                "tools": [{"type": "function", "name": "get_weather"}],
                "tool_choice": tc,
            });
            responses_request_to_chat("m", &body)
                .extra
                .get("tool_choice")
                .cloned()
        };

        for mode in ["auto", "none", "required"] {
            assert_eq!(with_choice(json!(mode)), Some(json!(mode)));
        }
        // An allowed_tools choice keeps its mode; chat cannot express the
        // subset restriction, so the subset is dropped.
        assert_eq!(
            with_choice(json!({
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "get_weather"}],
            })),
            Some(json!("required"))
        );
        assert_eq!(
            with_choice(json!({"type": "allowed_tools", "mode": "auto", "tools": []})),
            Some(json!("auto"))
        );
        assert_eq!(with_choice(json!({"type": "any"})), Some(json!("required")));
        // The three named forms all land on the one chat spelling — never
        // a Responses-only shape a non-OpenAI bridge could not read.
        for named in [
            json!({"type": "function", "name": "get_weather"}),
            json!({"type": "custom", "name": "get_weather"}),
            json!({"type": "tool", "name": "get_weather"}),
        ] {
            assert_eq!(
                with_choice(named),
                Some(json!({"type": "function", "function": {"name": "get_weather"}}))
            );
        }
        // A hosted-tool choice, an allowed_tools mode with no chat
        // counterpart, and a named form missing its name all drop.
        assert_eq!(with_choice(json!({"type": "file_search"})), None);
        assert_eq!(
            with_choice(json!({"type": "allowed_tools", "mode": "none"})),
            None
        );
        assert_eq!(with_choice(json!({"type": "function"})), None);
    }

    #[test]
    fn parallel_tool_calls_rides_along_with_a_surviving_tools_list() {
        let body = json!({
            "model": "m",
            "input": "hi",
            "tools": [{"type": "function", "name": "get_weather"}],
            "parallel_tool_calls": false,
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(chat.extra.get("parallel_tool_calls"), Some(&json!(false)));

        // `true` is forwarded as sent, not normalised away.
        let body = json!({
            "model": "m",
            "input": "hi",
            "tools": [{"type": "function", "name": "get_weather"}],
            "parallel_tool_calls": true,
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(chat.extra.get("parallel_tool_calls"), Some(&json!(true)));
    }

    #[test]
    fn parallel_tool_calls_is_dropped_when_no_tool_survives_translation() {
        // Same rule as `tool_choice`: a chat upstream rejects the field
        // without an accompanying `tools` list.
        let body = json!({
            "model": "m",
            "input": "hi",
            "tools": [{"type": "web_search_preview"}],
            "parallel_tool_calls": false,
        });
        let chat = responses_request_to_chat("m", &body);
        assert!(!chat.extra.contains_key("tools"));
        assert!(!chat.extra.contains_key("parallel_tool_calls"));
    }

    #[test]
    fn json_tool_output_reaches_the_upstream_as_a_json_string() {
        let outputs = [
            (
                json!({"temp": 21, "unit": "C"}),
                r#"{"temp":21,"unit":"C"}"#,
            ),
            (json!(42), "42"),
            (json!(true), "true"),
            // `null` and an absent output are the empty string, not "null".
            (json!(null), ""),
        ];
        for (output, expected) in outputs {
            let body = json!({
                "model": "m",
                "input": [{"type": "function_call_output", "call_id": "c1", "output": output}],
            });
            let chat = responses_request_to_chat("m", &body);
            let msg = chat.messages.last().unwrap();
            assert!(matches!(msg.role, Role::Tool));
            assert_eq!(msg.content.as_deref(), Some(expected));
            assert!(msg.content_blocks.is_none());
        }

        // A string output is untouched — it is not re-encoded with quotes.
        let body = json!({
            "model": "m",
            "input": [{"type": "function_call_output", "call_id": "c1", "output": "21C"}],
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(
            chat.messages.last().unwrap().content.as_deref(),
            Some("21C")
        );
    }

    #[test]
    fn a_json_array_tool_output_is_serialised_not_parsed_as_content_parts() {
        // A tool returning a list of records is a JSON array, not the
        // Responses content-part array it would otherwise be parsed as —
        // which recognised no part and emptied the whole tool message.
        let body = json!({
            "model": "m",
            "input": [{
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"id": 1, "name": "x"}, {"id": 2, "name": "y"}],
            }],
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(
            chat.messages.last().unwrap().content.as_deref(),
            Some(r#"[{"id":1,"name":"x"},{"id":2,"name":"y"}]"#)
        );

        // An array that IS content parts keeps the part handling: its
        // text reaches the model unquoted.
        let body = json!({
            "model": "m",
            "input": [{
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "21C"}],
            }],
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(
            chat.messages.last().unwrap().content.as_deref(),
            Some("21C")
        );

        // A mixed array keeps its parts as text and serialises every
        // element that is not one, in place — nothing the tool returned
        // is dropped on the floor.
        let body = json!({
            "model": "m",
            "input": [{
                "type": "function_call_output",
                "call_id": "c1",
                "output": [
                    {"type": "text", "text": "rows: "},
                    42,
                    {"total": 3},
                    "plain",
                ],
            }],
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(
            chat.messages.last().unwrap().content.as_deref(),
            Some(r#"rows: 42{"total":3}plain"#)
        );

        // An empty array is not a value worth serialising as "[]".
        let body = json!({
            "model": "m",
            "input": [{"type": "function_call_output", "call_id": "c1", "output": []}],
        });
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(chat.messages.last().unwrap().content.as_deref(), Some(""));
    }

    #[test]
    fn out_of_range_max_output_tokens_is_ignored_not_truncated() {
        // A value above u32::MAX must not wrap to a small/zero cap.
        let body = json!({"model": "m", "input": "hi", "max_output_tokens": 10_000_000_000u64});
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(chat.max_tokens, None);
    }

    // ── Non-streaming response translation ───────────────────────

    fn chat_response_with(
        text: Option<&str>,
        tool_calls: Option<Value>,
        fr: FinishReason,
    ) -> ChatResponse {
        let mut extra = Map::new();
        if let Some(tc) = tool_calls {
            extra.insert("tool_calls".into(), tc);
        }
        ChatResponse {
            id: "id".into(),
            model: "m".into(),
            message: ChatMessage {
                role: Role::Assistant,
                content: text.map(|s| s.to_string()),
                content_blocks: None,
                name: None,
                tool_call_id: None,
                extra,
            },
            finish_reason: fr,
            usage: UsageStats::new(11, 7),
        }
    }

    #[test]
    fn non_streaming_text_response_builds_message_output_and_usage() {
        let resp = chat_response_with(Some("hello"), None, FinishReason::Stop);
        let out = chat_response_to_responses_json(&resp, "opus-4.7", 100, &no_custom_tools());
        assert_eq!(out["object"], "response");
        assert_eq!(out["status"], "completed");
        assert_eq!(out["model"], "opus-4.7");
        let item = &out["output"][0];
        assert_eq!(item["type"], "message");
        assert_eq!(item["content"][0]["type"], "output_text");
        assert_eq!(item["content"][0]["text"], "hello");
        assert_eq!(out["usage"]["input_tokens"], 11);
        assert_eq!(out["usage"]["output_tokens"], 7);
        assert_eq!(out["usage"]["total_tokens"], 18);
    }

    #[test]
    fn non_streaming_tool_call_response_builds_function_call_item() {
        let tcs = json!([{"id": "call_9", "type": "function", "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}}]);
        let resp = chat_response_with(None, Some(tcs), FinishReason::ToolCalls);
        let out = chat_response_to_responses_json(&resp, "m", 1, &no_custom_tools());
        let item = &out["output"][0];
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["call_id"], "call_9");
        assert_eq!(item["name"], "shell");
        assert_eq!(item["arguments"], "{\"cmd\":\"ls\"}");
    }

    /// The caller registered `apply_patch` as a `custom` tool, so the call
    /// it gets back is a `custom_tool_call` item carrying the freeform
    /// `input` — not the single-string function wrapper the request side
    /// used to reach a chat upstream.
    #[test]
    fn a_call_to_a_custom_tool_returns_a_custom_tool_call_item() {
        let tcs = json!([{
            "id": "call_9",
            "type": "function",
            "function": {"name": "apply_patch", "arguments": "{\"content\":\"*** Begin Patch\"}"},
        }]);
        let resp = chat_response_with(None, Some(tcs), FinishReason::ToolCalls);
        let out = chat_response_to_responses_json(&resp, "m", 1, &custom_tools(&["apply_patch"]));
        let item = &out["output"][0];
        assert_eq!(item["type"], "custom_tool_call");
        assert_eq!(item["call_id"], "call_9");
        assert_eq!(item["name"], "apply_patch");
        assert_eq!(item["input"], "*** Begin Patch");
        assert_eq!(item["status"], "completed");
        assert!(item["id"].as_str().unwrap().starts_with("ctc_"));
        // The function-call spelling is gone, not carried alongside.
        assert!(item.get("arguments").is_none());
    }

    /// A model that ignored the single-string schema still has its payload
    /// delivered: the raw argument string becomes the input, because that
    /// is what the caller's freeform tool was going to receive either way.
    #[test]
    fn custom_tool_input_falls_back_to_the_raw_arguments() {
        let custom = custom_tools(&["apply_patch"]);
        for arguments in [
            "not json at all",
            "{\"other\":\"x\"}",
            "{\"content\":42}",
            // The model answered with its freeform payload verbatim and it
            // happens to be JSON carrying a `content` field. Unwrapping
            // that would deliver `"x"` and drop `keep`.
            "{\"content\":\"x\",\"keep\":1}",
        ] {
            let tcs = json!([{
                "id": "c1",
                "type": "function",
                "function": {"name": "apply_patch", "arguments": arguments},
            }]);
            let resp = chat_response_with(None, Some(tcs), FinishReason::ToolCalls);
            let out = chat_response_to_responses_json(&resp, "m", 1, &custom);
            assert_eq!(out["output"][0]["input"], arguments, "for {arguments}");
        }
    }

    /// One reply can mix both kinds, and each keeps its own item type —
    /// the set is consulted per call, not once per response.
    #[test]
    fn a_mixed_reply_keeps_each_call_on_its_own_item_type() {
        let tcs = json!([
            {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}},
            {"id": "c2", "type": "function", "function": {"name": "apply_patch", "arguments": "{\"content\":\"p\"}"}},
        ]);
        let resp = chat_response_with(None, Some(tcs), FinishReason::ToolCalls);
        let out = chat_response_to_responses_json(&resp, "m", 1, &custom_tools(&["apply_patch"]));
        assert_eq!(out["output"][0]["type"], "function_call");
        assert_eq!(out["output"][0]["arguments"], "{\"cmd\":\"ls\"}");
        assert_eq!(out["output"][1]["type"], "custom_tool_call");
        assert_eq!(out["output"][1]["input"], "p");
    }

    #[test]
    fn length_finish_maps_to_incomplete_status() {
        let resp = chat_response_with(Some("x"), None, FinishReason::Length);
        let out = chat_response_to_responses_json(&resp, "m", 1, &no_custom_tools());
        assert_eq!(out["status"], "incomplete");
        assert_eq!(out["incomplete_details"]["reason"], "max_output_tokens");
    }

    // ── Streaming encoder ────────────────────────────────────────

    fn content_chunk(text: &str) -> ChatChunk {
        ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                content: Some(text.into()),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        }
    }

    fn types_of(events: &[ResponsesSseEvent]) -> Vec<&'static str> {
        events.iter().map(|e| e.event_type).collect()
    }

    #[test]
    fn streaming_text_emits_canonical_event_sequence() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "opus-4.7", 0, no_custom_tools());
        let mut all: Vec<ResponsesSseEvent> = Vec::new();
        all.extend(enc.next_events(&content_chunk("Hel")));
        all.extend(enc.next_events(&content_chunk("lo")));
        // Finish chunk carrying usage (Anthropic attaches it here).
        all.extend(enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::Stop),
            usage: Some(UsageStats::new(5, 2)),
        }));
        let types = types_of(&all);
        assert_eq!(
            types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert!(enc.is_finished());
        let completed = all.last().unwrap();
        assert_eq!(
            completed.data["response"]["output"][0]["content"][0]["text"],
            "Hello"
        );
        assert_eq!(completed.data["response"]["usage"]["input_tokens"], 5);
        assert_eq!(completed.data["response"]["usage"]["output_tokens"], 2);
        // sequence_number is monotonic from 0.
        assert_eq!(all[0].data["sequence_number"], 0);
        assert_eq!(all[1].data["sequence_number"], 1);
    }

    #[test]
    fn streaming_completed_withheld_until_trailing_usage_frame() {
        // OpenAI-compat upstreams send usage AFTER the finish chunk.
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let _ = enc.next_events(&content_chunk("hi"));
        // Finish without usage → close items but NOT completed yet.
        let at_finish = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        });
        assert!(!types_of(&at_finish).contains(&"response.completed"));
        assert!(!enc.is_finished());
        // Trailing usage frame releases completed.
        let usage_frame = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: None,
            usage: Some(UsageStats::new(3, 4)),
        });
        assert_eq!(types_of(&usage_frame), vec!["response.completed"]);
        assert_eq!(usage_frame[0].data["response"]["usage"]["output_tokens"], 4);
    }

    #[test]
    fn streaming_tool_call_emits_function_call_events() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let chunk = ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![json!({
                    "index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "shell", "arguments": "{\"cmd\""},
                })]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        };
        let mut all = enc.next_events(&chunk);
        all.extend(enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![
                    json!({"index": 0, "function": {"arguments": ":\"ls\"}"}}),
                ]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        }));
        all.extend(enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::ToolCalls),
            usage: Some(UsageStats::new(4, 6)),
        }));
        let types = types_of(&all);
        assert!(types.contains(&"response.output_item.added"));
        assert!(types.contains(&"response.function_call_arguments.delta"));
        assert!(types.contains(&"response.function_call_arguments.done"));
        assert_eq!(*types.last().unwrap(), "response.completed");
        let completed = all.last().unwrap();
        let item = &completed.data["response"]["output"][0];
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["call_id"], "call_1");
        assert_eq!(item["arguments"], "{\"cmd\":\"ls\"}");
    }

    /// One custom-tool call, streamed: the item opens as a
    /// `custom_tool_call` with an empty `input`, the wrapper fragments are
    /// buffered rather than streamed, and the close emits exactly one
    /// unwrapped input delta, its done event, and the full item.
    #[test]
    fn streaming_custom_tool_call_emits_one_unwrapped_input_delta() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, custom_tools(&["apply_patch"]));
        let mut all = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![json!({
                    "index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "apply_patch", "arguments": "{\"content\":\"*** Be"},
                })]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        });
        all.extend(enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![
                    json!({"index": 0, "function": {"arguments": "gin Patch\"}"}}),
                ]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        }));
        all.extend(enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::ToolCalls),
            usage: Some(UsageStats::new(4, 6)),
        }));

        assert_eq!(
            types_of(&all),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.custom_tool_call_input.delta",
                "response.custom_tool_call_input.done",
                "response.output_item.done",
                "response.completed",
            ]
        );

        let added = &all[2].data;
        assert_eq!(added["item"]["type"], "custom_tool_call");
        assert_eq!(added["item"]["input"], "");
        assert_eq!(added["item"]["status"], "in_progress");
        let item_id = added["item"]["id"].as_str().unwrap().to_string();
        assert!(item_id.starts_with("ctc_"));

        assert_eq!(all[3].data["delta"], "*** Begin Patch");
        assert_eq!(all[3].data["item_id"], item_id);
        assert_eq!(all[4].data["input"], "*** Begin Patch");
        assert_eq!(all[4].data["item_id"], item_id);

        let done_item = &all[5].data["item"];
        assert_eq!(done_item["type"], "custom_tool_call");
        assert_eq!(done_item["id"], item_id);
        assert_eq!(done_item["call_id"], "call_1");
        assert_eq!(done_item["input"], "*** Begin Patch");
        assert_eq!(done_item["status"], "completed");

        let final_item = &all[6].data["response"]["output"][0];
        assert_eq!(final_item["type"], "custom_tool_call");
        assert_eq!(final_item["input"], "*** Begin Patch");

        // `sequence_number` runs unbroken across the custom-tool events.
        let seqs: Vec<u64> = all
            .iter()
            .map(|e| e.data["sequence_number"].as_u64().unwrap())
            .collect();
        assert_eq!(seqs, (0..all.len() as u64).collect::<Vec<_>>());
    }

    /// A custom tool never streams the function-call argument events —
    /// they carry the single-string wrapper, which is gateway plumbing the
    /// caller never asked to see.
    /// An upstream that splits `function.name` across chunks overwrites
    /// the name on each one. The item id and the item type are both
    /// derived from whether the name is a custom tool, so a later fragment
    /// completing the name must not change what an already-emitted
    /// `output_item.added` said — `.done` would name an item the client
    /// never saw opened.
    #[test]
    fn a_tool_name_completed_after_the_item_opened_keeps_its_announced_identity() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, custom_tools(&["apply_patch"]));
        // First fragment carries a PREFIX of the custom tool's name, so the
        // item opens as a plain function call.
        let mut all = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![json!({
                    "index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "apply_"},
                })]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        });
        all.extend(enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![
                    json!({"index": 0, "function": {"name": "apply_patch", "arguments": "{}"}}),
                ]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        }));
        all.extend(enc.force_finish());

        let added = all
            .iter()
            .find(|e| e.event_type == "response.output_item.added")
            .expect("item announced");
        let done = all
            .iter()
            .find(|e| e.event_type == "response.output_item.done")
            .expect("item closed");
        assert_eq!(added.data["item"]["id"], done.data["item"]["id"]);
        assert_eq!(added.data["item"]["type"], done.data["item"]["type"]);
        let final_item = &all.last().unwrap().data["response"]["output"][0];
        assert_eq!(final_item["id"], added.data["item"]["id"]);
        assert_eq!(final_item["type"], added.data["item"]["type"]);
    }

    #[test]
    fn streaming_custom_tool_call_emits_no_function_call_argument_events() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, custom_tools(&["apply_patch"]));
        let mut all = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![json!({
                    "index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "apply_patch", "arguments": "{\"content\":\"p\"}"},
                })]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        });
        all.extend(enc.force_finish());
        let types = types_of(&all);
        assert!(!types.contains(&"response.function_call_arguments.delta"));
        assert!(!types.contains(&"response.function_call_arguments.done"));
    }

    /// Both kinds in one stream keep their own item types and event
    /// families, at their own `output_index`.
    #[test]
    fn streaming_mixed_tool_calls_keep_their_own_event_families() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, custom_tools(&["apply_patch"]));
        let mut all = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![
                    json!({"index": 0, "id": "c1", "type": "function",
                           "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}}),
                    json!({"index": 1, "id": "c2", "type": "function",
                           "function": {"name": "apply_patch", "arguments": "{\"content\":\"p\"}"}}),
                ]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        });
        all.extend(enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::ToolCalls),
            usage: Some(UsageStats::new(4, 6)),
        }));
        let types = types_of(&all);
        assert!(types.contains(&"response.function_call_arguments.delta"));
        assert!(types.contains(&"response.function_call_arguments.done"));
        assert!(types.contains(&"response.custom_tool_call_input.delta"));
        assert!(types.contains(&"response.custom_tool_call_input.done"));

        let output = all.last().unwrap().data["response"]["output"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["arguments"], "{\"cmd\":\"ls\"}");
        assert!(output[0]["id"].as_str().unwrap().starts_with("fc_"));
        assert_eq!(output[1]["type"], "custom_tool_call");
        assert_eq!(output[1]["input"], "p");
        assert!(output[1]["id"].as_str().unwrap().starts_with("ctc_"));
    }

    /// A bare `[DONE]` or a usage-only stream carries nothing a response
    /// is made of; only content, reasoning, a tool call or a finish reason
    /// counts.
    #[test]
    fn only_a_renderable_chunk_counts_as_output() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let usage_only = ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: None,
            usage: Some(UsageStats::new(10, 0)),
        };
        assert!(enc.next_events(&usage_only).is_empty());
        assert!(!enc.has_output());
        let _ = enc.next_events(&content_chunk("hi"));
        assert!(enc.has_output());
    }

    #[test]
    fn tool_call_finish_without_usage_then_force_finish_does_not_double_close() {
        // Finish chunk lacks usage → done events emitted, completed withheld.
        // force_finish must NOT re-emit the per-item done events.
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let _ = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![json!({
                    "index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                })]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        });
        let at_finish = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
        });
        assert_eq!(
            types_of(&at_finish)
                .iter()
                .filter(|t| **t == "response.output_item.done")
                .count(),
            1
        );
        assert!(!enc.is_finished());
        let tail = enc.force_finish();
        // The trailing close emits only response.completed, not a second
        // round of done events.
        assert_eq!(types_of(&tail), vec!["response.completed"]);
    }

    /// `total_tokens` is echoed when nothing was converted and
    /// recomputed when something was — the two halves of one rule, so
    /// they are pinned together.
    ///
    /// Echoing unconditionally is how AISIX-Cloud#1447 reached the wire:
    /// an Anthropic upstream's total folds in cache tokens that its
    /// `input_tokens` excludes, so a client projected into OpenAI
    /// accounting read `40 + 10 = 150`. Recomputing unconditionally is
    /// the opposite mistake — it silently corrects away whatever a
    /// provider counted outside `prompt + completion`.
    #[test]
    fn streaming_total_tokens_is_echoed_unless_the_shape_converted() {
        fn closing_usage(usage: UsageStats) -> Value {
            let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
            let _ = enc.next_events(&content_chunk("hi"));
            let done = enc.next_events(&ChatChunk {
                id: "c".into(),
                model: "m".into(),
                delta: ChatDelta::default(),
                finish_reason: Some(FinishReason::Stop),
                usage: Some(usage),
            });
            done.last().unwrap().data["response"]["usage"].clone()
        }

        // No conversion: the provider's own total stands even though it
        // exceeds input + output.
        let echoed = closing_usage(UsageStats {
            prompt_tokens: 5,
            completion_tokens: 2,
            total_tokens: 11,
            ..Default::default()
        });
        assert_eq!(echoed["input_tokens"], 5);
        assert_eq!(echoed["output_tokens"], 2);
        assert_eq!(echoed["total_tokens"], 11);

        // Converted: the stored total was built under the upstream's own
        // accounting, so it is rebuilt from the projected fields.
        let converted = closing_usage(UsageStats::with_cache(40, 10, 30, 70));
        assert_eq!(converted["input_tokens"], 140);
        assert_eq!(converted["output_tokens"], 10);
        assert_eq!(converted["total_tokens"], 150);
        assert_eq!(converted["input_tokens_details"]["cached_tokens"], 70);
        assert_eq!(
            converted["input_tokens_details"]["cache_creation_tokens"],
            30
        );
    }

    #[test]
    fn streaming_length_finish_emits_incomplete_with_reason() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let _ = enc.next_events(&content_chunk("partial"));
        let done = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::Length),
            usage: Some(UsageStats::new(3, 9)),
        });
        let completed = done.last().unwrap();
        assert_eq!(completed.event_type, "response.incomplete");
        assert_eq!(completed.data["response"]["status"], "incomplete");
        assert_eq!(
            completed.data["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }

    /// `(event, data)` of every SSE event in a relay's output.
    fn sse_events(frames: &[u8]) -> Vec<(String, Value)> {
        std::str::from_utf8(frames)
            .unwrap()
            .split("\n\n")
            .filter(|e| !e.is_empty())
            .map(|e| {
                let (event, data) = e
                    .strip_prefix("event: ")
                    .and_then(|r| r.split_once("\ndata: "))
                    .expect("an `event:` line followed by one `data:` line");
                (
                    event.to_string(),
                    serde_json::from_str(data).expect("one JSON document"),
                )
            })
            .collect()
    }

    /// The Responses API defines its `error` event FLAT, discriminated on
    /// the top-level `type`, and the official SDKs parse it that way —
    /// `openai-python`'s `ResponseErrorEvent` is
    /// `{type, code, message, param, sequence_number}`. Nesting it under an
    /// `error` object (which is what every other SSE error in this crate
    /// does) would hand a `responses.stream()` client an event it cannot
    /// classify. Both of this relay's failure kinds are checked, because a
    /// client should not have to know which failure it hit.
    #[test]
    fn both_sse_error_frames_match_the_responses_api_error_event() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        for _ in 0..7 {
            enc.take_sequence_number();
        }
        let block = sse_events(&guardrail_failure_frames(
            &mut enc,
            true,
            true,
            Some("gr-block"),
            None,
        ));
        assert_eq!(block.len(), 1, "a terminal event already went out");
        let (event, block) = &block[0];
        assert_eq!(event, "error");
        assert_eq!(block["type"], "error");
        assert_eq!(block["code"], "content_filter");
        assert!(block["message"].as_str().unwrap().contains("gr-block"));
        assert_eq!(block["param"], serde_json::Value::Null);
        assert_eq!(block["sequence_number"], 7);

        // The upstream-failure frame on the same stream, same envelope. Its
        // message is JSON-escaped through serde rather than interpolated.
        let upstream = sse_events(&failure_frames(
            &mut enc,
            true,
            true,
            "upstream_error",
            "upstream_error",
            "boom \"quoted\"",
        ));
        let (_, upstream) = &upstream[0];
        assert_eq!(upstream["type"], "error");
        assert_eq!(upstream["code"], "upstream_error");
        assert_eq!(upstream["message"], "boom \"quoted\"");
        assert_eq!(upstream["sequence_number"], 8);

        // Exactly the SDK's field set, and nothing nested: an `error` key
        // here is the shape this deliberately does NOT use.
        for v in [block, upstream] {
            assert!(v.get("error").is_none());
            let keys: std::collections::BTreeSet<&str> =
                v.as_object().unwrap().keys().map(String::as_str).collect();
            assert_eq!(
                keys,
                ["code", "message", "param", "sequence_number", "type"]
                    .into_iter()
                    .collect::<std::collections::BTreeSet<_>>(),
            );
        }
    }

    /// The flat `error` frame is followed by `response.failed`, the event a
    /// client reads a failure from, numbered after it. A guardrail stop
    /// reports `invalid_prompt` there — re-running the turn meets the same
    /// guardrail — while the flat frame keeps its own code.
    #[test]
    fn a_failure_is_followed_by_response_failed_carrying_the_error() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let _ = enc.next_events(&content_chunk("partial"));
        let events = sse_events(&guardrail_failure_frames(
            &mut enc,
            true,
            false,
            Some("gr-block"),
            None,
        ));
        assert_eq!(
            events.iter().map(|(e, _)| e.as_str()).collect::<Vec<_>>(),
            ["error", "response.failed"]
        );
        let (error, failed) = (&events[0].1, &events[1].1);
        assert_eq!(error["code"], "content_filter");
        assert_eq!(failed["type"], "response.failed");
        assert_eq!(
            failed["sequence_number"].as_u64().unwrap(),
            error["sequence_number"].as_u64().unwrap() + 1
        );
        let response = &failed["response"];
        assert_eq!(response["id"], "resp_1");
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "invalid_prompt");
        assert_eq!(response["error"]["message"], error["message"]);
        // The output that was held back is not reported on a failure.
        assert_eq!(response["output"], json!([]));
        assert_eq!(response["usage"], Value::Null);
        assert!(enc.is_finished());
    }

    fn in_band_error(code: Option<&str>, kind: Option<&str>) -> aisix_gateway::BridgeError {
        aisix_gateway::BridgeError::UpstreamInBand {
            status: None,
            message: "upstream said no".into(),
            parsed: Some(Box::new(aisix_gateway::UpstreamErrorView {
                kind: kind.map(str::to_string),
                message: Some("upstream said no".into()),
                code: code.map(str::to_string),
                param: None,
            })),
            wire: aisix_gateway::UpstreamWire::Unknown,
        }
    }

    /// The codes a client acts on specifically pass through from the
    /// upstream's own in-band error; anything else reports the flat
    /// frame's code.
    #[test]
    fn response_failed_passes_through_the_codes_clients_act_on() {
        for code in [
            "context_length_exceeded",
            "insufficient_quota",
            "rate_limit_exceeded",
            "server_is_overloaded",
            "slow_down",
        ] {
            assert_eq!(upstream_failed_code(&in_band_error(Some(code), None)), code);
            assert_eq!(upstream_failed_code(&in_band_error(None, Some(code))), code);
        }
        assert_eq!(
            upstream_failed_code(&in_band_error(Some("1302"), Some("some_vendor_error"))),
            "upstream_in_band_error"
        );
        assert_eq!(
            upstream_failed_code(&aisix_gateway::BridgeError::Transport("eof".into())),
            "transport_error"
        );
    }

    // ── Reasoning on the bridged path ────────────────────────────

    fn reasoning_chunk(text: &str) -> ChatChunk {
        ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                reasoning_content: Some(text.into()),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        }
    }

    fn finish_chunk(usage: Option<UsageStats>) -> ChatChunk {
        ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::Stop),
            usage,
        }
    }

    /// A chat upstream that streamed nothing but its chain-of-thought still
    /// owes the client a complete `reasoning` item — opened, summarised,
    /// closed — rather than an empty response.
    #[test]
    fn streaming_reasoning_only_emits_a_closed_reasoning_item() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let mut all = enc.next_events(&reasoning_chunk("think"));
        all.extend(enc.next_events(&reasoning_chunk("ing")));
        all.extend(enc.next_events(&finish_chunk(Some(UsageStats::new(4, 6)))));
        assert_eq!(
            types_of(&all),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let added = &all[2];
        assert_eq!(added.data["output_index"], 0);
        assert_eq!(added.data["item"]["type"], "reasoning");
        let item_id = added.data["item"]["id"].as_str().unwrap().to_string();
        assert!(item_id.starts_with("rs_"), "reasoning ids are rs_-prefixed");
        assert_eq!(all[3].data["item_id"], item_id);
        assert_eq!(all[3].data["summary_index"], 0);
        assert_eq!(all[3].data["part"]["type"], "summary_text");
        assert_eq!(all[4].data["delta"], "think");
        assert_eq!(all[6].data["text"], "thinking");
        assert_eq!(all[7].data["part"]["text"], "thinking");
        assert_eq!(all[8].data["item"]["summary"][0]["text"], "thinking");
        // …and it is the only item in the completed response.
        let output = &all[9].data["response"]["output"];
        assert_eq!(output.as_array().unwrap().len(), 1);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["summary"][0]["text"], "thinking");
        // Sequence numbers keep counting across the reasoning events.
        let seqs: Vec<u64> = all
            .iter()
            .map(|e| e.data["sequence_number"].as_u64().unwrap())
            .collect();
        assert_eq!(seqs, (0..all.len() as u64).collect::<Vec<_>>());
    }

    /// Reasoning, then prose, then a tool call: each opens at the NEXT
    /// output_index, and the reasoning item is closed before the message
    /// item opens.
    #[test]
    fn streaming_reasoning_then_content_then_tool_call_advances_output_index() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let mut all = enc.next_events(&reasoning_chunk("why"));
        all.extend(enc.next_events(&content_chunk("because")));
        all.extend(enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![json!({
                    "index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                })]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        }));
        all.extend(enc.next_events(&finish_chunk(Some(UsageStats::new(1, 2)))));
        assert_eq!(
            types_of(&all),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added", // reasoning
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done", // closed by the content delta
                "response.reasoning_summary_part.done",
                "response.output_item.done",  // reasoning
                "response.output_item.added", // message
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_item.added", // function_call
                "response.function_call_arguments.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done", // message
                "response.function_call_arguments.done",
                "response.output_item.done", // function_call
                "response.completed",
            ]
        );
        assert_eq!(all[2].data["output_index"], 0, "reasoning leads");
        assert_eq!(all[8].data["output_index"], 1, "message follows it");
        assert_eq!(all[11].data["output_index"], 2, "then the tool call");
        let output = &all.last().unwrap().data["response"]["output"];
        let kinds: Vec<&str> = output
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, vec!["reasoning", "message", "function_call"]);
    }

    /// Reasoning that arrives after a message item is already open opens a
    /// SECOND reasoning item at the next output_index — the already-open
    /// message item keeps its own index and its own accumulated text.
    #[test]
    fn streaming_reasoning_after_content_opens_a_further_reasoning_item() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let mut all = enc.next_events(&content_chunk("first"));
        all.extend(enc.next_events(&reasoning_chunk("second thoughts")));
        all.extend(enc.next_events(&finish_chunk(Some(UsageStats::new(1, 1)))));
        let reasoning_added: Vec<&ResponsesSseEvent> = all
            .iter()
            .filter(|e| {
                e.event_type == "response.output_item.added"
                    && e.data["item"]["type"] == "reasoning"
            })
            .collect();
        assert_eq!(reasoning_added.len(), 1);
        assert_eq!(
            reasoning_added[0].data["output_index"], 1,
            "the message item kept index 0; reasoning takes the next one",
        );
        let output = &all.last().unwrap().data["response"]["output"];
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["content"][0]["text"], "first");
        assert_eq!(output[1]["type"], "reasoning");
        assert_eq!(output[1]["summary"][0]["text"], "second thoughts");
    }

    /// A bridged non-streaming response surfaces the upstream's
    /// chain-of-thought as a `reasoning` item ahead of the message item.
    #[test]
    fn non_streaming_reasoning_becomes_a_leading_reasoning_item() {
        let mut resp = chat_response_with(Some("42"), None, FinishReason::Stop);
        resp.message
            .extra
            .insert("reasoning_content".into(), json!("6 times 7"));
        let out = chat_response_to_responses_json(&resp, "m", 1, &no_custom_tools());
        assert_eq!(out["output"][0]["type"], "reasoning");
        assert!(out["output"][0]["id"].as_str().unwrap().starts_with("rs_"));
        assert_eq!(out["output"][0]["summary"][0]["type"], "summary_text");
        assert_eq!(out["output"][0]["summary"][0]["text"], "6 times 7");
        assert_eq!(out["output"][1]["type"], "message");
        assert_eq!(out["output"][1]["content"][0]["text"], "42");
    }

    /// An upstream that reported no reasoning adds no `reasoning` item —
    /// an empty one would render as a blank thinking block.
    #[test]
    fn non_streaming_without_reasoning_emits_no_reasoning_item() {
        let resp = chat_response_with(Some("42"), None, FinishReason::Stop);
        let out = chat_response_to_responses_json(&resp, "m", 1, &no_custom_tools());
        assert_eq!(out["output"][0]["type"], "message");
        assert!(out["output"]
            .as_array()
            .unwrap()
            .iter()
            .all(|i| i["type"] != "reasoning"));
    }

    /// The guardrail scan text is unchanged by reasoning: generated
    /// reasoning is out of output-guardrail scope on every /v1/responses
    /// path, so it must not reach the assembled assistant message.
    #[test]
    fn assembled_assistant_message_excludes_reasoning() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let _ = enc.next_events(&reasoning_chunk("SECRET"));
        let _ = enc.next_events(&content_chunk("visible"));
        let (text, tool_calls) = enc.assembled_assistant_message();
        assert_eq!(text, "visible");
        assert!(tool_calls.is_empty());
    }

    /// A stream whose upstream never sent a usage frame must report the
    /// SAME numbers to the client as the usage record gets — the encoder
    /// adopts the local estimate before the synthesized terminal event.
    #[test]
    fn force_finish_reports_the_estimate_the_usage_record_gets() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let _ = enc.next_events(&content_chunk("hello"));
        let _ = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        });
        // …the trailing usage frame never arrives; the relay hands the
        // encoder the same estimate it wrote to the usage record.
        enc.set_estimated_usage(11, 7);
        let events = enc.force_finish();
        let completed = events.last().unwrap();
        assert_eq!(completed.event_type, "response.completed");
        let usage = &completed.data["response"]["usage"];
        assert_eq!(usage["input_tokens"], 11);
        assert_eq!(usage["output_tokens"], 7);
        assert_eq!(usage["total_tokens"], 18);
        // Standard usage shape only — nothing tells the client it is an
        // estimate.
        let keys: std::collections::BTreeSet<&str> = usage
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "input_tokens",
                "input_tokens_details",
                "output_tokens",
                "output_tokens_details",
                "total_tokens",
            ]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>(),
        );
    }

    /// The estimate never overwrites what the upstream actually reported.
    /// The relay hands the encoder an estimate whenever it computed one, so
    /// a usage frame that landed WITHOUT a finish chunk — the shape an
    /// OpenAI-compatible upstream sends — must still win at force_finish.
    #[test]
    fn a_usage_frame_reporting_only_one_counter_still_gets_the_other_filled() {
        // A relay that streams `{prompt_tokens: 3, completion_tokens: 0}`
        // used to block the whole estimate, so the client read
        // `output_tokens: 0` while the usage record — which fills per
        // counter — billed the estimate.
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let _ = enc.next_events(&content_chunk("hi"));
        let mut partial = UsageStats::new(3, 0);
        partial.total_tokens = 3;
        let _ = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: None,
            usage: Some(partial),
        });
        enc.set_estimated_usage(3, 7);
        let events = enc.force_finish();
        let usage = &events.last().unwrap().data["response"]["usage"];
        assert_eq!(usage["input_tokens"], 3, "the reported counter stands");
        assert_eq!(usage["output_tokens"], 7, "the zero was filled");
        // The total the frame carried described the pre-fill counters.
        assert_eq!(usage["total_tokens"], 10);
    }

    #[test]
    fn set_estimated_usage_is_ignored_once_a_usage_frame_landed() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let _ = enc.next_events(&content_chunk("hi"));
        let _ = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: None,
            usage: Some(UsageStats::new(3, 4)),
        });
        enc.set_estimated_usage(99, 99);
        let events = enc.force_finish();
        let usage = &events.last().unwrap().data["response"]["usage"];
        assert_eq!(usage["input_tokens"], 3, "the frame was read, not guessed");
        assert_eq!(usage["output_tokens"], 4);
    }

    // ── Replayed reasoning ───────────────────────────────────────

    fn reasoning_item(summary: &str) -> Value {
        json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": summary}],
            "content": null,
            "encrypted_content": null,
        })
    }

    fn reasoning_of(m: &ChatMessage) -> Option<&str> {
        m.extra.get("reasoning_content").and_then(Value::as_str)
    }

    /// The Codex turn shape: the reasoning that produced an answer rides the
    /// answer's own assistant message as `reasoning_content`, never as
    /// visible content and never as a turn of its own.
    #[test]
    fn replayed_reasoning_rides_the_assistant_message_that_follows_it() {
        let chat = responses_request_to_chat(
            "m",
            &json!({"input": [
                {"role": "user", "content": "hi"},
                reasoning_item("the user greets me"),
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
            ]}),
        );
        assert_eq!(chat.messages.len(), 2);
        assert!(matches!(chat.messages[1].role, Role::Assistant));
        assert_eq!(chat.messages[1].content_str(), "hello");
        assert_eq!(reasoning_of(&chat.messages[1]), Some("the user greets me"));
    }

    /// Reasoning that led to tool calls rides the one assistant message that
    /// carries them, parallel calls included.
    #[test]
    fn replayed_reasoning_rides_the_tool_calls_it_led_to() {
        let chat = responses_request_to_chat(
            "m",
            &json!({"input": [
                {"role": "user", "content": "list and read"},
                reasoning_item("run two commands"),
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "a"},
                {"type": "function_call_output", "call_id": "c2", "output": "b"},
            ]}),
        );
        assert_eq!(chat.messages.len(), 4);
        let assistant = &chat.messages[1];
        assert_eq!(reasoning_of(assistant), Some("run two commands"));
        assert!(assistant.content.is_none());
        assert_eq!(assistant.extra["tool_calls"].as_array().unwrap().len(), 2);
        assert!(matches!(chat.messages[2].role, Role::Tool));
    }

    /// The Codex turn shape with tools: the model's text, then the calls it
    /// made in the same turn. They go back as ONE assistant message
    /// carrying both, with the turn's reasoning on it — the shape the model
    /// produced them in.
    #[test]
    fn assistant_text_and_the_calls_that_follow_it_are_one_turn() {
        let chat = responses_request_to_chat(
            "m",
            &json!({"input": [
                {"role": "user", "content": "list files"},
                reasoning_item("run ls"),
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Running ls."}]},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "a"},
                {"type": "function_call_output", "call_id": "c2", "output": "b"},
            ]}),
        );
        assert_eq!(chat.messages.len(), 4);
        let turn = &chat.messages[1];
        assert_eq!(turn.content_str(), "Running ls.");
        assert_eq!(reasoning_of(turn), Some("run ls"));
        let ids: Vec<&str> = turn.extra["tool_calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tc| tc["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["c1", "c2"]);
    }

    /// Consecutive reasoning items join in order; `content` text wins over
    /// the summary, `encrypted_content` is never read, and an item with no
    /// readable text replays nothing.
    #[test]
    fn replayed_reasoning_text_prefers_content_and_joins_consecutive_items() {
        let chat = responses_request_to_chat(
            "m",
            &json!({"input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "summary"}],
                 "content": [{"type": "reasoning_text", "text": " full thought "}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "a"}, {"type": "summary_text", "text": "b"}]},
                {"type": "reasoning", "summary": [], "encrypted_content": "ciphertext"},
                {"role": "assistant", "content": "done"},
            ]}),
        );
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(reasoning_of(&chat.messages[0]), Some("full thought\na\nb"));
        assert!(!serde_json::to_string(&chat.messages)
            .unwrap()
            .contains("ciphertext"));
    }

    /// A bare-string `content` is not a shape the input mask rewrites, so it
    /// is never replayed; the summary is used instead.
    #[test]
    fn a_bare_string_reasoning_content_is_not_replayed() {
        let chat = responses_request_to_chat(
            "m",
            &json!({"input": [
                {"type": "reasoning", "content": "raw a@x.com", "summary": [{"type": "summary_text", "text": "masked"}]},
                {"role": "assistant", "content": "done"},
            ]}),
        );
        assert_eq!(reasoning_of(&chat.messages[0]), Some("masked"));
    }

    /// Reasoning that no assistant message follows is still passed back, as
    /// an assistant message of its own ahead of the turn that came next.
    #[test]
    fn replayed_reasoning_without_a_following_answer_stays_its_own_message() {
        let chat = responses_request_to_chat(
            "m",
            &json!({"input": [
                {"role": "user", "content": "q1"},
                reasoning_item("thinking, then interrupted"),
                {"role": "user", "content": "q2"},
            ]}),
        );
        assert_eq!(chat.messages.len(), 3);
        assert!(matches!(chat.messages[1].role, Role::Assistant));
        assert!(chat.messages[1].content.is_none());
        assert_eq!(
            reasoning_of(&chat.messages[1]),
            Some("thinking, then interrupted")
        );
        assert!(matches!(chat.messages[2].role, Role::User));
        // Bridges with no slot for it skip exactly this message.
        assert!(chat.messages[1].is_reasoning_only());
    }

    // ── Namespace tools ──────────────────────────────────────────

    fn multi_agent_namespace() -> Value {
        json!({
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Tools for spawning and managing sub-agents.",
            "tools": [
                {"type": "function", "name": "spawn_agent", "description": "Spawn one.",
                 "strict": false, "parameters": {"type": "object", "properties": {"task": {"type": "string"}}}},
                {"type": "function", "name": "close_agent",
                 "parameters": {"type": "object", "properties": {"target": {"type": "string"}}}},
                {"type": "custom", "name": "not_a_function"},
            ],
        })
    }

    #[test]
    fn namespace_function_sub_tools_flatten_into_prefixed_chat_tools() {
        let chat = responses_request_to_chat(
            "m",
            &json!({"input": "go", "tools": [
                {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
                multi_agent_namespace(),
                {"type": "web_search", "external_web_access": true},
            ]}),
        );
        let tools = chat.extra["tools"].as_array().unwrap();
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "exec_command",
                "multi_agent_v1__spawn_agent",
                "multi_agent_v1__close_agent"
            ]
        );
        assert_eq!(
            tools[1]["function"]["description"],
            "Tools for spawning and managing sub-agents.\n\nSpawn one."
        );
        assert_eq!(
            tools[2]["function"]["description"],
            "Tools for spawning and managing sub-agents."
        );
        assert_eq!(
            tools[1]["function"]["parameters"]["properties"]["task"]["type"],
            "string"
        );
    }

    /// A sub-tool whose flattened name a top-level function already has is
    /// not offered twice, and a call to that name stays the top-level one.
    #[test]
    fn a_flattened_name_taken_by_a_top_level_function_is_not_offered_twice() {
        let body = json!({"input": "go", "tools": [
            {"type": "function", "name": "ns__a"},
            {"type": "namespace", "name": "ns", "tools": [{"type": "function", "name": "a"}]},
        ]});
        let chat = responses_request_to_chat("m", &body);
        assert_eq!(chat.extra["tools"].as_array().unwrap().len(), 1);
        let reply = ResponsesReplyContext::from_request(&body);
        assert!(matches!(
            reply.tool_call("ns__a"),
            ReplyToolCall::Function {
                name: "ns__a",
                namespace: None
            }
        ));
        // The bare name is unambiguous, so it still maps to the sub-tool.
        assert!(matches!(
            reply.tool_call("a"),
            ReplyToolCall::Function {
                name: "a",
                namespace: Some("ns")
            }
        ));
    }

    /// A replayed call to a namespace sub-tool goes back under the name the
    /// model was offered it as.
    #[test]
    fn a_replayed_namespace_call_uses_the_flattened_name() {
        let chat = responses_request_to_chat(
            "m",
            &json!({"input": [
                {"type": "function_call", "call_id": "c1", "name": "spawn_agent",
                 "namespace": "multi_agent_v1", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            ]}),
        );
        assert_eq!(
            chat.messages[0].extra["tool_calls"][0]["function"]["name"],
            "multi_agent_v1__spawn_agent"
        );
    }

    /// A forced choice of a namespace sub-tool names the tool the model
    /// was actually offered.
    #[test]
    fn a_forced_namespace_choice_names_the_flattened_tool() {
        let chat = responses_request_to_chat(
            "m",
            &json!({"input": "go", "tools": [multi_agent_namespace()],
                    "tool_choice": {"type": "function", "name": "spawn_agent", "namespace": "multi_agent_v1"}}),
        );
        assert_eq!(
            chat.extra["tool_choice"],
            json!({"type": "function", "function": {"name": "multi_agent_v1__spawn_agent"}})
        );
    }

    /// The request echo on a lifecycle event is not charged against an
    /// output guardrail's hold-back budget: an agent's instructions and
    /// tools are tens of kilobytes, repeated on three events per stream.
    #[test]
    fn the_request_echo_is_not_charged_to_the_hold_back_budget() {
        let cost_of_created = |request: Value| {
            let mut enc = ResponsesSseEncoder::new(
                "resp_1",
                "m",
                0,
                ResponsesReplyContext::from_request(&request),
            );
            let events = enc.next_events(&content_chunk("hi"));
            let created = &events[0];
            assert_eq!(created.event_type, "response.created");
            enc.buffer_cost(created, created.to_sse_string().len())
        };
        let small = cost_of_created(json!({}));
        let large = cost_of_created(json!({"instructions": "x".repeat(50_000)}));
        assert_eq!(small, large);
        // An output event is charged in full.
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let events = enc.next_events(&content_chunk("hi"));
        let delta = events
            .iter()
            .find(|e| e.event_type == "response.output_text.delta")
            .unwrap();
        let len = delta.to_sse_string().len();
        assert_eq!(enc.buffer_cost(delta, len), len);
    }

    fn namespace_reply() -> ResponsesReplyContext {
        ResponsesReplyContext::from_request(&json!({"tools": [multi_agent_namespace()]}))
    }

    #[test]
    fn non_streaming_namespace_call_returns_the_sub_tool_and_its_namespace() {
        let tcs = json!([
            {"id": "c1", "type": "function", "function": {"name": "multi_agent_v1__spawn_agent", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "close_agent", "arguments": "{}"}},
            {"id": "c3", "type": "function", "function": {"name": "shell", "arguments": "{}"}},
        ]);
        let resp = chat_response_with(None, Some(tcs), FinishReason::ToolCalls);
        let out = chat_response_to_responses_json(&resp, "m", 1, &namespace_reply());
        let output = out["output"].as_array().unwrap();
        assert_eq!(output[0]["name"], "spawn_agent");
        assert_eq!(output[0]["namespace"], "multi_agent_v1");
        // A model answering with the bare, unambiguous name is understood.
        assert_eq!(output[1]["name"], "close_agent");
        assert_eq!(output[1]["namespace"], "multi_agent_v1");
        // Anything else is an ordinary function call with no namespace.
        assert_eq!(output[2]["name"], "shell");
        assert!(output[2].get("namespace").is_none());
    }

    #[test]
    fn streaming_namespace_call_announces_and_closes_the_sub_tool() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, namespace_reply());
        let mut events = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![json!({
                    "index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "multi_agent_v1__spawn_agent", "arguments": "{\"task\":"},
                })]),
                ..Default::default()
            },
            finish_reason: None,
            usage: None,
        });
        events.extend(enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta {
                tool_calls: Some(vec![
                    json!({"index": 0, "function": {"arguments": "\"x\"}"}}),
                ]),
                ..Default::default()
            },
            finish_reason: Some(FinishReason::ToolCalls),
            usage: Some(UsageStats::new(3, 2)),
        }));
        let item_of = |t: &str| {
            events
                .iter()
                .find(|e| e.event_type == t)
                .map(|e| e.data["item"].clone())
                .unwrap()
        };
        for item in [
            item_of("response.output_item.added"),
            item_of("response.output_item.done"),
        ] {
            assert_eq!(item["type"], "function_call");
            assert_eq!(item["name"], "spawn_agent");
            assert_eq!(item["namespace"], "multi_agent_v1");
        }
        assert_eq!(
            item_of("response.output_item.done")["arguments"],
            "{\"task\":\"x\"}"
        );
        let completed = &events.last().unwrap().data["response"]["output"][0];
        assert_eq!(completed["name"], "spawn_agent");
        assert_eq!(completed["namespace"], "multi_agent_v1");
    }

    // ── Full Response object ─────────────────────────────────────

    /// Every top-level member the Responses API requires of a Response.
    const RESPONSE_MEMBERS: [&str; 30] = [
        "id",
        "object",
        "created_at",
        "completed_at",
        "status",
        "incomplete_details",
        "model",
        "previous_response_id",
        "instructions",
        "output",
        "error",
        "tools",
        "tool_choice",
        "truncation",
        "parallel_tool_calls",
        "text",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "top_logprobs",
        "temperature",
        "reasoning",
        "usage",
        "max_output_tokens",
        "max_tool_calls",
        "store",
        "background",
        "service_tier",
        "metadata",
        "safety_identifier",
    ];

    fn assert_full_response(r: &Value) {
        for key in RESPONSE_MEMBERS.iter().chain(["prompt_cache_key"].iter()) {
            assert!(r.get(*key).is_some(), "Response object lacks `{key}`: {r}");
        }
    }

    /// A Response object reports the request's own settings, and the API's
    /// default where the request left one out.
    #[test]
    fn the_response_object_echoes_the_request_and_defaults_the_rest() {
        let request = json!({
            "model": "m",
            "instructions": "be terse",
            "input": "hi",
            "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}, multi_agent_namespace()],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "temperature": 0.2,
            "max_output_tokens": 64,
            "reasoning": {"effort": "high", "summary": "auto"},
            "store": false,
            "prompt_cache_key": "session-1",
            "text": null,
            "previous_response_id": null,
        });
        let resp = chat_response_with(Some("hello"), None, FinishReason::Stop);
        let out = chat_response_to_responses_json(
            &resp,
            "m",
            1,
            &ResponsesReplyContext::from_request(&request),
        );
        assert_full_response(&out);
        assert_eq!(out["instructions"], "be terse");
        assert_eq!(out["parallel_tool_calls"], false);
        assert_eq!(out["temperature"], 0.2);
        assert_eq!(out["max_output_tokens"], 64);
        assert_eq!(
            out["reasoning"],
            json!({"effort": "high", "summary": "auto"})
        );
        assert_eq!(out["store"], false);
        assert_eq!(out["prompt_cache_key"], "session-1");
        // A function tool carries every member the API defines for it; any
        // other tool is echoed as sent.
        assert_eq!(
            out["tools"][0],
            json!({"type": "function", "name": "f", "parameters": {"type": "object"},
                   "description": null, "strict": null})
        );
        assert_eq!(out["tools"][1], multi_agent_namespace());
        // Defaults for what the request left out, or sent as null.
        assert_eq!(out["text"], json!({"format": {"type": "text"}}));
        assert_eq!(out["truncation"], "disabled");
        assert_eq!(out["top_p"], 1.0);
        assert_eq!(out["previous_response_id"], Value::Null);
        assert_eq!(out["metadata"], json!({}));
        assert_eq!(out["error"], Value::Null);
        assert_eq!(out["incomplete_details"], Value::Null);
        assert!(out["completed_at"].as_i64().is_some());
    }

    /// Every lifecycle event carries the full object too, `usage: null`
    /// until the response has finished.
    #[test]
    fn every_lifecycle_event_carries_the_full_response_object() {
        let mut enc = ResponsesSseEncoder::new(
            "resp_1",
            "m",
            0,
            ResponsesReplyContext::from_request(&json!({"instructions": "sys"})),
        );
        let mut events = enc.next_events(&content_chunk("hi"));
        events.extend(enc.next_events(&finish_chunk(Some(UsageStats::new(3, 1)))));
        let lifecycle: Vec<&ResponsesSseEvent> = events
            .iter()
            .filter(|e| e.data.get("response").is_some())
            .collect();
        assert_eq!(
            lifecycle.iter().map(|e| e.event_type).collect::<Vec<_>>(),
            [
                "response.created",
                "response.in_progress",
                "response.completed"
            ]
        );
        for e in &lifecycle {
            assert_full_response(&e.data["response"]);
            assert_eq!(e.data["response"]["instructions"], "sys");
        }
        assert_eq!(lifecycle[0].data["response"]["usage"], Value::Null);
        assert_eq!(lifecycle[0].data["response"]["completed_at"], Value::Null);
        assert_eq!(lifecycle[2].data["response"]["usage"]["input_tokens"], 3);

        let failed = ResponsesSseEncoder::new("resp_2", "m", 0, no_custom_tools())
            .failed_event("upstream_error", "boom");
        assert_full_response(&failed.data["response"]);
    }

    // ── Stream ends ──────────────────────────────────────────────

    /// Run `chunks` through the relay with no guardrail and return its SSE
    /// events plus the end-of-stream usage record.
    async fn relay(
        chunks: Vec<Result<ChatChunk, aisix_gateway::BridgeError>>,
    ) -> (Vec<(String, Value)>, ResponsesStreamCompletion) {
        let (tx, rx) = std::sync::mpsc::channel();
        let body = build_responses_bridge_stream(
            Box::pin(futures::stream::iter(chunks)),
            ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools()),
            Instant::now(),
            Instant::now(),
            None,
            false,
            usize::MAX,
            "m".to_string(),
            None,
            None,
            move |comp| tx.send(comp).unwrap(),
        );
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        (sse_events(&bytes), rx.recv().unwrap())
    }

    fn event_names(events: &[(String, Value)]) -> Vec<&str> {
        events.iter().map(|(e, _)| e.as_str()).collect()
    }

    /// A stream that ends before carrying any piece of a response — here a
    /// usage-only frame — fails with a retryable code instead of completing
    /// with an empty output. Nothing went out before the failure, so it
    /// opens the stream with `response.created` + `response.in_progress`.
    #[tokio::test]
    async fn an_empty_upstream_stream_fails_instead_of_completing() {
        let usage_only = ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: None,
            usage: Some(UsageStats::new(10, 0)),
        };
        let (events, comp) = relay(vec![Ok(usage_only)]).await;
        assert_eq!(event_names(&events), FAILED_BEFORE_ANY_OUTPUT);
        assert_opened_then_failed(&events);
        assert_eq!(events[2].1["code"], "upstream_error");
        let failed = &events[3].1["response"];
        assert_eq!(failed["status"], "failed");
        assert_eq!(failed["error"]["code"], "upstream_error");
        assert_eq!(failed["error"]["message"], events[2].1["message"]);
        // Accounted like any upstream failure mid-relay.
        assert!(!comp.reached_end);

        let (events, _) = relay(Vec::new()).await;
        assert_eq!(event_names(&events), FAILED_BEFORE_ANY_OUTPUT);
        assert_opened_then_failed(&events);
    }

    const FAILED_BEFORE_ANY_OUTPUT: [&str; 4] = [
        "response.created",
        "response.in_progress",
        "error",
        "response.failed",
    ];

    /// The opening pair carries the same in-progress Response the failure
    /// then ends, and the sequence runs 0..n without a gap.
    fn assert_opened_then_failed(events: &[(String, Value)]) {
        for (i, (_, data)) in events.iter().enumerate() {
            assert_eq!(data["sequence_number"], i as u64);
        }
        for (_, data) in &events[..2] {
            assert_eq!(data["response"]["id"], "resp_1");
            assert_eq!(data["response"]["status"], "in_progress");
            assert_eq!(data["response"]["output"], json!([]));
        }
        assert_eq!(events[3].1["response"]["id"], "resp_1");
        assert_eq!(events[3].1["response"]["status"], "failed");
    }

    /// An upstream failure before the first chunk is also the first thing
    /// the client receives, so it opens the stream the same way.
    #[tokio::test]
    async fn a_transport_error_before_any_chunk_opens_the_stream_then_fails() {
        let (events, _) = relay(vec![Err(aisix_gateway::BridgeError::Transport(
            "connection reset".into(),
        ))])
        .await;
        assert_eq!(event_names(&events), FAILED_BEFORE_ANY_OUTPUT);
        assert_opened_then_failed(&events);
        assert_eq!(events[2].1["code"], "transport_error");
    }

    /// A stream that produced content and then simply ended keeps
    /// completing: the caller has an answer, only its finish was not
    /// reported.
    #[tokio::test]
    async fn a_stream_with_content_but_no_finish_still_completes() {
        let (events, comp) = relay(vec![Ok(content_chunk("partial"))]).await;
        assert_eq!(event_names(&events).last(), Some(&"response.completed"));
        assert!(comp.reached_end);
    }

    /// The upstream sent its finish reason and the connection then dropped
    /// before the usage frame: nothing the caller asked for is missing, so
    /// the response completes rather than fails.
    #[tokio::test]
    async fn a_transport_error_after_the_finish_reason_completes_the_response() {
        let (events, comp) = relay(vec![
            Ok(content_chunk("whole answer")),
            Ok(finish_chunk(None)),
            Err(aisix_gateway::BridgeError::Transport(
                "unexpected EOF".into(),
            )),
        ])
        .await;
        let names = event_names(&events);
        assert_eq!(names.last(), Some(&"response.completed"));
        assert!(!names.contains(&"error"));
        assert!(comp.reached_end);
    }

    /// The same drop BEFORE the finish reason is a failure: the flat error
    /// frame, then `response.failed` carrying the same code and message.
    #[tokio::test]
    async fn a_transport_error_before_the_finish_reason_fails_the_response() {
        let (events, _) = relay(vec![
            Ok(content_chunk("half an ans")),
            Err(aisix_gateway::BridgeError::Transport(
                "unexpected EOF".into(),
            )),
        ])
        .await;
        let names = event_names(&events);
        assert_eq!(&names[names.len() - 2..], ["error", "response.failed"]);
        // Already opened by the content chunk: not opened a second time.
        assert_eq!(
            names.iter().filter(|n| **n == "response.created").count(),
            1
        );
        let (error, failed) = (&events[events.len() - 2].1, &events[events.len() - 1].1);
        assert_eq!(error["code"], "transport_error");
        assert_eq!(failed["response"]["error"]["code"], "transport_error");
        assert_eq!(failed["response"]["error"]["message"], error["message"]);
        assert_eq!(failed["response"]["output"], json!([]));
    }
}
