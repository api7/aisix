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
        // Reasoning items can't be replayed across providers — drop them.
        Some("reasoning") => {}
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
/// preceding assistant tool-call message when contiguous so parallel
/// `function_call` items land in one assistant turn (one `tool_calls`
/// array) — the standard OpenAI history shape every bridge expects.
fn push_tool_call(messages: &mut Vec<ChatMessage>, tc: Value) {
    if let Some(last) = messages.last_mut() {
        if matches!(last.role, Role::Assistant) && last.content.is_none() {
            if let Some(Value::Array(arr)) = last.extra.get_mut("tool_calls") {
                arr.push(tc);
                return;
            }
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
///   * hosted tools (`web_search*`, `file_search`, `code_interpreter`,
///     `mcp`, `computer_use*`, `image_generation`, …) have no chat
///     equivalent and are dropped.
///
/// Returns `None` when nothing translates so the field stays absent from
/// the wire.
fn responses_tools_to_chat(tools: &Value) -> Option<Value> {
    let arr = tools.as_array()?;
    let out: Vec<Value> = arr
        .iter()
        .filter_map(|t| match t.get("type").and_then(|v| v.as_str()) {
            Some("function") => {
                let name = t.get("name").and_then(|v| v.as_str())?;
                let mut func = Map::new();
                func.insert("name".to_string(), json!(name));
                if let Some(d) = t.get("description") {
                    func.insert("description".to_string(), d.clone());
                }
                if let Some(p) = t.get("parameters") {
                    func.insert("parameters".to_string(), p.clone());
                }
                Some(json!({"type": "function", "function": Value::Object(func)}))
            }
            Some("custom") => {
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
            _ => None,
        })
        .collect();
    (!out.is_empty()).then_some(Value::Array(out))
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

/// The names of the `custom` tools a Responses request declared.
///
/// The request side turns each of them into an ordinary function tool
/// ([`responses_tools_to_chat`]), so the reply arrives as a chat tool call
/// carrying nothing that says which Responses tool kind it came from. The
/// caller registered the tool as `custom` and is waiting for a
/// `custom_tool_call` item back, so both response translators need the
/// request's own tool list to tell the two kinds apart.
pub fn custom_tool_names(body: &Value) -> std::collections::BTreeSet<String> {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return std::collections::BTreeSet::new();
    };
    tools
        .iter()
        .filter(|t| t.get("type").and_then(Value::as_str) == Some("custom"))
        .filter_map(|t| t.get("name").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
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
/// id). `created_at` is a unix timestamp stamped by the caller.
/// `custom_tools` names the request's `custom` tools (see
/// [`custom_tool_names`]) so a call to one of them is returned as the
/// `custom_tool_call` item the caller registered it for.
pub fn chat_response_to_responses_json(
    resp: &ChatResponse,
    requested_model: &str,
    created_at: i64,
    custom_tools: &std::collections::BTreeSet<String>,
) -> Value {
    let (status, incomplete_reason) = responses_status(&resp.finish_reason);
    let output = build_output_items(
        message_reasoning_text(&resp.message),
        resp.message.content.as_deref(),
        resp.message
            .extra
            .get("tool_calls")
            .and_then(|v| v.as_array()),
        custom_tools,
    );

    let mut obj = json!({
        "id": format!("resp_{}", Uuid::new_v4().simple()),
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": requested_model,
        "output": output,
        "usage": responses_usage_json(&resp.usage),
    });
    if let Some(reason) = incomplete_reason {
        obj["incomplete_details"] = json!({"reason": reason});
    }
    obj
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
/// `custom_tool_call` for a call naming one of `custom_tools`,
/// `function_call` for everything else.
fn build_output_items(
    reasoning: Option<&str>,
    text: Option<&str>,
    tool_calls: Option<&Vec<Value>>,
    custom_tools: &std::collections::BTreeSet<String>,
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
            if custom_tools.contains(name) {
                output.push(json!({
                    "type": "custom_tool_call",
                    "id": format!("ctc_{}", Uuid::new_v4().simple()),
                    "call_id": call_id,
                    "name": name,
                    "input": unwrap_custom_tool_input(arguments),
                    "status": "completed",
                }));
            } else {
                output.push(json!({
                    "type": "function_call",
                    "id": format!("fc_{}", Uuid::new_v4().simple()),
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed",
                }));
            }
        }
    }
    output
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
            json!({
                "type": "function_call",
                "id": self.item_id(),
                "call_id": self.call_id,
                "name": self.name,
                "arguments": self.arguments,
                "status": "completed",
            })
        }
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
    /// The request's `custom` tool names (see [`custom_tool_names`]) — a
    /// call naming one of them streams as a `custom_tool_call` item.
    custom_tools: std::collections::BTreeSet<String>,
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
    /// `custom_tools` names the request's `custom` tools (see
    /// [`custom_tool_names`]); pass an empty set for a request that declared
    /// none.
    pub fn new(
        response_id: impl Into<String>,
        model_display_name: impl Into<String>,
        created_at: i64,
        custom_tools: std::collections::BTreeSet<String>,
    ) -> Self {
        Self {
            custom_tools,
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

    /// The bare response object embedded in lifecycle events.
    fn response_object(&self, status: &str, with_output: bool, with_usage: bool) -> Value {
        let mut obj = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model_display_name,
            "output": if with_output { Value::Array(self.final_output_items()) } else { json!([]) },
        });
        if with_usage {
            obj["usage"] = self.usage_value();
        }
        obj
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
            events.push(self.event(
                "response.created",
                json!({"response": self.response_object("in_progress", false, false)}),
            ));
            events.push(self.event(
                "response.in_progress",
                json!({"response": self.response_object("in_progress", false, false)}),
            ));
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
                            output_index,
                            arguments: String::new(),
                            item_added: false,
                        },
                    );
                }
                let custom = self.custom_tools.contains(name);
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
                    }
                }

                // Emit output_item.added once the call id + name are known.
                if !state.item_added && !state.call_id.is_empty() && !state.name.is_empty() {
                    state.item_added = true;
                    let (item_id, call_id, name, output_index, custom) = (
                        state.item_id(),
                        state.call_id.clone(),
                        state.name.clone(),
                        state.output_index,
                        state.custom,
                    );
                    let item = if custom {
                        json!({"type": "custom_tool_call", "id": item_id, "call_id": call_id, "name": name, "input": "", "status": "in_progress"})
                    } else {
                        json!({"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": "", "status": "in_progress"})
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
        let mut response = self.response_object(status, true, true);
        if let Some(reason) = reason {
            response["incomplete_details"] = json!({"reason": reason});
        }
        self.event(event_type, json!({"response": response}))
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Flush a clean close when the upstream stream ended without a finish
    /// chunk, or while the completed event was withheld for usage.
    pub fn force_finish(&mut self) -> Vec<ResponsesSseEvent> {
        if self.finished {
            return Vec::new();
        }
        let mut events = Vec::new();
        // No renderable signal ever arrived → synthesize the preamble so
        // the client still gets a well-formed (empty) response.
        if !self.sent_created {
            self.sent_created = true;
            events.push(self.event(
                "response.created",
                json!({"response": self.response_object("in_progress", false, false)}),
            ));
            events.push(self.event(
                "response.in_progress",
                json!({"response": self.response_object("in_progress", false, false)}),
            ));
        }
        let status = self.pending_status.take().unwrap_or("completed");
        let reason = self.pending_reason.take();
        events.extend(self.close_items());
        events.push(self.completed_event(status, reason));
        events
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
        // Stamped on the first bytes that actually leave for the client —
        // under hold-back that is the release, not the upstream chunk.
        macro_rules! downstream_mark {
            () => {
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
                            held_bytes += b.len();
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
                    yield Ok(bytes::Bytes::from(upstream_error_frame(
                        encoder.take_sequence_number(),
                        e.error_type(),
                        &e.to_string(),
                    )));
                    return;
                }
            }
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
                    held_bytes += b.len();
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
            yield Ok(bytes::Bytes::from(guardrail_error_frame(encoder.take_sequence_number(), None, Some(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED))));
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
                yield Ok(bytes::Bytes::from(guardrail_error_frame(encoder.take_sequence_number(), guardrail_name.as_deref(), unavailable.as_deref())));
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

/// Responses-API SSE `error` frame for an upstream failure mid-relay.
fn upstream_error_frame(seq: u64, error_type: &str, message: &str) -> String {
    responses_error_frame(seq, error_type, message)
}

/// Responses-API SSE `error` frame for an output-guardrail block. Carries the
/// firing guardrail's name (#519 B.4b) but never the matched-pattern detail.
fn guardrail_error_frame(
    seq: u64,
    guardrail_name: Option<&str>,
    unavailable: Option<&str>,
) -> String {
    responses_error_frame(
        seq,
        "content_filter",
        &crate::error::guardrail_block_message("response", guardrail_name, unavailable),
    )
}

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

    /// A request that declared no `custom` tools.
    fn no_custom_tools() -> BTreeSet<String> {
        BTreeSet::new()
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
    fn custom_tool_names_reads_only_the_custom_entries() {
        let body = json!({
            "tools": [
                {"type": "function", "name": "get_weather"},
                {"type": "custom", "name": "apply_patch"},
                {"type": "custom"},
                {"type": "web_search_preview"},
            ],
        });
        assert_eq!(
            custom_tool_names(&body),
            BTreeSet::from(["apply_patch".to_string()])
        );
        assert!(custom_tool_names(&json!({"input": "hi"})).is_empty());
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
        let out = chat_response_to_responses_json(
            &resp,
            "m",
            1,
            &BTreeSet::from(["apply_patch".to_string()]),
        );
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
        let custom = BTreeSet::from(["apply_patch".to_string()]);
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
        let out = chat_response_to_responses_json(
            &resp,
            "m",
            1,
            &BTreeSet::from(["apply_patch".to_string()]),
        );
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
        let mut enc = ResponsesSseEncoder::new(
            "resp_1",
            "m",
            0,
            BTreeSet::from(["apply_patch".to_string()]),
        );
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
        let mut enc = ResponsesSseEncoder::new(
            "resp_1",
            "m",
            0,
            BTreeSet::from(["apply_patch".to_string()]),
        );
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
        let mut enc = ResponsesSseEncoder::new(
            "resp_1",
            "m",
            0,
            BTreeSet::from(["apply_patch".to_string()]),
        );
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
        let mut enc = ResponsesSseEncoder::new(
            "resp_1",
            "m",
            0,
            BTreeSet::from(["apply_patch".to_string()]),
        );
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

    #[test]
    fn force_finish_on_empty_stream_emits_well_formed_completed() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0, no_custom_tools());
        let events = enc.force_finish();
        let types = types_of(&events);
        assert_eq!(
            types,
            vec![
                "response.created",
                "response.in_progress",
                "response.completed"
            ]
        );
        assert!(enc.is_finished());
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

    /// The Responses API defines its `error` event FLAT, discriminated on
    /// the top-level `type`, and the official SDKs parse it that way —
    /// `openai-python`'s `ResponseErrorEvent` is
    /// `{type, code, message, param, sequence_number}`. Nesting it under an
    /// `error` object (which is what every other SSE error in this crate
    /// does) would hand a `responses.stream()` client an event it cannot
    /// classify. Both of this relay's error frames are checked, because a
    /// client should not have to know which failure it hit.
    #[test]
    fn both_sse_error_frames_match_the_responses_api_error_event() {
        let payload_of = |frame: &str| -> serde_json::Value {
            let body = frame
                .strip_prefix("event: error\ndata: ")
                .and_then(|r| r.strip_suffix("\n\n"))
                .expect("an SSE error frame labelled `error`")
                .to_owned();
            serde_json::from_str(&body).expect("one JSON document")
        };

        let block = payload_of(&guardrail_error_frame(7, Some("gr-block"), None));
        assert_eq!(block["type"], "error");
        assert_eq!(block["code"], "content_filter");
        assert!(block["message"].as_str().unwrap().contains("gr-block"));
        assert_eq!(block["param"], serde_json::Value::Null);
        assert_eq!(block["sequence_number"], 7);

        // The upstream-failure frame on the same stream, same envelope. Its
        // message is JSON-escaped through serde rather than interpolated.
        let upstream = payload_of(&upstream_error_frame(
            8,
            "upstream_error",
            "boom \"quoted\"",
        ));
        assert_eq!(upstream["type"], "error");
        assert_eq!(upstream["code"], "upstream_error");
        assert_eq!(upstream["message"], "boom \"quoted\"");
        assert_eq!(upstream["sequence_number"], 8);

        // Exactly the SDK's field set, and nothing nested: an `error` key
        // here is the shape this deliberately does NOT use.
        for v in [&block, &upstream] {
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
}
