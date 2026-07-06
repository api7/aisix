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
//! carried (`instructions`, `input`, `tools`, `tool_choice`,
//! `temperature`, `top_p`, `max_output_tokens`, `stream`). OpenAI-only
//! knobs (`reasoning`, `store`, `previous_response_id`, `text`, …) are
//! dropped rather than forwarded — the downstream provider bridges flatten
//! unknown `extra` fields onto the upstream wire, where an OpenAI-only key
//! would 400 (e.g. Anthropic). Reasoning/thinking has no canonical bridge
//! mapping today, so it is intentionally not translated.

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
    // emitting it here is all that's needed.
    if let Some(tools) = body.get("tools").and_then(responses_tools_to_chat) {
        chat.extra.insert("tools".to_string(), tools);
    }
    if let Some(tc) = body
        .get("tool_choice")
        .and_then(responses_tool_choice_to_chat)
    {
        chat.extra.insert("tool_choice".to_string(), tc);
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
        // The tool result fed back by the caller → a `tool` role message.
        Some("function_call_output") => {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let output = item
                .get("output")
                .map(responses_content_text)
                .unwrap_or_default();
            messages.push(ChatMessage {
                role: Role::Tool,
                content: Some(output),
                content_blocks: None,
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
            let text = item
                .get("content")
                .map(responses_content_text)
                .unwrap_or_default();
            if text.is_empty() {
                return;
            }
            messages.push(match role {
                "assistant" => ChatMessage::assistant(text),
                "system" | "developer" => ChatMessage::system(text),
                _ => ChatMessage::user(text),
            });
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

/// Plain text of a Responses-API content slot: a bare string, or the
/// concatenation of the `text` of an array of typed parts
/// (`input_text` / `output_text` / `text`). Non-text parts are skipped.
fn responses_content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Translate Responses-API `tools` (flat function shape `{type:"function",
/// name, description, parameters}`) into OpenAI chat tools (`{type:
/// "function", function:{name, description, parameters}}`). Non-function
/// (hosted) tools have no chat equivalent and are dropped. Returns `None`
/// when nothing translates so the field stays absent from the wire.
fn responses_tools_to_chat(tools: &Value) -> Option<Value> {
    let arr = tools.as_array()?;
    let out: Vec<Value> = arr
        .iter()
        .filter_map(|t| {
            if t.get("type").and_then(|v| v.as_str()) != Some("function") {
                return None;
            }
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
        })
        .collect();
    (!out.is_empty()).then_some(Value::Array(out))
}

/// Translate Responses-API `tool_choice` to OpenAI chat shape:
/// `"auto"|"none"|"required"` pass through; `{type:"function", name}` →
/// `{type:"function", function:{name}}`. Hosted-tool choices have no chat
/// equivalent and drop to `None`.
fn responses_tool_choice_to_chat(tc: &Value) -> Option<Value> {
    match tc {
        Value::String(s) => Some(Value::String(s.clone())),
        Value::Object(o) => {
            if o.get("type").and_then(|v| v.as_str()) == Some("function") {
                let name = o.get("name").and_then(|v| v.as_str())?;
                Some(json!({"type": "function", "function": {"name": name}}))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Build the non-streaming Responses-API response object from a bridge
/// [`ChatResponse`]. `requested_model` is echoed back (not the upstream
/// id). `created_at` is a unix timestamp stamped by the caller.
pub fn chat_response_to_responses_json(
    resp: &ChatResponse,
    requested_model: &str,
    created_at: i64,
) -> Value {
    let (status, incomplete_reason) = responses_status(&resp.finish_reason);
    let output = build_output_items(
        resp.message.content.as_deref(),
        resp.message
            .extra
            .get("tool_calls")
            .and_then(|v| v.as_array()),
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

/// Assemble the `output` array: a `message` item carrying the assistant
/// text (when any), followed by one `function_call` item per tool call.
fn build_output_items(text: Option<&str>, tool_calls: Option<&Vec<Value>>) -> Vec<Value> {
    let mut output: Vec<Value> = Vec::new();
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
    output
}

/// Render usage in the Responses-API shape. `cached_tokens` takes whichever
/// of the OpenAI-normalized hit count or the Anthropic cache-read count is
/// present (the other is 0).
fn responses_usage_json(u: &UsageStats) -> Value {
    let total = if u.total_tokens > 0 {
        u.total_tokens
    } else {
        u.prompt_tokens.saturating_add(u.completion_tokens)
    };
    json!({
        "input_tokens": u.prompt_tokens,
        "input_tokens_details": {"cached_tokens": u.cached_prompt_tokens.max(u.cache_read_tokens)},
        "output_tokens": u.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": u.reasoning_tokens},
        "total_tokens": total,
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

/// Per-tool-call streaming state.
#[derive(Debug)]
struct ToolCallState {
    item_id: String,
    call_id: String,
    name: String,
    output_index: u32,
    arguments: String,
    item_added: bool,
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
    // Tool-call items keyed by the OpenAI delta index.
    tool_calls: std::collections::BTreeMap<u64, ToolCallState>,
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
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
}

impl ResponsesSseEncoder {
    pub fn new(
        response_id: impl Into<String>,
        model_display_name: impl Into<String>,
        created_at: i64,
    ) -> Self {
        Self {
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
            tool_calls: std::collections::BTreeMap::new(),
            pending_status: None,
            pending_reason: None,
            usage_seen: false,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
        }
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
            self.cache_creation_tokens = self.cache_creation_tokens.max(u.cache_creation_tokens);
            self.cache_read_tokens = self.cache_read_tokens.max(u.cache_read_tokens);
        }
    }

    fn usage_value(&self) -> Value {
        // `responses_usage_json` keeps a provider-supplied `total_tokens`
        // when present and only falls back to prompt+completion when it's 0.
        responses_usage_json(&UsageStats {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            cached_prompt_tokens: self.cached_prompt_tokens,
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
            items.push((
                tc.output_index,
                json!({
                    "type": "function_call",
                    "id": tc.item_id,
                    "call_id": tc.call_id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "status": "completed",
                }),
            ));
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
        let has_finish = chunk.finish_reason.is_some();

        let mut events = Vec::new();

        if !self.sent_created && (has_content || has_tools || has_finish) {
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
                            item_id: format!("fc_{}", Uuid::new_v4().simple()),
                            call_id: String::new(),
                            name: String::new(),
                            output_index,
                            arguments: String::new(),
                            item_added: false,
                        },
                    );
                }
                let state = self.tool_calls.get_mut(&oai_index).expect("just inserted");
                if !id.is_empty() {
                    state.call_id = id.to_string();
                }
                if !name.is_empty() {
                    state.name = name.to_string();
                }

                // Emit output_item.added once the call id + name are known.
                if !state.item_added && !state.call_id.is_empty() && !state.name.is_empty() {
                    state.item_added = true;
                    let (item_id, call_id, name, output_index) = (
                        state.item_id.clone(),
                        state.call_id.clone(),
                        state.name.clone(),
                        state.output_index,
                    );
                    events.push(self.event(
                        "response.output_item.added",
                        json!({
                            "output_index": output_index,
                            "item": {"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": "", "status": "in_progress"},
                        }),
                    ));
                }

                if !arguments.is_empty() {
                    let state = self.tool_calls.get_mut(&oai_index).expect("present");
                    state.arguments.push_str(arguments);
                    if state.item_added {
                        let (item_id, output_index) = (state.item_id.clone(), state.output_index);
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

    /// Emit the per-item `*.done` closing events for the open text + tool
    /// items. Idempotent: a no-op after the first call, so the finish chunk
    /// and a later `force_finish` (when the completed event was withheld for
    /// usage) don't double-emit the done events.
    fn close_items(&mut self) -> Vec<ResponsesSseEvent> {
        if self.items_closed {
            return Vec::new();
        }
        self.items_closed = true;
        let mut events = Vec::new();
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
            let (item_id, call_id, name, arguments, output_index) = {
                let s = self.tool_calls.get(&k).expect("present");
                (
                    s.item_id.clone(),
                    s.call_id.clone(),
                    s.name.clone(),
                    s.arguments.clone(),
                    s.output_index,
                )
            };
            events.push(self.event(
                "response.function_call_arguments.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "arguments": arguments,
                }),
            ));
            events.push(self.event(
                "response.output_item.done",
                json!({
                    "output_index": output_index,
                    "item": {"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": arguments, "status": "completed"},
                }),
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
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub reasoning_tokens: u32,
    pub cached_prompt_tokens: u32,
    pub cache_creation_tokens: u32,
    pub cache_read_tokens: u32,
    pub finish_reason: String,
    pub ttft_ms: u32,
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
}

struct CompleteOnDrop<F: FnOnce(ResponsesStreamCompletion)> {
    slot: Option<(F, ResponsesStreamCompletion)>,
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
        if let Some((f, comp)) = self.slot.take() {
            f(comp);
        }
    }
}

/// Wrap a bridge [`ChatChunkStream`] as a Responses-API SSE body, encoding
/// each chunk via [`ResponsesSseEncoder`]. An end-of-stream telemetry
/// callback fires from a Drop guard (so it runs on normal end and on client
/// disconnect).
///
/// When `output_guardrail` is `Some`, the encoded SSE is **held back** and
/// released only after the assembled assistant output passes the scan —
/// mirroring the verbatim `/v1/responses` path's secure BufferFull default
/// (#719), so a configured output block can't be bypassed by streaming a
/// non-OpenAI model. The scan reads the fully-reassembled text + tool calls
/// (not raw deltas), and the buffer is capped — an output guardrail must
/// never release content it couldn't fully buffer to scan, so an overflow
/// fails closed. With no output guardrail the bytes forward live.
#[allow(clippy::too_many_arguments)]
pub fn build_responses_bridge_stream(
    upstream: ChatChunkStream,
    encoder: ResponsesSseEncoder,
    started: Instant,
    output_guardrail: Option<Arc<aisix_guardrails::GuardrailChain>>,
    max_buffer_bytes: usize,
    model_label: String,
    // Largest content cap any content-capturing exporter wants
    // (AISIX-Cloud#947); `None` skips response-text accumulation entirely.
    content_cap: Option<u32>,
    on_complete: impl FnOnce(ResponsesStreamCompletion) + Send + 'static,
) -> axum::body::Body {
    use futures::StreamExt;

    let mut encoder = encoder;
    let stream = async_stream::stream! {
        let mut guard = CompleteOnDrop { slot: Some((on_complete, ResponsesStreamCompletion::default())) };
        let mut upstream = upstream;
        let mut first_chunk_seen = false;
        let buffering = output_guardrail.is_some();
        // Held SSE events when an output guardrail is attached; empty (and
        // unused) on the live-forward path.
        let mut held: Vec<bytes::Bytes> = Vec::new();
        let mut held_bytes = 0usize;
        let mut overflowed = false;
        while let Some(item) = upstream.next().await {
            match item {
                Ok(chunk) => {
                    if !first_chunk_seen
                        && (chunk.delta.content.is_some() || chunk.delta.tool_calls.is_some())
                    {
                        first_chunk_seen = true;
                        guard.comp().ttft_ms =
                            started.elapsed().as_millis().min(u32::MAX as u128) as u32;
                    }
                    {
                        let comp = guard.comp();
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
                        if let Some(u) = chunk.usage.as_ref() {
                            comp.prompt_tokens = comp.prompt_tokens.max(u.prompt_tokens);
                            comp.completion_tokens = comp.completion_tokens.max(u.completion_tokens);
                            comp.reasoning_tokens = comp.reasoning_tokens.max(u.reasoning_tokens);
                            comp.cached_prompt_tokens = comp.cached_prompt_tokens.max(u.cached_prompt_tokens);
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
                            yield Ok::<_, std::io::Error>(b);
                        }
                    }
                    if overflowed || encoder.is_finished() {
                        break;
                    }
                }
                Err(e) => {
                    let frame = format!(
                        "event: error\ndata: {{\"type\":\"error\",\"code\":\"{}\",\"message\":{}}}\n\n",
                        e.error_type(),
                        serde_json::to_string(&e.to_string()).unwrap_or_else(|_| "\"error\"".into()),
                    );
                    yield Ok(bytes::Bytes::from(frame));
                    return;
                }
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
                    yield Ok(b);
                }
            }
        }

        // Live-forward path: nothing held, nothing to scan.
        let Some(chain) = output_guardrail.as_ref() else { return; };

        // Buffer overflow: an output guardrail must not release content it
        // couldn't fully buffer to scan — fail closed (#719).
        if overflowed {
            tracing::warn!(
                guardrail_hook = "output",
                model = %model_label,
                max_buffer_bytes,
                "streaming /v1/responses (cross-provider) output exceeded buffer cap; failing closed",
            );
            guard.comp().guardrail_blocked = true;
            yield Ok(bytes::Bytes::from(guardrail_error_frame(None)));
            return;
        }

        // End-of-stream output guardrail (#719): scan the fully-reassembled
        // assistant output (canonical tool calls, so a literal split across
        // argument deltas can't slip through), then release or block.
        let (text, tool_calls) = encoder.assembled_assistant_message();
        if !text.is_empty() || !tool_calls.is_empty() {
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
                |g| match crate::redact::redact_responses_sse(g, &joined) {
                    Some((rewritten, counts)) => {
                        joined = rewritten;
                        seg_rewrote = true;
                        counts
                    }
                    None => crate::redact::RedactionCounts::new(),
                },
            )
            .await;
            guard.comp().monitor_hits.extend(seg_hits);
            if !seg_counts.is_empty() {
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
            if let aisix_guardrails::GuardrailVerdict::Block { reason, guardrail_name } = verdict {
                tracing::warn!(
                    guardrail_hook = "output",
                    model = %model_label,
                    reason = %reason,
                    "guardrail blocked streaming /v1/responses (cross-provider) response",
                );
                guard.comp().guardrail_blocked = true;
                yield Ok(bytes::Bytes::from(guardrail_error_frame(guardrail_name.as_deref())));
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
                yield Ok(bytes::Bytes::from(rewritten));
                return;
            }
        }
        // Release the held events verbatim.
        for b in held {
            yield Ok(b);
        }
    };
    axum::body::Body::from_stream(stream)
}

/// Responses-API SSE `error` frame for an output-guardrail block. Carries the
/// firing guardrail's name (#519 B.4b) but never the matched-pattern detail.
fn guardrail_error_frame(guardrail_name: Option<&str>) -> String {
    format!(
        "event: error\ndata: {}\n\n",
        json!({
            "type": "error",
            "code": "content_filter",
            "message": crate::error::guardrail_block_message("response", guardrail_name),
        })
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
            // OpenAI-only knobs must NOT leak into chat.extra (they'd 400 Anthropic).
            "reasoning": {"effort": "high"},
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
        assert!(!chat.extra.contains_key("reasoning"));
        assert!(!chat.extra.contains_key("store"));
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
        let out = chat_response_to_responses_json(&resp, "opus-4.7", 100);
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
        let out = chat_response_to_responses_json(&resp, "m", 1);
        let item = &out["output"][0];
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["call_id"], "call_9");
        assert_eq!(item["name"], "shell");
        assert_eq!(item["arguments"], "{\"cmd\":\"ls\"}");
    }

    #[test]
    fn length_finish_maps_to_incomplete_status() {
        let resp = chat_response_with(Some("x"), None, FinishReason::Length);
        let out = chat_response_to_responses_json(&resp, "m", 1);
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
        let mut enc = ResponsesSseEncoder::new("resp_1", "opus-4.7", 0);
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
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0);
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
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0);
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

    #[test]
    fn force_finish_on_empty_stream_emits_well_formed_completed() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0);
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
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0);
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

    #[test]
    fn streaming_preserves_provider_total_tokens() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0);
        let _ = enc.next_events(&content_chunk("hi"));
        let done = enc.next_events(&ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: ChatDelta::default(),
            finish_reason: Some(FinishReason::Stop),
            // Provider reports an authoritative total that differs from
            // prompt+completion (e.g. it counts tool/system overhead).
            usage: Some(UsageStats {
                prompt_tokens: 5,
                completion_tokens: 2,
                total_tokens: 11,
                ..Default::default()
            }),
        });
        let completed = done.last().unwrap();
        assert_eq!(completed.data["response"]["usage"]["total_tokens"], 11);
    }

    #[test]
    fn streaming_length_finish_emits_incomplete_with_reason() {
        let mut enc = ResponsesSseEncoder::new("resp_1", "m", 0);
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
}
