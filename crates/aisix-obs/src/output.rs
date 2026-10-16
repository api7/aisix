//! The assistant output a caller received, captured in order and by kind —
//! text, reasoning, tool calls — for exporters that report the conventions'
//! structured `gen_ai.output.messages`.
//!
//! It is read off the CLIENT-FACING wire (the response body, or the SSE
//! frames as they were handed to the caller), never off an upstream: that
//! is what makes it post-masking by construction, and the same whether the
//! request was served natively or bridged from another protocol.
//!
//! What it keeps, per wire:
//!
//! - OpenAI Chat: `reasoning_content` (or `reasoning`), `content`, and
//!   `tool_calls`.
//! - Anthropic Messages: `text`, `thinking` and `tool_use` blocks.
//!   `redacted_thinking` carries no readable text and is left out.
//! - OpenAI Responses: `message` text, the readable summary and text of a
//!   `reasoning` item (never its `encrypted_content`), and `function_call`.
//!
//! Every other output kind — a refusal, audio, a server-side tool call or
//! its result — has no exact counterpart among the conventions' part
//! types and is left out rather than reported as a near-match.

use std::collections::BTreeMap;

use serde_json::{json, Value};

/// The client-facing protocol an output was written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputWire {
    OpenAiChat,
    AnthropicMessages,
    OpenAiResponses,
}

#[derive(Debug, Clone, PartialEq)]
enum Part {
    Text(String),
    Reasoning(String),
    ToolCall {
        id: Option<String>,
        name: String,
        arguments: String,
    },
}

impl Part {
    fn len(&self) -> usize {
        match self {
            Part::Text(s) | Part::Reasoning(s) => s.len(),
            Part::ToolCall {
                id,
                name,
                arguments,
            } => id.as_deref().map_or(0, str::len) + name.len() + arguments.len(),
        }
    }

    fn to_json(&self) -> Option<Value> {
        Some(match self {
            Part::Text(s) if !s.is_empty() => json!({"type": "text", "content": s}),
            Part::Reasoning(s) if !s.is_empty() => json!({"type": "reasoning", "content": s}),
            Part::ToolCall {
                id,
                name,
                arguments,
            } if !name.is_empty() => {
                let arguments = serde_json::from_str::<Value>(arguments)
                    .unwrap_or_else(|_| Value::String(arguments.clone()));
                let mut part = json!({"type": "tool_call", "name": name, "arguments": arguments});
                if let Some(id) = id.as_deref().filter(|id| !id.is_empty()) {
                    part["id"] = Value::String(id.to_string());
                }
                part
            }
            _ => return None,
        })
    }
}

/// A captured output: its parts in the order the caller received them,
/// and whether the capture cap cut it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OutputCapture {
    parts: Vec<Part>,
    /// The output outgrew the capture cap, so the parts are incomplete.
    pub truncated: bool,
}

impl OutputCapture {
    /// The conventions' message parts, `None` when nothing was captured.
    pub fn parts_json(&self) -> Option<Vec<Value>> {
        let parts: Vec<Value> = self.parts.iter().filter_map(Part::to_json).collect();
        (!parts.is_empty()).then_some(parts)
    }
}

/// Where an output part is being assembled. Keys order the parts: the
/// position the protocol gives the part, then the order it first appeared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SlotKey {
    position: u64,
    sub: u64,
}

/// Assembles an [`OutputCapture`] from a response body or from the SSE
/// frames of a stream, bounded to `cap` bytes of captured text.
#[derive(Debug)]
pub struct OutputCollector {
    wire: OutputWire,
    cap: usize,
    slots: BTreeMap<SlotKey, Part>,
    /// Chat assigns slots in arrival order: a tool call keeps its own by
    /// index, and text or reasoning opens a new one whenever the kind
    /// changes, so interleaved reasoning stays where it was received.
    next_sub: u64,
    chat_tools: BTreeMap<u64, u64>,
    chat_run: Option<(&'static str, u64)>,
    /// Anthropic `tool_use` slots holding the `{}` a block opened with,
    /// which the first `input_json_delta` replaces rather than extends.
    placeholder_args: std::collections::BTreeSet<SlotKey>,
    truncated: bool,
    decoder: aisix_gateway::SseDecoder,
}

impl OutputCollector {
    pub fn new(wire: OutputWire, cap: usize) -> Self {
        Self {
            wire,
            cap,
            slots: BTreeMap::new(),
            next_sub: 0,
            chat_tools: BTreeMap::new(),
            chat_run: None,
            placeholder_args: Default::default(),
            truncated: false,
            decoder: aisix_gateway::SseDecoder::new(),
        }
    }

    /// One complete response body as the caller received it.
    pub fn from_body(wire: OutputWire, cap: usize, body: &Value) -> OutputCapture {
        let mut c = Self::new(wire, cap);
        c.observe_body(body);
        c.finish()
    }

    /// Bytes of an SSE response, as handed to the caller.
    pub fn feed_sse(&mut self, bytes: &[u8]) {
        let Ok(events) = self.decoder.feed(bytes) else {
            // A frame outgrew the decoder: what follows is lost, so the
            // capture is incomplete.
            self.truncated = true;
            return;
        };
        for event in events {
            if let aisix_gateway::SseEvent::Data(payload) = event {
                if let Ok(value) = serde_json::from_str::<Value>(&payload) {
                    self.observe_event(&value);
                }
            }
        }
    }

    pub fn finish(mut self) -> OutputCapture {
        if let Ok(Some(aisix_gateway::SseEvent::Data(payload))) = self.decoder.finish() {
            if let Ok(value) = serde_json::from_str::<Value>(&payload) {
                self.observe_event(&value);
            }
        }
        OutputCapture {
            parts: self.slots.into_values().collect(),
            truncated: self.truncated,
        }
    }

    fn captured_bytes(&self) -> usize {
        self.slots.values().map(Part::len).sum()
    }

    /// Room left under the cap for `len` more bytes; past it the capture
    /// is marked cut and stops growing.
    fn admit(&mut self, len: usize) -> bool {
        if self.truncated {
            return false;
        }
        if self.captured_bytes() + len > self.cap {
            self.truncated = true;
            return false;
        }
        true
    }

    fn append(&mut self, key: SlotKey, fresh: impl FnOnce() -> Part, text: &str) {
        if text.is_empty() || !self.admit(text.len()) {
            return;
        }
        match self.slots.entry(key).or_insert_with(fresh) {
            Part::Text(s) | Part::Reasoning(s) => s.push_str(text),
            Part::ToolCall { arguments, .. } => arguments.push_str(text),
        }
    }

    fn put(&mut self, key: SlotKey, part: Part) {
        let previous = self.slots.get(&key).map_or(0, Part::len);
        if self.truncated || self.captured_bytes() - previous + part.len() > self.cap {
            self.truncated = true;
            return;
        }
        self.slots.insert(key, part);
    }

    fn observe_body(&mut self, body: &Value) {
        match self.wire {
            OutputWire::OpenAiChat => {
                if let Some(message) = first_choice(body).and_then(|c| c.get("message")) {
                    self.observe_chat(message);
                }
            }
            OutputWire::AnthropicMessages => {
                for (i, block) in array(body.get("content")).iter().enumerate() {
                    self.observe_anthropic_block(i as u64, block);
                }
            }
            OutputWire::OpenAiResponses => {
                for (i, item) in array(body.get("output")).iter().enumerate() {
                    self.observe_responses_item(i as u64, item);
                }
            }
        }
    }

    fn observe_event(&mut self, event: &Value) {
        match self.wire {
            OutputWire::OpenAiChat => {
                if let Some(delta) = first_choice(event).and_then(|c| c.get("delta")) {
                    self.observe_chat(delta);
                }
            }
            OutputWire::AnthropicMessages => self.observe_anthropic_event(event),
            OutputWire::OpenAiResponses => self.observe_responses_event(event),
        }
    }

    // ── OpenAI Chat ──────────────────────────────────────────────────────

    fn fresh_sub(&mut self) -> u64 {
        let sub = self.next_sub;
        self.next_sub += 1;
        sub
    }

    /// The slot text or reasoning extends: the current run's while the kind
    /// is unchanged, a new one when it changes.
    fn chat_run_slot(&mut self, kind: &'static str) -> SlotKey {
        let sub = match self.chat_run {
            Some((k, sub)) if k == kind => sub,
            _ => {
                let sub = self.fresh_sub();
                self.chat_run = Some((kind, sub));
                sub
            }
        };
        SlotKey { position: 0, sub }
    }

    fn chat_tool_slot(&mut self, index: u64) -> SlotKey {
        let sub = match self.chat_tools.get(&index) {
            Some(sub) => *sub,
            None => {
                let sub = self.fresh_sub();
                self.chat_tools.insert(index, sub);
                sub
            }
        };
        self.chat_run = None;
        SlotKey { position: 0, sub }
    }

    /// A `message` (body) or a `delta` (stream): the same keys either way,
    /// a delta carrying a fragment of each.
    fn observe_chat(&mut self, m: &Value) {
        let reasoning = m
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| m.get("reasoning").and_then(Value::as_str));
        if let Some(text) = reasoning {
            let key = self.chat_run_slot("reasoning");
            self.append(key, || Part::Reasoning(String::new()), text);
        }
        match m.get("content") {
            Some(Value::String(text)) => {
                let key = self.chat_run_slot("text");
                self.append(key, || Part::Text(String::new()), text);
            }
            Some(Value::Array(parts)) => {
                for p in parts {
                    if let Some(text) = p.get("text").and_then(Value::as_str) {
                        let key = self.chat_run_slot("text");
                        self.append(key, || Part::Text(String::new()), text);
                    }
                }
            }
            _ => {}
        }
        for (position, call) in array(m.get("tool_calls")).iter().enumerate() {
            let index = call
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(position as u64);
            let key = self.chat_tool_slot(index);
            let id = call.get("id").and_then(Value::as_str);
            let function = call.get("function");
            let name = function.and_then(|f| f.get("name")).and_then(Value::as_str);
            if !self.slots.contains_key(&key) {
                let part = Part::ToolCall {
                    id: None,
                    name: String::new(),
                    arguments: String::new(),
                };
                self.put(key, part);
            }
            if let Some(Part::ToolCall {
                id: slot_id,
                name: slot_name,
                ..
            }) = self.slots.get_mut(&key)
            {
                if let Some(id) = id.filter(|id| !id.is_empty()) {
                    slot_id.get_or_insert_with(|| id.to_string());
                }
                if let Some(name) = name.filter(|n| !n.is_empty()) {
                    if slot_name.is_empty() {
                        slot_name.push_str(name);
                    }
                }
            }
            match function.and_then(|f| f.get("arguments")) {
                Some(Value::String(args)) => self.append(key, unreachable_part, args),
                Some(other) if !other.is_null() => {
                    self.append(key, unreachable_part, &other.to_string())
                }
                _ => {}
            }
        }
    }

    // ── Anthropic Messages ───────────────────────────────────────────────

    fn observe_anthropic_block(&mut self, index: u64, block: &Value) {
        let key = SlotKey {
            position: index,
            sub: 0,
        };
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                self.put(key, Part::Text(text.to_string()));
            }
            Some("thinking") => {
                let text = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                self.put(key, Part::Reasoning(text.to_string()));
            }
            Some("tool_use") => {
                // A streamed block opens with `input: {}` and fills it in
                // through `input_json_delta`, so an empty object is kept as
                // the arguments only until a delta arrives — a tool that
                // takes none really is called with `{}`.
                let arguments = match block.get("input") {
                    Some(input) if !input.is_null() => input.to_string(),
                    _ => String::new(),
                };
                if arguments == "{}" {
                    self.placeholder_args.insert(key);
                }
                self.put(
                    key,
                    Part::ToolCall {
                        id: block.get("id").and_then(Value::as_str).map(str::to_string),
                        name: block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        arguments,
                    },
                );
            }
            _ => {}
        }
    }

    fn observe_anthropic_event(&mut self, event: &Value) {
        let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
        let key = SlotKey {
            position: index,
            sub: 0,
        };
        match event.get("type").and_then(Value::as_str) {
            Some("content_block_start") => {
                if let Some(block) = event.get("content_block") {
                    self.observe_anthropic_block(index, block);
                }
            }
            Some("content_block_delta") => {
                let Some(delta) = event.get("delta") else {
                    return;
                };
                // Only into a slot the block's start opened: a delta of a
                // kind left out (a server tool's input) must not create one.
                if !self.slots.contains_key(&key) {
                    return;
                }
                let fragment = match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => delta.get("text"),
                    Some("thinking_delta") => delta.get("thinking"),
                    Some("input_json_delta") => delta.get("partial_json"),
                    _ => None,
                };
                if let Some(text) = fragment.and_then(Value::as_str) {
                    if self.placeholder_args.remove(&key) {
                        if let Some(Part::ToolCall { arguments, .. }) = self.slots.get_mut(&key) {
                            arguments.clear();
                        }
                    }
                    self.append(key, unreachable_part, text);
                }
            }
            _ => {}
        }
    }

    // ── OpenAI Responses ─────────────────────────────────────────────────

    fn observe_responses_item(&mut self, output_index: u64, item: &Value) {
        let key = |sub: u64| SlotKey {
            position: output_index,
            sub,
        };
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                for (i, c) in array(item.get("content")).iter().enumerate() {
                    if c.get("type").and_then(Value::as_str) == Some("output_text") {
                        let text = c.get("text").and_then(Value::as_str).unwrap_or("");
                        self.put(key(i as u64), Part::Text(text.to_string()));
                    }
                }
            }
            Some("reasoning") => {
                // Readable text only: the summary, then the reasoning text a
                // model returns in the clear. `encrypted_content` is opaque.
                let summary = array(item.get("summary"));
                for (i, s) in summary.iter().enumerate() {
                    if let Some(text) = s.get("text").and_then(Value::as_str) {
                        self.put(key(i as u64), Part::Reasoning(text.to_string()));
                    }
                }
                for (i, c) in array(item.get("content")).iter().enumerate() {
                    if let Some(text) = c.get("text").and_then(Value::as_str) {
                        self.put(
                            key(REASONING_TEXT_BASE + i as u64),
                            Part::Reasoning(text.to_string()),
                        );
                    }
                }
            }
            Some("function_call") => {
                self.put(
                    key(0),
                    Part::ToolCall {
                        id: item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        name: item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        arguments: item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    },
                );
            }
            _ => {}
        }
    }

    fn observe_responses_event(&mut self, event: &Value) {
        let output_index = event
            .get("output_index")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let key = |sub: u64| SlotKey {
            position: output_index,
            sub,
        };
        let delta = event.get("delta").and_then(Value::as_str).unwrap_or("");
        match event.get("type").and_then(Value::as_str) {
            Some("response.output_item.added") => {
                if let Some(item) = event.get("item") {
                    if item.get("type").and_then(Value::as_str) == Some("function_call") {
                        self.observe_responses_item(output_index, item);
                    }
                }
            }
            // The finished item is authoritative: it replaces whatever the
            // deltas assembled for it.
            Some("response.output_item.done") => {
                if let Some(item) = event.get("item") {
                    self.slots.retain(|k, _| k.position != output_index);
                    self.observe_responses_item(output_index, item);
                }
            }
            Some("response.output_text.delta") => {
                let sub = event
                    .get("content_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                self.append(key(sub), || Part::Text(String::new()), delta);
            }
            Some("response.reasoning_summary_text.delta") => {
                let sub = event
                    .get("summary_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                self.append(key(sub), || Part::Reasoning(String::new()), delta);
            }
            Some("response.reasoning_text.delta") => {
                let sub = event
                    .get("content_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                self.append(
                    key(REASONING_TEXT_BASE + sub),
                    || Part::Reasoning(String::new()),
                    delta,
                );
            }
            Some("response.function_call_arguments.delta") => {
                if self.slots.contains_key(&key(0)) {
                    self.append(key(0), unreachable_part, delta);
                }
            }
            _ => {}
        }
    }
}

/// The finish reason a Responses result reports, in OpenAI chat's
/// vocabulary (the one the usage event's `finish_reason` speaks for chat):
/// a `completed` response is `tool_calls` when its output asks for a tool
/// (a `function_call` or `custom_tool_call` item) and `stop` otherwise; an
/// `incomplete` one is `content_filter` when that is its reason and
/// `length` otherwise. Any other status (`failed`, `cancelled`, …) names
/// none.
pub fn responses_finish_reason(response: &Value) -> Option<&'static str> {
    match response.get("status").and_then(Value::as_str)? {
        "completed" => {
            let calls_tool = array(response.get("output")).iter().any(|item| {
                matches!(
                    item.get("type").and_then(Value::as_str),
                    Some("function_call") | Some("custom_tool_call")
                )
            });
            Some(if calls_tool { "tool_calls" } else { "stop" })
        }
        "incomplete" => {
            let reason = response
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str);
            Some(if reason == Some("content_filter") {
                "content_filter"
            } else {
                "length"
            })
        }
        _ => None,
    }
}

/// Reads a streamed Responses result's finish reason off the frames the
/// caller receives: the terminal `response.completed` / `response.incomplete`
/// / `response.failed` event carries the response object.
#[derive(Debug, Default)]
pub struct ResponsesFinishObserver {
    decoder: aisix_gateway::SseDecoder,
    finish: Option<&'static str>,
}

impl ResponsesFinishObserver {
    pub fn feed_sse(&mut self, bytes: &[u8]) {
        let Ok(events) = self.decoder.feed(bytes) else {
            return;
        };
        for event in events {
            let aisix_gateway::SseEvent::Data(payload) = event else {
                continue;
            };
            // Only the terminal events are worth a parse.
            if !(payload.contains("response.completed")
                || payload.contains("response.incomplete")
                || payload.contains("response.failed"))
            {
                continue;
            }
            if let Ok(value) = serde_json::from_str::<Value>(&payload) {
                if matches!(
                    value.get("type").and_then(Value::as_str),
                    Some("response.completed" | "response.incomplete" | "response.failed")
                ) {
                    self.finish = value.get("response").and_then(responses_finish_reason);
                }
            }
        }
    }

    pub fn finish(&self) -> Option<&'static str> {
        self.finish
    }
}

/// Orders a reasoning item's clear-text content after its summary.
const REASONING_TEXT_BASE: u64 = 1 << 32;

/// The `fresh` argument of an append into a slot that already exists.
fn unreachable_part() -> Part {
    Part::Text(String::new())
}

fn first_choice(v: &Value) -> Option<&Value> {
    let choices = v.get("choices")?.as_array()?;
    choices
        .iter()
        .find(|c| c.get("index").and_then(Value::as_u64).unwrap_or(0) == 0)
}

fn array(v: Option<&Value>) -> &[Value] {
    v.and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(c: &OutputCapture) -> Value {
        Value::Array(c.parts_json().unwrap_or_default())
    }

    fn sse(events: &[Value]) -> Vec<u8> {
        events
            .iter()
            .map(|e| format!("data: {e}\n\n"))
            .collect::<String>()
            .into_bytes()
    }

    #[test]
    fn chat_stream_keeps_reasoning_text_and_tool_calls_in_order() {
        let mut c = OutputCollector::new(OutputWire::OpenAiChat, 4096);
        let frames = sse(&[
            json!({"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think "}}]}),
            json!({"choices":[{"index":0,"delta":{"reasoning_content":"hard"}}]}),
            json!({"choices":[{"index":0,"delta":{"content":"Hi"}}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"tz\":"}}]}}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"UTC\"}"}}]}}]}),
        ]);
        // Split mid-frame: the decoder reassembles.
        let (a, b) = frames.split_at(frames.len() / 2);
        c.feed_sse(a);
        c.feed_sse(b);
        c.feed_sse(b"data: [DONE]\n\n");
        let out = c.finish();
        assert!(!out.truncated);
        assert_eq!(
            parts(&out),
            json!([
                {"type":"reasoning","content":"think hard"},
                {"type":"text","content":"Hi"},
                {"type":"tool_call","id":"call_1","name":"get_time","arguments":{"tz":"UTC"}},
            ])
        );
    }

    #[test]
    fn responses_finish_reason_follows_status_and_output() {
        let cases = [
            (
                json!({"status":"completed","output":[{"type":"message"}]}),
                Some("stop"),
            ),
            (
                json!({"status":"completed","output":[{"type":"function_call"}]}),
                Some("tool_calls"),
            ),
            (
                json!({"status":"completed","output":[{"type":"custom_tool_call"}]}),
                Some("tool_calls"),
            ),
            (
                json!({"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}),
                Some("length"),
            ),
            (
                json!({"status":"incomplete","incomplete_details":{"reason":"content_filter"}}),
                Some("content_filter"),
            ),
            (json!({"status":"failed"}), None),
            (json!({"status":"cancelled"}), None),
            (json!({}), None),
        ];
        for (response, expected) in cases {
            assert_eq!(responses_finish_reason(&response), expected, "{response}");
        }
        let mut o = ResponsesFinishObserver::default();
        o.feed_sse(&sse(&[
            json!({"type":"response.created","response":{"status":"in_progress"}}),
            json!({"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call"}]}}),
        ]));
        assert_eq!(o.finish(), Some("tool_calls"));
    }

    #[test]
    fn interleaved_reasoning_keeps_its_place() {
        let mut c = OutputCollector::new(OutputWire::OpenAiChat, 4096);
        c.feed_sse(&sse(&[
            json!({"choices":[{"index":0,"delta":{"reasoning_content":"A"}}]}),
            json!({"choices":[{"index":0,"delta":{"content":"x"}}]}),
            json!({"choices":[{"index":0,"delta":{"reasoning_content":"B"}}]}),
        ]));
        assert_eq!(
            parts(&c.finish()),
            json!([
                {"type":"reasoning","content":"A"},
                {"type":"text","content":"x"},
                {"type":"reasoning","content":"B"},
            ])
        );
    }

    #[test]
    fn a_tool_use_without_arguments_reports_an_empty_object() {
        let body = json!({"content":[{"type":"tool_use","id":"t","name":"now","input":{}}]});
        let out = OutputCollector::from_body(OutputWire::AnthropicMessages, 4096, &body);
        assert_eq!(
            parts(&out),
            json!([{"type":"tool_call","id":"t","name":"now","arguments":{}}])
        );
    }

    #[test]
    fn an_oversized_frame_marks_the_capture_cut() {
        let mut c = OutputCollector::new(OutputWire::OpenAiChat, 4096);
        c.feed_sse(&vec![b'x'; aisix_gateway::sse::MAX_SSE_FRAME_BYTES + 1]);
        assert!(c.finish().truncated);
    }

    #[test]
    fn unparseable_arguments_stay_a_string_and_the_cap_marks_truncation() {
        let body = json!({"choices":[{"index":0,"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{not json"}}]}}]});
        let out = OutputCollector::from_body(OutputWire::OpenAiChat, 4096, &body);
        assert_eq!(
            parts(&out),
            json!([{"type":"tool_call","id":"c","name":"f","arguments":"{not json"}])
        );

        let long = json!({"choices":[{"index":0,"message":{"content":"x".repeat(100)}}]});
        let out = OutputCollector::from_body(OutputWire::OpenAiChat, 10, &long);
        assert!(out.truncated);
    }

    #[test]
    fn anthropic_stream_keeps_thinking_text_and_tool_use_but_not_redacted_thinking() {
        let mut c = OutputCollector::new(OutputWire::AnthropicMessages, 4096);
        c.feed_sse(&sse(&[
            json!({"type":"message_start","message":{"id":"m"}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"ponder"}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"xx"}}),
            json!({"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Hello"}}),
            json!({"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"tu_1","name":"lookup","input":{}}}),
            json!({"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}),
            json!({"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"1}"}}),
        ]));
        assert_eq!(
            parts(&c.finish()),
            json!([
                {"type":"reasoning","content":"ponder"},
                {"type":"text","content":"Hello"},
                {"type":"tool_call","id":"tu_1","name":"lookup","arguments":{"q":1}},
            ])
        );
    }

    #[test]
    fn responses_body_skips_encrypted_reasoning_and_server_tools() {
        let body = json!({"output":[
            {"type":"reasoning","summary":[{"type":"summary_text","text":"why"}],"encrypted_content":"ENC"},
            {"type":"web_search_call","id":"ws","status":"completed"},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Answer"}]},
            {"type":"function_call","call_id":"call_9","name":"lookup","arguments":"{\"a\":1}"},
        ]});
        let out = OutputCollector::from_body(OutputWire::OpenAiResponses, 4096, &body);
        assert_eq!(
            parts(&out),
            json!([
                {"type":"reasoning","content":"why"},
                {"type":"text","content":"Answer"},
                {"type":"tool_call","id":"call_9","name":"lookup","arguments":{"a":1}},
            ])
        );
        assert!(!parts(&out).to_string().contains("ENC"));
    }

    #[test]
    fn responses_stream_assembles_deltas_and_lets_the_done_item_win() {
        let mut c = OutputCollector::new(OutputWire::OpenAiResponses, 4096);
        c.feed_sse(&sse(&[
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","summary":[]}}),
            json!({"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"plan"}),
            json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}),
            json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"x\":"}),
            json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":"2}"}),
            json!({"type":"response.output_text.delta","output_index":2,"content_index":0,"delta":"Do"}),
            json!({"type":"response.output_item.done","output_index":2,"item":{"type":"message","content":[{"type":"output_text","text":"Done"}]}}),
        ]));
        assert_eq!(
            parts(&c.finish()),
            json!([
                {"type":"reasoning","content":"plan"},
                {"type":"tool_call","id":"c1","name":"f","arguments":{"x":2}},
                {"type":"text","content":"Done"},
            ])
        );
    }
}
