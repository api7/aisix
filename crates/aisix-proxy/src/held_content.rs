//! What a streamed output guardrail's `max_buffer_bytes` measures (#513).
//!
//! While a stream is held back for output inspection (a hold-back
//! [`aisix_guardrails::StreamOutputPolicy`]), the cap bounds
//! the model-generated content held: assistant text, refusals, reasoning,
//! and tool-call arguments. SSE and JSON framing — event names, ids, indexes,
//! the envelope around each delta — is never counted, so the same response
//! trips the cap at the same point whichever route and wire protocol
//! carries it.
//!
//! Every hold-back route measures through this module so the definition
//! cannot drift between them. A payload that is not one JSON document
//! counts whole: nothing can separate its content from its envelope.
//!
//! Content is not memory, though: the held buffer keeps every frame whole,
//! and frames that carry no content (pings, snapshots, base64 media) count
//! nothing. So each hold-back also bounds the raw bytes it keeps, at
//! [`RAW_HOLD_FACTOR`] times the same cap, through [`HeldBuffer`]. Crossing
//! either bound is the same buffer-exceeded event.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

use aisix_gateway::ChatDelta;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Raw bytes a hold-back may keep, as a multiple of `max_buffer_bytes`
/// (32 MiB at the 256 KiB default). Above the framing an ordinary token
/// stream wraps around its content — OpenAI streams of CJK text measure
/// about 75× their content, on Chat Completions and Responses alike — so a
/// normal response still trips on content first.
pub(crate) const RAW_HOLD_FACTOR: usize = 128;

/// What one hold-back buffer holds: generated content (the cap
/// `max_buffer_bytes` names) and the raw bytes kept to hold it.
#[derive(Debug, Default)]
pub(crate) struct HeldBuffer {
    content: usize,
    raw: usize,
}

impl HeldBuffer {
    pub(crate) fn hold(&mut self, content: usize, raw: usize) {
        self.content = self.content.saturating_add(content);
        self.raw = self.raw.saturating_add(raw);
    }

    /// Whether admitting one more held frame would cross either bound.
    ///
    /// The relay uses this before decoding an unterminated terminal frame so
    /// the configured raw-buffer cap wins over its otherwise fail-closed
    /// malformed-frame handling.
    pub(crate) fn would_exceed_after(
        &self,
        content: usize,
        raw: usize,
        max_buffer_bytes: usize,
    ) -> bool {
        self.content.saturating_add(content) > max_buffer_bytes
            || self.raw.saturating_add(raw) > max_buffer_bytes.saturating_mul(RAW_HOLD_FACTOR)
    }

    /// Past either bound: the content cap, or the raw-byte guard derived
    /// from it.
    pub(crate) fn exceeds(&self, max_buffer_bytes: usize) -> bool {
        self.content > max_buffer_bytes
            || self.raw > max_buffer_bytes.saturating_mul(RAW_HOLD_FACTOR)
    }
}

/// Raw bytes held back right now, across every stream in the process.
static HOLDBACK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Bytes currently held back by streamed output guardrails, process-wide.
pub fn holdback_bytes() -> usize {
    HOLDBACK_BYTES.load(Ordering::Relaxed)
}

/// One hold-back buffer's share of [`holdback_bytes`].
///
/// Owned next to the buffer it describes and dropped with it, so a stream
/// that ends without releasing — a client that went away, a block, an
/// error — gives its share back without a release path having to.
#[derive(Debug, Default)]
pub(crate) struct HeldBytes(usize);

impl HeldBytes {
    pub(crate) fn add(&mut self, n: usize) {
        self.0 += n;
        HOLDBACK_BYTES.fetch_add(n, Ordering::Relaxed);
    }

    /// The buffer now holds exactly `n` bytes — after a partial release,
    /// or a rewrite in place that changed its length.
    pub(crate) fn set(&mut self, n: usize) {
        if n >= self.0 {
            HOLDBACK_BYTES.fetch_add(n - self.0, Ordering::Relaxed);
        } else {
            HOLDBACK_BYTES.fetch_sub(self.0 - n, Ordering::Relaxed);
        }
        self.0 = n;
    }

    pub(crate) fn clear(&mut self) {
        self.set(0);
    }
}

impl Drop for HeldBytes {
    fn drop(&mut self) {
        self.clear();
    }
}

/// The leading JSON values that fit in `cap` serialized bytes. Once one
/// does not fit, nothing after it is kept, so the values stay a prefix of
/// the stream.
#[derive(Debug, Default)]
pub(crate) struct BoundedValues {
    values: Vec<Value>,
    bytes: usize,
    full: bool,
}

impl BoundedValues {
    pub(crate) fn push(&mut self, v: &Value, cap: usize) {
        if self.full {
            return;
        }
        let len = json_len(v);
        if self.bytes.saturating_add(len) > cap {
            self.full = true;
            return;
        }
        self.bytes += len;
        self.values.push(v.clone());
    }

    /// Whether a value was refused for lack of room.
    pub(crate) fn is_full(&self) -> bool {
        self.full
    }

    /// The kept values, or `None` when there are none.
    pub(crate) fn take(&mut self) -> Option<Vec<Value>> {
        self.bytes = 0;
        (!self.values.is_empty()).then(|| std::mem::take(&mut self.values))
    }
}

fn json_len(v: &impl serde::Serialize) -> usize {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 += buf.len();
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut c = Count(0);
    let _ = serde_json::to_writer(&mut c, v);
    c.0
}

/// Raw size of a held chat chunk: its serialized length, which is what the
/// chunk occupies once rendered at release.
pub(crate) fn chat_chunk_raw(chunk: &aisix_gateway::ChatChunk) -> usize {
    json_len(chunk)
}

/// Held content in one normalised chat delta.
pub(crate) fn chat_delta(delta: &ChatDelta) -> usize {
    let text = delta.content.as_deref().map_or(0, str::len);
    let reasoning = delta.reasoning_content.as_deref().map_or(0, str::len);
    let tool_args: usize = delta
        .tool_calls
        .iter()
        .flatten()
        .map(|tc| {
            let function = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str);
            let custom = tc
                .get("custom")
                .and_then(|c| c.get("input"))
                .and_then(Value::as_str);
            function.map_or(0, str::len) + custom.map_or(0, str::len)
        })
        .sum();
    text + reasoning + tool_args
}

/// One stream event split the way the output guardrails read it: `scan`
/// is the generated text they inspect (assistant text, refusals, and
/// tool-call arguments), `reasoning` the generated reasoning they do not
/// inspect.
/// Both count toward the hold-back cap, so [`Parts::held`] is the cap's
/// measure and `scan` the scanner's input — one extraction for both.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Parts {
    pub(crate) scan: String,
    pub(crate) reasoning: usize,
}

impl Parts {
    pub(crate) fn held(&self) -> usize {
        self.scan.len() + self.reasoning
    }

    fn scan_str(&mut self, v: Option<&Value>) {
        if let Some(s) = v.and_then(Value::as_str) {
            self.scan.push_str(s);
        }
    }

    fn reasoning_str(&mut self, v: Option<&Value>) {
        self.reasoning += v.and_then(Value::as_str).map_or(0, str::len);
    }
}

#[derive(Clone, Copy)]
enum ResponsesCarrierMode {
    Delta,
    Snapshot,
}

const MAX_RESPONSES_HELD_CARRIERS: usize = 1_024;
const MAX_RESPONSES_HELD_ID_BYTES: usize = 512;

struct ResponsesHeldCarrier {
    bytes: usize,
    hash: Sha256,
}

/// Logical generated content currently retained by a Responses SSE
/// hold-back buffer. OpenAI emits the same text as deltas, direct `.done`
/// events, content-part/item snapshots, and the terminal response object.
/// The raw-byte cap still counts every frame, while this state prevents those
/// equivalent source carriers from consuming the content cap repeatedly.
#[derive(Default)]
pub(crate) struct ResponsesHeldContent {
    carriers: HashMap<String, ResponsesHeldCarrier>,
    saturated: bool,
}

impl ResponsesHeldContent {
    /// Returns `None` when this is not one parseable Responses SSE frame; the
    /// caller then uses the conservative generic held-content extraction.
    pub(crate) fn observe_sse_frame(&mut self, frame: &[u8]) -> Option<usize> {
        if self.saturated {
            return None;
        }
        let payload = crate::redact::frame_payload(frame)?;
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            return Some(0);
        }
        let event = serde_json::from_str::<Value>(payload).ok()?;
        self.observe_event(&event)
    }

    /// Records one parsed event. `None` asks the caller to use the generic
    /// content counter: either the ledger has reached its fixed key budget,
    /// or this event was the one that reached it.
    pub(crate) fn observe_event(&mut self, event: &Value) -> Option<usize> {
        if self.saturated {
            return None;
        }
        let held = self.observe_event_inner(event);
        (!self.saturated).then_some(held)
    }

    fn observe_event_inner(&mut self, event: &Value) -> usize {
        match event.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => self.observe_event_content(
                event,
                "text",
                event.get("delta").and_then(Value::as_str),
                ResponsesCarrierMode::Delta,
            ),
            Some("response.refusal.delta") => self.observe_event_content(
                event,
                "refusal",
                event.get("delta").and_then(Value::as_str),
                ResponsesCarrierMode::Delta,
            ),
            Some("response.output_text.done") => self.observe_event_content(
                event,
                "text",
                event.get("text").and_then(Value::as_str),
                ResponsesCarrierMode::Snapshot,
            ),
            Some("response.refusal.done") => self.observe_event_content(
                event,
                "refusal",
                event.get("refusal").and_then(Value::as_str),
                ResponsesCarrierMode::Snapshot,
            ),
            Some("response.function_call_arguments.delta") => self.observe_event_tool(
                event,
                "function_call",
                "arguments",
                event.get("delta").and_then(Value::as_str),
                ResponsesCarrierMode::Delta,
            ),
            Some("response.mcp_call_arguments.delta") => self.observe_event_tool(
                event,
                "mcp_call",
                "arguments",
                event.get("delta").and_then(Value::as_str),
                ResponsesCarrierMode::Delta,
            ),
            Some("response.custom_tool_call_input.delta") => self.observe_event_tool(
                event,
                "custom_tool_call",
                "input",
                event.get("delta").and_then(Value::as_str),
                ResponsesCarrierMode::Delta,
            ),
            Some("response.function_call_arguments.done") => {
                self.observe_event_tool(
                    event,
                    "function_call",
                    "name",
                    event.get("name").and_then(Value::as_str),
                    ResponsesCarrierMode::Snapshot,
                ) + self.observe_event_tool(
                    event,
                    "function_call",
                    "arguments",
                    event.get("arguments").and_then(Value::as_str),
                    ResponsesCarrierMode::Snapshot,
                )
            }
            Some("response.mcp_call_arguments.done") => {
                self.observe_event_tool(
                    event,
                    "mcp_call",
                    "name",
                    event.get("name").and_then(Value::as_str),
                    ResponsesCarrierMode::Snapshot,
                ) + self.observe_event_tool(
                    event,
                    "mcp_call",
                    "arguments",
                    event.get("arguments").and_then(Value::as_str),
                    ResponsesCarrierMode::Snapshot,
                )
            }
            Some("response.custom_tool_call_input.done") => {
                self.observe_event_tool(
                    event,
                    "custom_tool_call",
                    "name",
                    event.get("name").and_then(Value::as_str),
                    ResponsesCarrierMode::Snapshot,
                ) + self.observe_event_tool(
                    event,
                    "custom_tool_call",
                    "input",
                    event.get("input").and_then(Value::as_str),
                    ResponsesCarrierMode::Snapshot,
                )
            }
            Some("response.content_part.added" | "response.content_part.done") => event
                .get("part")
                .map(|part| self.observe_event_part(event, part, ResponsesCarrierMode::Snapshot))
                .unwrap_or(0),
            Some("response.output_item.added" | "response.output_item.done") => event
                .get("item")
                .map(|item| {
                    self.observe_item(
                        item,
                        response_item_base(event, item),
                        ResponsesCarrierMode::Snapshot,
                    )
                })
                .unwrap_or(0),
            Some("response.completed" | "response.incomplete" | "response.failed") => event
                .get("response")
                .and_then(|response| response.get("output"))
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .enumerate()
                        .map(|(index, item)| {
                            self.observe_item(
                                item,
                                response_item_base_at(item, index),
                                ResponsesCarrierMode::Snapshot,
                            )
                        })
                        .sum()
                })
                .unwrap_or(0),
            Some("response.reasoning_text.delta") => self.observe_event_reasoning(
                event,
                "content",
                event.get("content_index").and_then(Value::as_u64),
                event.get("delta").and_then(Value::as_str),
                ResponsesCarrierMode::Delta,
            ),
            Some("response.reasoning_summary_text.delta") => self.observe_event_reasoning(
                event,
                "summary",
                event.get("summary_index").and_then(Value::as_u64),
                event.get("delta").and_then(Value::as_str),
                ResponsesCarrierMode::Delta,
            ),
            Some("response.reasoning_text.done") => self.observe_event_reasoning(
                event,
                "content",
                event.get("content_index").and_then(Value::as_u64),
                event.get("text").and_then(Value::as_str),
                ResponsesCarrierMode::Snapshot,
            ),
            Some("response.reasoning_summary_text.done") => self.observe_event_reasoning(
                event,
                "summary",
                event.get("summary_index").and_then(Value::as_u64),
                event.get("text").and_then(Value::as_str),
                ResponsesCarrierMode::Snapshot,
            ),
            Some(
                "response.reasoning_summary_part.added" | "response.reasoning_summary_part.done",
            ) => event
                .get("part")
                .and_then(|part| part.get("text"))
                .and_then(Value::as_str)
                .map(|text| {
                    self.observe_event_reasoning(
                        event,
                        "summary",
                        event.get("summary_index").and_then(Value::as_u64),
                        Some(text),
                        ResponsesCarrierMode::Snapshot,
                    )
                })
                .unwrap_or(0),
            _ => 0,
        }
    }

    fn observe_event_content(
        &mut self,
        event: &Value,
        field: &str,
        text: Option<&str>,
        mode: ResponsesCarrierMode,
    ) -> usize {
        let key = response_event_base(event).and_then(|base| {
            event
                .get("content_index")
                .and_then(Value::as_u64)
                .map(|index| format!("{base}/content/{index}/{field}"))
        });
        self.observe_text(key, text, mode)
    }

    fn observe_event_part(
        &mut self,
        event: &Value,
        part: &Value,
        mode: ResponsesCarrierMode,
    ) -> usize {
        match part.get("type").and_then(Value::as_str) {
            Some("reasoning_text") => self.observe_event_reasoning(
                event,
                "content",
                event.get("content_index").and_then(Value::as_u64),
                part.get("text").and_then(Value::as_str),
                mode,
            ),
            Some("summary_text") => self.observe_event_reasoning(
                event,
                "summary",
                event.get("summary_index").and_then(Value::as_u64),
                part.get("text").and_then(Value::as_str),
                mode,
            ),
            _ => {
                let Some((field, text)) = responses_part_text(part) else {
                    return 0;
                };
                self.observe_event_content(event, field, Some(text), mode)
            }
        }
    }

    fn observe_event_tool(
        &mut self,
        event: &Value,
        tool_type: &str,
        field: &str,
        text: Option<&str>,
        mode: ResponsesCarrierMode,
    ) -> usize {
        let key = response_event_base(event).map(|base| format!("{base}/tool/{tool_type}/{field}"));
        self.observe_text(key, text, mode)
    }

    fn observe_event_reasoning(
        &mut self,
        event: &Value,
        group: &str,
        index: Option<u64>,
        text: Option<&str>,
        mode: ResponsesCarrierMode,
    ) -> usize {
        let key = response_event_base(event)
            .zip(index)
            .map(|(base, index)| format!("{base}/reasoning/{group}/{index}"));
        self.observe_text(key, text, mode)
    }

    fn observe_item(
        &mut self,
        item: &Value,
        base: Option<String>,
        mode: ResponsesCarrierMode,
    ) -> usize {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => item
                .get("content")
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .enumerate()
                        .map(|(index, part)| {
                            let Some((field, text)) = responses_part_text(part) else {
                                return 0;
                            };
                            self.observe_text(
                                base.as_ref()
                                    .map(|base| format!("{base}/content/{index}/{field}")),
                                Some(text),
                                mode,
                            )
                        })
                        .sum()
                })
                .unwrap_or(0),
            Some("reasoning") => ["content", "summary"]
                .into_iter()
                .map(|group| {
                    item.get(group)
                        .and_then(Value::as_array)
                        .map(|parts| {
                            parts
                                .iter()
                                .enumerate()
                                .map(|(index, part)| {
                                    self.observe_text(
                                        base.as_ref().map(|base| {
                                            format!("{base}/reasoning/{group}/{index}")
                                        }),
                                        part.get("text").and_then(Value::as_str),
                                        mode,
                                    )
                                })
                                .sum::<usize>()
                        })
                        .unwrap_or(0)
                })
                .sum(),
            Some("function_call" | "mcp_call") => {
                let tool_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
                self.observe_text(
                    base.as_ref()
                        .map(|base| format!("{base}/tool/{tool_type}/name")),
                    item.get("name").and_then(Value::as_str),
                    mode,
                ) + self.observe_text(
                    base.map(|base| format!("{base}/tool/{tool_type}/arguments")),
                    item.get("arguments").and_then(Value::as_str),
                    mode,
                )
            }
            Some("custom_tool_call") => {
                self.observe_text(
                    base.as_ref()
                        .map(|base| format!("{base}/tool/custom_tool_call/name")),
                    item.get("name").and_then(Value::as_str),
                    mode,
                ) + self.observe_text(
                    base.map(|base| format!("{base}/tool/custom_tool_call/input")),
                    item.get("input").and_then(Value::as_str),
                    mode,
                )
            }
            _ => 0,
        }
    }

    fn observe_text(
        &mut self,
        key: Option<String>,
        text: Option<&str>,
        mode: ResponsesCarrierMode,
    ) -> usize {
        let Some(text) = text.filter(|text| !text.is_empty()) else {
            return 0;
        };
        let Some(key) = key else {
            // A missing/conflicting coordinate must never borrow another
            // carrier's ledger entry. Charge it in full without retaining an
            // unbounded anonymous key.
            return text.len();
        };
        match mode {
            ResponsesCarrierMode::Delta => {
                if !self.carriers.contains_key(&key)
                    && self.carriers.len() >= MAX_RESPONSES_HELD_CARRIERS
                {
                    self.saturated = true;
                    return text.len();
                }
                match self.carriers.entry(key) {
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        let entry = entry.get_mut();
                        entry.bytes = entry.bytes.saturating_add(text.len());
                        entry.hash.update(text.as_bytes());
                        text.len()
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let mut hash = Sha256::new();
                        hash.update(text.as_bytes());
                        entry.insert(ResponsesHeldCarrier {
                            bytes: text.len(),
                            hash,
                        });
                        text.len()
                    }
                }
            }
            ResponsesCarrierMode::Snapshot => {
                if !self.carriers.contains_key(&key) {
                    if self.carriers.len() >= MAX_RESPONSES_HELD_CARRIERS {
                        self.saturated = true;
                        return text.len();
                    }
                    let mut hash = Sha256::new();
                    hash.update(text.as_bytes());
                    self.carriers.insert(
                        key,
                        ResponsesHeldCarrier {
                            bytes: text.len(),
                            hash,
                        },
                    );
                    return text.len();
                }
                let previous = self
                    .carriers
                    .get_mut(&key)
                    .expect("carrier was present immediately before lookup");
                if text.len() == previous.bytes
                    && Sha256::digest(text.as_bytes()) == previous.hash.clone().finalize()
                {
                    0
                } else {
                    let prefix_matches = text.get(..previous.bytes).is_some_and(|prefix| {
                        Sha256::digest(prefix.as_bytes()) == previous.hash.clone().finalize()
                    });
                    if prefix_matches {
                        let suffix = text
                            .get(previous.bytes..)
                            .expect("validated UTF-8 prefix boundary");
                        previous.bytes = text.len();
                        previous.hash.update(suffix.as_bytes());
                        suffix.len()
                    } else {
                        let mut hash = Sha256::new();
                        hash.update(text.as_bytes());
                        *previous = ResponsesHeldCarrier {
                            bytes: text.len(),
                            hash,
                        };
                        text.len()
                    }
                }
            }
        }
    }
}

fn response_event_base(event: &Value) -> Option<String> {
    response_base(
        event.get("item_id").and_then(Value::as_str),
        event.get("output_index").and_then(Value::as_u64),
    )
}

fn response_item_base(event: &Value, item: &Value) -> Option<String> {
    let top_level_id = event.get("item_id").and_then(Value::as_str);
    let item_id = item.get("id").and_then(Value::as_str)?;
    if top_level_id.is_some_and(|top_level_id| top_level_id != item_id) {
        return None;
    }
    response_base(
        Some(item_id),
        event.get("output_index").and_then(Value::as_u64),
    )
}

fn response_item_base_at(item: &Value, output_index: usize) -> Option<String> {
    response_base(
        item.get("id").and_then(Value::as_str),
        u64::try_from(output_index).ok(),
    )
}

fn response_base(item_id: Option<&str>, output_index: Option<u64>) -> Option<String> {
    let item_id = item_id?;
    let output_index = output_index?;
    (!item_id.is_empty() && item_id.len() <= MAX_RESPONSES_HELD_ID_BYTES)
        .then(|| format!("{}:{item_id}:{output_index}", item_id.len()))
}

fn responses_part_text(part: &Value) -> Option<(&'static str, &str)> {
    match part.get("type").and_then(Value::as_str) {
        Some("output_text" | "text" | "input_text") => part
            .get("text")
            .and_then(Value::as_str)
            .map(|text| ("text", text)),
        Some("refusal") => part
            .get("refusal")
            .and_then(Value::as_str)
            .map(|text| ("refusal", text)),
        _ => None,
    }
}

/// An Anthropic Messages stream event: `text` and `partial_json` deltas and
/// the text or tool input a `content_block_start` already carries are
/// scanned; `thinking` is reasoning.
pub(crate) fn anthropic_event_parts(v: &Value) -> Parts {
    let mut p = Parts::default();
    match v.get("type").and_then(Value::as_str) {
        Some("content_block_delta") => {
            let d = v.get("delta");
            p.scan_str(d.and_then(|d| d.get("text")));
            p.scan_str(d.and_then(|d| d.get("partial_json")));
            p.reasoning_str(d.and_then(|d| d.get("thinking")));
        }
        Some("content_block_start") => {
            let cb = v.get("content_block");
            p.scan_str(cb.and_then(|c| c.get("text")));
            if let Some(input) = cb
                .and_then(|c| c.get("input"))
                .filter(|i| !i.is_null() && i.as_object().is_none_or(|o| !o.is_empty()))
            {
                p.scan.push_str(&input.to_string());
            }
            p.reasoning_str(cb.and_then(|c| c.get("thinking")));
        }
        _ => {}
    }
    p
}

fn responses_part_parts(parts: &mut Parts, part: &Value, reasoning: bool) {
    let text = match part.get("type").and_then(Value::as_str) {
        Some("output_text" | "text" | "input_text") => part.get("text"),
        Some("refusal") => part.get("refusal"),
        Some("reasoning_text" | "summary_text") => {
            parts.reasoning_str(part.get("text"));
            return;
        }
        _ => return,
    };
    if reasoning {
        parts.reasoning_str(text);
    } else {
        parts.scan_str(text);
    }
}

fn responses_item_parts(parts: &mut Parts, item: &Value) {
    match item.get("type").and_then(Value::as_str) {
        Some("reasoning") => {
            for key in ["content", "summary"] {
                for part in item
                    .get(key)
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    responses_part_parts(parts, part, true);
                }
            }
        }
        Some("message") => {
            for part in item
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                responses_part_parts(parts, part, false);
            }
        }
        Some("function_call" | "mcp_call") => {
            parts.scan_str(item.get("name"));
            parts.scan_str(item.get("arguments"));
        }
        Some("custom_tool_call") => {
            parts.scan_str(item.get("name"));
            parts.scan_str(item.get("input"));
        }
        _ => {}
    }
}

/// An OpenAI Responses stream event. A stream can legally end on a direct
/// `.done`, content-part, output-item, or terminal snapshot event without a
/// preceding delta, so every client-visible carrier contributes to the output
/// scan. The generic held-content counter below is deliberately conservative;
/// held Responses routes use [`ResponsesHeldContent`] to avoid charging the
/// same identified carrier again. Reasoning remains held but outside the
/// output-guardrail scan.
pub(crate) fn responses_event_parts(v: &Value) -> Parts {
    let mut p = Parts::default();
    match v.get("type").and_then(Value::as_str) {
        Some(
            "response.output_text.delta"
            | "response.refusal.delta"
            | "response.function_call_arguments.delta"
            | "response.mcp_call_arguments.delta"
            | "response.custom_tool_call_input.delta",
        ) => p.scan_str(v.get("delta")),
        Some("response.output_text.done") => p.scan_str(v.get("text")),
        Some("response.refusal.done") => p.scan_str(v.get("refusal")),
        Some("response.function_call_arguments.done" | "response.mcp_call_arguments.done") => {
            p.scan_str(v.get("name"));
            p.scan_str(v.get("arguments"));
        }
        Some("response.custom_tool_call_input.done") => {
            p.scan_str(v.get("name"));
            p.scan_str(v.get("input"));
        }
        Some("response.content_part.added" | "response.content_part.done") => {
            if let Some(part) = v.get("part") {
                responses_part_parts(&mut p, part, false);
            }
        }
        Some("response.output_item.added" | "response.output_item.done") => {
            if let Some(item) = v.get("item") {
                responses_item_parts(&mut p, item);
            }
        }
        Some("response.completed" | "response.incomplete" | "response.failed") => {
            for item in v
                .get("response")
                .and_then(|response| response.get("output"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                responses_item_parts(&mut p, item);
            }
        }
        Some("response.reasoning_text.delta" | "response.reasoning_summary_text.delta") => {
            p.reasoning_str(v.get("delta"))
        }
        Some("response.reasoning_text.done" | "response.reasoning_summary_text.done") => {
            p.reasoning_str(v.get("text"))
        }
        Some("response.reasoning_summary_part.added" | "response.reasoning_summary_part.done") => {
            if let Some(part) = v.get("part") {
                responses_part_parts(&mut p, part, true);
            }
        }
        _ => {}
    }
    p
}

/// An OpenAI chat-completions stream chunk: every choice's `delta.content`
/// refusals, and tool-call arguments are scanned; `reasoning_content` (or
/// the `reasoning` spelling some relays use) is reasoning.
pub(crate) fn chat_chunk_parts(v: &Value) -> Parts {
    let mut p = Parts::default();
    for delta in v
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|c| c.get("delta"))
    {
        match delta.get("content") {
            Some(Value::String(s)) => p.scan.push_str(s),
            Some(Value::Array(parts)) => {
                for part in parts {
                    p.scan_str(part.get("text"));
                    if part.get("type").and_then(Value::as_str) == Some("refusal") {
                        p.scan_str(part.get("refusal"));
                    }
                }
            }
            _ => {}
        }
        p.scan_str(delta.get("refusal"));
        for tc in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            p.scan_str(tc.get("function").and_then(|f| f.get("arguments")));
            p.scan_str(tc.get("custom").and_then(|c| c.get("input")));
        }
        p.scan_str(delta.get("function_call").and_then(|f| f.get("arguments")));
        p.reasoning_str(delta.get("reasoning_content"));
        p.reasoning_str(delta.get("reasoning"));
    }
    p
}

/// A legacy completions stream chunk: every choice's `text` is scanned.
pub(crate) fn completions_chunk_parts(v: &Value) -> Parts {
    let mut p = Parts::default();
    for c in v
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        p.scan_str(c.get("text"));
    }
    p
}

/// Held content in one Anthropic Messages stream event.
pub(crate) fn anthropic_event(v: &Value) -> usize {
    anthropic_event_parts(v).held()
}

/// Held content in one OpenAI Responses stream event.
pub(crate) fn responses_event(v: &Value) -> usize {
    responses_event_parts(v).held()
}

/// Held content across every SSE frame in `frames` (complete frames; an
/// unterminated trailing fragment counts as one more frame).
pub(crate) fn sse_frames(frames: &[u8], per_event: fn(&Value) -> usize) -> usize {
    crate::redact::sse_frame_payloads(frames)
        .iter()
        .map(|payload| {
            let p = payload.trim();
            if p.is_empty() || p == "[DONE]" {
                return 0;
            }
            match serde_json::from_str::<Value>(p) {
                Ok(v) => per_event(&v),
                Err(_) => p.len(),
            }
        })
        .sum()
}

/// Held Responses content across every SSE frame in `frames`, with one
/// bounded source ledger shared by the complete frames and the final tail.
/// Once the ledger cannot safely identify more source carriers, fall back to
/// the generic counter rather than treating later content as free.
pub(crate) fn responses_sse_held_frames(ledger: &mut ResponsesHeldContent, frames: &[u8]) -> usize {
    crate::redact::sse_frame_payloads(frames)
        .iter()
        .map(|payload| {
            let payload = payload.trim();
            if payload.is_empty() || payload == "[DONE]" {
                return 0;
            }
            match serde_json::from_str::<Value>(payload) {
                Ok(event) => ledger
                    .observe_event(&event)
                    .unwrap_or_else(|| responses_event(&event)),
                Err(_) => payload.len(),
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_delta_counts_text_reasoning_and_tool_arguments_only() {
        let delta = ChatDelta {
            role: None,
            content: Some("abc".into()),
            reasoning_content: Some("de".into()),
            tool_calls: Some(vec![
                json!({"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":1}"}}),
            ]),
        };
        assert_eq!(chat_delta(&delta), 3 + 2 + 7);
    }

    #[test]
    fn chat_chunk_counts_refusals_and_legacy_tool_arguments_as_scanned_content() {
        let direct = "direct streamed refusal";
        let direct_parts = chat_chunk_parts(&json!({
            "choices": [{"delta": {"refusal": direct}}],
        }));
        assert_eq!(direct_parts.scan, direct);
        assert_eq!(direct_parts.held(), direct.len());

        let typed = "typed streamed refusal";
        let typed_parts = chat_chunk_parts(&json!({
            "choices": [{"delta": {"content": [{"type": "refusal", "refusal": typed}]}}],
        }));
        assert_eq!(typed_parts.scan, typed);
        assert_eq!(typed_parts.held(), typed.len());

        let opaque_parts = chat_chunk_parts(&json!({
            "choices": [{"delta": {"content": [{"type": "future_media", "refusal": typed}]}}],
        }));
        assert!(opaque_parts.scan.is_empty());
        assert_eq!(opaque_parts.held(), 0);

        let arguments = "legacy streamed tool arguments";
        let legacy_parts = chat_chunk_parts(&json!({
            "choices": [{"delta": {"function_call": {"arguments": arguments}}}],
        }));
        assert_eq!(legacy_parts.scan, arguments);
        assert_eq!(legacy_parts.held(), arguments.len());
    }

    #[test]
    fn anthropic_event_ignores_envelope_and_empty_tool_input() {
        let start = json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t","name":"lookup","input":{}}});
        assert_eq!(anthropic_event(&start), 0);
        let thinking = json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}});
        assert_eq!(anthropic_event(&thinking), 3);
        let args = json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\""}});
        assert_eq!(anthropic_event(&args), 4);
        let stop = json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}});
        assert_eq!(anthropic_event(&stop), 0);
    }

    #[test]
    fn responses_frames_count_every_authoritative_output_carrier() {
        let frames = concat!(
            "event: response.reasoning_summary_text.delta\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"think\"}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "event: response.refusal.delta\n",
            "data: {\"type\":\"response.refusal.delta\",\"delta\":\"no\"}\n\n",
            "event: response.output_text.done\n",
            "data: {\"type\":\"response.output_text.done\",\"text\":\"hi\"}\n\n",
            "data: [DONE]\n\n",
        );
        assert_eq!(
            sse_frames(frames.as_bytes(), responses_event),
            5 + 2 + 2 + 2
        );
    }

    fn responses_frame(event: Value) -> Vec<u8> {
        format!("data: {event}\n\n").into_bytes()
    }

    #[test]
    fn responses_held_content_counts_repeated_output_snapshots_once() {
        let text = "x".repeat(300);
        let frames = [
            responses_frame(json!({
                "type": "response.output_text.delta",
                "item_id": "message_1",
                "output_index": 0,
                "content_index": 0,
                "delta": text.as_str(),
            })),
            responses_frame(json!({
                "type": "response.output_text.done",
                "item_id": "message_1",
                "output_index": 0,
                "content_index": 0,
                "text": text.as_str(),
            })),
            responses_frame(json!({
                "type": "response.content_part.done",
                "item_id": "message_1",
                "output_index": 0,
                "content_index": 0,
                "part": { "type": "output_text", "text": text.as_str() },
            })),
            responses_frame(json!({
                "type": "response.output_item.done",
                "item_id": "message_1",
                "output_index": 0,
                "item": {
                    "id": "message_1",
                    "type": "message",
                    "content": [{ "type": "output_text", "text": text.as_str() }],
                },
            })),
            responses_frame(json!({
                "type": "response.completed",
                "response": {
                    "output": [{
                        "id": "message_1",
                        "type": "message",
                        "content": [{ "type": "output_text", "text": text.as_str() }],
                    }],
                },
            })),
        ];
        let mut ledger = ResponsesHeldContent::default();

        assert_eq!(
            responses_sse_held_frames(&mut ledger, &frames[0]),
            text.len(),
            "the initial delta establishes the logical carrier"
        );
        assert_eq!(
            responses_sse_held_frames(&mut ledger, &frames[1..].concat()),
            0,
            "done, part, item, and terminal forms share that carrier across reads"
        );
    }

    #[test]
    fn responses_held_content_deduplicates_reasoning_part_snapshots() {
        let text = "think";
        let frames = [
            responses_frame(json!({
                "type": "response.reasoning_text.delta",
                "item_id": "reasoning_1",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            })),
            responses_frame(json!({
                "type": "response.content_part.done",
                "item_id": "reasoning_1",
                "output_index": 0,
                "content_index": 0,
                "part": { "type": "reasoning_text", "text": text },
            })),
            responses_frame(json!({
                "type": "response.output_item.done",
                "item_id": "reasoning_1",
                "output_index": 0,
                "item": {
                    "id": "reasoning_1",
                    "type": "reasoning",
                    "content": [{ "type": "reasoning_text", "text": text }],
                },
            })),
        ];
        let mut ledger = ResponsesHeldContent::default();

        assert_eq!(
            frames
                .iter()
                .map(|frame| ledger.observe_sse_frame(frame).unwrap())
                .sum::<usize>(),
            text.len()
        );
    }

    #[test]
    fn responses_held_content_counts_extensions_and_unidentified_snapshots() {
        let mut ledger = ResponsesHeldContent::default();
        let delta = responses_frame(json!({
            "type": "response.output_text.delta",
            "item_id": "message_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "abc",
        }));
        let extended = responses_frame(json!({
            "type": "response.output_text.done",
            "item_id": "message_1",
            "output_index": 0,
            "content_index": 0,
            "text": "abcdef",
        }));
        let unkeyed = responses_frame(json!({
            "type": "response.output_text.done",
            "text": "abc",
        }));

        assert_eq!(ledger.observe_sse_frame(&delta), Some(3));
        assert_eq!(ledger.observe_sse_frame(&extended), Some(3));
        assert_eq!(ledger.observe_sse_frame(&unkeyed), Some(3));
        assert_eq!(ledger.observe_sse_frame(&unkeyed), Some(3));
    }

    #[test]
    fn held_buffer_trips_on_content_or_on_raw_bytes() {
        let mut b = HeldBuffer::default();
        b.hold(10, 10 * RAW_HOLD_FACTOR);
        assert!(!b.exceeds(10));
        b.hold(1, 0);
        assert!(b.exceeds(10), "content past the cap");
        let mut b = HeldBuffer::default();
        b.hold(0, 10 * RAW_HOLD_FACTOR + 1);
        assert!(b.exceeds(10), "content-free bytes past the raw guard");
    }

    #[test]
    fn bounded_values_stop_at_their_own_size() {
        let empty_delta = json!({"index": 0});
        let len = serde_json::to_string(&empty_delta).unwrap().len();
        let mut kept = BoundedValues::default();
        for _ in 0..1_000 {
            kept.push(&empty_delta, 10 * len);
        }
        assert_eq!(kept.take().map(|v| v.len()), Some(10));
        assert_eq!(kept.take(), None);
    }

    #[test]
    fn bounded_values_stay_a_prefix() {
        let mut kept = BoundedValues::default();
        // 6 bytes fit, 10 more don't, and the 3 after them would.
        kept.push(&json!("aaaa"), 10);
        kept.push(&json!("bbbbbbbb"), 10);
        kept.push(&json!("c"), 10);
        assert_eq!(kept.take(), Some(vec![json!("aaaa")]));
    }

    #[test]
    fn unparseable_payload_counts_whole() {
        assert_eq!(sse_frames(b"data: not json\n\n", responses_event), 8);
    }
}
