//! What a streamed output guardrail's `max_buffer_bytes` measures (#513).
//!
//! While a stream is held back for output inspection
//! ([`aisix_guardrails::StreamOutputPolicy::BufferFull`]), the cap bounds
//! the model-generated content held: assistant text, reasoning, and
//! tool-call arguments. SSE and JSON framing — event names, ids, indexes,
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

use aisix_gateway::ChatDelta;
use serde_json::Value;

/// Raw bytes a hold-back may keep, as a multiple of `max_buffer_bytes`
/// (16 MiB at the 256 KiB default). Far above the framing an ordinary
/// token stream wraps around its content, so a normal response still trips
/// on content first.
pub(crate) const RAW_HOLD_FACTOR: usize = 64;

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

    /// Past either bound: the content cap, or the raw-byte guard derived
    /// from it.
    pub(crate) fn exceeds(&self, max_buffer_bytes: usize) -> bool {
        self.content > max_buffer_bytes
            || self.raw > max_buffer_bytes.saturating_mul(RAW_HOLD_FACTOR)
    }
}

/// Raw size of a held chat chunk: its serialized length, which is what the
/// chunk occupies once rendered at release.
pub(crate) fn chat_chunk_raw(chunk: &aisix_gateway::ChatChunk) -> usize {
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
    let _ = serde_json::to_writer(&mut c, chunk);
    c.0
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
/// is the generated text they inspect (assistant text and tool-call
/// arguments), `reasoning` the generated reasoning they do not inspect.
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

/// An OpenAI Responses stream event. Only delta events count: the `.done`
/// events and the terminal `response.*` snapshot repeat content already
/// counted from its deltas.
pub(crate) fn responses_event_parts(v: &Value) -> Parts {
    let mut p = Parts::default();
    match v.get("type").and_then(Value::as_str) {
        Some(
            "response.output_text.delta"
            | "response.function_call_arguments.delta"
            | "response.mcp_call_arguments.delta"
            | "response.custom_tool_call_input.delta",
        ) => p.scan_str(v.get("delta")),
        Some("response.reasoning_text.delta" | "response.reasoning_summary_text.delta") => {
            p.reasoning_str(v.get("delta"))
        }
        _ => {}
    }
    p
}

/// An OpenAI chat-completions stream chunk: every choice's `delta.content`
/// and tool-call arguments are scanned; `reasoning_content` (or the
/// `reasoning` spelling some relays use) is reasoning.
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
                }
            }
            _ => {}
        }
        for tc in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            p.scan_str(tc.get("function").and_then(|f| f.get("arguments")));
            p.scan_str(tc.get("custom").and_then(|c| c.get("input")));
        }
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
    fn responses_frames_count_deltas_not_snapshots() {
        let frames = concat!(
            "event: response.reasoning_summary_text.delta\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"think\"}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "event: response.output_text.done\n",
            "data: {\"type\":\"response.output_text.done\",\"text\":\"hi\"}\n\n",
            "data: [DONE]\n\n",
        );
        assert_eq!(sse_frames(frames.as_bytes(), responses_event), 5 + 2);
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
    fn unparseable_payload_counts_whole() {
        assert_eq!(sse_frames(b"data: not json\n\n", responses_event), 8);
    }
}
