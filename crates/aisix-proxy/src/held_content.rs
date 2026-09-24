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

use aisix_gateway::ChatDelta;
use serde_json::Value;

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

/// Held content in one Anthropic Messages stream event: `text`, `thinking`
/// and `partial_json` deltas, plus any text, thinking or tool input a
/// `content_block_start` already carries.
pub(crate) fn anthropic_event(v: &Value) -> usize {
    let str_len = |o: Option<&Value>, key: &str| {
        o.and_then(|o| o.get(key))
            .and_then(Value::as_str)
            .map_or(0, str::len)
    };
    match v.get("type").and_then(Value::as_str) {
        Some("content_block_delta") => {
            let d = v.get("delta");
            str_len(d, "text") + str_len(d, "thinking") + str_len(d, "partial_json")
        }
        Some("content_block_start") => {
            let cb = v.get("content_block");
            let input = cb
                .and_then(|c| c.get("input"))
                .filter(|i| !i.is_null() && i.as_object().is_none_or(|o| !o.is_empty()))
                .map_or(0, |i| i.to_string().len());
            str_len(cb, "text") + str_len(cb, "thinking") + input
        }
        _ => 0,
    }
}

/// Held content in one OpenAI Responses stream event. Only the delta
/// events count: the `.done` events and the terminal `response.*` snapshot
/// repeat content already counted from its deltas.
pub(crate) fn responses_event(v: &Value) -> usize {
    match v.get("type").and_then(Value::as_str) {
        Some(
            "response.output_text.delta"
            | "response.refusal.delta"
            | "response.reasoning_text.delta"
            | "response.reasoning_summary_text.delta"
            | "response.function_call_arguments.delta"
            | "response.mcp_call_arguments.delta"
            | "response.custom_tool_call_input.delta",
        ) => v.get("delta").and_then(Value::as_str).map_or(0, str::len),
        _ => 0,
    }
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
    fn unparseable_payload_counts_whole() {
        assert_eq!(sse_frames(b"data: not json\n\n", responses_event), 8);
    }
}
