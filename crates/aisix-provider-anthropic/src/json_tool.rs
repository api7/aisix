//! Structured output for a Claude family without `output_config.format`,
//! on a platform's own Messages route.
//!
//! A `/v1/messages` caller asking Claude below 4.5 on Bedrock or Vertex AI
//! for structured output stays on the native Messages body: the schema
//! becomes the synthetic [`JSON_TOOL_NAME`] tool ([`inject`]), and the
//! answer's call to it is written back as the JSON text the caller asked
//! for ([`unwrap`]). Everything else the caller sent reaches the model as
//! written. The tool route answers once the call is complete, so a
//! streaming caller gets that answer as an SSE stream ([`into_sse_frames`]).

use aisix_gateway::structured_output::{
    apply_schema_limits, seal_object_schemas, ANTHROPIC_SCHEMA_LIMITS, JSON_TOOL_DESCRIPTION,
    JSON_TOOL_NAME,
};
use serde_json::{json, Value};

/// Move the body's structured-output request — the legacy top-level
/// `output_format`, which wins, or `output_config.format` — onto the
/// synthetic tool, and force it unless the caller set a `tool_choice` of
/// their own or turned thinking on (Anthropic rejects a forced choice
/// beside thinking). Returns whether the tool was injected; a format that
/// carries no JSON schema is left where it is.
pub fn inject(body: &mut Value) -> bool {
    let Some(obj) = body.as_object_mut() else {
        return false;
    };
    let legacy = obj.get("output_format").filter(|v| !v.is_null()).cloned();
    let format = legacy.clone().or_else(|| {
        obj.get("output_config")
            .and_then(|c| c.get("format"))
            .filter(|v| !v.is_null())
            .cloned()
    });
    let Some(mut schema) = format
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("json_schema"))
        .and_then(|f| f.get("schema").cloned())
        .filter(Value::is_object)
    else {
        return false;
    };
    seal_object_schemas(&mut schema);
    apply_schema_limits(&mut schema, &ANTHROPIC_SCHEMA_LIMITS);

    obj.remove("output_format");
    if let Some(Value::Object(config)) = obj.get_mut("output_config") {
        config.remove("format");
        if config.is_empty() {
            obj.remove("output_config");
        }
    }
    let tool = json!({
        "name": JSON_TOOL_NAME,
        "description": JSON_TOOL_DESCRIPTION,
        "input_schema": schema,
    });
    match obj.get_mut("tools") {
        Some(Value::Array(tools)) => tools.push(tool),
        _ => {
            obj.insert("tools".into(), Value::Array(vec![tool]));
        }
    }
    let thinking_on = obj
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|t| t != "disabled");
    let client_choice = obj.get("tool_choice").is_some_and(|c| !c.is_null());
    if !client_choice && !thinking_on {
        obj.insert(
            "tool_choice".into(),
            json!({"type": "tool", "name": JSON_TOOL_NAME}),
        );
    }
    true
}

/// Write a Messages answer's call to the synthetic tool back as text. With
/// no other tool call in the answer, the JSON is the answer: it replaces
/// the text blocks and the turn ends as `end_turn`. Beside a real tool
/// call the call is replaced in place and `tool_use` stands.
pub fn unwrap(message: &mut Value) {
    let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    let is_json_tool = |b: &Value| {
        b.get("type").and_then(Value::as_str) == Some("tool_use")
            && b.get("name").and_then(Value::as_str) == Some(JSON_TOOL_NAME)
    };
    if !content.iter().any(is_json_tool) {
        return;
    }
    let real_calls_remain = content
        .iter()
        .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_use") && !is_json_tool(b));
    let as_text = |b: &Value| {
        let input = b.get("input").cloned().unwrap_or_else(|| json!({}));
        json!({"type": "text", "text": input.to_string()})
    };
    if real_calls_remain {
        for block in content.iter_mut() {
            if is_json_tool(block) {
                *block = as_text(block);
            }
        }
        return;
    }
    let json_blocks: Vec<Value> = content
        .iter()
        .filter(|b| is_json_tool(b))
        .map(as_text)
        .collect();
    content.retain(|b| !is_json_tool(b) && b.get("type").and_then(Value::as_str) != Some("text"));
    content.extend(json_blocks);
    if message.get("stop_reason").and_then(Value::as_str) == Some("tool_use") {
        message["stop_reason"] = json!("end_turn");
    }
}

/// A complete Messages answer as the Anthropic SSE frames a stream of it
/// carries: `message_start` with the usage, each content block as
/// `content_block_start` / delta / `content_block_stop`, then
/// `message_delta` with the stop reason and usage, and `message_stop`. A
/// block kind with no delta form arrives whole in its `content_block_start`.
pub fn into_sse_frames(message: &Value) -> Vec<bytes::Bytes> {
    let frame =
        |event: &str, data: Value| bytes::Bytes::from(format!("event: {event}\ndata: {data}\n\n"));
    let usage = message.get("usage").cloned().unwrap_or_else(|| json!({}));
    let mut start = message.clone();
    if let Some(obj) = start.as_object_mut() {
        obj.insert("content".into(), json!([]));
        obj.insert("stop_reason".into(), Value::Null);
        obj.insert("stop_sequence".into(), Value::Null);
    }
    let mut frames = vec![frame(
        "message_start",
        json!({"type": "message_start", "message": start}),
    )];
    let blocks = message
        .get("content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for (index, block) in blocks.iter().enumerate() {
        let kind = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let (skeleton, deltas): (Value, Vec<Value>) = match kind {
            "text" => {
                let mut skeleton = block.clone();
                skeleton["text"] = json!("");
                let text = block.get("text").cloned().unwrap_or_else(|| json!(""));
                (skeleton, vec![json!({"type": "text_delta", "text": text})])
            }
            "thinking" => {
                let mut deltas = vec![json!({
                    "type": "thinking_delta",
                    "thinking": block.get("thinking").cloned().unwrap_or_else(|| json!("")),
                })];
                if let Some(signature) = block.get("signature").filter(|s| !s.is_null()) {
                    deltas.push(json!({"type": "signature_delta", "signature": signature}));
                }
                (
                    json!({"type": "thinking", "thinking": "", "signature": ""}),
                    deltas,
                )
            }
            "tool_use" => {
                let mut skeleton = block.clone();
                skeleton["input"] = json!({});
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                (
                    skeleton,
                    vec![json!({"type": "input_json_delta", "partial_json": input.to_string()})],
                )
            }
            _ => (block.clone(), Vec::new()),
        };
        frames.push(frame(
            "content_block_start",
            json!({"type": "content_block_start", "index": index, "content_block": skeleton}),
        ));
        for delta in deltas {
            frames.push(frame(
                "content_block_delta",
                json!({"type": "content_block_delta", "index": index, "delta": delta}),
            ));
        }
        frames.push(frame(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        ));
    }
    frames.push(frame(
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": message.get("stop_reason").cloned().unwrap_or(Value::Null),
                "stop_sequence": message.get("stop_sequence").cloned().unwrap_or(Value::Null),
            },
            "usage": usage,
        }),
    ));
    frames.push(frame("message_stop", json!({"type": "message_stop"})));
    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body() -> Value {
        json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "top_k": 5,
            "messages": [{"role": "user", "content": "weather?"}],
            "output_format": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        })
    }

    #[test]
    fn the_schema_becomes_a_forced_tool_and_the_rest_of_the_body_stays() {
        let mut b = body();
        assert!(inject(&mut b));
        assert!(b.get("output_format").is_none());
        assert_eq!(b["top_k"], 5);
        assert_eq!(b["tools"][0]["name"], JSON_TOOL_NAME);
        assert_eq!(
            b["tools"][0]["input_schema"]["additionalProperties"],
            json!(false)
        );
        assert_eq!(
            b["tool_choice"],
            json!({"type": "tool", "name": JSON_TOOL_NAME})
        );
    }

    #[test]
    fn a_callers_tool_choice_or_thinking_leaves_the_tool_unforced() {
        let mut b = body();
        b["tool_choice"] = json!({"type": "auto"});
        assert!(inject(&mut b));
        assert_eq!(b["tool_choice"], json!({"type": "auto"}));

        let mut b = body();
        b["thinking"] = json!({"type": "enabled", "budget_tokens": 1024});
        assert!(inject(&mut b));
        assert!(b.get("tool_choice").is_none());
    }

    #[test]
    fn output_config_format_is_moved_and_its_other_members_stay() {
        let mut b = body();
        let format = b.as_object_mut().unwrap().remove("output_format").unwrap();
        b["output_config"] = json!({"format": format, "effort": "high"});
        assert!(inject(&mut b));
        assert_eq!(b["output_config"], json!({"effort": "high"}));
    }

    #[test]
    fn the_tool_call_becomes_the_text_answer() {
        let mut m = json!({
            "content": [
                {"type": "thinking", "thinking": "hm", "signature": "s"},
                {"type": "text", "text": "Here you go"},
                {"type": "tool_use", "id": "t", "name": JSON_TOOL_NAME, "input": {"city": "Paris"}},
            ],
            "stop_reason": "tool_use",
        });
        unwrap(&mut m);
        assert_eq!(
            m["content"],
            json!([
                {"type": "thinking", "thinking": "hm", "signature": "s"},
                {"type": "text", "text": "{\"city\":\"Paris\"}"},
            ])
        );
        assert_eq!(m["stop_reason"], "end_turn");
    }

    #[test]
    fn beside_a_real_call_the_json_is_replaced_in_place() {
        let mut m = json!({
            "content": [
                {"type": "tool_use", "id": "a", "name": "lookup", "input": {}},
                {"type": "tool_use", "id": "t", "name": JSON_TOOL_NAME, "input": {"x": 1}},
            ],
            "stop_reason": "tool_use",
        });
        unwrap(&mut m);
        assert_eq!(m["content"][0]["name"], "lookup");
        assert_eq!(
            m["content"][1],
            json!({"type": "text", "text": "{\"x\":1}"})
        );
        assert_eq!(m["stop_reason"], "tool_use");
    }

    #[test]
    fn a_message_streams_as_its_events() {
        let m = json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "c",
            "content": [{"type": "text", "text": "{}"}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        });
        let text: String = into_sse_frames(&m)
            .iter()
            .map(|b| String::from_utf8(b.to_vec()).unwrap())
            .collect();
        let events: Vec<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("event: "))
            .collect();
        assert_eq!(
            events,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert!(
            text.contains(r#""text_delta","text":"{}""#)
                || text.contains(r#""text":"{}","type":"text_delta""#)
        );
        assert!(text.contains(r#""output_tokens":2"#));
    }
}
