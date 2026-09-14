//! Structured outputs: the parts every provider bridge translating an
//! OpenAI `response_format` needs to agree on.
//!
//! Providers reach JSON by two different routes and the gateway uses
//! both. Where the upstream has a native schema control (Anthropic's
//! `output_config.format`, Bedrock Converse's `outputConfig.textFormat`,
//! Gemini's `responseJsonSchema`) the schema goes straight onto the
//! wire. Where it does not, the schema rides a **synthetic tool** the
//! model is asked to call — [`JSON_TOOL_NAME`] — whose input *is* the
//! answer. The tool route needs the same two translations on the way
//! back everywhere it is used, so they live here rather than in one
//! provider crate: [`unwrap_json_tool_call`] turns the call back into
//! content, and [`response_into_fake_stream_chunks`] renders the
//! completed response as a stream, because a tool call cannot be
//! streamed before it is complete.
//!
//! [`seal_object_schemas`] and [`close_object_schemas`] are the two
//! schema normalisations these paths need; they differ only in whether
//! every declared property is forced into `required`.

use crate::chat::{ChatChunk, ChatDelta, ChatResponse, FinishReason, Role};

/// Name of the synthetic tool the tool route asks the model to call.
/// The response decoder recognises it by this name to translate the
/// call back into plain JSON content, so the two sides must agree.
pub const JSON_TOOL_NAME: &str = "json_tool_call";

/// Description carried on the synthetic tool.
pub const JSON_TOOL_DESCRIPTION: &str =
    "Respond by calling this tool with your answer as JSON matching its input schema.";

/// Pull the JSON schema out of an OpenAI `response_format`, verbatim.
///
/// Returns `None` for anything that is not a `json_schema` carrying a
/// non-null schema — `{"type":"json_object"}` and `{"type":"text"}`
/// included, neither of which names a schema to translate.
pub fn json_schema_from_response_format(
    response_format: &serde_json::Value,
) -> Option<serde_json::Value> {
    if response_format.get("type").and_then(|t| t.as_str())? != "json_schema" {
        return None;
    }
    response_format
        .get("json_schema")?
        .get("schema")
        .filter(|s| !s.is_null())
        .cloned()
}

/// Recursively set `additionalProperties: false` on every object schema,
/// leaving `required` exactly as the caller wrote it.
///
/// This is what Anthropic and Bedrock require: both reject an object
/// that does not close, and both list `required` as an ordinary,
/// optional JSON Schema keyword — a property left out of it stays
/// optional and simply sorts after the required ones in the output.
/// Forcing every property into `required` would silently promote a
/// caller's optional field to mandatory, which changes what the model
/// is allowed to answer.
pub fn seal_object_schemas(schema: &mut serde_json::Value) {
    walk_object_schemas(schema, false);
}

/// Recursively make every object schema satisfy OpenAI **strict** mode:
/// `additionalProperties: false`, and every declared property listed in
/// `required`. Strict mode defines optionality through a nullable type
/// rather than through `required`, so the promotion is part of the
/// contract there — unlike [`seal_object_schemas`].
pub fn close_object_schemas(schema: &mut serde_json::Value) {
    walk_object_schemas(schema, true);
}

fn walk_object_schemas(schema: &mut serde_json::Value, require_every_property: bool) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    if obj.get("type").and_then(|t| t.as_str()) == Some("object") {
        if let Some(properties) = obj.get("properties").and_then(|p| p.as_object()) {
            let required: Vec<serde_json::Value> =
                properties.keys().map(|k| k.as_str().into()).collect();
            obj.insert("additionalProperties".to_string(), false.into());
            if require_every_property {
                obj.insert("required".to_string(), required.into());
            }
        }
        if let Some(properties) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
            for property in properties.values_mut() {
                walk_object_schemas(property, require_every_property);
            }
        }
    }
    if let Some(items) = obj.get_mut("items") {
        walk_object_schemas(items, require_every_property);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(branches) = obj.get_mut(key).and_then(|b| b.as_array_mut()) {
            for branch in branches {
                walk_object_schemas(branch, require_every_property);
            }
        }
    }
    for key in ["$defs", "definitions"] {
        if let Some(defs) = obj.get_mut(key).and_then(|d| d.as_object_mut()) {
            for def in defs.values_mut() {
                walk_object_schemas(def, require_every_property);
            }
        }
    }
}

/// Undo the tool route: turn the model's call to the synthetic
/// [`JSON_TOOL_NAME`] tool back into the plain JSON content the caller
/// asked for. Only ever applied to a response whose request carried the
/// synthetic tool, so a caller's own tool of that name is never touched.
///
/// The call's arguments are already the JSON-encoded tool input, which
/// is exactly the document the schema describes. When it is the only
/// call the response becomes an ordinary text completion — no
/// `tool_calls`, and a tool-use finish reason demoted to `stop`, which
/// is what a client that never offered a tool must see. When the model
/// called real tools alongside it, those and their finish reason are
/// left untouched and the JSON is appended to the content.
pub fn unwrap_json_tool_call(resp: &mut ChatResponse) {
    let mut json_parts: Vec<String> = Vec::new();
    let mut real_calls_remain = false;
    if let Some(serde_json::Value::Array(calls)) = resp.message.extra.get_mut("tool_calls") {
        calls.retain(|call| {
            let name = call
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .unwrap_or_default();
            if name != JSON_TOOL_NAME {
                return true;
            }
            if let Some(args) = call.pointer("/function/arguments").and_then(|a| a.as_str()) {
                json_parts.push(args.to_string());
            }
            false
        });
        real_calls_remain = !calls.is_empty();
    }
    if json_parts.is_empty() {
        return;
    }
    if !real_calls_remain {
        resp.message.extra.remove("tool_calls");
        resp.finish_reason = FinishReason::Stop;
    }
    let json = json_parts.join("\n");
    resp.message.content = Some(match resp.message.content.take() {
        Some(text) if !text.is_empty() => format!("{text}\n{json}"),
        _ => json,
    });
}

/// Render a complete response as the chunk sequence a streaming client
/// expects: role, content, finish, usage.
///
/// The tool route cannot stream — the JSON only exists once the tool
/// call is complete — so a bridge runs that request non-streaming and
/// fake-streams the result through here. Keeping the usage on its own
/// terminal chunk matches what a real upstream emits, so the proxy's
/// accounting and every downstream encoder see an ordinary stream.
pub fn response_into_fake_stream_chunks(resp: ChatResponse) -> Vec<ChatChunk> {
    let ChatResponse {
        id,
        model,
        message,
        finish_reason,
        usage,
    } = resp;
    let chunk = |delta, finish_reason, usage| ChatChunk {
        id: id.clone(),
        model: model.clone(),
        delta,
        finish_reason,
        usage,
    };
    let tool_calls = message
        .extra
        .get("tool_calls")
        .and_then(|c| c.as_array())
        .cloned();
    vec![
        chunk(
            ChatDelta {
                role: Some(Role::Assistant),
                ..ChatDelta::default()
            },
            None,
            None,
        ),
        chunk(
            ChatDelta {
                content: Some(message.content.unwrap_or_default()),
                tool_calls,
                ..ChatDelta::default()
            },
            None,
            None,
        ),
        chunk(ChatDelta::default(), Some(finish_reason), None),
        chunk(ChatDelta::default(), None, Some(usage)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "pet": {
                    "type": "object",
                    "properties": {"kind": {"type": "string"}},
                    "required": ["kind"],
                },
            },
            "required": ["name"],
        })
    }

    #[test]
    fn sealing_closes_every_object_and_leaves_required_alone() {
        let mut schema = person_schema();
        seal_object_schemas(&mut schema);
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["pet"]["additionalProperties"], false);
        // `pet` was optional and stays optional — the whole point of the
        // seal-only variant.
        assert_eq!(schema["required"], serde_json::json!(["name"]));
        assert_eq!(
            schema["properties"]["pet"]["required"],
            serde_json::json!(["kind"])
        );
    }

    #[test]
    fn strict_closing_promotes_every_property_to_required() {
        let mut schema = person_schema();
        close_object_schemas(&mut schema);
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["required"], serde_json::json!(["name", "pet"]));
    }

    #[test]
    fn sealing_reaches_arrays_branches_and_definitions() {
        let mut schema = serde_json::json!({
            "type": "array",
            "items": {"type": "object", "properties": {"a": {"type": "string"}}},
            "anyOf": [{"type": "object", "properties": {"b": {"type": "string"}}}],
            "$defs": {"d": {"type": "object", "properties": {"c": {"type": "string"}}}},
        });
        seal_object_schemas(&mut schema);
        assert_eq!(schema["items"]["additionalProperties"], false);
        assert_eq!(schema["anyOf"][0]["additionalProperties"], false);
        assert_eq!(schema["$defs"]["d"]["additionalProperties"], false);
    }

    #[test]
    fn only_a_json_schema_response_format_yields_a_schema() {
        let schema = serde_json::json!({"type": "object"});
        assert_eq!(
            json_schema_from_response_format(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": true},
            })),
            Some(schema)
        );
        for other in [
            serde_json::json!({"type": "json_object"}),
            serde_json::json!({"type": "text"}),
            serde_json::json!({"type": "json_schema", "json_schema": {"name": "answer"}}),
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": null},
            }),
        ] {
            assert_eq!(json_schema_from_response_format(&other), None, "{other}");
        }
    }
}
