//! JSON ↔ Smithy `Document` conversion for the Bedrock Converse tool path.
//!
//! Converse models tool input schemas
//! (`toolConfig.tools[].toolSpec.inputSchema.json`) and tool-call
//! arguments (`toolUse.input`) as [`aws_smithy_types::Document`], while
//! the gateway's OpenAI-shape surface carries the same data as
//! [`serde_json::Value`]. The Bedrock SDK ships no serde bridge between
//! the two, so these two functions are the single conversion point used
//! by the request builder ([`crate::bridge`]'s `build_tool_config`) and
//! the response / stream translators.
//!
//! Reference (Converse `ToolInputSchema.json` / `ToolUseBlock.input` are
//! `Document`): <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolInputSchema.html>

use aws_smithy_types::{Document, Number};
use serde_json::Value;

/// Convert a [`serde_json::Value`] (OpenAI tool schema / arguments) into
/// the [`Document`] the Bedrock Converse SDK builders accept.
pub(crate) fn json_to_document(value: &Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(b) => Document::Bool(*b),
        Value::String(s) => Document::String(s.clone()),
        Value::Number(n) => {
            // Preserve the integer-vs-float distinction so a schema's
            // `3` doesn't reach the model as `3.0`. serde_json stores
            // non-negative integers as u64, negatives as i64, the rest
            // as f64; mirror that into Smithy's `Number`.
            if let Some(u) = n.as_u64() {
                Document::Number(Number::PosInt(u))
            } else if let Some(i) = n.as_i64() {
                Document::Number(Number::NegInt(i))
            } else {
                Document::Number(Number::Float(n.as_f64().unwrap_or_default()))
            }
        }
        Value::Array(items) => Document::Array(items.iter().map(json_to_document).collect()),
        Value::Object(map) => Document::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), json_to_document(v)))
                .collect(),
        ),
    }
}

/// Convert a [`Document`] (Converse `toolUse.input`) back into a
/// [`serde_json::Value`] so the gateway can render OpenAI-shape
/// `tool_calls[].function.arguments`.
pub(crate) fn document_to_json(doc: &Document) -> Value {
    match doc {
        Document::Null => Value::Null,
        Document::Bool(b) => Value::Bool(*b),
        Document::String(s) => Value::String(s.clone()),
        Document::Number(Number::PosInt(u)) => Value::from(*u),
        Document::Number(Number::NegInt(i)) => Value::from(*i),
        Document::Number(Number::Float(f)) => {
            // serde_json cannot represent NaN / ±Inf; fall back to Null
            // (matches `serde_json::to_value(f64::NAN)`).
            serde_json::Number::from_f64(*f)
                .map(Value::Number)
                .unwrap_or(Value::Null)
        }
        Document::Array(items) => Value::Array(items.iter().map(document_to_json).collect()),
        Document::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), document_to_json(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_nested_tool_schema() {
        // A realistic OpenAI `parameters` JSON Schema with mixed value
        // kinds must survive Value → Document → Value unchanged.
        let original = serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"enum": ["c", "f"]},
                "ratio": 0.5,
                "flags": [true, false, null]
            },
            "required": ["location"]
        });
        let back = document_to_json(&json_to_document(&original));
        assert_eq!(back, original);
    }

    #[test]
    fn preserves_integer_vs_float_distinction() {
        // Agents that `JSON.parse` the arguments must not see an integer
        // arrive as a float (or vice versa).
        let v = serde_json::json!({"pos": 3, "neg": -7, "frac": 1.5});
        let back = document_to_json(&json_to_document(&v));
        assert!(
            back["pos"].is_u64(),
            "positive int must stay integral: {back}"
        );
        assert!(
            back["neg"].is_i64(),
            "negative int must stay integral: {back}"
        );
        assert!(back["frac"].is_f64(), "float must stay fractional: {back}");
    }

    #[test]
    fn non_finite_float_degrades_to_null() {
        // Defensive: a Float(NaN) document (not producible from JSON, but
        // possible from a misbehaving upstream) must not panic.
        let doc = Document::Number(Number::Float(f64::NAN));
        assert_eq!(document_to_json(&doc), Value::Null);
    }
}
