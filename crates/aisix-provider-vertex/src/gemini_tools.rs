//! Function calling on Vertex's native Gemini wire (`:generateContent` /
//! `:streamGenerateContent`).
//!
//! The gateway carries tools in the OpenAI chat shape: `tools` /
//! `tool_choice` on the request, `tool_calls` on assistant turns and
//! `role: "tool"` results. This module translates each of those to the
//! Gemini shape and back:
//!
//! - `tools[].function` → one `{"functionDeclarations": [...]}` tool, the
//!   parameters narrowed to the OpenAPI `Schema` subset Gemini accepts
//!   ([`gemini_function_parameters`]);
//! - `tool_choice` → `toolConfig.functionCallingConfig`
//!   ([`gemini_tool_config`]);
//! - assistant `tool_calls` → `functionCall` parts, tool results →
//!   `functionResponse` parts;
//! - response `functionCall` parts → OpenAI `tool_calls`
//!   ([`tool_call_from_function_call`]).
//!
//! Thought signatures. Gemini 3 rejects history whose function calls lack
//! the `thoughtSignature` it returned with them, and an OpenAI-shaped
//! client has no field to carry one back in. The signature therefore
//! rides in the tool call id the client is handed (`<id>__thought__<sig>`)
//! and is split back out when the history returns. A Gemini 3 history
//! with no signature on a turn's first call (one written by another
//! model) gets Google's documented placeholder instead.
//! <https://ai.google.dev/gemini-api/docs/thought-signatures>

use aisix_gateway::{
    structured_output::{apply_schema_limits, SchemaLimits},
    BridgeError, ChatMessage,
};
use serde_json::{json, Map, Value};

/// Separator between a tool call id and the thought signature embedded
/// in it.
const THOUGHT_SIGNATURE_SEPARATOR: &str = "__thought__";

/// Base64 of `skip_thought_signature_validator`: the value Google
/// documents for function calls replayed into Gemini 3 without a
/// signature of their own (history from another model).
const PLACEHOLDER_THOUGHT_SIGNATURE: &str = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=";

/// Inline internal `$ref`s and nothing else. The rest of the narrowing
/// is the dedicated walk in [`gemini_function_parameters`].
const INLINE_REFS_ONLY: SchemaLimits = SchemaLimits {
    noted_constraints: &[],
    dropped_keywords: &[],
    allowed_min_items: None,
    relax_one_of: false,
    inline_internal_refs: true,
};

/// The members of Gemini's `Schema` type a function's parameters may
/// carry. Anything else is rejected by name upstream.
const SCHEMA_FIELDS: &[&str] = &[
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "default",
    "items",
    "minItems",
    "maxItems",
    "enum",
    "properties",
    "propertyOrdering",
    "required",
    "minProperties",
    "maxProperties",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "pattern",
    "example",
    "anyOf",
];

/// Whether `model` is Gemini 3 or newer — the generations that accept
/// (and return) `id` on `functionCall` / `functionResponse` and that
/// require thought signatures on replayed function calls. Older
/// generations reject the `id` field with a 400.
///
/// Only names recognisably older are excluded, so an alias such as
/// `gemini-flash-latest` that tracks the newest generation counts as
/// new.
pub(crate) fn gemini_3_or_newer(model: &str) -> bool {
    let name = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    if name.is_empty() || name.chars().all(|c| c.is_ascii_digit()) || name.starts_with("gemma-") {
        return false;
    }
    let Some(rest) = name.strip_prefix("gemini-") else {
        return true;
    };
    // A name segment ends at `-` or at the end of the name.
    let ends_segment = |s: &str| s.is_empty() || s.starts_with('-');
    let older_generation = |rest: &str| -> bool {
        let Some(after_major) = rest.strip_prefix(['1', '2']) else {
            return false;
        };
        if ends_segment(after_major) {
            return true;
        }
        after_major.strip_prefix('.').is_some_and(|minor| {
            let digits = minor.chars().take_while(char::is_ascii_digit).count();
            digits > 0 && ends_segment(&minor[digits..])
        })
    };
    let undated = |rest: &str| -> bool {
        if rest.strip_prefix("exp").is_some_and(ends_segment) {
            return true;
        }
        ["pro", "flash"].iter().any(|family| {
            rest.strip_prefix(family).is_some_and(|tail| {
                ends_segment(tail) && tail != "-latest" && tail != "-lite-latest"
            })
        })
    };
    !(older_generation(rest) || undated(rest))
}

/// `tools` → Gemini's `tools` array: every function tool becomes a
/// declaration in ONE `functionDeclarations` tool. Entries that are not
/// function tools (hosted tools of other providers) have no declaration
/// to become and are left out.
pub(crate) fn gemini_tools(tools: &Value) -> Result<Option<Vec<Value>>, BridgeError> {
    let Some(entries) = tools.as_array() else {
        return Ok(None);
    };
    let mut declarations = Vec::new();
    for entry in entries {
        // `{"type":"function","function":{...}}`, or the bare function
        // object some clients send.
        let function = match entry.get("function") {
            Some(f) => f,
            None if entry.get("name").is_some() => entry,
            None => continue,
        };
        let Some(name) = function.get("name").and_then(Value::as_str) else {
            continue;
        };
        let mut declaration = Map::new();
        declaration.insert("name".into(), name.into());
        if let Some(description) = function.get("description").filter(|d| !d.is_null()) {
            declaration.insert("description".into(), description.clone());
        }
        match function.get("parameters") {
            None | Some(Value::Null) => {}
            Some(Value::String(s)) if s.is_empty() => {
                declaration.insert("parameters".into(), json!({"type": "object"}));
            }
            Some(schema @ Value::Object(_)) => {
                declaration.insert("parameters".into(), gemini_function_parameters(schema)?);
            }
            Some(other) => {
                declaration.insert("parameters".into(), other.clone());
            }
        }
        declarations.push(Value::Object(declaration));
    }
    if declarations.is_empty() {
        return Ok(None);
    }
    Ok(Some(vec![json!({ "functionDeclarations": declarations })]))
}

/// `tool_choice` → `toolConfig`. `none` / `auto` / `required` become the
/// `NONE` / `AUTO` / `ANY` calling modes; a named function becomes `ANY`
/// restricted to that one name.
pub(crate) fn gemini_tool_config(choice: &Value) -> Result<Option<Value>, BridgeError> {
    let config = match choice {
        Value::String(mode) => {
            let mode = match mode.as_str() {
                "none" => "NONE",
                "auto" => "AUTO",
                "required" => "ANY",
                other => {
                    return Err(BridgeError::InvalidUpstreamConfig(format!(
                        "Gemini does not support tool_choice={other:?}; \
                         supported values are \"auto\", \"none\", \"required\" or a named function"
                    )))
                }
            };
            json!({ "mode": mode })
        }
        Value::Object(_) => {
            let name = choice
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            json!({ "mode": "ANY", "allowedFunctionNames": [name] })
        }
        _ => return Ok(None),
    };
    Ok(Some(json!({ "functionCallingConfig": config })))
}

/// Split the thought signature out of a tool call id this gateway minted.
fn split_tool_call_id(id: &str) -> (&str, Option<&str>) {
    match id.split_once(THOUGHT_SIGNATURE_SEPARATOR) {
        Some((id, signature)) => (id, Some(signature)),
        None => (id, None),
    }
}

/// The thought signature a replayed tool call carries: a
/// `provider_specific_fields.thought_signature` on the call or on its
/// function, else the one embedded in its id.
fn tool_call_thought_signature(call: &Value) -> Option<&str> {
    fn explicit(v: Option<&Value>) -> Option<&str> {
        v.and_then(|v| v.pointer("/provider_specific_fields/thought_signature"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    }
    explicit(Some(call))
        .or_else(|| explicit(call.get("function")))
        .or_else(|| {
            call.get("id")
                .and_then(Value::as_str)
                .and_then(|id| split_tool_call_id(id).1)
        })
}

/// An assistant turn's `tool_calls` → `functionCall` parts, each with
/// the thought signature it was returned with.
pub(crate) fn function_call_parts(
    tool_calls: &[Value],
    upstream_model: &str,
) -> Result<Vec<Value>, BridgeError> {
    let newer = gemini_3_or_newer(upstream_model);
    let mut parts = Vec::with_capacity(tool_calls.len());
    for call in tool_calls {
        let Some(function) = call.get("function") else {
            continue;
        };
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let args = match function.get("arguments") {
            None | Some(Value::Null) => json!({"type": "object"}),
            Some(Value::String(s)) if s.is_empty() => json!({"type": "object"}),
            Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| {
                BridgeError::InvalidUpstreamConfig(format!(
                    "assistant tool call {name:?} has arguments that are not valid JSON: {e}"
                ))
            })?,
            Some(other) => other.clone(),
        };
        let mut function_call = Map::new();
        if newer {
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                let (id, _) = split_tool_call_id(id);
                if !id.is_empty() {
                    function_call.insert("id".into(), id.into());
                }
            }
        }
        function_call.insert("name".into(), name.into());
        function_call.insert("args".into(), args);
        let mut part = Map::new();
        part.insert("functionCall".into(), Value::Object(function_call));
        // Gemini signs only the first call of a parallel batch, so only
        // that one gets the placeholder when the history has none.
        let signature = tool_call_thought_signature(call)
            .or_else(|| (newer && parts.is_empty()).then_some(PLACEHOLDER_THOUGHT_SIGNATURE));
        if let Some(signature) = signature {
            part.insert("thoughtSignature".into(), signature.into());
        }
        parts.push(Value::Object(part));
    }
    Ok(parts)
}

/// A `role: "tool"` result → a `functionResponse` part.
///
/// Gemini keys the response by function name, which an OpenAI tool
/// message does not carry: it is recovered from the most recent
/// assistant turn's call with the same id, else from the message's own
/// `name`. A JSON-object result is passed as the response itself;
/// anything else is wrapped as `{"content": <text>}`.
pub(crate) fn function_response_part(
    message: &ChatMessage,
    last_tool_calls: Option<&[Value]>,
    upstream_model: &str,
) -> Result<Value, BridgeError> {
    let call_id = message.tool_call_id.as_deref();
    let name = last_tool_calls
        .into_iter()
        .flatten()
        .find(|call| call_id.is_some() && call.get("id").and_then(Value::as_str) == call_id)
        .and_then(|call| call.pointer("/function/name"))
        .and_then(Value::as_str)
        .or(message.name.as_deref())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            BridgeError::InvalidUpstreamConfig(
                "tool message has no corresponding tool call in the preceding assistant turn"
                    .into(),
            )
        })?;
    let content = message.content_str();
    let trimmed = content.trim_start();
    let response = if trimmed.starts_with('{') || trimmed.starts_with('[') {
        match serde_json::from_str::<Value>(&content) {
            Ok(object @ Value::Object(_)) => object,
            _ => json!({ "content": content }),
        }
    } else {
        json!({ "content": content })
    };
    let mut function_response = Map::new();
    if gemini_3_or_newer(upstream_model) {
        if let Some(id) = call_id.map(|id| split_tool_call_id(id).0) {
            if !id.is_empty() {
                function_response.insert("id".into(), id.into());
            }
        }
    }
    function_response.insert("name".into(), name.into());
    function_response.insert("response".into(), response);
    Ok(json!({ "functionResponse": function_response }))
}

/// A response `functionCall` part → an OpenAI `tool_calls[]` entry
/// (without `index`; the streaming path adds it). The id is Gemini's own
/// when it sent one, else a minted `call_…`; a thought signature on the
/// part is appended to it so the client hands it back with the history.
pub(crate) fn tool_call_from_function_call(
    function_call: &GeminiFunctionCall,
    thought_signature: Option<&str>,
) -> Value {
    let mut id = function_call
        .id
        .clone()
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..28]));
    if let Some(signature) = thought_signature.filter(|s| !s.is_empty()) {
        id.push_str(THOUGHT_SIGNATURE_SEPARATOR);
        id.push_str(signature);
    }
    let arguments = function_call
        .args
        .as_ref()
        .map(Value::to_string)
        .unwrap_or_else(|| "{}".to_string());
    json!({
        "id": id,
        "type": "function",
        "function": { "name": function_call.name, "arguments": arguments },
    })
}

/// A `functionCall` part as Gemini returns it.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct GeminiFunctionCall {
    #[serde(default)]
    pub(crate) id: Option<String>,
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) args: Option<Value>,
}

/// Narrow a JSON Schema to the OpenAPI `Schema` subset Gemini accepts in
/// a function declaration's `parameters`:
///
/// - internal `$ref`s are inlined (the dialect has no references);
/// - a `null` branch of an `anyOf` becomes `nullable: true` on the rest;
/// - a type list (`["string","integer"]`) becomes an `anyOf`;
/// - `enum` survives only on string-typed nodes, and an empty-string
///   member becomes `null`;
/// - an array without `items` gets `{"type":"object"}` items, a node
///   with no type gets `type: object`, and empty `properties` are
///   removed;
/// - every member outside the `Schema` type is dropped, and an `anyOf`
///   node keeps nothing but the branches (its `title` / `description`
///   are copied into each).
pub(crate) fn gemini_function_parameters(schema: &Value) -> Result<Value, BridgeError> {
    let mut schema = schema.clone();
    apply_schema_limits(&mut schema, &INLINE_REFS_ONLY);
    anyof_null_to_nullable(&mut schema)?;
    type_lists_to_anyof(&mut schema);
    empty_enum_strings_to_null(&mut schema);
    enum_on_strings_only(&mut schema);
    default_array_items(&mut schema);
    default_object_type(&mut schema);
    Ok(keep_schema_fields(schema))
}

fn for_properties_and_items(obj: &mut Map<String, Value>, f: &mut dyn FnMut(&mut Value)) {
    if let Some(properties) = obj.get_mut("properties").and_then(Value::as_object_mut) {
        properties.values_mut().for_each(&mut *f);
    }
    if let Some(items) = obj.get_mut("items") {
        f(items);
    }
}

fn anyof_null_to_nullable(schema: &mut Value) -> Result<(), BridgeError> {
    let Some(obj) = schema.as_object_mut() else {
        return Ok(());
    };
    if let Some(branches) = obj.get_mut("anyOf").and_then(Value::as_array_mut) {
        let before = branches.len();
        branches.retain(|b| b.get("type").and_then(Value::as_str) != Some("null"));
        let had_null = branches.len() != before;
        for branch in branches.iter_mut() {
            if let Some(b) = branch.as_object_mut() {
                if b.is_empty() {
                    b.insert("type".into(), "object".into());
                }
            }
        }
        if branches.is_empty() {
            return Err(BridgeError::InvalidUpstreamConfig(
                "tool parameters: an anyOf with only a null type is not supported by Gemini".into(),
            ));
        }
        if had_null {
            for branch in branches.iter_mut() {
                if let Some(b) = branch.as_object_mut() {
                    b.insert("nullable".into(), true.into());
                }
            }
        }
    }
    let mut result = Ok(());
    for_properties_and_items(obj, &mut |sub| {
        if result.is_ok() {
            result = anyof_null_to_nullable(sub);
        }
    });
    result
}

fn type_lists_to_anyof(schema: &mut Value) {
    const TYPE_SPECIFIC: &[&str] = &[
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    ];
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    if let Some(Value::Array(types)) = obj.get("type").cloned() {
        if types.len() == 1 {
            obj.insert("type".into(), types[0].clone());
        } else if types.len() > 1 {
            let structured = types
                .iter()
                .any(|t| matches!(t.as_str(), Some("object" | "array")));
            let mut branches = Vec::new();
            for t in types.iter().filter_map(Value::as_str) {
                let mut branch = Map::new();
                branch.insert("type".into(), t.into());
                if t == "object" || t == "array" {
                    for field in TYPE_SPECIFIC {
                        if let Some(v) = obj.get(*field) {
                            branch.insert((*field).into(), v.clone());
                        }
                    }
                }
                branches.push(Value::Object(branch));
            }
            if structured {
                for field in TYPE_SPECIFIC {
                    obj.remove(*field);
                }
            }
            obj.remove("type");
            obj.insert("anyOf".into(), Value::Array(branches));
        }
    }
    for_properties_and_items(obj, &mut type_lists_to_anyof);
    if let Some(branches) = obj.get_mut("anyOf").and_then(Value::as_array_mut) {
        branches.iter_mut().for_each(type_lists_to_anyof);
    }
}

fn empty_enum_strings_to_null(schema: &mut Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    if let Some(members) = obj.get_mut("enum").and_then(Value::as_array_mut) {
        for member in members.iter_mut() {
            if member.as_str() == Some("") {
                *member = Value::Null;
            }
        }
    }
    for_properties_and_items(obj, &mut empty_enum_strings_to_null);
}

fn is_string_type(v: Option<&Value>) -> bool {
    v.and_then(Value::as_str)
        .is_some_and(|t| t.eq_ignore_ascii_case("string"))
}

fn enum_on_strings_only(schema: &mut Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    if obj.get("enum").is_some_and(Value::is_array) {
        let keep = is_string_type(obj.get("type"))
            || obj
                .get("anyOf")
                .and_then(Value::as_array)
                .is_some_and(|bs| bs.iter().any(|b| is_string_type(b.get("type"))));
        if !keep {
            obj.remove("enum");
        }
    }
    for_properties_and_items(obj, &mut enum_on_strings_only);
    if let Some(branches) = obj.get_mut("anyOf").and_then(Value::as_array_mut) {
        branches.iter_mut().for_each(enum_on_strings_only);
    }
}

fn default_array_items(schema: &mut Value) {
    match schema {
        Value::Object(obj) => {
            let is_array = obj
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| t.eq_ignore_ascii_case("array"));
            let items_missing = obj
                .get("items")
                .is_none_or(|i| i.as_object().is_some_and(Map::is_empty));
            if is_array && items_missing {
                obj.insert("items".into(), json!({"type": "object"}));
            }
            obj.values_mut().for_each(default_array_items);
        }
        Value::Array(values) => values.iter_mut().for_each(default_array_items),
        _ => {}
    }
}

fn default_object_type(schema: &mut Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    if !["type", "anyOf", "oneOf", "allOf"]
        .iter()
        .any(|k| obj.contains_key(*k))
    {
        obj.insert("type".into(), "object".into());
    }
    if obj.contains_key("properties") {
        if obj.get("required").is_some_and(Value::is_null) {
            obj.remove("required");
        }
        if obj["properties"].as_object().is_some_and(Map::is_empty) {
            obj.remove("properties");
            obj.remove("required");
        } else if let Some(properties) = obj.get_mut("properties").and_then(Value::as_object_mut) {
            properties.values_mut().for_each(default_object_type);
        }
        obj.insert("type".into(), "object".into());
    }
    if let Some(items) = obj.get_mut("items") {
        default_object_type(items);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(branches) = obj.get_mut(key).and_then(Value::as_array_mut) {
            branches.iter_mut().for_each(default_object_type);
        }
    }
}

fn keep_schema_fields(schema: Value) -> Value {
    let Value::Object(mut obj) = schema else {
        return schema;
    };
    if let Some(Value::Array(mut branches)) = obj
        .remove("anyOf")
        .filter(|b| b.as_array().is_some_and(|b| !b.is_empty()))
    {
        let title = obj.get("title").cloned();
        let description = obj.get("description").cloned();
        if (title.is_some() || description.is_some()) && branches.iter().all(Value::is_object) {
            for branch in branches.iter_mut().filter_map(Value::as_object_mut) {
                if let Some(t) = &title {
                    branch.insert("title".into(), t.clone());
                }
                if let Some(d) = &description {
                    branch.insert("description".into(), d.clone());
                }
            }
        }
        obj = Map::new();
        obj.insert("anyOf".into(), Value::Array(branches));
    }
    let mut out = Map::new();
    for (key, value) in obj {
        if !SCHEMA_FIELDS.contains(&key.as_str()) {
            continue;
        }
        let value = match (key.as_str(), value) {
            ("properties", Value::Object(properties)) => Value::Object(
                properties
                    .into_iter()
                    .map(|(name, sub)| (name, keep_schema_fields(sub)))
                    .collect(),
            ),
            ("format", format) => {
                if matches!(format.as_str(), Some("enum" | "date-time")) {
                    format
                } else {
                    continue;
                }
            }
            ("items", items @ Value::Object(_)) => keep_schema_fields(items),
            ("anyOf", Value::Array(branches)) => {
                Value::Array(branches.into_iter().map(keep_schema_fields).collect())
            }
            (_, value) => value,
        };
        out.insert(key, value);
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_gate_matches_gemini_3_and_later() {
        for older in [
            "gemini-2.5-flash",
            "gemini-2.0-flash-001",
            "gemini-1.5-pro",
            "gemini-2",
            "gemini-exp-1206",
            "gemini-pro",
            "gemini-flash-lite",
            "publishers/google/models/gemini-2.5-pro",
            "gemma-3-27b",
            "1234567890",
        ] {
            assert!(!gemini_3_or_newer(older), "{older}");
        }
        for newer in [
            "gemini-3-pro-preview",
            "gemini-3.1-flash",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-10-ultra",
        ] {
            assert!(gemini_3_or_newer(newer), "{newer}");
        }
    }

    #[test]
    fn tool_choice_maps_to_function_calling_modes() {
        let mode = |c: Value| gemini_tool_config(&c).unwrap().unwrap();
        assert_eq!(
            mode(json!("none")),
            json!({"functionCallingConfig": {"mode": "NONE"}})
        );
        assert_eq!(
            mode(json!("auto")),
            json!({"functionCallingConfig": {"mode": "AUTO"}})
        );
        assert_eq!(
            mode(json!("required")),
            json!({"functionCallingConfig": {"mode": "ANY"}})
        );
        assert_eq!(
            mode(json!({"type": "function", "function": {"name": "get_weather"}})),
            json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}})
        );
        assert!(matches!(
            gemini_tool_config(&json!("any")),
            Err(BridgeError::InvalidUpstreamConfig(_))
        ));
    }

    #[test]
    fn function_tools_become_one_declarations_tool() {
        let tools = gemini_tools(&json!([
            {"type": "function", "function": {
                "name": "get_weather", "description": "Weather", "strict": true,
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                               "required": ["city"], "additionalProperties": false}}},
            {"type": "function", "function": {"name": "now"}},
            {"type": "web_search_preview"},
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(
            tools,
            vec![json!({"functionDeclarations": [
                {"name": "get_weather", "description": "Weather",
                 "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                "required": ["city"]}},
                {"name": "now"},
            ]})]
        );
        assert_eq!(gemini_tools(&json!([])).unwrap(), None);
    }

    #[test]
    fn parameters_narrow_to_the_gemini_schema_subset() {
        let narrowed = gemini_function_parameters(&json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "unit": {"anyOf": [{"type": "string", "enum": ["c", "f", ""]}, {"type": "null"}],
                          "description": "Unit"},
                "days": {"type": ["integer", "null"], "minimum": 1, "exclusiveMinimum": 0},
                "tags": {"type": "array"},
                "level": {"type": "integer", "enum": [1, 2]},
                "mode": {"type": "string", "enum": ["fast", ""]},
                "when": {"type": "string", "format": "date-time"},
                "email": {"type": "string", "format": "email"},
                "place": {"$ref": "#/$defs/Place"},
                "meta": {"type": "object", "properties": {}},
            },
            "required": ["unit"],
            "additionalProperties": false,
            "$defs": {"Place": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }))
        .unwrap();
        assert_eq!(
            narrowed,
            json!({
                "type": "object",
                "properties": {
                    "unit": {"anyOf": [{"type": "string", "enum": ["c", "f", ""],
                                        "nullable": true, "description": "Unit"}]},
                    "days": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "tags": {"type": "array", "items": {"type": "object"}},
                    "level": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["fast", null]},
                    "when": {"type": "string", "format": "date-time"},
                    "email": {"type": "string"},
                    "place": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "meta": {"type": "object"},
                },
                "required": ["unit"],
            })
        );
    }

    #[test]
    fn an_anyof_of_only_null_is_refused() {
        assert!(matches!(
            gemini_function_parameters(&json!({"type": "object", "properties": {
                "x": {"anyOf": [{"type": "null"}]}}})),
            Err(BridgeError::InvalidUpstreamConfig(_))
        ));
    }

    fn call(id: &str, name: &str, arguments: &str) -> Value {
        json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}})
    }

    #[test]
    fn history_tool_calls_carry_their_signature_and_gemini_3_ids() {
        let calls = [
            call(
                "call_a__thought__SIGA",
                "get_weather",
                r#"{"city":"Paris"}"#,
            ),
            call("call_b", "get_weather", r#"{"city":"Rome"}"#),
        ];
        assert_eq!(
            function_call_parts(&calls, "gemini-2.5-flash").unwrap(),
            vec![
                json!({"functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
                       "thoughtSignature": "SIGA"}),
                json!({"functionCall": {"name": "get_weather", "args": {"city": "Rome"}}}),
            ]
        );
        assert_eq!(
            function_call_parts(&calls, "gemini-3-pro-preview").unwrap(),
            vec![
                json!({"functionCall": {"id": "call_a", "name": "get_weather", "args": {"city": "Paris"}},
                       "thoughtSignature": "SIGA"}),
                json!({"functionCall": {"id": "call_b", "name": "get_weather", "args": {"city": "Rome"}}}),
            ]
        );
    }

    #[test]
    fn unsigned_gemini_3_history_gets_the_placeholder_on_its_first_call_only() {
        let calls = [call("call_a", "f", "{}"), call("call_b", "f", "")];
        let parts = function_call_parts(&calls, "gemini-3-pro-preview").unwrap();
        assert_eq!(parts[0]["thoughtSignature"], PLACEHOLDER_THOUGHT_SIGNATURE);
        assert!(parts[1].get("thoughtSignature").is_none());
        assert_eq!(parts[1]["functionCall"]["args"], json!({"type": "object"}));
        let older = function_call_parts(&calls, "gemini-2.5-flash").unwrap();
        assert!(older[0].get("thoughtSignature").is_none());
    }

    #[test]
    fn unparseable_history_arguments_are_refused() {
        assert!(matches!(
            function_call_parts(&[call("c", "f", "{not json")], "gemini-2.5-flash"),
            Err(BridgeError::InvalidUpstreamConfig(_))
        ));
    }

    fn tool_message(id: &str, content: &str) -> ChatMessage {
        let mut m = ChatMessage::tool(content);
        m.tool_call_id = Some(id.into());
        m
    }

    #[test]
    fn tool_results_recover_the_function_name_and_wrap_text() {
        let calls = [call("call_a__thought__SIG", "get_weather", "{}")];
        let part = function_response_part(
            &tool_message("call_a__thought__SIG", "sunny"),
            Some(&calls),
            "gemini-2.5-flash",
        )
        .unwrap();
        assert_eq!(
            part,
            json!({"functionResponse": {"name": "get_weather", "response": {"content": "sunny"}}})
        );
        let part = function_response_part(
            &tool_message("call_a__thought__SIG", r#" {"temp": 21}"#),
            Some(&calls),
            "gemini-3-pro-preview",
        )
        .unwrap();
        assert_eq!(
            part,
            json!({"functionResponse": {"id": "call_a", "name": "get_weather", "response": {"temp": 21}}})
        );
        let part = function_response_part(&tool_message("x", "[1,2]"), None, "gemini-2.5-flash");
        assert!(matches!(part, Err(BridgeError::InvalidUpstreamConfig(_))));
    }

    #[test]
    fn response_calls_mint_ids_and_carry_the_signature() {
        let fc: GeminiFunctionCall =
            serde_json::from_value(json!({"name": "get_weather", "args": {"city": "Paris"}}))
                .unwrap();
        let tc = tool_call_from_function_call(&fc, Some("SIG"));
        let id = tc["id"].as_str().unwrap();
        let (bare, sig) = split_tool_call_id(id);
        assert!(bare.starts_with("call_") && bare.len() == 33, "{bare}");
        assert_eq!(sig, Some("SIG"));
        assert_eq!(tc["function"]["arguments"], r#"{"city":"Paris"}"#);
        let fc: GeminiFunctionCall =
            serde_json::from_value(json!({"id": "native-1", "name": "now"})).unwrap();
        let tc = tool_call_from_function_call(&fc, None);
        assert_eq!(tc["id"], "native-1");
        assert_eq!(tc["function"]["arguments"], "{}");
    }
}
