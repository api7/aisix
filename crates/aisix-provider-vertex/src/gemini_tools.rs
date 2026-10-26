//! Function calling on Vertex's native Gemini wire (`:generateContent` /
//! `:streamGenerateContent`).
//!
//! The gateway carries tools in the OpenAI chat shape: `tools` /
//! `tool_choice` on the request, `tool_calls` on assistant turns and
//! `role: "tool"` results. This module translates each of those to the
//! Gemini shape and back:
//!
//! - `tools[].function` → one `{"functionDeclarations": [...]}` tool
//!   ([`gemini_tools`]); the parameters go as the typed `parameters`
//!   schema when they fit it, else verbatim as `parametersJsonSchema`
//!   ([`gemini_function_parameters`]);
//! - `tool_choice` → `toolConfig.functionCallingConfig`
//!   ([`gemini_tool_config`]);
//! - assistant `tool_calls` → `functionCall` parts, tool results →
//!   `functionResponse` parts, both carrying the call id;
//! - response `functionCall` parts → OpenAI `tool_calls`
//!   ([`tool_call_from_function_call`]).
//!
//! Thought signatures. Gemini 3 rejects history whose function calls lack
//! the `thoughtSignature` it returned with them, and an OpenAI-shaped
//! client has no field to carry one back in. The signature therefore
//! rides in the tool call id the client is handed
//! (`<id>__thought__<signature>`, URL-safe base64 without padding so the
//! id stays `[A-Za-z0-9_-]`), and is split back out here when the history
//! returns. Every other bridge strips it
//! ([`aisix_gateway::tool_calls`]). A Gemini 3 history with no signature
//! on a turn's first call (one written by another model) gets Google's
//! documented placeholder instead.
//! <https://cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures>

use aisix_gateway::{
    structured_output::{apply_schema_limits, SchemaLimits},
    tool_calls::{split_thought_signature, strip_thought_signature, THOUGHT_SIGNATURE_SEPARATOR},
    BridgeError, ChatMessage,
};
use serde_json::{json, Map, Value};

/// Base64 of `skip_thought_signature_validator`: the value Google
/// documents for function calls replayed into Gemini 3 without a
/// signature of their own (history from another model).
const PLACEHOLDER_THOUGHT_SIGNATURE: &str = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=";

/// Longest tool call id this gateway hands out, and the character set it
/// keeps to: what OpenAI and Bedrock accept, so the base id survives a
/// conversation moving to another provider.
const MAX_TOOL_CALL_ID_LEN: usize = 64;

/// Inline internal `$ref`s and nothing else.
const INLINE_REFS_ONLY: SchemaLimits = SchemaLimits {
    noted_constraints: &[],
    dropped_keywords: &[],
    allowed_min_items: None,
    relax_one_of: false,
    inline_internal_refs: true,
};

/// The members of Gemini's `Schema` type a function's `parameters` may
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

/// JSON Schema keywords the typed `Schema` has no equivalent for. A schema
/// using any of them goes as `parametersJsonSchema` rather than losing
/// the constraint.
const UNTYPED_KEYWORDS: &[&str] = &[
    "oneOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "const",
    "contains",
    "patternProperties",
    "prefixItems",
    "propertyNames",
    "dependentSchemas",
    "dependentRequired",
    "unevaluatedProperties",
    "unevaluatedItems",
    "$ref",
    "$dynamicRef",
];

/// Whether `model` is Gemini 3 or newer — the generations that require a
/// thought signature on every replayed function call, and so get the
/// placeholder when the history has none.
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
/// declaration in ONE `functionDeclarations` tool. An OpenAI `custom` tool
/// takes free-form text, which Gemini has no declaration for, and is
/// dropped; a tool of any other type (another provider's hosted tool) is
/// refused.
///
/// The parameters go as `parametersJsonSchema`, verbatim, on the families
/// that take it ([`takes_parameters_json_schema`]); older ones get the typed
/// `parameters` schema where it can express them
/// ([`gemini_function_parameters`]).
pub(crate) fn gemini_tools(
    tools: &Value,
    upstream_model: &str,
) -> Result<Option<Vec<Value>>, BridgeError> {
    let entries = match tools {
        Value::Null => return Ok(None),
        Value::Array(entries) => entries,
        _ => return Err(BridgeError::InvalidRequest("tools must be an array".into())),
    };
    let json_schema = takes_parameters_json_schema(upstream_model);
    let mut declarations = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        let kind = entry.get("type").and_then(Value::as_str);
        // `{"type":"function","function":{...}}`, or the bare function
        // object some clients send.
        let function = match (kind, entry.get("function")) {
            (Some("custom"), _) => continue,
            (None | Some("function"), Some(f)) => f,
            (None, None) if entry.get("name").is_some() => entry,
            _ => {
                return Err(BridgeError::InvalidRequest(format!(
                    "tool type {:?} is not supported for this model; only function tools are",
                    kind.unwrap_or("<missing>")
                )))
            }
        };
        let Some(name) = function.get("name").and_then(Value::as_str) else {
            return Err(BridgeError::InvalidRequest(format!(
                "tools[{i}].function.name is required"
            )));
        };
        let mut declaration = Map::new();
        declaration.insert("name".into(), name.into());
        if let Some(description) = function.get("description").filter(|d| !d.is_null()) {
            declaration.insert("description".into(), description.clone());
        }
        match function.get("parameters") {
            None | Some(Value::Null) => {}
            Some(Value::String(s)) if s.is_empty() => {
                let key = if json_schema {
                    "parametersJsonSchema"
                } else {
                    "parameters"
                };
                declaration.insert(key.into(), json!({"type": "object"}));
            }
            Some(schema) if json_schema => {
                declaration.insert("parametersJsonSchema".into(), schema.clone());
            }
            Some(schema) => match gemini_function_parameters(schema) {
                GeminiParameters::Typed(typed) => {
                    declaration.insert("parameters".into(), typed);
                }
                GeminiParameters::JsonSchema(raw) => {
                    declaration.insert("parametersJsonSchema".into(), raw);
                }
            },
        }
        declarations.push(Value::Object(declaration));
    }
    if declarations.is_empty() {
        return Ok(None);
    }
    Ok(Some(vec![json!({ "functionDeclarations": declarations })]))
}

/// Whether `model` takes a function's JSON Schema verbatim as
/// `parametersJsonSchema`: Gemini 2.5 and later, which is where it was
/// verified live (2.5 Flash / Flash-Lite / Pro, 3 Flash, 3.1 Pro). Older
/// families get the typed `parameters` form.
pub(crate) fn takes_parameters_json_schema(model: &str) -> bool {
    if gemini_3_or_newer(model) {
        return true;
    }
    let name = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    name.strip_prefix("gemini-2.5")
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('-'))
}

/// `tool_choice` → `toolConfig`. `none` / `auto` / `required` become the
/// `NONE` / `AUTO` / `ANY` calling modes, a named function becomes `ANY`
/// restricted to that one name, and an `allowed_tools` subset in
/// `required` mode becomes `ANY` restricted to the subset's functions.
/// `null`, an `allowed_tools` subset in `auto` mode (Gemini restricts
/// names only when a call is forced) and a `custom` tool set no choice.
/// Any other shape is not a valid OpenAI `tool_choice` and is refused.
pub(crate) fn gemini_tool_config(choice: &Value) -> Result<Option<Value>, BridgeError> {
    let invalid = || {
        BridgeError::InvalidRequest(
            "tool_choice must be \"none\", \"auto\", \"required\" or a \
             {\"type\": \"function\", \"function\": {\"name\": …}} object"
                .into(),
        )
    };
    let config = match choice {
        Value::Null => return Ok(None),
        Value::String(mode) => {
            let mode = match mode.as_str() {
                "none" => "NONE",
                "auto" => "AUTO",
                "required" => "ANY",
                _ => return Err(invalid()),
            };
            json!({ "mode": mode })
        }
        Value::Object(o) => match o.get("type").and_then(Value::as_str) {
            None | Some("function") => {
                let Some(name) = choice
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .filter(|n| !n.is_empty())
                else {
                    return Err(BridgeError::InvalidRequest(
                        "tool_choice.function.name is required".into(),
                    ));
                };
                json!({ "mode": "ANY", "allowedFunctionNames": [name] })
            }
            Some("allowed_tools") => {
                let allowed = choice.get("allowed_tools");
                if allowed.and_then(|a| a.get("mode")).and_then(Value::as_str) != Some("required") {
                    return Ok(None);
                }
                let names: Vec<&str> = allowed
                    .and_then(|a| a.get("tools"))
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    // A `custom` tool is not declared to Gemini.
                    .filter(|t| t.get("type").and_then(Value::as_str) != Some("custom"))
                    .filter_map(|t| {
                        t.pointer("/function/name")
                            .or_else(|| t.get("name"))
                            .and_then(Value::as_str)
                    })
                    .collect();
                if names.is_empty() {
                    return Ok(None);
                }
                json!({ "mode": "ANY", "allowedFunctionNames": names })
            }
            Some("custom") => return Ok(None),
            Some(_) => return Err(invalid()),
        },
        _ => return Err(invalid()),
    };
    Ok(Some(json!({ "functionCallingConfig": config })))
}

/// Gemini's signature (standard base64) → the URL-safe, unpadded form
/// carried in an id.
fn encode_signature(signature: &str) -> String {
    signature
        .trim_end_matches('=')
        .chars()
        .map(|c| match c {
            '+' => '-',
            '/' => '_',
            c => c,
        })
        .collect()
}

/// The inverse of [`encode_signature`].
fn decode_signature(encoded: &str) -> String {
    let mut signature: String = encoded
        .chars()
        .map(|c| match c {
            '-' => '+',
            '_' => '/',
            c => c,
        })
        .collect();
    while !signature.len().is_multiple_of(4) {
        signature.push('=');
    }
    signature
}

/// The thought signature a replayed tool call carries: a
/// `provider_specific_fields.thought_signature` on the call or on its
/// function, else the one embedded in its id.
fn tool_call_thought_signature(call: &Value) -> Option<String> {
    fn explicit(v: Option<&Value>) -> Option<&str> {
        v.and_then(|v| v.pointer("/provider_specific_fields/thought_signature"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    }
    explicit(Some(call))
        .or_else(|| explicit(call.get("function")))
        .map(str::to_string)
        .or_else(|| {
            call.get("id")
                .and_then(Value::as_str)
                .and_then(|id| split_thought_signature(id).1)
                .filter(|s| !s.is_empty())
                .map(decode_signature)
        })
}

/// An assistant turn's `tool_calls` → `functionCall` parts, each with its
/// call id and the thought signature it was returned with.
pub(crate) fn function_call_parts(
    tool_calls: &[Value],
    upstream_model: &str,
) -> Result<Vec<Value>, BridgeError> {
    let needs_signature = gemini_3_or_newer(upstream_model);
    let mut parts = Vec::with_capacity(tool_calls.len());
    for call in tool_calls {
        let Some(function) = call.get("function") else {
            return Err(BridgeError::InvalidRequest(
                "assistant tool call is not a function call; only function tool calls \
                 can be replayed to this model"
                    .into(),
            ));
        };
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let args = match function.get("arguments") {
            None | Some(Value::Null) => json!({}),
            Some(Value::String(s)) if s.is_empty() => json!({}),
            Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| {
                BridgeError::InvalidRequest(format!(
                    "assistant tool call {name:?} has arguments that are not valid JSON: {e}"
                ))
            })?,
            Some(other) => other.clone(),
        };
        let mut function_call = Map::new();
        if let Some(id) = call
            .get("id")
            .and_then(Value::as_str)
            .map(|id| split_thought_signature(id).0)
            .filter(|id| !id.is_empty())
        {
            function_call.insert("id".into(), id.into());
        }
        function_call.insert("name".into(), name.into());
        function_call.insert("args".into(), args);
        let mut part = Map::new();
        part.insert("functionCall".into(), Value::Object(function_call));
        // Gemini signs only the first call of a parallel batch, so only
        // that one gets the placeholder when the history has none.
        let signature = tool_call_thought_signature(call).or_else(|| {
            (needs_signature && parts.is_empty()).then(|| PLACEHOLDER_THOUGHT_SIGNATURE.to_string())
        });
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
/// message does not carry: it is recovered from the latest earlier
/// assistant call with the same base id, else from the message's own
/// `name`. A JSON-object result is passed as the response itself;
/// anything else is wrapped as `{"content": <text>}`.
pub(crate) fn function_response_part(
    message: &ChatMessage,
    prior_tool_calls: &[&Value],
) -> Result<Value, BridgeError> {
    let call_id = message.tool_call_id.as_deref();
    let base_id = call_id.map(strip_thought_signature);
    let name = prior_tool_calls
        .iter()
        .rev()
        .find(|call| {
            base_id.is_some()
                && call
                    .get("id")
                    .and_then(Value::as_str)
                    .map(strip_thought_signature)
                    == base_id
        })
        .and_then(|call| call.pointer("/function/name"))
        .and_then(Value::as_str)
        .or(message.name.as_deref())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            BridgeError::InvalidRequest(
                "tool message has no corresponding tool call in an earlier assistant turn".into(),
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
    if let Some(id) = call_id
        .map(|id| split_thought_signature(id).0)
        .filter(|id| !id.is_empty())
    {
        function_response.insert("id".into(), id.into());
    }
    function_response.insert("name".into(), name.into());
    function_response.insert("response".into(), response);
    Ok(json!({ "functionResponse": function_response }))
}

/// Whether an id Gemini minted can be handed out as it is: short enough
/// and in the character set every upstream accepts.
fn portable_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_TOOL_CALL_ID_LEN
        && !id.contains(THOUGHT_SIGNATURE_SEPARATOR)
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// A response `functionCall` part → an OpenAI `tool_calls[]` entry
/// (without `index`; the streaming path adds it). The id is Gemini's own
/// when it sent a portable one, else a minted `call_…`; a thought
/// signature on the part is appended to it so the client hands it back
/// with the history.
pub(crate) fn tool_call_from_function_call(
    function_call: &GeminiFunctionCall,
    thought_signature: Option<&str>,
) -> Value {
    let mut id = function_call
        .id
        .clone()
        .filter(|id| portable_id(id))
        .unwrap_or_else(|| format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..28]));
    if let Some(signature) = thought_signature.filter(|s| !s.is_empty()) {
        id.push_str(THOUGHT_SIGNATURE_SEPARATOR);
        id.push_str(&encode_signature(signature));
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

/// Which declaration field a function's parameters go in.
#[derive(Debug, PartialEq)]
pub(crate) enum GeminiParameters {
    /// `parameters`: Gemini's typed OpenAPI `Schema`.
    Typed(Value),
    /// `parametersJsonSchema`: the caller's JSON Schema as written.
    JsonSchema(Value),
}

/// A function's JSON Schema → the declaration field it goes in.
///
/// A schema the typed `Schema` can express is converted to it:
///
/// - internal `$ref`s are inlined (the dialect has no references);
/// - `type: [X, "null"]` becomes `type: X` with `nullable: true`, and a
///   `null` branch of an `anyOf` becomes `nullable: true` on the rest;
/// - an array without `items` gets `{"type":"object"}` items, a node
///   with no type gets `type: object`, and empty `properties` are
///   removed;
/// - every member outside the `Schema` type is dropped
///   (`additionalProperties`, `$schema`, unsupported `format`s …), and an
///   `anyOf` node keeps nothing but the branches (its `title` /
///   `description` are copied into each).
///
/// One the typed form cannot express — `oneOf` / `allOf` and the other
/// [`UNTYPED_KEYWORDS`], a type list other than `[X]` / `[X, "null"]`, an
/// `enum` on anything but a string, a tuple `items`, a recursive `$ref` —
/// goes verbatim as `parametersJsonSchema` instead, so no constraint is
/// silently lost.
pub(crate) fn gemini_function_parameters(schema: &Value) -> GeminiParameters {
    let mut typed = schema.clone();
    apply_schema_limits(&mut typed, &INLINE_REFS_ONLY);
    if !fits_typed_schema(&typed) {
        return GeminiParameters::JsonSchema(schema.clone());
    }
    nullable_type_lists(&mut typed);
    anyof_null_to_nullable(&mut typed);
    default_array_items(&mut typed);
    default_object_type(&mut typed);
    GeminiParameters::Typed(keep_schema_fields(typed))
}

/// The non-null member of a `[X]` or `[X, "null"]` type list.
fn single_type(types: &[Value]) -> Option<&str> {
    let mut named = types
        .iter()
        .filter_map(Value::as_str)
        .filter(|t| *t != "null");
    let first = named.next()?;
    let all_strings = types.iter().all(Value::is_string);
    let fits = all_strings
        && named.next().is_none()
        && (types.len() == 1 || (types.len() == 2 && types.iter().any(|t| t == "null")));
    fits.then_some(first)
}

fn fits_typed_schema(schema: &Value) -> bool {
    let Some(obj) = schema.as_object() else {
        // `true` / `false` schemas have no typed spelling.
        return false;
    };
    if UNTYPED_KEYWORDS.iter().any(|k| obj.contains_key(*k)) {
        return false;
    }
    let ty = match obj.get("type") {
        None => None,
        Some(Value::String(t)) => Some(t.as_str()),
        Some(Value::Array(types)) => match single_type(types) {
            Some(t) => Some(t),
            None => return false,
        },
        Some(_) => return false,
    };
    if obj.contains_key("enum") && ty != Some("string") {
        return false;
    }
    if let Some(properties) = obj.get("properties") {
        match properties.as_object() {
            Some(properties) if properties.values().all(fits_typed_schema) => {}
            _ => return false,
        }
    }
    match obj.get("items") {
        None => {}
        Some(items @ Value::Object(_)) if fits_typed_schema(items) => {}
        Some(_) => return false,
    }
    if let Some(branches) = obj.get("anyOf") {
        let Some(branches) = branches.as_array() else {
            return false;
        };
        let non_null = branches
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) != Some("null"))
            .count();
        if non_null == 0 || !branches.iter().all(fits_typed_schema) {
            return false;
        }
    }
    true
}

fn for_each_typed_subschema(obj: &mut Map<String, Value>, f: &mut dyn FnMut(&mut Value)) {
    if let Some(properties) = obj.get_mut("properties").and_then(Value::as_object_mut) {
        properties.values_mut().for_each(&mut *f);
    }
    if let Some(items) = obj.get_mut("items") {
        f(items);
    }
    if let Some(branches) = obj.get_mut("anyOf").and_then(Value::as_array_mut) {
        branches.iter_mut().for_each(&mut *f);
    }
}

fn nullable_type_lists(schema: &mut Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    if let Some(Value::Array(types)) = obj.get("type") {
        let nullable = types.iter().any(|t| t == "null");
        if let Some(ty) = single_type(types).map(str::to_string) {
            obj.insert("type".into(), ty.into());
            if nullable {
                obj.insert("nullable".into(), true.into());
            }
        }
    }
    for_each_typed_subschema(obj, &mut nullable_type_lists);
}

fn anyof_null_to_nullable(schema: &mut Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    if let Some(branches) = obj.get_mut("anyOf").and_then(Value::as_array_mut) {
        let before = branches.len();
        branches.retain(|b| b.get("type").and_then(Value::as_str) != Some("null"));
        if branches.len() != before {
            for branch in branches.iter_mut().filter_map(Value::as_object_mut) {
                branch.insert("nullable".into(), true.into());
            }
        }
    }
    for_each_typed_subschema(obj, &mut anyof_null_to_nullable);
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
        assert_eq!(
            mode(
                json!({"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [
                    {"type": "function", "function": {"name": "get_weather"}},
                    {"type": "custom", "name": "grammar"},
                    {"type": "function", "function": {"name": "now"}},
                ]}})
            ),
            json!({"functionCallingConfig": {"mode": "ANY",
                                             "allowedFunctionNames": ["get_weather", "now"]}})
        );
        for unset in [
            Value::Null,
            json!({"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [
                {"type": "function", "function": {"name": "get_weather"}}]}}),
            json!({"type": "custom", "custom": {"name": "grammar"}}),
        ] {
            assert_eq!(gemini_tool_config(&unset).unwrap(), None, "{unset}");
        }
        let refusal = |c: Value| match gemini_tool_config(&c) {
            Err(BridgeError::InvalidRequest(m)) => m,
            other => panic!("{c}: {other:?}"),
        };
        for refused in [
            json!("any"),
            json!(1),
            json!(true),
            json!({"type": "hosted"}),
        ] {
            assert_eq!(
                refusal(refused),
                "tool_choice must be \"none\", \"auto\", \"required\" or a \
                 {\"type\": \"function\", \"function\": {\"name\": …}} object"
            );
        }
        assert_eq!(
            refusal(json!({"type": "function", "function": {}})),
            "tool_choice.function.name is required"
        );
    }

    #[test]
    fn function_tools_become_one_declarations_tool() {
        let tools = json!([
            {"type": "function", "function": {
                "name": "get_weather", "description": "Weather", "strict": true,
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                               "required": ["city"], "additionalProperties": false}}},
            {"type": "function", "function": {"name": "now"}},
            {"type": "function", "function": {"name": "pick", "parameters":
                {"type": "object", "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}}}},
        ]);
        assert_eq!(
            gemini_tools(&tools, "gemini-2.0-flash-001")
                .unwrap()
                .unwrap(),
            vec![json!({"functionDeclarations": [
                {"name": "get_weather", "description": "Weather",
                 "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                "required": ["city"]}},
                {"name": "now"},
                {"name": "pick", "parametersJsonSchema":
                    {"type": "object", "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}}},
            ]})]
        );
        assert_eq!(gemini_tools(&json!([]), "gemini-2.5-flash").unwrap(), None);
        assert_eq!(
            gemini_tools(&Value::Null, "gemini-2.5-flash").unwrap(),
            None
        );
    }

    #[test]
    fn gemini_2_5_and_later_take_the_parameters_verbatim_as_json_schema() {
        let parameters = json!({"type": "object", "properties": {"city": {"type": "string"}},
                                "required": ["city"], "additionalProperties": false});
        for model in [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
            "gemini-flash-latest",
        ] {
            let tools = json!([{"type": "function",
                                "function": {"name": "get_weather", "parameters": parameters}}]);
            assert_eq!(
                gemini_tools(&tools, model).unwrap().unwrap(),
                vec![json!({"functionDeclarations": [
                    {"name": "get_weather", "parametersJsonSchema": parameters}
                ]})],
                "{model}"
            );
        }
        for model in ["gemini-2.0-flash-001", "gemini-1.5-pro", "gemini-pro"] {
            assert!(!takes_parameters_json_schema(model), "{model}");
        }
    }

    #[test]
    fn a_custom_tool_is_dropped_and_another_tool_type_refused() {
        let tools = json!([
            {"type": "custom", "custom": {"name": "x"}},
            {"type": "function", "function": {"name": "now"}},
        ]);
        assert_eq!(
            gemini_tools(&tools, "gemini-2.5-flash").unwrap().unwrap(),
            vec![json!({"functionDeclarations": [{"name": "now"}]})]
        );
        let only_custom = json!([{"type": "custom", "custom": {"name": "x"}}]);
        assert_eq!(
            gemini_tools(&only_custom, "gemini-2.5-flash").unwrap(),
            None
        );
        assert!(matches!(
            gemini_tools(&json!([{"type": "web_search_preview"}]), "gemini-2.5-flash"),
            Err(BridgeError::InvalidRequest(_))
        ));
    }

    #[test]
    fn malformed_tools_are_refused_naming_the_field() {
        let message = |tools: Value| match gemini_tools(&tools, "gemini-2.5-flash") {
            Err(BridgeError::InvalidRequest(m)) => m,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            message(json!({"type": "function"})),
            "tools must be an array"
        );
        assert_eq!(
            message(json!([{"type": "function", "function": {"name": "a"}},
                           {"type": "function", "function": {}}])),
            "tools[1].function.name is required"
        );
    }

    #[test]
    fn a_schema_the_typed_form_can_express_is_converted() {
        let typed = gemini_function_parameters(&json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "unit": {"anyOf": [{"type": "string", "enum": ["c", "f", ""]}, {"type": "null"}],
                          "description": "Unit"},
                "days": {"type": ["integer", "null"], "minimum": 1},
                "mode": {"type": ["string", "null"], "enum": ["fast", ""]},
                "tags": {"type": "array"},
                "when": {"type": "string", "format": "date-time"},
                "email": {"type": "string", "format": "email"},
                "place": {"$ref": "#/$defs/Place"},
                "meta": {"type": "object", "properties": {}},
            },
            "required": ["unit"],
            "additionalProperties": false,
            "$defs": {"Place": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }));
        assert_eq!(
            typed,
            GeminiParameters::Typed(json!({
                "type": "object",
                "properties": {
                    "unit": {"anyOf": [{"type": "string", "enum": ["c", "f", ""],
                                        "nullable": true, "description": "Unit"}]},
                    "days": {"type": "integer", "nullable": true, "minimum": 1},
                    "mode": {"type": "string", "nullable": true, "enum": ["fast", ""]},
                    "tags": {"type": "array", "items": {"type": "object"}},
                    "when": {"type": "string", "format": "date-time"},
                    "email": {"type": "string"},
                    "place": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "meta": {"type": "object"},
                },
                "required": ["unit"],
            }))
        );
    }

    #[test]
    fn a_schema_the_typed_form_cannot_express_goes_as_json_schema() {
        let object = |prop: Value| json!({"type": "object", "properties": {"x": prop}});
        for prop in [
            json!({"oneOf": [{"type": "string"}, {"type": "integer"}]}),
            json!({"allOf": [{"type": "object"}]}),
            json!({"type": ["string", "integer"]}),
            json!({"type": "integer", "enum": [1, 2]}),
            json!({"type": "array", "items": [{"type": "string"}]}),
            json!({"anyOf": [{"type": "null"}]}),
            json!({"const": "a"}),
            json!({"$ref": "#/$defs/Missing"}),
        ] {
            let schema = object(prop.clone());
            assert_eq!(
                gemini_function_parameters(&schema),
                GeminiParameters::JsonSchema(schema),
                "{prop}"
            );
        }
    }

    fn call(id: &str, name: &str, arguments: &str) -> Value {
        json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}})
    }

    #[test]
    fn history_tool_calls_carry_their_id_and_decoded_signature() {
        // `+/=` in Gemini's signature travel as `-_` and no padding.
        let calls = [
            call(
                "call_a__thought__ab-c_d",
                "get_weather",
                r#"{"city":"Paris"}"#,
            ),
            call("call_b", "get_weather", r#"{"city":"Rome"}"#),
        ];
        let expected = vec![
            json!({"functionCall": {"id": "call_a", "name": "get_weather", "args": {"city": "Paris"}},
                   "thoughtSignature": "ab+c/d=="}),
            json!({"functionCall": {"id": "call_b", "name": "get_weather", "args": {"city": "Rome"}}}),
        ];
        assert_eq!(
            function_call_parts(&calls, "gemini-2.5-flash").unwrap(),
            expected
        );
        assert_eq!(
            function_call_parts(&calls, "gemini-3-pro-preview").unwrap(),
            expected
        );
    }

    #[test]
    fn unsigned_gemini_3_history_gets_the_placeholder_on_its_first_call_only() {
        let calls = [call("call_a", "f", "{}"), call("call_b", "f", "")];
        let parts = function_call_parts(&calls, "gemini-3-pro-preview").unwrap();
        assert_eq!(parts[0]["thoughtSignature"], PLACEHOLDER_THOUGHT_SIGNATURE);
        assert!(parts[1].get("thoughtSignature").is_none());
        let older = function_call_parts(&calls, "gemini-2.5-flash").unwrap();
        assert!(older[0].get("thoughtSignature").is_none());
    }

    #[test]
    fn a_call_without_arguments_replays_empty_args() {
        let calls = [
            json!({"id": "a", "type": "function", "function": {"name": "now", "arguments": ""}}),
            json!({"id": "b", "type": "function", "function": {"name": "now"}}),
        ];
        for part in function_call_parts(&calls, "gemini-2.5-flash").unwrap() {
            assert_eq!(part["functionCall"]["args"], json!({}));
        }
    }

    #[test]
    fn unparseable_history_arguments_are_refused() {
        assert!(matches!(
            function_call_parts(&[call("c", "f", "{not json")], "gemini-2.5-flash"),
            Err(BridgeError::InvalidRequest(_))
        ));
    }

    fn tool_message(id: &str, content: &str) -> ChatMessage {
        let mut m = ChatMessage::tool(content);
        m.tool_call_id = Some(id.into());
        m
    }

    #[test]
    fn tool_results_recover_the_function_name_and_carry_the_id() {
        let calls = [call("call_a__thought__SIG", "get_weather", "{}")];
        let prior: Vec<&Value> = calls.iter().collect();
        let part =
            function_response_part(&tool_message("call_a__thought__SIG", "sunny"), &prior).unwrap();
        assert_eq!(
            part,
            json!({"functionResponse": {"id": "call_a", "name": "get_weather",
                                        "response": {"content": "sunny"}}})
        );
        let part = function_response_part(
            &tool_message("call_a__thought__SIG", r#" {"temp": 21}"#),
            &prior,
        )
        .unwrap();
        assert_eq!(part["functionResponse"]["response"], json!({"temp": 21}));
        let part = function_response_part(&tool_message("x", "[1,2]"), &[]);
        assert!(matches!(part, Err(BridgeError::InvalidRequest(_))));
    }

    #[test]
    fn response_calls_get_portable_ids_and_carry_the_signature() {
        let fc: GeminiFunctionCall =
            serde_json::from_value(json!({"name": "get_weather", "args": {"city": "Paris"}}))
                .unwrap();
        let tc = tool_call_from_function_call(&fc, Some("ab+c/d=="));
        let id = tc["id"].as_str().unwrap();
        let (bare, sig) = split_thought_signature(id);
        assert!(bare.starts_with("call_") && bare.len() == 33, "{bare}");
        assert_eq!(sig, Some("ab-c_d"));
        assert!(id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'));
        assert_eq!(tc["function"]["arguments"], r#"{"city":"Paris"}"#);

        let fc: GeminiFunctionCall =
            serde_json::from_value(json!({"id": "native-1", "name": "now"})).unwrap();
        let tc = tool_call_from_function_call(&fc, None);
        assert_eq!(tc["id"], "native-1");
        assert_eq!(tc["function"]["arguments"], "{}");

        for unportable in ["has space", "a/b", &"x".repeat(65)] {
            let fc: GeminiFunctionCall =
                serde_json::from_value(json!({"id": unportable, "name": "now"})).unwrap();
            let tc = tool_call_from_function_call(&fc, None);
            assert!(
                tc["id"].as_str().unwrap().starts_with("call_"),
                "{unportable}"
            );
        }
    }

    #[test]
    fn a_result_finds_its_call_by_base_id_in_any_earlier_turn() {
        let first = call("call_a__thought__SIG", "get_weather", "{}");
        let second = call("call_b", "get_time", "{}");
        let prior = [&first, &second];
        // The suffix was stripped on the way through another hop.
        let part = function_response_part(&tool_message("call_a", "sunny"), &prior).unwrap();
        assert_eq!(part["functionResponse"]["name"], "get_weather");
    }

    #[test]
    fn a_non_function_tool_call_in_history_is_refused() {
        let custom =
            json!({"id": "c", "type": "custom", "custom": {"name": "apply_patch", "input": "x"}});
        assert!(matches!(
            function_call_parts(&[custom], "gemini-2.5-flash"),
            Err(BridgeError::InvalidRequest(_))
        ));
    }
}
