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

/// Whether a schema node describes an object and therefore has to be
/// sealed. `type` is not always the bare string `"object"`: the
/// canonical strict-mode spelling of an optional nested object is the
/// union `["object", "null"]`, and a node carrying `properties` with no
/// `type` at all is still an object schema. Missing either leaves that
/// node — and everything under it, since the walk would not recurse —
/// open, which the providers that require sealing reject outright.
fn is_object_schema(obj: &serde_json::Map<String, serde_json::Value>) -> bool {
    match obj.get("type") {
        Some(serde_json::Value::String(ty)) => ty == "object",
        Some(serde_json::Value::Array(types)) => types.iter().any(|t| t.as_str() == Some("object")),
        // No `type`, but `properties` can only describe an object.
        None => obj.contains_key("properties"),
        _ => false,
    }
}

fn walk_object_schemas(schema: &mut serde_json::Value, require_every_property: bool) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    if is_object_schema(obj) {
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

/// The subset of JSON Schema one provider's constrained decoder accepts.
///
/// Every provider that compiles a schema into a decoding grammar
/// supports only a subset of JSON Schema and returns a 400 for anything
/// outside it — so a schema that worked against an OpenAI upstream
/// fails outright once the gateway starts forwarding it. Rather than
/// hand that error to a caller who did nothing wrong, each edge narrows
/// the schema to what its provider takes, and says in the schema itself
/// what it had to drop.
pub struct SchemaLimits {
    /// Scalar constraint keywords the provider rejects. Each is removed
    /// and recorded in that node's `description`, so the constraint is
    /// still stated to the model even though it is no longer enforced by
    /// the decoder.
    pub noted_constraints: &'static [&'static str],
    /// Keywords the provider rejects that say nothing a sentence can
    /// carry — structural combinators and applicators. Removed quietly.
    pub dropped_keywords: &'static [&'static str],
    /// `minItems` values the provider accepts. `None` = all of them.
    pub allowed_min_items: Option<&'static [u64]>,
    /// Rewrite `oneOf` into `anyOf`. The providers here document
    /// `anyOf` and not `oneOf`; for constraining *output* the
    /// difference (exactly-one vs at-least-one) does not bind, since a
    /// document the model produces matches whichever branch it followed.
    /// Renaming keeps the alternatives, which dropping would not.
    pub relax_one_of: bool,
    /// Inline internal `$ref`s and remove the definition blocks they
    /// point at. For providers whose schema dialect has no `$ref` at
    /// all; the ones that document internal references keep theirs.
    pub inline_internal_refs: bool,
}

/// What Anthropic's structured outputs accept, per the "JSON Schema
/// limitations" section of their structured-outputs guide. Bedrock
/// documents the same subset for both its Converse `outputConfig` and
/// the Anthropic Messages `/invoke` body, so both edges use this.
///
/// Internal `$ref` / `$defs` / `definitions` are supported by both and
/// are left in place. Recursive schemas and external `$ref`s are not,
/// and nothing this can do would make them legal, so they are left for
/// the upstream to reject.
pub const ANTHROPIC_SCHEMA_LIMITS: SchemaLimits = SchemaLimits {
    noted_constraints: &[
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
        "uniqueItems",
    ],
    dropped_keywords: &[],
    allowed_min_items: Some(&[0, 1]),
    relax_one_of: true,
    inline_internal_refs: false,
};

/// What Gemini's older `responseSchema` dialect accepts. It is an
/// OpenAPI 3.0 `Schema` object, not JSON Schema: unknown members are
/// rejected by name, there is no `$ref`, and the applicator keywords
/// have no equivalent. Numeric and string bounds *are* part of that
/// dialect, so unlike Anthropic they survive.
pub const GEMINI_OPENAPI_SCHEMA_LIMITS: SchemaLimits = SchemaLimits {
    noted_constraints: &[
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "uniqueItems",
    ],
    dropped_keywords: &[
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "const",
        "contains",
        "patternProperties",
        "prefixItems",
        "unevaluatedProperties",
    ],
    allowed_min_items: None,
    relax_one_of: true,
    inline_internal_refs: true,
};

/// Narrow `schema` to what `limits` says the provider accepts.
pub fn apply_schema_limits(schema: &mut serde_json::Value, limits: &SchemaLimits) {
    if limits.inline_internal_refs {
        inline_internal_refs(schema);
    }
    narrow_schema_node(schema, limits);
}

fn narrow_schema_node(schema: &mut serde_json::Value, limits: &SchemaLimits) {
    if let Some(array) = schema.as_array_mut() {
        for item in array {
            narrow_schema_node(item, limits);
        }
        return;
    }
    let Some(obj) = schema.as_object_mut() else {
        return;
    };

    // Collect the constraints being removed in the order they are
    // declared on the limits, so the note reads the same every time.
    let mut notes: Vec<String> = Vec::new();
    for key in limits.noted_constraints {
        if let Some(value) = obj.remove(*key) {
            notes.push(format!("{key}: {}", render_constraint(&value)));
        }
    }
    if let Some(allowed) = limits.allowed_min_items {
        let out_of_range = obj
            .get("minItems")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|v| !allowed.contains(&v));
        if out_of_range {
            if let Some(value) = obj.remove("minItems") {
                notes.push(format!("minItems: {}", render_constraint(&value)));
            }
        }
    }
    if !notes.is_empty() {
        let note = notes.join(", ");
        let merged = match obj.get("description").and_then(|d| d.as_str()) {
            Some(existing) if !existing.is_empty() => format!("{existing} ({note})"),
            _ => note,
        };
        obj.insert("description".to_string(), merged.into());
    }

    for key in limits.dropped_keywords {
        obj.remove(*key);
    }
    if limits.relax_one_of {
        if let Some(branches) = obj.remove("oneOf") {
            obj.entry("anyOf").or_insert(branches);
        }
    }

    for value in obj.values_mut() {
        narrow_schema_node(value, limits);
    }
}

/// Render a constraint value for the description note. Strings keep
/// their quotes off; everything else is its compact JSON form.
fn render_constraint(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Replace every internal `$ref` with the definition it names and drop
/// the definition blocks, for dialects that have no `$ref`.
///
/// A `$ref` this cannot resolve — external, or recursive past
/// [`MAX_REF_DEPTH`] — is left exactly as it came in. Nothing this
/// function could do would make such a schema legal, so the upstream's
/// own rejection is the honest outcome.
fn inline_internal_refs(schema: &mut serde_json::Value) {
    // `$defs` and `definitions` are separate namespaces — a schema may
    // define the same name in both — so the map is keyed by the pointer
    // that reaches each one, not by the bare name.
    let mut defs: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for block in ["$defs", "definitions"] {
        if let Some(entries) = schema.get(block).and_then(|d| d.as_object()) {
            for (name, definition) in entries {
                defs.insert(format!("{block}/{name}"), definition.clone());
            }
        }
    }
    if defs.is_empty() {
        return;
    }
    substitute_refs(schema, &defs, 0);
    if let Some(obj) = schema.as_object_mut() {
        obj.remove("$defs");
        obj.remove("definitions");
    }
}

/// How many times one `$ref` chain is followed before giving up. A
/// recursive definition is the only way to exceed it.
const MAX_REF_DEPTH: usize = 8;

fn substitute_refs(
    node: &mut serde_json::Value,
    defs: &serde_json::Map<String, serde_json::Value>,
    depth: usize,
) {
    if let Some(array) = node.as_array_mut() {
        for item in array {
            substitute_refs(item, defs, depth);
        }
        return;
    }
    let Some(obj) = node.as_object_mut() else {
        return;
    };
    if let Some(reference) = obj.get("$ref").and_then(|r| r.as_str()) {
        let Some(name) = internal_ref_name(reference) else {
            return; // external reference: not ours to resolve
        };
        let Some(definition) = defs.get(name) else {
            return;
        };
        if depth >= MAX_REF_DEPTH {
            return; // recursive: leave the `$ref` and let the upstream say so
        }
        let mut expanded = definition.clone();
        substitute_refs(&mut expanded, defs, depth + 1);
        *node = expanded;
        return;
    }
    for value in obj.values_mut() {
        substitute_refs(value, defs, depth);
    }
}

/// The `<block>/<name>` key a `#/$defs/Name` or `#/definitions/Name`
/// pointer resolves to. `None` for anything else, which includes every
/// external reference.
fn internal_ref_name(reference: &str) -> Option<&str> {
    let path = reference.strip_prefix("#/")?;
    let (block, name) = path.split_once('/')?;
    if !matches!(block, "$defs" | "definitions") || name.is_empty() || name.contains('/') {
        return None;
    }
    Some(path)
}

/// Undo the tool route: turn the model's call to the synthetic
/// [`JSON_TOOL_NAME`] tool back into the plain JSON content the caller
/// asked for. Only ever applied to a response whose request carried the
/// synthetic tool, so a caller's own tool of that name is never touched.
///
/// The call's arguments are already the JSON-encoded tool input, which
/// is exactly the document the schema describes.
///
/// When it is the only call the JSON **replaces** the content: the
/// caller asked for a document they can parse, and a model that
/// narrated before calling the tool ("Sure, here you go:") would
/// otherwise leave them with a string that is not JSON. Any prose is
/// dropped, `tool_calls` with it, and a tool-use finish reason is
/// demoted to `stop` — what a client that never offered a tool must
/// see. Prose is likeliest exactly where the tool could not be forced
/// (a caller's own `tool_choice`, extended thinking, a Converse family
/// with no `toolChoice`), so this is not a rare shape.
///
/// When the model called real tools alongside it, the caller *did* ask
/// for tool calls and is parsing the response themselves, so those
/// calls and their finish reason are left untouched and the JSON is
/// appended to whatever text came with them.
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
    let json = json_parts.join("\n");
    if !real_calls_remain {
        resp.message.extra.remove("tool_calls");
        resp.finish_reason = FinishReason::Stop;
        resp.message.content = Some(json);
        return;
    }
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
    // The non-streaming `tool_calls` shape carries no `index`, but the
    // streaming one must: OpenAI SDKs accumulate by it, and this repo's
    // Anthropic SSE re-encoder reads it to key each `content_block`,
    // folding every index-less call onto block 0. Number them densely
    // in arrival order, leaving any index a decoder already assigned.
    let tool_calls = message
        .extra
        .get("tool_calls")
        .and_then(|c| c.as_array())
        .map(|calls| {
            calls
                .iter()
                .enumerate()
                .map(|(i, call)| {
                    let mut call = call.clone();
                    if let Some(obj) = call.as_object_mut() {
                        obj.entry("index").or_insert(i.into());
                    }
                    call
                })
                .collect()
        });
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
    fn sealing_recognises_union_typed_and_untyped_object_nodes() {
        // `["object","null"]` is how strict mode spells an optional
        // nested object, and a node with `properties` and no `type` is
        // still an object. Missing either leaves the whole subtree open
        // and the provider rejects the request.
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "nullable": {
                    "type": ["object", "null"],
                    "properties": {"a": {"type": "string"}},
                },
                "untyped": {"properties": {"b": {"type": "string"}}},
            },
        });
        seal_object_schemas(&mut schema);
        assert_eq!(
            schema["properties"]["nullable"]["additionalProperties"],
            false
        );
        assert_eq!(
            schema["properties"]["nullable"]["properties"]["a"]["type"],
            "string"
        );
        assert_eq!(
            schema["properties"]["untyped"]["additionalProperties"],
            false
        );
    }

    // ── provider schema subsets ───────────────────────────────────

    #[test]
    fn anthropic_limits_strip_every_unsupported_constraint_and_say_so() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "description": "the age",
                    "minimum": 1,
                    "maximum": 120,
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 121,
                    "multipleOf": 1,
                },
                "name": {"type": "string", "minLength": 2, "maxLength": 20},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 9,
                    "uniqueItems": true,
                },
            },
        });
        apply_schema_limits(&mut schema, &ANTHROPIC_SCHEMA_LIMITS);

        let age = &schema["properties"]["age"];
        for keyword in [
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        ] {
            assert!(age.get(keyword).is_none(), "{keyword} must be stripped");
        }
        // An existing description keeps its own text and gains the note.
        assert_eq!(
            age["description"],
            concat!(
                "the age (minimum: 1, maximum: 120, exclusiveMinimum: 0, ",
                "exclusiveMaximum: 121, multipleOf: 1)"
            )
        );

        let name = &schema["properties"]["name"];
        assert!(name.get("minLength").is_none());
        assert!(name.get("maxLength").is_none());
        // No description to begin with: the note becomes one.
        assert_eq!(name["description"], "minLength: 2, maxLength: 20");

        let tags = &schema["properties"]["tags"];
        assert!(tags.get("maxItems").is_none());
        assert!(tags.get("uniqueItems").is_none());
        // `minItems` is supported only at 0 and 1, so 3 goes too.
        assert!(tags.get("minItems").is_none());
        assert_eq!(
            tags["description"],
            "maxItems: 9, uniqueItems: true, minItems: 3"
        );
    }

    #[test]
    fn anthropic_limits_keep_the_min_items_values_the_provider_takes() {
        for kept in [0, 1] {
            let mut schema =
                serde_json::json!({"type": "array", "items": {"type": "string"}, "minItems": kept});
            apply_schema_limits(&mut schema, &ANTHROPIC_SCHEMA_LIMITS);
            assert_eq!(schema["minItems"], kept, "minItems {kept} is supported");
            assert!(schema.get("description").is_none());
        }
    }

    #[test]
    fn one_of_is_relaxed_to_any_of_rather_than_dropped() {
        // Neither provider documents `oneOf`; dropping it would take the
        // alternatives with it, so the branches move to `anyOf`.
        let mut schema = serde_json::json!({
            "oneOf": [{"type": "string"}, {"type": "integer"}],
        });
        apply_schema_limits(&mut schema, &ANTHROPIC_SCHEMA_LIMITS);
        assert!(schema.get("oneOf").is_none());
        assert_eq!(
            schema["anyOf"],
            serde_json::json!([{"type": "string"}, {"type": "integer"}])
        );
    }

    #[test]
    fn anthropic_limits_leave_internal_references_in_place() {
        // Anthropic and Bedrock both document internal `$ref`; only the
        // dialects without one need inlining.
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {"pet": {"$ref": "#/$defs/Pet"}},
            "$defs": {"Pet": {"type": "object", "properties": {"kind": {"type": "string"}}}},
        });
        apply_schema_limits(&mut schema, &ANTHROPIC_SCHEMA_LIMITS);
        assert_eq!(schema["properties"]["pet"]["$ref"], "#/$defs/Pet");
        assert!(schema["$defs"]["Pet"].is_object());
    }

    #[test]
    fn gemini_limits_inline_internal_references_and_drop_the_blocks() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "pet": {"$ref": "#/$defs/Pet"},
                "other": {"$ref": "#/definitions/Pet"},
            },
            "$defs": {"Pet": {"type": "object", "properties": {"kind": {"type": "string"}}}},
            "definitions": {"Pet": {"type": "string"}},
        });
        apply_schema_limits(&mut schema, &GEMINI_OPENAPI_SCHEMA_LIMITS);
        assert_eq!(schema["properties"]["pet"]["type"], "object");
        assert_eq!(
            schema["properties"]["pet"]["properties"]["kind"]["type"],
            "string"
        );
        assert_eq!(schema["properties"]["other"]["type"], "string");
        assert!(schema.get("$defs").is_none());
        assert!(schema.get("definitions").is_none());
    }

    #[test]
    fn an_external_or_recursive_reference_is_left_for_the_upstream_to_reject() {
        // Nothing inlining can do makes either legal, so the request
        // goes as written and the provider says why.
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {"remote": {"$ref": "https://example.com/Pet.json"}},
            "$defs": {"Pet": {"type": "string"}},
        });
        apply_schema_limits(&mut schema, &GEMINI_OPENAPI_SCHEMA_LIMITS);
        assert_eq!(
            schema["properties"]["remote"]["$ref"],
            "https://example.com/Pet.json"
        );

        let mut recursive = serde_json::json!({
            "$ref": "#/$defs/Node",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                },
            },
        });
        apply_schema_limits(&mut recursive, &GEMINI_OPENAPI_SCHEMA_LIMITS);
        // Expansion stops at the depth cap rather than looping; the
        // innermost `$ref` survives and Vertex rejects it.
        let json = recursive.to_string();
        assert!(json.contains("#/$defs/Node"), "{json}");
    }

    #[test]
    fn gemini_limits_drop_the_applicators_the_dialect_has_no_member_for() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"type": "string", "const": "x", "multipleOf": 2, "uniqueItems": true},
            },
            "allOf": [{"type": "object"}],
            "not": {"type": "null"},
            "if": {"type": "object"},
            "then": {"type": "object"},
            "else": {"type": "object"},
            "patternProperties": {"^a": {"type": "string"}},
            "prefixItems": [{"type": "string"}],
        });
        apply_schema_limits(&mut schema, &GEMINI_OPENAPI_SCHEMA_LIMITS);
        for dropped in [
            "allOf",
            "not",
            "if",
            "then",
            "else",
            "patternProperties",
            "prefixItems",
        ] {
            assert!(schema.get(dropped).is_none(), "{dropped} must be dropped");
        }
        let a = &schema["properties"]["a"];
        assert!(a.get("const").is_none());
        assert!(a.get("multipleOf").is_none());
        assert_eq!(a["description"], "multipleOf: 2, uniqueItems: true");
        // Bounds ARE part of the OpenAPI dialect, so they survive.
        let mut bounded = serde_json::json!({"type": "integer", "minimum": 1, "maximum": 9});
        apply_schema_limits(&mut bounded, &GEMINI_OPENAPI_SCHEMA_LIMITS);
        assert_eq!(bounded["minimum"], 1);
        assert_eq!(bounded["maximum"], 9);
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
