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

/// Every member of a schema object that itself holds a schema, or a
/// collection of them.
///
/// Both walkers in this module run off this one list. They used to carry
/// their own, which is how they came to disagree and how each came to
/// miss the applicator keywords entirely: a `maximum` under an
/// `if`/`then` branch stayed on the wire, and the object a
/// `Dict[str, Model]` field compiles to — `additionalProperties` in its
/// schema form — was never sealed.
///
/// Three member shapes are handled: a map whose *values* are schemas
/// (`properties`, `$defs`, `patternProperties`, `dependentSchemas`), a
/// list of schemas (`anyOf`, `prefixItems`), and a single schema
/// (`items`, `not`, `contains`, …) — with `items` also taking the
/// draft-07 tuple form, so both shapes are tried for every one of those.
/// A map's keys are never schemas and never keywords: they are names the
/// caller chose.
fn for_each_subschema(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    visit: &mut impl FnMut(&mut serde_json::Value),
) {
    const SCHEMA_MAPS: &[&str] = &[
        "properties",
        "patternProperties",
        "dependentSchemas",
        "$defs",
        "definitions",
    ];
    const SCHEMA_LISTS: &[&str] = &["anyOf", "oneOf", "allOf", "prefixItems"];
    const SCHEMA_VALUES: &[&str] = &[
        "items",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
    ];

    for key in SCHEMA_MAPS {
        if let Some(entries) = obj.get_mut(*key).and_then(|m| m.as_object_mut()) {
            for entry in entries.values_mut() {
                visit(entry);
            }
        }
    }
    for key in SCHEMA_LISTS {
        if let Some(entries) = obj.get_mut(*key).and_then(|l| l.as_array_mut()) {
            for entry in entries {
                visit(entry);
            }
        }
    }
    for key in SCHEMA_VALUES {
        match obj.get_mut(*key) {
            // The tuple form of `items`.
            Some(serde_json::Value::Array(entries)) => {
                for entry in entries {
                    visit(entry);
                }
            }
            // `additionalProperties: false` and friends are booleans,
            // not schemas.
            Some(value) if value.is_object() => visit(value),
            _ => {}
        }
    }
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
    }
    for_each_subschema(obj, &mut |sub| {
        walk_object_schemas(sub, require_every_property)
    });
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
    // Everything the `Schema` type has no member for, taken from
    // Google's own discovery document rather than from prose:
    // `GET https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta`,
    // whose `schemas.Schema.properties` is exactly `anyOf`, `default`,
    // `description`, `enum`, `example`, `format`, `items`, `maxItems`,
    // `maxLength`, `maxProperties`, `maximum`, `minItems`, `minLength`,
    // `minProperties`, `minimum`, `nullable`, `pattern`, `properties`,
    // `propertyOrdering`, `required`, `title`, `type`. Anything else is
    // rejected by name, so it cannot simply ride along.
    dropped_keywords: &[
        "additionalProperties",
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
        "readOnly",
        "writeOnly",
        "deprecated",
        "contentEncoding",
        "contentMediaType",
        "$schema",
        "$id",
        "$comment",
        "$anchor",
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

    // Every position that holds a schema, from the one list both walkers
    // read. A blind walk over the members would read the keys of
    // `properties` as keywords, so a caller whose document has a field
    // called `minimum` or `const` would lose that field and gain a
    // `description` built out of its own property names.
    for_each_subschema(obj, &mut |sub| narrow_schema_node(sub, limits));
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
    // Inlining runs on a copy under a global expansion budget, and the
    // schema is only adopted if it finished inside it. A recursive
    // definition with several alternatives multiplies at every level —
    // a few hundred bytes of schema can expand into hundreds of
    // megabytes — and this runs synchronously while a caller waits, so
    // the budget has to bound the total work, not just the depth of one
    // chain. On overrun nothing is rewritten: the `$ref`s go upstream
    // as the caller wrote them and the provider rejects what it cannot
    // resolve, which is the same outcome as any other unresolvable
    // reference here.
    // Each definition's serialised size, measured once. The count alone
    // does not bound the work: a definition carrying a few hundred
    // kilobytes of `description` reaches a hundred megabytes well inside
    // any sane expansion count, and that document is then serialised
    // again on its way upstream.
    let mut budget = RefBudget {
        expansions: MAX_REF_EXPANSIONS,
        bytes: MAX_REF_BYTES,
        sizes: defs
            .iter()
            .map(|(name, definition)| (name.clone(), definition.to_string().len()))
            .collect(),
    };
    let mut working = schema.clone();
    if !substitute_refs(&mut working, &defs, 0, &mut budget) {
        tracing::debug!("leaving $ref in place: inlining exceeded its expansion budget");
        return;
    }
    if let Some(obj) = working.as_object_mut() {
        obj.remove("$defs");
        obj.remove("definitions");
    }
    *schema = working;
}

/// How many times one `$ref` chain is followed before giving up. Bounds
/// the depth of a single chain; [`MAX_REF_EXPANSIONS`] bounds the whole
/// job, which is what a recursive definition with several alternatives
/// actually blows through.
const MAX_REF_DEPTH: usize = 8;

/// Total `$ref` expansions allowed for one schema. Comfortably above
/// any hand-written or generated schema — a large typed model produces
/// tens — and far below the point where expansion costs real time.
const MAX_REF_EXPANSIONS: usize = 2_000;

/// Total bytes of definition allowed to be spliced in. The count bounds
/// how many times a definition is copied; this bounds how large the
/// copies are, which is the half that decides how much memory the
/// expanded schema — and its re-serialisation on the way upstream —
/// occupies.
const MAX_REF_BYTES: usize = 4 * 1024 * 1024;

/// What one inlining pass is allowed to spend.
struct RefBudget {
    expansions: usize,
    bytes: usize,
    /// Serialised size of each definition, by the key that reaches it.
    sizes: std::collections::HashMap<String, usize>,
}

impl RefBudget {
    /// Charge one expansion of `name`. `false` once either half is gone.
    fn charge(&mut self, name: &str) -> bool {
        let Some(expansions) = self.expansions.checked_sub(1) else {
            return false;
        };
        let size = self.sizes.get(name).copied().unwrap_or(0);
        let Some(bytes) = self.bytes.checked_sub(size) else {
            return false;
        };
        self.expansions = expansions;
        self.bytes = bytes;
        true
    }
}

/// Expand every resolvable internal `$ref` in `node`. Returns `false`
/// when `budget` ran out, in which case `node` is left partly rewritten
/// and the caller must discard it.
fn substitute_refs(
    node: &mut serde_json::Value,
    defs: &serde_json::Map<String, serde_json::Value>,
    depth: usize,
    budget: &mut RefBudget,
) -> bool {
    if let Some(array) = node.as_array_mut() {
        for item in array {
            if !substitute_refs(item, defs, depth, budget) {
                return false;
            }
        }
        return true;
    }
    let Some(obj) = node.as_object_mut() else {
        return true;
    };
    if let Some(reference) = obj.get("$ref").and_then(|r| r.as_str()) {
        let Some(name) = internal_ref_name(reference) else {
            return true; // external reference: not ours to resolve
        };
        let Some(definition) = defs.get(name) else {
            return true;
        };
        if depth >= MAX_REF_DEPTH {
            return true; // recursive chain: leave the `$ref` in place
        }
        if !budget.charge(name) {
            return false;
        }
        let mut expanded = definition.clone();
        if !substitute_refs(&mut expanded, defs, depth + 1, budget) {
            return false;
        }
        *node = expanded;
        return true;
    }
    for value in obj.values_mut() {
        if !substitute_refs(value, defs, depth, budget) {
            return false;
        }
    }
    true
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
    fn a_property_named_like_a_keyword_is_not_mistaken_for_one() {
        // `properties` is a map of caller-chosen names, not of schema
        // keywords. Walking it blindly deletes a field called `minimum`
        // and builds a `description` out of the caller's own field
        // names — so a perfectly ordinary document schema comes out
        // missing members.
        let document = serde_json::json!({
            "type": "object",
            "properties": {
                "minimum": {"type": "number"},
                "maximum": {"type": "number"},
                "maxLength": {"type": "integer"},
                "const": {"type": "string"},
                "if": {"type": "boolean"},
                "allOf": {"type": "string"},
                "uniqueItems": {"type": "boolean"},
            },
        });
        for (label, limits) in [
            ("anthropic", &ANTHROPIC_SCHEMA_LIMITS),
            ("gemini", &GEMINI_OPENAPI_SCHEMA_LIMITS),
        ] {
            let mut schema = document.clone();
            apply_schema_limits(&mut schema, limits);
            assert_eq!(
                schema["properties"], document["properties"],
                "{label}: every property must survive untouched"
            );
            assert!(
                schema.get("description").is_none(),
                "{label}: no note belongs on a node with no constraints"
            );
        }
    }

    #[test]
    fn tuple_form_items_are_reached_by_both_walkers() {
        // Draft-07 spells a positional array as `items: [schema, …]`.
        // That is a list of schemas, not a schema, so a walker that
        // treats it as one skips every element — leaving those objects
        // open and their constraints on the wire.
        let document = serde_json::json!({
            "type": "array",
            "items": [
                {"type": "object", "properties": {"a": {"type": "string", "maxLength": 4}}},
                {"type": "integer", "minimum": 2},
            ],
        });

        let mut sealed = document.clone();
        seal_object_schemas(&mut sealed);
        assert_eq!(sealed["items"][0]["additionalProperties"], false);

        let mut narrowed = document.clone();
        apply_schema_limits(&mut narrowed, &ANTHROPIC_SCHEMA_LIMITS);
        assert!(narrowed["items"][0]["properties"]["a"]
            .get("maxLength")
            .is_none());
        assert_eq!(
            narrowed["items"][0]["properties"]["a"]["description"],
            "maxLength: 4"
        );
        assert!(narrowed["items"][1].get("minimum").is_none());
        assert_eq!(narrowed["items"][1]["description"], "minimum: 2");
    }

    #[test]
    fn a_branching_recursive_schema_is_left_alone_rather_than_expanded() {
        // Each level multiplies by the number of alternatives, so a few
        // hundred bytes can expand into hundreds of megabytes — on the
        // request path, with a caller waiting. The budget bounds the
        // whole job, and on overrun nothing is rewritten.
        let branches: Vec<serde_json::Value> = (0..7)
            .map(|i| serde_json::json!({format!("child{i}"): {"$ref": "#/$defs/Node"}}))
            .collect();
        let mut properties = serde_json::Map::new();
        for branch in &branches {
            for (k, v) in branch.as_object().unwrap() {
                properties.insert(k.clone(), v.clone());
            }
        }
        let schema = serde_json::json!({
            "$ref": "#/$defs/Node",
            "$defs": {"Node": {"type": "object", "properties": properties}},
        });

        let mut narrowed = schema.clone();
        let started = std::time::Instant::now();
        apply_schema_limits(&mut narrowed, &GEMINI_OPENAPI_SCHEMA_LIMITS);
        let elapsed = started.elapsed();

        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "inlining must not run away: took {elapsed:?}"
        );
        // Untouched: the `$ref` and its definition both survive, so the
        // upstream gets the schema as written and says why it cannot
        // take it.
        assert_eq!(narrowed["$ref"], "#/$defs/Node");
        assert!(narrowed["$defs"]["Node"].is_object());
        assert!(
            narrowed.to_string().len() < 4_096,
            "nothing should have been expanded"
        );
    }

    #[test]
    fn an_ordinary_nested_schema_still_inlines_under_the_budget() {
        // The budget must not be so tight that real schemas stop
        // inlining — the generated ones nest a handful of models deep.
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"$ref": "#/$defs/Inner"},
                "b": {"$ref": "#/$defs/Inner"},
                "c": {"type": "array", "items": {"$ref": "#/$defs/Inner"}},
            },
            "$defs": {
                "Inner": {"type": "object", "properties": {"leaf": {"$ref": "#/$defs/Leaf"}}},
                "Leaf": {"type": "string"},
            },
        });
        apply_schema_limits(&mut schema, &GEMINI_OPENAPI_SCHEMA_LIMITS);
        assert!(schema.get("$defs").is_none());
        assert_eq!(
            schema["properties"]["a"]["properties"]["leaf"]["type"],
            "string"
        );
        assert_eq!(
            schema["properties"]["c"]["items"]["properties"]["leaf"]["type"],
            "string"
        );
    }

    /// Every position a sub-schema can sit in, with the node to place
    /// there and the pointer that reaches it once placed. Both walkers
    /// run off one list, so a position missing from that list is missing
    /// from both — which is how the applicator keywords came to be
    /// skipped by each of them at once.
    fn subschema_positions() -> Vec<(&'static str, serde_json::Value, &'static str)> {
        let object = serde_json::json!({"type": "object", "properties": {"n": {"type": "string"}}});
        vec![
            (
                "properties",
                serde_json::json!({"child": object}),
                "/properties/child",
            ),
            ("items", object.clone(), "/items"),
            ("items (tuple)", serde_json::json!([object]), "/items/0"),
            ("prefixItems", serde_json::json!([object]), "/prefixItems/0"),
            ("anyOf", serde_json::json!([object]), "/anyOf/0"),
            ("allOf", serde_json::json!([object]), "/allOf/0"),
            ("$defs", serde_json::json!({"D": object}), "/$defs/D"),
            (
                "definitions",
                serde_json::json!({"D": object}),
                "/definitions/D",
            ),
            (
                "additionalProperties",
                object.clone(),
                "/additionalProperties",
            ),
            (
                "patternProperties",
                serde_json::json!({"^a": object}),
                "/patternProperties/^a",
            ),
            (
                "dependentSchemas",
                serde_json::json!({"a": object}),
                "/dependentSchemas/a",
            ),
            ("not", object.clone(), "/not"),
            ("if", object.clone(), "/if"),
            ("then", object.clone(), "/then"),
            ("else", object.clone(), "/else"),
            ("contains", object.clone(), "/contains"),
            ("propertyNames", object.clone(), "/propertyNames"),
            (
                "unevaluatedProperties",
                object.clone(),
                "/unevaluatedProperties",
            ),
            ("unevaluatedItems", object, "/unevaluatedItems"),
        ]
    }

    #[test]
    fn sealing_reaches_every_sub_schema_position() {
        for (name, value, pointer) in subschema_positions() {
            let mut schema = serde_json::json!({});
            schema[name.split_whitespace().next().unwrap()] = value;
            seal_object_schemas(&mut schema);
            assert_eq!(
                schema
                    .pointer(pointer)
                    .and_then(|s| s.get("additionalProperties")),
                Some(&serde_json::Value::Bool(false)),
                "{name}: the schema under {pointer} was never sealed"
            );
        }
    }

    #[test]
    fn narrowing_reaches_every_sub_schema_position() {
        for (name, value, pointer) in subschema_positions() {
            let key = name.split_whitespace().next().unwrap();
            // Put a constraint the Anthropic subset rejects on the leaf,
            // so reaching the node is observable.
            let mut value = value;
            let leaf_pointer = pointer.trim_start_matches(&format!("/{key}")).to_string();
            let leaf = match leaf_pointer.as_str() {
                "" => &mut value,
                p => value.pointer_mut(p).unwrap(),
            };
            leaf["properties"]["n"]["maxLength"] = 7.into();

            let mut schema = serde_json::json!({});
            schema[key] = value;
            apply_schema_limits(&mut schema, &ANTHROPIC_SCHEMA_LIMITS);

            let leaf = schema
                .pointer(&format!("{pointer}/properties/n"))
                .unwrap_or_else(|| panic!("{name}: {pointer} vanished"));
            assert!(
                leaf.get("maxLength").is_none(),
                "{name}: maxLength under {pointer} reached the wire"
            );
            assert_eq!(leaf["description"], "maxLength: 7", "{name}");
        }
    }

    #[test]
    fn a_definition_too_large_to_splice_is_left_as_a_reference() {
        // The expansion COUNT alone does not bound the work. This schema
        // is nowhere near the count budget — twenty references, no
        // recursion — but each one splices in 400KB, so the document
        // that would go upstream is megabytes of duplicated text, and it
        // is serialised again on the way out.
        let big = "x".repeat(400 * 1024);
        let mut properties = serde_json::Map::new();
        for i in 0..20 {
            properties.insert(format!("f{i}"), serde_json::json!({"$ref": "#/$defs/Big"}));
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": properties,
            "$defs": {"Big": {"type": "string", "description": big}},
        });

        let mut narrowed = schema.clone();
        apply_schema_limits(&mut narrowed, &GEMINI_OPENAPI_SCHEMA_LIMITS);

        // Sent exactly as the caller wrote it, references intact; the
        // provider says why it cannot take them.
        assert_eq!(narrowed["properties"]["f0"]["$ref"], "#/$defs/Big");
        assert!(narrowed["$defs"]["Big"].is_object());
        assert!(
            narrowed.to_string().len() < 2 * big.len(),
            "nothing should have been spliced in"
        );
    }

    #[test]
    fn a_handful_of_large_references_still_inlines() {
        // The byte budget must not be so tight that an ordinary schema
        // with a couple of well-documented models stops inlining.
        let text = "x".repeat(8 * 1024);
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"$ref": "#/$defs/Doc"},
                "b": {"$ref": "#/$defs/Doc"},
            },
            "$defs": {"Doc": {"type": "string", "description": text}},
        });
        let mut narrowed = schema;
        apply_schema_limits(&mut narrowed, &GEMINI_OPENAPI_SCHEMA_LIMITS);
        assert!(narrowed.get("$defs").is_none());
        assert_eq!(narrowed["properties"]["a"]["type"], "string");
        assert_eq!(narrowed["properties"]["b"]["type"], "string");
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
