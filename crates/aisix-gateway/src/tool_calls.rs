//! Tool-call handling every bridge shares.
//!
//! Gemini's thought signature rides in the tool-call id the gateway hands a
//! client (`<id>__thought__<signature>`, the signature URL-safe base64
//! without padding), because no client protocol has another field that
//! brings it back with the history. Only the Vertex Gemini bridge reads it.
//! Every other upstream must see the base id alone: OpenAI caps tool-call
//! ids at 64 characters and Bedrock also restricts them to
//! `[A-Za-z0-9_-]`, so a conversation that started on Gemini and moves to
//! another model would otherwise be rejected.

use std::borrow::Cow;

use futures::StreamExt;
use serde_json::{json, Value};

use crate::{
    BridgeError, ChatChunk, ChatChunkStream, ChatFormat, ChatMessage, ChatResponse, FinishReason,
    Role,
};

/// The tool declarations of a Responses request: its top-level `tools`,
/// then the `tools` of every `additional_tools` input item, in input order.
/// A client may declare its tools either way (the Codex CLI's lite mode
/// sends no top-level list at all), so every reader that translates or
/// counts a request's tools reads them through here.
///
/// A tool declared again replaces its earlier declaration and keeps the
/// earlier one's position: the Codex CLI leaves each `additional_tools` item
/// in the history and, when a tool's schema changes, appends another item
/// re-declaring it. A namespace declared again is merged member by member,
/// its header (description) taken from the later declaration. A tool without
/// a name is kept as is.
pub fn responses_declared_tools<'a>(
    tools: Option<&'a Value>,
    input: Option<&'a Value>,
) -> Vec<Cow<'a, Value>> {
    let additional = input
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| is_responses_additional_tools_item(item))
        .filter_map(|item| item.get("tools").and_then(Value::as_array))
        .flatten();
    let declared = tools
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .chain(additional);

    let mut out: Vec<Cow<'a, Value>> = Vec::new();
    let mut position: std::collections::HashMap<(bool, &'a str), usize> = Default::default();
    for tool in declared {
        let is_namespace = tool.get("type").and_then(Value::as_str) == Some("namespace");
        let Some(name) = tool.get("name").and_then(Value::as_str) else {
            out.push(Cow::Borrowed(tool));
            continue;
        };
        match position.get(&(is_namespace, name)) {
            Some(&i) if is_namespace => {
                let merged = merge_namespace(&out[i], tool);
                out[i] = Cow::Owned(merged);
            }
            Some(&i) => out[i] = Cow::Borrowed(tool),
            None => {
                position.insert((is_namespace, name), out.len());
                out.push(if is_namespace && has_repeated_member(tool) {
                    Cow::Owned(merge_namespace(&json!({}), tool))
                } else {
                    Cow::Borrowed(tool)
                });
            }
        }
    }
    out
}

fn namespace_members(namespace: &Value) -> &[Value] {
    namespace
        .get("tools")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

fn member_name(member: &Value) -> Option<&str> {
    member.get("name").and_then(Value::as_str)
}

fn has_repeated_member(namespace: &Value) -> bool {
    let mut seen = std::collections::HashSet::new();
    namespace_members(namespace)
        .iter()
        .filter_map(member_name)
        .any(|name| !seen.insert(name))
}

/// `later` declared over `earlier`: `later`'s header, and `earlier`'s
/// members with each one `later` declares again replaced in place and the
/// new ones appended.
fn merge_namespace(earlier: &Value, later: &Value) -> Value {
    let mut members: Vec<Value> = Vec::new();
    for member in namespace_members(earlier)
        .iter()
        .chain(namespace_members(later))
    {
        let existing = member_name(member)
            .and_then(|name| members.iter().position(|m| member_name(m) == Some(name)));
        match existing {
            Some(i) => members[i] = member.clone(),
            None => members.push(member.clone()),
        }
    }
    let mut merged = later.clone();
    merged["tools"] = Value::Array(members);
    merged
}

/// A Responses `input` item that declares tools rather than carrying a turn.
pub fn is_responses_additional_tools_item(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("additional_tools")
}

/// Separates a tool-call id from the thought signature appended to it.
pub const THOUGHT_SIGNATURE_SEPARATOR: &str = "__thought__";

/// `id` split into its base id and the thought signature appended to it.
pub fn split_thought_signature(id: &str) -> (&str, Option<&str>) {
    match id.split_once(THOUGHT_SIGNATURE_SEPARATOR) {
        Some((base, signature)) => (base, Some(signature)),
        None => (id, None),
    }
}

/// `id` without a thought-signature suffix.
pub fn strip_thought_signature(id: &str) -> &str {
    split_thought_signature(id).0
}

fn has_signature(v: Option<&Value>) -> bool {
    v.and_then(Value::as_str)
        .is_some_and(|id| id.contains(THOUGHT_SIGNATURE_SEPARATOR))
}

fn strip_in_place(v: Option<&mut Value>) {
    if let Some(Value::String(id)) = v {
        if let Some((base, _)) = id.split_once(THOUGHT_SIGNATURE_SEPARATOR) {
            *id = base.to_string();
        }
    }
}

/// `req` with every tool-call id (assistant `tool_calls[].id`, tool
/// `tool_call_id`) reduced to its base id. Borrowed back unchanged when no
/// id carries a signature, which is every request that never touched a
/// Gemini model.
pub fn chat_without_thought_signatures(req: Cow<'_, ChatFormat>) -> Cow<'_, ChatFormat> {
    let signed = req.messages.iter().any(|m| {
        m.tool_call_id
            .as_deref()
            .is_some_and(|id| id.contains(THOUGHT_SIGNATURE_SEPARATOR))
            || m.extra
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| calls.iter().any(|c| has_signature(c.get("id"))))
    });
    if !signed {
        return req;
    }
    let mut req = req.into_owned();
    for m in &mut req.messages {
        if let Some(id) = m.tool_call_id.as_mut() {
            let base = strip_thought_signature(id).len();
            id.truncate(base);
        }
        if let Some(calls) = m.extra.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for call in calls {
                strip_in_place(call.get_mut("id"));
            }
        }
    }
    Cow::Owned(req)
}

/// What a raw-body key holds, for the two walkers below: they must agree, or
/// a signed body takes the path that does not strip it.
enum BodyField {
    ToolCallId,
    /// A call's own arguments: the caller's data, whatever keys they hold.
    Arguments,
    Nested,
}

fn body_field(key: &str, value: &Value) -> BodyField {
    match (key, value) {
        ("id" | "call_id" | "tool_call_id" | "tool_use_id", _) => BodyField::ToolCallId,
        ("arguments", _) | ("input", Value::Object(_)) => BodyField::Arguments,
        _ => BodyField::Nested,
    }
}

/// Whether [`strip_thought_signatures_in_body`] would change `body`: some
/// tool-call id in it carries a signature. Lets a caller that only holds
/// the body borrowed skip copying it when there is nothing to strip.
pub fn body_has_thought_signatures(body: &Value) -> bool {
    match body {
        Value::Object(obj) => obj.iter().any(|(key, value)| match body_field(key, value) {
            BodyField::ToolCallId => has_signature(Some(value)),
            BodyField::Arguments => false,
            BodyField::Nested => body_has_thought_signatures(value),
        }),
        Value::Array(items) => items.iter().any(body_has_thought_signatures),
        _ => false,
    }
}

/// Reduce every tool-call id in a raw request body (Anthropic Messages
/// `tool_use.id` / `tool_result.tool_use_id`, Responses `call_id`, chat
/// `tool_calls[].id` / `tool_call_id`) to its base id, wherever it sits.
/// A call's own arguments (`tool_use.input`, `arguments`) are the caller's
/// data and are left alone, whatever keys they hold.
pub fn strip_thought_signatures_in_body(body: &mut Value) {
    match body {
        Value::Object(obj) => {
            for (key, value) in obj.iter_mut() {
                match body_field(key, value) {
                    BodyField::ToolCallId => strip_in_place(Some(value)),
                    BodyField::Arguments => {}
                    BodyField::Nested => strip_thought_signatures_in_body(value),
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(strip_thought_signatures_in_body),
        _ => {}
    }
}

/// How a request that used the deprecated function-calling fields is
/// answered, returned by [`legacy_functions_as_tools`].
#[derive(Debug, Clone, Copy, Default)]
#[must_use = "the reply must be rendered back in the shape the caller asked for"]
pub struct LegacyFunctionCalling {
    /// The caller declared `functions` (and no `tools`), so it reads the
    /// answer's call from `message.function_call`.
    legacy_reply: bool,
}

/// Rewrite the deprecated `functions` / `function_call` request fields and
/// the history turns they produce into their `tools` equivalents, for a
/// bridge whose upstream has no slot for the legacy shape but has one for
/// tools:
///
/// - `functions[]` → `tools[]` of `{type: "function", function}` (appended
///   when the caller also sent `tools`);
/// - `function_call` → `tool_choice` (`"none"` / `"auto"` as is, `{name}`
///   as a named function choice), unless the caller sent `tool_choice`;
/// - an assistant turn's `function_call` → a one-entry `tool_calls`, and
///   the `function` result turn that follows → a `tool` result answering
///   that call's id.
///
/// The returned [`LegacyFunctionCalling`] renders the answer back in the
/// legacy shape when the caller asked in it. OpenAI-wire upstreams take
/// the legacy fields themselves and never go through this.
///
/// A field whose shape is not one the legacy API defines is refused with a
/// 400 naming it.
pub fn legacy_functions_as_tools(
    req: Cow<'_, ChatFormat>,
) -> Result<(Cow<'_, ChatFormat>, LegacyFunctionCalling), BridgeError> {
    // A legacy field set to `null` is absent: an SDK replaying a turn it
    // dumped writes `function_call: null` beside its content.
    const LEGACY: [&str; 2] = ["functions", "function_call"];
    let set = |v: Option<&Value>| v.is_some_and(|v| !v.is_null());
    let declares = LEGACY.iter().any(|k| set(req.extra.get(*k)));
    let replays = req
        .messages
        .iter()
        .any(|m| m.role == Role::Function || set(m.extra.get("function_call")));
    let nulls = LEGACY
        .iter()
        .any(|k| req.extra.get(*k).is_some_and(Value::is_null))
        || req
            .messages
            .iter()
            .any(|m| m.extra.get("function_call").is_some_and(Value::is_null));
    if !declares && !replays && !nulls {
        return Ok((req, LegacyFunctionCalling::default()));
    }
    let mut req = req.into_owned();
    req.extra
        .retain(|k, v| !(v.is_null() && LEGACY.contains(&k.as_str())));
    for message in &mut req.messages {
        if message
            .extra
            .get("function_call")
            .is_some_and(Value::is_null)
        {
            message.extra.remove("function_call");
        }
    }
    let legacy_reply = req.extra.contains_key("functions") && !set(req.extra.get("tools"));

    if let Some(functions) = req.extra.remove("functions") {
        let Value::Array(functions) = functions else {
            return Err(BridgeError::InvalidRequest(
                "`functions` must be an array of function definitions".into(),
            ));
        };
        let mapped = functions
            .into_iter()
            .map(|function| json!({"type": "function", "function": function}));
        match req.extra.get_mut("tools") {
            Some(Value::Array(tools)) => tools.extend(mapped),
            _ => {
                req.extra
                    .insert("tools".into(), Value::Array(mapped.collect()));
            }
        }
    }
    if let Some(function_call) = req.extra.remove("function_call") {
        let choice = match &function_call {
            Value::String(mode) if mode == "none" || mode == "auto" => function_call.clone(),
            Value::Object(o) if o.get("name").is_some_and(Value::is_string) => {
                json!({"type": "function", "function": {"name": o["name"]}})
            }
            _ => {
                return Err(BridgeError::InvalidRequest(
                    "`function_call` must be \"none\", \"auto\" or {\"name\": \"...\"}".into(),
                ))
            }
        };
        req.extra.entry("tool_choice").or_insert(choice);
    }

    let mut last_call_id: Option<String> = None;
    for (index, message) in req.messages.iter_mut().enumerate() {
        if let Some(call) = message.extra.remove("function_call") {
            let Some(name) = call.get("name").and_then(Value::as_str) else {
                return Err(BridgeError::InvalidRequest(format!(
                    "`messages[{index}].function_call.name` must be a string"
                )));
            };
            let id = format!("call_function_{index}");
            let arguments = call.get("arguments").cloned().unwrap_or(json!("{}"));
            let tool_call = json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            });
            match message.extra.get_mut("tool_calls") {
                Some(Value::Array(calls)) => calls.push(tool_call),
                _ => {
                    message
                        .extra
                        .insert("tool_calls".into(), json!([tool_call]));
                }
            }
            last_call_id = Some(id);
        }
        if message.role == Role::Function {
            message.role = Role::Tool;
            if message.tool_call_id.is_none() {
                message.tool_call_id = Some(
                    last_call_id
                        .clone()
                        .unwrap_or_else(|| format!("call_function_{index}")),
                );
            }
        }
    }
    Ok((Cow::Owned(req), LegacyFunctionCalling { legacy_reply }))
}

impl LegacyFunctionCalling {
    /// `resp` in the shape the caller asked in: with the first tool call
    /// of each choice as `message.function_call` and a `tool_calls` finish
    /// as `function_call`. The legacy shape holds one call per turn, so
    /// any further call has no slot.
    pub fn reply(self, mut resp: ChatResponse) -> ChatResponse {
        if !self.legacy_reply {
            return resp;
        }
        legacy_message(&mut resp.message, &mut resp.finish_reason);
        for choice in &mut resp.more_choices {
            legacy_message(&mut choice.message, &mut choice.finish_reason);
        }
        resp
    }

    /// [`Self::reply`] for a stream: each tool-call delta of the first call
    /// becomes a `delta.function_call` fragment.
    pub fn reply_stream(self, stream: ChatChunkStream) -> ChatChunkStream {
        if !self.legacy_reply {
            return stream;
        }
        Box::pin(stream.map(|chunk| chunk.map(legacy_chunk)))
    }
}

fn legacy_finish(finish: &mut FinishReason) {
    if *finish == FinishReason::ToolCalls {
        *finish = FinishReason::Other("function_call".into());
    }
}

fn legacy_message(message: &mut ChatMessage, finish: &mut FinishReason) {
    if let Some(Value::Array(calls)) = message.extra.remove("tool_calls") {
        if let Some(function) = calls
            .into_iter()
            .next()
            .and_then(|c| c.get("function").cloned())
        {
            message.extra.insert("function_call".into(), function);
        }
    }
    legacy_finish(finish);
}

fn legacy_chunk(mut chunk: ChatChunk) -> ChatChunk {
    if let Some(calls) = chunk.delta.tool_calls.take() {
        let first = calls
            .into_iter()
            .find(|c| c.get("index").and_then(Value::as_u64).unwrap_or(0) == 0);
        if let Some(function) = first.and_then(|c| c.get("function").cloned()) {
            chunk.delta.extra.insert("function_call".into(), function);
        }
    }
    if let Some(finish) = chunk.finish_reason.as_mut() {
        legacy_finish(finish);
    }
    chunk
}

/// The text a guardrail scans for a `tool_calls` array: every call as it
/// is, minus its id. An id is an opaque handle that links a call to its
/// result; it carries no content, and scanning it only produces false
/// positives — Gemini ids carry hundreds of bytes of thought signature, and
/// upstream-generated ids are random strings a keyword rule can match by
/// chance.
pub fn tool_calls_scan_text(tool_calls: &Value) -> String {
    match tool_calls {
        Value::Array(calls) => blocks_scan_text(calls),
        other => other.to_string(),
    }
}

/// The top-level keys that hold a tool-call id on any wire this gateway
/// speaks: a chat call's and an Anthropic block's `id`, a Responses
/// `call_id`, and the result-side back-references.
pub const TOOL_CALL_ID_KEYS: [&str; 4] = ["id", "call_id", "tool_use_id", "tool_call_id"];

/// [`tool_calls_scan_text`] over any list of tool-call-bearing objects
/// (an Anthropic `content` array included): each one serialized without its
/// id keys, as a JSON array.
pub fn blocks_scan_text<'a>(blocks: impl IntoIterator<Item = &'a Value>) -> String {
    struct WithoutIds<'a>(&'a Value);
    impl serde::Serialize for WithoutIds<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeMap;
            match self.0 {
                Value::Object(obj) => {
                    let mut map = serializer.serialize_map(None)?;
                    for (k, v) in obj {
                        if !TOOL_CALL_ID_KEYS.contains(&k.as_str()) {
                            map.serialize_entry(k, v)?;
                        }
                    }
                    map.end()
                }
                other => other.serialize(serializer),
            }
        }
    }
    let blocks: Vec<WithoutIds<'_>> = blocks.into_iter().map(WithoutIds).collect();
    serde_json::to_string(&blocks).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(body: &Value) -> Vec<Value> {
        responses_declared_tools(body.get("tools"), body.get("input"))
            .into_iter()
            .map(Cow::into_owned)
            .collect()
    }

    /// A tool declared again replaces the earlier declaration in place; a
    /// namespace declared again is merged member by member.
    #[test]
    fn a_redeclared_tool_replaces_the_earlier_declaration_in_place() {
        let body = json!({
            "tools": [{"type": "function", "name": "top", "description": "old"}],
            "input": [
                {"type": "additional_tools", "tools": [
                    {"type": "namespace", "name": "functions", "description": "v1", "tools": [
                        {"type": "custom", "name": "exec", "description": "old exec"},
                        {"type": "function", "name": "wait"},
                    ]},
                    {"type": "web_search"},
                ]},
                {"role": "user", "content": "hi"},
                {"type": "additional_tools", "tools": [
                    {"type": "namespace", "name": "functions", "description": "v2", "tools": [
                        {"type": "custom", "name": "exec", "description": "new exec"},
                        {"type": "function", "name": "apply"},
                    ]},
                    {"type": "custom", "name": "top", "description": "new"},
                    {"type": "web_search"},
                ]},
            ],
        });
        assert_eq!(
            declared(&body),
            vec![
                json!({"type": "custom", "name": "top", "description": "new"}),
                json!({"type": "namespace", "name": "functions", "description": "v2", "tools": [
                    {"type": "custom", "name": "exec", "description": "new exec"},
                    {"type": "function", "name": "wait"},
                    {"type": "function", "name": "apply"},
                ]}),
                json!({"type": "web_search"}),
                json!({"type": "web_search"}),
            ]
        );
    }

    #[test]
    fn a_member_repeated_inside_one_namespace_is_declared_once() {
        let body = json!({"tools": [{"type": "namespace", "name": "ns", "tools": [
            {"type": "function", "name": "a", "description": "first"},
            {"type": "function", "name": "a", "description": "second"},
        ]}]});
        assert_eq!(
            declared(&body),
            vec![json!({"type": "namespace", "name": "ns", "tools": [
                {"type": "function", "name": "a", "description": "second"},
            ]})]
        );
    }
    use serde_json::json;

    #[test]
    fn signed_ids_are_reduced_to_their_base_everywhere() {
        let req: ChatFormat = serde_json::from_value(json!({
            "model": "m",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_a__thought__c2ln", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_a__thought__c2ln", "content": "ok"}
            ]
        }))
        .unwrap();
        let stripped = chat_without_thought_signatures(Cow::Borrowed(&req));
        assert!(matches!(stripped, Cow::Owned(_)));
        assert_eq!(stripped.messages[0].extra["tool_calls"][0]["id"], "call_a");
        assert_eq!(stripped.messages[0].extra["tool_calls"][1]["id"], "call_b");
        assert_eq!(stripped.messages[1].tool_call_id.as_deref(), Some("call_a"));
        assert!(matches!(
            chat_without_thought_signatures(Cow::Borrowed(&stripped)),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn raw_bodies_lose_the_suffix_on_every_id_field() {
        let mut body = json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_a__thought__c2ln", "name": "f",
                 "input": {"id": "kept__thought__as-is"}},
                {"type": "text", "text": "an id__thought__ in prose stays"}
            ]}, {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_a__thought__c2ln", "content": "ok"}
            ]}],
            "input": [{"type": "function_call", "call_id": "call_b__thought__eHl6"}]
        });
        assert!(body_has_thought_signatures(&body));
        strip_thought_signatures_in_body(&mut body);
        assert!(!body_has_thought_signatures(&body));
        assert_eq!(body["messages"][0]["content"][0]["id"], "call_a");
        assert_eq!(
            body["messages"][0]["content"][0]["input"]["id"],
            "kept__thought__as-is"
        );
        assert_eq!(
            body["messages"][0]["content"][1]["text"],
            "an id__thought__ in prose stays"
        );
        assert_eq!(body["messages"][1]["content"][0]["tool_use_id"], "call_a");
        assert_eq!(body["input"][0]["call_id"], "call_b");
    }

    /// The check looks where the strip rewrites and nowhere else: a
    /// signature-looking value inside a call's own arguments or in prose
    /// does not count.
    #[test]
    fn signature_check_ignores_arguments_and_prose() {
        let body = json!({"input": [
            {"type": "function_call", "call_id": "call_b", "arguments": "{\"id\":\"x__thought__y\"}"},
            {"type": "tool_use", "id": "call_c", "input": {"id": "x__thought__y"}},
            {"type": "message", "content": [{"type": "input_text", "text": "id__thought__"}]}
        ]});
        assert!(!body_has_thought_signatures(&body));
    }

    #[test]
    fn legacy_function_calling_maps_onto_tools_and_back() {
        let req: ChatFormat = serde_json::from_value(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null,
                 "function_call": {"name": "w", "arguments": "{\"c\":\"x\"}"}},
                {"role": "function", "name": "w", "content": "75F"}
            ],
            "functions": [{"name": "w", "parameters": {"type": "object"}}],
            "function_call": {"name": "w"}
        }))
        .unwrap();
        let (mapped, legacy) = legacy_functions_as_tools(Cow::Borrowed(&req)).unwrap();
        assert!(!mapped.extra.contains_key("functions"));
        assert_eq!(mapped.extra["tools"][0]["function"]["name"], "w");
        assert_eq!(mapped.extra["tool_choice"]["function"]["name"], "w");
        let call_id = &mapped.messages[1].extra["tool_calls"][0]["id"];
        assert_eq!(mapped.messages[2].role, Role::Tool);
        assert_eq!(mapped.messages[2].tool_call_id.as_deref(), call_id.as_str());

        let mut message = ChatMessage::assistant("");
        message.extra.insert(
            "tool_calls".into(),
            json!([{"id": "t", "type": "function", "function": {"name": "w", "arguments": "{}"}}]),
        );
        let resp = legacy.reply(ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message,
            finish_reason: FinishReason::ToolCalls,
            logprobs: None,
            more_choices: Vec::new(),
            usage: crate::UsageStats::default(),
            service_tier: None,
        });
        assert_eq!(resp.message.extra["function_call"]["name"], "w");
        assert!(!resp.message.extra.contains_key("tool_calls"));
        assert_eq!(
            resp.finish_reason,
            FinishReason::Other("function_call".into())
        );
    }

    #[test]
    fn null_legacy_fields_are_absent() {
        let req: ChatFormat = serde_json::from_value(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello", "function_call": null, "tool_calls": null},
                {"role": "user", "content": "again"}
            ],
            "functions": null,
            "function_call": null
        }))
        .unwrap();
        let (mapped, legacy) = legacy_functions_as_tools(Cow::Borrowed(&req)).unwrap();
        assert!(!mapped.extra.contains_key("functions"));
        assert!(!mapped.extra.contains_key("function_call"));
        assert!(!mapped.extra.contains_key("tools"));
        assert!(!mapped.extra.contains_key("tool_choice"));
        assert!(!mapped.messages[1].extra.contains_key("function_call"));
        assert!(!legacy.legacy_reply);
    }

    #[test]
    fn a_function_call_of_another_shape_is_refused() {
        let mut req = ChatFormat::new("m", vec![]);
        req.extra.insert("function_call".into(), json!(42));
        assert!(matches!(
            legacy_functions_as_tools(Cow::Borrowed(&req)),
            Err(BridgeError::InvalidRequest(m)) if m.contains("function_call")
        ));
    }

    #[test]
    fn scan_text_carries_no_tool_call_id() {
        let text = tool_calls_scan_text(&json!([
            {"id": "call_a__thought__c2ln", "type": "function",
             "function": {"name": "lookup", "arguments": "{\"id\":\"arg_id\"}"}}
        ]));
        assert!(!text.contains("call_a") && !text.contains("c2ln"), "{text}");
        assert!(text.contains("lookup") && text.contains("arg_id"), "{text}");
        let blocks = blocks_scan_text(&[
            json!({"type": "tool_use", "id": "toolu_x", "name": "f", "input": {"id": "in_id"}}),
            json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_y", "content": []}),
        ]);
        assert!(
            !blocks.contains("toolu_x") && !blocks.contains("srvtoolu_y"),
            "{blocks}"
        );
        assert!(blocks.contains("in_id"), "{blocks}");
    }
}
