use std::collections::{HashMap, HashSet, VecDeque};

use aisix_gateway::ChatFormat;
use serde_json::{json, Map, Value};

use crate::ProxyError;

fn invalid(message: impl Into<String>) -> ProxyError {
    ProxyError::InvalidRequest(message.into())
}

// ProtoJSON parsers accept the proto field name as well as its lowerCamelCase name.
fn field<'a>(object: &'a Value, camel: &str, snake: &str) -> Option<&'a Value> {
    object.get(camel).or_else(|| object.get(snake))
}

pub(super) fn number(value: &Value) -> Option<f64> {
    value.as_f64().or_else(|| value.as_str()?.parse().ok())
}

fn array(value: &Value) -> Result<&[Value], ProxyError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("expected a JSON array"))
}

fn string<'a>(obj: &'a Value, key: &str) -> Result<&'a str, ProxyError> {
    obj.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("expected a string for {key}")))
}

// Gemini Schema uses ProtoJSON int64 strings for size constraints. Walk
// schema positions only: an example, default or property name is user data.
fn json_schema(mut schema: Value) -> Result<Value, ProxyError> {
    let Some(obj) = schema.as_object_mut() else {
        return Ok(schema);
    };
    if let Some(Value::String(ty)) = obj.get_mut("type") {
        ty.make_ascii_lowercase();
    }
    for (name, snake) in [
        ("minLength", "min_length"),
        ("maxLength", "max_length"),
        ("minItems", "min_items"),
        ("maxItems", "max_items"),
        ("minProperties", "min_properties"),
        ("maxProperties", "max_properties"),
    ] {
        if let Some(v) = obj.remove(name).or_else(|| obj.remove(snake)) {
            let count = v
                .as_u64()
                .or_else(|| v.as_str()?.parse::<u64>().ok())
                .ok_or_else(|| {
                    invalid(format!("expected a nonnegative integer for schema {name}"))
                })?;
            obj.insert(name.into(), count.into());
        }
    }
    obj.remove("propertyOrdering");
    obj.remove("property_ordering");
    if let Some(Value::Object(properties)) = obj.get_mut("properties") {
        for prop in properties.values_mut() {
            *prop = json_schema(prop.take())?;
        }
    }
    if let Some(items) = obj.get_mut("items") {
        *items = json_schema(items.take())?;
    }
    if let Some(mut options) = obj.remove("anyOf").or_else(|| obj.remove("any_of")) {
        if let Some(options) = options.as_array_mut() {
            for option in options {
                *option = json_schema(option.take())?;
            }
        }
        obj.insert("anyOf".into(), options);
    }
    if obj.remove("nullable") == Some(Value::Bool(true)) {
        if let Some(ty) = obj.get_mut("type") {
            *ty = json!([ty, "null"]);
        }
    }
    Ok(schema)
}

fn schema(
    obj: &Value,
    raw: (&str, &str),
    typed: (&str, &str),
) -> Result<Option<Value>, ProxyError> {
    let value = match field(obj, raw.0, raw.1) {
        Some(raw) => Some(raw.clone()),
        None => field(obj, typed.0, typed.1)
            .cloned()
            .map(json_schema)
            .transpose()?,
    };
    if value
        .as_ref()
        .is_some_and(|s| !s.is_object() && !s.is_boolean())
    {
        return Err(invalid("expected a JSON schema"));
    }
    Ok(value)
}

fn media(part: &Value) -> Result<Option<Value>, ProxyError> {
    let inline = field(part, "inlineData", "inline_data");
    let file = field(part, "fileData", "file_data");
    let Some(data) = inline.or(file) else {
        return Ok(None);
    };
    let mime = field(data, "mimeType", "mime_type")
        .and_then(Value::as_str)
        .unwrap_or("application/octet-stream");
    let url = if inline.is_some() {
        format!("data:{mime};base64,{}", string(data, "data")?)
    } else {
        field(data, "fileUri", "file_uri")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("expected fileUri"))?
            .to_owned()
    };
    Ok(Some(if mime.starts_with("image/") {
        json!({"type":"image_url","image_url":{"url":url}})
    } else if inline.is_some() {
        json!({"type":"file","file":{"file_data":url}})
    } else {
        json!({"type":"file","file":{"file_id":url}})
    }))
}

fn content(blocks: Vec<Value>) -> Value {
    if blocks.iter().all(|b| b["type"] == "text") {
        Value::String(blocks.iter().filter_map(|b| b["text"].as_str()).collect())
    } else {
        Value::Array(blocks)
    }
}

pub(super) fn execution_slot(part: &Value) -> Option<(&'static str, &'static str)> {
    [
        ("executableCode", "code"),
        ("executable_code", "code"),
        ("codeExecutionResult", "output"),
        ("code_execution_result", "output"),
    ]
    .into_iter()
    .find(|&(key, child)| part.get(key).and_then(|v| v.get(child)).is_some())
}

fn contents(body: &Value) -> &[Value] {
    body.get("contents").map_or(&[], |v| {
        v.as_array()
            .map(Vec::as_slice)
            .unwrap_or_else(|| std::slice::from_ref(v))
    })
}

pub(crate) fn has_execution(body: &Value) -> bool {
    field(body, "systemInstruction", "system_instruction")
        .into_iter()
        .chain(contents(body))
        .flat_map(|turn| turn["parts"].as_array().into_iter().flatten())
        .any(|part| execution_slot(part).is_some())
}

#[derive(Default)]
struct History<'a> {
    pending: HashMap<String, VecDeque<String>>,
    explicit_ids: HashSet<&'a str>,
    next: usize,
}

impl History<'_> {
    fn messages(
        &mut self,
        turn: &Value,
        role: &str,
        out: &mut Vec<Value>,
        policy_view: bool,
    ) -> Result<(), ProxyError> {
        let mut blocks = Vec::new();
        let mut calls = Vec::new();
        let mut results = Vec::new();
        let mut thoughts = String::new();
        let mut execution_text = String::new();
        for part in array(&turn["parts"])? {
            if let Some(call) = field(part, "functionCall", "function_call") {
                let name = string(call, "name")?;
                let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
                if role != "assistant" || !args.is_object() {
                    return Err(invalid("invalid functionCall"));
                }
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| loop {
                        let id = format!("call_{}", self.next);
                        self.next += 1;
                        if !self.explicit_ids.contains(id.as_str()) {
                            break id;
                        }
                    });
                self.pending
                    .entry(name.into())
                    .or_default()
                    .push_back(id.clone());
                let mut call = json!({"id":id,"type":"function","function":{"name":name,"arguments":args.to_string()}});
                if let Some(signature) = field(part, "thoughtSignature", "thought_signature") {
                    call["extra_content"] = json!({"google":{"thought_signature":signature}});
                }
                calls.push(call);
            } else if let Some(result) = field(part, "functionResponse", "function_response") {
                let name = string(result, "name")?;
                let queue = self.pending.entry(name.into()).or_default();
                let id = if let Some(id) = result.get("id").and_then(Value::as_str) {
                    if let Some(i) = queue.iter().position(|v| v == id) {
                        queue.remove(i);
                    }
                    id.to_owned()
                } else {
                    queue.pop_front().ok_or_else(|| {
                        invalid(format!(
                            "functionResponse has no matching functionCall for {name}"
                        ))
                    })?
                };
                let response = result
                    .get("response")
                    .filter(|v| v.is_object())
                    .ok_or_else(|| invalid("expected functionResponse.response object"))?;
                let mut blocks = vec![json!({"type":"text","text":response.to_string()})];
                if let Some(parts) = result.get("parts") {
                    for media_part in array(parts)? {
                        if let Some(block) = media(media_part)? {
                            blocks.push(block);
                        }
                    }
                }
                results.push(json!({"role":"tool","tool_call_id":id,"content":content(blocks)}));
            } else if let Some(text) = part.get("text") {
                let text = text
                    .as_str()
                    .ok_or_else(|| invalid("expected text string"))?;
                if role == "assistant" && part["thought"] == true {
                    thoughts.push_str(text);
                } else {
                    blocks.push(json!({"type":"text","text":text}));
                }
            } else if let Some((key, child)) = execution_slot(part) {
                if policy_view {
                    if let Some(text) = part[key][child].as_str() {
                        execution_text.push('\n');
                        execution_text.push_str(text);
                    }
                }
            } else if let Some(block) = media(part)? {
                blocks.push(block);
            }
        }
        // Tool replies must immediately follow the assistant's calls, even when
        // a Gemini user turn combines those replies with a new user message.
        out.extend(results);
        if !execution_text.is_empty() {
            blocks.push(json!({"type":"text","text":execution_text}));
        }
        if !blocks.is_empty() || !calls.is_empty() || !thoughts.is_empty() {
            let mut msg = json!({"role":role,"content":if blocks.is_empty() {Value::Null} else {content(blocks)}});
            if !calls.is_empty() {
                msg["tool_calls"] = calls.into();
            }
            if !thoughts.is_empty() {
                msg["reasoning_content"] = thoughts.into();
            }
            out.push(msg);
        }
        Ok(())
    }
}

fn messages(body: &Value, policy_view: bool) -> Result<Vec<Value>, ProxyError> {
    if body.get("contents").is_none() {
        return Err(invalid("expected contents"));
    }
    let mut messages = Vec::new();
    let contents = contents(body);
    let explicit_ids = contents
        .iter()
        .flat_map(|turn| turn["parts"].as_array().into_iter().flatten())
        .filter_map(|p| {
            field(p, "functionCall", "function_call")
                .or_else(|| field(p, "functionResponse", "function_response"))
        })
        .filter_map(|call| call["id"].as_str())
        .collect();
    let mut history = History {
        explicit_ids,
        ..Default::default()
    };
    if let Some(system) =
        field(body, "systemInstruction", "system_instruction").filter(|v| !v.is_null())
    {
        history.messages(system, "system", &mut messages, policy_view)?;
    }
    for turn in contents {
        let role = match turn.get("role").and_then(Value::as_str).unwrap_or("user") {
            "model" => "assistant",
            "user" => "user",
            _ => return Err(invalid("expected contents role user or model")),
        };
        history.messages(turn, role, &mut messages, policy_view)?;
    }
    Ok(messages)
}

pub(crate) fn policy_input(model: &str, body: &Value) -> Result<ChatFormat, ProxyError> {
    serde_json::from_value(json!({"model":model,"messages":messages(body, true)?}))
        .map_err(|e| invalid(e.to_string()))
}

pub(crate) fn translate(
    model: &str,
    streaming: bool,
    body: &Value,
) -> Result<ChatFormat, ProxyError> {
    let messages = messages(body, false)?;
    let mut req = json!({"model":model, "stream":streaming,"messages":messages});
    if streaming {
        req["stream_options"] = json!({"include_usage":true});
    }
    let cfg = field(body, "generationConfig", "generation_config").unwrap_or(&Value::Null);
    if !cfg.is_null() && !cfg.is_object() {
        return Err(invalid("expected generationConfig object"));
    }
    for (camel, snake, key) in [
        ("temperature", "temperature", "temperature"),
        ("maxOutputTokens", "max_output_tokens", "max_tokens"),
        ("topP", "top_p", "top_p"),
        ("stopSequences", "stop_sequences", "stop"),
        ("candidateCount", "candidate_count", "n"),
        ("presencePenalty", "presence_penalty", "presence_penalty"),
        ("frequencyPenalty", "frequency_penalty", "frequency_penalty"),
        ("responseLogprobs", "response_logprobs", "logprobs"),
        ("logprobs", "logprobs", "top_logprobs"),
        ("seed", "seed", "seed"),
    ] {
        if let Some(v) = field(cfg, camel, snake) {
            req[key] = match key {
                "temperature" | "top_p" | "presence_penalty" | "frequency_penalty" => {
                    let n = number(v)
                        .and_then(serde_json::Number::from_f64)
                        .ok_or_else(|| invalid(format!("expected number for {camel}")))?;
                    Value::Number(n)
                }
                "max_tokens" | "n" | "top_logprobs" | "seed" => {
                    let n = v
                        .as_i64()
                        .or_else(|| v.as_str()?.parse::<i64>().ok())
                        .ok_or_else(|| invalid(format!("expected integer for {camel}")))?;
                    Value::from(n)
                }
                "stop"
                    if !(v.is_string()
                        || v.as_array()
                            .is_some_and(|arr| arr.iter().all(Value::is_string))) =>
                {
                    return Err(invalid("expected stopSequences strings"))
                }
                "logprobs" if !v.is_boolean() => {
                    return Err(invalid("expected responseLogprobs boolean"))
                }
                _ => v.clone(),
            };
        }
    }
    if let Some(v) = field(cfg, "responseModalities", "response_modalities") {
        req["modalities"] = array(v)?
            .iter()
            .filter_map(Value::as_str)
            .filter(|v| matches!(*v, "TEXT" | "AUDIO"))
            .map(str::to_ascii_lowercase)
            .collect();
        if req["modalities"] == json!([]) {
            req.as_object_mut().unwrap().remove("modalities");
        }
    }
    let mime = field(cfg, "responseMimeType", "response_mime_type").and_then(Value::as_str);
    if matches!(mime, None | Some("application/json")) {
        if let Some(schema) = schema(
            cfg,
            ("responseJsonSchema", "response_json_schema"),
            ("responseSchema", "response_schema"),
        )? {
            req["response_format"] =
                json!({"type":"json_schema","json_schema":{"name":"response","schema":schema}});
        } else if mime.is_some() {
            req["response_format"] = json!({"type":"json_object"});
        }
    }
    let thinking = field(cfg, "thinkingConfig", "thinking_config").unwrap_or(&Value::Null);
    if !thinking.is_null() && !thinking.is_object() {
        return Err(invalid("expected thinkingConfig object"));
    }
    if let Some(level) = field(thinking, "thinkingLevel", "thinking_level").and_then(Value::as_str)
    {
        req["reasoning_effort"] = level.to_ascii_lowercase().into();
    }
    if let Some(tools) = body.get("tools") {
        let mut declarations = Vec::new();
        for tool in array(tools)? {
            if let Some(functions) = field(tool, "functionDeclarations", "function_declarations") {
                for decl in array(functions)? {
                    let mut func = Map::from_iter([(
                        "name".into(),
                        Value::String(string(decl, "name")?.into()),
                    )]);
                    if let Some(v) = decl.get("description") {
                        func.insert("description".into(), v.clone());
                    }
                    if let Some(v) = schema(
                        decl,
                        ("parametersJsonSchema", "parameters_json_schema"),
                        ("parameters", "parameters"),
                    )? {
                        func.insert("parameters".into(), v);
                    }
                    declarations.push(json!({"type":"function","function":func}));
                }
            }
        }
        if !declarations.is_empty() {
            req["tools"] = declarations.into();
        }
    }
    if let Some(config) = field(body, "toolConfig", "tool_config")
        .and_then(|cfg| field(cfg, "functionCallingConfig", "function_calling_config"))
    {
        let names = field(config, "allowedFunctionNames", "allowed_function_names");
        let choice = match config.get("mode").and_then(Value::as_str).unwrap_or("AUTO") {
            "ANY" if names.is_some_and(|n| n.as_array().is_some_and(|a| a.len() == 1)) => {
                json!({"type":"function","function":{"name":names.unwrap()[0]}})
            }
            mode @ ("ANY" | "AUTO" | "VALIDATED")
                if names.is_some_and(|n| n.as_array().is_some_and(|a| !a.is_empty())) =>
            {
                let tools: Vec<_> = array(names.unwrap())?
                    .iter()
                    .map(|n| json!({"type":"function","function":{"name":n}}))
                    .collect();
                json!({"type":"allowed_tools","allowed_tools":{"mode":if mode=="ANY" {"required"} else {"auto"},"tools":tools}})
            }
            "ANY" => json!("required"),
            "AUTO" | "VALIDATED" | "MODE_UNSPECIFIED" => json!("auto"),
            "NONE" => json!("none"),
            _ => return Err(invalid("unknown functionCallingConfig.mode")),
        };
        req["tool_choice"] = choice;
    }
    let mut chat: ChatFormat = serde_json::from_value(req).map_err(|e| invalid(e.to_string()))?;
    chat.thoughts_requested = field(thinking, "includeThoughts", "include_thoughts")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    chat.thinking_budget_tokens =
        field(thinking, "thinkingBudget", "thinking_budget").and_then(Value::as_u64);
    chat.top_k = field(cfg, "topK", "top_k").and_then(Value::as_u64);
    Ok(chat)
}

pub(crate) fn redact(
    chain: &dyn aisix_guardrails::Guardrail,
    body: &mut Value,
) -> crate::redact::RedactionCounts {
    use crate::redact::{redact_value_strings, Direction, RedactionCounts};
    let mut counts = RedactionCounts::new();
    if !chain.redacts_input() {
        return counts;
    }
    let mut walk = |turn: &mut Value, dir| {
        let model_turn = turn["role"] == "model";
        let Some(parts) = turn["parts"].as_array_mut() else {
            return;
        };
        let has_media = parts.iter().any(|p| {
            field(p, "inlineData", "inline_data").is_some()
                || field(p, "fileData", "file_data").is_some()
        });
        // Match the Chat message channels: all-text content and assistant thoughts
        // are concatenated, while a multimodal message retains its typed blocks.
        for thought in [false, true] {
            let slots: Vec<_> = parts
                .iter_mut()
                .filter(|p| (model_turn && p["thought"] == true) == thought)
                .filter_map(|p| p.get_mut("text").filter(|v| v.is_string()))
                .collect();
            if slots.len() < 2 || (has_media && !thought) {
                for slot in slots {
                    redact_value_strings(chain, dir, slot, &mut counts);
                }
            } else {
                let mut joined = Value::String(slots.iter().filter_map(|s| s.as_str()).collect());
                let mut local = RedactionCounts::new();
                redact_value_strings(chain, dir, &mut joined, &mut local);
                if !local.is_empty() {
                    let mut remaining = joined.as_str().unwrap();
                    let last = slots.len() - 1;
                    for (i, slot) in slots.into_iter().enumerate() {
                        let boundary = if i == last {
                            remaining.len()
                        } else {
                            remaining.floor_char_boundary(
                                slot.as_str().unwrap().len().min(remaining.len()),
                            )
                        };
                        let (head, tail) = remaining.split_at(boundary);
                        *slot = Value::from(head);
                        remaining = tail;
                    }
                    crate::redact::merge_counts(&mut counts, local);
                }
            }
        }
        for part in parts {
            if let Some((key, child)) = execution_slot(part) {
                redact_value_strings(chain, dir, &mut part[key][child], &mut counts);
            }
            for (field, child) in [
                ("functionCall", "args"),
                ("function_call", "args"),
                ("functionResponse", "response"),
                ("function_response", "response"),
            ] {
                crate::redact::offer_tool_name(
                    chain,
                    dir,
                    part.get(field).and_then(|v| v.get("name")),
                );
                if let Some(v) = part.get_mut(field).and_then(|parent| parent.get_mut(child)) {
                    redact_value_strings(chain, dir, v, &mut counts);
                }
            }
        }
    };
    for key in ["systemInstruction", "system_instruction"] {
        if let Some(system) = body.get_mut(key).filter(|v| !v.is_null()) {
            walk(system, Direction::InputHistory);
        }
    }
    if let Some(contents) = body.get_mut("contents") {
        if let Some(turns) = contents.as_array_mut() {
            let end = turns.len().saturating_sub(1);
            let start = turns[..end]
                .iter()
                .rposition(|m| m["role"] == "model")
                .map_or(0, |i| i + 1);
            for (i, t) in turns.iter_mut().enumerate() {
                walk(
                    t,
                    if i < start {
                        Direction::InputHistory
                    } else {
                        Direction::Input
                    },
                );
            }
        } else {
            walk(contents, Direction::Input);
        }
    }
    counts
}

pub(crate) fn with_model_effort<'a>(
    body: &'a Value,
    model: &aisix_core::Model,
) -> std::borrow::Cow<'a, Value> {
    use aisix_core::EffortAction;
    use std::borrow::Cow;
    let key = |v: &Value, camel: &'static str, snake: &'static str| {
        if v.get(camel).is_none() && v.get(snake).is_some() {
            snake
        } else {
            camel
        }
    };
    let generation = key(body, "generationConfig", "generation_config");
    let thinking = key(&body[generation], "thinkingConfig", "thinking_config");
    let cfg = &body[generation][thinking];
    let level = key(cfg, "thinkingLevel", "thinking_level");
    if cfg.get(level).is_none() && field(cfg, "thinkingBudget", "thinking_budget").is_some() {
        return Cow::Borrowed(body);
    }
    let effort = cfg[level].as_str().map(str::to_ascii_lowercase);
    match model.mapped_effort(effort.as_deref()) {
        EffortAction::Keep => Cow::Borrowed(body),
        EffortAction::Set(v) if effort.as_deref() == Some(v) => Cow::Borrowed(body),
        EffortAction::Set(v) => {
            let mut changed = body.clone();
            changed[generation][thinking][level] = v.to_ascii_uppercase().into();
            Cow::Owned(changed)
        }
        EffortAction::Remove if cfg.get(level).is_some() => {
            let mut changed = body.clone();
            changed[generation][thinking]
                .as_object_mut()
                .unwrap()
                .remove(level);
            Cow::Owned(changed)
        }
        EffortAction::Remove => Cow::Borrowed(body),
    }
}
