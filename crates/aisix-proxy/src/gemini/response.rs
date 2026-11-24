use std::collections::BTreeMap;

use serde_json::{json, Value};

fn usage(u: &Value) -> Value {
    let mut meta = json!({});
    for (path, name) in [
        ("/prompt_tokens", "promptTokenCount"),
        ("/completion_tokens", "candidatesTokenCount"),
        ("/total_tokens", "totalTokenCount"),
        (
            "/prompt_tokens_details/cached_tokens",
            "cachedContentTokenCount",
        ),
        (
            "/completion_tokens_details/reasoning_tokens",
            "thoughtsTokenCount",
        ),
    ] {
        if let Some(v) = u.pointer(path) {
            meta[name] = v.clone();
        }
    }
    if let (Some(all), Some(thoughts)) = (
        meta["candidatesTokenCount"].as_u64(),
        meta["thoughtsTokenCount"].as_u64(),
    ) {
        meta["candidatesTokenCount"] = all.saturating_sub(thoughts).into();
    }
    meta
}

fn envelope(chat: &Value, candidates: Vec<Value>) -> Value {
    let mut result = json!({"candidates":candidates});
    if let Some(id) = chat.get("id") {
        result["responseId"] = id.clone();
    }
    if let Some(model) = chat.get("model") {
        result["modelVersion"] = model.clone();
    }
    if let Some(u) = chat.get("usage").filter(|v| !v.is_null()) {
        result["usageMetadata"] = usage(u);
    }
    result
}

fn finish(reason: &str) -> &str {
    match reason {
        "length" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        "stop" | "tool_calls" | "function_call" => "STOP",
        _ => "OTHER",
    }
}

fn candidate(choice: &Value, parts: Vec<Value>, malformed: bool) -> Value {
    let mut candidate = json!({"index":choice["index"],"content":{"role":"model","parts":parts}});
    if malformed {
        candidate["finishReason"] = "MALFORMED_FUNCTION_CALL".into();
    } else if let Some(reason) = choice["finish_reason"].as_str() {
        candidate["finishReason"] = finish(reason).into();
    }
    if let Some(tokens) = choice
        .pointer("/logprobs/content")
        .and_then(Value::as_array)
    {
        let token = |v: &Value| json!({"token":v["token"],"logProbability":v["logprob"]});
        let chosen: Vec<_> = tokens.iter().map(token).collect();
        let top: Vec<_> = tokens
            .iter()
            .map(|v| {
                let candidates: Vec<_> = v["top_logprobs"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(token)
                    .collect();
                json!({"candidates":candidates})
            })
            .collect();
        candidate["logprobsResult"] = json!({"chosenCandidates":chosen,"topCandidates":top});
    }
    candidate
}

fn image_part(block: &Value) -> Option<Value> {
    use aisix_gateway::content_part::{parse_part, ContentPart, MediaSource};
    if block["type"] != "image_url" {
        return None;
    }
    let ContentPart::Media(media) = parse_part(block).ok()? else {
        return None;
    };
    Some(match media.source {
        MediaSource::Inline { data, mime_type } => {
            json!({"inlineData":{"mimeType":mime_type,"data":data}})
        }
        MediaSource::Uri { uri, mime_type, .. } => {
            let mut file = json!({"fileUri":uri});
            if let Some(mime) = mime_type {
                file["mimeType"] = mime.into();
            }
            json!({"fileData":file})
        }
    })
}

fn message_parts(msg: &Value, requested_audio_mime: Option<&str>) -> Vec<Value> {
    let mut parts = Vec::new();
    if let Some(text) = msg["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
        parts.push(json!({"text":text,"thought":true}));
    }
    if let Some(text) = msg["content"].as_str().filter(|s| !s.is_empty()) {
        parts.push(json!({"text":text}));
    }
    if let Some(blocks) = msg["content"].as_array() {
        for block in blocks {
            if let Some(image) = image_part(block) {
                parts.push(image);
            } else if let Some(text) = block["text"].as_str() {
                parts.push(json!({"text":text}));
            }
        }
    }
    if let Some(images) = msg["images"].as_array() {
        parts.extend(images.iter().filter_map(image_part));
    }
    if let Some(text) = msg["refusal"].as_str() {
        parts.push(json!({"text":text}));
    }
    let mime = match msg.pointer("/audio/format").and_then(Value::as_str) {
        Some(format) => aisix_gateway::chat::audio_mime_type(format)
            .or_else(|| format.starts_with("audio/").then_some(format)),
        None => requested_audio_mime,
    };
    if let (Some(data), Some(mime)) = (msg.pointer("/audio/data").and_then(Value::as_str), mime) {
        parts.push(json!({"inlineData":{"mimeType":mime,"data":data}}));
    }
    parts
}

#[derive(Default)]
struct FunctionCall {
    id: String,
    name: String,
    arguments: String,
    signature: Option<Value>,
}

impl FunctionCall {
    fn append(&mut self, call: &Value) {
        let repeated_id = call["id"].as_str() == Some(&self.id);
        if let Some(s) = call["id"].as_str() {
            if !repeated_id {
                self.id.push_str(s);
            }
        }
        if let Some(s) = call.pointer("/function/name").and_then(Value::as_str) {
            if !repeated_id || self.name != s {
                self.name.push_str(s);
            }
        }
        if let Some(s) = call.pointer("/function/arguments").and_then(Value::as_str) {
            self.arguments.push_str(s);
        }
        if let Some(s) = call
            .pointer("/extra_content/google/thought_signature")
            .or_else(|| call.pointer("/provider_specific_fields/thought_signature"))
        {
            self.signature = Some(s.clone());
        }
    }

    fn part(self) -> Option<Value> {
        if self.name.is_empty() {
            return None;
        }
        let args = if self.arguments.is_empty() {
            json!({})
        } else {
            serde_json::from_str(&self.arguments).ok()?
        };
        if !args.is_object() {
            return None;
        }
        let mut part = json!({"functionCall":{"name":self.name,"args":args}});
        if !self.id.is_empty() {
            part["functionCall"]["id"] = self.id.into();
        }
        if let Some(s) = self.signature {
            part["thoughtSignature"] = s;
        }
        Some(part)
    }
}

pub(super) fn unary(chat: Value, audio_mime: Option<&str>) -> Value {
    let mut candidates = Vec::new();
    for choice in chat["choices"].as_array().into_iter().flatten() {
        let msg = &choice["message"];
        let mut parts = message_parts(msg, audio_mime);
        let mut malformed = false;
        for call in msg["tool_calls"].as_array().into_iter().flatten() {
            let mut function = FunctionCall::default();
            function.append(call);
            match function.part() {
                Some(part) => parts.push(part),
                None => malformed = true,
            }
        }
        candidates.push(candidate(choice, parts, malformed));
    }
    envelope(&chat, candidates)
}

#[derive(Default)]
pub(super) struct Stream {
    calls: BTreeMap<u64, BTreeMap<u64, FunctionCall>>,
    pending_finish: Option<Value>,
}

impl Stream {
    pub(super) fn chunk(&mut self, chat: Value, audio_mime: Option<&str>) -> Option<Value> {
        let mut candidates = Vec::new();
        for choice in chat["choices"].as_array().into_iter().flatten() {
            let delta = &choice["delta"];
            let mut parts = message_parts(delta, audio_mime);
            let index = choice["index"].as_u64().unwrap_or(0);
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let calls = self.calls.entry(index).or_default();
                calls
                    .entry(call["index"].as_u64().unwrap_or(0))
                    .or_default()
                    .append(call);
            }
            let mut malformed = false;
            if choice["finish_reason"].is_string() {
                for (_, call) in self.calls.remove(&index).unwrap_or_default() {
                    match call.part() {
                        Some(p) => parts.push(p),
                        None => malformed = true,
                    }
                }
            }
            if !parts.is_empty() || choice["finish_reason"].is_string() {
                candidates.push(candidate(choice, parts, malformed));
            }
        }
        if candidates.is_empty() && chat["usage"].is_null() {
            return None;
        }
        if candidates.is_empty() {
            if let Some(mut finish) = self.pending_finish.take() {
                finish["usageMetadata"] = usage(&chat["usage"]);
                return Some(finish);
            }
        }
        let terminal =
            !candidates.is_empty() && candidates.iter().all(|v| v["finishReason"].is_string());
        let frame = envelope(&chat, candidates);
        if terminal && chat["usage"].is_null() {
            self.pending_finish.replace(frame)
        } else {
            Some(frame)
        }
    }

    pub(super) fn finish(&mut self) -> Option<Value> {
        self.pending_finish.take()
    }
}
