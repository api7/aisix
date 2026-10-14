//! Facts the OTLP span encoder reports beside a [`crate::UsageEvent`], under
//! the OpenTelemetry GenAI and HTTP semantic conventions (semantic-conventions
//! v1.41.1, the last release that defines `gen_ai.*` in that repository).
//!
//! None of this is serialized: the usage event is the control plane's wire
//! contract, and the request parameters, the dispatched upstream and the
//! inbound HTTP request shape matter only to a trace backend. They ride the
//! exporter path as [`SpanFacts`] on the sink record instead.

use std::sync::Arc;

use serde_json::{json, Map, Value};

/// Export-only facts about one usage event's request. Every field is
/// optional: a family that never parses a GenAI request, or a request that
/// never resolved a target, simply reports less.
#[derive(Debug, Clone, Default)]
pub struct SpanFacts {
    /// The inbound HTTP request, for the SERVER span.
    pub http: Option<Arc<HttpServerFacts>>,
    /// The caller's GenAI request parameters, for the inference span.
    pub request: Option<Arc<GenAiRequestParams>>,
    /// The upstream this event's attempt was addressed to.
    pub upstream: Option<UpstreamFacts>,
}

/// The inbound HTTP request, as the HTTP server span conventions name it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpServerFacts {
    /// `http.request.method`: the method when it is one of the RFC 9110 /
    /// RFC 5789 methods, otherwise `_OTHER` (with the original kept in
    /// `method_original`).
    pub method: &'static str,
    pub method_original: Option<String>,
    /// `url.path`, capped.
    pub path: String,
    /// `http.route`: only when the path IS a static route the gateway
    /// serves. A prefix-matched or parameterised path has no template the
    /// router reports, and the conventions forbid substituting the path.
    pub route: Option<&'static str>,
    /// `url.scheme` of the hop that reached the gateway.
    pub scheme: Option<&'static str>,
}

/// Longest `url.path` exported. Paths are caller-chosen; a trace backend
/// stores the value on every span of the request's root.
const MAX_URL_PATH_BYTES: usize = 512;

impl HttpServerFacts {
    pub fn new(
        method: &str,
        path: &str,
        route: Option<&'static str>,
        scheme: Option<&'static str>,
    ) -> Self {
        const KNOWN: [&str; 10] = [
            "CONNECT", "DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "QUERY", "TRACE",
        ];
        let known = KNOWN.iter().copied().find(|m| *m == method);
        Self {
            method: known.unwrap_or("_OTHER"),
            method_original: known.is_none().then(|| capped(method, 64).to_string()),
            path: capped(path, MAX_URL_PATH_BYTES).to_string(),
            route,
            scheme,
        }
    }

    /// The SERVER span name: `{method} {route}`, `{method}` without a
    /// route, and `HTTP` in place of an unknown method.
    pub fn span_name(&self) -> String {
        let method = if self.method == "_OTHER" {
            "HTTP"
        } else {
            self.method
        };
        match self.route {
            Some(route) => format!("{method} {route}"),
            None => method.to_string(),
        }
    }
}

/// The upstream one attempt was addressed to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpstreamFacts {
    /// `gen_ai.provider.name`, already mapped onto the conventions'
    /// well-known values (see [`genai_provider_name`]). Empty when the
    /// upstream is not a GenAI provider the gateway can name.
    pub provider_name: String,
    /// `gen_ai.request.model`: the model name sent to the upstream.
    pub request_model: String,
    /// `server.address` / `server.port` of the upstream.
    pub server_address: String,
    pub server_port: Option<u16>,
    /// `request_model` was filled from the caller's own alias (a wildcard
    /// row), so it is caller-chosen cardinality: the span name leaves it
    /// out, the attribute keeps it.
    pub request_model_caller_named: bool,
}

impl UpstreamFacts {
    /// Fill `server.address` / `server.port` from an upstream base URL.
    /// The port is the explicit one, else the scheme's default; a URL that
    /// does not parse leaves both unset rather than guessing.
    pub fn with_base_url(mut self, base: &str) -> Self {
        if let Ok(url) = reqwest::Url::parse(base.trim()) {
            if let Some(host) = url.host_str() {
                self.server_address = host.trim_matches(['[', ']']).to_string();
                self.server_port = url.port_or_known_default();
            }
        }
        self
    }
}

/// `gen_ai.provider.name` for a dispatched upstream.
///
/// The value is the conventions' discriminator for which provider's
/// telemetry flavour a span follows, so a platform that hosts other vendors'
/// models wins over the vendor: Claude reached through Bedrock is
/// `aws.bedrock`. Otherwise the catalog vendor id maps onto the well-known
/// value it names, and a vendor the conventions do not list keeps its own id,
/// which they allow as a custom value.
pub fn genai_provider_name(vendor: &str, adapter: Option<aisix_core::Adapter>) -> String {
    use aisix_core::Adapter;
    match adapter {
        Some(Adapter::Bedrock) => return "aws.bedrock".into(),
        Some(Adapter::Vertex) => return "gcp.vertex_ai".into(),
        Some(Adapter::AzureOpenai) => return "azure.ai.openai".into(),
        Some(Adapter::Openai) | Some(Adapter::Anthropic) | None => {}
    }
    let vendor = vendor.trim().to_ascii_lowercase();
    let known = match vendor.as_str() {
        "openai" => "openai",
        "anthropic" => "anthropic",
        "amazon-bedrock" | "bedrock" => "aws.bedrock",
        "azure" | "azure-openai" | "azure_openai" => "azure.ai.openai",
        "google" | "gemini" => "gcp.gemini",
        "google-vertex" | "vertex" | "vertex_ai" => "gcp.vertex_ai",
        "cohere" => "cohere",
        "deepseek" => "deepseek",
        "groq" => "groq",
        "mistral" | "mistralai" | "mistral_ai" => "mistral_ai",
        "perplexity" => "perplexity",
        "xai" | "x_ai" => "x_ai",
        "ibm" | "watsonx" | "ibm.watsonx.ai" => "ibm.watsonx.ai",
        // A legacy row with no vendor is dispatched by its adapter family.
        "" => match adapter {
            Some(Adapter::Openai) => "openai",
            Some(Adapter::Anthropic) => "anthropic",
            _ => return String::new(),
        },
        _ => return vendor,
    };
    known.to_string()
}

/// The caller's GenAI request parameters, read from the inbound body in the
/// protocol it was written in.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenAiRequestParams {
    pub max_tokens: Option<i64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<f64>,
    pub stop_sequences: Vec<String>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub seed: Option<i64>,
    /// `n`, reported only when it is not 1 (the conventions' condition).
    pub choice_count: Option<i64>,
    pub stream: bool,
    /// `gen_ai.output.type`, only when the request names an output format.
    pub output_type: Option<&'static str>,
    pub encoding_formats: Vec<String>,
    pub embedding_dimensions: Option<i64>,
    /// `openai.request.service_tier`, when the request set one other than
    /// `auto`. Reported only on spans whose provider is `openai`.
    pub openai_service_tier: Option<String>,
}

fn num(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64)
}

fn int(v: Option<&Value>) -> Option<i64> {
    v.and_then(Value::as_i64)
}

fn strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|i| i.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

fn service_tier(body: &Map<String, Value>) -> Option<String> {
    body.get("service_tier")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty() && *t != "auto")
        .map(|t| capped(t, 64).to_string())
}

/// The OpenAI-wire `response_format` / Responses `text.format` type, read
/// as the conventions' output modality.
fn output_type_of(format_type: Option<&str>) -> Option<&'static str> {
    match format_type? {
        "json_object" | "json_schema" => Some("json"),
        "text" => Some("text"),
        _ => None,
    }
}

impl GenAiRequestParams {
    /// `/v1/chat/completions` (OpenAI Chat Completions).
    pub fn from_openai_chat(body: &Map<String, Value>) -> Self {
        Self {
            max_tokens: int(body.get("max_completion_tokens")).or(int(body.get("max_tokens"))),
            temperature: num(body.get("temperature")),
            top_p: num(body.get("top_p")),
            top_k: num(body.get("top_k")),
            stop_sequences: strings(body.get("stop")),
            frequency_penalty: num(body.get("frequency_penalty")),
            presence_penalty: num(body.get("presence_penalty")),
            seed: int(body.get("seed")),
            choice_count: int(body.get("n")).filter(|n| *n != 1),
            stream: body.get("stream").and_then(Value::as_bool) == Some(true),
            output_type: output_type_of(
                body.get("response_format")
                    .and_then(|f| f.get("type"))
                    .and_then(Value::as_str),
            ),
            openai_service_tier: service_tier(body),
            ..Self::default()
        }
    }

    /// `/v1/completions` (OpenAI legacy Completions).
    pub fn from_openai_completions(body: &Map<String, Value>) -> Self {
        Self {
            max_tokens: int(body.get("max_tokens")),
            temperature: num(body.get("temperature")),
            top_p: num(body.get("top_p")),
            stop_sequences: strings(body.get("stop")),
            frequency_penalty: num(body.get("frequency_penalty")),
            presence_penalty: num(body.get("presence_penalty")),
            seed: int(body.get("seed")),
            choice_count: int(body.get("n")).filter(|n| *n != 1),
            stream: body.get("stream").and_then(Value::as_bool) == Some(true),
            openai_service_tier: service_tier(body),
            ..Self::default()
        }
    }

    /// `/v1/messages` (Anthropic Messages).
    pub fn from_anthropic_messages(body: &Map<String, Value>) -> Self {
        Self {
            max_tokens: int(body.get("max_tokens")),
            temperature: num(body.get("temperature")),
            top_p: num(body.get("top_p")),
            top_k: num(body.get("top_k")),
            stop_sequences: strings(body.get("stop_sequences")),
            stream: body.get("stream").and_then(Value::as_bool) == Some(true),
            ..Self::default()
        }
    }

    /// `/v1/responses` (OpenAI Responses).
    pub fn from_openai_responses(body: &Map<String, Value>) -> Self {
        Self {
            max_tokens: int(body.get("max_output_tokens")),
            temperature: num(body.get("temperature")),
            top_p: num(body.get("top_p")),
            stream: body.get("stream").and_then(Value::as_bool) == Some(true),
            output_type: output_type_of(
                body.get("text")
                    .and_then(|t| t.get("format"))
                    .and_then(|f| f.get("type"))
                    .and_then(Value::as_str),
            ),
            openai_service_tier: service_tier(body),
            ..Self::default()
        }
    }
}

fn capped(value: &str, max_bytes: usize) -> &str {
    let mut end = max_bytes.min(value.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

/// Captured content converted to the conventions' structured attributes.
/// Each is the JSON-string form the conventions allow on spans.
#[derive(Debug, Default, PartialEq)]
pub struct StructuredContent {
    pub input_messages: Option<String>,
    pub system_instructions: Option<String>,
    pub tool_definitions: Option<String>,
    pub output_messages: Option<String>,
}

/// Which inbound wire a captured prompt was written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptWire {
    OpenAiChat,
    AnthropicMessages,
    OpenAiResponses,
    OpenAiCompletions,
}

impl PromptWire {
    /// The wire of an event's captured prompt, from the operation the
    /// gateway stamped on it. `None` for a payload that is not a message
    /// list (embeddings, images, audio, jobs, tools, agents).
    pub fn for_operation(operation: &str) -> Option<Self> {
        match operation {
            "chat" => Some(Self::OpenAiChat),
            "messages" => Some(Self::AnthropicMessages),
            "responses" => Some(Self::OpenAiResponses),
            "completions" => Some(Self::OpenAiCompletions),
            _ => None,
        }
    }
}

/// Convert a captured inbound request body and the assembled response text
/// into the structured content attributes. A prompt that does not parse as
/// the expected wire yields no input attributes at all — never a partial or
/// malformed list. The output message needs the finish reason the schema
/// requires, so it is omitted when the event carries none.
pub fn structured_content(
    wire: PromptWire,
    prompt: &str,
    response: &str,
    finish_reason: &str,
) -> StructuredContent {
    let mut out = StructuredContent::default();
    if let Ok(Value::Object(body)) = serde_json::from_str::<Value>(prompt) {
        let converted = match wire {
            PromptWire::OpenAiChat => openai_chat_input(&body),
            PromptWire::AnthropicMessages => anthropic_input(&body),
            PromptWire::OpenAiResponses => responses_input(&body),
            PromptWire::OpenAiCompletions => completions_input(&body),
        };
        if let Some((messages, system, tools)) = converted {
            out.input_messages = Some(Value::Array(messages).to_string());
            out.system_instructions = system.map(|s| Value::Array(s).to_string());
            out.tool_definitions = tools
                .filter(|t| !t.is_empty())
                .map(|t| Value::Array(t).to_string());
        }
    }
    if !response.is_empty() && !finish_reason.is_empty() {
        out.output_messages = Some(
            json!([{
                "role": "assistant",
                "parts": [text_part(response)],
                "finish_reason": finish_reason,
            }])
            .to_string(),
        );
    }
    out
}

type Converted = (Vec<Value>, Option<Vec<Value>>, Option<Vec<Value>>);

fn text_part(text: &str) -> Value {
    json!({"type": "text", "content": text})
}

/// `data:<mime>;base64,<payload>` → (mime, payload).
fn data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, payload) = rest.split_once(',')?;
    let mime = meta.strip_suffix(";base64")?;
    Some((mime, payload))
}

fn media_part(url: &str, modality: &str) -> Value {
    match data_url(url) {
        Some((mime, payload)) => json!({
            "type": "blob", "modality": modality, "mime_type": mime, "content": payload,
        }),
        None => json!({"type": "uri", "modality": modality, "uri": url}),
    }
}

/// Tool-call arguments as the object the conventions expect, deserialized
/// from the wire's JSON string when it is one.
fn arguments(raw: Option<&Value>) -> Value {
    match raw {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
        }
        Some(other) => other.clone(),
        None => Value::Null,
    }
}

/// A part the conventions have no specific shape for: the wire's own block,
/// which already carries the `type` a generic part requires.
fn generic_part(block: &Value) -> Option<Value> {
    block
        .get("type")
        .and_then(Value::as_str)
        .map(|_| block.clone())
}

fn openai_content_parts(content: Option<&Value>) -> Option<Vec<Value>> {
    match content {
        None | Some(Value::Null) => Some(Vec::new()),
        Some(Value::String(s)) => Some(vec![text_part(s)]),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") | Some("input_text") | Some("output_text") => {
                    p.get("text").and_then(Value::as_str).map(text_part)
                }
                Some("image_url") => p
                    .get("image_url")
                    .and_then(|i| i.get("url").or(Some(i)))
                    .and_then(Value::as_str)
                    .map(|u| media_part(u, "image")),
                Some("input_image") => match p.get("image_url").and_then(Value::as_str) {
                    Some(u) => Some(media_part(u, "image")),
                    None => p
                        .get("file_id")
                        .and_then(Value::as_str)
                        .map(|id| json!({"type": "file", "modality": "image", "file_id": id})),
                },
                Some("input_audio") => {
                    let audio = p.get("input_audio")?;
                    let data = audio.get("data").and_then(Value::as_str)?;
                    let mut part = json!({"type": "blob", "modality": "audio", "content": data});
                    if let Some(format) = audio.get("format").and_then(Value::as_str) {
                        part["mime_type"] = Value::String(format!("audio/{format}"));
                    }
                    Some(part)
                }
                _ => generic_part(p),
            })
            .collect(),
        Some(_) => None,
    }
}

fn openai_tools(tools: Option<&Value>) -> Option<Vec<Value>> {
    let tools = tools?.as_array()?;
    Some(
        tools
            .iter()
            .filter_map(|t| {
                let kind = t.get("type").and_then(Value::as_str)?;
                // Chat nests the function; Responses flattens it.
                let name = t
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .or_else(|| t.get("name"))
                    .and_then(Value::as_str)?;
                Some(json!({"type": kind, "name": name}))
            })
            .collect(),
    )
}

fn openai_chat_input(body: &Map<String, Value>) -> Option<Converted> {
    let messages = body.get("messages")?.as_array()?;
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role = m.get("role").and_then(Value::as_str)?;
        let mut parts = Vec::new();
        if role == "tool" {
            let mut part = json!({
                "type": "tool_call_response",
                "response": m.get("content").cloned().unwrap_or(Value::Null),
            });
            if let Some(id) = m.get("tool_call_id").and_then(Value::as_str) {
                part["id"] = Value::String(id.to_string());
            }
            parts.push(part);
        } else {
            if let Some(reasoning) = m.get("reasoning_content").and_then(Value::as_str) {
                parts.push(json!({"type": "reasoning", "content": reasoning}));
            }
            parts.extend(openai_content_parts(m.get("content"))?);
            for call in m
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let function = call.get("function");
                let name = function
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)?;
                let mut part = json!({
                    "type": "tool_call",
                    "name": name,
                    "arguments": arguments(function.and_then(|f| f.get("arguments"))),
                });
                if let Some(id) = call.get("id").and_then(Value::as_str) {
                    part["id"] = Value::String(id.to_string());
                }
                parts.push(part);
            }
        }
        let mut message = json!({"role": role, "parts": parts});
        if let Some(name) = m.get("name").and_then(Value::as_str) {
            message["name"] = Value::String(name.to_string());
        }
        out.push(message);
    }
    // Chat carries its system prompt inside the message list, which is
    // where the conventions want history-borne instructions recorded.
    Some((out, None, openai_tools(body.get("tools"))))
}

fn anthropic_blocks(content: Option<&Value>) -> Option<Vec<Value>> {
    match content {
        Some(Value::String(s)) => Some(vec![text_part(s)]),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .map(|b| match b.get("type").and_then(Value::as_str) {
                Some("text") => b.get("text").and_then(Value::as_str).map(text_part),
                Some("thinking") => b
                    .get("thinking")
                    .and_then(Value::as_str)
                    .map(|t| json!({"type": "reasoning", "content": t})),
                Some("image") | Some("document") => {
                    let modality = if b["type"] == "image" {
                        "image"
                    } else {
                        "document"
                    };
                    let source = b.get("source")?;
                    match source.get("type").and_then(Value::as_str) {
                        Some("base64") => Some(json!({
                            "type": "blob",
                            "modality": modality,
                            "mime_type": source.get("media_type").cloned().unwrap_or(Value::Null),
                            "content": source.get("data").and_then(Value::as_str)?,
                        })),
                        Some("url") => Some(json!({
                            "type": "uri",
                            "modality": modality,
                            "uri": source.get("url").and_then(Value::as_str)?,
                        })),
                        _ => generic_part(b),
                    }
                }
                Some("tool_use") => {
                    let mut part = json!({
                        "type": "tool_call",
                        "name": b.get("name").and_then(Value::as_str)?,
                        "arguments": b.get("input").cloned().unwrap_or(Value::Null),
                    });
                    if let Some(id) = b.get("id").and_then(Value::as_str) {
                        part["id"] = Value::String(id.to_string());
                    }
                    Some(part)
                }
                Some("tool_result") => {
                    let mut part = json!({
                        "type": "tool_call_response",
                        "response": b.get("content").cloned().unwrap_or(Value::Null),
                    });
                    if let Some(id) = b.get("tool_use_id").and_then(Value::as_str) {
                        part["id"] = Value::String(id.to_string());
                    }
                    Some(part)
                }
                _ => generic_part(b),
            })
            .collect(),
        _ => None,
    }
}

fn anthropic_input(body: &Map<String, Value>) -> Option<Converted> {
    let messages = body.get("messages")?.as_array()?;
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role = m.get("role").and_then(Value::as_str)?;
        out.push(json!({"role": role, "parts": anthropic_blocks(m.get("content"))?}));
    }
    let system = match body.get("system") {
        None | Some(Value::Null) => None,
        Some(system) => Some(anthropic_blocks(Some(system))?),
    };
    let tools = body.get("tools").and_then(Value::as_array).map(|tools| {
        tools
            .iter()
            .filter_map(|t| {
                let name = t.get("name").and_then(Value::as_str)?;
                // A client tool carries no `type` (or `custom`); a server
                // tool names its versioned kind.
                let kind = match t.get("type").and_then(Value::as_str) {
                    None | Some("custom") => "function",
                    Some(kind) => kind,
                };
                Some(json!({"type": kind, "name": name}))
            })
            .collect()
    });
    Some((out, system, tools))
}

fn responses_input(body: &Map<String, Value>) -> Option<Converted> {
    let messages = match body.get("input")? {
        Value::String(s) => vec![json!({"role": "user", "parts": [text_part(s)]})],
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let kind = item.get("type").and_then(Value::as_str);
                match kind {
                    Some("function_call") => {
                        let mut part = json!({
                            "type": "tool_call",
                            "name": item.get("name").and_then(Value::as_str)?,
                            "arguments": arguments(item.get("arguments")),
                        });
                        if let Some(id) = item.get("call_id").and_then(Value::as_str) {
                            part["id"] = Value::String(id.to_string());
                        }
                        out.push(json!({"role": "assistant", "parts": [part]}));
                    }
                    Some("function_call_output") => {
                        let mut part = json!({
                            "type": "tool_call_response",
                            "response": item.get("output").cloned().unwrap_or(Value::Null),
                        });
                        if let Some(id) = item.get("call_id").and_then(Value::as_str) {
                            part["id"] = Value::String(id.to_string());
                        }
                        out.push(json!({"role": "tool", "parts": [part]}));
                    }
                    None | Some("message") => {
                        let role = item.get("role").and_then(Value::as_str)?;
                        out.push(json!({
                            "role": role,
                            "parts": openai_content_parts(item.get("content"))?,
                        }));
                    }
                    // A reference names an earlier item without saying
                    // whose it was: no role can be claimed for it.
                    Some("item_reference") => return None,
                    // Reasoning items and server-side tool calls replayed
                    // from an earlier turn are the model's own output: the
                    // item itself, as a generic part of an assistant turn.
                    Some(_) => out.push(json!({"role": "assistant", "parts": [item.clone()]})),
                }
            }
            out
        }
        _ => return None,
    };
    let system = body
        .get("instructions")
        .and_then(Value::as_str)
        .map(|s| vec![text_part(s)]);
    Some((messages, system, openai_tools(body.get("tools"))))
}

fn completions_input(body: &Map<String, Value>) -> Option<Converted> {
    // A prompt array is a batch of independent prompts, not one history.
    let prompt = body.get("prompt")?.as_str()?;
    Some((
        vec![json!({"role": "user", "parts": [text_part(prompt)]})],
        None,
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn provider_name_prefers_the_hosting_platform_then_the_vendor() {
        use aisix_core::Adapter;
        assert_eq!(
            genai_provider_name("openai", Some(Adapter::Openai)),
            "openai"
        );
        assert_eq!(
            genai_provider_name("anthropic", Some(Adapter::Bedrock)),
            "aws.bedrock"
        );
        assert_eq!(genai_provider_name("google-vertex", None), "gcp.vertex_ai");
        assert_eq!(genai_provider_name("xai", Some(Adapter::Openai)), "x_ai");
        assert_eq!(
            genai_provider_name("wafer.ai", Some(Adapter::Openai)),
            "wafer.ai"
        );
        assert_eq!(
            genai_provider_name("", Some(Adapter::Anthropic)),
            "anthropic"
        );
        assert_eq!(genai_provider_name("", None), "");
    }

    #[test]
    fn chat_params_read_the_openai_names() {
        let p = GenAiRequestParams::from_openai_chat(&obj(json!({
            "max_completion_tokens": 50, "max_tokens": 10, "temperature": 0.2,
            "top_p": 0.9, "stop": "END", "n": 2, "seed": 7, "stream": true,
            "frequency_penalty": 0.1, "presence_penalty": 0.3,
            "response_format": {"type": "json_schema"}, "service_tier": "flex",
        })));
        assert_eq!(p.max_tokens, Some(50));
        assert_eq!(p.stop_sequences, vec!["END"]);
        assert_eq!(p.choice_count, Some(2));
        assert_eq!(p.output_type, Some("json"));
        assert!(p.stream);
        assert_eq!(p.openai_service_tier.as_deref(), Some("flex"));
        let single =
            GenAiRequestParams::from_openai_chat(&obj(json!({"n": 1, "service_tier": "auto"})));
        assert_eq!(single.choice_count, None);
        assert_eq!(single.openai_service_tier, None);
    }

    #[test]
    fn responses_and_completions_params_read_their_own_names() {
        let r = GenAiRequestParams::from_openai_responses(&obj(json!({
            "max_output_tokens": 300, "temperature": 1.0, "stream": true,
            "text": {"format": {"type": "json_schema"}},
        })));
        assert_eq!(r.max_tokens, Some(300));
        assert_eq!(r.output_type, Some("json"));
        assert!(r.stream);
        let c = GenAiRequestParams::from_openai_completions(&obj(json!({
            "max_tokens": 16, "stop": ["\n", "END"], "n": 3, "seed": 1,
        })));
        assert_eq!(c.max_tokens, Some(16));
        assert_eq!(c.stop_sequences, vec!["\n", "END"]);
        assert_eq!(c.choice_count, Some(3));
        assert!(!c.stream);
    }

    #[test]
    fn responses_content_maps_instructions_and_function_items() {
        let body = json!({
            "instructions": "be kind",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                {"type": "function_call", "call_id": "c1", "name": "w", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "sunny"}
            ],
            "tools": [{"type": "function", "name": "w"}, {"type": "web_search"}]
        });
        let c = structured_content(PromptWire::OpenAiResponses, &body.to_string(), "", "");
        let input: Value = serde_json::from_str(c.input_messages.as_deref().unwrap()).unwrap();
        assert_eq!(input[0]["parts"][0]["content"], "weather?");
        assert_eq!(input[1]["parts"][0]["type"], "tool_call");
        assert_eq!(input[2]["role"], "tool");
        assert_eq!(
            c.system_instructions.as_deref(),
            Some(r#"[{"content":"be kind","type":"text"}]"#)
        );
        // A tool with no name cannot be a valid definition; it is left out.
        assert_eq!(
            c.tool_definitions.as_deref(),
            Some(r#"[{"name":"w","type":"function"}]"#)
        );
    }

    #[test]
    fn server_facts_follow_the_http_naming_rules() {
        let known =
            HttpServerFacts::new("POST", "/v1/messages", Some("/v1/messages"), Some("http"));
        assert_eq!(known.span_name(), "POST /v1/messages");
        let unrouted = HttpServerFacts::new("GET", "/passthrough/x/y", None, None);
        assert_eq!(unrouted.span_name(), "GET");
        let odd = HttpServerFacts::new("BREW", "/", None, None);
        assert_eq!(odd.method, "_OTHER");
        assert_eq!(odd.method_original.as_deref(), Some("BREW"));
        assert_eq!(odd.span_name(), "HTTP");
    }

    #[test]
    fn upstream_address_comes_from_the_base_url() {
        let f = UpstreamFacts::default().with_base_url("https://api.example.com/v1");
        assert_eq!(f.server_address, "api.example.com");
        assert_eq!(f.server_port, Some(443));
        let f = UpstreamFacts::default().with_base_url("http://[::1]:8080");
        assert_eq!(f.server_address, "::1");
        assert_eq!(f.server_port, Some(8080));
        let f = UpstreamFacts::default().with_base_url("not a url");
        assert_eq!(f.server_address, "");
        assert_eq!(f.server_port, None);
    }

    #[test]
    fn anthropic_content_maps_system_tools_and_tool_blocks() {
        let body = json!({
            "system": "be brief",
            "tools": [{"name": "get_weather", "input_schema": {}}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "rainy"}
                ]}
            ]
        });
        let c = structured_content(
            PromptWire::AnthropicMessages,
            &body.to_string(),
            "It rains.",
            "end_turn",
        );
        let input: Value = serde_json::from_str(c.input_messages.as_deref().unwrap()).unwrap();
        assert_eq!(input[1]["parts"][0]["type"], "tool_call");
        assert_eq!(input[1]["parts"][0]["arguments"]["city"], "Paris");
        assert_eq!(input[2]["parts"][0]["type"], "tool_call_response");
        assert_eq!(
            c.system_instructions.as_deref(),
            Some(r#"[{"content":"be brief","type":"text"}]"#)
        );
        assert_eq!(
            c.tool_definitions.as_deref(),
            Some(r#"[{"name":"get_weather","type":"function"}]"#)
        );
        let output: Value = serde_json::from_str(c.output_messages.as_deref().unwrap()).unwrap();
        assert_eq!(output[0]["finish_reason"], "end_turn");
    }

    #[test]
    fn unparseable_prompt_or_missing_finish_reason_yields_nothing() {
        let c = structured_content(PromptWire::OpenAiChat, "{\"messages\": [", "x", "");
        assert_eq!(c, StructuredContent::default());
    }

    #[test]
    fn openai_chat_tool_call_arguments_are_deserialized() {
        let body = json!({"messages": [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}
        ]});
        let c = structured_content(PromptWire::OpenAiChat, &body.to_string(), "", "");
        let input: Value = serde_json::from_str(c.input_messages.as_deref().unwrap()).unwrap();
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[1]["parts"][0]["arguments"]["a"], 1);
        assert_eq!(input[2]["role"], "tool");
        assert_eq!(input[3]["parts"][1]["type"], "blob");
        assert_eq!(input[3]["parts"][1]["mime_type"], "image/png");
        assert!(c.output_messages.is_none());
    }
}
