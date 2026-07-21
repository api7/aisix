//! Local token estimation for usage telemetry (AISIX-Cloud#1074).
//!
//! When an upstream response — streaming or not — carries no `usage`
//! block, the emit paths fall back to counting tokens locally so usage
//! events, post-stream TPM/TPD accounting, and metrics record real
//! numbers instead of silent zeros. Estimation fills **telemetry only**:
//! the client-visible response body is never rewritten, and upstream
//! values always win when non-zero (per-field `or` semantics, matching
//! the reference stream-rebuild implementations). Any event that carries
//! an estimated component sets `UsageEvent::usage_estimated`.
//!
//! Counting follows the OpenAI cookbook message scheme: 3 tokens per
//! message, +1 per `name`, +3 reply priming, tool definitions rendered
//! to the de-facto TypeScript-ish form (+9, −4 when a system message is
//! present). The completion side counts the accumulated output text as
//! plain text with no overhead.
//!
//! Encoding selection mirrors tiktoken's model map (`gpt-4o`/`o1`/... →
//! `o200k_base`, `gpt-4`/`gpt-3.5` → `cl100k_base`) and falls back to
//! `cl100k_base` for unknown/non-OpenAI models. Counts for models with
//! proprietary tokenizers (Claude, Gemini, ...) are therefore
//! approximations — the `usage_estimated` flag exists so consumers can
//! tell these numbers from upstream-reported ones.

use serde_json::Value;
use tiktoken_rs::tokenizer::{get_tokenizer, Tokenizer};
use tiktoken_rs::CoreBPE;

/// Bound on the output text accumulated for completion-side estimation.
/// ~1 MiB of UTF-8 comfortably exceeds the largest real model outputs
/// (128K tokens ≈ 0.5 MiB of English text); past the cap the estimate
/// becomes a lower bound instead of the buffer growing without limit.
pub const OUTPUT_ACCUMULATION_CAP: usize = 1 << 20;

const TOKENS_PER_MESSAGE: u32 = 3;
const TOKENS_PER_NAME: u32 = 1;
/// "Every reply is primed with `<|start|>assistant<|message|>`."
const REPLY_PRIMING: u32 = 3;
const TOOLS_OVERHEAD: u32 = 9;
/// Rendered tool definitions absorb part of the system message frame.
const TOOLS_WITH_SYSTEM_DISCOUNT: u32 = 4;
/// Image content block without size information: the fixed low-detail
/// cost. `detail: high` costs `85 + 170×tiles`, but tiling needs the
/// image dimensions, which the gateway never fetches — those fall back
/// to [`IMAGE_TOKENS_DEFAULT`].
const IMAGE_TOKENS_LOW: u32 = 85;
const IMAGE_TOKENS_DEFAULT: u32 = 250;

/// The request-side input for prompt-token estimation, captured (moved,
/// not copied where the call site allows) before the response starts and
/// tokenized only if estimation actually runs.
pub enum PromptInput {
    /// OpenAI-shaped chat request (`/v1/chat/completions`).
    Chat(Box<aisix_gateway::chat::ChatFormat>),
    /// Raw Anthropic `/v1/messages` request body.
    Anthropic(Value),
    /// Raw `/v1/responses` request body.
    Responses(Value),
}

/// Prompt shape + the model name that selects the tokenizer encoding.
pub struct Estimator {
    model: String,
    input: PromptInput,
}

/// Outcome of [`fill_missing`]: the (possibly filled) token counts and
/// whether estimation supplied any non-zero component.
pub struct Filled {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub estimated: bool,
}

impl Estimator {
    pub fn new(model: impl Into<String>, input: PromptInput) -> Self {
        Self {
            model: model.into(),
            input,
        }
    }

    /// Count the prompt side of the captured request.
    pub fn count_prompt(&self) -> u32 {
        let bpe = bpe_for(&self.model);
        match &self.input {
            PromptInput::Chat(req) => count_chat_prompt(bpe, req),
            PromptInput::Anthropic(body) => count_anthropic_prompt(bpe, body),
            PromptInput::Responses(body) => count_responses_prompt(bpe, body),
        }
    }

    pub fn count_output(&self, text: &str) -> u32 {
        count_text(&self.model, text)
    }
}

/// Count `text` as plain text with the model's encoding — the
/// completion-side rule (no message overhead).
pub fn count_text(model: &str, text: &str) -> u32 {
    enc(bpe_for(model), text)
}

/// Per-field `or` fill: upstream values win when non-zero; zeros are
/// replaced with local counts. `output_text: None` disables the
/// completion side (nothing was delivered, or nothing was accumulated).
/// Sub-counters (cache/reasoning) are never estimated.
pub fn fill_missing(
    est: &Estimator,
    upstream_prompt: u32,
    upstream_completion: u32,
    output_text: Option<&str>,
) -> Filled {
    let mut filled = Filled {
        prompt_tokens: upstream_prompt,
        completion_tokens: upstream_completion,
        estimated: false,
    };
    if upstream_prompt == 0 {
        let n = est.count_prompt();
        if n > 0 {
            filled.prompt_tokens = n;
            filled.estimated = true;
        }
    }
    if upstream_completion == 0 {
        if let Some(text) = output_text {
            let n = est.count_output(text);
            if n > 0 {
                filled.completion_tokens = n;
                filled.estimated = true;
            }
        }
    }
    filled
}

fn bpe_for(model: &str) -> &'static CoreBPE {
    match get_tokenizer(model) {
        Some(Tokenizer::O200kHarmony) => tiktoken_rs::o200k_harmony_singleton(),
        Some(Tokenizer::O200kBase) => tiktoken_rs::o200k_base_singleton(),
        Some(Tokenizer::P50kBase) => tiktoken_rs::p50k_base_singleton(),
        Some(Tokenizer::P50kEdit) => tiktoken_rs::p50k_edit_singleton(),
        Some(Tokenizer::R50kBase | Tokenizer::Gpt2) => tiktoken_rs::r50k_base_singleton(),
        Some(Tokenizer::Cl100kBase) | None => tiktoken_rs::cl100k_base_singleton(),
    }
}

fn clamp(n: usize) -> u32 {
    n.min(u32::MAX as usize) as u32
}

/// Encode in slices of this size. The tokenizer's regex engine is
/// superlinear on long unbroken runs (a multi-MiB single piece burns
/// seconds of CPU) and its vendored wrapper unwraps regex failures —
/// a ~1 MiB whitespace run panics with a fancy-regex StackOverflow.
/// Slicing linearizes the cost and keeps any failure small; each
/// boundary can split at most one token (+1 per slice, negligible).
const ENCODE_SLICE_BYTES: usize = 64 * 1024;

/// Total bytes exactly tokenized per [`enc`] call; the tail beyond it
/// extrapolates at ~4 bytes/token. Real prompts sit far below this
/// (≈256K tokens of English), so accuracy is unaffected — the cap
/// exists so an adversarial multi-MiB piece can't pin a runtime
/// worker inside a Drop guard.
const EXACT_COUNT_BUDGET: usize = 1 << 20;

/// Estimated-token fallback rate for text the exact pass skipped or
/// the tokenizer failed on: the ~4-bytes-per-token rule of thumb.
fn approx_tokens(bytes: usize) -> u32 {
    clamp(bytes / 4)
}

/// Panic-proof, cost-bounded token count. Estimation runs on the
/// request hot path and inside Drop guards (where a second panic
/// during an unwind aborts the process), so the tokenizer must never
/// panic out of here and must never burn unbounded CPU: encode in
/// bounded slices, catch any tokenizer panic, and extrapolate the
/// remainder past the exact-count budget.
fn enc(bpe: &CoreBPE, text: &str) -> u32 {
    let mut n: u32 = 0;
    let mut rest = text;
    let mut budget = EXACT_COUNT_BUDGET;
    while !rest.is_empty() {
        if budget == 0 {
            return n.saturating_add(approx_tokens(rest.len()));
        }
        let mut cut = ENCODE_SLICE_BYTES.min(rest.len());
        while !rest.is_char_boundary(cut) {
            cut += 1;
        }
        let (head, tail) = rest.split_at(cut);
        let counted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bpe.encode_ordinary(head).len()
        }))
        .map(clamp)
        .unwrap_or_else(|_| approx_tokens(head.len()));
        n = n.saturating_add(counted);
        budget = budget.saturating_sub(cut);
        rest = tail;
    }
    n
}

// ---------------------------------------------------------------------
// Chat (`/v1/chat/completions`)
// ---------------------------------------------------------------------

fn count_chat_prompt(bpe: &CoreBPE, req: &aisix_gateway::chat::ChatFormat) -> u32 {
    let mut n: u32 = 0;
    let mut has_system = false;
    for m in &req.messages {
        let role = role_label(&m.role);
        if role == "system" {
            has_system = true;
        }
        n = n.saturating_add(TOKENS_PER_MESSAGE);
        n = n.saturating_add(enc(bpe, role));
        if let Some(blocks) = m.content_blocks.as_deref() {
            n = n.saturating_add(count_content_blocks(bpe, blocks));
        } else if let Some(text) = m.content.as_deref() {
            n = n.saturating_add(enc(bpe, text));
        }
        if let Some(name) = m.name.as_deref() {
            n = n
                .saturating_add(TOKENS_PER_NAME)
                .saturating_add(enc(bpe, name));
        }
        if let Some(id) = m.tool_call_id.as_deref() {
            n = n.saturating_add(enc(bpe, id));
        }
        for (key, value) in &m.extra {
            match key.as_str() {
                "tool_calls" | "function_call" => {
                    n = n.saturating_add(count_tool_call_arguments(bpe, value));
                }
                _ => {
                    if let Some(s) = value.as_str() {
                        n = n.saturating_add(enc(bpe, s));
                    }
                }
            }
        }
    }
    n.saturating_add(count_request_extra(
        bpe,
        req.extra.get("tools"),
        req.extra.get("tool_choice"),
        has_system,
    ))
}

fn role_label(role: &aisix_gateway::chat::Role) -> &'static str {
    match role {
        aisix_gateway::chat::Role::System => "system",
        aisix_gateway::chat::Role::User => "user",
        aisix_gateway::chat::Role::Assistant => "assistant",
        aisix_gateway::chat::Role::Tool => "tool",
    }
}

/// History `tool_calls` / legacy `function_call`: only the call
/// `arguments` strings are counted; the function *names* are covered by
/// the rendered tool definitions.
fn count_tool_call_arguments(bpe: &CoreBPE, value: &Value) -> u32 {
    let mut n: u32 = 0;
    let calls: &[Value] = match value {
        Value::Array(a) => a.as_slice(),
        v @ Value::Object(_) => std::slice::from_ref(v),
        _ => return 0,
    };
    for call in calls {
        let args = call
            .get("function")
            .and_then(|f| f.get("arguments"))
            .or_else(|| call.get("arguments"));
        if let Some(Value::String(s)) = args {
            n = n.saturating_add(enc(bpe, s));
        }
    }
    n
}

// ---------------------------------------------------------------------
// Anthropic (`/v1/messages`)
// ---------------------------------------------------------------------

fn count_anthropic_prompt(bpe: &CoreBPE, body: &Value) -> u32 {
    let mut n: u32 = 0;
    let mut has_system = false;
    if let Some(system) = body.get("system") {
        if !system.is_null() {
            has_system = true;
            n = n.saturating_add(TOKENS_PER_MESSAGE);
            n = n.saturating_add(enc(bpe, "system"));
            n = n.saturating_add(count_content_value(bpe, system));
        }
    }
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for m in messages {
            n = n.saturating_add(TOKENS_PER_MESSAGE);
            if let Some(role) = m.get("role").and_then(|r| r.as_str()) {
                n = n.saturating_add(enc(bpe, role));
            }
            if let Some(content) = m.get("content") {
                n = n.saturating_add(count_content_value(bpe, content));
            }
        }
    }
    n.saturating_add(count_request_extra(
        bpe,
        body.get("tools"),
        body.get("tool_choice"),
        has_system,
    ))
}

// ---------------------------------------------------------------------
// Responses (`/v1/responses`)
// ---------------------------------------------------------------------

/// The Responses API has no cookbook counting scheme; items are close
/// enough to chat messages that each item takes the per-message overhead
/// and its text-bearing fields are counted like message content.
fn count_responses_prompt(bpe: &CoreBPE, body: &Value) -> u32 {
    let mut n: u32 = 0;
    let mut has_system = false;
    if let Some(instructions) = body.get("instructions").and_then(|i| i.as_str()) {
        has_system = true;
        n = n.saturating_add(TOKENS_PER_MESSAGE);
        n = n.saturating_add(enc(bpe, "system"));
        n = n.saturating_add(enc(bpe, instructions));
    }
    match body.get("input") {
        Some(Value::String(text)) => {
            n = n.saturating_add(TOKENS_PER_MESSAGE);
            n = n.saturating_add(enc(bpe, "user"));
            n = n.saturating_add(enc(bpe, text));
        }
        Some(Value::Array(items)) => {
            for item in items {
                n = n.saturating_add(TOKENS_PER_MESSAGE);
                match item.get("role").and_then(|r| r.as_str()) {
                    Some(role) => {
                        // Message-shaped item.
                        if role == "system" || role == "developer" {
                            has_system = true;
                        }
                        n = n.saturating_add(enc(bpe, role));
                        if let Some(content) = item.get("content") {
                            n = n.saturating_add(count_content_value(bpe, content));
                        }
                    }
                    None => {
                        // Non-message item (function_call, function_call_output,
                        // reasoning, ...): count its text-bearing fields.
                        n = n.saturating_add(count_string_fields(
                            bpe,
                            item,
                            &["type", "id", "status"],
                        ));
                    }
                }
            }
        }
        _ => {}
    }
    n.saturating_add(count_request_extra(
        bpe,
        body.get("tools"),
        body.get("tool_choice"),
        has_system,
    ))
}

// ---------------------------------------------------------------------
// Shared content walkers
// ---------------------------------------------------------------------

/// `content` that is either a bare string or an array of typed blocks.
fn count_content_value(bpe: &CoreBPE, content: &Value) -> u32 {
    match content {
        Value::String(s) => enc(bpe, s),
        Value::Array(blocks) => count_content_blocks(bpe, blocks),
        _ => 0,
    }
}

/// Typed content blocks: OpenAI chat parts (`text`, `image_url`),
/// Anthropic blocks (`text`, `image`, `tool_use`, `tool_result`,
/// `thinking`), and Responses parts (`input_text`, `output_text`,
/// `refusal`). Unknown block types count 0 — the emit path must never
/// fail on an unrecognized shape.
fn count_content_blocks(bpe: &CoreBPE, blocks: &[Value]) -> u32 {
    let mut n: u32 = 0;
    for block in blocks {
        if let Some(s) = block.as_str() {
            n = n.saturating_add(enc(bpe, s));
            continue;
        }
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text" | "input_text" | "output_text") => {
                if let Some(s) = block.get("text").and_then(|t| t.as_str()) {
                    n = n.saturating_add(enc(bpe, s));
                }
            }
            Some("refusal") => {
                if let Some(s) = block.get("refusal").and_then(|t| t.as_str()) {
                    n = n.saturating_add(enc(bpe, s));
                }
            }
            Some("image_url") => {
                let detail = block
                    .get("image_url")
                    .and_then(|i| i.get("detail"))
                    .and_then(|d| d.as_str())
                    .unwrap_or("auto");
                n = n.saturating_add(match detail {
                    "low" | "auto" => IMAGE_TOKENS_LOW,
                    // `high` needs the image dimensions to tile; the
                    // gateway never fetches them, so use the flat default.
                    _ => IMAGE_TOKENS_DEFAULT,
                });
            }
            Some("image" | "input_image") => {
                n = n.saturating_add(IMAGE_TOKENS_DEFAULT);
            }
            Some("tool_use" | "tool_result") => {
                n = n.saturating_add(count_string_fields(
                    bpe,
                    block,
                    &["type", "id", "tool_use_id", "cache_control", "is_error"],
                ));
            }
            Some("thinking") => {
                if let Some(s) = block.get("thinking").and_then(|t| t.as_str()) {
                    n = n.saturating_add(enc(bpe, s));
                }
            }
            _ => {}
        }
    }
    n
}

/// Count the text carried by an arbitrary JSON object: every string
/// field outside `skip`, with non-string composites (e.g. a `tool_use`
/// `input` object) counted via their compact JSON serialization.
fn count_string_fields(bpe: &CoreBPE, value: &Value, skip: &[&str]) -> u32 {
    let Some(obj) = value.as_object() else {
        return 0;
    };
    let mut n: u32 = 0;
    for (key, v) in obj {
        if skip.contains(&key.as_str()) {
            continue;
        }
        match v {
            Value::String(s) => n = n.saturating_add(enc(bpe, s)),
            Value::Array(_) | Value::Object(_) => {
                if let Ok(s) = serde_json::to_string(v) {
                    n = n.saturating_add(enc(bpe, &s));
                }
            }
            _ => {}
        }
    }
    n
}

/// Reply priming + tool definitions + `tool_choice`.
fn count_request_extra(
    bpe: &CoreBPE,
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    has_system: bool,
) -> u32 {
    let mut n = REPLY_PRIMING;
    let tools = tools.and_then(|t| t.as_array()).filter(|t| !t.is_empty());
    if let Some(tools) = tools {
        n = n.saturating_add(enc(bpe, &format_tool_definitions(tools)));
        n = n.saturating_add(TOOLS_OVERHEAD);
        if has_system {
            n = n.saturating_sub(TOOLS_WITH_SYSTEM_DISCOUNT);
        }
    }
    match tool_choice {
        Some(Value::String(s)) if s == "none" => n = n.saturating_add(1),
        Some(tc @ Value::Object(_)) => {
            n = n.saturating_add(7);
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .or_else(|| tc.get("name"))
                .and_then(|v| v.as_str());
            if let Some(name) = name {
                n = n.saturating_add(enc(bpe, name));
            }
        }
        _ => {}
    }
    n
}

/// Render tool definitions in the TypeScript-namespace form the OpenAI
/// tokenizer overhead was measured against. Accepts both the OpenAI
/// shape (`{type, function: {name, description, parameters}}`) and flat
/// shapes (Anthropic `{name, description, input_schema}`, Responses
/// `{type, name, description, parameters}`).
fn format_tool_definitions(tools: &[Value]) -> String {
    let mut lines: Vec<String> = vec!["namespace functions {".into(), String::new()];
    for tool in tools {
        if !tool.is_object() {
            continue;
        }
        let function = match tool.get("function") {
            Some(f @ Value::Object(_)) => f.clone(),
            _ => {
                let params = tool
                    .get("input_schema")
                    .or_else(|| tool.get("parameters"))
                    .cloned()
                    .filter(|p| p.is_object())
                    .unwrap_or_else(|| Value::Object(Default::default()));
                serde_json::json!({
                    "name": tool.get("name").cloned().unwrap_or(Value::Null),
                    "description": tool.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": params,
                })
            }
        };
        let Some(name) = function.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        if let Some(desc) = function.get("description").and_then(|d| d.as_str()) {
            if !desc.is_empty() {
                lines.push(format!("// {desc}"));
            }
        }
        let parameters = function.get("parameters").cloned().unwrap_or(Value::Null);
        let has_properties = parameters
            .get("properties")
            .and_then(|p| p.as_object())
            .is_some_and(|p| !p.is_empty());
        if has_properties {
            lines.push(format!("type {name} = (_: {{"));
            lines.push(format_object_parameters(&parameters, 0));
            lines.push("}) => any;".into());
        } else {
            lines.push(format!("type {name} = () => any;"));
        }
        lines.push(String::new());
    }
    lines.push("} // namespace functions".into());
    lines.join("\n")
}

fn format_object_parameters(parameters: &Value, indent: usize) -> String {
    let Some(properties) = parameters.get("properties").and_then(|p| p.as_object()) else {
        return String::new();
    };
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(|r| r.as_array())
        .map(|r| r.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let mut lines: Vec<String> = Vec::new();
    for (key, props) in properties {
        if let Some(desc) = props.get("description").and_then(|d| d.as_str()) {
            lines.push(format!("// {desc}"));
        }
        let question = if required.contains(&key.as_str()) {
            ""
        } else {
            "?"
        };
        lines.push(format!("{key}{question}: {},", format_type(props, indent)));
    }
    let pad = " ".repeat(indent);
    lines
        .iter()
        .map(|l| format!("{pad}{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_type(props: &Value, indent: usize) -> String {
    let enum_values = |props: &Value| -> Option<String> {
        props.get("enum").and_then(|e| e.as_array()).map(|items| {
            items
                .iter()
                .map(|i| match i {
                    Value::String(s) => format!("\"{s}\""),
                    other => format!("\"{other}\""),
                })
                .collect::<Vec<_>>()
                .join(" | ")
        })
    };
    match props.get("type").and_then(|t| t.as_str()) {
        Some("string") => enum_values(props).unwrap_or_else(|| "string".into()),
        Some("array") => match props.get("items") {
            Some(items) => format!("{}[]", format_type(items, indent)),
            None => "any[]".into(),
        },
        Some("object") => format!("{{\n{}\n}}", format_object_parameters(props, indent + 2)),
        Some("integer" | "number") => enum_values(props).unwrap_or_else(|| "number".into()),
        Some("boolean") => "boolean".into(),
        Some("null") => "null".into(),
        _ => "any".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::chat::{ChatFormat, ChatMessage, Role};
    use serde_json::json;

    fn chat_est(model: &str, req: ChatFormat) -> Estimator {
        Estimator::new(model, PromptInput::Chat(Box::new(req)))
    }

    fn msg(role: Role, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: Some(content.to_string()),
            content_blocks: None,
            name: None,
            tool_call_id: None,
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn encoding_selection_mirrors_tiktoken_model_map() {
        assert!(std::ptr::eq(
            bpe_for("gpt-4o-2024-08-06"),
            tiktoken_rs::o200k_base_singleton(),
        ));
        assert!(std::ptr::eq(
            bpe_for("gpt-4"),
            tiktoken_rs::cl100k_base_singleton(),
        ));
        // Unknown / non-OpenAI models fall back to cl100k_base.
        assert!(std::ptr::eq(
            bpe_for("claude-sonnet-4-5"),
            tiktoken_rs::cl100k_base_singleton(),
        ));
        assert!(std::ptr::eq(
            bpe_for(""),
            tiktoken_rs::cl100k_base_singleton(),
        ));
    }

    #[test]
    fn count_text_plain() {
        assert_eq!(count_text("gpt-4", ""), 0);
        // cl100k_base: "Hello" + " world" = 2 tokens.
        assert_eq!(count_text("gpt-4", "Hello world"), 2);
        // Unknown model uses the same fallback encoding.
        assert_eq!(count_text("some-proxy-model", "Hello world"), 2);
    }

    /// Audit HIGH-1 regression: a ~1 MiB unbroken whitespace run makes
    /// the tokenizer's regex engine fail (StackOverflow) and its
    /// vendored wrapper panic. `enc` slices the input and catches the
    /// panic, so estimation must return a sane count instead of
    /// panicking (which inside a Drop guard during an unwind would
    /// abort the whole process).
    #[test]
    fn adversarial_whitespace_run_does_not_panic() {
        let evil = " ".repeat(1 << 20) + "x";
        let n = count_text("gpt-4", &evil);
        assert!(n > 0, "a 1 MiB run must still produce a count, got {n}");
    }

    /// Text past the exact-count budget extrapolates at ~4 bytes/token
    /// instead of burning unbounded CPU: the count keeps growing with
    /// input size beyond the budget.
    #[test]
    fn exact_count_budget_extrapolates_tail() {
        let base = "hello world ".repeat((1 << 20) / 12 + 1); // ≈ budget-sized
        let doubled = base.repeat(2);
        let n1 = count_text("gpt-4", &base);
        let n2 = count_text("gpt-4", &doubled);
        assert!(n2 > n1, "tail beyond the budget must still be counted");
        // The extrapolated tail (~4 B/token) stays within 2× of the
        // exact rate for this corpus (~3 B/token), so the total is the
        // right order of magnitude.
        assert!(n2 < n1 * 3);
    }

    #[test]
    fn chat_prompt_counts_cookbook_overhead() {
        // One user message: 3 (per-message) + 1 ("user") + 1 ("Hello")
        // + 3 (reply priming) = 8.
        let req = ChatFormat::new("gpt-4", vec![msg(Role::User, "Hello")]);
        assert_eq!(chat_est("gpt-4", req).count_prompt(), 8);
    }

    #[test]
    fn chat_prompt_counts_name_and_tool_calls() {
        let mut named = msg(Role::User, "Hello");
        named.name = Some("alice".into());
        let base = ChatFormat::new("gpt-4", vec![msg(Role::User, "Hello")]);
        let with_name = ChatFormat::new("gpt-4", vec![named]);
        let base_n = chat_est("gpt-4", base).count_prompt();
        let name_n = chat_est("gpt-4", with_name).count_prompt();
        // +1 name overhead +count("alice").
        assert_eq!(
            name_n,
            base_n + TOKENS_PER_NAME + count_text("gpt-4", "alice")
        );

        let mut assistant = msg(Role::Assistant, "");
        assistant.content = None;
        assistant.extra.insert(
            "tool_calls".into(),
            json!([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }]),
        );
        let req = ChatFormat::new("gpt-4", vec![assistant]);
        let n = chat_est("gpt-4", req).count_prompt();
        // 3 + count("assistant") + count(arguments) + 3 priming; the id and
        // function name are intentionally not counted.
        let expected = TOKENS_PER_MESSAGE
            + count_text("gpt-4", "assistant")
            + count_text("gpt-4", "{\"city\":\"Paris\"}")
            + REPLY_PRIMING;
        assert_eq!(n, expected);
    }

    #[test]
    fn tool_definitions_render_and_add_overhead() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["c", "f"]}
                    },
                    "required": ["city"]
                }
            }
        }]);
        let rendered = format_tool_definitions(tools.as_array().unwrap());
        assert_eq!(
            rendered,
            "namespace functions {\n\n// Get the weather\ntype get_weather = (_: {\n\
             // City name\ncity: string,\nunit?: \"c\" | \"f\",\n}) => any;\n\n\
             } // namespace functions"
        );

        let mut req = ChatFormat::new("gpt-4", vec![msg(Role::User, "Hi")]);
        req.extra.insert("tools".into(), tools.clone());
        let with_tools = chat_est("gpt-4", req).count_prompt();
        let plain = chat_est(
            "gpt-4",
            ChatFormat::new("gpt-4", vec![msg(Role::User, "Hi")]),
        )
        .count_prompt();
        assert_eq!(
            with_tools,
            plain + count_text("gpt-4", &rendered) + TOOLS_OVERHEAD
        );

        // Anthropic flat tool shape renders identically.
        let anthropic_tools = json!([{
            "name": "get_weather",
            "description": "Get the weather",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["city"]
            }
        }]);
        assert_eq!(
            format_tool_definitions(anthropic_tools.as_array().unwrap()),
            rendered
        );
    }

    #[test]
    fn tools_with_system_message_discount() {
        let tools = json!([{"type": "function", "function": {"name": "f"}}]);
        let mut with_system = ChatFormat::new(
            "gpt-4",
            vec![msg(Role::System, "Be brief"), msg(Role::User, "Hi")],
        );
        with_system.extra.insert("tools".into(), tools.clone());
        let mut without_system = ChatFormat::new("gpt-4", vec![msg(Role::User, "Hi")]);
        without_system.extra.insert("tools".into(), tools);
        let sys_msg_cost =
            TOKENS_PER_MESSAGE + count_text("gpt-4", "system") + count_text("gpt-4", "Be brief");
        assert_eq!(
            chat_est("gpt-4", with_system).count_prompt(),
            chat_est("gpt-4", without_system).count_prompt() + sys_msg_cost
                - TOOLS_WITH_SYSTEM_DISCOUNT
        );
    }

    #[test]
    fn anthropic_prompt_counts_system_blocks_and_tools() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "system": "Be helpful",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "tool_use", "id": "tu_1", "name": "get_weather",
                     "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "sunny"}
                ]}
            ]
        });
        let est = Estimator::new("claude-sonnet-4-5", PromptInput::Anthropic(body));
        let m = |s: &str| count_text("claude-sonnet-4-5", s);
        let expected = (TOKENS_PER_MESSAGE + m("system") + m("Be helpful"))
            + (TOKENS_PER_MESSAGE + m("user") + m("Hello"))
            + (TOKENS_PER_MESSAGE
                + m("assistant")
                + m("Hi")
                + m("get_weather")
                + m("{\"city\":\"Paris\"}"))
            + (TOKENS_PER_MESSAGE + m("user") + m("sunny"))
            + REPLY_PRIMING;
        assert_eq!(est.count_prompt(), expected);
    }

    #[test]
    fn responses_prompt_counts_string_and_item_inputs() {
        let simple = json!({"model": "gpt-4o", "input": "Hello"});
        let est = Estimator::new("gpt-4o", PromptInput::Responses(simple));
        let m = |s: &str| count_text("gpt-4o", s);
        assert_eq!(
            est.count_prompt(),
            TOKENS_PER_MESSAGE + m("user") + m("Hello") + REPLY_PRIMING
        );

        let items = json!({
            "model": "gpt-4o",
            "instructions": "Be brief",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "get_weather", "arguments": "{}", "status": "completed"}
            ]
        });
        let est = Estimator::new("gpt-4o", PromptInput::Responses(items));
        let expected = (TOKENS_PER_MESSAGE + m("system") + m("Be brief"))
            + (TOKENS_PER_MESSAGE + m("user") + m("Hello"))
            + (TOKENS_PER_MESSAGE + m("call_1") + m("get_weather") + m("{}"))
            + REPLY_PRIMING;
        assert_eq!(est.count_prompt(), expected);
    }

    #[test]
    fn fill_missing_or_semantics() {
        let est = chat_est(
            "gpt-4",
            ChatFormat::new("gpt-4", vec![msg(Role::User, "Hello")]),
        );

        // Upstream reported both: untouched, not estimated.
        let f = fill_missing(&est, 12, 34, Some("Hello world"));
        assert_eq!(
            (f.prompt_tokens, f.completion_tokens, f.estimated),
            (12, 34, false)
        );

        // Both missing: both filled.
        let f = fill_missing(&est, 0, 0, Some("Hello world"));
        assert_eq!(f.prompt_tokens, 8);
        assert_eq!(f.completion_tokens, 2);
        assert!(f.estimated);

        // Partial: only the missing side fills.
        let f = fill_missing(&est, 12, 0, Some("Hello world"));
        assert_eq!(
            (f.prompt_tokens, f.completion_tokens, f.estimated),
            (12, 2, true)
        );

        // No output text (nothing delivered): completion stays 0, prompt fills.
        let f = fill_missing(&est, 0, 0, None);
        assert_eq!(
            (f.prompt_tokens, f.completion_tokens, f.estimated),
            (8, 0, true)
        );

        // Empty output text estimates to 0 — not marked estimated for it.
        let f = fill_missing(&est, 5, 0, Some(""));
        assert_eq!(
            (f.prompt_tokens, f.completion_tokens, f.estimated),
            (5, 0, false)
        );
    }
}
