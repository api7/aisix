//! The parts of a Converse request built from a chat request's top-level
//! fields: `additionalModelRequestFields`, the Bedrock-native request blocks
//! (`guardrailConfig`, `performanceConfig`, `serviceTier`,
//! `requestMetadata`, `outputConfig`), and the per-publisher translation of
//! `reasoning_effort`, `stop`, `max_completion_tokens`, `top_k` and
//! `parallel_tool_calls`.
//!
//! References:
//! - <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html>
//! - <https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html>

use aisix_gateway::{BridgeError, ChatFormat};
use aws_sdk_bedrockruntime::types::{
    GuardrailConfiguration, GuardrailStreamConfiguration, GuardrailStreamProcessingMode,
    GuardrailTrace, JsonSchemaDefinition, OutputConfig, OutputFormat, OutputFormatStructure,
    OutputFormatType, PerformanceConfigLatency, PerformanceConfiguration, PromptVariableValues,
    ServiceTier, ServiceTierType,
};
use serde_json::{Map, Value};

use crate::bridge::{capability_id, strip_region_prefix};
use crate::openai_compat::is_openai_gpt;

/// OpenAI Chat Completions parameters with no Converse field of their
/// own. A publisher whose Converse contract defines one takes it as an
/// `additionalModelRequestFields` member ([`passes_through`]); every
/// other publisher has no place for it.
const OPENAI_UNTRANSLATED: &[&str] = &[
    "audio",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "prompt_cache_key",
    "prompt_cache_retention",
    "response_format",
    "safety_identifier",
    "seed",
    "store",
    "top_logprobs",
    "user",
    "verbosity",
    "web_search_options",
];

/// Top-level request fields that are never forwarded as
/// `additionalModelRequestFields`: the OpenAI parameters Converse carries in
/// its own fields or the gateway translates, the fields the gateway
/// consumes itself, and the Bedrock-native blocks that have their own place
/// in the request.
const TRANSLATED: &[&str] = &[
    // OpenAI Chat Completions, carried by Converse's own fields
    "function_call",
    "functions",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "model",
    "reasoning_effort",
    "service_tier",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    // Translated or consumed by the gateway
    "thinking",
    "output_config",
    "anthropic_beta",
    "context_management",
    "cache_control",
    "top_k",
    "extra_body",
    "extra_headers",
    "model_id",
    // Bedrock-native request blocks and top-level Converse fields
    "additionalModelRequestFields",
    "additionalModelResponseFieldPaths",
    "promptVariables",
    "guardrailConfig",
    "performanceConfig",
    "serviceTier",
    "requestMetadata",
    "outputConfig",
];

/// The OpenAI models on Converse: gpt-oss, GPT and Grok. AWS's OpenAI
/// model page: "Include any other fields in the
/// `additionalModelRequestFields` object."
fn is_openai_family(id: &str) -> bool {
    id.contains("gpt-oss") || is_openai_gpt(id) || id.starts_with("xai.grok-")
}

/// Whether an untranslated OpenAI parameter reaches this model as an
/// `additionalModelRequestFields` member, per its publisher's Bedrock
/// request body. `n`, `logprobs` and `top_logprobs` never do: Converse
/// answers with one message and no log probabilities, so what they ask
/// for would be billed and lost. A `json_schema` `response_format` becomes
/// the request's tool or output config instead.
fn passes_through(key: &str, req: &ChatFormat, id: &str) -> bool {
    if is_openai_family(id) {
        return match key {
            "n" | "logprobs" | "top_logprobs" => false,
            "response_format" => {
                req.extra
                    .get("response_format")
                    .and_then(|rf| rf.get("type"))
                    .and_then(Value::as_str)
                    != Some("json_schema")
            }
            _ => true,
        };
    }
    // Cohere Command R / R+: `seed`, `frequency_penalty`, `presence_penalty`.
    if id.starts_with("cohere.command-r") {
        return matches!(key, "seed" | "frequency_penalty" | "presence_penalty");
    }
    // AI21 Jamba: `frequency_penalty`, `presence_penalty`.
    if id.starts_with("ai21.jamba") {
        return matches!(key, "frequency_penalty" | "presence_penalty");
    }
    false
}

/// The caller's own `additionalModelRequestFields`: the object they sent
/// under that name, merged with every top-level field that is not an
/// OpenAI parameter or one the gateway handles, and with the OpenAI
/// parameters this model's Converse contract takes there.
pub(crate) fn caller_additional_fields(req: &ChatFormat, model_id: &str) -> Map<String, Value> {
    let id = base_id(model_id);
    let mut fields = match req.extra.get("additionalModelRequestFields") {
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    };
    for (key, value) in &req.extra {
        let key = key.as_str();
        let forward = if TRANSLATED.contains(&key) {
            false
        } else if OPENAI_UNTRANSLATED.contains(&key) {
            passes_through(key, req, &id)
        } else {
            true
        };
        if forward {
            fields.insert(key.to_string(), value.clone());
        }
    }
    fields
}

/// `additionalModelResponseFieldPaths`: the JSON pointers into the model's
/// response that Converse returns as `additionalModelResponseFields`.
fn additional_response_field_paths(req: &ChatFormat) -> Result<Option<Vec<String>>, BridgeError> {
    const WANT: &str = "must be an array of strings";
    match req.extra.get("additionalModelResponseFieldPaths") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
            .map(Some)
            .ok_or_else(|| invalid("additionalModelResponseFieldPaths", WANT)),
        Some(_) => Err(invalid("additionalModelResponseFieldPaths", WANT)),
    }
}

/// `promptVariables`: the values for a Prompt management prompt's
/// variables, each `{"text": "..."}`.
fn prompt_variables(req: &ChatFormat) -> Result<Vec<(String, PromptVariableValues)>, BridgeError> {
    const WANT: &str = "values must be {\"text\": string} objects";
    let Some(vars) = object(req, "promptVariables")? else {
        return Ok(Vec::new());
    };
    vars.iter()
        .map(|(name, value)| {
            let text = value
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("promptVariables", WANT))?;
            Ok((name.clone(), PromptVariableValues::Text(text.to_string())))
        })
        .collect()
}

/// The Bedrock-native blocks of a Converse request.
#[derive(Default)]
pub(crate) struct NativeBlocks {
    pub guardrail: Option<Map<String, Value>>,
    pub performance: Option<PerformanceConfiguration>,
    pub service_tier: Option<ServiceTier>,
    pub request_metadata: Vec<(String, String)>,
    pub output_config: Option<OutputConfig>,
    pub response_field_paths: Option<Vec<String>>,
    pub prompt_variables: Vec<(String, PromptVariableValues)>,
}

fn invalid(field: &str, why: &str) -> BridgeError {
    BridgeError::InvalidRequest(format!("{field} {why}"))
}

fn object<'a>(
    req: &'a ChatFormat,
    field: &str,
) -> Result<Option<&'a Map<String, Value>>, BridgeError> {
    match req.extra.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(m)) => Ok(Some(m)),
        Some(_) => Err(invalid(field, "must be an object")),
    }
}

fn string_field<'a>(m: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    m.get(key).and_then(Value::as_str)
}

impl NativeBlocks {
    pub(crate) fn from_request(req: &ChatFormat) -> Result<Self, BridgeError> {
        let mut blocks = NativeBlocks {
            guardrail: object(req, "guardrailConfig")?.cloned(),
            ..Default::default()
        };
        if let Some(guardrail) = &blocks.guardrail {
            if string_field(guardrail, "guardrailIdentifier").is_none()
                || string_field(guardrail, "guardrailVersion").is_none()
            {
                return Err(invalid(
                    "guardrailConfig",
                    "must carry guardrailIdentifier and guardrailVersion",
                ));
            }
        }
        if let Some(perf) = object(req, "performanceConfig")? {
            let mut b = PerformanceConfiguration::builder();
            if let Some(latency) = string_field(perf, "latency") {
                b = b.latency(PerformanceConfigLatency::from(latency));
            }
            blocks.performance = Some(b.build());
        }
        // `serviceTier` is the Bedrock-native object; OpenAI's
        // `service_tier` string maps onto it, `auto` as Bedrock's default.
        let tier = match object(req, "serviceTier")? {
            Some(tier) => Some(
                string_field(tier, "type")
                    .ok_or_else(|| invalid("serviceTier", "must carry a type"))?
                    .to_string(),
            ),
            None => match req.extra.get("service_tier") {
                None | Some(Value::Null) => None,
                Some(Value::String(t)) => match t.as_str() {
                    "auto" => Some("default".to_string()),
                    "default" | "flex" | "priority" => Some(t.clone()),
                    // A tier Bedrock has no counterpart for.
                    _ => None,
                },
                Some(_) => return Err(invalid("service_tier", "must be a string")),
            },
        };
        if let Some(tier) = tier {
            blocks.service_tier = Some(
                ServiceTier::builder()
                    .r#type(ServiceTierType::from(tier.as_str()))
                    .build()
                    .map_err(|e| invalid("serviceTier", &e.to_string()))?,
            );
        }
        if let Some(metadata) = object(req, "requestMetadata")? {
            for (k, v) in metadata {
                let v = v
                    .as_str()
                    .ok_or_else(|| invalid("requestMetadata", "values must be strings"))?;
                blocks.request_metadata.push((k.clone(), v.to_string()));
            }
        }
        if let Some(config) = object(req, "outputConfig")? {
            blocks.output_config = output_config(config)?;
        }
        blocks.response_field_paths = additional_response_field_paths(req)?;
        blocks.prompt_variables = prompt_variables(req)?;
        Ok(blocks)
    }

    pub(crate) fn converse_guardrail(&self) -> Option<GuardrailConfiguration> {
        let g = self.guardrail.as_ref()?;
        let mut b = GuardrailConfiguration::builder()
            .set_guardrail_identifier(string_field(g, "guardrailIdentifier").map(str::to_string))
            .set_guardrail_version(string_field(g, "guardrailVersion").map(str::to_string));
        if let Some(trace) = string_field(g, "trace") {
            b = b.trace(GuardrailTrace::from(trace));
        }
        Some(b.build())
    }

    pub(crate) fn stream_guardrail(&self) -> Option<GuardrailStreamConfiguration> {
        let g = self.guardrail.as_ref()?;
        let mut b = GuardrailStreamConfiguration::builder()
            .set_guardrail_identifier(string_field(g, "guardrailIdentifier").map(str::to_string))
            .set_guardrail_version(string_field(g, "guardrailVersion").map(str::to_string));
        if let Some(trace) = string_field(g, "trace") {
            b = b.trace(GuardrailTrace::from(trace));
        }
        if let Some(mode) = string_field(g, "streamProcessingMode") {
            b = b.stream_processing_mode(GuardrailStreamProcessingMode::from(mode));
        }
        Some(b.build())
    }
}

fn output_config(config: &Map<String, Value>) -> Result<Option<OutputConfig>, BridgeError> {
    let Some(text_format) = config.get("textFormat").and_then(Value::as_object) else {
        return Ok(None);
    };
    let schema = text_format
        .get("structure")
        .and_then(|s| s.get("jsonSchema"))
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("outputConfig", "textFormat must carry structure.jsonSchema"))?;
    let schema_text = match schema.get("schema") {
        Some(Value::String(s)) => s.clone(),
        Some(v @ Value::Object(_)) => v.to_string(),
        _ => return Err(invalid("outputConfig", "jsonSchema must carry a schema")),
    };
    let mut definition = JsonSchemaDefinition::builder().schema(schema_text);
    if let Some(name) = string_field(schema, "name") {
        definition = definition.name(name);
    }
    if let Some(description) = string_field(schema, "description") {
        definition = definition.description(description);
    }
    let format = OutputFormat::builder()
        .r#type(OutputFormatType::JsonSchema)
        .structure(OutputFormatStructure::JsonSchema(
            definition
                .build()
                .map_err(|e| invalid("outputConfig", &e.to_string()))?,
        ))
        .build()
        .map_err(|e| invalid("outputConfig", &e.to_string()))?;
    Ok(Some(OutputConfig::builder().text_format(format).build()))
}

/// The model id without its cross-region prefix, read through an ARN.
fn base_id(model_id: &str) -> String {
    strip_region_prefix(capability_id(model_id)).to_ascii_lowercase()
}

fn is_nova_2(model_id: &str) -> bool {
    base_id(model_id).starts_with("amazon.nova-2-")
}

/// `reasoning_effort` in the shape each non-Claude publisher takes on
/// Converse, as one `additionalModelRequestFields` member: gpt-oss, Grok,
/// DeepSeek V3, Z.AI GLM and Kimi K3 take it verbatim, OpenAI GPT-5.x/6 as
/// `reasoning.effort`, Amazon Nova 2 as `reasoningConfig`. DeepSeek R1
/// always reasons and rejects every shape, and other publishers have none,
/// so it is dropped for them. Claude's translation lives with the
/// Anthropic request builder.
pub(crate) fn reasoning_field(
    req: &ChatFormat,
    model_id: &str,
) -> Result<Option<(&'static str, Value)>, BridgeError> {
    let effort = match req.extra.get("reasoning_effort") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(effort)) => effort.as_str(),
        Some(_) => return Err(invalid("reasoning_effort", "must be a string")),
    };
    let id = base_id(model_id);
    if id.contains("deepseek.r1") {
        return Ok(None);
    }
    if id.contains("gpt-oss")
        || id.contains("deepseek")
        || id.starts_with("xai.grok-")
        || id.starts_with("zai.")
        || id.starts_with("moonshotai.kimi-k3")
    {
        return Ok(Some((
            "reasoning_effort",
            Value::String(effort.to_string()),
        )));
    }
    if is_openai_gpt(&id) {
        return Ok(Some(("reasoning", serde_json::json!({"effort": effort}))));
    }
    if is_nova_2(model_id) {
        // Nova 2 reasons at low, medium or high, or not at all; any other
        // tier has no `reasoningConfig` and is dropped.
        return Ok(match effort {
            "none" => Some(("reasoningConfig", serde_json::json!({"type": "disabled"}))),
            "low" | "medium" | "high" => Some((
                "reasoningConfig",
                serde_json::json!({"type": "enabled", "maxReasoningEffort": effort}),
            )),
            _ => None,
        });
    }
    Ok(None)
}

/// `top_k` where the publisher takes it on Converse: Claude as an
/// additional field, Nova inside its own `inferenceConfig`. Dropped
/// elsewhere.
pub(crate) fn top_k_field(req: &ChatFormat, model_id: &str) -> Option<(&'static str, Value)> {
    let top_k = match req.extra.get("top_k").filter(|v| !v.is_null()) {
        Some(v) => v.clone(),
        None => req.top_k?.into(),
    };
    let id = base_id(model_id);
    if id.starts_with("anthropic.") {
        Some(("top_k", top_k))
    } else if id.starts_with("amazon.nova") {
        Some(("inferenceConfig", serde_json::json!({"topK": top_k})))
    } else {
        None
    }
}

/// `stop` as Converse `stopSequences`. A lone empty string is skipped, as
/// Converse rejects it. Every other valid `stop` is sent, also to the
/// models AWS refuses it for, whose error then reaches the caller. A
/// Claude caller's native `stop_sequences`, top-level or inside its own
/// `additionalModelRequestFields`, wins even when `null`, as on the
/// Anthropic route: it reaches the model as sent there, and Converse
/// rejects a request that also sets `stopSequences`.
///
/// Converse is built from typed fields, so a `stop` it cannot carry cannot
/// be forwarded for the upstream to reject: anything else that is not a
/// non-empty string or an array of them is refused here, on every model.
pub(crate) fn stop_sequences(
    req: &ChatFormat,
    model_id: &str,
) -> Result<Option<Vec<String>>, BridgeError> {
    const WANT: &str = "must be a non-empty string or an array of non-empty strings";
    let stops = match req.extra.get("stop") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(Value::String(s)) => Some(vec![s.clone()]),
        Some(Value::Array(items)) => {
            let seqs = items
                .iter()
                .map(|v| v.as_str().filter(|s| !s.is_empty()).map(str::to_string))
                .collect::<Option<Vec<String>>>()
                .ok_or_else(|| invalid("stop", WANT))?;
            (!seqs.is_empty()).then_some(seqs)
        }
        Some(_) => return Err(invalid("stop", WANT)),
    };
    if base_id(model_id).starts_with("anthropic.")
        && caller_additional_fields(req, model_id).contains_key("stop_sequences")
    {
        return Ok(None);
    }
    Ok(stops)
}

/// Refuse a `max_completion_tokens` Converse cannot carry: anything but a
/// non-negative integer (or `null`). Converse is built from typed fields,
/// so the value cannot be forwarded for the upstream to reject.
pub(crate) fn check_max_completion_tokens(req: &ChatFormat) -> Result<(), BridgeError> {
    match req.extra.get("max_completion_tokens") {
        None | Some(Value::Null) => Ok(()),
        Some(v) if v.as_u64().is_some() => Ok(()),
        Some(_) => Err(invalid(
            "max_completion_tokens",
            "must be a non-negative integer",
        )),
    }
}

/// The output-token cap: `max_completion_tokens`, else `max_tokens`.
pub(crate) fn max_tokens(req: &ChatFormat) -> Option<u32> {
    req.extra
        .get("max_completion_tokens")
        .and_then(Value::as_u64)
        .map(|v| v.min(u32::MAX as u64) as u32)
        .or(req.max_tokens)
}

/// Whether a request's `response_format` tool is forced on this model. A
/// forced `toolChoice` is what makes the reply JSON, but Converse rejects
/// the field on models without tool-choice support, so the synthetic tool
/// is merely offered there. The set follows the models that support a
/// forced tool choice on Bedrock.
pub(crate) fn forces_json_tool(model_id: &str) -> bool {
    let id = base_id(model_id);
    let prefixed = |p: &str| id.starts_with(p);
    (prefixed("anthropic.") || prefixed("amazon.nova") || prefixed("xai.") || prefixed("zai."))
        || (prefixed("openai.") && !id.contains("gpt-oss-safeguard"))
        || (prefixed("qwen.") && !prefixed("qwen.qwen3-next") && !prefixed("qwen.qwen3-vl"))
        || prefixed("deepseek.v3")
        || prefixed("moonshotai.kimi-k2.5")
        || prefixed("moonshotai.kimi-k3")
        || prefixed("minimax.minimax-m2.")
        || prefixed("nvidia.nemotron-nano-3")
        || prefixed("nvidia.nemotron-super-3")
        || prefixed("mistral.devstral-2")
        || prefixed("mistral.mistral-large-2407")
}

/// Rewrite tool history into text when the request sends no tools: Converse
/// rejects `toolUse` / `toolResult` blocks without a `toolConfig`. An
/// assistant's calls become `[tool call <id>: <name>(<arguments>)]` lines
/// after its text, and a tool result becomes a user turn
/// `[tool result for <id>: <text>]`.
pub(crate) fn neutralize_tool_history(req: &ChatFormat) -> Option<ChatFormat> {
    use aisix_gateway::{ChatMessage, Role};
    let has_tool_blocks = req.messages.iter().any(|m| {
        matches!(m.role, Role::Tool)
            || (matches!(m.role, Role::Assistant)
                && m.extra
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .is_some_and(|c| !c.is_empty()))
    });
    if !has_tool_blocks {
        return None;
    }
    tracing::debug!("request has tool history but sends no tools; rewriting it as text");
    let mut out = req.clone();
    out.messages = req
        .messages
        .iter()
        .map(|m| match m.role {
            Role::Assistant => {
                let Some(calls) = m
                    .extra
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .filter(|c| !c.is_empty())
                else {
                    return m.clone();
                };
                let mut lines: Vec<String> = Vec::new();
                let text = m.content_str();
                if !text.is_empty() {
                    lines.push(text.to_string());
                }
                for call in calls {
                    let f = call.get("function");
                    let name = f
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .filter(|n| !n.is_empty())
                        .unwrap_or("unknown_tool");
                    let args = f
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let label = match call.get("id").and_then(Value::as_str) {
                        Some(id) if !id.is_empty() => format!("tool call {id}"),
                        _ => "tool call".to_string(),
                    };
                    lines.push(format!("[{label}: {name}({args})]"));
                }
                ChatMessage::assistant(lines.join("\n"))
            }
            Role::Tool => {
                let target = m
                    .tool_call_id
                    .as_deref()
                    .or(m.name.as_deref())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("unknown");
                let content = m.content_str();
                let text = content.trim();
                let text = if text.is_empty() {
                    "<non-text tool result omitted>"
                } else {
                    text
                };
                ChatMessage::user(format!("[tool result for {target}: {text}]"))
            }
            _ => m.clone(),
        })
        .collect();
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(extra: Value) -> ChatFormat {
        let mut r = ChatFormat::new("m", vec![aisix_gateway::ChatMessage::user("hi")]);
        r.extra = extra.as_object().unwrap().clone();
        r
    }

    #[test]
    fn unknown_fields_join_the_callers_additional_fields_and_openai_ones_do_not() {
        let r = req(serde_json::json!({
            "additionalModelRequestFields": {"a": 1},
            "custom_knob": true,
            "seed": 7,
            "user": "u",
            "guardrailConfig": {"guardrailIdentifier": "g", "guardrailVersion": "1"},
        }));
        let fields = caller_additional_fields(&r, "meta.llama3-3-70b-instruct-v1:0");
        assert_eq!(
            serde_json::Value::Object(fields),
            serde_json::json!({"a": 1, "custom_knob": true})
        );
    }

    #[test]
    fn reasoning_effort_takes_each_publishers_shape() {
        let r = req(serde_json::json!({"reasoning_effort": "high"}));
        assert_eq!(
            reasoning_field(&r, "openai.gpt-oss-120b-1:0").unwrap(),
            Some(("reasoning_effort", serde_json::json!("high")))
        );
        assert_eq!(
            reasoning_field(&r, "global.openai.gpt-6.1-sol").unwrap(),
            Some(("reasoning", serde_json::json!({"effort": "high"})))
        );
        assert_eq!(
            reasoning_field(&r, "us.amazon.nova-2-lite-v1:0").unwrap(),
            Some((
                "reasoningConfig",
                serde_json::json!({"type": "enabled", "maxReasoningEffort": "high"})
            ))
        );
        assert_eq!(reasoning_field(&r, "deepseek.r1-v1:0").unwrap(), None);
        assert_eq!(
            reasoning_field(&r, "meta.llama3-3-70b-instruct-v1:0").unwrap(),
            None
        );
        assert_eq!(
            reasoning_field(&r, "zai.glm-5.3").unwrap(),
            Some(("reasoning_effort", serde_json::json!("high")))
        );
        assert_eq!(
            reasoning_field(&r, "moonshotai.kimi-k3").unwrap(),
            Some(("reasoning_effort", serde_json::json!("high")))
        );
        // Nova 2: "none" turns reasoning off, a tier it lacks is dropped.
        let none = req(serde_json::json!({"reasoning_effort": "none"}));
        assert_eq!(
            reasoning_field(&none, "amazon.nova-2-lite-v1:0").unwrap(),
            Some(("reasoningConfig", serde_json::json!({"type": "disabled"})))
        );
        let xhigh = req(serde_json::json!({"reasoning_effort": "xhigh"}));
        assert_eq!(
            reasoning_field(&xhigh, "amazon.nova-2-lite-v1:0").unwrap(),
            None
        );
        let wrong = req(serde_json::json!({"reasoning_effort": 3}));
        assert!(matches!(
            reasoning_field(&wrong, "amazon.nova-2-lite-v1:0"),
            Err(BridgeError::InvalidRequest(m)) if m.starts_with("reasoning_effort ")
        ));
    }

    #[test]
    fn openai_fields_reach_the_models_whose_converse_contract_takes_them() {
        let r = req(serde_json::json!({
            "seed": 7,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.25,
            "user": "u",
            "logit_bias": {"1": 2},
            "parallel_tool_calls": false,
            "response_format": {"type": "json_object"},
            "n": 2,
            "logprobs": true,
            "top_logprobs": 2,
        }));
        let keys = |id: &str| {
            let mut k: Vec<String> = caller_additional_fields(&r, id).keys().cloned().collect();
            k.sort();
            k
        };
        for id in [
            "openai.gpt-oss-120b-1:0",
            "openai.gpt-6-sol",
            "xai.grok-4.6",
        ] {
            assert_eq!(
                keys(id),
                [
                    "frequency_penalty",
                    "logit_bias",
                    "parallel_tool_calls",
                    "presence_penalty",
                    "response_format",
                    "seed",
                    "user"
                ],
                "{id}"
            );
        }
        assert_eq!(
            keys("cohere.command-r-plus-v1:0"),
            ["frequency_penalty", "presence_penalty", "seed"]
        );
        assert_eq!(
            keys("ai21.jamba-1-5-large-v1:0"),
            ["frequency_penalty", "presence_penalty"]
        );
        assert!(keys("meta.llama3-3-70b-instruct-v1:0").is_empty());
        // A json_schema format becomes the tool or output config instead.
        let schema = req(serde_json::json!({"response_format": {"type": "json_schema"}}));
        assert!(caller_additional_fields(&schema, "openai.gpt-oss-120b-1:0").is_empty());
    }

    #[test]
    fn a_claude_callers_native_stop_sequences_wins_over_stop() {
        let claude = "us.anthropic.claude-haiku-4-5-20251001-v1:0";
        let both = req(serde_json::json!({"stop": ["delta"], "stop_sequences": ["beta"]}));
        assert_eq!(stop_sequences(&both, claude).unwrap(), None);
        assert_eq!(
            caller_additional_fields(&both, claude).get("stop_sequences"),
            Some(&serde_json::json!(["beta"]))
        );
        let nested = req(serde_json::json!({
            "stop": ["delta"],
            "additionalModelRequestFields": {"stop_sequences": ["beta"]},
        }));
        assert_eq!(stop_sequences(&nested, claude).unwrap(), None);
        // A null native field is still the caller's statement, as on the
        // Anthropic route.
        let null_native = req(serde_json::json!({"stop": ["delta"], "stop_sequences": null}));
        assert_eq!(stop_sequences(&null_native, claude).unwrap(), None);
        assert_eq!(
            caller_additional_fields(&null_native, claude).get("stop_sequences"),
            Some(&serde_json::Value::Null)
        );
        // Other publishers keep `stop` as before.
        assert_eq!(
            stop_sequences(&both, "us.amazon.nova-lite-v1:0").unwrap(),
            Some(vec!["delta".into()])
        );
    }

    #[test]
    fn stop_sequences_reach_every_model() {
        let r = req(serde_json::json!({"stop": ["END"]}));
        for id in [
            "meta.llama3-3-70b-instruct-v1:0",
            "openai.gpt-oss-20b-1:0",
            "us.openai.gpt-oss-120b-1:0",
            "us.openai.gpt-6-luna",
            "xai.grok-4.6",
        ] {
            assert_eq!(
                stop_sequences(&r, id).unwrap(),
                Some(vec!["END".into()]),
                "{id}"
            );
        }
        assert_eq!(
            stop_sequences(
                &req(serde_json::json!({"stop": ""})),
                "amazon.nova-lite-v1:0"
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn an_invalid_stop_or_max_completion_tokens_is_refused() {
        let nova = "amazon.nova-lite-v1:0";
        for id in [
            nova,
            "openai.gpt-oss-20b-1:0",
            "us.openai.gpt-6-luna",
            "xai.grok-4.6",
        ] {
            for stop in [
                serde_json::json!(5),
                serde_json::json!({"a": 1}),
                serde_json::json!(["a", 1]),
                serde_json::json!(["a", ""]),
            ] {
                let err = stop_sequences(&req(serde_json::json!({"stop": stop})), id).unwrap_err();
                assert!(
                    matches!(&err, BridgeError::InvalidRequest(m) if m.starts_with("stop ")),
                    "{id} {stop}: {err:?}"
                );
            }
        }
        assert_eq!(
            stop_sequences(&req(serde_json::json!({"stop": null})), nova).unwrap(),
            None
        );
        assert_eq!(
            stop_sequences(&req(serde_json::json!({"stop": []})), nova).unwrap(),
            None
        );
        for cap in [
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("100"),
        ] {
            let err = check_max_completion_tokens(&req(serde_json::json!({
                "max_completion_tokens": cap
            })))
            .unwrap_err();
            assert!(
                matches!(&err, BridgeError::InvalidRequest(m) if m.starts_with("max_completion_tokens ")),
                "{cap}: {err:?}"
            );
        }
        check_max_completion_tokens(&req(serde_json::json!({"max_completion_tokens": 0}))).unwrap();
    }
}
