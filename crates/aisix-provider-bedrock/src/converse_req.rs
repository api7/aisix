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
    OutputFormatType, PerformanceConfigLatency, PerformanceConfiguration, ServiceTier,
    ServiceTierType,
};
use serde_json::{Map, Value};

use crate::bridge::{capability_id, strip_region_prefix};
use crate::openai_compat::{chat_completions_family, is_openai_gpt};

/// Top-level request fields that are never forwarded as
/// `additionalModelRequestFields`: every OpenAI Chat Completions parameter,
/// the fields the gateway translates or consumes itself, and the
/// Bedrock-native blocks that have their own place in the request.
/// Anything else the caller sent is a model-specific parameter, and reaches
/// the model there.
const NOT_FORWARDED: &[&str] = &[
    // OpenAI Chat Completions
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "metadata",
    "modalities",
    "model",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning_effort",
    "response_format",
    "safety_identifier",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "verbosity",
    "web_search_options",
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
    // Bedrock-native request blocks
    "additionalModelRequestFields",
    "guardrailConfig",
    "performanceConfig",
    "serviceTier",
    "requestMetadata",
    "outputConfig",
];

/// The caller's own `additionalModelRequestFields`: the object they sent
/// under that name, merged with every top-level field that is not an
/// OpenAI parameter or one the gateway handles.
pub(crate) fn caller_additional_fields(req: &ChatFormat) -> Map<String, Value> {
    let mut fields = match req.extra.get("additionalModelRequestFields") {
        Some(Value::Object(m)) => m.clone(),
        _ => Map::new(),
    };
    for (key, value) in &req.extra {
        if !NOT_FORWARDED.contains(&key.as_str()) {
            fields.insert(key.clone(), value.clone());
        }
    }
    fields
}

/// The Bedrock-native blocks of a Converse request.
#[derive(Default)]
pub(crate) struct NativeBlocks {
    pub guardrail: Option<Map<String, Value>>,
    pub performance: Option<PerformanceConfiguration>,
    pub service_tier: Option<ServiceTier>,
    pub request_metadata: Vec<(String, String)>,
    pub output_config: Option<OutputConfig>,
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
            None => match req.extra.get("service_tier").and_then(Value::as_str) {
                Some("auto") => Some("default".to_string()),
                Some(t @ ("default" | "flex" | "priority")) => Some(t.to_string()),
                _ => None,
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
/// Converse, as one `additionalModelRequestFields` member: gpt-oss and
/// DeepSeek V3 take it verbatim, OpenAI GPT-5.x/6 as `reasoning.effort`,
/// Amazon Nova 2 as `reasoningConfig`. DeepSeek R1 always reasons and
/// rejects every shape, and other publishers have none, so it is dropped
/// for them. Claude's translation lives with the Anthropic request builder.
pub(crate) fn reasoning_field(
    req: &ChatFormat,
    model_id: &str,
) -> Result<Option<(&'static str, Value)>, BridgeError> {
    let Some(effort) = req.extra.get("reasoning_effort").and_then(Value::as_str) else {
        return Ok(None);
    };
    let id = base_id(model_id);
    if id.contains("deepseek.r1") {
        return Ok(None);
    }
    if id.contains("gpt-oss") || id.contains("deepseek") {
        return Ok(Some((
            "reasoning_effort",
            Value::String(effort.to_string()),
        )));
    }
    if is_openai_gpt(&id) {
        return Ok(Some(("reasoning", serde_json::json!({"effort": effort}))));
    }
    if is_nova_2(model_id) {
        if !matches!(effort, "low" | "medium" | "high") {
            return Err(BridgeError::InvalidRequest(format!(
                "Invalid reasoning_effort value '{effort}' for Nova 2 models. \
                 Supported values: low, medium, high"
            )));
        }
        return Ok(Some((
            "reasoningConfig",
            serde_json::json!({"type": "enabled", "maxReasoningEffort": effort}),
        )));
    }
    Ok(None)
}

/// `top_k` where the publisher takes it on Converse: Claude as an
/// additional field, Nova inside its own `inferenceConfig`. Dropped
/// elsewhere.
pub(crate) fn top_k_field(req: &ChatFormat, model_id: &str) -> Option<(&'static str, Value)> {
    let top_k = req.extra.get("top_k").filter(|v| !v.is_null())?.clone();
    let id = base_id(model_id);
    if id.starts_with("anthropic.") {
        Some(("top_k", top_k))
    } else if id.starts_with("amazon.nova") {
        Some(("inferenceConfig", serde_json::json!({"topK": top_k})))
    } else {
        None
    }
}

/// `stop` as Converse `stopSequences`. An empty string is skipped, as
/// Converse rejects it. Grok and GPT 5.6 and newer refuse stop sequences on
/// every route, so they get none.
pub(crate) fn stop_sequences(req: &ChatFormat, model_id: &str) -> Option<Vec<String>> {
    if chat_completions_family(capability_id(model_id)) {
        return None;
    }
    match req.extra.get("stop")? {
        Value::String(s) if !s.is_empty() => Some(vec![s.clone()]),
        Value::Array(items) => {
            let seqs: Vec<String> = items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect();
            (!seqs.is_empty()).then_some(seqs)
        }
        _ => None,
    }
}

/// The output-token cap: `max_completion_tokens`, else `max_tokens`. The
/// OpenAI-family models on Bedrock (and Kimi K3) reject a cap below 16,
/// so a smaller one is raised to it.
pub(crate) fn max_tokens(req: &ChatFormat, model_id: &str) -> Option<u32> {
    let cap = req
        .extra
        .get("max_completion_tokens")
        .and_then(Value::as_u64)
        .map(|v| v.min(u32::MAX as u64) as u32)
        .or(req.max_tokens)?;
    let id = capability_id(model_id);
    let needs_floor =
        is_openai_gpt(id) || id.contains("xai.grok-") || id.contains("moonshotai.kimi-k3");
    Some(if needs_floor { cap.max(16) } else { cap })
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
        let fields = caller_additional_fields(&r);
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
        let bad = req(serde_json::json!({"reasoning_effort": "xhigh"}));
        assert!(reasoning_field(&bad, "amazon.nova-2-lite-v1:0").is_err());
    }

    #[test]
    fn stop_sequences_skip_the_families_that_refuse_them() {
        let r = req(serde_json::json!({"stop": "END"}));
        assert_eq!(
            stop_sequences(&r, "meta.llama3-3-70b-instruct-v1:0"),
            Some(vec!["END".into()])
        );
        assert_eq!(stop_sequences(&r, "us.openai.gpt-6-luna"), None);
        assert_eq!(
            stop_sequences(
                &req(serde_json::json!({"stop": ""})),
                "amazon.nova-lite-v1:0"
            ),
            None
        );
    }
}
