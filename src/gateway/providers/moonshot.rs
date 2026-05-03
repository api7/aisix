//! Moonshot AI currently publishes a single official OpenAI-compatible endpoint,
//! so both moonshotai and moonshotai-cn share the same wire behavior.
//!
//! Docs:
//! - https://platform.kimi.com/docs/api/overview.md
//! - https://platform.kimi.com/docs/api/chat.md
//! - https://platform.kimi.com/docs/api/models-overview.md
//! - https://platform.kimi.com/docs/guide/migrating-from-openai-to-kimi.md

use http::{HeaderMap, HeaderValue, header::AUTHORIZATION};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::gateway::{
    error::{GatewayError, Result},
    provider_instance::ProviderAuth,
    traits::{ChatTransform, ProviderCapabilities, ProviderMeta},
    types::openai::ChatCompletionRequest,
};

pub const IDENTIFIER: &str = "moonshotai";
pub const CN_IDENTIFIER: &str = "moonshotai-cn";

const DEFAULT_BASE_URL: &str = "https://api.moonshot.cn/v1";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MoonshotAiProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MoonshotAiCnProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct MoonshotAi;
pub struct MoonshotAiCn;

impl ProviderMeta for MoonshotAi {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        DEFAULT_BASE_URL
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        build_auth_headers(self.name(), auth)
    }
}

impl ProviderMeta for MoonshotAiCn {
    fn name(&self) -> &'static str {
        CN_IDENTIFIER
    }

    fn default_base_url(&self) -> &'static str {
        DEFAULT_BASE_URL
    }

    fn build_auth_headers(&self, auth: &ProviderAuth) -> Result<HeaderMap> {
        build_auth_headers(self.name(), auth)
    }
}

impl ChatTransform for MoonshotAi {
    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        transform_request(request)
    }
}

impl ChatTransform for MoonshotAiCn {
    fn transform_request(&self, request: &ChatCompletionRequest) -> Result<Value> {
        transform_request(request)
    }
}

impl ProviderCapabilities for MoonshotAi {}

impl ProviderCapabilities for MoonshotAiCn {}

fn build_auth_headers(identifier: &str, auth: &ProviderAuth) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    let value = HeaderValue::from_str(&format!("Bearer {}", auth.api_key_for(identifier)?))
        .map_err(|error| GatewayError::Validation(error.to_string()))?;
    headers.insert(AUTHORIZATION, value);
    Ok(headers)
}

fn transform_request(request: &ChatCompletionRequest) -> Result<Value> {
    let mut body = serde_json::to_value(request)
        .map_err(|error| GatewayError::Transform(error.to_string()))?;

    let Value::Object(map) = &mut body else {
        return Ok(body);
    };

    convert_legacy_functions(map)?;
    convert_legacy_function_call(map)?;
    validate_tool_choice(map)?;

    let model = map
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| GatewayError::Validation("moonshot providers require a string model".into()))?
        .to_string();

    apply_model_specific_quirks(map, model.as_str());
    validate_generic_constraints(map)?;

    Ok(body)
}

fn convert_legacy_functions(map: &mut serde_json::Map<String, Value>) -> Result<()> {
    let Some(functions) = map.remove("functions") else {
        return Ok(());
    };

    if map.contains_key("tools") {
        return Ok(());
    }

    match functions {
        Value::Null => Ok(()),
        Value::Array(functions) => {
            let tools = functions
                .into_iter()
                .map(|function| {
                    let mut tool = serde_json::Map::new();
                    tool.insert("type".into(), Value::String("function".into()));
                    tool.insert("function".into(), function);
                    Value::Object(tool)
                })
                .collect();
            map.insert("tools".into(), Value::Array(tools));
            Ok(())
        }
        _ => Err(GatewayError::Validation(
            "moonshot providers expect legacy functions to be an array".into(),
        )),
    }
}

fn convert_legacy_function_call(map: &mut serde_json::Map<String, Value>) -> Result<()> {
    let Some(function_call) = map.remove("function_call") else {
        return Ok(());
    };

    if map.contains_key("tool_choice") {
        return Ok(());
    }

    match function_call {
        Value::Null => Ok(()),
        Value::String(mode) if matches!(mode.as_str(), "none" | "auto") => {
            map.insert("tool_choice".into(), Value::String(mode));
            Ok(())
        }
        Value::String(mode) => Err(GatewayError::Validation(format!(
            "moonshot providers only document tool_choice values \"none\" and \"auto\"; unsupported legacy function_call value \"{mode}\""
        ))),
        Value::Object(_) => Err(GatewayError::Validation(
            "moonshot providers do not document forced function_call objects; use tools with tool_choice set to \"auto\" or \"none\"".into(),
        )),
        _ => Err(GatewayError::Validation(
            "moonshot providers expect legacy function_call to be a string or null".into(),
        )),
    }
}

fn validate_tool_choice(map: &serde_json::Map<String, Value>) -> Result<()> {
    let Some(tool_choice) = map.get("tool_choice") else {
        return Ok(());
    };

    match tool_choice {
        Value::String(mode) if matches!(mode.as_str(), "none" | "auto") => Ok(()),
        Value::String(mode) if mode == "required" => Err(GatewayError::Validation(
            "moonshot providers do not support tool_choice=\"required\"".into(),
        )),
        Value::String(mode) => Err(GatewayError::Validation(format!(
            "moonshot providers only document tool_choice values \"none\" and \"auto\", got \"{mode}\""
        ))),
        Value::Object(_) => Err(GatewayError::Validation(
            "moonshot providers do not document object-form tool_choice".into(),
        )),
        Value::Null => Ok(()),
        _ => Err(GatewayError::Validation(
            "moonshot providers expect tool_choice to be a string, object, or null".into(),
        )),
    }
}

fn apply_model_specific_quirks(map: &mut serde_json::Map<String, Value>, model: &str) {
    match model {
        // kimi-k2.6 exposes thinking as an extra-body extension and documents
        // sampling controls as fixed rather than user-tunable.
        "kimi-k2.6" => {
            map.remove("temperature");
            map.remove("top_p");
            map.remove("n");
            map.remove("presence_penalty");
            map.remove("frequency_penalty");
        }
        // kimi-k2.5 documents a fixed temperature that depends on thinking mode,
        // so omit user-supplied temperature and let the model choose the correct value.
        "kimi-k2.5" => {
            map.remove("temperature");
        }
        _ => {}
    }
}

fn validate_generic_constraints(map: &serde_json::Map<String, Value>) -> Result<()> {
    if let Some(temperature) = map.get("temperature").and_then(Value::as_f64)
        && !(0.0..=1.0).contains(&temperature)
    {
        return Err(GatewayError::Validation(format!(
            "moonshot providers require temperature to be within [0, 1], got {temperature}"
        )));
    }

    if let Some(top_p) = map.get("top_p").and_then(Value::as_f64)
        && !(0.0..=1.0).contains(&top_p)
    {
        return Err(GatewayError::Validation(format!(
            "moonshot providers require top_p to be within [0, 1], got {top_p}"
        )));
    }

    if let Some(n) = map.get("n").and_then(Value::as_u64) {
        if !(1..=5).contains(&n) {
            return Err(GatewayError::Validation(format!(
                "moonshot providers require n to be within [1, 5], got {n}"
            )));
        }

        if map.get("temperature").and_then(Value::as_f64) == Some(0.0) && n > 1 {
            return Err(GatewayError::Validation(
                "moonshot providers reject n > 1 when temperature is 0".into(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::{MoonshotAi, MoonshotAiCn};
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{ChatTransform, ProviderMeta},
        types::openai::ChatCompletionRequest,
    };

    #[test]
    fn provider_metadata_and_urls_are_correct() {
        let global = MoonshotAi;
        let cn = MoonshotAiCn;
        let global_headers = global
            .build_auth_headers(&ProviderAuth::ApiKey("moonshot-key".into()))
            .unwrap();
        let cn_headers = cn
            .build_auth_headers(&ProviderAuth::ApiKey("moonshot-cn-key".into()))
            .unwrap();

        assert_eq!(global.name(), "moonshotai");
        assert_eq!(cn.name(), "moonshotai-cn");
        assert_eq!(global.default_base_url(), "https://api.moonshot.cn/v1");
        assert_eq!(cn.default_base_url(), "https://api.moonshot.cn/v1");
        assert_eq!(global_headers["authorization"], "Bearer moonshot-key");
        assert_eq!(cn_headers["authorization"], "Bearer moonshot-cn-key");
        assert_eq!(
            global.build_url(global.default_base_url(), "ignored"),
            "https://api.moonshot.cn/v1/chat/completions"
        );
        assert_eq!(
            cn.build_url(cn.default_base_url(), "ignored"),
            "https://api.moonshot.cn/v1/chat/completions"
        );
    }

    #[test]
    fn transform_request_converts_legacy_function_fields() {
        let provider = MoonshotAi;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "moonshot-v1-128k",
            "messages": [{"role": "user", "content": "hello"}],
            "functions": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}}
                }
            ],
            "function_call": "auto"
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed.get("functions"), None);
        assert_eq!(transformed.get("function_call"), None);
        assert_eq!(transformed["tool_choice"], "auto");
        assert_eq!(transformed["tools"][0]["type"], "function");
        assert_eq!(transformed["tools"][0]["function"]["name"], "search");
    }

    #[test]
    fn transform_request_rejects_required_tool_choice() {
        let provider = MoonshotAi;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "moonshot-v1-128k",
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": "required"
        }))
        .unwrap();

        let error = provider.transform_request(&request).unwrap_err();

        assert_matches!(
            error,
            crate::gateway::error::GatewayError::Validation(message)
                if message.contains("tool_choice") && message.contains("required")
        );
    }

    #[test]
    fn transform_request_strips_fixed_k26_sampling_fields() {
        let provider = MoonshotAi;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "kimi-k2.6",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.8,
            "top_p": 0.7,
            "n": 3,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "thinking": {"type": "disabled"}
        }))
        .unwrap();

        let transformed = provider.transform_request(&request).unwrap();

        assert_eq!(transformed.get("temperature"), None);
        assert_eq!(transformed.get("top_p"), None);
        assert_eq!(transformed.get("n"), None);
        assert_eq!(transformed.get("presence_penalty"), None);
        assert_eq!(transformed.get("frequency_penalty"), None);
        assert_eq!(transformed["thinking"]["type"], "disabled");
    }

    #[test]
    fn transform_request_rejects_temperature_above_one() {
        let provider = MoonshotAiCn;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "moonshot-v1-128k",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 1.5
        }))
        .unwrap();

        let error = provider.transform_request(&request).unwrap_err();

        assert_matches!(
            error,
            crate::gateway::error::GatewayError::Validation(message)
                if message.contains("temperature") && message.contains("[0, 1]")
        );
    }

    #[test]
    fn transform_request_rejects_multiple_choices_when_temperature_is_zero() {
        let provider = MoonshotAi;
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "moonshot-v1-128k",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.0,
            "n": 2
        }))
        .unwrap();

        let error = provider.transform_request(&request).unwrap_err();

        assert_matches!(
            error,
            crate::gateway::error::GatewayError::Validation(message)
                if message.contains("n > 1") && message.contains("temperature is 0")
        );
    }
}
