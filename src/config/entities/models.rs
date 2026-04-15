use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use serde::{Deserialize, Serialize, de::Error};
use serde_json::json;
use utoipa::ToSchema;

use super::{ConfigProvider, EntityStore, IndexFn};
use crate::{
    config::entities::{
        ResourceEntry,
        types::{HasRateLimit, RateLimit, RateLimitMetric},
    },
    gateway::providers::{configs, identifiers},
    utils::jsonschema::format_evaluation_error,
};

static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema#",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "model": {
                "type": "string",
                "pattern": MODELS_PATTERN
            },
            "provider_config": {"type": "object"},
            "timeout": {
                "type": "integer",
                "minimum": 0
            },
            "rate_limit": {"type": "object"}
        },
        "required": ["name", "model", "provider_config"],
        "additionalProperties": false,
        "allOf": [
            {
                "if": {
                    "properties": {
                        "model": { "pattern": "^(anthropic|deepseek|gemini|openai)/.+$" }
                    },
                    "required": ["model"]
                },
                "then": {
                    "properties": {
                        "provider_config": { "$ref": "#/$defs/openai_compatible" }
                    }
                }
            }
        ],
        "$defs": {
            "openai_compatible": {
                "type": "object",
                "required": ["api_key"],
                "properties": {
                    "api_key": {"type": "string"},
                    "api_base": {"type": "string"}
                },
                "additionalProperties": false
            }
        }
    })
});
pub static SCHEMA_VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| jsonschema::validator_for(&SCHEMA).expect("Invalid JSON schema for Model"));

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(untagged)]
pub enum ProviderConfig {
    Anthropic(configs::AnthropicProviderConfig),
    DeepSeek(configs::DeepSeekProviderConfig),
    Gemini(configs::GeminiProviderConfig),
    OpenAI(configs::OpenAIProviderConfig),
}

impl ProviderConfig {
    pub fn from_json(
        provider: &str,
        json_value: &serde_json::Value,
    ) -> Result<Self, serde_json::Error> {
        match provider {
            identifiers::ANTHROPIC => {
                let config =
                    serde_json::from_value::<configs::AnthropicProviderConfig>(json_value.clone())?;
                Ok(ProviderConfig::Anthropic(config))
            }
            identifiers::DEEPSEEK => {
                let config =
                    serde_json::from_value::<configs::DeepSeekProviderConfig>(json_value.clone())?;
                Ok(ProviderConfig::DeepSeek(config))
            }
            identifiers::GEMINI => {
                let config =
                    serde_json::from_value::<configs::GeminiProviderConfig>(json_value.clone())?;
                Ok(ProviderConfig::Gemini(config))
            }
            identifiers::OPENAI => {
                let config =
                    serde_json::from_value::<configs::OpenAIProviderConfig>(json_value.clone())?;
                Ok(ProviderConfig::OpenAI(config))
            }
            _ => Err(serde_json::Error::custom(format!(
                "Unknown provider type: {}",
                provider
            ))),
        }
    }
}

pub static MODELS_PATTERN: &str = "^(anthropic|deepseek|gemini|openai)/.+$";
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ProviderModel {
    #[serde(skip)]
    pub provider: String,
    #[serde(skip)]
    pub name: String,

    #[serde(rename = "model")]
    #[schema(pattern = "^(anthropic|deepseek|gemini|openai)/.+$")]
    pub original_model: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Model {
    pub name: String,

    #[serde(flatten)]
    #[schema(inline)]
    pub model: ProviderModel,
    pub provider_config: ProviderConfig,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimit>,
}

impl<'de> Deserialize<'de> for Model {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ModelRaw {
            name: String,
            model: String,
            provider_config: serde_json::Value,
            timeout: Option<u64>,
            rate_limit: Option<RateLimit>,
        }

        let raw = ModelRaw::deserialize(deserializer)?;

        let mut model_parts = raw.model.split('/');
        let provider = model_parts.next().unwrap_or("").to_lowercase();
        let provider_model = model_parts.next().unwrap_or("").to_string();
        if provider.is_empty() || provider_model.is_empty() {
            return Err(D::Error::custom(format!(
                "Invalid model format for {}: {}",
                raw.name, raw.model
            )));
        }

        let provider_config = match ProviderConfig::from_json(&provider, &raw.provider_config) {
            Ok(config) => config,
            Err(err) => {
                return Err(D::Error::custom(format!(
                    "Failed to parse provider_config for model {}: {}",
                    raw.name, err
                )));
            }
        };

        Ok(Model {
            name: raw.name,
            model: ProviderModel {
                provider,
                name: provider_model,
                original_model: raw.model,
            },
            provider_config,
            timeout: raw.timeout,
            rate_limit: raw.rate_limit,
        })
    }
}

impl HasRateLimit for ResourceEntry<Model> {
    fn rate_limit(&self) -> Option<RateLimit> {
        self.rate_limit.clone()
    }

    fn rate_limit_key(&self, metric: RateLimitMetric) -> String {
        format!("model:{}:{}", self.name, metric)
    }
}

fn validate(key: &str, value: &Model) -> Result<(), String> {
    let evaluation = SCHEMA_VALIDATOR.evaluate(
        &serde_json::to_value(value)
            .map_err(|e| format!("Failed to serialize model for validation: {}", e))?,
    );
    if !evaluation.flag().valid {
        return Err(format!(
            r#"JSON schema validation error on model "{key}": {}"#,
            format_evaluation_error(&evaluation)
        ));
    }

    Ok(())
}

#[derive(Clone)]
pub struct ModelsStore {
    store: EntityStore<Model>,
}

static INDEX_FNS: &[IndexFn<Model>] = &[("by_name", |m: &Model| Some(m.name.clone()))];

impl ModelsStore {
    pub async fn new(provider: Arc<dyn ConfigProvider + Send + Sync>) -> Self {
        Self {
            store: EntityStore::new(provider, "/models/", "models", Some(validate), INDEX_FNS)
                .await,
        }
    }

    pub fn list(&self) -> Arc<HashMap<String, ResourceEntry<Model>>> {
        self.store.list()
    }

    pub fn get_by_name(&self, name: &str) -> Option<ResourceEntry<Model>> {
        self.store.get_by_secondary("by_name", name)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{SCHEMA, SCHEMA_VALIDATOR, format_evaluation_error};
    use crate::config::entities::models::MODELS_PATTERN;

    #[test]
    fn test_valid_jsonschema() {
        assert!(jsonschema::meta::is_valid(&SCHEMA));
    }

    #[rstest::rstest]
    #[case::ok(json!({
        "name": "test",
        "model": "openai/gpt-5",
        "provider_config": { "api_key": "test_key" },
    }), true, None)]
    #[case::missing_name(json!({
        "model": "openai/gpt-5",
        "provider_config": { "api_key": "test_key" },
    }), false, Some(r#"property "/" validation failed: "name" is a required property"#.to_string()))]
    #[case::missing_model(json!({
        "name": "test",
        "provider_config": {},
    }), false, Some(r#"property "/" validation failed: "model" is a required property"#.to_string()))]
    #[case::missing_provider_config(json!({
        "name": "test",
        "model": "deepseek/deepseek-chat",
    }), false, Some(r#"property "/" validation failed: "provider_config" is a required property"#.to_string()))]
    #[case::invalid_name_type(json!({
        "name": 123,
        "model": "openai/gpt-5",
        "provider_config": { "api_key": "test_key" },
    }), false, Some(r#"property "/name" validation failed: 123 is not of type "string""#.to_string()))]
    #[case::invalid_model_type(json!({
        "name": "test",
        "model": 123,
        "provider_config": {},
    }), false, Some(r#"property "/model" validation failed: 123 is not of type "string"
property "/provider_config" validation failed: "api_key" is a required property"#.to_string()))]
    #[case::invalid_model_pattern(json!({
        "name": "test",
        "model": "invalid",
        "provider_config": {},
    }), false, Some(format!(r#"property "/model" validation failed: "invalid" does not match "{}""#, MODELS_PATTERN)))]
    #[case::invalid_provider_config_type(json!({
        "name": "test",
        "model": "openai/gpt-5",
        "provider_config": 123,
    }), false, Some(r#"property "/provider_config" validation failed: 123 is not of type "object"
property "/provider_config" validation failed: 123 is not of type "object""#.to_string()))]
    #[case::invalid_provider_config_for_specific_vendor(json!({
        "name": "test",
        "model": "deepseek/deepseek-chat",
        "provider_config": {},
    }), false, Some(r#"property "/provider_config" validation failed: "api_key" is a required property"#.to_string()))]
    #[case::invalid_provider_config_additional_property(json!({
        "name": "test",
        "model": "deepseek/deepseek-chat",
        "provider_config": {
            "api_key": "test_key",
            "additional": "not allowed"
        },
    }), false, Some(r#"property "/provider_config" validation failed: Additional properties are not allowed ('additional' was unexpected)"#.to_string()))]
    #[case::invalid_root_additional_property(json!({
        "name": "test",
        "model": "deepseek/deepseek-chat",
        "provider_config": { "api_key": "test_key" },
        "extra": "not allowed"
    }), false, Some(r#"property "/" validation failed: Additional properties are not allowed ('extra' was unexpected)"#.to_string()))]
    #[case::ok_with_rate_limit(json!({
        "name": "test",
        "model": "openai/gpt-5",
        "provider_config": { "api_key": "test_key" },
        "rate_limit": {},
    }), true, None)]
    #[case::invalid_rate_limit_type(json!({
        "name": "test",
        "model": "openai/gpt-5",
        "provider_config": { "api_key": "test_key" },
        "rate_limit": 123,
    }), false, Some(r#"property "/rate_limit" validation failed: 123 is not of type "object""#.to_string()))]
    fn schemas(
        #[case] input: serde_json::Value,
        #[case] ok: bool,
        #[case] expected_error: Option<String>,
    ) {
        let evaluation = SCHEMA_VALIDATOR.evaluate(&input);

        assert_eq!(evaluation.flag().valid, ok, "unexpected evaluation result");
        if !ok {
            assert_eq!(
                format_evaluation_error(&evaluation),
                expected_error.unwrap(),
                "unexpected error message"
            );
        }
    }
}
