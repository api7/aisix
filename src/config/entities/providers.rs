use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::{ConfigProvider, EntityStore, ResourceEntry};
use crate::{
    gateway::providers::{configs, identifiers},
    utils::jsonschema::format_evaluation_error,
};

static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("providers-schema.json"))
        .expect("Invalid JSON document for Provider schema")
});
pub static SCHEMA_VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| jsonschema::validator_for(&SCHEMA).expect("Invalid JSON schema for Provider"));

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", content = "config")]
pub enum ProviderConfig {
    #[serde(rename = "anthropic")]
    Anthropic(configs::AnthropicProviderConfig),
    #[serde(rename = "azure")]
    Azure(configs::AzureProviderConfig),
    #[serde(rename = "bedrock")]
    Bedrock(configs::BedrockProviderConfig),
    #[serde(rename = "cohere")]
    Cohere(configs::CohereProviderConfig),
    #[serde(rename = "deepseek")]
    DeepSeek(configs::DeepSeekProviderConfig),
    #[serde(rename = "fireworks-ai")]
    FireworksAi(configs::FireworksAiProviderConfig),
    #[serde(rename = "gemini")]
    Gemini(configs::GeminiProviderConfig),
    #[serde(rename = "groq")]
    Groq(configs::GroqProviderConfig),
    #[serde(rename = "xai")]
    Xai(configs::XaiProviderConfig),
    #[serde(rename = "mistral")]
    Mistral(configs::MistralProviderConfig),
    #[serde(rename = "moonshotai")]
    MoonshotAi(configs::MoonshotAiProviderConfig),
    #[serde(rename = "moonshotai-cn")]
    MoonshotAiCn(configs::MoonshotAiCnProviderConfig),
    #[serde(rename = "openai")]
    OpenAI(configs::OpenAIProviderConfig),
    #[serde(rename = "openrouter")]
    OpenRouter(configs::OpenRouterProviderConfig),
    #[serde(rename = "zhipuai")]
    ZhipuAi(configs::ZhipuAiProviderConfig),
}

impl ProviderConfig {
    pub fn provider_type(&self) -> &'static str {
        match self {
            Self::Anthropic(_) => identifiers::ANTHROPIC,
            Self::Azure(_) => identifiers::AZURE,
            Self::Bedrock(_) => identifiers::BEDROCK,
            Self::Cohere(_) => identifiers::COHERE,
            Self::DeepSeek(_) => identifiers::DEEPSEEK,
            Self::FireworksAi(_) => identifiers::FIREWORKS_AI,
            Self::Gemini(_) => identifiers::GEMINI,
            Self::Groq(_) => identifiers::GROQ,
            Self::Xai(_) => identifiers::XAI,
            Self::Mistral(_) => identifiers::MISTRAL,
            Self::MoonshotAi(_) => identifiers::MOONSHOT_AI,
            Self::MoonshotAiCn(_) => identifiers::MOONSHOT_AI_CN,
            Self::OpenAI(_) => identifiers::OPENAI,
            Self::OpenRouter(_) => identifiers::OPENROUTER,
            Self::ZhipuAi(_) => identifiers::ZHIPUAI,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Provider {
    pub name: String,

    #[serde(flatten)]
    #[schema(inline)]
    pub provider: ProviderConfig,
}

impl Provider {
    pub fn provider_type(&self) -> &'static str {
        self.provider.provider_type()
    }
}

fn validate(key: &str, value: &Provider) -> Result<(), String> {
    let evaluation = SCHEMA_VALIDATOR.evaluate(
        &serde_json::to_value(value)
            .map_err(|e| format!("Failed to serialize provider for validation: {}", e))?,
    );
    if !evaluation.flag().valid {
        return Err(format!(
            r#"JSON schema validation error on provider "{key}": {}"#,
            format_evaluation_error(&evaluation)
        ));
    }

    Ok(())
}

#[derive(Clone)]
pub struct ProvidersStore {
    store: EntityStore<Provider>,
}

impl ProvidersStore {
    pub async fn new(provider: Arc<dyn ConfigProvider + Send + Sync>) -> Self {
        Self {
            store: EntityStore::new(provider, "/providers/", "providers", Some(validate), &[])
                .await,
        }
    }

    pub fn list(&self) -> Arc<HashMap<String, ResourceEntry<Provider>>> {
        self.store.list()
    }

    pub fn get_by_id(&self, id: &str) -> Option<ResourceEntry<Provider>> {
        self.store.get(id)
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::{SCHEMA, SCHEMA_VALIDATOR, format_evaluation_error};

    #[test]
    fn test_valid_jsonschema() {
        assert!(jsonschema::meta::is_valid(&SCHEMA));
    }

    #[rstest::rstest]
    #[case::openai_ok(json!({
        "name": "openai-primary",
        "type": "openai",
        "config": { "api_key": "test_key" }
    }), true, None)]
    #[case::azure_ok(json!({
        "name": "azure-primary",
        "type": "azure",
        "config": {
            "api_key": "test_key",
            "api_base": "https://example-resource.openai.azure.com"
        }
    }), true, None)]
    #[case::bedrock_ok(json!({
        "name": "bedrock-primary",
        "type": "bedrock",
        "config": {
            "region": "us-east-1",
            "access_key_id": "AKIA123",
            "secret_access_key": "secret"
        }
    }), true, None)]
    #[case::cohere_ok(json!({
        "name": "cohere-primary",
        "type": "cohere",
        "config": { "api_key": "test_key" }
    }), true, None)]
    #[case::fireworks_ai_ok(json!({
        "name": "fireworks-primary",
        "type": "fireworks-ai",
        "config": { "api_key": "test_key" }
    }), true, None)]
    #[case::openrouter_ok(json!({
        "name": "openrouter-primary",
        "type": "openrouter",
        "config": { "api_key": "test_key" }
    }), true, None)]
    #[case::groq_ok(json!({
        "name": "groq-primary",
        "type": "groq",
        "config": { "api_key": "test_key" }
    }), true, None)]
    #[case::xai_ok(json!({
        "name": "xai-primary",
        "type": "xai",
        "config": { "api_key": "test_key" }
    }), true, None)]
    #[case::mistral_ok(json!({
        "name": "mistral-primary",
        "type": "mistral",
        "config": { "api_key": "test_key" }
    }), true, None)]
    #[case::moonshotai_ok(json!({
        "name": "moonshot-primary",
        "type": "moonshotai",
        "config": { "api_key": "test_key" }
    }), true, None)]
    #[case::moonshotai_cn_ok(json!({
        "name": "moonshot-cn-primary",
        "type": "moonshotai-cn",
        "config": { "api_key": "test_key" }
    }), true, None)]
    #[case::zhipuai_ok(json!({
        "name": "zhipu-primary",
        "type": "zhipuai",
        "config": { "api_key": "test_key" }
    }), true, None)]
    #[case::missing_type(json!({
        "name": "openai-primary",
        "config": { "api_key": "test_key" }
    }), false, Some(r#"property "/" validation failed: "type" is a required property"#.to_string()))]
    #[case::invalid_openai_config(json!({
        "name": "openai-primary",
        "type": "openai",
        "config": {}
    }), false, Some(r#"property "/config" validation failed: "api_key" is a required property"#.to_string()))]
    #[case::invalid_azure_config(json!({
        "name": "azure-primary",
        "type": "azure",
        "config": { "api_key": "test_key" }
    }), false, Some(r#"property "/config" validation failed: "api_base" is a required property"#.to_string()))]
    #[case::invalid_bedrock_config(json!({
        "name": "bedrock-primary",
        "type": "bedrock",
        "config": {
            "access_key_id": "AKIA123",
            "secret_access_key": "secret"
        }
    }), false, Some(r#"property "/config" validation failed: "region" is a required property"#.to_string()))]
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

    #[test]
    fn deserialize_provider_preserves_type_information() {
        let provider: super::Provider = serde_json::from_value(json!({
            "name": "openai-primary",
            "type": "openai",
            "config": {
                "api_key": "test_key",
                "api_base": "https://api.openai.com/v1"
            }
        }))
        .unwrap();

        assert_eq!(provider.name, "openai-primary");
        assert_eq!(provider.provider_type(), "openai");
        assert_matches!(provider.provider, super::ProviderConfig::OpenAI(_));
    }
}
