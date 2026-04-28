use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::{ConfigProvider, EntityStore, IndexFn, ResourceEntry};
use crate::{
    config::entities::{
        Provider, ResourceRegistry,
        types::{HasRateLimit, RateLimit, RateLimitMetric},
    },
    utils::jsonschema::format_evaluation_error,
};

static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("models-schema.json"))
        .expect("Invalid JSON document for Model schema")
});
pub static SCHEMA_VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| jsonschema::validator_for(&SCHEMA).expect("Invalid JSON schema for Model"));

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Model {
    pub name: String,
    pub provider_id: String,
    pub model: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimit>,
}

impl Model {
    /// Get provider of current model
    pub fn provider(&self, resources: &ResourceRegistry) -> Option<ResourceEntry<Provider>> {
        resources.providers.get_by_id(&self.provider_id)
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

    #[test]
    fn test_valid_jsonschema() {
        assert!(jsonschema::meta::is_valid(&SCHEMA));
    }

    #[rstest::rstest]
    #[case::ok(json!({
        "name": "test",
        "provider_id": "openai-primary",
        "model": "gpt-5"
    }), true, None)]
    #[case::ok_with_rate_limit(json!({
        "name": "test",
        "provider_id": "bedrock-primary",
        "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "rate_limit": {}
    }), true, None)]
    #[case::missing_name(json!({
        "provider_id": "openai-primary",
        "model": "gpt-5"
    }), false, Some(r#"property "/" validation failed: "name" is a required property"#.to_string()))]
    #[case::missing_provider_id(json!({
        "name": "test",
        "model": "gpt-5"
    }), false, Some(r#"property "/" validation failed: "provider_id" is a required property"#.to_string()))]
    #[case::missing_model(json!({
        "name": "test",
        "provider_id": "openai-primary"
    }), false, Some(r#"property "/" validation failed: "model" is a required property"#.to_string()))]
    #[case::invalid_name_type(json!({
        "name": 123,
        "provider_id": "openai-primary",
        "model": "gpt-5"
    }), false, Some(r#"property "/name" validation failed: 123 is not of type "string""#.to_string()))]
    #[case::invalid_provider_id_type(json!({
        "name": "test",
        "provider_id": 123,
        "model": "gpt-5"
    }), false, Some(r#"property "/provider_id" validation failed: 123 is not of type "string""#.to_string()))]
    #[case::invalid_model_type(json!({
        "name": "test",
        "provider_id": "openai-primary",
        "model": 123
    }), false, Some(r#"property "/model" validation failed: 123 is not of type "string""#.to_string()))]
    #[case::invalid_root_additional_property(json!({
        "name": "test",
        "provider_id": "openai-primary",
        "model": "gpt-5",
        "extra": "not allowed"
    }), false, Some(r#"property "/" validation failed: Additional properties are not allowed ('extra' was unexpected)"#.to_string()))]
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
    fn deserialize_model_preserves_provider_reference_and_model_name() {
        let model: super::Model = serde_json::from_value(json!({
            "name": "test",
            "provider_id": "bedrock-primary",
            "model": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "timeout": 30000
        }))
        .unwrap();

        assert_eq!(model.name, "test");
        assert_eq!(model.provider_id, "bedrock-primary");
        assert_eq!(
            model.model,
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        );
        assert_eq!(model.timeout, Some(30000));
    }
}
