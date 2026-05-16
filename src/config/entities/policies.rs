use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use cel::Program;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::{ConfigProvider, EntityStore, IndexFn, ResourceEntry};
use crate::utils::jsonschema::format_evaluation_error;

static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("policies-schema.json"))
        .expect("Invalid JSON document for Policy schema")
});
pub static SCHEMA_VALIDATOR: LazyLock<jsonschema::Validator> =
    LazyLock::new(|| jsonschema::validator_for(&SCHEMA).expect("Invalid JSON schema for Policy"));

fn default_enabled() -> bool {
    true
}

fn default_policy_stages() -> Vec<PolicyStage> {
    vec![PolicyStage::Input, PolicyStage::Output]
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PolicyStage {
    Input,
    Output,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct GuardrailPolicyAction {
    #[serde(default = "default_policy_stages")]
    pub stages: Vec<PolicyStage>,
    pub guardrail_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(tag = "type", content = "config")]
pub enum PolicyAction {
    #[serde(rename = "guardrail")]
    Guardrail(GuardrailPolicyAction),
}

impl PolicyAction {
    fn guardrail_ids(&self) -> &[String] {
        match self {
            Self::Guardrail(config) => &config.guardrail_ids,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Policy {
    pub name: String,

    #[serde(default = "default_enabled")]
    pub enabled: bool,

    #[serde(default)]
    pub priority: i32,

    pub when: String,

    pub actions: Vec<PolicyAction>,
}

impl Policy {
    pub fn referenced_guardrail_ids(&self) -> impl Iterator<Item = &str> {
        self.actions
            .iter()
            .flat_map(|action| action.guardrail_ids().iter().map(String::as_str))
    }
}

pub(crate) fn validate_policy_definition(key: &str, value: &Policy) -> Result<(), String> {
    let evaluation = SCHEMA_VALIDATOR.evaluate(
        &serde_json::to_value(value)
            .map_err(|error| format!("Failed to serialize policy for validation: {error}"))?,
    );
    if !evaluation.flag().valid {
        return Err(format!(
            r#"JSON schema validation error on policy "{key}": {}"#,
            format_evaluation_error(&evaluation)
        ));
    }

    Program::compile(&value.when)
        .map_err(|error| format!(r#"CEL validation error on policy "{key}": {error}"#))?;

    Ok(())
}

#[derive(Clone)]
pub struct PoliciesStore {
    store: EntityStore<Policy>,
}

static INDEX_FNS: &[IndexFn<Policy>] = &[("by_name", |policy: &Policy| Some(policy.name.clone()))];

impl PoliciesStore {
    pub async fn new(provider: Arc<dyn ConfigProvider + Send + Sync>) -> Self {
        Self {
            store: EntityStore::new(
                provider,
                "/policies/",
                "policies",
                Some(validate_policy_definition),
                INDEX_FNS,
            )
            .await,
        }
    }

    pub fn list(&self) -> Arc<HashMap<String, ResourceEntry<Policy>>> {
        self.store.list()
    }

    pub fn get_by_id(&self, id: &str) -> Option<ResourceEntry<Policy>> {
        self.store.get(id)
    }

    pub fn get_by_name(&self, name: &str) -> Option<ResourceEntry<Policy>> {
        self.store.get_by_secondary("by_name", name)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::{
        Policy, SCHEMA, SCHEMA_VALIDATOR, format_evaluation_error, validate_policy_definition,
    };

    #[test]
    fn test_valid_jsonschema() {
        assert!(jsonschema::meta::is_valid(&SCHEMA));
    }

    #[rstest::rstest]
    #[case::ok_minimal(json!({
        "name": "tenant-a-bedrock-default",
        "when": "auth.api_key.id == 'tenant-a' && provider.type == 'bedrock'",
        "actions": [{
            "type": "guardrail",
            "config": {
                "guardrail_ids": ["gr-bedrock-default"]
            }
        }]
    }), true, None)]
    #[case::ok_with_explicit_defaults(json!({
        "name": "responses-session-review",
        "enabled": true,
        "priority": 80,
        "when": "route.format == 'responses' && input.messages.size() > 20",
        "actions": [{
            "type": "guardrail",
            "config": {
                "stages": ["input"],
                "guardrail_ids": ["gr-session-review"]
            }
        }]
    }), true, None)]
    #[case::missing_name(json!({
        "when": "true",
        "actions": [{
            "type": "guardrail",
            "config": {
                "stages": ["input"],
                "guardrail_ids": ["gr-input"]
            }
        }]
    }), false, Some(r#"property "/" validation failed: "name" is a required property"#.to_string()))]
    #[case::invalid_stage(json!({
        "name": "invalid-stage",
        "when": "true",
        "actions": [{
            "type": "guardrail",
            "config": {
                "stages": ["tool_call"],
                "guardrail_ids": ["gr-input"]
            }
        }]
    }), false, Some(r#"property "/actions/0/config/stages/0" validation failed: "tool_call" is not one of "input" or "output""#.to_string()))]
    #[case::duplicate_guardrail_ids(json!({
        "name": "duplicate-guardrails",
        "when": "true",
        "actions": [{
            "type": "guardrail",
            "config": {
                "stages": ["input"],
                "guardrail_ids": ["gr-input", "gr-input"]
            }
        }]
    }), false, Some(r#"property "/actions/0/config/guardrail_ids" validation failed: ["gr-input","gr-input"] has non-unique elements"#.to_string()))]
    #[case::invalid_root_additional_property(json!({
        "name": "extra-field",
        "when": "true",
        "actions": [{
            "type": "guardrail",
            "config": {
                "stages": ["input"],
                "guardrail_ids": ["gr-input"]
            }
        }],
        "extra": true
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
    fn validate_policy_definition_rejects_invalid_cel() {
        let policy: Policy = serde_json::from_value(json!({
            "name": "broken-cel",
            "when": "route.format ==",
            "actions": [{
                "type": "guardrail",
                "config": {
                    "stages": ["input"],
                    "guardrail_ids": ["gr-input"]
                }
            }]
        }))
        .unwrap();

        let error = validate_policy_definition("broken-cel", &policy).unwrap_err();

        assert!(error.contains("CEL validation error on policy \"broken-cel\""));
    }

    #[test]
    fn deserialize_policy_defaults_enabled_and_priority() {
        let policy: Policy = serde_json::from_value(json!({
            "name": "defaults",
            "when": "true",
            "actions": [{
                "type": "guardrail",
                "config": {
                    "guardrail_ids": ["gr-input"]
                }
            }]
        }))
        .unwrap();

        assert_eq!(policy.name, "defaults");
        assert!(policy.enabled);
        assert_eq!(policy.priority, 0);
        assert_eq!(
            policy.actions,
            vec![super::PolicyAction::Guardrail(
                super::GuardrailPolicyAction {
                    stages: vec![super::PolicyStage::Input, super::PolicyStage::Output],
                    guardrail_ids: vec!["gr-input".to_string()],
                }
            )]
        );
        assert_eq!(
            policy.referenced_guardrail_ids().collect::<Vec<_>>(),
            vec!["gr-input"]
        );
    }
}
