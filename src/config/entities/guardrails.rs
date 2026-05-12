use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::{ConfigProvider, EntityStore, ResourceEntry};
use crate::{
    guardrail::guardrails::{configs::BedrockGuardrailConfig, identifiers},
    utils::jsonschema::format_evaluation_error,
};

static SCHEMA: LazyLock<serde_json::Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("guardrails-schema.json"))
        .expect("Invalid JSON document for Guardrail schema")
});
pub static SCHEMA_VALIDATOR: LazyLock<jsonschema::Validator> = LazyLock::new(|| {
    jsonschema::validator_for(&SCHEMA).expect("Invalid JSON schema for Guardrail")
});

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", content = "config")]
pub enum GuardrailConfig {
    #[serde(rename = "bedrock")]
    Bedrock(BedrockGuardrailConfig),
}

impl GuardrailConfig {
    pub fn guardrail_type(&self) -> &'static str {
        match self {
            Self::Bedrock(_) => identifiers::BEDROCK,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Guardrail {
    pub name: String,

    #[serde(flatten)]
    #[schema(inline)]
    pub guardrail: GuardrailConfig,
}

impl Guardrail {
    pub fn guardrail_type(&self) -> &'static str {
        self.guardrail.guardrail_type()
    }
}

fn validate(key: &str, value: &Guardrail) -> Result<(), String> {
    let evaluation = SCHEMA_VALIDATOR.evaluate(
        &serde_json::to_value(value)
            .map_err(|error| format!("Failed to serialize guardrail for validation: {}", error))?,
    );
    if !evaluation.flag().valid {
        return Err(format!(
            r#"JSON schema validation error on guardrail "{key}": {}"#,
            format_evaluation_error(&evaluation)
        ));
    }

    Ok(())
}

#[derive(Clone)]
pub struct GuardrailsStore {
    store: EntityStore<Guardrail>,
}

impl GuardrailsStore {
    pub async fn new(provider: Arc<dyn ConfigProvider + Send + Sync>) -> Self {
        Self {
            store: EntityStore::new(provider, "/guardrails/", "guardrails", Some(validate), &[])
                .await,
        }
    }

    pub fn list(&self) -> Arc<HashMap<String, ResourceEntry<Guardrail>>> {
        self.store.list()
    }

    pub fn get_by_id(&self, id: &str) -> Option<ResourceEntry<Guardrail>> {
        self.store.get(id)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    use anyhow::Result;
    use assert_matches::assert_matches;
    use async_trait::async_trait;
    use pretty_assertions::assert_eq;
    use serde_json::json;
    use tokio::sync::mpsc;

    use super::{
        Guardrail, GuardrailConfig, GuardrailsStore, SCHEMA, SCHEMA_VALIDATOR,
        format_evaluation_error,
    };
    use crate::{
        config::{ConfigEvent, ConfigEventReceiver, ConfigProvider, GetEntry, PutEntry},
        guardrail::guardrails::configs::BedrockGuardrailConfig,
    };

    struct MockProvider {
        data: Vec<(String, Vec<u8>)>,
        watch_rx: Mutex<Option<ConfigEventReceiver>>,
    }

    impl MockProvider {
        fn new(data: Vec<(&str, Vec<u8>)>, rx: ConfigEventReceiver) -> Arc<Self> {
            Arc::new(Self {
                data: data
                    .into_iter()
                    .map(|(key, value)| (key.to_string(), value))
                    .collect(),
                watch_rx: Mutex::new(Some(rx)),
            })
        }
    }

    #[async_trait]
    impl ConfigProvider for MockProvider {
        async fn get_all_raw(
            &self,
            _prefix: Option<&str>,
        ) -> Result<Vec<GetEntry<Vec<u8>>>, String> {
            Ok(self
                .data
                .iter()
                .enumerate()
                .map(|(index, (key, value))| GetEntry {
                    key: key.clone(),
                    value: value.clone(),
                    create_revision: index as i64 + 1,
                    mod_revision: index as i64 + 1,
                })
                .collect())
        }

        async fn get_raw(&self, _key: &str) -> Result<Option<GetEntry<Vec<u8>>>, String> {
            Ok(None)
        }

        async fn put_raw(&self, _key: &str, _value: Vec<u8>) -> Result<PutEntry<Vec<u8>>, String> {
            Ok(PutEntry::Created)
        }

        async fn delete(&self, _key: &str) -> Result<i64, String> {
            Ok(0)
        }

        async fn watch(&self, _prefix: Option<&str>) -> anyhow::Result<ConfigEventReceiver> {
            Ok(self
                .watch_rx
                .lock()
                .unwrap()
                .take()
                .expect("MockProvider::watch called more than once"))
        }

        async fn shutdown(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn bedrock_guardrail(name: &str, identifier: &str, version: &str) -> Guardrail {
        Guardrail {
            name: name.to_string(),
            guardrail: GuardrailConfig::Bedrock(BedrockGuardrailConfig {
                identifier: identifier.to_string(),
                version: version.to_string(),
                region: "us-east-1".into(),
                access_key_id: "AKIA123".into(),
                secret_access_key: "secret".into(),
                session_token: None,
                endpoint: None,
            }),
        }
    }

    fn raw(value: &impl serde::Serialize) -> Vec<u8> {
        serde_json::to_vec(value).unwrap()
    }

    fn put_event(key: &str, value: &impl serde::Serialize, rev: i64) -> ConfigEvent {
        ConfigEvent::Put((key.into(), serde_json::to_vec(value).unwrap(), rev))
    }

    #[test]
    fn test_valid_jsonschema() {
        assert!(jsonschema::meta::is_valid(&SCHEMA));
    }

    #[rstest::rstest]
    #[case::bedrock_ok(json!({
        "name": "bedrock-prod",
        "type": "bedrock",
        "config": {
            "identifier": "guardrail-123",
            "version": "1",
            "region": "us-east-1",
            "access_key_id": "AKIA123",
            "secret_access_key": "secret"
        }
    }), true, None)]
    #[case::missing_type(json!({
        "name": "bedrock-prod",
        "config": {
            "identifier": "guardrail-123",
            "version": "1",
            "region": "us-east-1",
            "access_key_id": "AKIA123",
            "secret_access_key": "secret"
        }
    }), false, Some(r#"property "/" validation failed: "type" is a required property"#.to_string()))]
    #[case::missing_identifier(json!({
        "name": "bedrock-prod",
        "type": "bedrock",
        "config": {
            "version": "1",
            "region": "us-east-1",
            "access_key_id": "AKIA123",
            "secret_access_key": "secret"
        }
    }), false, Some(r#"property "/config" validation failed: "identifier" is a required property"#.to_string()))]
    #[case::missing_version(json!({
        "name": "bedrock-prod",
        "type": "bedrock",
        "config": {
            "identifier": "guardrail-123",
            "region": "us-east-1",
            "access_key_id": "AKIA123",
            "secret_access_key": "secret"
        }
    }), false, Some(r#"property "/config" validation failed: "version" is a required property"#.to_string()))]
    #[case::missing_region(json!({
        "name": "bedrock-prod",
        "type": "bedrock",
        "config": {
            "identifier": "guardrail-123",
            "version": "1",
            "access_key_id": "AKIA123",
            "secret_access_key": "secret"
        }
    }), false, Some(r#"property "/config" validation failed: "region" is a required property"#.to_string()))]
    #[case::invalid_root_additional_property(json!({
        "name": "bedrock-prod",
        "type": "bedrock",
        "config": {
            "identifier": "guardrail-123",
            "version": "1",
            "region": "us-east-1",
            "access_key_id": "AKIA123",
            "secret_access_key": "secret"
        },
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
    fn deserialize_guardrail_preserves_type_information() {
        let guardrail: Guardrail = serde_json::from_value(json!({
            "name": "bedrock-prod",
            "type": "bedrock",
            "config": {
                "identifier": "guardrail-123",
                "version": "1",
                "region": "us-east-1",
                "access_key_id": "AKIA123",
                "secret_access_key": "secret"
            }
        }))
        .unwrap();

        assert_eq!(guardrail.name, "bedrock-prod");
        assert_eq!(guardrail.guardrail_type(), "bedrock");
        assert_matches!(guardrail.guardrail, GuardrailConfig::Bedrock(_));
    }

    #[tokio::test]
    async fn guardrails_store_loads_full_snapshot_with_relative_ids() {
        let (tx, rx) = mpsc::channel(8);
        let provider = MockProvider::new(
            vec![(
                "/guardrails/gr-primary",
                raw(&bedrock_guardrail("bedrock-prod", "guardrail-123", "1")),
            )],
            rx,
        );

        let store = GuardrailsStore::new(provider).await;

        let entry = store
            .get_by_id("gr-primary")
            .expect("guardrail should be available after full load");
        assert_eq!(entry.name, "bedrock-prod");
        assert_eq!(entry.guardrail_type(), "bedrock");
        assert_eq!(store.list().len(), 1);

        drop(tx);
    }

    #[tokio::test]
    async fn guardrails_store_skips_schema_invalid_entries_and_applies_watch_put() {
        let (tx, rx) = mpsc::channel(8);
        let provider = MockProvider::new(
            vec![
                (
                    "/guardrails/gr-valid",
                    raw(&bedrock_guardrail("bedrock-prod", "guardrail-123", "1")),
                ),
                (
                    "/guardrails/gr-invalid",
                    raw(&json!({
                        "name": "bedrock-invalid",
                        "type": "bedrock",
                        "config": {
                            "identifier": "",
                            "version": "1",
                            "region": "us-east-1",
                            "access_key_id": "AKIA123",
                            "secret_access_key": "secret"
                        }
                    })),
                ),
            ],
            rx,
        );

        let store = GuardrailsStore::new(provider).await;

        assert!(store.get_by_id("gr-valid").is_some());
        assert!(store.get_by_id("gr-invalid").is_none());

        tx.send(put_event(
            "/aisix/guardrails/gr-live",
            &bedrock_guardrail("bedrock-live", "guardrail-live", "2"),
            10,
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        let live = store
            .get_by_id("gr-live")
            .expect("guardrail should be added from watch event");
        assert_eq!(live.name, "bedrock-live");
        assert_eq!(live.guardrail_type(), "bedrock");
    }
}
